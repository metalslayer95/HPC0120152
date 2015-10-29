#include "malloc.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "time.h"
#include "stdio.h"
#include "cuda.h"
#include "stdlib.h"
#define HANDLE_ERROR( err ) ( HandleError( err, __FILE__, __LINE__ ) )
# define BLOCK_DIM 512

static void HandleError( cudaError_t err, const char *file, int line )
{
  if (err != cudaSuccess)
  {
    printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
           file, line );
    exit( EXIT_FAILURE );
  }
}
#define Mask_size 3
#define BLOCKSIZE 32
#define TILE_SIZE 32

using namespace std;

using namespace cv;

__constant__ char Global_Mask[Mask_size*Mask_size];
__device__ unsigned char clamp(int value){
	if(value < 0)
		value = 0;
	else
		if(value > 255)
			value = 255;
	return value;
}
__global__ void sobelFilter(unsigned char *In, int Row, int Col, unsigned int Mask_Width,char *Mask,unsigned char *Out){
	unsigned int row = blockIdx.y*blockDim.y+threadIdx.y;
	unsigned int col = blockIdx.x*blockDim.x+threadIdx.x;
	int Pvalue = 0;
	int N_start_point_row = row - (Mask_Width/2);
	int N_start_point_col = col - (Mask_Width/2);
	for(int i = 0; i < Mask_Width; i++){
		for(int j = 0; j < Mask_Width; j++ ){
			if((N_start_point_col + j >=0 && N_start_point_col + j < Row)&&(N_start_point_row + i >=0 && N_start_point_row + i < Col)){
			Pvalue += In[(N_start_point_row + i)*Row+(N_start_point_col + j)] * Mask[i*Mask_Width+j];
			}
		}
	}
	Out[row*Row+col] = clamp(Pvalue);
}
__global__ void sobelFilterConstant(unsigned char *In, int Row, int Col, unsigned int Mask_Width,unsigned char *Out){
	unsigned int row = blockIdx.y*blockDim.y+threadIdx.y;
	unsigned int col = blockIdx.x*blockDim.x+threadIdx.x;
	int Pvalue = 0;
	int N_start_point_row = row - (Mask_Width/2);
	int N_start_point_col = col - (Mask_Width/2);
	for(int i = 0; i < Mask_Width; i++){
		for(int j = 0; j < Mask_Width; j++ ){
			if((N_start_point_col + j >=0 && N_start_point_col + j < Row)&&(N_start_point_row + i >=0 && N_start_point_row + i < Col)){
				Pvalue += In[(N_start_point_row + i)*Row+(N_start_point_col + j)] * Global_Mask[i*Mask_Width+j];
			}
		}
	}
	Out[row*Row+col] = clamp(Pvalue);
}
__global__ void sobel_Compartida(unsigned char *In, unsigned char *Out,int maskWidth, int width, int height){
	__shared__ float N_ds[TILE_SIZE + Mask_size - 1][TILE_SIZE+ Mask_size - 1];
	int n = Mask_size/2;
	int dest = threadIdx.y*TILE_SIZE+threadIdx.x, destY = dest / (TILE_SIZE+Mask_size-1), destX = dest % (TILE_SIZE+Mask_size-1),
	srcY = blockIdx.y * TILE_SIZE + destY - n, srcX = blockIdx.x * TILE_SIZE + destX - n,
	src = (srcY * width + srcX);
	if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
	N_ds[destY][destX] = In[src];
	else
	N_ds[destY][destX] = 0;
	// Second batch loading
	dest = threadIdx.y * TILE_SIZE + threadIdx.x + TILE_SIZE * TILE_SIZE;
	destY = dest /(TILE_SIZE + Mask_size - 1), destX = dest % (TILE_SIZE + Mask_size - 1);
	srcY = blockIdx.y * TILE_SIZE + destY - n;
	srcX = blockIdx.x * TILE_SIZE + destX - n;
	src = (srcY * width + srcX);
	if (destY < TILE_SIZE + Mask_size - 1) {
		if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
			N_ds[destY][destX] = In[src];
		else
			N_ds[destY][destX] = 0;
		}
	__syncthreads();
	int accum = 0;
	int y, x;
	for (y = 0; y < maskWidth; y++)
		for (x = 0; x < maskWidth; x++)
			accum += N_ds[threadIdx.y + y][threadIdx.x + x] * Global_Mask[y * maskWidth + x];
		y = blockIdx.y * TILE_SIZE + threadIdx.y;
		x = blockIdx.x * TILE_SIZE + threadIdx.x;
		if (y < height && x < width)
			Out[(y * width + x)] = clamp(accum);
		__syncthreads();
	}
	__global__ void gray(unsigned char *In, unsigned char *Out,int Row, int Col){
		int row = blockIdx.y*blockDim.y+threadIdx.y;
		int col = blockIdx.x*blockDim.x+threadIdx.x;
		if((row < Col) && (col < Row)){
			Out[row*Row+col] = In[(row*Row+col)*3+2]*0.299 + In[(row*Row+col)*3+1]*0.587+ In[(row*Row+col)*3]*0.114;
		}
		}
		// :::::::::::::::::::::::::::::::::::Clock Function::::::::::::::::::::::::::::
		double diffclock(clock_t clock1,clock_t clock2){
		double diffticks=clock2-clock1;
		double diffms=(diffticks)/(CLOCKS_PER_SEC/1); // /1000 mili
		return diffms;
	}

void d_convolution2d(Mat image,unsigned char *In,unsigned char *h_Out,char *h_Mask,int Mask_Width,int Row,int Col,int op){
	// Variables
	int size_of_rgb = sizeof(unsigned char)*Row*Col*image.channels();
	int size_of_Gray = sizeof(unsigned char)*Row*Col; // sin canales alternativos
	int Mask_size_of_bytes = sizeof(char)*(Mask_size*Mask_size);
	unsigned char *d_In,*d_Out,*d_sobelOut;
	char *d_Mask;
	float Blocksize=BLOCKSIZE;
	// Memory Allocation in device
	cudaMalloc((void**)&d_In,size_of_rgb);
	cudaMalloc((void**)&d_Out,size_of_Gray);
	cudaMalloc((void**)&d_Mask,Mask_size_of_bytes);
	cudaMalloc((void**)&d_sobelOut,size_of_Gray);
	// Memcpy Host to device
	cudaMemcpy(d_In,In,size_of_rgb, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Mask,h_Mask,Mask_size_of_bytes,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Global_Mask,h_Mask,Mask_size_of_bytes); // avoid cache coherence
	// Thread logic and Kernel call
	dim3 dimGrid(ceil(Row/Blocksize),ceil(Col/Blocksize),1);
	dim3 dimBlock(Blocksize,Blocksize,1);
	gray<<<dimGrid,dimBlock>>>(d_In,d_Out,Row,Col); // pasando a escala de grices.
	cudaDeviceSynchronize();
	if(op==1){
	sobelFilter<<<dimGrid,dimBlock>>>(d_Out,Row,Col,Mask_size,d_Mask,d_sobelOut);
	}
	if(op==2){
	sobelFilterConstant<<<dimGrid,dimBlock>>>(d_Out,Row,Col,Mask_size,d_sobelOut);
	}
	if(op==3){
	sobelFilterShared<<<dimGrid,dimBlock>>>(d_Out,d_sobelOut,3,Row,Col);
	}
	cudaMemcpy (h_Out,d_sobelOut,size_of_Gray,cudaMemcpyDeviceToHost);
	cudaFree(d_In);
	cudaFree(d_Out);
	cudaFree(d_Mask);
	cudaFree(d_sobelOut);
}



main () 
{
	 double T1,T2; // Time flags
	clock_t start,end;// Time flags
	int Mask_Width = Mask_size;
	char h_Mask[] = {-1,0,1,-2,0,2,-1,0,1};
	Mat image,result_image;
	image = imread("./inputs/img1.jpg");
	Size s = image.size();
	int Row = s.width;
	int Col = s.height;
	unsigned char * In = (unsigned char*)malloc( sizeof(unsigned char)*Row*Col*image.channels());
	unsigned char * h_Out = (unsigned char *)malloc( sizeof(unsigned char)*Row*Col);
	In = image.data;
	start = clock();
	d_convolution2d(image,In,h_Out,h_Mask,Mask_Width,Row,Col,3);
	end = clock();
	T1=diffclock(start,end);
	cout<<" Result Parallel"<<" At "<<T1<<",Seconds"<<endl;
	Mat gray_image_opencv, grad_x, abs_grad_x;
	start = clock();
	cvtColor(image, gray_image_opencv, CV_BGR2GRAY);
	Sobel(gray_image_opencv,grad_x,CV_8UC1,1,0,3,1,0,BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);
	end = clock();
	T2=diffclock(start,end);
	cout<<" Result secuential"<<" At "<<T2<<",Seconds"<<endl;
	cout<<"Total acceleration "<<T2/T1<<"X"<<endl;
	result_image.create(Col,Row,CV_8UC1);
	result_image.data = h_Out;
	imwrite("./outputs/1088328019.png",result_image);
	return 0;
}
