#include "malloc.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "time.h"
#include "stdio.h"
#include "cuda.h"
#include <bits/stdc++.h>
#include <highgui.h>
#include <cv.h>
#include "stdlib.h"
#define Mask_size 3
#define BLOCKSIZE 32
#define TILE_SIZE 32
#define HANDLE_ERROR( err ) ( HandleError( err, __FILE__, __LINE__ ) )

using namespace std;

using namespace cv;

__constant__ char Global_Mask[Mask_size*Mask_size];

static void HandleError( cudaError_t err, const char *file, int line )
{
  if (err != cudaSuccess)
  {
    printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
           file, line );
    exit( EXIT_FAILURE );
  }
}

__device__ unsigned char in_Range(int valor){
	if(valor < 0)
		valor = 0;
	else
		if(valor > 255)
			valor = 255;
	return valor;
}

__global__ void gray(unsigned char *In, unsigned char *Out,int Row, int Col){
		int row = blockIdx.y*blockDim.y+threadIdx.y;
		int col = blockIdx.x*blockDim.x+threadIdx.x;
		if((row < Col) && (col < Row)){
			Out[row*Row+col] = In[(row*Row+col)*3+2]*0.299 + In[(row*Row+col)*3+1]*0.587+ In[(row*Row+col)*3]*0.114;
		}
		}
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
			Out[(y * width + x)] = in_Range(accum);
		__syncthreads();
}

void sobel_Operator(Mat image,unsigned char *In,unsigned char *h_Out,char *h_Mask,int Mask_Width,int Row,int Col){
	// Variables
	int tamano_RGB = sizeof(unsigned char)*Row*Col*image.channels();
	int tamano_Gris = sizeof(unsigned char)*Row*Col; // sin canales alternativos
	int Mask_size_of_bytes = sizeof(char)*(Mask_size*Mask_size);
	unsigned char *d_In,*d_Out,*d_sobelOut;
	char *d_Mask;
	float Blocksize=BLOCKSIZE;
	// Memory Allocation in device
	cudaMalloc((void**)&d_In,tamano_RGB);
	cudaMalloc((void**)&d_Out,tamano_Gris);
	cudaMalloc((void**)&d_Mask,Mask_size_of_bytes);
	cudaMalloc((void**)&d_sobelOut,tamano_Gris);
	HANDLE_ERROR (cudaMemcpy(d_In,In,tamano_RGB, cudaMemcpyHostToDevice));
	HANDLE_ERROR (cudaMemcpy(d_Mask,h_Mask,Mask_size_of_bytes,cudaMemcpyHostToDevice));
	HANDLE_ERROR (cudaMemcpyToSymbol(Global_Mask,h_Mask,Mask_size_of_bytes)); 
	clock_t start,end;  
	float tiempo;
    int i;
	dim3 dimGrid(ceil(Row/Blocksize),ceil(Col/Blocksize),1);
	dim3 dimBlock(Blocksize,Blocksize,1);
	for(i=0;i<20;i++)
	{
		start = clock();
		gray<<<dimGrid,dimBlock>>>(d_In,d_Out,Row,Col); 
		
		cudaDeviceSynchronize();

		sobel_Compartida<<<dimGrid,dimBlock>>>(d_Out,d_sobelOut,3,Row,Col);

		HANDLE_ERROR (cudaMemcpy (h_Out,d_sobelOut,tamano_Gris,cudaMemcpyDeviceToHost));
	    end = clock();

		tiempo = ((double) (end - start)) / CLOCKS_PER_SEC;
		printf("%d\n",tiempo);
	}
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
	  int i;
	for(i = 0 ; i < 20; i++)
	{
	image = imread("./inputs/img1.jpg");
	Size s = image.size();
	int Row = s.width;
	int Col = s.height;
	unsigned char * In = (unsigned char*)malloc( sizeof(unsigned char)*Row*Col*image.channels());
	unsigned char * h_Out = (unsigned char *)malloc( sizeof(unsigned char)*Row*Col);
	In = image.data;
    sobel_Operator(image,In,h_Out,h_Mask,Mask_Width,Row,Col);
	result_image.create(Col,Row,CV_8UC1);
	result_image.data = h_Out;
	imwrite("./outputs/1088328019.png",result_image);
	}
	return 0;
}

   