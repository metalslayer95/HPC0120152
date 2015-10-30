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
#define MASK_SIZE 3
#define BLOCKSIZE 32

using namespace std;

using namespace cv;

__device__ unsigned char in_Range(int value){
	if(value < 0)
		value = 0;
	else
		if(value > 255)
			value = 255;
	return value;
}

__global__ void sobel_Global(unsigned char *In, int Row, int Col, unsigned int mask_Width,char *Mask,unsigned char *Out){
	unsigned int row = blockIdx.y*blockDim.y+threadIdx.y;
	unsigned int col = blockIdx.x*blockDim.x+threadIdx.x;
	int Pvalue = 0;
	int N_start_point_row = row - (mask_Width/2);
	int N_start_point_col = col - (mask_Width/2);
	for(int i = 0; i < mask_Width; i++){
		for(int j = 0; j < mask_Width; j++ ){
			if((N_start_point_col + j >=0 && N_start_point_col + j < Row)&&(N_start_point_row + i >=0 && N_start_point_row + i < Col)){
			Pvalue += In[(N_start_point_row + i)*Row+(N_start_point_col + j)] * Mask[i*mask_Width+j];
			}
		}
	}
	Out[row*Row+col] = in_Range(Pvalue);
}

__global__ void gray(unsigned char *In, unsigned char *Out,int Row, int Col){
		int row = blockIdx.y*blockDim.y+threadIdx.y;
		int col = blockIdx.x*blockDim.x+threadIdx.x;
		if((row < Col) && (col < Row)){
			Out[row*Row+col] = In[(row*Row+col)*3+2]*0.299 + In[(row*Row+col)*3+1]*0.587+ In[(row*Row+col)*3]*0.114;
		}
}

void sobel_Operator(Mat image,unsigned char *In,unsigned char *h_Out,char *h_Mask,int mask_Width,int Row,int Col){
	// Variables
	int tamano_RGB = sizeof(unsigned char)*Row*Col*image.channels();
	int tamano_Gris = sizeof(unsigned char)*Row*Col; // sin canales alternativos
	int tamano_Mascara = sizeof(char)*(MASK_SIZE*MASK_SIZE);
	unsigned char *d_In,*d_Out,*d_sobelOut;
	char *d_Mask;
	float blocksize=BLOCKSIZE;
	// Memory Allocation in device
	cudaMalloc((void**)&d_In,tamano_RGB);
	cudaMalloc((void**)&d_Out,tamano_Gris);
	cudaMalloc((void**)&d_Mask,tamano_Mascara);
	cudaMalloc((void**)&d_sobelOut,tamano_Gris);
	// Memcpy Host to device
	cudaMemcpy(d_In,In,tamano_RGB, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Mask,h_Mask,tamano_Mascara,cudaMemcpyHostToDevice);
	
	dim3 dimGrid(ceil(Row/blocksize),ceil(Col/blocksize),1);
	dim3 dimBlock(blocksize,blocksize,1);
  clock_t start,end;  
  float tiempo;
  int i;
  for(i=0;i<20;i++)
  {
	start = clock();
    gray<<<dimGrid,dimBlock>>>(d_In,d_Out,Row,Col); // pasando a escala de grices.
	cudaDeviceSynchronize();

	sobel_Global<<<dimGrid,dimBlock>>>(d_Out,Row,Col,MASK_SIZE,d_Mask,d_sobelOut);
	
	cudaMemcpy (h_Out,d_sobelOut,tamano_Gris,cudaMemcpyDeviceToHost);
    end = clock();

    tiempo = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("%f\n",tiempo);
  }
    cudaFree(d_In);
	cudaFree(d_Out);
	cudaFree(d_Mask);
	cudaFree(d_sobelOut);
}



main () 
{
	int mask_Width = MASK_SIZE;
    int Row, Col;
	char h_Mask[] = {-1,0,1,-2,0,2,-1,0,1};
	Mat image,result_image;  
	image = imread("./inputs/img3.jpg");
	Size s = image.size();
	Row = s.width;
	Col = s.height;
	unsigned char * In = (unsigned char*)malloc( sizeof(unsigned char)*Row*Col*image.channels());
	unsigned char * h_Out = (unsigned char *)malloc( sizeof(unsigned char)*Row*Col);
	In = image.data;
    sobel_Operator(image,In,h_Out,h_Mask,mask_Width,Row,Col);
	result_image.create(Col,Row,CV_8UC1);
	result_image.data = h_Out;
	imwrite("./outputs/1088328019.png",result_image);

	return 0;
}

   