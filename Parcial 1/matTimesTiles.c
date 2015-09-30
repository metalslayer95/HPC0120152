#include "malloc.h"
#include "time.h"
#include "stdio.h"
#include "cuda.h"
#include "stdlib.h"
#define N 20
#define ACols 100
#define BRows 100
#define BCols 60
#define ARows 60
#define TILE_DIM 32
#define HANDLE_ERROR( err ) ( HandleError( err, __FILE__, __LINE__ ) )
# define BLOCK_DIM 6

static void HandleError( cudaError_t err, const char *file, int line )
{
  if (err != cudaSuccess)
  {
    printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
           file, line );
    exit( EXIT_FAILURE );
  }
}

__global__ void MatMul(float* A, float* B, float* C, int CRows, int CCols) {

    float CValue = 0;

    int Row = blockIdx.y*TILE_DIM + threadIdx.y;
    int Col = blockIdx.x*TILE_DIM + threadIdx.x;

    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    for (int k = 0; k < (TILE_DIM + ACols - 1)/TILE_DIM; k++) {

         if (k*TILE_DIM + threadIdx.x < ACols && Row < ARows)   
           As[threadIdx.y][threadIdx.x] = A[Row*ACols + k*TILE_DIM + threadIdx.x];
         else                                                   
           As[threadIdx.y][threadIdx.x] = 0.0;

         if (k*TILE_DIM + threadIdx.y < BRows && Col < BCols)   
           Bs[threadIdx.y][threadIdx.x] = B[(k*TILE_DIM + threadIdx.y)*BCols + Col];
         else                                                   
           Bs[threadIdx.y][threadIdx.x] = 0.0;

         __syncthreads();

         for (int n = 0; n < TILE_DIM; ++n) CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];

         __syncthreads();
    }

    if (Row < CRows && Col < CCols) C[((blockIdx.y * blockDim.y + threadIdx.y)*CCols)+(blockIdx.x*blockDim.x)+threadIdx.x]=CValue;
}


// CUDA Kernel for Vector Addition
__global__ void matrix_Multiplication( const float *dev_a , const float *dev_b , float *dev_c)
{
  int col = blockIdx.x*blockDim.x+threadIdx.x;
  int row = blockIdx.y*blockDim.y+threadIdx.y;
  if ( col < N && row < N) // check the boundry condition for the threads
  {
			int Pvalue = 0;
     for(int k = 0 ; k < N ; k++ )
     {
        Pvalue+= dev_a[row * N + k ] * dev_b[k*N+ col];
     }
  		dev_c[row * N + col] = Pvalue;
  }
}

void initialize(float *vec1, int n , int m)
{
  int i;
  //printf("Reach\n");

 	srand(time(NULL));
  for ( i = 0; i< n*m; i++)
  {
  	vec1[i] = 1.0;//rand() % (1+10-0) + 0;
  }
}

void printTimes(float *a,float *b,float *c)
{
   int i,aux = ARows*BCols-5;
  for ( i = aux; i < ARows*BCols ; i++) 
  { 
  	printf("%d = %f\n",i,c[i]);
  } 
}

main () 
{
  float *Host_a,*Host_b,*Host_c;
  float *dev_a,*dev_b,*dev_c;
  clock_t begin, end;
  double time_spent;

  Host_a = NULL;
  Host_b = NULL;
  Host_c = NULL;
  Host_a = (float *) malloc ( sizeof(float) * ACols*ARows);
  Host_b = (float *) malloc ( sizeof(float) * BCols*BRows);
  Host_c = (float *) malloc ( sizeof(float) * ARows*BCols);
  initialize(Host_a,ACols,ARows);
  initialize(Host_b,BCols,BRows);
  //Allocate the memory on the GPU
  //printf("Reach\n");

  HANDLE_ERROR ( cudaMalloc((void **)&dev_a , ACols*ARows*sizeof(float) ) );
  HANDLE_ERROR ( cudaMalloc((void **)&dev_b , BCols*BRows*sizeof(float) ) );
  HANDLE_ERROR ( cudaMalloc((void **)&dev_c , ARows*BCols*sizeof(float) ) );
  dim3 dimBlock(TILE_DIM, TILE_DIM,1);
  dim3 dimGrid((int)ceil((float)ARows/(float)dimBlock.x),(int)ceil((float)BCols/(float)dimBlock.y),1);
  begin = clock();
 //Copy Host array to Device array
  HANDLE_ERROR (cudaMemcpy (dev_a , Host_a , ACols*ARows*sizeof(float) , cudaMemcpyHostToDevice));
  HANDLE_ERROR (cudaMemcpy (dev_b , Host_b , BCols*BRows*sizeof(float) , cudaMemcpyHostToDevice));


  //Make a call to GPU kernel
  //matrix_Multiplication_Tiles <<< dimGrid, dimBlock  >>> (dev_a , dev_b , dev_c ) ;
	MatMul <<< dimGrid,dimBlock >>> (dev_a,dev_b,dev_c,ARows,BCols);

  //Copy back to Host array from Device array
  HANDLE_ERROR (cudaMemcpy(Host_c , dev_c , ARows*BCols*sizeof(float) , cudaMemcpyDeviceToHost));

  end = clock();
  printTimes(Host_a,Host_b,Host_c);
  time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("%f\n",time_spent);
  free(Host_a);
  free(Host_b);
  free(Host_c);
}
