#include "malloc.h"
#include "time.h"
#include "stdio.h"
#include "cuda.h"
#include "stdlib.h"
#define N 512
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



// CUDA Kernel for Vector Addition
__global__ void matrix_Multiplication( const int *dev_a , const int *dev_b , int *dev_c)
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

void initialize(int *vec1,int *vec2)
{
  int i;
 	srand(time(NULL));
  for ( i = 0; i< N*N; i++)
  {
  	vec1[i] = 1;//rand() % (1+10-0) + 0;
  	vec2[i] = 1;//rand() % (1+20-0) + 0;
  }
}

void printTimes(int *a,int *b,int *c)
{
   int i;
  for ( i = N*N-5; i < N*N ; i++) // ROW * ROW
  { 
  	printf("%d = %d\n",i,c[i]);
  } 
}

main () 
{
  int *Host_a,*Host_b,*Host_c;
  int *dev_a,*dev_b,*dev_c;
  clock_t begin, end;
  double time_spent;

  Host_a = NULL;
  Host_b = NULL;
  Host_c = NULL;
  Host_a = (int *) malloc ( sizeof(int) * N*N);
  Host_b = (int *) malloc ( sizeof(int) * N*N);
  Host_c = (int *) malloc ( sizeof(int) * N*N);
  initialize(Host_a,Host_b);
  //Allocate the memory on the GPU
  HANDLE_ERROR ( cudaMalloc((void **)&dev_a , N*N*sizeof(int) ) );
  HANDLE_ERROR ( cudaMalloc((void **)&dev_b , N*N*sizeof(int) ) );
  HANDLE_ERROR ( cudaMalloc((void **)&dev_c , N*N*sizeof(int) ) );
  dim3 dimBlock(32, 32,1);
  dim3 dimGrid((int)ceil((float)N/(float)dimBlock.x),(int)ceil((float)N/(float)dimBlock.y),1);
  begin = clock();
 //Copy Host array to Device array
  HANDLE_ERROR (cudaMemcpy (dev_a , Host_a , N*N*sizeof(int) , cudaMemcpyHostToDevice));
  HANDLE_ERROR (cudaMemcpy (dev_b , Host_b , N*N*sizeof(int) , cudaMemcpyHostToDevice));


  //Make a call to GPU kernel
  matrix_Multiplication <<< dimGrid, dimBlock  >>> (dev_a , dev_b , dev_c ) ;


  //Copy back to Host array from Device array
  HANDLE_ERROR (cudaMemcpy(Host_c , dev_c , N*N*sizeof(int) , cudaMemcpyDeviceToHost));

  end = clock();
  printTimes(Host_a,Host_b,Host_c);
  time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("Se ha demorado %f segundos.\n",time_spent);
  free(Host_a);
  free(Host_b);
  free(Host_c);
}
