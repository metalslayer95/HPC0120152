# define COL 2
# define ROW 5
# define N 10
# define BLOCK_DIM 512
#include "malloc.h"
#include "time.h"
#include "stdio.h"
#include "stdlib.h"
#define HANDLE_ERROR( err ) ( HandleError( err, __FILE__, __LINE__ ) )

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
__global__ void matrix_Addition ( const int *dev_a , const int *dev_b , int *dev_c)
{
  int col = blockIdx.x*blockDim.x+threadIdx.x;
  int row = blockIdx.y*blockDim.y+threadIdx.y;
  int index = col + row * N;

  if ( col < N && row < N) // check the boundry condition for the threads
    dev_c [index] = dev_a[index] + dev_b[index] ;

}


void initialize(int *vec1,int *vec2)
{
  int i;
 	srand(time(NULL));
  for ( i = 0; i< N*N; i++)
  {
  	vec1[i] = rand() % (1+10-0) + 0;
  	vec2[i] = rand() % (1+10-0) + 0;
  }
}

void printAdd(int *a,int *b,int *c)
{
   int i;
  for ( i = 0; i < N*N ; i++)
  { 
  	printf("%d + %d = %d\n",a[i],b[i],c[i]);
  } 
}

main () 
{
  int *Host_a,*Host_b,*Host_c;
  int *dev_a,*dev_b,*dev_c;
  clock_t begin, end;
	double time_spent;
  //dim3 grid(16,16); // grid = 16 x 16 blocks
	//dim3 block(100,1); 
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
 	dim3 dimBlock(COL, ROW);
	dim3 dimGrid((int)ceil((float)N/(float)dimBlock.x),(int)ceil((float)N/(float)dimBlock.y));
  begin = clock();
  //Copy Host array to Device array
  HANDLE_ERROR (cudaMemcpy (dev_a , Host_a , N*N*sizeof(int) , cudaMemcpyHostToDevice));
  HANDLE_ERROR (cudaMemcpy (dev_b , Host_b , N*N*sizeof(int) , cudaMemcpyHostToDevice));


  //Make a call to GPU kernel
  matrix_Addition <<< dimGrid, dimBlock  >>> (dev_a , dev_b , dev_c ) ;


  //Copy back to Host array from Device array
  HANDLE_ERROR (cudaMemcpy(Host_c , dev_c , N*N*sizeof(int) , cudaMemcpyDeviceToHost));

  end = clock();
  printAdd(Host_a,Host_b,Host_c);
  time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("Se ha demorado %f segundos.\n",time_spent);
  free(Host_a);
  free(Host_b);
  free(Host_c);
}
