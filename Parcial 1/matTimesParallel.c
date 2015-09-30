#include "malloc.h"
#include "time.h"
#include "stdio.h"
#include "cuda.h"
#include "stdlib.h"
#define N 3
#define Ar 20
#define Ac 5
#define Br 5
#define Bc 20
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

__global__ void matMulti(int *dev_a, int *dev_b, int *dev_c){
  
	int row = blockIdx.y*blockDim.y+threadIdx.y;    
	int col = blockIdx.x*blockDim.x+threadIdx.x;   
	int cont,k;
  
  
	if(( row < Ar ) && ( col < Bc )){
		cont=0;
			for( k = 0 ; k < Ac ; k++){
				cont+= dev_a[row * Ac + k]*dev_b[ k * Bc + col];
			}
			dev_c[ row * Bc + col] = cont;
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

void initialize(int *vec1,int n, int m)
{
  int i;
 	srand(time(NULL));
  for ( i = 0; i< n*m; i++)
  {
  	vec1[i] = 1;//rand() % (1+10-0) + 0;
  }
}

void printTimes(int *a,int *b,int *c)
{
   int i;
  for ( i = Bc*Ar-5; i < Bc*Ar ; i++) // ROW * ROW
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
  Host_a = (int *) malloc ( sizeof(int) * Ar*Ac);
  Host_b = (int *) malloc ( sizeof(int) * Br*Bc);
  Host_c = (int *) malloc ( sizeof(int) * Ar*Bc);
  initialize(Host_a,Ar,Ac);
  initialize(Host_b,Br,Bc);
  //Allocate the memory on the GPU
  HANDLE_ERROR ( cudaMalloc((void **)&dev_a , Ar*Ac*sizeof(int) ) );
  HANDLE_ERROR ( cudaMalloc((void **)&dev_b , Br*Bc*sizeof(int) ) );
  HANDLE_ERROR ( cudaMalloc((void **)&dev_c , Ar*Bc*sizeof(int) ) );
  dim3 dimBlock(32, 32,1);
  dim3 dimGrid((int)ceil((float)N/(float)dimBlock.x),(int)ceil((float)N/(float)dimBlock.y),1);
  begin = clock();
 //Copy Host array to Device array
  HANDLE_ERROR (cudaMemcpy (dev_a , Host_a , Ar*Ac*sizeof(int) , cudaMemcpyHostToDevice));
  HANDLE_ERROR (cudaMemcpy (dev_b , Host_b , Br*Bc*sizeof(int) , cudaMemcpyHostToDevice));


  //Make a call to GPU kernel
  matMulti <<< dimGrid, dimBlock  >>> (dev_a , dev_b , dev_c ) ;


  //Copy back to Host array from Device array
  HANDLE_ERROR (cudaMemcpy(Host_c , dev_c , Ar*Bc*sizeof(int) , cudaMemcpyDeviceToHost));

  end = clock();
  printTimes(Host_a,Host_b,Host_c);
  time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("Se ha demorado %f segundos.\n",time_spent);
  free(Host_a);
  free(Host_b);
  free(Host_c);
}
