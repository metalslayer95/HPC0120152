#include "malloc.h"
#include "time.h"
#include "stdio.h"
#include "cuda.h"
#include "stdlib.h"
#define Ar 4
#define Ac 4
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

__global__ void matSum(int *dev_a){
  
	int row = blockIdx.y*blockDim.y+threadIdx.y;    
	int col = blockIdx.x*blockDim.x+threadIdx.x;   
  int index  = row * Ac + col;   
  int index2 = (row+((Ac)/2)) * Ac + col; 
  //printf("%d\t",index);
	if((index < (Ar*Ac+1)/2) && (index2 < Ar*Ac) ){
 			 printf("%d %d\t",dev_a[index2],index2);
    	 printf("\n");
				dev_a[index] = dev_a[index]+dev_a[index2];
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

void print(int *vec)
{
  int i;
  for ( i = 0 ; i < Ac*Ar;i++)
    printf("%d\n",vec[i]);
  
}

main () 
{
  int *Host_a;
  int *dev_a;
  clock_t begin, end;
  double time_spent;

  Host_a = NULL;
  Host_a = (int *) malloc ( sizeof(int) * Ar*Ac);
  initialize(Host_a,Ar,Ac);
  //Allocate the memory on the GPU
  HANDLE_ERROR ( cudaMalloc((void **)&dev_a , Ar*Ac*sizeof(int) ) );
  dim3 dimBlock(Ar,Ac,1);
  dim3 dimGrid((int)ceil((float)Ar/(float)dimBlock.x),(int)ceil((float)Ac/(float)dimBlock.y),1);
  //dim3 dimGrid(32,32,1);
  begin = clock();
 //Copy Host array to Device array
  HANDLE_ERROR (cudaMemcpy (dev_a , Host_a , Ar*Ac*sizeof(int) , cudaMemcpyHostToDevice));


  //Make a call to GPU kernel
  matSum <<< dimGrid, dimBlock  >>> (dev_a ) ;


  //Copy back to Host array from Device array
  HANDLE_ERROR (cudaMemcpy(Host_a , dev_a , Ar*Ac*sizeof(int) , cudaMemcpyDeviceToHost));

  end = clock();
  print(Host_a);
  time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("%f\n",time_spent);
  free(Host_a);
}
