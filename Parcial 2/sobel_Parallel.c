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

__global__ void sobel_Filter(){

}


main () 
{
  int *Host_a,*Host_b,*Host_c;
  int *dev_a,*dev_b,*dev_c;
  clock_t begin, end;
  double time_spent;
  Mat src, src_gray;
  Mat grad;
  src = imread( "./inputs/img1.jpg" );
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
  dim3 dimGrid((int)ceil((float)Ar/(float)dimBlock.x),(int)ceil((float)Bc/(float)dimBlock.y),1);
  begin = clock();
 //Copy Host array to Device array
  HANDLE_ERROR (cudaMemcpy (dev_a , Host_a , Ar*Ac*sizeof(int) , cudaMemcpyHostToDevice));
  HANDLE_ERROR (cudaMemcpy (dev_b , Host_b , Br*Bc*sizeof(int) , cudaMemcpyHostToDevice));


  //Make a call to GPU kernel
  matMulti <<< dimGrid, dimBlock  >>> (dev_a , dev_b , dev_c ) ;


  //Copy back to Host array from Device array
  HANDLE_ERROR (cudaMemcpy(Host_c , dev_c , Ar*Bc*sizeof(int) , cudaMemcpyDeviceToHost));

  end = clock();

  imwrite("./outputs/1088328019.png",grad);
  time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("%f\n",time_spent);
  free(Host_a);
  free(Host_b);
  free(Host_c);
}
