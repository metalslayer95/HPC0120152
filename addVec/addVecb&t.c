		 #include <stdio.h>

		#define HANDLE_ERROR( err ) ( HandleError( err, __FILE__, __LINE__ ) )
		# define num_block 1024
		# define N 500000
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
		__global__ void Vector_Addition ( const float *dev_a , const float *dev_b , float *dev_c)
		{
			  //Get the id of thread within a block
			  //unsigned short tid = threadIdx.x ;
			  unsigned short tid = blockIdx.x*blockDim.x+threadIdx.x ;
			 
			  if ( tid < N ) // check the boundry condition for the threads
					dev_c [tid] = dev_a[tid] + dev_b[tid] ;

		}


		int main (void)
		{

			  //Host array
			  float Host_a[N], Host_b[N], Host_c[N];

				clock_t begin, end;
				double time_spent;
			  //Device array
			  float *dev_a , *dev_b, *dev_c ;

			  //Allocate the memory on the GPU
			  HANDLE_ERROR ( cudaMalloc((void **)&dev_a , N*sizeof(float) ) );
			  HANDLE_ERROR ( cudaMalloc((void **)&dev_b , N*sizeof(float) ) );
			  HANDLE_ERROR ( cudaMalloc((void **)&dev_c , N*sizeof(float) ) );

			  //fill the Host array with random elements on the CPU
			  for ( int i = 0; i <N ; i++ )
			  {
					Host_a[i] = -i ;
					Host_b[i] = i*i ; 
			  }
			  
			  int dimGrid = (int)ceil((float)N/(float)num_block);
				begin = clock();
			  //Copy Host array to Device array
			  HANDLE_ERROR (cudaMemcpy (dev_a , Host_a , N*sizeof(float) , cudaMemcpyHostToDevice));
			  HANDLE_ERROR (cudaMemcpy (dev_b , Host_b , N*sizeof(float) , cudaMemcpyHostToDevice));


			  //Make a call to GPU kernel
			  Vector_Addition <<< dimGrid, num_block  >>> (dev_a , dev_b , dev_c ) ;


			  //Copy back to Host array from Device array
			  HANDLE_ERROR (cudaMemcpy(Host_c , dev_c , N*sizeof(float) , cudaMemcpyDeviceToHost));

		  end = clock();
			  //Display the result
			  //for ( int i = 0; i<N; i++ )
				//		  printf ("%f + %f = %f\n", Host_a[i] , Host_b[i] , Host_c[i] ) ;

			  time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
			  printf("Se ha demorado %f segundos.\n",time_spent);
			  //Free the Device array memory
			  cudaFree (dev_a) ;
			  cudaFree (dev_b) ;
			  cudaFree (dev_c) ;

			  return 0 ;

		}