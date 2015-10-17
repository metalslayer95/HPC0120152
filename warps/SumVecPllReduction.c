# include <bits/stdc++.h>
# include <cuda.h>

#define SIZE 600000000// Global Size
#define BLOCK_SIZE 1024
using namespace std;

//::::::::::::::::::::::::::::::::::::::::::GPU::::::::::::::::::::::::::::::::

// :::: Kernel

__global__ void vecSum(double *g_idata,double *g_odata,int l){ // Sequential Addressing technique

  __shared__ double sdata[BLOCK_SIZE];
  // each thread loads one element from global to shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<l){ // bad thing -> severely punished performance.
    sdata[tid] = g_idata[i];
  }else{
    sdata[tid] = 0.0;
  }

  __syncthreads();
  // do reduction in shared mem
  for(unsigned int s=blockDim.x/2; s > 0; s >>= 1) {
    if(tid < s){
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  // write result for this block to global mem
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// :::: Calls
void d_VectorMult(double *vec1,double *Total){
  double * d_Vec1;
  double * d_Total;
  double Blocksize=BLOCK_SIZE; // Block of 1Dim

  cudaMalloc((void**)&d_Vec1,SIZE*sizeof(double));
  cudaMalloc((void**)&d_Total,SIZE*sizeof(double));

  cudaMemcpy(d_Vec1,vec1,SIZE*sizeof(double),cudaMemcpyHostToDevice);

    cudaMemcpy(d_Total,Total,SIZE*sizeof(double),cudaMemcpyHostToDevice);

    int temp=SIZE;

    while(temp>1){
      dim3 dimBlock(Blocksize,1,1);
      int grid=ceil(temp/Blocksize);
      dim3 dimGrid(grid,1,1);

      vecSum<<<dimGrid,dimBlock>>>(d_Vec1,d_Total,temp);
      cudaDeviceSynchronize();

      cudaMemcpy(d_Vec1,d_Total,SIZE*sizeof(double),cudaMemcpyDeviceToDevice);
      temp=ceil(temp/Blocksize);
    }

    cudaMemcpy(Total,d_Total,SIZE*sizeof(double),cudaMemcpyDeviceToHost);

    cudaFree(d_Vec1);
    cudaFree(d_Total);
  }

  //::::::::::::::::::::::::::::::::::::::::::CPU::::::::::::::::::::::::::::::::

  void h_sum_vec(double *vec1, double *all){
    for(int i=0;i<SIZE;i++) *all+=vec1[i];
  }

  //:::::::::::::::::::::::::::: Rutinary Functions

  void fill_Vec(double *vec,double Value){
    for(int i =0 ; i<SIZE ; i++) vec[i]=Value;
  }

  void Show_vec(double *vec){
    for (int i=0;i<SIZE;i++){
      if(i%10==0 && i!=0){
        cout<<endl;
      }
      cout<<"["<<vec[i]<<"] ";
    }
    cout<<endl;
  }



  // :::::::::::::::::::::::::::::::::::Clock Function::::::::::::::::::::::::::::
  double diffclock(clock_t clock1,clock_t clock2){
    double diffticks=clock2-clock1;
    double diffms=(diffticks)/(CLOCKS_PER_SEC); // /1000 mili
    return diffms;
  }

  // :::::::::::::::::::::::::::::::::::::::Main::::::::::::::::::::::::::::::::.

int main(){

    double T1,T2; 
    double *vec1 = (double*)malloc((SIZE)*sizeof(double)); // Elements to compute. CPU way
    double *total2 = (double*)malloc((SIZE)*sizeof(double)); // GPU
    double *total1 = (double*)malloc(sizeof(double)); // Total Variables.

    fill_Vec(vec1,1.0);
    fill_Vec(total2,0.0);

    // Secuential
    clock_t start = clock();
    h_sum_vec(vec1,total1);
    clock_t end = clock();
    T1=diffclock(start,end);
    cout<<"Resultado secuencial: "<<*total1<<" en "<<T1<<",segundos"<<endl;
    // Parallel
    start = clock();
    d_VectorMult(vec1,total2);
    end = clock();
    T2=diffclock(start,end);
    cout<<"Resultado en paralelo: "<<total2[0]<<" en "<<T2<<",segundos"<<endl;
    cout<<"Aceleracion total: "<<T1/T2<<",X"<<endl;

    free(vec1);
    free(total2);

    return 0;
}