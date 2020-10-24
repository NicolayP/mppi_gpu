#include "gpu_cmake.hpp"
#include <iostream>
#include <assert.h>

__global__ void kernel(float* o, float* a, float* b, int n){
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if(tid < n){
    o[tid] = a[tid] + b[tid];
  }
}

int main(){
  float *h_o, *h_a, *h_b, *d_o, *d_a, *d_b;
  size_t bytes = sizeof(float)*N;

  int block_size = 256;
  int grid_size = ((N + block_size)/block_size);

  h_o = (float*) malloc(bytes);
  h_a = (float*) malloc(bytes);
  h_b = (float*) malloc(bytes);

  cudaMalloc((void**)&d_o, bytes);
  cudaMalloc((void**)&d_a, bytes);
  cudaMalloc((void**)&d_b, bytes);

  for(int i=0; i < N; i++){
    h_a[i] = 1.0f;
    h_b[i] = 2.0f;
  }

  cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

  kernel<<<grid_size,block_size>>>(d_o, d_a, d_b, N);

  cudaMemcpy(h_o, d_o, bytes, cudaMemcpyDeviceToHost);

  for (int i=0; i < N; i++){
      assert(fabs(h_o[i] - h_a[i] - h_b[i]) < ERR_TOL);
  }
  std::cout << "Test passed" << std::endl;

  cudaFree(d_o);
  cudaFree(d_a);
  cudaFree(d_b);
  free(h_o);
  free(h_a);
  free(h_b);

  return 0;
}
