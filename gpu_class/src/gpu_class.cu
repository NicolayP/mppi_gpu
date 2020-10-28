#include "gpu_class.hpp"
#include <iostream>
/*__global__ void kernel(Model* model, int n){
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  model->sim_gpu<<< 1 + n/256, 256>>>();
  cudaDeviceSynchronize();
}*/

__global__ void sim_gpu_kernel(ModelGpu* d_models, int* x_, int n_);

__global__ void set_data(ModelGpu* d_models, int* x, int n_);


__host__ __device__ ModelGpu::ModelGpu(){
  x_ = 0;
  n_ = STEPS;
}

__host__ __device__ void ModelGpu::advance(){
  for (int i=0; i < n_; i++){
    x_++;
  }
}

__host__ __device__ void ModelGpu::init(int x){
  x_ = x;
  n_ = STEPS;
  return;
}

__host__ __device__ void ModelGpu::setX(int x){
  x_ = x;
}

__host__ __device__ void ModelGpu::setN(int n){
  n_ = n;
}

__host__ __device__ int ModelGpu::getX(){ return x_;}

__host__ __device__ int ModelGpu::getN(){ return n_;}


Model::Model(int n){
  n_ = n;
  bytes_ = sizeof(int)*n_;
  cudaMalloc((void**)&d_models, sizeof(ModelGpu)*n_);
  cudaMalloc((void**)&d_x, bytes_);
  cudaDeviceSynchronize();
}

Model::~Model(){
  cudaFree((void*) d_x);
  cudaFree((void*) d_models);
}

void Model::sim(){
  //sim_gpu_kernel<<< 1 + n_/256, 256>>>(d_models, d_x, n_);
  sim_gpu_kernel<<<1 + n_/256, 256>>>(d_models, d_x, n_);
  cudaDeviceSynchronize();
}

void Model::memcpy_set_data(int* x){
  cudaMemcpy(d_x, x, bytes_, cudaMemcpyHostToDevice);
  std::cout << "Setting data after cuda memcyp... : ";
  set_data<<<1 + n_/256, 256>>>(d_models, d_x, n_);
  std::cout << "Done" << std::endl;
  cudaDeviceSynchronize();
}

void Model::memcpy_get_data(int *x){
  cudaMemcpy(x, d_x, bytes_, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
}

__global__ void sim_gpu_kernel(ModelGpu* d_models, int* x_, int n_){
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if(tid < n_){
    d_models[tid].advance();
    x_[tid] = d_models[tid].getX();
  }
}

__global__ void set_data(ModelGpu* d_models, int* d_x, int n_){
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if(tid < n_){
    d_models[tid].init(d_x[tid]);
  }
}
