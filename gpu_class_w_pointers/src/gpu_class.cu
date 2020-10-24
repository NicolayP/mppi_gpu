#include "gpu_class.hpp"
#include <iostream>
/*__global__ void kernel(Model* model, int n){
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  model->sim_gpu<<< 1 + n/256, 256>>>();
  cudaDeviceSynchronize();
}*/


__host__ __device__ ModelGpu::ModelGpu(){
  x_ = 0;
  steps_ = STEPS;
}

__host__ __device__ void ModelGpu::advance(){
  for (int i=1; i < steps_; i++){
    x_[i] = x_[i-1] + 1;
  }
}

__host__ __device__ void ModelGpu::init(int* x, int init){
  x_ = x;
  x_[0] = init;
  steps_ = STEPS;
  return;
}

__host__ __device__ void ModelGpu::set_x(int* x){
  x_ = x;
}

__host__ __device__ void ModelGpu::set_steps(int steps){
  /*
   * DO NOT USE ATM, when steps change, we need to update
   * the pointer x for the extra allocate space. As all the data
   * is represented in a continous array failling to do so will
   * produce a seg fault and probably leave the memory in a inconsistant
   * state.
   */
  steps_ = steps;
}

__host__ __device__ int* ModelGpu::get_x(){ return x_;}

__host__ __device__ int ModelGpu::get_steps(){ return steps_;}


Model::Model(int nb_sim, int steps){
  n_sim_ = nb_sim;
  steps_ = steps;

  /*
   * just for convinience, ultimatly replace with a
   * template type associated with the class wich will
   * represent the mppi domain.
   */
  bytes_ = sizeof(int)*steps_*n_sim_;

  // *Allocate the data on tahe GPU.*

  // allocate space for all our simulation objects.
  cudaMalloc((void**)&d_models, sizeof(ModelGpu)*n_sim_);
  // allocate space for the init_state array. int* x[n_sim]
  cudaMalloc((void**)&d_x_i, sizeof(int)*n_sim_);
  // allocate data space, continous in memeory so int* x[n_sim*steps_]
  cudaMalloc((void**)&d_x, sizeof(int)*n_sim_*steps_);
  // set the memory with 0s.
  cudaMemset((void*)d_x, 0, sizeof(int)*n_sim_*steps_);
  cudaDeviceSynchronize();
  std::cout << "Simulation objects created" << std::endl;
}

Model::~Model(){
  cudaFree(d_x);
  cudaFree(d_x_i);
  cudaFree(d_models);
}

void Model::sim(){
  // launch 1 thread per simulation. Can later consider to
  // add dimensions to the block and thread of the kernel
  // to // enven more the code inside the simulation.
  // using blockDim.y & blockDim.z, blockIdx.y & blockIdx.x
  // and threadIdx.y & threadIdx.z.
  std::cout << "Starting simulations..." << std::endl;
  sim_gpu_kernel_<<<1 + n_sim_/256, 256>>>(d_models, n_sim_);
  std::cout << "simulations finished!" << std::endl;
  cudaDeviceSynchronize();
}

void Model::memcpy_set_data(int* x){
  cudaMemcpy(d_x_i, x, sizeof(int)*n_sim_, cudaMemcpyHostToDevice);
  std::cout << "Setting inital state of the sims... : ";
  set_data_<<<1 + n_sim_/256, 256>>>(d_models, d_x_i, d_x, n_sim_, steps_);
  std::cout << "Done" << std::endl;
  cudaDeviceSynchronize();
}

void Model::memcpy_get_data(int* x_all){
  cudaMemcpy(x_all, d_x, bytes_, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
}

__global__ void sim_gpu_kernel_(ModelGpu* d_models, int n_sim_){
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if(tid < n_sim_){
    d_models[tid].advance();
  }
}

__global__ void set_data_(ModelGpu* d_models,
                          int* d_x_i,
                          int* d_x,
                          int n_sim,
                          int steps){
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if(tid < n_sim){
    d_models[tid].init(&d_x[tid*steps], d_x_i[tid]);
  }
}
