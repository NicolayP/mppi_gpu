#ifndef __CUDA_CLASS_H__
#define __CUDA_CLASS_H__

#define STEPS 200

class ModelGpu{
public:
  __host__ __device__ void advance();
  __host__ __device__ void set_x(int* x);
  __host__ __device__ void set_steps(int steps);
  __host__ __device__ int* get_x();
  __host__ __device__ int get_steps();
  __host__ __device__ ModelGpu();
  __host__ __device__ void init(int* x, int init);

private:
  int* x_;
  int steps_;
};

// TODO: add cudaError_t attribute to recorde the allocation
// and execution state of the device as reported by cuda.
class Model{
public:
  Model(int n, int steps);
  ~Model();
  void sim();
  void memcpy_set_data(int* x);
  void memcpy_get_data(int* x_all);
  //void set_steps(int steps);
  //int get_steps();
  //void set_nb_sim(int n);
  //int get_nb_sim();
private:
  int n_sim_;
  int steps_;
  size_t bytes_;
  int* d_x;
  // value to set up inital state vector.
  int* d_x_i;
  ModelGpu* d_models;

};

/*
 * Set of global function that the class Model will use to
 * run kernels.
 */
__global__ void sim_gpu_kernel_(ModelGpu* d_models, int n_);

__global__ void set_data_(ModelGpu* d_models,
                          int* d_x_i,
                          int* d_x,
                          int n_sim,
                            int steps);

#endif
