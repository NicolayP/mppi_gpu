#ifndef __CUDA_CLASS_H__
#define __CUDA_CLASS_H__

#define STEPS 10

class ModelGpu{
public:
  __host__ __device__ void advance();
  __host__ __device__ void setX(int x);
  __host__ __device__ void setN(int n);
  __host__ __device__ int getX();
  __host__ __device__ int getN();
  __host__ __device__ ModelGpu();
  __host__ __device__ void init(int x);

private:
  int x_;
  int n_;
};

class Model{
public:
  Model(int n);
  ~Model();
  void sim();
  void memcpy_set_data(int* x);
  void memcpy_get_data(int* x);
private:
  int n_;
  size_t bytes_;
  int* d_x;
  ModelGpu* d_models;

};

#endif
