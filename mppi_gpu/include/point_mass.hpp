#ifndef __CUDA_CLASS_H__
#define __CUDA_CLASS_H__

#define STEPS 5
#define TOL 1e-6



/*
 * Simple Model Class that will be use to generate
 * Samples. Ultimatly this should be a pure virtual
 * Class with a template for the type of state and
 * for the timestep.
 */
class PointMassModelGpu{
public:
  __host__ __device__ PointMassModelGpu();
  __host__ __device__ void init(float* x,
                                float init,
                                float* u,
                                float* x_gain,
                                int x_size,
                                float* u_gain,
                                int u_size);
  __host__ __device__ void step();
  __host__ __device__ void run();
  __host__ __device__ void set_state(float* x);
  __host__ __device__ void set_horizon(int horizon);
  __host__ __device__ float* get_state();
  __host__ __device__ int get_horizon();

private:
  // Current timestep
  int t_;
  // Horizon
  int tau_;

  int dt_;
  // Action pointer.
  float* u_;
  // State pointer.
  float* x_;

  // LTI:
  float* x_gain_;
  float* u_gain_;
  int x_size_;
  int u_size_;
};

// TODO: add cudaError_t attribute to recorde the allocation
// and execution state of the device as reported by cuda.
class PointMassModel{
public:
  PointMassModel(int n, int steps, float dt);
  ~PointMassModel();
  void sim();
  void memcpy_set_data(float* x, float* u);
  void memcpy_get_data(float* x_all);
  //void set_steps(int steps);
  //int get_steps();
  //void set_nb_sim(int n);
  //int get_nb_sim();
private:
  int n_sim_;
  int steps_;
  size_t bytes_;
  float* d_x;
  float* d_u;
  // value to set up inital state vector.
  float* d_x_i;
  PointMassModelGpu* d_models;


  float* state_gain;
  int state_dim;
  float* act_gain;
  int act_dim;

  float dt_;
};

/*
 * Set of global function that the class Model will use to
 * run kernels.
 */
__global__ void sim_gpu_kernel_(PointMassModelGpu* d_models, int n_);

__global__ void set_data_(PointMassModelGpu* d_models,
                          float* d_x_i,
                          float* d_x,
                          float* d_u,
                          int n_sim,
                          int steps,
                          float* state_gain,
                          int state_dim,
                          float* act_gain,
                          int act_dim);

#endif
