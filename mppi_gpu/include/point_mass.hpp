#ifndef __CUDA_CLASS_HPP__
#define __CUDA_CLASS_HPP__

#include <curand.h>
#include <curand_kernel.h>
#include "cost.hpp"


#define STEPS 200
#define TOL 1e-6

// Called inside constructor
#define CUDA_CALL_CONST(x) do { if((x) != cudaSuccess) {\
    printf("Error at %s %d\n",__FILE__, __LINE__);\
    }} while(0)

#define CUDA_CALL(x) do { if((x) != cudaSuccess) {\
    printf("Error at %s %d\n",__FILE__, __LINE__);\
    return EXIT_FAILURE}} while(0)



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
                                   float* e,
                                   float* x_gain,
                                   int x_size,
                                   float* u_gain,
                                   int u_size,
                                   float* w,
                                   float* goal,
                                   float lambda);
     __host__ __device__ void step(curandState* state);
     __host__ __device__ float run(curandState* state);
     __host__ __device__ void set_state(float* x);
     __host__ __device__ void set_horizon(int horizon);
     __host__ __device__ float* get_state();
     __host__ __device__ int get_horizon();

 private:
     // Current timestep
     int _t;
     // Horizon
     int _tau;

     float _dt;
     // Action pointer.
     float* _u;
     // State pointer.
     float* _x;

     // LTI:
     float* _x_gain;
     float* _u_gain;
     int _x_size;
     int _u_size;

     float* _g;
     float* _w;

     float* _e;

     //sigma and its inverse.
     float* _s;
     float* _inv_s;

     // Cost object
     Cost _cost;
     // contains cumulative cost.
     float _c;

 };

 // TODO: add cudaError_t attribute to recorde the allocation
 // and execution state of the device as reported by cuda.
 class PointMassModel{
 public:
     PointMassModel(size_t nb_sim, size_t steps, float dt);
     ~PointMassModel();
     void sim();
     void memcpy_set_data(float* x, float* u, float* goal, float* w);
     void memcpy_get_data(float* x_all, float* e);
     void min_beta();
     void nabla();
     void weights();
     //void set_steps(int steps);
     //int get_steps();
     //void set_nb_sim(int n);
     //int get_nb_sim();
 private:
     size_t n_sim_;
     size_t steps_;
     size_t bytes_;

     float* d_x;
     float* d_u;
     float* d_e;
     // value to set up inital state vector.
     float* d_x_i;
     float* d_cost;

     float* d_beta;
     float* _d_beta;

     float* d_nabla;
     float* _d_nabla;

     float* d_lambda;
     float* d_weights;

     PointMassModelGpu* d_models;

     /* Goal vector passed to the cost function */
     float* d_g;
     /* Weight vector */
     float* d_w;

     float* state_gain;
     size_t state_dim;
     float* act_gain;
     size_t act_dim;

     float _dt;

     curandState* rng_states;
 };

 /*
 * Set of global function that the class Model will use to
 * run kernels.
 */
 __global__ void sim_gpu_kernel_(PointMassModelGpu* d_models,
     size_t n_,
     float* d_u,
     float* d_cost,
     curandState* rng_states);

 __global__ void min_red(float* v, float* beta, int n);

 __global__ void sum_red_exp(float* v, float* lambda, float* beta, float* v_r, int n);

__global__ void sum_red(float* v, float* v_r, int n);

__global__ void weights_kernel(float* v, float* v_r, float* lambda_1, float* beta, float* nabla_1, size_t n);

__global__ void set_data_(PointMassModelGpu* d_models,
                          float* d_x_i,
                          float* d_x,
                          float* d_u,
                          float* d_e,
                          size_t n_sim,
                          size_t steps,
                          float* state_gain,
                          size_t state_dim,
                          float* act_gain,
                          size_t act_dim,
                          curandState* rng_states,
                          float* goal,
                          float* w,
                          float* lambda);

#endif
