#ifndef __CUDA_CLASS_HPP__
#define __CUDA_CLASS_HPP__

#include <curand.h>
#include <curand_kernel.h>
#include "cost.hpp"

#include <stdlib.h>
#include <stdio.h>

#include <iostream>
#include <fstream>


#define STEPS 200
#define TOL 1e-6

// Called inside constructor
/*
#define CUDA_CALL_CONST(x) do { if((x) != cudaSuccess) {\
    printf("Error at %s %d\n",__FILE__, __LINE__);\
    }} while(0)
*/
#define CUDA_CALL(x) do { if((x) != cudaSuccess) {\
    printf("Error at %s %d\n",__FILE__, __LINE__);\
    return EXIT_FAILURE}} while(0)


#define CUDA_CALL_CONST(x) do {\
    cudaError_t err = (x);\
    if(err != cudaSuccess) {\
        printf("API error failed %s:%d Returned: %d\n", \
        __FILE__, __LINE__, err);\
    exit(1);\
}} while(0)

/*
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
void gpuAssert(cudaError_t code, char* file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) getchar();
   }
}*/


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
                                   float lambda,
                                   int id);
     __host__ __device__ void step(curandState* state);
     __host__ __device__ float run(curandState* state);
     __host__ __device__ void save_e();
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

     // local memory to e.
     float* _e;
     // pointer to global memory of e.
     float* glob_e;

     //sigma and its inverse.
     float* _s;
     float* _inv_s;

     // Cost object
     Cost _cost;
     // contains cumulative cost.
     float _c;
     int _id;

 };

 // TODO: add cudaError_t attribute to recorde the allocation
 // and execution state of the device as reported by cuda.
 class PointMassModel{
 public:
     PointMassModel(int nb_sim, int steps, float dt);
     ~PointMassModel();
     void sim();
     void memcpy_set_data(float* x, float* u, float* goal, float* w);
     void memcpy_get_data(float* x_all, float* e);
     void get_inf(float* x, float* u, float* e, float* cost, float* beta, float* nabla, float* weight);
     void exp();
     void min_beta();
     void nabla();
     void weights();
     void update_act();
     void update_act_id();
     //void set_steps(int steps);
     //int get_steps();
     //void set_nb_sim(int n);
     //int get_nb_sim();
 private:
     int n_sim_;
     int steps_;
     int bytes_;

     float* d_x;
     float* d_u;
     float* d_u_swap;
     float* d_e;

     // value to set up inital state vector.
     float* d_x_i;
     float* d_cost;

     float* d_exp;

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
     float* h_w;

     float* state_gain;
     int state_dim;
     float* act_gain;
     int act_dim;

     float _dt;

     curandState* rng_states;
 };

 /*
 * Set of global function that the class Model will use to
 * run kernels.
 */
__global__ void sim_gpu_kernel_(PointMassModelGpu* d_models,
     int n_,
     float* d_u,
     float* d_cost,
     curandState* rng_states);

__global__ void exp_red(float* out, float* cost, float* lambda, float* beta, float* nabla, int size);

__global__ void min_red(float* v, float* beta, int n);

__global__ void sum_red_exp(float* v, float* lambda, float* beta, float* v_r, int n);

__global__ void sum_red(float* v, float* v_r, int n);

__global__ void weights_kernel(float* v, float* v_r, float* lambda_1, float* beta, float* nabla_1, int n);

__global__ void copy_act(float* u, float* tmp, int t, int act_dim);

__global__ void update_act_kernel(float* u,
                                  float* weights,
                                  float* e,
                                  int steps,
                                  int t,
                                  int act_dim,
                                  int n);

__global__ void set_data_(PointMassModelGpu* d_models,
                          float* d_x_i,
                          float* d_x,
                          float* d_u,
                          float* d_e,
                          int n_sim,
                          int steps,
                          float* state_gain,
                          int state_dim,
                          float* act_gain,
                          int act_dim,
                          curandState* rng_states,
                          float* goal,
                          float* w,
                          float* lambda);

__global__ void print_x(float* x, int steps, int samples, int s_dim);

__global__ void print_u(float* u, int steps, int a_dim);

__global__ void print_e(float* e, int steps, int samples, int a_dim);

__global__ void print_beta(float* beta, int size);

__global__ void print_nabla(float* nabla, int size);

__global__ void print_lam(float* lamb, int size);

__global__ void print_weights(float* weights, int samples);

__global__ void print_costs(float* costs, int samples);

__global__ void print_exp(float* exp, int samples);

__global__ void shift_act(float* u, float* u_swap, int a_dim, int samples);

__global__ void update_act_id_kernel(int steps, int t, int a_dim, int samples);



#endif
