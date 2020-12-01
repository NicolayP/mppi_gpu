#ifndef __CUDA_CLASS_HPP__
#define __CUDA_CLASS_HPP__

#include <curand.h>
#include <curand_kernel.h>
#include "cost.hpp"
#include "point_mass_gpu.hpp"

#include <stdlib.h>
#include <stdio.h>

#include <iostream>
#include <fstream>


#define TOL 1e-6




 // TODO: add cudaError_t attribute to recorde the allocation
 // and execution state of the device as reported by cuda.
 class PointMassModel{
 public:
     PointMassModel (int nb_sim,
                     int steps,
                     float dt,
                     int state_dim,
                     int act_dim,
                     bool verbose=false);
     ~PointMassModel ();
     void get_act (float* next_act);
     void memcpy_set_data (float* x, float* u, float* goal, float* w);
     void get_x (float* x);
     void memcpy_get_data (float* x_all, float* e);
     void get_inf (float* x,
                   float* u,
                   float* e,
                   float* cost,
                   float* beta,
                   float* nabla,
                   float* weight);
     void set_x (float* x);
     void get_u (float* u);

 private:
     void sim ();
     void exp ();
     void beta ();
     void nabla ();
     void weights ();
     void update_act ();

     int _n_sim;
     int _steps;
     int _bytes;
     /* State and action dimensions. */
     int _state_dim;
     int _act_dim;

     /* Pointer to state in global memory. */
     float* _x;

     /* Array to initial state.  */
     float* _x_i;

     /* Pointer to action sequence in global memory. */
     float* _u;
     /* buffer used when shifting action. */
     float* _u_swap;

     /* Pointer to noise in global memory. */
     float* _e;

     /* Costs array of every simulation. */
     float* _cost;

     /* Array to exponenetial of $$ -\frac{1}{\lambda} (S(\Epsilon^k) - \Beta) $$
      * Mostly used to debug and verify solutions.
      */
     float* _exp;

     /* Pointer to beta, only beta[0] is filled with a value */
     float* _beta;

     /* Pointer to nabla, only nable[0] is filled with a value */
     float* _nabla;

     /* Pointer to lambda. */
     float* _lambda;

     /* Pointer to the weights of each path. */
     float* _weights;

     /* Pointer to the simulations. */
     PointMassModelGpu* _models;

     /* Goal vector passed to the cost function. */
     float* _g;

     /* Weight vector for the cost function. */
     float* _w;

     /* State and action gain for LTI system. */
     float* _state_gain;
     float* _act_gain;

     /* Timestep for integration. */
     float _dt;

     /* Pointer to the random generator state. One for each simulation. */
     curandState* _rng_states;

     /* Verbosity variable. */
     bool _verb;
 };

 /*
 * Set of global function that the class Model will use to
 * run kernels.
 */
__global__ void sim_gpu_kernel_ (PointMassModelGpu* models,
                                 int n,
                                 float* u,
                                 float* cost,
                                 curandState* rng_states);

__global__ void exp_red (float* out,
                         float* cost,
                         float* lambda,
                         float* beta,
                         int size);

__global__ void min_red (float* v, float* beta, int n);

__global__ void sum_red_exp (float* v,
                             float* lambda,
                             float* beta,
                             float* v_r,
                             int n);

__global__ void sum_red (float* v, float* v_r, int n);

__global__ void weights_kernel (float* v,
                                float* v_r,
                                float* lambda_1,
                                float* beta,
                                float* nabla_1,
                                int n);

__global__ void copy_act (float* u, float* tmp, int t, int act_dim);

__global__ void update_act_kernel (float* u,
                                   float* weights,
                                   float* e,
                                   int steps,
                                   int t,
                                   int act_dim,
                                   int n);

__global__ void set_data (PointMassModelGpu* models,
                          float* x_i,
                          float* x,
                          float* u,
                          float* e,
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

__global__ void shift_act (float* u, float* u_swap, int a_dim, int samples);

__global__ void set_x_kernel (PointMassModelGpu* models, float* x_i, int n);

__global__ void sum_red_adim (float* v, float* v_r, int n, int a_dim);


#endif
