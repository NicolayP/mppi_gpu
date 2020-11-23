#ifndef __MPPI_UTILS_HPP__
#define __MPPI_UTILS_HPP__

#include <stdlib.h>
#include <stdio.h>

/* Set of printing function for the device. Should always run with one thread
   and one block.
 */

__global__ void print_x (float* x, int steps, int samples, int s_dim);

__global__ void print_u (float* u, int steps, int a_dim);

__global__ void print_e (float* e, int steps, int samples, int a_dim);

__global__ void print_beta (float* beta, int size);

__global__ void print_nabla (float* nabla, int size);

__global__ void print_lam (float* lamb, int size);

__global__ void print_weights (float* weights, int samples);

__global__ void print_costs (float* costs, int samples);

__global__ void print_exp (float* exp, int samples);

__global__ void print_sum_weights (float* w, int samples);

#endif
