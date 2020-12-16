#include "mppi_utils.hpp"

/* Set of printing function for the device. Should always run with one thread
   and one block.
 */

__global__ void print_x (float* x, int steps, int samples, int s_dim) {
    int id(0);
    for (int k = 0; k < samples; k++)
    {
        for (int j=0; j < steps; j++)
        {
            for (int i=0; i < s_dim; i++)
            {
                id = k*steps*s_dim + j*s_dim + i;
                printf("x[%d]: %f ", id, x[id]);
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("\n");
}

__global__ void print_u (float* u, int steps, int a_dim) {
    int id(0);
    for (int j=0; j < steps; j++)
    {
        for (int i=0; i < a_dim; i++)
        {
            id = j*a_dim + i;
            printf("u[%d]: %f ", id, u[id]);
        }
        printf("\n");
    }
    printf("\n");
}

__global__ void print_e (float* e, int steps, int samples, int a_dim) {
    int id(0);
    for (int k = 0; k < samples; k++)
    {
        for (int j=0; j < steps; j++)
        {
            for (int i=0; i < a_dim; i++)
            {
                id = k*steps*a_dim + j*a_dim + i;
                printf("e[%d]: %f ", id, e[id]);
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("\n");
}

__global__ void print_beta (float* beta, int size) {
    for (int i=0; i < size; i++)
    {
        printf("Beta[%d]: %f\n", i, beta[i]);
    }
}

__global__ void print_nabla (float* nabla, int size) {
    for (int i=0; i < size; i++)
    {
        printf("Nabla[%d]: %f\n", i, nabla[i]);
    }
}

__global__ void print_lam (float* lam, int size) {
    for (int i=0; i < size; i++)
    {
        printf("lambda[%d]: %f\n",i, lam[i]);
    }
}

__global__ void print_weights (float* weights, int samples) {
    for (int i=0; i < samples; i++)
    {
        printf("weights[%d]: %f\n", i, weights[i]);
    }
}

__global__ void print_costs (float* costs, int samples) {
    for (int i=0; i < samples; i++)
    {
        printf("costs[%d]: %f\n", i, costs[i]);
    }
}

__global__ void print_exp (float* exp, int samples) {
    for (int i=0; i<samples; i++)
    {
        printf("exp[%d]: %f\n", i, exp[i]);
    }
}

__global__ void print_sum_weights (float* w, int samples) {
    float sum(0);
    for (int i = 0; i < samples; i++) {
        sum += w[i];
    }
    printf("weight sum: %f\n", sum);
}
