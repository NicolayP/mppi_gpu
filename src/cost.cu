#include "cost.hpp"
#include <stdio.h>
#include <stdlib.h>


__host__ __device__ Cost::Cost () {

}

__host__ __device__ Cost::Cost (float* w,
                               int w_size,
                               float* goal,
                               int goal_size,
                               float lambda,
                               float* inv_s,
                               int u_size) {
    _w = w;
    _w_size = w_size;
    _goal = goal;
    _goal_size = goal_size;
    _lambda = lambda;
    _u_size = u_size;
    _inv_s = inv_s;
}

__host__ __device__ void Cost::init (float* w,
                                    int w_size,
                                    float* goal,
                                    int goal_size,
                                    float lambda,
                                    float* inv_s,
                                    int u_size) {
    _w = w;
    _w_size = w_size;
    _goal = goal;
    _goal_size = goal_size;
    _lambda = lambda;
    _u_size = u_size;
    _inv_s = inv_s;
}

__host__ __device__ float Cost::step_cost (float* x, float* u, float* e, int id, int t) {
    float res(0);

    for (int i=0; i < _u_size; i++) {
        res += u[i]*_inv_s[i]*e[i];
    }
    res *= _lambda;

    for (int i=0; i < _w_size; i++) {
        res += (x[i] - _goal[i])*_w[i]*(x[i] - _goal[i]);
    }

    return res;
}

__host__ __device__ float Cost::final_cost (float* x, int id) {
    float res(0);

    for (int i=0; i < _w_size; i++) {
        res += (x[i] - _goal[i])*_w[i]*(x[i] - _goal[i]);
    }
    return res;
}
