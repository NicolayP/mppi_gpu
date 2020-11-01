#include "cost.hpp"


__host__ __device__ Cost::Cost(){

}

__host__ __device__ Cost::Cost(float* w,
                               size_t w_size,
                               float* goal,
                               size_t goal_size,
                               float lambda,
                               float* inv_s,
                               size_t u_size)
{
    _w = w;
    _w_size = w_size;
    _goal = goal;
    _goal_size = goal_size;
    _lambda = lambda;
    _u_size = u_size;
    _inv_s = inv_s;
}

__host__ __device__ void Cost::init(float* w,
                                    size_t w_size,
                                    float* goal,
                                    size_t goal_size,
                                    float lambda,
                                    float* inv_s,
                                    size_t u_size)
{
    _w = w;
    _w_size = w_size;
    _goal = goal;
    _goal_size = goal_size;
    _lambda = lambda;
    _u_size = u_size;
    _inv_s = inv_s;
}

__host__ __device__ float Cost::step_cost(float* x, float* u, float* e)
{
    float res(0);
    for(int i=0; i < _w_size; i++)
    {
        res += (x[i] - _goal[i])*_w[i]*(x[i] - _goal[i]);
    }

    for(int i=0; i < _u_size; i++)
    {
        res += u[i]*_inv_s[i]*e[i];
    }

    return _lambda*res;
}

__host__ __device__ float Cost::final_cost(float* x)
{
    float res(0);
    for (int i=0; i < _w_size; i++){
        res += (x[i] - _goal[i])*_w[i]*(x[i] - _goal[i]);
    }
    return res;
}
