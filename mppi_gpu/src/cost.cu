#include "cost.hpp"

__host__ __device__ Cost::Cost(float* w, inf w_size, float* goal, int goal_dim, float lambda, int u_dim)
{
    _w = w;
    _w_size = w_size;
    _goal = goal;
    _goal_dim = goal_dim;
    _lambda = lambda;
    _u_dim = u_dim;
    _inv_s = inv_s;
}

__host__ __device__ void Cost::init(float* w, inf w_size, float* goal, int goal_dim, float lambda, int u_dim)
{
    _w = w;
    _w_size = w_size;
    _goal = goal;
    _goal_dim = goal_dim;
    _lambda = lambda;
    _u_dim = u_dim;
    _inv_s = inv_s
}

__host__ __device__ float Cost::step_cost(float* x, float* u, float* e)
{
    float res(0);
    for(int i=0; i < _w_size; i++)
    {
        res += (x[i] - _goal[i])*w[i]*(x[i] - _goal[i]);
    }

    for(int i=0; i < _u_dim; i++)
    {
        res += u[i]*inv_s[i]*e[i];
    }

    return _lambda*res;
}

__host__ __device__ float Cost::final_cost(float* x)
{
    float res(0);
    for (int i=0; i < w_size; i++){
        res += (x[i] - _goal[i])*w[i]*(x[i] - _goal[i]);
    }
    return res;
}
