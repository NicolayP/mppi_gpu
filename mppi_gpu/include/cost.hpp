#ifndef __COST_CLASS_HPP_
#define __COST_CLASS_HPP_

class Cost{
public:
    __host__ __device__ Cost(float* w, inf w_size, float* goal, int goal_dim, float lambda);
    __host__ __device__ void init(float* w, inf w_size, float* goal, int goal_dim, float lambda);
    __host__ __device__ float step_cost(float* x, float* u);
    __host__ __device__ float final_cost(float* x);
private:
    float* _w;
    int _w_size;
    float* goal;
    int _goal_dim;
    float _lambda;
    /* inverse of sigma */
    float* _inv_s;
}

#endif
