#ifndef __COST_CLASS_HPP_
#define __COST_CLASS_HPP_

class Cost{
public:
    __host__ __device__ Cost();

    __host__ __device__ Cost(float* w,
                             int w_size,
                             float* goal,
                             int goal_size,
                             float lambda,
                             float* inv_s,
                             int u_size);

    __host__ __device__ void init(float* w,
                                  int w_size,
                                  float* goal,
                                  int goal_size,
                                  float lambda,
                                  float* inv_s,
                                  int u_size);

    __host__ __device__ float step_cost(float* x, float* u, float* e, int id, int t);
    __host__ __device__ float final_cost(float* x, int id);
private:
    float* _w;
    int _w_size;
    float* _goal;
    int _goal_size;
    float _lambda;
    /* inverse of sigma */
    float* _inv_s;
    int _u_size;
};

#endif
