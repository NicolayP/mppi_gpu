#include "point_mass_gpu.hpp"
#include <stdio.h>
#include <stdlib.h>


__host__ __device__ PointMassModelGpu::PointMassModelGpu () {
    _x = nullptr;
    _u = nullptr;
    _e = nullptr;
    _tau = 0;
    _t = 1;
    _id = 0;
    /*
    // Local copy of the weight and the goal.
    float* w = (float*) malloc(sizeof(float)*x_size);
    float* g = (float*) malloc(sizeof(float)*x_size);
    for (int i = 0; i < x_size; i++)
    {
        w[i] = weight[i];
        g[i] = goal[i];
    }
    _cost = Cost(w, x_size, g, x_size, lambda);
    _c = 0;*/
}

__host__ __device__ void PointMassModelGpu::init (float* x,
                                                 float* init,
                                                 float* u,
                                                 float* e,
                                                 int steps,
                                                 float* x_gain,
                                                 int x_size,
                                                 float* u_gain,
                                                 int u_size,
                                                 float* weight,
                                                 float* goal,
                                                 float lambda,
                                                 int id,
                                                 bool verbose) {
    // TODO: cache x* in sm memory for faster access
    _x = x;
    _u = u;
    set_x(init);
    _tau = steps;
    // Point the gain pointers to the right address
    _x_gain = x_gain;
    _x_size = x_size;

    _u_gain = u_gain;
    _u_size = u_size;

    _t = 1;
    // Local copy of the weight and the goal.
    _w = (float*) malloc(sizeof(float)*_x_size);
    _g = (float*) malloc(sizeof(float)*_x_size);
    // local copy of the error for faster access
    _e = (float*) malloc(sizeof(float)*_tau*_u_size);
    glob_e = e;

    _inv_s = (float*) malloc(sizeof(float)*_u_size);
    _inv_s[0] = 1.0;
    _inv_s[1] = 1.0;

    for (int i = 0; i < _x_size; i++)
    {
        _w[i] = weight[i];
        _g[i] = goal[i];
    }

    for (int i = 0; i < _tau; i++){
        _e[i*_u_size + 0] = e[i*_u_size + 0];
        _e[i*_u_size + 1] = e[i*_u_size + 1];
    }
    _cost = Cost(_w, _x_size, _g, _x_size, lambda, _inv_s, u_size);
    _c = 0;
    _id = id;

    _verb = verbose;
    return;
}

__host__ __device__ void PointMassModelGpu::step (curandState* state, int t) {

#ifdef __CUDA_ARCH__
    for (int i=0; i < _u_size; i++) {
        _e[(t)*_u_size + i] = 0.025* curand_normal(state);
        if (_verb) {
            printf("id: %d, _e[%d]: %f\n", _id, (t)*_u_size + i, _e[(t)*_u_size + i]);
        }
    }

#else
    _e[(t)*_u_size] += 0; //cpu random uniform;
    _e[(t)*_u_size + 1] += 0; //cpu random uniform;
#endif

    for(int i=0; i < 2; i++){
        _x[(t+1)*_x_size+i] = _x_gain[0]*_x[(t)*_x_size+i] +
        _x_gain[1]*_x[(t)*_x_size+i+2] +
        _u_gain[0]*(_u[(t)*_u_size + i] + _e[(t)*_u_size + i]);

        _x[(t+1)*_x_size+i+2] = _x_gain[2]*_x[(t)*_x_size+i] +
        _x_gain[3]*_x[(t)*_x_size+i+2] +
        _u_gain[1]*(_u[(t)*_u_size + i] + _e[(t)*_u_size + i]);
    }
    _c += _cost.step_cost(&_x[(t+1)*_x_size], &_u[(t)*_u_size], &_e[(t)*_u_size], _id, t);
    //printf("_c[%d]: %f\n", _id, _c);
}

__host__ __device__ float PointMassModelGpu::run (curandState* state) {
    for (int t = 0; t < _tau; t++ ){
        step(state, t);
    }
    _c += _cost.final_cost(&_x[(_tau-1)*_x_size], _id);
    //printf("_c[%d, %d]: %f\n", (_tau-1), _id, _c);
    // save action to global pointer
    save_e();
    return _c;
}

__host__ __device__ void PointMassModelGpu::save_e () {
    for (int t = 0; t < _tau; t++)
    {
        glob_e[(t)*_u_size] = _e[(t)*_u_size];
        glob_e[(t)*_u_size + 1] = _e[(t)*_u_size + 1];
    }
}

__host__ __device__ void PointMassModelGpu::set_state (float* x) {
    _x = x;
}

__host__ __device__ void PointMassModelGpu::set_horizon (int horizon) {
    /*
    * DO NOT USE ATM, when steps change, we need to update
    * the pointer x for the extra allocate space. As all the data
    * is represented in a continous array failling to do so will
    * produce a seg fault and probably leave the memory in a inconsistant
    * state.
    */
    _tau = horizon;
}

__host__ __device__ void PointMassModelGpu::set_x (float* x) {
    for(int i =0; i < _x_size; i++)
    {
        _x[i] = x[i];
    }
}

__host__ __device__ float* PointMassModelGpu::get_state () { return _x;}

__host__ __device__ int PointMassModelGpu::get_horizon () { return _tau;}
