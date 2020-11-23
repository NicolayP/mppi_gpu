#ifndef __POINT_MASS_GPU_HPP__
#define __POINT_MASS_GPU_HPP__

#include <curand.h>
#include <curand_kernel.h>
#include "cost.hpp"

#include <stdlib.h>
#include <stdio.h>


/*
 * Simple Model Class that will be use to generate
 * Samples. Ultimatly this should be a pure virtual
 * Class with a template for the type of state and
 * for the timestep.
 */

class PointMassModelGpu {

public:
    __host__ __device__ PointMassModelGpu ();
    __host__ __device__ void init (float* x,
                                   float* init,
                                   float* u,
                                   float* e,
                                   int steps,
                                   float* x_gain,
                                   int x_size,
                                   float* u_gain,
                                   int u_size,
                                   float* w,
                                   float* goal,
                                   float lambda,
                                   int id,
                                   bool verbose=false);
    __host__ __device__ void step (curandState* state, int t);
    __host__ __device__ float run (curandState* state);
    __host__ __device__ void save_e ();
    __host__ __device__ void set_x (float* x);
    __host__ __device__ void set_state (float* x);
    __host__ __device__ void set_horizon (int horizon);
    __host__ __device__ float* get_state ();
    __host__ __device__ int get_horizon ();

private:
    // Current timestep
    int _t;

    /* Time horizon */
    int _tau;

    /* Timestep */
    float _dt;

    /* Action sequence shared with all the device classes. READ ONLY! */
    float* _u;

    /* Simulation state pointer.  */
    float* _x;

    /* State and action gain of the LTI point mass system. */
    float* _x_gain;
    int _x_size;

    float* _u_gain;
    int _u_size;

    /* Reference Goal. */
    float* _g;
    /* Weight for the cost function. */
    float* _w;

    /* Local copy of noise pointer. */
    float* _e;
    /* Pointer to noise in global memory. */
    float* glob_e;

    /* System noise */
    float* _s;
    float* _inv_s;

    /* Cost object */
    Cost _cost;

    /* contains cumulative cost. */
    float _c;

    /* Simulation indentifier, mostly used for debugging purposes. */
    int _id;

    /* Verbose indicator. */
    bool _verb;

};

#endif
