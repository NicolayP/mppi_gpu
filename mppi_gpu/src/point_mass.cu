#include "point_mass.hpp"

#include <iostream>


__host__ __device__ PointMassModelGpu::PointMassModelGpu(){
    _x = nullptr;
    _u = nullptr;
    _e = nullptr;
    _tau = STEPS;
    _t = 1;
    float lambda = 1.0;
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

__host__ __device__ void PointMassModelGpu::init(float* x,
                                                 float init,
                                                 float* u,
                                                 float* e,
                                                 float* x_gain,
                                                 int x_size,
                                                 float* u_gain,
                                                 int u_size,
                                                 float* weight,
                                                 float* goal){
    // TODO: cache x* in sm memory for faster access
    _x = x;
    _u = u;
    _x[0] = init;
    _tau = STEPS;
    // Point the gain pointers to the right address
    _x_gain = x_gain;
    _x_size = x_size;

    _u_gain = u_gain;
    _u_size = u_size;

    _t = 1;
    float lambda = 1.0;
    // Local copy of the weight and the goal.
    _w = (float*) malloc(sizeof(float)*_x_size);
    _g = (float*) malloc(sizeof(float)*_x_size);
    // local copy of the error for faster access
    _e = (float*) malloc(sizeof(float)*STEPS*_u_size);

    _inv_s = (float*) malloc(sizeof(float)*_u_size);
    _inv_s[0] = 1.0;
    _inv_s[1] = 1.0;

    for (int i = 0; i < _x_size; i++)
    {
        _w[i] = weight[i];
        _g[i] = goal[i];
    }
    for (int i = 0; i < STEPS; i++){
        _e[i*_u_size + 0] = e[i*_u_size + 0];
        _e[i*_u_size + 1] = e[i*_u_size + 1];
    }
    _cost = Cost::Cost(_w, _x_size, _g, _x_size, lambda, _inv_s, u_size);
    _c = 0;
    return;
}

__host__ __device__ void PointMassModelGpu::step(curandState* state){

#ifdef __CUDA_ARCH__
    _e[(_t-1)*_u_size] += curand_normal(state);
    _e[(_t-1)*_u_size + 1] += curand_normal(state);
#else
    _e[(_t-1)*_u_size] += 0; //cpu random uniform;
    _e[(_t-1)*_u_size + 1] += 0; //cpu random uniform;
#endif
    for(int i=0; i < 2; i++){
        _x_[_t*_x_size+i] = _x_gain[0]*_x_[(_t-1)*_x_size+i] +
        _x_gain[1]*_x[(_t-1)*_x_size+i+2] +
        _u_gain[0]*(_u[(_t-1)*_u_size + i] + _e[(_t-1)*_u_size + i]);

        _x[_t*_x_size+i+2] = _x_gain[2]*_x[(_t-1)*_x_size+i] +
        _x_gain[3]*_x[(_t-1)*_x_size+i+2] +
        _u_gain[1]*(_u[(_t-1)*_u_size + i] + _e[(_t-1)*_u_size + i]);
    }
    _c += _cost.step_cost(&_x[_t*_x_size], &_u[(_t-1)*_u_size], &_e[(_t-1)*_u_size]);
}

__host__ __device__ void PointMassModelGpu::run(curandState* state){
    for (_t = 1; _t < _tau; _t++ ){
        step(state);
    }
    _c += _cost.final_cost(&_x[_t*_x_size]);
    printf("%f\n", _c);
}

__host__ __device__ void PointMassModelGpu::set_state(float* x){
    _x = x;
}

__host__ __device__ void PointMassModelGpu::set_horizon(int horizon){
    /*
    * DO NOT USE ATM, when steps change, we need to update
    * the pointer x for the extra allocate space. As all the data
    * is represented in a continous array failling to do so will
    * produce a seg fault and probably leave the memory in a inconsistant
    * state.
    */
    _tau = horizon;
}

__host__ __device__ float* PointMassModelGpu::get_state(){ return _x;}

__host__ __device__ int PointMassModelGpu::get_horizon(){ return _tau;}


PointMassModel::PointMassModel(size_t nb_sim, size_t steps, float dt){
    n_sim_ = nb_sim;
    steps_ = steps;
    act_dim = 2;
    state_dim = 4;

    _dt = dt;


    /*
    * just for convinience, ultimatly replace with a
    * template type associated with the class wich will
    * represent the mppi domain.
    */
    bytes_ = sizeof(float)*steps_*n_sim_*state_dim;

    //host data used to send data to memory.
    float state_[4];
    float act_[2];

    act_[0] = _dt*_dt/2.0;
    act_[1] = _dt;
    state_[0] = 1;
    state_[1] = _dt;
    state_[2] = 0;
    state_[3] = 1;

    // *Allocate the data on tahe GPU.*
    std::cout << "Allocating Space... : " << std::flush;
    // allocate space for all our simulation objects.
    cudaMalloc((void**)&d_models, sizeof(PointMassModelGpu)*n_sim_);
    // allocate space for the init_state array. int* x[n_sim]
    cudaMalloc((void**)&d_x_i, sizeof(float)*n_sim_*state_dim);
    // allocate data space, continous in memeory so int* x[n_sim*steps_]
    cudaMalloc((void**)&d_x, sizeof(float)*n_sim_*steps_*state_dim);
    // set the memory with 0s.
    cudaMemset((void*)d_x, 0, sizeof(float)*n_sim_*steps_*state_dim);
    // allocate space for action.
    cudaMalloc((void**)&d_e, sizeof(float)*n_sim_*steps_*act_dim);

    cudaMemset((void*)d_e, 0, sizeof(float)*n_sim_*steps_*act_dim);

    cudaMalloc((void**)&d_u, sizeof(float)*steps_*act_dim);

    // Set gain memory
    cudaMalloc((void**)&state_gain, sizeof(float)*state_dim);
    cudaMalloc((void**)&act_gain, sizeof(float)*act_dim);

    cudaMemcpy(state_gain, state_, sizeof(float)*state_dim, cudaMemcpyHostToDevice);
    cudaMemcpy(act_gain, act_, sizeof(float)*act_dim, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&rng_states, sizeof(curandState_t)*n_sim_);

    cudaMalloc((void**)&d_g, sizeof(float)*state_dim);
    cudaMalloc((void**)&d_w, sizeof(float)*state_dim);

    cudaDeviceSynchronize();
    std::cout << "Done" << std::endl;
}

PointMassModel::~PointMassModel(){
    cudaFree((void*)d_x);
    cudaFree((void*)d_x_i);
    cudaFree((void*)d_u);
    cudaFree((void*)d_e);
    cudaFree((void*)state_gain);
    cudaFree((void*)act_gain);
    cudaFree((void*)d_models);
    cudaFree((void*)rng_states);
}

void PointMassModel::sim(){
    // launch 1 thread per simulation. Can later consider to
    // add dimensions to the block and thread of the kernel
    // to // enven more the code inside the simulation.
    // using blockDim.y & blockDim.z, blockIdx.y & blockIdx.x
    // and threadIdx.y & threadIdx.z.
    std::cout << "Starting simulations... : " << std::flush;
    sim_gpu_kernel_<<<1 + n_sim_/256, 256>>>(d_models, n_sim_, d_e, rng_states);
    std::cout << "simulations finished!" << std::endl;
    cudaDeviceSynchronize();
}

void PointMassModel::memcpy_set_data(float* x, float* u, float* goal, float* w){
    cudaMemcpy(d_x_i, x, sizeof(float)*n_sim_*state_dim, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, u, sizeof(float)*n_sim_*act_dim*steps_, cudaMemcpyHostToDevice);
    cudaMemcpy(d_g, goal, sizeof(float)*state_dim, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, w, sizeof(float)*state_dim, cudaMemcpyHostToDevice);
    std::cout << "Setting inital state of the sims... : " << std::flush;
    set_data_<<<1 + n_sim_/256, 256>>>(d_models,
                                        d_x_i,
                                        d_x,
                                        d_u,
                                        d_e,
                                        n_sim_,
                                        steps_,
                                        state_gain,
                                        state_dim,
                                        act_gain,
                                        act_dim,
                                        rng_states,
                                        d_g,
                                        d_w);
    std::cout << "Done" << std::endl;
    cudaDeviceSynchronize();
}

void PointMassModel::memcpy_get_data(float* x_all, float* e){
    cudaMemcpy(x_all, d_x, bytes_, cudaMemcpyDeviceToHost);
    cudaMemcpy(e, d_e, sizeof(float)*n_sim_*act_dim*steps_, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}

__global__ void sim_gpu_kernel_(PointMassModelGpu* d_models,
                                size_t n_sim_,
                                float* d_u,
                                curandState* rng_states)
{

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < n_sim_){
        /* local copy of rng state for faster generation. */
        curandState localState = rng_states[tid];
        d_models[tid].run(&localState);
        /* copy back the state ohterwise the rng state is not working. */
        rng_states[tid] = localState;
        /*
        for (int i = 0; i < STEPS; i++){
            printf("%d, u_x: %f\n", tid*(STEPS*2) + i*2 + 0, d_u[tid*(STEPS*2) + i*2 + 0]);
            printf("%d, u_y: %f\n", tid*(STEPS*2) + i*2 + 1, d_u[tid*(STEPS*2) + i*2 + 1]);
        }
        */
    }
    // sync thread.

    // Find min on the thread

    // get total min

    // compute normalisation term.

    // compute weight.

    // update actions

    // slide actions.
}

__global__ void set_data_(PointMassModelGpu* d_models,
                          float* d_x_i,
                          float* d_x,
                          float* d_u,
                          float* d_e,
                          size_t n_sim,
                          size_t steps,
                          float* state_gain,
                          size_t state_dim,
                          float* act_gain,
                          size_t act_dim,
                          curandState* rng_states,
                          float* goal,
                          float* w)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < n_sim){
        curand_init(tid, tid, tid, &rng_states[tid]);
        d_models[tid].init(&d_x[tid*steps*state_dim],
                            d_x_i[tid],
                            d_u,
                            &d_e[tid*steps*act_dim],
                            state_gain,
                            state_dim,
                            act_gain,
                            act_dim,
                            w,
                            goal);
    }
}
