#include "point_mass.hpp"

#include <iostream>


__host__ __device__ PointMassModelGpu::PointMassModelGpu(){
    x_ = nullptr;
    u_ = nullptr;
    tau_ = STEPS;
    t_ = 1;
}

__host__ __device__ void PointMassModelGpu::init(float* x,
                                                 float init,
                                                 float* u,
                                                 float* x_gain,
                                                 int x_size,
                                                 float* u_gain,
                                                 int u_size){
    // TODO: cache x* in sm memory for faster access
    x_ = x;
    u_ = u;
    x_[0] = init;
    tau_ = STEPS;
    // Point the gain pointers to the right address
    x_gain_ = x_gain;
    x_size_ = x_size;

    u_gain_ = u_gain;
    u_size_ = u_size;

    t_ = 1;
    return;
}

__host__ __device__ void PointMassModelGpu::step(curandState* state){

#ifdef __CUDA_ARCH__
    u_[(t_-1)*u_size_] += curand_normal(state);
    u_[(t_-1)*u_size_ + 1] += curand_normal(state);
#else
    u_[(t_-1)*u_size_] += 0; //cpu random uniform;
    u_[(t_-1)*u_size_ + 1] += 0; //cpu random uniform;
#endif
    for(int i=0; i < 2; i++){
        x_[t_*x_size_+i] = x_gain_[0]*x_[(t_-1)*x_size_+i] +
        x_gain_[1]*x_[(t_-1)*x_size_+i+2] +
        u_gain_[0]*u_[(t_-1)*u_size_ + i];

        x_[t_*x_size_+i+2] = x_gain_[2]*x_[(t_-1)*x_size_+i] +
        x_gain_[3]*x_[(t_-1)*x_size_+i+2] +
        u_gain_[1]*u_[(t_-1)*u_size_ + i];
    }
}

__host__ __device__ void PointMassModelGpu::run(curandState* state){
    for (t_ = 1; t_ < tau_; t_++ ){
        step(state);
    }
}

__host__ __device__ void PointMassModelGpu::set_state(float* x){
    x_ = x;
}

__host__ __device__ void PointMassModelGpu::set_horizon(int horizon){
    /*
    * DO NOT USE ATM, when steps change, we need to update
    * the pointer x for the extra allocate space. As all the data
    * is represented in a continous array failling to do so will
    * produce a seg fault and probably leave the memory in a inconsistant
    * state.
    */
    tau_ = horizon;
}

__host__ __device__ float* PointMassModelGpu::get_state(){ return x_;}

__host__ __device__ int PointMassModelGpu::get_horizon(){ return tau_;}


PointMassModel::PointMassModel(int nb_sim, int steps, float dt){
    n_sim_ = nb_sim;
    steps_ = steps;
    act_dim = 2;
    state_dim = 4;

    dt_ = dt;


    /*
    * just for convinience, ultimatly replace with a
    * template type associated with the class wich will
    * represent the mppi domain.
    */
    bytes_ = sizeof(int)*steps_*n_sim_*state_dim;

    //host data used to send data to memory.
    float state_[4];
    float act_[2];

    act_[0] = dt_*dt_/2.0;
    act_[1] = dt_;
    state_[0] = 1;
    state_[1] = dt_;
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
    cudaMalloc((void**)&d_u, sizeof(float)*n_sim_*steps_*act_dim);

    // Set gain memory
    cudaMalloc((void**)&state_gain, sizeof(float)*state_dim);
    cudaMalloc((void**)&act_gain, sizeof(float)*act_dim);

    cudaMemcpy(state_gain, state_, sizeof(float)*state_dim, cudaMemcpyHostToDevice);
    cudaMemcpy(act_gain, act_, sizeof(float)*act_dim, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&rng_states, sizeof(curandState_t)*n_sim_);

    cudaDeviceSynchronize();
    std::cout << "Done" << std::endl;
}

PointMassModel::~PointMassModel(){
    cudaFree((void*)d_x);
    cudaFree((void*)d_x_i);
    cudaFree((void*)d_u);
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
    sim_gpu_kernel_<<<1 + n_sim_/256, 256>>>(d_models, n_sim_, d_u, rng_states);
    std::cout << "simulations finished!" << std::endl;
    cudaDeviceSynchronize();
}

void PointMassModel::memcpy_set_data(float* x, float* u){
    cudaMemcpy(d_x_i, x, sizeof(float)*n_sim_*state_dim, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, u, sizeof(float)*n_sim_*act_dim*steps_, cudaMemcpyHostToDevice);
    std::cout << "Setting inital state of the sims... : " << std::flush;
    set_data_<<<1 + n_sim_/256, 256>>>(d_models,
                                        d_x_i,
                                        d_x,
                                        d_u,
                                        n_sim_,
                                        steps_,
                                        state_gain,
                                        state_dim,
                                        act_gain,
                                        act_dim,
                                        rng_states);
    std::cout << "Done" << std::endl;
    cudaDeviceSynchronize();
}

void PointMassModel::memcpy_get_data(float* x_all, float* u){
    cudaMemcpy(x_all, d_x, bytes_, cudaMemcpyDeviceToHost);
    cudaMemcpy(u, d_u, sizeof(float)*n_sim_*act_dim*steps_, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}

__global__ void sim_gpu_kernel_(PointMassModelGpu* d_models,
                                int n_sim_,
                                float* d_u,
                                curandState* rng_states){

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
                          int n_sim,
                          int steps,
                          float* state_gain,
                          int state_dim,
                          float* act_gain,
                          int act_dim,
                          curandState* rng_states){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < n_sim){
        curand_init(tid*64, tid, tid, &rng_states[tid]);
        d_models[tid].init(&d_x[tid*steps*state_dim], d_x_i[tid], &d_u[tid*steps*act_dim], state_gain, 4, act_gain, 2);
    }
}
