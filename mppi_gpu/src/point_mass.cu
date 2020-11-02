#include "point_mass.hpp"

#include <iostream>
#include <math.h>

#define SIZE 256
#define SHMEM_SIZE 256


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
                                                 float* goal,
                                                 float lambda)
{
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

__host__ __device__ void PointMassModelGpu::step(curandState* state)
{

#ifdef __CUDA_ARCH__
    _e[(_t-1)*_u_size] += curand_normal(state);
    _e[(_t-1)*_u_size + 1] += curand_normal(state);
#else
    _e[(_t-1)*_u_size] += 0; //cpu random uniform;
    _e[(_t-1)*_u_size + 1] += 0; //cpu random uniform;
#endif
    for(int i=0; i < 2; i++){
        _x[_t*_x_size+i] = _x_gain[0]*_x[(_t-1)*_x_size+i] +
        _x_gain[1]*_x[(_t-1)*_x_size+i+2] +
        _u_gain[0]*(_u[(_t-1)*_u_size + i] + _e[(_t-1)*_u_size + i]);

        _x[_t*_x_size+i+2] = _x_gain[2]*_x[(_t-1)*_x_size+i] +
        _x_gain[3]*_x[(_t-1)*_x_size+i+2] +
        _u_gain[1]*(_u[(_t-1)*_u_size + i] + _e[(_t-1)*_u_size + i]);
    }
    _c += _cost.step_cost(&_x[_t*_x_size], &_u[(_t-1)*_u_size], &_e[(_t-1)*_u_size]);
}

__host__ __device__ float PointMassModelGpu::run(curandState* state)
{
    for (_t = 1; _t < _tau; _t++ ){
        step(state);
    }
    _c += _cost.final_cost(&_x[_t*_x_size]);
    return _c;
}

__host__ __device__ void PointMassModelGpu::set_state(float* x)
{
    _x = x;
}

__host__ __device__ void PointMassModelGpu::set_horizon(int horizon)
{
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


PointMassModel::PointMassModel(size_t nb_sim, size_t steps, float dt)
{
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

    // container for the min value and the normalisation term.
    cudaMalloc((void**)&d_beta, sizeof(float));
    cudaMalloc((void**)&d_nabla, sizeof(float));
    cudaMalloc((void**)&d_lambda, sizeof(float));

    float lambda[0];
    lambda[0] = 1.;

    cudaMemcpy(d_lambda, lambda, sizeof(float), cudaMemcpyHostToDevice);


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

PointMassModel::~PointMassModel()
{
    cudaFree((void*)d_x);
    cudaFree((void*)d_x_i);
    cudaFree((void*)d_u);
    cudaFree((void*)d_e);
    cudaFree((void*)d_beta);
    cudaFree((void*)d_nabla);
    cudaFree((void*)state_gain);
    cudaFree((void*)act_gain);
    cudaFree((void*)d_models);
    cudaFree((void*)rng_states);
}

void PointMassModel::sim()
{
    // launch 1 thread per simulation. Can later consider to
    // add dimensions to the block and thread of the kernel
    // to // enven more the code inside the simulation.
    // using blockDim.y & blockDim.z, blockIdx.y & blockIdx.x
    // and threadIdx.y & threadIdx.z.
    std::cout << "Starting simulations... : " << std::flush;
    sim_gpu_kernel_<<<1 + n_sim_/SIZE, SIZE>>>(d_models, n_sim_, d_e, rng_states);
    std::cout << "simulations finished!" << std::endl;
    std::cout << "Compute min cost... :" << std::flush;
    // find min cost
    min_beta();
    std::cout << "Done" << std::endl;
    std::cout << "Compute nabla... :" << std::flush;
    nabla();
    std::cout << "Done" << std::endl;
    // get normalisation
    //sum_red<<<>>>();
    //compute weights
    //weight<<<>>>();
    // compute new set of actions.
    //action<<<>>>();
    cudaDeviceSynchronize();
}

void PointMassModel::memcpy_set_data(float* x, float* u, float* goal, float* w)
{
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
                                        d_w,
                                        d_lambda);
    std::cout << "Done" << std::endl;
    cudaDeviceSynchronize();
}

void PointMassModel::memcpy_get_data(float* x_all, float* e)
{
    cudaMemcpy(x_all, d_x, bytes_, cudaMemcpyDeviceToHost);
    cudaMemcpy(e, d_e, sizeof(float)*n_sim_*act_dim*steps_, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}

void PointMassModel::min_beta()
{
    std::cout << "Finding beta" << std::endl;
    float* _d_beta;
    size_t _n_sim(n_sim_);
    // TB Size
    int BLOCK_SIZE = SIZE;

    // Grid Size (cut in half) (No padding)
    int GRID_SIZE = _n_sim / BLOCK_SIZE / 2 + 1;

    // THIS shouldn't change size during the controller iterations.
    // should verify this and then allocate data in the init to improve
    // computation time
    cudaMalloc((void**)&_d_beta, sizeof(float)*GRID_SIZE);

    std::cout << "Allocated device beta memory" << std::endl;

    if (GRID_SIZE == 1)
    {
        std::cout << "Run only once...: " << std::flush;
        min_red << <1, BLOCK_SIZE >> > (d_cost, _d_beta, _n_sim);
        std::cout << "Done" << std::endl;
    }
    else
    {
        // insure at least one pass.
        std::cout << "Run first...: " << std::flush;
        min_red << <GRID_SIZE, BLOCK_SIZE >> > (d_cost, _d_beta, _n_sim);
        std::cout << "Done" << std::endl;

        _n_sim = GRID_SIZE;
        GRID_SIZE = _n_sim / BLOCK_SIZE / 2 + 1 ;

        while (GRID_SIZE - 1 > 1 )
        {
            min_red << <GRID_SIZE, BLOCK_SIZE >> > (_d_beta, _d_beta, _n_sim);
            _n_sim = GRID_SIZE;
            GRID_SIZE = _n_sim / BLOCK_SIZE / 2 + 1 ;
        }
        std::cout << "Run last...: " << std::flush;
        min_red << <1, BLOCK_SIZE >> > (_d_beta, _d_beta, _n_sim);
        std::cout << "Done" << std::endl;
    }

    cudaMemcpy(d_beta, _d_beta, sizeof(float), cudaMemcpyDeviceToDevice);
    //d_beta[0] = _d_beta[0];

    cudaFree(_d_beta);
}

void PointMassModel::nabla()
{
    float* _d_nabla;
    size_t _n_sim(n_sim_);
    // TB Size
    int BLOCK_SIZE = SIZE;

    // Grid Size (cut in half) (No padding)
    int GRID_SIZE = _n_sim / BLOCK_SIZE / 2 + 1;

    cudaMalloc((void**)&_d_nabla, sizeof(float)*GRID_SIZE);


    if (GRID_SIZE == 1)
    {
        sum_red_exp << <1, BLOCK_SIZE >> > (d_cost, d_lambda, d_beta, _d_nabla, _n_sim);
    }
    else
    {
        // insure at least one pass.
        sum_red_exp << <GRID_SIZE, BLOCK_SIZE >> > (d_cost, d_lambda, d_beta, _d_nabla, _n_sim);

        _n_sim = GRID_SIZE;
        GRID_SIZE = _n_sim / BLOCK_SIZE / 2 + 1 ;

        while (GRID_SIZE - 1 > 1 )
        {
            sum_red << <GRID_SIZE, BLOCK_SIZE >> > (_d_nabla, _d_nabla, _n_sim);
            _n_sim = GRID_SIZE;
            GRID_SIZE = _n_sim / BLOCK_SIZE / 2 + 1 ;
        }
        sum_red << <1, BLOCK_SIZE >> > (_d_nabla, _d_nabla, _n_sim);
    }
    cudaMemcpy(d_nabla, _d_nabla, sizeof(float), cudaMemcpyDeviceToDevice);
    cudaFree(_d_nabla);
}

__global__ void sim_gpu_kernel_(PointMassModelGpu* d_models,
                                size_t n_sim_,
                                float* d_u,
                                curandState* rng_states)
{

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    float* cost = (float*) malloc(sizeof(float)*n_sim_);
    if(tid < n_sim_){
        /* local copy of rng state for faster generation. */
        curandState localState = rng_states[tid];
        cost[tid] = d_models[tid].run(&localState);
        /* copy back the state ohterwise the rng state is not working. */
        rng_states[tid] = localState;
        /*
        for (int i = 0; i < STEPS; i++){
            printf("%d, u_x: %f\n", tid*(STEPS*2) + i*2 + 0, d_u[tid*(STEPS*2) + i*2 + 0]);
            printf("%d, u_y: %f\n", tid*(STEPS*2) + i*2 + 1, d_u[tid*(STEPS*2) + i*2 + 1]);
        }
        */
    }
    // replace with a block sync threadrather than device.

    // Find min on the thread

    // get total min

    // compute normalisation term.

    // compute weight.

    // update actions

    // slide actions.
}

__global__ void min_red(float* v, float* beta, int n)
{
    // Allocate shared memory
	__shared__ int partial_min[SHMEM_SIZE];

	// Calculate thread ID
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Load elements AND do first add of reduction
	// Vector now 2x as long as number of threads, so scale i
	int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	// Store first partial result instead of just the elements
    if (i + blockDim.x < n)
    {
	   partial_min[threadIdx.x] = v[i] < v[i + blockDim.x] ? v[i] : v[i + blockDim.x];
    }
    else if (i < n)
    {
       partial_min[threadIdx.x] = v[i];
    }
    else
    {
        partial_min[threadIdx.x] = INFINITY;
    }
    //partial_sum[threadIdx.x] = v[i] + v[i + blockDim.x];

	__syncthreads();

	// Start at 1/2 block stride and divide by two each iteration
	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		// Each thread does work unless it is further than the stride
		if (threadIdx.x < s) {
			partial_min[threadIdx.x] += partial_min[threadIdx.x + s];
		}
		__syncthreads();
	}

	// Let the thread 0 for this block write it's result to main memory
	// Result is inexed by this block
	if (threadIdx.x == 0) {
		beta[blockIdx.x] = partial_min[0];
	}
}

__global__ void sum_red_exp(float* v, float* lambda, float* beta, float* v_r, int n)
{
    // Allocate shared memory
    __shared__ int partial_sum[SHMEM_SIZE];

    // Calculate thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Load elements AND do first add of reduction
    // Vector now 2x as long as number of threads, so scale i
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    // Store first partial result instead of just the elements
    if (i + blockDim.x < n)
    {
       v[i] = exp(1./lambda[0] * (v[i] - beta[0]));
       v[i + blockDim.x] = exp(1./lambda[0] * (v[i + blockDim.x] - beta[0]));
       partial_sum[threadIdx.x] = v[i] + v[i + blockDim.x];
    }
    else if (i < n)
    {
       v[i] = exp(1./lambda[0] * (v[i] - beta[0]));
       partial_sum[threadIdx.x] = v[i];
    }
    else
    {
        partial_sum[threadIdx.x] = 0;
    }
    //partial_sum[threadIdx.x] = v[i] + v[i + blockDim.x];

    __syncthreads();

    // Start at 1/2 block stride and divide by two each iteration
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        // Each thread does work unless it is further than the stride
        if (threadIdx.x < s) {
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Let the thread 0 for this block write it's result to main memory
    // Result is inexed by this block
    if (threadIdx.x == 0) {
        v_r[blockIdx.x] = partial_sum[0];
    }
}

__global__ void sum_red(float* v, float* v_r, int n)
{
    // Allocate shared memory
    __shared__ int partial_sum[SHMEM_SIZE];

    // Calculate thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Load elements AND do first add of reduction
    // Vector now 2x as long as number of threads, so scale i
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    // Store first partial result instead of just the elements
    if (i + blockDim.x < n)
    {
       partial_sum[threadIdx.x] = v[i] + v[i + blockDim.x];
    }
    else if (i < n)
    {
       partial_sum[threadIdx.x] = v[i];
    }
    else
    {
        partial_sum[threadIdx.x] = 0;
    }
    //partial_sum[threadIdx.x] = v[i] + v[i + blockDim.x];

    __syncthreads();

    // Start at 1/2 block stride and divide by two each iteration
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        // Each thread does work unless it is further than the stride
        if (threadIdx.x < s) {
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Let the thread 0 for this block write it's result to main memory
    // Result is inexed by this block
    if (threadIdx.x == 0) {
        v_r[blockIdx.x] = partial_sum[0];
    }
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
                          float* w,
                          float* lambda)
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
                            goal,
                            lambda[0]);
    }
}
