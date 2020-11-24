#include "point_mass.hpp"
#include "mppi_utils.hpp"

#include <iostream>
#include <math.h>

#include <assert.h>

#include <cstdio>
#include <ctime>
#include <chrono>
#include <unistd.h>


#define SIZE 256
#define SHMEM_SIZE 256


PointMassModel::PointMassModel (int nb_sim,
                                int steps,
                                float dt,
                                int state_dim,
                                int act_dim,
                                bool verbose) {
    _n_sim = nb_sim;
    _steps = steps;
    _state_dim = state_dim;
    _act_dim = act_dim;

    _dt = dt;


    /*
    * just for convinience, ultimatly replace with a
    * template type associated with the class wich will
    * represent the mppi domain.
    */
    _bytes = sizeof(float)*(_steps+1)*_n_sim*_state_dim;

    //host data used to send data to memory.
    float state[4];
    float act[2];

    _verb = verbose;

    act[0] = _dt*_dt/2.0;
    act[1] = _dt;
    state[0] = 1;
    state[1] = _dt;
    state[2] = 0;
    state[3] = 1;

    float lambda[0];
    lambda[0] = 1.;

    int GRID_SIZE = _n_sim / SIZE / 2 + 1;
    // *Allocate the data on tahe GPU.*
    std::cout << "Allocating Space... : " << std::flush;
    // allocate space for all our simulation objects.
    CUDA_CALL_CONST(cudaMalloc((void**)&_models, sizeof(PointMassModelGpu)*_n_sim));
    // allocate space for the init_state array. int* x[n_sim]
    CUDA_CALL_CONST(cudaMalloc((void**)&_x_i, sizeof(float)*_n_sim*_state_dim));
    // allocate data space, continous in memeory so int* x[n_sim*steps_]
    CUDA_CALL_CONST(cudaMalloc((void**)&_x, sizeof(float)*_n_sim*(_steps+1)*_state_dim));
    // set the memory with 0s.
    CUDA_CALL_CONST(cudaMemset((void*)_x, 0, sizeof(float)*_n_sim*(_steps+1)*_state_dim));
    // allocate space for action.
    CUDA_CALL_CONST(cudaMalloc((void**)&_e, sizeof(float)*_n_sim*_steps*_act_dim));

    CUDA_CALL_CONST(cudaMemset((void*)_e, 0, sizeof(float)*_n_sim*_steps*_act_dim));

    CUDA_CALL_CONST(cudaMalloc((void**)&_u, sizeof(float)*_steps*_act_dim));
    CUDA_CALL_CONST(cudaMalloc((void**)&_u_swap, sizeof(float)*_steps*_act_dim));

    CUDA_CALL_CONST(cudaMalloc((void**)&_cost, sizeof(float)*_n_sim));
    CUDA_CALL_CONST(cudaMalloc((void**)&_exp, sizeof(float)*_n_sim));

    // container for the min value and the normalisation term.
    CUDA_CALL_CONST(cudaMalloc((void**)&_beta, sizeof(float)));

    CUDA_CALL_CONST(cudaMalloc((void**)&_nabla, sizeof(float)));

    CUDA_CALL_CONST(cudaMalloc((void**)&_lambda, sizeof(float)));

    CUDA_CALL_CONST(cudaMalloc((void**)&_weights, sizeof(float)*_n_sim));

    CUDA_CALL_CONST(cudaMemcpy(_lambda, lambda, sizeof(float), cudaMemcpyHostToDevice));


    // Set gain memory
    CUDA_CALL_CONST(cudaMalloc((void**)&_state_gain, sizeof(float)*_state_dim));
    CUDA_CALL_CONST(cudaMalloc((void**)&_act_gain, sizeof(float)*_act_dim));

    CUDA_CALL_CONST(cudaMemcpy(_state_gain, state, sizeof(float)*_state_dim, cudaMemcpyHostToDevice));
    CUDA_CALL_CONST(cudaMemcpy(_act_gain, act, sizeof(float)*_act_dim, cudaMemcpyHostToDevice));

    CUDA_CALL_CONST(cudaMalloc((void**)&_rng_states, sizeof(curandState_t)*_n_sim));

    CUDA_CALL_CONST(cudaMalloc((void**)&_g, sizeof(float)*_state_dim));
    CUDA_CALL_CONST(cudaMalloc((void**)&_w, sizeof(float)*_state_dim));


    //CUDA_CALL_CONST(cudaMalloc((void**)&v_r, sizeof(float)*))

    cudaDeviceSynchronize();
    std::cout << "Done" << std::endl;
}

PointMassModel::~PointMassModel () {
    CUDA_CALL_CONST(cudaFree((void*)_x));
    CUDA_CALL_CONST(cudaFree((void*)_x_i));
    CUDA_CALL_CONST(cudaFree((void*)_u));
    CUDA_CALL_CONST(cudaFree((void*)_u_swap));
    CUDA_CALL_CONST(cudaFree((void*)_e));
    CUDA_CALL_CONST(cudaFree((void*)_cost));
    CUDA_CALL_CONST(cudaFree((void*)_exp));
    CUDA_CALL_CONST(cudaFree((void*)_beta));
    CUDA_CALL_CONST(cudaFree((void*)_nabla));
    CUDA_CALL_CONST(cudaFree((void*)_lambda));
    CUDA_CALL_CONST(cudaFree((void*)_weights));
    CUDA_CALL_CONST(cudaFree((void*)_models));
    CUDA_CALL_CONST(cudaFree((void*)_g));
    CUDA_CALL_CONST(cudaFree((void*)_w));
    CUDA_CALL_CONST(cudaFree((void*)_state_gain));
    CUDA_CALL_CONST(cudaFree((void*)_act_gain));
    CUDA_CALL_CONST(cudaFree((void*)_rng_states));

}

void PointMassModel::get_act (float* next_act) {
    /* launch 1 thread per simulation. Can later consider to
     * add dimensions to the block and thread of the kernel
     * to // enven more the code inside the simulation.
     * using blockDim.y & blockDim.z, blockIdx.y & blockIdx.x
     * and threadIdx.y & threadIdx.z. */
    if (_verb) {
        std::cout << "Print x before Sim" << std::endl;
        print_x<<<1, 1>>>(_x, (_steps+1), _n_sim, _state_dim);
        cudaDeviceSynchronize();
    }

    sim();
    cudaDeviceSynchronize();

    if (_verb) {
        std::cout << "Print x after Sim" << std::endl;
        print_x<<<1, 1>>>(_x, (_steps+1), _n_sim, _state_dim);
        cudaDeviceSynchronize();
        std::cout << "Print u" << std::endl;
        print_u<<<1, 1>>>(_u, _steps, _act_dim);
        cudaDeviceSynchronize();
        std::cout << "Print e" << std::endl;
        std::cout << _act_dim << std::endl;

        print_e<<<1, 1>>>(_e, _steps, _n_sim, _act_dim);
        cudaDeviceSynchronize();
    }


    beta();
    if (_verb) {
        print_beta<<<1, 1>>>(_beta, 1);
        std::cout << std::endl;
    }
    cudaDeviceSynchronize();

    exp();
    if (_verb) {
        print_exp<<<1, 1>>>(_exp, _n_sim);
        std::cout << std::endl;
    }
    cudaDeviceSynchronize();

    nabla();
    if (_verb) {
        print_nabla<<<1, 1>>>(_nabla, 1);
        std::cout << std::endl;
    }
    cudaDeviceSynchronize();

    weights();
    if (_verb) {
        print_sum_weights<<<1, 1>>>(_weights, _n_sim);
        std::cout << std::endl;
    }
    cudaDeviceSynchronize();

    // CURRENT BOTTLENECK!!!
    update_act();

    if (_verb) {
        print_u<<<1, 1>>>(_u, _steps, _act_dim);
        std::cout << std::endl;
    }
    cudaDeviceSynchronize();

    CUDA_CALL_CONST(cudaMemcpy(next_act, _u, sizeof(float)*_act_dim, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();

    shift_act<<<1 + _n_sim/SIZE, SIZE>>>(_u, _u_swap, _act_dim, _steps);
    CUDA_CALL_CONST(cudaMemcpy(_u, _u_swap, sizeof(float)*_act_dim*_steps, cudaMemcpyDeviceToDevice));

    cudaDeviceSynchronize();

}

void PointMassModel::memcpy_set_data (float* x, float* u, float* goal, float* w) {
    CUDA_CALL_CONST(cudaMemcpy(_x_i, x, sizeof(float)*_state_dim, cudaMemcpyHostToDevice));
    CUDA_CALL_CONST(cudaMemcpy(_u, u, sizeof(float)*_act_dim*_steps, cudaMemcpyHostToDevice));
    CUDA_CALL_CONST(cudaMemcpy(_g, goal, sizeof(float)*_state_dim, cudaMemcpyHostToDevice));
    CUDA_CALL_CONST(cudaMemcpy(_w, w, sizeof(float)*_state_dim, cudaMemcpyHostToDevice));
    std::cout << "Setting inital state of the sims... : " << std::flush;
    set_data<<<1 + _n_sim/SIZE, SIZE>>>(_models,
                                        _x_i,
                                        _x,
                                        _u,
                                        _e,
                                        _n_sim,
                                        _steps,
                                        _state_gain,
                                        _state_dim,
                                        _act_gain,
                                        _act_dim,
                                        _rng_states,
                                        _g,
                                        _w,
                                        _lambda);
    std::cout << "Done" << std::endl;
    cudaDeviceSynchronize();
}

void PointMassModel::memcpy_get_data (float* x_all, float* e) {
    CUDA_CALL_CONST(cudaMemcpy(x_all, _x, _bytes, cudaMemcpyDeviceToHost));
    CUDA_CALL_CONST(cudaMemcpy(e, _e, sizeof(float)*_n_sim*_act_dim*_steps, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
}

void PointMassModel::get_inf (float* x,
                              float* u,
                              float* e,
                              float* cost,
                              float* beta,
                              float* nabla,
                              float* w) {
    // get all the info to look at the results and debug if necessary.
    std::cout << "Collect informations: " << std::endl;
    //std::cout << "N " << _n_sim << " STEPS: " << _steps << " State dim: " << _state_dim << std::endl;
    //std::cout << "X " << std::endl;
    CUDA_CALL_CONST(cudaMemcpy(x, _x, sizeof(float)*(_steps+1)*_n_sim*_state_dim, cudaMemcpyDeviceToHost));
    //std::cout << "U " << std::endl;
    CUDA_CALL_CONST(cudaMemcpy(u, _u, sizeof(float)*_steps*_act_dim, cudaMemcpyDeviceToHost));
    //std::cout << "E " << std::endl;
    CUDA_CALL_CONST(cudaMemcpy(e, _e, sizeof(float)*_n_sim*_steps*_act_dim, cudaMemcpyDeviceToHost));
    //std::cout << "Cost " << std::endl;
    CUDA_CALL_CONST(cudaMemcpy(cost, _cost, sizeof(float)*_n_sim, cudaMemcpyDeviceToHost));
    //std::cout << "Beta " << std::endl;
    CUDA_CALL_CONST(cudaMemcpy(beta, _beta, sizeof(float), cudaMemcpyDeviceToHost));
    //std::cout << "Nabla " << std::endl;
    CUDA_CALL_CONST(cudaMemcpy(nabla, _nabla, sizeof(float), cudaMemcpyDeviceToHost));
    //std::cout << "Weights " << std::endl;
    CUDA_CALL_CONST(cudaMemcpy(w, _weights, sizeof(float)*_n_sim, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();

}

void PointMassModel::sim () {
    sim_gpu_kernel_<<<1 + _n_sim/SIZE, SIZE>>>(_models, _n_sim, _e, _cost, _rng_states);
}

void PointMassModel::beta () {
    int n(_n_sim);
    // TB Size
    int BLOCK_SIZE = SIZE;
    // Grid Size (cut in half) (No padding)
    int GRID_SIZE = n / BLOCK_SIZE / 2 + 1;

    float* v_r;
    CUDA_CALL_CONST(cudaMalloc((void**)&v_r, sizeof(float)*GRID_SIZE));
    CUDA_CALL_CONST(cudaMemset((void*)v_r, 0, sizeof(float)*GRID_SIZE));

    if (GRID_SIZE == 1)
    {
        //std::cout << "Run only once" << std::endl;
        min_red << <1, BLOCK_SIZE >> > (_cost, v_r, n);
    }
    else
    {
        //std::cout << "Run first" << std::endl;
        // insure at least one pass.
        min_red << <GRID_SIZE, BLOCK_SIZE >> > (_cost, v_r, n);

        n = GRID_SIZE;
        GRID_SIZE = n / BLOCK_SIZE / 2 + 1 ;
        cudaDeviceSynchronize();
        while (GRID_SIZE - 1 > 1 )
        {
            //std::cout << "Mid" << std::endl;
            min_red << <GRID_SIZE, BLOCK_SIZE >> > (v_r, v_r, n);
            n = GRID_SIZE;
            GRID_SIZE = n / BLOCK_SIZE / 2 + 1 ;
            cudaDeviceSynchronize();
        }
        //std::cout << "Run last" << std::endl;
        min_red << <1, BLOCK_SIZE >> > (v_r, v_r, n);
    }

    cudaDeviceSynchronize();

    //std::cout << "Copy beta: " << std::endl;
    //float h_beta[1];
    CUDA_CALL_CONST(cudaMemcpy(_beta, v_r, sizeof(float), cudaMemcpyDeviceToDevice));
    //CUDA_CALL_CONST(cudaMemcpy(h_beta, d_beta, sizeof(float), cudaMemcpyDeviceToHost));
    //std::cout << "beta: " << h_beta[0] << std::endl;
    CUDA_CALL_CONST(cudaFree((void*)v_r));
}

void PointMassModel::exp () {
    exp_red <<<1 + _n_sim/SIZE, SIZE>>> (_exp, _cost, _lambda, _beta, _n_sim);
}

void PointMassModel::nabla () {
    int n(_n_sim);

    int BLOCK_SIZE = SIZE;
    int GRID_SIZE = n / BLOCK_SIZE / 2 + 1;

    float* v_r;
    CUDA_CALL_CONST(cudaMalloc((void**)&v_r, sizeof(float)*GRID_SIZE));
    CUDA_CALL_CONST(cudaMemset((void*)v_r, 0, sizeof(float)*GRID_SIZE));

    //std::cout << "Find norm term " << " with: "
    //          << "GRID_SIZE: " << GRID_SIZE << ", "
    //          << "BLOCK_SIZE: " << BLOCK_SIZE << std::endl;

    if (GRID_SIZE == 1)
    {
    //    std::cout << "Run only once" << std::endl;
        sum_red <<<1, BLOCK_SIZE >>> (_exp, v_r, n);
    }
    else
    {
    //    std::cout << "Run first" << std::endl;
        // insure at least one pass.
        sum_red <<<GRID_SIZE, BLOCK_SIZE >>> (_exp, v_r, n);
        cudaDeviceSynchronize();

        n = GRID_SIZE;
        GRID_SIZE = n / BLOCK_SIZE / 2 + 1 ;

        while (GRID_SIZE - 1 > 1 )
        {
    //        std::cout << "Mid" << std::endl;
            sum_red <<<GRID_SIZE, BLOCK_SIZE >>> (v_r, v_r, n);
            n = GRID_SIZE;
            GRID_SIZE = n / BLOCK_SIZE / 2 + 1 ;
            cudaDeviceSynchronize();
        }
    //    std::cout << "Run last" << std::endl;
        sum_red <<<1, BLOCK_SIZE >>> (v_r, v_r, n);
    }

    cudaDeviceSynchronize();

    //std::cout << "Copy nabla: " << std::endl;
    //float h_nabla[1];
    CUDA_CALL_CONST(cudaMemcpy(_nabla, v_r, sizeof(float), cudaMemcpyDeviceToDevice));
    //CUDA_CALL_CONST(cudaMemcpy(h_nabla, d_nabla, sizeof(float), cudaMemcpyDeviceToHost));
    //std::cout << "nabla: " << h_nabla[0] << std::endl;
    CUDA_CALL_CONST(cudaFree((void*)v_r));
}

void PointMassModel::weights () {
    weights_kernel<<<1 + _n_sim/SIZE, SIZE>>>(_cost, _weights, _lambda, _beta, _nabla, _n_sim);
    cudaDeviceSynchronize();
}

void PointMassModel::update_act () {
    int n(_n_sim);
    int BLOCK_SIZE = SIZE;
    int GRID_SIZE = n / BLOCK_SIZE / 2 / _act_dim  + 1;
    float* v_r;
//    std::cout << "allocate data" << std::endl;
    CUDA_CALL_CONST(cudaMalloc((void**)&v_r, sizeof(float)*GRID_SIZE*_act_dim));
    CUDA_CALL_CONST(cudaMemset((void*)v_r, 0, sizeof(float)*GRID_SIZE*_act_dim));
//    std::cout << "Data allocated" << std::endl;

    // This is a problem, should parallelize this.
    for (int t=0; t < _steps; t++)
    {
        // TB Size
        n = _n_sim;
        int BLOCK_SIZE = SIZE;
        // Grid Size (cut in half) (No padding)
        int GRID_SIZE = n / BLOCK_SIZE / 2 / _act_dim + 1;


//        std::cout << "Starting update at t: " << t << " with: "
//                  << "GRID_SIZE: " << GRID_SIZE << ", "
//                  << "BLOCK_SIZE: " << BLOCK_SIZE << std::endl;
        if (GRID_SIZE == 1)
        {
            //print_e<<<1, 1>>>(d_e, STEPS, _n_sim, act_dim);
            //std::cout << "Only before : " << t << std::endl;
            //print_u<<<1, 1>>>(v_r, GRID_SIZE, act_dim);
            //std::cout << "Run only once" << std::endl;
            //std::cout << "steps: " << STEPS << " t: " << t << " a_dim: " << act_dim << " n: " << _n_sim << std::endl;
            update_act_kernel <<<1, BLOCK_SIZE >>> (v_r, _weights, _e, _steps, t, _act_dim, n);
            cudaDeviceSynchronize();

            //std::cout << "Only : " << t << std::endl;
            //print_u<<<1, 1>>>(v_r, GRID_SIZE, act_dim);
            //cudaDeviceSynchronize();

        }
        else
        {
            //std::cout << "Run first" << std::endl;
            // insure at least one pass.
            //std::cout << "steps: " << STEPS << " t: " << t << " a_dim: " << act_dim << " n: " << _n_sim << std::endl;
            update_act_kernel <<<GRID_SIZE, BLOCK_SIZE >>> (v_r, _weights, _e, _steps, t, _act_dim, n);

            n = GRID_SIZE;
            GRID_SIZE = n / BLOCK_SIZE / 2 / _act_dim  + 1 ;
            cudaDeviceSynchronize();

            //std::cout << "First : " << t << std::endl;
            //print_u<<<1, 1>>>(v_r, GRID_SIZE, act_dim);
            //cudaDeviceSynchronize();


            while (GRID_SIZE - 1 > 1 )
            {
                //std::cout << "Mid" << std::endl;
                sum_red << <GRID_SIZE, BLOCK_SIZE >> > (v_r, v_r, n);
                n = GRID_SIZE;
                GRID_SIZE = n / BLOCK_SIZE / 2 / _act_dim  + 1 ;
                cudaDeviceSynchronize();

                /*
                std::cout << "Midle : " << t << std::endl;
                print_u<<<1, 1>>>(v_r, GRID_SIZE, act_dim);
                cudaDeviceSynchronize();*/


            }
            //std::cout << "Run last" << std::endl;
            //std::cout << "steps: " << STEPS << " t: " << t << " a_dim: " << act_dim << " n: " << _n_sim << std::endl;
            sum_red <<<1, BLOCK_SIZE >>> (v_r, v_r, n);
            cudaDeviceSynchronize();
            /*
            std::cout << "Last :" << t << std::endl;
            print_u<<<1, 1>>>(v_r, GRID_SIZE, act_dim);
            cudaDeviceSynchronize();
            */

        }
        //std::cout << "Copy at t: " << t << std::endl;
        /*
        std::cout << "Before copy :" << t << std::endl;
        print_u<<<1, 1>>>(v_r, 1, act_dim);
        cudaDeviceSynchronize();
        */
        copy_act<<< 1 + n/SIZE, SIZE >>>(_u, v_r, t, _act_dim);
        cudaDeviceSynchronize();
        //std::cout << "Done" << std::endl;

    }

    CUDA_CALL_CONST(cudaFree((void*)v_r));

    cudaDeviceSynchronize();
}

void PointMassModel::set_x (float* h_x) {
    CUDA_CALL_CONST(cudaMemcpy(_x_i, h_x, sizeof(float)*_state_dim, cudaMemcpyHostToDevice));
    set_x_kernel<<<1 + _n_sim/SIZE, SIZE>>>(_models, _x_i, _n_sim);
    CUDA_CALL_CONST(cudaDeviceSynchronize());
}

void PointMassModel::get_u (float* u) {
    CUDA_CALL_CONST(cudaMemcpy(u, _u, sizeof(float)*_steps*_act_dim, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
}

__global__ void sim_gpu_kernel_ (PointMassModelGpu* models,
                                int n_sim,
                                float* u,
                                float* cost,
                                curandState* rng_states) {

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < n_sim){
        /* local copy of rng state for faster generation. */
        curandState localState = rng_states[tid];
        cost[tid] = models[tid].run(&localState);

        /* copy back the state ohterwise the rng state is not working. */
        rng_states[tid] = localState;
    }
}

__global__ void exp_red (float* out,
                         float* cost,
                         float* lambda,
                         float* beta,
                         int size) {
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if(tid < size)
    {
        out[tid] = expf(-(1/lambda[0]) * (cost[tid] - beta[0]));
        if (false) {
            printf("1/lambda: %f, cost[%d]: %f, beta: %f, res[%d]: %f,  arg: %f\n",
                    1/lambda[0],
                    tid,
                    cost[tid],
                    beta[0],
                    tid,
                    out[tid],
                    (cost[tid] - beta[0]));
        }
    }

}

__global__ void min_red (float* v, float* beta, int n) {
    // Allocate shared memory
	__shared__ float partial_min[SHMEM_SIZE];

	// Load elements AND do first add of reduction
	// Vector now 2x as long as number of threads, so scale i
	int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	// Store first partial result instead of just the elements
    if (i + blockDim.x < n)
    {
        partial_min[threadIdx.x] = v[i] < v[i + blockDim.x] ? v[i] : v[i + blockDim.x];
        //printf("v[%d]: %f, v[%d]: %f\n", i, v[i], i + blockDim.x, v[i+ blockDim.x]);
    }
    else if (i < n)
    {
        partial_min[threadIdx.x] = v[i];
        //printf("v[%d]: %f\n", i, v[i]);
    }
    else
    {
        //printf("HERE\n");
        partial_min[threadIdx.x] = INFINITY;
    }
    //partial_sum[threadIdx.x] = v[i] + v[i + blockDim.x];

	__syncthreads();

	// Start at 1/2 block stride and divide by two each iteration
	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		// Each thread does work unless it is further than the stride
		if (threadIdx.x < s) {
			partial_min[threadIdx.x] = partial_min[threadIdx.x] < partial_min[threadIdx.x + s] ? partial_min[threadIdx.x] : partial_min[threadIdx.x + s];
		}
		__syncthreads();
	}

	// Let the thread 0 for this block write it's result to main memory
	// Result is inexed by this block
	if (threadIdx.x == 0) {
		beta[blockIdx.x] = partial_min[0];
        //printf("partial_min[0]: %f\n", partial_min[0]);
	}
}

__global__ void sum_red_exp (float* v,
                             float* lambda_1,
                             float* beta,
                             float* v_r,
                             int n) {
    // Allocate shared memory
    __shared__ float partial_sum[SHMEM_SIZE];

    // Load elements AND do first add of reduction
    // Vector now 2x as long as number of threads, so scale i
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    // Store first partial result instead of just the elements
    if (i + blockDim.x < n)
    {
       v[i] = expf(lambda_1[0] * (v[i] - beta[0]));
       v[i + blockDim.x] = expf(lambda_1[0] * (v[i + blockDim.x] - beta[0]));
       partial_sum[threadIdx.x] = v[i] + v[i + blockDim.x];
       //printf("v[%d]: %f, v[%d]: %f\n", i, v[i], i + blockDim.x, v[i+ blockDim.x]);
       //printf("partial_sim[%d]: %f\n", threadIdx.x, partial_sum[threadIdx.x] );
    }
    else if (i < n)
    {
       v[i] = expf(lambda_1[0] * (v[i] - beta[0]));
       partial_sum[threadIdx.x] = v[i];
       //printf("v[%d]: %f\n", i, v[i]);
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
        //printf("res[%d]: %f\n", blockIdx.x, v_r[blockIdx.x]);
    }
}

__global__ void sum_red (float* v, float* v_r, int n) {
    // Allocate shared memory
    __shared__ float partial_sum[SHMEM_SIZE];

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

__global__ void weights_kernel (float* v,
                                float* v_r,
                                float* lambda_1,
                                float* beta,
                                float* nabla_1,
                                int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n){
        v_r[tid] = 1.0/nabla_1[0] * expf(-(1.0/lambda_1[0])*(v[tid] - beta[0]));
        //printf("weight[%d]: %f, nabla_1: %f, 1/nabla_1: %f\n", tid, v_r[tid], nabla_1[0], 1/nabla_1[0]);
    }
}

__global__ void copy_act (float* u, float* tmp, int t, int act_dim) {
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if (tid < act_dim)
        //printf("u[%d]: %f\n", (tid + t*act_dim), u[tid + t*act_dim]);
        u[tid + t*act_dim] += tmp[tid];
}

__global__ void set_data (PointMassModelGpu* d_models,
                          float* d_x_i,
                          float* d_x,
                          float* d_u,
                          float* d_e,
                          int n_sim,
                          int steps,
                          float* state_gain,
                          int state_dim,
                          float* act_gain,
                          int act_dim,
                          curandState* rng_states,
                          float* goal,
                          float* w,
                          float* lambda) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < n_sim){
        curand_init(tid, tid, tid, &rng_states[tid]);
        d_models[tid].init(&d_x[tid*(steps+1)*state_dim],
                            d_x_i,
                            d_u,
                            &d_e[tid*steps*act_dim],
                            STEPS,
                            state_gain,
                            state_dim,
                            act_gain,
                            act_dim,
                            w,
                            goal,
                            lambda[0],
                            tid);
    }
}

__global__ void set_x_kernel (PointMassModelGpu* d_models, float* x_i, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n)
    {
        d_models[tid].set_x(x_i);
    }
}

__global__ void shift_act (float* u, float* u_swap, int a_dim, int samples) {
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if(tid < samples - 1)
    {
        for(int j=0; j < a_dim; j++)
        {
            u_swap[(tid)*a_dim+j] = u[(tid+1)*a_dim+j];
        }
    }

    // last element of shifited array is initalized with a repeating scheme
    if(tid == samples - 1){
        for(int j=0; j < a_dim; j++)
        {
            u_swap[(tid)*a_dim+j] = u[(tid)*a_dim+j];
        }
    }


}

// First implementation. Usually T << K so parallelize over K first. BOTTLENECK OF THE CODE
// NEEDS A LOT OF INVESTIGATION
__global__ void update_act_kernel (float* v_r,
                                  float* w,
                                  float* e,
                                  int steps,
                                  int t,
                                  int act_dim,
                                  int n) {

    // Allocate shared memory
    __shared__ float partial_acts[SHMEM_SIZE*2];

    int i = blockIdx.x * (blockDim.x*2) * act_dim + t * act_dim + threadIdx.x*steps*act_dim;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int shift = blockDim.x*steps*act_dim;
    int max_size = n*act_dim*steps;
    /*
     * If the array is long enough, one thread can compute two elements
     * of the output array at the time
     */
    if (i + shift < max_size)
    {
        for (int j = 0; j < act_dim; j++)
        {
            partial_acts[threadIdx.x*act_dim + j] = w[k]*e[i + j] +
                                                    w[k]*e[i + blockDim.x*steps*act_dim + j];
            //printf("e[%d]: %f\n", i+j, e[i+j]);
            //printf("e[%d]: %f\n", i+ blockDim.x*steps*act_dim +j, e[i+blockDim.x*steps*act_dim+j]);
        }
    }
    /*
     * If the shift is out of bound only compute one element.
     */
    else if (i < max_size )
    {
        for (int j = 0; j < act_dim; j++)
        {
            partial_acts[threadIdx.x*act_dim + j] = w[k]*e[i + j];
            //printf("e[%d]: %f, w[%d]: %f\n", i+j, e[i+j], k, w[k]);
        }
    }
    else
    {
        for (int j = 0; j < act_dim; j++)
        {
            partial_acts[threadIdx.x*act_dim + j] = 0;
        }
    }
    //partial_sum[threadIdx.x] = v[i] + v[i + blockDim.x];
    __syncthreads();

    // Start at 1/2 block stride and divide by two each iteration
    for (int s = blockDim.x * act_dim / 2; s > 1; s >>= 1)
    {
        // Each thread does work unless it is further than the stride
        if (threadIdx.x*act_dim  < s)
        {
            for (int j = 0; j < act_dim; j++)
            {
                partial_acts[threadIdx.x*act_dim + j] += partial_acts[threadIdx.x*act_dim + j + s];
            }
        }
        __syncthreads();
    }

    // Let the thread 0 for this block write it's result to main memory
    // Result is inexed by this block
    if (threadIdx.x == 0)
    {
        //printf("FUCK YOU CUNT");
        for(int j = 0; j < act_dim; j++)
        {
            v_r[blockIdx.x*act_dim + j] = partial_acts[j];
            //printf("Id: %d, val: %f\n",blockIdx.x*act_dim + j, partial_acts[j]);
        }
    }
}
