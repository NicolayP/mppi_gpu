#include "point_mass.hpp"

#include <iostream>
#include <math.h>

#include <assert.h>

#include <cstdio>
#include <ctime>
#include <chrono>
#include <unistd.h>


#define SIZE 256
#define SHMEM_SIZE 256


__host__ __device__ PointMassModelGpu::PointMassModelGpu () {
    _x = nullptr;
    _u = nullptr;
    _e = nullptr;
    _tau = STEPS;
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

PointMassModel::PointMassModel (int nb_sim, int steps, float dt, bool verbose) {
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
    bytes_ = sizeof(float)*(steps_+1)*n_sim_*state_dim;

    //host data used to send data to memory.
    float state_[4];
    float act_[2];

    _verb = verbose;

    act_[0] = _dt*_dt/2.0;
    act_[1] = _dt;
    state_[0] = 1;
    state_[1] = _dt;
    state_[2] = 0;
    state_[3] = 1;

    float lambda[0];
    lambda[0] = 1.;

    int GRID_SIZE = n_sim_ / SIZE / 2 + 1;
    // *Allocate the data on tahe GPU.*
    std::cout << "Allocating Space... : " << std::flush;
    // allocate space for all our simulation objects.
    CUDA_CALL_CONST(cudaMalloc((void**)&d_models, sizeof(PointMassModelGpu)*n_sim_));
    // allocate space for the init_state array. int* x[n_sim]
    CUDA_CALL_CONST(cudaMalloc((void**)&d_x_i, sizeof(float)*n_sim_*state_dim));
    // allocate data space, continous in memeory so int* x[n_sim*steps_]
    CUDA_CALL_CONST(cudaMalloc((void**)&d_x, sizeof(float)*n_sim_*(steps_+1)*state_dim));
    // set the memory with 0s.
    CUDA_CALL_CONST(cudaMemset((void*)d_x, 0, sizeof(float)*n_sim_*(steps_+1)*state_dim));
    // allocate space for action.
    CUDA_CALL_CONST(cudaMalloc((void**)&d_e, sizeof(float)*n_sim_*steps_*act_dim));

    CUDA_CALL_CONST(cudaMemset((void*)d_e, 0, sizeof(float)*n_sim_*steps_*act_dim));

    CUDA_CALL_CONST(cudaMalloc((void**)&d_u, sizeof(float)*steps_*act_dim));
    CUDA_CALL_CONST(cudaMalloc((void**)&d_u_swap, sizeof(float)*steps_*act_dim));

    CUDA_CALL_CONST(cudaMalloc((void**)&d_cost, sizeof(float)*n_sim_));
    CUDA_CALL_CONST(cudaMalloc((void**)&d_exp, sizeof(float)*n_sim_));

    // container for the min value and the normalisation term.
    CUDA_CALL_CONST(cudaMalloc((void**)&d_beta, sizeof(float)));
    // used for the reduction algorithm
    CUDA_CALL_CONST(cudaMalloc((void**)&_d_beta, sizeof(float)*GRID_SIZE));
    CUDA_CALL_CONST(cudaMemset((void*)_d_beta, 0, sizeof(float)*GRID_SIZE));

    CUDA_CALL_CONST(cudaMalloc((void**)&d_nabla, sizeof(float)));
    // used for the reduction algorithm
    CUDA_CALL_CONST(cudaMalloc((void**)&_d_nabla, sizeof(float)*GRID_SIZE));
    CUDA_CALL_CONST(cudaMemset((void*)_d_nabla, 0, sizeof(float)*GRID_SIZE));

    CUDA_CALL_CONST(cudaMalloc((void**)&d_lambda, sizeof(float)));

    CUDA_CALL_CONST(cudaMalloc((void**)&d_weights, sizeof(float)*n_sim_));

    CUDA_CALL_CONST(cudaMemcpy(d_lambda, lambda, sizeof(float), cudaMemcpyHostToDevice));


    // Set gain memory
    CUDA_CALL_CONST(cudaMalloc((void**)&state_gain, sizeof(float)*state_dim));
    CUDA_CALL_CONST(cudaMalloc((void**)&act_gain, sizeof(float)*act_dim));

    CUDA_CALL_CONST(cudaMemcpy(state_gain, state_, sizeof(float)*state_dim, cudaMemcpyHostToDevice));
    CUDA_CALL_CONST(cudaMemcpy(act_gain, act_, sizeof(float)*act_dim, cudaMemcpyHostToDevice));

    CUDA_CALL_CONST(cudaMalloc((void**)&rng_states, sizeof(curandState_t)*n_sim_));

    CUDA_CALL_CONST(cudaMalloc((void**)&d_g, sizeof(float)*state_dim));
    CUDA_CALL_CONST(cudaMalloc((void**)&d_w, sizeof(float)*state_dim));


    //CUDA_CALL_CONST(cudaMalloc((void**)&v_r, sizeof(float)*))

    cudaDeviceSynchronize();
    std::cout << "Done" << std::endl;
}

PointMassModel::~PointMassModel () {
    CUDA_CALL_CONST(cudaFree((void*)d_x));
    CUDA_CALL_CONST(cudaFree((void*)d_x_i));
    CUDA_CALL_CONST(cudaFree((void*)d_u));
    CUDA_CALL_CONST(cudaFree((void*)d_u_swap));
    CUDA_CALL_CONST(cudaFree((void*)d_e));
    CUDA_CALL_CONST(cudaFree((void*)d_cost));
    CUDA_CALL_CONST(cudaFree((void*)d_exp));
    CUDA_CALL_CONST(cudaFree((void*)d_beta));
    CUDA_CALL_CONST(cudaFree((void*)_d_beta));
    CUDA_CALL_CONST(cudaFree((void*)d_nabla));
    CUDA_CALL_CONST(cudaFree((void*)_d_nabla));
    CUDA_CALL_CONST(cudaFree((void*)state_gain));
    CUDA_CALL_CONST(cudaFree((void*)act_gain));
    CUDA_CALL_CONST(cudaFree((void*)d_models));
    CUDA_CALL_CONST(cudaFree((void*)rng_states));

}

void PointMassModel::sim (float* next_act) {
    //std::chrono::time_point<std::chrono::system_clock> t1;
    //std::chrono::time_point<std::chrono::system_clock> t2;
    //std::chrono::duration<double, std::milli> fp_ms;
    //double delta;
    // launch 1 thread per simulation. Can later consider to
    // add dimensions to the block and thread of the kernel
    // to // enven more the code inside the simulation.
    // using blockDim.y & blockDim.z, blockIdx.y & blockIdx.x
    // and threadIdx.y & threadIdx.z.
    //std::cout << "Running simulations... : " << std::flush;
    //t1 = std::chrono::system_clock::now();
    if (_verb) {
        std::cout << "Print x before Sim" << std::endl;
        print_x<<<1, 1>>>(d_x, (STEPS+1), n_sim_, state_dim);
        cudaDeviceSynchronize();
    }

    sim_gpu_kernel_<<<1 + n_sim_/SIZE, SIZE>>>(d_models, n_sim_, d_e, d_cost, rng_states);
    cudaDeviceSynchronize();

    //t2 = std::chrono::system_clock::now();
    //fp_ms = t2 - t1;
    //delta = fp_ms.count();
    //std::cout << "Done" << std::endl;

    //std::cout << "Sim execution time: " << delta << "ms" << std::endl;
    if (_verb) {
        std::cout << "Print x after Sim" << std::endl;
        print_x<<<1, 1>>>(d_x, (STEPS+1), n_sim_, state_dim);
        cudaDeviceSynchronize();
    }
    /*
    std::cout << "Print u" << std::endl;
    print_u<<<1, 1>>>(d_u, STEPS, act_dim);
    cudaDeviceSynchronize();
    std::cout << "Print e" << std::endl;
    print_e<<<1, 1>>>(d_e, STEPS, n_sim_, act_dim);
    cudaDeviceSynchronize();*/

    // find min cost
    //std::cout << "Compute min cost... : " << std::flush;

    //t1 = std::chrono::system_clock::now();

    min_beta();
    if (_verb) {
        print_beta<<<1, 1>>>(d_beta, 1);
        std::cout << std::endl;
    }
    cudaDeviceSynchronize();
    //t2 = std::chrono::system_clock::now();
    //fp_ms = t2 - t1;
    //delta = fp_ms.count();
    //std::cout << "Done" << std::endl;
    //std::cout << "Beta execution time: " << delta << "ms" << std::endl;
    //std::cout << "Compute nabla... : " << std::flush;
    //t1 = std::chrono::system_clock::now();

    exp();
    if (_verb) {
        print_exp<<<1, 1>>>(d_exp, n_sim_);
        std::cout << std::endl;
    }
    cudaDeviceSynchronize();
    //t2 = std::chrono::system_clock::now();
    //fp_ms = t2 - t1;
    //delta = fp_ms.count();
    //std::cout << "Exp execution time: " << delta << "ms" << std::endl;
    //t1 = std::chrono::system_clock::now();

    nabla();
    if (_verb) {
        print_nabla<<<1, 1>>>(d_nabla, 1);
        std::cout << std::endl;
    }
    cudaDeviceSynchronize();
    //t2 = std::chrono::system_clock::now();
    //fp_ms = t2 - t1;
    //delta = fp_ms.count();
    //std::cout << "Done" << std::endl;
    //std::cout << "Nabla execution time: " << delta << "ms" << std::endl;
    //compute weights
    //std::cout << "Compute weights... : " << std::flush;
    //t1 = std::chrono::system_clock::now();


    weights();
    if (true) {
        print_sum_weights<<<1, 1>>>(d_weights, n_sim_);
        std::cout << std::endl;
    }
    cudaDeviceSynchronize();

    //t2 = std::chrono::system_clock::now();
    //fp_ms = t2 - t1;
    //delta = fp_ms.count();

    //std::cout << "Done" << std::endl;

    //std::cout << "Weights execution time: " << delta << "ms" << std::endl;

    //std::cout << "compute new actions... : " << std::flush;

    //t1 = std::chrono::system_clock::now();
    // CURRENT BOTTLENECK!!!
    update_act();
    //t2 = std::chrono::system_clock::now();
    //fp_ms = t2 - t1;
    //delta = fp_ms.count();
    //std::cout << "Done" << std::endl;
    //std::cout << "Update action execution time: " << delta << "ms" << std::endl;
    //std::cout << "send action... : " << std::flush;

    if (_verb) {
        print_u<<<1, 1>>>(d_u, STEPS, act_dim);
        std::cout << std::endl;
    }
    cudaDeviceSynchronize();

    //t1 = std::chrono::system_clock::now();
    CUDA_CALL_CONST(cudaMemcpy(next_act, d_u, sizeof(float)*act_dim, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    //t2 = std::chrono::system_clock::now();
    //fp_ms = t2 - t1;
    //delta = fp_ms.count();
    //std::cout << "Done" << std::endl;
    //std::cout << "Copy act execution time: " << delta << "ms" << std::endl;
    //std::cout << "shift and init... : " << std::flush;


    //t1 = std::chrono::system_clock::now();
    shift_act<<<1 + n_sim_/SIZE, SIZE>>>(d_u, d_u_swap, act_dim, STEPS);
    CUDA_CALL_CONST(cudaMemcpy(d_u, d_u_swap, sizeof(float)*act_dim*STEPS, cudaMemcpyDeviceToDevice));
    //t2 = std::chrono::system_clock::now();
    //fp_ms = t2 - t1;
    //delta = fp_ms.count();
    //std::cout << "Done" << std::endl;
    //std::cout << "Shift & Init execution time: " << delta << "ms" << std::endl;

    cudaDeviceSynchronize();

}

void PointMassModel::memcpy_set_data (float* x, float* u, float* goal, float* w) {
    CUDA_CALL_CONST(cudaMemcpy(d_x_i, x, sizeof(float)*state_dim, cudaMemcpyHostToDevice));
    CUDA_CALL_CONST(cudaMemcpy(d_u, u, sizeof(float)*act_dim*steps_, cudaMemcpyHostToDevice));
    CUDA_CALL_CONST(cudaMemcpy(d_g, goal, sizeof(float)*state_dim, cudaMemcpyHostToDevice));
    CUDA_CALL_CONST(cudaMemcpy(d_w, w, sizeof(float)*state_dim, cudaMemcpyHostToDevice));
    std::cout << "Setting inital state of the sims... : " << std::flush;
    set_data<<<1 + n_sim_/256, 256>>>(d_models,
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

void PointMassModel::memcpy_get_data (float* x_all, float* e) {
    CUDA_CALL_CONST(cudaMemcpy(x_all, d_x, bytes_, cudaMemcpyDeviceToHost));
    CUDA_CALL_CONST(cudaMemcpy(e, d_e, sizeof(float)*n_sim_*act_dim*steps_, cudaMemcpyDeviceToHost));
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
    std::cout << "WTF" << std::endl;
    CUDA_CALL_CONST(cudaMemcpy(x, d_x, sizeof(float)*(steps_+1)*n_sim_*state_dim, cudaMemcpyDeviceToHost));
    std::cout << "WTF" << std::endl;
    CUDA_CALL_CONST(cudaMemcpy(u, d_u, sizeof(float)*steps_*act_dim, cudaMemcpyDeviceToHost));
    std::cout << "WTF" << std::endl;
    CUDA_CALL_CONST(cudaMemcpy(e, d_e, sizeof(float)*n_sim_*steps_*act_dim, cudaMemcpyDeviceToHost));
    std::cout << "WTF" << std::endl;
    CUDA_CALL_CONST(cudaMemcpy(cost, d_cost, sizeof(float)*n_sim_, cudaMemcpyDeviceToHost));
    std::cout << "WTF" << std::endl;
    CUDA_CALL_CONST(cudaMemcpy(beta, d_beta, sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "WTF" << std::endl;
    CUDA_CALL_CONST(cudaMemcpy(nabla, d_nabla, sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "WTF" << std::endl;
    CUDA_CALL_CONST(cudaMemcpy(w, d_weights, sizeof(float)*n_sim_, cudaMemcpyDeviceToHost));
    std::cout << "WTF" << std::endl;
}

void PointMassModel::min_beta () {
    int _n_sim(n_sim_);
    // TB Size
    int BLOCK_SIZE = SIZE;

    // Grid Size (cut in half) (No padding)
    int GRID_SIZE = _n_sim / BLOCK_SIZE / 2 + 1;

    // THIS shouldn't change size during the controller iterations.
    // should verify this and then allocate data in the init to improve
    // computation time
    //std::cout << "Find min path " << " with: "
    //          << "GRID_SIZE: " << GRID_SIZE << ", "
    //          << "BLOCK_SIZE: " << BLOCK_SIZE << std::endl;


    if (GRID_SIZE == 1)
    {
        //std::cout << "Run only once" << std::endl;
        min_red << <1, BLOCK_SIZE >> > (d_cost, _d_beta, _n_sim);
    }
    else
    {
        //std::cout << "Run first" << std::endl;
        // insure at least one pass.
        min_red << <GRID_SIZE, BLOCK_SIZE >> > (d_cost, _d_beta, _n_sim);

        _n_sim = GRID_SIZE;
        GRID_SIZE = _n_sim / BLOCK_SIZE / 2 + 1 ;
        cudaDeviceSynchronize();
        while (GRID_SIZE - 1 > 1 )
        {
            //std::cout << "Mid" << std::endl;
            min_red << <GRID_SIZE, BLOCK_SIZE >> > (_d_beta, _d_beta, _n_sim);
            _n_sim = GRID_SIZE;
            GRID_SIZE = _n_sim / BLOCK_SIZE / 2 + 1 ;
            cudaDeviceSynchronize();
        }
        //std::cout << "Run last" << std::endl;
        min_red << <1, BLOCK_SIZE >> > (_d_beta, _d_beta, _n_sim);
    }

    cudaDeviceSynchronize();

    //std::cout << "Copy beta: " << std::endl;
    //float h_beta[1];
    CUDA_CALL_CONST(cudaMemcpy(d_beta, _d_beta, sizeof(float), cudaMemcpyDeviceToDevice));
    //CUDA_CALL_CONST(cudaMemcpy(h_beta, d_beta, sizeof(float), cudaMemcpyDeviceToHost));
    //std::cout << "beta: " << h_beta[0] << std::endl;
}

void PointMassModel::exp () {
    exp_red <<<1 + n_sim_/SIZE, SIZE>>> (d_exp, d_cost, d_lambda, d_beta, _d_nabla, n_sim_);
}

void PointMassModel::nabla () {
    int _n_sim(n_sim_);
    // TB Size
    int BLOCK_SIZE = SIZE;

    // Grid Size (cut in half) (No padding)
    int GRID_SIZE = _n_sim / BLOCK_SIZE / 2 + 1;

    //std::cout << "Find norm term " << " with: "
    //          << "GRID_SIZE: " << GRID_SIZE << ", "
    //          << "BLOCK_SIZE: " << BLOCK_SIZE << std::endl;

    if (GRID_SIZE == 1)
    {
    //    std::cout << "Run only once" << std::endl;
        sum_red <<<1, BLOCK_SIZE >>> (d_exp, _d_nabla, _n_sim);
    }
    else
    {
    //    std::cout << "Run first" << std::endl;
        // insure at least one pass.
        sum_red <<<GRID_SIZE, BLOCK_SIZE >>> (d_exp, _d_nabla, _n_sim);
        cudaDeviceSynchronize();

        _n_sim = GRID_SIZE;
        GRID_SIZE = _n_sim / BLOCK_SIZE / 2 + 1 ;

        while (GRID_SIZE - 1 > 1 )
        {
    //        std::cout << "Mid" << std::endl;
            sum_red <<<GRID_SIZE, BLOCK_SIZE >>> (_d_nabla, _d_nabla, _n_sim);
            _n_sim = GRID_SIZE;
            GRID_SIZE = _n_sim / BLOCK_SIZE / 2 + 1 ;
            cudaDeviceSynchronize();
        }
    //    std::cout << "Run last" << std::endl;
        sum_red <<<1, BLOCK_SIZE >>> (_d_nabla, _d_nabla, _n_sim);
    }

    cudaDeviceSynchronize();

    //std::cout << "Copy nabla: " << std::endl;
    //float h_nabla[1];
    CUDA_CALL_CONST(cudaMemcpy(d_nabla, _d_nabla, sizeof(float), cudaMemcpyDeviceToDevice));
    //CUDA_CALL_CONST(cudaMemcpy(h_nabla, d_nabla, sizeof(float), cudaMemcpyDeviceToHost));
    //std::cout << "nabla: " << h_nabla[0] << std::endl;

}

void PointMassModel::weights () {
    weights_kernel<<<1 + n_sim_/SIZE, SIZE>>>(d_cost, d_weights, d_lambda, d_beta, d_nabla, n_sim_);
    cudaDeviceSynchronize();
}

void PointMassModel::update_act () {
    int _n_sim(n_sim_);
    int BLOCK_SIZE = SIZE;
    int GRID_SIZE = _n_sim / BLOCK_SIZE / 2 / act_dim  + 1;
    float* v_r;
//    std::cout << "allocate data" << std::endl;
    CUDA_CALL_CONST(cudaMalloc((void**)&v_r, sizeof(float)*GRID_SIZE*act_dim));
    CUDA_CALL_CONST(cudaMemset((void*)v_r, 0, sizeof(float)*GRID_SIZE*act_dim));
//    std::cout << "Data allocated" << std::endl;

    // This is a problem, should parallelize this.
    for (int t=0; t < steps_; t++)
    {
        // TB Size
        int _n_sim(n_sim_);
        int BLOCK_SIZE = SIZE;
        // Grid Size (cut in half) (No padding)
        int GRID_SIZE = _n_sim / BLOCK_SIZE / 2 / act_dim + 1;


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
            update_act_kernel <<<1, BLOCK_SIZE >>> (v_r, d_weights, d_e, STEPS, t, act_dim, _n_sim);
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
            update_act_kernel <<<GRID_SIZE, BLOCK_SIZE >>> (v_r, d_weights, d_e, STEPS, t, act_dim, _n_sim);

            _n_sim = GRID_SIZE;
            GRID_SIZE = _n_sim / BLOCK_SIZE / 2 / act_dim  + 1 ;
            cudaDeviceSynchronize();

            //std::cout << "First : " << t << std::endl;
            //print_u<<<1, 1>>>(v_r, GRID_SIZE, act_dim);
            //cudaDeviceSynchronize();


            while (GRID_SIZE - 1 > 1 )
            {
                //std::cout << "Mid" << std::endl;
                sum_red << <GRID_SIZE, BLOCK_SIZE >> > (v_r, v_r, _n_sim);
                _n_sim = GRID_SIZE;
                GRID_SIZE = _n_sim / BLOCK_SIZE / 2 / act_dim  + 1 ;
                cudaDeviceSynchronize();

                /*
                std::cout << "Midle : " << t << std::endl;
                print_u<<<1, 1>>>(v_r, GRID_SIZE, act_dim);
                cudaDeviceSynchronize();*/


            }
            //std::cout << "Run last" << std::endl;
            //std::cout << "steps: " << STEPS << " t: " << t << " a_dim: " << act_dim << " n: " << _n_sim << std::endl;
            sum_red <<<1, BLOCK_SIZE >>> (v_r, v_r, _n_sim);
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
        copy_act<<< 1 + _n_sim/SIZE, SIZE >>>(d_u, v_r, t, act_dim);
        cudaDeviceSynchronize();
        //std::cout << "Done" << std::endl;

    }


    cudaDeviceSynchronize();
    CUDA_CALL_CONST(cudaFree((void*)v_r));
}

void PointMassModel::set_x (float* h_x) {
    CUDA_CALL_CONST(cudaMemcpy(d_x_i, h_x, sizeof(float)*state_dim, cudaMemcpyHostToDevice));
    set_x_kernel<<<1 + n_sim_/SIZE, SIZE>>>(d_models, d_x_i, n_sim_);
    CUDA_CALL_CONST(cudaDeviceSynchronize());
}

__global__ void sim_gpu_kernel_ (PointMassModelGpu* d_models,
                                int n_sim_,
                                float* d_u,
                                float* cost,
                                curandState* rng_states) {

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < n_sim_){
        /* local copy of rng state for faster generation. */
        curandState localState = rng_states[tid];
        cost[tid] = d_models[tid].run(&localState);

        /* copy back the state ohterwise the rng state is not working. */
        rng_states[tid] = localState;
    }
}

__global__ void exp_red (float* out,
                         float* cost,
                         float* lambda,
                         float* beta,
                         float* nabla,
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
    const int a_dim(act_dim);

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

__global__ void print_x (float* x, int steps, int samples, int s_dim) {
    int id(0);
    for (int k = 0; k < samples; k++)
    {
        for (int j=0; j < steps; j++)
        {
            for (int i=0; i < s_dim; i++)
            {
                id = k*steps*s_dim + j*s_dim + i;
                printf("x[%d]: %f ", id, x[id]);
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("\n");
}

__global__ void print_u (float* u, int steps, int a_dim) {
    int id(0);
    for (int j=0; j < steps; j++)
    {
        for (int i=0; i < a_dim; i++)
        {
            id = j*a_dim + i;
            printf("u[%d]: %f ", id, u[id]);
        }
        printf("\n");
    }
    printf("\n");
}

__global__ void print_e (float* e, int steps, int samples, int a_dim) {
    int id(0);
    for (int k = 0; k < samples; k++)
    {
        for (int j=0; j < steps; j++)
        {
            for (int i=0; i < a_dim; i++)
            {
                id = k*steps*a_dim + j*a_dim + i;
                printf("e[%d]: %f ", id, e[id]);
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("\n");
}

__global__ void print_beta (float* beta, int size) {
    for (int i=0; i < size; i++)
    {
        printf("Beta[%d]: %f\n", i, beta[i]);
    }
}

__global__ void print_nabla (float* nabla, int size) {
    for (int i=0; i < size; i++)
    {
        printf("Nabla[%d]: %f\n", i, nabla[i]);
    }
}

__global__ void print_lam (float* lam, int size) {
    for (int i=0; i < size; i++)
    {
        printf("lambda[%d]: %f\n",i, lam[i]);
    }
}

__global__ void print_weights (float* weights, int samples) {
    for (int i=0; i < samples; i++)
    {
        printf("weights[%d]: %f\n", i, weights[i]);
    }
}

__global__ void print_costs (float* costs, int samples) {
    for (int i=0; i < samples; i++)
    {
        printf("costs[%d]: %f\n", i, costs[i]);
    }
}

__global__ void print_exp (float* exp, int samples) {
    for (int i=0; i<samples; i++)
    {
        printf("exp[%d]: %f\n", i, exp[i]);
    }
}

__global__ void print_sum_weights (float* w, int samples) {
    float sum(0);
    for (int i = 0; i < samples; i++) {
        sum += w[i];
    }
    printf("weight sum: %f\n", sum);
}
