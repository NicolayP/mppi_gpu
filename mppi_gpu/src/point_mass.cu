#include "point_mass.hpp"

#include <iostream>
#include <math.h>

#define SIZE 8
#define SHMEM_SIZE 8


__host__ __device__ PointMassModelGpu::PointMassModelGpu(){
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
                                                 float lambda,
                                                 int id)
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
    glob_e = e;

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
    _cost = Cost(_w, _x_size, _g, _x_size, lambda, _inv_s, u_size);
    _c = 0;
    _id = id;
    return;
}

__host__ __device__ void PointMassModelGpu::step(curandState* state)
{

#ifdef __CUDA_ARCH__
    _e[(_t-1)*_u_size] += curand_normal(state);
    _e[(_t-1)*_u_size + 1] += curand_normal(state);
    //printf("_e[%d]: %f\n", (_t-1)*_u_size, _e[(_t-1)*_u_size]);
    //printf("_e[%d]: %f\n", (_t-1)*_u_size + 1, _e[(_t-1)*_u_size + 1]);

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
    _c += _cost.step_cost(&_x[_t*_x_size], &_u[(_t-1)*_u_size], &_e[(_t-1)*_u_size], _id, _t);
    printf("_c[%d]: %f\n", _id, _c);
}

__host__ __device__ float PointMassModelGpu::run(curandState* state)
{
    for (_t = 1; _t < _tau; _t++ ){
        step(state);
    }
    _c += _cost.final_cost(&_x[(_tau-1)*_x_size], _id);
    printf("_c[%d, %d]: %f\n", (_tau-1), _id, _c);
    // save action to global pointer
    save_e();
    return _c;
}

__host__ __device__ void PointMassModelGpu::save_e()
{
    for (int t = 0; t < _tau; t++)
    {
        glob_e[(t-1)*_u_size] = _e[(t-1)*_u_size];
        glob_e[(t-1)*_u_size + 1] = _e[(t-1)*_u_size + 1];
    }
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


PointMassModel::PointMassModel(int nb_sim, int steps, float dt)
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
    CUDA_CALL_CONST(cudaMalloc((void**)&d_x, sizeof(float)*n_sim_*steps_*state_dim));
    // set the memory with 0s.
    CUDA_CALL_CONST(cudaMemset((void*)d_x, 0, sizeof(float)*n_sim_*steps_*state_dim));
    // allocate space for action.
    CUDA_CALL_CONST(cudaMalloc((void**)&d_e, sizeof(float)*n_sim_*steps_*act_dim));

    CUDA_CALL_CONST(cudaMemset((void*)d_e, 0, sizeof(float)*n_sim_*steps_*act_dim));

    CUDA_CALL_CONST(cudaMalloc((void**)&d_u, sizeof(float)*steps_*act_dim));

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

PointMassModel::~PointMassModel()
{
    CUDA_CALL_CONST(cudaFree((void*)d_x));
    CUDA_CALL_CONST(cudaFree((void*)d_x_i));
    CUDA_CALL_CONST(cudaFree((void*)d_u));
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

void PointMassModel::sim()
{
    // launch 1 thread per simulation. Can later consider to
    // add dimensions to the block and thread of the kernel
    // to // enven more the code inside the simulation.
    // using blockDim.y & blockDim.z, blockIdx.y & blockIdx.x
    // and threadIdx.y & threadIdx.z.
    std::cout << "Running simulations... : " << std::endl;
    sim_gpu_kernel_<<<1 + n_sim_/SIZE, SIZE>>>(d_models, n_sim_, d_e, d_cost, rng_states);
    cudaDeviceSynchronize();
    std::cout << "Print x" << std::endl;
    print_x<<<1, 1>>>(d_x, STEPS, n_sim_, state_dim);
    cudaDeviceSynchronize();
    std::cout << "Print u" << std::endl;
    print_u<<<1, 1>>>(d_u, STEPS, act_dim);
    cudaDeviceSynchronize();
    std::cout << "Print e" << std::endl;
    print_e<<<1, 1>>>(d_e, STEPS, n_sim_, act_dim);
    cudaDeviceSynchronize();
    std::cout << "Print cost" << std::endl;
    print_costs<<<1, 1>>>(d_cost, n_sim_);
    cudaDeviceSynchronize();
    std::cout << "Done" << std::endl;


    // find min cost
    std::cout << "Compute min cost... : " << std::endl;
    min_beta();
    print_beta<<<1, 1>>>(d_beta, 1);
    cudaDeviceSynchronize();
    std::cout << "Done" << std::endl;

    std::cout << "Compute nabla... : " << std::endl;
    exp();
    print_exp<<<1, 1>>>(d_exp, n_sim_);
    cudaDeviceSynchronize();

    nabla();
    print_nabla<<<1, 1>>>(d_nabla, 1);
    cudaDeviceSynchronize();
    std::cout << "Done" << std::endl;

    //compute weights
    std::cout << "Compute weights... : " << std::flush;
    weights();
    print_weights<<<1, 1>>>(d_weights, n_sim_);
    cudaDeviceSynchronize();
    std::cout << "Done" << std::endl;

    std::cout << "compute new actions... : " << std::endl;
    update_act();
    print_u<<<1, 1>>>(d_u, STEPS, act_dim);
    cudaDeviceSynchronize();
    std::cout << "Done" << std::endl;

    /*
    std::cout << "send action... : " << std::flush;
    std::cout << "Not Implemented yet" << std::endl;

    std::cout << "shift and init... : " << std::flush;
    std::cout << "Not Implemented yet" << std::endl;
    */
    cudaDeviceSynchronize();
}

void PointMassModel::memcpy_set_data(float* x, float* u, float* goal, float* w)
{
    CUDA_CALL_CONST(cudaMemcpy(d_x_i, x, sizeof(float)*n_sim_*state_dim, cudaMemcpyHostToDevice));
    CUDA_CALL_CONST(cudaMemcpy(d_e, u, sizeof(float)*n_sim_*act_dim*steps_, cudaMemcpyHostToDevice));
    CUDA_CALL_CONST(cudaMemcpy(d_g, goal, sizeof(float)*state_dim, cudaMemcpyHostToDevice));
    CUDA_CALL_CONST(cudaMemcpy(d_w, w, sizeof(float)*state_dim, cudaMemcpyHostToDevice));
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
    CUDA_CALL_CONST(cudaMemcpy(x_all, d_x, bytes_, cudaMemcpyDeviceToHost));
    CUDA_CALL_CONST(cudaMemcpy(e, d_e, sizeof(float)*n_sim_*act_dim*steps_, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
}

void PointMassModel::get_inf(float* x,
                             float* u,
                             float* e,
                             float* cost,
                             float* beta,
                             float* nabla,
                             float* w)
{
    // get all the info to look at the results and debug if necessary.
    CUDA_CALL_CONST(cudaMemcpy(x, d_x, sizeof(float)*steps_*n_sim_*state_dim, cudaMemcpyDeviceToHost));
    CUDA_CALL_CONST(cudaMemcpy(u, d_u, sizeof(float)*steps_*act_dim, cudaMemcpyDeviceToHost));
    CUDA_CALL_CONST(cudaMemcpy(e, d_e, sizeof(float)*n_sim_*steps_*act_dim, cudaMemcpyDeviceToHost));
    CUDA_CALL_CONST(cudaMemcpy(cost, d_cost, sizeof(float)*n_sim_, cudaMemcpyDeviceToHost));
    CUDA_CALL_CONST(cudaMemcpy(beta, d_beta, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CALL_CONST(cudaMemcpy(nabla, d_nabla, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CALL_CONST(cudaMemcpy(w, d_weights, sizeof(float)*n_sim_, cudaMemcpyDeviceToHost));
}

void PointMassModel::min_beta()
{
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

void PointMassModel::exp()
{
    exp_red <<<1 + n_sim_/SIZE, SIZE>>> (d_exp, d_cost, d_lambda, d_beta, _d_nabla, n_sim_);
}

void PointMassModel::nabla()
{
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

void PointMassModel::weights()
{
    weights_kernel<<<1 + n_sim_/SIZE, SIZE>>>(d_cost, d_weights, d_lambda, d_beta, d_nabla, n_sim_);
    cudaDeviceSynchronize();
}

void PointMassModel::update_act_id(){
    int n_sim(n_sim_);
    //int BLOCK_SIZE = SIZE;
    //int GRID_SIZE = n_sim / BLOCK_SIZE / 2 / act_dim + 1;
    for (int t=0; t < steps_; t++)
    {
        int BLOCK_SIZE = SIZE;
        int GRID_SIZE = n_sim / BLOCK_SIZE / 2/ act_dim + 1;
        std::cout << "GRID_SIZE: " << GRID_SIZE << std::endl;

        if (GRID_SIZE==1)
        {
            update_act_id_kernel<<<1 , BLOCK_SIZE>>>(STEPS, t, act_dim, n_sim);
        }
        else
        {
            update_act_id_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(STEPS, t, act_dim, n_sim);

            /*n_sim = GRID_SIZE;
            GRID_SIZE = n_sim / BLOCK_SIZE / 2 / act_dim + 1;
            while (GRID_SIZE - 1 > 1)
            {
                update_act_id_kernel<<GRID_SIZE, BLOCK_SIZE>>>(STEPS, t, act_dim, _n_sim);
                n_sim = GRID_SIZE;
                GRID_SIZE = n_sim / BLOCK_SIZE / 2 / act_dim + 1;
            }
            update_act_id_kernel<<<1, BLOCK_SIZE>>>(STEPS, t, act_dim, _n_sim);
            */
        }
    }
}

void PointMassModel::update_act()
{
    int _n_sim(n_sim_);
    int BLOCK_SIZE = SIZE;
    int GRID_SIZE = _n_sim / BLOCK_SIZE / 2 / act_dim  + 1;
    float* v_r;
//    std::cout << "allocate data" << std::endl;
    CUDA_CALL_CONST(cudaMalloc((void**)&v_r, sizeof(float)*GRID_SIZE*act_dim));
//    std::cout << "Data allocated" << std::endl;
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
            //std::cout << "Run only once" << std::endl;
            //std::cout << "steps: " << STEPS << " t: " << t << " a_dim: " << act_dim << " n: " << _n_sim << std::endl;
            update_act_kernel <<<1, BLOCK_SIZE >>> (v_r, d_weights, d_e, STEPS, t, act_dim, _n_sim);
            cudaDeviceSynchronize();
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

            while (GRID_SIZE - 1 > 1 )
            {
                //std::cout << "Mid" << std::endl;
                sum_red << <GRID_SIZE, BLOCK_SIZE >> > (v_r, v_r, _n_sim);
                _n_sim = GRID_SIZE;
                GRID_SIZE = _n_sim / BLOCK_SIZE / 2 / act_dim  + 1 ;
                cudaDeviceSynchronize();
            }
            //std::cout << "Run last" << std::endl;
            //std::cout << "steps: " << STEPS << " t: " << t << " a_dim: " << act_dim << " n: " << _n_sim << std::endl;
            sum_red <<<1, BLOCK_SIZE >>> (v_r, v_r, _n_sim);
            cudaDeviceSynchronize();
        }
        std::cout << "Copy at t: " << t << std::endl;

        copy_act<<< 1, BLOCK_SIZE >>>(d_u, v_r, t, act_dim);
        cudaDeviceSynchronize();
        std::cout << "Done" << std::endl;

    }

    float h_u[STEPS*2];
    CUDA_CALL_CONST(cudaMemcpy(h_u, d_u, sizeof(float)*steps_*act_dim, cudaMemcpyDeviceToHost));
    for (int j=0; j<act_dim; j++)
    {
        std::cout << "h_u[" << j << "]: ";
        for (int i = 0; i < steps_; i++)
        {
            std::cout << h_u[j*act_dim + i] << " ";
        }
        std::cout << std::endl;
    }

    cudaDeviceSynchronize();
    CUDA_CALL_CONST(cudaFree((void*)v_r));
}

__global__ void sim_gpu_kernel_(PointMassModelGpu* d_models,
                                int n_sim_,
                                float* d_u,
                                float* cost,
                                curandState* rng_states)
{

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < n_sim_){
        /* local copy of rng state for faster generation. */
        curandState localState = rng_states[tid];
        cost[tid] = d_models[tid].run(&localState);
        //printf("tid: %d, cost[tid]: %f\n", tid, cost[tid]);
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

__global__ void exp_red(float* out, float* cost, float* lambda, float* beta, float* nabla, int size)
{
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if(tid < size)
    {
        out[tid] = expf(-lambda[0] * (cost[tid] - beta[0]));
    }

}

__global__ void min_red(float* v, float* beta, int n)
{
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

__global__ void sum_red_exp(float* v, float* lambda_1, float* beta, float* v_r, int n)
{
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

__global__ void sum_red(float* v, float* v_r, int n)
{
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

__global__ void weights_kernel(float* v, float* v_r, float* lambda_1, float* beta, float* nabla_1, int n)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n){
        v_r[tid] = 1.0/nabla_1[0] * expf(-(1.0/lambda_1[0])*(v[tid] - beta[0]));
        //printf("weight[%d]: %f, nabla_1: %f, 1/nabla_1: %f\n", tid, v_r[tid], nabla_1[0], 1/nabla_1[0]);
    }
}

__global__ void copy_act(float* u, float* tmp, int t, int act_dim)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    //printf("WHY???\n");
    //printf("%d\n", tid);
    if (tid < act_dim)
        //printf("u[%d]: %f\n", (tid + t*act_dim), u[tid + t*act_dim]);
        u[tid + t*act_dim] += tmp[tid];
}
// First implementation. Usually T << K so parallelize over K first.
__global__ void update_act_kernel(float* v_r,
                                  float* w,
                                  float* e,
                                  int steps,
                                  int t,
                                  int act_dim,
                                  int n)
{
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

__global__ void print_x(float* x, int steps, int samples, int s_dim)
{
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

__global__ void print_u(float* u, int steps, int a_dim)
{
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

__global__ void print_e(float* e, int steps, int samples, int a_dim)
{
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

__global__ void print_beta(float* beta, int size)
{
    for (int i=0; i < size; i++)
    {
        printf("Beta[%d]: %f\n", i, beta[i]);
    }
}

__global__ void print_nabla(float* nabla, int size)
{
    for (int i=0; i < size; i++)
    {
        printf("Nabla[%d]: %f\n", i, nabla[i]);
    }
}

__global__ void print_lam(float* lam, int size)
{
    for (int i=0; i < size; i++)
    {
        printf("lambda[%d]: %f\n",i, lam[i]);
    }
}

__global__ void print_weights(float* weights, int samples)
{
    for (int i=0; i < samples; i++)
    {
        printf("weights[%d]: %f\n", i, weights[i]);
    }
}

__global__ void print_costs(float* costs, int samples)
{
    for (int i=0; i < samples; i++)
    {
        printf("costs[%d]: %f\n", i, costs[i]);
    }
}

__global__ void print_exp(float* exp, int samples)
{
    for (int i=0; i<samples; i++)
    {
        printf("exp[%d]: %f\n", i, exp[i]);
    }
}

__global__ void update_act_id_kernel(int steps, int t, int act_dim, int samples)
{
    int i = blockIdx.x * (blockDim.x*2) * act_dim + t * act_dim + threadIdx.x*steps*act_dim;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int shift = blockDim.x*steps*act_dim;
    int max_size = samples*act_dim*steps;

    if (i + shift < max_size)
    {
        for (int j = 0; j < act_dim; j++)
        {
            printf("Assigne 2 bl: %d, th: %d, i: %d, k: %d, shift: %d, max_size: %d, t: %d, a_dim: %d, samples: %d, stpes: %d, p[%d], w[%d], e[%d], e[%d]\n",
                blockIdx.x,
                threadIdx.x,
                i,
                k,
                shift,
                max_size,
                t,
                act_dim,
                samples,
                steps,
                threadIdx.x*act_dim + j,
                k,
                i+j,
                i + blockDim.x*steps*act_dim + j);
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
            printf("Assigne 1 bl: %d, th: %d, i: %d, k: %d, shift: %d, max_size: %d, t: %d, a_dim: %d, samples: %d, stpes: %d, p[%d], w[%d], e[%d] \n",
                blockIdx.x,
                threadIdx.x,
                i,
                k,
                shift,
                max_size,
                t,
                act_dim,
                samples,
                steps,
                threadIdx.x*act_dim + j,
                k,
                i+j);
        }
    }
    else
    {
        for (int j = 0; j < act_dim; j++)
        {
            printf("Assign 0 p[%d]\n", threadIdx.x*act_dim + j);
        }
    }
    //partial_sum[threadIdx.x] = v[i] + v[i + blockDim.x];
    __syncthreads();

    // Start at 1/2 block stride and divide by two each iteration
    for (int s = blockDim.x * act_dim / 2; s > 1; s >>= 1)
    {
        // Each thread does work unless it is further than the stride
        if (threadIdx.x*act_dim < s)
        {
            for (int j = 0; j < act_dim; j++)
            {
                printf("bl: %d, th: %d, i: %d, k: %d, shift: %d, max_size: %d, t: %d, a_dim: %d, samples: %d, stpes: %d, s: %d, p[%d], p[%d]\n",
                    blockIdx.x,
                    threadIdx.x,
                    i,
                    k,
                    shift,
                    max_size,
                    t,
                    act_dim,
                    samples,
                    steps,
                    s,
                    threadIdx.x*act_dim + j,
                    threadIdx.x*act_dim + j + s);
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
            printf("v_r[%d] p[%d]\n", blockIdx.x*act_dim + j, j);
        }
    }
}

__global__ void set_data_(PointMassModelGpu* d_models,
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
                            lambda[0],
                            tid);
    }
}
