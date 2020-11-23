#include "point_mass.hpp"
#include "cost.hpp"
#include <math.h>
#include <assert.h>

#define MAX_N 60
#define MAX_T 100
#define MAX_A 2
#define SIZE 256

void test_exp () {
    for (int n=0; n < MAX_N; n++) {
        float* h_out = (float*) malloc(sizeof(float)*n);
        float* h_cost = (float*) malloc(sizeof(float)*n);
        for (int i=0; i < n; i++) {
            h_cost[i] = i;
        }
        float h_lambda[1] = {1.0};
        float h_beta[1] = {0.25};
        float h_nabla[1] = {0.5};

        float* d_out;
        float* d_cost;
        float* d_lambda;
        float* d_beta;
        float* d_nabla;

        cudaMalloc((void**) &d_out, sizeof(float)*n);
        cudaMalloc((void**) &d_cost, sizeof(float)*n);
        cudaMalloc((void**) &d_lambda, sizeof(float));
        cudaMalloc((void**) &d_beta, sizeof(float));
        cudaMalloc((void**) &d_nabla, sizeof(float));

        cudaMemcpy(d_cost, h_cost, sizeof(float)*n, cudaMemcpyHostToDevice);
        cudaMemcpy(d_lambda, h_lambda, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_beta, h_beta, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_nabla, h_nabla, sizeof(float), cudaMemcpyHostToDevice);

        exp_red<<<1 + n/SIZE, SIZE>>>(d_out, d_cost, d_lambda, d_beta, d_nabla, n);

        cudaMemcpy(h_out, d_out, sizeof(float)*n, cudaMemcpyDeviceToHost);

        float tmp;
        for (int i=0; i < n; i++) {
            tmp = exp(-h_lambda[0] * (h_cost[i] - h_beta[0]));
            assert( (tmp - h_out[i]) < TOL);
        }

        cudaFree(d_out);
        cudaFree(d_cost);
        cudaFree(d_lambda);
        cudaFree(d_beta);
        cudaFree(d_nabla);

        free(h_out);
        free(h_cost);
    }
    std::cout << "Test exp passed" << std::endl;
}

void test_beta () {

}

void test_nalba () {

}

void test_sum_red () {

}

void test_weight () {

}

void init_update_act_data (float* u, float* w, float* e, int n, int t, int a) {
    for (int k=0; k < n; k++) {
        for (int j=0; j < t; j++) {
            for (int i=0; i < a; i++) {
                e[k*t*a + j*a + i] = 0.25*(k*t*a + j*a + i);
            }
        }
    }

    for (int k=0; k < n; k++) {
        w[k] = 0.5*k;
    }

    for (int j=0; j < t; j++) {
        for (int i=0; i < a; i++) {
            u[j*a + i] = 0.75*(j*a + i);
        }
    }
}

void update_act_cpu (float* u, float* w, float* e, int n, int t, int a) {
    for (int k=0; k < n; k++) {
        for (int j=0; j < t; j++) {
            for (int i=0; i < a; i++) {
                u[j*a + i] += w[k]*e[k*t*a + j*a + i];
            }
        }
    }
}

void update_act_gpu (float* u, float* w, float* e, int n, int t, int a) {
    int BLOCK_SIZE = SIZE;
    int GRID_SIZE = n / BLOCK_SIZE / 2 / a  + 1;
    float* v_r;
//    std::cout << "allocate data" << std::endl;
    CUDA_CALL_CONST(cudaMalloc((void**)&v_r, sizeof(float)*GRID_SIZE*a));
//    std::cout << "Data allocated" << std::endl;

    // This is a problem, should parallelize this.
    for (int j=0; j < t; j++)
    {
        // TB Size
        int BLOCK_SIZE = SIZE;
        // Grid Size (cut in half) (No padding)
        int GRID_SIZE = n / BLOCK_SIZE / 2 / a + 1;


//        std::cout << "Starting update at t: " << t << " with: "
//                  << "GRID_SIZE: " << GRID_SIZE << ", "
//                  << "BLOCK_SIZE: " << BLOCK_SIZE << std::endl;
        if (GRID_SIZE == 1)
        {
            //std::cout << "Run only once" << std::endl;
            //std::cout << "steps: " << STEPS << " t: " << t << " a_dim: " << act_dim << " n: " << _n_sim << std::endl;
            update_act_kernel <<<1, BLOCK_SIZE >>> (v_r, w, e, t, j, a, n);
            cudaDeviceSynchronize();
        }
        else
        {
            //std::cout << "Run first" << std::endl;
            // insure at least one pass.
            //std::cout << "steps: " << STEPS << " t: " << t << " a_dim: " << act_dim << " n: " << _n_sim << std::endl;
            update_act_kernel <<<GRID_SIZE, BLOCK_SIZE >>> (v_r, w, e, t, j, a, n);

            n = GRID_SIZE;
            GRID_SIZE = n / BLOCK_SIZE / 2 / a  + 1 ;
            cudaDeviceSynchronize();

            while (GRID_SIZE - 1 > 1 )
            {
                //std::cout << "Mid" << std::endl;
                sum_red << <GRID_SIZE, BLOCK_SIZE >> > (v_r, v_r, n);
                n = GRID_SIZE;
                GRID_SIZE = n / BLOCK_SIZE / 2 / a  + 1 ;
                cudaDeviceSynchronize();
            }
            //std::cout << "Run last" << std::endl;
            //std::cout << "steps: " << STEPS << " t: " << t << " a_dim: " << act_dim << " n: " << _n_sim << std::endl;
            sum_red <<<1, BLOCK_SIZE >>> (v_r, v_r, n);
            cudaDeviceSynchronize();
        }
    //std::cout << "Copy at t: " << t << std::endl;

    copy_act<<< 1 + n/SIZE, SIZE >>>(u, v_r, j, a);
    cudaDeviceSynchronize();
    //std::cout << "Done" << std::endl;
    }

    cudaDeviceSynchronize();
    CUDA_CALL_CONST(cudaFree(v_r));

}

bool comp_update_act (float* d_u, float* h_u, int n, int t, int a) {
    for (int j=0; j < t; j++) {
        for (int i=0; i < a; i++) {
            if(! (fabs(d_u[j*a + i] - h_u[j*a + i]) < TOL)) {
                return false;
            }
        }
    }
    return true;
}

void test_update_act () {
    int a = MAX_A;
    for (int n=1; n < MAX_N; n++) {
        for (int t=1; t < MAX_T; t++) {

            // init data
            float* h_u = (float*) malloc(sizeof(float)*t*a);
            float* h_w = (float*) malloc(sizeof(float)*n);
            float* h_e = (float*) malloc(sizeof(float)*t*a*n);

            float* o_u = (float*) malloc(sizeof(float)*t*a);

            init_update_act_data(h_u, h_w, h_e, n, t, a);

            float* d_u;
            float* d_w;
            float* d_e;

            cudaMalloc((void**) &d_u, sizeof(float)*t*a);
            cudaMalloc((void**) &d_w, sizeof(float)*n);
            cudaMalloc((void**) &d_e, sizeof(float)*t*a*n);

            cudaMemcpy(d_u, h_u, sizeof(float)*t*a, cudaMemcpyHostToDevice);
            cudaMemcpy(d_w, h_w, sizeof(float)*n, cudaMemcpyHostToDevice);
            cudaMemcpy(d_e, h_e, sizeof(float)*t*a*n, cudaMemcpyHostToDevice);

            update_act_cpu(h_u, h_w, h_e, n, t, a);
            //std::cout << "One " << std::endl;
            update_act_gpu(d_u, d_w, d_e, n, t, a);
            //std::cout << "Two " << std::endl;
            cudaMemcpy(o_u, d_u, sizeof(float)*t*a, cudaMemcpyDeviceToHost);

            if (!comp_update_act(o_u, h_u, n, t, a)) {
                std::cout << "Test failed for: n = " << n
                          << " t = " << t << std::endl;
            }

            free(h_u);
            free(h_w);
            free(h_e);
            free(o_u);
            cudaFree(d_u);
            cudaFree(d_w);
            cudaFree(d_e);

        }
    }
    std::cout << "Update Act test passed" << std::endl;
}

int main(int argc, char const *argv[]) {
    test_exp();
    test_beta();
    test_nalba();
    test_sum_red();
    test_weight();
    test_update_act();
    return 0;
}
