#include "point_mass.hpp"
#include "cost.hpp"
#include <math.h>
#include <assert.h>

#define MAX_N 2000
#define T 2000
#define A 16
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
    for (int k=0; k < n; n++) {
        for (int j=0; j < t; j++) {
            for (int i=0; i < a; i++) {
                
            }
        }
    }
}

void test_update_act () {
    for (int n=0; n < MAX_N; n++) {
        for (int t=0; t < MAX_T; t++) {
            for (int a=0; a < MAX_A; a++) {

                // init data
                float* h_u = (float*) malloc(sizeof(float)*t*a);
                float* h_w = (float*) malloc(sizeof(float)*n);
                float* h_e = (float*) malloc(sizeof(float)*t*a*n);

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
            }
        }
    }
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
