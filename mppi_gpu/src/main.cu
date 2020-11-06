#include "point_mass.hpp"
#include <iostream>
#include <fstream>

#include <assert.h>

#include <cstdio>
#include <ctime>
#include <chrono>
#include <unistd.h>

/*
* Same as previous example only this time the parallized
* objects contain a pointer to the data. This will show
* us how to implment a recording of a path for MPPI.
*
* This examples run a simple linear simulation in parallel.
* All this is wrapped in a model class which will later be our
* controller. The Wrapper creates simulation classes on the device.
* It then sets the input data on the classes (can be run in // too).
* finally it runs the simulation in parallel and collects the data
* on the host device.
*/

/*
bool test_sim_gpu(float* data,
                  int samples,
                  int state_dim,
                  int act_dim,
                  float dt,
                  float act,
                  float init){

    std::chrono::time_point<std::chrono::system_clock> t1;
    std::chrono::time_point<std::chrono::system_clock> t2;
    std::chrono::duration<double, std::milli> fp_ms;
    double delta;

    PointMassModelGpu* models = new PointMassModelGpu[samples];
    float** x;
    float* u;
    float x_gain[state_dim];
    float u_gain[act_dim];

    u_gain[0] = dt*dt/2.0;
    u_gain[1] = dt;
    x_gain[0] = 1;
    x_gain[1] = dt;
    x_gain[2] = 0;
    x_gain[3] = 1;

    x = (float**) malloc(sizeof(float*)*samples);
    u = (float*) malloc(sizeof(float)*STEPS*act_dim);

    for(int i=0; i < STEPS; i++){
        u[i*act_dim + 0] = act;
        u[i*act_dim + 1] = act;
    }

    for(int i=0; i < samples; i++){
        x[i] = (float*) malloc(sizeof(float)*STEPS*state_dim);
        x[i][0] = init;
        x[i][1] = init;
        x[i][2] = init;
        x[i][3] = init;
        models[i].init(x[i], 0, u, x_gain, state_dim, u_gain, act_dim);
    }

    t1 = std::chrono::system_clock::now();

    // run the same code for on the cpu to evaluate the improvement.
    for(int i=0; i < samples; i++){
        models[i].run();
    }

    t2 = std::chrono::system_clock::now();
    fp_ms = t2 - t1;
    delta = fp_ms.count();

    std::cout << "Sequencial execution time: " << delta << "ms" << std::endl;

    bool error=true;
    // free the memory.
    for(int i=0; i<samples; i++){
        for (int j=0; j<STEPS; j++){
            for (int k=0; k<state_dim; k++){
                if (!fabs(data[i*STEPS*state_dim + j*state_dim + k] - x[i][j*state_dim + k]) < TOL ){
                    error=false;
                }
            }
        }
    }
    for(int i=0; i < samples; i++){
        free(x[i]);
    }
    free(x);
    free(u);
    delete models;
    return error;
}
*/

void to_csv(std::string filename,
            float* x,
            float* u,
            int sample,
            int size,
            int s_dim,
            int a_dim)
{
    std::cout << "Saving data to file...: " << std::flush;
    std::ofstream outfile;
    // create a name for the file output

    outfile.open(filename);

    outfile << "sample" << "," << "x" << "," << "y" << "," << "x_dot" << ","
            << "y_dot" << "," << "u_x" << "," << "u_y" << std::endl;
    for (int i=0; i < sample; i++){
        for (int j=0; j < size; j++){
            outfile << i << ","
                    << x[i*size*s_dim + j*s_dim + 0] << ","
                    << x[i*size*s_dim + j*s_dim + 1] << ","
                    << x[i*size*s_dim + j*s_dim + 2] << ","
                    << x[i*size*s_dim + j*s_dim + 3] << ","
                    << u[i*size*a_dim + j*a_dim + 0] << ","
                    << u[i*size*a_dim + j*a_dim + 1] << std::endl;
        }
    }
    outfile.close();
    std::cout << "Done" << std::endl;
    return;
}

void to_csv2(std::string filename,
            float* x,
            float* u,
            float* e,
            float* cost,
            float* beta,
            float* nabla,
            float* w,
            int sample,
            int size,
            int s_dim,
            int a_dim)
{
    std::cout << "Saving data to file...: " << std::flush;
    std::ofstream outfile;
    // create a name for the file output

    outfile.open(filename);

    outfile << "sample" << "," << "x" << "," << "y" << "," << "x_dot" << ","
            << "y_dot" << "," << "u_x" << "," << "u_y" << std::endl;
    for (int i=0; i < sample; i++){
        for (int j=0; j < size; j++){
            outfile << i << ","
                    << x[i*size*s_dim + j*s_dim + 0] << ","
                    << x[i*size*s_dim + j*s_dim + 1] << ","
                    << x[i*size*s_dim + j*s_dim + 2] << ","
                    << x[i*size*s_dim + j*s_dim + 3] << ","
                    << e[i*size*a_dim + j*a_dim + 0] << ","
                    << e[i*size*a_dim + j*a_dim + 1] << std::endl;
        }
    }

    for(int d=0; d < a_dim; d++){
        outfile << "u[" << d << "]: ";
        for(int i=0; i < STEPS; i++){
            outfile << "," << u[i*a_dim + d];
        }
        outfile << std::endl;
    }


    outfile << "w: ";
    for (int samp=0; samp < sample; samp++){
        outfile << "," << w[samp];
    }
    outfile << std::endl;

    outfile.close();
    std::cout << "Done" << std::endl;
    return;
}

int main(){


    std::chrono::time_point<std::chrono::system_clock> t1;
    std::chrono::time_point<std::chrono::system_clock> t2;
    std::chrono::duration<double, std::milli> fp_ms;
    double delta;

    int act_dim = 2;
    int state_dim = 4;

    int n = 4;

    float* x = (float*) malloc(sizeof(float)*n*STEPS*state_dim);
    float* u = (float*) malloc(sizeof(float)*STEPS*act_dim);
    float* e = (float*) malloc(sizeof(float)*n*STEPS*act_dim);
    float* cost = (float*) malloc(sizeof(float)*n);
    float* beta = (float*) malloc(sizeof(float));
    float* nabla = (float*) malloc(sizeof(float));
    float* weight = (float*) malloc(sizeof(float)*n);

    float dt = 0.1;

    //bool test = false;
    bool save = true;
    std::string filename("to_plot.csv");

    PointMassModel* model = new PointMassModel(n, STEPS, dt);
    /*
    * The state data stored on host. In this example,
    * the state is only one scalar but is stored on a
    * array with all the following states. Thus we need a int[n]
    * array to the input data.
    */
    float* h_x;
    float* h_u;
    /*
    * This variable stores the output result. In this
    * example it is a n*steps array but it will be continous
    * in device memory. so a int* array will be easier to work with.
    */
    float* h_o;
    float* h_e;

    float* goal;
    float* w;
    // allocate and init and res data.
    h_x = (float*) malloc(sizeof(float)*n*state_dim);
    h_u = (float*) malloc(sizeof(float)*STEPS*act_dim);

    h_o = (float*) malloc(sizeof(float)*n*STEPS*state_dim);
    h_e = (float*) malloc(sizeof(float)*n*STEPS*act_dim);

    goal = (float*) malloc(sizeof(float)*state_dim);
    goal[0] = 1.0;
    goal[1] = 0.0;
    goal[2] = 0.0;
    goal[3] = 0.0;

    w = (float*) malloc(sizeof(float)*state_dim);
    w[0] = 1.0;
    w[1] = 1.0;
    w[2] = 1.0;
    w[3] = 1.0;

    for (int i=0; i < n; i++){
        h_x[i*state_dim+0] = 0.;
        h_x[i*state_dim+1] = 0.;
        h_x[i*state_dim+2] = 0.;
        h_x[i*state_dim+3] = 0.;
    }

    for (int j=0; j < STEPS; j++){
        h_u[(j*act_dim)+0] = 0.;
        h_u[(j*act_dim)+1] = 0.;
    }
    // send the data on the device.
    model->memcpy_set_data(h_x, h_u, goal, w);


    t1 = std::chrono::system_clock::now();

    // run the multiple simulation on the device.
    model->sim();


    t2 = std::chrono::system_clock::now();
    fp_ms = t2 - t1;
    delta = fp_ms.count();


    std::cout << "GPU execution time: " << delta << "ms" << std::endl;

    model->get_inf(x, u, e, cost, beta, nabla, weight);
    // get the data from the device.
    model->memcpy_get_data(h_o, h_e);
    /*
    for (int i = 0; i < n; i ++){
        for (int j=0; j < STEPS; j++){
            std::cout << "t: " << j
                          << " u_x: " << h_u[i*STEPS*act_dim + j*act_dim + 0]
                          << " u_y: " << h_u[i*STEPS*act_dim + j*act_dim + 1] << std::endl;
        }
    }
    */

    if(save){
        //to_csv(filename, h_o, h_e, n, STEPS, state_dim, act_dim);
        to_csv2(filename, x, u, h_e, cost, beta, nabla, weight, n, STEPS, state_dim, act_dim);
    }
    /*if(test){
        //if(test_sim_gpu(h_o, n, state_dim, act_dim, dt, 0.01, 0.0)){
        //    std::cout << "Test passed!" << std::endl;
        //}
    }*/

    std::cout << "Freeing memory... : " << std::flush;
    free(h_x);
    free(h_o);
    free(h_u);
    std::cout << "Done" << std::endl;

    delete model;
    cudaDeviceReset();
    //cuCtxDestroy();
}
