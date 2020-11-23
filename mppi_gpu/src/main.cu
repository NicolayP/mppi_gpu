#include "point_mass.hpp"
#include "mppi_env.hpp"
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
            << "y_dot" << "," << "u_x" << "," << "u_y";
    for(int d=0; d < a_dim; d++)
    {
        outfile << "," << "u[" << d << "]";
    }
    outfile << "," << "c" <<  "," << "w" << std::endl;

    for (int i=0; i < sample; i++){
        for (int j=0; j < size + 1 ; j++){
            outfile << i << ","
                    << x[i*(size+1)*s_dim + j*s_dim + 0] << ","
                    << x[i*(size+1)*s_dim + j*s_dim + 1] << ","
                    << x[i*(size+1)*s_dim + j*s_dim + 2] << ","
                    << x[i*(size+1)*s_dim + j*s_dim + 3] << ","
                    << e[i*size*a_dim + j*a_dim + 0] << ","
                    << e[i*size*a_dim + j*a_dim + 1];
            // U is of size steps
            if(i < 1) {
                outfile << "," << u[j*a_dim + 0] << "," << u[j*a_dim + 1];
            }else
                outfile << ", , ";
            if (i*size + j < sample) {
                outfile << "," << cost[i*size+j] << "," << w[i*size+j];
            }
            outfile << std::endl;
        }
    }


    outfile.close();
    std::cout << "Done" << std::endl;
    return;
}

int main(){
    char* modelFile = "../envs/point_mass.xml";
    char* mjkey = "../lib/contrib/mjkey.txt";


    std::chrono::time_point<std::chrono::system_clock> t1;
    std::chrono::time_point<std::chrono::system_clock> t2;
    std::chrono::duration<double, std::milli> fp_ms;
    double delta;

    int act_dim = 2;
    int state_dim = 4;
    int n = 3000;

    float* x = (float*) malloc(sizeof(float)*n*(STEPS+1)*state_dim);
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
    PointMassEnv env = PointMassEnv(modelFile, mjkey, true);

    PointMassModel* model = new PointMassModel(n, STEPS, dt, state_dim, act_dim, false);
    bool done=false;
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
    x = (float*) malloc(sizeof(float)*state_dim);
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
    w[2] = 0.025;
    w[3] = 0.025;

    env.get_x(x);

    for (int j=0; j < STEPS; j++){
        h_u[(j*act_dim)+0] = 0.;
        h_u[(j*act_dim)+1] = 0.;
    }
    // send the data on the device.

    float* next_act = (float*) malloc(sizeof(float)*act_dim);

    model->memcpy_set_data(x, h_u, goal, w);


    // run the multiple simulation on the device.
    while(!done){

        t1 = std::chrono::system_clock::now();
        model->sim(next_act);
        t2 = std::chrono::system_clock::now();
        fp_ms += t2 - t1;

        done = env.simulate(next_act);
        std::cout << "next_act: " << next_act[0] << ", " << next_act[1] << '\n';
        env.get_x(x);
        model->set_x(x);
    }

    //send act to sim;

    // collect new state;

    // set state in controller;

    // next step


    delta = fp_ms.count();


    std::cout << "GPU execution time: " << delta << "ms" << std::endl;

    //model->get_inf(x, u, e, cost, beta, nabla, weight);
    // get the data from the device.
    //model->memcpy_get_data(h_o, h_e);

    if(save){
        model->get_inf(h_x, h_u, h_e, cost, beta, nabla, weight);
        to_csv2(filename, h_x, h_u, h_e, cost, beta, nabla, weight, n, STEPS, state_dim, act_dim);
    }

    std::cout << "Freeing memory... : " << std::flush;
    free(h_x);
    free(h_o);
    free(h_u);
    std::cout << "Done" << std::endl;

    delete model;
    cudaDeviceReset();
}
