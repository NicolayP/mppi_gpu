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

bool test_sim_gpu(float* data,
              size_t samples,
              size_t state_dim,
              size_t act_dim,
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

void to_csv(std::string filename, float* x, size_t size, size_t sample){
  std::cout << "Saving data to file...: " << std::flush;
  std::ofstream outfile;
  // create a name for the file output

  outfile.open(filename);

  outfile << "sample" << "," << "x" << "," << "y" << "," << "x_dot" << "," << "y_dot" << std::endl;
  for (int j=0; j < sample; j++){
    for (int i=0; i < size; i++){
      outfile << j << "," << x[i*4 + 0] << "," << x[i*4 + 1] << "," << x[i*4 + 2] << "," << x[i*4 + 3] << std::endl;
    }
  }
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

  int n = 1;

  float dt = 1.;

  bool test = false;
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
  // allocate and init and res data.
  h_x = (float*) malloc(sizeof(float)*n*state_dim);
  h_u = (float*) malloc(sizeof(float)*n*STEPS*act_dim);

  h_o = (float*) malloc(sizeof(float)*n*STEPS*state_dim);
  for (int i=0; i < n; i++){
    h_x[i*state_dim+0] = 0.;
    h_x[i*state_dim+1] = 0.;
    h_x[i*state_dim+2] = 0.;
    h_x[i*state_dim+3] = 0.;
    for (int j=0; j < STEPS; j++){
      h_u[(i*STEPS*act_dim)+(j*act_dim)+0] = 0.01;
      h_u[(i*STEPS*act_dim)+(j*act_dim)+1] = 0.005;
    }
  }
  // send the data on the device.
  model->memcpy_set_data(h_x, h_u);

  t1 = std::chrono::system_clock::now();

  // run the multiple simulation on the device.
  model->sim();


  t2 = std::chrono::system_clock::now();
  fp_ms = t2 - t1;
  delta = fp_ms.count();

  std::cout << "GPU execution time: " << delta << "ms" << std::endl;

  // get the data from the device.
  model->memcpy_get_data(h_o);

  if(save){
    to_csv(filename, h_o, STEPS, n);
  }

  if(test){
    if(test_sim_gpu(h_o, n, state_dim, act_dim, dt, 0.01, 0.0)){
      std::cout << "Test passed!" << std::endl;
    }
  }

  std::cout << "Freeing memory... : " << std::flush;
  free(h_x);
  free(h_o);
  free(h_u);
  std::cout << "Done" << std::endl;

  delete model;
  cudaDeviceReset();
  //cuCtxDestroy();
}
