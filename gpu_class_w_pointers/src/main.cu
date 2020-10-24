#include "gpu_class.hpp"
#include <iostream>
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


int main(){

  std::chrono::time_point<std::chrono::system_clock> t1;
  std::chrono::time_point<std::chrono::system_clock> t2;
  std::chrono::duration<double, std::milli> fp_ms;
  double delta;


  int n = 2000;

  /*
   * copy of our models on host. Should ultimatly
   * be removed and the models object should be stored on
   * device only.
   */
  ModelGpu* models = new ModelGpu[n];
  /*
   * Model Gpu wrapper, this will allow to offer one entry
   * interface to the controller that is not device or host specific.
   */
  Model model = Model(n, STEPS);
  /*
   * The state data stored on host. In this example,
   * the state is only one scalar but is stored on a
   * array with all the following states. Thus we need a int[n]
   * array to the input data.
   */

  int* h_x;

  /*
   * This variable stores the output result. In this
   * example it is a n*steps array but it will be continous
   * in device memory. so a int* array will be easier to work with.
   */

  int* h_o;

  // allocate and init and res data.
  h_x = (int*) malloc(sizeof(int)*n);
  h_o = (int*) malloc(sizeof(int)*n*STEPS);
  for (int i=0; i < n; i++){
    h_x[i] = 1;
  }
  // send the data on the device.
  model.memcpy_set_data(h_x);

  t1 = std::chrono::system_clock::now();

  // run the multiple simulation on the device.
  model.sim();


  t2 = std::chrono::system_clock::now();
  fp_ms = t2 - t1;
  delta = fp_ms.count();


  // get the data from the device.
  model.memcpy_get_data(h_o);

  // Test if the value in h_x is as expected.
  for(int i=0; i < n; i++){
    for(int j = 0; j < STEPS; j++){
      assert(h_o[i*STEPS+j] == j+1);
    }
  }
  std::cout << "GPU: Test passed" << std::endl;
  std::cout << "GPU execution time: " << delta << "ms" << std::endl;

  /*
  std::cout << "Data: " << std::endl;
  for (int i = 0; i < n; i++){
    for (int j = 0; j < STEPS; j++){
      std::cout << h_o[i*STEPS+j] << " ";
    }
    std::cout << std::endl;
  }*/

  int** x;
  x = (int**) malloc(sizeof(int*)*n);
  for(int i=0; i < n; i++){
    x[i] = (int*) malloc(sizeof(int)*STEPS);
    x[i][0] = 1;
    models[i].set_x(x[i]);
  }

  t1 = std::chrono::system_clock::now();

  // run the same code for on the cpu to evaluate the improvement.
  for(int i=0; i < n; i++){
    models[i].advance();
  }

  t2 = std::chrono::system_clock::now();
  fp_ms = t2 - t1;
  delta = fp_ms.count();

  for(int i=0; i < n; i++){
    for(int j = 0; j < STEPS; j++){
      assert(x[i][j] == j+1);
    }
  }
  std::cout << "Sequencial: Test passed" << std::endl;
  std::cout << "Sequencial execution time: " << delta << "ms" << std::endl;

  // free the memory.
  free(h_x);
}
