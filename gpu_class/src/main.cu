#include "gpu_class.hpp"
#include <iostream>
#include <assert.h>

#include <cstdio>
#include <ctime>
#include <chrono>
#include <unistd.h>




int main(){

  std::chrono::time_point<std::chrono::system_clock> t1;
  std::chrono::time_point<std::chrono::system_clock> t2;
  std::chrono::duration<double, std::milli> fp_ms;
  double delta;


  int n = 10;

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
  Model* model = new Model(n);
  /*
   * The state data stored on host. In this toy example,
   * the state is only one scalar. Thus we need a int[n]
   * array to store everything.
   */
  int* h_x;
  // allocate and init data.
  h_x = (int*) malloc(sizeof(int)*n);

  for (int i=0; i < n; i++){
    h_x[i] = 1;
  }
  // send the data on the device.
  model->memcpy_set_data(h_x);

  t1 = std::chrono::system_clock::now();

  // run the multiple simulation on the device.
  model->sim();


  t2 = std::chrono::system_clock::now();
  fp_ms = t2 - t1;
  delta = fp_ms.count();
  std::cout << "GPU: " << delta << "ms" << std::endl;


  // get the data from the device.
  model->memcpy_get_data(h_x);

  // Test if the value in h_x is as expected.
  for(int i=0; i < n; i++){
    assert(h_x[i] == STEPS+1);
  }
  std::cout << "Test passed" << std::endl;

    t1 = std::chrono::system_clock::now();

  // run the same code for on the cpu to evaluate the improvement.
  for(int i=0; i < n; i++){
    models[i].advance();
  }


  t2 = std::chrono::system_clock::now();
  fp_ms = t2 - t1;
  delta = fp_ms.count();
  std::cout << "Sequencial: " << delta << "ms" << std::endl;


  // free the memory.
  free(h_x);
  delete models;
  delete model;

  cudaDeviceReset();
}
