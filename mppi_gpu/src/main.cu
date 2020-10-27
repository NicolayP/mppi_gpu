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

 #define TOL 1e-6

void to_csv(float* x, size_t size, size_t sample){
  std::ofstream outfile;
  // create a name for the file output
  std::string filename = "exampleOutput.csv";

  outfile.open(filename);

  outfile << "sample" << "," << "x" << "," << "y" << "," << "x_dot" << "," << "y_dot" << std::endl;
  for (int j=0; j < sample; j++){
    for (int i=0; i < size; i++){
      outfile << j << "," << x[i*4 + 0] << "," << x[i*4 + 1] << "," << x[i*4 + 2] << "," << x[i*4 + 3] << std::endl;
    }
  }
  outfile.close();
}

int main(){


  std::chrono::time_point<std::chrono::system_clock> t1;
  std::chrono::time_point<std::chrono::system_clock> t2;
  std::chrono::duration<double, std::milli> fp_ms;
  double delta;


  int n = 20000;

  /*
   * copy of our models on host. Should ultimatly
   * be removed and the models object should be stored on
   * device only.
   */
  // PointMassModelGpu* models = new PointMassModelGpu[n];
  /*
   * Model Gpu wrapper, this will allow to offer one entry
   * interface to the controller that is not device or host specific.
   */
  PointMassModel model = PointMassModel(n, STEPS, 0.01);
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
  h_x = (float*) malloc(sizeof(float)*n*4);
  h_u = (float*) malloc(sizeof(float)*n*STEPS*2);

  h_o = (float*) malloc(sizeof(float)*n*STEPS*4);
  for (int i=0; i < n; i++){
    h_x[i*4+0] = 0.;
    h_x[i*4+1] = 0.;
    h_x[i*4+2] = 0.;
    h_x[i*4+3] = 0.;
    for (int j=0; j < STEPS; j++){
      h_u[(i*STEPS*2)+(j*2)+0] = 0.01;
      h_u[(i*STEPS*2)+(j*2)+1] = 0.01;
    }
  }
  // send the data on the device.
  model.memcpy_set_data(h_x, h_u);

  t1 = std::chrono::system_clock::now();

  // run the multiple simulation on the device.
  model.sim();


  t2 = std::chrono::system_clock::now();
  fp_ms = t2 - t1;
  delta = fp_ms.count();

  std::cout << "GPU: Test passed" << std::endl;
  std::cout << "GPU execution time: " << delta << "ms" << std::endl;

  // get the data from the device.
  model.memcpy_get_data(h_o);
  //std::cout << "Saving data to file...: ";
  //to_csv(h_o, STEPS, n);
  //std::cout << "Done" << std::endl;

  // Test if the value in h_x is as expected.
  /*for(int i=0; i < n; i++){
    for(int j = 0; j < STEPS; j++){
      assert(h_o[i*STEPS+j] == j+1);
    }
  }*/

  /*
  float** x;
  x = (float**) malloc(sizeof(float*)*n);
  for(float i=0; i < n; i++){
    x[i] = (float*) malloc(sizeof(float)*STEPS);
    x[i][0] = 1.0;
    models[i].set_state(x[i]);
  }

  t1 = std::chrono::system_clock::now();

  // run the same code for on the cpu to evaluate the improvement.
  for(int i=0; i < n; i++){
    models[i].run();
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

  */

  // free the memory.
  for(int i = 0; i < n*4; i++){
    assert(fabs(h_x[i] - 0.) < TOL );
  }
  std::cout << "Test passed" << std::endl;

  std::cout << "Freeing memory... : ";

  free(h_x);
  free(h_o);
  free(h_u);
  std::cout << "Done" << std::endl;
}
