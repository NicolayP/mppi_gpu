#include <stdlib.h>
#include <iostream>

#include "mppi_env.hpp"
#include "mppi_cont.hpp"
#include "mppi_cost.hpp"
#include "mppi_model.hpp"

int main(int argc, char const *argv[]) {
  /* parse argument:
   *  -config_file.
   *  --eval_out (eval mode, need a output dir as well)
   *  -env_file. (model and environment file)
   *  ** For later **
   *  -tpye (string with the type of controller)
   */
  // Store Hpyer parameters in a config file.
  // load the hyper parameters.
  char* modelFile = "../envs/point_mass.xml";
  char* mjkey = "../lib/contrib/mjkey.txt";
  PointMassEnv env = PointMassEnv(modelFile, mjkey, false);
  PointMassModel model = PointMassModel("some/config/file", 0.01);
  env.simulate();
  std::cout << "Creating model" << std::endl;
  float u[2] = {0.01, 0.005};
  float x[4] = {0.0, 0.0, 0.0, 0.0};
  float x1[4] = {0.0, 0.0, 0.0, 0.0};
  float x2[4] = {0.0, 0.0, 0.0, 0.0};
  model.predict(x1, u);
  env.step(x2, u);
  std::cout << "Finished simulation" << std::endl;

  //env.simulate();

  /* TODO: install GLFW 3.3.2 otherwise glfwSwapInterval(1) is not working correctly. !!!! */

  //model.~PointMassModel();

  // create simulation.
  // create cost.
  // create controller.
  // start loop.
  return 0;
}
