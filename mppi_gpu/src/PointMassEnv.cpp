#include "mppi_env.hpp"
#include <iostream>

#include <sys/stat.h>

#include <cstdio>
#include <chrono>
#include <unistd.h>
typedef std::chrono::high_resolution_clock Clock;

inline bool is_file (const char* name)
{
  struct stat buffer;
  return (stat (name, &buffer) == 0);
}

PointMassEnv::PointMassEnv(const char* modelFile, const char* mjkey, bool view)
{
  std::cout << GLFW_VERSION_MAJOR << "." << GLFW_VERSION_MINOR << "." << GLFW_VERSION_REVISION << std::endl;
  std::cout << glfwGetVersionString() << std::endl;
  info = string("Point Mass Environment info");
  view_ = view;

  if (!is_file(mjkey)){
    std::cout << "The activation file doesn't exist" << std::endl;
  }

  if (!is_file(modelFile)){
    std::cout << "The model file doesn't exist" << std::endl;
  }

  mj_activate(mjkey);

  m = mj_loadXML(modelFile, NULL, NULL, 0);

  if(m == NULL){
    std::cout << "couldn't load xml" << std::endl;
  }

  d = mj_makeData(m);

  std::cout << "Loaded model and Data" << '\n';

  if(view_){
    // init GLFW, create window, make OpenGL context current, request v-sync
    glfwInit();
    window = glfwCreateWindow(1200, 900, "Demo", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // initialize visualization data structures
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);

    // create scene and context
    mjv_makeScene(m, &scn, 2000);
    mjr_makeContext(m, &con, mjFONTSCALE_100);
  }

  _simend = d->time + 0.1;
}

PointMassEnv::~PointMassEnv()
{
  // close GLFW, free visualization storage
  glfwTerminate();
  mjv_freeScene(&scn);
  mjr_freeContext(&con);
  // deallocate existing mjModel
  mj_deleteModel(m);

  // deallocate existing mjData
  mj_deleteData(d);
}

string PointMassEnv::print() const
{
  return Env::print();
}

bool PointMassEnv::simulate(float* u)
{
  // run main loop, target real-time simulation and 60 fps rendering

  if( !glfwWindowShouldClose(window) && d->time < _simend ){
    //std::cout << "d->time: " << d->time << " simed: " << simend << '\n';
    // advance interactive simulation for 1/60 sec
    //  Assuming MuJoCo can simulate faster than real-time, which it usually can,
    //  this loop will finish on time for the next frame to be rendered at 60 fps.
    //  Otherwise add a cpu timer and exit this loop when it is time to render.
    auto t1 = Clock::now();
    auto t2 = Clock::now();
    if (view_){
      t1 = Clock::now();
    }
    mjtNum simstart = d->time;

    d->ctrl[0] = u[0];
    d->ctrl[1] = u[1];

    while( d->time - simstart < 1.0/60.0 ){
      mj_step(m, d);
      //std::cout << "d->time: " << d->time << " simed: " << simstart << " freq: " << 1./6. << std::endl;
    }
    //std::cout << "DONE" << std::endl;
    if(view_){

      // get framebuffer viewport
      mjrRect viewport = {0, 0, 0, 0};
      glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

      // update scene and render
      mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
      mjr_render(viewport, &scn, &con);

      /* Bit of code to have the sim ~real time as for whatever reason
       * glfwSwapBuffers does not seem to be blocking dispite version 3.4.0
       * of glfw and setting glfwSwapInterval(1) on my 60hz monitor.
       */

      t2 = Clock::now();
      std::chrono::duration<double, std::milli> fp_ms = t2 - t1;
      double delta = fp_ms.count();
      if (delta < 1.0/60.0*1000){
        usleep((1.0/60.0*1000 - delta)*1000);
      }
      // swap OpenGL buffers (blocking call due to v-sync)
      // V-sync not working
      glfwSwapBuffers(window);


      // process pending GUI events, call GLFW callbacks
      glfwPollEvents();
    }
    return false;
  }
  return true;
}

void PointMassEnv::step(float* x, float* u)
{
  d->ctrl[0] = u[0];
  d->ctrl[1] = u[1];
  mj_step(m, d);
  x[0] = d->qpos[0];
  x[1] = d->qpos[1];
  x[2] = d->qvel[0];
  x[3] = d->qvel[1];
}

void PointMassEnv::get_x(float* x)
{
    x[0] = d->qpos[0];
    x[1] = d->qpos[1];
    x[2] = d->qpos[2];
    x[3] = d->qpos[3];
}
