#include "mppi_env.hpp"
#include <iostream>

#include <sys/stat.h>

#include <cstdio>
#include <chrono>
#include <unistd.h>
typedef std::chrono::high_resolution_clock Clock;

mjModel* m;
mjData* d;

mjvCamera cam;                      // abstract camera
mjvOption opt;                      // visualization options
mjvScene scn;                       // abstract scene
mjrContext con;                     // custom GPU context
GLFWwindow* window;

// mouse interaction
bool button_left = false;
bool button_middle = false;
bool button_right = false;
bool save_data = false;
double lastx = 0;
double lasty = 0;


void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods);
void mouse_button(GLFWwindow* window, int button, int act, int mods);
void mouse_move(GLFWwindow* window, double xpos, double ypos);
void scroll(GLFWwindow* window, double xoffset, double yoffset);

inline bool is_file (const char* name) {
  struct stat buffer;
  return (stat (name, &buffer) == 0);
}

PointMassEnv::PointMassEnv(const char* modelFile, const char* mjkey, bool view) {
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

  std::cout << "Loaded model and Data" << std::endl;

  if(view_){
    button_left = false;
    button_middle = false;
    button_right = false;
    save_data = false;
    lastx = 0;
    lasty = 0;
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

    glfwSetKeyCallback(window, keyboard);
    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetScrollCallback(window, scroll);

    // create scene and context
    mjv_makeScene(m, &scn, 2000);
    mjr_makeContext(m, &con, mjFONTSCALE_100);
  }
  // empty step to initalize everything.
  mj_step(m, d);

  _simend = d->time + 0.0001 + 10;
}

PointMassEnv::~PointMassEnv() {
  // close GLFW, free visualization storage
  glfwTerminate();
  mjv_freeScene(&scn);
  mjr_freeContext(&con);
  // deallocate existing mjModel
  mj_deleteModel(m);

  // deallocate existing mjData
  mj_deleteData(d);
}

string PointMassEnv::print() const {
  return Env::print();
}

bool PointMassEnv::simulate(float* u) {
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

    for (int i=0; i < m->nu; i++) {
        d->ctrl[i] = u[i];
    }


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

void PointMassEnv::step(float* x, float* u) {
    for (int i=0; i < m->nu; i++) {
        d->ctrl[i] = u[i];
    }
    mj_step(m, d);

    for (int i=0; i < m->nq; i++) {
        x[i] = d->qpos[i];
    }

    for (int i=0; i < m->nv; i++) {
        x[i+m->nq] = d->qvel[i];
    }
}

void PointMassEnv::get_x(float* x) {
    for (int i=0; i < m->nq; i++) {
        x[i] = d->qpos[i];
    }

    for (int i=0; i <  m->nv; i++) {
        x[i+m->nq] = d->qvel[i];
    }
}

// mouse move callback
void mouse_move(GLFWwindow* window, double xpos, double ypos)
{
    // no buttons down: nothing to do
    if( !button_left && !button_middle && !button_right )
        return;

    // compute mouse displacement, save
    double dx = xpos - lastx;
    double dy = ypos - lasty;
    lastx = xpos;
    lasty = ypos;

    // get current window size
    int width, height;
    glfwGetWindowSize(window, &width, &height);

    // get shift key state
    bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)==GLFW_PRESS ||
                      glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT)==GLFW_PRESS);

    // determine action based on mouse button
    mjtMouse action;
    if( button_right )
        action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    else if( button_left )
        action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    else
        action = mjMOUSE_ZOOM;

    // move camera
    mjv_moveCamera(m, action, dx/height, dy/height, &scn, &cam);
}


// scroll callback
void scroll(GLFWwindow* window, double xoffset, double yoffset)
{
    // emulate vertical mouse motion = 5% of window height
    mjv_moveCamera(m, mjMOUSE_ZOOM, 0, 0.05*yoffset, &scn, &cam);
}


// keyboard callback
void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods)
{
    // backspace: reset simulation
    if( act==GLFW_PRESS && key==GLFW_KEY_END)
    {
        save_data = true;
    }
}


// mouse button callback
void mouse_button(GLFWwindow* window, int button, int act, int mods)
{
    // update button state
    button_left =   (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)==GLFW_PRESS);
    button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE)==GLFW_PRESS);
    button_right =  (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT)==GLFW_PRESS);

    // update mouse position
    glfwGetCursorPos(window, &lastx, &lasty);
}
