#ifndef _MPPI_ENV_H_
#define _MPPI_ENV_H_

#include <iostream>
#include <string>
#include "mujoco.h"
#include <GLFW/glfw3.h>

using namespace std;


class Env{
public:
    friend ostream& operator<<(ostream& os, const Env& env);
    virtual string print() const;
protected:
    string info;

};

class PointMassEnv : public Env{
public:
    PointMassEnv(const char* modelFile, const char* mjkey, bool view=false);
    virtual ~PointMassEnv();
    string print() const override;
    bool simulate(float* u);
    void step(float* x, float* u);
    void get_x(float* x);

protected:
    bool view_;
    mjtNum _simend;


};


#endif
