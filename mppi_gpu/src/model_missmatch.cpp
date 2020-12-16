#include "mppi_env.hpp"
#include <iostream>
#include <fstream>
#include <random>

#include <assert.h>

#include <cstdio>
#include <ctime>
#include <chrono>
#include <unistd.h>

#include <tclap/CmdLine.h>
#include <yaml-cpp/yaml.h>

using namespace std;

string model_file = "../envs/point_mass2d.xml";
string mjkey_file = "../lib/contrib/mjkey.txt";

float dt = 0.1;

float _x_gain[4] = {1, dt, 0, 1};
float _u_gain[2] = {dt*dt/2.0, dt};

void step_model (float* x_next, float* x, float* u, size_t a, size_t d, size_t t) {

    for(int i=0; i < a; i++){
        x_next[i] = _x_gain[0]*x[i] +
        _x_gain[1]*x[i+d/2] +
        _u_gain[0]*u[i];

        x_next[i+d/2] = _x_gain[2]*x[i] +
        _x_gain[3]*x[i+d/2] +
        _u_gain[1]*u[i];

    }
}

void gen_inputs(float* u, size_t n){
    default_random_engine gen;
    normal_distribution<float> dist(0.0, 1.0);
    // allocate random values between [0-1]
    for (size_t i=0; i < n; i++) {
        u[i] = dist(gen);
    }
}

void run_world (vector<vector<float>> &traj_world, float* u, size_t n, size_t a, size_t d) {
    PointMassEnv env = PointMassEnv(model_file.c_str(), mjkey_file.c_str(), true);
    vector<float> tmp;
    float x[d];

    env.get_x(x);
    for (size_t i=0; i < d; i++){
        tmp.push_back(x[i]);
    }
    traj_world.push_back(tmp);
    tmp.clear();

    for (size_t t=0; t < n; t++) {
        env.simulate(&u[t*a]);
        env.get_x(x);
        for (size_t i=0; i < d; i++){
            tmp.push_back(x[i]);
        }
        traj_world.push_back(tmp);
        tmp.clear();

    }
}


void run_sim (vector<vector<float>> &traj_sim, float* u, size_t n, size_t a, size_t d) {
    vector<float> tmp;
    float x[d];
    for (size_t i=0; i<d; i++) {
        x[i] = 0;
    }

    for (size_t i=0; i<d; i++){
        tmp.push_back(x[i]);
    }
    traj_sim.push_back(tmp);
    tmp.clear();

    float x_next[d];
    for (size_t t=0; t < n; t++) {
        step_model(x_next, x, &u[t*a], a, d, t);

        for (size_t i=0; i < d; i++){
            x[i] = x_next[i];
            tmp.push_back(x[i]);
        }
        traj_sim.push_back(tmp);
        tmp.clear();
    }
}



void save_trajs (const string &filename,
                 const vector<vector<float>> &traj_world,
                 const vector<vector<float>> &traj_sim) {
    ofstream outfile;
    outfile.open(filename);
    outfile << "x_s," << "y_s," << "v_x_s," << "v_y_s,"
            << "x_w," << "y_w," << "v_x_w," << "v_y_w," << endl;

    for (size_t i=0; i<traj_world.size(); i++) {
        for (size_t j=0; j<traj_world[i].size(); j++) {
            outfile << traj_world[i][j] << ",";
        }

        for (size_t j=0; j<traj_sim[i].size(); j++) {
            outfile << traj_sim[i][j] << ",";
        }

        outfile << endl;
    }
}

int main(int argc, char const *argv[]) {
    size_t n = 100;
    size_t a = 2;
    size_t d = 4;

    string filename = "missmatch.csv";

    float inputs[a*n];
    vector<vector<float>> traj_world;
    vector<vector<float>> traj_sim;

    gen_inputs(inputs, n*a);
    run_world(traj_world, inputs, n, a, d);

    for (auto t=traj_world.begin(); t!=traj_world.end(); t++) {
        for (auto x=(*t).begin(); x!=(*t).end(); x++) {
            cout << *x << " ";
        }
        cout << endl;
    }

    run_sim(traj_sim, inputs, n, a, d);

    for (auto t=traj_sim.begin(); t!=traj_sim.end(); t++) {
        for (auto x=(*t).begin(); x!=(*t).end(); x++) {
            cout << *x << " ";
        }
        cout << endl;
    }

    save_trajs(filename, traj_world, traj_sim);
    return 0;
}
