#include "point_mass.hpp"
#include "mppi_env.hpp"
#include "mppi_utils.hpp"
#include <iostream>
#include <fstream>

#include <assert.h>

#include <cstdio>
#include <ctime>
#include <chrono>
#include <unistd.h>

#include <tclap/CmdLine.h>
#include <yaml-cpp/yaml.h>
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


void to_csv_traj (std::string filename,
                  std::vector<std::vector<float>> x,
                  std::vector<std::vector<float>> u) {
    std::ofstream outfile;

    outfile.open(filename);
    std::cout << "Saving traj to file...: " << std::endl;

    outfile << "x" << "," << "y" << "," << "vx" << "," << "vy" << ","
            << "ux" << "," << "uy" << ',' << "size_x" << "," << "size_u"<< std::endl;

    outfile << x[0][0] << "," << x[0][1] << ","
            << x[0][2] << "," << x[0][3] << ","
            << u[0][0] << "," << u[0][1] << ","
            << x.size() << "," << u.size() << std::endl;
    for (int i = 1; i < u.size(); i++) {
        outfile << x[i][0] << "," << x[i][1] << ","
                << x[i][2] << "," << x[i][3] << ","
                << u[i][0] << "," << u[i][1] << std::endl;
    }
    outfile << x[u.size()][0] << "," << x[u.size()][1] << ","
            << x[u.size()][2] << "," << x[u.size()][3];
    outfile.close();
    std::cout << x.size() << " " << u.size() << std::endl;
    return;
}

void to_csv (std::string filename,
            float* x,
            float* u,
            int sample,
            int size,
            int s_dim,
            int a_dim) {
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

void to_csv2 (std::string filename,
             float* x,
             float* u,
             float* u_prev,
             float* e,
             float* cost,
             float* beta,
             float* nabla,
             float* w,
             int sample,
             int size,
             int s_dim,
             int a_dim) {

    std::cout << "Saving data to file...: " << std::flush;
    std::ofstream outfile;
    // create a name for the file output

    outfile.open(filename);

    outfile << "sample" << "," << "x" << "," << "y" << "," << "x_dot" << ","
            << "y_dot" << "," << "e_x" << "," << "e_y";
    for(int d=0; d < a_dim; d++)
    {
        outfile << "," << "u[" << d << "]";
    }

    for(int d=0; d < a_dim; d++)
    {
        outfile << "," << "u_prev[" << d << "]";
    }

    outfile << "," << "c" <<  "," << "w" << std::endl;

    for (int i=0; i < sample; i++){
        for (int j=0; j < size + 1 ; j++){
            outfile << i << ","
                    << x[i*(size+1)*s_dim + j*s_dim + 0] << ","
                    << x[i*(size+1)*s_dim + j*s_dim + 1] << ","
                    << x[i*(size+1)*s_dim + j*s_dim + 2] << ","
                    << x[i*(size+1)*s_dim + j*s_dim + 3] << ",";
            if (j < size) {
                outfile << e[i*size*a_dim + j*a_dim + 0] << ","
                        << e[i*size*a_dim + j*a_dim + 1];
            } else {
                outfile << ", ";
            }
            // U is of size steps
            if(i < 1 && j < size) {
                outfile << "," << u[j*a_dim + 0] << "," << u[j*a_dim + 1];
                outfile << "," << u_prev[j*a_dim + 0] << "," << u_prev[j*a_dim + 1];
            } else {
                outfile << ", , , , ";
            }

            if (i*(size+1) + j < sample) {
                outfile << "," << cost[i*(size+1)+j] << "," << w[i*(size+1)+j];
            }
            outfile << std::endl;
        }
    }


    outfile.close();
    std::cout << "Done" << std::endl;
    return;
}

void parse_argument (int argc,
                     char const* argv[],
                     std::string& config,
                     std::string& mjkey,
                     std::string& stepfile,
                     std::string& trajfile);

void parse_config (std::string& config_file,
                   std::string& model_file,
                   int& samples,
                   int& state_dim,
                   int& act_dim,
                   int& horizon,
                   float& dt,
                   float& lambda,
                   float** noise,
                   float** init,
                   float** max_a,
                   float** goal,
                   std::string& cost_type,
                   float** cost_q);

void init_controller_var (int const& samples,
                          int const& state_dim,
                          int const& act_dim,
                          int const& horizon,
                          float** next_act,
                          float** init_state,
                          float** init_actions,
                          float** cost,
                          float** beta,
                          float** nabla,
                          float** weight);

void free_controller_memory(float** next_act,
                            float** init_state,
                            float** init_actions,
                            float** cost,
                            float** beta,
                            float** nabla,
                            float** weight);

void free_paresd_data(float** noise,
                      float** init_next,
                      float** max_a,
                      float** goal,
                      float** cost_q);

void init_action_seq(float* init_actions, int act_dim, int steps);

void verify_parse (int n,
                   int state_dim,
                   int act_dim,
                   int steps,
                   float dt,
                   float lambda,
                   float* noise,
                   float* init_next,
                   float* max_a,
                   float* cost_q,
                   float* goal);

int main (int argc, char const* argv[]) {

    std::string config_file;
    std::string mjkey_file;
    std::string out_step_file;
    std::string out_traj_file;
    std::string model_file;
    std::string cost_type;
    // will store the next action.
    float* next_act;

    std::size_t t(0);

    int n;
    int state_dim;
    int act_dim;
    int steps;
    float dt;
    float lambda;
    float* noise;
    float* init_next;
    float* max_a;
    float* cost_q;
    float* goal;

    float* init_state;
    float* init_actions;
    float* cost;
    float* beta;
    float* nabla;
    float* weight;

    bool save_step = false;
    bool save_traj = true;
    bool done = false;

    std::vector<std::vector<float>> u;
    std::vector<std::vector<float>> x;
    std::vector<float> tmp_u;
    std::vector<float> tmp_x;


    parse_argument(argc, argv, config_file, mjkey_file, out_step_file, out_traj_file);

    parse_config(config_file,
                 model_file,
                 n,
                 state_dim,
                 act_dim,
                 steps,
                 dt,
                 lambda,
                 &noise,
                 &init_next,
                 &max_a,
                 &goal,
                 cost_type,
                 &cost_q);

    init_controller_var(n,
                         state_dim,
                         act_dim,
                         steps,
                         &next_act,
                         &init_state,
                         &init_actions,
                         &cost,
                         &beta,
                         &nabla,
                         &weight);

    if (config_file == "../config/mppi-config-test.yaml") {
        verify_parse(n,
                 state_dim,
                 act_dim,
                 steps,
                 dt,
                 lambda,
                 noise,
                 init_next,
                 max_a,
                 cost_q,
                 goal);
    }

    PointMassEnv env = PointMassEnv(model_file.c_str(), mjkey_file.c_str(), true);

    PointMassModel* model = new PointMassModel(n, steps, dt, state_dim, act_dim, false);

    env.get_x(init_state);
    init_action_seq(init_actions, act_dim, steps);

    model->memcpy_set_data(init_state, init_actions, goal, cost_q);

    for (int i=0; i < state_dim; i++) {
        tmp_x.push_back(init_state[i]);
    }
    x.push_back(tmp_x);
    tmp_x.clear();

    // run the multiple simulation on the device.
    float* u_prev = (float*) malloc(sizeof(float)*steps*act_dim);
    while(!done){
        model->get_u(u_prev);
        //t1 = std::chrono::system_clock::now();
        model->get_act(next_act);
        //t2 = std::chrono::system_clock::now();
        //fp_ms += t2 - t1;
        std::cout << "next_act: ";
        for (int i=0; i < act_dim; i++) {
            std::cout << next_act[i] << " ";

        }
        std::cout << std::endl;
        done = env.simulate(next_act);
        env.get_x(init_state);

        for (int i=0; i < act_dim; i++) {
            tmp_u.push_back(next_act[i]);
        }
        u.push_back(tmp_u);
        tmp_u.clear();

        for (int i=0; i < state_dim; i++) {
            tmp_x.push_back(init_state[i]);
        }
        x.push_back(tmp_x);
        tmp_x.clear();

        if (save_step) {
            float* h_x = (float*) malloc(sizeof(float)*n*(steps+1)*state_dim);
            float* h_u = (float*) malloc(sizeof(float)*steps*act_dim);
            float* h_e = (float*) malloc(sizeof(float)*n*steps*act_dim);
            //float* u_prev = (float*) malloc(sizeof(float)*steps*act_dim);

            model->get_inf(h_x, h_u, h_e, cost, beta, nabla, weight);
            std::cout << out_step_file + std::to_string(t) << std::endl;
            to_csv2(out_step_file + std::to_string(t), h_x, h_u, u_prev, h_e, cost, beta, nabla, weight, n, steps, state_dim, act_dim);
            free(h_x);
            free(h_u);
            free(h_e);
        }
        //std::cout << "next_act: " << next_act[0] << ", " << next_act[1] << std::endl;
        //std::cout << "Position: " << init_state[0] << ", " << init_state[1] << std::endl;

        model->set_x(init_state);

        t += 1;
    }

    if (save_traj) {
        to_csv_traj(out_traj_file, x, u);
    }

    free(u_prev);

    std::cout << "Freeing memory... : " << std::flush;
    free_controller_memory(&next_act,
                           &init_state,
                           &init_actions,
                           &cost,
                           &beta,
                           &nabla,
                           &weight);
    free_paresd_data(&noise, &init_next, &max_a, &goal, &cost_q);
    delete model;
    std::cout << "Done" << std::endl;

}

void parse_argument (int argc,
                     char const* argv[],
                     std::string& config,
                     std::string& mjkey,
                     std::string& stepfile,
                     std::string& trajfile) {
    try {
        TCLAP::CmdLine cmd("Mppi controller", ' ', "0.0");
        TCLAP::ValueArg<std::string> configArg("c",
                                               "config",
                                               "Config file",
                                               false,
                                               "../config/point_mass.yaml",
                                               "string",
                                               cmd);

        TCLAP::ValueArg<std::string> mjkeyArg("k",
                                              "key",
                                              "Mujoco key file",
                                              false,
                                              "../lib/contrib/mjkey.txt",
                                              "string",
                                              cmd);

        TCLAP::ValueArg<std::string> outStepArg("s",
                                                "step-save",
                                                "Step save output file",
                                                false,
                                                "to_plot.csv",
                                                "string",
                                                cmd);

        TCLAP::ValueArg<std::string> ouTrajtArg("t",
                                                "traj-save",
                                                "traj save output file",
                                                false,
                                                "traj_to_plot.csv",
                                                "string",
                                                cmd);

        cmd.parse(argc, argv);

        config = configArg.getValue();
        mjkey = mjkeyArg.getValue();
        stepfile = outStepArg.getValue();
        trajfile = ouTrajtArg.getValue();

        std::cout << "Argument parsed" << std::endl;

    } catch (TCLAP::ArgException &e) {
        std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
    }
}

void parse_config (std::string& config_file,
                   std::string& model_file,
                   int& samples,
                   int& state_dim,
                   int& act_dim,
                   int& horizon,
                   float& dt,
                   float& lambda,
                   float** noise,
                   float** init,
                   float** max_a,
                   float** goal,
                   std::string& cost_type,
                   float** cost_q) {
    float* tmp_noise;
    float* tmp_init;
    float* tmp_max_a;
    float* tmp_goal;
    float* tmp_cost_q;

    YAML::Node config = YAML::LoadFile(config_file);

    /* env section */
    if (!config["env"])  {
        std::cout << "Please provide a env file in the config file" << std::endl;
        exit(1);
    }
    model_file = config["env"].as<std::string>();

    /* Sample section */
    if (!config["samples"])  {
        std::cout << "Please provide the number of samples in the config file" << std::endl;
        exit(1);
    }
    samples = config["samples"].as<int>();

    /* State section */
    if (!config["state-dim"])  {
        std::cout << "Please provide the state dimension in the config file" << std::endl;
        exit(1);
    }
    state_dim = config["state-dim"].as<int>();

    /* Action section */
    if (!config["action-dim"])  {
        std::cout << "Please provide the action dimension in the config file" << std::endl;
        exit(1);
    }
    act_dim = config["action-dim"].as<int>();

    /* Horizon section */
    if (!config["horizon"])  {
        std::cout << "Please provide the prediction horizon in the config file" << std::endl;
        exit(1);
    }
    horizon = config["horizon"].as<int>();

    /* Timestep section */
    if (!config["dt"])  {
        std::cout << "Please provide the time step in the config file" << std::endl;
        exit(1);
    }
    dt = config["dt"].as<float>();

    /* Lambda section */
    if (!config["lambda"])  {
        std::cout << "Please provide a env file in the config file" << std::endl;
        exit(1);
    }
    lambda = config["lambda"].as<float>();

    /* Noise section */
    {
        if (!config["noise"])  {
            std::cout << "Please provide a noise vector in the config file, should be a array of size action-dim" << std::endl;
            exit(1);
        }
        if (config["noise"].size() != act_dim) {
            std::cout << "Warning: the cost function weights matrix is larger than the action dimension ";
        }
        tmp_noise = (float*) malloc(sizeof(float)*config["noise"].size());
    }


    /* Init action section */
    {
        if (!config["init-act"])  {
            std::cout << "Please provide a init vector in the config file, should be a array of size action-dim" << std::endl;
            exit(1);
        }
        if (config["init-act"].size() != act_dim) {
            std::cout << "Warning: the cost function weights matrix is larger than the action dimension ";
        }
        tmp_init = (float*) malloc(sizeof(float)*config["max-a"].size());
    }


    /* Max action section */
    {
        if (!config["max-a"])  {
            std::cout << "Please provide a max input vector in the config file, should be a array of size action-dim" << std::endl;
            exit(1);
        }
        if (config["max-a"].size() != act_dim) {
            std::cout << "Warning: the input limit is different than the action dimension " << std::endl;
        }
        tmp_max_a = (float*) malloc(sizeof(float)*config["max-a"].size());
    }


    for (std::size_t i=0; i < config["max-a"].size(); i++) {
        tmp_noise[i] = config["noise"][i].as<float>();
        tmp_init[i] = config["init-act"][i].as<float>();
        tmp_max_a[i] = config["max-a"][i].as<float>();
    }

    /* Goal section */
    {
        if (!config["goal"])  {
            std::cout << "Please provide a goal vector in the config file, should be a array of size action-dim" << std::endl;
            exit(1);
        }
        if (config["goal"].size() != state_dim) {
            std::cout << "Warning: the goal size is different than the state dimension " << std::endl;
        }
        tmp_goal = (float*) malloc(sizeof(float)*config["goal"].size());

        for (std::size_t i=0; i < config["goal"].size(); i++) {
            tmp_goal[i] = config["goal"][i].as<float>();
        }
    }

    /* Cost related section  */
    {
        if (!config["cost"])  {
            std::cout << "Please provide cost function in the config file." << std::endl;
            exit(1);
        }

        if (!config["cost"]["type"]) {
            std::cout << "Please provide cost function type in the config file. Currently supported: quadratic " << std::endl;
            exit(1);
        }
        cost_type = config["cost"]["type"].as<std::string>();

        if (!config["cost"]["w"]) {
            std::cout << "Please provide cost function type in the config file. Currently supported: quadratic " << std::endl;
            exit(1);
        }
        if (config["cost"]["w"].size() != state_dim) {
            std::cout << "Warning: the cost function weights matrix is different than the state dimension " << std::endl;
        }
        tmp_cost_q = (float*) malloc(sizeof(float)*config["cost"]["w"].size());

        for (std::size_t i=0; i< config["cost"]["w"].size(); i++) {
            tmp_cost_q[i] = config["cost"]["w"][i].as<float>();
        }
    }

    *noise = tmp_noise;
    *init = tmp_init;
    *max_a = tmp_max_a;
    *goal = tmp_goal;
    *cost_q = tmp_cost_q;

    tmp_noise = nullptr;
    tmp_init = nullptr;
    tmp_max_a = nullptr;
    tmp_cost_q = nullptr;
    tmp_goal = nullptr;

    std::cout << "N " << samples << " steps: " << horizon << " State dim: " << state_dim << std::endl;

}

void init_controller_var (int const& samples,
                          int const& state_dim,
                          int const& act_dim,
                          int const& horizon,
                          float** next_act,
                          float** init_state,
                          float** init_actions,
                          float** cost,
                          float** beta,
                          float** nabla,
                          float** weight) {
      *next_act = (float*) malloc(sizeof(float)*act_dim);
      *init_state = (float*) malloc(sizeof(float)*state_dim);
      *init_actions = (float*) malloc(sizeof(float)*horizon*act_dim);
      *cost = (float*) malloc(sizeof(float)*samples);
      *beta = (float*) malloc(sizeof(float));
      *nabla = (float*) malloc(sizeof(float));
      *weight = (float*) malloc(sizeof(float)*samples);
}

void free_controller_memory(float** next_act,
                            float** init_state,
                            float** init_actions,
                            float** cost,
                            float** beta,
                            float** nabla,
                            float** weight) {
    free(*next_act);
    free(*init_state);
    free(*init_actions);
    free(*cost);
    free(*beta);
    free(*nabla);
    free(*weight);
}

void free_paresd_data(float** noise,
                      float** init_next,
                      float** max_a,
                      float** goal,
                      float** cost_q) {
    free(*noise);
    free(*init_next);
    free(*max_a);
    free(*goal);
    free(*cost_q);
}

void init_action_seq(float* init_actions, int action_dim, int steps) {
    for (int i=0; i < steps; i++) {
        for( int j=0; j < action_dim; j++) {
            init_actions[i*action_dim + j] = 0.;
        }
    }
}

void verify_parse (int n,
                   int state_dim,
                   int act_dim,
                   int steps,
                   float dt,
                   float lambda,
                   float* noise,
                   float* init_next,
                   float* max_a,
                   float* cost_q,
                   float* goal) {
    assert(n == 3);
    assert(state_dim == 4);
    assert(act_dim == 2);
    assert(steps == 12);

    assert(fabs(dt - 0.1) < TOL);
    assert(fabs(lambda - 1.5) < TOL);
    assert(fabs(max_a[0] - 1.2) < TOL);
    assert(fabs(max_a[1] - 1.3) < TOL);

    assert(fabs(noise[0] - 0.24) < TOL);
    assert(fabs(noise[1] - 0.26) < TOL);


    assert(fabs(init_next[0] - 0.1) < TOL);
    assert(fabs(init_next[1] - 0.2) < TOL);

    assert(fabs(cost_q[0] - 1) < TOL);
    assert(fabs(cost_q[1] - 2) < TOL);
    assert(fabs(cost_q[2] - 0.5) < TOL);
    assert(fabs(cost_q[3] - 0.75) < TOL);

    assert(fabs(goal[0] - 1) < TOL);
    assert(fabs(goal[1] - 2) < TOL);
    assert(fabs(goal[2] - 3) < TOL);
    assert(fabs(goal[3] - 4) < TOL);

    std::cout << "Test passed" << std::endl;
}
