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


void to_csv (std::string filename,
            float* x,
            float* u,
            int sample,
            int size,
            int s_dim,
            int a_dim)
{
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
            }else
                outfile << ", , , , ";
            if (i*size + j < sample) {
                outfile << "," << cost[i*size+j] << "," << w[i*size+j];
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
                     std::string& outfile);

void parse_config (std::string& config_file,
                   std::string& model_file,
                   int& samples,
                   int& state_dim,
                   int& act_dim,
                   int& horizon,
                   float& lambda,
                   float** noise,
                   float** init,
                   float** max_a,
                   std::string& cost_type,
                   float** cost_q);

int main (int argc, char const* argv[]) {

    std::string config_file;
    std::string mjkeyFile;
    std::string outFile;
    std::string model_file;
    std::string cost_type;

    int n(0);
    int state_dim(0);
    int act_dim(0);
    int steps(0);
    float lambda;
    float* noise;
    float* init;
    float* max_a;
    float* cost_q;

    parse_argument(argc, argv, config_file, mjkeyFile, outFile);

    std::cout << "Config: " << config_file << std::endl;
    std::cout << "MjKey: " << mjkeyFile << std::endl;
    std::cout << "Outfile: " << outFile << std::endl;

    std::cout << max_a << std::endl;
    parse_config(config_file,
                 model_file,
                 n,
                 state_dim,
                 act_dim,
                 steps,
                 lambda,
                 &noise,
                 &init,
                 &max_a,
                 cost_type,
                 &cost_q);
    std::cout << max_a << std::endl;

    std::cout << "Parse config output: " << model_file << " "
              << n << " "
              << state_dim << " "
              << act_dim << " "
              << steps << " "
              << lambda << " " << std::endl;

    std::cout << "max_a: ";
    for (int i = 0; i < act_dim; i++) {
        std::cout << max_a[i] << ' ';
    }
    std::cout << std::endl;

    std::cout << "Init: ";
    for (int i = 0; i < act_dim; i++) {
        std::cout << init[i] << ' ';
    }
    std::cout << std::endl;

    std::cout << "Noise: ";
    for (int i = 0; i < act_dim; i++) {
        std::cout << noise[i] << ' ';
    }
    std::cout << std::endl;

    std::cout << "Cost_q: ";
    for (int i = 0; i < state_dim; i++) {
        std::cout << cost_q[i] << ' ';
    }
    std::cout << std::endl;

    //char*  model_file = "../envs/point_mass.xml";
    //char* mjkey = "../lib/contrib/mjkey.txt";
    std::cout << "N " << n << " STEPS: " << steps << " State dim: " << state_dim << std::endl;

    float* x = (float*) malloc(sizeof(float)*state_dim);
    float* cost = (float*) malloc(sizeof(float)*n);
    float* beta = (float*) malloc(sizeof(float));
    float* nabla = (float*) malloc(sizeof(float));
    float* weight = (float*) malloc(sizeof(float)*n);

    /*
    * The state data stored on host. In this example,
    * the state is only one scalar but is stored on a
    * array with all the following states. Thus we need a int[n]
    * array to the input data.
    */

    /*
    float* h_x = (float*) malloc(sizeof(float)*n*(STEPS+1)*state_dim);
    float* h_u = (float*) malloc(sizeof(float)*STEPS*act_dim);
    float* h_e = (float*) malloc(sizeof(float)*n*STEPS*act_dim);
    float* u_prev = (float*) malloc(sizeof(float)*STEPS*act_dim);

    float* goal = (float*) malloc(sizeof(float)*state_dim);
    float* w = (float*) malloc(sizeof(float)*state_dim);
    // allocate and init and res data.
    goal[0] = 1.0;
    goal[1] = 0.0;
    goal[2] = 0.0;
    goal[3] = 0.0;

    w[0] = 50.0;
    w[1] = 50.0;
    w[2] = 0.25;
    w[3] = 0.25;

    float dt = 0.1;

    //bool test = false;
    bool save = true;
    std::string filename("to_plot.csv");
    PointMassEnv env = PointMassEnv(model_file, mjkey, true);

    PointMassModel* model = new PointMassModel(n, STEPS, dt, state_dim, act_dim, false);
    bool done=false;

    env.get_x(x);

    for (int j=0; j < STEPS; j++){
        h_u[(j*act_dim)+0] = 0.;
        h_u[(j*act_dim)+1] = 0.;
    }
    // send the data on the device.

    float* next_act = (float*) malloc(sizeof(float)*act_dim);

    model->memcpy_set_data(x, h_u, goal, w);


    // run the multiple simulation on the device.
    while(!done){
        model->get_u(u_prev);
        //t1 = std::chrono::system_clock::now();
        model->get_act(next_act);
        //t2 = std::chrono::system_clock::now();
        //fp_ms += t2 - t1;

        done = env.simulate(next_act);
        std::cout << "next_act: " << next_act[0] << ", " << next_act[1] << '\n';
        env.get_x(x);
        model->set_x(x);
    }

    //send act to sim;

    // collect new state;

    // set state in controller;

    // next step


    //delta = fp_ms.count();


    //std::cout << "GPU execution time: " << delta << "ms" << std::endl;

    if(save){
        model->get_inf(h_x, h_u, h_e, cost, beta, nabla, weight);
        to_csv2(filename, h_x, h_u, u_prev, h_e, cost, beta, nabla, weight, n, STEPS, state_dim, act_dim);
    }

    std::cout << "Freeing memory... : " << std::flush;
    free(h_x);
    free(h_u);
    free(h_e);
    free(x);
    free(w);
    free(goal);
    std::cout << "Done" << std::endl;

    delete model;
    cudaDeviceReset();

    */
}

void parse_argument (int argc,
                     char const* argv[],
                     std::string& config,
                     std::string& mjkey,
                     std::string& outfile) {
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

        TCLAP::ValueArg<std::string> outArg("o",
                                            "out",
                                            "Outpute file",
                                            false,
                                            "to_plot.csv",
                                            "string",
                                            cmd);

        cmd.parse(argc, argv);

        config = configArg.getValue();
        mjkey = mjkeyArg.getValue();
        outfile = outArg.getValue();

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
                   float& lambda,
                   float** noise,
                   float** init,
                   float** max_a,
                   std::string& cost_type,
                   float** cost_q) {
    float* tmp_noise;
    float* tmp_init;
    float* tmp_max_a;
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
        tmp_noise = (float*) malloc(sizeof(float)*config["max-a"].size());
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
        tmp_noise[i] = config["max-a"][i].as<float>();
        tmp_init[i] = config["init-act"][i].as<float>();
        tmp_max_a[i] = config["noise"][i].as<float>();
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
    *cost_q = tmp_cost_q;

    tmp_noise = nullptr;
    tmp_init = nullptr;
    tmp_max_a = nullptr;
    tmp_cost_q = nullptr;


}
