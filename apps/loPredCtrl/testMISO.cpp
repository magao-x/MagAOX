#define EIGEN_DONT_PARALLELIZE
#include "testMISO.hpp"

#include <random>
#include <cmath>

#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>

#include <omp.h>

int main(int argc, char **argv){
    omp_set_dynamic(0);      // Do not allow runtime changes
    omp_set_num_threads(16); // Use exactly 16 OpenMP threads

    std::default_random_engine generator;
    std::normal_distribution<DDSPC::realT> distribution(0, 1.0);

    int num_steps = 1000;
    DDSPC::realT x[num_steps] = {0.0};

    for(int i=0; i<num_steps; i++){
        x[i] = std::sin(2 * 3.14 * 20.0 * i / 1000.0);
    }

    DDSPC::realT gain = -0.5;
    DDSPC::realT gamma = 1.000;
    DDSPC::realT initial_regularization = 100.0;
    DDSPC::realT initial_covariance = 1e-5;

    int num_history = 5;
    int num_future = 2;
    int num_actuators = 1;
    int num_modes = 32 * 16;

    DDSPC::Matrix measurement;
    measurement.resize(num_actuators,1);

    DDSPC::Matrix exploration_noise;
    exploration_noise.resize(num_actuators,1);

    // DDSPC::PredictiveController controllers[num_modes];  // Array of 3 Student objects
    std::vector<DDSPC::PredictiveController> controllers;
    controllers.reserve(num_modes);
    for(int i = 0; i < num_modes; ++i) {
        controllers.emplace_back(num_actuators, num_history, num_future, gain, gamma, initial_regularization, initial_covariance);
        controllers.back().use_qrd = true;  // Set use_qrd to true for each controller
    }

    std::vector<std::vector<DDSPC::realT>> signal(num_actuators, std::vector<DDSPC::realT>(num_steps, 0.0));
    std::vector<std::vector<DDSPC::realT>> err(num_actuators, std::vector<DDSPC::realT>(num_steps, 0.0));

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    double command_calc = 0.0;
    double update_controller = 0.0;

    for(int i=0; i < (num_steps - 1); i++){
        if(i == 250){
            for(int m = 0; m < num_modes; ++m) {
                controllers[m].set_regularization(10.0);
            }
        }

        if(i == 500){
            for(int m = 0; m < num_modes; ++m) {
                controllers[m].set_regularization(0.1);
            }
        }

        for(int k=0; k<num_actuators; k++){
            if(i < 500){
                exploration_noise(k,0) = 0.1 * distribution(generator);
            } else {
                exploration_noise(k,0) = 0.0;
            }

            err[k][i] = x[i] + signal[k][i];
            measurement(k,0) = err[k][i] + 0. * distribution(generator);
        }
       
        begin = std::chrono::steady_clock::now();
        #pragma omp parallel
        for(int m = 0; m < (num_modes/16); ++m) {
            for(int ki = 0; ki < 16; ++ki){
                DDSPC::Matrix new_command = controllers[32 * m + ki].calculate_command(measurement, exploration_noise);
            }
        }
        end = std::chrono::steady_clock::now();
        command_calc += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();

        if((i+1) > (num_future + num_history)){
            begin = std::chrono::steady_clock::now();
            #pragma omp parallel
            for(int m = 0; m < (num_modes/32); ++m) {
                for(int ki = 0; ki < 32; ++ki){
                    controllers[32 * m + ki].update_system();
                    controllers[32 * m + ki].update_controller();
                } 
            }
            end = std::chrono::steady_clock::now();
            update_controller += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();

        }
    }
    std::cout << "Timing RLS Method: " << std::endl;
    std::cout << command_calc / num_steps / 1000.0 << "   " << update_controller / num_steps / 1000.0 << std::endl;


    
    {
        printf("Thread %d\n", omp_get_thread_num());
    }
    return 0;
}