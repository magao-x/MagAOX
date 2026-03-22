#define EIGEN_DONT_PARALLELIZE
#include "testPredCtrl.hpp"
#include "ar_controller.hpp"

#include <random>
#include <cmath>

#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>

double variance(double data[], int size, int num_skip=0){
    double mean = 0.0;
    for(int i=num_skip; i<size; i++){
        mean += data[i];
    }
    mean /= (size - num_skip);
    std::cout << "Mean: " << mean << std::endl;

    double var = 0.0;
    for(int i=num_skip; i<size; i++){
        var += (data[i] - mean) * (data[i] - mean);
    }

    return var / (size - num_skip);
}

double standard_dev(double data[], int size, int num_skip=0){
    return std::sqrt(variance(data, size, num_skip));
}


float variance(float data[], int size, int num_skip=0){
    float mean = 0.0;
    for(int i=num_skip; i<size; i++){
        mean += data[i];
    }
    mean /= (size - num_skip);
    std::cout << "Mean: " << mean << std::endl;

    float var = 0.0;
    for(int i=num_skip; i<size; i++){
        var += (data[i] - mean) * (data[i] - mean);
    }

    return var / (size - num_skip);
}

float standard_dev(float data[], int size, int num_skip=0){
    return std::sqrt(variance(data, size, num_skip));
}

void write_to_file(std::string filename, double data[], int size){
    std::ofstream out(filename);
    for(int i=0; i<size; i++){
        out << data[i];
        if(i < (size - 1))
            out << ',';
    }
}


void write_to_file(std::string filename, float data[], int size){
    std::ofstream out(filename);
    for(int i=0; i<size; i++){
        out << data[i];
        if(i < (size - 1))
            out << ',';
    }
}

int main(int argc, char **argv){
    std::default_random_engine generator;
    std::normal_distribution<DDSPC::realT> distribution(0, 1.0);

    int num_steps = 10000;
    DDSPC::realT x[num_steps] = {0.0};
    DDSPC::realT err[num_steps] = {0.0};
    DDSPC::realT signal[num_steps] = {0.0};

    DDSPC::realT err_pc[num_steps] = {0.0};
    DDSPC::realT signal_pc[num_steps] = {0.0};

    for(int i=0; i<num_steps; i++){
        x[i] = std::sin(2 * 3.14 * 20.0 * i / 1000.0);
    }

    DDSPC::realT gain = 0.5;
    DDSPC::realT gamma = 1.001;
    DDSPC::realT initial_regularization = 100.0;
    DDSPC::realT initial_covariance = 1.e5;

    int num_history = 10;
    int num_future = 3;
    int num_actuators = 2;

    DDSPC::Matrix measurement;
    measurement.resize(num_actuators,1);

    DDSPC::Matrix exploration_noise;
    exploration_noise.resize(num_actuators,1);

    DDSPC::PredictiveController controller = DDSPC::PredictiveController(num_actuators, num_history, num_future, gain, gamma, initial_regularization, initial_covariance);

    std::vector<std::vector<DDSPC::realT>> signal(num_actuators, std::vector<DDSPC::realT>(num_steps, 0.0));
    std::vector<std::vector<DDSPC::realT>> err(num_actuators, std::vector<DDSPC::realT>(num_steps, 0.0));
    std::vector<std::vector<DDSPC::realT>> signal_pc(num_actuators, std::vector<DDSPC::realT>(num_steps, 0.0));
    std::vector<std::vector<DDSPC::realT>> err_pc(num_actuators, std::vector<DDSPC::realT>(num_steps, 0.0));

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    double command_calc = 0.0;
    double update_system = 0.0;
    double update_controller = 0.0;
    for(int i=0; i < (num_steps - 1); i++){
        if(i == 250)
            controller.set_regularization(1.0);
        if(i == 500)
            controller.set_regularization(0.1);

        for(int k=0; k<num_actuators; k++){
            if(i < 500){
                exploration_noise(k,0) = 0.1 * distribution(generator);
            } else {
                exploration_noise(k,0) = 0.0;
            }

            err[k][i] = x[i] + signal[k][i];
            signal[k][i+1] = signal[k][i] - gain * err[k][i] + exploration_noise(k, 0);

            err_pc[k][i] = x[i] + signal_pc[k][i];
            measurement(k,0) = err_pc[k][i];
        }

        begin = std::chrono::steady_clock::now();
        DDSPC::Matrix new_command = controller.calculate_command(measurement, exploration_noise);
        end = std::chrono::steady_clock::now();
        command_calc += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();

        for(int k=0; k<num_actuators; k++){
            signal_pc[k][i + 1] = signal_pc[k][i] + new_command(k,0);
        }

        if((i+1) > (num_future + num_history)){
            begin = std::chrono::steady_clock::now();
            controller.update_system();
            end = std::chrono::steady_clock::now();
            update_system += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();

            begin = std::chrono::steady_clock::now();
            controller.update_controller();
            end = std::chrono::steady_clock::now();
            update_controller += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
        }
    }

    std::cout << command_calc / num_steps / 1000.0 << "  " << update_system / num_steps / 1000.0 << "  " << update_controller / num_steps / 1000.0 << std::endl;

    std::cout<< standard_dev(x, num_steps, 500) << std::endl;
    std::cout<< standard_dev(err[0].data(), num_steps, 500) << std::endl;
    std::cout<< standard_dev(err_pc[0].data(), num_steps, 500) << std::endl;

    write_to_file("x.csv", x, num_steps);
    write_to_file("err.csv", err[0].data(), num_steps);
    write_to_file("err_pc.csv", err_pc[0].data(), num_steps);

    return 0;

}