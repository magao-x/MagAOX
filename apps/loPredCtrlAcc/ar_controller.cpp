#include "ar_controller.hpp"
#include <algorithm>
#include <fstream>
#include <stdexcept>

namespace DDSPC
{

PredictiveController::PredictiveController(int num_actuators, int num_history, int num_future, realT gain, realT gamma, realT initial_regularization, realT initial_covariance, int num_accel_channels, int accel_history){
    // Settable properties
    _num_modes = num_actuators;
    _num_history = num_history;
    _num_future = num_future;
    _num_accel_channels = std::max(0, num_accel_channels);
    _num_accel_history = std::max(0, accel_history);
    _gain = gain;
    _delta_max = 0.5;
    _regularization = initial_regularization;

    // Derived properties
    num_predictors = _num_future * _num_modes;
    num_accel_features = _num_accel_history * _num_accel_channels;
    num_features = (_num_future - 1 + 2 * _num_history) * _num_modes + num_accel_features;
    num_state_features = (2 * _num_history - 1) * _num_modes + num_accel_features;
    num_correlations = _num_future * _num_modes;

    // Data buffers
    buffer_size = find_next_power_of_2(2 * (num_history + num_future + _num_accel_history + 1));

    measurement_head = 0;
    measurement_buffer.resize(buffer_size, _num_modes);
    measurement_buffer.setZero();

    command_head = 0;
    command_buffer.resize(buffer_size, _num_modes);
    command_buffer.setZero();

    accel_head = 0;
    accel_buffer.resize(buffer_size, _num_accel_channels);
    accel_buffer.setZero();

    // The learner
    rls = new RecursiveLeastSquares(num_predictors, num_features, gamma, initial_covariance);
    qrd_rls = new QRDRecursiveLeastSquares(num_predictors, num_features, gamma, initial_covariance);

    // Initializing the controller
    controller.resize(_num_modes, num_state_features);
    controller.setZero();

    integrator.resize(_num_modes, num_state_features);
    integrator.setZero();

    for(int i=0; i < _num_modes; i++){
        int index = (_num_history - 1) * _num_modes + num_accel_features + i;
        integrator(i, index) = _gain;
    }

    // Set the regularization matrix
    regularization_matrix_01.resize(num_correlations, num_correlations);
    regularization_matrix_01.setZero();

    regularization_matrix_02.resize(num_correlations, num_correlations);
    regularization_matrix_02.setZero();

    for(int i = 0; i<num_correlations; i++){
        regularization_matrix_01(i, i) = initial_regularization;
        regularization_matrix_02(i, i) = initial_regularization;
    }

    regularization_matrix = &regularization_matrix_01;
};


PredictiveController::~PredictiveController(){
    delete rls;
    delete qrd_rls;
}

void PredictiveController::reset(){
    controller.resize(_num_modes, num_state_features);
    controller.setZero();
    integrator.resize(_num_modes, num_state_features);
    integrator.setZero();

    for(int i=0; i < _num_modes; i++){
        int index = (_num_history - 1) * _num_modes + num_accel_features + i;
        integrator(i, index) = _gain;
    }

    rls->reset();
    qrd_rls->reset();

    reset_buffers();

}

void PredictiveController::reset_buffers(){
    measurement_head = 0;
    measurement_buffer.resize(buffer_size, _num_modes);
    measurement_buffer.setZero();

    command_head = 0;
    command_buffer.resize(buffer_size, _num_modes);
    command_buffer.setZero();

    accel_head = 0;
    accel_buffer.resize(buffer_size, _num_accel_channels);
    accel_buffer.setZero();

}

void PredictiveController::set_regularization(realT new_regularization){
   if(use_regularization_matrix_01){
        for(int i = 0; i<num_correlations; i++){
            regularization_matrix_02(i, i) = new_regularization;
        }
        use_regularization_matrix_01 = false;
   }else{
        for(int i = 0; i<num_correlations; i++){
            regularization_matrix_01(i, i) = new_regularization;
        }
        use_regularization_matrix_01 = true;
   }
   do_switch_regularization_matrix = true;
}

Matrix PredictiveController::get_measurement_future(){
    Matrix future_vec;
    future_vec.resize(_num_future * _num_modes, 1);

    for(int i=0; i<_num_future; i++){
        auto dat = measurement_buffer.row((measurement_head - i - 1) & (buffer_size - 1));
        for(int j=0; j < _num_modes; j++){
            future_vec(i * _num_modes + j, 0) = dat(0, j);
        }
    }

    return future_vec;
}

Matrix PredictiveController::get_measurement_past(){
    Matrix past_vec;
    past_vec.resize(_num_history * _num_modes, 1);

    // This is a smarter way to ravel the data!
    // VectorXd B(Map<VectorXd>(A.data(), A.cols()*A.rows()));
    for(int i=0; i<_num_history; i++){
        auto dat = measurement_buffer.row((measurement_head - i - _num_future - 1) & (buffer_size - 1));
        for(int j=0; j < _num_modes; j++){
            past_vec(i * _num_modes + j, 0) = dat(0, j);
        }
    }

    return past_vec;
}

Matrix PredictiveController::get_command_future(int skip_cmds=0){
    Matrix future_vec;
    if(true){
        future_vec.resize((_num_future - skip_cmds) * _num_modes, 1);

        for(int i=0; i < (_num_future - skip_cmds); i++){
            int offset = skip_cmds * _num_modes;
            auto dat = command_buffer.row((command_head - i - 1 - offset) & (buffer_size - 1));
            for(int j=0; j < _num_modes; j++){
                future_vec(i * _num_modes + j, 0) = dat(0, j);
            }
        }
    }else{
        future_vec.resize(_num_future * _num_modes, 1);

        for(int i=0; i<_num_future; i++){
            auto dat = command_buffer.row((command_head - i - 1) & (buffer_size - 1));
            for(int j=0; j < _num_modes; j++){
                future_vec(i * _num_modes + j, 0) = dat(0, j);
            }
        }
    }

    return future_vec;
}

Matrix PredictiveController::get_command_past(){
    Matrix past_vec;
    past_vec.resize(_num_history * _num_modes, 1);

    for(int i=0; i<_num_history; i++){
        auto dat = command_buffer.row((command_head - i - _num_future - 1) & (buffer_size - 1));
        for(int j=0; j < _num_modes; j++){
            past_vec(i * _num_modes + j) = dat(0, j);
        }
    }

    return past_vec;
}

Matrix PredictiveController::get_accelerometer_past(){
    Matrix past_vec;
    past_vec.resize(num_accel_features, 1);
    past_vec.setZero();

    if(num_accel_features == 0){
        return past_vec;
    }

    for(int i=0; i < _num_accel_history; i++){
        auto dat = accel_buffer.row((accel_head - i - 1) & (buffer_size - 1));
        for(int j=0; j < _num_accel_channels; j++){
            past_vec(i * _num_accel_channels + j, 0) = dat(0, j);
        }
    }

    return past_vec;
}

void PredictiveController::push_accelerometer_sample(const Matrix &new_acceleration){
    if(_num_accel_channels <= 0){
        return;
    }

    if(new_acceleration.rows() != _num_accel_channels || new_acceleration.cols() != 1){
        return;
    }

    for(int j=0; j < _num_accel_channels; j++){
        accel_buffer((accel_head & (buffer_size - 1)), j) = new_acceleration(j, 0);
    }
    accel_head++;
}

Matrix PredictiveController::get_current_measurement_past(int num_steps){
    Matrix past_vec;
    past_vec.resize(num_steps * _num_modes, 1);

    for(int i=0; i<num_steps; i++){
        auto dat = measurement_buffer.row((measurement_head - i - 1) & (buffer_size - 1));
        for(int j=0; j < _num_modes; j++){
            past_vec(i * _num_modes + j) = dat(0, j);
        }
    }

    return past_vec;
}

Matrix PredictiveController::get_current_command_past(int num_steps){
    Matrix past_vec;
    past_vec.resize(num_steps * _num_modes, 1);

    for(int i=0; i<num_steps; i++){
        auto dat = command_buffer.row((command_head - i - 1) & (buffer_size - 1));
        for(int j=0; j < _num_modes; j++){
            past_vec(i * _num_modes + j) = dat(0, j);
        }
    }

    return past_vec;
}

void PredictiveController::update_system(){
    Matrix future_cmd = get_command_future(1);
    Matrix past_cmd = get_command_past();
    Matrix past_accel = get_accelerometer_past();
    Matrix future_measurement = get_measurement_future();
    Matrix past_measurement = get_measurement_past();

    Matrix prediction_vector;
    prediction_vector.resize(past_measurement.rows() + past_cmd.rows() + past_accel.rows() + future_cmd.rows(), 1);
    prediction_vector << future_cmd, past_cmd, past_accel, past_measurement;

    if(use_qrd){
        qrd_rls->update(&prediction_vector, &future_measurement);
    }else{
        rls->update(&prediction_vector, &future_measurement);
    }
    
}

void PredictiveController::update_controller(){
    Matrix H = get_prediction_matrix().transpose() * get_prediction_matrix();
    Matrix H11 = H.block(0, 0, num_correlations, num_correlations);
    Matrix H21 = H.block(0, num_correlations, num_correlations, H.cols() - num_correlations);

    if (do_switch_regularization_matrix){
        if(use_regularization_matrix_01){
            regularization_matrix = &regularization_matrix_01;
        }else{
            regularization_matrix = &regularization_matrix_02;
        }

        do_switch_regularization_matrix = false;
    }

    Matrix full_controller = -1 * (H11 + H11.maxCoeff() * (*regularization_matrix)).inverse() * H21;
    controller = full_controller.block(full_controller.rows() - _num_modes, 0, _num_modes, full_controller.cols());
}

Matrix PredictiveController::calculate_command(Matrix new_measurement, Matrix exploration_noise){
    for(int j=0; j<_num_modes; j++){
        measurement_buffer( (measurement_head & (buffer_size-1)), j ) = new_measurement(j,0);
    }
    measurement_head++;

    Matrix past_command = get_current_command_past(_num_history - 1);
    Matrix past_accel = get_accelerometer_past();
    Matrix past_measurement = get_current_measurement_past(_num_history);

    Matrix past_vec;
    past_vec.resize(num_state_features, 1);
    past_vec << past_command, past_accel, past_measurement;

    Matrix new_delta = (controller + integrator) * past_vec + exploration_noise;

    if(false){
        for(int i=0; i<_num_modes; i++){
            if( new_delta(i,0) > _delta_max )
                new_delta(i, 0) = _delta_max;

            if( new_delta(i,0) < -_delta_max )
                new_delta(i, 0) = -_delta_max;
        }
    }

    for(int j=0; j<_num_modes; j++){
        command_buffer( (command_head & (buffer_size - 1)), j ) = new_delta(j,0);
    }
    command_head++;

    return new_delta;
}

void PredictiveController::save_state(const std::string &filename) {
    // Save metadata in a JSON sidecar file
    std::string metadata_file = filename + ".meta";
    std::ofstream ofs(metadata_file);
    if (!ofs.is_open()) {
        throw std::runtime_error("Could not open file for save_state metadata: " + metadata_file);
    }

    ofs << "{\n";
    ofs << "  \"num_modes\": " << _num_modes << ",\n";
    ofs << "  \"num_future\": " << _num_future << ",\n";
    ofs << "  \"num_history\": " << _num_history << ",\n";
    ofs << "  \"gain\": " << _gain << ",\n";
    ofs << "  \"delta_max\": " << _delta_max << ",\n";
    ofs << "  \"regularization\": " << _regularization << ",\n";
    ofs << "}\n";
    ofs.close();

    // Save matrices using utils helpers
    DDSPC::save_matrix(filename + ".controller", controller);
    rls->save_state(filename + ".rls");
    qrd_rls->save_state(filename + ".qrd_rls");
}

void PredictiveController::load_state(const std::string &filename) {
    std::string metadata_file = filename + ".meta";
    std::ifstream ifs(metadata_file);
    if (!ifs.is_open()) {
        throw std::runtime_error("Could not open file for load_state metadata: " + metadata_file);
    }

    std::string line;
    std::getline(ifs, line); // {

    std::getline(ifs, line); _num_modes = std::stoi(DDSPC::parse_json_value(line));
    std::getline(ifs, line); _num_future = std::stoi(DDSPC::parse_json_value(line));
    std::getline(ifs, line); _num_history = std::stoi(DDSPC::parse_json_value(line));
    std::getline(ifs, line); _gain = static_cast<realT>(std::stod(DDSPC::parse_json_value(line)));
    std::getline(ifs, line); _delta_max = static_cast<realT>(std::stod(DDSPC::parse_json_value(line)));
    std::getline(ifs, line); _regularization = static_cast<realT>(std::stod(DDSPC::parse_json_value(line)));

    ifs.close();

    controller = DDSPC::load_matrix(filename + ".controller");
    set_regularization(_regularization);
    rls->load_state(filename + ".rls");
    qrd_rls->load_state(filename + ".qrd_rls");
    reset_buffers();
}

}
