#ifndef PCARC_HPP
#define PCARC_HPP

#include <Eigen/Dense>

#include "utils.hpp"
#include "recursive_least_squares.hpp"
#include "qrd_rls.hpp"

namespace DDSPC
{

class PredictiveController{

	private:
        RecursiveLeastSquares* rls;
        QRDRecursiveLeastSquares* qrd_rls;

        uint buffer_size;
        uint measurement_head {0};
        uint command_head {0};
        Matrix measurement_buffer;
        Matrix command_buffer;

        Matrix* regularization_matrix;
        bool use_regularization_matrix_01 {true};
        bool do_switch_regularization_matrix {false};
        Matrix regularization_matrix_01;
        Matrix regularization_matrix_02;

        Matrix controller;
        Matrix integrator;

        int _num_modes;
        int _num_future;
        int _num_history;
        realT _gain;
        realT _delta_max;
        realT _regularization;

        int num_predictors;
        int num_features;
        int num_correlations;

	public:
        bool use_qrd {false};
        
        PredictiveController(int num_actuators, int num_history, int num_future, realT gain, realT gamma, realT initial_regularization, realT initial_covariance);
		~PredictiveController();

        void set_regularization(realT new_regularization);
        inline Matrix get_prediction_matrix(){
            if(use_qrd){
                return qrd_rls->prediction_matrix;
            }else{
                return rls->prediction_matrix;
            }
        };

        void reset();
        void reset_buffers();

        Matrix get_measurement_future();
        Matrix get_measurement_past();

        Matrix get_command_future(int skip_cmds);
        Matrix get_command_past();

        Matrix get_current_command_past(int num_steps);
        Matrix get_current_measurement_past(int num_steps);

        void get_current_past();

        Matrix calculate_command(Matrix new_measurement, Matrix exploration_noise);
        void update_system();
        void update_controller();

        void save_state(const std::string &filename);
        void load_state(const std::string &filename);
};

}

#endif
