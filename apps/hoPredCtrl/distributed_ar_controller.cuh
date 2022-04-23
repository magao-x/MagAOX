#ifndef PCDDSC_CUH
#define PCDDSC_CUH

#include "cuda_runtime.h"
#include "cublas_v2.h"     // if you need CUBLAS v2, include before magma.h

#include "recursive_least_squares.cuh"
#include "new_matrix.cuh"

namespace DDSPC
{

class DistributedAutoRegressiveController{

	public:
		RecursiveLeastSquares* rls;
		
		Matrix *measurement_buffer;
		Matrix *command_buffer;
		Matrix *phi;
		Matrix *xf;
		
		Matrix* newest_measurement;
		Matrix *command;
		Matrix *delta_command;
		Matrix *H, *H11, *H12, *invH11, *invH11sub, *controller, *full_controller;
		Matrix *condition_matrix;
		Matrix *wp;

		cublasHandle_t* handle;

		int nhistory, nfuture, nmodes, nfeatures;
		unsigned long long int buffer_size;
		unsigned long long int buffer_index;

		Matrix* lambda;
		float gamma;

		bool use_predictor;
		float int_gain;
		float int_leakage;

		DistributedAutoRegressiveController(cublasHandle_t* new_handle, int num_history, int num_future, int num_modes, float new_gamma, float* new_lambda, float P0);
		~DistributedAutoRegressiveController();

		void set_handle(cublasHandle_t* new_handle);

		// void add_measurement(float* new_measurement);
		void set_new_regularization(float* new_lambda);
		inline void set_new_gamma(float new_gamma){
			gamma = new_gamma;
		};

		inline void set_integrator(bool new_use_predictor, float new_int_gain, float new_int_leakage){
				use_predictor = new_use_predictor;
				int_gain = new_int_gain;
				int_leakage = new_int_leakage;
		};

		void reset_data_buffer();
		void reset_controller();

		void add_measurement(Matrix* new_measurement);
		void update_predictor();
		void update_controller();

		void save_controller_state(std::string filename);
		
		inline Matrix* get_command(){
			return command;
		};

		Matrix* get_new_control_command(float clip_val, Matrix* exploration_signal);

};

}

#endif