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

		DistributedAutoRegressiveController(cublasHandle_t* new_handle, int num_history, int num_future, int num_modes, float gamma, float* new_lambda, float P0);
		~DistributedAutoRegressiveController();

		void set_handle(cublasHandle_t* new_handle);

		// void add_measurement(float* new_measurement);
		void set_new_regularization(float* new_lambda);
		void reset_data_buffer();
		void reset_controller();

		void add_measurement(Matrix* new_measurement);
		void update_predictor();
		void update_controller();
		Matrix* get_command(float clip_val, Matrix* exploration_signal);

};

}

#endif