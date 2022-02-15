#ifndef RDDSC_CUH
#define RDDSC_CUH

#include "cuda_runtime.h"
#include "cublas_v2.h"     // if you need CUBLAS v2, include before magma.h

#include "new_matrix.cuh"
#include "distributed_ar_controller.cuh"
#include "utils.cuh"

namespace DDSPC
{

class PredictiveController{
	private:
		cublasHandle_t handle;

		DistributedAutoRegressiveController* controller;

		int m_num_history;
		int m_num_future;
		int m_num_modes;
		int m_num_measurements;

		float m_gamma;
		float* m_lambda;
		float m_P0;

		unsigned long long int m_exploration_buffer_size;
		unsigned long long int m_exploration_index;

		Matrix* m_wfs_measurement;	// contains the new wavefront sensor measurement
		Matrix* m_measurement; 		// contains the new wfs measurement in modal space
		Matrix* m_exploration_signal;
		Matrix* m_command;	
		Matrix* m_interaction_matrix;
		Matrix* m_exploration_buffer;

	protected:

	public:

		PredictiveController(int num_history, int num_future, int num_modes, int num_measurements, float gamma, float lambda, float P0);
		~PredictiveController();

		// Just direct function wrappers
		void reset_data_buffer(){controller->reset_data_buffer();};
		void reset_controller(){controller->reset_controller();};
		void update_predictor(){controller->update_predictor();};
		void update_controller(){controller->update_controller();};

		// Training signal
		void get_next_exploration_signal();
		void create_exploration_buffer(float rms, int exploration_buffer_size);

		// New wrapper functions
		void set_new_regularization(float new_lambda);
		void set_interaction_matrix(float* interaction_matrix);
		void add_measurement(float* new_wfs_measurement);
		float* get_command(float clip_val);
		
};


}

#endif