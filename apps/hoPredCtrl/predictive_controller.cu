#include "predictive_controller.cuh"
#include <iostream>

namespace DDSPC
{	
	PredictiveController::PredictiveController(int num_history, int num_future, int num_modes, int num_measurements, float gamma, float lambda, float P0){
		cudaError_t cudaerr = cudaSetDevice(0);
		check_cuda_error(cudaerr);

		cublasStatus_t stat = cublasCreate(&handle);
		if (stat != CUBLAS_STATUS_SUCCESS)
			printf ("CUBLAS initialization failed\n");

		m_num_history = num_history;
		m_num_future = num_future;
		m_num_modes = num_modes;
		m_num_measurements = num_measurements;

		m_gamma = gamma;
		m_P0 = P0;

		m_lambda = new float[num_modes];
		for(int i=0; i<num_modes; ++i)
			m_lambda[i] = lambda;

		m_wfs_measurement = new Matrix(0.0, num_measurements, 1);
		m_wfs_measurement->set_handle(&handle);

		//m_measurement = make_col_vector(0.0, num_modes, 1);		
		m_measurement = new Matrix(1.0, num_modes, 1);
		m_measurement->set_handle(&handle);

		m_exploration_signal = make_col_vector(0.0, num_modes, 1);
		m_exploration_signal->set_handle(&handle);
		// for(int kk=0; kk<5;kk++){
		//	m_exploration_signal->cpu_data[0][kk] = 0.01 * kk;
		// }
		m_exploration_signal->to_gpu();
		
		m_exploration_buffer = nullptr;
		
		m_command = make_col_vector(0.0, num_modes, 1);
		m_command->set_handle(&handle);

		m_interaction_matrix = new Matrix((float) 1.0, num_modes, num_measurements, 1);
		m_interaction_matrix->set_handle(&handle);
		m_interaction_matrix->to_gpu();

		controller = new DistributedAutoRegressiveController(&handle, m_num_history, m_num_future, m_num_modes, m_gamma, m_lambda, m_P0);
		controller->set_handle(&handle);
	};

	PredictiveController::~PredictiveController(){

		cublasDestroy(handle);
		delete [] m_lambda;
		//delete controller;
		
		delete m_wfs_measurement;
		
		delete m_measurement;
		delete m_exploration_signal;
		delete m_command;
		delete m_interaction_matrix;
		
		if (m_exploration_buffer){
			delete m_exploration_buffer;
		}

	};

	void PredictiveController::create_exploration_buffer(float rms, int exploration_buffer_size){
		std::cout << " Create buffer with size: " << exploration_buffer_size << std::endl;
		m_exploration_index = 0;
		// Create the buffer size
		// We do this by finding the next power of two!
		// With the bit shifts we also make a bit mask which will make it easier to cycle through the buffer.
		auto num_bits = find_next_power_of_2(exploration_buffer_size + 1);
		m_exploration_buffer_size = exploration_buffer_size;
		// for(unsigned long long int i =0; i < num_bits ; i++){
		// 	m_exploration_buffer_size |= 1 << i;
		// }
		
		// First clean the current_buffer and only if it exists
		if(m_exploration_buffer){
			delete m_exploration_buffer;
		}

		// Replace with random matrix
		m_exploration_buffer = make_random_binary_matrix(rms, m_num_modes, 1, m_exploration_buffer_size + 1);
	}

	void PredictiveController::get_next_exploration_signal(){
		if(m_exploration_index < m_exploration_buffer_size){
			// Copy the data from the data buffer to the new vector
			cudaMemcpy(m_exploration_signal->gpu_data[0], m_exploration_buffer->gpu_data[m_exploration_index], m_num_modes * sizeof(float), cudaMemcpyDeviceToDevice);
			
			//gpu_col_copy(m_exploration_signal, 0, 0, m_exploration_buffer, 0, m_exploration_index & m_exploration_buffer_size);
			m_exploration_index += 1;
			// if(m_exploration_index == m_exploration_buffer_size){
			// 	m_exploration_index -= m_exploration_buffer_size;
			// }
		}else{
			cudaMemset(m_exploration_signal->gpu_data[0], 0, m_num_modes * sizeof(float));
		}

	}

	void PredictiveController::set_new_regularization(float lambda){
		for(int i=0; i < m_num_modes; ++i){
			m_lambda[i] = lambda;
		}
		controller->set_new_regularization(m_lambda);
	};
	
	void PredictiveController::set_interaction_matrix(float* interaction_matrix){
		cpu_full_copy(m_interaction_matrix, interaction_matrix);
		m_interaction_matrix->to_gpu();
	};

	void PredictiveController::add_measurement(float* new_wfs_measurement){
		// Copy measurement to GPU
		cpu_full_copy(m_wfs_measurement, new_wfs_measurement);
		m_wfs_measurement->to_gpu();
		
		m_interaction_matrix->dot(m_wfs_measurement, m_measurement);

		// Add data to controller
		controller->add_measurement(m_measurement);
	};

	float* PredictiveController::get_command(float clip_val){
		get_next_exploration_signal();
		
		// Determine the command vector
		m_command = controller->get_command(clip_val, m_exploration_signal);
		
		// Copy into shmimstream
		return m_command->cpu_data[0];
	};
	
}