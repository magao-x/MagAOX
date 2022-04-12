#include "predictive_controller.cuh"
#include <iostream>

namespace DDSPC
{	
	PredictiveController::PredictiveController(int num_history, int num_future, int num_modes, int num_measurements, float gamma, float lambda, float P0, int num_actuators){
		cudaError_t cudaerr = cudaSetDevice(0);
		check_cuda_error(cudaerr);

		cublasStatus_t stat = cublasCreate(&handle);
		if (stat != CUBLAS_STATUS_SUCCESS)
			printf ("CUBLAS initialization failed\n");

		m_num_history = num_history;
		m_num_future = num_future;
		m_num_modes = num_modes;
		m_num_measurements = num_measurements;
		m_num_actuators = num_actuators;
		m_exploration_buffer_size = 20;

		m_gamma = gamma;
		m_P0 = P0;

		m_lambda = new float[num_modes];
		for(int i=0; i<num_modes; ++i)
			m_lambda[i] = lambda;

		m_wfs_measurement = new Matrix(0.0, num_measurements, 1);
		m_wfs_measurement->set_handle(&handle);

		//m_measurement = make_col_vector(0.0, num_modes, 1);		
		m_measurement = new Matrix(0.0, num_modes, 1);
		m_measurement->set_handle(&handle);

		m_exploration_signal = make_col_vector(0.0, num_modes, 1);
		m_exploration_signal->set_handle(&handle);
		// for(int kk=0; kk<5;kk++){
		//	m_exploration_signal->cpu_data[0][kk] = 0.01 * kk;
		// }
		m_exploration_signal->to_gpu();
		m_exploration_buffer = nullptr;
		

		//
		// m_voltages = make_col_vector(0.0, num_actuators, 1);
		m_voltages = new Matrix(0.0, num_actuators, 1);
		m_voltages->set_handle(&handle);

		m_interaction_matrix = new Matrix((float) 1.0, num_modes, num_measurements, 1);
		m_interaction_matrix->set_handle(&handle);
		m_interaction_matrix->to_gpu();

		m_mode_mapping_matrix = new Matrix((float) 1.0, num_actuators, num_modes, 1);
		m_mode_mapping_matrix->set_handle(&handle);
		m_mode_mapping_matrix->to_gpu();

		controller = new DistributedAutoRegressiveController(&handle, m_num_history, m_num_future, m_num_modes, m_gamma, m_lambda, m_P0);
		controller->set_handle(&handle);
		
		// m_command = new Matrix(0.0, num_modes, 1);
		// m_command->set_handle(&handle);

		// This was okay!
		m_command = controller->get_command();

	};

	PredictiveController::~PredictiveController(){

		cublasDestroy(handle);
		delete [] m_lambda;
		//delete controller;
		
		delete m_wfs_measurement;
		
		delete m_measurement;
		delete m_exploration_signal;
		delete m_command;
		delete m_voltages;

		delete m_interaction_matrix;
		delete m_mode_mapping_matrix;
		
		if (m_exploration_buffer){
			delete m_exploration_buffer;
		}

	};

	void PredictiveController::create_exploration_buffer(float rms, int exploration_buffer_size){
		std::cout << " Create buffer with size: " << exploration_buffer_size << std::endl;
		m_exploration_index = 0;
		m_exploration_buffer_size = exploration_buffer_size;

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
			m_exploration_index += 1;
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

	void PredictiveController::set_new_gamma(float new_gamma){
		controller->set_new_regularization(m_lambda);
	};

	void PredictiveController::set_interaction_matrix(float* interaction_matrix){
		cpu_full_copy(m_interaction_matrix, interaction_matrix);
		m_interaction_matrix->to_gpu();
	};

	void PredictiveController::set_mapping_matrix(float* mapping_matrix){
		cpu_full_copy(m_mode_mapping_matrix, mapping_matrix);
		m_mode_mapping_matrix->to_gpu();
		std::cout << "shape of mode mapping matrix: ";
		m_mode_mapping_matrix->print_shape();
		
		std::cout << "shape of command: ";
		m_command->print_shape();
		
		std::cout << "shape of voltages: ";
		m_voltages->print_shape();

		std::cout << "shape of interaction matrix: ";
		m_interaction_matrix->print_shape();

		std::cout << "shape of wfs output: ";
		m_wfs_measurement->print_shape();

		std::cout << "shape of measurement output: ";
		m_measurement->print_shape();
	};

	void PredictiveController::add_measurement(float* new_wfs_measurement){
		// Copy measurement to GPU
		cpu_full_copy(m_wfs_measurement, new_wfs_measurement);
		m_wfs_measurement->to_gpu();
		
		m_interaction_matrix->dot(m_wfs_measurement, m_measurement);

		// Add data to controller
		controller->add_measurement(m_measurement);
	};

	void PredictiveController::set_zero(){
		m_command->set_to_zero();
	};

	float* PredictiveController::get_command(float clip_val){
		get_next_exploration_signal();
		
		// Determine the command vector
		m_command = controller->get_new_control_command(clip_val, m_exploration_signal);

		// We need to add a modal coefficients to actuators mapping here.
		m_mode_mapping_matrix->dot(m_command, m_voltages);
		m_voltages->to_cpu();

		// Copy into shmimstream
		return m_voltages->cpu_data[0];
	};
	
}