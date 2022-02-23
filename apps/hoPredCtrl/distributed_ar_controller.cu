#include "device_launch_parameters.h"
#include "utils.cuh"
#include "distributed_ar_controller.cuh"
#include "new_matrix.cuh"
#include <iostream>

namespace DDSPC
{

DistributedAutoRegressiveController::DistributedAutoRegressiveController(cublasHandle_t* new_handle, int num_history, int num_future, int num_modes, float gamma, float* new_lambda, float P0){
	nhistory = num_history;
	nfuture = num_future;
	nmodes = num_modes;
	
	// Use all future DM commands except the most recent because that one will only have an effect in later measurements
	nfeatures = nfuture - 1 + 2 * nhistory;
	
	std::cout << "Nhist :: " << nhistory << std::endl;
	std::cout << "Nfut :: " << nfuture << std::endl;
	std::cout << "Nmodes :: " << nmodes << std::endl;
	std::cout << "Nfeatures :: " << nfeatures << std::endl;


	gamma = gamma;
	buffer_size = 0;

	// Create the buffer size
	// We do this by finding the next power of two!
	// With the bit shifts we also make a bit mask which will make
	// it easier to cycle through the buffer
	auto num_bits = find_next_power_of_2(nhistory + nfuture + 2);
	for(unsigned long long int i =0; i < num_bits ; i++){
		buffer_size |= 1 << i;
	}

	std::cout << "Measurement and command buffer" << std::endl;
	measurement_buffer = new Matrix(0.0, num_modes, 1, buffer_size + 1);
	command_buffer = new Matrix(0.0, num_modes, 1, buffer_size + 1);
	
	// The RLS learning vectors
	std::cout << "Recursive least squares" << std::endl;
	rls = new RecursiveLeastSquares(new_handle, nfeatures, nfuture, num_modes, gamma, P0);
	std::cout << "Phi, Xf, Wp" << std::endl;
	phi = new Matrix(0.0, nfeatures, 1, num_modes);
	xf = new Matrix(0.0, nfuture, 1, num_modes);
	wp = new Matrix(0.0, nfeatures - nfuture, 1, num_modes);
	
	// The output
	std::cout << "Command and Delta Command" << std::endl;
	command = new Matrix(0.0, 1, 1, num_modes);

	int ncontrol = 1;
	delta_command = new Matrix(0.0, ncontrol, 1, num_modes);
	
	std::cout << "H, H11, H12, invH11" << std::endl;
	H = new Matrix(0.0, nfeatures, nfeatures, nmodes);
	H11 = new Matrix(0.0, nfuture, nfuture, nmodes);
	H12 = new Matrix(0.0, nfuture, nfeatures - nfuture, nmodes);
	invH11 = new Matrix(0.0, nfuture, nfuture, nmodes);
	
	// Setup the condition matrix	
	std::cout << "lambda" << std::endl;
	lambda = new Matrix(0.0, 1, 1, num_modes);
	cpu_full_copy(lambda, new_lambda);
	
	std::cout << "condition matrix" << std::endl;
	condition_matrix = make_identity_matrix(0.0, nfuture, nmodes);
	// set_identity_matrix(Matrix* destination, float value, int size, int batch_size=1);
	// copy_to_identity_matrix(condition_matrix, lambda, nfuture, nmodes);
	
	std::cout << "invH11, controller and full controller" << std::endl;
	invH11sub = new Matrix(0.0, 1, 1, nmodes);
	controller = new Matrix(0.0, 1, nfeatures - nfuture, nmodes);
	full_controller = new Matrix(0.0, nfuture, nfeatures - nfuture, nmodes);

	newest_measurement = nullptr;

	buffer_index = 0;
}


DistributedAutoRegressiveController::~DistributedAutoRegressiveController(){
	delete measurement_buffer;
	delete command;
	delete delta_command;
	delete phi;
	delete xf;
	delete command_buffer;
	delete wp;

	delete lambda;
	delete condition_matrix;

	delete H;
	delete H11;
	delete H12;
	delete invH11;
	delete invH11sub;
	delete controller;
	delete full_controller;
	delete rls;
}


void DistributedAutoRegressiveController::set_handle(cublasHandle_t* new_handle)
{
	handle = new_handle;
	//rls.set_handle(new_handle);
	measurement_buffer->set_handle(new_handle);
	command_buffer->set_handle(new_handle);
	command->set_handle(new_handle);
	delta_command->set_handle(new_handle);
	phi->set_handle(new_handle);
	xf->set_handle(new_handle);
	lambda->set_handle(new_handle);

	H->set_handle(new_handle);
	H11->set_handle(new_handle);
	invH11->set_handle(new_handle);
	H12->set_handle(new_handle);
	controller->set_handle(new_handle);
	full_controller->set_handle(new_handle);
	wp->set_handle(new_handle);
};

void DistributedAutoRegressiveController::reset_data_buffer(){
	buffer_index = 0;
	
	measurement_buffer->set_to_zero();
	command_buffer->set_to_zero();
	command->set_to_zero();
	delta_command->set_to_zero();
	phi->set_to_zero();
	xf->set_to_zero();
	wp->set_to_zero();
}


void DistributedAutoRegressiveController::reset_controller(){
	H->set_to_zero();
	H11->set_to_zero();
	H12->set_to_zero();
	invH11->set_to_zero();
	
	invH11sub->set_to_zero();

	controller->set_to_zero();
	full_controller->set_to_zero();

	rls->reset();

}


void DistributedAutoRegressiveController::add_measurement(Matrix* new_measurement){
	
	newest_measurement = new_measurement;

	// Copy the new measurement into our data buffer
	gpu_col_copy(measurement_buffer, 0, buffer_index & buffer_size, new_measurement);
	// measurement_buffer->print(true);
	
	// Create the feature vector
	// Copy the future and past commands
	for(int i=0; i < (nfuture - 1 + nhistory); i++){
		// Copies x into y
		cublasXcopy(*handle, nmodes,
			command_buffer->gpu_data[(buffer_index-1-i) & buffer_size], 1,
			&phi->gpu_data[0][i], nfeatures);
		
	}

	// Start by copying the past measurements
	int phi_offset = nfuture-1 + nhistory;
	int data_offset = nfuture;
	for(int i=0; i <nhistory; i++){
		// Copies x into y
		cublasXcopy(*handle, nmodes,
			measurement_buffer->gpu_data[(buffer_index-data_offset-i) & buffer_size], 1,
			&phi->gpu_data[0][i + phi_offset], nfeatures);	
	}

	// Copy the future measurements
	for(int i=0; i <nfuture; i++){
		// Copies x into y
		cublasXcopy(*handle, nmodes,
			measurement_buffer->gpu_data[(buffer_index-i) & buffer_size], 1,
			&xf->gpu_data[0][i], nfuture);
			
	}
	
	// Create the past vector for the control command
	for(int i=0; i < nhistory-1; i++){
		// We take the N-1 past commands
		cublasXcopy(*handle, nmodes,
			command_buffer->gpu_data[(buffer_index-1-i) & buffer_size], 1,
			&wp->gpu_data[0][i], nfeatures - nfuture);
	}

	int offset = nhistory-1;
	for(int i=0; i < nhistory; i++){
		// We take the N past measurements
		cublasXcopy(*handle, nmodes,
			measurement_buffer->gpu_data[(buffer_index-i) & buffer_size], 1,
			&wp->gpu_data[0][i+offset], nfeatures - nfuture);
	}

}


void DistributedAutoRegressiveController::update_predictor(){

	// Estimate the future wavefront
	rls->update(phi, xf);
}


void DistributedAutoRegressiveController::update_controller(){
	/*
		All steps have been verified by comparing with python.
	*/

	// Calculate the controller (B.T.dot(A))
	// Get the prediction matrix from the RLS
	Matrix * A = rls->A;
	
	// A consists of two submatrices, As and Bs.
	// As does the prediction and Bs the system dynamics
	// Bs columns 0-(nfuture-1) and As columns (nfuture-1) to the end
	float alpha = 1.0;
	float beta = 0.0;

	// Calculate B.T.dot( A )
	cublasXgemmStridedBatched(*handle,
		CUBLAS_OP_T,
		CUBLAS_OP_N,
		H12->nrows_, H12->ncols_, A->nrows_,
		&alpha, 
		A->gpu_data[0], A->nrows_, A->size_,
		A->gpu_data[0] + A->nrows_* nfuture, A->nrows_, A->size_,
		&beta,
		H12->gpu_data[0], H12->nrows_, H12->size_,
		H12->batch_size_);
	
	// Calculate B.T.dot( B )
	cublasXgemmStridedBatched(*handle,
		CUBLAS_OP_T,
		CUBLAS_OP_N,
		H11->nrows_, H11->ncols_, A->nrows_,
		&alpha, 
		A->gpu_data[0], A->nrows_, A->size_,
		A->gpu_data[0], A->nrows_, A->size_,
		&beta,
		H11->gpu_data[0], H11->nrows_, H11->size_,
		H11->batch_size_);
	// This can change to a single MVM and a two smart memcpy's
		
	// Invert B.T.dot(B) submatrix
	H11->add(condition_matrix);
	H11->inverse(invH11);
		
	alpha = -1.0;
	cublasXgemmStridedBatched(*handle,
		CUBLAS_OP_N,
		CUBLAS_OP_N,
		controller->nrows_, controller->ncols_, invH11->ncols_,
		&alpha, 
		invH11->gpu_data[0] + nfuture - 1, invH11->nrows_, invH11->size_,
		H12->gpu_data[0], H12->nrows_, H12->size_,
		&beta,
		controller->gpu_data[0], controller->nrows_, controller->size_,
		controller->batch_size_);
	
	// invH11->dot(H12, full_controller, -1.0, 0.0, CUBLAS_OP_N, CUBLAS_OP_N);

	// std::cout << "CUDA H11 " << std::endl;
	// H11->print(true);
	// std::cout << "CUDA invH11 " << std::endl;
	// invH11->print(true);
	// std::cout << "CUDA H12 " << std::endl;
	// H12->print(true);

	// invH11->dot(H12, controller, -1.0, 0.0, CUBLAS_OP_N, CUBLAS_OP_N);
	// std::cout << "CUDA controller " << std::endl;
	// controller->print(true);

	// invH11sub->dot(H12, controller, -1.0, 0.0, CUBLAS_OP_N, CUBLAS_OP_N);
	
}


void DistributedAutoRegressiveController::set_new_regularization(float* new_lambda){
	// Setup the condition matrix	
	cpu_full_copy(lambda, new_lambda);
	copy_to_identity_matrix(condition_matrix, lambda, nfuture, nmodes);
}


__global__ void clip_array(float* x, float clip_value, int n){
	/*
		calculates:
			z = y / x
	*/
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int k = index; k < n; k += stride) {
		// z[BIDX2C(i, 0, k, element_size, 1)] = y[BIDX2C(i, 0, k, element_size, 1)] / x[k];
		x[k] = x[k] < -clip_value ? -clip_value : x[k];
		x[k] = x[k] > clip_value ? clip_value : x[k];
	}
}


Matrix* DistributedAutoRegressiveController::get_command(float clip_val, Matrix* exploration_signal){
	
	// Calculate the dot product
	controller->dot(wp, delta_command, 1.0, 0.0, CUBLAS_OP_N, CUBLAS_OP_N);
	delta_command->add(exploration_signal);
	clip_array <<<8*32, 64 >>>(delta_command->gpu_data[0], clip_val, delta_command->total_size_);
	
	// Copy the new measurement into our data buffer
	// gpu_col_copy(command_buffer, 0, buffer_index & buffer_size, delta_command);
	// gpu_col_copy(measurement_buffer, 0, buffer_index & buffer_size, new_measurement);
	
	// Integrate on the delta_command into command
	command->add(delta_command);
	// command->scale(0.98);
	// command->add(newest_measurement, -0.6);
	// Transfer the data back to the cpu
	command->to_cpu();

	buffer_index++;

	return command;
}


}