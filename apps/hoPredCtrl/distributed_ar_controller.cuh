#ifndef PCDDSC_CUH
#define PCDDSC_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"     // if you need CUBLAS v2, include before magma.h

#include "utility_functions.cuh"
#include "matrix.cuh"
#include "recursive_least_squares.cuh"

namespace DDSPC
{

template <class T>
class DistributedAutoRegressiveController{

	public:
		RecursiveLeastSquares<T> *rls;
		
		Matrix<T> *measurement_buffer;
		Matrix<T> *command_buffer;
		Matrix<T> *phi;
		Matrix<T> *xf;

		Matrix<T> *command;
		Matrix<T> *delta_command;
		Matrix<T> *H, *H11, *H12, *invH11, *invH11sub, *controller, *full_controller;
		Matrix<T> *condition_matrix;
		Matrix<T> *wp;

		cublasHandle_t* handle;

		int nhistory, nfuture, nmodes, nfeatures;
		unsigned long long int buffer_size;
		unsigned long long int buffer_index;

		Matrix<T>* lambda;
		T gamma;

		DistributedAutoRegressiveController(cublasHandle_t* new_handle, int num_history, int num_future, int num_modes, T gamma, T* new_lambda, T P0);
		~DistributedAutoRegressiveController();

		void set_handle(cublasHandle_t* new_handle){
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

		// void add_measurement(T* new_measurement);
		void set_new_regularization(T* new_lambda);
		void reset_data_buffer();
		void reset_controller();

		void add_measurement(Matrix<T>* new_measurement);
		void update_predictor();
		void update_controller();
		Matrix<T>* get_command(T clip_val, Matrix<T>* exploration_signal);

};

template <class T>
DistributedAutoRegressiveController<T>::DistributedAutoRegressiveController(cublasHandle_t* new_handle, int num_history, int num_future, int num_modes, T gamma, T* new_lambda, T P0){
	nhistory = num_history;
	nfuture = num_future;
	nmodes = num_modes;
	
	// Use all future DM commands except the most recent because that one will only have an effect in later measurements
	nfeatures = nfuture - 1 + 2 * nhistory;
	
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

	measurement_buffer = new Matrix<T>(0.0, num_modes, 1, buffer_size + 1);
	command_buffer = new Matrix<T>(0.0, num_modes, 1, buffer_size + 1);
	
	// The RLS learning vectors
	rls = new RecursiveLeastSquares<T>(new_handle, nfeatures, nfuture, num_modes, gamma, P0);
	phi = new Matrix<T>(0.0, nfeatures, 1, num_modes);
	xf = new Matrix<T>(0.0, nfuture, 1, num_modes);
	wp = new Matrix<T>(0.0, nfeatures - nfuture, 1, num_modes);
	
	// The output
	command = new Matrix<T>(0.0, 1, 1, num_modes);

	int ncontrol = 1;
	delta_command = new Matrix<T>(0.0, ncontrol, 1, num_modes);
	
	H = new Matrix<T>(0.0, nfeatures, nfeatures, nmodes);
	H11 = new Matrix<T>(0.0, nfuture, nfuture, nmodes);
	H12 = new Matrix<T>(0.0, nfuture, nfeatures - nfuture, nmodes);
	invH11 = new Matrix<T>(0.0, nfuture, nfuture, nmodes);
	
	// Setup the condition matrix	
	lambda = new Matrix<T>(0.0, 1, 1, num_modes);
	cpu_full_copy(lambda, new_lambda);
	condition_matrix = make_identity_matrix<T>(0.0, nfuture, nmodes);
	copy_to_identity_matrix(condition_matrix, lambda, nfuture, nmodes);

	invH11sub = new Matrix<T>(0.0, 1, 1, nmodes);
	controller = new Matrix<T>(0.0, 1, nfeatures - nfuture, nmodes);
	full_controller = new Matrix<T>(0.0, nfuture, nfeatures - nfuture, nmodes);

	buffer_index = 0;
}

template <class T>
DistributedAutoRegressiveController<T>::~DistributedAutoRegressiveController(){
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

template <class T>
void DistributedAutoRegressiveController<T>::reset_data_buffer(){
	buffer_index = 0;
	
	measurement_buffer->set_to_zero();
	command_buffer->set_to_zero();
	command->set_to_zero();
	delta_command->set_to_zero();
	phi->set_to_zero();
	xf->set_to_zero();
	wp->set_to_zero();
}

template <class T>
void DistributedAutoRegressiveController<T>::reset_controller(){
	H->set_to_zero();
	H11->set_to_zero();
	H12->set_to_zero();
	invH11->set_to_zero();
	
	invH11sub->set_to_zero();

	controller->set_to_zero();
	full_controller->set_to_zero();

	rls->reset();

}

template <class T>
void DistributedAutoRegressiveController<T>::add_measurement(Matrix<T>* new_measurement){
	
	// Copy the new measurement into our data buffer
	gpu_col_copy(measurement_buffer, 0, buffer_index & buffer_size, new_measurement);
	// measurement_buffer->print(true);

	// Create the feature vector
	// Copy the future and past commands
	for(int i=0; i <(nfuture - 1 + nhistory); i++){
		// Copies x into y
		cublasXcopy(*handle, nmodes,
			command_buffer->gpu_data[(buffer_index-1-i) & buffer_size], 1,
			&phi->gpu_data[0][i], nfeatures);
		
	}
	
	// Start by copying the past measurements
	/*
		This is verified with python.
	*/
	int phi_offset = nfuture-1 + nhistory;
	int data_offset = nfuture;
	for(int i=0; i <nhistory; i++){
		// Copies x into y
		cublasXcopy(*handle, nmodes,
			measurement_buffer->gpu_data[(buffer_index-data_offset-i) & buffer_size], 1,
			&phi->gpu_data[0][i + phi_offset], nfeatures);	
	}

	// Copy the future measurements
	/*
		This is now validated by comparing with python implementation.
	*/
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

template <class T>
void DistributedAutoRegressiveController<T>::update_predictor(){

	// Estimate the future wavefront
	rls->update(phi, xf);
}

template <class T>
void DistributedAutoRegressiveController<T>::update_controller(){
	/*
		All steps have been verified by comparing with python.
	*/

	// Calculate the controller (B.T.dot(A))
	// Get the prediction matrix from the RLS
	Matrix<T> * A = rls->A;
	
	// A consists of two submatrices, As and Bs.
	// As does the prediction and Bs the system dynamics
	// Bs columns 0-(nfuture-1) and As columns (nfuture-1) to the end
	T alpha = 1.0;
	T beta = 0.0;

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

template <class T>
void DistributedAutoRegressiveController<T>::set_new_regularization(T* new_lambda){
	// Setup the condition matrix	
	cpu_full_copy(lambda, new_lambda);
	copy_to_identity_matrix(condition_matrix, lambda, nfuture, nmodes);
}

template <class T>
__global__ void clip_array(T* x, T clip_value, int n){
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

template <class T>
Matrix<T>* DistributedAutoRegressiveController<T>::get_command(T clip_val, Matrix<T>* exploration_signal){
	// Calculate the dot product
	controller->dot(wp, delta_command, 1.0, 0.0, CUBLAS_OP_N, CUBLAS_OP_N);
	delta_command->add(exploration_signal);
	clip_array<T> <<<8*32, 64 >>>(delta_command->gpu_data[0], clip_val, delta_command->total_size_);
	
	// Copy the new measurement into our data buffer
	gpu_col_copy(command_buffer, 0, buffer_index & buffer_size, delta_command);
	// gpu_col_copy(measurement_buffer, 0, buffer_index & buffer_size, new_measurement);

	// Integrate on the delta_command into command
	command->add(delta_command);

	// Transfer the data back to the cpu
	command->to_cpu();

	buffer_index++;

	return command;
}

}

#endif