#ifndef PCMATRIX_CUH
#define PCMATRIX_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"
#include <stdio.h>
#include <iostream>
#include <random>
#include <algorithm>

#include "utility_functions.cuh"

// #include <pybind11/pybind11.h>
// #include <pybind11/numpy.h>
// namespace py = pybind11;

namespace DDSPC
{

template <class T>
class Matrix{
	private:
		cublasHandle_t* handle;

	public:	
		int size_;
		size_t element_size_;
		int total_size_;

		int batch_size_;
		int nrows_;
		int ncols_;
			
		T** cpu_data; // Host side pointers to host data
		T** gpu_data; // Host side pointers to device data
		T** dev_gpu_data; // Device side pointers to device data
		int *info;
	
		Matrix(T initialization_value, int num_rows, int num_columns, int num_batches=1);
		~Matrix();

		void set_to_zero();

		// Data manipulations functions
		long int num_elements(){
			return ncols_ * nrows_;
		}
		void shift_columns_cpu();

		void set_handle(cublasHandle_t* new_handle){
			handle = new_handle;
		};
		
		void set(T value, int row_index, int column_index, int batch_index=1){
			cpu_data[0][BIDX2C(row_index, column_index, batch_index, nrows_, ncols_)] = value;
			// cpu_data[batch_index][IDX2C(row_index, column_index, nrows_)] = value;
		}

		T get(int row_index, int column_index, int batch_index=1){
			return cpu_data[0][BIDX2C(row_index, column_index, batch_index, nrows_, ncols_)];
			// return cpu_data[batch_index][IDX2C(row_index, column_index, nrows_)];
		}

		void dot(Matrix* other, Matrix* output, T alpha=1.0, T beta=0.0, cublasOperation_t opA=CUBLAS_OP_N, cublasOperation_t opB=CUBLAS_OP_N,  bool use_strided=true);
		void inverse(Matrix* output);

		void add(Matrix* other, float value=1);
		void subtract(Matrix* other, float value=-1){
			add(other, value);
		};

		void divide_by_scalar(Matrix* scalar, Matrix* out=nullptr);

		// Data transfer commands
		void to_gpu(){
			// Copy one contineous block of memory
			// A pointer just point to the beginnen of the data
			cudaMemcpy(gpu_data[0], cpu_data[0], element_size_ * size_ * batch_size_, cudaMemcpyHostToDevice);
			cudaDeviceSynchronize();
		};

		void to_cpu(){
			cudaMemcpy(cpu_data[0], gpu_data[0], element_size_ * size_ * batch_size_, cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();
		};

		// I/O functions
		void print_shape(){
			std::cout << "(" << nrows_ << "," << ncols_ << "," << batch_size_ << ")";
		}

		void print(bool print_gpu=false){
			if(print_gpu){
				gpu_print_buffer <<<1, 1 >>>(gpu_data[0], batch_size_, nrows_, ncols_);
				cudaDeviceSynchronize();
			}else{
				print_batch_buffer(cpu_data, batch_size_, nrows_, ncols_);
			}
		}	
		
		T* get_data_ptr(){
			return cpu_data[0];
		}
		// py::array_t<T> to_python();

};


template <class T>
Matrix<T>::Matrix(T initialization_value, int num_rows, int num_columns, int num_batches){

	ncols_ = num_columns;
	nrows_ = num_rows;
	batch_size_ = num_batches;
	
	size_ = num_columns * num_rows;
	total_size_ = size_ * batch_size_;

	element_size_ = sizeof(T);
	
	// Allocate the pointers
	cpu_data = new T*[batch_size_];
	
	// Create a contiguous block of memory in the host
	// With contigous memory we can easily switch between Batch and Strided cuBlas operations
	T* contigous_memory = new T[size_ * batch_size_];

	// Initialize the CPU data
	for(size_t n = 0; n < batch_size_; n++){
		cpu_data[n] = &contigous_memory[size_ * n];

		for(size_t i = 0; i < size_; i++){
			contigous_memory[size_ * n + i] = initialization_value;
		}
	}

	// Host side pointers
	gpu_data = new T*[batch_size_];
	// Allocate gpu memory and use the first host side pointer as start of
	// the contigeous memory.
	cudaMalloc((void**)&gpu_data[0], element_size_ * size_ * batch_size_);
	
	// Now set all subsequent pointers
	for(size_t n = 0; n < batch_size_; n++){
		gpu_data[n] = gpu_data[0] + size_ * n;
	}
	
	// Allocate device side pointers
	cudaError_t err1 = cudaMalloc((void**)&dev_gpu_data, batch_size_ * sizeof(*gpu_data));

	// And copy the host side pointers to the device
	// now dev_gpu_data is device side pointers that point to the data on the gpu!
	cudaMemcpy(dev_gpu_data, gpu_data, batch_size_ * sizeof(*gpu_data), cudaMemcpyHostToDevice);

	// Transfer to host side data to the gpu
	to_gpu();

	// Allocate memory for the information for batch processes
	cudaMalloc(&info, batch_size_ * sizeof(int));
}

template <class T>
Matrix<T>::~Matrix(){
	
	if(cpu_data != nullptr){
		// Free host side data
		if(cpu_data[0] != nullptr)
			delete cpu_data[0];

		// Free host side pointers to host side data
		delete cpu_data;
	}

	if(gpu_data != nullptr){
		// Free device side data
		if(gpu_data[0] != nullptr)
			cudaFree(gpu_data[0]);

		// Free host side pointers to device data
		delete gpu_data;
	}
	
	// Free device side pointers to device data
	if(dev_gpu_data != nullptr )
		cudaFree(dev_gpu_data);
	
	if(info != nullptr)
		cudaFree(info);
}

template <class T>
void Matrix<T>::dot(Matrix* other, Matrix* output, T alpha, T beta, cublasOperation_t opA, cublasOperation_t opB, bool use_strided){
	int nrows_A = (opA == CUBLAS_OP_N) ? nrows_ : ncols_;
	int ncols_A = (opA == CUBLAS_OP_N) ? ncols_ : nrows_;
	
	// int nrows_B = (opB == CUBLAS_OP_N) ? other->nrows_ : other->ncols_;
	int ncols_B = (opB == CUBLAS_OP_N) ? other->ncols_ : other->nrows_;

	if( use_strided ){
		cublasXgemmStridedBatched(*handle,
			opA, 
			opB,
			nrows_A, ncols_B, ncols_A,
			&alpha,
			gpu_data[0], nrows_,
			size_,
			other->gpu_data[0], other->nrows_,
			other->size_,
			&beta,
			output->gpu_data[0], output->nrows_, 
			output->size_, 
			batch_size_);
	}else{
		cublasXgemmBatched(*handle,
			opA,
			opB,
			nrows_A, ncols_B, ncols_A,
			&alpha, 
			dev_gpu_data, nrows_,
			other->dev_gpu_data, other->nrows_,
			&beta,
			output->dev_gpu_data, output->nrows_,
			batch_size_);

	}
	  
}

template <class T>
void Matrix<T>::set_to_zero(){
	/*
	*/
	
	// Set the cpu data to zero
	memset(cpu_data[0], 0, size_ * batch_size_ * sizeof(T));
	
	// And copy the zero'd data to the GPU.
	// This can probably be replaced with a cudamemset if a speedup is required.
	to_gpu();

}

template <class T>
void Matrix<T>::inverse(Matrix* other){
	/*
		cuBLAS assumes column-major format with dimensions nxn.
		Other functions use row-major layout.
		So we effectively get inverse of the transpose of the input.
		so this does :: inv(A.T)
	*/
	cublasXmatinvBatched(*handle, nrows_, dev_gpu_data, other->dev_gpu_data, batch_size_, info);
}

template <class T>
void Matrix<T>::add(Matrix* other, float value){
	//add_gpu<<<32 * 8, 64>>>(other->gpu_data, gpu_data, size);
	cublasXaxpy(*handle, total_size_, value, other->gpu_data[0], gpu_data[0]);
}

template <class T>
void Matrix<T>::divide_by_scalar(Matrix* other, Matrix* out){
	divide_scalar_gpu<T><<<32 * 8, 64>>>(other->gpu_data[0], gpu_data[0], out->gpu_data[0], size_, batch_size_);
}

template <class T>
void Matrix<T>::shift_columns_cpu(){
	T reset_value = 0.0;
	for (int k=0; k < batch_size_; k++) {
		for(int i=0; i<nrows_; i++){
			for(int j=(ncols_-1); j>0; j--){
				// We walk in reverse through the buffer to make sure we copy the data correctly
				// If we walk forward we overwrite everything with the first element

				int current_index = BIDX2C(i, j, k, nrows_, ncols_);
				int previous_index = BIDX2C(i, j-1, k, nrows_, ncols_);
				cpu_data[0][current_index] = cpu_data[0][previous_index];
			}
			
			int current_index = BIDX2C(i, 0, k, nrows_, ncols_);
			cpu_data[0][current_index] = reset_value;
		}	
	}
}


template <class T>
Matrix<T>* make_identity_matrix(T value, int size, int batch_size=1){
	
	Matrix<T>* new_matrix = new Matrix<T>(0.0, size, size, batch_size);
	
	
	for(size_t k=0; k<batch_size; k++){
		for(size_t i=0; i<size; i++){
			// set(T value, int row_index, int column_index, int batch_index=1
			new_matrix->set(value, i, i, k);
		}
	}
	
	new_matrix->to_gpu();

	return new_matrix;
}

template <class T>
Matrix<T>* make_identity_matrix(T* value, int size, int batch_size=1){
	
	Matrix<T>* new_matrix = new Matrix<T>(0.0, size, size, batch_size);
	
	
	for(size_t k=0; k<batch_size; k++){
		for(size_t i=0; i<size; i++){
			// set(T value, int row_index, int column_index, int batch_index=1
			new_matrix->set(value[k], i, i, k);
		}
	}
	
	new_matrix->to_gpu();

	return new_matrix;
}


template <class T>
Matrix<T>* make_identity_matrix(Matrix<T>* value, int size, int batch_size=1){
	/*
		Creates a batch of identity matrices with <value> on the diagonal.
		The values for each matrix are different because value is a
	*/

	Matrix<T>* new_matrix = new Matrix<T>(0.0, size, size, batch_size);
	
	
	for(size_t k=0; k<batch_size; k++){
		for(size_t i=0; i<size; i++){
			// set(T value, int row_index, int column_index, int batch_index=1
			new_matrix->set(value->cpu_data[0][k], i, i, k);
		}
	}
	
	new_matrix->to_gpu();

	return new_matrix;
}

template <class T>
Matrix<T>* make_identity_matrix_from_scalar(T value, int size, int batch_size=1){
	/*
		Creates a batch of identity matrices with <value> on the diagonal.
	*/

	Matrix<T>* new_matrix = new Matrix<T>(0.0, size, size, batch_size);
	
	
	for(size_t k=0; k<batch_size; k++){
		for(size_t i=0; i<size; i++){
			// set(T value, int row_index, int column_index, int batch_index=1
			new_matrix->set(value, i, i, k);
		}
	}
	
	new_matrix->to_gpu();

	return new_matrix;
}

template <class T>
void copy_to_identity_matrix(Matrix<T>* destination, Matrix<T>* source, int size, int batch_size=1){
	/*
		This functions creates a batch of identity matrices from the vector <source>.
		Each element of <source> is the value of a new diagonal matrix.
	*/

	for(size_t k=0; k<batch_size; k++){
		for(size_t i=0; i<size; i++){
			// set(T value, int row_index, int column_index, int batch_index=1
			destination->set(source->cpu_data[0][k], i, i, k);
		}
	}
	destination->to_gpu();
}

template <class T>
void set_identity_matrix(Matrix<T>* destination, T value, int size, int batch_size=1){
	/*
		This functions set a batch of identity matrices with the same value on the diagonal.
	*/

	for(size_t k=0; k<batch_size; k++){
		for(size_t i=0; i<size; i++){
			// set(T value, int row_index, int column_index, int batch_index=1
			destination->set(value, i, i, k);
		}
	}
	destination->to_gpu();
}

template <class T>
Matrix<T>* make_row_vector(T value, int size, int batch_size=1){
	
	Matrix<T>* new_matrix = new Matrix<T>(0.0, size, 1, batch_size);
	
	for(size_t k=0; k<batch_size; k++){
		for(size_t i=0; i<size; i++){
			new_matrix->set(value, i, 0, k);
		}
	}	
	new_matrix->to_gpu();

	return new_matrix;
}


template <class T>
Matrix<T>* make_random_row_vector(T standard_deviation, int size, int batch_size=1){
	// Hmmm this should be defined somewhere else because it is initialized with the same seed.
	std::default_random_engine generator;
	std::normal_distribution<T> distribution(0.0, standard_deviation);

	Matrix<T>* new_matrix = new Matrix<T>(0.0, size, 1, batch_size);
	
	for(size_t k=0; k<batch_size; k++){
		for(size_t i=0; i<size; i++){
			new_matrix->set(distribution(generator), i, 0, k);
		}
	}	
	new_matrix->to_gpu();

	return new_matrix;
}

template <class T>
Matrix<T>* make_col_vector(T value, int size, int batch_size=1){
	
	Matrix<T>* new_matrix = new Matrix<T>(0.0, 1, size, batch_size);
	
	for(size_t k=0; k<batch_size; k++){
		for(size_t i=0; i<size; i++){
			new_matrix->set(value, 0, i, k);
		}
	}	
	new_matrix->to_gpu();

	return new_matrix;
}

template <class T>
Matrix<T>* make_random_col_vector(T standard_deviation, int size, int batch_size=1){
	// Hmmm this should be defined somewhere else because it is initialized with the same seed.
	std::default_random_engine generator;
	std::normal_distribution<T> distribution(0.0, standard_deviation);

	Matrix<T>* new_matrix = new Matrix<T>(0.0, 1, size, batch_size);
	
	for(size_t k=0; k<batch_size; k++){
		for(size_t i=0; i<size; i++){
			new_matrix->set(distribution(generator), 0, i, k);
		}
	}	
	new_matrix->to_gpu();

	return new_matrix;
}

template <class T>
Matrix<T>* make_random_matrix(T standard_deviation, int nrows, int ncols, int batch_size=1){
	
	// Hmmm this should be defined somewhere else because it is initialized with the same seed.
	std::default_random_engine generator;
	std::normal_distribution<T> distribution(0.0, standard_deviation);

	Matrix<T>* new_matrix = new Matrix<T>(0, nrows, ncols, batch_size);

	for(size_t k=0; k<batch_size; k++){
		for(size_t i=0; i<nrows; i++){
			for(size_t j=0; j<ncols; j++){
				new_matrix->set(distribution(generator), i, i, k);
			}
		}
	}
	
	new_matrix->to_gpu();

	return new_matrix;
}

/*
template <class T>
py::array_t<T> Matrix<T>::to_python(){
	// Move data to the cpu
	to_cpu();

	auto result = py::array(py::buffer_info(
		cpu_data[0], // Pointer to data (nullptr -> ask NumPy to allocate!)
		element_size_,    // Size of one item
		py::format_descriptor<T>::value, // Buffer format
		1,          // How many dimensions?
		{ total_size_}, 	// Number of elements for each dimension
		{ element_size_ }  // Strides for each dimension
	));

	return result;
};
*/

template <class T>
void cpu_matrix_copy(Matrix<T>* destination, Matrix<T>* source){
	std::copy(source->cpu_data[0], source->cpu_data[0] + source->total_size_, destination->cpu_data[0]);
}

// This is the fast copy for column-major data-layouts
template <class T>
void cpu_col_copy(Matrix<T>* destination, int dcol_index, int dbatch_index, Matrix<T>* source, int scol_index=0, int sbatch_index=0){
	int sindex = IDX2C(0, scol_index, source->nrows_);
	int dindex = IDX2C(0, dcol_index, destination->nrows_);

	std::copy(source->cpu_data[sbatch_index] + sindex, source->cpu_data[sbatch_index] + sindex + source->nrows_, destination->cpu_data[dbatch_index] + dindex);
}

// This is the fast copy for column-major data-layouts
template <class T>
void cpu_col_copy(Matrix<T>* destination, int dcol_index, int dbatch_index, T * source){
	int dindex = IDX2C(0, dcol_index, destination->nrows_);
	std::copy(source, source + destination->nrows_, destination->cpu_data[dbatch_index] + dindex);
}

// This is the fast copy for column-major data-layouts
template <class T>
void cpu_full_copy(Matrix<T>* destination, T * source){
	std::copy(source, source + destination->total_size_, destination->cpu_data[0]);
}

template <class T>
void gpu_col_copy(Matrix<T>* destination, int dcol_index, int dbatch_index, Matrix<T>* source, int scol_index=0, int sbatch_index=0){
	int sindex = IDX2C(0, scol_index, source->nrows_);
	int dindex = IDX2C(0, dcol_index, destination->nrows_);
	cudaMemcpy(destination->gpu_data[dbatch_index] + dindex, source->gpu_data[sbatch_index] + sindex, destination->nrows_ * sizeof(T), cudaMemcpyDeviceToDevice);
	
}

/*
template <class T>
void cpu_row_copy(Matrix<T>* destination, Matrix<T>* source, int row_index){
	std::copy(source->cpu_data[0], source->cpu_data[0] + source->total_size_, destination->cpu_data[0]);
}
*/

// This is also a fast copy because each individual batch has a column-major layout
// and has contigeous memory!
template <class T>
void cpu_batch_copy(Matrix<T>* destination, Matrix<T>* source, int batch_index){
	std::copy(source->cpu_data[batch_index], source->cpu_data[batch_index] + source->size_, destination->cpu_data[batch_index]);
}

template <class T>
void gpu_batch_copy(Matrix<T>* destination, Matrix<T>* source, int batch_index){
	cudaMemcpy(destination->cpu_data[batch_index], source->gpu_data[batch_index], source->size_ * sizeof(T), cudaMemcpyDeviceToDevice);
}

typedef Matrix<float> sMatrix;
typedef Matrix<double> dMatrix;

}

#endif