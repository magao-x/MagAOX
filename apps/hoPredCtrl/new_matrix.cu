#include "new_matrix.cuh"
#include <random>
#include <iostream>

namespace DDSPC
{

Matrix::Matrix(float initialization_value, int num_rows, int num_columns, int num_batches){

	ncols_ = num_columns;
	nrows_ = num_rows;
	batch_size_ = num_batches;
	std::cout << "Making a matrix of (" << nrows_ << " , " << ncols_ << " , " << batch_size_ << " )" << std::endl;

	size_ = num_columns * num_rows;
	total_size_ = size_ * batch_size_;

	element_size_ = sizeof(float);
	
	// Allocate the pointers
	cpu_data = new float* [batch_size_];
	
	// Create a contiguous block of memory in the host
	// With contigous memory we can easily switch between Batch and Strided cuBlas operations
	float* contigous_memory = new float[size_ * batch_size_];

	// Initialize the CPU data
	for(size_t n = 0; n < batch_size_; n++){
		cpu_data[n] = &contigous_memory[size_ * n];

		for(size_t i = 0; i < size_; i++){
			contigous_memory[size_ * n + i] = initialization_value;
		}
	}

	// Host side pointers
	gpu_data = new float* [batch_size_];
	// Allocate gpu memory and use the first host side pointer as start of
	// the contigeous memory.
	cudaError_t err = cudaMalloc((void**)&gpu_data[0], element_size_ * size_ * batch_size_);
	check_cuda_error(err);
	
	// Now set all subsequent pointers
	for(size_t n = 0; n < batch_size_; n++){
		gpu_data[n] = gpu_data[0] + size_ * n;
	}
	
	// Allocate device side pointers
	cudaError_t err1 = cudaMalloc((void**)&dev_gpu_data, batch_size_ * sizeof(*gpu_data));
	check_cuda_error(err1);

	// And copy the host side pointers to the device
	// now dev_gpu_data is device side pointers that point to the data on the gpu!
	cudaError_t err2 = cudaMemcpy(dev_gpu_data, gpu_data, batch_size_ * sizeof(*gpu_data), cudaMemcpyHostToDevice);
	check_cuda_error(err2);

	// Transfer to host side data to the gpu
	to_gpu();

	// Allocate memory for the information for batch processes
	cudaError_t err3 = cudaMalloc(&info, batch_size_ * sizeof(int));
	check_cuda_error(err3);

	// std::cout << "Print gpu buffer::" <<std::endl;
	// print(true);
}

Matrix::~Matrix(){
	
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
	
	// print(true);

	if(info != nullptr)
		cudaFree(info);
}

void Matrix::to_gpu(){
	// Copy one contineous block of memory A pointer just point to the beginning of the data
	cudaError_t err3 = cudaMemcpy(gpu_data[0], cpu_data[0], element_size_ * size_ * batch_size_, cudaMemcpyHostToDevice);
	check_cuda_error(err3);
	cudaDeviceSynchronize();
}

void Matrix::to_cpu(){
	cudaError_t err3 = cudaMemcpy(cpu_data[0], gpu_data[0], element_size_ * size_ * batch_size_, cudaMemcpyDeviceToHost);
	check_cuda_error(err3);
	cudaDeviceSynchronize();
}

void Matrix::print(bool print_gpu){
	//if(print_gpu){
	
	gpu_print_buffer <<<1, 1 >>>(gpu_data[0], batch_size_, nrows_, ncols_, 5, 5);
	cudaDeviceSynchronize();

	//}else{
	// print_batch_buffer(cpu_data, batch_size_, nrows_, ncols_);
	//}
}	

void Matrix::shift_columns_cpu(){
	float reset_value = 0.0;
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

void Matrix::divide_by_scalar(Matrix* other, Matrix* out){
	divide_scalar_gpu<<<32 * 8, 64>>>(other->gpu_data[0], gpu_data[0], out->gpu_data[0], size_, batch_size_);
}

void Matrix::set_to_zero(){
	// Set the cpu data to zero
	memset(cpu_data[0], 0, size_ * batch_size_ * sizeof(float));
	
	// And copy the zero'd data to the GPU.
	to_gpu();
}

void Matrix::add(Matrix* other, float value){
	cublasXaxpy(*handle, total_size_, value, other->gpu_data[0], gpu_data[0]);
}

void Matrix::scale(float scale_param){
	cublasXscal(*handle, total_size_, scale_param, gpu_data[0]);
}

void Matrix::subtract(Matrix* other, float value){
	add(other, value);
}


void Matrix::dot(Matrix* other, Matrix* output, float alpha, float beta, cublasOperation_t opA, cublasOperation_t opB, bool use_strided){
	int nrows_A = (opA == CUBLAS_OP_N) ? nrows_ : ncols_;
	int ncols_A = (opA == CUBLAS_OP_N) ? ncols_ : nrows_;
	int ncols_B = (opB == CUBLAS_OP_N) ? other->ncols_ : other->nrows_;
	// std::cout << nrows_A << " " << ncols_A << " " << ncols_B << std::endl;

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

void Matrix::inverse(Matrix* other){
	/*
		cuBLAS assumes column-major format with dimensions nxn.
		Other functions use row-major layout.
		So we effectively get inverse of the transpose of the input.
		so this does :: inv(A.T)
	*/
	cublasXmatinvBatched(*handle, nrows_, dev_gpu_data, other->dev_gpu_data, batch_size_, info);
}


Matrix* make_identity_matrix(float value, int size, int batch_size){
	
	Matrix* new_matrix = new Matrix(0.0, size, size, batch_size);
	
	
	for(size_t k=0; k<batch_size; k++){
		for(size_t i=0; i<size; i++){
			// set(T value, int row_index, int column_index, int batch_index=1
			new_matrix->set(value, i, i, k);
		}
	}
	
	new_matrix->to_gpu();

	return new_matrix;
}

Matrix* make_identity_matrix(float* value, int size, int batch_size){
	
	Matrix* new_matrix = new Matrix(0.0, size, size, batch_size);
	
	
	for(size_t k=0; k<batch_size; k++){
		for(size_t i=0; i<size; i++){
			// set(T value, int row_index, int column_index, int batch_index=1
			new_matrix->set(value[k], i, i, k);
		}
	}
	
	new_matrix->to_gpu();

	return new_matrix;
}

Matrix* make_identity_matrix(Matrix* value, int size, int batch_size){
	/*
		Creates a batch of identity matrices with <value> on the diagonal.
		The values for each matrix are different because value is a
	*/

	Matrix* new_matrix = new Matrix(0.0, size, size, batch_size);
	
	
	for(size_t k=0; k<batch_size; k++){
		for(size_t i=0; i<size; i++){
			// set(T value, int row_index, int column_index, int batch_index=1
			new_matrix->set(value->cpu_data[0][k], i, i, k);
		}
	}
	
	new_matrix->to_gpu();

	return new_matrix;
}


Matrix* make_identity_matrix_from_scalar(float value, int size, int batch_size){
	/*
		Creates a batch of identity matrices with <value> on the diagonal.
	*/

	Matrix* new_matrix = new Matrix(0.0, size, size, batch_size);
	
	
	for(size_t k=0; k<batch_size; k++){
		for(size_t i=0; i<size; i++){
			// set(T value, int row_index, int column_index, int batch_index=1
			new_matrix->set(value, i, i, k);
		}
	}
	
	new_matrix->to_gpu();

	return new_matrix;
}

void set_identity_matrix(Matrix* destination, float value, int size, int batch_size){
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

void copy_to_identity_matrix(Matrix* destination, Matrix* source, int size, int batch_size){
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

// This is the fast copy for column-major data-layouts
void cpu_full_copy(Matrix* destination, float * source){
	std::copy(source, source + destination->total_size_, destination->cpu_data[0]);
}

// This is the fast copy for column-major data-layouts
void cpu_full_copy(float * destination, Matrix* source){
	std::copy(source->cpu_data[0], source->cpu_data[0] + source->total_size_, destination);
}

void gpu_col_copy(Matrix* destination, int dcol_index, int dbatch_index, Matrix* source, int scol_index, int sbatch_index){
	int sindex = IDX2C(0, scol_index, source->nrows_);
	int dindex = IDX2C(0, dcol_index, destination->nrows_);
	cudaMemcpy(destination->gpu_data[dbatch_index] + dindex, source->gpu_data[sbatch_index] + sindex, destination->nrows_ * sizeof(float), cudaMemcpyDeviceToDevice);
}


Matrix* make_col_vector(float value, int size, int batch_size){
	
	Matrix* new_matrix = new Matrix(value, 1, size, batch_size);
	
	// for(size_t k=0; k<batch_size; k++){
	//	for(size_t i=0; i<size; i++){
	//		new_matrix->set(value, 0, i, k);
	//	}
	// }	
	// new_matrix->to_gpu();

	return new_matrix;
}

Matrix* make_col_vector(float* value, int size, int batch_size){
	
	Matrix* new_matrix = new Matrix(0.0, 1, size, batch_size);
	
	for(size_t k=0; k<batch_size; k++){
		for(size_t i=0; i<size; i++){
			new_matrix->set(value[i], 0, i, k);
		}
	}	
	new_matrix->to_gpu();

	return new_matrix;
}

Matrix* make_random_col_vector(float standard_deviation, int size, int batch_size){
	// Hmmm this should be defined somewhere else because it is initialized with the same seed.
	std::default_random_engine generator;
	std::normal_distribution<float> distribution(0.0, standard_deviation);

	Matrix* new_matrix = new Matrix(0.0, 1, size, batch_size);
	
	for(size_t k=0; k<batch_size; k++){
		for(size_t i=0; i<size; i++){
			new_matrix->set(distribution(generator), 0, i, k);
		}
	}	
	new_matrix->to_gpu();

	return new_matrix;
}

Matrix* make_random_matrix(float standard_deviation, int nrows, int ncols, int batch_size){
	
	// Hmmm this should be defined somewhere else because it is initialized with the same seed.
	std::default_random_engine generator;
	std::normal_distribution<float> distribution(0.0, standard_deviation);

	Matrix* new_matrix = new Matrix(0, nrows, ncols, batch_size);

	for(size_t k=0; k<batch_size; k++){
		for(size_t i=0; i<nrows; i++){
			for(size_t j=0; j<ncols; j++){
				new_matrix->set(distribution(generator), i, j, k);
			}
		}
	}
	
	new_matrix->to_gpu();

	return new_matrix;
}


Matrix* make_random_binary_matrix(float standard_deviation, int nrows, int ncols, int batch_size){
	
	// Hmmm this should be defined somewhere else because it is initialized with the same seed.
	std::default_random_engine generator;
	std::normal_distribution<float> distribution(0.0, standard_deviation);

	Matrix* new_matrix = new Matrix(0, nrows, ncols, batch_size);

	for(size_t k=0; k<batch_size; k++){
		for(size_t i=0; i<nrows; i++){
			for(size_t j=0; j<ncols; j++){
				float new_val = distribution(generator);
				if(new_val > 0){
					new_matrix->set(standard_deviation, i, j, k);
				}else{
					new_matrix->set(-standard_deviation, i, j, k);
				}
			
			}
		}
	}
	
	new_matrix->to_gpu();

	return new_matrix;
}

}