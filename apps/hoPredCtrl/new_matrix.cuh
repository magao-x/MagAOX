#ifndef PCMATRIX_CUH
#define PCMATRIX_CUH

#include <iostream>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "utils.cuh"

namespace DDSPC
{


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
			
		float** cpu_data; // Host side pointers to host data
		float** gpu_data; // Host side pointers to device data
		float** dev_gpu_data; // Device side pointers to device data
		int *info;
	
		Matrix(float initialization_value, int num_rows, int num_columns, int num_batches=1);
		~Matrix();

		void set_to_zero();
		void divide_by_scalar(Matrix* scalar, Matrix* out=nullptr);
		
		void transpose();
		/*
		// Data manipulations functions
		// I/O functions
		void print(bool print_gpu=false);
		
		*/

		void shift_columns_cpu();

		void dot(Matrix* other, Matrix* output, float alpha=1.0, float beta=0.0, cublasOperation_t opA=CUBLAS_OP_N, cublasOperation_t opB=CUBLAS_OP_N,  bool use_strided=true);
		void inverse(Matrix* output);
		void add(Matrix* other, float value=1);
		void subtract(Matrix* other, float value=-1);
		void scale(float scale_param);
		
		void print(bool print_gpu);
		
		long int num_elements(){
			return ncols_ * nrows_;
		}

		void set_handle(cublasHandle_t* new_handle){
			handle = new_handle;
		};
		
		void set(float value, int row_index, int column_index, int batch_index=1){
			cpu_data[0][BIDX2C(row_index, column_index, batch_index, nrows_, ncols_)] = value;
		}

		float get(int row_index, int column_index, int batch_index=1){
			return cpu_data[0][BIDX2C(row_index, column_index, batch_index, nrows_, ncols_)];
		}

		void print_shape(){
			std::cout << "(" << nrows_ << "," << ncols_ << "," << batch_size_ << ")\n";
		}

		float* get_data_ptr(){
			return cpu_data[0];
		}
		
		// Data transfer commands
		void to_gpu();
		void to_cpu();

		//
		void to_file(std::string filename);
		void from_file(std::string filename);

};

Matrix* make_identity_matrix(float value, int size, int batch_size=1);
Matrix* make_identity_matrix(float* value, int size, int batch_size=1);
Matrix* make_identity_matrix(Matrix* value, int size, int batch_size=1);

Matrix* make_identity_matrix_from_scalar(float value, int size, int batch_size=1);
void set_identity_matrix(Matrix* destination, float value, int size, int batch_size=1);

void copy_to_identity_matrix(Matrix* destination, Matrix* source, int size, int batch_size=1);

void cpu_full_copy(Matrix* destination, float * source);
void cpu_full_copy(float * destination, Matrix* source);
void gpu_col_copy(Matrix* destination, int dcol_index, int dbatch_index, Matrix* source, int scol_index=0, int sbatch_index=0);

Matrix* make_col_vector(float value, int size, int batch_size=1);
Matrix* make_col_vector(float* value, int size, int batch_size=1);
Matrix* make_random_col_vector(float standard_deviation, int size, int batch_size=1);
Matrix* make_random_matrix(float standard_deviation, int nrows, int ncols, int batch_size=1);
Matrix* make_random_binary_matrix(float standard_deviation, int nrows, int ncols, int batch_size=1);
}

#endif