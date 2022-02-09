#ifndef PCUTIL_CUH
#define PCUTIL_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"
#include <stdio.h>
#include <iostream>

namespace DDSPC
{

/*
	The following two defines are used to get the correct index for StridedBatched CUDA functions.
*/
// Transform the 2D index to the correct 1D array index for column-major layout.
// i is row, j is column, ld is the number of rows
#define IDX2C(i, j, nrow) (((j) * (nrow)) + (i))
#define BIDX2C(i, j, k, nrow, ncol) (((j) * (nrow)) + (i) + ((k) * (ncol) * (nrow)))

/*
	Shift bits to the right until the number becomes 0.
	This will count how many right shifts we have done.
	And how many bits are used.
	 
*/
uint find_next_power_of_2(int sample){
    uint num_bits = 0;
    
    do{
        sample >>= 1;
        ++num_bits;
    } while(sample);
    
    return num_bits;
};

/*
	Change this function so that batch count is the last parameter.
	Do not forget to change this in all subsequent code!
*/
template <class T_ELEM>
void print_batch_buffer(T_ELEM* data, int batch_count, int nrow, int ncol) {
	for (int k = 0; k < batch_count; k++) {

		for (int i = 0; i < nrow; i++) {
			std::cout << "[";

			for (int j = 0; j < ncol; j++) {
				int index = IDX2C(i, j, nrow);
				std::cout << data[k][index];

				//int index = BIDX2C(i, j, k, nrow, ncol);
				//std::cout << data[k][index];
				if (j != (ncol - 1))
					std::cout << ", ";
			}
			std::cout << "]" << std::endl;
		}
		std::cout << std::endl;
	}
};

/*
cublasOperation_t transa,
                                  cublasOperation_t transb,
                                  int m, int n, int k,
                                  const float           *alpha,
                                  const float           *A, int lda,
                                  long long int          strideA,
                                  const float           *B, int ldb,
                                  long long int          strideB,
                                  const float           *beta,
                                  float                 *C, int ldc,
                                  long long int          strideC,
								  int batchCount)
								  
*/
static inline cublasStatus_t cublasXgemmStridedBatched(cublasHandle_t handle,
	cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
	float *alpha, const float *A, int lda, long long int strideA,
	const float *B, int ldb, long long int strideB,
	float *beta,
	float *C, int ldc, long long int strideC,
	int batchCount)
{
	return cublasSgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
}

static inline cublasStatus_t cublasXgemmStridedBatched(cublasHandle_t handle,
	cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
	double *alpha, const double *A, int lda, long long int strideA,
	const double *B, int ldb, long long int strideB,
	double *beta,
	double *C, int ldc, long long int strideC,
	int batchCount)
{
	return cublasDgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
}

/*
	cublasHandle_t handle,
	cublasOperation_t transa,
	cublasOperation_t transb,
	int m, int n, int k,
	const float           *alpha,
	const float           *Aarray[], int lda,
	const float           *Barray[], int ldb,
	const float           *beta,
	float           *Carray[], int ldc,
	int batchCount
*/
static inline cublasStatus_t cublasXgemmBatched(cublasHandle_t handle,
	cublasOperation_t transa,
	cublasOperation_t transb,
	int m, int n, int k,
	float *alpha,
	float ** Aarray, int lda,
	float ** Barray, int ldb,
	float *beta,
	float ** Carray, int ldc,
	int batchCount)
{
return cublasSgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
}

/*
	cublasHandle_t handle,
	cublasOperation_t transa,
	cublasOperation_t transb,
	int m, int n, int k,
	const double          *alpha,
	const double          *Aarray[], int lda,
	const double          *Barray[], int ldb,
	const double          *beta,
	double          *Carray[], int ldc,
	int batchCount
*/
static inline cublasStatus_t cublasXgemmBatched(cublasHandle_t handle,
	cublasOperation_t transa,
	cublasOperation_t transb,
	int m, int n, int k,
	double *alpha,
	double ** Aarray, int lda,
	double ** Barray, int ldb,
	double *beta,
	double ** Carray, int ldc,
	int batchCount)
{
return cublasDgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
}

/*
cublasHandle_t handle,
int n,
const double *A[],
int lda,
double *Ainv[],
int lda_inv,
int *info,
int batchSize
*/
static inline cublasStatus_t cublasXmatinvBatched(cublasHandle_t handle,
	int n,
	double ** A,
	double ** Ainv,
	int batchCount,
	int* info)
{
	return cublasDmatinvBatched(handle, n, A, n, Ainv, n, info , batchCount);
}

static inline cublasStatus_t cublasXmatinvBatched(cublasHandle_t handle,
	int n,
	float ** A,
	float ** Ainv,
	int batchCount,
	int* info)
{
	return cublasSmatinvBatched(handle, n, A, n, Ainv, n, info , batchCount);
}

/*
*/
static inline cublasStatus_t cublasXaxpy(cublasHandle_t handle, int n, float alpha, float *x, float *y)
{
	return cublasSaxpy(handle, n, &alpha, x, 1, y, 1);
}
// (cublasHandle_t, int, float *, float **, float **)
static inline cublasStatus_t cublasXaxpy(cublasHandle_t handle, int n, double alpha, double *x, double *y)
{
	return cublasDaxpy(handle, n, &alpha, x, 1, y, 1);
}


static inline cublasStatus_t cublasXcopy(cublasHandle_t handle, int n, float *x, int incx, float *y, int incy)
{
	return cublasScopy(handle, n, x, incx, y, incy);
}

static inline cublasStatus_t cublasXcopy(cublasHandle_t handle, int n, double *x, int incx, double *y, int incy)
{
	return cublasDcopy(handle, n, x, incx, y, incy);
}

template <class T>
__global__ void divide_scalar_gpu(T* x, T* y, T* z, int element_size, int batch_size){
	/*
		calculates:
			z = y / x
	*/
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int k = index; k < batch_size; k += stride) {
		for(int i=0; i < element_size; i++){
			z[BIDX2C(i, 0, k, element_size, 1)] = y[BIDX2C(i, 0, k, element_size, 1)] / x[k]; 
		}
		
		
	}
}

//__global__ void create_submatrix(float* H, float* Hsub, float rcond, int nsub, int nrow, int ncol, int num_modes, int orow, int ocol);

template <class T>
__global__ void create_submatrix(T* H, T* Hsub, T rcond, int nrow, int ncol, int num_modes, int nsubr, int nsubc, int orow, int ocol) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	int ind = 0;
	int sub_ind = 0;

	for (int k = index; k < num_modes; k += stride) {
		for (int i = 0; i < nsubr; i++) {
			for (int j = 0; j < nsubc; j++) {
				ind = BIDX2C(i + orow, j + ocol, k, nrow, ncol);
				sub_ind = BIDX2C(i, j, k, nsubr, nsubc);
				Hsub[sub_ind] = H[ind];
				
				if (i == j) {
					Hsub[sub_ind] += rcond;
				}

			}
		}
	}

}

/*
Some helper functions
*/
__global__ void gpu_print_buffer(float* data, int batch_count, int nrow, int ncol) {
	// int index = blockIdx.x * blockDim.x + threadIdx.x;
	// int stride = blockDim.x * gridDim.x;

	// The index steps through the number of modes
	//printf("[");
	for (int k = 0; k < batch_count; k++) {

		for (int i = 0; i < nrow; i++) {
			printf("[");

			for (int j = 0; j < ncol; j++) {

				int index = BIDX2C(i, j, k, nrow, ncol);
				printf("%f", data[index]);
				// int index = IDX2C(i, j, nrow);
				// printf("%f", data[k][index]);

				if (j != (ncol - 1))
					printf(", ");
			}
			printf("]\n");
		}
	}
	//printf("]\n");
};

__global__ void gpu_print_buffer(double* data, int batch_count, int nrow, int ncol) {
	// int index = blockIdx.x * blockDim.x + threadIdx.x;
	// int stride = blockDim.x * gridDim.x;

	// The index steps through the number of modes
	//printf("[");
	for (int k = 0; k < batch_count; k++) {

		for (int i = 0; i < nrow; i++) {
			printf("[");

			for (int j = 0; j < ncol; j++) {
				int index = BIDX2C(i, j, k, nrow, ncol);
				printf("%f", data[index]);
				
				// int index = IDX2C(i, j, nrow);
				// printf("%f", data[k][index]);

				if (j != (ncol - 1))
					printf(", ");
			}
			printf("]\n");
		}
	}
	//printf("]\n");
};

/*
	This is now obsolete
*/
// __global__ void stupid_3x3matrix_inversion(float* A, float* invA, int num_modes);
// __global__ void stupid_2x2matrix_inversion(float* A, float* invA, int num_modes);
// __global__ void stupid_1x1matrix_inversion(float* A, float* invA, int num_modes);

}

#endif