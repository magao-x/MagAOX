#ifndef PCUTIL_CUH
#define PCUTIL_CUH

#include "cuda_runtime.h"
#include "cublas_v2.h"

namespace DDSPC
{

/*
	The following two defines are used to get the correct index for StridedBatched CUDA functions.
*/
// Transform the 2D index to the correct 1D array index for column-major layout.
// i is row, j is column, ld is the number of rows
#define IDX2C(i, j, nrow) (((j) * (nrow)) + (i))
#define BIDX2C(i, j, k, nrow, ncol) (((j) * (nrow)) + (i) + ((k) * (ncol) * (nrow)))

__global__ void divide_scalar_gpu(float* x, float* y, float* z, int element_size, int batch_size);
__global__ void gpu_print_buffer(float* data, int batch_count, int nrow, int ncol, int row_max=-1, int col_max=-1);
void print_batch_buffer(float* data, int batch_count, int nrow, int ncol);

void check_cuda_error(cudaError_t cudaerr);

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

static inline uint find_next_power_of_2(int sample){
    uint num_bits = 0;
    
    do{
        sample >>= 1;
        ++num_bits;
    } while(sample);
    
    return num_bits;
}


}

#endif