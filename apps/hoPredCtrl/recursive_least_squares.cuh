#ifndef PCRLS_CUH
#define PCRLS_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"

#include "utils.cuh"
#include "new_matrix.cuh"

namespace DDSPC
{

/*
	This is just a generic Recursive Least Squares implementation.
	The auto-regressive model will be implemented in a different class.
	This allows for better reuse of the Recursive Least Squares.
*/


class RecursiveLeastSquares{

	private:
		

	public:
		Matrix* K; // Gain matrix
		float gamma; // The forgetting factor
		float _initial_covariance;
		int _num_features;
		int _batch_size;

		Matrix* gamma_vec;
		Matrix* err; // The a-priori prediction error
		Matrix* xtP;
		Matrix* cn;
		cublasHandle_t* handle;
		
		Matrix* A; // Prediction matrix
		Matrix* P; // Inverse covariance
		
		RecursiveLeastSquares(cublasHandle_t* new_handle, int num_features, int num_predictors, int batch_size, float forgetting_factor, float P0);
		~RecursiveLeastSquares();

		void update(Matrix *x, Matrix *y);
		void reset();
		void save_state(std::string filaname);
};

}

#endif
