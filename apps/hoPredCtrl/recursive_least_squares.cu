#include "recursive_least_squares.cuh"

namespace DDSPC
{

RecursiveLeastSquares::RecursiveLeastSquares(cublasHandle_t* new_handle, int num_features, int num_predictors, int batch_size, float forgetting_factor, float P0){
	// A = make_random_matrix(1e-7, num_predictors, num_features, batch_size);
	A = new Matrix(0.0, num_predictors, num_features, batch_size);
	P = make_identity_matrix_from_scalar(P0, num_features, batch_size);
	
	gamma = forgetting_factor;
	_initial_covariance = P0;
	_num_features = num_features;
	_batch_size = batch_size;

	// K is a row vector
	K = new Matrix(0, 1, num_features, batch_size);
	// err is a column vector
	err = new Matrix(0, num_predictors, 1, batch_size);
	// xtP is a row vector
	xtP = new Matrix(0, 1, num_features, batch_size);

	// cn is a scalar
	cn = new Matrix(0, 1, 1, batch_size);
	// gamma_vec is a scalar field
	gamma_vec = new Matrix(gamma, 1, 1, batch_size);

	handle = new_handle;

	A->set_handle(new_handle);
	K->set_handle(new_handle);
	P->set_handle(new_handle);
	gamma_vec->set_handle(new_handle);
	err->set_handle(new_handle);
	xtP->set_handle(new_handle);
	cn->set_handle(new_handle);
};


RecursiveLeastSquares::~RecursiveLeastSquares(){
	delete A;
	delete K;
	delete P;
	delete err;
	delete xtP;
	delete cn;
	delete gamma_vec;
}


void RecursiveLeastSquares::reset(){
	A->set_to_zero();
	K->set_to_zero();
	err->set_to_zero();
	xtP->set_to_zero();
	cn->set_to_zero();
	
	// delete P;
	// P = make_identity_matrix_from_scalar(P0, num_features, batch_size);		
	P->set_to_zero();
	set_identity_matrix(P, _initial_covariance, _num_features, _batch_size);
}


void RecursiveLeastSquares::update(Matrix *x, Matrix *y){
	// Total time ~	0.275 ms
	
	// This seems to take 0.03ms on average
	x->dot(P, xtP, 1.0, 0.0, CUBLAS_OP_T, CUBLAS_OP_N);

	// calculate the gain vector
	// 0.248328 -> 0.22522 = 0.0231
	xtP->dot(x, cn);
	
	// The computation time is within the error of 0.005
	cn->add(gamma_vec);
	xtP->divide_by_scalar(cn, K);

	// Calculate the a-priori error
	// This takes about 0.025 - 0.030 ms
	A->dot(x, err, -1.0, 0.0, CUBLAS_OP_N, CUBLAS_OP_N);
	// 0.03 ms -> this can probably be made a bit more efficient
	err->add(y);

	// Total time ~	0.275 ms
	// Update the A matrix
	// Without A update : 0.218 ms
	// So A update takes about 0.06 ms
	err->dot(K, A, 1.0, 1.0, CUBLAS_OP_N, CUBLAS_OP_N);

	// Update the P matrix
	// Without P update 0.079 ms
	// So P update takes about 0.16 ms.
	xtP->dot(K, P, -1.0 / gamma, 1.0 / gamma, CUBLAS_OP_T, CUBLAS_OP_N);
}

}