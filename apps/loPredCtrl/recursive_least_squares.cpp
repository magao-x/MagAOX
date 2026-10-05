#include "recursive_least_squares.hpp"
#include <string>

namespace DDSPC
{

RecursiveLeastSquares::RecursiveLeastSquares(int num_predictors, int num_features, realT forgetting_factor, realT initial_covariance){

	_gamma = forgetting_factor;
	_inverse_gamma = 1 / _gamma;

	_initial_covariance = initial_covariance;
	_num_features = num_features;
	_num_predictors = num_predictors;

	// Set size of all the arrays
	prediction_matrix.resize(_num_predictors, _num_features);
	prediction_matrix.setZero();

    prediction_output.resize(_num_predictors, 1);
    prediction_output.setZero();

	inverse_covariance.resize(_num_features, _num_features);
	inverse_covariance.setZero();
	for(int i=0; i < _num_features; i++)
		inverse_covariance(i, i) = _initial_covariance;

	err.resize(_num_predictors, 1);
	err.setZero();

	K.resize(1, _num_features);
	K.setZero();
};


RecursiveLeastSquares::~RecursiveLeastSquares(){

}

void RecursiveLeastSquares::reset(){
	prediction_matrix.setZero();
	err.setZero();
	K.setZero();

	inverse_covariance.resize(_num_features, _num_features);
	inverse_covariance.setZero();
	for(int i=0; i < _num_features; i++)
		inverse_covariance(i, i) = _initial_covariance;
}

// I want to change this interface to make it easier to use.
void RecursiveLeastSquares::update(eigenImage<realT> *x, eigenImage<realT> *y){
	Matrix _x = (*x).matrix();
    err = (*y).matrix();
    err -= prediction_matrix * _x;

	xtP = (_inverse_gamma * _x).transpose() * inverse_covariance;
	realT cn = 1 + (xtP * _x)(0,0);
    K = xtP;
    K /= cn;
    prediction_matrix += err * K;

    inverse_covariance *= _inverse_gamma;
    inverse_covariance -= K.transpose() * xtP;
}

void RecursiveLeastSquares::update(Matrix *x, Matrix *y){
	Matrix _x = (*x);
    err = (*y);
    err -= prediction_matrix * _x;

	xtP = (_inverse_gamma * _x).transpose() * inverse_covariance;
	realT cn = 1 + (xtP * _x)(0,0);
    K = xtP;
    K /= cn;
    prediction_matrix += err * K;

    inverse_covariance *= _inverse_gamma;
    inverse_covariance -= K.transpose() * xtP;
}


// Matrix RecursiveLeastSquares::predict(eigenImage<realT> *x){
//    return prediction_matrix * (*x).matrix();
//}

void RecursiveLeastSquares::save_state(std::string filename){

}

}