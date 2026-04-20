#include "recursive_least_squares.hpp"
#include <string>
#include <stdexcept>
#include <fstream>

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
		inverse_covariance(i, i) = 1 / _initial_covariance;

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
		inverse_covariance(i, i) = 1 / _initial_covariance;
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


void RecursiveLeastSquares::save_state(const std::string &filename){
    std::string metadata_file = filename + ".meta";
    std::ofstream ofs(metadata_file);
    if(!ofs.is_open()){
        throw std::runtime_error("Could not open file for save_state metadata: " + metadata_file);
    }
    ofs << "{\n";
    ofs << "  \"gamma\": " << _gamma << ",\n";
    ofs << "  \"inverse_gamma\": " << _inverse_gamma << ",\n";
    ofs << "  \"initial_covariance\": " << _initial_covariance << ",\n";
    ofs << "  \"num_features\": " << _num_features << ",\n";
    ofs << "  \"num_predictors\": " << _num_predictors << "\n";
    ofs << "}\n";
    ofs.close();

    DDSPC::save_matrix(filename + ".prediction_matrix", prediction_matrix);
    DDSPC::save_matrix(filename + ".inverse_covariance", inverse_covariance);
}

void RecursiveLeastSquares::load_state(const std::string &filename){
    std::string metadata_file = filename + ".meta";
    std::ifstream ifs(metadata_file);
    if(!ifs.is_open()){
        throw std::runtime_error("Could not open file for load_state metadata: " + metadata_file);
    }

    std::string line;
    std::getline(ifs, line); // {

    std::getline(ifs, line); _gamma = static_cast<realT>(std::stod(DDSPC::parse_json_value(line)));
    std::getline(ifs, line); _inverse_gamma = static_cast<realT>(std::stod(DDSPC::parse_json_value(line)));
    std::getline(ifs, line); _initial_covariance = static_cast<realT>(std::stod(DDSPC::parse_json_value(line)));
    std::getline(ifs, line); _num_features = std::stoi(DDSPC::parse_json_value(line));
    std::getline(ifs, line); _num_predictors = std::stoi(DDSPC::parse_json_value(line));

    ifs.close();

    prediction_matrix = DDSPC::load_matrix(filename + ".prediction_matrix");
    inverse_covariance = DDSPC::load_matrix(filename + ".inverse_covariance");
}

}