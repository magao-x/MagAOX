#include "qrd_rls.hpp"
#include <string>
#include <stdexcept>
#include <fstream>
#include <cmath>

namespace DDSPC
{

QRDRecursiveLeastSquares::QRDRecursiveLeastSquares(int num_predictors, int num_features, realT forgetting_factor, realT initial_covariance){

	_gamma = forgetting_factor;
    _lambda = std::sqrt(_gamma);
	_initial_covariance = initial_covariance;
    _delta = std::sqrt(_initial_covariance);

	_num_features = num_features;
	_num_predictors = num_predictors;

	// Set size of all the arrays
	R_.resize(_num_features, _num_features);
	R_.setZero();

    for(int i=0; i < _num_features; i++)
		R_(i, i) = _delta;

    prediction_matrix.resize(_num_predictors, _num_features);
    prediction_matrix.setZero();

    A_.resize(_num_features + 1, _num_features);
    A_.setZero();

	err.resize(_num_predictors, 1);
	err.setZero();

    z.resize(_num_features, 1);
	z.setZero();

    g.resize(_num_features, 1);
	g.setZero();

    prediction_output.resize(_num_predictors, 1);
    prediction_output.setZero();

};


QRDRecursiveLeastSquares::~QRDRecursiveLeastSquares(){

}

void QRDRecursiveLeastSquares::reset(){
	prediction_matrix.setZero();
	err.setZero();
	prediction_matrix.setZero();
    A_.setZero();

	R_.setZero();
    for(int i=0; i < _num_features; i++)
		R_(i, i) = _delta;
}


// ------------------------------------------------------------
// Givens rotation (branch-light)
// ------------------------------------------------------------
EIGEN_STRONG_INLINE
void QRDRecursiveLeastSquares::givens(realT a, realT b, realT& c, realT& s) {
    if (b == realT(0)) { c = realT(1); s = realT(0); return; }
    const realT r = std::hypot(a, b);
    c = a / r;
    s = -b / r;
}

// ------------------------------------------------------------
// Apply Givens to zero last row
// ------------------------------------------------------------
EIGEN_STRONG_INLINE
void QRDRecursiveLeastSquares::apply_givens() {
    int N = _num_features;

    for (int j = 0; j < N; ++j) {
        for (int i = N; i > j; --i) {
            realT a = A_(i-1, j);
            realT b = A_(i,   j);
            if (b == realT(0)) continue;

            realT c, s;
            givens(a, b, c, s);

            // rotate rows i-1 and i for columns j..N-1
            // unrolled inner loop for better SIMD
            int k = j;
            for (; k <= N-4; k += 4) {
                // #pragma unroll
                for (int u = 0; u < 4; ++u) {
                    realT r1 = A_(i-1, k+u);
                    realT r2 = A_(i,   k+u);
                    A_(i-1, k+u) = c * r1 - s * r2;
                    A_(i,   k+u) = s * r1 + c * r2;
                }
            }
            for (; k < N; ++k) {
                realT r1 = A_(i-1, k);
                realT r2 = A_(i,   k);
                A_(i-1, k) = c * r1 - s * r2;
                A_(i,   k) = s * r1 + c * r2;
            }
        }
    }
}

// ------------------------------------------------------------
// Triangular solves (no alloc, unrolled-friendly)
// ------------------------------------------------------------
EIGEN_STRONG_INLINE
void QRDRecursiveLeastSquares::solve_Rt(Matrix& x, Matrix& z) {
    int N = _num_features;

    for (int i = 0; i < N; ++i) {
        realT s = x(i);
        for (int k = 0; k < i; ++k)
            s -= R_(k, i) * z(k);
        z(i) = s / R_(i, i);
    }
}

EIGEN_STRONG_INLINE
void QRDRecursiveLeastSquares::solve_R(Matrix& z, Matrix& g) {
    int N = _num_features;

    for (int i = N-1; i >= 0; --i) {
        realT s = z(i);
        for (int k = i+1; k < N; ++k)
            s -= R_(i, k) * g(k);
        g(i) = s / R_(i, i);
    }
}

// I want to change this interface to make it easier to use.
void QRDRecursiveLeastSquares::update(eigenImage<realT> *x, eigenImage<realT> *y){
    Matrix _x = (*x).matrix();
    err = (*y).matrix();
    update(&_x, &err);
}

void QRDRecursiveLeastSquares::update(Matrix *x, Matrix *y){
    int N = _num_features;

    Matrix _x = (*x);
    err = (*y);

    // 1) Forgetting
    R_ *= _lambda; // Eigen vectorizes

    // 2) Build augmented A = [R; x^T]
    A_.topRows(N) = R_;
    A_.row(N) = _x.transpose();

    // 3) QR update
    // if (method_ == QRD_RLS_Method::Givens) {
    apply_givens();

    // Extract R (upper part already in A_)
    R_ = A_.topRows(N);

    // 4) Output y = W x
    prediction_output = prediction_matrix * _x; // vectorized GEMV
    err -= prediction_output;

    // 5) Gain g (shared across outputs)
    z.setZero();
    g.setZero();
    solve_Rt(_x, z);
    solve_R(z, g);

    // 6) Rank-1 update: W += e * g^T  (M x 1 times 1 x N)
    // Eigen will use vectorized outer product
    prediction_matrix += err * g.transpose();
}

void QRDRecursiveLeastSquares::save_state(const std::string &filename){
    std::cout << filename << std::endl;
    /*
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
    */
}

void QRDRecursiveLeastSquares::load_state(const std::string &filename){
    std::cout << filename << std::endl;
    /*
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
    */
}

}