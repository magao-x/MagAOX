#ifndef PC_QRDRLS_HPP
#define PC_QRDRLS_HPP

#include <Eigen/Dense>
#include <mx/improc/eigenCube.hpp>
#include <mx/improc/eigenImage.hpp>
using namespace mx::improc;
#include "utils.hpp"
#include "recursive_least_squares.hpp"

namespace DDSPC
{

enum class QRD_RLS_Method { Householder, Givens };

/*
	This is just a generic Recursive Least Squares implementation.
*/

class QRDRecursiveLeastSquares{

	private:

	public:
		int _num_features;
        int _num_predictors;

		realT _gamma;
		realT _lambda;
		realT _delta;
		realT _initial_covariance;
		QRD_RLS_Method method_;

		Matrix R_;     // Upper-triangular
		Matrix prediction_matrix;     // Weights (M x N)
		Matrix A_;     // Augmented for QR
		
		Matrix err;         // The a-priori prediction error
		Matrix prediction_output;
		Matrix z;
		Matrix g;

		QRDRecursiveLeastSquares(int num_predictors, int num_features, realT forgetting_factor, realT inverse_covariance);
		~QRDRecursiveLeastSquares();

		void givens(realT a, realT b, realT& c, realT& s);
		void apply_givens();
		void solve_Rt(Matrix& x, Matrix& z);
		void solve_R(Matrix& x, Matrix& z);

        // This interface might need to change
		void update(eigenImage<realT> *x, eigenImage<realT> *y);
		void update(Matrix *x, Matrix *y);

        //Matrix* predict(eigenImage<realT> *x);
		void reset();
		void save_state(const std::string &filename);
		void load_state(const std::string &filename);
};

}

#endif
