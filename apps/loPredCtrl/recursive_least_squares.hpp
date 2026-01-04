#ifndef PCRLS_HPP
#define PCRLS_HPP

#include <Eigen/Dense>
#include <mx/improc/eigenCube.hpp>
#include <mx/improc/eigenImage.hpp>
using namespace mx::improc;
#include "utils.hpp"

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
		realT _gamma;
        realT _inverse_gamma;
		realT _initial_covariance;
		int _num_features;
        int _num_predictors;

        Matrix K;           // The gain matrix
		Matrix err;         // The a-priori prediction error
		Matrix xtP;         // A convenience variable


		Matrix prediction_matrix;         // Prediction matrix
		Matrix inverse_covariance;       // Inverse covariance

        Matrix prediction_output;

		RecursiveLeastSquares(int num_predictors, int num_features, realT forgetting_factor, realT inverse_covariance);
		~RecursiveLeastSquares();

        // This interface might need to change
		void update(eigenImage<realT> *x, eigenImage<realT> *y);
		void update(Matrix *x, Matrix *y);

        //Matrix* predict(eigenImage<realT> *x);
		void reset();
		void save_state(std::string filaname);
};

}

#endif
