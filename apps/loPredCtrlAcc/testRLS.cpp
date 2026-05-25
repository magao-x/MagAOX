#define EIGEN_DONT_PARALLELIZE
#include "testRLS.hpp"

#include <random>
#include <chrono>

int main(int argc, char **argv){
    std::default_random_engine generator;
    std::normal_distribution<DDSPC::realT> distribution(0, 1.0);


    DDSPC::Matrix W_true;
    W_true.resize(2, 4);
    W_true.setZero();

    W_true(1,0) = -0.2;
    W_true(1,1) = 0.4;
    W_true(1,2) = 0.1;
    W_true(1,3) = -0.5;

    W_true(0,0) = 0.5;
    W_true(0,1) = -0.3;
    W_true(0,2) = 0.8;
    W_true(0,3) = 0.1;

    DDSPC::print_matrix(W_true, "W_true");

    // Setup the RLS
    int n = 2;
    int m = 4;
    DDSPC::realT forgetting_factor = 0.999;
    DDSPC::realT delta = 1.0;
    
    DDSPC::RecursiveLeastSquares rls = DDSPC::RecursiveLeastSquares(n, m, forgetting_factor, delta);
    DDSPC::QRDRecursiveLeastSquares qrd_rls = DDSPC::QRDRecursiveLeastSquares(n, m, forgetting_factor, delta);
    
    for(int j=0; j < 200; j++){
        DDSPC::Matrix x = DDSPC::Matrix::Random(m, 1);
        DDSPC::Matrix y = W_true * x + 0.01 * DDSPC::Matrix::Random(n, 1); // Add some noise
  
        rls.update(&x, &y);
        qrd_rls.update(&x, &y);

        if(j % 20 == 0){
            std::cout << "RLS error norm: " << (rls.prediction_matrix - W_true).norm() << std::endl;
            std::cout << "QRD RLS error norm: " << (qrd_rls.prediction_matrix - W_true).norm() << std::endl;
        }
    }

    return 0;

}