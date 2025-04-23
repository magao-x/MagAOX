#ifndef PCUTILS_HPP
#define PCUTILS_HPP

#include <Eigen/Dense>
#include <string>
#include <iostream>
#include <fstream>

namespace DDSPC
{
    typedef float realT;
    typedef Eigen::Matrix<realT, Eigen::Dynamic, Eigen::Dynamic> Matrix;

    inline void print_shape_matrix(Matrix mat, std::string name=""){
        std::cout << name << " Matrix is of size " << mat.rows() << "x" << mat.cols() << std::endl;
    };

    void print_matrix(Matrix mat, std::string name);

    static inline uint64_t find_next_power_of_2(int sample){
        uint64_t num_bits = 0;
        
        do{
            sample >>= 1;
            ++num_bits;
        } while(sample);
        
        return 1 << num_bits;
    };

    void save_matrix(std::string fileName, Matrix mat);
    Matrix load_matrix(std::string fileName);
}

#endif // utils_hpp