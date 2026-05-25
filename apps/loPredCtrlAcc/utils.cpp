#include <Eigen/Dense>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>

#include "utils.hpp"

namespace DDSPC
{
    void save_matrix(std::string fileName, Matrix mat){
        // https://eigen.tuxfamily.org/dox/structEigen_1_1IOFormat.html
        const static Eigen::IOFormat CSVFormat (Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");

        std::ofstream file(fileName);
        if (file.is_open()){
            file << mat.format(CSVFormat);
            file.close();
        }
    }

    void print_matrix(Matrix mat, std::string name=""){
        if(name.length() > 0)
            std::cout << name << std::endl;

        for(int i=0; i < mat.rows(); i++){
            std::cout << "[";
            for(int j=0; j < mat.cols(); j++){
                std::cout << mat(i, j);
                if(j < (mat.cols() - 1))
                    std::cout << ",";
            }
            std::cout << "]";
            std::cout << std::endl;
        }
    }

    Matrix load_matrix(std::string fileName){

        std::vector<realT> matrixEntries;
        std::ifstream matrixDataFile(fileName);
        std::string matrixRowString;
        std::string matrixEntry;
        int matrixRowNumber = 0;

        while (getline(matrixDataFile, matrixRowString)){

            std::stringstream matrixRowStringStream(matrixRowString);

            while (getline(matrixRowStringStream, matrixEntry, ',')){
                matrixEntries.push_back(static_cast<realT>(stod(matrixEntry)));
            }
            matrixRowNumber++;
        }

        return  Eigen::Map<Matrix>(matrixEntries.data(), matrixRowNumber, matrixEntries.size() / matrixRowNumber);

    }

    std::string trim(const std::string &s){
        auto start = s.find_first_not_of(" \t\n\r");
        if(start == std::string::npos) return "";
        auto end = s.find_last_not_of(" \t\n\r");
        return s.substr(start, end - start + 1);
    }

    std::string parse_json_value(const std::string &line){
        auto colon = line.find(':');
        if(colon == std::string::npos){
            throw std::runtime_error("Invalid metadata line in JSON metadata");
        }
        std::string value = line.substr(colon + 1);
        value = trim(value);
        if(!value.empty() && value.back() == ','){
            value.pop_back();
            value = trim(value);
        }
        if(!value.empty() && value.front() == '"' && value.back() == '"'){
            value = value.substr(1, value.size() - 2);
        }
        return value;
    }
}