#include <iostream>
#include "device_launch_parameters.h"
#include "utils.cuh"

namespace DDSPC
{

//	cudaError_t cudaerr = cudaDeviceSynchronize();
void check_cuda_error(cudaError_t cudaerr ){
	if (cudaerr != cudaSuccess)
		printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
}

__global__ void divide_scalar_gpu(float* x, float* y, float* z, int element_size, int batch_size){
	/*
		calculates:
			z = y / x
	*/
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int k = index; k < batch_size; k += stride) {
		for(int i=0; i < element_size; i++){
			z[BIDX2C(i, 0, k, element_size, 1)] = y[BIDX2C(i, 0, k, element_size, 1)] / x[k]; 
		}
	}
}

__global__ void gpu_print_buffer(float* data, int batch_count, int nrow, int ncol, int row_max, int col_max) {
	// int index = blockIdx.x * blockDim.x + threadIdx.x;
	// int stride = blockDim.x * gridDim.x;

	if(row_max == -1 || row_max > nrow){
		row_max = nrow;
	}

	if(col_max == -1 || col_max > ncol){
		col_max = ncol;
	}

	// The index steps through the number of modes
	//printf("[");
	for (int k = 0; k < batch_count; k++) {

		for (int i = 0; i < row_max; i++) {
			printf("[");

			for (int j = 0; j < col_max; j++) {
				int index = BIDX2C(i, j, k, nrow, ncol);
				printf("%f", data[index]);
				
				// int index = IDX2C(i, j, nrow);
				// printf("%f", data[k][index]);

				if (j != (ncol - 1))
					printf(", ");
			}
			printf("]\n");
		}
	}
	//printf("]\n");
};

/*
void print_batch_buffer(float* data, int batch_count, int nrow, int ncol) {
	for (int k = 0; k < batch_count; k++) {

		for (int i = 0; i < nrow; i++) {
			std::cout << "[";

			for (int j = 0; j < ncol; j++) {
				int index = IDX2C(i, j, nrow);
				std::cout << data[k][index];

				//int index = BIDX2C(i, j, k, nrow, ncol);
				//std::cout << data[k][index];
				if (j != (ncol - 1))
					std::cout << ", ";
			}
			std::cout << "]" << std::endl;
		}
		std::cout << std::endl;
	}
};
*/

}