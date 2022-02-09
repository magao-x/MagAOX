#include "utility_functions.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <iostream>
#include <random>
#include <vector>
#include <string>

namespace DDSPC
{

__global__ void gpu_print_buffer(float* data, int batch_count, int nrow, int ncol) {
	// int index = blockIdx.x * blockDim.x + threadIdx.x;
	// int stride = blockDim.x * gridDim.x;

	// The index steps through the number of modes
	//printf("[");
	for (int k = 0; k < batch_count; k++) {

		for (int i = 0; i < nrow; i++) {
			printf("[");

			for (int j = 0; j < ncol; j++) {

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

__global__ void gpu_print_buffer(double* data, int batch_count, int nrow, int ncol) {
	// int index = blockIdx.x * blockDim.x + threadIdx.x;
	// int stride = blockDim.x * gridDim.x;

	// The index steps through the number of modes
	//printf("[");
	for (int k = 0; k < batch_count; k++) {

		for (int i = 0; i < nrow; i++) {
			printf("[");

			for (int j = 0; j < ncol; j++) {
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

__global__ void stupid_3x3matrix_inversion(float* A, float* invA, int num_modes) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int k = index; k < num_modes; k += stride) {

		float a = A[BIDX2C(0, 0, k, 3, 3)];
		float b = A[BIDX2C(0, 1, k, 3, 3)];
		float c = A[BIDX2C(0, 2, k, 3, 3)];
		float d = A[BIDX2C(1, 0, k, 3, 3)];
		float e = A[BIDX2C(1, 1, k, 3, 3)];
		float f = A[BIDX2C(1, 2, k, 3, 3)];
		float g = A[BIDX2C(2, 0, k, 3, 3)];
		float h = A[BIDX2C(2, 1, k, 3, 3)];
		float i = A[BIDX2C(2, 2, k, 3, 3)];

		float det = (a * e * i + b * f * g + c * d * h - c * e * g - a * f * h - b * d * i);

		invA[BIDX2C(0, 0, k, 3, 3)] = (e*i-f*h)/det;
		invA[BIDX2C(0, 1, k, 3, 3)] = (h*c-i*b)/det;
		invA[BIDX2C(0, 2, k, 3, 3)] = (b*f-c*e)/det;

		invA[BIDX2C(1, 0, k, 3, 3)] = (g*f-d*i) / det;
		invA[BIDX2C(1, 1, k, 3, 3)] = (a*i-g*c) / det;
		invA[BIDX2C(1, 2, k, 3, 3)] = (d*c-a*f) / det;

		invA[BIDX2C(2, 0, k, 3, 3)] = (d*h-g*e) / det;
		invA[BIDX2C(2, 1, k, 3, 3)] = (g*b-a*h) / det;
		invA[BIDX2C(2, 2, k, 3, 3)] = (a*e-d*b) / det;

	}
}

__global__ void stupid_2x2matrix_inversion(float* A, float* invA, int num_modes) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int k = index; k < num_modes; k += stride) {

		float a = A[BIDX2C(0, 0, k, 3, 3)];
		float b = A[BIDX2C(0, 1, k, 3, 3)];
		float c = A[BIDX2C(1, 0, k, 3, 3)];
		float d = A[BIDX2C(1, 1, k, 3, 3)];

		float det = a * d - b * c;

		invA[BIDX2C(0, 0, k, 3, 3)] = d/det;
		invA[BIDX2C(0, 1, k, 3, 3)] = -b/det;
		invA[BIDX2C(1, 0, k, 3, 3)] = -c/det;
		invA[BIDX2C(1, 1, k, 3, 3)] = a/det;

	}
}

__global__ void stupid_1x1matrix_inversion(float* A, float* invA, int num_modes) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int k = index; k < num_modes; k += stride) {
		invA[BIDX2C(0, 0, k, 3, 3)] = 1/A[BIDX2C(0, 0, k, 3, 3)];
	}
}

uint find_next_power_of_2(int sample){
    uint num_bits = 0;
    
    do{
        sample >>= 1;
        ++num_bits;
    } while(sample);
    
    return num_bits;
};


}