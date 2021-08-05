#include <iostream>
#include <string>
#include <complex>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <fftw3.h>

void mean_subtraction(double *image, int ROWS, int COLS);

void point_multiply0( fftw_complex *mat1, fftw_complex *matf, int rows, int cols);

void point_multiply(fftw_complex *mat1, fftw_complex *mat2, fftw_complex *matf, int rows, int cols);

void locate_max(int ROWS, int COLS, double *mat, int *delta_i, int *delta_j);

void image0_fft_fill(fftw_complex *image0_fft_FDZP, fftw_complex *image_ftt, int rows, int cols);

void crop_image(double *dest, double *src, int IMROWS, int IMCOLS, int x, int y, int SZ);
