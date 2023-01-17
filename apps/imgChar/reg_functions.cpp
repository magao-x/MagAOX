/****************************************
* Author: Jacob Trzaska                 *
* Purpose: Functions to aid in image    *
*          registration                 *
****************************************/
#include "reg_functions.hpp"



/***************************************************
*  Function: mean_subtraction()                    *
*  Description: Removes the mean from the image    *
*  ---------------------------------------------   *
*  List of Arguments:                              *
*  double*  image - the image to mean subtracted    *
*  int      ROWS - number of rows in image         *
*  int      COLS - numbers of cols in image        *
***************************************************/

void mean_subtraction(double *image, int ROWS, int COLS)
{
    double mean=0;
    
    for(int j=0; j<ROWS; j++)
      for(int i=0; i<COLS; i++)
        mean += image[i+j*COLS];

    mean = mean/(ROWS*COLS);

    for(int j=0; j<ROWS; j++)
      for(int i=0; i<COLS; i++)
        image[i+j*COLS] = image[i+j*COLS] - mean;

}




/********************************************************************************************
*  Function: point_multiply()                                                               *
*  Description: Puts the product of fft of cropped image0 and image in an array for ifft    *
*  -----------------------------------------------------------                              *
*  List of Arguments:                                                                       *
*  fftw_complex*  mat1 - fft of cropped image0                                             *
*  fftw_complex*  mat2 - fft of cropped image                                              * 
*  fftw_complex*  matf - extended product of mat1 & mat2                                   *
*  int            rows - number of rows in mat1 & mat2                                      *
*  int            cols - number of cols in mat1 & mat2                                      *
*  int       cols_fdzp - number of cols in matf                                             *
*  int    memory block - spacing between the two segments of memory                         *
********************************************************************************************/

void point_multiply(fftw_complex *mat1, fftw_complex *mat2, fftw_complex *matf, int rows, int cols)
{
    for (int i=0; i<rows; i++)
      for (int j=0; j<cols; j++)
        {
          (matf[j+i*cols])[0] = (mat1[j+i*cols])[0]*(mat2[j+i*cols])[0] - (mat1[j+i*cols])[1]*(mat2[j+i*cols])[1];
          (matf[j+i*cols])[1] = (mat1[j+i*cols])[1]*(mat2[j+i*cols])[0] + (mat1[j+i*cols])[0]*(mat2[j+i*cols])[1];        
        }

}




/*******************************************************************
*  Function: locate_max()                                          *
*  Description: finds max pixel location                           *
*  ----------------------------------------------                  *
*  List of Arguments:                                              *
*  int        ROWS - number of rows in image                       *
*  int        COLS - number of cols in image                       *
*  double*     mat  - image matrix (results of cross-correlation)   *
*  int *delta_i    - row index of max correlation                  *
*  int *delta_j    - column index of max correlation               *
*******************************************************************/

void locate_max(int ROWS, int COLS, double *mat, int *delta_i, int *delta_j)
{
    double max = fabs(mat[0]);

        for ( int i=0; i<10; i++ )
          for ( int j=0; j<10; j++ )
            if ( fabs(mat[j+i*COLS]) > max )
              {
                *delta_i=i; *delta_j=j; max = fabs(mat[j+i*COLS]);
              }

         for ( int i=ROWS-10; i<ROWS; i++ )
          for ( int j=COLS-10; j<COLS; j++ )
            if ( fabs(mat[j+i*COLS]) > max )
              {
                *delta_i=i-ROWS; *delta_j=j-COLS; max = fabs(mat[j+i*COLS]);
              } 
        

        for ( int i=0; i<10; i++ )
          for ( int j=COLS-10; j<COLS; j++ )
            if ( fabs(mat[j+i*COLS]) > max )
              { 
                *delta_i=i; *delta_j=j-COLS; max = fabs(mat[j+i*COLS]);
              }
    
         for ( int i=ROWS-10; i<ROWS; i++ )
          for ( int j=0; j<10; j++ )
            if ( fabs(mat[j+i*COLS]) > max )
              {
                *delta_i=i-ROWS; *delta_j=j; max = fabs(mat[j+i*COLS]);
              } 
}







/*********************************************************
*  Function: image0_fft_FDZP()                           *
*  Description: copies image0's fft into a new array     *
*  -------------------------------------------------     *
*  List of Arguments:                                    *
*  fftw_complex*  image0_fft_FDZP - copy of image fft   *
*  fftw_complex*        image_fft - fft of image        *
*  int                       rows - number of rows       *
*  int                       cols - number of cols       *
*********************************************************/

void image0_fft_fill(fftw_complex *image0_fft_FDZP, fftw_complex *image_fft, int rows, int cols)
{
    for (int i=0; i<rows; i++)
      for (int j=0; j<cols; j++)
        {
          (image0_fft_FDZP[j+i*cols])[0] = (image_fft[j+i*cols])[0];
          (image0_fft_FDZP[j+i*cols])[1] = -(image_fft[j+i*cols])[1];
        }


}


/*******************************************************
*  Function: crop_image()                              *
*  Description: Crops out the PSF from the image       *
*  -----------------------------------------------     *
*  List of Arugments:                                  *
*  double* dest - the cropped image                    *
*  double*  src - the image being cropped              *
*  int   IMROWS - number of rows in original image     *
*  int     ROWS - number of rows in cropped image      *
*  int   lxFill - first column index to be cropped     *
*  int   uxFill - last column index to be cropped      *
*  int   lyFill - first row index to be cropped        *
*  int   uyFill - last row index to be cropped         *  
*******************************************************/

void crop_image(double *dest, double *src, int IMROWS, int IMCOLS, int x, int y, int SZ)
{ 
    int lx=x-SZ, ly=y-SZ;
    int array_x[2*SZ+1], array_y[2*SZ+1];
    
    for (int i=0; i<2*SZ+1; i++) {
        array_x[i] = ( (lx+i<0) ? IMCOLS+(lx+i) : lx+i );
        array_y[i] = ( (ly+i<0) ? IMROWS+(ly+i) : ly+i );
    }

    for (int i=0; i<2*SZ+1; i++)
        for (int j=0; j<2*SZ+1; j++) {
            dest[j+i*(2*SZ+1)] = src[array_x[j]+array_y[i]*IMCOLS];
        }

}
