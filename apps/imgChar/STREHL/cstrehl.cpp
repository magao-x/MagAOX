/**
 * \author  Jacob Trzaska
 * \file    cstrehl.cpp
 * \date    2021-09-23
 */
#include <iostream>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <array>
#include <vector>
#include <complex>
#include <fftw3.h>


/* Parameters */
#define MSKROWS     (1024)               //Unitless
#define MSKCOLS     (1024)               //Unitless
#define MSKELEMENTS (MSKROWS * MSKCOLS)  //Unitless

#define MODROWS     (128)                //Unitless
#define MODCOLS     (128)                //Unitless
#define MODELEMENTS (MODROWS * MODCOLS)  //Unitless

#define DPX         (962.56)             //Unitless

#define NUMRADII    (61)                 //Unitless
#define FNUM        (69.0)               //Unitless
#define PIXEL_SIZE  (4.8)                //Units - um

//Array containing WFS wavelengths.
std::array<double, 2> wlArray {0.791, 0.837};  //Units - um


/* Function Decalarations */
static void
fft_center(double * im, size_t rows, size_t cols);

double
strehlAmp(double * image, size_t rows, size_t cols);


/* Begin Program */

int main()
{
      /* Table for the Strehl amplitude values. */
      std::vector<double> saTable(wlArray.size());

      /* Read in the pupil mask. */ 
      double * pupil = new double[MSKELEMENTS];
      FILE *fptr = fopen("magMask.dat", "rb");
      if(!fptr)
      {
         delete pupil;
         return;
      }
      int stat = fread(pupil, sizeof(double), MSKELEMENTS, fptr);
      if (stat <= 0)
      {
         delete pupil; //just for codacy.
         fclose(fptr);
         return 1;
      }
      fclose(fptr);

      
      for (int n {0}; n < wlArray.size(); ++n)
      {
         int64_t NPX { (int64_t) ((wlArray[n] * FNUM * DPX) / PIXEL_SIZE) };

         /* Table of modulation radii (in pixels). */
         std::vector<double> radii(NUMRADII);
         for (size_t i {0}; i < NUMRADII; ++i)
            radii[i] = ((0.1 * NPX) / static_cast<double>(DPX)) 
                     * i;

         /* Initialize the memory buffers */
         double * psfR
             = new double[NPX * NPX];
         std::complex<double> * psfC 
             = new std::complex<double>[NPX * NPX];
         std::complex<double> * mask 
             = new std::complex<double>[NPX * NPX];

         fftw_plan planF = fftw_plan_dft_2d(NPX, NPX, 
                              reinterpret_cast<fftw_complex*>(mask), 
                              reinterpret_cast<fftw_complex*>(psfC),
                              FFTW_FORWARD,
                              FFTW_MEASURE);

         /* Calculate the PSF from magMask data */
         for (size_t i{0}; i < (NPX * NPX); ++i)
             mask[i] = 0;

        for (size_t i{0}; i < MSKROWS; ++i)
             for (size_t j{0}; j < MSKCOLS; ++j)
                  mask[j + i * NPX] = pupil[j + i * MSKCOLS];

         fftw_execute(planF);

         for (size_t i{0}; i < (NPX * NPX); ++i)
             psfR[i] = abs(psfC[i]) * abs(psfC[i]);

         fft_center(psfR, NPX, NPX);


         /* Crop the PSF into a N x N grid */
         size_t N {MODCOLS};
         double * im = new double[N * N];

         for (size_t i {NPX/2 - N/2}; i < NPX/2 + N/2; ++i)
         {
            for (size_t j {NPX/2 - N/2}; j < NPX/2 + N/2; ++j)
            {
               im[(j - NPX/2 + N/2) + (i - NPX/2 + N/2) * N] = psfR[j + i * NPX];
            }
         }

         saTable[n] = strehlAmp(im, N, N);
 
         fftw_destroy_plan(planF);
         delete[] psfR;
         delete[] psfC;
         delete[] mask;
         delete[] im;
      }

      /* Write to binary file */
      double * sa_data = new double[wlArray.size()];
      for (size_t r {0}; r < wlArray.size(); ++r)
               sa_data[r] = saTable[r];
    
      FILE * sa_ptr = fopen("sa.dat", "wb"); 
      fwrite(sa_data, sizeof(double), wlArray.size(), sa_ptr);
      fclose(sa_ptr);

      delete[] sa_data;
      delete[] pupil;

      return 0;
}



/// Centers the image
/** Shifts the image of a 2D FFT so that the center of the
  * image corresponds to the Zero frequency bin
  *
  */
static void fft_center(double * im, size_t rows, size_t cols)
{
     double temp;

     for (size_t j{0}; j < cols/2; ++j)
          for (size_t i{0}; i < rows; ++i)
          {
               temp = im[j + (i * cols)];
               im[j + (i * cols)] = im[(j + cols/2) + (i * cols)];
               im[(j + cols/2) + (i * cols)] = temp;
          }
    
     for (size_t i{0}; i < rows/2; ++i)
          for (size_t j{0}; j < cols; ++j)
          {
               temp = im[j + (i * cols)];
               im[j + (i * cols)] = im[j + (i + (rows / 2)) * cols];
               im[j + (i + (rows / 2)) * cols] = temp;
          }

}


double strehlAmp(double * image, size_t rows, size_t cols)
{
   double max { image[0] };
   double tot { 0 };

   for (size_t n {0}; n < rows * cols; ++n)
   {
      if (image[n] > max)
         max = image[n];
      tot += image[n];
   }
   return max / tot;
}
