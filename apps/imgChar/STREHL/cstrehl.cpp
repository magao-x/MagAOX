/**
 * \author  Jacob Trzaska
 * \file    cstrehl.cpp
 * \date    2021-09-12
 */
#include <iostream>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <array>
#include <vector>
#include <complex>
#include <fftw3.h>

#include <x86intrin.h>

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

static double *
modulate(double * im, size_t rows, size_t cols, double radius);

static void
estimateCentroid(double * im, size_t rows, size_t cols,  double * centroid);

static double
getStrehlMod(double * im, size_t rows, size_t cols, double * centroid);

static double
getFluxTotal(double * im, size_t rows, size_t cols);

static void
imshift(double * im, size_t rows, size_t cols, int xshift, int yshift);

static void
imshiftMov(double * im, size_t rows, size_t cols, int xshift, int yshift, 
           double * mod);



/* Begin Program */

int main()
{
      /* Table for the Strehl amplitude values. */
      std::vector<double> saTable(NUMRADII * wlArray.size());

      /* Read in the pupil mask. */ 
      double * temp = new double[MSKELEMENTS];
      FILE *fptr = fopen("magMask.dat", "rb");
      int stat = fread(temp, sizeof(double), MSKELEMENTS, fptr);
      if (stat <= 0)
          return 1;
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
                  mask[j + i * NPX] = temp[j + i * MSKCOLS];

         fftw_execute(planF);

         for (size_t i{0}; i < (NPX * NPX); ++i)
             psfR[i] = abs(psfC[i]) * abs(psfC[i]);

         fft_center(psfR, NPX, NPX);


         /* Crop the PSF into a N x N grid */
         size_t N {MODCOLS};
         double * im0 = new double[N * N];

         for (size_t i {NPX/2 - N/2}; i < NPX/2 + N/2; ++i)
         {
            for (size_t j {NPX/2 - N/2}; j < NPX/2 + N/2; ++j)
            {
               im0[(j - NPX/2 + N/2) + (i - NPX/2 + N/2) * N] = psfR[j + i * NPX];
            }
         }
 
         /*** Time to modulate the PSF. */
         double * modIm {nullptr};
         double  ctr[2] {0, 0};     

         double sMax   {0};
         double sTotal {0};

         for (size_t i {0}; i < NUMRADII; ++i)
         {
            // Revolve the point spread function in a circle
            modIm = modulate(im0, N, N, radii[i]);

            // Calculate the theoretical Strehl amplitude
            estimateCentroid(modIm, N, N, ctr);
            sMax = getStrehlMod(modIm, N, N, ctr);
            sTotal = getFluxTotal(modIm, N, N);
            saTable[i + n * NUMRADII] = sMax / sTotal;
        
            // reset modIm for the next iteration 
            delete[] modIm;
            modIm = nullptr; 
         }

         fftw_destroy_plan(planF);
         delete[] psfR;
         delete[] psfC;
         delete[] mask;
         delete[] im0;
      }

      /* Write to binary file */
      double * sa_data = new double[NUMRADII * wlArray.size()];
      for (size_t r {0}; r < NUMRADII * wlArray.size(); ++r)
               sa_data[r] = saTable[r];
    
      FILE * sa_ptr = fopen("sa.dat", "wb"); 
      fwrite(sa_data, sizeof(double), NUMRADII * wlArray.size(), sa_ptr);
      fclose(sa_ptr);

      delete[] sa_data;
      delete[] temp;

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


/// Modulate the input image
/** Revolves the image in a circle with radius given by the 
  * input parameter 'radius'.
  *
  */
static double * 
modulate(double * im, size_t rows, size_t cols, double radius)
{
     const size_t NUM_ANGLES {1200};
     std::vector<double> angles(NUM_ANGLES);
     for (size_t i{0}; i < NUM_ANGLES; ++i)
          angles[i] = (2 * M_PI * i) / NUM_ANGLES;

     std::vector<std::vector<int>> dr(2); //[0] -> x, [1] -> y
     dr[0].resize(NUM_ANGLES);
     dr[1].resize(NUM_ANGLES);
     
     for (size_t i{0}; i < NUM_ANGLES; ++i)
     {
          dr[0][i] = (radius * cos(angles[i]));
          dr[1][i] = (radius * sin(angles[i]));
     }

     double * temp = new double[rows * cols];
     memset(temp, 0, sizeof(double) * rows * cols);
     double * mod = new double[rows * cols];

     for (size_t a {0}; a < NUM_ANGLES; ++a)
     {
          imshiftMov(im, rows, cols, dr[0][a], dr[1][a], mod);
          for (size_t i {0}; i < rows * cols; ++i)
              temp[i] += mod[i];
     }

     delete[] mod;

     return temp; 
}


static void
estimateCentroid(double * im, size_t rows, size_t cols,  double * centroid)
{
     double xctr {0};
     double yctr {0};
     double tot {0};

     for (size_t i {0}; i < rows; ++i)
          for (size_t j {0}; j < cols; ++j)
          {
               tot  += im[j + i * cols];
               xctr += im[j + i * cols] * j;
               yctr += im[j + i * cols] * i;
          }

     centroid[0] = xctr / tot;
     centroid[1] = yctr / tot;
}


static double
getStrehlMod(double * im, size_t rows, size_t cols, double * centroid)
{
     // Setup arrays to track maximum image values
     const size_t numlines  = 512;
     double * heights       = new double[numlines]; 
     double * search_angles = new double[numlines];

     for (int i = 0; i < numlines; ++i)
     {
          search_angles[i] = (2.0 * M_PI * i) / numlines;
     }

     // Launch rays to get maximum points for Airy peaks
     __m128i xtrack = _mm_set1_epi32(centroid[0]);
     __m128i ytrack = _mm_set1_epi32(centroid[1]);
     __m256d amplitude = _mm256_setzero_pd();

     __m128 iteratorC = _mm_set1_ps(0.0);
     __m128 iteratorS = _mm_set1_ps(0.0);
     __m128 cos_angle, sin_angle;
     __m128 ones = _mm_set1_ps(1.0);
     __m256d comp;  
     
     __m128i zeros = _mm_set1_epi32(0);
     __m128i maskX1, maskX2, maskY1, maskY2;
    
     int *vx = (int *) &xtrack;
     int *vy = (int *) &ytrack;

     for (int bin = 0; bin < numlines; bin += 4) // make sure (numlines % 4 == 0)
     { 
          cos_angle = _mm256_cvtpd_ps(
                         _mm256_set_pd( cos(search_angles[bin + 0]), 
                                        cos(search_angles[bin + 1]),  
                                        cos(search_angles[bin + 2]), 
                                        cos(search_angles[bin + 3])
                                      )
                                );

          sin_angle = _mm256_cvtpd_ps(
                         _mm256_set_pd( sin(search_angles[bin + 0]),
                                        sin(search_angles[bin + 1]), 
                                        sin(search_angles[bin + 2]),
                                        sin(search_angles[bin + 3])
                                      )
                                     );
          CONTINUE_SEARCH:
              // Mask the break condition for each channel to determine when
              // to end the "while" loop 
              maskX1 = _mm_cmpgt_epi32(_mm_set1_epi32(cols-1), xtrack);
              maskX2 = _mm_cmpgt_epi32(xtrack, zeros);
              maskY1 = _mm_cmpgt_epi32(_mm_set1_epi32(rows-1), ytrack);
              maskY2 = _mm_cmpgt_epi32(ytrack, zeros);

              maskX1 = _mm_and_si128(maskX1, maskX2);
              maskY1 = _mm_and_si128(maskY1, maskY2);
              maskX1 = _mm_and_si128(maskX1, maskY1);
               
              if ( _mm_movemask_ps(_mm_castsi128_ps(maskX1)) == 0) {
                  goto EXIT_SEARCH; 
              }

              comp = _mm256_set_pd(
                          im[vx[3] + (vy[3] * cols)], 
                          im[vx[2] + (vy[2] * cols)],
                          im[vx[1] + (vy[1] * cols)], 
                          im[vx[0] + (vy[0] * cols)]
                          );
              amplitude = _mm256_max_pd(comp, amplitude);
                 
              iteratorC = _mm_and_ps(_mm_castsi128_ps(maskX1), cos_angle);
              iteratorS = _mm_and_ps(_mm_castsi128_ps(maskX1), sin_angle);

              xtrack = _mm_add_epi32(xtrack, _mm_cvtps_epi32(iteratorC));
              ytrack = _mm_add_epi32(ytrack, _mm_cvtps_epi32(iteratorS));
              goto CONTINUE_SEARCH;

          EXIT_SEARCH: 
              heights[bin + 0] = amplitude[0];
              heights[bin + 1] = amplitude[1];
              heights[bin + 2] = amplitude[2];
              heights[bin + 3] = amplitude[3];

              amplitude = _mm256_setzero_pd();
              xtrack = _mm_set1_epi32(centroid[0]);
              ytrack = _mm_set1_epi32(centroid[1]);
     } 

     // Perform averaging and get Strehl ratio
     double strehlAmplitude {0};
     for (size_t i {0}; i < numlines; ++i)
     {
          strehlAmplitude += heights[i]; 
     }
     
     strehlAmplitude /= numlines; 

     delete[] heights;
     delete[] search_angles;

     return strehlAmplitude;
}


static double
getFluxTotal(double * im, size_t rows, size_t cols)
{
     double totalFlux {0};
     for(size_t i {0}; i < rows; ++i)
     {
          for (size_t j {0}; j < cols; ++j)
              totalFlux += im[i + j * cols]; 
     }

     return totalFlux;
}


static void
imshift(double * im, size_t rows, size_t cols, int xshift, int yshift)
{
     double temp {0};

     switch (yshift > 0)
     {
          case true:
             for (size_t j {0}; j < cols; ++j)
             {
                  for (size_t c {0}; c < yshift; ++c)
                  {
                       temp = im[j + (rows - 1) * cols];
                       for (size_t i {rows-1}; i > 0; --i)
                       {
                            im[j + i * cols] = im[j + (i - 1) * cols];
                       }
                       im[j] = temp;
                  }
             }
             break;

          case false :
             for (size_t j {0}; j < cols; ++j)
             {
                  for (size_t c {0}; c < -yshift; ++c)
                  {
                       temp = im[j];
                       for (size_t i {0}; i < rows - 1; ++i)
                       {
                            im[j + i * cols] = im[j + (i + 1) * cols];
                       }
                       im[j + (rows - 1) * cols] = temp;
                  }
             }
             break;
     }

     switch (xshift > 0)
     {
          case true:
             for (size_t i {0}; i < rows; ++i)
             {
                  for (size_t c {0}; c < xshift; ++c)
                  {
                       temp = im[(cols - 1) + i * cols];
                       for (size_t j {cols - 1}; j > 0; --j)
                       {
                            im[j + i * cols] = im[(j - 1) + i * cols];
                       }
                       im[i * cols] = temp;
                  }
             }
             break;

          case false :
             for (size_t i {0}; i < rows; ++i)
             {
                  for (size_t c {0}; c < -xshift; ++c)
                  {
                       temp = im[i * cols];
                       for (size_t j {0}; j < cols - 1; ++j)
                       {
                            im[j + i * cols] = im[(j + 1) + i * cols];
                       }
                       im[(cols - 1) + i * cols] = temp;
                  }
             }
             break;
     }

}



static void
imshiftMov(double * im, size_t rows, size_t cols, int xshift, int yshift, double * mod)
{
     double temp {0};
     memcpy(mod, im, rows * cols * sizeof(double));

     switch (yshift > 0)
     {
          case true:
             for (size_t j {0}; j < cols; ++j)
             {
                  for (size_t c {0}; c < yshift; ++c)
                  {
                       temp = mod[j + (rows - 1) * cols];
                       for (size_t i {rows-1}; i > 0; --i)
                       {
                            mod[j + i * cols] = mod[j + (i - 1) * cols];
                       }
                       mod[j] = temp;
                  }
             }
             break;

          case false :
             for (size_t j {0}; j < cols; ++j)
             {
                  for (size_t c {0}; c < -yshift; ++c)
                  {
                       temp = mod[j];
                       for (size_t i {0}; i < rows - 1; ++i)
                       {
                            mod[j + i * cols] = mod[j + (i + 1) * cols];
                       }
                       mod[j + (rows - 1) * cols] = temp;
                  }
             }
             break;
     }

     switch (xshift > 0)
     {
          case true:
             for (size_t i {0}; i < rows; ++i)
             {
                  for (size_t c {0}; c < xshift; ++c)
                  {
                       temp = mod[(cols - 1) + i * cols];
                       for (size_t j {cols - 1}; j > 0; --j)
                       {
                            mod[j + i * cols] = mod[(j - 1) + i * cols];
                       }
                       mod[i * cols] = temp;
                  }
             }
             break;

          case false :
             for (size_t i {0}; i < rows; ++i)
             {
                  for (size_t c {0}; c < -xshift; ++c)
                  {
                       temp = mod[i * cols];
                       for (size_t j {0}; j < cols - 1; ++j)
                       {
                            mod[j + i * cols] = mod[(j + 1) + i * cols];
                       }
                       mod[(cols - 1) + i * cols] = temp;
                  }
             }
             break;
     }
}
