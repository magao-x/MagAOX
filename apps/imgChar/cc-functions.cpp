#include "cc-functions.hpp"

#define NEW_TEST 1

void GaussFit(size_t ROWS,    // number of rows in input 
              size_t COLS,    // number of cols in input 
              double *input,  // input image 
              size_t SZ,      // determines size of crop region
              double *result  // holds results of the fit 
              )
{
   size_t range = ((2 * SZ) + 1);

   Eigen::MatrixXd coeff(6,6);
   Eigen::VectorXd b(6);
   int x[range];
   int y[range];
   int di {0}, dj {0};
   double *image2 = (double *) malloc((2*SZ+1)*(2*SZ+1)*sizeof(double));
    
   // initialize matrix 
   for (size_t i {0}; i < 6; ++i)
      for (size_t j {0}; j < 6; ++j)
         coeff(i,j) = 0;

   for (size_t i {0}; i < 6; ++i) { b(i) = 0; }
 
   locate_max(ROWS, COLS, input, &di, &dj);
   // crop (2*SZ+1)x(2*SZ+1) into image
   crop_image(image2, input, ROWS, COLS, dj, di, SZ);

   // linearize the data in the (2*SZ+1)x(2*SZ+1) grid
   for (size_t i {0}; i < range; ++i)
      for (size_t j {0}; j < range; ++j)
         image2[j + i * range] = log(image2[j+i*range]/(ROWS*COLS));

   for (size_t i {0}; i < range; ++i) {
      x[i] = dj-SZ+i;
      y[i] = di-SZ+i;
   } 


   /* Setup the least-square regression matrix */

   // Calculate the single-sum elements
   double tx, ty;
   coeff(5,5) = range * range;

   for (size_t a {0}; a < range; ++a) {
      tx = x[a] * x[a];
      ty = y[a] * y[a];
      coeff(0,0) += tx *tx;
      coeff(2,2) += ty * ty;
      coeff(3,3) += tx;
      coeff(4,4) += ty;
      coeff(4,5) += y[a];
      coeff(3,5) += x[a];
      coeff(0,3) += tx * x[a];
      coeff(2,4) += ty * y[a];
   }


      coeff(0,0) = range * coeff(0,0);
      coeff(2,2) = range * coeff(2,2);
      coeff(3,3) = range * coeff(3,3);
      coeff(4,4) = range * coeff(4,4);
      coeff(4,5) = range * coeff(4,5);
      coeff(3,5) = range * coeff(3,5);
      coeff(0,3) = range * coeff(0,3);
      coeff(2,4) = range * coeff(2,4);

   // Use redundancy in upper triangle
   coeff(0,5) = coeff(3,3);
   coeff(2,5) = coeff(4,4);

   // Calculate the remaining diagonal and upper triangular terms
   double txy;
   for (size_t i {0}; i < range; ++i)
      for (size_t j {0}; j < range; ++j) {
         tx  = x[j] * x[j];
         ty  = y[i] * y[i];
         txy = x[j] * y[i];
         coeff(0,1) += tx * txy;
         coeff(1,1) += tx * ty;
         coeff(1,2) += txy * ty;
         coeff(1,3) += tx * y[i];
         coeff(2,3) += x[j] * ty;
         coeff(3,4) += txy;

         b(0) +=   tx * image2[j + i * range];
         b(1) +=  txy * image2[j + i * range];
         b(2) +=   ty * image2[j + i * range];
         b(3) += x[j] * image2[j + i * range];
         b(4) += y[i] * image2[j + i * range];
         b(5) +=        image2[j + i * range];
      }

   // Use redundancy of upper triangle
   coeff(0,2) = coeff(1,1);
   coeff(0,4) = coeff(1,3);
   coeff(1,4) = coeff(2,3);
   coeff(1,5) = coeff(3,4);

   // Use matrix symmetry
   coeff(1,0) = coeff(0,1);
   coeff(2,0) = coeff(0,2);
   coeff(3,0) = coeff(0,3);
   coeff(4,0) = coeff(0,4);
   coeff(5,0) = coeff(0,5);

   coeff(2,1) = coeff(1,2);
   coeff(3,1) = coeff(1,3);
   coeff(4,1) = coeff(1,4);
   coeff(5,1) = coeff(1,5);

   coeff(3,2) = coeff(2,3);
   coeff(4,2) = coeff(2,4);
   coeff(5,2) = coeff(2,5);

   coeff(4,3) = coeff(3,4);
   coeff(5,3) = coeff(3,5);

   coeff(5,4) = coeff(4,5);   

   // solve system with Eigen tools; calculate shifts
   Eigen::VectorXd v(6);
   v = coeff.fullPivHouseholderQr().solve(b);                    
   result[0] = (v(0)*v(4) - 2*v(1)*v(3)) / (v(1)*v(1) - 4*v(0)*v(2)); // y-component
   result[1] = (2*v(2)*v(3) - v(1)*v(4)) / (v(1)*v(1) - 4*v(0)*v(2)); // x-component
}




double getStrehlMod(double *detector, size_t ROWS, size_t COLS, size_t xctr, size_t yctr)
{
        size_t numBytesD = sizeof(double);
     // Setup arrays to track maximum image values
        size_t numlines = 256;
        double *heights = (double *)malloc(numBytesD * numlines) ; 
        double *search_angles = (double *)malloc(numBytesD * numlines) ;

        for (size_t i {0}; i < numlines; ++i)
          {
            search_angles[i] = (2.0 * M_PI * i) / numlines;
          }
    
     // Launch rays to get maximum points for Airy peaks
        __m128i xtrack = _mm_set1_epi32(xctr) ;     // Intrinsics allow us to
        __m128i ytrack = _mm_set1_epi32(yctr) ;     // calculate four max
        __m256d amplitude = _mm256_setzero_pd() ;   // amplitudes at once

        __m128 iteratorC = _mm_set1_ps(0.0) ;
        __m128 iteratorS = _mm_set1_ps(0.0) ;
        __m128 cos_angle, sin_angle;
        __m256d comp;  
     
        __m128i zeros = _mm_set1_epi32(0) ;
        __m128i maskX1, maskX2, maskY1, maskY2;
        int *vx = (int *) &xtrack;
        int *vy = (int *) &ytrack;
        double *h = (double *) &amplitude;

        for (size_t bin = 0; bin < numlines; bin += 4) // need numlines % 4 == 0
          { 
            cos_angle = _mm256_cvtpd_ps(
                                     _mm256_set_pd(cos(search_angles[bin]), 
                                                   cos(search_angles[bin+1]),  
                                                   cos(search_angles[bin+2]), 
                                                   cos(search_angles[bin+3]))
                                        ) ;

            sin_angle = _mm256_cvtpd_ps(
                                     _mm256_set_pd(sin(search_angles[bin]),
                                                   sin(search_angles[bin+1]), 
                                                   sin(search_angles[bin+2]),
                                                   sin(search_angles[bin+3]))
                                       ) ;
            CONTINUE_SEARCH:
                // Mask the break condition for each channel to determine
                // when to end the "while" loop 
                maskX1 = _mm_cmpgt_epi32(_mm_set1_epi32(COLS-1), xtrack) ;
                maskX2 = _mm_cmpgt_epi32(xtrack, zeros) ;
                maskY1 = _mm_cmpgt_epi32(_mm_set1_epi32(ROWS-1), ytrack) ;
                maskY2 = _mm_cmpgt_epi32(ytrack, zeros) ;

                maskX1 = _mm_and_si128(maskX1, maskX2) ;
                maskY1 = _mm_and_si128(maskY1, maskY2) ;
                maskX1 = _mm_and_si128(maskX1, maskY1) ;
                
                if (_mm_movemask_ps(_mm_castsi128_ps(maskX1)) == 0) {
                    goto EXIT_SEARCH;                               
                }
                comp = _mm256_set_pd(detector[vx[3] + (vy[3] * COLS)], 
                                     detector[vx[2] + (vy[2] * COLS)],
                                     detector[vx[1] + (vy[1] * COLS)], 
                                     detector[vx[0] + (vy[0] * COLS)]
                            ) ;  
                amplitude = _mm256_max_pd(comp, amplitude) ;
                 
                iteratorC = _mm_and_ps(_mm_castsi128_ps(maskX1), cos_angle) ;
                iteratorS = _mm_and_ps(_mm_castsi128_ps(maskX1), sin_angle) ;

                xtrack = _mm_add_epi32(xtrack, _mm_cvtps_epi32(iteratorC)) ;
                ytrack = _mm_add_epi32(ytrack, _mm_cvtps_epi32(iteratorS)) ;
                goto CONTINUE_SEARCH;

            EXIT_SEARCH: 
            heights[bin] = h[0];
            heights[bin+1] = h[1];
            heights[bin+2] = h[2];
            heights[bin+3] = h[3];

            amplitude = _mm256_setzero_pd() ;
            xtrack = _mm_set1_epi32(xctr) ;
            ytrack = _mm_set1_epi32(yctr) ;
          } 

     // Perform averaging and get Strehl ratio
        double strehl = 0;
        for (size_t i=0; i < numlines; ++i)
          {
            strehl += heights[i]; 
          }
        strehl /= numlines;

        double flux = 0;
        for (size_t i = 0; i < ROWS * COLS; ++i)
          {
            flux += detector[i];
          }

        strehl /= flux;   // strehl = pk_flux / total_flux
        strehl /= 1;//fPSF;

        free(heights);
        free(search_angles); 
 
        return strehl;
}



void copy_image(double* in, void * image, size_t rows, size_t cols, int datatype) {

   size_t IMEND = rows * cols; 

   switch (datatype) {
      case _DATATYPE_UINT8:
      for (size_t i = 0; i < IMEND; ++i)
         in[i] = ((uint8_t *) image)[i];
      break; 

      case _DATATYPE_INT8:
      for (size_t i = 0; i < IMEND; ++i)
         in[i] = ((int8_t *) image)[i];
      break; 

      case _DATATYPE_UINT16:
      for (size_t i = 0; i < IMEND; ++i)
         in[i] = ((uint16_t *) image)[i];
      break; 

      case _DATATYPE_INT16:
      for (size_t i = 0; i < IMEND; ++i)
         in[i] = ((int16_t *) image)[i];
      break;

      case _DATATYPE_UINT32: 
      for (size_t i = 0; i < IMEND; ++i)
         in[i] = ((uint32_t *) image)[i];
      break;

      case _DATATYPE_INT32:
      for (size_t i = 0; i < IMEND; ++i)
         in[i] = ((int32_t *) image)[i];
      break;
      
      case _DATATYPE_UINT64:                                                                                                                                                                                 for (size_t i = 0; i < IMEND; ++i)
         in[i] = ((uint64_t *) image)[i];
      break;

      case _DATATYPE_INT64:
      for (size_t i = 0; i < IMEND; ++i)
         in[i] = ((int64_t *) image)[i];
      break;

      case _DATATYPE_FLOAT:
      for (size_t i = 0; i < IMEND; ++i)
         in[i] = ((float *) image)[i];
      break;

      case _DATATYPE_DOUBLE:
      for (size_t i = 0; i < IMEND; ++i)
         in[i] = ((double *) image)[i];
      break;     
      
      default:
      fprintf(stderr, "Error: Image type not supported.\n");
      exit(EXIT_FAILURE);

   }
}
