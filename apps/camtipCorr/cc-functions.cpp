#include "cc-functions.hpp"


void GaussFit(size_t ROWS,    // number of rows in input 
              size_t COLS,    // number of cols in input 
              double *input,  // input image 
              size_t SZ,      // determines size of crop region
              double *result  // holds results of the fit 
              )
{
        Eigen::MatrixXd coeff(4,4);
        Eigen::VectorXd b(4);
        int x[2*SZ+1];
        int y[2*SZ+1];
        int di=0, dj=0;
        double *image2 = (double *) malloc((2*SZ+1)*(2*SZ+1)*sizeof(double));
    
        // initialize matrix 
        for (size_t i=0; i<4; i++)
                for (size_t j=0; j<4; j++)
                        coeff(i,j) = 0;

        for (size_t i=0; i<4; i++) { b(i)=0; }
 
        locate_max(ROWS, COLS, input, &di, &dj);
        // crop (2*SZ+1)x(2*SZ+1) into image
        crop_image(image2, input, ROWS, COLS, dj, di, SZ);

        // linearize the data in the (2*SZ+1)x(2*SZ+1) grid
        for (size_t i=0; i<2*SZ+1; i++)
                for (size_t j=0; j<2*SZ+1; j++)
                        image2[j+i*(2*SZ+1)] 
                                = log(image2[j+i*(2*SZ+1)]/(ROWS*COLS));

        for (size_t i=0; i<2*SZ+1; i++) {
                x[i] = dj-SZ+i;
                y[i] = di-SZ+i;
        } 


        // setup least squares matrix and vector
        coeff(0,0) = (2*SZ+1) * (2*SZ+1);

        for (size_t a=0; a<(2*SZ+1); a++){
                coeff(0,1) += x[a];
                coeff(0,2) += y[a];
                coeff(1,1) += x[a]*x[a];
                coeff(2,2) += y[a]*y[a];
        }

        coeff(0,1) = (2*SZ+1)*coeff(0,1);
        coeff(0,2) = (2*SZ+1)*coeff(0,2);
        coeff(1,1) = (2*SZ+1)*coeff(1,1);
        coeff(2,2) = (2*SZ+1)*coeff(2,2);

        coeff(0,3) = (coeff(1,1) + coeff(2,2));

        double t;
        for (size_t i=0; i<(2*SZ+1); i++)
                for (size_t j=0; j<(2*SZ+1); j++){
                        coeff(1,2) += x[j]*y[i];
                        t = x[j]*x[j]+y[i]*y[i];
                        coeff(1,3) += t*x[j];
                        coeff(2,3) += t*y[i];
                        coeff(3,3) += t*t;

                        b(0) += image2[j+i*(2*SZ+1)];
                        b(1) += x[j]*image2[j+i*(2*SZ+1)];
                        b(2) += y[i]*image2[j+i*(2*SZ+1)];
                        b(3) += t*image2[j+i*(2*SZ+1)];
                }

        // exploit matrix symmetry
        coeff(1,0)=coeff(0,1); coeff(2,0)=coeff(0,2); coeff(3,0)=coeff(0,3);
        coeff(2,1)=coeff(1,2); coeff(3,1)=coeff(1,3); coeff(3,2)=coeff(2,3);

        // solve system with Eigen tools; calculate shifts
        Eigen::VectorXd v(4);
        v = coeff.fullPivHouseholderQr().solve(b);                    
        result[0] = -0.5*v(2)/v(3); // y-component
        result[1] = -0.5*v(1)/v(3); // x-component
}




double getStrehlMod(double *detector, int ROWS, int COLS, int xctr, int yctr)
{
        size_t numBytesD = sizeof(double);
     // Setup arrays to track maximum image values
        size_t numlines = 256;
        double *heights = (double *)malloc(numBytesD * numlines) ; 
        double *search_angles = (double *)malloc(numBytesD * numlines) ;

        for (int i = 0; i < numlines; ++i)
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
        __m128 ones = _mm_set1_ps(1.0) ;
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



// static void center_estimate(IMAGE newImage, int* &xtr, int* &yctr);

void copy_image(double* in, IMAGE* image) {

   size_t IMEND = image->md[0].size[0] * image->md[0].size[1]; 

   switch (image->md[0].datatype) {
      case _DATATYPE_UINT8:
      for (size_t i = 0; i < IMEND; ++i)
         in[i] = image->array.UI8[i];
      break; 

      case _DATATYPE_INT8:
      for (size_t i = 0; i < IMEND; ++i)
         in[i] = image->array.SI8[i];
      break; 

      case _DATATYPE_UINT16:
      for (size_t i = 0; i < IMEND; ++i)
         in[i] = image->array.UI16[i];
      break; 

      case _DATATYPE_INT16:
      for (size_t i = 0; i < IMEND; ++i)
         in[i] = image->array.SI16[i];
      break;

      case _DATATYPE_UINT32: 
      for (size_t i = 0; i < IMEND; ++i)
         in[i] = image->array.UI32[i];
      break;

      case _DATATYPE_INT32:
      for (size_t i = 0; i < IMEND; ++i)
         in[i] = image->array.SI32[i];
      break;
      
      case _DATATYPE_UINT64:                                                                                                                                                                                 for (size_t i = 0; i < IMEND; ++i)
         in[i] = image->array.UI64[i];
      break;

      case _DATATYPE_INT64:
      for (size_t i = 0; i < IMEND; ++i)
         in[i] = image->array.SI64[i];
      break;

      case _DATATYPE_FLOAT:
      for (size_t i = 0; i < IMEND; ++i)
         in[i] = image->array.F[i];
      break;

      case _DATATYPE_DOUBLE:
      for (size_t i = 0; i < IMEND; ++i)
         in[i] = image->array.D[i];
      break;     
      
      default:
      fprintf(stderr, "Error: Image type not supported.\n");
      exit(EXIT_FAILURE);

   }
}
