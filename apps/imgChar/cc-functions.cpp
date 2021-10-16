#include "cc-functions.hpp"

#define NEW_TEST 1

std::vector<double> 
GaussFit(size_t ROWS,    // number of rows in input 
         size_t COLS,    // number of cols in input 
         double *input,  // input image 
         size_t SZ      // determines size of crop region
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

   std::vector<double> res(5);
   double theta { 0.5 * atan(2 * v[1] / (v[0] - v[2])) };
   double sec   { 1.0 / cos(2 * theta) };

   res[0] = (2*v(0)*v(4) - v(1)*v(3)) / (v(1)*v(1) - 4*v(0)*v(2));    // y-mean
   res[1] = (2*v(2)*v(3) - v(1)*v(4)) / (v(1)*v(1) - 4*v(0)*v(2));    // x-mean
   res[2] = 1.0 / sqrt(-v[0] - v[2] + (v[0] - v[2]) * sec);  // sigma_y
   res[3] = 1.0 / sqrt(-v[0] - v[2] + (v[2] - v[0]) * sec);  // sigma_x
   res[4] = theta; // Tilt angle

   return res;
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
      
      case _DATATYPE_UINT64:
      for (size_t i = 0; i < IMEND; ++i)
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
      fprintf(stderr, "Error: Image datatype not supported.\n");
      exit(EXIT_FAILURE);

   }
}


void copy_image0(double* in, void * image, size_t rows, size_t cols, 
                 int datatype, size_t * xctr, size_t * yctr) {

   xctr[0] = 0;
   yctr[0] = 0;
   double tot {0}; 

   switch (datatype) {
      case _DATATYPE_UINT8:
      for (size_t i = 0; i < rows; ++i)
         for (size_t j = 0; j < cols; ++j)
         {
            in[j + i * cols] = ((uint8_t *) image)[j + i * cols];
            tot += in[j + i * cols];
            xctr[0] += (in[j + i * cols] * j);
            yctr[0] += (in[j + i * cols] * i);
         }
      break; 

      case _DATATYPE_INT8:
      for (size_t i = 0; i < rows; ++i)
         for (size_t j = 0; j < cols; ++j)
         {
            in[j + i * cols] = ((int8_t *) image)[j + i * cols];
            tot += in[j + i * cols];
            xctr[0] += (in[j + i * cols] * j);
            yctr[0] += (in[j + i * cols] * i);
         }
      break; 

      case _DATATYPE_UINT16:
      for (size_t i = 0; i < rows; ++i)
         for (size_t j = 0; j < cols; ++j)
         {
            in[j + i * cols] = ((uint16_t *) image)[j + i * cols];
            tot += in[j + i * cols];
            xctr[0] += (in[j + i * cols] * j);
            yctr[0] += (in[j + i * cols] * i);
         }
      break; 

      case _DATATYPE_INT16:
      for (size_t i = 0; i < rows; ++i)
         for (size_t j = 0; j < cols; ++j)
         {
            in[j + i * cols] = ((int16_t *) image)[j + i * cols];
            tot += in[j + i * cols];
            xctr[0] += (in[j + i * cols] * j);
            yctr[0] += (in[j + i * cols] * i);
         }
      break;

      case _DATATYPE_UINT32: 
      for (size_t i = 0; i < rows; ++i)
         for (size_t j = 0; j < cols; ++j)
         {
            in[j + i * cols] = ((uint32_t *) image)[j + i * cols];
            tot += in[j + i * cols];
            xctr[0] += (in[j + i * cols] * j);
            yctr[0] += (in[j + i * cols] * i);
         }
      break;

      case _DATATYPE_INT32:
      for (size_t i = 0; i < rows; ++i)
         for (size_t j = 0; j < cols; ++j)
         {
            in[j + i * cols] = ((int32_t *) image)[j + i * cols];
            tot += in[j + i * cols];
            xctr[0] += (in[j + i * cols] * j);
            yctr[0] += (in[j + i * cols] * i);
         }
      break;
      
      case _DATATYPE_UINT64:
      for (size_t i = 0; i < rows; ++i)
         for (size_t j = 0; j < cols; ++j)
         {
            in[j + i * cols] = ((uint64_t *) image)[j + i * cols];
            tot += in[j + i * cols];
            xctr[0] += (in[j + i * cols] * j);
            yctr[0] += (in[j + i * cols] * i);
         }
      break;

      case _DATATYPE_INT64:
      for (size_t i = 0; i < rows; ++i)
         for (size_t j = 0; j < cols; ++j)
         {
            in[j + i * cols] = ((int64_t *) image)[j + i * cols];
            tot += in[j + i * cols];
            xctr[0] += (in[j + i * cols] * j);
            yctr[0] += (in[j + i * cols] * i);
         }
      break;

      case _DATATYPE_FLOAT:
      for (size_t i = 0; i < rows; ++i)
         for (size_t j = 0; j < cols; ++j)
         {
            in[j + i * cols] = ((float *) image)[j + i * cols];
            tot += in[j + i * cols];
            xctr[0] += (in[j + i * cols] * j);
            yctr[0] += (in[j + i * cols] * i);
         }
      break;

      case _DATATYPE_DOUBLE:
      for (size_t i = 0; i < rows; ++i)
         for (size_t j = 0; j < cols; ++j)
         {
            in[j + i * cols] = ((double *) image)[j + i * cols];
            tot += in[j + i * cols];
            xctr[0] += (in[j + i * cols] * j);
            yctr[0] += (in[j + i * cols] * i);
         }
      break;     
      
      default:
      fprintf(stderr, "Error: Image datatype not supported.\n");
      exit(EXIT_FAILURE);

   }

   xctr[0] /= tot;
   yctr[0] /= tot;

}


double strehlAmp(void * image, size_t IMEND, int datatype) {

   double maximum   { 0 };
   double totalFlux { 0 };

   switch (datatype) {
      case _DATATYPE_UINT8:
      for (size_t i = 0; i < IMEND; ++i)
      {
         if (maximum < ((uint8_t *) image)[i])
            maximum = ((uint8_t *) image)[i];
         totalFlux += ((uint8_t *) image)[i];
      }
      return maximum;
      break; 

      case _DATATYPE_INT8:
      for (size_t i = 0; i < IMEND; ++i)
      {
         if (maximum < ((int8_t *) image)[i])
            maximum = ((int8_t *) image)[i];
         totalFlux += ((int8_t *) image)[i];
      }
      return maximum;
      break; 

      case _DATATYPE_UINT16:
      for (size_t i = 0; i < IMEND; ++i)
      {
         if (maximum < ((uint16_t *) image)[i])
            maximum = ((uint16_t *) image)[i];
         totalFlux += ((uint16_t *) image)[i];
      }
      return maximum;
      break; 

      case _DATATYPE_INT16:
      for (size_t i = 0; i < IMEND; ++i)
      {
         if (maximum < ((int16_t *) image)[i])
            maximum = ((int16_t *) image)[i];
         totalFlux += ((int16_t *) image)[i];
      }
      return maximum;
      break;

      case _DATATYPE_UINT32: 
      for (size_t i = 0; i < IMEND; ++i)
      {
         if (maximum < ((uint32_t *) image)[i])
            maximum = ((uint32_t *) image)[i];
         totalFlux += ((uint32_t *) image)[i];
      }
      return maximum;
      break;

      case _DATATYPE_INT32:
      for (size_t i = 0; i < IMEND; ++i)
      {
         if (maximum < ((int32_t *) image)[i])
            maximum = ((int32_t *) image)[i];
         totalFlux += ((int32_t *) image)[i];
      }
      return maximum;
      break;
      
      case _DATATYPE_UINT64:
      for (size_t i = 0; i < IMEND; ++i)
      {
         if (maximum < ((uint64_t *) image)[i])
            maximum = ((uint64_t *) image)[i];
         totalFlux += ((uint64_t *) image)[i];
      }
      return maximum;
      break;

      case _DATATYPE_INT64:
      for (size_t i = 0; i < IMEND; ++i)
      {
         if (maximum < ((int64_t *) image)[i])
            maximum = ((int64_t *) image)[i];
         totalFlux += ((int64_t *) image)[i];
      }
      return maximum;
      break;

      case _DATATYPE_FLOAT:
      for (size_t i = 0; i < IMEND; ++i)
      {
         if (maximum < ((float *) image)[i])
            maximum = ((float *) image)[i];
         totalFlux += ((float *) image)[i];
      }
      return maximum;
      break;

      case _DATATYPE_DOUBLE:
      for (size_t i = 0; i < IMEND; ++i)
      {
         if (maximum < ((double *) image)[i])
            maximum = ((double *) image)[i];
         totalFlux += ((double *) image)[i];
      }
      return maximum;
      break;     
      
      default:
      fprintf(stderr, "Error: Image datatype not supported.\n");
      exit(EXIT_FAILURE);

   }
}
