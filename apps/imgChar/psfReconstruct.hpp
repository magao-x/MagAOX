// psfReconstruct.hpp
// Jacob Trzaska
// 2021-10-15

#pragma once

// C++ Libraries
#include <complex>
#include <cstring>
#include <Eigen/Dense>

// C Libraries
#include <fftw3.h>



/// Calculate the Strehl amplitude of a given Point Spread Function
/**
 *
 *
 *
 */
double strehlAmp(double * im, size_t rows, size_t cols)
{
     double totalFlux { 0 };
     double max { im[0] };

     for (size_t i {0}; i < rows; ++i)
          for (size_t j {0}; j < cols; ++j)
          {
               if (im[j + i * cols] > max) {
                    max = im[j + i * cols];
               }

               totalFlux += im[j + i * cols];
          }

     return max / totalFlux;
}


/// Create a spatial representation of the the modulation mask
/**
 *
 *
 *
 */
void 
createModMask(double * mask, int64_t nx, int64_t ny, double R)
{
     memset(mask, 0, nx * ny * sizeof(double));

     int64_t idxX { (int64_t) R };
     int64_t idxY { 0 };
     double  ang0 { 0 };
     double  ang1 { 0 };
     double  dphi { 0 };

     while (idxX >= 0) {
          if ( (idxX * idxX + (idxY+1) * (idxY+1)) < R * R) {
               ang1 = asin((idxY+1) / R);
               dphi = ang1 - ang0;
               mask[idxX + idxY * nx] = dphi;
               mask[(nx-idxX-1) + idxY * nx] = dphi;
               mask[idxX + (ny-idxY-1) * nx] = dphi;
               mask[(nx-idxX-1) + (ny-idxY-1) * nx] = dphi;
               idxY += 1;
          }
          else if ( (idxX * idxX + (idxY+1) * (idxY+1)) > R * R) {
               ang1 = acos(idxX / R);
               dphi = ang1 - ang0;
               mask[idxX + idxY * nx] = dphi;
               mask[(nx-idxX-1) + idxY * nx] = dphi;
               mask[idxX + (ny-idxY-1) * nx] = dphi;
               mask[(nx-idxX-1) + (ny-idxY-1) * nx] = dphi;
               idxX -= 1;
          }
          else {
               ang1 = acos(idxX / R);
               dphi = ang1 - ang0;
               mask[idxX + idxY * nx] = dphi;
               mask[(nx-idxX-1) + idxY * nx] = dphi;
               mask[idxX + (ny-idxY-1) * nx] = dphi;
               mask[(nx-idxX-1) + (ny-idxY-1) * nx] = dphi;
               idxX -= 1;
               idxY += 1;
          }

          ang0 = ang1;
     }
}



/// Create a spatial representation of the modulation mask
/*
 *
 *
 *
 */
double * 
createModMask(int64_t nx, int64_t ny, double R)
{
     double * mask { new double[nx * ny] };
     memset(mask, 0, nx * ny * sizeof(double));

     int64_t idxX { (int64_t) R };
     int64_t idxY { 0 };
     double  ang0 { 0 };
     double  ang1 { 0 };
     double  dphi { 0 };

     while (idxX >= 0) {
          if ( (idxX * idxX + (idxY+1) * (idxY+1)) < R * R) {
               ang1 = asin((idxY+1) / R);
               dphi = ang1 - ang0;
               mask[idxX + idxY * nx] = dphi;
               mask[(nx-idxX-1) + idxY * nx] = dphi;
               mask[idxX + (ny-idxY-1) * nx] = dphi;
               mask[(nx-idxX-1) + (ny-idxY-1) * nx] = dphi;
               idxY += 1;
          }
          else if ( (idxX * idxX + (idxY+1) * (idxY+1)) > R * R) {
               ang1 = acos(idxX / R);
               dphi = ang1 - ang0;
               mask[idxX + idxY * nx] = dphi;
               mask[(nx-idxX-1) + idxY * nx] = dphi;
               mask[idxX + (ny-idxY-1) * nx] = dphi;
               mask[(nx-idxX-1) + (ny-idxY-1) * nx] = dphi;
               idxX -= 1;
          }
          else {
               ang1 = acos(idxX / R);
               dphi = ang1 - ang0;
               mask[idxX + idxY * nx] = dphi;
               mask[(nx-idxX-1) + idxY * nx] = dphi;
               mask[idxX + (ny-idxY-1) * nx] = dphi;
               mask[(nx-idxX-1) + (ny-idxY-1) * nx] = dphi;
               idxX -= 1;
               idxY += 1;
          }

          ang0 = ang1;
     }

     return mask;
}


/// Apply the Weiner filter the Optical Transfer Function of interest
/**
 * Assumes white gauassian noise. 
 *
 */
void applyWienerFilter(fftw_complex * H, fftw_complex * OTF, uint64_t nx, uint64_t ny)
{
     double absH { 0 };
     double absO { 0 };
     
     double noiseFloor { 0 };
     for (size_t i { ny/2 - 4 }; i <= ny/2 + 4; ++i)
          for (size_t j { nx/2 -7 }; j < nx/2 + 1; ++j) {
               noiseFloor += ( (OTF[j + i * (nx/2 + 1)])[0] * (OTF[j + i * (nx/2 + 1)])[0] 
                           +   (OTF[j + i * (nx/2 + 1)])[1] * (OTF[j + i * (nx/2 + 1)])[1] );
          }
     
     noiseFloor /= (8 * 7);
     
     for (size_t i { 0 }; i < ny; ++i)
          for (size_t j { 0 }; j < nx/2 + 1; ++j)
          {
               absH = (OTF[j + i * (nx/2 + 1)])[0] * (OTF[j + i * (nx/2 + 1)])[0] 
                    + (OTF[j + i * (nx/2 + 1)])[1] * (OTF[j + i * (nx/2 + 1)])[1]; 
               absO = ( (OTF[j + i * (nx/2 + 1)])[0] == 0.0 && (OTF[j + i * (nx/2 + 1)])[1] == 0.0) ? 1e-100 : absH;

               (OTF[j + i * (nx/2 + 1)])[0] *= (  (H[j + i * (nx/2 + 1)])[0] / (absH * absH + noiseFloor / (absO * absO) ) );
               (OTF[j + i * (nx/2 + 1)])[1] *= ( -(H[j + i * (nx/2 + 1)])[1] / (absH * absH + noiseFloor / (absO * absO) ) );
          }
}

/// Mask out high frequencies in the provided Optical Transfer Function
/**
 *
 *
 * FCO is the index of the first pixel to be masked out
 */
void 
maskHighFreq(fftw_complex * otf, size_t nx, size_t ny, size_t FCO)
{
     size_t colSz { nx/2 + 1};
     size_t CCO   { 0 };

     // Handle the 0 - index row
     memset(otf + FCO, 0.0, sizeof(fftw_complex) * (colSz-FCO) );

     // Handle the remaining rows
     for (size_t i { 1 }; i < FCO; ++i)
     {
          CCO = (size_t) sqrt(FCO * FCO - i * i);
          ++CCO;

          memset(otf + CCO + colSz * i, 0.0,
                    sizeof(fftw_complex) * (colSz - CCO) );
          memset(otf + CCO + colSz * (ny - i) , 0.0, 
                    sizeof(fftw_complex) * (colSz - CCO) );
     }

     // Zero the center of the OTF
     memset(otf + colSz * FCO, 0.0, sizeof(fftw_complex) * colSz * (ny - FCO - FCO - 1));
}



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



double locate_max(size_t ROWS, size_t COLS, double *mat, size_t *delta_i, size_t *delta_j)
{
    double max = fabs(mat[0]);

    for ( int i=0; i<ROWS; i++ )
        for ( int j=0; j<COLS; j++ )
          if ( fabs(mat[j+i*COLS]) > max )
          {
              *delta_i=i; *delta_j=j; max = fabs(mat[j+i*COLS]);
          }

     return max;
}



std::vector<double> psfGaussFit(size_t ROWS,     // number of rows in input 
                                size_t COLS,     // number of cols in input 
                                double * input   // input image 
                               )
{
   /* Initialize Eigen matrix */
   Eigen::MatrixXd coeff(6,6);
   Eigen::VectorXd b(6);
   
   for (size_t i {0}; i < 6; ++i)
      for (size_t j {0}; j < 6; ++j)
         coeff(i,j) = 0;

   for (size_t i {0}; i < 6; ++i)
   { b(i) = 0; }
 
   /* Setup the linearized image */
   std::vector<size_t> xpos;
   std::vector<size_t> ypos;
   size_t ymax {0}, xmax {0};
   std::vector<double> image2;   
 
   double max { locate_max(ROWS, COLS, input, &ymax, &xmax) };

   // Find the radius needed for the fit
   double n { 3.5 }; // adjustable parameter
   size_t xrad { 0 }, yrad { 0 }, rad { 0 };

   for (size_t y { ymax }; y < ROWS; ++y)
      if (input[xmax + y * COLS] <= max / n) {
         yrad = (y - ymax < 0) ? (ymax - y)
                               : (y - ymax);
         break;
      } 

   for (size_t x { xmax }; x < COLS; ++x)
      if (input[x + ymax * COLS] <= max / n) {
         xrad = (x - xmax < 0) ? (xmax - x)
                               : (x - xmax);
         break;
      } 

   rad = (xrad > yrad) ? xrad : yrad;

   // Crop out the fit region
   for (size_t y { ymax - rad }; y <= (ymax + rad); ++y)
      for (size_t x { xmax - rad }; x <= (xmax + rad); ++x) {
         if ( (x-xmax)*(x-xmax) + (y-ymax)*(y-ymax) <= rad*rad ) {
            image2.push_back(input[x + y * COLS]);
            xpos.push_back(x); 
            ypos.push_back(y);
         }
      }

   // linearize the data in the grid
   for (auto & v : image2)
      v = std::log(v); 


   /* Setup the least-square regression matrix */

   // Calculate the single-sum elements
   double tx, ty;

   for (size_t a {0}; a < image2.size(); ++a) {
      tx = xpos[a] * xpos[a];
      ty = ypos[a] * ypos[a];
      coeff(0,0) += tx *tx;
      coeff(2,2) += ty * ty;
      coeff(3,3) += tx;
      coeff(4,4) += ty;
      coeff(4,5) += ypos[a];
      coeff(3,5) += xpos[a];
      coeff(0,3) += tx * xpos[a];
      coeff(2,4) += ty * ypos[a];
      coeff(5,5) += 1.0;
   }

   // Use redundancy in upper triangle
   coeff(0,5) = coeff(3,3);
   coeff(2,5) = coeff(4,4);

   // Calculate the remaining diagonal and upper triangular terms
   double txy;
   for (size_t a {0}; a < image2.size(); ++a) {
      tx  = xpos[a] * xpos[a];
      ty  = ypos[a] * ypos[a];
      txy = xpos[a] * ypos[a];

      coeff(0,1) += tx      * txy;
      coeff(1,1) += tx      * ty;
      coeff(1,2) += txy     * ty;
      coeff(1,3) += tx      * ypos[a];
      coeff(2,3) += xpos[a] * ty;
      coeff(3,4) += txy;

      b(0) +=   tx    * image2[a];
      b(1) +=  txy    * image2[a];
      b(2) +=   ty    * image2[a];
      b(3) += xpos[a] * image2[a];
      b(4) += ypos[a] * image2[a];
      b(5) +=           image2[a];
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
   double theta { 0.5 * atan(v[1] / (v[0] - v[2])) };

   if (v[0] == v[2])
   {
     res[0] = (2*v(0)*v(4) - v(1)*v(3)) / (v(1)*v(1) - 4*v(0)*v(2));    // y-mean
     res[1] = (2*v(2)*v(3) - v(1)*v(4)) / (v(1)*v(1) - 4*v(0)*v(2));    // x-mean
     res[2] = 1.0 / sqrt( -2 * v[2]);                                   // sigma_y
     res[3] = 1.0 / sqrt( -2 * v[0]);                                   // sigma_x
     res[4] = 0.0;                                                      // Tilt angle
   }
   else
   {
     res[0] = (2*v(0)*v(4) - v(1)*v(3)) / (v(1)*v(1) - 4*v(0)*v(2));                                           // y-mean
     res[1] = (2*v(2)*v(3) - v(1)*v(4)) / (v(1)*v(1) - 4*v(0)*v(2));                                           // x-mean
     res[2] = sqrt( 0.5 * cos(2*theta) / (v[0] * sin(theta) * sin(theta) - v[2] * cos(theta) * cos(theta)) );  // sigma_y
     res[3] = sqrt( 0.5 * cos(2*theta) / (v[2] * sin(theta) * sin(theta) - v[0] * cos(theta) * cos(theta)) );  // sigma_x
     res[4] = theta;                                                                                           // Tilt angle
   }

     std::cout << "fwhm_x = " << 2*sqrt(2*std::log(2)) / sqrt( cos(theta) * cos(theta) / (res[2]*res[2]) + sin(theta) * sin(theta) / (res[3]*res[3])) << "\n";

   return res;
}
