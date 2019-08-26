/** \file pupilFitter.hpp
  * \brief The MagAO-X Pyramid Pupil Fitter class header
  *
  * \ingroup pupilFit_files
  */

#ifndef pupilFitter_hpp
#define pupilFitter_hpp

#include <mx/improc/eigenImage.hpp>


namespace MagAOX
{
namespace app
{
   
/// Struct to perform centration and measure diameter of Pyramid pupils
template<typename realT>
struct pupilFitter
{
   mx::improc::eigenImage<realT> m_quad;
   mx::improc::eigenImage<realT> m_circ;
   
   unsigned m_rows {0}; ///< [in] the size of a quad, in rows
   unsigned m_cols {0}; ///< [in] the size of a quad, in cols
   
   float m_med[4] = {0,0,0,0};
   float m_avgx[4] = {0,0,0,0};
   float m_avgy[4] = {0,0,0,0};
   float m_avgr[4] = {0,0,0,0};
   
   float m_thresh {0.5};
   
   std::vector<realT> m_pixs;
   
   pupilFitter() {}
   
   int setSize( unsigned rows, ///< [in] the new size of a quad, in rows
                unsigned cols  ///< [in] the new size of a quad, in cols
              );
   
   int threshold( mx::improc::eigenImage<realT> & im );
   
   int getQuad( mx::improc::eigenImage<realT> & quad,
                mx::improc::eigenImage<realT> & im,
                int quadNo
              );
   
   int putQuad( mx::improc::eigenImage<realT> & im,
                mx::improc::eigenImage<realT> & quad,
                int quadNo
              );
   
   int outerpix( float & avgx,
                 float & avgy,
                 float & avgr,
                 int quadNo
               );
   
   int fit( mx::improc::eigenImage<realT> & im, 
            mx::improc::eigenImage<realT> & edged
          );
   
   //int emitRegion( const std::string fname );
   
};

template<typename realT>
int pupilFitter<realT>::setSize( unsigned rows,
                                 unsigned cols
                               )
{
   m_rows = rows;
   m_cols = cols;
   
   m_quad.resize(m_rows,m_cols);
   m_circ.resize(m_rows,m_cols);
   
   m_pixs.resize(m_rows*m_cols);
   
   return 0;
}
   
template<typename realT>
int pupilFitter<realT>::threshold( mx::improc::eigenImage<realT> & im )

{
   for(size_t i =0; i< (size_t) im.rows(); ++i)
   {
      for(size_t j=0; j< (size_t) im.cols(); ++j)
      {
         if(im(i,j) >= m_thresh) im(i,j) = 1;
         else im(i,j) = 0;
      }
   }
   
   return 0;
}

template<typename realT>
int pupilFitter<realT>::getQuad( mx::improc::eigenImage<realT> & quad,
                              mx::improc::eigenImage<realT> & im,
                              int quadNo
                            )
{
   
   if(im.rows() != 2*m_rows || im.cols() != 2*m_cols)
   {
      return -1;
   }
   
   size_t i0=0, j0=0;
   
   if(quadNo == 1) i0 = m_rows;
   if(quadNo == 2) j0 = m_cols;
   if(quadNo == 3) 
   {
      i0 = m_rows;
      j0 = m_cols;
   }
   
   for(size_t i =i0; i< i0+m_rows; ++i)
   {
      for(size_t j=j0; j<j0+m_cols;++j)
      {
         quad(i-i0, j-j0) = im(i,j);
         
      }
   }
   
   return 0;
}

template<typename realT>
int pupilFitter<realT>::putQuad( mx::improc::eigenImage<realT> & im,
                              mx::improc::eigenImage<realT> & quad,
                              int quadNo
                            )
{
   if(im.rows() != 2*m_rows || im.cols() != 2*m_cols)
   {
      return -1;
   }
   
   size_t i0=0, j0=0;
   
   if(quadNo == 1) i0 = m_rows;
   if(quadNo == 2) j0 = m_cols;
   if(quadNo == 3) 
   {
      i0 = m_rows;
      j0 = m_cols;
   }
   
   for(size_t i =i0; i< i0+m_rows; ++i)
   {
      for(size_t j=j0; j<j0+m_cols;++j)
      {
         im(i,j) = quad(i-i0, j-j0);;
      }
   }
   
   return 0;
}

template<typename realT>
int pupilFitter<realT>::outerpix( float & avgx,
                               float & avgy,
                               float & avgr,
                               int quadNo
                             )
{
   
   m_circ.setZero();

   /* bottom half*/
   for(size_t i =0; i< m_rows; ++i)
   {
      bool found = false;
      
      size_t j0 = 0.5*m_cols;
      if(i > .25*m_cols && i < .75*m_cols) j0 += .25*m_cols;
      if( m_quad(i,(int) j0) == 0) found = true;
      
      for(size_t j= j0+1; j< m_cols;++j)
      {

         if(m_quad(i,j) == 0 && !found)
         {
            m_circ(i , (j-1)) = 1;
            found = true;
         }
         if(found) break;
               
      }
      if(found == false)
      {
         m_circ(i, m_cols-1) = 1;
      }
      
      found = false;
      for(size_t j= 0; j < 0.5*m_cols;++j)
      {
         if(m_quad(i,j) == 1 && !found)
         {
            m_circ(i,j) = 1;
            found = true;
         }
      }
   }
   
   /* left half */
   for(size_t j =0; j< m_cols; ++j)
   {
      bool found = false;
      
      size_t i0 = 0.5*m_rows;
      
      if(j > .25*m_cols && j < .75*m_cols) i0 += .25*m_rows;
      
      if(m_quad((int) i0,j) == 0) found = true;
      
      for(size_t i= i0; i< m_rows;++i)
      {
         if(found) break;
         if(m_quad(i,j) == 0 && !found)
         {
            m_circ(i-1,j) = 1;
            found = true;
         }
      }
            
      found = false;
      for(size_t i= 0; i < 0.5*m_rows;++i)
      {
         if(m_quad(i,j) == 1 && !found)
         {
            m_circ(i,j) = 1;
            found = true;
         }
         if(found) break;         
      }
   }/**/
   
   int npix = 0;
   avgx = 0;
   avgy = 0;
   
   for(size_t i = 0; i < m_rows; ++i)
   {
      for(size_t j = 0; j < (size_t) m_cols; ++j)
      {
         if(m_circ(i,j) == 1)
         {
            ++npix;
            avgx += i;
            avgy += j;
         }
      }
   }
      
   avgx /= npix;
   avgy /= npix;

   avgr = 0;
   
   for(size_t i = 0; i < (size_t) m_circ.rows(); ++i)
   {
      for(size_t j = 0; j < (size_t) m_circ.cols(); ++j)
      {
         if(m_circ(i,j) == 1)
         {
            avgr += sqrt( pow(i-avgx,2) + pow(j-avgy,2)) + 0.055;
         }
      }
   }

   avgr /= npix;
   
   size_t i0=0, j0=0;
   
   if(quadNo == 1) i0 = m_rows;
   if(quadNo == 2) j0 = m_cols;
   if(quadNo == 3) 
   {
      i0 = m_rows;
      j0 = m_cols;
   }
   
   avgx += i0;
   avgy += j0;
      
   return 0;
}

template<typename realT>
int pupilFitter<realT>::fit( mx::improc::eigenImage<realT> & im, 
                             mx::improc::eigenImage<realT> & edged
                           )
{
   
   // 1) normalize by median of pupil
   for(int i=0; i< 4; ++i)
   {
      getQuad(m_quad, im, i);
      
      for(size_t j=0; j < m_rows*m_cols; ++j) m_pixs[j] = m_quad.data()[j];
      
      ///\todo this should use std::nth
      std::sort(m_pixs.begin(), m_pixs.end());

      m_med[i] = m_pixs[0.6266*m_rows*m_cols]; //This should be the median of the pupils if the right size and contained in the quad
      
      m_quad/=m_med[i]; 
      putQuad(im, m_quad, i);
   }
   
   // 2) apply the threshold, after which the image is 1s or 0s.
   threshold(im);

   // 3) for each quad, find the outer-most 1 pixel in each direction.
   for(int i=0; i< 4; ++i)
   {
      getQuad(m_quad, im, i);
      outerpix(m_avgx[i], m_avgy[i], m_avgr[i], i);
   
      putQuad(edged, m_circ, i);
   }
   return 0;
}
   
/*
int pupFitter::emitRegion( const std::string fname )
{
   
   std::ofstream fout;
   
   fout.open(fname);
   
   fout << "# Region file format: DS9 version 4.1\n";
   fout << "global color=green dashlist=8 3 width=1 font=\"helvetica 10 normal roman\" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n";
   fout << "physical\n";
   
   for(int i=0; i< 4; ++i)
   {
      fout << "circle(" << x[i]+1 << "," << y[i]+1 << "," << r[i] << ") # color=red width=3\n";
      char txt[6];
      snprintf(txt, 6, "%0.2f", 2*r[i]);
      fout << "# text(" << x[i] - 15 << "," <<  y[i] + 10 << ") color=red font=\"helvetica 24 normal roman\" text={ " << txt << "}\n";
      
      double x0 = 29.5;
      double y0 = 29.5;
      if(i==1 || i == 3) x0 = 89.5;
      if(i==2 || i == 3) y0 = 89.5;
      snprintf(txt, 6, "%0.2f", sqrt(pow(x[i]-x0,2) + pow(y[i]-y0,2)) );
      fout << "# text(" << x[i] +7 << "," <<  y[i] -3 << ") color=red font=\"helvetica 16 normal roman\" text={ " << txt << "}\n";
   }

   fout << "line(" << x[0]+1 << "," << y[0]+1 << "," << x[1]+1 << "," << y[1]+1 <<  ") # line=0 0 color=red width=3\n";
   fout << "line(" << x[1]+1 << "," << y[1]+1 << "," << x[3]+1 << "," << y[3]+1 <<  ") # line=0 0 color=red width=3\n";
   fout << "line(" << x[2]+1 << "," << y[2]+1 << "," << x[3]+1 << "," << y[3]+1 <<  ") # line=0 0 color=red width=3\n";
   fout << "line(" << x[0]+1 << "," << y[0]+1 << "," << x[2]+1 << "," << y[2]+1 <<  ") # line=0 0 color=red width=3\n";


   fout << "circle(30.5,90.5,28)\n";
   fout << "circle(90.5,90.5,28)\n";
   fout << "circle(90.5,30.5,28)\n";
   fout << "circle(30.5,30.5,28)\n";
   fout << "line(30.5,120,30.5,1) # line=0 0\n";
   fout << "line(90.5,120,90.5,1) # line=0 0\n";
   fout << "line(1,90.5,120,90.5) # line=0 0\n";
   fout << "line(1,30.5,120,30.5) # line=0 0\n";

   fout.close();
   
   return 0;
}*/



} //namespace app
} //namespace MagAOX

#endif //pupilFitter_hpp
