/** \file pupilFitter.hpp
  * \brief The MagAO-X Pyramid Pupil Fitter class header
  *
  * \ingroup pupilFit_files
  */

#ifndef pupilFitter_hpp
#define pupilFitter_hpp

#include <mx/improc/eigenImage.hpp>
#include <mx/improc/imageUtils.hpp>
#include <mx/improc/circleOuterpix.hpp>
#include <mx/improc/imageTransforms.hpp>

namespace MagAOX
{
namespace app
{
   
/// Struct to perform centration and measure diameter of Pyramid pupils
template<typename realT>
struct pupilFitter
{
   mx::improc::eigenImage<realT> m_quad;
   mx::improc::eigenImage<realT> m_quadMag;
   mx::improc::eigenImage<realT> m_circ;
   mx::improc::eigenImage<realT> m_circMag;
   
   unsigned m_rows {0}; ///< [in] the size of a quad, in rows
   unsigned m_cols {0}; ///< [in] the size of a quad, in cols
   
   int m_numPupils {4}; ///< the number of pupils.  Default is 4, 3 is also supported.
   float m_pupMedIndex {0.6866}; ///< the index of the illuminated pupil median in a sorted array.
   
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
   
   /// Returns the quadrant starting coordinates
   int quadCoords( size_t & i0, ///< [out] the i coordinate of the lower-left corner of quad 
                   size_t & j0, ///< [out] the j coordinate of the lower-left corner of auad
                   int quadNo   ///< [in] the quadrant number
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
   
   //Set up for the magnified version
   m_quadMag.resize(m_quad.rows()*10, m_quad.cols()*10);
   m_circMag.resize(m_circ.rows()*10, m_circ.cols()*10);

   m_pixs.resize(m_quadMag.rows()*m_quadMag.cols());
   
   return 0;
}
   
template<typename realT>
int pupilFitter<realT>::quadCoords( size_t & i0,
                                    size_t & j0,
                                    int quadNo  
                                  )
{
   if(m_numPupils == 3)
   {
      i0 = 0;
      j0 = 0;
      
      if(quadNo == 1) 
      {
         i0 = m_rows;
      }
      if(quadNo == 2)
      {
         i0 = 0.5*m_rows;
         j0 = m_cols;
      }
   }
   else
   {
      i0=0;
      j0=0;
   
      if(quadNo == 1) i0 = m_rows;
      if(quadNo == 2) j0 = m_cols;
      if(quadNo == 3) 
      {
         i0 = m_rows;
         j0 = m_cols;
      }
      
   }
   
   return 0;
}

template<typename realT>
int pupilFitter<realT>::threshold( mx::improc::eigenImage<realT> & im )
{
   for(int i =0; i< im.rows(); ++i)
   {
      for(int j=0; j< im.cols(); ++j)
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
   quadCoords(i0, j0, quadNo);
   
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
   quadCoords(i0, j0, quadNo);
   
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
   
   realT x0, y0, avgr0;

   mx::improc::circleOuterpix(x0, y0, avgr0, avgx, avgy, avgr, m_circMag, m_quadMag);
   
   avgx/=10.0;
   avgy/=10.0;
   avgr/=10.0;

   size_t i0=0, j0=0;
   quadCoords(i0, j0, quadNo);
   
   avgx += i0;
   avgy += j0;
   
   return 0;
   
}



template<typename realT>
int pupilFitter<realT>::fit( mx::improc::eigenImage<realT> & im, 
                             mx::improc::eigenImage<realT> & edged
                           )
{
   mx::improc::eigenImage<realT>  imin = im;
   im.setZero();
   for(int i=0; i< m_numPupils; ++i)
   {
      // 0) magnify the image
      getQuad(m_quad, imin, i);
      
      mx::improc::imageMagnify(m_quadMag, m_quad, mx::improc::bilinearTransform<realT>());

      // 1) normalize by median of pupil

      for(size_t j=0; j < m_pixs.size(); ++j) m_pixs[j] = m_quadMag.data()[j];
      
      std::sort(m_pixs.begin(), m_pixs.end());

      m_med[i] = m_pixs[m_pupMedIndex*m_pixs.size()]; //This should be the median of the pupils if the right size and contained in the quad
      
      m_quadMag/=m_med[i]; 
      
      // 2) Threshold the normalized quad
      threshold(m_quadMag);

      // 3) Find outer pixels and the radius
      outerpix(m_avgx[i], m_avgy[i], m_avgr[i], i);
   
      // 4) De-magnify and prepare for putting on the streams
      mx::improc::imageRebinSum( m_quad, m_quadMag, true);
      mx::improc::imageRebinSum( m_circ, m_circMag);
      putQuad(im, m_quad, i);
      putQuad(edged, m_circ, i);
   }
   
   return 0;
}
} //namespace app
} //namespace MagAOX

#endif //pupilFitter_hpp
