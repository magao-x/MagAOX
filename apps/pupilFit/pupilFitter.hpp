/** \file pupilFitter.hpp
 * \brief The MagAO-X Pyramid Pupil Fitter class header
 *
 * \ingroup pupilFit_files
 */

#ifndef pupilFitter_hpp
#define pupilFitter_hpp

#include <mx/math/floatUtils.hpp>

#include <mx/improc/eigenImage.hpp>
#include <mx/improc/imageUtils.hpp>
#include <mx/improc/circleOuterpix.hpp>
#include <mx/improc/imageTransforms.hpp>

namespace MagAOX
{
namespace app
{

/// Struct to perform centration and measure diameter of Pyramid pupils
template <typename realT>
struct pupilFitter
{
    /// Workspace for one pupil region at the input sampling.
    mx::improc::eigenImage<realT> m_quad;

    /// Magnified, normalized, then thresholded pupil workspace.
    mx::improc::eigenImage<realT> m_quadMag;

    /// Rebinned edge mask for one pupil region.
    mx::improc::eigenImage<realT> m_circ;

    /// Magnified edge mask used to measure the pupil circle.
    mx::improc::eigenImage<realT> m_circMag;

    unsigned m_rows{ 0 }; ///< Allocated pupil-region height in pixels.
    unsigned m_cols{ 0 }; ///< Allocated pupil-region width in pixels.

    int   m_numPupils{ 4 }; ///< the number of pupils.  Default is 4, 3 is also supported.

    /// Sorted-pixel quantile used to estimate the local background.
    float m_bgMedIndex{ 0.1867 };
    float m_pupMedIndex{ 0.6867 }; ///< the index of the illuminated pupil median in a sorted array.


    /// Background estimates in active quadrant order.
    float m_bg[4]   = { 0, 0, 0, 0 };

    /// Illuminated-pupil intensity estimates in active quadrant order.
    float m_med[4]  = { 0, 0, 0, 0 };

    /// Measured full-image x centers in active quadrant order.
    float m_avgx[4] = { 0, 0, 0, 0 };

    /// Measured full-image y centers in active quadrant order.
    float m_avgy[4] = { 0, 0, 0, 0 };

    /// Measured pupil radii in input-image pixels.
    float m_avgr[4] = { 0, 0, 0, 0 };


    /// Threshold applied after local background subtraction and normalization.
    float m_thresh{ 0.5 };


    /// Sorted magnified pixels used to estimate intensity quantiles.
    std::vector<realT> m_pixs;


    /// Construct an unallocated fitter with the default four-pupil geometry.
    pupilFitter()
    {
    }

    /// Allocate workspaces for a pupil region and its tenfold magnification.
    int setSize( unsigned rows /**< [in] region height in pixels */,
                 unsigned cols /**< [in] region width in pixels */ );


    /// Return the starting coordinates of a region in the full image.
    int quadCoords( size_t &i0 /**< [out] first row of the pupil region */,
                    size_t &j0 /**< [out] first column of the pupil region */,
                    int quadNo /**< [in] zero-based pupil index */ );


    /// Replace normalized intensities with a binary pupil mask.
    int threshold( mx::improc::eigenImage<realT> &im /**< [in/out] normalized pupil image */ );


    /// Extract one pupil region from a full image.
    int getQuad( mx::improc::eigenImage<realT> &quad /**< [out] preallocated region workspace */,
                 mx::improc::eigenImage<realT> &im /**< [in] full input image */,
                 int quadNo /**< [in] zero-based pupil index */ );


    /// Copy a processed pupil region back into a full image.
    int putQuad( mx::improc::eigenImage<realT> &im /**< [out] preallocated full image */,
                 mx::improc::eigenImage<realT> &quad /**< [in] processed pupil region */,
                 int quadNo /**< [in] zero-based pupil index */ );


    /// Measure the full-image center and radius from the magnified pupil edge.
    int outerpix( float &avgx /**< [out] full-image x center */,
                  float &avgy /**< [out] full-image y center */,
                  float &avgr /**< [out] radius at input-image sampling */,
                  int quadNo /**< [in] zero-based pupil index */ );


    /// Fit every active pupil, rejecting invalid dimensions, settings, pixels, or measurements.
    /** \returns 0 on success, -1 on failure. On failure, callers must discard all measurement arrays. */
    int fit( mx::improc::eigenImage<realT> &im /**< [in/out] full image, replaced by thresholded pupils */,
             mx::improc::eigenImage<realT> &edged /**< [out] resized full-image edge mask */ );

};

template <typename realT>
int pupilFitter<realT>::setSize( unsigned rows, unsigned cols )
{
    m_rows = rows;
    m_cols = cols;

    m_quad.resize( m_rows, m_cols );
    m_circ.resize( m_rows, m_cols );

    // Set up for the magnified version
    m_quadMag.resize( m_quad.rows() * 10, m_quad.cols() * 10 );
    m_circMag.resize( m_circ.rows() * 10, m_circ.cols() * 10 );

    m_pixs.resize( m_quadMag.rows() * m_quadMag.cols() );

    return 0;
}

template <typename realT>
int pupilFitter<realT>::quadCoords( size_t &i0, size_t &j0, int quadNo )
{
    if( m_numPupils == 3 )
    {
        i0 = 0;
        j0 = 0;

        if( quadNo == 1 )
        {
            i0 = m_rows;
        }
        if( quadNo == 2 )
        {
            i0 = 0.5 * m_rows;
            j0 = m_cols;
        }
    }
    else
    {
        i0 = 0;
        j0 = 0;

        if( quadNo == 1 )
            i0 = m_rows;
        if( quadNo == 2 )
            j0 = m_cols;
        if( quadNo == 3 )
        {
            i0 = m_rows;
            j0 = m_cols;
        }
    }

    return 0;
}

template <typename realT>
int pupilFitter<realT>::threshold( mx::improc::eigenImage<realT> &im )
{
    for( int i = 0; i < im.rows(); ++i )
    {
        for( int j = 0; j < im.cols(); ++j )
        {
            if( im( i, j ) >= m_thresh )
                im( i, j ) = 1;
            else
                im( i, j ) = 0;
        }
    }

    return 0;
}

template <typename realT>
int pupilFitter<realT>::getQuad( mx::improc::eigenImage<realT> &quad, mx::improc::eigenImage<realT> &im, int quadNo )
{

    if( im.rows() != 2 * m_rows || im.cols() != 2 * m_cols )
    {
        return -1;
    }

    size_t i0 = 0, j0 = 0;
    quadCoords( i0, j0, quadNo );

    for( size_t i = i0; i < i0 + m_rows; ++i )
    {
        for( size_t j = j0; j < j0 + m_cols; ++j )
        {
            quad( i - i0, j - j0 ) = im( i, j );
        }
    }

    return 0;
}

template <typename realT>
int pupilFitter<realT>::putQuad( mx::improc::eigenImage<realT> &im, mx::improc::eigenImage<realT> &quad, int quadNo )
{
    if( im.rows() != 2 * m_rows || im.cols() != 2 * m_cols )
    {
        return -1;
    }

    size_t i0 = 0, j0 = 0;
    quadCoords( i0, j0, quadNo );

    for( size_t i = i0; i < i0 + m_rows; ++i )
    {
        for( size_t j = j0; j < j0 + m_cols; ++j )
        {
            im( i, j ) = quad( i - i0, j - j0 );
            ;
        }
    }

    return 0;
}

template <typename realT>
int pupilFitter<realT>::outerpix( float &avgx, float &avgy, float &avgr, int quadNo )
{

    realT x0, y0, avgr0;

    if(mx::improc::circleOuterpix( x0, y0, avgr0, avgx, avgy, avgr, m_circMag, m_quadMag ) < 0) return -1;

    avgx /= 10.0;
    avgy /= 10.0;
    avgr /= 10.0;

    size_t i0 = 0, j0 = 0;
    quadCoords( i0, j0, quadNo );

    avgx += i0;
    avgy += j0;

    return 0;
}

template <typename realT>
int pupilFitter<realT>::fit( mx::improc::eigenImage<realT> &im, mx::improc::eigenImage<realT> &edged )
{
    try
    {

        if( (m_numPupils != 3 && m_numPupils != 4) || m_rows == 0 || m_cols == 0 ||
            im.rows() != 2*m_rows || im.cols() != 2*m_cols ||
            !mx::math::isFinite(m_thresh) || m_thresh <= 0 || m_thresh >= 1 ||
            !mx::math::isFinite(m_bgMedIndex) || !mx::math::isFinite(m_pupMedIndex) ||
            m_bgMedIndex < 0 || m_bgMedIndex >= m_pupMedIndex || m_pupMedIndex >= 1 ) return -1;

        for(Eigen::Index i=0; i<im.size(); ++i)
            if(!mx::math::isFinite(im.data()[i])) return -1;

        edged.resize(im.rows(), im.cols());
        edged.setZero();
        mx::improc::eigenImage<realT> imin = im;
        im.setZero();
        for( int i = 0; i < m_numPupils; ++i )
        {
            // 0) magnify the image
            if(getQuad( m_quad, imin, i ) < 0) return -1;

            mx::improc::imageMagnify( m_quadMag, m_quad, mx::improc::bilinearTransform<realT>() );

            // 1) normalize by median of pupil

            for( size_t j = 0; j < m_pixs.size(); ++j )
            {
                m_pixs[j] = m_quadMag.data()[j];
            }

            std::sort( m_pixs.begin(), m_pixs.end() );

            m_bg[i] = m_pixs[m_bgMedIndex * m_pixs.size()];

            m_med[i] = m_pixs[m_pupMedIndex * m_pixs.size()]; // This should be the median of the pupils if the right
                                                              // size and contained in the quad

            if( !mx::math::isFinite(m_med[i]) || !mx::math::isFinite(m_bg[i]) || m_med[i] <= m_bg[i] ) return -1;
            m_quadMag = ( m_quadMag - m_bg[i] ) / ( m_med[i] - m_bg[i] );

            // 2) Threshold the normalized quad
            threshold( m_quadMag );

            // 3) Find outer pixels and the radius
            if(outerpix( m_avgx[i], m_avgy[i], m_avgr[i], i ) < 0 ||
               !mx::math::isFinite(m_avgx[i]) || !mx::math::isFinite(m_avgy[i]) ||
               !mx::math::isFinite(m_avgr[i]) || m_avgr[i] <= 0 ) return -1;

            // 4) De-magnify and prepare for putting on the streams
            mx::improc::imageRebinSum( m_quad, m_quadMag, true );
            mx::improc::imageRebinSum( m_circ, m_circMag );
            if(putQuad( im, m_quad, i ) < 0 || putQuad( edged, m_circ, i ) < 0) return -1;
        }

        return 0;
    }
    catch( const std::exception &e )
    {
        std::cerr << __FILE__ << ' ' << __LINE__ << ' ' << e.what() << '\n';
        return -1;
    }
}
} // namespace app
} // namespace MagAOX

#endif // pupilFitter_hpp
