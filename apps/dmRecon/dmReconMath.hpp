/** \file dmReconMath.hpp
 * \brief Validated mask selection and private inverse construction for dmRecon.
 * \ingroup dmRecon_files
 */

#ifndef dmReconMath_hpp
#define dmReconMath_hpp

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <mx/improc/eigenCube.hpp>
#include <mx/improc/eigenImage.hpp>
#include <mx/math/eigenLapack.hpp>

namespace MagAOX
{
namespace app
{
namespace dmReconMath
{

/// Classify float bits without relying on floating-point assumptions made by fast-math builds.
inline bool finite( float value /**< [in] Value to classify. */ ) noexcept;

/// Check every element of a private buffer; an empty buffer is finite.
inline bool allFinite( const float *data /**< [in] Contiguous private values, non-null when count is nonzero. */,
                       size_t       count /**< [in] Number of values. */ ) noexcept;

/// Select exact unit mask entries in their existing column-major storage order.
/** Returns false for an empty, nonfinite or inactive mask, preserving the previous selection. */
inline bool maskIndices( std::vector<size_t> &indices /**< [out] Validated active pixel indices on success. */,
                         const float         *mask /**< [in] Contiguous column-major mask values. */,
                         size_t               count /**< [in] Number of mask pixels. */ );

/// Validate a nonzero perfect-square pixel count representable by a stream dimension.
inline bool squareSize( uint32_t &side /**< [out] Square side length on success. */,
                        size_t    count /**< [in] Number of pixels. */ ) noexcept;

/// Build and validate an inverse privately, preserving the previous inverse on failure.
/** Uses the same response normalization and mxlib SVD as dmRecon. Returns zero on success and -1 on invalid input,
 * failed decomposition or nonfinite output. The full-SVD wrapper requires at least as many active pixels as modes.
 */
inline int
buildInverse( mx::improc::eigenImage<float> &inverse /**< [out] Complete finite inverse on success. */,
              float &condition /**< [out] Condition estimate, initialized before decomposition. */,
              int   &rejected /**< [out] Rejected mode count, initialized before decomposition. */,
              const mx::improc::eigenCube<float>  &modes /**< [in] Private input mode cube. */,
              const std::vector<size_t>           &indices /**< [in] Active output-pixel indices. */,
              const mx::improc::eigenImage<float> &response /**< [in] Optional response matrix; empty disables it. */,
              int          inverseNumModes /**< [in] Number of singular modes to retain, or zero for all. */,
              std::string &error /**< [out] Failure explanation, empty on success. */ );

inline bool finite( float value ) noexcept
{
    static_assert( sizeof( float ) == sizeof( uint32_t ) && std::numeric_limits<float>::is_iec559 );
    return ( std::bit_cast<uint32_t>( value ) & 0x7f800000U ) != 0x7f800000U;
}

inline bool allFinite( const float *data, size_t count ) noexcept
{
    if( count != 0 && data == nullptr )
    {
        return false;
    }

    uint32_t invalid = 0;

    for( size_t index = 0; index < count; ++index )
    {
        invalid |= !finite( data[index] );
    }

    return invalid == 0;
}

inline bool maskIndices( std::vector<size_t> &indices, const float *mask, size_t count )
{
    if( count == 0 || !allFinite( mask, count ) )
    {
        return false;
    }

    std::vector<size_t> candidate;
    candidate.reserve( count );

    for( size_t index = 0; index < count; ++index )
    {
        if( mask[index] == 1 )
        {
            candidate.push_back( index );
        }
    }

    if( candidate.empty() )
    {
        return false;
    }

    indices.swap( candidate );
    return true;
}

inline bool squareSize( uint32_t &side, size_t count ) noexcept
{
    if( count == 0 )
    {
        return false;
    }

    const auto root = static_cast<uint64_t>( std::sqrt( static_cast<long double>( count ) ) );

    if( root == 0 || root > std::numeric_limits<uint32_t>::max() || count / root != root || count % root != 0 )
    {
        return false;
    }

    side = static_cast<uint32_t>( root );
    return true;
}

inline int buildInverse( mx::improc::eigenImage<float>       &inverse,
                         float                               &condition,
                         int                                 &rejected,
                         const mx::improc::eigenCube<float>  &modes,
                         const std::vector<size_t>           &indices,
                         const mx::improc::eigenImage<float> &response,
                         int                                  inverseNumModes,
                         std::string                         &error )
{
    condition = 1;
    rejected  = 0;
    error.clear();

    if( modes.rows() <= 0 || modes.cols() <= 0 || modes.planes() <= 0 || indices.empty() ||
        indices.size() < static_cast<size_t>( modes.planes() ) || inverseNumModes < 0 ||
        inverseNumModes > modes.planes() )
    {
        error = "invalid mode dimensions, active pixel count or inverse mode count";
        return -1;
    }

    if( static_cast<size_t>( modes.rows() ) > std::numeric_limits<size_t>::max() / static_cast<size_t>( modes.cols() ) )
    {
        error = "mode image is too large";
        return -1;
    }

    const size_t sourcePixels = static_cast<size_t>( modes.rows() ) * static_cast<size_t>( modes.cols() );

    if( sourcePixels > std::numeric_limits<size_t>::max() / static_cast<size_t>( modes.planes() ) ||
        !allFinite( modes.data(), sourcePixels * static_cast<size_t>( modes.planes() ) ) )
    {
        error = "mode cube is too large or contains nonfinite values";
        return -1;
    }

    size_t     outputPixels = sourcePixels;
    const bool useResponse  = response.size() != 0;

    if( useResponse )
    {
        uint32_t side = 0;

        if( response.cols() != static_cast<Eigen::Index>( sourcePixels ) || !squareSize( side, response.rows() ) ||
            !allFinite( response.data(), response.size() ) )
        {
            error = "response dimensions do not match the modes or contain nonfinite values";
            return -1;
        }

        outputPixels = response.rows();
    }

    for( size_t index = 0; index < indices.size(); ++index )
    {
        if( indices[index] >= outputPixels || ( index != 0 && indices[index] <= indices[index - 1] ) )
        {
            error = "active pixel indices are outside the mode image or not strictly increasing";
            return -1;
        }
    }

    mx::improc::eigenImage<float> maskedModes( indices.size(), modes.planes() );
    mx::improc::eigenImage<float> transformed;

    for( Eigen::Index plane = 0; plane < modes.planes(); ++plane )
    {
        const float *pixels = modes.image( plane ).data();

        if( useResponse )
        {
            Eigen::Map<const Eigen::Matrix<float, -1, 1>> source( pixels, sourcePixels );
            transformed       = ( response.matrix() * source ).array();
            const float norm  = std::sqrt( transformed.square().sum() / indices.size() );
            const float scale = std::sqrt( modes.image( plane ).square().sum() / sourcePixels );

            if( !finite( norm ) || !finite( scale ) || norm <= 0 || scale <= 0 )
            {
                error = "response mode normalization is zero or nonfinite";
                return -1;
            }

            const float factor = scale / norm;

            if( !finite( factor ) || factor <= 0 )
            {
                error = "response mode normalization factor is zero or nonfinite";
                return -1;
            }

            transformed *= factor;

            if( !allFinite( transformed.data(), transformed.size() ) )
            {
                error = "normalized response mode contains nonfinite values";
                return -1;
            }

            pixels = transformed.data();
        }

        for( size_t index = 0; index < indices.size(); ++index )
        {
            maskedModes( index, plane ) = pixels[indices[index]];
        }
    }

    mx::improc::eigenImage<float> candidate;
    float                         maxCondition = -static_cast<float>( inverseNumModes );
    const int result = mx::math::eigenPseudoInverse( candidate, condition, rejected, maskedModes, maxCondition );

    if( result != 0 )
    {
        error = "eigenPseudoInverse failed with status " + std::to_string( result );
        return -1;
    }

    if( candidate.rows() != modes.planes() || candidate.cols() != static_cast<Eigen::Index>( indices.size() ) ||
        !finite( condition ) || !allFinite( candidate.data(), candidate.size() ) )
    {
        error = "inverse dimensions or numerical values are invalid";
        return -1;
    }

    inverse.swap( candidate );
    return 0;
}

} // namespace dmReconMath
} // namespace app
} // namespace MagAOX

#endif // dmReconMath_hpp
