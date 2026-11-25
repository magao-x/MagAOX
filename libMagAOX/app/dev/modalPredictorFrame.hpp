/** \file modalPredictorFrame.hpp
 * \brief Shared mode-major predictor frame used by modalGainOpt and modalFilter.
 */

#ifndef libMagAOX_app_dev_modalPredictorFrame_hpp
#define libMagAOX_app_dev_modalPredictorFrame_hpp

#include <cstddef>
#include <span>

#include <mx/math/floatUtils.hpp>

namespace MagAOX::app::modalPredictor
{

/// Maximum supported tap count on either side of the controller.
inline constexpr size_t maxTaps = 1000;

/// Number of scalars before the two equally sized tap arrays in a mode column.
inline constexpr size_t headerSize = 4;

/// Compute the number of floats per mode.
inline constexpr size_t width( size_t capacity /**< [in] Stored taps per array. */ )
{
    return headerSize + 2 * capacity;
}

/// Test the size of a mode column before interpreting its data.
inline constexpr bool validWidth( size_t columns /**< [in] Number of floats in each mode. */ )
{
    return columns >= headerSize && columns <= width( maxTaps ) && ( columns - headerSize ) % 2 == 0;
}

/// A validated mode with a gain, a command multiplier and its two tap sequences.
struct Mode
{
    float                  gain{ 0 }; ///< PC gain factor in command units.
    float                  mult{ 1 }; ///< PC command-history multiplier factor.
    std::span<const float> a;         ///< IIR taps on previous commands, most recent first.
    std::span<const float> b;         ///< FIR taps on WFS error, beginning with the current sample.

    /// Whether the mode uses predictive control instead of the SI controller.
    bool enabled() const
    {
        return !a.empty() && !b.empty();
    }
};

/// Read one mode. Two zero counts explicitly select the SI controller.
inline bool decode( Mode                  &mode /**< [out] Borrowed mode data valid for the lifetime of the frame. */,
                    std::span<const float> data /**< [in] Complete mode column. */ )
{
    if( !validWidth( data.size() ) )
    {
        return false;
    }

    const size_t capacity = ( data.size() - headerSize ) / 2;

    for( size_t i = 0; i < headerSize; ++i )
    {
        if( !mx::math::isFinite( data[i] ) || data[i] < 0 )
        {
            return false;
        }
    }

    if( data[2] > capacity || data[3] > capacity )
    {
        return false;
    }

    const auto na = static_cast<size_t>( data[2] );
    const auto nb = static_cast<size_t>( data[3] );

    if( data[2] != na || data[3] != nb || ( na == 0 ) != ( nb == 0 ) )
    {
        return false;
    }

    mode = { data[0], data[1], data.subspan( headerSize, na ), data.subspan( headerSize + capacity, nb ) };

    for( const auto taps : { mode.a, mode.b } )
    {
        for( float tap : taps )
        {
            if( !mx::math::isFinite( tap ) )
            {
                return false;
            }
        }
    }

    return true;
}

} // namespace MagAOX::app::modalPredictor

#endif // libMagAOX_app_dev_modalPredictorFrame_hpp
