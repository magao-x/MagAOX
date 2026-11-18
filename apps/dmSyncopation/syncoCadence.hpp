/** \file syncoCadence.hpp
 * \brief Checked fixed-point timing arithmetic for immutable shadow-scheduler grids.
 */
#ifndef dmSyncopation_syncoCadence_hpp
#define dmSyncopation_syncoCadence_hpp

#include <limits>

#include "../../libMagAOX/common/dmSyncoSimulation.hpp"

namespace MagAOX
{
namespace app
{
namespace synco
{

/// Positive monotonic time with a nearest-nanosecond value and its exact signed binary remainder.
struct GridTime
{
    int64_t ns{ 0 };          ///< Rounded monotonic nanoseconds; zero denotes an unrepresentable result.
    int64_t fractionQ32{ 0 }; ///< Residual in [-periodScale/2, periodScale/2), preserving subnanosecond phase.
};

/// Round a checked positive binary timestamp without losing the residual required for future extrapolation.
inline GridTime roundGrid( __int128 value /**< [in] timestamp in scaled nanoseconds */ );

/// Extrapolate between genuine counter labels with the period frozen for the selected source.
inline GridTime shiftGrid( GridTime base /**< [in] rounded origin and residual */,
                           uint64_t from /**< [in] genuine origin counter */,
                           uint64_t to /**< [in] genuine target counter */,
                           uint64_t periodQ32 /**< [in] exact scaled nanoseconds per cycle */ );

/// Place a subframe directly within a frozen fractional cycle rather than repeatedly adding rounded offsets.
inline GridTime stepGrid( GridTime base /**< [in] frozen main grid and residual */,
                          uint64_t periodQ32 /**< [in] exact frozen scaled period */,
                          unsigned step /**< [in] requested nonnegative subframe */,
                          unsigned count /**< [in] positive total step count */ );

inline GridTime roundGrid( __int128 value )
{
    const __int128 scale = MagAOX::dmSyncoSimulation::periodScale;

    if( value <= 0 || value > __int128( std::numeric_limits<int64_t>::max() ) * scale + scale / 2 - 1 )
    {
        return {};
    }

    const __int128 rounded = ( value + scale / 2 ) / scale;

    if( rounded <= 0 )
    {
        return {};
    }

    return { int64_t( rounded ), int64_t( value - rounded * scale ) };
}

inline GridTime shiftGrid( GridTime base, uint64_t from, uint64_t to, uint64_t periodQ32 )
{
    const int64_t half = int64_t( MagAOX::dmSyncoSimulation::periodScale / 2 );

    if( base.ns <= 0 || base.fractionQ32 < -half || base.fractionQ32 >= half || !periodQ32 ||
        periodQ32 > uint64_t( 1000000000 ) * MagAOX::dmSyncoSimulation::periodScale )
    {
        return {};
    }

    return roundGrid( __int128( base.ns ) * MagAOX::dmSyncoSimulation::periodScale + base.fractionQ32 +
                      ( __int128( to ) - __int128( from ) ) * periodQ32 );
}

inline GridTime stepGrid( GridTime base, uint64_t periodQ32, unsigned step, unsigned count )
{
    const int64_t half = int64_t( MagAOX::dmSyncoSimulation::periodScale / 2 );

    if( base.ns <= 0 || base.fractionQ32 < -half || base.fractionQ32 >= half || !periodQ32 || !count ||
        periodQ32 > uint64_t( 1000000000 ) * MagAOX::dmSyncoSimulation::periodScale )
    {
        return {};
    }

    return roundGrid( __int128( base.ns ) * MagAOX::dmSyncoSimulation::periodScale + base.fractionQ32 +
                      __int128( periodQ32 ) * step / count );
}

} // namespace synco
} // namespace app
} // namespace MagAOX
#endif
