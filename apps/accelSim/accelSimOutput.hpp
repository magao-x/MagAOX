/** \file accelSimOutput.hpp
 * \brief Select published ADC counts without changing the simulated physical state.
 * \ingroup accelSim_files
 */
#ifndef accelSimOutput_hpp
#define accelSimOutput_hpp

#include "accelSimModel.hpp"

namespace MagAOX
{
namespace app
{
namespace accelSimDetail
{

/// Select the acquired pair or calibrated zero-acceleration counts; never change the model or its clamp counters.
std::array<uint16_t, 2> adcOutput( const sample     &value /**< [in] Acquired physical/ADC sample. */,
                                   const parameters &config /**< [in] Validated startup ADC calibration. */,
                                   bool              zeroOutput = false /**< [in] Replace only the published pair. */ );

inline std::array<uint16_t, 2> adcOutput( const sample &value, const parameters &config, bool zeroOutput )
{
    if( !zeroOutput )
    {
        return value.m_adc;
    }

    uint64_t   low = 0, high = 0;
    const auto zero = adcCounts( 0, config, low, high );
    return { zero, zero };
}

} // namespace accelSimDetail
} // namespace app
} // namespace MagAOX
#endif // accelSimOutput_hpp
