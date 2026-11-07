/** \file accelFFCtrlPhase_test.hpp
 * \brief Shared offline per-band output calibration of unchanged harmonic estimator results.
 * \ingroup accelFFCtrl_unit_test
 */
#ifndef accelFFCtrlPhase_test_hpp
#define accelFFCtrlPhase_test_hpp

#include "../accelFFCtrlModel.hpp"

namespace libXWCTest
{
namespace accelFFCtrlTest
{
/// \cond ACCELFFCTRL_PHASE_FIXTURE
using MagAOX::app::accelFFCtrlDetail::estimatorState;
using MagAOX::app::accelFFCtrlDetail::prediction;

/// One band maps one declared sensor's cosine/sine coordinates into two output coordinates.
struct phaseCalibration
{
    unsigned        m_axis{ 0 }; ///< ADC0 at the lower carrier, ADC1 at the upper carrier in the paired policy.
    Eigen::Matrix2d m_weights;   ///< Output rows and cosine/sine columns; units belong to the supplied fixture.
};

/// Evaluate the existing target phasor; lever=1 retains the historical count-normalized replay basis.
inline Eigen::Vector2d quadratures( const prediction &input /**< [in] original harmonic prediction */,
                                    double            frequency /**< [in] exact carrier Hz */,
                                    unsigned          axis /**< [in] sensor index 0 or 1 */,
                                    double lever = 1 /**< [in] explicit numerical or simulated physical lever */ )
{
    const double omega = 6.28318530717958647692 * frequency;
    const double phase = input.m_phase[axis] + omega * ( ( input.m_targetTimeNs - input.m_sampleTimeNs ) * 1e-9 );
    return ( -input.m_amplitude[axis] / ( omega * omega ) / lever ) *
           Eigen::Vector2d( std::cos( phase ), std::sin( phase ) );
}

/// Change output coordinates only; retain all source, target, fit and validity state.
inline prediction calibrated( const prediction       &input /**< [in] unchanged estimator result */,
                              double                  frequency /**< [in] exact fitted carrier Hz */,
                              const phaseCalibration &calibration /**< [in] earlier-fitted numerical coefficients */,
                              double lever = 1 /**< [in] explicit basis lever; one for count-normalized fixtures */ )
{
    prediction result = input;
    result.m_pointing = {};
    if( input.m_state != estimatorState::ready )
        return result;
    if( calibration.m_axis >= 2 || !std::isfinite( lever ) || lever <= 0 )
    {
        result.m_state = estimatorState::numericalFault;
        return result;
    }
    const Eigen::Vector2d mapped = calibration.m_weights * quadratures( input, frequency, calibration.m_axis, lever );
    if( !mapped.allFinite() )
        result.m_state = estimatorState::numericalFault;
    else
        result.m_pointing = { mapped[0], mapped[1] };
    return result;
}
/// \endcond
} // namespace accelFFCtrlTest
} // namespace libXWCTest
#endif
