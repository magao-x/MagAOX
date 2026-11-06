/** \file accelFFCtrlCancellation_test.hpp
 * \brief Explicit synthetic pupil calibration shared by offline cancellation tests and export.
 * \ingroup accelFFCtrl_unit_test
 */
#ifndef accelFFCtrlCancellation_test_hpp
#define accelFFCtrlCancellation_test_hpp

#include "../offline/accelFFCtrlCancellation.hpp"
#include "../../accelSim/accelSimDisturbance.hpp"

namespace libXWCTest
{
namespace accelFFCtrlTest
{
/// \cond ACCELFFCTRL_CANCELLATION_FIXTURES
namespace cancellation
{
using namespace MagAOX::app::accelFFCtrlDetail;
using namespace MagAOX::app::accelFFCtrlOffline;
namespace sim            = MagAOX::app::accelSimDetail;
constexpr double  pi     = 3.14159265358979323846;
constexpr int64_t period = 125000, origin = 1000000000, epoch = 1700000000000000000;

/// Eight explicit pupil quadrature points, with rotated orthogonal RMS-one surface-micron modes.
struct calibration
{
    Eigen::Matrix<double, 8, 2> m_modes; ///< Synthetic surface image B [micron/coefficient], not a deployed DM grid.
    Eigen::Matrix2d m_mapping, m_plant;  ///< Reused projector's modal K and nominal inverse pointing response.

    /// Fit synthetic diameter=6.5m, basis rotation=30deg using the existing physical TT projector.
    calibration();
};

inline calibration::calibration()
{
    constexpr double    diameter = 6.5, angle = pi / 6;
    std::vector<double> x( 8 ), y( 8 ), b0( 8 ), b1( 8 ), weights( 8, 1 );
    for( size_t n = 0; n < 8; ++n )
    {
        x[n]            = diameter / ( 2 * std::sqrt( 2. ) ) * std::cos( n * pi / 4 );
        y[n]            = diameter / ( 2 * std::sqrt( 2. ) ) * std::sin( n * pi / 4 );
        b0[n]           = 4 / diameter * ( std::cos( angle ) * x[n] + std::sin( angle ) * y[n] );
        b1[n]           = 4 / diameter * ( -std::sin( angle ) * x[n] + std::cos( angle ) * y[n] );
        m_modes( n, 0 ) = b0[n];
        m_modes( n, 1 ) = b1[n];
    }
    const sim::tipTiltProjection projection( b0, b1, x, y, weights, 1e-12 );
    Eigen::Matrix<double, 8, 2>  response;
    for( size_t n = 0; n < 8; ++n )
        for( size_t axis = 0; axis < 2; ++axis )
            response( n, axis ) = projection.response()[axis][n];
    m_mapping = m_modes.colPivHouseholderQr().solve( response );
    m_plant   = m_mapping.inverse();
}

/// A fabricated ready value for algebraic unit tests/oracle baselines only; never a controller input.
inline prediction idealPrediction( std::array<double, 2> theta /**< [in] analytic target angles [rad] */,
                                   int64_t               now /**< [in] query ns */,
                                   int64_t               horizon /**< [in] prediction horizon ns */ )
{
    prediction p;
    p.m_state        = estimatorState::ready;
    p.m_pointing     = theta;
    p.m_sampleTimeNs = now - period;
    p.m_targetTimeNs = now + horizon;
    p.m_sequence     = 1 + ( now - origin ) / period;
    p.m_generation   = 1;
    p.m_rcond        = 1;
    return p;
}

/// Prepare deliberately explicit synthetic experiment controls; production defaults are never read or edited.
inline correctionParameters controls( double  gain /**< [in] gain in [0,1] */,
                                      int64_t horizon = 1000000 /**< [in] horizon beyond query ns */ )
{
    correctionParameters c;
    c.m_gain      = gain;
    c.m_horizonNs = horizon;
    c.m_maxModal  = { .1, .1 };
    return c;
}
} // namespace cancellation
/// \endcond
} // namespace accelFFCtrlTest
} // namespace libXWCTest
#endif
