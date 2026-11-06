/** \file accelFFCtrlAO_test.hpp
 * \brief Synthetic leaky AO integrator for offline interaction experiments, not a deployed controller model.
 * \ingroup accelFFCtrl_unit_test
 */
#ifndef accelFFCtrlAO_test_hpp
#define accelFFCtrlAO_test_hpp

#include "accelFFCtrlCancellation_test.hpp"

namespace libXWCTest
{
namespace accelFFCtrlTest
{
/// \cond ACCELFFCTRL_AO_FIXTURE
/// Deliberate test model: caller supplies only a previously acquired synthetic WFS residual.
class syntheticIntegrator
{
    Eigen::Matrix2d m_mapping; ///< Synthetic WFS radians to modal coefficients.
    Eigen::Vector2d m_state{
        Eigen::Vector2d::Zero() }; ///< Integrated AO contribution, not an incremental channel write.
    double m_gain, m_leak;         ///< Test-only integrator parameters, never inferred from deployed configuration.

  public:
    /// Validate explicit synthetic calibration/gain; zero default gain leaves the AO contribution zero.
    syntheticIntegrator( const Eigen::Matrix2d &mapping /**< [in] coefficients per WFS radian */,
                         double                 gain = 0 /**< [in] test gain in [0,1] */,
                         double                 leak = 0 /**< [in] per-query leak in [0,1] */ );

    /// Update from a previous WFS sample and return the entire integrated contribution.
    Eigen::Vector2d update( const std::array<double, 2> &residual /**< [in] earlier measured WFS residual radians */ );

    /// Optional synthetic back-calculation; it models neither the real combiner nor hardware saturation handling.
    void track( const Eigen::Vector2d &excess /**< [in] total proposed minus applied coefficients */,
                double                 weight /**< [in] illustrative back-calculation weight in [0,1] */ );

    /// Clear only the synthetic AO state.
    void reset();
};

inline syntheticIntegrator::syntheticIntegrator( const Eigen::Matrix2d &mapping, double gain, double leak )
    : m_mapping( mapping ), m_gain( gain ), m_leak( leak )
{
    if( !mapping.allFinite() || !std::isfinite( gain ) || gain < 0 || gain > 1 || !std::isfinite( leak ) || leak < 0 ||
        leak > 1 )
        throw std::invalid_argument( "invalid synthetic integrator" );
}

inline Eigen::Vector2d syntheticIntegrator::update( const std::array<double, 2> &residual )
{
    const Eigen::Vector2d measured( residual[0], residual[1] );
    if( !measured.allFinite() )
        throw std::invalid_argument( "invalid synthetic WFS residual" );
    m_state = ( 1 - m_leak ) * m_state - m_gain * m_mapping * measured;
    if( !m_state.allFinite() )
        throw std::overflow_error( "synthetic integrator overflow" );
    return m_state;
}

inline void syntheticIntegrator::track( const Eigen::Vector2d &excess, double weight )
{
    if( !excess.allFinite() || !std::isfinite( weight ) || weight < 0 || weight > 1 )
        throw std::invalid_argument( "invalid synthetic anti-windup" );
    m_state -= weight * excess;
}

inline void syntheticIntegrator::reset()
{
    m_state.setZero();
}
/// \endcond
} // namespace accelFFCtrlTest
} // namespace libXWCTest
#endif
