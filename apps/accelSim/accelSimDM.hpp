/** \file accelSimDM.hpp
 * \brief Precomputed physical DM synthesis independent of transport and operator state.
 * \ingroup accelSim_files
 */
#ifndef accelSimDM_hpp
#define accelSimDM_hpp

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>

#include "accelSimDisturbance.hpp"

namespace MagAOX
{
namespace app
{
namespace accelSimDetail
{

constexpr double maxDMScale = 100; ///< Upper limit of the linear DM-only amplitude scale.

/// Result of the stateless staging adapter retained for existing controller test callers.
enum class dmState
{
    disabled,
    active,
    numericalFault,
    streamFault
};

/// How the 2x2 mode-coefficient calibration of the DM injection was obtained.
enum class dmCalibrationMethod
{
    none,        ///< Not prepared.
    fit,         ///< Least-squares fit of the modes to the physical tilt surface over fitting weights.
    coefficients ///< Explicitly configured coefficients; no tilt surface is fitted.
};

/// Return a stable name for a calibration method.
const char *dmCalibrationName( dmCalibrationMethod method /**< [in] calibration method */ );

/// Calibrated physical-pointing synthesis with an independent amplitude scale.
/** The command is always c0*mode0 + c1*mode1 over the full supplied planes, with modal coefficients c = K*theta for
 * simulated pointing theta [rad]. Calibration only determines the 2x2 matrix K; no actuator value is masked, zeroed
 * or renormalized. One thread (accelSim's DM output thread) owns this helper; it never publishes or manages output
 * state.
 */
class dmInjection
{
    std::optional<tipTiltBasis> m_basis;   ///< Validated precomputed angular responses and reusable command.
    std::vector<float>          m_command; ///< Complete scaled contribution, allocated only during preparation.
    dmCalibrationMethod         m_method{ dmCalibrationMethod::none }; ///< Source of m_mapping.
    Eigen::Matrix2d       m_mapping{ Eigen::Matrix2d::Zero() }; ///< K: rows modes, columns simulated X/Y [coef/rad].
    std::array<double, 2> m_residual{ NAN, NAN }; ///< Relative weighted fit residual per axis; NaN when not fitted.
    size_t                m_fitPixels{ 0 };       ///< Pixels with nonzero fitting weight; zero when not fitted.

  public:
    /// Construct uncalibrated, without stream access.
    dmInjection() = default;

    /// Fit K to the physical tilt surface of the fixed DM-pupil geometry before acquisition starts.
    /** The weights only select and weight the fit samples; the output is the full modes times K everywhere. An empty
     * mask uses uniform weights of 1.0 over the entire 50x50 plane. The fit must reproduce the physical tilt surface
     * within the residual tolerance over the weighted pixels; this is unchanged with or without a mask.
     */
    void prepare( const std::vector<double> &mode0 /**< [in] first surface-micron TT plane, native order */,
                  const std::vector<double> &mode1 /**< [in] second surface-micron TT plane */,
                  const std::vector<double> &mask /**< [in] fitting weights, or empty for uniform weights */,
                  double                     residual /**< [in] accepted relative fit residual */ );

    /// Use an explicitly configured K without fitting a tilt surface; the modes and K must both have rank two.
    void prepareCoefficients( const std::vector<double> &mode0 /**< [in] first surface-micron TT plane */,
                              const std::vector<double> &mode1 /**< [in] second surface-micron TT plane */,
                              const Eigen::Matrix2d     &mapping /**< [in] K [mode coefficient/rad] */ );

    /// Return whether calibration has been successfully prepared.
    bool ready() const;

    /// Return the calibration method of the latest successful preparation.
    dmCalibrationMethod method() const;

    /// Return K of the latest successful preparation [mode coefficient/rad].
    const Eigen::Matrix2d &mapping() const;

    /// Return the relative fit residual of each simulated axis; NaN for explicit coefficients.
    const std::array<double, 2> &residual() const;

    /// Return the number of pixels with nonzero fitting weight; zero for explicit coefficients.
    size_t fitPixels() const;

    /// Return a complete finite scaled command; invalid input throws without publishing.
    const std::vector<float> &
    synthesize( const sample &value /**< [in] physical sample; only noise-free pointing is used */,
                double        scale /**< [in] independent DM amplitude scale in [0,maxDMScale] */ );

    /// Preserve existing staging call sites without timing gates, revisions or latched state.
    template <typename publisherT>
    dmState service( const sample &value /**< [in] completed physical sample */,
                     uint64_t      request /**< [in] legacy enable request; only bit zero is used */,
                     double        scale /**< [in] independent amplitude scale in [0,maxDMScale] */,
                     bool /**< [in] unused former timing qualification */,
                     publisherT &publish /**< [in,out] private command staging callback */ );

  private:
    /// Clear the previous calibration before a new preparation.
    void clear();
};

inline const char *dmCalibrationName( dmCalibrationMethod method )
{
    switch( method )
    {
    case dmCalibrationMethod::fit:
        return "fit";
    case dmCalibrationMethod::coefficients:
        return "coefficients";
    default:
        return "none";
    }
}

inline void dmInjection::clear()
{
    m_basis.reset();
    m_method    = dmCalibrationMethod::none;
    m_mapping   = Eigen::Matrix2d::Zero();
    m_residual  = { NAN, NAN };
    m_fitPixels = 0;
}

inline void dmInjection::prepare( const std::vector<double> &mode0,
                                  const std::vector<double> &mode1,
                                  const std::vector<double> &mask,
                                  double                     residual )
{
    clear();
    if( mode0.size() != 2500 || mode1.size() != 2500 || ( !mask.empty() && mask.size() != 2500 ) )
        throw std::invalid_argument( "invalid DM calibration geometry" );
    std::vector<double> x( 2500 ), y( 2500 );
    for( size_t n = 0; n < 2500; ++n )
    {
        x[n] = ( double( n % 50 ) - 24.5 ) * 6.5 / 48;
        y[n] = ( double( n / 50 ) - 24.5 ) * 6.5 / 48;
    }
    const std::vector<double> weights = mask.empty() ? std::vector<double>( 2500, 1. ) : mask;
    tipTiltProjection         projection( mode0, mode1, x, y, weights, residual );
    m_basis.emplace( 50, 50, projection.response()[0], projection.response()[1] );
    m_command.assign( 2500, 0 );
    m_method    = dmCalibrationMethod::fit;
    m_mapping   = projection.mapping();
    m_residual  = projection.relativeResidual();
    m_fitPixels = size_t( std::count_if( weights.begin(), weights.end(), []( double w ) { return w > 0; } ) );
}

inline void dmInjection::prepareCoefficients( const std::vector<double> &mode0,
                                              const std::vector<double> &mode1,
                                              const Eigen::Matrix2d     &mapping )
{
    clear();
    if( mode0.size() != 2500 || mode1.size() != 2500 )
        throw std::invalid_argument( "invalid DM calibration geometry" );
    if( !mapping.allFinite() || !( std::abs( mapping.determinant() ) > 1e-12 * mapping.squaredNorm() ) )
        throw std::invalid_argument( "dm.coefficients must be a finite rank-two 2x2 matrix" );
    Eigen::Matrix2d gram = Eigen::Matrix2d::Zero();
    for( size_t n = 0; n < 2500; ++n )
    {
        if( !std::isfinite( mode0[n] ) || !std::isfinite( mode1[n] ) )
            throw std::invalid_argument( "nonfinite TT modes" );
        gram += Eigen::Vector2d( mode0[n], mode1[n] ) * Eigen::RowVector2d( mode0[n], mode1[n] );
    }
    if( !( std::abs( gram.determinant() ) > 1e-12 * gram.squaredNorm() ) )
        throw std::invalid_argument( "rank-deficient TT modes" );
    std::array<std::vector<double>, 2> response{ std::vector<double>( 2500 ), std::vector<double>( 2500 ) };
    for( size_t axis = 0; axis < 2; ++axis )
        for( size_t pixel = 0; pixel < 2500; ++pixel ) // The same expression as tipTiltProjection's response.
            response[axis][pixel] = mode0[pixel] * mapping( 0, axis ) + mode1[pixel] * mapping( 1, axis );
    m_basis.emplace( 50, 50, response[0], response[1] );
    m_command.assign( 2500, 0 );
    m_method  = dmCalibrationMethod::coefficients;
    m_mapping = mapping;
}

inline bool dmInjection::ready() const
{
    return m_basis.has_value();
}

inline dmCalibrationMethod dmInjection::method() const
{
    return m_method;
}

inline const Eigen::Matrix2d &dmInjection::mapping() const
{
    return m_mapping;
}

inline const std::array<double, 2> &dmInjection::residual() const
{
    return m_residual;
}

inline size_t dmInjection::fitPixels() const
{
    return m_fitPixels;
}

inline const std::vector<float> &dmInjection::synthesize( const sample &value, double scale )
{
    if( !ready() || !std::isfinite( scale ) || scale < 0 || scale > maxDMScale )
        throw std::invalid_argument( "invalid DM calibration or scale" );
    // The existing basis rejects nonfinite pointing and float overflow before returning a command.
    const auto &command = m_basis->synthesize( pointingDisturbance( value ) );
    for( size_t n = 0; n < command.size(); ++n )
        m_command[n] = static_cast<float>( command[n] * scale );
    return m_command;
}

template <typename publisherT>
dmState dmInjection::service( const sample &value, uint64_t request, double scale, bool, publisherT &publish )
{
    if( !( request & 1 ) )
        return dmState::disabled;
    try
    {
        return publish( synthesize( value, scale ) ) ? dmState::active : dmState::streamFault;
    }
    catch( const std::exception & )
    {
        return dmState::numericalFault;
    }
}

} // namespace accelSimDetail
} // namespace app
} // namespace MagAOX
#endif
