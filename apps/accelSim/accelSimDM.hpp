/** \file accelSimDM.hpp
 * \brief Precomputed physical DM synthesis independent of transport and operator state.
 * \ingroup accelSim_files
 */
#ifndef accelSimDM_hpp
#define accelSimDM_hpp

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

/// Calibrated physical-pointing synthesis with an independent amplitude scale.
/** One thread (accelSim's DM output thread) owns this helper; it never publishes or manages output state. */
class dmInjection
{
    std::optional<tipTiltBasis> m_basis;   ///< Validated precomputed angular responses and reusable command.
    std::vector<float>          m_command; ///< Complete scaled contribution, allocated only during preparation.

  public:
    /// Construct uncalibrated, without stream access.
    dmInjection() = default;

    /// Prepare fixed DM-pupil geometry and the existing validated projector before acquisition starts.
    /** The mask only weights the 2x2 fit of mode coefficients to physical pointing; the output is always the full
     * supplied modes times that mapping and is never spatially masked. An empty mask fits over the modes' own
     * support (pixels where either mode is nonzero) with uniform weights; pixels outside the cube's support carry
     * no modal surface and would otherwise bias the fit. The unchanged residual tolerance rejects a support over
     * which the modes are not the requested physical tilt.
     */
    void prepare( const std::vector<double> &mode0 /**< [in] first surface-micron TT plane, native order */,
                  const std::vector<double> &mode1 /**< [in] second surface-micron TT plane */,
                  const std::vector<double> &mask /**< [in] dmmask fitting weights, or empty for mode support */,
                  double                     residual /**< [in] accepted relative fit residual */ );

    /// Return uniform fitting weights over the pixels where either mode is nonzero.
    static std::vector<double> modeSupport( const std::vector<double> &mode0 /**< [in] first mode plane */,
                                            const std::vector<double> &mode1 /**< [in] second mode plane */ );

    /// Return whether calibration has been successfully prepared.
    bool ready() const;

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
};

inline void dmInjection::prepare( const std::vector<double> &mode0,
                                  const std::vector<double> &mode1,
                                  const std::vector<double> &mask,
                                  double                     residual )
{
    m_basis.reset();
    if( mode0.size() != 2500 || mode1.size() != 2500 || ( !mask.empty() && mask.size() != 2500 ) )
        throw std::invalid_argument( "invalid DM calibration geometry" );
    std::vector<double> x( 2500 ), y( 2500 );
    for( size_t n = 0; n < 2500; ++n )
    {
        x[n] = ( double( n % 50 ) - 24.5 ) * 6.5 / 48;
        y[n] = ( double( n / 50 ) - 24.5 ) * 6.5 / 48;
    }
    tipTiltProjection projection( mode0, mode1, x, y, mask.empty() ? modeSupport( mode0, mode1 ) : mask, residual );
    m_basis.emplace( 50, 50, projection.response()[0], projection.response()[1] );
    m_command.assign( 2500, 0 );
}

inline std::vector<double> dmInjection::modeSupport( const std::vector<double> &mode0,
                                                    const std::vector<double> &mode1 )
{
    if( mode0.size() != mode1.size() )
        throw std::invalid_argument( "mismatched DM mode planes" );
    std::vector<double> weights( mode0.size() );
    for( size_t n = 0; n < mode0.size(); ++n )
        weights[n] = mode0[n] != 0 || mode1[n] != 0;
    return weights;
}

inline bool dmInjection::ready() const
{
    return m_basis.has_value();
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
