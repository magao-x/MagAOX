/** \file accelSimDM.hpp
 * \brief Precomputed, single-owner DM injection state independent of shared memory.
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

/// Sample-boundary state; faults latch until a new explicit control revision.
enum class dmState
{
    disabled,
    active,
    calibrationFault,
    streamFault,
    numericalFault,
    timingFault
};

/// Same-sample synthesis with an independent amplitude scale and no queued commands.
/** One thread owns this object, including disable/zero; shutdown may take ownership only after joining it.
 * A publisher accepts a complete vector and returns success without waiting for a consumer.
 */
class dmInjection
{
    std::optional<tipTiltBasis> m_basis;      ///< Validated precomputed angular responses and reusable command.
    std::vector<float>          m_command;    ///< Complete scaled contribution, allocated only during preparation.
    double                      m_bound{ 0 }; ///< Maximum response magnitude, used to reject overflow before synthesis.
    uint64_t m_revision{ 0 };              ///< Last consumed packed request: revision in high bits, enable in bit zero.
    bool     m_zeroPending{ false };       ///< A nonzero-capable publication requires an explicit final zero.
    dmState  m_state{ dmState::disabled }; ///< Latched owner-thread state.

  public:
    /// Construct uncalibrated and disabled, without stream access.
    dmInjection() = default;

    /// Prepare fixed DM-pupil geometry and the existing validated projector before acquisition starts.
    void prepare( const std::vector<double> &mode0 /**< [in] first surface-micron TT plane, native order */,
                  const std::vector<double> &mode1 /**< [in] second surface-micron TT plane */,
                  const std::vector<double> &mask /**< [in] deployed dmmask fitting weights */,
                  double                     residual /**< [in] accepted relative fit residual */ );

    /// Return whether calibration has been successfully prepared.
    bool ready() const;

    /// Return current single-owner state.
    dmState state() const;

    /// Publish an explicit zero; failure remains pending for a later retry.
    template <typename publisherT>
    bool zero( publisherT &publish /**< [in,out] complete-image publisher */ );

    /// Consume one completed ADC sample; control changes never replay an older nonzero command.
    template <typename publisherT>
    dmState service( const sample &value /**< [in] exact sample already published to acc000 */,
                     uint64_t      request /**< [in] monotonically revised enable request */,
                     double        scale /**< [in] DM-only scale in [0,1] */,
                     bool          healthy /**< [in] rate, synchronization, and calibration remain valid */,
                     publisherT   &publish /**< [in,out] active complete-image publisher */ );
};

inline void dmInjection::prepare( const std::vector<double> &mode0,
                                  const std::vector<double> &mode1,
                                  const std::vector<double> &mask,
                                  double                     residual )
{
    m_basis.reset();
    if( mode0.size() != 2500 || mode1.size() != 2500 || mask.size() != 2500 )
        throw std::invalid_argument( "invalid DM calibration geometry" );
    std::vector<double> x( 2500 ), y( 2500 );
    for( size_t n = 0; n < 2500; ++n )
    {
        x[n] = ( double( n % 50 ) - 24.5 ) * 6.5 / 48;
        y[n] = ( double( n / 50 ) - 24.5 ) * 6.5 / 48;
    }
    tipTiltProjection projection( mode0, mode1, x, y, mask, residual );
    m_bound = 0;
    for( const auto &plane : projection.response() )
        for( double v : plane )
            m_bound = std::max( m_bound, std::abs( v ) );
    m_basis.emplace( 50, 50, projection.response()[0], projection.response()[1] );
    m_command.assign( 2500, 0 );
}

inline bool dmInjection::ready() const
{
    return m_basis.has_value();
}

inline dmState dmInjection::state() const
{
    return m_state;
}

template <typename publisherT>
bool dmInjection::zero( publisherT &publish )
{
    if( !m_basis )
        return false;
    std::fill( m_command.begin(), m_command.end(), 0 );
    m_zeroPending = !publish( m_command );
    if( m_zeroPending )
        m_state = dmState::streamFault;
    return !m_zeroPending;
}

template <typename publisherT>
dmState dmInjection::service( const sample &value, uint64_t request, double scale, bool healthy, publisherT &publish )
{
    if( request < m_revision )
        return m_state; // A stale control revision must never rearm after a completed zero.
    if( request != m_revision )
    {
        m_revision = request;
        m_state    = dmState::disabled;
    }
    if( !( request & 1 ) )
    {
        if( m_zeroPending )
            zero( publish );
        return m_state;
    }
    if( m_state != dmState::disabled && m_state != dmState::active )
    {
        if( m_zeroPending )
            zero( publish );
        return m_state;
    }
    if( !ready() )
        return m_state = dmState::calibrationFault;
    if( !healthy )
        m_state = dmState::timingFault;
    else if( !std::isfinite( scale ) || scale < 0 || scale > 1 || !std::isfinite( value.m_pointing[0] ) ||
             !std::isfinite( value.m_pointing[1] ) ||
             ( std::abs( value.m_pointing[0] ) * arcsecToRad + std::abs( value.m_pointing[1] ) * arcsecToRad ) *
                     m_bound >
                 double( std::numeric_limits<float>::max() ) / 2 )
        m_state = dmState::numericalFault;
    if( m_state != dmState::disabled && m_state != dmState::active )
    {
        zero( publish );
        return m_state;
    }
    // Finite/overflow checks above make the existing synthesizer's throwing paths unreachable here.
    const auto &command = m_basis->synthesize( pointingDisturbance( value ) );
    for( size_t n = 0; n < command.size(); ++n )
        m_command[n] = static_cast<float>( command[n] * scale );
    m_zeroPending = true;
    if( !publish( m_command ) )
    {
        m_state = dmState::streamFault;
        zero( publish );
    }
    else
        m_state = dmState::active;
    return m_state;
}

} // namespace accelSimDetail
} // namespace app
} // namespace MagAOX
#endif
