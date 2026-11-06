/** \file accelFFCtrlCancellation.hpp
 * \brief Offline-only feed-forward modal proposal and synthetic delayed DM response; no publication API.
 * \ingroup accelFFCtrl_files
 */
#ifndef accelFFCtrlCancellation_hpp
#define accelFFCtrlCancellation_hpp

#include <span>
#include <vector>

#include "../accelFFCtrlModel.hpp"

namespace MagAOX
{
namespace app
{
namespace accelFFCtrlOffline
{
using accelFFCtrlDetail::prediction;

/// Experiment controls only; amplitude bounds are synthetic modal coefficients, not instrument protection.
struct correctionParameters
{
    double                m_gain{ 0 };  ///< Opposing feed-forward gain, in [0,1], disabled by default.
    std::array<double, 2> m_maxModal{}; ///< Explicit nonnegative absolute modal bounds [coefficient]; synthetic modes
                                        ///< have 1 micron surface RMS at coefficient one.
    int64_t               m_horizonNs{ 0 }; ///< Target beyond query now; must match every supplied prediction exactly.
    int64_t m_maxAgeNs{ 2000000 }, m_maxPredictionNs{ 50000000 }; ///< Freshness and target-validity bounds.
};

/// A fresh proposal replaces the previous proposal even when disabled or inhibited.
enum class correctionState
{
    disabled,
    inhibited,
    active,
    limited
};

/// Owned two-mode proposal; invalid/disabled proposals always contain zero, never a held command.
struct correction
{
    std::array<double, 2> m_modal{}; ///< Complete instantaneous modal contribution; never an accumulated increment.
    correctionState       m_state{ correctionState::inhibited }; ///< Numerical proposal status, not a hardware enable.
    int64_t               m_targetNs{ 0 }; ///< Validated prediction target; zero when disabled/inhibited.
    double                m_scale{ 0 };    ///< Common amplitude-bound scalar, after the configured gain.
};

/// Fixed-storage radians-to-modal conversion, intentionally excluded from the live application.
class feedForward
{
    Eigen::Matrix2d m_mapping; ///< Explicit supplied TT calibration [modal coefficient/rad], no default calibration.
    correctionParameters m_config; ///< Immutable controls; constructing a replacement requires no history.

  public:
    /// Validate the explicit calibration and experiment controls; construction is outside sample processing.
    feedForward( const Eigen::Matrix2d      &mapping /**< [in] calibrated modal rows, pointing-axis columns */,
                 const correctionParameters &config = {} /**< [in] gain, bounds and exact prediction horizon */ );

    /// Sum coherent ready bands and return -gain*M*theta; any fault inhibits the entire new proposal.
    correction
    propose( std::span<const prediction> bands /**< [in] predictions with identical source/target identity */,
             int64_t now /**< [in] actual query time in the predictions' monotonic domain */ ) const;
};

/// Synthetic modal plant with exact integer-step delay and first-order zero-order-held input response.
/** For bandwidth>0: x[n]=a*x[n-1]+(1-a)*u[n-d-1], a=exp(-2*pi*bandwidth*dt).
 * This evolves the preceding interval: a new command cannot change an earlier state.
 * Bandwidth=0 explicitly selects the ideal instantaneous delayed plant x[n]=u[n-d].
 * Output angle is the supplied plant matrix times x; no WFS feedback loop is simulated.
 */
class dmResponse
{
    Eigen::Matrix2d              m_mapping; ///< Synthetic physical response [pointing rad/modal coefficient].
    std::vector<Eigen::Vector2d> m_history; ///< Startup-allocated delay line, containing only issued commands.
    Eigen::Vector2d m_modal{ Eigen::Vector2d::Zero() }; ///< Physical modal response, including natural decay.
    size_t          m_cursor{ 0 }, m_delay{ 0 };        ///< Next command slot and integer delay steps.
    double          m_pole{ 0 };                        ///< Prepared pole; ignored by the explicitly ideal plant.
    bool            m_ideal{ false }; ///< Zero bandwidth selects the separate ideal-response convention.

  public:
    /// Prepare an explicitly calibrated synthetic plant, with no clocks, streams or simulator truth input.
    dmResponse( const Eigen::Matrix2d &mapping /**< [in] physical pointing per modal coefficient */,
                int64_t                periodNs /**< [in] fixed simulation time step [ns] */,
                int64_t                delayNs /**< [in] nonnegative pure delay, exactly divisible by periodNs */,
                double bandwidthHz /**< [in] first-order -3dB bandwidth; zero means ideal instantaneous response */ );

    /// Clear only this simulated plant, including its delayed command history.
    void reset();

    /// Advance exactly one simulation time step; invalid input clears the simulated plant and throws.
    std::array<double, 2> step( const std::array<double, 2> &command /**< [in] current held modal contribution */ );
};

inline feedForward::feedForward( const Eigen::Matrix2d &mapping, const correctionParameters &config )
    : m_mapping( mapping ), m_config( config )
{
    if( !mapping.allFinite() || !mapping.fullPivLu().isInvertible() || !std::isfinite( config.m_gain ) ||
        config.m_gain < 0 || config.m_gain > 1 || config.m_horizonNs < 0 || config.m_maxAgeNs <= 0 ||
        config.m_maxPredictionNs <= 0 || config.m_horizonNs > config.m_maxPredictionNs )
        throw std::invalid_argument( "invalid offline correction calibration/controls" );
    for( double limit : config.m_maxModal )
        if( !std::isfinite( limit ) || limit < 0 )
            throw std::invalid_argument( "invalid offline modal bound" );
}

inline correction feedForward::propose( std::span<const prediction> bands, int64_t now ) const
{
    correction result;
    if( m_config.m_gain == 0 )
    {
        result.m_state = correctionState::disabled;
        return result;
    }
    if( bands.empty() || now < 0 || now > std::numeric_limits<int64_t>::max() - m_config.m_horizonNs )
        return result;
    Eigen::Vector2d pointing = Eigen::Vector2d::Zero();
    const auto     &first    = bands.front();
    for( const auto &p : bands )
    {
        if( p.m_state != accelFFCtrlDetail::estimatorState::ready || p.m_sampleTimeNs <= 0 || p.m_sampleTimeNs > now ||
            now - p.m_sampleTimeNs > m_config.m_maxAgeNs || p.m_targetTimeNs != now + m_config.m_horizonNs ||
            p.m_targetTimeNs - p.m_sampleTimeNs > m_config.m_maxPredictionNs ||
            p.m_sampleTimeNs != first.m_sampleTimeNs || p.m_sequence != first.m_sequence ||
            p.m_generation != first.m_generation || !p.m_sequence || !p.m_generation || !std::isfinite( p.m_rcond ) ||
            p.m_rcond <= 0 )
            return result;
        for( size_t axis = 0; axis < 2; ++axis )
        {
            if( !std::isfinite( p.m_pointing[axis] ) || !std::isfinite( p.m_acceleration[axis] ) ||
                !std::isfinite( p.m_amplitude[axis] ) || p.m_amplitude[axis] < 0 || !std::isfinite( p.m_phase[axis] ) ||
                !std::isfinite( p.m_offset[axis] ) )
                return result;
            pointing[axis] += p.m_pointing[axis];
        }
    }
    const Eigen::Vector2d modal = -m_config.m_gain * ( m_mapping * pointing );
    if( !modal.allFinite() )
        return result;
    double scale = 1;
    for( size_t axis = 0; axis < 2; ++axis )
        if( std::abs( modal[axis] ) > m_config.m_maxModal[axis] )
            scale = std::min( scale, m_config.m_maxModal[axis] / std::abs( modal[axis] ) );
    for( size_t axis = 0; axis < 2; ++axis )
        result.m_modal[axis] = scale * modal[axis];
    result.m_state    = scale < 1 ? correctionState::limited : correctionState::active;
    result.m_targetNs = now + m_config.m_horizonNs;
    result.m_scale    = scale;
    return result;
}

inline dmResponse::dmResponse( const Eigen::Matrix2d &mapping, int64_t periodNs, int64_t delayNs, double bandwidthHz )
    : m_mapping( mapping ), m_ideal( bandwidthHz == 0 )
{
    if( !mapping.allFinite() || periodNs <= 0 || delayNs < 0 || delayNs % periodNs || !std::isfinite( bandwidthHz ) ||
        bandwidthHz < 0 || delayNs / periodNs > 1000000 )
        throw std::invalid_argument( "invalid offline plant calibration/timing" );
    m_delay = size_t( delayNs / periodNs );
    m_history.assign( m_delay + 2, Eigen::Vector2d::Zero() );
    m_pole = std::exp( -6.28318530717958647692 * bandwidthHz * ( periodNs * 1e-9 ) );
}

inline void dmResponse::reset()
{
    std::fill( m_history.begin(), m_history.end(), Eigen::Vector2d::Zero() );
    m_modal.setZero();
    m_cursor = 0;
}

inline std::array<double, 2> dmResponse::step( const std::array<double, 2> &command )
{
    const Eigen::Vector2d input( command[0], command[1] );
    if( !input.allFinite() )
    {
        reset();
        throw std::invalid_argument( "nonfinite offline plant command" );
    }
    m_history[m_cursor] = input;
    const size_t index  = ( m_cursor + m_history.size() - m_delay - unsigned( !m_ideal ) ) % m_history.size();
    m_modal             = m_ideal ? m_history[index] : ( m_pole * m_modal + ( 1 - m_pole ) * m_history[index] ).eval();
    m_cursor            = ( m_cursor + 1 ) % m_history.size();
    const Eigen::Vector2d pointing = m_mapping * m_modal;
    if( !pointing.allFinite() )
    {
        reset();
        throw std::overflow_error( "offline plant response overflow" );
    }
    return { pointing[0], pointing[1] };
}

} // namespace accelFFCtrlOffline
} // namespace app
} // namespace MagAOX
#endif
