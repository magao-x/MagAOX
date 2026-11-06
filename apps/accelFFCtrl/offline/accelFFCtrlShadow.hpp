/** \file accelFFCtrlShadow.hpp
 * \brief Host-memory-only absolute modal proposals with source, time and calibration provenance.
 * \ingroup accelFFCtrl_files
 */
#ifndef accelFFCtrlShadow_hpp
#define accelFFCtrlShadow_hpp

#include "accelFFCtrlCancellation.hpp"

namespace MagAOX
{
namespace app
{
namespace accelFFCtrlOffline
{

/// Numerical evidence label only; none of these records authorizes an actuator connection.
enum class shadowCalibration
{
    synthetic,
    archivedPupilOnly
};

/// Additional shadow lifecycle faults; ordinary numerical inhibition remains in correction::m_state.
enum class shadowFault
{
    none,
    queryRollback,
    sourceChange,
    counterRollback,
    resetBarrier,
    sampleChange
};

/// One complete absolute proposal, including explicit zeros and exact integer source/time identities.
struct shadowRecord
{
    uint64_t m_record{ 0 }, m_epoch{ 0 }, m_inputRevision{ 0 }, m_calibrationId{ 0 }; ///< Log identity/provenance.
    uint64_t m_sourceCounter{ 0 }, m_sourceGeneration{ 0 }; ///< Prediction source, not DM publication counters.
    int64_t  m_queryNs{ 0 }, m_sampleNs{ 0 }, m_predictionTargetNs{ 0 }, m_requestedTargetNs{ 0 }; ///< Monotonic ns.
    std::array<double, 2> m_pointing{},
        m_dmPointing{};    ///< Numerically valid input sum and mapped DM-pupil radians, including at zero gain.
    correction m_proposal; ///< Replaces the entire contribution; never accumulated, queued or published.
    bool m_validPrediction{ false }; ///< Numerically qualified prediction, also audited when proposal gain is zero.
    shadowFault       m_fault{ shadowFault::none }; ///< Latched source/query fault, or current reset barrier.
    shadowCalibration m_evidence{ shadowCalibration::synthetic }; ///< Manifest must identify the supplied maps.
};

/// Offline wrapper around the existing numerical proposer; no filesystem, shared memory, clocks or actuator API.
/** Coordinate map C must explicitly map prediction radians to the calibrated DM-pupil axes.
 * K is the supplied coefficient/radian modal map; the proposer receives K*C.
 * Generation/counter rollback and query-time rollback latch inhibition until reset().
 * Counter advances between prediction queries are normal and do not count ADC loss.
 */
class shadowCommand
{
    feedForward          m_proposer;    ///< Existing gain, numerical validity, horizon and common-scalar bounds.
    feedForward          m_validator;   ///< Zero-bound numerical validation for disabled-gain diagnostics only.
    Eigen::Matrix2d      m_coordinates; ///< Explicit C; no assumed sky/WFS rotation or reflection.
    correctionParameters m_config;      ///< Immutable experiment settings; changes require a new owner/manifest.
    shadowCalibration    m_evidence;    ///< Synthetic or archived-pupil-only qualification, never physical enable.
    uint64_t             m_calibrationId, m_record{ 0 }, m_epoch{ 0 },
        m_revision{ 0 };                              ///< Stable manifest key and lifetime log state.
    uint64_t    m_generation{ 0 }, m_counter{ 0 };    ///< Last accepted proposal's physical source identity.
    int64_t     m_lastQuery{ -1 }, m_resetTime{ -1 }; ///< Monotonic query ordering and post-reset acquisition barrier.
    int64_t     m_lastSample{ -1 }; ///< An accepted source counter retains one immutable acquisition timestamp.
    shadowFault m_fault{ shadowFault::none }; ///< Integrity fault latch, cleared only by explicit reset.

    /// Preserve all timing bounds and force a zero-output numerical audit independent of proposal gain.
    static correctionParameters validityParameters( const correctionParameters &config /**< [in] proposal settings */ );

  public:
    /// Construct with separately explicit coordinate/modal maps and a caller-owned calibration manifest identity.
    shadowCommand( const Eigen::Matrix2d &coordinates /**< [in] pointing radians to DM-pupil radians */,
                   const Eigen::Matrix2d &modalMap /**< [in] modal coefficients per DM-pupil radian */,
                   uint64_t calibrationId /**< [in] nonzero identity keyed to an external provenance manifest */,
                   shadowCalibration           evidence /**< [in] numerical fixture qualification only */,
                   const correctionParameters &config = {} /**< [in] gain defaults to zero; finite explicit bounds */ );

    /// Clear source latches, advance epoch, and require an acquisition newer than this reset before nonzero output.
    void reset( int64_t  now /**< [in] current monotonic ns, at or after the last query */,
                uint64_t inputRevision /**< [in] owner-applied estimator revision, logged without changing maps */ );

    /// Produce a new absolute host-memory record; every disabled, faulted or unavailable record contains zero.
    shadowRecord propose( std::span<const prediction> bands /**< [in] unchanged numerical prediction interface */,
                          int64_t                     now /**< [in] actual query monotonic ns */ );
};

inline correctionParameters shadowCommand::validityParameters( const correctionParameters &config )
{
    auto result       = config;
    result.m_gain     = 1;
    result.m_maxModal = { 0, 0 };
    return result;
}

inline shadowCommand::shadowCommand( const Eigen::Matrix2d      &coordinates,
                                     const Eigen::Matrix2d      &modalMap,
                                     uint64_t                    calibrationId,
                                     shadowCalibration           evidence,
                                     const correctionParameters &config )
    : m_proposer( modalMap * coordinates, config ), m_validator( modalMap * coordinates, validityParameters( config ) ),
      m_coordinates( coordinates ), m_config( config ), m_evidence( evidence ), m_calibrationId( calibrationId )
{
    if( !calibrationId || !coordinates.allFinite() || !coordinates.fullPivLu().isInvertible() ||
        !modalMap.allFinite() || !modalMap.fullPivLu().isInvertible() )
        throw std::invalid_argument( "invalid shadow coordinate/modal provenance" );
}

inline void shadowCommand::reset( int64_t now, uint64_t inputRevision )
{
    if( now < 0 || now < m_lastQuery || inputRevision < m_revision || m_epoch == std::numeric_limits<uint64_t>::max() )
        throw std::invalid_argument( "invalid shadow reset ordering" );
    ++m_epoch;
    m_revision  = inputRevision;
    m_resetTime = m_lastQuery = now;
    m_generation = m_counter = 0;
    m_lastSample             = -1;
    m_fault                  = shadowFault::none;
}

inline shadowRecord shadowCommand::propose( std::span<const prediction> bands, int64_t now )
{
    if( m_record == std::numeric_limits<uint64_t>::max() )
        throw std::overflow_error( "shadow log identity overflow" );
    shadowRecord result;
    result.m_record        = ++m_record;
    result.m_epoch         = m_epoch;
    result.m_inputRevision = m_revision;
    result.m_calibrationId = m_calibrationId;
    result.m_evidence      = m_evidence;
    result.m_queryNs       = now;
    if( now >= 0 && now <= std::numeric_limits<int64_t>::max() - m_config.m_horizonNs )
        result.m_requestedTargetNs = now + m_config.m_horizonNs;
    if( !bands.empty() )
    {
        const auto &p               = bands.front();
        result.m_sourceCounter      = p.m_sequence;
        result.m_sourceGeneration   = p.m_generation;
        result.m_sampleNs           = p.m_sampleTimeNs;
        result.m_predictionTargetNs = p.m_targetTimeNs;
    }
    if( now < 0 || now < m_lastQuery )
        m_fault = shadowFault::queryRollback;
    else
        m_lastQuery = now;
    if( m_fault != shadowFault::none )
    {
        result.m_fault = m_fault;
        return result;
    }
    const auto proposal = m_proposer.propose( bands, now );
    const auto validity = proposal.m_state == correctionState::disabled ? m_validator.propose( bands, now ) : proposal;
    if( validity.m_state == correctionState::active || validity.m_state == correctionState::limited )
    {
        if( m_generation && result.m_sourceGeneration != m_generation )
            m_fault = shadowFault::sourceChange;
        else if( m_generation && result.m_sourceCounter < m_counter )
            m_fault = shadowFault::counterRollback;
        else if( m_generation && ( result.m_sampleNs < m_lastSample ||
                                   ( result.m_sourceCounter == m_counter && result.m_sampleNs != m_lastSample ) ||
                                   ( result.m_sourceCounter > m_counter && result.m_sampleNs <= m_lastSample ) ) )
            m_fault = shadowFault::sampleChange;
        if( m_fault != shadowFault::none )
        {
            result.m_fault = m_fault;
            return result;
        }
        if( result.m_sampleNs <= m_resetTime )
        {
            result.m_fault = shadowFault::resetBarrier;
            return result;
        }
        m_generation = result.m_sourceGeneration;
        m_counter    = result.m_sourceCounter;
        m_lastSample = result.m_sampleNs;
        for( const auto &p : bands )
            for( size_t axis = 0; axis < 2; ++axis )
                result.m_pointing[axis] += p.m_pointing[axis];
        const Eigen::Vector2d mapped = m_coordinates * Eigen::Vector2d( result.m_pointing[0], result.m_pointing[1] );
        if( !mapped.allFinite() )
            return result;
        result.m_dmPointing      = { mapped[0], mapped[1] };
        result.m_validPrediction = true;
    }
    result.m_proposal = proposal;
    return result;
}

} // namespace accelFFCtrlOffline
} // namespace app
} // namespace MagAOX
#endif
