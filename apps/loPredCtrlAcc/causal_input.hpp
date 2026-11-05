/** \file causal_input.hpp
 * \brief Bounded, stream-independent pairing of explicit WFS exposures and verified applied-input intervals.
 */
#ifndef PCCINPUT_HPP
#define PCCINPUT_HPP

#include <deque>
#include "shadow_controller.hpp"

namespace DDSPC
{

/// Provenance asserted by a recording adapter; prepared/accepted commands do not prove physical application.
enum class InputEvidence
{
    Unverified,
    Requested,
    DriverPrepared,
    DriverAccepted,
    VerifiedApplied,
    SyntheticApplied
};

/// Return the declared provenance without conflating synthetic data or SDK acceptance with physical readback.
const char *input_evidence( InputEvidence evidence /**< [in] Source provenance. */ );

/// Explain the latest builder outcome without claiming native stream verification.
enum class InputReason
{
    UnverifiedContract,
    AwaitingRecords,
    Ready,
    Reset,
    InvalidPayload,
    InvalidTime,
    WrongIdentity,
    UnverifiedCommand,
    OutOfOrder,
    CounterGap,
    TimingGap,
    MissingCoverage,
    AmbiguousExposure,
    UnsupportedLatency,
    Stale,
    Overflow,
    ClockReversal
};

/// Return a stable diagnostic description of input synchronization state.
const char *input_reason( InputReason reason /**< [in] Builder outcome. */ );

/// Immutable discrete-time and source contract, supplied by an independently verified recording adapter.
/** All nanoseconds are in one established monotonic clock epoch. No wall-clock conversion or delay estimation
 * is performed here. Changing source generation, clock mapping, or modal calibration requires a new builder.
 */
struct InputConfig
{
    /// Residual and calibrated absolute applied-input mode count.
    int m_modes{ 1 };

    /// Optional prebuilt feature count; zero disables assisted features in records, not the standard model.
    int m_accelChannels{ 0 };

    /// Expected WFS stream generation, never inferred from a counter reset.
    std::uint64_t m_wfsGeneration{ 0 };

    /// Expected applied-input source generation.
    std::uint64_t m_commandGeneration{ 0 };

    /// Established common monotonic clock mapping/version, including any clock-offset calibration.
    std::uint64_t m_clockEpoch{ 0 };

    /// Shared modal ordering/units/calibration version for both residuals and projected applied inputs.
    std::uint64_t m_layout{ 0 };

    /// Expected WFS exposure-start spacing in nanoseconds.
    std::int64_t m_periodNs{ 0 };

    /// Expected exposure duration in nanoseconds.
    std::int64_t m_exposureNs{ 0 };

    /// Allowed duration/spacing deviation, strictly below half a period.
    std::int64_t m_jitterNs{ 0 };

    /// Maximum age of pending/current evidence, including the one-frame assembly delay.
    std::int64_t m_maxAgeNs{ 0 };

    /// Maximum accepted uncertainty of exposure and applied-interval boundaries.
    std::int64_t m_maxUncertaintyNs{ 0 };

    /// Established optical-response lag estimate; its bounded error must be included in WFS uncertainty.
    std::int64_t m_responseLagNs{ 0 };

    /// Minimum decision-to-application delay for a potentially changeable next input.
    std::int64_t m_minCommandDelayNs{ 0 };

    /// Bound each pending source queue; overflow breaks histories rather than overwriting evidence.
    std::size_t m_capacity{ 128 };

    /// Explicitly synthetic or externally verified provenance; no default trust.
    InputEvidence m_evidence{ InputEvidence::Unverified };

    /// Exposure/clock/latency interpretation has been established independently.
    bool m_timingVerified{ false };

    /// Projection, units, sign, absolute-input convention and included DM channels are established.
    bool m_mappingVerified{ false };
};

/// Common metadata for immutable, consistently captured source records.
struct InputStamp
{
    /// Source instance identity, not the ImageStreamIO slot number.
    std::uint64_t m_generation{ 0 };

    /// Source frame/certificate sequence; must advance by one within each source generation.
    std::uint64_t m_counter{ 0 };

    /// Common clock mapping/version; REALTIME is not silently equated with MONOTONIC.
    std::uint64_t m_clockEpoch{ 0 };

    /// Modal calibration/order/units version.
    std::uint64_t m_layout{ 0 };

    /// Producer availability/publication time in the established clock.
    std::int64_t m_availableNs{ 0 };

    /// Local acquisition of this immutable record, separate from physical measurement/application.
    std::int64_t m_receivedNs{ 0 };

    /// Boundary uncertainty, including clock mapping and (for WFS) exposure/response-lag calibration error.
    std::int64_t m_uncertaintyNs{ 0 };

    /// Adapter verified payload and metadata belong to one stable snapshot.
    bool m_snapshotValid{ false };
};

/// Residual y_i and the features actually available by origin i's frozen decision deadline.
struct TimedWfsRecord
{
    /// WFS source identity and availability metadata.
    InputStamp m_stamp;

    /// Observed modal residual, not a DM request or simulator state.
    Matrix m_residual;

    /// Physical half-open exposure window [start,end), independently established.
    std::int64_t m_exposureStartNs{ 0 }, m_exposureEndNs{ 0 };

    /// Frozen origin deadline, at or after local WFS availability.
    std::int64_t m_decisionNs{ 0 };

    /// Optional already synchronized origin feature vector; no raw accelerometer decoding is provided.
    Matrix m_acceleration;

    /// Physical acquisition and origin availability of optional features.
    std::int64_t m_accelAcquiredNs{ 0 }, m_accelAvailableNs{ 0 };

    /// Optional feature builder's causal/freshness assertion, independently rechecked against the deadline.
    bool m_accelValid{ false };
};

/// Closed, confirmed interval of actual held modal input, never extrapolated from a request or SDK success.
/** The adapter must prove completeness, clipping/combiner semantics, mapping and effective timing. Confirmation
 * is available only at/after the interval ends. SDK return alone cannot construct this certificate.
 */
struct AppliedInputRecord
{
    /// Applied-input certificate source identity and confirmation availability.
    InputStamp m_stamp;

    /// Constant absolute modal input over the entire certified interval, after limits/mapping.
    Matrix m_applied;

    /// Request time is diagnostic only and never used to select a WFS input.
    std::int64_t m_requestedNs{ 0 };

    /// Confirmed half-open physical hold interval, with bounded boundary uncertainty.
    std::int64_t m_effectiveStartNs{ 0 }, m_effectiveEndNs{ 0 };

    /// Explicit command provenance, checked against the session contract.
    InputEvidence m_evidence{ InputEvidence::Unverified };
};

/// Audit trail linking y_i/u_i to the following exposure; both learners receive the same m_record.
struct PairedInputRecord
{
    /// Phase 1 record: u_i is the applied input affecting y_(i+1).
    ShadowRecord m_record;

    /// Immutable origin identity and deadline.
    InputStamp m_origin;

    /// Following WFS frame whose exposure establishes this u_i association.
    InputStamp m_target;

    /// Exact applied certificate used, with its provenance.
    InputStamp m_command;

    /// Applied-input provenance, retained instead of labeling synthetic data as hardware readback.
    InputEvidence m_evidence{ InputEvidence::Unverified };

    /// Origin deadline and response-aligned following exposure window.
    std::int64_t m_decisionNs{ 0 }, m_inputStartNs{ 0 }, m_inputEndNs{ 0 };
};

/// One builder operation; a discontinuity must invalidate both model histories before any returned records.
struct InputBuildResult
{
    /// The supplied event/tick was accepted; waiting is distinct from rejecting data.
    bool m_accepted{ false };

    /// Both histories must break; previously learned coefficients may be retained.
    bool m_discontinuity{ false };

    /// Latest acceptance/waiting/rejection explanation.
    InputReason m_reason{ InputReason::AwaitingRecords };

    /// Chronological complete records, bounded by queue capacity.
    std::vector<PairedInputRecord> m_records;
};

/// Bounded single-owner shadow/replay builder; no stream, clock read, output sink, or hardware access.
/** Emits (y_i,u_i) only after exposure i+1 and its held input are confirmed. This extra assembly delay is
 * intentional for training/shadow diagnostics; this is not a real-time correction input adapter.
 * Counter/time/identity faults clear pending records. High-water counters persist until a new builder/session.
 */
class CausalInputBuilder
{
  private:
    /// Metadata high-water state, retained across invalidation to reject replayed observations.
    struct Cursor
    {
        /// A valid source record has been observed.
        bool m_have{ false };

        /// Last accepted source identity and availability.
        InputStamp m_stamp;

        /// Source physical window and WFS deadline.
        std::int64_t m_startNs{ 0 }, m_endNs{ 0 }, m_decisionNs{ 0 };
    };

    /// Immutable source and timing contract.
    InputConfig m_config;

    /// Bounded immutable WFS observations awaiting a certified next input.
    std::deque<TimedWfsRecord> m_wfs;

    /// Bounded non-overlapping verified hold intervals.
    std::deque<AppliedInputRecord> m_commands;

    /// Source-specific sequence/time high-water state.
    Cursor m_lastWfs, m_lastCommand;

    /// A local monotonic operation time has been observed.
    bool m_haveNow{ false };

    /// Clock reversal latches until a newly established session is constructed.
    bool m_clockFault{ false };

    /// Reject clock reversal rather than silently remapping timestamps.
    std::int64_t m_lastNowNs{ 0 };

    /// Latest synchronization outcome.
    InputReason m_reason{ InputReason::UnverifiedContract };

    /// Clear queues and all outputs from this operation, retaining source high-water evidence.
    void reject( InputBuildResult &result /**< [in,out] Operation to invalidate. */,
                 InputReason       reason /**< [in] Fault explanation. */ );

    /// Check contract trust, operation time and pending-source age before accepting an event.
    bool prepare( InputBuildResult &result /**< [in,out] Operation result. */,
                  std::int64_t      nowNs /**< [in] Current established monotonic time. */ );

    /// Validate source identity and common timestamp ordering without mutation.
    bool valid_stamp( const InputStamp &stamp /**< [in] Immutable metadata. */,
                      std::uint64_t     generation /**< [in] Expected source generation. */,
                      std::int64_t      nowNs /**< [in] Operation time. */ ) const;

    /// Emit only complete contiguous exposure/input associations; never average command transitions.
    void drain( InputBuildResult &result /**< [in,out] Operation and audit records. */,
                std::int64_t      nowNs /**< [in] Current established monotonic time. */ );

    /// Publish this operation's diagnostic state.
    InputBuildResult finish( InputBuildResult result /**< [in] Completed operation. */ );

  public:
    /// Validate numerical configuration; unverified provenance/timing remain gated on every input.
    explicit CausalInputBuilder( const InputConfig &config /**< [in] Fixed source/timing contract. */ );

    /// Accept an immutable WFS record and attempt chronological pairing.
    InputBuildResult push_wfs( const TimedWfsRecord &record /**< [in] Exposure, residual and origin features. */,
                               std::int64_t          nowNs /**< [in] Established monotonic operation time. */ );

    /// Accept a closed applied-input certificate and attempt chronological pairing.
    InputBuildResult push_command( const AppliedInputRecord &record /**< [in] Confirmed input interval. */,
                                   std::int64_t              nowNs /**< [in] Established monotonic operation time. */ );

    /// Detect stale pending/current input during silence; caller must schedule this in the established clock.
    InputBuildResult advance( std::int64_t nowNs /**< [in] Established monotonic operation time. */ );

    /// Explicitly break pending histories without permitting counter replay in this session.
    InputBuildResult invalidate();

    /// Return the latest builder outcome for application diagnostics.
    InputReason reason() const;

    /// Return the session's explicit source provenance, including synthetic/unverified states.
    InputEvidence evidence() const;
};

} // namespace DDSPC
#endif // PCCINPUT_HPP
