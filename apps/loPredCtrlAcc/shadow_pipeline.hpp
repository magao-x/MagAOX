/** \file shadow_pipeline.hpp
 * \brief Platform-independent application input/prediction path with confirmed-input-only learning.
 */
#ifndef PCSHADOW_PIPELINE_HPP
#define PCSHADOW_PIPELINE_HPP
#include "causal_input.hpp"
#include "shadow_policy.hpp"

namespace DDSPC
{
/// Decoded ADC evidence in an explicitly established common clock and calibration version.
struct ShadowAdcRecord
{
    AdcSample     m_sample; ///< Owned result of actual AdcCapture processing, never a model proposal.
    std::uint64_t m_clockEpoch{ 0 }, m_calibration{ 0 }; ///< Immutable session mappings established by the adapter.
    std::int64_t  m_uncertaintyNs{ 0 };                  ///< Acquisition/clock bound established by the adapter.
    bool          m_snapshotValid{ false }; ///< Decoder and adapter established payload/metadata consistency.
};

/// Scalar-only snapshot; eligibility is diagnostic and cannot authorize a physical output.
struct ShadowPipelineStatus
{
    InputReason   m_inputReason{ InputReason::AwaitingRecords }; ///< Common record pairing status.
    FeatureReason m_featureReason{ FeatureReason::Incomplete };  ///< Current frozen ADC window status.
    PolicyMode m_requested{ PolicyMode::Standard }, m_effective{ PolicyMode::Safe }; ///< Intent and eligible candidate.
    const char *m_reason{ "awaiting_origin" };                  ///< Stable fallback/recovery/unavailability reason.
    bool     m_candidateValid{ false }, m_commonValid{ false }; ///< Current decision eligibility, not output authority.
    double   m_step{ 0 }; ///< Maximum per-mode change of the diagnostic candidate, including unavailability ramps.
    unsigned m_validStreak{ 0 }; ///< Consecutive fully qualified assisted decisions.
    std::uint64_t m_index{ 0 }, m_decisions{ 0 }, m_fallbacks{ 0 },
        m_recoveries{ 0 };                                 ///< Decision/transition evidence.
    std::uint64_t m_adcGeneration{ 0 }, m_adcSegment{ 0 }; ///< Current decoded source incarnation.
    std::int64_t  m_originNs{ -1 }, m_featureNs{ -1 },
        m_availableNs{ -1 }; ///< Frozen decision, effective acquisition, receipt.
};

/// Single-owner bounded orchestration; no runtime, clock reads, output sink, INDI, or native transport.
/** A WFS origin freezes ADC features. Only CausalInputBuilder's later confirmed records reach process().
 * The borrowed controller outlives this object. Callers serialize all entry points.
 */
class ShadowPipeline
{
    ShadowController      &m_controller;    ///< Borrowed actual independent learners; never trained on candidates.
    ShadowConfig           m_models;        ///< Immutable history and mode dimensions, matching the borrowed pair.
    InputConfig            m_inputConfig;   ///< Explicit source, modal and clock contract.
    FeatureConfig          m_featureConfig; ///< Separate ADC coverage and freshness limits.
    CausalInputBuilder     m_inputs; ///< Historical origin/confirmed-input pairing with conservative exposure coverage.
    ShadowPolicy           m_policy; ///< Shared fallback, hysteresis and diagnostic slew limiter.
    std::uint64_t          m_calibration;    ///< Required ADC calibration identity, independent of source restarts.
    std::vector<AdcSample> m_adc, m_scratch; ///< Bounded chronological ADC ring and preallocated window copy.
    std::size_t            m_adcNext{ 0 }, m_adcCount{ 0 }; ///< Ring insertion and retained count.
    AdcSample              m_lastAdc; ///< High-water metadata, used to reject duplicates and source discontinuities.
    bool         m_haveAdc{ false }, m_havePair{ false }, m_clockFault{ false }; ///< Source/history/clock validity.
    std::int64_t m_lastNow{ -1 };        ///< Common-clock monotonic admission high-water.
    Matrix       m_y, m_u, m_a;          ///< Bounded completed-record histories, indexed by their own circular head.
    std::vector<unsigned char> m_aValid; ///< Feature history validity in the current ADC incarnation.
    std::size_t                m_head{ 0 }, m_count{ 0 }, m_depth; ///< Completed-record ring and capacity.
    std::uint64_t              m_lastPair{ 0 }; ///< Latest confirmed origin identity, never a proposal sequence.
    Matrix m_standardState, m_assistedState, m_previous, m_zero; ///< Preallocated prediction and diagnostic workspaces.
    TimedWfsRecord       m_origin;   ///< Owned current WFS origin, whose caller-supplied ADC fields are overwritten.
    AccelFeature         m_feature;  ///< Reused shared aggregate, frozen before delayed input confirmation.
    ControlEvidence      m_evidence; ///< Actual cached-prediction admission and candidates.
    ControlDecision      m_decision; ///< Diagnostic candidate only; never used to construct an AppliedInputRecord.
    ShadowPipelineStatus m_status;   ///< Coherent scalar diagnostics for this origin.

    /// Reject monotonic clock reversal, invalidating both histories until a new pipeline/session is configured.
    bool admit_time( std::int64_t nowNs /**< [in] Common monotonic time. */ );

    /// Clear only assisted feature history after ADC loss/restart; common learning remains independent.
    void break_adc();

    /// Invalidate common histories/candidates without resetting learned coefficients or requested training.
    void break_common();

    /// Consume only builder-certified learning records, preserving actual input and origin-frozen features.
    bool consume( InputBuildResult result /**< [in] Pairing evidence and any common discontinuity. */ );

    /// Fill a canonical current state from WFS plus completed historical records; never future measurements.
    void state( Matrix       &output /**< [out] Preallocated canonical state. */,
                const Matrix &current /**< [in] Current measured WFS residual. */,
                bool          assisted /**< [in] Include current/past decoded acceleration. */ );

  public:
    /// Validate a complete contract and allocate bounded storage; no inputs, learners or hardware are started.
    ShadowPipeline(
        ShadowController    &controller /**< [in,out] Actual owning app's independent learners. */,
        const ShadowConfig  &models /**< [in] Matching learner dimensions. */,
        const InputConfig   &inputs /**< [in] Explicitly synthetic or externally verified input contract. */,
        const FeatureConfig &features /**< [in] ADC timestamp-window policy. */,
        const PolicyConfig  &policy /**< [in] Candidate bounds and recovery hysteresis. */,
        PolicyMode           requested /**< [in] Initial diagnostic request for this session. */,
        std::uint64_t        calibration /**< [in] Expected nonzero ADC calibration identity. */,
        std::size_t          capacity = 128 /**< [in] Retained ADC samples, at least enough for a feature window. */ );

    /// Retain only consistent, available decoded ADC input; source restarts break assistance, not standard learning.
    bool push_adc( const ShadowAdcRecord &record /**< [in] Actual decoded sample and established mappings. */,
                   std::int64_t           nowNs /**< [in] Frontend availability time. */ );

    /// Freeze one current origin, submit completed historical records, and generate an in-memory candidate.
    bool push_wfs( const TimedWfsRecord &record /**< [in] Current WFS exposure/identity; ADC fields are ignored. */,
                   std::int64_t          nowNs /**< [in] Must equal this origin's frozen decision time. */ );

    /// Accept an independently confirmed input interval; this is the only source of learning commands.
    bool push_command( const AppliedInputRecord &record /**< [in] Synthetic or externally verified held input. */,
                       std::int64_t              nowNs /**< [in] Confirmation receipt time. */ );

    /// Expire silent/stale common sources; no decision can be made retroactively.
    bool advance( std::int64_t nowNs /**< [in] Current common monotonic time. */ );

    /// Explicitly invalidate admission/history after raw callbacks, resets or shutdown.
    void invalidate();

    /// Change the requested mode; the next decision still enforces every qualification and slew gate.
    void request( PolicyMode mode /**< [in] Standard or Assisted, without changing either learned model. */ );

    /// Return scalar diagnostics without allocating or publishing anything.
    ShadowPipelineStatus status() const;

    /// Return provenance supplied by the immutable input contract.
    InputEvidence evidence() const;

    /// Copy the current candidate into caller-owned memory, only when it is qualified.
    bool candidate( Matrix &output /**< [out] Pre-sized mode column; zeroed when unavailable. */ ) const;
};
} // namespace DDSPC
#endif
