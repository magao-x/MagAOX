/** \file multirate_shadow.hpp
 * \brief Qualified multirate DDSPC assembly; raw publication never grants exposure or applied-input provenance.
 */
#ifndef PC_MULTIRATE_SHADOW_HPP
#define PC_MULTIRATE_SHADOW_HPP
#include "native_observation.hpp"
#include "causal_input.hpp"
#include <memory>

namespace DDSPC
{
/// Exact native incarnation binding; neither a stream name nor similar timestamps establish this identity.
struct NativeBinding
{
    std::uint64_t m_device{ 0 }, m_inode{ 0 }, m_image{ 0 }, m_generation{ 0 }; ///< File and publisher identity.

    /// Match one original native record, permitting absent image inode only for the ADC cursor.
    bool matches( const NativeObservation &record /**< [in] Immutable source snapshot. */ ) const;
};

/// Upstream owns the semaphore/delay phase lock; this consumer verifies its counter contract and live status.
struct UpstreamSyncContract
{
    bool m_referenceIsModal{ true }; ///< Use the measured WFS modal counter; no camera-parent reconstruction.
    bool m_enabled{ false }; ///< Explicit alternative to physical-exposure reconstruction, never inferred from rates.
    std::uint64_t m_epoch{ 0 }, m_modalAnchor{ 0 }, m_referenceAnchor{ 0 }, m_adcAnchor{ 0 }; ///< Reviewed 4:1 anchors.
    std::int64_t  m_statusMaxAgeNs{ 1000000000 }, m_featureMaxAgeNs{ 1000000 }; ///< Local receipt age limits.
    unsigned      m_recoveryFrames{ 8 }; ///< Consecutive complete locked frames before assistance can recover.
    bool          m_combinedRequestModel{
        false }; ///< Identify request-to-measurement dynamics, without claiming physical application.
};

/// Independently established source contract; deliberately not loadable from an INDI/configuration trust switch.
struct MultirateContract
{
    UpstreamSyncContract m_upstream; ///< Optional upstream synchronized measurement/command-index convention.
    InputConfig          m_inputs;   ///< Layout, local receipt epoch and declared command domain; strict legacy timing.
    std::array<NativeBinding, 4> m_sources; ///< ADC, modal, camera and combined-command source bindings.
    bool                         m_clockVerified{ false }, m_exposureVerified{ false }, m_associationVerified{ false },
        m_scheduleVerified{ false };    ///< Independent evidence; declarations alone do not authenticate a producer.
    std::string  m_provenance;          ///< Reviewed producer/clock/calibration/schedule evidence reference.
    std::int64_t m_wfsEndOffsetNs{ 0 }; ///< Verified exposure end relative to the command-grid origin.
    std::int64_t m_adcPhaseNs{ 0 };     ///< Verified ADC phase relative to that grid, not camera publication.
};

/// Admit a real physical or explicitly declared request-domain contract type, never synthetic/live promotion.
bool native_shadow_evidence( const MultirateContract &contract /**< [in] Reviewed adapter contract. */ );

/// Bounded worker settings; no device/output name or actuation enable exists.
struct OnlineShadowConfig
{
    ShadowConfig m_model; ///< Existing RLS/QR/law configuration; native models use four command slots.
    std::size_t  m_queueCapacity{ 2048 }, m_adcCapacity{ 4096 }, m_pendingCapacity{ 32 }; ///< Fixed memory bounds.
    unsigned     m_periodUs{ 5000 },
        m_budgetUs{ 1000 };               ///< Soft wall-time duty budget; individual solves are not preemptible.
    bool        m_expectUpstream{ true }; ///< Native admission convention; not a declaration that a producer is locked.
    std::size_t m_predictionCapacity{ 0 }; ///< Optional bounded scalar-score queue; zero disables tracing.
    std::string m_predictionFile;          ///< Optional exclusive private CSV, drained on an ordinary recorder thread.
    std::uint64_t m_predictionMaxRows{ 2000000 }; ///< Hard recording bound; reaching it never changes model inputs.
};

/// WFS origin with upstream counter evidence or strict camera/exposure evidence; synco cutoffs remain independent.
struct OnlineOrigin
{
    InputStamp m_stamp; ///< Modal source and mapped availability/receipt; snapshot validity is independently checked.
    std::uint64_t        m_cameraCounter{ 0 }; ///< Actual camera-origin association, never guessed from cnt1.
    std::array<float, 2> m_residual{};         ///< Observed selected calibrated modes, never interpolated/up-sampled.
    std::int64_t m_exposureStartNs{ 0 }, m_exposureEndNs{ 0 }, m_gridNs{ 0 }, m_decisionNs{ 0 }; ///< Distinct events.
    std::array<std::int64_t, 8> m_cutoffsNs{}; ///< Actual current/next frame input-ready cutoffs, not slack estimates.
    unsigned m_prefixSlots{ 0 }; ///< Zero to four already committed oldest slots, excluded from optimization.
    std::array<float, 8>
                 m_prefix{}; ///< Chronological slot-major modal values, separate from future applied certificates.
    std::int64_t m_prefixAvailableNs{ 0 }; ///< When the immutable prefix became known.
    bool m_prefixVerified{ false }; ///< Independent schedule/commit proof, not inferred from command publication.
    std::uint64_t m_syncEpoch{ 0 },
        m_firstAdcCounter{ 0 };                      ///< Upstream counter-phase witness for this reference frame.
    bool m_synchronized{ false }, m_locked{ false }; ///< Observed upstream mode and lock, never a DDSPC lock algorithm.
    std::int64_t m_syncStatusReceivedNs{
        -1 }; ///< Local receipt of the producer's status, not a producer clock mapping.
    std::int64_t m_predictionDeadlineNs{ -1 }; ///< Optional diagnostic compute deadline, separate from DM slot cutoffs.
};

/// Complete observed combined-request input for one WFS frame; it is not physically applied-command telemetry.
struct OnlineCommandFrame
{
    InputStamp    m_stamp; ///< Certificate sequence/layout/local receipt epoch; separate from source request counts.
    std::uint64_t m_wfsCounter{ 0 }; ///< Explicit frame/slot sampling association from the command-history adapter.
    std::array<std::uint64_t, 4> m_counters{}, m_hashes{}; ///< Original combined snapshots; a held request may repeat.
    std::array<std::int64_t, 4>  m_sampledNs{}; ///< Actual local sampling cutoffs of the declared request operator.
    std::array<float, 8> m_values{}; ///< Chronological slot-major calibrated modal requests, never two DM pixels.
    bool m_complete{ false };        ///< Adapter established complete request history and its slot sampling convention.
    InputEvidence m_evidence{
        InputEvidence::Unverified }; ///< Requested or explicitly synthetic, never promoted to applied.
};

/// Closed physical-input certificate; raw combined requests or SDK returns cannot construct this record.
struct OnlineAppliedSlot
{
    InputStamp    m_stamp;          ///< Independently established certificate identity and confirmation receipt.
    std::uint64_t m_dmCounter{ 0 }; ///< Original combined request identity, also required in retained observations.
    std::int64_t  m_startNs{ 0 }, m_endNs{ 0 }; ///< Confirmed effective hold interval, after limits/response mapping.
    std::array<float, 2> m_values{}; ///< Calibrated total applied modal input, not two arbitrarily selected DM pixels.
    InputEvidence        m_evidence{ InputEvidence::Unverified }; ///< Physical or explicitly synthetic provenance.
};

/// Non-actuating worker readback, including precise qualification and computational admission state.
struct OnlineShadowStatus
{
    ShadowStatus m_standard, m_assisted; ///< Independent actual native learner states.
    bool m_qualified{ false }, m_clock{ false }, m_exposure{ false }, m_association{ false }, m_schedule{ false },
        m_candidateValid{ false }, m_running{ false }, m_overrun{ false }; ///< Separate trust/readiness/deadline gates.
    bool m_upstream{ false },
        m_syncValid{ false }; ///< Counter contract selected; current assistance lock/history state.
    bool m_predictionValid{ false }, m_predictionLate{ false }; ///< Diagnostic state, independent of eligible DM plans.
    InputEvidence m_inputEvidence{ InputEvidence::Unverified }; ///< Preserve requested/applied distinction.
    std::uint64_t m_forecasts{ 0 },
        m_lateForecasts{ 0 }; ///< Frozen diagnostic origins, distinct from eligible DM plans.
    const char   *m_reason{ "source_host_mapping_unqualified" }; ///< Stable inhibition description.
    ShadowMode    m_requested{ ShadowMode::Standard }, m_effective{ ShadowMode::Standard }; ///< Request/fallback state.
    std::uint64_t m_raw{ 0 }, m_records{ 0 }, m_candidates{ 0 }, m_drops{ 0 },
        m_rejections{ 0 };                           ///< Distinct work/loss.
    double        m_computeUs{ -1 }, m_peakUs{ -1 }; ///< Actual worker model time; unavailable before numerical work.
    std::size_t   m_rawQueued{ 0 }, m_qualifiedQueued{ 0 }; ///< Pending work, not retained source history.
    std::size_t   m_predictionsQueued{ 0 }; ///< Pending diagnostic scores, independent of acquisition queues.
    std::uint64_t m_predictionDrops{ 0 };   ///< Diagnostic losses never invalidate or fabricate model input.
    const char   *m_adapterReason{ "native_adapter_disabled" };                 ///< Actual live packet assembly gate.
    std::uint64_t m_preparedPlans{ 0 }, m_shadowSlots{ 0 }, m_missedSlots{ 0 }; ///< Private producer diagnostics.
    double m_callbackUs{ -1 }, m_callbackPeakUs{ -1 }; ///< Copy/validation only, excluding model work and SDK writes.
    std::uint64_t m_traceRows{ 0 };                    ///< Actually written numerical records, not queue occupancy.
    bool          m_traceIoError{ false },
        m_traceLimit{ false };                          ///< Separate recorder failures/limits, never training evidence.
    std::int64_t  m_originNs{ -1 }, m_deadlineNs{ -1 }; ///< Actual latest frozen origin/cutoff, not a clock mapping.
    std::uint64_t m_fastOpportunities{ 0 }, m_fastPredictions{ 0 }, m_wfsReceived{ 0 }; ///< Distinct multirate events.
    std::array<std::uint64_t, 4> m_phasePredictions{}; ///< Accepted fresh ADC predictions at each upstream phase.
    double m_fastUs{ -1 }, m_fastPeakUs{ -1 };         ///< Measured fast-owner inference/render latency, not WCET.
};

/// Serialized qualified assembler reused by the native worker and deterministic tests; no I/O or publisher API.
class MultirateShadow
{
    struct Impl;
    std::unique_ptr<Impl> m_impl; ///< Sole bounded histories, independent models and retained diagnostic plans.

  public:
    /// Validate/allocate; production permits only externally verified physical evidence.
    explicit MultirateShadow(
        const OnlineShadowConfig &config /**< [in] Bounded numerical settings. */,
        InputEvidence evidence = InputEvidence::VerifiedApplied /**< [in] Explicit test-only synthetic opt-in. */ );

    /// Destroy only owned numerical state.
    ~MultirateShadow();

    /// Forward optional scalar diagnostics from the same native learners, with no extra prediction calculation.
    void prediction_sink( ShadowPredictionSink *sink /**< [in] Borrowed nonblocking consumer. */ );

    /// Install a reviewed source contract; unknown/missing evidence leaves both learners inhibited.
    bool qualify( const MultirateContract &contract /**< [in] Independent immutable provenance. */ );

    /// Retain individual source records and invalidate histories on loss/restart; never infer qualification.
    void observe( const NativeObservation &record /**< [in] Original ADC/WFS/camera/DM observation. */ );

    /// Freeze causal individual ADC history and, if qualified/ready, calculate next-origin diagnostic plans.
    bool origin( const OnlineOrigin &record /**< [in] Actual source/exposure/schedule record. */,
                 std::int64_t        nowNs /**< [in] Established processing time, at/after receipt. */ );

    /// Confirm a complete held slot and train only after four independent certificates cover a frame.
    bool applied( const OnlineAppliedSlot &record /**< [in] Actual applied input, never a generated proposal. */,
                  std::int64_t             nowNs /**< [in] Established confirmation processing time. */ );

    /// Train from complete counter-associated combined requests in the explicitly declared request-domain model.
    bool commands( const OnlineCommandFrame &record /**< [in] Independently observed request trajectory. */,
                   std::int64_t              nowNs /**< [in] Local processing time. */ );

    /// Expire stale input/candidates and reject computation completed after a slot's input-ready cutoff.
    void advance( std::int64_t nowNs /**< [in] Actual established monotonic time. */ );

    /// Reject late computation before publishing its readback; ordinary later expiry must not erase valid scores.
    void finish_origin( std::int64_t nowNs /**< [in] Actual completion time of the latest origin calculation. */ );

    /// Break histories after queue loss, preserving coefficients but revoking readiness.
    void invalidate( const char *reason /**< [in] Stable diagnostic literal. */ );

    /// Request diagnostic selection; assistance may fall back only to a qualified ready standard model.
    void request( ShadowMode mode /**< [in] Standard or assisted; no output authority. */ );

    /// Return a bounded readback; caller serializes access.
    OnlineShadowStatus status() const;

    /// Copy the full newest-first future plan only while its current qualified deadline remains valid.
    bool candidate( ShadowMode mode /**< [in] Independent model. */,
                    Matrix    &plan /**< [out] Owned diagnostic copy. */ ) const;

    /// Compute an uncommitted hypothetical plan ahead of polling; this grants no schedule or output eligibility.
    bool prepare_plan( ShadowMode mode /**< [in] Independent ready model. */,
                       Matrix    &plan /**< [out] Complete newest-first private plan. */ );

    /// Inspect actual canonical coefficients for deterministic independence/indexing tests.
    Matrix coefficients( ShadowMode mode /**< [in] Independent model. */ );
};
} // namespace DDSPC
#endif
