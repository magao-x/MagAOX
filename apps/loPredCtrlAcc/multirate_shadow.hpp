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

/// Independently established source contract; deliberately not loadable from an INDI/configuration trust switch.
struct MultirateContract
{
    InputConfig m_inputs; ///< Clock/layout, exposure, uncertainty, response and applied-evidence contract.
    std::array<NativeBinding, 4> m_sources; ///< ADC, modal, camera and combined-command source bindings.
    bool                         m_clockVerified{ false }, m_exposureVerified{ false }, m_associationVerified{ false },
        m_scheduleVerified{ false };    ///< Independent evidence; declarations alone do not authenticate a producer.
    std::string  m_provenance;          ///< Reviewed producer/clock/calibration/schedule evidence reference.
    std::int64_t m_wfsEndOffsetNs{ 0 }; ///< Verified exposure end relative to the command-grid origin.
    std::int64_t m_adcPhaseNs{ 0 };     ///< Verified ADC phase relative to that grid, not camera publication.
};

/// Bounded worker settings; no device/output name or actuation enable exists.
struct OnlineShadowConfig
{
    ShadowConfig m_model; ///< Existing RLS/QR/law configuration; native models use four command slots.
    std::size_t  m_queueCapacity{ 2048 }, m_adcCapacity{ 4096 }, m_pendingCapacity{ 32 }; ///< Fixed memory bounds.
    unsigned     m_periodUs{ 5000 },
        m_budgetUs{ 1000 }; ///< Soft wall-time duty budget; individual solves are not preemptible.
};

/// WFS origin plus separately established camera/exposure and immutable synco slot evidence.
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
    const char   *m_reason{ "source_host_mapping_unqualified" };           ///< Stable inhibition description.
    ShadowMode    m_requested{ ShadowMode::Standard }, m_effective{ ShadowMode::Standard }; ///< Request/fallback state.
    std::uint64_t m_raw{ 0 }, m_records{ 0 }, m_candidates{ 0 }, m_drops{ 0 },
        m_rejections{ 0 };                          ///< Distinct work/loss.
    double       m_computeUs{ -1 }, m_peakUs{ -1 }; ///< Actual worker model time; unavailable before numerical work.
    std::size_t  m_rawQueued{ 0 }, m_qualifiedQueued{ 0 }; ///< Pending work, not retained source history.
    std::int64_t m_originNs{ -1 }, m_deadlineNs{ -1 };     ///< Actual latest frozen origin/cutoff, not a clock mapping.
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

    /// Inspect actual canonical coefficients for deterministic independence/indexing tests.
    Matrix coefficients( ShadowMode mode /**< [in] Independent model. */ );
};
} // namespace DDSPC
#endif
