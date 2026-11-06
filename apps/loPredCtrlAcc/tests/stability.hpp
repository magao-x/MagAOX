/** \file stability.hpp
 * \brief Offline-only estimator candidate and causal recovery supervisor; no application integration or output sink.
 */
#ifndef loPredCtrlAcc_stability_hpp
#define loPredCtrlAcc_stability_hpp
#include "vibration.hpp"
#include <memory>

namespace libXWCTest::loPredCtrlAccTest
{
/// Distinguish model-sourced corrections from explicitly authorized simulation recovery actions.
enum class SimAction
{
    Hold,
    Model,
    Calibration,
    Neutral,
    Probe
};
/// Finite offline recovery states; no state enables physical commands.
enum class SimRecovery
{
    Tracking,
    Neutral,
    Waiting,
    Training,
    Ramping,
    Disabled
};
/// Explicit candidate settings; default policy preserves Phase 10 behavior.
struct StabilityPolicy
{
    bool   m_symmetricRls{ false };  ///< Use the pinned independent candidate, never alter the reference estimator.
    bool   m_neutralReturn{ false }; ///< Permit model-independent slew toward synthetic zero after sustained rejection.
    bool   m_retrain{ false };       ///< Authorize bounded synthetic calibration probes, offline only.
    bool   m_confidence{ false }; ///< Compare matured forecast errors to the initial calibration's empirical quantile.
    double m_quantile{ .99 }; ///< Declared empirical false-alarm design quantile, not a hidden coefficient threshold.
};
/// Worker facade selects an independently compiled candidate while retaining the same input/publication protocol.
class StabilityWorker
{
    struct Impl;
    std::unique_ptr<Impl> m_impl; ///< One exclusive original or candidate model pair.
  public:
    /// Allocate immutable dimensions and bounded queues outside virtual frame decisions.
    StabilityWorker( const DDSPC::ShadowConfig &config /**< [in] Existing numerical contract. */,
                     std::size_t                capacity /**< [in] Queue capacity. */,
                     bool                       symmetric /**< [in] Test-only estimator candidate. */ );
    /// Release privately owned numerical buffers.
    ~StabilityWorker();
    /// Admit actual input records with explicit per-lane results.
    DDSPC::OfflineAdmission submit_record( const DDSPC::OfflineRecord &record /**< [in] Actual synthetic histories. */,
                                           std::uint64_t               now /**< [in] Virtual monotonic time. */ );
    /// Service one immutable queued record.
    bool manual_step( DDSPC::ShadowMode mode /**< [in] Independent lane. */, std::uint64_t now /**< [in] Time. */ );
    /// Invalidate old publications and reset one owning learner.
    void reset( DDSPC::ShadowMode mode /**< [in] Independent lane. */ );
    /// Apply unchanged snapshot-age, causality, rank and readiness checks.
    bool predict( DDSPC::ShadowMode         mode /**< [in] Independent lane. */,
                  std::uint64_t             epoch /**< [in] Source epoch. */,
                  std::uint64_t             sequence /**< [in] State origin. */,
                  std::uint64_t             stateNs /**< [in] State time. */,
                  std::uint64_t             now /**< [in] Query time. */,
                  std::uint64_t             maxAge /**< [in] Evidence-age bound. */,
                  const DDSPC::Matrix      &state /**< [in] State. */,
                  DDSPC::Matrix            &output /**< [out] Proposal only. */,
                  DDSPC::OfflinePrediction &result /**< [out] Gate. */ );
    /// Copy coherent diagnostics for the actual selected model.
    bool read_snapshot( DDSPC::ShadowMode       mode /**< [in] Lane. */,
                        DDSPC::OfflineSnapshot &out /**< [out] Fixed buffers. */ );
    /// Read bounded worker counters.
    DDSPC::OfflineWorkerStats stats( DDSPC::ShadowMode mode /**< [in] Lane. */ ) const;
};

/// Causal command supervisor learns limits/confidence solely from earlier calibration and observed forecasts.
class RecoverySupervisor
{
    StabilityPolicy m_policy;           ///< Explicit offline authority; never inferred from a failed model.
    int  m_features, m_delay, m_budget; ///< Rank-sized validation window, history drain and bounded probe budget.
    int  m_bad{ 0 }, m_probeFrames{ 0 }, m_rampFrames{ 0 }; ///< Bounded episode counters.
    bool m_reset{ false }; ///< One pending owner-reset request, acknowledged by the simulated frontend.
    SimRecovery   m_state{ SimRecovery::Tracking }; ///< Explicit supervisor state.
    DDSPC::Matrix m_step, m_previous; ///< Per-axis maximum observed calibration slew and previous calibration input.
    std::vector<double> m_calibration, m_errors; ///< Finite calibration errors and rolling current evidence.
    double              m_threshold{ 0 };        ///< Frozen empirical calibration bound; zero until fitted.
    bool m_boundReady{ false }; ///< A legitimate zero error bound is distinct from missing calibration evidence.
    /// Enforce a calibrated per-axis slew bound toward an explicitly requested synthetic target.
    DDSPC::Matrix slew( const DDSPC::Matrix &held /**< [in] Last request. */,
                        const DDSPC::Matrix &target /**< [in] Model-independent or validated target. */ ) const;

  public:
    /// Set dimensions and bounded retraining authority from the declared simulation protocol.
    RecoverySupervisor( const VibrationConfig &config /**< [in] Dimensions and initial calibration duration. */,
                        int                    features /**< [in] Existing full-rank feature count. */,
                        const StabilityPolicy &policy /**< [in] Explicit recovery authority. */ );
    /// Observe an actual initial calibration command, never a hypothetical model request.
    void calibrate( const DDSPC::Matrix &command /**< [in] Actual bounded initial probe. */ );
    /// Score only origin-frozen predictions whose WFS targets have become available.
    void observe( double squaredError /**< [in] Mean squared error per target scalar. */,
                  bool   calibration /**< [in] Origin belongs to initial calibration. */ );
    /// Choose a slew-limited target; the simulation applies its final inward amplitude clip.
    DDSPC::Matrix command( bool                 valid /**< [in] Original predictor gate. */,
                           bool                 healthy /**< [in] Fresh input pipeline. */,
                           const DDSPC::Matrix &proposal /**< [in] Candidate feedback plus the normal tiny probe. */,
                           const DDSPC::Matrix &held /**< [in] Last actual request. */,
                           const DDSPC::Matrix &probe /**< [in] Explicit independent calibration excitation. */,
                           double               limit /**< [in] Synthetic amplitude bound. */,
                           SimAction           &action /**< [out] Provenance. */ );
    /// Consume one reset request; repeated invalid frames cannot repeatedly reset the owning learner.
    bool take_reset();
    /// Inspect explicit state without changing any model.
    SimRecovery state() const;
    /// Return the frozen calibration error bound.
    double threshold() const;
    /// Inspect the maximum actual calibration command increments.
    const DDSPC::Matrix &step_bound() const;
};
} //namespace libXWCTest::loPredCtrlAccTest
#endif
