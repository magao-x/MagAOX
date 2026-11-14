/** \file shadow_policy.hpp
 * \brief Shared requested/effective DDSPC selection, recovery hysteresis and diagnostic candidate constraints.
 */
#ifndef PCSHADOW_POLICY_HPP
#define PCSHADOW_POLICY_HPP
#include "adc_features.hpp"
namespace DDSPC
{
/// Effective candidate selection; Safe is unavailable for actuation, with an optional diagnostic ramp toward zero.
enum class PolicyMode
{
    Safe,
    Standard,
    Assisted
};

/// Return a stable CSV description of shadow candidate selection.
const char *policy_mode( PolicyMode mode /**< [in] Requested or effective mode. */ );

/// Immutable switching and common command constraints.
struct PolicyConfig
{
    int      m_modes{ 2 };           ///< WFS/input dimensions; independent of the two ADC channels.
    unsigned m_recoveryFrames{ 10 }; ///< Consecutive fully qualified ADC/WFS/model decisions before assisted recovery.
    double   m_limit{ 3 }, m_maxStep{ .2 }; ///< Per-axis absolute and per-WFS-step bounds, shared by both policies.
};

/// Evidence available at a single frozen decision; proposals come only from actual predict_cached calls.
struct ControlEvidence
{
    /// Allocate the two proposal columns before processing decisions.
    explicit ControlEvidence( int modes = 2 /**< [in] Positive WFS/input dimensions. */ );

    bool m_commonValid{ false }, m_featuresValid{ false };   ///< WFS/applied timing and complete ADC feature history.
    bool m_standardReady{ false }, m_assistedReady{ false }; ///< Current prediction calls succeeded.
    FeatureReason m_featureReason{ FeatureReason::Incomplete }; ///< Current window's qualification outcome.
    DDSPC::Matrix m_standard, m_assisted;                       ///< Absolute proposals; never plant truth.
};

/// A single bounded command and an explicit reason, retained even when authority does not change.
struct ControlDecision
{
    /// Allocate one diagnostic candidate column before processing decisions.
    explicit ControlDecision( int modes = 2 /**< [in] Positive WFS/input dimensions. */ );

    PolicyMode    m_requested{ PolicyMode::Standard }, m_effective{ PolicyMode::Safe }; ///< Intent versus selection.
    const char   *m_reason{ "initial" }; ///< Stable reason for fallback, recovery, or safe output.
    unsigned      m_validStreak{ 0 };    ///< Consecutive fully qualified assisted decisions.
    DDSPC::Matrix m_command;             ///< Diagnostic candidate; only the offline plant may explicitly consume it.
    bool          m_candidateValid{ false }; ///< A qualified model produced a candidate; never grants output authority.
    bool          m_switched{ false };       ///< Effective candidate selection changed at this decision.
    double        m_step{ 0 };               ///< Maximum per-axis change from the preceding bounded candidate/input.
};

/// Shadow policy with no publishers, runtime, INDI properties, or Legacy access; caller serializes operations.
class ShadowPolicy
{
    PolicyConfig  m_config;                                     ///< Validated bounds and hysteresis.
    PolicyMode    m_requested, m_effective{ PolicyMode::Safe }; ///< Requested and qualified effective selection.
    DDSPC::Matrix m_target;                                     ///< Preallocated proposal-plus-probe workspace.
    unsigned      m_validStreak{ 0 }; ///< Recovery qualification, reset on any failed assisted prerequisite.

  public:
    /// Reset qualification and effective selection without touching learners or producing output.
    void reset();

    /// Change requested assistance without resetting fits, the previous candidate, or effective-mode evidence.
    void request( PolicyMode mode /**< [in] Standard or Assisted; does not grant readiness or output authority. */ );

    /// Validate the initial request and command policy without any external action.
    ShadowPolicy( PolicyMode          requested /**< [in] Standard or Assisted. */,
                  const PolicyConfig &config /**< [in] Common limits and recovery dwell. */ );

    /// Select a candidate, add common exploration only to usable proposals, then enforce absolute/slew bounds.
    ControlDecision decide( const ControlEvidence &evidence /**< [in] Current causal model and sensor evidence. */,
                            const DDSPC::Matrix &previous /**< [in] Previous diagnostic or simulated-consumed input. */,
                            const DDSPC::Matrix &exploration /**< [in] Common independent evaluation probe. */ );

    /// Fill caller-owned decision storage; only existing finite model proposals can qualify.
    void decide( const ControlEvidence &evidence /**< [in] Current causal source/model evidence. */,
                 const Matrix          &previous /**< [in] Previous diagnostic or simulated-consumed command. */,
                 const Matrix    &exploration /**< [in] Explicit independent probe; zero in application shadow. */,
                 ControlDecision &output /**< [out] Preallocated decision of configured shape. */ );
};

/// Bound every command, including calibration, safe transitions and ordinary predictions.
DDSPC::Matrix bounded_command( const DDSPC::Matrix &request /**< [in] Finite absolute command request. */,
                               const DDSPC::Matrix &previous /**< [in] Last consumed bounded command. */,
                               const PolicyConfig  &config /**< [in] Common absolute and slew limits. */ );

/// Apply common absolute/slew constraints into caller-owned storage, without allocation.
void bounded_command( const Matrix       &request /**< [in] Finite absolute request. */,
                      const Matrix       &previous /**< [in] Previous bounded input. */,
                      const PolicyConfig &config /**< [in] Dimensions and bounds. */,
                      Matrix             &output /**< [out] Preallocated result, disjoint from request/previous. */ );

} // namespace DDSPC
#endif
