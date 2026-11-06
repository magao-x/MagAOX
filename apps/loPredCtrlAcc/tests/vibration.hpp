/** \file vibration.hpp
 * \brief Deterministic closed-loop numerical experiment; synthetic units and host-memory commands only.
 */
#ifndef loPredCtrlAcc_vibration_hpp
#define loPredCtrlAcc_vibration_hpp
#include "async_shadow.hpp"
#include <array>
#include <string>

namespace libXWCTest::loPredCtrlAccTest
{
struct StabilityPolicy;
/// Immutable experiment settings; none are instrument calibration or operational defaults.
struct VibrationConfig
{
    std::string m_name{ "mixed" };                ///< Artifact identity, not a device/stream name.
    int      m_train{ 2400 }, m_evaluate{ 8000 }; ///< Common open-loop identification and closed-loop frames at 2 kHz.
    int      m_history{ 5 }, m_future{ 3 }, m_accelHistory{ 5 }; ///< Unmodified general DDSPC dimensions.
    unsigned m_seed{ 100 }; ///< Exogenous noise identity; common to every controller.
    bool     m_peak28{ true }, m_peak56{ true }, m_broadband{ false }, m_qrd{ false }; ///< Source/learner choices.
    double   m_coherence{ 1 }, m_accelNoise{ .03 }, m_wfsNoise{ .03 }; ///< Sensor mixture and independent noise scales.
    double   m_lambda{ .02 }, m_prior{ .001 }, m_gamma{ .9995 }; ///< Literal causal / relative Legacy penalty and RLS.
    double   m_dmPole{ .1 }, m_dmScale{ 1 }, m_controlGain{ 1 }, m_limit{ 4 }; ///< Synthetic DM response and safety.
    int m_dmExtraFrames{ 0 }, m_wfsDelayFrames{ 0 }, m_sensorDelayFrames{ 0 }; ///< Additional physical/sample delays.
    int m_publicationFrames{ 0 }, m_resetFrame{ -1 }, m_stallFrame{ -1 }, m_stallFrames{ 0 }; ///< Virtual workers.
    int m_missingFrame{ -1 }, m_missingFrames{ 0 };       ///< Missing source samples, starting at absolute frame.
    std::uint64_t m_jitterNs{ 0 }, m_maxAgeNs{ 2000000 }; ///< Sensor timestamp/availability jitter; evidence age gate.
    std::size_t   m_capacity{ 32 };                       ///< Bounded actual prototype queues.
    const StabilityPolicy *m_stability{ nullptr }; ///< Caller-owned policy, alive for the run; null preserves Phase 10.
    int m_freezeFrame{ -1 }, m_freezeFrames{ 0 };  ///< Explicit temporary hold of the simulated applied input only.
};

/// One common immutable source realization, generated without consulting controller state.
struct VibrationSource
{
    std::vector<DDSPC::Matrix> m_disturbance, m_acceleration, m_wfsNoise, m_exploration; ///< Physical/noise arrays.
    std::vector<std::uint64_t> m_availableNs, m_sensorNs, m_sensorAvailableNs;           ///< Explicit monotonic timing.
    std::vector<int>           m_sensorIndex; ///< Latest causal available sample identity, -1 when absent/stale.
};

/// Auditable per-frame result; all fields refer solely to this simulated trajectory.
struct VibrationFrame
{
    DDSPC::Matrix m_residual, m_measured, m_requested, m_applied; ///< True/noisy residual and bounded input history.
    bool          m_valid{ false }, m_ready{ false }, m_newCommand{ false }; ///< Gate and actual proposal admission.
    std::uint64_t m_version{ 0 }, m_ageNs{ 0 }, m_modelSequence{ 0 };        ///< Exact publication evidence used.
    int           m_origin{ -1 }, m_reason{ 0 }; ///< Latest available WFS origin and prototype rejection reason.
    double        m_predictionSquared{ 0 };      ///< Post-hoc conditional forecast error, never a decision feature.
    int           m_predictionCount{ 0 };        ///< Number of scalar forecast targets scored at this decision.
    double        m_thetaNorm{ 0 },
        m_dNorm{ 0 };   ///< Actual origin snapshot coefficient magnitudes, for convergence diagnostics.
    int    m_rank{ 0 }; ///< Existing normalized readiness rank; no substitute closed-loop stability test.
    int    m_action{ 0 }, m_recovery{ 0 }; ///< Explicit SimAction/SimRecovery values for offline provenance.
    bool   m_correctionAccepted{ false };  ///< A fresh model, rather than a recovery action, supplied the request.
    double m_confidenceBound{ 0 }, m_slewX{ 0 }, m_slewY{ 0 }; ///< Calibrated bounds; never instrument calibration.
};

/// Complete finite result; no real-time or physical performance claim follows from virtual worker service.
struct VibrationResult
{
    std::array<std::vector<VibrationFrame>, 5> m_frames;            ///< None, oracle, Legacy, standard, assisted.
    std::array<std::uint64_t, 5> m_overflows{}, m_invalidChanges{}; ///< Queue rejects and forbidden gate violations.
};

/// Return the virtual WFS origin; u_i is held after this instant and influences y_(i+1).
std::uint64_t vibration_time( int frame /**< [in] Nonnegative source index. */ );

/// Generate oscillators, causal force innovations, and independently seeded measurement/excitation records.
VibrationSource vibration_source( const VibrationConfig &config /**< [in] Explicit synthetic contract. */ );

/// Replay identical exogenous data through five separate numerical plants/controllers.
VibrationResult vibration_run( const VibrationConfig &config /**< [in] Experiment parameters. */,
                               const VibrationSource &source /**< [in] Shared immutable realization. */ );
/// Execute the verbatim committed Phase 10 loop, linked under a distinct symbol as the compatibility default.
VibrationResult committed_vibration_run( const VibrationConfig &config /**< [in] Original settings. */,
                                         const VibrationSource &source /**< [in] Identical source. */ );
} // namespace libXWCTest::loPredCtrlAccTest
#endif
