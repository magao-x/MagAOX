/** \file offline_ddspc.hpp
 * \brief Memory-only 8-kHz ADC/2-kHz WFS experiment using the actual independent shadow learners.
 * \ingroup loPredCtrlAcc_unit_test
 */
#ifndef loPredCtrlAcc_offline_ddspc_hpp
#define loPredCtrlAcc_offline_ddspc_hpp
#include "offline_adc.hpp"
#include "offline_control.hpp"
#include "../shadow_controller.hpp"

/// \cond OFFLINE_TEST_SUPPORT
namespace libXWCTest::loPredCtrlAccTest
{
/// Explicit synthetic plant and identification parameters; none are operational calibration.
struct AdcDdspcConfig
{
    SyntheticAdcConfig  m_adc;        ///< SI ADC waveform, encoder and observation faults.
    DDSPC::ShadowConfig m_controller; ///< Actual production causal learner dimensions and parameters.
    FeatureConfig       m_features;   ///< Strict timestamp-window qualification, separate from ADC transport health.
    PolicyConfig        m_policy;     ///< Test-only fallback, hysteresis and common command bounds.
    int    m_train{ 2000 }, m_evaluate{ 4000 };          ///< Fixed 2-kHz WFS steps; ADC cadence has its own timestamps.
    double m_coupling{ 1 }, m_pole{ .8 }, m_dmGain{ 1 }; ///< Synthetic WFS response, one-frame applied-input delay.
    double m_wfsNoise{ .005 }, m_trainProbe{ .3 }, m_evalProbe{ .02 }; ///< Independent Gaussian noise/excitation.
    bool   m_uncorrelatedAdc{ false }; ///< Substitute independent white ADC measurements without changing the plant.
    int m_wfsInvalidStart{ 0 }, m_wfsInvalidCount{ 0 }; ///< Optional invalid WFS interval; physical data still advance.

    /// Establish modest default dimensions and strict QR identification for this offline experiment.
    AdcDdspcConfig();
};

/// Frozen-decision evidence for control transitions, timing and actual command consumption.
struct AdcDdspcDecision
{
    int             m_index{ 0 };      ///< Absolute WFS index; traces contain evaluation decisions only.
    std::int64_t    m_decisionNs{ 0 }; ///< WFS deadline in the established common monotonic clock.
    AccelFeature    m_feature;         ///< Actual decoded aggregate and its causal timing evidence.
    ControlDecision m_control;         ///< Requested/effective mode, reason, hysteresis and consumed command.
    bool            m_commonValid{ false }, m_featuresValid{ false };   ///< Current source/history eligibility.
    bool            m_standardReady{ false }, m_assistedReady{ false }; ///< Actual current cached-prediction admission.
};

/// Periodic observed training evidence from the actual selected model.
struct AdcDdspcTrainingPoint
{
    int                 m_index{ 0 };     ///< WFS identity, including calibration.
    DDSPC::ShadowStatus m_status;         ///< Actual readiness, rank, updates and conditional RMSE.
    double              m_thetaNorm{ 0 }; ///< Canonical prediction coefficient norm at this instant.
};

/// One controller's trajectory and finite, explicitly conditional prediction diagnostics.
struct AdcDdspcTrajectory
{
    std::array<std::vector<double>, 2> m_residual; ///< Evaluation-only true residual, in arbitrary synthetic WFS units.
    double        m_variance{ 0 }, m_predictionRmse{ 0 }, m_thetaNorm{ 0 }, m_maxCommand{ 0 }; ///< Measured summaries.
    double        m_commandRms{ 0 }, m_power28{ 0 }, m_power56{ 0 }, m_maxSwitchStep{ 0 };     ///< Evaluation evidence.
    std::uint64_t m_longestFallback{ 0 }; ///< Longest assisted-unavailable episode, including safe output.
    std::uint64_t m_fallbackFrames{ 0 }, m_safeFrames{ 0 }; ///< Standard fallback and safe-ramp time separately.
    std::uint64_t m_updates{ 0 }, m_predictions{ 0 }, m_commands{ 0 }, m_blocked{ 0 },
        m_faults{ 0 }; ///< Gate evidence.
    std::uint64_t m_gateViolations{
        0 }; ///< Assisted model proposals admitted with incomplete ADC evidence; must stay0.
    std::int64_t     m_firstReady{ -1 }, m_lastReady{ -1 }; ///< Absolute WFS indices proving warmup/recovery.
    DDSPC::AdcStatus m_adcStatus; ///< Actual capture diagnostics, including loss and retention accounting.
    std::vector<AdcDdspcTrainingPoint> m_learning; ///< Every100th WFS step, first readiness and final fit evidence.
    std::vector<AdcDdspcDecision> m_decisions; ///< Every evaluation decision, including unavailable-model decisions.
};

/// Identical exogenous realizations in three separate plants, with independently learned standard/assisted fits.
struct AdcDdspcResult
{
    std::array<AdcDdspcTrajectory, 3> m_runs; ///< Open loop, standard feedback and assisted feedback.
};

/// Run synthetic virtual time only; decoded ADC samples are the sole source of assisted features.
AdcDdspcResult adc_ddspc_run( const AdcDdspcConfig &config /**< [in] Bounded, explicit simulation contract. */ );
} // namespace libXWCTest::loPredCtrlAccTest
/// \endcond
#endif
