/** \file qualified_shadow_replay.hpp
 * \brief Deterministic qualified synthetic telemetry through actual ADC and native multirate DDSPC code.
 * \ingroup loPredCtrlAcc_unit_test
 */
#ifndef loPredCtrlAcc_qualified_shadow_replay_hpp
#define loPredCtrlAcc_qualified_shadow_replay_hpp
#include "offline_adc.hpp"
#include "../multirate_shadow.hpp"

/// \cond QUALIFIED_REPLAY_SUPPORT
namespace libXWCTest::loPredCtrlAccTest
{
/// Bounded test-only oracle; no field confers real telemetry qualification.
struct QualifiedReplayConfig
{
    std::size_t m_frames{ 3000 };            ///< Number of actual 2-kHz synthetic measurements.
    unsigned    m_future{ 1 }, m_seed{ 73 }; ///< Matched target horizon and independent realization.
    bool m_qrd{ true }, m_uncorrelated{ false }, m_duplicateChannels{ false }; ///< Learner and negative controls.
    std::size_t m_changeCommandAt{ 0 }; ///< Future-input perturbation for origin-only leakage regression.
    std::size_t m_changeWfsAt{ 0 };     ///< Perturb only targets at/after this counter; zero disables leakage check.
    bool m_noAdcAnchor{ false }; ///< Standard model must train even before an accelerometer is attached/qualified.
    bool m_upstream{ true }, m_timerMode{ false }, m_wrongPhase{ false },
        m_futureAdc{ false };                                          ///< Explicit synchronization cases.
    std::size_t m_syncLossStart{ 0 }, m_syncLossCount{ 0 };            ///< Upstream lock loss, not local phase control.
    bool        m_exposureEvidence{ true }, m_appliedEvidence{ true }; ///< Missing proof must leave training inhibited.
};

/// Owned offline artifacts; future inputs never originate in either diagnostic controller.
struct QualifiedReplayResult
{
    DDSPC::OnlineShadowStatus m_status;               ///< Actual two-model result, not externally forced readiness.
    DDSPC::AdcStatus          m_adc;                  ///< Actual decoder counters and rates.
    DDSPC::Matrix             m_standard, m_assisted; ///< Final canonical coefficients for independence comparisons.
    std::vector<DDSPC::ShadowPrediction> m_scores;    ///< Per-target prequential conditional traces.
    std::vector<double> m_computeUs; ///< Unpaced entire frame assembly/numerical times, not target deadlines.
    std::vector<double> m_numericalUs, m_maxJobUs; ///< Worker-equivalent model jobs, excluding synthetic acquisition.
    std::vector<std::array<int, 4>> m_readiness;   ///< Per-frame standard/assisted readiness and empirical ranks.
    std::size_t m_candidates{ 0 }, m_scoreDrops{ 0 }; ///< Diagnostic generation and retention evidence.
};

/// Generate a declared causal test system with independent four-slot actual inputs and native ADC conversion.
QualifiedReplayResult
replay_qualified( const QualifiedReplayConfig &config /**< [in] Fixed finite synthetic scenario. */ );
} // namespace libXWCTest::loPredCtrlAccTest
/// \endcond
#endif
