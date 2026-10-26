/** \file dmSyncoSimulation.hpp
 * \brief Read-only syncopation policy simulation snapshots and conditional step outcomes.
 */
#ifndef common_dmSyncoSimulation_hpp
#define common_dmSyncoSimulation_hpp

#include <array>
#include <cstddef>
#include <cstdint>

namespace MagAOX
{
namespace dmSyncoSimulation
{

constexpr size_t maxSteps = 16; ///< Fixed capacity shared with the provider schedule layout.

/// Counts for one step index; modeled copies and triggers never address a DM stream.
struct Step
{
    uint64_t considered{ 0 };    ///< Committed occurrences evaluated once.
    uint64_t wouldRun{ 0 };      ///< Occurrences eligible under the explicitly configured model bounds.
    uint64_t skipped{ 0 };       ///< Occurrences unavailable before any modeled copy starts.
    uint64_t earlyMain{ 0 };     ///< Passive preparations cancelled by an already observed main.
    uint64_t missingMain{ 0 };   ///< Active slots without their genuine counter-labelled main.
    uint64_t missingOutput{ 0 }; ///< Slots without a coherent main-to-combined association.
    uint64_t readerHold{ 0 };    ///< Slots whose actual or modeled reader hold prevents the fixed target.
    uint64_t expired{ 0 };       ///< Obsolete slots or unavailable future timing envelopes.
    uint64_t overlap{ 0 };       ///< Observations that could overlap a modeled copy/reader interval.
};

/// Coherent snapshot copied outside monitor callbacks for INDI and telemetry.
/** A valid prediction is conditional on configured timing bounds. It neither
 * qualifies a physical earliest-main reserve nor confirms an actual auxiliary
 * delivery. All timestamps and residuals use same-host monotonic nanoseconds.
 */
struct Summary
{
    bool     enabled{ false };             ///< Startup-only simulation selection.
    bool     valid{ false };               ///< A fresh counter-labelled grid is available.
    bool     feasible{ false };            ///< The nominal complete cycle fits configured bounds.
    uint64_t session{ 1 };                 ///< Changes after explicit reset or stream identity replacement.
    uint64_t sequence{ 0 };                ///< Distinct accepted primary observations in this session.
    int64_t  nowNs{ 0 };                   ///< Last evaluation on the monotonic scheduling clock.
    int64_t  lastMainNs{ 0 };              ///< Last accepted genuine primary observation.
    uint64_t counter{ 0 };                 ///< Last accepted genuine primary counter, never inferred from time.
    int      steps{ 0 };                   ///< Fixed requested count, including passive step zero.
    int64_t  periodNs{ 0 };                ///< Current robust nominal period.
    int64_t  gridNs{ 0 };                  ///< Frozen prediction for the most recently observed primary counter.
    int64_t  phaseNs{ 0 };                 ///< Actual observation minus its prediction made before that observation.
    int64_t  phaseMinNs{ 0 };              ///< Minimum qualified pre-observation prediction residual this session.
    int64_t  phaseMaxNs{ 0 };              ///< Maximum qualified pre-observation prediction residual this session.
    uint64_t phaseCount{ 0 };              ///< Observations with an independently frozen phase prediction.
    uint64_t periodSamples{ 0 };           ///< Qualified interval samples retained by the bounded estimator.
    uint64_t phaseSamples{ 0 };            ///< Counter/observation pairs retained by the phase estimator.
    int64_t  nextGridNs{ 0 };              ///< Next committed main-grid target, or zero while unavailable.
    int64_t  nextActiveNs{ 0 };            ///< Next committed unexpired active target, or zero when unavailable.
    int64_t  earlyReserveNs{ 0 };          ///< Assumed configured reserve; simulation does not qualify this value.
    int64_t  deliveryNs{ 0 };              ///< Assumed delivery allowance after an observed or modeled publication.
    int64_t  keepoutNs{ 0 };               ///< Assumed controller processing and final-reader lifetime.
    int64_t  guardNs{ 0 };                 ///< Additional reader/delivery guard.
    int64_t  snapshotNs{ 0 };              ///< Assumed source-snapshot budget.
    int64_t  copyNs{ 0 };                  ///< Assumed whole-batch copy/publication budget.
    int64_t  leadNs{ 0 };                  ///< Nominal passive-copy completion lead.
    int64_t  observationNs{ 0 };           ///< Assumed publication-to-primary-observation uncertainty.
    int64_t  visibilityNs{ 0 };            ///< Required visibility before the earliest plausible main.
    int64_t  reserveNs{ 0 };               ///< Preparation reserve before the hard passive-copy cutoff.
    uint64_t window{ 0 };                  ///< Configured period and phase sample capacity.
    double   phaseGain{ 0 };               ///< Bounded phase-estimate correction gain.
    int64_t  phaseSlewNs{ 0 };             ///< Maximum correction to an unadvertised future cycle.
    int64_t  cadenceMinPeriodNs{ 0 };      ///< Whole-cycle minimum from the configured total-cadence sanity rating.
    uint64_t duplicates{ 0 };              ///< Unchanged-counter primary notifications ignored.
    uint64_t gaps{ 0 };                    ///< Forward primary-counter gaps, excluding their intervals from estimation.
    uint64_t resets{ 0 };                  ///< Counter rollback or explicit identity resets.
    uint64_t invalid{ 0 };                 ///< Incoherent, out-of-order or unrepresentable observations.
    uint64_t excluded{ 0 };                ///< Invalid or broadly implausible period samples, not timing faults.
    uint64_t outputGaps{ 0 };              ///< Missing/ambiguous combined-counter progression.
    uint64_t associations{ 0 };            ///< Coherent actual primary-to-combined observations accepted.
    std::array<Step, maxSteps> outcomes{}; ///< Current-session conditional outcomes indexed by step.
};

} // namespace dmSyncoSimulation
} // namespace MagAOX
#endif
