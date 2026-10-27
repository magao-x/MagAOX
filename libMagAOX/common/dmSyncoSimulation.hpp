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

constexpr size_t traceCapacity = 256; ///< Fixed event FIFO; full queues drop new evidence without changing the model.

constexpr size_t traceBatchCapacity = 64; ///< Maximum events in one telemetry record.

constexpr size_t estimatorHistoryCapacity = 256; ///< Pre-trigger accepted-primary history, covering the largest window.

constexpr size_t estimatorPostSamples = 64; ///< Subsequent accepted primaries retained after the latest overlap.

constexpr size_t estimatorTraceCapacity = 4096; ///< Export FIFO; diagnostic saturation never changes model decisions.

constexpr size_t estimatorBatchCapacity = 512; ///< Maximum estimator samples in one ordinary-thread emission.

/// Monitor path supplying retrospective evidence; both paths can observe the same main.
enum class OverlapOrigin
{
    primary  = 0,
    combined = 1
};

/// Last modeled interval containing the observation, with half-open boundaries.
enum class OverlapWindow
{
    copy       = 0,
    readerHold = 1
};

/// Optional caller context; zero means unavailable and producer clocks are never treated as monotonic.
struct Observation
{
    uint64_t streamInode{ 0 };          ///< Observed stream metadata identity.
    uint64_t triggerInode{ 0 };         ///< Combined trace's named primary identity, when available.
    int64_t  receiptNs{ 0 };            ///< Same-host monotonic semaphore-receipt observation.
    int64_t  callbackNs{ 0 };           ///< Same-host monotonic monitor callback entry.
    int64_t  lockRequestNs{ 0 };        ///< Same-host monotonic cycle-lock request.
    int64_t  lockAcquiredNs{ 0 };       ///< Same-host monotonic cycle-lock acquisition.
    int64_t  publicationNs{ 0 };        ///< Observed stream publication on its producer clock; not monotonic.
    int64_t  primaryPublicationNs{ 0 }; ///< Associated primary publication on its producer clock; not monotonic.
    int64_t  startNs{ 0 };              ///< Combined trace start on its producer clock; not monotonic.
    int64_t  updateNs{ 0 };             ///< Combined trace update on its producer clock; not monotonic.
};

/// One overlap observation; the session/planCounter/step/mainCounter tuple supports offline deduplication.
struct OverlapEvent
{
    uint64_t      id{ 0 };              ///< Process-lifetime evidence sequence, including dropped records.
    uint64_t      session{ 0 };         ///< Model session owning this event, retained across queue resets.
    uint64_t      mainCounter{ 0 };     ///< Actual primary counter supplying the evidence.
    uint64_t      combinedCounter{ 0 }; ///< Current combined counter, or last observed counter on the primary path.
    uint64_t      planCounter{ 0 };     ///< Counter labelling the last modeled copy's cycle.
    uint64_t      lastMainCounter{ 0 }; ///< Most recent accepted primary counter before this evidence.
    uint64_t      planEstimatorId{ 0 }; ///< Estimator record that created the modeled copy's frozen plan.
    uint64_t      lastEstimatorId{ 0 }; ///< Latest completed estimator record before this evidence.
    int           step{ 0 };            ///< Index of the last modeled copy.
    OverlapOrigin origin{ OverlapOrigin::primary }; ///< Monitor path; repeated origins are deliberately retained.
    OverlapWindow window{ OverlapWindow::copy };    ///< Copy or active-reader interval containing the evidence.
    int64_t       observedNs{ 0 };                  ///< Same-host monotonic evidence time.
    int64_t       mainGridNs{ 0 };   ///< Committed grid or extrapolation from the last frozen cycle; zero unavailable.
    int64_t       planGridNs{ 0 };   ///< Frozen main prediction for the last modeled copy's cycle.
    int64_t       periodNs{ 0 };     ///< Period frozen with that modeled copy's cycle.
    int64_t       copyBeginNs{ 0 };  ///< Same-host monotonic modeled copy start.
    int64_t       copyEndNs{ 0 };    ///< Same-host monotonic modeled copy completion.
    int64_t       readerEndNs{ 0 };  ///< Modeled active-reader end; zero for passive preparation.
    int64_t       lastMainNs{ 0 };   ///< Previous accepted primary observation on the monotonic clock.
    int64_t       lastOutputNs{ 0 }; ///< Latest associated combined observation on the monotonic clock.
    Observation   observation;       ///< Raw monitor/producer context supplied by the caller.
};

/// Process-lifetime FIFO accounting; detected equals drained plus dropped plus queued.
struct TraceStatus
{
    bool     enabled{ false };          ///< Model selection; queued older-session events remain drainable after reset.
    uint64_t detected{ 0 };             ///< All overlap evidence, including records rejected by the full FIFO.
    uint64_t drained{ 0 };              ///< Records removed from the FIFO for a telemetry attempt.
    uint64_t dropped{ 0 };              ///< New records discarded because the fixed FIFO was full.
    uint64_t queued{ 0 };               ///< Events awaiting a telemetry attempt.
    uint64_t capacity{ traceCapacity }; ///< Fixed FIFO capacity.
    uint64_t logFailed{ 0 };            ///< App-owned count of drained events whose telemetry submission failed.
};

/// One accepted-primary estimator update; values come from the original calculations without recomputing medians.
struct EstimatorSample
{
    uint64_t id{ 0 };                    ///< Process-lifetime accepted-primary sample ID, including unselected history.
    uint64_t session{ 0 };               ///< Model session owning this update.
    uint64_t sequence{ 0 };              ///< Accepted-primary sequence within the session.
    uint64_t counter{ 0 };               ///< Genuine main counter supplying this update.
    int64_t  observedNs{ 0 };            ///< Same-host monotonic observation.
    int64_t  publicationIntervalNs{ 0 }; ///< Raw producer-clock elapsed interval; zero means unavailable.
    bool     sequential{ false };        ///< Counter is the previous accepted main plus one.
    bool     intervalAccepted{ false };  ///< Input interval was admitted to the period-estimator ring.
    uint64_t periodSamples{ 0 };         ///< Retained eligible publication intervals after this update.
    uint64_t phaseSamples{ 0 };          ///< Retained counter/observation pairs after this update.
    int64_t  periodBeforeNs{ 0 };        ///< Selected period before incorporating this primary.
    int64_t  periodNs{ 0 };              ///< Selected period after incorporating this primary.
    int64_t  predictedNs{ 0 };           ///< Frozen prediction for this main before observing it; zero unavailable.
    uint64_t nextCounter{ 0 };           ///< Counter of the attempted future commitment; zero if none.
    uint64_t sourcePlanCounter{ 0 };     ///< Frozen cycle extrapolated to obtain the future base grid.
    uint64_t sourceEstimatorId{ 0 };     ///< Estimator sample that created that source plan.
    int64_t  sourceGridNs{ 0 };          ///< Source plan's frozen monotonic main target.
    int64_t  baseGridNs{ 0 };            ///< Future grid before applying the phase correction.
    int64_t  candidateNs{ 0 };           ///< Median future phase candidate, using the updated estimator.
    double   phaseGain{ 0 };             ///< Gain used for this update, retained across queued configuration epochs.
    int64_t  phaseSlewNs{ 0 };           ///< Maximum absolute phase correction for this update.
    int64_t  rawCorrectionNs{ 0 };       ///< Unclamped correction truncated toward zero to integer nanoseconds.
    int64_t  appliedCorrectionNs{ 0 };   ///< Original bounded correction applied to the base grid.
    int64_t  committedGridNs{ 0 };       ///< Newly committed future target; zero if no commitment succeeded.
    bool     committed{ false };         ///< A new future plan was actually committed by this update.
};

/// Process-lifetime triggered estimator coverage; selected equals drained plus dropped plus queued.
struct EstimatorStatus
{
    bool     enabled{ false };                   ///< Shadow estimator collection is selected.
    uint64_t observed{ 0 };                      ///< All completed accepted-primary estimator samples.
    uint64_t triggers{ 0 };                      ///< Overlap observations requesting a pre/post history window.
    uint64_t selected{ 0 };                      ///< Unique samples selected for export, including FIFO losses.
    uint64_t drained{ 0 };                       ///< Selected samples removed for a telemetry attempt.
    uint64_t dropped{ 0 };                       ///< Selected samples discarded because the export FIFO was full.
    uint64_t queued{ 0 };                        ///< Samples awaiting telemetry.
    uint64_t capacity{ estimatorTraceCapacity }; ///< Fixed export FIFO capacity.
    uint64_t logFailed{ 0 }; ///< App-owned count of drained samples lost to detected telemetry failures.
    uint64_t history{ 0 };   ///< Currently retained pre-trigger samples; reset clears this history.
    uint64_t historyCapacity{ estimatorHistoryCapacity }; ///< Fixed pre-trigger ring capacity.
    uint64_t postRemaining{ 0 }; ///< Accepted-primary samples still to export after the latest trigger.
};

/// Bounded drained telemetry batch; status describes the FIFO after this drain.
struct TraceBatch
{
    std::array<OverlapEvent, traceBatchCapacity> events{};   ///< Initialized prefix contains drained event records.
    size_t                                       count{ 0 }; ///< Initialized events; never exceeds traceBatchCapacity.
    TraceStatus status; ///< Process-lifetime collection and telemetry-failure accounting.
    std::array<EstimatorSample, estimatorBatchCapacity> estimator{}; ///< Selected pre/post estimator updates.
    size_t          estimatorCount{ 0 }; ///< Initialized estimator prefix, bounded by estimatorBatchCapacity.
    EstimatorStatus estimatorStatus;     ///< Independent estimator-history coverage and loss accounting.
};

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
