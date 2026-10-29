/** \file dmReaderTiming.hpp
 * \brief Qualified reader-protection measurements joining primary, combined and independent controller endpoint
 * observations and semaphore-wait returns.
 */
#ifndef common_dmReaderTiming_hpp
#define common_dmReaderTiming_hpp

#include <array>
#include <cstddef>
#include <cstdint>

#include "dmCompletion.hpp"
#include "timingStats.hpp"

namespace MagAOX
{
namespace dmReaderTiming
{

/// Exact raw producer identity; external publication fields are never mixed with local monotonic times.
struct Identity
{
    uint64_t inode{ 0 };   ///< Shared input inode.
    uint64_t counter{ 0 }; ///< Input publication counter.
    int64_t  sec{ 0 };     ///< Raw producer writetime seconds.
    int64_t  nsec{ 0 };    ///< Raw producer writetime nanoseconds.
};

/// Coherent primary observation captured in its owning monitor thread.
struct Primary
{
    Identity input;           ///< Primary command identity.
    int64_t  observedNs{ 0 }; ///< Local monotonic primary observation time.
};

/// Combined observation with a named primary trace link and frozen modeled hold budget.
struct Combined
{
    Identity input;               ///< Combined publication identity consumed by the DM controller.
    uint64_t primaryInode{ 0 };   ///< Named primary stream inode reported by the trace.
    uint64_t primaryCounter{ 0 }; ///< Named primary publication counter reported by the trace.
    int64_t  startNs{ 0 };        ///< External producer-clock trace start, in nanoseconds.
    int64_t  observedNs{ 0 };     ///< Local monotonic combined observation time.
    int64_t  budgetNs{ 0 };       ///< Frozen modeled reader-protection budget, in nanoseconds.
};

/// Coherent dedicated completion record captured by its own observer.
struct Completed
{
    MagAOX::dmCompletion::Record record;          ///< Versioned controller record with exact combined-input identity.
    int64_t                      observedNs{ 0 }; ///< Local monotonic completion-stream observation time.
    int64_t receiptNs{ 0 }; ///< Local monotonic semaphore-wait return; zero means unavailable in legacy captures.
};

/// Independent timing populations; offset and slack remain signed, all statistics use seconds.
enum class Metric : uint8_t
{
    completionOffset,
    requiredHold,
    observedHold,
    processing,
    notification,
    modelSlack,
    wake,    ///< Finish-to-wait-return delay for a valid original endpoint observation.
    capture, ///< Wait-return-to-coherent-capture delay, paired with the same wake population.
    count
};

/// Full provenance for one successfully qualified three-way join.
struct Sample
{
    Primary   primary;             ///< Named primary observation used by the join.
    Combined  combined;            ///< Exact combined observation and frozen budget used by the join.
    Completed completed;           ///< Qualified controller completion and its observation time.
    int64_t   baseNs{ 0 };         ///< Later primary/combined observation, matching the existing protection baseline.
    int64_t   offsetNs{ 0 };       ///< Signed controller finish minus baseline.
    int64_t   requiredNs{ 0 };     ///< Nonnegative hold until software reads finished.
    int64_t   observedHoldNs{ 0 }; ///< Nonnegative hold until completion was observed.
    int64_t   slackNs{ 0 };        ///< Frozen modeled budget minus signed completion offset.
};

/// Lifetime observation quality; misses describe observed evidence rather than inferred physical frames.
struct Counters
{
    uint64_t events{ 0 };           ///< Source observations consumed, including rejected ones.
    uint64_t primary{ 0 };          ///< Primary observations consumed.
    uint64_t combined{ 0 };         ///< Combined observations consumed.
    uint64_t completion{ 0 };       ///< Completion observations consumed.
    uint64_t matched{ 0 };          ///< Successfully qualified three-way joins.
    uint64_t busy{ 0 };             ///< Busy controller records, never qualified as completion.
    uint64_t unavailable{ 0 };      ///< Unavailable controller records.
    uint64_t error{ 0 };            ///< Controller error records.
    uint64_t unqualified{ 0 };      ///< Unqualified controller records, including no-send and synthetic callbacks.
    uint64_t invalid{ 0 };          ///< Invalid timestamps, identities, provenance or completed-record semantics.
    uint64_t incoherent{ 0 };       ///< Source captures explicitly failed metadata consistency checks.
    uint64_t duplicates{ 0 };       ///< Repeated or backward source counter/sequence observations.
    uint64_t sequenceGaps{ 0 };     ///< Missing completion attempts inferred from sequence advances.
    uint64_t counterGaps{ 0 };      ///< Nonconsecutive source-counter advances, counted as occurrences.
    uint64_t sourceResets{ 0 };     ///< Attachment or completion publisher/session/epoch reset transitions.
    uint64_t staleEpoch{ 0 };       ///< Queued observations discarded after source attachment reset.
    uint64_t missingPrimary{ 0 };   ///< Removed completed/combined pairs lacking a named primary observation.
    uint64_t missingCombined{ 0 };  ///< Removed completion records lacking their combined observation.
    uint64_t identityMismatch{ 0 }; ///< Same inode/counter had different raw publication identity or ordering.
    uint64_t evicted{ 0 };          ///< Pending or matched observations removed by a capacity limit or source reset.
    uint64_t expired{ 0 }; ///< Retained source observations aged out, including reusable already-matched primaries.
    uint64_t droppedPrimary{ 0 };    ///< Primary capture handoffs dropped because their queue was full.
    uint64_t droppedCombined{ 0 };   ///< Combined capture handoffs dropped because their queue was full.
    uint64_t droppedCompletion{ 0 }; ///< Completion capture handoffs dropped because their queue was full.
    uint64_t sdkResults{ 0 };        ///< Distinct release attempts with an explicitly observed backend outcome.
    uint64_t sdkSuccess{ 0 };        ///< Observed actual SDK successes; reader release alone never increments this.
    uint64_t sdkFailed{
        0 }; ///< Observed backend failures or exceptions, including outcomes received after reader release.
    uint64_t receiptUnavailable{ 0 }; ///< First retained completions without a semaphore-wait return timestamp.
    uint64_t receiptInvalid{ 0 }; ///< First retained completions whose nonzero receipt lies outside finish-to-capture.
    uint64_t firstEnriched{ 0 };  ///< First retained reader-release snapshots already containing an SDK outcome.
};

/// Consumer-owned trailing statistics and lifetime worst-case evidence.
struct Summary
{
    MagAOX::dmCompletion::Kind kind{
        MagAOX::dmCompletion::Kind::completion }; ///< Endpoint protocol measured independently.

    bool     enabled{ false };       ///< Queues and consumer storage were configured before the producers started.
    uint64_t session{ 0 };           ///< Join epoch, incremented when attachment or publisher provenance changes.
    uint64_t sequence{ 0 };          ///< Consumer summary update sequence.
    int64_t  nowNs{ 0 };             ///< Local monotonic evaluation time.
    double   windowSec{ 0 };         ///< Configured trailing retention duration in seconds.
    size_t   capacity{ 0 };          ///< Maximum per-source queue and pending population, and retained matched samples.
    size_t   queued{ 0 };            ///< Remaining capture handoffs after the bounded drain.
    size_t   pendingPrimary{ 0 };    ///< Unexpired primary observations retained for out-of-order joins.
    size_t   pendingCombined{ 0 };   ///< Unexpired combined observations awaiting completion or primary evidence.
    size_t   pendingCompletion{ 0 }; ///< Unexpired qualified completion records awaiting their other observations.
    double   span{ 0 };              ///< Coverage of retained completion-observation timestamps, in seconds.
    double   age{ -1 };              ///< Age of newest qualified completion observation, or -1 when unavailable.
    std::array<MagAOX::timing::Stats, static_cast<size_t>( Metric::count )> metrics{}; ///< Independent populations.
    Counters total;               ///< Lifetime quality counts since configure.
    bool     worstValid{ false }; ///< A lifetime matched sample establishes the required-hold maximum.
    Sample   worst; ///< Exact lifetime evidence for the largest required hold, including the frozen budget.
    int64_t  maxRequiredHoldNs{ 0 }; ///< Lifetime maximum nonnegative software-reader hold.
    int64_t  maxObservedHoldNs{ 0 }; ///< Lifetime maximum nonnegative completion-observation hold.
    uint64_t overBudget{ 0 };        ///< Lifetime qualified joins whose signed completion offset exceeds their budget.
    bool     worstNotificationValid{ false }; ///< A lifetime matched sample establishes the largest notification delay.
    Sample   worstNotification; ///< Exact first-seen evidence for the largest finish-to-observation delay.
};

/// Convert one stable metric identifier to its array index.
inline constexpr size_t index( Metric metric /**< [in] Timing population identifier. */ );

/// Return the stable telemetry name of one timing population.
inline const char *metricName( Metric metric /**< [in] Timing population identifier. */ );

inline constexpr size_t index( Metric metric )
{
    return static_cast<size_t>( metric );
}

inline const char *metricName( Metric metric )
{
    constexpr const char *names[] = { "completionOffset",
                                      "requiredHold",
                                      "observedHold",
                                      "processing",
                                      "notification",
                                      "modelSlack",
                                      "wake",
                                      "capture" };
    return index( metric ) < index( Metric::count ) ? names[index( metric )] : "unknown";
}

} // namespace dmReaderTiming
} // namespace MagAOX
#endif
