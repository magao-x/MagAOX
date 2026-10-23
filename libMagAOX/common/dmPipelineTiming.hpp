/** \file dmPipelineTiming.hpp
 * \brief Best-effort upstream AO trace timing conditioned on primary publication excursions.
 */
#ifndef common_dmPipelineTiming_hpp
#define common_dmPipelineTiming_hpp

#include "dmTiming.hpp"

namespace MagAOX
{
namespace dmPipelineTiming
{

constexpr size_t maxStages = 10; ///< Maximum existing process-trace rows captured per observation.

using Stats = MagAOX::dmTiming::Stats; ///< Duration population; change populations also permit negative values.

/// Cumulative observation-quality counts, independent of the trailing statistics window.
struct Counters
{
    uint64_t events{ 0 };          ///< Events consumed, including unavailable observations.
    uint64_t receipts{ 0 };        ///< Events carrying a real successful semaphore receipt.
    uint64_t synthetic{ 0 };       ///< Events without a successful semaphore receipt.
    uint64_t rejected{ 0 };        ///< Real receipts rejected before callback processing.
    uint64_t duplicates{ 0 };      ///< Duplicate primary output counters.
    uint64_t counterGaps{ 0 };     ///< Nonconsecutive primary output counter changes.
    uint64_t invalidMetadata{ 0 }; ///< Unstable or malformed publication metadata.
    uint64_t missingTrace{ 0 };    ///< No populated process-trace row was present.
    uint64_t invalidTrace{ 0 };    ///< Unstable, malformed, nonprogressing or unreceived trace chains.
    uint64_t topologyChanges{ 0 }; ///< Populated row identities changed between consecutive observations.
    uint64_t truncated{ 0 };       ///< Source advertised more rows than the bounded capture can retain.
    uint64_t dropped{ 0 };         ///< Events lost because the preallocated handoff queue was full.
};

/// One producer stage identified by trace position, writer and its trigger-input inode.
struct Stage
{
    uint64_t rowIndex{ 0 };   ///< Zero is the primary writer; increasing rows are further upstream.
    int64_t  pid{ 0 };        ///< Writer PID reported by this trace row.
    uint64_t inputInode{ 0 }; ///< Trigger-input inode, not the inode of this stage's output.

    Stats execution; ///< Current stage start to output-update duration.
    Stats handoff;   ///< Next upstream row update to current stage start; unavailable for the last row.

    Stats lateExecution;  ///< Execution when the current primary publication interval is late.
    Stats lateHandoff;    ///< Handoff when the current primary publication interval is late.
    Stats earlyExecution; ///< Execution when the current primary publication interval is early.
    Stats earlyHandoff;   ///< Handoff when the current primary publication interval is early.

    Stats lateExecutionChange;  ///< Signed execution change from the previous chain during a late interval.
    Stats lateHandoffChange;    ///< Signed handoff change from the previous chain during a late interval.
    Stats earlyExecutionChange; ///< Signed execution change from the previous chain during an early interval.
    Stats earlyHandoffChange;   ///< Signed handoff change from the previous chain during an early interval.
};

/// Consumer-owned snapshot; trace associations are consistency checked rather than frame provenance.
struct Summary
{
    bool     enabled{ false }; ///< Capture storage and a trailing observation window were configured.
    uint64_t session{ 0 };     ///< Monitor attachment epoch.
    uint64_t sequence{ 0 };    ///< Consumer update sequence.
    int64_t  nowNs{ 0 };       ///< Local monotonic evaluation timestamp.
    double   windowSec{ 0 };   ///< Configured trailing observation duration in seconds.
    double   span{ 0 };        ///< Retained local observation coverage in seconds.
    double   age{ -1 };        ///< Newest local observation age in seconds; -1 when unavailable.
    size_t   capacity{ 0 };    ///< Bounded queue and retained-event capacity.
    size_t   queued{ 0 };      ///< Queue occupancy after the consumer drain.
    double   median{ 0 };      ///< Median qualified primary publication interval in seconds.
    double   threshold{ 0 };   ///< Excursion distance from the median in seconds; zero disables conditional statistics.
    uint64_t intervals{ 0 };   ///< Qualified primary publication intervals, including ordinary tails.
    uint64_t late{ 0 };        ///< Qualified publication intervals strictly above median plus threshold.
    uint64_t early{ 0 };       ///< Qualified publication intervals strictly below median minus threshold.

    Stats sourceInterval;     ///< Consecutive oldest represented stage starts; boundary timing, not camera acquisition.
    Stats lateSourceInterval; ///< Oldest-stage start interval when the primary publication interval is late.
    Stats earlySourceInterval; ///< Oldest-stage start interval when the primary publication interval is early.
    Stats
        latePublicationChange; ///< Signed change in writetime minus row-zero update during late publication intervals.
    Stats earlyPublicationChange; ///< Signed change in writetime minus row-zero update during early publication
                                  ///< intervals.

    size_t                       stageCount{ 0 }; ///< Populated rows in the latest coherent observed topology.
    std::array<Stage, maxStages> stages{};        ///< Statistics use only the latest observed topology.
    Counters total; ///< Quality counts since configuration, including observations outside the window.
};

} // namespace dmPipelineTiming
} // namespace MagAOX
#endif
