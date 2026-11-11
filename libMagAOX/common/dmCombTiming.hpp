/** \file dmCombTiming.hpp
 * \brief Best-effort combined DM stream trace timing and collection quality.
 */
#ifndef common_dmCombTiming_hpp
#define common_dmCombTiming_hpp

#include "dmTiming.hpp"

namespace MagAOX
{
namespace dmCombTiming
{

/// Stable duration identifiers; publication and trace durations retain the producer's own clock.
enum class Metric : uint8_t
{
    publicationInterval,
    receiptInterval,
    execution,
    publication,
    primaryStart,
    primaryPublication,
    count
};

using Stats = MagAOX::dmTiming::Stats; ///< Independent duration population with local observation coverage.

/// Cumulative observation-quality counts; none certifies atomic stream/frame association.
struct Counters
{
    uint64_t events{ 0 };          ///< Timing observations consumed.
    uint64_t receipts{ 0 };        ///< Observations with a real semaphore receipt timestamp.
    uint64_t synthetic{ 0 };       ///< Observations without a semaphore receipt.
    uint64_t rejected{ 0 };        ///< Receipts rejected before the pixel callback.
    uint64_t duplicates{ 0 };      ///< Repeated observed output counters.
    uint64_t counterGaps{ 0 };     ///< Nonconsecutive output-counter changes.
    uint64_t invalidMetadata{ 0 }; ///< Unusable or unstable metadata observations.
    uint64_t missingTrace{ 0 };    ///< No first process-trace entry is available.
    uint64_t invalidTrace{ 0 };    ///< Unstable trace, invalid writer or unordered/invalid timestamps.
    uint64_t repeatedStart{ 0 };   ///< Process-start timestamp failed to advance for the same writer.
    uint64_t unlinkedPrimary{ 0 }; ///< Trace trigger inode does not identify the expected primary stream.
    uint64_t invalidPrimary{ 0 };  ///< Linked upstream trace is missing or has invalid/unordered timestamps.
    uint64_t repeatedPrimary{ 0 }; ///< Linked primary counter or upstream update failed to advance.
    uint64_t dropped{ 0 };         ///< Timing handoffs dropped when the bounded queue was full.
};

/// Consumer-owned aggregate; each metric exposes its own availability and population.
struct Summary
{
    uint64_t session{ 0 };     ///< Monitor attachment epoch.
    uint64_t sequence{ 0 };    ///< Summary update sequence.
    int64_t  nowNs{ 0 };       ///< Local monotonic evaluation time.
    double   windowSec{ 0 };   ///< Configured trailing window in seconds.
    double   span{ 0 };        ///< Retained local observation coverage in seconds.
    double   age{ -1 };        ///< Age of newest local observation, or -1 when unavailable.
    size_t   capacity{ 0 };    ///< Queue and retained-event cap.
    size_t   queued{ 0 };      ///< Handoff occupancy after the bounded consumer drain.
    bool     enabled{ false }; ///< Collection was enabled and preallocated.

    std::array<Stats, static_cast<size_t>( Metric::count )> metrics{}; ///< Independent timing populations.

    Counters total; ///< Cumulative quality counters since configure.

    int64_t  writerPid{ 0 };      ///< Writer reported by the latest accepted trace observation.
    uint64_t triggerInode{ 0 };   ///< Trigger inode reported by that trace, without atomic identity guarantees.
    uint64_t primaryCounter{ 0 }; ///< Named input counter reported by the trace; primary only when its inode is linked.
};

/// Convert a stable metric identifier to its population index.
inline constexpr size_t index( Metric metric /**< [in] Metric identifier. */ );

/// Get the stable telemetry name for a metric.
inline const char *metricName( Metric metric /**< [in] Metric identifier. */ );

inline constexpr size_t index( Metric metric )
{
    return static_cast<size_t>( metric );
}

inline const char *metricName( Metric metric )
{
    constexpr const char *names[] = {
        "publicationInterval", "receiptInterval", "execution", "publication", "primaryStart", "primaryPublication" };

    return index( metric ) < index( Metric::count ) ? names[index( metric )] : "unknown";
}

} // namespace dmCombTiming
} // namespace MagAOX
#endif
