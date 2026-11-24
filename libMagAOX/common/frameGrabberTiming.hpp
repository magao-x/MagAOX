/** \file frameGrabberTiming.hpp
 * \brief Paired camera timestamps, optional acquisition stages and ordinary-thread summary records.
 */
#ifndef common_frameGrabberTiming_hpp
#define common_frameGrabberTiming_hpp

#include <array>
#include <cstddef>
#include <cstdint>
#include <ctime>

#include "timingStats.hpp"

namespace MagAOX
{
namespace fgTiming
{
/// Shared seconds-based population, preserving existing framegrabber timing units.
using Stats = MagAOX::timing::Stats;

/// Stable local endpoint identifiers; capture order may defer rearm into the post-publication hook.
enum class Marker : uint8_t
{
    waitEnter,
    waitReturn,
    rearmReturn,
    validated,
    loadDone,
    writeStampDone,
    mainPostDone,
    syncPostDone, ///< Post-publication hook return; optional rearm can follow its notification before this endpoint.
    loopTailDone,
    rearmEnter,
    timeoutEnter,
    timeoutReturn,
    count
};

/// Independent populations; acquisitionToWrite retains its legacy external-clock meaning.
enum class Metric : uint8_t
{
    acquisitionInterval,
    publicationInterval,
    acquisitionToWrite,
    waitReturnInterval,
    writeEndpointInterval,
    wait,
    rearm,
    validation,
    conversion,
    writeBracket,
    mainPost,
    syncPost,
    loopTail,
    waitToWrite,
    previousTail,
    serialOverlap,
    timeoutQuery,
    count
};

/// Result of one acquisition attempt, independent of available diagnostic markers.
enum class Outcome : uint8_t
{
    published,
    rejected,
    failed
};

/// Availability of the three compatible framegrabber populations.
enum class Availability : uint8_t
{
    disabled,
    warmingUp,
    valid,
    stale,
    unavailable
};

/// Cumulative qualification counts; gaps count observations rather than inferred missing frames.
struct Counters
{
    uint64_t events{ 0 };         ///< Consumed acquisition attempts.
    uint64_t published{ 0 };      ///< Attempts that published output pixels.
    uint64_t rejected{ 0 };       ///< Deliberately discarded attempts.
    uint64_t failed{ 0 };         ///< SDK/processing/publication failures.
    uint64_t duplicates{ 0 };     ///< Repeated output counters in adjacent events.
    uint64_t counterGaps{ 0 };    ///< Nonunit output-counter observations.
    uint64_t frameGaps{ 0 };      ///< Nonunit validated OCAM-frame observations, excluding exact unsigned wrap.
    uint64_t invalid{ 0 };        ///< Invalid markers or external timestamp observations.
    uint64_t bracketWide{ 0 };    ///< Events exceeding a configured positive write-sample bracket limit.
    uint64_t dropped{ 0 };        ///< Acquisition events lost because the queue was full.
    uint64_t controls{ 0 };       ///< Consumed completed serial transactions.
    uint64_t controlFailed{ 0 };  ///< Transactions whose caller reported failure.
    uint64_t controlInvalid{ 0 }; ///< Invalid/backward transaction endpoints.
    uint64_t controlDropped{ 0 }; ///< Transactions lost because their queue was full.
};

/// Fixed producer event with paired timestamps and optional local stage markers.
struct Event
{
    uint64_t sequence{ 0 };        ///< Attempt sequence, including queue losses.
    uint64_t session{ 0 };         ///< Acquisition/configuration epoch.
    uint64_t frameNumber{ 0 };     ///< Validated raw camera frame number when supplied.
    uint64_t outputCounter{ 0 };   ///< Counter of the published output frame.
    uint64_t streamInode{ 0 };     ///< Output-stream metadata identity.
    uint32_t mode{ 0 };            ///< Caller-defined immutable mode tag for this attempt.
    uint32_t binX{ 0 };            ///< Configured software binning factor, zero when unknown.
    uint32_t binY{ 0 };            ///< Configured software binning factor, zero when unknown.
    bool     frameValid{ false };  ///< Raw camera frame number was validated.
    bool     outputValid{ false }; ///< Caller attached published frame metadata.
    timespec atime{};              ///< Original external SDK timestamp, unchanged.
    timespec writetime{};          ///< Existing output CLOCK_REALTIME timestamp, unchanged.
    std::array<int64_t, static_cast<size_t>( Marker::count )> markers{};  ///< Local monotonic ns.
    uint32_t                                                  valid{ 0 }; ///< Available local stage marker bits.
    uint32_t                                                  invalidMarkers{ 0 }; ///< Failed/backward local captures.
    Outcome                                                   outcome{ Outcome::rejected }; ///< Attempt result.
    int64_t anchorNs{ -1 }; ///< Consumer-assigned local retention anchor; never read from producer storage.
};

/// Completed ordinary control operation; its producer is serialized separately from acquisition.
struct ControlEvent
{
    int64_t entered{ -1 };    ///< Local monotonic transaction entry.
    int64_t completed{ -1 };  ///< Local monotonic transaction completion.
    bool    success{ false }; ///< Caller-qualified transaction result.
};

/// Raw direction of a monotonic publication excursion against the preceding rolling median.
enum class TraceDirection : uint8_t
{
    late,
    early
};

/// Why a fixed chronological neighborhood stopped accepting following frames.
enum class TraceEnd : uint8_t
{
    complete,
    boundary,
    expired
};

/// Consumer-selected excursion with two preceding and two following acquisition attempts.
/** Index two is the trigger. Selection uses at least three strictly preceding
 * consecutive intervals within the configured bounded trailing history. Wide or
 * absent write brackets do not censor raw selection; qualified reports their
 * independent attribution status. Completed-control overlap is only what the
 * consumer knows when finalizing this record, never full in-flight coverage.
 */
struct TraceRecord
{
    uint64_t               id{ 0 };            ///< Monotonically increasing detected-excursion identity.
    uint8_t                present{ 0 };       ///< Available neighborhood positions, bits zero through four.
    bool                   qualified{ false }; ///< Trigger and preceding write brackets pass attribution checks.
    TraceDirection         direction{ TraceDirection::late }; ///< Signed raw interval direction.
    TraceEnd               end{ TraceEnd::complete };         ///< Completion or interruption of following neighbors.
    int64_t                medianNs{ 0 };          ///< Prior-interval median, midpoint rounded down to integer ns.
    uint64_t               baselineIntervals{ 0 }; ///< Number of strictly preceding intervals in this baseline.
    int64_t                intervalNs{ 0 };        ///< Trigger's monotonic write-endpoint interval, ns.
    int64_t                thresholdNs{ 0 };       ///< Strict distance used for raw selection, ns.
    int64_t                bracketLimitNs{ 0 };    ///< Independent attribution limit; zero means undecided.
    int64_t                evaluatedNs{ 0 };       ///< Consumer monotonic finalization time, ns.
    int64_t                controlCoverageNs{ 0 }; ///< Evaluation cutoff for observed completed controls only.
    std::array<Event, 5>   events{};               ///< Raw ordered attempts with all available clocks and provenance.
    std::array<int64_t, 5> serialOverlapNs{ -1, -1, -1, -1, -1 }; ///< Known completed-control overlap per frame, ns.
};

/// Bounded consumer trace retention and cumulative diagnostic-loss accounting.
struct TraceStatus
{
    bool     enabled{ false };     ///< Stages, positive capacity and an excursion threshold of at least one ns exist.
    size_t   capacity{ 0 };        ///< Maximum ready plus pending neighborhoods.
    size_t   ready{ 0 };           ///< Finalized neighborhoods awaiting ordinary-thread delivery.
    size_t   pending{ 0 };         ///< Neighborhoods still awaiting following frames.
    uint64_t detected{ 0 };        ///< Raw excursions identified, including capacity losses.
    uint64_t recorded{ 0 };        ///< Complete or partial neighborhoods finalized since configure.
    uint64_t dropped{ 0 };         ///< Newest neighborhoods rejected because retention was full.
    uint64_t incomplete{ 0 };      ///< Finalized neighborhoods missing one or more requested neighbors.
    uint64_t unclassified{ 0 };    ///< Valid intervals observed before three preceding baseline intervals exist.
    uint64_t discontinuities{ 0 }; ///< Observed epoch, stream, attempt, counter or marker continuity breaks.
    uint64_t eventDropped{ 0 };    ///< Acquisition handoff losses since configure.
    uint64_t controlDropped{ 0 };  ///< Completed-control handoff losses since configure.
};

/// Coherent ordinary-thread cache; raw metrics are independent of conditional attribution.
struct Summary
{
    uint64_t     session{ 0 };           ///< Current acquisition epoch.
    uint64_t     sequence{ 0 };          ///< Consumer summary sequence.
    int64_t      nowNs{ 0 };             ///< Local monotonic evaluation time.
    int64_t      newestNs{ -1 };         ///< Newest retained local anchor.
    double       windowSec{ 0 };         ///< Configured trailing duration, seconds.
    double       span{ 0 };              ///< Retained local coverage, seconds.
    double       age{ -1 };              ///< Newest observation age, seconds; -1 when unavailable.
    size_t       capacity{ 0 };          ///< Maximum retained events per channel.
    size_t       queued{ 0 };            ///< Acquisition queue population after bounded drain.
    size_t       controlQueued{ 0 };     ///< Serial queue population after bounded drain.
    bool         enabled{ false };       ///< Paired collection is configured.
    bool         stagesEnabled{ false }; ///< Optional stage timestamps and control observations are configured.
    bool         fgValid{ false };       ///< All three legacy framegrabber populations have sample variance.
    Availability state{ Availability::disabled };                      ///< Compatible population availability.
    std::array<Stats, static_cast<size_t>( Metric::count )> metrics{}; ///< Raw independent populations.
    std::array<Stats, static_cast<size_t>( Metric::count )>
        lateChange{}; ///< Signed current-minus-previous durations for late intervals.
    std::array<Stats, static_cast<size_t>( Metric::count )>
             earlyChange{};              ///< Signed current-minus-previous durations for early intervals.
    Stats    lateWaitReturnInterval;     ///< SDK-return intervals for the qualified late-output cohort.
    Stats    earlyWaitReturnInterval;    ///< SDK-return intervals for the qualified early-output cohort.
    Stats    lateWriteEndpointInterval;  ///< Monotonic write intervals for the qualified late-output cohort.
    Stats    earlyWriteEndpointInterval; ///< Monotonic write intervals for the qualified early-output cohort.
    Stats    latePreviousTail;           ///< Previous write endpoint to current wait entry for late outputs.
    Stats    earlyPreviousTail;          ///< Previous write endpoint to current wait entry for early outputs.
    Stats    serialDuration;             ///< Completed serial durations; overlap alone does not establish causation.
    double   median{ 0 };                ///< Qualified raw monotonic write-endpoint interval median, seconds.
    double   threshold{ 0 };             ///< Excursion distance from that median, seconds.
    double   bracketLimit{ 0 };   ///< Positive maximum accepted write bracket; zero leaves attribution undecided.
    uint64_t intervals{ 0 };      ///< Consecutive published monotonic endpoint intervals.
    uint64_t qualified{ 0 };      ///< Intervals with two sufficiently narrow complete brackets.
    uint64_t ambiguous{ 0 };      ///< Other raw intervals; never censored from raw populations.
    uint64_t late{ 0 };           ///< Qualified intervals strictly above median plus threshold.
    uint64_t early{ 0 };          ///< Qualified intervals strictly below median minus threshold.
    uint64_t serialOverlaps{ 0 }; ///< Retained attempts overlapping a completed observed serial operation.
    uint64_t streamInode{ 0 };    ///< Latest retained output metadata identity.
    uint64_t lastFrame{ 0 };      ///< Latest supplied camera frame number.
    uint64_t lastCounter{ 0 };    ///< Latest supplied output counter.
    uint32_t mode{ 0 };           ///< Latest caller-supplied mode tag.
    uint32_t binX{ 0 };           ///< Latest software binning factor.
    uint32_t binY{ 0 };           ///< Latest software binning factor.
    Counters total;               ///< Cumulative observed outcomes and losses since configure.
    Counters delta;               ///< Changes since the preceding summary.
};

/// Convert a metric identifier to its stable array position.
inline constexpr size_t metricIndex( Metric metric /**< [in] Population identifier. */ );

/// Convert a marker identifier to its stable array position.
inline constexpr size_t markerIndex( Marker marker /**< [in] Endpoint identifier. */ );

/// Return one stable human-readable metric name.
inline const char *metricName( Metric metric /**< [in] Population identifier. */ );

/// Validate available nonnegative endpoints in an immediate or explicitly deferred rearm sequence.
inline bool validMarkerOrder(
    const std::array<int64_t, static_cast<size_t>( Marker::count )> &markers /**< [in] Local endpoint values. */,
    uint32_t valid /**< [in] Available endpoint bits; unknown bits are rejected. */,
    bool     allowIncompleteDeferred = false /**< [in] Permit missing entry after a reported capture failure. */ );

/// Return the stable quality-member sequence used by readbacks and deltas.
inline const std::array<uint64_t Counters::*, 14> &counterFields();

/// Return quality names in the same order as counterFields.
inline const std::array<const char *, 14> &counterNames();

inline constexpr size_t metricIndex( Metric metric )
{
    return static_cast<size_t>( metric );
}
inline constexpr size_t markerIndex( Marker marker )
{
    return static_cast<size_t>( marker );
}
inline const char *metricName( Metric metric )
{
    constexpr const char *names[] = { "acquisitionInterval",
                                      "publicationInterval",
                                      "acquisitionToWrite",
                                      "waitReturnInterval",
                                      "writeEndpointInterval",
                                      "wait",
                                      "rearm",
                                      "validation",
                                      "conversion",
                                      "writeBracket",
                                      "mainPost",
                                      "syncPost",
                                      "loopTail",
                                      "waitToWrite",
                                      "previousTail",
                                      "serialOverlap",
                                      "timeoutQuery" };
    return metricIndex( metric ) < metricIndex( Metric::count ) ? names[metricIndex( metric )] : "unknown";
}
inline bool validMarkerOrder( const std::array<int64_t, static_cast<size_t>( Marker::count )> &markers,
                              uint32_t                                                         valid,
                              bool                                                             allowIncompleteDeferred )
{
    constexpr uint32_t known = ( 1u << markerIndex( Marker::count ) ) - 1;

    if( valid & ~known )
    {
        return false;
    }

    constexpr std::array<Marker, markerIndex( Marker::count )> immediate{ Marker::waitEnter,
                                                                          Marker::waitReturn,
                                                                          Marker::rearmEnter,
                                                                          Marker::rearmReturn,
                                                                          Marker::timeoutEnter,
                                                                          Marker::timeoutReturn,
                                                                          Marker::validated,
                                                                          Marker::loadDone,
                                                                          Marker::writeStampDone,
                                                                          Marker::mainPostDone,
                                                                          Marker::syncPostDone,
                                                                          Marker::loopTailDone };
    constexpr std::array<Marker, markerIndex( Marker::count )> deferred{ Marker::waitEnter,
                                                                         Marker::waitReturn,
                                                                         Marker::timeoutEnter,
                                                                         Marker::timeoutReturn,
                                                                         Marker::validated,
                                                                         Marker::loadDone,
                                                                         Marker::writeStampDone,
                                                                         Marker::mainPostDone,
                                                                         Marker::rearmEnter,
                                                                         Marker::rearmReturn,
                                                                         Marker::syncPostDone,
                                                                         Marker::loopTailDone };

    constexpr std::array orders{ immediate, deferred };

    for( size_t path = 0; path < orders.size(); ++path )
    {
        if( path && !allowIncompleteDeferred && !( valid & ( 1u << markerIndex( Marker::rearmEnter ) ) ) )
        {
            continue;
        }

        int64_t previous = -1;
        bool    ordered  = true;

        for( Marker marker : orders[path] )
        {
            const size_t index = markerIndex( marker );

            if( valid & ( 1u << index ) )
            {
                if( markers[index] < 0 || markers[index] < previous )
                {
                    ordered = false;
                    break;
                }

                previous = markers[index];
            }
        }

        if( ordered )
        {
            return true;
        }
    }

    return false;
}
inline const std::array<uint64_t Counters::*, 14> &counterFields()
{
    static constexpr std::array<uint64_t Counters::*, 14> fields{ &Counters::events,
                                                                  &Counters::published,
                                                                  &Counters::rejected,
                                                                  &Counters::failed,
                                                                  &Counters::duplicates,
                                                                  &Counters::counterGaps,
                                                                  &Counters::frameGaps,
                                                                  &Counters::invalid,
                                                                  &Counters::bracketWide,
                                                                  &Counters::dropped,
                                                                  &Counters::controls,
                                                                  &Counters::controlFailed,
                                                                  &Counters::controlInvalid,
                                                                  &Counters::controlDropped };
    return fields;
}
inline const std::array<const char *, 14> &counterNames()
{
    static constexpr std::array<const char *, 14> names{ "events",
                                                         "published",
                                                         "rejected",
                                                         "failed",
                                                         "duplicates",
                                                         "counterGaps",
                                                         "frameGaps",
                                                         "invalid",
                                                         "bracketWide",
                                                         "dropped",
                                                         "controls",
                                                         "controlFailed",
                                                         "controlInvalid",
                                                         "controlDropped" };
    return names;
}
} // namespace fgTiming
} // namespace MagAOX
#endif
