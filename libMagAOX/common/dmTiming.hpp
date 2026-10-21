/** \file dmTiming.hpp
 * \brief Shared identifiers and aggregate records for DM controller timing.
 */
#ifndef common_dmTiming_hpp
#define common_dmTiming_hpp

#include <array>
#include <cstddef>
#include <cstdint>

namespace MagAOX
{
namespace dmTiming
{

/// Stable metric identifiers; all duration statistics use seconds.
enum class Metric : uint8_t
{
    receiptInterval,
    sendInterval,
    receiptSend,
    dispatch,
    actuator,
    sdk,
    saturation,
    delta,
    post,
    throughDelta,
    callback,
    receiptCallback,
    publicationInterval,
    count
};

/// Local monotonic endpoints, independent of external image timestamps.
enum class Marker : uint8_t
{
    receipt,
    entry,
    actuatorStart,
    actuatorEnd,
    sdkStart,
    sdkEnd,
    satStart,
    satEnd,
    deltaStart,
    deltaEnd,
    postStart,
    postEnd,
    finish,
    count
};

/// Outcome of the callback; a later failure does not erase an earlier submission.
enum class Outcome : uint8_t
{
    unknown,
    success,
    failed,
    partial,
    noSend,
    rejected
};

/// Availability of the compatible three-population timing summary.
enum class Availability : uint8_t
{
    disabled,
    warmingUp,
    valid,
    stale,
    unavailable
};

/// Aggregate of one independently valid metric population.
struct Stats
{
    uint64_t count{ 0 };     ///< Number of retained observations.
    double   mean{ 0 };      ///< Mean duration, in seconds; unavailable when count is zero.
    double   stddev{ 0 };    ///< Sample standard deviation, in seconds; unavailable below two observations.
    double   min{ 0 };       ///< Minimum duration in seconds.
    double   max{ 0 };       ///< Maximum duration in seconds.
    double   span{ 0 };      ///< Local endpoint/observation coverage in seconds, also for publication intervals.
    bool     valid{ false }; ///< True when sample variance is defined.
};

/// Quality counts; counter gaps count observations, not an inferred number of missing images.
struct Counters
{
    uint64_t events{ 0 };      ///< Timing events received by the consumer.
    uint64_t receipts{ 0 };    ///< Events containing a successful semaphore wait return.
    uint64_t submitted{ 0 };   ///< Events with a qualified successful submission endpoint.
    uint64_t failed{ 0 };      ///< Failed callbacks, possibly following a successful submission.
    uint64_t partial{ 0 };     ///< Commands with only a subset of their required writes completed.
    uint64_t noSend{ 0 };      ///< Callbacks intentionally making no submission.
    uint64_t synthetic{ 0 };   ///< Initial-image callbacks lacking a semaphore receipt.
    uint64_t rejected{ 0 };    ///< Receipts rejected before callback processing.
    uint64_t duplicates{ 0 };  ///< Repeated observed input counters.
    uint64_t counterGaps{ 0 }; ///< Nonconsecutive observed input-counter changes.
    uint64_t invalid{ 0 };     ///< Failed or backward local timestamp captures.
    uint64_t dropped{ 0 };     ///< Timing events dropped because the handoff queue was full.
};

/// Fixed-size command-thread handoff; no external timestamp implies verified frame identity.
struct Event
{
    uint64_t sequence{ 0 }; ///< Producer sequence, including events subsequently dropped.
    uint64_t session{ 0 };  ///< Monitor attachment/reset epoch.
    uint64_t counter{ 0 };  ///< Input counter observed by the callback, when available.

    int64_t publicationNs{ -1 }; ///< External writetime in ns; best-effort metadata, not atomic frame identity.

    bool    counterValid{ false };       ///< Whether the input-counter observation is available.
    bool    synthetic{ false };          ///< No successful semaphore wait preceded this event.
    bool    submitted{ false };          ///< SDK/transport success was explicitly qualified.
    Outcome outcome{ Outcome::unknown }; ///< Callback result independent of submission success.

    std::array<int64_t, static_cast<size_t>( Marker::count )> markers{};  ///< Monotonic nanoseconds.
    uint32_t                                                  valid{ 0 }; ///< Bit mask of available endpoints.
    uint32_t invalidMarkers{ 0 };                                         ///< Invalid timestamp captures in this event.
};

/// Consumer-owned coherent snapshot for telemetry and ordinary app logic.
struct Summary
{
    uint64_t session{ 0 };   ///< Current monitor epoch.
    uint64_t sequence{ 0 };  ///< Consumer summary sequence.
    int64_t  nowNs{ 0 };     ///< Monotonic evaluation time.
    int64_t  newestNs{ -1 }; ///< Newest retained event endpoint, or -1 when unavailable.

    double windowSec{ 0 }; ///< Configured trailing elapsed-time window in seconds.
    double span{ 0 };      ///< Actual retained endpoint span in seconds.
    double age{ -1 };      ///< Age of newest retained event in seconds; -1 when unavailable.
    size_t capacity{ 0 };  ///< Configured event/window cap.
    size_t queued{ 0 };    ///< Queue occupancy after the bounded drain.

    bool         enabled{ false };                ///< Collection is configured and preallocated.
    bool         fgValid{ false };                ///< All three populations required by telem_fgtimings are valid.
    Availability state{ Availability::disabled }; ///< Explicit compatible-summary availability.

    std::array<Stats, static_cast<size_t>( Metric::count )> metrics{}; ///< Independent metric populations.

    Counters total; ///< Cumulative quality counts since configure.
    Counters delta; ///< Changes since the preceding consumer summary.
};

/// Convert a stable metric identifier to its array index.
inline constexpr size_t metricIndex( Metric metric /**< [in] Metric identifier. */ );

/// Convert an endpoint identifier to its array index.
inline constexpr size_t markerIndex( Marker marker /**< [in] Endpoint identifier. */ );

/// Return the stable telemetry name for one metric.
inline const char *metricName( Metric metric /**< [in] Metric identifier. */ );

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
    constexpr const char *names[] = { "receiptInterval",
                                      "sendInterval",
                                      "receiptSend",
                                      "dispatch",
                                      "actuator",
                                      "sdk",
                                      "saturation",
                                      "delta",
                                      "post",
                                      "throughDelta",
                                      "callback",
                                      "receiptCallback",
                                      "publicationInterval" };

    return metricIndex( metric ) < metricIndex( Metric::count ) ? names[metricIndex( metric )] : "unknown";
}

} // namespace dmTiming
} // namespace MagAOX
#endif
