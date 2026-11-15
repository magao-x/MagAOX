/** \file loopTiming.hpp
 * \brief Bounded temporal association and statistics for loop timestamp observations.
 */
#ifndef loopTimingMonitor_loopTiming_hpp
#define loopTimingMonitor_loopTiming_hpp

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>

#include <mx/math/floatUtils.hpp>

#include "../../libMagAOX/common/timingStats.hpp"

namespace MagAOX
{
namespace app
{
namespace loopTiming
{

/// Lifetime observation-quality counts, preserved across association resets.
struct Counters
{
    uint64_t cameraMissed{ 0 };       ///< Camera publications skipped between observed counters.
    uint64_t sdkMissed{ 0 };          ///< SDK diagnostic publications skipped between observed counters.
    uint64_t counterRegressions{ 0 }; ///< Counter resets or wraps; missing counts cannot be inferred.
    uint64_t invalid{ 0 };            ///< Invalid timestamps, changed identities or out-of-bound latencies.
    uint64_t ambiguous{ 0 };          ///< Camera intervals containing multiple or boundary-overlapping SDK events.
    uint64_t unmatched{ 0 };          ///< Closed or expired camera intervals without a qualified SDK candidate.
    uint64_t overflow{ 0 };           ///< Pending-history resets or retained samples evicted by capacity.
    uint64_t accepted{ 0 };           ///< Qualified latency samples since configuration.
    uint64_t resets{ 0 };             ///< Explicit or detected breaks in association continuity.
};

/// Ordinary-thread snapshot; all duration populations use seconds.
struct Summary
{
    timing::Stats cameraIntervals; ///< Consecutive camera intervals in the camera's original clock.
    timing::Stats sdkIntervals;    ///< Consecutive SDK-start intervals in MONOTONIC.
    timing::Stats latency;         ///< Qualified camera-to-SDK timestamp differences in MONOTONIC.
    Counters      quality;         ///< Lifetime quality counts, including observations no longer retained.
    int64_t       newestNs{ 0 };   ///< Most recent accepted SDK endpoint, retained after window expiry.
    double        age{ -1 };       ///< Seconds since newestNs; negative before any accepted latency.
    uint64_t      epoch{ 1 };      ///< Association generation, changed whenever continuity is reset.
};

/// Associate timestamp-ordered sources under an externally established sub-period latency contract.
/** One relevant SDK submission per camera frame and true latency shorter than
 * one frame period are operating assumptions, not conclusions of this helper.
 * A later SDK timestamp closes each camera interval before it is classified.
 * Methods are serialized by one ordinary application thread; no stream access,
 * clock calls or synchronization occur here.
 */
class Monitor
{
  private:
    /// One observed endpoint, including camera mapping bounds and original-clock timestamp.
    struct Endpoint
    {
        uint64_t counter{ 0 }; ///< Producer's observed publication sequence.
        int64_t  stampNs{ 0 }; ///< Endpoint in the common MONOTONIC clock.
        int64_t  lowerNs{ 0 }; ///< Inclusive earliest endpoint allowed by mapping uncertainty.
        int64_t  upperNs{ 0 }; ///< Inclusive latest endpoint allowed by mapping uncertainty.
        int64_t  rawNs{ 0 };   ///< Timestamp in the original producer clock for cadence statistics.
    };

    /// One duration retained until its final endpoint leaves the statistics window.
    struct Sample
    {
        int64_t durationNs{ 0 }; ///< Integer difference before conversion to seconds.
        int64_t firstNs{ 0 };    ///< Earlier MONOTONIC coverage endpoint.
        int64_t lastNs{ 0 };     ///< Later MONOTONIC coverage endpoint and expiry key.
    };

    int64_t m_maxLatencyNs{ 0 };  ///< Strict upper latency bound derived from configured maximum FPS.
    int64_t m_windowNs{ 0 };      ///< Trailing statistics window in MONOTONIC nanoseconds.
    size_t  m_capacity{ 0 };      ///< Independent bound on each pending history and retained population.
    int64_t m_nowNs{ 0 };         ///< Previous summary time, used to detect local clock regressions.
    int64_t m_sdkCoverageNs{ 0 }; ///< Earliest SDK start observed in the current continuous association epoch.

    Endpoint m_lastCamera; ///< Last camera observation used for counter continuity and cadence.
    Endpoint m_lastSdk;    ///< Last SDK observation used for counter continuity and cadence.

    std::deque<Endpoint> m_cameras; ///< Pending camera endpoints awaiting a closing SDK watermark.
    std::deque<Endpoint> m_sdks;    ///< Pending ordered SDK starts; never reused by two accepted pairs.

    std::deque<Sample> m_cameraIntervals; ///< Accepted consecutive camera intervals.
    std::deque<Sample> m_sdkIntervals;    ///< Accepted consecutive SDK-start intervals.
    std::deque<Sample> m_latencies;       ///< Qualified camera-to-SDK differences.

    Summary m_summary; ///< Cached ordinary-thread statistics and lifetime quality.

    /// Validate one source observation and break both pending histories when continuity is lost.
    bool observe( const Endpoint &event /**< [in] New endpoint to validate. */,
                  bool            cameraSource /**< [in] True for the camera; false for the SDK source. */ );

    /// Finalize camera intervals whose complete SDK candidate population has become visible.
    void match();

    /// Retain one sample with bounded storage and explicit oldest-sample eviction counts.
    void retain( std::deque<Sample> &population /**< [in,out] One duration population. */,
                 const Sample       &sample /**< [in] Qualified integer duration and coverage. */ );

    /// Expire old samples and recompute numerically stable duration statistics.
    timing::Stats statistics( std::deque<Sample> &population /**< [in,out] Retained duration population. */,
                              int64_t cutoffNs /**< [in] Endpoints at or before this time have expired. */ );

  public:
    /// Configure finite positive bounds and clear all state, including lifetime quality counters.
    bool configure( double maxFPS /**< [in] Maximum supported camera rate, Hz. */,
                    double windowSec /**< [in] Positive trailing statistics window, seconds. */,
                    size_t capacity /**< [in] Maximum pending endpoints or samples per population; at least three. */ );

    /// Start a new association epoch while preserving lifetime quality counters.
    void reset( bool clearStats = true /**< [in] Also discard retained statistics and freshness when true. */ );

    /// Observe a camera endpoint and optional original-clock timestamp for independent cadence statistics.
    void camera( uint64_t counter /**< [in] Camera publication counter. */,
                 int64_t  stampNs /**< [in] Camera timestamp mapped into MONOTONIC. */,
                 int64_t  lowerNs /**< [in] Inclusive lower mapping/quantization bound. */,
                 int64_t  upperNs /**< [in] Inclusive upper mapping/quantization bound. */,
                 int64_t  rawStampNs = 0 /**< [in] Original-clock timestamp, or zero to use stampNs. */ );

    /// Observe one successful SDK start, independent of its later diagnostic publication time.
    void sdk( uint64_t counter /**< [in] SDK diagnostic publication counter. */,
              int64_t  startNs /**< [in] Original SDK-entry timestamp in MONOTONIC. */ );

    /// Expire old history and update cached statistics even when sources have stopped advancing.
    void update( int64_t nowNs /**< [in] Current MONOTONIC time, nanoseconds. */ );

    /// Return the cached populations and current lifetime quality counters.
    const Summary &summary() const;
};

inline bool Monitor::configure( double maxFPS, double windowSec, size_t capacity )
{
    const double periodNs = 1e9 / maxFPS;
    const double windowNs = windowSec * 1e9;

    if( !mx::math::isFinite( maxFPS ) || maxFPS <= 0 || !mx::math::isFinite( periodNs ) || periodNs < 1 ||
        periodNs >= double( std::numeric_limits<int64_t>::max() ) || !mx::math::isFinite( windowSec ) ||
        windowSec <= 0 || !mx::math::isFinite( windowNs ) || windowNs < 1 ||
        windowNs >= double( std::numeric_limits<int64_t>::max() ) || capacity < 3 )
    {
        return false;
    }

    m_maxLatencyNs = int64_t( periodNs );
    m_windowNs     = int64_t( windowNs );
    m_capacity     = capacity;
    reset();
    m_summary = {};
    return true;
}

inline void Monitor::reset( bool clearStats )
{
    m_cameras.clear();
    m_sdks.clear();
    m_lastCamera    = {};
    m_lastSdk       = {};
    m_nowNs         = 0;
    m_sdkCoverageNs = 0;
    ++m_summary.quality.resets;
    m_summary.epoch = m_summary.epoch == UINT64_MAX ? 1 : m_summary.epoch + 1;

    if( clearStats )
    {
        m_cameraIntervals.clear();
        m_sdkIntervals.clear();
        m_latencies.clear();
        m_summary.cameraIntervals = {};
        m_summary.sdkIntervals    = {};
        m_summary.latency         = {};
        m_summary.newestNs        = 0;
        m_summary.age             = -1;
    }
}

inline bool Monitor::observe( const Endpoint &event, bool cameraSource )
{
    if( !m_capacity )
    {
        return false;
    }

    auto &previous = cameraSource ? m_lastCamera : m_lastSdk;

    if( event.lowerNs <= 0 || event.stampNs < event.lowerNs || event.upperNs < event.stampNs || event.rawNs <= 0 )
    {
        ++m_summary.quality.invalid;
        reset();
        return false;
    }

    if( previous.stampNs )
    {
        if( event.counter == previous.counter )
        {
            if( event.rawNs != previous.rawNs )
            {
                ++m_summary.quality.invalid;
                reset();
            }

            return false;
        }

        if( event.counter < previous.counter )
        {
            ++m_summary.quality.counterRegressions;
            reset();
        }
        else if( event.counter - previous.counter > 1 )
        {
            auto &missed = cameraSource ? m_summary.quality.cameraMissed : m_summary.quality.sdkMissed;
            missed += event.counter - previous.counter - 1;

            if( event.stampNs <= previous.stampNs || event.rawNs <= previous.rawNs )
            {
                ++m_summary.quality.invalid;
                reset();
                return false;
            }

            reset( false );
        }
        else if( event.stampNs <= previous.stampNs || event.rawNs <= previous.rawNs )
        {
            ++m_summary.quality.invalid;
            reset();
            return false;
        }
    }

    auto &history = cameraSource ? m_cameras : m_sdks;

    if( history.size() == m_capacity )
    {
        ++m_summary.quality.overflow;
        reset( false );
    }

    if( previous.stampNs )
    {
        retain( cameraSource ? m_cameraIntervals : m_sdkIntervals,
                { event.rawNs - previous.rawNs, previous.stampNs, event.stampNs } );
    }

    previous = event;
    history.push_back( event );

    if( !cameraSource && !m_sdkCoverageNs )
    {
        m_sdkCoverageNs = event.stampNs;
    }

    match();
    return true;
}

inline void Monitor::camera( uint64_t counter, int64_t stampNs, int64_t lowerNs, int64_t upperNs, int64_t rawStampNs )
{
    observe( { counter, stampNs, lowerNs, upperNs, rawStampNs ? rawStampNs : stampNs }, true );
}

inline void Monitor::sdk( uint64_t counter, int64_t startNs )
{
    observe( { counter, startNs, startNs, startNs, startNs }, false );
}

inline void Monitor::retain( std::deque<Sample> &population, const Sample &sample )
{
    if( population.size() == m_capacity )
    {
        population.pop_front();
        ++m_summary.quality.overflow;
    }

    population.push_back( sample );
}

inline void Monitor::match()
{
    while( m_cameras.size() > 1 && !m_sdks.empty() )
    {
        const auto &first = m_cameras[0];
        const auto &next  = m_cameras[1];

        if( m_sdks.back().stampNs <= next.upperNs )
        {
            return;
        }

        while( !m_sdks.empty() && m_sdks.front().stampNs < first.lowerNs )
        {
            m_sdks.pop_front();
        }

        size_t  candidates  = 0;
        int64_t candidateNs = 0;
        bool    ambiguous   = first.upperNs >= next.lowerNs;

        for( const auto &event : m_sdks )
        {
            if( event.stampNs > next.upperNs )
            {
                break;
            }

            if( event.stampNs <= first.upperNs || event.stampNs >= next.lowerNs )
            {
                ambiguous = true;
            }
            else
            {
                ++candidates;
                candidateNs = event.stampNs;
            }
        }

        if( m_sdkCoverageNs >= first.lowerNs )
        {
            ++m_summary.quality.unmatched;
        }
        else if( ambiguous || candidates > 1 )
        {
            ++m_summary.quality.ambiguous;
        }
        else if( candidates == 0 )
        {
            ++m_summary.quality.unmatched;
        }
        else
        {
            const auto boundNs = std::min( m_maxLatencyNs, next.stampNs - first.stampNs );

            if( candidateNs - first.lowerNs >= boundNs )
            {
                ++m_summary.quality.invalid;
            }
            else
            {
                retain( m_latencies, { candidateNs - first.stampNs, first.stampNs, candidateNs } );
                ++m_summary.quality.accepted;
                m_summary.newestNs = candidateNs;
            }
        }

        while( !m_sdks.empty() && m_sdks.front().stampNs < next.lowerNs )
        {
            m_sdks.pop_front();
        }

        m_cameras.pop_front();
    }
}

inline timing::Stats Monitor::statistics( std::deque<Sample> &population, int64_t cutoffNs )
{
    while( !population.empty() && population.front().lastNs <= cutoffNs )
    {
        population.pop_front();
    }

    timing::Accumulator accumulator;

    for( const auto &sample : population )
    {
        accumulator.add( sample.durationNs * 1e-9, sample.firstNs, sample.lastNs );
    }

    return accumulator.stats();
}

inline void Monitor::update( int64_t nowNs )
{
    if( nowNs <= 0 || ( m_nowNs && nowNs < m_nowNs ) )
    {
        ++m_summary.quality.invalid;
        reset();
        return;
    }

    m_nowNs             = nowNs;
    const auto cutoffNs = nowNs - m_windowNs;

    while( !m_cameras.empty() && m_cameras.front().stampNs <= cutoffNs )
    {
        m_cameras.pop_front();
        ++m_summary.quality.unmatched;
    }

    while( !m_sdks.empty() && m_sdks.front().stampNs <= cutoffNs )
    {
        m_sdks.pop_front();
    }

    m_summary.cameraIntervals = statistics( m_cameraIntervals, cutoffNs );
    m_summary.sdkIntervals    = statistics( m_sdkIntervals, cutoffNs );
    m_summary.latency         = statistics( m_latencies, cutoffNs );
    m_summary.age             = m_summary.newestNs ? ( nowNs - m_summary.newestNs ) * 1e-9 : -1;
}

inline const Summary &Monitor::summary() const
{
    return m_summary;
}

} // namespace loopTiming
} // namespace app
} // namespace MagAOX
#endif // loopTimingMonitor_loopTiming_hpp
