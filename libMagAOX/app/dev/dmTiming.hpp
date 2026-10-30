/** \file dmTiming.hpp
 * \brief Bounded, ownership-safe DM timing handoff and consumer-side statistics.
 */
#ifndef dev_dmTiming_hpp
#define dev_dmTiming_hpp

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <ctime>
#include <deque>
#include <limits>
#include <vector>

#include <mx/math/floatUtils.hpp>

#include "../../common/dmTiming.hpp"
#include "../../common/spscQueue.hpp"

namespace MagAOX
{
namespace app
{
namespace dev
{

/// Capture fixed-size timing events without blocking the command thread.
/** Configure only before the producer starts or after it has stopped. Producer
 * capture methods and update/summary must each have one owning thread. In detached
 * mode a separate finalizer exclusively owns submitEvent; the capture owner must
 * detach every event before beginning another event or changing sessions. Never
 * mix legacy finishEvent calls with a concurrent detached submission producer. The consumer
 * never reads a producer-owned slot and the producer never overwrites an unread
 * slot. Consumer-side window allocation and statistics are outside the RT path.
 */
class dmTiming
{
  protected:
    MagAOX::spscQueue<MagAOX::dmTiming::Event> m_queue; ///< Ownership-safe fixed-event handoff.

    std::atomic<uint64_t> m_session{ 1 }; ///< Producer epoch, also visible when a reset produces no event.

    uint64_t m_eventSequence{ 0 }; ///< Producer-only event sequence, including full-queue losses.

    MagAOX::dmTiming::Event m_event; ///< Producer-owned event under construction.

    bool m_active{ false }; ///< Producer has an unfinished receipt or synthetic callback.

    bool m_enabled{ false }; ///< Immutable while producer/consumer threads are running.

    int64_t m_windowNs{ 0 }; ///< Configured consumer window duration.

    double m_excursionSec{ 0 }; ///< Immutable positive ordered-interval threshold; zero disables that analysis.

    std::vector<int64_t> m_intervalScratch; ///< Consumer-only median workspace, reserved before collection starts.

    std::deque<MagAOX::dmTiming::Event> m_window; ///< Consumer-owned retained events.

    MagAOX::dmTiming::Summary m_summary; ///< Consumer-owned snapshot; forced recording reads this cache.

    MagAOX::dmTiming::Counters m_total; ///< Consumer-owned cumulative quality counts.

    MagAOX::dmTiming::Counters m_previous; ///< Counts at the preceding summary.

    uint64_t m_counterSession{ 0 }; ///< Epoch of the previous observed input counter.

    uint64_t m_counterSequence{ 0 }; ///< Event sequence of the previous counter observation.

    uint64_t m_counter{ 0 }; ///< Previous observed input counter.

    bool m_counterValid{ false }; ///< Previous counter is suitable for quality comparison.

    bool m_everEvent{ false }; ///< Current epoch previously supplied an event, for stale reporting.

    /// Count one consumed event independently of its retention in the time window.
    void countEvent( const MagAOX::dmTiming::Event &event /**< [in] Completed command-thread event. */ );

    /// Recompute independent metric populations from consumer-owned events.
    void calculate( int64_t nowNs /**< [in] Monotonic window evaluation time. */ );

    /// Require two real, contiguous publication observations within the same local retention window.
    static bool orderedPublication( const MagAOX::dmTiming::Event &previous /**< [in] Earlier observation. */,
                                    const MagAOX::dmTiming::Event &event /**< [in] Following observation. */,
                                    int64_t                        cutoff /**< [in] Oldest accepted local receipt. */,
                                    int64_t                        nowNs /**< [in] Latest accepted local receipt. */ );

    /// Recompute publication medians and directed adjacent-interval populations outside the producer thread.
    void calculateIntervalPairs( int64_t cutoff /**< [in] Oldest accepted local receipt. */,
                                 int64_t nowNs /**< [in] Latest accepted local receipt. */ );

  public:
    /// Read monotonic nanoseconds without changing errno; return -1 on clock failure.
    static int64_t monotonicNowNs();

    /// Allocate handoff storage and reset all state while producer and consumer are stopped.
    void
    configure( double windowSec /**< [in] Positive elapsed-time window; zero disables. */,
               size_t capacity /**< [in] Maximum queued and retained events; zero disables. */,
               double excursionSec = 0 /**< [in] Positive publication excursion threshold; zero disables pairs. */ );

    /// Return whether collection was configured with nonzero storage and duration.
    bool enabled() const;

    /// Return whether the producer owns an unfinished event.
    bool active() const;

    /// Start a real receipt or synthetic callback, timestamping a real receipt immediately.
    void beginEvent( bool hasReceipt /**< [in] Successful semaphore wait preceded this event. */ );

    /// Start an event with a supplied receipt timestamp, primarily for deterministic validation.
    void beginEventAt( bool    hasReceipt /**< [in] Whether the event has a real receipt. */,
                       int64_t receiptNs /**< [in] Monotonic receipt timestamp, or -1 on capture failure. */ );

    /// Record one endpoint with an errno-preserving clock read.
    void mark( MagAOX::dmTiming::Marker marker /**< [in] Endpoint to record. */ );

    /// Record one endpoint supplied in monotonic integer nanoseconds.
    void markAt( MagAOX::dmTiming::Marker marker /**< [in] Endpoint to record. */,
                 int64_t                  ns /**< [in] Monotonic timestamp, or -1 on clock failure. */ );

    /// Validate and record one endpoint in caller-owned detached storage.
    static void markEvent( MagAOX::dmTiming::Event &event /**< [in,out] Exclusively owned event. */,
                           MagAOX::dmTiming::Marker marker /**< [in] Endpoint to record. */,
                           int64_t                  ns /**< [in] Monotonic timestamp, or -1 on clock failure. */ );

    /// Import one immutable worker result while allowing SDK execution to overlap monitor staging.
    void importSdk( int64_t                   startNs /**< [in] Actual worker SDK entry timestamp. */,
                    int64_t                   endNs /**< [in] Actual worker SDK return timestamp. */,
                    MagAOX::dmTiming::Outcome outcome /**< [in] Actual submission outcome. */,
                    int64_t dispatchNs  = -1 /**< [in] Monitor request-publication timestamp, or unavailable. */,
                    int64_t waitStartNs = -1 /**< [in] Monitor SDK-result wait start, or unavailable. */,
                    int64_t waitEndNs   = -1 /**< [in] Monitor SDK-result wait return, or unavailable. */,
                    int64_t readyNs = -1 /**< [in] Worker ready for selection before idle wait, or unavailable. */ );

    /// Import SDK endpoints into caller-owned storage using the same causal checks as importSdk.
    static void
    importSdkEvent( MagAOX::dmTiming::Event  &event /**< [in,out] Exclusively owned event. */,
                    int64_t                   startNs /**< [in] Actual SDK entry timestamp. */,
                    int64_t                   endNs /**< [in] Actual SDK return timestamp. */,
                    MagAOX::dmTiming::Outcome outcome /**< [in] Actual backend submission outcome. */,
                    int64_t dispatchNs  = -1 /**< [in] Request-publication timestamp, or unavailable. */,
                    int64_t waitStartNs = -1 /**< [in] Result wait start, or unavailable. */,
                    int64_t waitEndNs   = -1 /**< [in] Result wait return, or unavailable. */,
                    int64_t readyNs     = -1 /**< [in] Worker ready before selection/idle wait, or unavailable. */ );

    /// Attach an observed input counter, without asserting verified frame identity.
    void inputCounter( uint64_t counter /**< [in] Stable counter observation supplied by the monitor. */ );

    /// Attach best-effort external writetime with the observed counter, without asserting atomic frame identity.
    void inputPublication(
        int64_t publicationNs /**< [in] Positive external writetime in nanoseconds; otherwise unavailable. */ );

    /// Set callback outcome; success qualifies submission and later failures preserve it.
    void outcome( MagAOX::dmTiming::Outcome outcome /**< [in] Qualified backend or callback result. */ );

    /// Set the result of an exclusively owned event while retaining prior successful submission evidence.
    static void outcomeEvent( MagAOX::dmTiming::Event  &event /**< [in,out] Exclusively owned event. */,
                              MagAOX::dmTiming::Outcome outcome /**< [in] Qualified callback result. */ );

    /// Return the currently qualified callback outcome.
    MagAOX::dmTiming::Outcome outcome() const;

    /// Explicitly qualify or revoke submission status independently of callback outcome.
    void submitted( bool success = true /**< [in] Actual SDK/transport success was established. */ );

    /// Transfer the active event by value without queue publication; return an empty event when inactive.
    MagAOX::dmTiming::Event detachEvent();

    /// Publish an immutable event from the one handoff producer, dropping without waiting when full.
    /** The capture and submission owners must transfer event ownership externally.
     * This method never accesses the capture owner's current event or sequence.
     */
    void submitEvent( const MagAOX::dmTiming::Event &event /**< [in] Finished detached event. */ );

    /// Publish one finished event, or drop it without waiting when the queue is full.
    void finishEvent();

    /// Finish a pending receipt as rejected and begin a new attachment epoch.
    void newSession();

    /// Drain at most the captured queue population and update the trailing-window summary.
    void update( int64_t nowNs = monotonicNowNs() /**< [in] Monotonic evaluation time. */ );

    /// Return the coherent consumer-owned cache, without producer storage access.
    const MagAOX::dmTiming::Summary &summary() const;
};

inline int64_t dmTiming::monotonicNowNs()
{
    int      savedErrno = errno;
    timespec ts{};
    int      rv = clock_gettime( CLOCK_MONOTONIC, &ts );
    errno       = savedErrno;

    return rv == 0 ? static_cast<int64_t>( ts.tv_sec ) * 1000000000LL + ts.tv_nsec : -1;
}

inline void dmTiming::configure( double windowSec, size_t capacity, double excursionSec )
{
    static_assert( std::atomic<uint64_t>::is_always_lock_free, "DM timing handoff requires lock-free positions" );

    m_enabled      = mx::math::isFinite( windowSec ) && windowSec > 0 && windowSec < 9e9 && capacity > 0;
    m_windowNs     = m_enabled ? static_cast<int64_t>( windowSec * 1e9 ) : 0;
    m_enabled      = m_enabled && m_windowNs > 0;
    m_excursionSec = m_enabled && mx::math::isFinite( excursionSec ) && excursionSec > 0 ? excursionSec : 0;

    m_queue.configure( m_enabled ? capacity : 0 );
    m_intervalScratch.clear();

    if( m_excursionSec > 0 )
    {
        m_intervalScratch.reserve( capacity );
    }

    m_session.store( 1, std::memory_order_relaxed );

    m_eventSequence = 0;
    m_event         = {};
    m_active        = false;

    m_window.clear();
    m_summary           = {};
    m_summary.windowSec = m_enabled ? windowSec : 0;
    m_summary.capacity  = m_queue.capacity();
    m_summary.enabled   = m_enabled;
    m_summary.session   = 1;
    m_summary.state = m_enabled ? MagAOX::dmTiming::Availability::warmingUp : MagAOX::dmTiming::Availability::disabled;
    m_summary.intervalPairs.enabled   = m_excursionSec > 0;
    m_summary.intervalPairs.threshold = m_excursionSec;

    m_total        = {};
    m_previous     = {};
    m_counterValid = false;
    m_everEvent    = false;
}

inline bool dmTiming::enabled() const
{
    return m_enabled;
}

inline bool dmTiming::active() const
{
    return m_active;
}

inline void dmTiming::beginEvent( bool hasReceipt )
{
    if( !m_enabled )
    {
        return;
    }

    beginEventAt( hasReceipt, hasReceipt ? monotonicNowNs() : -1 );
}

inline void dmTiming::beginEventAt( bool hasReceipt, int64_t receiptNs )
{
    if( !m_enabled )
    {
        return;
    }

    if( m_active )
    {
        outcome( MagAOX::dmTiming::Outcome::rejected );
        finishEvent();
    }

    m_event           = {};
    m_event.sequence  = ++m_eventSequence;
    m_event.session   = m_session.load( std::memory_order_relaxed );
    m_event.synthetic = !hasReceipt;
    m_active          = true;

    if( hasReceipt )
    {
        markAt( MagAOX::dmTiming::Marker::receipt, receiptNs );
    }
}

inline void dmTiming::mark( MagAOX::dmTiming::Marker marker )
{
    if( m_active )
    {
        markAt( marker, monotonicNowNs() );
    }
}

inline void dmTiming::markAt( MagAOX::dmTiming::Marker marker, int64_t ns )
{
    if( m_active )
    {
        markEvent( m_event, marker, ns );
    }
}

inline void dmTiming::markEvent( MagAOX::dmTiming::Event &event, MagAOX::dmTiming::Marker marker, int64_t ns )
{
    using namespace MagAOX::dmTiming;

    if( markerIndex( marker ) >= markerIndex( Marker::count ) )
    {
        return;
    }

    uint32_t bit     = 1u << markerIndex( marker );
    bool     ordered = ns >= 0;

    for( size_t i = 0; ordered && i < event.markers.size(); ++i )
    {
        if( ( event.valid & ( 1u << i ) ) && event.markers[i] > ns )
        {
            ordered = false;
        }
    }

    if( !ordered )
    {
        event.valid &= ~bit;
        ++event.invalidMarkers;

        return;
    }

    event.markers[markerIndex( marker )] = ns;
    event.valid |= bit;
}

inline void dmTiming::importSdk( int64_t                   startNs,
                                 int64_t                   endNs,
                                 MagAOX::dmTiming::Outcome result,
                                 int64_t                   dispatchNs,
                                 int64_t                   waitStartNs,
                                 int64_t                   waitEndNs,
                                 int64_t                   readyNs )
{
    if( m_active )
    {
        importSdkEvent( m_event, startNs, endNs, result, dispatchNs, waitStartNs, waitEndNs, readyNs );
    }
}

inline void dmTiming::importSdkEvent( MagAOX::dmTiming::Event  &event,
                                      int64_t                   startNs,
                                      int64_t                   endNs,
                                      MagAOX::dmTiming::Outcome result,
                                      int64_t                   dispatchNs,
                                      int64_t                   waitStartNs,
                                      int64_t                   waitEndNs,
                                      int64_t                   readyNs )
{
    using namespace MagAOX::dmTiming;

    const auto endpoint = [&event]( Marker marker )
    {
        const auto index = markerIndex( marker );
        return ( event.valid & ( 1u << index ) ) ? event.markers[index] : int64_t( -1 );
    };
    const auto store = [&event]( Marker marker, int64_t ns )
    {
        const auto index     = markerIndex( marker );
        event.markers[index] = ns;
        event.valid |= 1u << index;
    };
    const auto reject = [&event]( Marker first, Marker last )
    {
        event.valid &= ~( ( 1u << markerIndex( first ) ) | ( 1u << markerIndex( last ) ) );
        ++event.invalidMarkers;
    };
    const auto finish = endpoint( Marker::finish );
    const auto preparation =
        std::max( { endpoint( Marker::receipt ), endpoint( Marker::entry ), endpoint( Marker::actuatorEnd ) } );

    if( dispatchNs >= 0 )
    {
        if( dispatchNs < preparation || ( startNs >= 0 && dispatchNs > startNs ) ||
            ( finish >= 0 && dispatchNs > finish ) )
        {
            reject( Marker::dispatch, Marker::dispatch );
        }
        else
        {
            store( Marker::dispatch, dispatchNs );
        }
    }

    const auto dispatch = endpoint( Marker::dispatch );
    const bool sdkValid = startNs >= 0 && endNs >= startNs && startNs >= preparation &&
                          ( dispatch < 0 || startNs >= dispatch ) && ( finish < 0 || endNs <= finish );

    if( sdkValid )
    {
        store( Marker::sdkStart, startNs );
        store( Marker::sdkEnd, endNs );
    }
    else
    {
        reject( Marker::sdkStart, Marker::sdkEnd );
    }

    if( waitStartNs >= 0 || waitEndNs >= 0 )
    {
        const auto release = std::max( { endpoint( Marker::entry ),
                                         endpoint( Marker::dispatch ),
                                         endpoint( Marker::readRelease ),
                                         endpoint( Marker::stagingEnd ) } );

        if( waitStartNs < 0 || waitEndNs < 0 || waitStartNs < release || waitEndNs < waitStartNs ||
            ( sdkValid && waitEndNs < endNs ) || ( finish >= 0 && waitEndNs > finish ) )
        {
            reject( Marker::waitStart, Marker::waitEnd );
        }
        else
        {
            store( Marker::waitStart, waitStartNs );
            store( Marker::waitEnd, waitEndNs );
        }
    }

    if( readyNs >= 0 )
    {
        if( !sdkValid || readyNs < endNs || ( finish >= 0 && readyNs > finish ) )
        {
            reject( Marker::sdkReady, Marker::sdkReady );
        }
        else
        {
            store( Marker::sdkReady, readyNs );
        }
    }

    if( result == Outcome::success )
    {
        event.submitted = true;
    }

    if( event.outcome != Outcome::failed && event.outcome != Outcome::partial && event.outcome != Outcome::rejected )
    {
        outcomeEvent( event, result );
    }
}

inline void dmTiming::inputCounter( uint64_t counter )
{
    if( !m_active )
    {
        return;
    }

    m_event.counter      = counter;
    m_event.counterValid = true;
}

inline void dmTiming::inputPublication( int64_t publicationNs )
{
    if( !m_active )
    {
        return;
    }

    m_event.publicationNs = publicationNs > 0 ? publicationNs : -1;
}

inline void dmTiming::outcome( MagAOX::dmTiming::Outcome outcome )
{
    if( !m_active )
    {
        return;
    }

    outcomeEvent( m_event, outcome );
}

inline void dmTiming::outcomeEvent( MagAOX::dmTiming::Event &event, MagAOX::dmTiming::Outcome outcome )
{
    event.outcome = outcome;

    if( outcome == MagAOX::dmTiming::Outcome::success )
    {
        event.submitted = true;
    }
}

inline MagAOX::dmTiming::Outcome dmTiming::outcome() const
{
    return m_event.outcome;
}

inline void dmTiming::submitted( bool success )
{
    if( m_active )
    {
        m_event.submitted = success;
    }
}

inline MagAOX::dmTiming::Event dmTiming::detachEvent()
{
    if( !m_active )
    {
        return {};
    }

    m_active = false;
    return m_event;
}

inline void dmTiming::submitEvent( const MagAOX::dmTiming::Event &event )
{
    if( m_enabled && event.session != 0 && event.sequence != 0 )
    {
        m_queue.push( event );
    }
}

inline void dmTiming::finishEvent()
{
    if( m_active )
    {
        submitEvent( detachEvent() );
    }
}

inline void dmTiming::newSession()
{
    if( !m_enabled )
    {
        return;
    }

    if( m_active )
    {
        outcome( MagAOX::dmTiming::Outcome::rejected );
        finishEvent();
    }

    m_session.fetch_add( 1, std::memory_order_release );
}

inline void dmTiming::countEvent( const MagAOX::dmTiming::Event &event )
{
    using namespace MagAOX::dmTiming;

    ++m_total.events;
    m_total.receipts += ( event.valid & ( 1u << markerIndex( Marker::receipt ) ) ) != 0;
    m_total.submitted += event.submitted;
    m_total.synthetic += event.synthetic;
    m_total.failed += event.outcome == Outcome::failed;
    m_total.partial += event.outcome == Outcome::partial;
    m_total.noSend += event.outcome == Outcome::noSend;
    m_total.rejected += event.outcome == Outcome::rejected;
    m_total.invalid += event.invalidMarkers;

    if( event.counterValid && m_counterValid && event.session == m_counterSession &&
        event.sequence == m_counterSequence + 1 )
    {
        m_total.duplicates += event.counter == m_counter;
        m_total.counterGaps += event.counter != m_counter && event.counter != m_counter + 1;
    }

    m_counter         = event.counter;
    m_counterValid    = event.counterValid;
    m_counterSession  = event.session;
    m_counterSequence = event.sequence;
}

inline void dmTiming::update( int64_t nowNs )
{
    using namespace MagAOX::dmTiming;

    ++m_summary.sequence;
    m_summary.nowNs = nowNs;

    if( !m_enabled )
    {
        return;
    }

    uint64_t session = m_session.load( std::memory_order_acquire );

    if( session != m_summary.session )
    {
        m_window.clear();
        m_summary.session = session;
        m_everEvent       = false;
    }

    const size_t population = m_queue.queued();

    for( size_t index = 0; index < population; ++index )
    {
        Event event;

        if( !m_queue.pop( event ) )
        {
            break;
        }

        countEvent( event );

        if( event.session > m_summary.session )
        {
            m_window.clear();
            m_summary.session = event.session;
            m_everEvent       = false;
        }

        if( event.session == m_summary.session )
        {
            m_window.push_back( event );

            if( m_window.size() > m_queue.capacity() )
            {
                m_window.pop_front();
            }

            m_everEvent = true;
        }
    }

    // A reset may race the bounded drain without yet publishing a new event.
    session = m_session.load( std::memory_order_acquire );

    if( session > m_summary.session )
    {
        m_window.clear();
        m_summary.session = session;
        m_everEvent       = false;
    }

    m_summary.queued = m_queue.queued();
    m_total.dropped  = m_queue.dropped();
    m_summary.total  = m_total;

    constexpr uint64_t Counters::*fields[] = { &Counters::events,
                                               &Counters::receipts,
                                               &Counters::submitted,
                                               &Counters::failed,
                                               &Counters::partial,
                                               &Counters::noSend,
                                               &Counters::synthetic,
                                               &Counters::rejected,
                                               &Counters::duplicates,
                                               &Counters::counterGaps,
                                               &Counters::invalid,
                                               &Counters::dropped };

    for( auto field : fields )
    {
        m_summary.delta.*field = m_total.*field - m_previous.*field;
    }

    m_previous = m_total;

    calculate( nowNs );
}

inline void dmTiming::calculate( int64_t nowNs )
{
    using namespace MagAOX::dmTiming;

    // Local arrays and Welford accumulation require no per-metric allocation.
    std::array<double, metricIndex( Metric::count )>  sumSquares{};
    std::array<int64_t, metricIndex( Metric::count )> earliest;
    std::array<int64_t, metricIndex( Metric::count )> latest;
    earliest.fill( std::numeric_limits<int64_t>::max() );
    latest.fill( -1 );

    m_summary.metrics                 = {};
    m_summary.newestNs                = -1;
    m_summary.age                     = -1;
    m_summary.span                    = 0;
    m_summary.intervalPairs           = {};
    m_summary.intervalPairs.enabled   = m_excursionSec > 0;
    m_summary.intervalPairs.threshold = m_excursionSec;

    if( nowNs < 0 )
    {
        m_summary.fgValid = false;
        m_summary.state   = Availability::unavailable;

        return;
    }

    int64_t oldestNs = std::numeric_limits<int64_t>::max();
    int64_t cutoff   = nowNs >= m_windowNs ? nowNs - m_windowNs : 0;

    auto endpoint = []( const Event &event, Marker marker ) -> int64_t
    {
        return event.valid & ( 1u << markerIndex( marker ) ) ? event.markers[markerIndex( marker )] : -1;
    };

    auto newest = [&]( const Event &event )
    {
        int64_t value = -1;

        for( size_t i = 0; i < event.markers.size(); ++i )
        {
            if( event.valid & ( 1u << i ) )
            {
                value = std::max( value, event.markers[i] );
            }
        }

        return value;
    };

    while( !m_window.empty() && newest( m_window.front() ) < cutoff )
    {
        m_window.pop_front();
    }

    calculateIntervalPairs( cutoff, nowNs );

    auto addValue = [&]( Metric metric, int64_t durationNs, int64_t a, int64_t b )
    {
        if( durationNs < 0 || a < cutoff || b < a || b > nowNs )
        {
            return;
        }

        size_t index = metricIndex( metric );
        Stats &stats = m_summary.metrics[index];
        double value = durationNs * 1e-9;

        ++stats.count;
        double difference = value - stats.mean;
        stats.mean += difference / stats.count;
        sumSquares[index] += difference * ( value - stats.mean );

        stats.min       = stats.count == 1 ? value : std::min( stats.min, value );
        stats.max       = stats.count == 1 ? value : std::max( stats.max, value );
        earliest[index] = std::min( earliest[index], a );
        latest[index]   = std::max( latest[index], b );
    };

    auto add = [&]( Metric metric, int64_t a, int64_t b )
    {
        if( a >= 0 && b >= a )
        {
            addValue( metric, b - a, a, b );
        }
    };

    int64_t      previousReceipt = -1, previousSend = -1;
    uint64_t     previousSequence = 0;
    const Event *previousEvent    = nullptr;

    for( const Event &event : m_window )
    {
        if( previousSequence && event.sequence != previousSequence + 1 )
        {
            previousReceipt = previousSend = -1;
        }

        previousSequence = event.sequence;

        for( size_t i = 0; i < event.markers.size(); ++i )
        {
            int64_t value = event.markers[i];

            if( ( event.valid & ( 1u << i ) ) && value >= cutoff && value <= nowNs )
            {
                oldestNs           = std::min( oldestNs, value );
                m_summary.newestNs = std::max( m_summary.newestNs, value );
            }
        }

        int64_t receipt = endpoint( event, Marker::receipt );
        int64_t send    = endpoint( event, Marker::sdkEnd );

        add( Metric::receiptInterval, previousReceipt, receipt );
        previousReceipt = receipt;

        if( event.submitted )
        {
            add( Metric::sendInterval, previousSend, send );
            add( Metric::receiptSend, receipt, send );
            previousSend = send;
        }
        else if( event.outcome == Outcome::unknown )
        {
            previousSend = -1;
        }

        if( previousEvent && event.sequence == previousEvent->sequence + 1 && event.session == previousEvent->session &&
            event.counterValid && previousEvent->counterValid && event.counter > previousEvent->counter &&
            event.counter == previousEvent->counter + 1 && previousEvent->publicationNs > 0 &&
            event.publicationNs > previousEvent->publicationNs )
        {
            int64_t previousObservation = endpoint( *previousEvent, Marker::receipt );
            int64_t observation         = receipt;

            if( previousObservation < 0 )
            {
                previousObservation = endpoint( *previousEvent, Marker::entry );
            }

            if( observation < 0 )
            {
                observation = endpoint( event, Marker::entry );
            }

            addValue( Metric::publicationInterval,
                      event.publicationNs - previousEvent->publicationNs,
                      previousObservation,
                      observation );
        }

        previousEvent = &event;

        constexpr Metric metrics[] = { Metric::dispatch,
                                       Metric::actuator,
                                       Metric::sdk,
                                       Metric::saturation,
                                       Metric::delta,
                                       Metric::post,
                                       Metric::throughDelta,
                                       Metric::callback,
                                       Metric::receiptCallback,
                                       Metric::sdkDispatch,
                                       Metric::readerRelease,
                                       Metric::staging,
                                       Metric::publication,
                                       Metric::sdkWait,
                                       Metric::sdkTail,
                                       Metric::sdkService };
        constexpr Marker starts[]  = { Marker::receipt,
                                       Marker::actuatorStart,
                                       Marker::sdkStart,
                                       Marker::satStart,
                                       Marker::deltaStart,
                                       Marker::postStart,
                                       Marker::entry,
                                       Marker::entry,
                                       Marker::receipt,
                                       Marker::dispatch,
                                       Marker::entry,
                                       Marker::stagingStart,
                                       Marker::publicationStart,
                                       Marker::waitStart,
                                       Marker::sdkEnd,
                                       Marker::sdkStart };
        constexpr Marker ends[]    = { Marker::entry,
                                       Marker::actuatorEnd,
                                       Marker::sdkEnd,
                                       Marker::satEnd,
                                       Marker::deltaEnd,
                                       Marker::postEnd,
                                       Marker::deltaEnd,
                                       Marker::finish,
                                       Marker::finish,
                                       Marker::sdkStart,
                                       Marker::readRelease,
                                       Marker::stagingEnd,
                                       Marker::publicationEnd,
                                       Marker::waitEnd,
                                       Marker::sdkReady,
                                       Marker::sdkReady };

        for( size_t i = 0; i < sizeof( metrics ) / sizeof( metrics[0] ); ++i )
        {
            add( metrics[i], endpoint( event, starts[i] ), endpoint( event, ends[i] ) );
        }
    }

    for( size_t i = 0; i < m_summary.metrics.size(); ++i )
    {
        Stats &stats = m_summary.metrics[i];
        stats.valid  = stats.count > 1;

        if( stats.valid )
        {
            stats.stddev = std::sqrt( std::max( 0.0, sumSquares[i] / ( stats.count - 1 ) ) );
        }

        if( stats.count )
        {
            stats.span = ( latest[i] - earliest[i] ) * 1e-9;
        }
    }

    if( m_summary.newestNs >= 0 )
    {
        m_summary.age  = ( nowNs - m_summary.newestNs ) * 1e-9;
        m_summary.span = ( m_summary.newestNs - oldestNs ) * 1e-9;
    }

    m_summary.fgValid = m_summary.metrics[metricIndex( Metric::receiptInterval )].valid &&
                        m_summary.metrics[metricIndex( Metric::sendInterval )].valid &&
                        m_summary.metrics[metricIndex( Metric::receiptSend )].valid;

    if( m_summary.fgValid )
    {
        m_summary.state = Availability::valid;
    }
    else if( m_window.empty() && m_everEvent )
    {
        m_summary.state = Availability::stale;
    }
    else if( m_summary.metrics[metricIndex( Metric::receiptInterval )].valid &&
             !m_summary.metrics[metricIndex( Metric::receiptSend )].count )
    {
        m_summary.state = Availability::unavailable;
    }
    else
    {
        m_summary.state = Availability::warmingUp;
    }
}

inline bool dmTiming::orderedPublication( const MagAOX::dmTiming::Event &previous,
                                          const MagAOX::dmTiming::Event &event,
                                          int64_t                        cutoff,
                                          int64_t                        nowNs )
{
    using namespace MagAOX::dmTiming;
    const uint32_t receipt = 1u << markerIndex( Marker::receipt );

    return !previous.synthetic && !event.synthetic && previous.outcome != Outcome::rejected &&
           event.outcome != Outcome::rejected && !previous.invalidMarkers && !event.invalidMarkers &&
           ( previous.valid & receipt ) && ( event.valid & receipt ) && previous.counterValid && event.counterValid &&
           event.session == previous.session && event.sequence > previous.sequence &&
           event.sequence == previous.sequence + 1 && event.counter > previous.counter &&
           event.counter == previous.counter + 1 && previous.publicationNs > 0 &&
           event.publicationNs > previous.publicationNs && previous.markers[markerIndex( Marker::receipt )] >= cutoff &&
           event.markers[markerIndex( Marker::receipt )] >= previous.markers[markerIndex( Marker::receipt )] &&
           event.markers[markerIndex( Marker::receipt )] <= nowNs;
}

inline void dmTiming::calculateIntervalPairs( int64_t cutoff, int64_t nowNs )
{
    using namespace MagAOX::dmTiming;
    auto &summary = m_summary.intervalPairs;

    if( !summary.enabled )
    {
        return;
    }

    m_intervalScratch.clear();
    const Event *previous = nullptr;

    for( const auto &event : m_window )
    {
        if( previous && orderedPublication( *previous, event, cutoff, nowNs ) )
        {
            m_intervalScratch.push_back( event.publicationNs - previous->publicationNs );
        }

        previous = &event;
    }

    summary.intervals = m_intervalScratch.size();

    if( !summary.intervals )
    {
        return;
    }

    const auto middle = m_intervalScratch.begin() + m_intervalScratch.size() / 2;
    std::nth_element( m_intervalScratch.begin(), middle, m_intervalScratch.end() );
    long double medianNs = *middle;

    if( m_intervalScratch.size() % 2 == 0 )
    {
        const int64_t lower = *std::max_element( m_intervalScratch.begin(), middle );
        medianNs            = lower + ( medianNs - lower ) / 2;
    }

    summary.median                     = static_cast<double>( medianNs * 1e-9L );
    const long double      thresholdNs = m_excursionSec * 1e9;
    std::array<Stats *, 4> populations{
        &summary.afterLate, &summary.afterEarly, &summary.lateEarlySum, &summary.earlyLateSum };
    std::array<double, 4>  sumSquares{};
    std::array<int64_t, 4> earliest, latest;
    earliest.fill( std::numeric_limits<int64_t>::max() );
    latest.fill( -1 );

    auto add = [&]( size_t index, int64_t duration, int64_t a, int64_t b )
    {
        auto  &stats = *populations[index];
        double value = duration * 1e-9;
        ++stats.count;
        double difference = value - stats.mean;
        stats.mean += difference / stats.count;
        sumSquares[index] += difference * ( value - stats.mean );
        stats.min       = stats.count == 1 ? value : std::min( stats.min, value );
        stats.max       = stats.count == 1 ? value : std::max( stats.max, value );
        earliest[index] = std::min( earliest[index], a );
        latest[index]   = std::max( latest[index], b );
    };

    previous                      = nullptr;
    const Event *first            = nullptr;
    int64_t      previousInterval = -1;

    for( const auto &event : m_window )
    {
        const bool    qualified = previous && orderedPublication( *previous, event, cutoff, nowNs );
        const int64_t interval  = qualified ? event.publicationNs - previous->publicationNs : -1;

        if( qualified && previousInterval > 0 )
        {
            ++summary.pairs;
            const long double firstResidual = previousInterval - medianNs;
            const long double nextResidual  = interval - medianNs;
            const int64_t     a             = previous->markers[markerIndex( Marker::receipt )];
            const int64_t     b             = event.markers[markerIndex( Marker::receipt )];

            if( firstResidual > thresholdNs )
            {
                add( 0, interval, a, b );

                if( nextResidual < -thresholdNs )
                {
                    add( 2,
                         event.publicationNs - first->publicationNs,
                         first->markers[markerIndex( Marker::receipt )],
                         b );
                }
            }
            else if( firstResidual < -thresholdNs )
            {
                add( 1, interval, a, b );

                if( nextResidual > thresholdNs )
                {
                    add( 3,
                         event.publicationNs - first->publicationNs,
                         first->markers[markerIndex( Marker::receipt )],
                         b );
                }
            }
        }

        first            = qualified ? previous : nullptr;
        previous         = &event;
        previousInterval = interval;
    }

    for( size_t index = 0; index < populations.size(); ++index )
    {
        auto &stats = *populations[index];
        stats.valid = stats.count > 1;

        if( stats.valid )
        {
            stats.stddev = std::sqrt( std::max( 0.0, sumSquares[index] / ( stats.count - 1 ) ) );
        }

        if( stats.count )
        {
            stats.span = ( latest[index] - earliest[index] ) * 1e-9;
        }
    }
}

inline const MagAOX::dmTiming::Summary &dmTiming::summary() const
{
    return m_summary;
}

} // namespace dev
} // namespace app
} // namespace MagAOX
#endif
