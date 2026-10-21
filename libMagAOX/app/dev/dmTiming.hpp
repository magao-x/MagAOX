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

namespace MagAOX
{
namespace app
{
namespace dev
{

/// Capture fixed-size timing events without blocking the command thread.
/** Configure only before the producer starts or after it has stopped. Producer
 * methods and update/summary must each have one owning thread. The consumer
 * never reads a producer-owned slot and the producer never overwrites an unread
 * slot. Consumer-side window allocation and statistics are outside the RT path.
 */
class dmTiming
{
  protected:
    std::vector<MagAOX::dmTiming::Event> m_queue; ///< Slots owned through release/acquire queue positions.

    std::atomic<uint64_t> m_head{ 0 }; ///< Next producer position, published after writing its slot.

    std::atomic<uint64_t> m_tail{ 0 }; ///< Next consumer position, released after copying its slot.

    std::atomic<uint64_t> m_dropped{ 0 }; ///< Producer drop count, observed without producer payload access.

    std::atomic<uint64_t> m_session{ 1 }; ///< Producer epoch, also visible when a reset produces no event.

    uint64_t m_eventSequence{ 0 }; ///< Producer-only event sequence, including full-queue losses.

    MagAOX::dmTiming::Event m_event; ///< Producer-owned event under construction.

    bool m_active{ false }; ///< Producer has an unfinished receipt or synthetic callback.

    bool m_enabled{ false }; ///< Immutable while producer/consumer threads are running.

    int64_t m_windowNs{ 0 }; ///< Configured consumer window duration.

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

  public:
    /// Read monotonic nanoseconds without changing errno; return -1 on clock failure.
    static int64_t monotonicNowNs();

    /// Allocate handoff storage and reset all state while producer and consumer are stopped.
    void configure( double windowSec /**< [in] Positive elapsed-time window; zero disables. */,
                    size_t capacity /**< [in] Maximum queued and retained events; zero disables. */ );

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

    /// Attach an observed input counter, without asserting verified frame identity.
    void inputCounter( uint64_t counter /**< [in] Stable counter observation supplied by the monitor. */ );

    /// Set callback outcome; success qualifies submission and later failures preserve it.
    void outcome( MagAOX::dmTiming::Outcome outcome /**< [in] Qualified backend or callback result. */ );

    /// Return the currently qualified callback outcome.
    MagAOX::dmTiming::Outcome outcome() const;

    /// Explicitly qualify or revoke submission status independently of callback outcome.
    void submitted( bool success = true /**< [in] Actual SDK/transport success was established. */ );

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

inline void dmTiming::configure( double windowSec, size_t capacity )
{
    static_assert( std::atomic<uint64_t>::is_always_lock_free, "DM timing handoff requires lock-free positions" );
    m_enabled  = mx::math::isFinite( windowSec ) && windowSec > 0 && windowSec < 9e9 && capacity > 0;
    m_windowNs = m_enabled ? static_cast<int64_t>( windowSec * 1e9 ) : 0;
    m_enabled  = m_enabled && m_windowNs > 0;
    m_queue.clear();
    m_queue.resize( m_enabled ? capacity : 0 );
    m_head.store( 0, std::memory_order_relaxed );
    m_tail.store( 0, std::memory_order_relaxed );
    m_dropped.store( 0, std::memory_order_relaxed );
    m_session.store( 1, std::memory_order_relaxed );
    m_eventSequence = 0;
    m_event         = {};
    m_active        = false;
    m_window.clear();
    m_summary           = {};
    m_summary.windowSec = m_enabled ? windowSec : 0;
    m_summary.capacity  = m_queue.size();
    m_summary.enabled   = m_enabled;
    m_summary.session   = 1;
    m_summary.state = m_enabled ? MagAOX::dmTiming::Availability::warmingUp : MagAOX::dmTiming::Availability::disabled;
    m_total         = {};
    m_previous      = {};
    m_counterValid  = false;
    m_everEvent     = false;
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
    using namespace MagAOX::dmTiming;
    if( !m_active || markerIndex( marker ) >= markerIndex( Marker::count ) )
    {
        return;
    }
    uint32_t bit     = 1u << markerIndex( marker );
    bool     ordered = ns >= 0;
    for( size_t i = 0; ordered && i < m_event.markers.size(); ++i )
    {
        if( ( m_event.valid & ( 1u << i ) ) && m_event.markers[i] > ns )
        {
            ordered = false;
        }
    }
    if( !ordered )
    {
        m_event.valid &= ~bit;
        ++m_event.invalidMarkers;
        return;
    }
    m_event.markers[markerIndex( marker )] = ns;
    m_event.valid |= bit;
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

inline void dmTiming::outcome( MagAOX::dmTiming::Outcome outcome )
{
    if( !m_active )
    {
        return;
    }
    m_event.outcome = outcome;
    if( outcome == MagAOX::dmTiming::Outcome::success )
    {
        m_event.submitted = true;
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

inline void dmTiming::finishEvent()
{
    if( !m_active )
    {
        return;
    }
    m_active      = false;
    uint64_t head = m_head.load( std::memory_order_relaxed );
    uint64_t tail = m_tail.load( std::memory_order_acquire );
    if( head - tail == m_queue.size() )
    {
        m_dropped.fetch_add( 1, std::memory_order_relaxed );
        return;
    }
    m_queue[head % m_queue.size()] = m_event;
    m_head.store( head + 1, std::memory_order_release );
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
    uint64_t       tail = m_tail.load( std::memory_order_relaxed );
    const uint64_t head = m_head.load( std::memory_order_acquire );
    while( tail != head )
    {
        Event event = m_queue[tail % m_queue.size()];
        ++tail;
        m_tail.store( tail, std::memory_order_release );
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
            if( m_window.size() > m_queue.size() )
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
    m_summary.queued                       = static_cast<size_t>( m_head.load( std::memory_order_acquire ) - tail );
    m_total.dropped                        = m_dropped.load( std::memory_order_relaxed );
    m_summary.total                        = m_total;
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
    m_summary.metrics  = {};
    m_summary.newestNs = -1;
    m_summary.age      = -1;
    m_summary.span     = 0;
    if( nowNs < 0 )
    {
        m_summary.fgValid = false;
        m_summary.state   = Availability::unavailable;
        return;
    }
    int64_t oldestNs = std::numeric_limits<int64_t>::max();
    int64_t cutoff   = nowNs >= m_windowNs ? nowNs - m_windowNs : 0;
    auto    endpoint = []( const Event &event, Marker marker ) -> int64_t
    { return event.valid & ( 1u << markerIndex( marker ) ) ? event.markers[markerIndex( marker )] : -1; };
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
    auto add = [&]( Metric metric, int64_t a, int64_t b )
    {
        if( a < cutoff || b < a || b > nowNs )
        {
            return;
        }
        size_t index = metricIndex( metric );
        Stats &stats = m_summary.metrics[index];
        double value = ( b - a ) * 1e-9;
        ++stats.count;
        double difference = value - stats.mean;
        stats.mean += difference / stats.count;
        sumSquares[index] += difference * ( value - stats.mean );
        stats.min       = stats.count == 1 ? value : std::min( stats.min, value );
        stats.max       = stats.count == 1 ? value : std::max( stats.max, value );
        earliest[index] = std::min( earliest[index], a );
        latest[index]   = std::max( latest[index], b );
    };
    int64_t  previousReceipt = -1, previousSend = -1;
    uint64_t previousSequence = 0;
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
        constexpr Metric metrics[] = { Metric::dispatch,
                                       Metric::actuator,
                                       Metric::sdk,
                                       Metric::saturation,
                                       Metric::delta,
                                       Metric::post,
                                       Metric::throughDelta,
                                       Metric::callback,
                                       Metric::receiptCallback };
        constexpr Marker starts[]  = { Marker::receipt,
                                       Marker::actuatorStart,
                                       Marker::sdkStart,
                                       Marker::satStart,
                                       Marker::deltaStart,
                                       Marker::postStart,
                                       Marker::entry,
                                       Marker::entry,
                                       Marker::receipt };
        constexpr Marker ends[]    = { Marker::entry,
                                       Marker::actuatorEnd,
                                       Marker::sdkEnd,
                                       Marker::satEnd,
                                       Marker::deltaEnd,
                                       Marker::postEnd,
                                       Marker::deltaEnd,
                                       Marker::finish,
                                       Marker::finish };
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

inline const MagAOX::dmTiming::Summary &dmTiming::summary() const
{
    return m_summary;
}

} // namespace dev
} // namespace app
} // namespace MagAOX
#endif
