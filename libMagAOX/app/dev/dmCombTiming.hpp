/** \file dmCombTiming.hpp
 * \brief Bounded observational collection of combined-stream publication and process-trace timing.
 */
#ifndef dev_dmCombTiming_hpp
#define dev_dmCombTiming_hpp

#include <ImageStreamIO/ImageStruct.h>

#include "../../common/dmCombTiming.hpp"
#include "dmTiming.hpp"

namespace MagAOX
{
namespace app
{
namespace dev
{

/// Observe existing producer metadata without modifying its stream, counters, traces or timing settings.
/** Configure only while both threads are stopped. One monitor thread owns all
 * producer methods; ordinary app logic owns update/summary. Trace association is
 * consistency checked and remains best effort, never atomic frame provenance.
 */
class dmCombTiming
{
  protected:
    /// Only fields used to qualify durations are copied from each shared process-trace row.
    struct Trace
    {
        int64_t  pid{ 0 };     ///< Writer reported by this row.
        uint64_t inode{ 0 };   ///< Trigger inode reported by this row.
        uint64_t counter{ 0 }; ///< Input counter reported by this row.
        timespec start{};      ///< External-clock process-start timestamp.
        timespec update{};     ///< External-clock publication-trace timestamp.
    };

    /// Fixed-size producer handoff, augmented with qualified consumer-side durations once copied.
    struct Event
    {
        uint64_t sequence{ 0 };           ///< Producer sequence including dropped observations.
        uint64_t session{ 0 };            ///< Attachment epoch.
        uint64_t counter{ 0 };            ///< Observed output counter.
        uint64_t expectedPrimary{ 0 };    ///< Expected primary inode at observation.
        int64_t  observed{ -1 };          ///< Local callback observation timestamp.
        int64_t  receipt{ -1 };           ///< Local successful-wait timestamp, or unavailable.
        int64_t  writetime{ -1 };         ///< External-clock output publication timestamp.
        Trace    trace;                   ///< First process-trace row.
        Trace    primary;                 ///< Upstream process-trace row, when available.
        bool     rejected{ false };       ///< Notification was rejected before callback processing.
        bool     metadataValid{ false };  ///< Metadata values passed the repeated-read bracket.
        bool     tracePresent{ false };   ///< A first trace row was available.
        bool     traceStable{ false };    ///< Both copies of the first row agree.
        bool     primaryPresent{ false }; ///< An upstream row was available.
        bool     primaryStable{ false };  ///< Both upstream-row copies agree.
        uint8_t  valid{ 0 };              ///< Consumer-qualified per-event duration bit mask.
        std::array<int64_t, MagAOX::dmCombTiming::index( MagAOX::dmCombTiming::Metric::count )> durations{};
        ///< External-clock differences; never subtracted from local receipt/observation timestamps.
    };

    std::vector<Event> m_queue; ///< Preallocated SPSC slots, never overwritten while unread.

    std::atomic<uint64_t> m_head{ 0 }; ///< Producer position released after writing a complete slot.

    std::atomic<uint64_t> m_tail{ 0 }; ///< Consumer position released after copying a complete slot.

    std::atomic<uint64_t> m_dropped{ 0 }; ///< Producer-side full-queue losses.

    std::atomic<uint64_t> m_session{ 1 }; ///< Attachment epoch, including resets that publish no event.

    uint64_t m_producerSequence{ 0 }; ///< Producer-only observation sequence.

    bool m_enabled{ false }; ///< Immutable while producer and consumer run.

    int64_t m_windowNs{ 0 }; ///< Configured trailing elapsed-time window.

    std::deque<Event> m_window; ///< Consumer-owned retained observation population.

    MagAOX::dmCombTiming::Summary m_summary; ///< Coherent ordinary-thread cache.

    uint64_t m_analysisSession{ 0 }; ///< Epoch of the last consumer analysis.

    uint64_t m_analysisSequence{ 0 }; ///< Sequence of the last analyzed event.

    uint64_t m_counter{ 0 }; ///< Last valid observed output counter.

    bool m_haveCounter{ false }; ///< Last counter can be compared with this observation.

    int64_t m_writer{ 0 }; ///< Writer associated with the process-start baseline.

    int64_t m_start{ -1 }; ///< Last accepted output's valid process-start timestamp.

    bool m_haveStart{ false }; ///< A same-writer process-start baseline exists.

    uint64_t m_primaryInode{ 0 }; ///< Trigger inode of the linked primary baseline.

    uint64_t m_primaryCounter{ 0 }; ///< Last qualified linked primary counter.

    int64_t m_primaryWriter{ 0 }; ///< Upstream writer associated with the linked-primary baseline.

    int64_t m_primaryUpdate{ -1 }; ///< Last valid upstream update; a frozen row cannot establish fresh latency.

    bool m_havePrimary{ false }; ///< A comparable linked-primary counter baseline exists.

    /// Normalize a positive external timestamp, rejecting malformed or overflowing timespec values.
    static int64_t timestamp( const timespec &stamp /**< [in] External-clock timestamp. */ );

    /// Copy only the process-trace fields consumed by these diagnostics.
    static Trace trace( const volatile STREAM_PROC_TRACE &source /**< [in] Shared process-trace row. */ );

    /// Compare all used trace fields without reading struct padding.
    static bool same( const Trace &a /**< [in] First observation. */, const Trace &b /**< [in] Second observation. */ );

    /// Select the local elapsed-time coverage anchor, never an external publication clock.
    static int64_t anchor( const Event &event /**< [in] Completed observation. */ );

    /// Publish a fixed-size event without waiting for queue capacity.
    void enqueue( Event &event /**< [in,out] Producer-owned observation. */ );

    /// Clear consumer freshness/association baselines after an epoch, loss or unavailable observation.
    void resetBaselines();

    /// Count one copied event and qualify fresh, ordered trace durations exactly once.
    void qualify( Event &event /**< [in,out] Consumer-owned observation. */,
                  bool   identity /**< [in] Update displayed identity for the current attachment. */ );

    /// Recompute statistics using consumer-owned events and local-time coverage gates.
    void calculate( int64_t nowNs /**< [in] Monotonic evaluation time. */ );

  public:
    /// Preallocate and reset collection while both owner threads are stopped; either zero disables.
    void configure( double windowSec /**< [in] Trailing elapsed-time window in seconds. */,
                    size_t capacity /**< [in] Queue and retained-observation cap. */ );

    /// Return whether collection is enabled and preallocated.
    bool enabled() const;

    /// Begin a new attachment epoch on the monitor thread.
    void newSession();

    /// Capture best-effort metadata and trace copies without allocating, blocking or reading another clock.
    void observe( const IMAGE &image /**< [in] Attached combined-stream descriptor. */,
                  ino_t        expectedPrimaryInode /**< [in] Current primary inode, or zero when unknown. */,
                  int64_t      observedNs /**< [in] Local callback observation timestamp. */,
                  int64_t receiptNs = -1 /**< [in] Successful-wait timestamp; -1 identifies a synthetic callback. */ );

    /// Retain an actual successful receipt rejected by monitor metadata/state checks.
    void rejected( int64_t receiptNs /**< [in] Successful-wait timestamp. */ );

    /// Drain a captured bounded queue population and refresh ordinary-thread statistics.
    void update( int64_t nowNs = dmTiming::monotonicNowNs() /**< [in] Local monotonic evaluation timestamp. */ );

    /// Return the coherent consumer-owned summary.
    const MagAOX::dmCombTiming::Summary &summary() const;
};

inline int64_t dmCombTiming::timestamp( const timespec &stamp )
{
    if( stamp.tv_sec < 0 || stamp.tv_nsec < 0 || stamp.tv_nsec >= 1000000000 ||
        stamp.tv_sec > ( std::numeric_limits<int64_t>::max() - stamp.tv_nsec ) / 1000000000 )
    {
        return -1;
    }

    int64_t value = static_cast<int64_t>( stamp.tv_sec ) * 1000000000 + stamp.tv_nsec;

    return value > 0 ? value : -1;
}

inline dmCombTiming::Trace dmCombTiming::trace( const volatile STREAM_PROC_TRACE &source )
{
    Trace result;
    result.pid     = source.procwrite_PID;
    result.inode   = source.trigger_inode;
    result.counter = source.cnt0;
    result.start   = { source.ts_procstart.tv_sec, source.ts_procstart.tv_nsec };
    result.update  = { source.ts_streamupdate.tv_sec, source.ts_streamupdate.tv_nsec };

    return result;
}

inline bool dmCombTiming::same( const Trace &a, const Trace &b )
{
    return a.pid == b.pid && a.inode == b.inode && a.counter == b.counter && a.start.tv_sec == b.start.tv_sec &&
           a.start.tv_nsec == b.start.tv_nsec && a.update.tv_sec == b.update.tv_sec &&
           a.update.tv_nsec == b.update.tv_nsec;
}

inline int64_t dmCombTiming::anchor( const Event &event )
{
    return event.receipt >= 0 ? event.receipt : event.observed;
}

inline void dmCombTiming::configure( double windowSec, size_t capacity )
{
    static_assert( std::atomic<uint64_t>::is_always_lock_free, "Combined timing handoff requires lock-free positions" );

    m_enabled  = mx::math::isFinite( windowSec ) && windowSec > 0 && windowSec < 9e9 && capacity > 0;
    m_windowNs = m_enabled ? static_cast<int64_t>( windowSec * 1e9 ) : 0;
    m_enabled  = m_enabled && m_windowNs > 0;
    m_queue.clear();
    m_queue.resize( m_enabled ? capacity : 0 );
    m_head.store( 0, std::memory_order_relaxed );
    m_tail.store( 0, std::memory_order_relaxed );
    m_dropped.store( 0, std::memory_order_relaxed );
    m_session.store( 1, std::memory_order_relaxed );
    m_producerSequence = 0;
    m_window.clear();
    m_summary           = {};
    m_summary.session   = 1;
    m_summary.enabled   = m_enabled;
    m_summary.windowSec = m_enabled ? windowSec : 0;
    m_summary.capacity  = m_queue.size();
    m_analysisSession   = 0;
    m_analysisSequence  = 0;
    resetBaselines();
}

inline bool dmCombTiming::enabled() const
{
    return m_enabled;
}

inline void dmCombTiming::newSession()
{
    if( m_enabled )
    {
        m_session.fetch_add( 1, std::memory_order_release );
    }
}

inline void dmCombTiming::enqueue( Event &event )
{
    event.sequence = ++m_producerSequence;
    event.session  = m_session.load( std::memory_order_relaxed );
    uint64_t head  = m_head.load( std::memory_order_relaxed );
    uint64_t tail  = m_tail.load( std::memory_order_acquire );

    if( head - tail == m_queue.size() )
    {
        m_dropped.fetch_add( 1, std::memory_order_relaxed );

        return;
    }

    m_queue[head % m_queue.size()] = event;
    m_head.store( head + 1, std::memory_order_release );
}

inline void
dmCombTiming::observe( const IMAGE &image, ino_t expectedPrimaryInode, int64_t observedNs, int64_t receiptNs )
{
    if( !m_enabled )
    {
        return;
    }

    Event event;
    event.expectedPrimary         = expectedPrimaryInode;
    event.observed                = observedNs;
    event.receipt                 = receiptNs;
    const volatile auto *metadata = image.md;

    if( metadata && metadata->write == 0 )
    {
        event.counter = metadata->cnt0;
        timespec published{ metadata->writetime.tv_sec, metadata->writetime.tv_nsec };
        event.writetime      = timestamp( published );
        const auto rows      = metadata->NBproctrace;
        event.tracePresent   = image.streamproctrace && rows > 0;
        event.primaryPresent = image.streamproctrace && rows > 1;

        if( event.tracePresent )
        {
            event.trace = trace( image.streamproctrace[0] );
        }

        if( event.primaryPresent )
        {
            event.primary = trace( image.streamproctrace[1] );
        }

        std::atomic_thread_fence( std::memory_order_acquire );
        event.traceStable   = event.tracePresent && same( event.trace, trace( image.streamproctrace[0] ) );
        event.primaryStable = event.primaryPresent && same( event.primary, trace( image.streamproctrace[1] ) );
        event.metadataValid = metadata->write == 0 && metadata->cnt0 == event.counter &&
                              metadata->writetime.tv_sec == published.tv_sec &&
                              metadata->writetime.tv_nsec == published.tv_nsec && metadata->NBproctrace == rows &&
                              event.writetime > 0 && anchor( event ) >= 0;
    }

    enqueue( event );
}

inline void dmCombTiming::rejected( int64_t receiptNs )
{
    if( !m_enabled )
    {
        return;
    }

    Event event;
    event.receipt  = receiptNs;
    event.observed = receiptNs;
    event.rejected = true;
    enqueue( event );
}

inline void dmCombTiming::resetBaselines()
{
    m_haveCounter = false;
    m_haveStart   = false;
    m_havePrimary = false;
}

inline void dmCombTiming::qualify( Event &event, bool identity )
{
    using namespace MagAOX::dmCombTiming;
    auto &quality = m_summary.total;
    ++quality.events;
    quality.receipts += event.receipt >= 0;
    quality.synthetic += event.receipt < 0;

    if( event.session != m_analysisSession || event.sequence != m_analysisSequence + 1 )
    {
        resetBaselines();
    }

    m_analysisSession  = event.session;
    m_analysisSequence = event.sequence;

    if( event.rejected )
    {
        ++quality.rejected;
        resetBaselines();

        return;
    }

    if( !event.metadataValid )
    {
        ++quality.invalidMetadata;
        resetBaselines();

        return;
    }

    bool duplicate = m_haveCounter && event.counter == m_counter;
    bool gap       = m_haveCounter && !duplicate && ( event.counter <= m_counter || event.counter != m_counter + 1 );
    quality.duplicates += duplicate;
    quality.counterGaps += gap;
    m_counter     = event.counter;
    m_haveCounter = true;

    if( gap )
    {
        m_haveStart = m_havePrimary = false;
    }

    if( duplicate )
    {
        return;
    }

    if( !event.tracePresent )
    {
        ++quality.missingTrace;
        m_haveStart = m_havePrimary = false;

        return;
    }

    int64_t start  = timestamp( event.trace.start );
    int64_t update = timestamp( event.trace.update );

    if( !event.traceStable || event.trace.pid <= 0 || start <= 0 || update < start || event.writetime < update )
    {
        ++quality.invalidTrace;
        m_haveStart = m_havePrimary = false;

        return;
    }

    if( identity )
    {
        m_summary.writerPid      = event.trace.pid;
        m_summary.triggerInode   = event.trace.inode;
        m_summary.primaryCounter = event.trace.counter;
    }

    bool sameWriter = m_haveStart && m_writer == event.trace.pid;
    bool freshStart = sameWriter && start > m_start;
    quality.repeatedStart += sameWriter && !freshStart;
    m_writer    = event.trace.pid;
    m_start     = start;
    m_haveStart = true;

    auto duration = [&]( Metric metric, int64_t value )
    {
        event.durations[index( metric )] = value;
        event.valid |= 1u << index( metric );
    };

    if( freshStart )
    {
        duration( Metric::execution, update - start );
        duration( Metric::publication, event.writetime - update );
    }

    if( event.expectedPrimary == 0 || event.trace.inode != event.expectedPrimary )
    {
        ++quality.unlinkedPrimary;
        m_havePrimary = false;

        return;
    }

    int64_t upstream = timestamp( event.primary.update );

    if( !event.primaryPresent || !event.primaryStable || event.primary.pid <= 0 || upstream <= 0 || upstream > start )
    {
        ++quality.invalidPrimary;
        m_havePrimary = false;

        return;
    }

    bool comparable =
        m_havePrimary && sameWriter && m_primaryInode == event.trace.inode && m_primaryWriter == event.primary.pid;
    bool freshPrimary = comparable && event.trace.counter > m_primaryCounter && upstream > m_primaryUpdate;
    quality.repeatedPrimary += comparable && !freshPrimary;
    m_primaryInode   = event.trace.inode;
    m_primaryCounter = event.trace.counter;
    m_primaryWriter  = event.primary.pid;
    m_primaryUpdate  = upstream;
    m_havePrimary    = true;

    if( freshStart && freshPrimary )
    {
        duration( Metric::primaryStart, start - upstream );
        duration( Metric::primaryPublication, event.writetime - upstream );
    }
}

inline void dmCombTiming::update( int64_t nowNs )
{
    ++m_summary.sequence;
    m_summary.nowNs = nowNs;

    if( !m_enabled )
    {
        return;
    }

    auto session = m_session.load( std::memory_order_acquire );

    if( session != m_summary.session )
    {
        m_window.clear();
        m_summary.session      = session;
        m_summary.writerPid    = 0;
        m_summary.triggerInode = m_summary.primaryCounter = 0;
    }

    uint64_t       tail = m_tail.load( std::memory_order_relaxed );
    const uint64_t head = m_head.load( std::memory_order_acquire );

    while( tail != head )
    {
        Event event = m_queue[tail % m_queue.size()];
        ++tail;
        m_tail.store( tail, std::memory_order_release );

        if( event.session > m_summary.session )
        {
            m_window.clear();
            m_summary.session      = event.session;
            m_summary.writerPid    = 0;
            m_summary.triggerInode = m_summary.primaryCounter = 0;
        }

        qualify( event, event.session == m_summary.session );

        if( event.session == m_summary.session )
        {
            m_window.push_back( event );

            if( m_window.size() > m_queue.size() )
            {
                m_window.pop_front();
            }
        }
    }

    session = m_session.load( std::memory_order_acquire );

    if( session > m_summary.session )
    {
        m_window.clear();
        m_summary.session      = session;
        m_summary.writerPid    = 0;
        m_summary.triggerInode = m_summary.primaryCounter = 0;
    }

    m_summary.queued        = static_cast<size_t>( m_head.load( std::memory_order_acquire ) - tail );
    m_summary.total.dropped = m_dropped.load( std::memory_order_relaxed );
    calculate( nowNs );
}

inline void dmCombTiming::calculate( int64_t nowNs )
{
    using namespace MagAOX::dmCombTiming;
    std::array<double, index( Metric::count )>  sumSquares{};
    std::array<int64_t, index( Metric::count )> earliest, latest;
    earliest.fill( std::numeric_limits<int64_t>::max() );
    latest.fill( -1 );
    m_summary.metrics = {};
    m_summary.span    = 0;
    m_summary.age     = -1;

    if( nowNs < 0 )
    {
        return;
    }

    int64_t cutoff = nowNs >= m_windowNs ? nowNs - m_windowNs : 0;

    while( !m_window.empty() && anchor( m_window.front() ) < cutoff )
    {
        m_window.pop_front();
    }

    auto add = [&]( Metric metric, int64_t valueNs, int64_t a, int64_t b )
    {
        if( valueNs < 0 || a < cutoff || b < a || b > nowNs )
        {
            return;
        }

        size_t position = index( metric );
        auto  &stats    = m_summary.metrics[position];
        double value    = valueNs * 1e-9;
        ++stats.count;
        double difference = value - stats.mean;
        stats.mean += difference / stats.count;
        sumSquares[position] += difference * ( value - stats.mean );
        stats.min          = stats.count == 1 ? value : std::min( stats.min, value );
        stats.max          = stats.count == 1 ? value : std::max( stats.max, value );
        earliest[position] = std::min( earliest[position], a );
        latest[position]   = std::max( latest[position], b );
    };

    int64_t      oldest = std::numeric_limits<int64_t>::max(), newest = -1;
    const Event *previous = nullptr;

    for( const auto &event : m_window )
    {
        int64_t observed = anchor( event );

        if( observed >= cutoff && observed <= nowNs )
        {
            oldest = std::min( oldest, observed );
            newest = std::max( newest, observed );
        }

        if( previous && event.sequence == previous->sequence + 1 && event.session == previous->session )
        {
            if( previous->receipt >= 0 && event.receipt >= previous->receipt )
            {
                add( Metric::receiptInterval, event.receipt - previous->receipt, previous->receipt, event.receipt );
            }

            if( previous->metadataValid && event.metadataValid && event.counter > previous->counter &&
                event.counter == previous->counter + 1 && event.writetime > previous->writetime )
            {
                add(
                    Metric::publicationInterval, event.writetime - previous->writetime, anchor( *previous ), observed );
            }
        }

        for( size_t i = index( Metric::execution ); i < index( Metric::count ); ++i )
        {
            if( event.valid & ( 1u << i ) )
            {
                add( Metric( i ), event.durations[i], observed, observed );
            }
        }

        previous = &event;
    }

    for( size_t i = 0; i < m_summary.metrics.size(); ++i )
    {
        auto &stats = m_summary.metrics[i];
        stats.valid = stats.count > 1;

        if( stats.valid )
        {
            stats.stddev = std::sqrt( std::max( 0.0, sumSquares[i] / ( stats.count - 1 ) ) );
        }

        if( stats.count )
        {
            stats.span = ( latest[i] - earliest[i] ) * 1e-9;
        }
    }

    if( newest >= 0 )
    {
        m_summary.span = ( newest - oldest ) * 1e-9;
        m_summary.age  = ( nowNs - newest ) * 1e-9;
    }
}

inline const MagAOX::dmCombTiming::Summary &dmCombTiming::summary() const
{
    return m_summary;
}

} // namespace dev
} // namespace app
} // namespace MagAOX
#endif
