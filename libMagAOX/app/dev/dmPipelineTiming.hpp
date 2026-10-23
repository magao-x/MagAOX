/** \file dmPipelineTiming.hpp
 * \brief Bounded collection of existing upstream process traces and publication-conditioned stage timing.
 */
#ifndef dev_dmPipelineTiming_hpp
#define dev_dmPipelineTiming_hpp

#include <ImageStreamIO/ImageStruct.h>

#include "../../common/dmPipelineTiming.hpp"
#include "dmTiming.hpp"

namespace MagAOX
{
namespace app
{
namespace dev
{

/// Capture existing trace metadata without changing the AO stream or adding producer clocks.
/** Configure with both owner threads stopped. One monitor thread owns observe,
 * rejected and newSession; ordinary app logic owns update and summary. Full
 * populated trace chains must be stable, ordered and progressing. Existing MILK
 * process timing must be enabled to refresh process-start timestamps. Repeated
 * read brackets improve consistency but do not prove consumed-frame identity.
 */
class dmPipelineTiming
{
  protected:
    /// Fields needed to identify, qualify and compare one existing shared trace row.
    struct Trace
    {
        int64_t  pid{ 0 };     ///< Writer PID, zero for an unused row.
        uint64_t inode{ 0 };   ///< Trigger-input stream inode.
        uint64_t counter{ 0 }; ///< Trigger-input counter, compared independently for each row.
        int      status{ 0 };  ///< Existing trigger result; linked rows must report received.
        int64_t  start{ -1 };  ///< Existing external-clock process-start timestamp in ns.
        int64_t  update{ -1 }; ///< Existing external-clock output-update timestamp in ns.
    };

    /// Fixed-size bounded handoff; qualifications are set only by the consumer.
    struct Event
    {
        uint64_t sequence{ 0 };                                         ///< Producer sequence including dropped events.
        uint64_t session{ 0 };                                          ///< Monitor attachment epoch.
        uint64_t counter{ 0 };                                          ///< Primary output counter.
        int64_t  observed{ -1 };                                        ///< Local callback observation time.
        int64_t  receipt{ -1 };                                         ///< Local successful-wait time, or unavailable.
        int64_t  published{ -1 };                                       ///< External-clock primary writetime.
        size_t   rows{ 0 };                                             ///< Number of populated captured trace rows.
        std::array<Trace, MagAOX::dmPipelineTiming::maxStages> trace{}; ///< Fixed captured trace chain.
        bool rejected{ false };      ///< Notification rejected before callback processing.
        bool metadataValid{ false }; ///< Publication metadata passed repeated-read checks.
        bool traceStable{ false };   ///< Both copies of every captured row agree.
        bool truncated{ false };     ///< Source row capacity exceeded the capture bound.
        bool traceValid{ false };    ///< Coherent populated chain passed intrinsic validation.
        bool qualified{ false };     ///< Real consecutive receipt has a fresh comparable chain.
    };

    std::vector<Event> m_queue; ///< Preallocated SPSC slots, never overwritten while unread.

    std::atomic<uint64_t> m_head{ 0 }; ///< Producer position published after writing a complete slot.

    std::atomic<uint64_t> m_tail{ 0 }; ///< Consumer position released after copying a slot.

    std::atomic<uint64_t> m_dropped{ 0 }; ///< Full-queue observations lost without waiting.

    std::atomic<uint64_t> m_session{ 1 }; ///< Attachment epoch, including epochs without observations.

    uint64_t m_producerSequence{ 0 }; ///< Producer-only sequence including queue losses.

    bool m_enabled{ false }; ///< Immutable capture configuration while the owner threads run.

    int64_t m_windowNs{ 0 }; ///< Trailing local observation duration.

    std::deque<Event> m_window; ///< Consumer-owned retained observations.

    std::vector<int64_t> m_scratch; ///< Consumer median workspace reserved during configuration.

    MagAOX::dmPipelineTiming::Summary m_summary; ///< Coherent consumer-owned snapshot.

    Event m_previous; ///< Last consumed coherent baseline, independent of retained-window expiry.

    bool m_havePrevious{ false }; ///< Previous event can establish trace freshness.

    Event m_counterPrevious; ///< Last metadata observation used for duplicate/gap counts.

    bool m_haveCounter{ false }; ///< Previous metadata counter is comparable.

    /// Normalize a positive external timestamp without overflowing integer nanoseconds.
    static int64_t timestamp( const timespec &value /**< [in] External-clock timestamp. */ );

    /// Copy only trace fields used by timing qualification.
    static Trace trace( const volatile STREAM_PROC_TRACE &source /**< [in] Existing shared trace row. */ );

    /// Compare used trace fields without consulting struct padding.
    static bool same( const Trace &a /**< [in] First capture. */, const Trace &b /**< [in] Second capture. */ );

    /// Require row count, writer and trigger identity to match between observations.
    static bool topology( const Event &a /**< [in] Earlier chain. */, const Event &b /**< [in] Later chain. */ );

    /// Return the local coverage anchor without mixing producer and observer clock domains.
    static int64_t anchor( const Event &event /**< [in] Captured observation. */ );

    /// Require two consecutive real publication observations within the current local window.
    static bool ordered( const Event &a /**< [in] Earlier observation. */,
                         const Event &b /**< [in] Later observation. */,
                         int64_t      cutoff /**< [in] Oldest retained local timestamp. */,
                         int64_t      nowNs /**< [in] Latest accepted local timestamp. */ );

    /// Publish one fixed event without allocating or waiting for queue capacity.
    void enqueue( Event &event /**< [in,out] Completed producer capture. */ );

    /// Count and qualify a copied observation exactly once outside the monitor thread.
    void qualify( Event &event /**< [in,out] Consumer-owned observation. */ );

    /// Recompute trailing medians and per-stage populations using only consumer-owned data.
    void calculate( int64_t nowNs /**< [in] Monotonic evaluation timestamp. */ );

  public:
    /// Allocate bounded storage and reset all state while both owner threads are stopped.
    void configure(
        double windowSec /**< [in] Positive trailing window in seconds; zero disables capture. */,
        size_t capacity /**< [in] Maximum queued and retained observations; zero disables capture. */,
        double excursionSec /**< [in] Positive distance from publication median; zero disables classification. */ );

    /// Return whether bounded capture storage is configured.
    bool enabled() const;

    /// Start a new monitor attachment epoch on the producer thread.
    void newSession();

    /// Capture existing metadata and traces without allocation, blocking or new clock reads.
    void
    observe( const IMAGE &image /**< [in] Attached primary stream descriptor. */,
             int64_t      observedNs /**< [in] Already captured local callback time. */,
             int64_t receiptNs = -1 /**< [in] Already captured successful-wait time, or -1 for a synthetic event. */ );

    /// Record a real receipt rejected before the normal callback.
    void rejected( int64_t receiptNs /**< [in] Existing successful-wait timestamp. */ );

    /// Drain a bounded queue snapshot and refresh the ordinary-thread summary.
    void update( int64_t nowNs = dmTiming::monotonicNowNs() /**< [in] Local monotonic evaluation timestamp. */ );

    /// Return the coherent consumer-owned snapshot.
    const MagAOX::dmPipelineTiming::Summary &summary() const;
};

inline int64_t dmPipelineTiming::timestamp( const timespec &value )
{
    if( value.tv_sec < 0 || value.tv_nsec < 0 || value.tv_nsec >= 1000000000 ||
        value.tv_sec > ( std::numeric_limits<int64_t>::max() - value.tv_nsec ) / 1000000000 )
    {
        return -1;
    }

    int64_t result = static_cast<int64_t>( value.tv_sec ) * 1000000000 + value.tv_nsec;

    return result > 0 ? result : -1;
}

inline dmPipelineTiming::Trace dmPipelineTiming::trace( const volatile STREAM_PROC_TRACE &source )
{
    Trace result;
    result.pid     = source.procwrite_PID;
    result.inode   = source.trigger_inode;
    result.counter = source.cnt0;
    result.status  = source.triggerstatus;
    result.start   = timestamp( timespec{ source.ts_procstart.tv_sec, source.ts_procstart.tv_nsec } );
    result.update  = timestamp( timespec{ source.ts_streamupdate.tv_sec, source.ts_streamupdate.tv_nsec } );

    return result;
}

inline bool dmPipelineTiming::same( const Trace &a, const Trace &b )
{
    return a.pid == b.pid && a.inode == b.inode && a.counter == b.counter && a.status == b.status &&
           a.start == b.start && a.update == b.update;
}

inline bool dmPipelineTiming::topology( const Event &a, const Event &b )
{
    if( a.rows != b.rows )
    {
        return false;
    }

    for( size_t i = 0; i < a.rows; ++i )
    {
        if( a.trace[i].pid != b.trace[i].pid || a.trace[i].inode != b.trace[i].inode )
        {
            return false;
        }
    }

    return true;
}

inline int64_t dmPipelineTiming::anchor( const Event &event )
{
    return event.receipt >= 0 ? event.receipt : event.observed;
}

inline bool dmPipelineTiming::ordered( const Event &a, const Event &b, int64_t cutoff, int64_t nowNs )
{
    return a.metadataValid && b.metadataValid && !a.rejected && !b.rejected && a.receipt >= cutoff &&
           b.receipt > a.receipt && b.receipt <= nowNs && a.session == b.session && b.sequence == a.sequence + 1 &&
           b.counter > a.counter && b.counter == a.counter + 1 && b.published > a.published;
}

inline void dmPipelineTiming::configure( double windowSec, size_t capacity, double excursionSec )
{
    static_assert( std::atomic<uint64_t>::is_always_lock_free, "Pipeline timing requires lock-free queue positions" );

    m_enabled  = mx::math::isFinite( windowSec ) && windowSec > 0 && windowSec < 9e9 && capacity > 0;
    m_windowNs = m_enabled ? static_cast<int64_t>( windowSec * 1e9 ) : 0;
    m_enabled  = m_enabled && m_windowNs > 0;
    m_queue.clear();
    m_queue.resize( m_enabled ? capacity : 0 );
    m_scratch.clear();
    m_scratch.reserve( m_queue.size() );
    m_window.clear();
    m_head.store( 0, std::memory_order_relaxed );
    m_tail.store( 0, std::memory_order_relaxed );
    m_dropped.store( 0, std::memory_order_relaxed );
    m_session.store( 1, std::memory_order_relaxed );
    m_producerSequence = 0;
    m_havePrevious = m_haveCounter = false;
    m_summary                      = {};
    m_summary.enabled              = m_enabled;
    m_summary.session              = 1;
    m_summary.windowSec            = m_enabled ? windowSec : 0;
    m_summary.capacity             = m_queue.size();
    m_summary.threshold = m_enabled && mx::math::isFinite( excursionSec ) && excursionSec > 0 ? excursionSec : 0;
}

inline bool dmPipelineTiming::enabled() const
{
    return m_enabled;
}

inline void dmPipelineTiming::newSession()
{
    if( m_enabled )
    {
        m_session.fetch_add( 1, std::memory_order_release );
    }
}

inline void dmPipelineTiming::enqueue( Event &event )
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

inline void dmPipelineTiming::observe( const IMAGE &image, int64_t observedNs, int64_t receiptNs )
{
    if( !m_enabled )
    {
        return;
    }

    Event event;
    event.observed                = observedNs;
    event.receipt                 = receiptNs;
    const volatile auto *metadata = image.md;

    if( metadata && metadata->write == 0 )
    {
        event.counter = metadata->cnt0;
        timespec published{ metadata->writetime.tv_sec, metadata->writetime.tv_nsec };
        event.published   = timestamp( published );
        auto   rows       = metadata->NBproctrace;
        size_t captured   = std::min<size_t>( rows, MagAOX::dmPipelineTiming::maxStages );
        event.truncated   = rows > MagAOX::dmPipelineTiming::maxStages;
        event.traceStable = image.streamproctrace && captured > 0;

        if( image.streamproctrace )
        {
            for( size_t i = 0; i < captured; ++i )
            {
                event.trace[i] = trace( image.streamproctrace[i] );
            }
        }

        std::atomic_thread_fence( std::memory_order_acquire );

        if( image.streamproctrace )
        {
            for( size_t i = 0; i < captured; ++i )
            {
                event.traceStable = event.traceStable && same( event.trace[i], trace( image.streamproctrace[i] ) );
            }
        }

        event.rows = captured;

        while( event.rows && event.trace[event.rows - 1].pid == 0 )
        {
            --event.rows;
        }

        event.metadataValid = metadata->write == 0 && metadata->cnt0 == event.counter &&
                              metadata->writetime.tv_sec == published.tv_sec &&
                              metadata->writetime.tv_nsec == published.tv_nsec && metadata->NBproctrace == rows &&
                              event.published > 0 && anchor( event ) >= 0;
    }

    enqueue( event );
}

inline void dmPipelineTiming::rejected( int64_t receiptNs )
{
    if( !m_enabled )
    {
        return;
    }

    Event event;
    event.observed = event.receipt = receiptNs;
    event.rejected                 = true;
    enqueue( event );
}

inline void dmPipelineTiming::qualify( Event &event )
{
    auto &quality = m_summary.total;
    ++quality.events;
    quality.receipts += event.receipt >= 0;
    quality.synthetic += event.receipt < 0;
    quality.truncated += event.truncated;

    if( m_havePrevious && ( event.session != m_previous.session || event.sequence != m_previous.sequence + 1 ) )
    {
        m_havePrevious = false;
    }

    if( m_haveCounter &&
        ( event.session != m_counterPrevious.session || event.sequence != m_counterPrevious.sequence + 1 ) )
    {
        m_haveCounter = false;
    }

    if( event.rejected || !event.metadataValid )
    {
        quality.rejected += event.rejected;
        quality.invalidMetadata += !event.rejected;
        m_haveCounter = m_havePrevious = false;

        return;
    }

    bool duplicate = m_haveCounter && event.counter == m_counterPrevious.counter;
    bool gap       = m_haveCounter && !duplicate &&
               ( event.counter <= m_counterPrevious.counter || event.counter != m_counterPrevious.counter + 1 );
    quality.duplicates += duplicate;
    quality.counterGaps += gap;
    m_counterPrevious = event;
    m_haveCounter     = true;

    if( duplicate || event.receipt < 0 )
    {
        m_havePrevious = false;

        return;
    }

    if( event.rows == 0 )
    {
        ++quality.missingTrace;
        m_havePrevious = false;

        return;
    }

    bool valid = event.traceStable;

    for( size_t i = 0; i < event.rows; ++i )
    {
        const auto &row = event.trace[i];
        valid = valid && row.pid > 0 && row.start > 0 && row.update >= row.start && row.update <= event.published &&
                ( row.inode == 0 || row.status == 2 );

        if( i + 1 < event.rows )
        {
            valid = valid && event.trace[i + 1].update <= row.start && row.inode != 0;
        }
    }

    if( !valid )
    {
        ++quality.invalidTrace;
        m_havePrevious = false;

        return;
    }

    event.traceValid = true;
    bool comparable  = m_havePrevious && !gap && topology( m_previous, event );

    if( m_havePrevious && !topology( m_previous, event ) )
    {
        ++quality.topologyChanges;
    }

    if( comparable )
    {
        valid = ordered( m_previous, event, 0, std::numeric_limits<int64_t>::max() );

        for( size_t i = 0; i < event.rows; ++i )
        {
            const auto &previous = m_previous.trace[i];
            const auto &row      = event.trace[i];
            valid                = valid && row.start > previous.start && row.update > previous.update &&
                    ( row.inode == 0 || ( row.counter > previous.counter && row.counter == previous.counter + 1 ) );
        }

        if( !valid )
        {
            ++quality.invalidTrace;
            event.traceValid = false;
            m_havePrevious   = false;

            return;
        }

        event.qualified = true;
    }

    m_previous     = event;
    m_havePrevious = true;
}

inline void dmPipelineTiming::update( int64_t nowNs )
{
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
        m_havePrevious = m_haveCounter = false;
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
            m_summary.session = event.session;
            m_havePrevious = m_haveCounter = false;
        }

        qualify( event );

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
        m_summary.session = session;
        m_havePrevious = m_haveCounter = false;
    }

    m_summary.queued        = static_cast<size_t>( m_head.load( std::memory_order_acquire ) - tail );
    m_summary.total.dropped = m_dropped.load( std::memory_order_relaxed );
    calculate( nowNs );
}

inline void dmPipelineTiming::calculate( int64_t nowNs )
{
    using namespace MagAOX::dmPipelineTiming;
    m_summary.stages                 = {};
    m_summary.sourceInterval         = {};
    m_summary.lateSourceInterval     = {};
    m_summary.earlySourceInterval    = {};
    m_summary.latePublicationChange  = {};
    m_summary.earlyPublicationChange = {};
    m_summary.stageCount             = 0;
    m_summary.median                 = 0;
    m_summary.intervals = m_summary.late = m_summary.early = 0;
    m_summary.age                                          = -1;
    m_summary.span                                         = 0;
    m_scratch.clear();

    if( nowNs < 0 )
    {
        return;
    }

    int64_t cutoff = nowNs >= m_windowNs ? nowNs - m_windowNs : 0;

    while( !m_window.empty() && anchor( m_window.front() ) < cutoff )
    {
        m_window.pop_front();
    }

    const Event *previous = nullptr;
    const Event *latest   = nullptr;
    int64_t      oldest = std::numeric_limits<int64_t>::max(), newest = -1;

    for( const auto &event : m_window )
    {
        int64_t observed = anchor( event );

        if( observed >= cutoff && observed <= nowNs )
        {
            oldest = std::min( oldest, observed );
            newest = std::max( newest, observed );

            if( event.traceValid )
            {
                latest = &event;
            }
        }

        if( previous && ordered( *previous, event, cutoff, nowNs ) )
        {
            m_scratch.push_back( event.published - previous->published );
        }

        previous = &event;
    }

    if( newest >= 0 )
    {
        m_summary.span = ( newest - oldest ) * 1e-9;
        m_summary.age  = ( nowNs - newest ) * 1e-9;
    }

    m_summary.intervals = m_scratch.size();

    if( !m_scratch.empty() )
    {
        auto middle = m_scratch.begin() + m_scratch.size() / 2;
        std::nth_element( m_scratch.begin(), middle, m_scratch.end() );
        double median = *middle;

        if( m_scratch.size() % 2 == 0 )
        {
            median = 0.5 * median + 0.5 * *std::max_element( m_scratch.begin(), middle );
        }

        m_summary.median = median * 1e-9;
    }

    if( latest )
    {
        m_summary.stageCount = latest->rows;

        for( size_t i = 0; i < latest->rows; ++i )
        {
            auto &stage      = m_summary.stages[i];
            stage.rowIndex   = i;
            stage.pid        = latest->trace[i].pid;
            stage.inputInode = latest->trace[i].inode;
        }
    }

    constexpr size_t                                        populations = 10;
    std::array<std::array<double, populations>, maxStages>  squares{};
    std::array<std::array<int64_t, populations>, maxStages> first{}, last{};

    for( auto &row : first )
    {
        row.fill( std::numeric_limits<int64_t>::max() );
    }

    auto fields = []( Stage &stage )
    {
        return std::array<Stats *, populations>{ &stage.execution,
                                                 &stage.handoff,
                                                 &stage.lateExecution,
                                                 &stage.lateHandoff,
                                                 &stage.earlyExecution,
                                                 &stage.earlyHandoff,
                                                 &stage.lateExecutionChange,
                                                 &stage.lateHandoffChange,
                                                 &stage.earlyExecutionChange,
                                                 &stage.earlyHandoffChange };
    };

    auto add = [&]( size_t row, size_t metric, int64_t valueNs, int64_t a, int64_t b )
    {
        auto   pointers = fields( m_summary.stages[row] );
        auto  &stats    = *pointers[metric];
        double value    = valueNs * 1e-9;
        ++stats.count;
        double difference = value - stats.mean;
        stats.mean += difference / stats.count;
        squares[row][metric] += difference * ( value - stats.mean );
        stats.min          = stats.count == 1 ? value : std::min( stats.min, value );
        stats.max          = stats.count == 1 ? value : std::max( stats.max, value );
        first[row][metric] = std::min( first[row][metric], a );
        last[row][metric]  = std::max( last[row][metric], b );
    };

    std::array<Stats *, 5> boundary{ &m_summary.sourceInterval,
                                     &m_summary.lateSourceInterval,
                                     &m_summary.earlySourceInterval,
                                     &m_summary.latePublicationChange,
                                     &m_summary.earlyPublicationChange };
    std::array<double, 5>  boundarySquares{};
    std::array<int64_t, 5> boundaryFirst, boundaryLast{};
    boundaryFirst.fill( std::numeric_limits<int64_t>::max() );

    auto addBoundary = [&]( size_t index, int64_t valueNs, int64_t a, int64_t b )
    {
        auto  &stats = *boundary[index];
        double value = valueNs * 1e-9;
        ++stats.count;
        double difference = value - stats.mean;
        stats.mean += difference / stats.count;
        boundarySquares[index] += difference * ( value - stats.mean );
        stats.min            = stats.count == 1 ? value : std::min( stats.min, value );
        stats.max            = stats.count == 1 ? value : std::max( stats.max, value );
        boundaryFirst[index] = std::min( boundaryFirst[index], a );
        boundaryLast[index]  = std::max( boundaryLast[index], b );
    };

    previous = nullptr;

    for( const auto &event : m_window )
    {
        bool consecutive = previous && ordered( *previous, event, cutoff, nowNs );
        int  direction   = 0;

        if( consecutive && m_summary.threshold > 0 )
        {
            double interval = ( event.published - previous->published ) * 1e-9;
            direction       = interval > m_summary.median + m_summary.threshold   ? 1
                              : interval < m_summary.median - m_summary.threshold ? -1
                                                                                  : 0;
            m_summary.late += direction > 0;
            m_summary.early += direction < 0;
        }

        int64_t observed = anchor( event );

        if( latest && event.qualified && event.traceValid && observed >= cutoff && observed <= nowNs &&
            topology( *latest, event ) )
        {
            if( consecutive && previous->traceValid && topology( *previous, event ) )
            {
                int64_t sourceInterval = event.trace[event.rows - 1].start - previous->trace[event.rows - 1].start;
                addBoundary( 0, sourceInterval, anchor( *previous ), observed );

                if( direction )
                {
                    int64_t publicationChange = ( event.published - event.trace[0].update ) -
                                                ( previous->published - previous->trace[0].update );
                    addBoundary( direction > 0 ? 1 : 2, sourceInterval, anchor( *previous ), observed );
                    addBoundary( direction > 0 ? 3 : 4, publicationChange, anchor( *previous ), observed );
                }
            }

            for( size_t i = 0; i < event.rows; ++i )
            {
                int64_t execution = event.trace[i].update - event.trace[i].start;
                add( i, 0, execution, observed, observed );

                if( i + 1 < event.rows )
                {
                    add( i, 1, event.trace[i].start - event.trace[i + 1].update, observed, observed );
                }

                if( direction && previous->traceValid && topology( *previous, event ) )
                {
                    size_t  base           = direction > 0 ? 2 : 4;
                    size_t  change         = direction > 0 ? 6 : 8;
                    int64_t priorExecution = previous->trace[i].update - previous->trace[i].start;
                    add( i, base, execution, observed, observed );
                    add( i, change, execution - priorExecution, anchor( *previous ), observed );

                    if( i + 1 < event.rows )
                    {
                        int64_t handoff      = event.trace[i].start - event.trace[i + 1].update;
                        int64_t priorHandoff = previous->trace[i].start - previous->trace[i + 1].update;
                        add( i, base + 1, handoff, observed, observed );
                        add( i, change + 1, handoff - priorHandoff, anchor( *previous ), observed );
                    }
                }
            }
        }

        previous = &event;
    }

    for( size_t i = 0; i < boundary.size(); ++i )
    {
        auto &stats = *boundary[i];
        stats.valid = stats.count > 1;

        if( stats.valid )
        {
            stats.stddev = std::sqrt( std::max( 0.0, boundarySquares[i] / ( stats.count - 1 ) ) );
        }

        if( stats.count )
        {
            stats.span = ( boundaryLast[i] - boundaryFirst[i] ) * 1e-9;
        }
    }

    for( size_t i = 0; i < m_summary.stageCount; ++i )
    {
        auto pointers = fields( m_summary.stages[i] );

        for( size_t j = 0; j < populations; ++j )
        {
            auto &stats = *pointers[j];
            stats.valid = stats.count > 1;

            if( stats.valid )
            {
                stats.stddev = std::sqrt( std::max( 0.0, squares[i][j] / ( stats.count - 1 ) ) );
            }

            if( stats.count )
            {
                stats.span = ( last[i][j] - first[i][j] ) * 1e-9;
            }
        }
    }
}

inline const MagAOX::dmPipelineTiming::Summary &dmPipelineTiming::summary() const
{
    return m_summary;
}

} // namespace dev
} // namespace app
} // namespace MagAOX
#endif
