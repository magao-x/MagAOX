/** \file frameGrabberTiming.hpp
 * \brief Ownership-safe paired framegrabber timing and optional consumer-side stage attribution.
 */
#ifndef dev_frameGrabberTiming_hpp
#define dev_frameGrabberTiming_hpp

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <ctime>
#include <deque>
#include <limits>
#include <utility>
#include <vector>

#include <mx/math/floatUtils.hpp>

#include "../../common/frameGrabberTiming.hpp"
#include "../../common/spscQueue.hpp"

namespace MagAOX
{
namespace app
{
namespace dev
{
/// Capture paired camera timestamps without blocking or exposing live payload storage.
/** Configure while both owners are stopped. Acquisition methods have one producer;
 * control() has a separately serialized producer. Only ordinary update/summary
 * access retained histories. Optional stage disablement adds no diagnostic clocks.
 */
class frameGrabberTiming
{
    MagAOX::spscQueue<fgTiming::Event>        m_queue;         ///< Fixed acquisition handoff.
    MagAOX::spscQueue<fgTiming::ControlEvent> m_controls;      ///< Independent serialized-control handoff.
    std::atomic<uint64_t>                     m_session{ 1 };  ///< Acquisition epoch, visible even without a new frame.
    uint64_t                                  m_sequence{ 0 }; ///< Producer attempt sequence, including losses.
    fgTiming::Event                           m_event;         ///< Producer-owned current attempt.
    std::array<uint8_t, fgTiming::markerIndex( fgTiming::Marker::count )>
            m_captureOrder{}; ///< First-capture positions, preserving overwrite validation independently of marker IDs.
    uint8_t m_captureCount{ 0 }; ///< Number of distinct successfully captured endpoints in this attempt.
    bool    m_active{ false };   ///< Producer currently owns an attempt.
    bool    m_enabled{ false };  ///< Paired timing enabled; immutable during acquisition.
    bool    m_stages{ false };   ///< Optional markers enabled; immutable during acquisition.
    int64_t m_windowNs{ 0 };     ///< Immutable local retention duration.
    int64_t m_bracketNs{ 0 };    ///< Positive attribution uncertainty bound; zero is undecided.
    uint32_t                           m_mode{ 0 };     ///< Producer's current mode tag.
    uint32_t                           m_binX{ 0 };     ///< Producer's current digital binning factor.
    uint32_t                           m_binY{ 0 };     ///< Producer's current digital binning factor.
    std::deque<fgTiming::Event>        m_window;        ///< Consumer-owned acquisition history.
    std::deque<fgTiming::ControlEvent> m_controlWindow; ///< Consumer-owned completed controls.
    std::vector<int64_t>               m_intervals;     ///< Consumer median workspace, reserved before acquisition.
    fgTiming::Summary                  m_summary;       ///< Cached ordinary-thread snapshot.
    fgTiming::Counters                 m_total;         ///< Consumer cumulative quality counts.
    fgTiming::Counters                 m_previous;      ///< Counts at preceding summary.
    fgTiming::Event                    m_previousEvent; ///< Last consumed event, for continuous quality accounting.
    bool                               m_havePrevious{ false }; ///< Previous event exists for continuity checks.
    bool                               m_everEvent{ false };    ///< Current session previously delivered events.
    fgTiming::TraceStatus              m_traceStatus;           ///< Consumer-only bounded excursion accounting.
    std::vector<fgTiming::TraceRecord> m_traceReady;   ///< Finalized records awaiting ordinary-thread serialization.
    std::vector<fgTiming::TraceRecord> m_tracePending; ///< Accepted triggers awaiting two following neighbors.
    std::array<fgTiming::Event, 2>     m_traceNeighbors{};        ///< Most recent continuous attempts, oldest first.
    size_t                             m_traceNeighborCount{ 0 }; ///< Available preceding attempts.
    uint64_t m_traceSequence{ 0 }; ///< Last classified sequence; repeated summaries never select twice.
    std::vector<std::pair<int64_t, int64_t>> m_traceIntervals; ///< Fixed ring of preceding interval and endpoint ns.
    std::vector<int64_t>                     m_traceSorted;    ///< Reserved sorted median workspace, consumer only.
    size_t                                   m_traceIntervalHead{ 0 };  ///< Oldest valid ring position.
    size_t                                   m_traceIntervalCount{ 0 }; ///< Number of preceding retained intervals.

    /// Convert a normalized nonnegative external timestamp to integer nanoseconds.
    static int64_t timestamp( const timespec &stamp /**< [in] External timestamp. */ );

    /// Return an available local endpoint or -1.
    static int64_t endpoint( const fgTiming::Event &event /**< [in] Consumer-owned event. */,
                             fgTiming::Marker       marker /**< [in] Local endpoint identifier. */ );

    /// Require adjacent successful events from one output identity and epoch.
    static bool consecutive( const fgTiming::Event &previous /**< [in] Earlier attempt. */,
                             const fgTiming::Event &event /**< [in] Later attempt. */ );

    /// Compute one nonnegative same-event local duration; false means incomplete endpoints.
    static bool duration( const fgTiming::Event &event /**< [in] Attempt to inspect. */,
                          fgTiming::Metric       metric /**< [in] Duration population. */,
                          int64_t               &value /**< [out] Duration, ns. */ );

    /// Count one consumed acquisition event, without reading producer-owned state.
    void count( const fgTiming::Event &event /**< [in] Consumer-owned attempt. */ );

    /// Calculate all retained independent populations and qualified changes.
    void calculate( int64_t nowNs /**< [in] Local monotonic evaluation time. */ );

    /// Compute the union of completed serial intervals intersecting one local frame span.
    int64_t serialOverlap( int64_t entered /**< [in] Frame's local interval start. */,
                           int64_t completed /**< [in] Frame's local interval end. */ ) const;

    /// Finish one accepted neighborhood using only completed control observations available now.
    void traceFinish( fgTiming::TraceRecord record /**< [in] Owned neighborhood to finalize. */,
                      fgTiming::TraceEnd    end /**< [in] Completion or interruption reason. */,
                      int64_t               nowNs /**< [in] Consumer monotonic evaluation time. */ );

    /// Flush pending neighborhoods and discard the preceding-interval baseline at a continuity boundary.
    void traceBoundary( fgTiming::TraceEnd end /**< [in] Interruption reason. */,
                        int64_t            nowNs /**< [in] Consumer monotonic evaluation time. */ );

    /// Classify one never-before-processed event before trailing-summary expiry.
    void traceEvent( const fgTiming::Event &event /**< [in] Consumer-owned acquisition attempt. */,
                     int64_t                nowNs /**< [in] Consumer monotonic evaluation time. */ );

    /// Remove the oldest preceding baseline interval from both fixed ring and sorted workspace.
    void tracePopInterval();

  public:
    /// Read CLOCK_MONOTONIC without changing errno; return -1 on failure.
    static int64_t monotonicNowNs();

    /// Read CLOCK_REALTIME on the ordinary thread without changing errno.
    static int64_t realtimeNowNs();

    /// Allocate initialized handoff storage while producer and consumer are stopped.
    void configure( double windowSec /**< [in] Positive trailing duration; zero disables collection. */,
                    size_t capacity /**< [in] Maximum queued/retained events per channel. */,
                    bool   stageEnabled = false /**< [in] Enable optional local stage clocks. */,
                    double excursionSec = 20e-6 /**< [in] Positive excursion distance; zero disables classification. */,
                    double bracketLimitSec = 0 /**< [in] Positive maximum accepted clock bracket; zero is undecided. */,
                    size_t traceMax =
                        0 /**< [in] Maximum ready plus pending records; zero or sub-ns threshold disables tracing. */ );

    /// Return whether paired timing collection is configured.
    bool enabled() const;

    /// Return whether optional local stage capture is configured.
    bool stagesEnabled() const;

    /// Return whether an acquisition attempt is under construction.
    bool active() const;

    /// Set immutable-per-attempt numeric mode provenance, without string allocation.
    void context( uint32_t mode /**< [in] Caller-defined mode tag. */,
                  uint32_t binX /**< [in] Digital X factor, zero when unknown. */,
                  uint32_t binY /**< [in] Digital Y factor, zero when unknown. */ );

    /// Begin an attempt, reading the local wait-enter clock only when stages are enabled.
    void begin();

    /// Begin an attempt with a caller-supplied deterministic local timestamp.
    void beginAt( int64_t waitEnterNs /**< [in] Local wait entry, ns; ignored when stages are disabled. */ );

    /// Capture one optional local endpoint without changing errno.
    void mark( fgTiming::Marker marker /**< [in] Endpoint identifier. */ );

    /// Attach a deterministic optional local endpoint.
    void markAt( fgTiming::Marker marker /**< [in] Endpoint identifier. */,
                 int64_t          nowNs /**< [in] Local monotonic ns; negative marks a failed capture. */ );

    /// Attach a validated raw camera frame number.
    void frame( uint64_t number /**< [in] Camera frame identity, including exact unsigned wrap. */ );

    /// Attach original paired timestamps and output identity after metadata completion.
    void published( const timespec &atime /**< [in] Original SDK acquisition timestamp. */,
                    const timespec &writetime /**< [in] Existing realtime publication timestamp. */,
                    uint64_t        counter /**< [in] Published output counter. */,
                    uint64_t        inode = 0 /**< [in] Output metadata inode, zero when unavailable. */ );

    /// Publish the current fixed event, counting loss without waiting for the consumer.
    void finish( fgTiming::Outcome outcome = fgTiming::Outcome::published /**< [in] Actual attempt result. */ );

    /// End a pending attempt as rejected and advance the producer epoch without resizing storage.
    void newSession();

    /// Publish a completed control operation; callers must serialize this channel's producers.
    void control( int64_t entered /**< [in] Local transaction entry, ns. */,
                  int64_t completed /**< [in] Local transaction completion, ns. */,
                  bool    success /**< [in] Caller-qualified transaction result. */ );

    /// Drain captured queue populations and update consumer-owned history and statistics.
    void update( int64_t nowMono = monotonicNowNs() /**< [in] Local monotonic evaluation time. */,
                 int64_t nowReal = realtimeNowNs() /**< [in] Realtime age anchor for paired-only samples. */ );

    /// Return the cached coherent summary; never inspect producer-owned payload storage.
    const fgTiming::Summary &summary() const;

    /// Return ready chronological neighborhoods; only the ordinary consumer may access this vector.
    const std::vector<fgTiming::TraceRecord> &traceRecords() const;

    /// Return consumer-only cumulative trace selection, retention and handoff-loss counts.
    const fgTiming::TraceStatus &traceStatus() const;

    /// Acknowledge all ready neighborhoods after ordinary-thread delivery without clearing pending neighbors.
    void clearTraceRecords();
};

inline int64_t frameGrabberTiming::timestamp( const timespec &stamp )
{
    if( stamp.tv_sec < 0 || stamp.tv_nsec < 0 || stamp.tv_nsec >= 1000000000 ||
        stamp.tv_sec > ( std::numeric_limits<int64_t>::max() - stamp.tv_nsec ) / 1000000000 )
    {
        return -1;
    }

    return static_cast<int64_t>( stamp.tv_sec ) * 1000000000LL + stamp.tv_nsec;
}

inline int64_t frameGrabberTiming::monotonicNowNs()
{
    const int saved = errno;
    timespec  stamp{};
    const int result = clock_gettime( CLOCK_MONOTONIC, &stamp );
    errno            = saved;
    return result == 0 ? timestamp( stamp ) : -1;
}

inline int64_t frameGrabberTiming::realtimeNowNs()
{
    const int saved = errno;
    timespec  stamp{};
    const int result = clock_gettime( CLOCK_REALTIME, &stamp );
    errno            = saved;
    return result == 0 ? timestamp( stamp ) : -1;
}

inline void frameGrabberTiming::configure(
    double windowSec, size_t capacity, bool stageEnabled, double excursionSec, double bracketLimitSec, size_t traceMax )
{
    m_enabled   = mx::math::isFinite( windowSec ) && windowSec > 0 && windowSec < 9e9 && capacity > 0;
    m_windowNs  = m_enabled ? static_cast<int64_t>( windowSec * 1e9 ) : 0;
    m_enabled   = m_enabled && m_windowNs > 0;
    m_stages    = m_enabled && stageEnabled;
    m_bracketNs = mx::math::isFinite( bracketLimitSec ) && bracketLimitSec > 0 && bracketLimitSec < 9e9
                      ? static_cast<int64_t>( bracketLimitSec * 1e9 )
                      : 0;
    m_queue.configure( m_enabled ? capacity : 0 );
    m_controls.configure( m_stages ? capacity : 0 );
    m_intervals.clear();
    m_intervals.reserve( m_stages ? capacity : 0 );
    m_window.clear();
    m_controlWindow.clear();
    m_session.store( 1, std::memory_order_relaxed );
    m_sequence = 0;
    m_event    = {};
    m_active   = false;
    m_mode = m_binX = m_binY = 0;
    m_summary                = {};
    m_summary.enabled        = m_enabled;
    m_summary.stagesEnabled  = m_stages;
    m_summary.session        = 1;
    m_summary.capacity       = m_queue.capacity();
    m_summary.windowSec      = m_enabled ? windowSec : 0;
    m_summary.threshold      = mx::math::isFinite( excursionSec ) && excursionSec > 0 ? excursionSec : 0;
    m_summary.bracketLimit   = m_bracketNs * 1e-9;
    m_summary.state          = m_enabled ? fgTiming::Availability::warmingUp : fgTiming::Availability::disabled;
    m_total                  = {};
    m_previous               = {};
    m_previousEvent          = {};
    m_havePrevious           = false;
    m_everEvent              = false;
    m_traceStatus            = {};
    m_traceStatus.enabled    = m_stages && traceMax > 0 && m_summary.threshold * 1e9L >= 1;
    m_traceStatus.capacity   = m_traceStatus.enabled ? traceMax : 0;
    m_traceReady.clear();
    m_traceReady.reserve( m_traceStatus.capacity );
    m_tracePending.clear();
    m_tracePending.reserve( m_traceStatus.capacity );
    m_traceNeighbors     = {};
    m_traceNeighborCount = 0;
    m_traceSequence      = 0;
    m_traceIntervalHead  = 0;
    m_traceIntervalCount = 0;
    m_traceIntervals.assign( m_traceStatus.enabled ? capacity : 0, {} );
    m_traceSorted.clear();
    m_traceSorted.reserve( m_traceIntervals.size() );
}

inline bool frameGrabberTiming::enabled() const
{
    return m_enabled;
}
inline bool frameGrabberTiming::stagesEnabled() const
{
    return m_stages;
}
inline bool frameGrabberTiming::active() const
{
    return m_active;
}

inline void frameGrabberTiming::context( uint32_t mode, uint32_t binX, uint32_t binY )
{
    m_mode = mode;
    m_binX = binX;
    m_binY = binY;
}

inline void frameGrabberTiming::begin()
{
    if( m_enabled )
    {
        beginAt( m_stages ? monotonicNowNs() : -1 );
    }
}

inline void frameGrabberTiming::beginAt( int64_t waitEnterNs )
{
    if( !m_enabled )
    {
        return;
    }

    if( m_active )
    {
        finish( fgTiming::Outcome::rejected );
    }

    m_event          = {};
    m_captureOrder   = {};
    m_captureCount   = 0;
    m_event.sequence = ++m_sequence;
    m_event.session  = m_session.load( std::memory_order_relaxed );
    m_event.mode     = m_mode;
    m_event.binX     = m_binX;
    m_event.binY     = m_binY;
    m_active         = true;
    markAt( fgTiming::Marker::waitEnter, waitEnterNs );
}

inline void frameGrabberTiming::mark( fgTiming::Marker marker )
{
    if( m_stages && m_active )
    {
        markAt( marker, monotonicNowNs() );
    }
}

inline void frameGrabberTiming::markAt( fgTiming::Marker marker, int64_t nowNs )
{
    if( !m_stages || !m_active || fgTiming::markerIndex( marker ) >= fgTiming::markerIndex( fgTiming::Marker::count ) )
    {
        return;
    }

    const size_t  index    = fgTiming::markerIndex( marker );
    bool          ordered  = nowNs >= 0;
    const uint8_t position = m_captureOrder[index] ? m_captureOrder[index] : m_captureCount + 1;

    for( size_t prior = 0; ordered && prior < m_event.markers.size(); ++prior )
    {
        if( prior != index && ( m_event.valid & ( 1u << prior ) ) &&
            ( m_captureOrder[prior] < position ? m_event.markers[prior] > nowNs : m_event.markers[prior] < nowNs ) )
        {
            ordered = false;
        }
    }

    if( !ordered )
    {
        m_event.valid &= ~( 1u << index );
        ++m_event.invalidMarkers;
        return;
    }

    m_event.markers[index] = nowNs;
    m_event.valid |= 1u << index;

    if( !m_captureOrder[index] )
    {
        m_captureOrder[index] = ++m_captureCount;
    }
}

inline void frameGrabberTiming::frame( uint64_t number )
{
    if( m_active )
    {
        m_event.frameNumber = number;
        m_event.frameValid  = true;
    }
}

inline void
frameGrabberTiming::published( const timespec &atime, const timespec &writetime, uint64_t counter, uint64_t inode )
{
    if( m_active )
    {
        m_event.atime         = atime;
        m_event.writetime     = writetime;
        m_event.outputCounter = counter;
        m_event.streamInode   = inode;
        m_event.outputValid   = true;
    }
}

inline void frameGrabberTiming::finish( fgTiming::Outcome outcome )
{
    if( m_active )
    {
        m_event.outcome = outcome;
        m_active        = false;
        m_queue.push( m_event );
    }
}

inline void frameGrabberTiming::newSession()
{
    if( !m_enabled )
    {
        return;
    }

    if( m_active )
    {
        finish( fgTiming::Outcome::rejected );
    }

    m_session.fetch_add( 1, std::memory_order_release );
}

inline void frameGrabberTiming::control( int64_t entered, int64_t completed, bool success )
{
    if( m_stages )
    {
        m_controls.push( { entered, completed, success } );
    }
}

inline int64_t frameGrabberTiming::endpoint( const fgTiming::Event &event, fgTiming::Marker marker )
{
    const size_t index = fgTiming::markerIndex( marker );
    return event.valid & ( 1u << index ) ? event.markers[index] : -1;
}

inline bool frameGrabberTiming::consecutive( const fgTiming::Event &previous, const fgTiming::Event &event )
{
    const bool frameStep = !previous.frameValid || !event.frameValid || event.frameNumber == previous.frameNumber + 1 ||
                           ( previous.frameNumber == UINT32_MAX && event.frameNumber == 0 );
    return previous.outcome == fgTiming::Outcome::published && event.outcome == fgTiming::Outcome::published &&
           previous.outputValid && event.outputValid && previous.session == event.session &&
           event.sequence == previous.sequence + 1 && event.outputCounter > previous.outputCounter &&
           event.outputCounter == previous.outputCounter + 1 && previous.streamInode == event.streamInode && frameStep;
}

inline bool frameGrabberTiming::duration( const fgTiming::Event &event, fgTiming::Metric metric, int64_t &value )
{
    using namespace fgTiming;
    Marker             first, last;
    constexpr uint32_t modern = ( 1u << markerIndex( Marker::rearmEnter ) ) |
                                ( 1u << markerIndex( Marker::timeoutEnter ) ) |
                                ( 1u << markerIndex( Marker::timeoutReturn ) );
    const bool legacy = !( event.valid & modern ) && event.invalidMarkers == 0;

    switch( metric )
    {
    case Metric::wait:
        first = Marker::waitEnter;
        last  = Marker::waitReturn;
        break;
    case Metric::rearm:
        if( endpoint( event, Marker::rearmEnter ) < 0 && !legacy )
        {
            return false;
        }

        first = endpoint( event, Marker::rearmEnter ) >= 0 ? Marker::rearmEnter : Marker::waitReturn;
        last  = Marker::rearmReturn;
        break;
    case Metric::validation:
        if( endpoint( event, Marker::timeoutReturn ) < 0 && !legacy )
        {
            return false;
        }

        first = endpoint( event, Marker::timeoutReturn ) >= 0 ? Marker::timeoutReturn : Marker::rearmReturn;
        last  = Marker::validated;
        break;
    case Metric::conversion:
        first = Marker::validated;
        last  = Marker::loadDone;
        break;
    case Metric::writeBracket:
        first = Marker::loadDone;
        last  = Marker::writeStampDone;
        break;
    case Metric::mainPost:
        first = Marker::writeStampDone;
        last  = Marker::mainPostDone;
        break;
    case Metric::syncPost:
        if( endpoint( event, Marker::rearmEnter ) < 0 &&
            ( endpoint( event, Marker::rearmReturn ) >= endpoint( event, Marker::mainPostDone ) ||
              ( event.invalidMarkers && endpoint( event, Marker::rearmReturn ) < 0 ) ) )
        {
            return false;
        }

        first = Marker::mainPostDone;
        last  = endpoint( event, Marker::rearmEnter ) >= endpoint( event, Marker::mainPostDone ) &&
                       endpoint( event, Marker::mainPostDone ) >= 0
                    ? Marker::rearmEnter
                    : Marker::syncPostDone;
        break;
    case Metric::loopTail:
        first = Marker::syncPostDone;
        last  = Marker::loopTailDone;
        break;
    case Metric::waitToWrite:
        first = Marker::waitReturn;
        last  = Marker::writeStampDone;
        break;
    case Metric::timeoutQuery:
        first = Marker::timeoutEnter;
        last  = Marker::timeoutReturn;
        break;
    default:
        return false;
    }

    const int64_t a = endpoint( event, first );
    const int64_t b = endpoint( event, last );

    if( a < 0 || b < a )
    {
        return false;
    }

    value = b - a;
    return true;
}

inline void frameGrabberTiming::count( const fgTiming::Event &event )
{
    using namespace fgTiming;
    ++m_total.events;
    m_total.published += event.outputValid;
    m_total.rejected += event.outcome == Outcome::rejected;
    m_total.failed += event.outcome == Outcome::failed;
    m_total.invalid += event.invalidMarkers;

    if( event.outputValid && ( timestamp( event.atime ) <= 0 || timestamp( event.writetime ) <= 0 ) )
    {
        ++m_total.invalid;
    }

    int64_t bracket = 0;

    if( m_bracketNs > 0 && duration( event, Metric::writeBracket, bracket ) && bracket > m_bracketNs )
    {
        ++m_total.bracketWide;
    }

    if( m_havePrevious && event.session == m_previousEvent.session && event.sequence == m_previousEvent.sequence + 1 )
    {
        if( m_previousEvent.outputValid && event.outputValid && event.streamInode == m_previousEvent.streamInode )
        {
            m_total.duplicates += event.outputCounter == m_previousEvent.outputCounter;
            m_total.counterGaps += event.outputCounter != m_previousEvent.outputCounter &&
                                   event.outputCounter != m_previousEvent.outputCounter + 1;
        }

        if( m_previousEvent.frameValid && event.frameValid )
        {
            m_total.frameGaps += event.frameNumber != m_previousEvent.frameNumber + 1 &&
                                 !( m_previousEvent.frameNumber == UINT32_MAX && event.frameNumber == 0 );
        }
    }

    m_previousEvent = event;
    m_havePrevious  = true;
}

inline void frameGrabberTiming::traceFinish( fgTiming::TraceRecord record, fgTiming::TraceEnd end, int64_t nowNs )
{
    record.end               = end;
    record.evaluatedNs       = nowNs;
    record.controlCoverageNs = nowNs;

    for( size_t index = 0; index < record.events.size(); ++index )
    {
        if( record.present & ( 1u << index ) )
        {
            record.serialOverlapNs[index] =
                serialOverlap( endpoint( record.events[index], fgTiming::Marker::waitEnter ),
                               endpoint( record.events[index], fgTiming::Marker::writeStampDone ) );
        }
    }

    m_traceStatus.incomplete += record.present != 0x1f;
    ++m_traceStatus.recorded;
    m_traceReady.push_back( record );
    m_traceStatus.ready = m_traceReady.size();
}

inline void frameGrabberTiming::traceBoundary( fgTiming::TraceEnd end, int64_t nowNs )
{
    for( const auto &record : m_tracePending )
    {
        traceFinish( record, end, nowNs );
    }

    m_tracePending.clear();
    m_traceStatus.pending = 0;
    m_traceNeighborCount  = 0;
    m_traceIntervalHead   = 0;
    m_traceIntervalCount  = 0;
    m_traceSorted.clear();
}

inline void frameGrabberTiming::tracePopInterval()
{
    const auto found =
        std::lower_bound( m_traceSorted.begin(), m_traceSorted.end(), m_traceIntervals[m_traceIntervalHead].first );
    m_traceSorted.erase( found );
    m_traceIntervalHead = ( m_traceIntervalHead + 1 ) % m_traceIntervals.size();
    --m_traceIntervalCount;
}

inline void frameGrabberTiming::traceEvent( const fgTiming::Event &event, int64_t nowNs )
{
    using namespace fgTiming;

    if( !m_traceStatus.enabled || event.sequence <= m_traceSequence || event.anchorNs > nowNs )
    {
        return;
    }

    m_traceSequence            = event.sequence;
    const Event  *previous     = m_traceNeighborCount ? &m_traceNeighbors[m_traceNeighborCount - 1] : nullptr;
    const int64_t written      = endpoint( event, Marker::writeStampDone );
    const int64_t priorWritten = previous ? endpoint( *previous, Marker::writeStampDone ) : -1;
    const bool continuous = previous && consecutive( *previous, event ) && priorWritten >= 0 && written > priorWritten;

    if( previous && !continuous )
    {
        ++m_traceStatus.discontinuities;
        traceBoundary( TraceEnd::boundary, nowNs );
        previous = nullptr;
    }

    if( continuous )
    {
        for( auto &record : m_tracePending )
        {
            const size_t next   = record.present & ( 1u << 3 ) ? 4 : 3;
            record.events[next] = event;
            record.present |= 1u << next;
        }

        while( !m_tracePending.empty() && ( m_tracePending.front().present & ( 1u << 4 ) ) )
        {
            traceFinish( m_tracePending.front(), TraceEnd::complete, nowNs );
            m_tracePending.erase( m_tracePending.begin() );
        }

        const int64_t cutoff = written >= m_windowNs ? written - m_windowNs : 0;

        while( m_traceIntervalCount && m_traceIntervals[m_traceIntervalHead].second < cutoff )
        {
            tracePopInterval();
        }

        const int64_t interval = written - priorWritten;

        if( m_traceSorted.size() >= 3 )
        {
            const size_t      middle    = m_traceSorted.size() / 2;
            const int64_t     upper     = m_traceSorted[middle];
            const int64_t     lower     = m_traceSorted.size() % 2 ? upper : m_traceSorted[middle - 1];
            const int64_t     median    = lower + ( upper - lower ) / 2;
            const long double threshold = m_summary.threshold * 1e9L;
            const long double residual  = static_cast<long double>( interval ) - median;

            if( threshold > 0 && ( residual > threshold || residual < -threshold ) )
            {
                const uint64_t id = ++m_traceStatus.detected;

                if( m_traceReady.size() + m_tracePending.size() >= m_traceStatus.capacity )
                {
                    ++m_traceStatus.dropped;
                }
                else
                {
                    TraceRecord record;
                    record.id                = id;
                    record.direction         = residual > 0 ? TraceDirection::late : TraceDirection::early;
                    record.medianNs          = median;
                    record.baselineIntervals = m_traceSorted.size();
                    record.intervalNs        = interval;
                    record.thresholdNs =
                        static_cast<int64_t>( std::min( threshold, static_cast<long double>( INT64_MAX ) ) );
                    record.bracketLimitNs = m_bracketNs;
                    record.events[2]      = event;
                    record.present        = 1u << 2;

                    for( size_t index = 0; index < m_traceNeighborCount; ++index )
                    {
                        const size_t destination   = 2 - m_traceNeighborCount + index;
                        record.events[destination] = m_traceNeighbors[index];
                        record.present |= 1u << destination;
                    }

                    int64_t firstBracket = 0, nextBracket = 0;
                    record.qualified = !previous->invalidMarkers && !event.invalidMarkers &&
                                       timestamp( previous->writetime ) > 0 && timestamp( event.writetime ) > 0 &&
                                       m_bracketNs > 0 && duration( *previous, Metric::writeBracket, firstBracket ) &&
                                       duration( event, Metric::writeBracket, nextBracket ) &&
                                       firstBracket <= m_bracketNs && nextBracket <= m_bracketNs;
                    m_tracePending.push_back( record );
                }
            }
        }
        else
        {
            ++m_traceStatus.unclassified;
        }

        if( m_traceIntervalCount == m_traceIntervals.size() )
        {
            tracePopInterval();
        }

        const size_t destination      = ( m_traceIntervalHead + m_traceIntervalCount ) % m_traceIntervals.size();
        m_traceIntervals[destination] = { interval, written };
        ++m_traceIntervalCount;
        m_traceSorted.insert( std::lower_bound( m_traceSorted.begin(), m_traceSorted.end(), interval ), interval );
    }

    if( m_traceNeighborCount == m_traceNeighbors.size() )
    {
        m_traceNeighbors[0] = m_traceNeighbors[1];
        m_traceNeighbors[1] = event;
    }
    else
    {
        m_traceNeighbors[m_traceNeighborCount++] = event;
    }

    m_traceStatus.pending = m_tracePending.size();
}

inline void frameGrabberTiming::update( int64_t nowMono, int64_t nowReal )
{
    using namespace fgTiming;
    ++m_summary.sequence;
    m_summary.nowNs = nowMono;

    if( !m_enabled )
    {
        return;
    }

    const int64_t cutoff  = nowMono >= m_windowNs ? nowMono - m_windowNs : 0;
    uint64_t      session = m_session.load( std::memory_order_acquire );

    const size_t controls = m_controls.queued();

    for( size_t index = 0; index < controls; ++index )
    {
        ControlEvent event;

        if( !m_controls.pop( event ) )
        {
            break;
        }

        ++m_total.controls;
        m_total.controlFailed += !event.success;

        if( event.entered < 0 || event.completed < event.entered )
        {
            ++m_total.controlInvalid;
            continue;
        }

        m_controlWindow.push_back( event );

        if( m_controlWindow.size() > m_queue.capacity() )
        {
            m_controlWindow.pop_front();
        }
    }

    // A frame captured after the preceding evaluation time was retained for this update.
    for( const auto &event : m_window )
    {
        traceEvent( event, nowMono );
    }

    if( session != m_summary.session )
    {
        m_window.clear();
        m_summary.session = session;
        m_everEvent       = false;
    }

    const size_t frames = m_queue.queued();

    for( size_t index = 0; index < frames; ++index )
    {
        Event event;

        if( !m_queue.pop( event ) )
        {
            break;
        }

        count( event );

        event.anchorNs = -1;

        for( size_t marker = 0; marker < markerIndex( Marker::count ); ++marker )
        {
            if( event.valid & ( 1u << marker ) )
            {
                event.anchorNs = std::max( event.anchorNs, event.markers[marker] );
            }
        }

        // Selection precedes summary retention and expiry, including older queued epochs.
        traceEvent( event, nowMono );

        if( event.session > m_summary.session )
        {
            m_window.clear();
            m_summary.session = event.session;
            m_everEvent       = false;
        }

        if( event.session != m_summary.session )
        {
            continue;
        }

        if( event.anchorNs < 0 )
        {
            const int64_t write = timestamp( event.writetime );

            if( event.outputValid && write > 0 && nowReal > 0 && nowMono >= 0 &&
                ( nowReal >= write ? nowReal - write <= nowMono
                                   : write - nowReal <= std::numeric_limits<int64_t>::max() - nowMono ) )
            {
                // A frame arriving after captured evaluation time stays retained for the next update.
                event.anchorNs = nowReal >= write ? nowMono - ( nowReal - write ) : nowMono + ( write - nowReal );
            }
            else
            {
                event.anchorNs = nowMono;

                if( event.outputValid && write > 0 )
                {
                    ++m_total.invalid;
                }
            }
        }

        m_window.push_back( event );

        if( m_window.size() > m_queue.capacity() )
        {
            m_window.pop_front();
        }

        m_everEvent = true;
    }

    session = m_session.load( std::memory_order_acquire );

    if( session > m_summary.session )
    {
        m_window.clear();
        m_summary.session = session;
        m_everEvent       = false;
    }

    if( m_traceNeighborCount && m_traceNeighbors[m_traceNeighborCount - 1].session < session )
    {
        ++m_traceStatus.discontinuities;
        traceBoundary( TraceEnd::boundary, nowMono );
    }
    else if( m_traceNeighborCount && m_traceNeighbors[m_traceNeighborCount - 1].anchorNs < cutoff )
    {
        traceBoundary( TraceEnd::expired, nowMono );
    }

    m_traceStatus.eventDropped   = m_queue.dropped();
    m_traceStatus.controlDropped = m_controls.dropped();

    m_window.erase( std::remove_if( m_window.begin(),
                                    m_window.end(),
                                    [=]( const Event &event )
                                    {
                                        return event.anchorNs < cutoff;
                                    } ),
                    m_window.end() );
    m_controlWindow.erase( std::remove_if( m_controlWindow.begin(),
                                           m_controlWindow.end(),
                                           [=]( const ControlEvent &event )
                                           {
                                               return event.completed < cutoff;
                                           } ),
                           m_controlWindow.end() );
    m_summary.queued        = m_queue.queued();
    m_summary.controlQueued = m_controls.queued();
    m_total.dropped         = m_queue.dropped();
    m_total.controlDropped  = m_controls.dropped();
    m_summary.total         = m_total;

    for( auto field : counterFields() )
    {
        m_summary.delta.*field = m_total.*field - m_previous.*field;
    }

    m_previous = m_total;
    calculate( nowMono );
}

inline int64_t frameGrabberTiming::serialOverlap( int64_t entered, int64_t completed ) const
{
    if( entered < 0 || completed < entered )
    {
        return -1;
    }

    int64_t total       = 0;
    int64_t previousEnd = entered;

    for( const auto &event : m_controlWindow )
    {
        if( event.completed > m_summary.nowNs )
        {
            continue;
        }

        const int64_t first = std::max( std::max( entered, previousEnd ), event.entered );
        const int64_t last  = std::min( completed, event.completed );

        if( last > first )
        {
            total += last - first;
            previousEnd = last;
        }
    }

    return total;
}

inline void frameGrabberTiming::calculate( int64_t nowNs )
{
    using namespace fgTiming;
    constexpr size_t                       count = metricIndex( Metric::count );
    std::array<timing::Accumulator, count> raw{}, late{}, early{};
    timing::Accumulator                    serial;
    timing::Accumulator                    lateReturn, earlyReturn, lateWrite, earlyWrite, lateTail, earlyTail;
    m_summary.metrics     = {};
    m_summary.lateChange  = {};
    m_summary.earlyChange = {};
    m_summary.newestNs    = -1;
    m_summary.span        = 0;
    m_summary.age         = -1;
    m_summary.fgValid     = false;
    m_summary.median      = 0;
    m_summary.intervals = m_summary.qualified = m_summary.ambiguous = m_summary.late = m_summary.early = 0;
    m_summary.serialOverlaps                                                                           = 0;
    m_summary.streamInode = m_summary.lastFrame = m_summary.lastCounter = 0;
    m_summary.mode = m_summary.binX = m_summary.binY = 0;
    m_intervals.clear();
    const Event *previous          = nullptr;
    const Event *previousPublished = nullptr;
    int64_t      oldest            = std::numeric_limits<int64_t>::max();

    for( const auto &control : m_controlWindow )
    {
        if( control.completed > nowNs )
        {
            continue;
        }

        serial.add( ( control.completed - control.entered ) * 1e-9, control.entered, control.completed );
    }

    m_summary.serialDuration = serial.stats();

    for( const auto &event : m_window )
    {
        if( event.anchorNs > nowNs )
        {
            continue;
        }

        oldest             = std::min( oldest, event.anchorNs );
        m_summary.newestNs = std::max( m_summary.newestNs, event.anchorNs );
        if( event.outputValid )
        {
            m_summary.streamInode = event.streamInode;
            m_summary.lastCounter = event.outputCounter;
        }

        if( event.frameValid )
        {
            m_summary.lastFrame = event.frameNumber;
        }
        m_summary.mode            = event.mode;
        m_summary.binX            = event.binX;
        m_summary.binY            = event.binY;
        const int64_t acquisition = timestamp( event.atime );
        const int64_t publication = timestamp( event.writetime );
        const bool    published   = event.outputValid;

        if( published && acquisition > 0 && publication > 0 )
        {
            raw[metricIndex( Metric::acquisitionToWrite )].add(
                ( publication - acquisition ) * 1e-9, event.anchorNs, event.anchorNs );
        }

        for( size_t index = metricIndex( Metric::wait ); index < metricIndex( Metric::count ); ++index )
        {
            int64_t value = 0;

            if( duration( event, static_cast<Metric>( index ), value ) )
            {
                raw[index].add( value * 1e-9, event.anchorNs, event.anchorNs );
            }
        }

        const int64_t overlap =
            serialOverlap( endpoint( event, Marker::waitEnter ), endpoint( event, Marker::writeStampDone ) );

        if( overlap >= 0 )
        {
            raw[metricIndex( Metric::serialOverlap )].add( overlap * 1e-9, event.anchorNs, event.anchorNs );
            m_summary.serialOverlaps += overlap > 0;
        }

        if( published )
        {
            if( previousPublished && previousPublished->session == event.session &&
                previousPublished->streamInode == event.streamInode &&
                event.outputCounter > previousPublished->outputCounter &&
                event.outputCounter == previousPublished->outputCounter + 1 )
            {
                const int64_t priorAcquisition = timestamp( previousPublished->atime );
                const int64_t priorPublication = timestamp( previousPublished->writetime );

                if( acquisition >= priorAcquisition && priorAcquisition > 0 )
                {
                    raw[metricIndex( Metric::acquisitionInterval )].add(
                        ( acquisition - priorAcquisition ) * 1e-9, previousPublished->anchorNs, event.anchorNs );
                }

                if( publication >= priorPublication && priorPublication > 0 )
                {
                    raw[metricIndex( Metric::publicationInterval )].add(
                        ( publication - priorPublication ) * 1e-9, previousPublished->anchorNs, event.anchorNs );
                }
            }

            previousPublished = &event;
        }

        if( previous && consecutive( *previous, event ) )
        {
            const int64_t returned    = endpoint( event, Marker::waitReturn );
            const int64_t priorReturn = endpoint( *previous, Marker::waitReturn );

            if( priorReturn >= 0 && returned > priorReturn )
            {
                raw[metricIndex( Metric::waitReturnInterval )].add(
                    ( returned - priorReturn ) * 1e-9, previous->anchorNs, event.anchorNs );
            }

            const int64_t written      = endpoint( event, Marker::writeStampDone );
            const int64_t priorWritten = endpoint( *previous, Marker::writeStampDone );

            if( priorWritten >= 0 && written > priorWritten )
            {
                raw[metricIndex( Metric::writeEndpointInterval )].add(
                    ( written - priorWritten ) * 1e-9, previous->anchorNs, event.anchorNs );
                m_intervals.push_back( written - priorWritten );
            }

            const int64_t entered = endpoint( event, Marker::waitEnter );

            if( priorWritten >= 0 && entered >= priorWritten )
            {
                raw[metricIndex( Metric::previousTail )].add(
                    ( entered - priorWritten ) * 1e-9, previous->anchorNs, event.anchorNs );
            }
        }

        previous = &event;
    }

    m_summary.intervals = m_intervals.size();

    if( !m_intervals.empty() )
    {
        auto middle = m_intervals.begin() + m_intervals.size() / 2;
        std::nth_element( m_intervals.begin(), middle, m_intervals.end() );
        long double median = *middle;

        if( m_intervals.size() % 2 == 0 )
        {
            const int64_t lower = *std::max_element( m_intervals.begin(), middle );
            median              = lower + ( median - lower ) / 2;
        }

        m_summary.median = static_cast<double>( median * 1e-9L );
    }

    previous = nullptr;

    for( const auto &event : m_window )
    {
        if( event.anchorNs > nowNs )
        {
            continue;
        }

        if( previous && consecutive( *previous, event ) )
        {
            const int64_t a = endpoint( *previous, Marker::writeStampDone );
            const int64_t b = endpoint( event, Marker::writeStampDone );

            if( a >= 0 && b > a )
            {
                int64_t    firstBracket = 0, nextBracket = 0;
                const bool qualified = !previous->invalidMarkers && !event.invalidMarkers &&
                                       timestamp( previous->writetime ) > 0 && timestamp( event.writetime ) > 0 &&
                                       m_bracketNs > 0 && duration( *previous, Metric::writeBracket, firstBracket ) &&
                                       duration( event, Metric::writeBracket, nextBracket ) &&
                                       firstBracket <= m_bracketNs && nextBracket <= m_bracketNs;

                if( !qualified )
                {
                    ++m_summary.ambiguous;
                }
                else
                {
                    ++m_summary.qualified;
                    const double residual = ( b - a ) * 1e-9 - m_summary.median;
                    const bool   isLate   = m_summary.threshold > 0 && residual > m_summary.threshold;
                    const bool   isEarly  = m_summary.threshold > 0 && residual < -m_summary.threshold;
                    m_summary.late += isLate;
                    m_summary.early += isEarly;

                    if( isLate || isEarly )
                    {
                        auto &writePopulation = isLate ? lateWrite : earlyWrite;
                        writePopulation.add( ( b - a ) * 1e-9, previous->anchorNs, event.anchorNs );
                        const int64_t returned    = endpoint( event, Marker::waitReturn );
                        const int64_t priorReturn = endpoint( *previous, Marker::waitReturn );

                        if( priorReturn >= 0 && returned > priorReturn )
                        {
                            auto &returnPopulation = isLate ? lateReturn : earlyReturn;
                            returnPopulation.add(
                                ( returned - priorReturn ) * 1e-9, previous->anchorNs, event.anchorNs );
                        }

                        const int64_t entered = endpoint( event, Marker::waitEnter );

                        if( entered >= a )
                        {
                            auto &tailPopulation = isLate ? lateTail : earlyTail;
                            tailPopulation.add( ( entered - a ) * 1e-9, previous->anchorNs, event.anchorNs );
                        }
                    }

                    for( size_t index = metricIndex( Metric::wait ); index < metricIndex( Metric::count ); ++index )
                    {
                        int64_t first = 0, second = 0;

                        if( ( isLate || isEarly ) && duration( *previous, static_cast<Metric>( index ), first ) &&
                            duration( event, static_cast<Metric>( index ), second ) )
                        {
                            auto &population = isLate ? late[index] : early[index];
                            population.add( ( second - first ) * 1e-9, previous->anchorNs, event.anchorNs );
                        }
                    }
                }
            }
        }

        previous = &event;
    }

    for( size_t index = 0; index < count; ++index )
    {
        m_summary.metrics[index]     = raw[index].stats();
        m_summary.lateChange[index]  = late[index].stats();
        m_summary.earlyChange[index] = early[index].stats();
    }

    m_summary.lateWaitReturnInterval     = lateReturn.stats();
    m_summary.earlyWaitReturnInterval    = earlyReturn.stats();
    m_summary.lateWriteEndpointInterval  = lateWrite.stats();
    m_summary.earlyWriteEndpointInterval = earlyWrite.stats();
    m_summary.latePreviousTail           = lateTail.stats();
    m_summary.earlyPreviousTail          = earlyTail.stats();

    if( m_summary.newestNs >= 0 )
    {
        m_summary.age  = ( nowNs - m_summary.newestNs ) * 1e-9;
        m_summary.span = ( m_summary.newestNs - oldest ) * 1e-9;
    }

    m_summary.fgValid = m_summary.metrics[metricIndex( Metric::acquisitionInterval )].valid &&
                        m_summary.metrics[metricIndex( Metric::publicationInterval )].valid &&
                        m_summary.metrics[metricIndex( Metric::acquisitionToWrite )].valid;
    m_summary.state = m_summary.fgValid                 ? Availability::valid
                      : m_window.empty() && m_everEvent ? Availability::stale
                      : m_everEvent                     ? Availability::unavailable
                                                        : Availability::warmingUp;
}

inline const fgTiming::Summary &frameGrabberTiming::summary() const
{
    return m_summary;
}

inline const std::vector<fgTiming::TraceRecord> &frameGrabberTiming::traceRecords() const
{
    return m_traceReady;
}

inline const fgTiming::TraceStatus &frameGrabberTiming::traceStatus() const
{
    return m_traceStatus;
}

inline void frameGrabberTiming::clearTraceRecords()
{
    m_traceReady.clear();
    m_traceStatus.ready = 0;
}
} // namespace dev
} // namespace app
} // namespace MagAOX
#endif
