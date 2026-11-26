/** \file ocamFrameDelay.hpp
 * \brief Consumer-side OCAM cadence checks and rolling metadata snapshots.
 */
#ifndef ocamFrameDelay_hpp
#define ocamFrameDelay_hpp

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <deque>
#include <fcntl.h>
#include <filesystem>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

#include <mx/math/floatUtils.hpp>

#include "../../libMagAOX/common/frameGrabberTiming.hpp"

namespace MagAOX::app
{
/// Compare OCAM timing against its acquisition rate and retain compact chronological history.
class ocamFrameDelay
{
  public:
    static constexpr size_t   endpoints = 3;             ///< DMA, SDK return and stream write, in that order.
    static constexpr size_t   checks    = 2 * endpoints; ///< Single intervals first, followed by integrated spans.
    static constexpr unsigned allChecks = ( 1u << checks ) - 1; ///< Mask for every supported check.
    static constexpr std::array<const char *, checks> names{
        "atime", "arrival", "write", "atime_sum", "arrival_sum", "write_sum" }; ///< INDI element and warning labels.

    /// Fixed-width version 1 disk row. A missing arrival uses the preceding valid time as its retention anchor.
    struct Row
    {
        int64_t  atime;    ///< EDT timestamp, ns, or -1.
        int64_t  arrival;  ///< SDK-return monotonic time or the most recent valid retention anchor, ns.
        int64_t  write;    ///< Stream write timestamp, ns, or -1.
        uint64_t sequence; ///< Timing attempt sequence, including gaps caused by queue loss.
        uint64_t counter;  ///< Output stream counter, when attached.
        uint32_t frame;    ///< OCAM frame number, when attached.
        uint32_t flags;    ///< Outcome in bits 0–1, with frame, output, arrival and clock-step flags in bits 2–5.
    };

    /// Configuration and stream context that takes effect at the given attempt sequence.
    struct Epoch
    {
        uint64_t sequence; ///< First attempt observed with this context.
        uint64_t session;  ///< Framegrabber acquisition epoch.
        uint64_t inode;    ///< Identity of the output stream, zero while unknown.
        double   fps;      ///< Effective rate during this epoch, Hz.
        uint64_t mode;     ///< OCAM SDK mode tag.
        uint64_t binX;     ///< Software binning along X.
        uint64_t binY;     ///< Software binning along Y.
    };

    /// The latest ordinary-thread cadence measurements, independent of warning enablement.
    struct Reading
    {
        double                      fps{ 0 };     ///< Effective frame rate in Hz, zero if unavailable.
        size_t                      samples{ 0 }; ///< Number of adjacent intervals required for the integrated test.
        std::array<int64_t, checks> elapsed{ -1, -1, -1, -1, -1, -1 }; ///< Measured spans in ns, or -1.
        std::array<int64_t, checks> excess{};     ///< Signed excess in ns for checks with a valid measured span.
        unsigned                    latched{ 0 }; ///< Checks latched since the last successful reconfiguration.
    };

  private:
    using Chunk = std::vector<Row>; ///< Consumer-owned rows which become immutable when sealed for a snapshot.
    static constexpr size_t   chunkRows  = 4096;    ///< Maximum rows per allocation and writer batch.
    static constexpr unsigned hasFrame   = 1u << 2; ///< A raw OCAM frame number was supplied.
    static constexpr unsigned hasOutput  = 1u << 3; ///< An output counter was supplied.
    static constexpr unsigned hasArrival = 1u << 4; ///< Arrival is an actual clock reading.
    static constexpr unsigned clockStep  = 1u << 5; ///< Possible external clock step near this attempt.

    /// One bounded snapshot view into sealed metadata.
    struct Span
    {
        std::shared_ptr<const Chunk> chunk; ///< Keeps these immutable rows alive until written.
        size_t                       first; ///< First included row index.
        size_t                       last;  ///< One past the last included row.
    };

  public:
    /// Frozen metadata handed from the ordinary consumer to the file writer.
    struct Snapshot
    {
        std::deque<Span>   spans;            ///< Chronological views of shared chunks.
        std::vector<Epoch> epochs;           ///< Context for the rows, including the epoch at the start of the file.
        uint64_t           id{ 0 };          ///< Manual request identifier within this process.
        int64_t            cutoff{ 0 };      ///< Monotonic time the manual request was accepted, ns.
        int64_t            requestedAt{ 0 }; ///< Realtime request time for log correlation, ns.
        uint64_t           boundary{ 0 };    ///< Last attempt sequence covered at the acquisition boundary.
        uint64_t           dropped{ 0 };     ///< Total producer handoff losses since startup.
        uint64_t           trimmed{ 0 };     ///< Total capacity evictions since startup.
        size_t             rows{ 0 };        ///< Number of saved metadata rows.
        int64_t            duration{ 0 };    ///< Requested history length, ns.
        int64_t            coverage{ 0 };    ///< Monotonic time from earliest to latest saved row, ns.
    };

    /// Set the retention budget and warning threshold before consuming events.
    void configure( unsigned duration /**< [in] Retention duration in seconds. */,
                    double   maxFPS /**< [in] Maximum configured camera mode rate. */,
                    double   fraction /**< [in] Warning excess in units of one frame period. */ );

    /// Record one delivered event and return the checks that latched on it.
    unsigned observe( const fgTiming::Event &event /**< [in] Raw timing event. */,
                      unsigned               enabled /**< [in] Current warning enable mask. */ );

    /// Reset measurement continuity and latches at a successful restart.
    void restart( uint64_t sequence /**< [in] First new attempt sequence, or zero if no restart occurred. */ );

    /// Return consumer-owned measurements and latches.
    const Reading &reading() const;

    /// Return the configured threshold fraction of one period.
    double fraction() const;

    /// Drop history older than the requested interval and release expired chunks.
    void expire( int64_t now /**< [in] Monotonic retention reference, ns. */ );

    /// Seal and share the history preceding a manual request.
    void freeze( Snapshot &snapshot /**< [in,out] Manual request metadata and output snapshot. */ );

    /// Write and rotate one snapshot on a non-real-time thread. Throws on an I/O failure.
    static std::filesystem::path write( Snapshot                     snapshot /**< [in] Sealed, owned snapshot. */,
                                        const std::filesystem::path &directory /**< [in] Dump directory. */,
                                        const std::string &camera /**< [in] Instance name for naming and rotation. */ );

  private:
    std::deque<std::shared_ptr<Chunk>> m_history;    ///< Bounded chronological shared history chunks.
    std::shared_ptr<Chunk>             m_open;       ///< Mutable tail, sealed before exposing views to the writer.
    std::deque<Epoch>                  m_epochs;     ///< Context for the current history span.
    size_t                             m_first{ 0 }; ///< First retained row of the front chunk.
    size_t                             m_count{ 0 }; ///< Retained history row count.
    size_t   m_capacity{ 0 };   ///< Maximum retained rows including allowance for rejected attempts.
    size_t   m_maxSamples{ 0 }; ///< Largest integrated window supported by the configured history budget.
    uint64_t m_trimmed{ 0 };    ///< Capacity-evicted rows since startup.
    int64_t  m_duration{ 0 };   ///< Monotonic time retention budget in ns.
    int64_t  m_anchor{ 0 };     ///< Most recent observed valid monotonic arrival.
    double   m_fraction{ 1 };   ///< Warning threshold in units of frame periods.
    Reading  m_reading;         ///< Latest cadence measurements and warning latches.
    std::array<long double, checks> m_limits{};     ///< Inclusive span thresholds cached at the effective rate.
    std::array<long double, checks> m_nominal{};    ///< Nominal spans rounded to nanoseconds at the effective rate.
    fgTiming::Event                 m_previous;     ///< Prior observation for frame/stream continuity tests.
    uint64_t                        m_restart{ 0 }; ///< Most recent restart applied by the consumer.
    std::array<std::deque<int64_t>, endpoints> m_samples; ///< Latest contiguous timestamps, one deque per endpoint.

    /// Convert a valid normalized external timestamp to ns, returning -1 if unavailable.
    static int64_t timestamp( const timespec &stamp /**< [in] Original SDK or realtime timestamp. */ );

    /// Test adjacency in all frame identities and the known effective FPS.
    static bool adjacent( const fgTiming::Event &previous /**< [in] Earlier observation. */,
                          const fgTiming::Event &event /**< [in] Later observation. */ );

    /// Clear all three cadence baselines and invalidate the current intervals.
    void resetSamples();

    /// Remove the oldest retained row and any superseded epoch context.
    void pop();
};

inline void ocamFrameDelay::configure( unsigned duration, double maxFPS, double fraction )
{
    if( duration == 0 || !mx::math::isFinite( maxFPS ) || maxFPS <= 0 || !mx::math::isFinite( fraction ) ||
        fraction <= 0 || static_cast<long double>( duration ) * maxFPS * 1.1L >= std::numeric_limits<size_t>::max() )
    {
        throw std::invalid_argument( "Invalid frame delay duration, mode rate or warning fraction" );
    }

    m_duration   = static_cast<int64_t>( duration ) * 1000000000;
    m_capacity   = static_cast<size_t>( std::ceil( static_cast<long double>( duration ) * maxFPS * 1.1L ) );
    m_maxSamples = static_cast<size_t>( std::ceil( maxFPS ) );
    m_fraction   = fraction;
}

inline int64_t ocamFrameDelay::timestamp( const timespec &stamp )
{
    if( stamp.tv_sec < 0 || stamp.tv_nsec < 0 || stamp.tv_nsec >= 1000000000 ||
        ( stamp.tv_sec == 0 && stamp.tv_nsec == 0 ) ||
        stamp.tv_sec > ( std::numeric_limits<int64_t>::max() - stamp.tv_nsec ) / 1000000000 )
    {
        return -1;
    }

    return static_cast<int64_t>( stamp.tv_sec ) * 1000000000 + stamp.tv_nsec;
}

inline bool ocamFrameDelay::adjacent( const fgTiming::Event &previous, const fgTiming::Event &event )
{
    return previous.sequence != 0 && event.sequence == previous.sequence + 1 && event.session == previous.session &&
           event.fps == previous.fps && event.streamInode == previous.streamInode && previous.frameValid &&
           event.frameValid && previous.outputValid && event.outputValid &&
           event.outcome == fgTiming::Outcome::published && previous.outcome == fgTiming::Outcome::published &&
           event.outputCounter == previous.outputCounter + 1 &&
           static_cast<uint32_t>( event.frameNumber ) - static_cast<uint32_t>( previous.frameNumber ) == 1;
}

inline void ocamFrameDelay::resetSamples()
{
    for( auto &samples : m_samples )
    {
        samples.clear();
    }

    m_reading.elapsed.fill( -1 );
}

inline void ocamFrameDelay::restart( uint64_t sequence )
{
    if( sequence > m_restart )
    {
        m_restart  = sequence;
        m_previous = {};
        m_reading  = {};
        resetSamples();
    }
}

inline const ocamFrameDelay::Reading &ocamFrameDelay::reading() const
{
    return m_reading;
}

inline double ocamFrameDelay::fraction() const
{
    return m_fraction;
}

inline unsigned ocamFrameDelay::observe( const fgTiming::Event &event, unsigned enabled )
{
    restart( event.restart );
    const size_t marker  = fgTiming::markerIndex( fgTiming::Marker::waitReturn );
    const auto   arrival = event.valid & ( 1u << marker ) ? event.markers[marker] : event.arrivalNs;
    const std::array<int64_t, endpoints> stamps{ timestamp( event.atime ), arrival, timestamp( event.writetime ) };
    const unsigned                       alreadyLatched = m_reading.latched;
    const bool                           continuous     = adjacent( m_previous, event );
    const bool rateKnown    = mx::math::isFinite( event.fps ) && event.fps > 0 && event.fps <= m_maxSamples;
    const bool published    = event.outcome == fgTiming::Outcome::published && event.outputValid && event.frameValid;
    const bool rateChanged  = rateKnown && event.fps != m_reading.fps;
    bool       suspectClock = false;
    m_reading.fps           = rateKnown ? event.fps : 0;
    m_reading.samples       = rateKnown ? static_cast<size_t>( std::max( 1.0, std::round( event.fps ) ) ) : 0;
    m_reading.elapsed.fill( -1 );
    m_reading.excess.fill( 0 );

    if( !continuous )
    {
        resetSamples();
    }

    if( rateChanged )
    {
        const long double period = 1e9L / event.fps;

        for( size_t check = 0; check < checks; ++check )
        {
            const size_t intervals = check < endpoints ? 1 : m_reading.samples;
            m_limits[check]        = std::round( ( intervals + static_cast<long double>( m_fraction ) ) * period );
            m_nominal[check]       = std::round( intervals * period );
        }
    }

    if( rateKnown && published )
    {
        for( size_t endpoint = 0; endpoint < endpoints; ++endpoint )
        {
            auto      &samples = m_samples[endpoint];
            const auto stamp   = stamps[endpoint];

            if( stamp < 0 || ( !samples.empty() && stamp < samples.back() ) )
            {
                suspectClock |= stamp >= 0 && endpoint != 1;
                samples.clear();
            }

            if( stamp < 0 )
            {
                continue;
            }

            if( !samples.empty() )
            {
                m_reading.elapsed[endpoint] = stamp - samples.back();
            }

            if( samples.size() == m_reading.samples )
            {
                m_reading.elapsed[endpoints + endpoint] = stamp - samples.front();
                samples.pop_front();
            }

            samples.push_back( stamp );
        }

        const long double period = 1e9L / event.fps;

        for( size_t check = 0; check < checks; ++check )
        {
            if( m_reading.elapsed[check] < 0 )
            {
                continue;
            }

            const auto nominal =
                static_cast<int64_t>( std::min( m_nominal[check], static_cast<long double>( INT64_MAX ) ) );
            m_reading.excess[check] = m_reading.elapsed[check] - nominal;
            const unsigned bit      = 1u << check;

            if( ( enabled & bit ) && m_reading.elapsed[check] >= m_limits[check] )
            {
                m_reading.latched |= bit;
            }
        }

        const auto &elapsed = m_reading.elapsed;
        suspectClock |= elapsed[0] >= 0 && elapsed[1] >= 0 && elapsed[2] >= 0 && elapsed[0] - elapsed[1] >= period &&
                        elapsed[2] - elapsed[1] >= period && std::abs( elapsed[0] - elapsed[2] ) < period / 2;
    }

    const auto  inode = event.outputValid ? event.streamInode : ( m_epochs.empty() ? 0 : m_epochs.back().inode );
    const Epoch epoch{ event.sequence, event.session, inode, m_reading.fps, event.mode, event.binX, event.binY };

    if( m_epochs.empty() || m_epochs.back().session != epoch.session || m_epochs.back().inode != epoch.inode ||
        m_epochs.back().fps != epoch.fps || m_epochs.back().mode != epoch.mode || m_epochs.back().binX != epoch.binX ||
        m_epochs.back().binY != epoch.binY )
    {
        m_epochs.push_back( epoch );
    }

    if( !m_open || m_open->size() == chunkRows )
    {
        auto chunk = std::make_shared<Chunk>();
        chunk->reserve( chunkRows );
        m_history.push_back( chunk );
        m_open = std::move( chunk );
    }

    m_anchor         = std::max( m_anchor, arrival );
    const auto flags = static_cast<unsigned>( event.outcome ) | ( event.frameValid ? hasFrame : 0 ) |
                       ( event.outputValid ? hasOutput : 0 ) | ( arrival >= 0 ? hasArrival : 0 ) |
                       ( suspectClock ? clockStep : 0 );
    m_open->push_back( { stamps[0],
                         m_anchor,
                         stamps[2],
                         event.sequence,
                         event.outputCounter,
                         static_cast<uint32_t>( event.frameNumber ),
                         flags } );
    ++m_count;

    if( m_count > m_capacity )
    {
        pop();
        ++m_trimmed;
    }

    m_previous = event;
    return m_reading.latched & ~alreadyLatched;
}

inline void ocamFrameDelay::pop()
{
    ++m_first;
    --m_count;

    if( m_first == m_history.front()->size() )
    {
        if( m_open == m_history.front() )
        {
            m_open.reset();
        }

        m_history.pop_front();
        m_first = 0;
    }

    if( m_history.empty() )
    {
        m_epochs.clear();
        return;
    }

    const auto sequence = ( *m_history.front() )[m_first].sequence;

    while( m_epochs.size() > 1 && m_epochs[1].sequence <= sequence )
    {
        m_epochs.pop_front();
    }
}

inline void ocamFrameDelay::expire( int64_t now )
{
    if( now >= m_duration )
    {
        const auto oldest = now - m_duration;

        while( !m_history.empty() && ( *m_history.front() )[m_first].arrival < oldest )
        {
            pop();
        }
    }
}

inline void ocamFrameDelay::freeze( Snapshot &snapshot )
{
    m_open.reset();
    expire( snapshot.cutoff );
    snapshot.duration = m_duration;
    snapshot.trimmed  = m_trimmed;
    size_t first      = m_first;

    for( const auto &chunk : m_history )
    {
        const auto end =
            std::partition_point( chunk->begin() + first,
                                  chunk->end(),
                                  [&]( const Row &row )
                                  {
                                      return row.sequence <= snapshot.boundary && row.arrival <= snapshot.cutoff;
                                  } );
        const size_t last = end - chunk->begin();

        if( last > first )
        {
            snapshot.spans.push_back( { chunk, first, last } );
            snapshot.rows += last - first;
        }

        if( end != chunk->end() )
        {
            break;
        }

        first = 0;
    }

    if( !snapshot.spans.empty() )
    {
        const auto &head        = snapshot.spans.front();
        const auto &tail        = snapshot.spans.back();
        snapshot.coverage       = ( *tail.chunk )[tail.last - 1].arrival - ( *head.chunk )[head.first].arrival;
        const auto lastSequence = ( *tail.chunk )[tail.last - 1].sequence;

        for( const auto &epoch : m_epochs )
        {
            if( epoch.sequence > lastSequence )
            {
                break;
            }

            snapshot.epochs.push_back( epoch );
        }
    }
}

inline std::filesystem::path
ocamFrameDelay::write( Snapshot snapshot, const std::filesystem::path &directory, const std::string &camera )
{
    namespace fs = std::filesystem;
    static_assert( std::endian::native == std::endian::little );
    static_assert( sizeof( Row ) == 48 && sizeof( Epoch ) == 56 );
    const std::string prefix      = camera + "-frame-delay-";
    const auto        destination = directory / ( prefix + std::to_string( snapshot.requestedAt ) + "-" +
                                           std::to_string( getpid() ) + "-" + std::to_string( snapshot.id ) + ".bin" );
    fs::create_directories( directory );

    const std::string temp = destination.string() + ".tmp";
    // Match the normal app logger's fopen mode and let the process umask apply.
    const int fd = ::open( temp.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0666 );

    if( fd < 0 )
    {
        throw std::runtime_error( "Cannot create frame history file " + destination.string() );
    }

    FILE *file = fdopen( fd, "wb" );

    if( !file )
    {
        close( fd );
        unlink( temp.c_str() );
        throw std::runtime_error( "Cannot open frame history file " + destination.string() );
    }

    constexpr std::array<char, 8> magic{ 'O', 'C', 'A', 'M', 'F', 'D', '0', '1' };
    const std::array<uint64_t, 9> header{ snapshot.id,
                                          static_cast<uint64_t>( snapshot.cutoff ),
                                          static_cast<uint64_t>( snapshot.requestedAt ),
                                          snapshot.boundary,
                                          snapshot.dropped,
                                          snapshot.trimmed,
                                          snapshot.rows,
                                          snapshot.epochs.size(),
                                          static_cast<uint64_t>( snapshot.duration ) };
    bool                          ok = fwrite( magic.data(), 1, magic.size(), file ) == magic.size() &&
              fwrite( header.data(), sizeof( header[0] ), header.size(), file ) == header.size();

    if( ok && !snapshot.epochs.empty() )
    {
        ok = fwrite( snapshot.epochs.data(), sizeof( Epoch ), snapshot.epochs.size(), file ) == snapshot.epochs.size();
    }

    while( ok && !snapshot.spans.empty() )
    {
        const auto &span  = snapshot.spans.front();
        const auto  count = span.last - span.first;
        ok                = fwrite( span.chunk->data() + span.first, sizeof( Row ), count, file ) == count;
        snapshot.spans.pop_front();
    }

    if( fclose( file ) != 0 || !ok || rename( temp.c_str(), destination.c_str() ) != 0 )
    {
        unlink( temp.c_str() );
        throw std::runtime_error( "Cannot finish frame history file " + destination.string() );
    }

    std::vector<fs::path> files;

    for( const auto &entry : fs::directory_iterator( directory ) )
    {
        const auto &path = entry.path();

        if( entry.is_regular_file() && path.extension() == ".bin" && path.filename().string().starts_with( prefix ) )
        {
            files.push_back( path );
        }
    }

    std::sort( files.begin(), files.end() );

    for( size_t i = 0; i + 5 < files.size(); ++i )
    {
        fs::remove( files[i] );
    }

    return destination;
}
} //namespace MagAOX::app

#endif // ocamFrameDelay_hpp
