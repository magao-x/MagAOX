/** \file native_observer.cpp
 * \brief Read-only native snapshots, bounded ADC forwarding and exclusive private observation recording.
 */
#include "native_observer.hpp"
#include <ImageStreamIO/ImageStreamIO.h>
#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace DDSPC
{
namespace
{
/// Capture actual local monotonic nanoseconds; an unavailable clock is never invented.
std::int64_t mono_ns()
{
    timespec t{};
    if( clock_gettime( CLOCK_MONOTONIC, &t ) != 0 )
        throw std::runtime_error( "Native observation monotonic clock failed" );
    return std::int64_t( t.tv_sec ) * 1000000000 + t.tv_nsec;
}
/// Preserve malformed native timestamps as unavailable.
std::int64_t raw_ns( const timespec &t /**< [in] Original source metadata. */ )
{
    if( t.tv_sec < 0 || t.tv_nsec < 0 || t.tv_nsec >= 1000000000 ||
        t.tv_sec > ( std::numeric_limits<std::int64_t>::max() - t.tv_nsec ) / 1000000000 )
        return -1;
    return std::int64_t( t.tv_sec ) * 1000000000 + t.tv_nsec;
}
/// Attach a bounded local clock bracket only for an explicitly declared source clock.
void clock_context( NativeObservation &r /**< [in,out] Owned source record. */,
                    ObservationClock   domain /**< [in] Explicit metadata publication clock. */ )
{
    const auto before = mono_ns();
    timespec   t{};
    int        result = -1;
    if( domain == ObservationClock::Realtime )
        result = clock_gettime( CLOCK_REALTIME, &t );
    if( domain == ObservationClock::Monotonic )
        result = clock_gettime( CLOCK_MONOTONIC, &t );
#ifdef CLOCK_TAI
    if( domain == ObservationClock::Tai )
        result = clock_gettime( CLOCK_TAI, &t );
#endif
    const auto after   = mono_ns();
    r.m_clockBracketNs = after - before;
    r.m_clockNowNs     = result == 0 ? raw_ns( t ) : -1;
    r.m_receivedNs = r.m_availableNs = after;
}
/// Owned metadata surrounding a source copy; original allocation geometry is never expanded.
struct NativeHeader
{
    std::uint64_t m_counter{ 0 }, m_counter1{ 0 }, m_index{ 0 },
        m_inode{ 0 }; ///< Uninterpreted producer counters/identity.
    std::uint32_t m_axes{ 0 }, m_width{ 0 }, m_height{ 0 }, m_depth{ 0 },
        m_datatype{ 0 };                              ///< Original extent and format.
    int          m_location{ 0 }, m_sem{ 0 };         ///< CPU accessibility and producer liveness.
    bool         m_writing{ true };                   ///< Producer update in progress.
    std::int64_t m_acquired{ -1 }, m_published{ -1 }; ///< Raw source clock values.
    /// Compare the complete bracket without using struct padding.
    bool operator==( const NativeHeader &other /**< [in] Closing metadata copy. */ ) const = default;
};
/// Read an ImageStreamIO metadata identity when supported; never substitute a filesystem inode.
template <typename Metadata>
std::uint64_t image_inode( const Metadata &md /**< [in] Native metadata view. */ )
{
    if constexpr( requires { md.inode; } )
        return std::uint64_t( md.inode );
    else
        return 0;
}
/// Existing-only mapping, with no semaphore registration, publisher API or hardware handle.
class NativeInput
{
    IMAGE         m_image{}; ///< Borrowed producer allocation, mapped only while its backing identity remains current.
    std::string   m_path;    ///< Existing backing path; never created, unlinked or replaced.
    bool          m_open{ false }; ///< Sole mapping owner.
    std::uint64_t m_device{ 0 }, m_inode{ 0 }, m_generation{ 0 },
        m_counter{ 0 };          ///< Observer attachment/counter progression.
    NativeHeader m_allocation;   ///< Immutable original layout bounds.
    std::int64_t m_retryNs{ 0 }; ///< Per-source reconnect backoff, without blocking other inputs.

    /// Copy native layout and raw timestamps around a bounded payload copy.
    NativeHeader header() const;

  public:
    /// Close only this observer's attachment.
    ~NativeInput();

    /// Copy an event/sample (1), duplicate (0), absent (-4), busy (-1), race (-2) or incompatible layout (-3).
    int sample( const ObservationStream          &config /**< [in] Existing source and declared clocks. */,
                ObservationSource                 source /**< [in] Fixed input role. */,
                const std::array<std::size_t, 2> &indices /**< [in] Selected modal entries. */,
                std::int64_t                      nowNs /**< [in] Observer monotonic poll time. */,
                NativeObservation                &record /**< [out] Owned event or stable sample. */,
                std::span<float> payload /**< [out] Optional bounded complete DM payload workspace. */ );

    /// Close without touching producer counters, pixels, timestamps or stream lifetime.
    void close();
};
NativeHeader NativeInput::header() const
{
    NativeHeader h;
    if( !m_open || !m_image.md )
        return h;
    const volatile auto *md = m_image.md;
    h.m_writing             = md->write != 0;
    std::atomic_thread_fence( std::memory_order_seq_cst );
    h.m_axes      = md->naxis;
    h.m_width     = md->size[0];
    h.m_height    = h.m_axes > 1 ? md->size[1] : 1;
    h.m_depth     = h.m_axes > 2 ? md->size[2] : 1;
    h.m_datatype  = md->datatype;
    h.m_location  = md->location;
    h.m_sem       = md->sem;
    h.m_counter   = md->cnt0;
    h.m_counter1  = md->cnt1;
    h.m_index     = h.m_axes > 2 ? h.m_counter1 : 0;
    h.m_inode     = image_inode( *md );
    h.m_acquired  = raw_ns( timespec{ md->atime.tv_sec, md->atime.tv_nsec } );
    h.m_published = raw_ns( timespec{ md->writetime.tv_sec, md->writetime.tv_nsec } );
    std::atomic_thread_fence( std::memory_order_seq_cst );
    h.m_writing = h.m_writing || md->write != 0;
    return h;
}
NativeInput::~NativeInput()
{
    close();
}
void NativeInput::close()
{
    if( m_open )
        ImageStreamIO_closeIm( &m_image );
    m_image   = {};
    m_open    = false;
    m_counter = 0;
}
int NativeInput::sample( const ObservationStream          &c,
                         ObservationSource                 source,
                         const std::array<std::size_t, 2> &indices,
                         std::int64_t                      now,
                         NativeObservation                &r,
                         std::span<float>                  payload )
{
    r.m_source     = source;
    r.m_receivedNs = r.m_availableNs = now;
    struct stat st{};
    if( m_open && ( stat( m_path.c_str(), &st ) != 0 || std::uint64_t( st.st_dev ) != m_device ||
                    std::uint64_t( st.st_ino ) != m_inode || !m_image.md || m_image.md->sem <= 0 ) )
    {
        r.m_event      = ObservationEvent::Disconnected;
        r.m_device     = m_device;
        r.m_fileInode  = m_inode;
        r.m_generation = m_generation;
        close();
        m_retryNs = now + 10000000;
        return 1;
    }
    if( !m_open )
    {
        if( c.m_name.empty() || now < m_retryNs )
            return -4;
        m_retryNs = now + 10000000;
        char        path[1024];
        struct stat after{};
        if( ImageStreamIO_filename( path, sizeof( path ), c.m_name.c_str() ) != 0 || stat( path, &st ) != 0 ||
            ImageStreamIO_openIm( &m_image, c.m_name.c_str() ) != 0 )
            return -4;
        m_open = true;
        if( stat( path, &after ) != 0 || st.st_dev != after.st_dev || st.st_ino != after.st_ino || !m_image.md ||
            !m_image.array.raw )
        {
            close();
            return -4;
        }
        m_path   = path;
        m_device = st.st_dev;
        m_inode  = st.st_ino;
        ++m_generation;
        m_allocation   = header();
        r.m_event      = ObservationEvent::Attached;
        r.m_generation = m_generation;
        r.m_device     = m_device;
        r.m_fileInode  = m_inode;
        r.m_imageInode = m_allocation.m_inode;
        r.m_width      = m_allocation.m_width;
        r.m_height     = m_allocation.m_height;
        r.m_depth      = m_allocation.m_depth;
        r.m_datatype   = m_allocation.m_datatype;
        return 1;
    }
    const auto h = header();
    if( h.m_writing )
        return -1;
    const auto count = std::uint64_t( h.m_width ) * h.m_height;
    if( h.m_axes < 1 || h.m_axes > 3 || h.m_location != -1 || !h.m_width || !h.m_height || !h.m_depth ||
        h.m_depth > 65536 || h.m_index >= h.m_depth || h.m_width != m_allocation.m_width ||
        h.m_height != m_allocation.m_height || h.m_depth != m_allocation.m_depth || h.m_axes != m_allocation.m_axes ||
        h.m_datatype != m_allocation.m_datatype || h.m_counter == std::numeric_limits<std::uint64_t>::max() )
        return -3;
    const bool endpoint = source == ObservationSource::Completion || source == ObservationSource::Sdk;
    if( endpoint && ( h.m_datatype != _DATATYPE_UINT64 || count != 16 || h.m_depth != 1 ) )
        return -3;
    if( ( source == ObservationSource::Modal || source == ObservationSource::Dm ) &&
        ( h.m_datatype != _DATATYPE_FLOAT || count > 65536 ||
          ( source == ObservationSource::Modal &&
            ( h.m_height != 1 || std::max( indices[0], indices[1] ) >= count ) ) ) )
        return -3;
    if( source == ObservationSource::Camera &&
        ( count > 16777216 || ( h.m_datatype != _DATATYPE_UINT8 && h.m_datatype != _DATATYPE_UINT16 ) ) )
        return -3;
    if( !h.m_counter )
        return -4;
    if( h.m_counter == m_counter && h.m_inode == m_allocation.m_inode )
        return 0;
    r.m_device         = m_device;
    r.m_fileInode      = m_inode;
    r.m_imageInode     = h.m_inode;
    r.m_counter        = h.m_counter;
    r.m_counter1       = h.m_counter1;
    r.m_hasCounter1    = true;
    r.m_width          = h.m_width;
    r.m_height         = h.m_height;
    r.m_depth          = h.m_depth;
    r.m_datatype       = h.m_datatype;
    r.m_acquiredRawNs  = h.m_acquired;
    r.m_publishedRawNs = h.m_published;
    r.m_elements       = count;
    if( h.m_depth > 1 )
    {
        if( !m_image.atimearray || !m_image.writetimearray || !m_image.cntarray )
            return -3;
        if( m_image.cntarray[h.m_index] != h.m_counter )
            return -2;
        r.m_acquiredRawNs  = raw_ns( m_image.atimearray[h.m_index] );
        r.m_publishedRawNs = raw_ns( m_image.writetimearray[h.m_index] );
    }
    if( endpoint )
    {
        const auto *words = static_cast<const std::uint64_t *>( m_image.array.raw );
        for( std::size_t n = 0; n < 16; ++n )
            r.m_wire[n] = __atomic_load_n( words + n, __ATOMIC_ACQUIRE );
    }
    else if( source == ObservationSource::Modal || source == ObservationSource::Dm )
    {
        const auto   *values = static_cast<const float *>( m_image.array.raw ) + h.m_index * count;
        std::uint64_t hash   = 14695981039346656037ULL;
        long double   square = 0;
        r.m_min              = std::numeric_limits<double>::infinity();
        r.m_max              = -r.m_min;
        for( std::size_t n = 0; n < count; ++n )
        {
            float value;
            std::memcpy( &value, values + n, sizeof( value ) );
            if( source == ObservationSource::Dm && count <= payload.size() )
                payload[n] = value;
            std::uint32_t bits;
            std::memcpy( &bits, &value, sizeof( bits ) );
            hash = ( hash ^ bits ) * 1099511628211ULL;
            if( !std::isfinite( value ) )
                r.m_flags |= ObservationInvalidPayload;
            else
            {
                r.m_min = std::min( r.m_min, double( value ) );
                r.m_max = std::max( r.m_max, double( value ) );
                square += double( value ) * value;
            }
            for( std::size_t axis = 0; axis < 2; ++axis )
                if( n == ( source == ObservationSource::Modal ? indices[axis] : axis ) )
                    r.m_values[axis] = value;
        }
        r.m_rms         = std::sqrt( double( square / count ) );
        r.m_payloadHash = hash;
    }
    std::atomic_thread_fence( std::memory_order_seq_cst );
    if( header() != h || ( h.m_depth > 1 && ( m_image.cntarray[h.m_index] != h.m_counter ||
                                              raw_ns( m_image.atimearray[h.m_index] ) != r.m_acquiredRawNs ||
                                              raw_ns( m_image.writetimearray[h.m_index] ) != r.m_publishedRawNs ) ) )
        return -2;
    if( h.m_counter < m_counter || h.m_inode != m_allocation.m_inode )
        ++m_generation;
    m_allocation.m_inode = h.m_inode;
    m_counter            = h.m_counter;
    r.m_generation       = m_generation;
    clock_context( r, endpoint ? ObservationClock::Unknown : observation_clock( c.m_publicationClock ) );
    return 1;
}

/// Write one explicit fixed-size record; no raw C++ layout/padding is serialized.
bool write_record( FILE                    *file /**< [in,out] Private exclusive CSV. */,
                   const NativeObservation &r /**< [in] Owned evidence. */ )
{
    const int result = std::fprintf( file,
                                     "%u,%u,%u,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%u,%u,%u,%u,%lld,%lld,%lld,%lld,%lld,"
                                     "%lld,%lld,%lld,%lld,%llu,%llu,%.17g,%.17g,%u,%u,%.17g,%.17g,%.17g",
                                     unsigned( r.m_source ),
                                     unsigned( r.m_event ),
                                     r.m_flags,
                                     (unsigned long long)r.m_device,
                                     (unsigned long long)r.m_fileInode,
                                     (unsigned long long)r.m_imageInode,
                                     (unsigned long long)r.m_generation,
                                     (unsigned long long)r.m_segment,
                                     (unsigned long long)r.m_counter,
                                     (unsigned long long)r.m_missing,
                                     r.m_width,
                                     r.m_height,
                                     r.m_depth,
                                     r.m_datatype,
                                     (long long)r.m_acquiredRawNs,
                                     (long long)r.m_publishedRawNs,
                                     (long long)r.m_acquiredMonoNs,
                                     (long long)r.m_publishedMonoNs,
                                     (long long)r.m_receivedNs,
                                     (long long)r.m_availableNs,
                                     (long long)r.m_clockNowNs,
                                     (long long)r.m_clockBracketNs,
                                     (long long)r.m_publicationAgeNs,
                                     (unsigned long long)r.m_elements,
                                     (unsigned long long)r.m_payloadHash,
                                     r.m_values[0],
                                     r.m_values[1],
                                     unsigned( r.m_raw[0] ),
                                     unsigned( r.m_raw[1] ),
                                     r.m_min,
                                     r.m_max,
                                     r.m_rms );
    if( result < 0 )
        return false;
    for( auto word : r.m_wire )
        if( std::fprintf( file, ",%llu", (unsigned long long)word ) < 0 )
            return false;
    return std::fprintf( file, ",%u,%llu\n", unsigned( r.m_hasCounter1 ), (unsigned long long)r.m_counter1 ) >= 0;
}

/// Write complete original FLOAT command values with an explicit little-endian 17-word identity/time header.
bool write_command( FILE                    *file /**< [in,out] Exclusive command journal, never a native stream. */,
                    const NativeObservation &r /**< [in] Stable metadata, not an applied-input certificate. */,
                    std::span<const float>   values /**< [in] Complete original command image. */ )
{
    static_assert( std::endian::native == std::endian::little && sizeof( float ) == 4 &&
                   std::numeric_limits<float>::is_iec559 );
    const std::array<std::uint64_t, 17> header{ r.m_elements,
                                                r.m_device,
                                                r.m_fileInode,
                                                r.m_imageInode,
                                                r.m_generation,
                                                r.m_counter,
                                                std::uint64_t( r.m_acquiredRawNs ),
                                                std::uint64_t( r.m_publishedRawNs ),
                                                std::uint64_t( r.m_receivedNs ),
                                                std::uint64_t( r.m_availableNs ),
                                                r.m_width,
                                                r.m_height,
                                                r.m_depth,
                                                r.m_datatype,
                                                r.m_flags,
                                                r.m_payloadHash,
                                                r.m_counter1 };
    return std::fwrite( header.data(), sizeof( header ), 1, file ) == 1 &&
           std::fwrite( values.data(), sizeof( float ), values.size(), file ) == values.size();
}
/// Create a new private artifact via an already opened directory; never follow or replace an existing output.
FILE *exclusive_file( int         directory /**< [in] Private owned directory descriptor. */,
                      const char *name /**< [in] Fixed observer artifact basename. */ )
{
    const int fd = openat( directory, name, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0600 );
    if( fd < 0 )
        throw std::runtime_error( std::string( "Cannot exclusively create observation artifact: " ) + name );
    FILE *file = fdopen( fd, "w" );
    if( !file )
    {
        ::close( fd );
        throw std::runtime_error( "Cannot open observation file stream" );
    }
    return file;
}
} // namespace

struct NativeObserver::Impl
{
    NativeObservationConfig             m_config;   ///< Immutable source selections and private recording limits.
    NativeObservationCapture            m_capture;  ///< Input-worker-only diagnostics and bounded identity joins.
    mutable std::mutex                  m_mutex;    ///< Short snapshot lock, never held for source or file I/O.
    NativeObservationStatus             m_status;   ///< Ordinary-thread diagnostic copy.
    NativeObservationQueue              m_queue;    ///< Fixed SPSC recording slots; input never waits for file I/O.
    std::unique_ptr<NativeCommandQueue> m_commands; ///< Optional independent bounded complete-command queue.
    std::vector<float> m_commandReadPayload, m_commandWritePayload; ///< Independent preallocated worker workspaces.
    std::atomic<std::uint64_t> m_commandsWritten{ 0 }, m_commandOversize{ 0 }; ///< Payload results, not transport loss.
    std::atomic<std::uint64_t> m_written{ 0 }; ///< Records actually written, independent of accepted snapshots.
    std::atomic<bool>          m_running{ false }, m_reading{ false }, m_finished{ false },
        m_ioError{ false };         ///< Lifecycle and writer failure.
    std::thread m_worker, m_writer; ///< Joined before the borrowed ADC reader or queue can be released.
    AdcReader  *m_adc{ nullptr };   ///< Borrowed existing reader; the observer never owns/starts/stops it.
    FILE       *m_csv{ nullptr }, *m_summary{ nullptr },
        *m_commandFile{ nullptr };                 ///< Exclusive private artifacts, owned by the writer.
    std::array<char, 65536> m_fileBuffer{};        ///< Fixed stdio buffer, configured before the writer starts.
    std::array<char, 65536> m_commandFileBuffer{}; ///< Independent full-command output buffer.

    /// Allocate bounded storage without I/O or worker startup.
    explicit Impl( const NativeObservationConfig &config /**< [in] Validated startup settings. */ );
    /// Enqueue or count a lost journal record; never block acquisition.
    void enqueue( const NativeObservation &record /**< [in] Owned record. */ );
    /// Observe existing sources and forward only bounded ADC batches to the journal, never a learner.
    void run();
    /// Drain the record queue and finish a private summary; no native source or model access.
    void write();
    /// Return a coherent ordinary-thread snapshot.
    NativeObservationStatus snapshot() const;
};
NativeObserver::Impl::Impl( const NativeObservationConfig &config )
    : m_config( config ), m_capture( config ), m_queue( config.m_queueCapacity )
{
    if( config.m_recordDmCommands )
    {
        m_commands = std::make_unique<NativeCommandQueue>( config.m_commandQueueCapacity, config.m_commandMaxElements );
        m_commandReadPayload.resize( config.m_commandMaxElements );
        m_commandWritePayload.resize( config.m_commandMaxElements );
    }
}
void NativeObserver::Impl::enqueue( const NativeObservation &record )
{
    m_queue.push( record );
}
NativeObservationStatus NativeObserver::Impl::snapshot() const
{
    std::lock_guard<std::mutex> lock( m_mutex ); // mutex scope
    auto                        s = m_status;
    s.m_queueDrops                = m_queue.drops();
    s.m_written                   = m_written.load();
    s.m_finished                  = m_finished.load();
    s.m_ioError                   = m_ioError.load();
    s.m_commandsWritten           = m_commandsWritten.load();
    s.m_commandDrops              = m_commands ? m_commands->drops() : 0;
    s.m_commandOversize           = m_commandOversize.load();
    return s;
}
void NativeObserver::Impl::run()
{
    try
    {
        std::array<NativeInput, observationSources - 1> inputs;
        AdcStatus                                       latestAdc;
        AdcCursor                                       cursor;
        bool                                            adcConnected = false;
        std::array<AdcSample, 8>                        samples;
        auto                                           &payload = m_commandReadPayload;
        const auto                                      start   = mono_ns();
        const auto                                      end     = start + std::int64_t( m_config.m_durationSec * 1e9 );
        { //mutex scope
            std::lock_guard<std::mutex> lock( m_mutex );
            m_status.m_startedNs = start;
        }
        while( m_running.load( std::memory_order_acquire ) && !m_ioError.load() && mono_ns() < end )
        {
            const auto now = mono_ns();
            for( unsigned batch = 0; batch < 4; ++batch )
            {
                const auto copied = m_adc->copy_after( cursor, samples, now );
                latestAdc         = copied.m_status;
                cursor            = copied.m_next;
                if( adcConnected && latestAdc.m_state == AdcState::Disconnected )
                {
                    NativeObservation r;
                    r.m_source     = ObservationSource::Adc;
                    r.m_event      = ObservationEvent::Disconnected;
                    r.m_device     = cursor.m_device;
                    r.m_fileInode  = cursor.m_inode;
                    r.m_generation = cursor.m_generation;
                    r.m_receivedNs = r.m_availableNs = now;
                    enqueue( m_capture.accept( r ) );
                    adcConnected = false;
                }
                if( copied.m_discontinuity || copied.m_missing )
                {
                    NativeObservation r;
                    r.m_source     = ObservationSource::Adc;
                    r.m_event      = ObservationEvent::AdcBreak;
                    r.m_receivedNs = r.m_availableNs = now;
                    r.m_missing                      = copied.m_missing;
                    r.m_generation                   = cursor.m_generation;
                    r.m_segment                      = cursor.m_segment;
                    r.m_device                       = cursor.m_device;
                    r.m_fileInode                    = cursor.m_inode;
                    enqueue( m_capture.accept( r ) );
                }
                for( std::size_t n = 0; n < copied.m_count; ++n )
                {
                    adcConnected        = true;
                    const auto       &s = samples[n];
                    NativeObservation r;
                    r.m_source          = ObservationSource::Adc;
                    r.m_device          = copied.m_status.m_device;
                    r.m_fileInode       = copied.m_status.m_inode;
                    r.m_generation      = s.m_generation;
                    r.m_segment         = s.m_segment;
                    r.m_counter         = s.m_frame.m_counter;
                    r.m_acquiredRawNs   = s.m_frame.m_acquiredRealNs;
                    r.m_publishedRawNs  = s.m_frame.m_writtenRealNs;
                    r.m_acquiredMonoNs  = s.m_acquiredNs;
                    r.m_publishedMonoNs = s.m_publishedNs;
                    clock_context( r, ObservationClock::Realtime );
                    r.m_receivedNs =
                        s.m_receivedNs; // Keep worker receipt distinct from this frontend's actual availability.
                    r.m_values   = s.m_acceleration;
                    r.m_raw      = s.m_frame.m_raw;
                    r.m_elements = 2;
                    r.m_width    = 2;
                    r.m_height   = 1;
                    r.m_depth    = copied.m_status.m_source.m_depth;
                    r.m_datatype = _DATATYPE_UINT16;
                    enqueue( m_capture.accept( r ) );
                }
                if( !copied.m_more )
                    break;
            }
            // Observe command identities before matching any completion/SDK records from this poll.
            for( const auto index : { 3u, 2u, 1u, 4u, 5u } )
            {
                if( m_config.m_streams[index].m_name.empty() )
                    continue;
                NativeObservation r;
                const auto        outcome = inputs[index - 1].sample( m_config.m_streams[index],
                                                                      ObservationSource( index ),
                                                                      m_config.m_modalIndices,
                                                                      mono_ns(),
                                                                      r,
                                                                      payload );
                if( outcome == 1 )
                {
                    r = m_capture.accept( r );
                    enqueue( r );
                    if( m_commands && r.m_source == ObservationSource::Dm && r.m_event == ObservationEvent::Sample )
                    {
                        if( r.m_elements > payload.size() )
                            ++m_commandOversize;
                        else
                            m_commands->push( r, std::span<const float>( payload.data(), r.m_elements ) );
                    }
                }
                else if( outcome != -4 )
                    m_capture.poll( ObservationSource( index ), outcome == -1, outcome == -2, outcome == -3 );
            }
            m_capture.tick( mono_ns() );
            { //mutex scope
                std::lock_guard<std::mutex> lock( m_mutex );
                const auto                  startNs = m_status.m_startedNs;
                m_status                            = m_capture.status();
                m_status.m_adcCapture               = latestAdc;
                m_status.m_startedNs                = startNs;
            }
            std::this_thread::sleep_for( std::chrono::microseconds( m_config.m_pollUs ) );
        }
    }
    catch( ... )
    {
        m_ioError.store( true );
    }
    { //mutex scope
        std::lock_guard<std::mutex> lock( m_mutex );
        try
        {
            m_status.m_stoppedNs = mono_ns();
        }
        catch( ... )
        {
            m_status.m_stoppedNs = -1;
        }
    }
    m_reading.store( false, std::memory_order_release );
}
void NativeObserver::Impl::write()
{
    auto &payload = m_commandWritePayload;
    // Drain independently so an empty metadata queue cannot strand a final command snapshot.
    const auto drainCommand = [&]()
    {
        NativeObservation command;
        if( !m_commands || !m_commands->pop( command, payload ) )
            return false;
        if( !m_ioError.load() &&
            write_command( m_commandFile, command, std::span<const float>( payload.data(), command.m_elements ) ) )
            ++m_commandsWritten;
        else
            m_ioError.store( true );
        return true;
    };
    while( true )
    {
        drainCommand();
        NativeObservation record;
        if( !m_queue.pop( record ) )
        {
            // Recheck after acquiring worker completion so a final publication cannot remain undrained.
            if( m_reading.load( std::memory_order_acquire ) )
            {
                std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
                continue;
            }
            if( !m_queue.pop( record ) )
            {
                while( drainCommand() )
                {
                }
                break;
            }
        }
        if( !m_ioError.load() && write_record( m_csv, record ) )
            ++m_written;
        else
            m_ioError.store( true );
    }
    const bool flushed = std::fflush( m_csv ) == 0;
    const bool closed  = std::fclose( m_csv ) == 0;
    if( !flushed || !closed )
        m_ioError.store( true );
    m_csv = nullptr;
    if( m_commandFile )
    {
        const bool payloadFlushed = std::fflush( m_commandFile ) == 0;
        const bool payloadClosed  = std::fclose( m_commandFile ) == 0;
        if( !payloadFlushed || !payloadClosed )
            m_ioError.store( true );
        m_commandFile = nullptr;
    }
    const auto s = snapshot();
    std::fprintf( m_summary,
                  "{\"schema\":1,\"observation_only\":true,\"training_enabled\":false,\"exposure_qualified\":false,"
                  "\"physical_application_qualified\":false,\"started_ns\":%lld,\"stopped_ns\":%lld,\"queue_drops\":%"
                  "llu,\"written\":%llu,\"adc_missing\":%llu,\"adc_breaks\":%llu,\"io_error\":%s,",
                  (long long)s.m_startedNs,
                  (long long)s.m_stoppedNs,
                  (unsigned long long)s.m_queueDrops,
                  (unsigned long long)s.m_written,
                  (unsigned long long)s.m_adcMissing,
                  (unsigned long long)s.m_adcBreaks,
                  s.m_ioError ? "true" : "false" );
    std::fprintf( m_summary,
                  "\"dm_commands_enabled\":%s,\"dm_commands_written\":%llu,\"dm_command_drops\":%llu,"
                  "\"dm_command_oversize\":%llu,\"dm_command_queue_capacity\":%zu,\"dm_command_max_elements\":%zu,",
                  m_config.m_recordDmCommands ? "true" : "false",
                  (unsigned long long)s.m_commandsWritten,
                  (unsigned long long)s.m_commandDrops,
                  (unsigned long long)s.m_commandOversize,
                  m_config.m_commandQueueCapacity,
                  m_config.m_commandMaxElements );
    const auto &a = s.m_adcCapture;
    std::fprintf( m_summary,
                  "\"duration_sec\":%.17g,\"poll_us\":%u,\"queue_capacity\":%zu,\"modal_indices\":[%zu,%zu],"
                  "\"adc_capture\":{\"state\":\"%s\",\"samples\":%llu,\"dropped\":%llu,\"invalid\":%llu,"
                  "\"timing_gaps\":%llu,\"restarts\":%llu,\"overwrites\":%llu,\"rate_hz\":%.17g},\"sources\":[",
                  m_config.m_durationSec,
                  m_config.m_pollUs,
                  m_config.m_queueCapacity,
                  m_config.m_modalIndices[0],
                  m_config.m_modalIndices[1],
                  adc_state( a.m_state ),
                  (unsigned long long)a.m_samples,
                  (unsigned long long)a.m_dropped,
                  (unsigned long long)a.m_invalid,
                  (unsigned long long)a.m_timingGaps,
                  (unsigned long long)a.m_restarts,
                  (unsigned long long)a.m_overwrites,
                  a.m_rateHz );
    for( std::size_t i = 0; i < observationSources; ++i )
    {
        const auto &v = s.m_sources[i];
        const auto &c = m_config.m_streams[i];
        std::fprintf(
            m_summary,
            "%s{\"role\":\"%s\",\"stream\":\"%s\",\"acquisition_clock\":\"%s\",\"publication_clock\":\"%s\","
            "\"samples\":%llu,\"counter_gaps\":%llu,\"restarts\":%llu,\"protocol_restarts\":%llu,\"clock_"
            "unknown\":%llu,\"clock_mismatches\":%llu,\"invalid\":%llu,\"protocol_invalid\":%llu,\"protocol_"
            "incomplete\":%llu,\"sequence_gaps\":%llu,\"association_failures\":%llu,\"metadata_"
            "associations\":%llu,\"busy\":%llu,\"races\":%llu,\"incompatible\":%llu,\"received_rate_hz\":%.17g}",
            i ? "," : "",
            observation_source_name( ObservationSource( i ) ),
            c.m_name.c_str(),
            c.m_acquisitionClock.c_str(),
            c.m_publicationClock.c_str(),
            (unsigned long long)v.m_samples,
            (unsigned long long)v.m_counterGaps,
            (unsigned long long)v.m_restarts,
            (unsigned long long)v.m_publisherRestarts,
            (unsigned long long)v.m_clockUnknown,
            (unsigned long long)v.m_clockMismatches,
            (unsigned long long)v.m_invalid,
            (unsigned long long)v.m_protocolInvalid,
            (unsigned long long)v.m_protocolIncomplete,
            (unsigned long long)v.m_sequenceGaps,
            (unsigned long long)v.m_associationFailures,
            (unsigned long long)v.m_associations,
            (unsigned long long)v.m_busy,
            (unsigned long long)v.m_races,
            (unsigned long long)v.m_incompatible,
            v.m_receivedRateHz );
    }
    const bool ended         = std::fprintf( m_summary, "]}\n" ) >= 0 && std::ferror( m_summary ) == 0;
    const bool summaryClosed = std::fclose( m_summary ) == 0;
    if( !ended || !summaryClosed )
        m_ioError.store( true );
    m_summary = nullptr;
    m_finished.store( true, std::memory_order_release );
}
NativeObserver::NativeObserver( const NativeObservationConfig &config ) : m_impl( std::make_unique<Impl>( config ) )
{
}
NativeObserver::~NativeObserver()
{
    stop();
}
void NativeObserver::start( AdcReader &adc )
{
    if( m_impl->m_worker.joinable() || m_impl->m_writer.joinable() )
        throw std::logic_error( "Observer already started" );
    const int   directory = open( m_impl->m_config.m_directory.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW );
    struct stat info{};
    if( directory < 0 || fstat( directory, &info ) != 0 || info.st_uid != getuid() )
    {
        if( directory >= 0 )
            ::close( directory );
        throw std::runtime_error( "Observation directory must exist and belong to this user" );
    }
    try
    {
        m_impl->m_csv     = exclusive_file( directory, "observations.csv" );
        m_impl->m_summary = exclusive_file( directory, "summary.json" );
        if( m_impl->m_commands )
        {
            m_impl->m_commandFile = exclusive_file( directory, "dm_commands.bin" );
            setvbuf(
                m_impl->m_commandFile, m_impl->m_commandFileBuffer.data(), _IOFBF, m_impl->m_commandFileBuffer.size() );
            if( std::fwrite( "DDDM0001", 8, 1, m_impl->m_commandFile ) != 1 )
                throw std::runtime_error( "Full-command journal header write failed" );
        }
    }
    catch( ... )
    {
        if( m_impl->m_csv )
        {
            std::fclose( m_impl->m_csv );
            m_impl->m_csv = nullptr;
        }
        if( m_impl->m_summary )
        {
            std::fclose( m_impl->m_summary );
            m_impl->m_summary = nullptr;
        }
        if( m_impl->m_commandFile )
        {
            std::fclose( m_impl->m_commandFile );
            m_impl->m_commandFile = nullptr;
        }
        ::close( directory );
        throw;
    }
    ::close( directory );
    setvbuf( m_impl->m_csv, m_impl->m_fileBuffer.data(), _IOFBF, m_impl->m_fileBuffer.size() );
    if( std::fprintf(
            m_impl->m_csv,
            "source,event,flags,device,file_inode,image_inode,generation,segment,counter,missing,width,height,depth,"
            "datatype,acquired_raw_ns,published_raw_ns,acquired_mono_ns,published_mono_ns,received_mono_ns,available_"
            "mono_ns,clock_now_ns,clock_bracket_ns,publication_age_ns,elements,payload_hash,value0,value1,raw0,raw1,"
            "min,max,rms,w0,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15,has_counter1,counter1\n" ) < 0 )
        throw std::runtime_error( "Observation CSV header write failed" );
    m_impl->m_adc = &adc;
    m_impl->m_running.store( true );
    m_impl->m_reading.store( true );
    try
    {
        m_impl->m_writer = std::thread( &Impl::write, m_impl.get() );
        m_impl->m_worker = std::thread( &Impl::run, m_impl.get() );
    }
    catch( ... )
    {
        m_impl->m_reading.store( false );
        stop();
        throw;
    }
}
void NativeObserver::stop()
{
    m_impl->m_running.store( false, std::memory_order_release );
    if( m_impl->m_worker.joinable() )
        m_impl->m_worker.join();
    if( m_impl->m_writer.joinable() )
        m_impl->m_writer.join();
    if( m_impl->m_csv )
    {
        std::fclose( m_impl->m_csv );
        m_impl->m_csv = nullptr;
    }
    if( m_impl->m_summary )
    {
        std::fclose( m_impl->m_summary );
        m_impl->m_summary = nullptr;
    }
    if( m_impl->m_commandFile )
    {
        std::fclose( m_impl->m_commandFile );
        m_impl->m_commandFile = nullptr;
    }
    m_impl->m_adc = nullptr;
}
NativeObservationStatus NativeObserver::status() const
{
    return m_impl->snapshot();
}
} // namespace DDSPC
