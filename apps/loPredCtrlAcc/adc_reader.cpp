/** \file adc_reader.cpp
 * \brief Existing-only ImageStreamIO ADC acquisition, adapted from accelFFCtrl's bounded transport.
 */
#include "adc_reader.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <limits>
#include <mutex>
#include <thread>
#include <semaphore.h>
#include <sys/stat.h>
#include <unistd.h>
#include <ImageStreamIO/ImageStreamIO.h>

namespace DDSPC
{
namespace
{

/// Read a realtime/monotonic bracket; a clock failure produces an invalid observation.
AdcClock read_clock()
{
    timespec before{}, real{}, after{};
    if( clock_gettime( CLOCK_MONOTONIC, &before ) != 0 || clock_gettime( CLOCK_REALTIME, &real ) != 0 ||
        clock_gettime( CLOCK_MONOTONIC, &after ) != 0 )
    {
        return {};
    }
    const auto first = std::int64_t( before.tv_sec ) * 1000000000 + before.tv_nsec;
    const auto last  = std::int64_t( after.tv_sec ) * 1000000000 + after.tv_nsec;
    return { std::int64_t( real.tv_sec ) * 1000000000 + real.tv_nsec,
             first + ( last - first ) / 2,
             ( last - first + 1 ) / 2 };
}

/// Convert native timestamps without overflow, preserving invalid metadata as -1.
std::int64_t timestamp_ns( const timespec &value /**< [in] Original producer time. */ )
{
    if( value.tv_sec < 0 || value.tv_nsec < 0 || value.tv_nsec >= 1000000000 ||
        value.tv_sec > ( std::numeric_limits<std::int64_t>::max() - value.tv_nsec ) / 1000000000 )
    {
        return -1;
    }
    return std::int64_t( value.tv_sec ) * 1000000000 + value.tv_nsec;
}

/// Sole native input owner; copied from accelFFCtrl's stream pattern without any estimator/output dependency.
class NativeAdcSource : public AdcSource
{
    IMAGE         m_image{};         ///< Existing producer mapping, never created, destroyed, updated, or written.
    bool          m_open{ false };   ///< This owner must release the mapping.
    int           m_semaphore{ -1 }; ///< Only this reader's claimed notification slot.
    std::string   m_path;            ///< Backing identity checked before reading an existing mapping.
    std::uint32_t m_depth{ 0 };      ///< Original allocation bound, never increased from mutable metadata.
    std::uint64_t m_device{ 0 }, m_inode{ 0 }; ///< Backing-file identity.

  public:
    /// Release only the reader mapping and its semaphore bookkeeping.
    ~NativeAdcSource() override;

    /// Open an existing input and claim one reader slot, with no producer or device operations.
    bool open( const std::string &name /**< [in] Validated non-DM source name. */ );

    /// Check backing replacement/removal and producer cleanup without reconnecting silently.
    bool connected() const;

    /// Return the original backing device.
    std::uint64_t device() const;

    /// Return the original backing inode.
    std::uint64_t inode() const;

    /// Snapshot current geometry, counter and write state.
    AdcHeader header() const override;

    /// Copy one in-bounds raw pair and matching original metadata.
    AdcFrame frame( std::size_t index /**< [in] Temporal plane. */ ) const override;

    /// Capture local clocks after copying producer data.
    AdcClock clock() const override;

    /// Bounded notification polling; never waits indefinitely for a stopped producer.
    void wait() const;

    /// Close without modifying producer payload/counters or unlinking its image.
    void close();
};

NativeAdcSource::~NativeAdcSource()
{
    close();
}

bool NativeAdcSource::open( const std::string &name )
{
    close();
    char        path[1024];
    struct stat before{}, after{};
    if( ImageStreamIO_filename( path, sizeof( path ), name.c_str() ) != 0 || stat( path, &before ) != 0 ||
        ImageStreamIO_openIm( &m_image, name.c_str() ) != 0 )
    {
        return false;
    }
    m_open = true;
    if( stat( path, &after ) != 0 || before.st_dev != after.st_dev || before.st_ino != after.st_ino || !m_image.md ||
        !m_image.semptr || !m_image.semReadPID || m_image.md->sem <= 0 )
    {
        close();
        return false;
    }
    m_depth     = m_image.md->naxis == 3 ? m_image.md->size[2] : 1;
    m_semaphore = ImageStreamIO_getsemwaitindex( &m_image, 5 );
    if( m_semaphore < 0 || m_semaphore >= m_image.md->sem || !m_image.semptr[m_semaphore] ||
        m_image.semptr[m_semaphore] == SEM_FAILED )
    {
        close();
        return false;
    }
    ImageStreamIO_semflush( &m_image, m_semaphore );
    m_path   = path;
    m_device = before.st_dev;
    m_inode  = before.st_ino;
    return true;
}

bool NativeAdcSource::connected() const
{
    struct stat current{};
    return m_open && m_image.md && m_image.md->sem > 0 && stat( m_path.c_str(), &current ) == 0 &&
           std::uint64_t( current.st_dev ) == m_device && std::uint64_t( current.st_ino ) == m_inode;
}

std::uint64_t NativeAdcSource::device() const
{
    return m_device;
}

std::uint64_t NativeAdcSource::inode() const
{
    return m_inode;
}

AdcHeader NativeAdcSource::header() const
{
    AdcHeader result;
    if( !m_open || !m_image.md )
    {
        return result;
    }
    const volatile auto *md = m_image.md;
    result.m_write          = md->write != 0;
    std::atomic_thread_fence( std::memory_order_seq_cst );
    result.m_axes       = md->naxis;
    result.m_width      = md->size[0];
    result.m_height     = result.m_axes > 1 ? md->size[1] : 0;
    result.m_depth      = result.m_axes == 3 ? md->size[2] : 1;
    result.m_counter    = md->cnt0;
    result.m_index      = result.m_axes == 3 ? md->cnt1 : 0;
    result.m_uint16     = md->datatype == _DATATYPE_UINT16;
    result.m_cpu        = md->location == -1;
    const bool temporal = ( md->imagetype & 0xF000F ) == ( CIRCULAR_BUFFER | ZAXIS_TEMPORAL );
    result.m_valid =
        m_image.array.raw && result.m_depth == m_depth &&
        ( m_depth == 1 || ( temporal && m_image.cntarray && m_image.atimearray && m_image.writetimearray ) );
    std::atomic_thread_fence( std::memory_order_seq_cst );
    result.m_write = result.m_write || md->write != 0;
    return result;
}

AdcFrame NativeAdcSource::frame( std::size_t index ) const
{
    AdcFrame result;
    if( !m_open || index >= m_depth || !m_image.array.raw )
    {
        return result;
    }
    std::atomic_thread_fence( std::memory_order_seq_cst );
    std::memcpy( result.m_raw.data(),
                 static_cast<const char *>( m_image.array.raw ) + index * 2 * sizeof( std::uint16_t ),
                 2 * sizeof( std::uint16_t ) );
    if( m_depth > 1 )
    {
        result.m_counter        = m_image.cntarray[index];
        result.m_acquiredRealNs = timestamp_ns( m_image.atimearray[index] );
        result.m_writtenRealNs  = timestamp_ns( m_image.writetimearray[index] );
    }
    else
    {
        result.m_counter        = m_image.md->cnt0;
        result.m_acquiredRealNs = timestamp_ns( m_image.md->atime );
        result.m_writtenRealNs  = timestamp_ns( m_image.md->writetime );
    }
    std::atomic_thread_fence( std::memory_order_seq_cst );
    return result;
}

AdcClock NativeAdcSource::clock() const
{
    return read_clock();
}

void NativeAdcSource::wait() const
{
    // Bounded polling also works on macOS, which has no sem_clockwait. Posts do not identify unique payloads.
    if( m_open && m_semaphore >= 0 )
    {
        for( unsigned n = 0; n < 256; ++n )
        {
            if( sem_trywait( m_image.semptr[m_semaphore] ) != 0 )
            {
                break;
            }
        }
    }
    std::this_thread::sleep_for( std::chrono::microseconds( 20 ) );
}

void NativeAdcSource::close()
{
    if( m_open )
    {
        if( m_semaphore >= 0 && m_image.semReadPID && m_image.md && m_semaphore < m_image.md->sem &&
            m_image.semReadPID[m_semaphore] == getpid() )
        {
            m_image.semReadPID[m_semaphore] = 0;
        }
        ImageStreamIO_closeIm( &m_image );
    }
    m_image     = {};
    m_open      = false;
    m_semaphore = -1;
    m_depth     = 0;
    m_device = m_inode = 0;
}

} // namespace

struct AdcReader::Impl
{
    AdcConfig          m_config;  ///< Immutable source name and calibration.
    mutable std::mutex m_mutex;   ///< Serializes short capture updates and diagnostic/ring copies, never model work.
    AdcCapture         m_capture; ///< Sole sample storage and validation state.
    std::atomic<bool>  m_running{ false }; ///< Worker stop request.
    std::thread        m_worker;           ///< Joined before capture state is destroyed.

    /// Allocate capture state without attaching any stream.
    explicit Impl( const AdcConfig &config /**< [in] Validated startup configuration. */ );

    /// Own all native acquisition and release the reader mapping on exit.
    void run();
};

AdcReader::Impl::Impl( const AdcConfig &config ) : m_config( config ), m_capture( config )
{
}

void AdcReader::Impl::run()
{
    NativeAdcSource source;
    bool            attached = false;
    while( m_running.load( std::memory_order_acquire ) )
    {
        if( !attached )
        {
            if( source.open( m_config.m_stream ) )
            {
                attached = true;
                { // mutex scope
                    std::lock_guard<std::mutex> lock( m_mutex );
                    m_capture.attached( source.device(), source.inode() );
                }
            }
            else
            {
                std::this_thread::sleep_for( std::chrono::milliseconds( 10 ) );
                continue;
            }
        }
        if( !source.connected() )
        {
            source.close();
            attached = false;
            { // mutex scope
                std::lock_guard<std::mutex> lock( m_mutex );
                m_capture.disconnected();
            }
            continue;
        }
        { // mutex scope
            std::lock_guard<std::mutex> lock( m_mutex );
            m_capture.drain( source );
        }
        source.wait();
    }
    { // mutex scope
        std::lock_guard<std::mutex> lock( m_mutex );
        m_capture.disconnected();
    }
}

AdcReader::AdcReader( const AdcConfig &config ) : m_impl( std::make_unique<Impl>( config ) )
{
}

AdcReader::~AdcReader()
{
    stop();
}

void AdcReader::start()
{
    if( m_impl->m_worker.joinable() )
    {
        return;
    }
    m_impl->m_running.store( true, std::memory_order_release );
    m_impl->m_worker = std::thread( &Impl::run, m_impl.get() );
}

void AdcReader::stop()
{
    m_impl->m_running.store( false, std::memory_order_release );
    if( m_impl->m_worker.joinable() )
    {
        m_impl->m_worker.join();
    }
}

AdcStatus AdcReader::status() const
{
    std::lock_guard<std::mutex> lock( m_impl->m_mutex ); // mutex scope
    return m_impl->m_capture.status();
}

std::vector<AdcSample> AdcReader::samples() const
{
    std::lock_guard<std::mutex> lock( m_impl->m_mutex ); // mutex scope
    return m_impl->m_capture.samples();
}

AdcBatch AdcReader::copy_after( const AdcCursor &cursor, std::span<AdcSample> output, std::int64_t availableByNs ) const
{
    std::lock_guard<std::mutex> lock( m_impl->m_mutex ); // mutex scope
    return m_impl->m_capture.copy_after( cursor, output, availableByNs );
}

} // namespace DDSPC
