/** \file loopTimingStream.hpp
 * \brief Metadata-only image polling without semaphore synchronization.
 */
#ifndef loopTimingMonitor_loopTimingStream_hpp
#define loopTimingMonitor_loopTimingStream_hpp

#include <cstdint>
#include <limits>
#include <string>

#include <sys/stat.h>

#include "../../libMagAOX/app/dev/dmSdkTimingStream.hpp"

namespace MagAOX
{
namespace app
{
namespace loopTiming
{

/// Attach and observe one timestamp stream from a single polling-thread owner.
/** No operation claims, waits on, flushes or posts a semaphore. Camera reads
 * qualify the existing producer protocol with repeated metadata checks; they
 * do not establish consumed-frame provenance. SDK reads use the explicit
 * metadata-only publication contract. Filesystem checks are caller-paced.
 */
class Stream
{
  public:
    /// One stable source observation in its producer's declared clock.
    struct Snapshot
    {
        uint64_t counter{ 0 }; ///< Publication count, independent of the other source's count.

        uint64_t inode{ 0 }; ///< Producer incarnation from image metadata.

        int64_t atimeNs{ 0 }; ///< Camera acquisition or original SDK entry timestamp.

        int64_t writetimeNs{ 0 }; ///< Producer publication timestamp, not a latency endpoint.

        uint64_t missed{ 0 }; ///< Skipped publications preceding this observation, when inferable.
    };

    /// Poll outcome; only updated observations extend a continuous association.
    enum class Result
    {
        unchanged,
        updated,
        gap,
        reset,
        busy,
        invalid
    };

    /// Start detached without owning any shared resource.
    Stream() = default;

    /// Keep attachment ownership unique.
    Stream( const Stream &other /**< [in] Attachment whose ownership cannot be copied. */ ) = delete;

    /// Keep attachment ownership unique.
    Stream &operator=( const Stream &other /**< [in] Attachment whose ownership cannot be copied. */ ) = delete;

    /// Close the reader mapping without destroying the producer's image.
    ~Stream() noexcept;

    /// Attach a local named image and baseline any coherent existing observation.
    int open( const std::string &name /**< [in] Stream name without directory components. */,
              bool               sdk = false /**< [in] Require the dedicated monotonic SDK publication contract. */ );

    /// Release this reader only; the producer and its metadata remain unchanged.
    void close() noexcept;

    /// Return whether a stream is attached.
    bool isOpen() const noexcept;

    /// Check the backing path for replacement without changing the attachment.
    bool checkIdentity() const noexcept;

    /// Forget continuity and baseline the current image without accepting a sample.
    void baseline() noexcept;

    /// Read one bounded metadata snapshot and classify source progression.
    Result poll(
        Snapshot &snapshot /**< [out] Coherent observation for unchanged, updated, gap or reset outcomes. */ ) noexcept;

  private:
    IMAGE m_image{}; ///< Reader mapping, exclusively owned by the caller's polling thread.

    std::string m_filename; ///< Backing path checked only at the caller's slower lifecycle cadence.

    dev_t m_device{ 0 }; ///< Filesystem device of the attached descriptor.

    ino_t m_inode{ 0 }; ///< Filesystem inode of the attached descriptor.

    bool m_sdk{ false }; ///< Require the dedicated SDK schema rather than generic camera metadata.

    bool m_havePrevious{ false }; ///< A coherent observation establishes this attachment's baseline.

    Snapshot m_previous; ///< Last accepted metadata identity for progression checks.

    /// Read a coherent snapshot without changing continuity state.
    Result read( Snapshot &snapshot /**< [out] Stable metadata when updated is returned. */ ) const noexcept;

    /// Convert a normalized nonnegative timespec without integer overflow.
    static int64_t timestamp( int64_t sec /**< [in] Whole producer-clock seconds. */,
                              int64_t nsec /**< [in] Fractional nanoseconds. */ ) noexcept;
};

inline Stream::~Stream() noexcept
{
    close();
}

inline int Stream::open( const std::string &name, bool sdk )
{
    close();

    if( name.empty() || name == "." || name == ".." || name.size() >= STRINGMAXLEN_IMAGE_NAME ||
        name.find_first_not_of( "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" ) !=
            std::string::npos )
    {
        return -1;
    }

    char        filename[4096];
    struct stat status{};
    IMAGE       attached{};

    if( ImageStreamIO_filename( filename, sizeof( filename ), name.c_str() ) != IMAGESTREAMIO_SUCCESS ||
        stat( filename, &status ) != 0 || !S_ISREG( status.st_mode ) ||
        ImageStreamIO_openIm( &attached, name.c_str() ) != IMAGESTREAMIO_SUCCESS )
    {
        return -1;
    }

    m_image = attached;

    if( !m_image.md || fstat( m_image.shmfd, &status ) != 0 ||
        ( sdk && !dev::dmSdkTimingStream::compatible( &m_image ) ) )
    {
        close();
        return -1;
    }

    m_filename = filename;
    m_device   = status.st_dev;
    m_inode    = status.st_ino;
    m_sdk      = sdk;

    if( !checkIdentity() )
    {
        close();
        return -1;
    }

    baseline();
    return 0;
}

inline void Stream::close() noexcept
{
    if( m_image.md )
    {
        ImageStreamIO_closeIm( &m_image );
    }

    m_image = {};
    m_filename.clear();
    m_device       = 0;
    m_inode        = 0;
    m_sdk          = false;
    m_havePrevious = false;
    m_previous     = {};
}

inline bool Stream::isOpen() const noexcept
{
    return m_image.md != nullptr;
}

inline bool Stream::checkIdentity() const noexcept
{
    struct stat status{};
    return isOpen() && stat( m_filename.c_str(), &status ) == 0 && status.st_dev == m_device &&
           status.st_ino == m_inode;
}

inline int64_t Stream::timestamp( int64_t sec, int64_t nsec ) noexcept
{
    if( sec < 0 || nsec < 0 || nsec >= 1000000000 || sec > ( std::numeric_limits<int64_t>::max() - nsec ) / 1000000000 )
    {
        return 0;
    }

    return sec * 1000000000 + nsec;
}

inline Stream::Result Stream::read( Snapshot &snapshot ) const noexcept
{
    if( !isOpen() )
    {
        return Result::invalid;
    }

    const auto *metadata = m_image.md;

    if( __atomic_load_n( &metadata->write, __ATOMIC_ACQUIRE ) != 0 ||
        __atomic_load_n( &metadata->cnt0, __ATOMIC_ACQUIRE ) == 0 )
    {
        return Result::busy;
    }

    if( m_sdk )
    {
        dev::dmSdkTimingStream::Snapshot value;
        const auto                       counter = __atomic_load_n( &metadata->cnt0, __ATOMIC_ACQUIRE );

        if( !dev::dmSdkTimingStream::read( &m_image, value ) )
        {
            if( !dev::dmSdkTimingStream::compatible( &m_image ) )
            {
                return Result::invalid;
            }

            const auto start     = timestamp( __atomic_load_n( &metadata->atime.tv_sec, __ATOMIC_ACQUIRE ),
                                          __atomic_load_n( &metadata->atime.tv_nsec, __ATOMIC_ACQUIRE ) );
            const auto published = timestamp( __atomic_load_n( &metadata->writetime.tv_sec, __ATOMIC_ACQUIRE ),
                                              __atomic_load_n( &metadata->writetime.tv_nsec, __ATOMIC_ACQUIRE ) );
            const bool stable    = __atomic_load_n( &metadata->write, __ATOMIC_ACQUIRE ) == 0 &&
                                counter == __atomic_load_n( &metadata->cnt0, __ATOMIC_ACQUIRE );
            return stable && ( start <= 0 || published < start ) ? Result::invalid : Result::busy;
        }

        snapshot = { value.counter, value.inode, value.startNs, value.publishedNs, 0 };
        return Result::updated;
    }

    for( unsigned attempt = 0; attempt < 3; ++attempt )
    {
        if( __atomic_load_n( &metadata->write, __ATOMIC_ACQUIRE ) != 0 )
        {
            continue;
        }

        Snapshot next;
        next.counter         = __atomic_load_n( &metadata->cnt0, __ATOMIC_ACQUIRE );
        next.inode           = __atomic_load_n( &metadata->inode, __ATOMIC_ACQUIRE );
        const auto atimeSec  = __atomic_load_n( &metadata->atime.tv_sec, __ATOMIC_ACQUIRE );
        const auto atimeNsec = __atomic_load_n( &metadata->atime.tv_nsec, __ATOMIC_ACQUIRE );
        const auto wtimeSec  = __atomic_load_n( &metadata->writetime.tv_sec, __ATOMIC_ACQUIRE );
        const auto wtimeNsec = __atomic_load_n( &metadata->writetime.tv_nsec, __ATOMIC_ACQUIRE );

        if( atimeSec != __atomic_load_n( &metadata->atime.tv_sec, __ATOMIC_ACQUIRE ) ||
            atimeNsec != __atomic_load_n( &metadata->atime.tv_nsec, __ATOMIC_ACQUIRE ) ||
            wtimeSec != __atomic_load_n( &metadata->writetime.tv_sec, __ATOMIC_ACQUIRE ) ||
            wtimeNsec != __atomic_load_n( &metadata->writetime.tv_nsec, __ATOMIC_ACQUIRE ) ||
            next.inode != __atomic_load_n( &metadata->inode, __ATOMIC_ACQUIRE ) ||
            __atomic_load_n( &metadata->write, __ATOMIC_ACQUIRE ) != 0 ||
            next.counter != __atomic_load_n( &metadata->cnt0, __ATOMIC_ACQUIRE ) )
        {
            continue;
        }

        next.atimeNs     = timestamp( atimeSec, atimeNsec );
        next.writetimeNs = timestamp( wtimeSec, wtimeNsec );

        if( next.inode == 0 || next.atimeNs <= 0 || next.writetimeNs <= 0 )
        {
            return Result::invalid;
        }

        snapshot = next;
        return Result::updated;
    }

    return Result::busy;
}

inline void Stream::baseline() noexcept
{
    Snapshot current;
    m_havePrevious = read( current ) == Result::updated;
    m_previous     = m_havePrevious ? current : Snapshot{};
}

inline Stream::Result Stream::poll( Snapshot &snapshot ) noexcept
{
    Snapshot   next;
    const auto result = read( next );

    if( result != Result::updated )
    {
        return result;
    }

    snapshot = next;

    if( !m_havePrevious )
    {
        m_havePrevious = true;
        m_previous     = next;
        return Result::unchanged;
    }

    const auto previous = m_previous;

    if( next.inode != previous.inode || next.counter < previous.counter )
    {
        m_previous = next;
        return Result::reset;
    }

    if( next.counter == previous.counter )
    {
        return next.atimeNs == previous.atimeNs && next.writetimeNs == previous.writetimeNs ? Result::unchanged
                                                                                            : Result::invalid;
    }

    m_previous = next;

    if( next.atimeNs <= previous.atimeNs || next.writetimeNs <= previous.writetimeNs )
    {
        return Result::invalid;
    }

    snapshot.missed = next.counter - previous.counter - 1;
    return snapshot.missed ? Result::gap : Result::updated;
}

} // namespace loopTiming
} // namespace app
} // namespace MagAOX

#endif // loopTimingMonitor_loopTimingStream_hpp
