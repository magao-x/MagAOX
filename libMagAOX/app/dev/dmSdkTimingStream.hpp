/** \file dmSdkTimingStream.hpp
 * \brief Optional semaphore-free publication of existing DM SDK entry timestamps.
 */
#ifndef dev_dmSdkTimingStream_hpp
#define dev_dmSdkTimingStream_hpp

#include <cerrno>
#include <cstring>
#include <limits>
#include <string>

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <ImageStreamIO/ImageStreamIO.h>

#include "dmTiming.hpp"

namespace MagAOX
{
namespace app
{
namespace dev
{

/// Publish metadata from one post-processing owner without semaphore operations.
/** Open and close only while the writer is stopped. The unused one-byte image
 * has no reader semaphores. Both timestamps use CLOCK_MONOTONIC: atime retains
 * SDK entry and writetime records this later publication. A fresh inode marks
 * each publisher incarnation; cnt0 counts successful published SDK entries.
 * ImageStreamIO reserves logging-semaphore storage, which this helper never uses.
 */
class dmSdkTimingStream
{
  public:
    /// One qualified metadata observation; counters are unrelated to camera frame numbers.
    struct Snapshot
    {
        uint64_t inode{ 0 };       ///< Publisher incarnation identified by its backing stream inode.
        uint64_t counter{ 0 };     ///< Published successful SDK entry count within this incarnation.
        int64_t  startNs{ 0 };     ///< Original monotonic SDK entry timestamp.
        int64_t  publishedNs{ 0 }; ///< Monotonic time when post-processing published that entry.
    };

    /// Start disabled without owning any shared resource.
    dmSdkTimingStream() = default;

    /// Keep the publisher's stream and lifetime lock uniquely owned.
    dmSdkTimingStream( const dmSdkTimingStream &other /**< [in] Publisher whose ownership cannot be copied. */ ) =
        delete;

    /// Keep the publisher's stream and lifetime lock uniquely owned.
    dmSdkTimingStream &
    operator=( const dmSdkTimingStream &other /**< [in] Publisher whose ownership cannot be copied. */ ) = delete;

    /// Release the attachment after its sole writer has stopped.
    ~dmSdkTimingStream() noexcept;

    /// Open a fresh exclusive publisher; an empty name disables it and succeeds.
    int open( const std::string &name /**< [in] Dedicated stream name without directory components. */ );

    /// Release resources without destroying semaphores or modifying an attached reader's last snapshot.
    void close() noexcept;

    /// Publish an existing SDK entry in bounded post-processing work; disabled publication succeeds.
    bool publish( int64_t startNs /**< [in] Positive monotonic native SDK entry timestamp. */ ) noexcept;

    /// Return whether a diagnostic stream is attached for publication.
    bool enabled() const noexcept;

    /// Recognize the fixed metadata-only layout and immutable clock/protocol keyword.
    static bool compatible( const IMAGE *image /**< [in] Attached stream to validate. */ ) noexcept;

    /// Read stable metadata in at most three attempts; initial zero metadata is unavailable.
    static bool read( const IMAGE *image /**< [in] Independently attached reader image. */,
                      Snapshot    &snapshot /**< [out] Qualified observation when true. */ ) noexcept;

  private:
    IMAGE m_stream{}; ///< Attachment written only by the post-processing owner after startup.

    int m_lock{ -1 }; ///< Exclusive nonblocking lifetime lock shared by cooperating publishers.

    /// Convert a normalized positive timespec without integer overflow.
    static int64_t timestamp( int64_t sec /**< [in] Whole monotonic seconds. */,
                              int64_t nsec /**< [in] Fractional nanoseconds. */ ) noexcept;
};

inline dmSdkTimingStream::~dmSdkTimingStream() noexcept
{
    close();
}

inline bool dmSdkTimingStream::compatible( const IMAGE *image ) noexcept
{
    return image && image->md && image->array.UI8 && image->kw && image->md->location == -1 &&
           image->md->datatype == _DATATYPE_UINT8 && image->md->naxis == 2 && image->md->size[0] == 1 &&
           image->md->size[1] == 1 && image->md->sem == 0 && image->md->NBkw == 1 && image->kw[0].type == 'S' &&
           std::strncmp( image->kw[0].name, "DM_SDK_TIME", sizeof( image->kw[0].name ) ) == 0 &&
           std::strncmp( image->kw[0].value.valstr, "monotonic-v1", sizeof( image->kw[0].value.valstr ) ) == 0;
}

inline int64_t dmSdkTimingStream::timestamp( int64_t sec, int64_t nsec ) noexcept
{
    if( sec < 0 || nsec < 0 || nsec >= 1000000000 || sec > ( std::numeric_limits<int64_t>::max() - nsec ) / 1000000000 )
    {
        return 0;
    }

    return sec * 1000000000 + nsec;
}

inline bool dmSdkTimingStream::read( const IMAGE *image, Snapshot &snapshot ) noexcept
{
    if( !compatible( image ) )
    {
        return false;
    }

    const auto *metadata = image->md;

    for( unsigned attempt = 0; attempt < 3; ++attempt )
    {
        if( __atomic_load_n( &metadata->write, __ATOMIC_ACQUIRE ) != 0 )
        {
            continue;
        }

        Snapshot next;
        next.inode               = __atomic_load_n( &metadata->inode, __ATOMIC_ACQUIRE );
        next.counter             = __atomic_load_n( &metadata->cnt0, __ATOMIC_ACQUIRE );
        const auto startSec      = __atomic_load_n( &metadata->atime.tv_sec, __ATOMIC_ACQUIRE );
        const auto startNsec     = __atomic_load_n( &metadata->atime.tv_nsec, __ATOMIC_ACQUIRE );
        const auto publishedSec  = __atomic_load_n( &metadata->writetime.tv_sec, __ATOMIC_ACQUIRE );
        const auto publishedNsec = __atomic_load_n( &metadata->writetime.tv_nsec, __ATOMIC_ACQUIRE );

        if( startSec != __atomic_load_n( &metadata->atime.tv_sec, __ATOMIC_ACQUIRE ) ||
            startNsec != __atomic_load_n( &metadata->atime.tv_nsec, __ATOMIC_ACQUIRE ) ||
            publishedSec != __atomic_load_n( &metadata->writetime.tv_sec, __ATOMIC_ACQUIRE ) ||
            publishedNsec != __atomic_load_n( &metadata->writetime.tv_nsec, __ATOMIC_ACQUIRE ) ||
            __atomic_load_n( &metadata->write, __ATOMIC_ACQUIRE ) != 0 ||
            next.counter != __atomic_load_n( &metadata->cnt0, __ATOMIC_ACQUIRE ) )
        {
            continue;
        }

        next.startNs     = timestamp( startSec, startNsec );
        next.publishedNs = timestamp( publishedSec, publishedNsec );

        if( next.inode == 0 || next.counter == 0 || next.startNs <= 0 || next.publishedNs < next.startNs )
        {
            return false;
        }

        snapshot = next;
        return true;
    }

    return false;
}

inline bool dmSdkTimingStream::enabled() const noexcept
{
    return m_stream.md != nullptr;
}

inline int dmSdkTimingStream::open( const std::string &name )
{
    close();

    if( name.empty() )
    {
        return 0;
    }

    if( name == "." || name == ".." || name.size() >= STRINGMAXLEN_IMAGE_NAME ||
        name.find_first_not_of( "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" ) !=
            std::string::npos )
    {
        return -1;
    }

    char filename[4096];

    if( ImageStreamIO_filename( filename, sizeof( filename ), name.c_str() ) != IMAGESTREAMIO_SUCCESS )
    {
        return -1;
    }

    m_lock = ::open(
        ( std::string( filename ) + ".dmSdkTiming.lock" ).c_str(), O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0660 );

    if( m_lock < 0 || flock( m_lock, LOCK_EX | LOCK_NB ) != 0 )
    {
        close();
        return -1;
    }

    struct stat previous{};
    const bool  exists = lstat( filename, &previous ) == 0;

    if( exists )
    {
        IMAGE old{};

        if( !S_ISREG( previous.st_mode ) || ImageStreamIO_openIm( &old, name.c_str() ) != IMAGESTREAMIO_SUCCESS )
        {
            close();
            return -1;
        }

        struct stat attached{};
        const bool  owned = compatible( &old ) && fstat( old.shmfd, &attached ) == 0 &&
                           attached.st_dev == previous.st_dev && attached.st_ino == previous.st_ino;
        ImageStreamIO_closeIm( &old );

        if( !owned )
        {
            close();
            return -1;
        }
    }
    else if( errno != ENOENT )
    {
        close();
        return -1;
    }

    const auto now       = dmTiming::monotonicNowNs();
    const auto temporary = ".dmSdkTiming_" + std::to_string( getpid() ) + "_" + std::to_string( now );
    char       temporaryFilename[4096];
    uint32_t   size[]{ 1, 1 };
    IMAGE      created{};

    if( now <= 0 ||
        ImageStreamIO_filename( temporaryFilename, sizeof( temporaryFilename ), temporary.c_str() ) !=
            IMAGESTREAMIO_SUCCESS ||
        ImageStreamIO_createIm_gpu( &created, temporary.c_str(), 2, size, _DATATYPE_UINT8, -1, 1, 0, 1, 0, 0 ) !=
            IMAGESTREAMIO_SUCCESS )
    {
        close();
        return -1;
    }

    m_stream = created;
    std::strncpy( m_stream.name, name.c_str(), sizeof( m_stream.name ) - 1 );
    std::strncpy( m_stream.md->name, name.c_str(), sizeof( m_stream.md->name ) - 1 );
    std::strcpy( m_stream.kw[0].name, "DM_SDK_TIME" );
    m_stream.kw[0].type = 'S';
    std::strcpy( m_stream.kw[0].value.valstr, "monotonic-v1" );
    m_stream.md->atime     = {};
    m_stream.md->writetime = {};

    struct stat current{};
    const bool  replaced = exists && ( lstat( filename, &current ) != 0 || current.st_dev != previous.st_dev ||
                                      current.st_ino != previous.st_ino || unlink( filename ) != 0 );

    if( replaced || link( temporaryFilename, filename ) != 0 )
    {
        unlink( temporaryFilename );
        close();
        return -1;
    }

    unlink( temporaryFilename );
    return 0;
}

inline void dmSdkTimingStream::close() noexcept
{
    const int savedErrno = errno;

    if( m_stream.md )
    {
        ImageStreamIO_closeIm( &m_stream );
        m_stream = {};
    }

    if( m_lock >= 0 )
    {
        ::close( m_lock );
        m_lock = -1;
    }

    errno = savedErrno;
}

inline bool dmSdkTimingStream::publish( int64_t startNs ) noexcept
{
    if( !enabled() )
    {
        return true;
    }

    const auto now     = dmTiming::monotonicNowNs();
    const auto counter = __atomic_load_n( &m_stream.md->cnt0, __ATOMIC_RELAXED );

    if( startNs <= 0 || now < startNs || counter == UINT64_MAX )
    {
        return false;
    }

    __atomic_store_n( &m_stream.md->write, uint8_t( 1 ), __ATOMIC_SEQ_CST );
    __atomic_store_n( &m_stream.md->atime.tv_sec, time_t( startNs / 1000000000 ), __ATOMIC_RELEASE );
    __atomic_store_n( &m_stream.md->atime.tv_nsec, long( startNs % 1000000000 ), __ATOMIC_RELEASE );
    __atomic_store_n( &m_stream.md->writetime.tv_sec, time_t( now / 1000000000 ), __ATOMIC_RELEASE );
    __atomic_store_n( &m_stream.md->writetime.tv_nsec, long( now % 1000000000 ), __ATOMIC_RELEASE );
    __atomic_store_n( &m_stream.md->cnt0, counter + 1, __ATOMIC_RELEASE );
    __atomic_store_n( &m_stream.md->write, uint8_t( 0 ), __ATOMIC_RELEASE );
    return true;
}

} // namespace dev
} // namespace app
} // namespace MagAOX
#endif // dev_dmSdkTimingStream_hpp
