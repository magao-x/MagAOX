/** \file dmTimingEventStream.hpp
 * \brief Bounded semaphore-free history of SDK endpoints and qualified main labels.
 */
#ifndef dev_dmTimingEventStream_hpp
#define dev_dmTimingEventStream_hpp

#include <atomic>
#include <cerrno>
#include <cstring>
#include <ctime>
#include <string>

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <ImageStreamIO/ImageStreamIO.h>

#include "../../common/dmTimingEvents.hpp"

namespace MagAOX
{
namespace app
{
namespace dev
{

/// Publish fixed records from one owner into a bounded history without reader notifications.
/** Lifecycle follows the SDK timestamp publisher: a cooperating writer holds an
 * exclusive lifetime file lock and installs a fresh backing inode on open.
 * Each slot has an atomic transport sequence around atomic payload words. The
 * image counter is the last committed transport sequence, independent of the
 * producer's possibly gapped Record::sequence. Publication reads no clock and
 * performs no allocation or semaphore operation. ImageStreamIO still reserves
 * unused logging-semaphore storage. Open/close require the writer to be stopped.
 */
class dmTimingEventStream
{
  public:
    static constexpr size_t minCapacity = 4;                         ///< Smallest retained transport history.
    static constexpr size_t maxCapacity = 1048576;                   ///< Largest accepted fixed allocation, in records.
    static constexpr size_t slotWords   = dmTimingEvents::words + 1; ///< Commit sequence followed by one payload.

    /// One bounded reader attempt; gap carries the first surviving record and an exact overwrite count.
    enum class Result
    {
        unchanged,
        updated,
        gap,
        reset,
        busy,
        invalid
    };

    /// Independent cursor over one attached ring, owned by one polling thread.
    class Reader
    {
      public:
        /// Use the same outcome names through either the publisher or reader type.
        using Result = dmTimingEventStream::Result;

        /// Start detached without a shared mapping or cursor.
        Reader() = default;

        /// Keep each mapping and cursor uniquely owned.
        Reader( const Reader &other /**< [in] Reader whose mapping cannot be copied. */ ) = delete;

        /// Keep each mapping and cursor uniquely owned.
        Reader &operator=( const Reader &other /**< [in] Reader whose mapping cannot be copied. */ ) = delete;

        /// Release only this reader's attachment.
        ~Reader() noexcept;

        /// Attach a compatible ring and baseline its current committed head.
        int open( const std::string &name /**< [in] Stream name without directory components. */ );

        /// Release the mapping without changing the producer or other readers.
        void close() noexcept;

        /// Return whether this reader has an attached ring.
        bool isOpen() const noexcept;

        /// Check the backing path for replacement at the caller's lifecycle cadence.
        bool checkIdentity() const noexcept;

        /// Skip currently committed history and establish a new cursor baseline.
        void baseline() noexcept;

        /// Read at most one coherent record with three bounded attempts and no system calls.
        Result poll( dmTimingEvents::Record &record /**< [out] Valid record for updated or gap results. */ ) noexcept;

        /// Return the attached metadata incarnation, or zero while detached.
        uint64_t inode() const noexcept;

        /// Return overwritten transport records for the last gap; producer-sequence gaps are caller-owned.
        uint64_t missed() const noexcept;

      private:
        IMAGE       m_image{};        ///< Reader-owned mapping, never written by this reader.
        std::string m_filename;       ///< Resolved backing path for slower identity checks.
        dev_t       m_device{ 0 };    ///< Filesystem device captured from the attached descriptor.
        ino_t       m_fileInode{ 0 }; ///< Filesystem inode captured from the attached descriptor.
        uint64_t    m_identity{ 0 };  ///< Metadata incarnation represented by this cursor.
        uint64_t    m_cursor{ 0 };    ///< Last returned or explicitly baselined transport sequence.
        uint64_t    m_missed{ 0 };    ///< Exact overwritten count from the most recent poll gap.
        size_t      m_capacity{ 0 };  ///< Immutable slot count validated at attachment.

        /// Attach after clearing any old mapping; the public entrypoint contains allocation failures.
        int openImpl( const std::string &name /**< [in] Validated local stream name. */ );
    };

    /// Start disabled without owning any shared resource.
    dmTimingEventStream() = default;

    /// Keep the publisher mapping and lifetime lock uniquely owned.
    dmTimingEventStream( const dmTimingEventStream &other /**< [in] Publisher whose ownership cannot be copied. */ ) =
        delete;

    /// Keep the publisher mapping and lifetime lock uniquely owned.
    dmTimingEventStream &
    operator=( const dmTimingEventStream &other /**< [in] Publisher whose ownership cannot be copied. */ ) = delete;

    /// Release resources after the sole publisher has stopped.
    ~dmTimingEventStream() noexcept;

    /// Open an exclusive fresh ring; an empty name disables publication and succeeds.
    int open( const std::string &name /**< [in] Dedicated stream name without directory components. */,
              size_t             capacity = 32768 /**< [in] Retained records, between minCapacity and maxCapacity. */ );

    /// Release the mapping and lifetime lock without destroying shared semaphore storage.
    void close() noexcept;

    /// Return whether publication is configured and attached.
    bool enabled() const noexcept;

    /// Commit one valid increasing producer sequence; disabled publication succeeds without work.
    bool
    publish( const dmTimingEvents::Record &record /**< [in] Immutable endpoint or classification to copy. */ ) noexcept;

    /// Recognize the immutable versioned ring layout and bounded dimensions.
    static bool compatible( const IMAGE *image /**< [in] Attached image to validate. */ ) noexcept;

  private:
    IMAGE    m_stream{};          ///< Ring mapping written only by its single publisher.
    int      m_lock{ -1 };        ///< Exclusive lifetime lock shared by cooperating publishers.
    size_t   m_capacity{ 0 };     ///< Immutable retained slot count.
    uint64_t m_published{ 0 };    ///< Last committed contiguous transport sequence.
    uint64_t m_lastProducer{ 0 }; ///< Last successfully published producer logical sequence.

    /// Create after clearing any old mapping; the public entrypoint contains allocation failures.
    int openImpl( const std::string &name /**< [in] Dedicated local stream name. */,
                  size_t             capacity /**< [in] Requested bounded slot count. */ );

    /// Validate a local stream name and resolve its backing filename.
    static bool filename( const std::string &name /**< [in] Dedicated local stream name. */,
                          char              *path /**< [out] Destination for the backing path. */,
                          size_t             size /**< [in] Size of path storage. */ );
};

inline bool dmTimingEventStream::filename( const std::string &name, char *path, size_t size )
{
    return !name.empty() && name != "." && name != ".." && name.size() < STRINGMAXLEN_IMAGE_NAME &&
           name.find_first_not_of( "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" ) ==
               std::string::npos &&
           ImageStreamIO_filename( path, size, name.c_str() ) == IMAGESTREAMIO_SUCCESS;
}

inline bool dmTimingEventStream::compatible( const IMAGE *image ) noexcept
{
    return image && image->md && image->array.UI64 && image->kw && image->md->shared == 1 &&
           image->md->location == -1 && image->md->datatype == _DATATYPE_UINT64 && image->md->naxis == 2 &&
           image->md->size[0] == slotWords && image->md->size[1] >= minCapacity && image->md->size[1] <= maxCapacity &&
           image->md->nelement == slotWords * image->md->size[1] &&
           image->md->imdatamemsize == sizeof( uint64_t ) * slotWords * image->md->size[1] && image->md->sem == 0 &&
           image->md->NBkw == 1 && image->kw[0].type == 'S' &&
           std::strncmp( image->kw[0].name, "DM_TIMING_RING", sizeof( image->kw[0].name ) ) == 0 &&
           std::strncmp( image->kw[0].value.valstr, "ring-v1", sizeof( image->kw[0].value.valstr ) ) == 0;
}

inline dmTimingEventStream::~dmTimingEventStream() noexcept
{
    close();
}

inline int dmTimingEventStream::open( const std::string &name, size_t capacity )
{
    close();

    try
    {
        return openImpl( name, capacity );
    }
    catch( ... )
    {
        close();
        return -1;
    }
}

inline int dmTimingEventStream::openImpl( const std::string &name, size_t capacity )
{

    if( name.empty() )
    {
        return 0;
    }

    char path[4096];

    if( capacity < minCapacity || capacity > maxCapacity || !filename( name, path, sizeof( path ) ) )
    {
        return -1;
    }

    m_lock = ::open(
        ( std::string( path ) + ".dmTimingEvents.lock" ).c_str(), O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0660 );

    if( m_lock < 0 || flock( m_lock, LOCK_EX | LOCK_NB ) != 0 )
    {
        close();
        return -1;
    }

    struct stat previous{};
    const bool  exists = lstat( path, &previous ) == 0;

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

    timespec now{};

    if( clock_gettime( CLOCK_MONOTONIC, &now ) != 0 )
    {
        close();
        return -1;
    }

    const auto temporary = ".dmTimingEvents_" + std::to_string( getpid() ) + "_" + std::to_string( now.tv_sec ) + "_" +
                           std::to_string( now.tv_nsec );
    char     temporaryPath[4096];
    uint32_t size[]{ uint32_t( slotWords ), uint32_t( capacity ) };
    IMAGE    created{};

    if( !filename( temporary, temporaryPath, sizeof( temporaryPath ) ) ||
        ImageStreamIO_createIm_gpu( &created, temporary.c_str(), 2, size, _DATATYPE_UINT64, -1, 1, 0, 1, 0, 0 ) !=
            IMAGESTREAMIO_SUCCESS )
    {
        close();
        return -1;
    }

    m_stream = created;
    std::strncpy( m_stream.name, name.c_str(), sizeof( m_stream.name ) - 1 );
    std::strncpy( m_stream.md->name, name.c_str(), sizeof( m_stream.md->name ) - 1 );
    std::strcpy( m_stream.kw[0].name, "DM_TIMING_RING" );
    m_stream.kw[0].type = 'S';
    std::strcpy( m_stream.kw[0].value.valstr, "ring-v1" );
    m_stream.md->atime     = {};
    m_stream.md->writetime = {};
    m_stream.md->cnt0      = 0;
    m_stream.md->write     = 0;
    std::memset( m_stream.array.UI64, 0, sizeof( uint64_t ) * slotWords * capacity );

    struct stat current{};
    const bool  replaced = exists && ( lstat( path, &current ) != 0 || current.st_dev != previous.st_dev ||
                                      current.st_ino != previous.st_ino || unlink( path ) != 0 );

    if( replaced || link( temporaryPath, path ) != 0 )
    {
        unlink( temporaryPath );
        close();
        return -1;
    }

    unlink( temporaryPath );
    m_capacity = capacity;
    return 0;
}

inline void dmTimingEventStream::close() noexcept
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

    m_capacity  = 0;
    m_published = m_lastProducer = 0;
    errno                        = savedErrno;
}

inline bool dmTimingEventStream::enabled() const noexcept
{
    return m_stream.md != nullptr;
}

inline bool dmTimingEventStream::publish( const dmTimingEvents::Record &record ) noexcept
{
    static_assert( std::atomic<uint64_t>::is_always_lock_free, "Shared event slots require lock-free uint64 atomics" );

    if( !enabled() )
    {
        return true;
    }

    if( !dmTimingEvents::valid( record ) || record.sequence <= m_lastProducer || m_published == UINT64_MAX )
    {
        return false;
    }

    const auto sequence = m_published + 1;
    auto      *slot     = m_stream.array.UI64 + ( ( sequence - 1 ) % m_capacity ) * slotWords;
    const auto wire     = dmTimingEvents::encode( record );
    __atomic_store_n( slot, uint64_t( 0 ), __ATOMIC_SEQ_CST );

    for( size_t index = 0; index < wire.size(); ++index )
    {
        __atomic_store_n( slot + index + 1, wire[index], __ATOMIC_RELEASE );
    }

    __atomic_store_n( slot, sequence, __ATOMIC_RELEASE );
    __atomic_store_n( &m_stream.md->cnt0, sequence, __ATOMIC_RELEASE );
    m_published    = sequence;
    m_lastProducer = record.sequence;
    return true;
}

inline dmTimingEventStream::Reader::~Reader() noexcept
{
    close();
}

inline int dmTimingEventStream::Reader::open( const std::string &name )
{
    close();

    try
    {
        return openImpl( name );
    }
    catch( ... )
    {
        close();
        return -1;
    }
}

inline int dmTimingEventStream::Reader::openImpl( const std::string &name )
{
    char        path[4096];
    struct stat status{};
    IMAGE       attached{};

    if( !filename( name, path, sizeof( path ) ) || stat( path, &status ) != 0 || !S_ISREG( status.st_mode ) ||
        ImageStreamIO_openIm( &attached, name.c_str() ) != IMAGESTREAMIO_SUCCESS )
    {
        return -1;
    }

    m_image = attached;

    if( !compatible( &m_image ) || fstat( m_image.shmfd, &status ) != 0 || m_image.md->inode == 0 )
    {
        close();
        return -1;
    }

    m_filename  = path;
    m_device    = status.st_dev;
    m_fileInode = status.st_ino;
    m_capacity  = m_image.md->size[1];

    if( !checkIdentity() )
    {
        close();
        return -1;
    }

    baseline();
    return 0;
}

inline void dmTimingEventStream::Reader::close() noexcept
{
    const int savedErrno = errno;

    if( m_image.md )
    {
        ImageStreamIO_closeIm( &m_image );
        m_image = {};
    }

    m_filename.clear();
    m_device    = 0;
    m_fileInode = 0;
    m_identity = m_cursor = m_missed = 0;
    m_capacity                       = 0;
    errno                            = savedErrno;
}

inline bool dmTimingEventStream::Reader::isOpen() const noexcept
{
    return m_image.md != nullptr;
}

inline bool dmTimingEventStream::Reader::checkIdentity() const noexcept
{
    struct stat status{};
    return isOpen() && stat( m_filename.c_str(), &status ) == 0 && status.st_dev == m_device &&
           status.st_ino == m_fileInode;
}

inline void dmTimingEventStream::Reader::baseline() noexcept
{
    m_cursor   = isOpen() ? __atomic_load_n( &m_image.md->cnt0, __ATOMIC_ACQUIRE ) : 0;
    m_identity = isOpen() ? __atomic_load_n( &m_image.md->inode, __ATOMIC_ACQUIRE ) : 0;
    m_missed   = 0;
}

inline uint64_t dmTimingEventStream::Reader::inode() const noexcept
{
    return m_identity;
}

inline uint64_t dmTimingEventStream::Reader::missed() const noexcept
{
    return m_missed;
}

inline dmTimingEventStream::Result dmTimingEventStream::Reader::poll( dmTimingEvents::Record &record ) noexcept
{
    m_missed = 0;

    if( !isOpen() )
    {
        return Result::invalid;
    }

    for( unsigned attempt = 0; attempt < 3; ++attempt )
    {
        const auto head = __atomic_load_n( &m_image.md->cnt0, __ATOMIC_ACQUIRE );

        if( __atomic_load_n( &m_image.md->inode, __ATOMIC_ACQUIRE ) != m_identity || head < m_cursor )
        {
            baseline();
            return Result::reset;
        }

        if( head == m_cursor )
        {
            return Result::unchanged;
        }

        const auto  missed   = head - m_cursor > m_capacity ? head - m_cursor - m_capacity : 0;
        const auto  sequence = m_cursor + missed + 1;
        const auto *slot     = m_image.array.UI64 + ( ( sequence - 1 ) % m_capacity ) * slotWords;

        if( __atomic_load_n( slot, __ATOMIC_ACQUIRE ) != sequence )
        {
            continue;
        }

        dmTimingEvents::Wire wire;

        for( size_t index = 0; index < wire.size(); ++index )
        {
            wire[index] = __atomic_load_n( slot + index + 1, __ATOMIC_ACQUIRE );
        }

        if( __atomic_load_n( slot, __ATOMIC_ACQUIRE ) != sequence )
        {
            continue;
        }

        if( !dmTimingEvents::decode( wire, record ) )
        {
            return Result::invalid;
        }

        m_cursor = sequence;
        m_missed = missed;
        return missed ? Result::gap : Result::updated;
    }

    return Result::busy;
}

} // namespace dev
} // namespace app
} // namespace MagAOX
#endif // dev_dmTimingEventStream_hpp
