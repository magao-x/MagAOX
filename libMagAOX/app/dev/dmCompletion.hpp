/** \file dmCompletion.hpp
 * \brief Optional bounded DM reader-completion publication on a dedicated shared stream.
 */
#ifndef dev_dmCompletion_hpp
#define dev_dmCompletion_hpp

#include <cerrno>
#include <cstring>
#include <limits>
#include <string>

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <ImageStreamIO/ImageStreamIO.h>

#include "../../common/dmCompletion.hpp"
#include "dmTiming.hpp"

namespace MagAOX
{
namespace app
{
namespace dev
{

/// Publish successful software reader completion independently of timing telemetry.
/** Open/close only while the command thread is stopped. One command thread owns
 * begin/finish/reset; ordinary app logic may read snapshot concurrently. The
 * stream's own semaphores are notifications, and the coherent versioned record
 * is authoritative. No input stream, input semaphore or hardware state changes.
 * Existing compatible streams are reused so attached readers survive restart.
 */
class dmCompletion
{
  protected:
    /// Stable identity sampled without modifying combined-input metadata.
    struct Identity
    {
        uint64_t inode{ 0 };   ///< Metadata inode, not an inferred channel identity.
        uint64_t counter{ 0 }; ///< Stable combined-input counter.
        int64_t  sec{ 0 };     ///< Raw producer publication seconds.
        int64_t  nsec{ 0 };    ///< Raw producer publication nanoseconds.
    };

    IMAGE m_stream{}; ///< Dedicated completion image, immutable attachment while the command thread runs.

    int m_lock{ -1 }; ///< Lifetime-exclusive publisher flock; readers do not acquire it.

    bool m_enabled{ false }; ///< Publication is configured and the compatible image is attached.

    bool m_active{ false }; ///< A begin call awaits its corresponding finish.

    bool m_qualified{ false }; ///< This attempt began with a genuine, new stable input identity.

    bool m_haveInput{ false }; ///< The previous genuine coherent identity can qualify progression.

    Identity m_previous; ///< Command-thread-only progression baseline.

    MagAOX::dmCompletion::Record m_record; ///< Command-thread-only record under construction.

    /// Check the dedicated stream's fixed CPU layout without touching its data.
    static bool compatible( const IMAGE *image /**< [in] Image whose layout is checked. */ ) noexcept;

    /// Read a repeated stable input identity without waiting for an in-progress writer.
    static bool identity( const IMAGE *image /**< [in] Combined command stream. */,
                          Identity    &value /**< [out] Coherent identity when available. */ ) noexcept;

    /// Compare every field of the raw combined-input identity.
    static bool equal( const Identity &first /**< [in] Earlier identity. */,
                       const Identity &second /**< [in] Later identity. */ ) noexcept;

    /// Store one coherent record and optionally notify dedicated completion-stream readers.
    void publish( bool notify /**< [in] Post completion/invalidation notifications when true. */ ) noexcept;

  public:
    /// Start with publication disabled and no attached shared resources.
    dmCompletion() = default;

    /// Keep publisher attachment ownership unique to one helper.
    dmCompletion( const dmCompletion &other /**< [in] Publisher whose ownership cannot be copied. */ ) = delete;

    /// Keep publisher attachment ownership unique to one helper.
    dmCompletion &
    operator=( const dmCompletion &other /**< [in] Publisher whose ownership cannot be copied. */ ) = delete;

    /// Close the optional publisher after its producer thread has stopped.
    ~dmCompletion() noexcept;

    /// Open an exclusive publisher; an empty name disables publication and succeeds.
    int open( const std::string &name /**< [in] Dedicated ImageStreamIO name, without directory components. */ );

    /// Publish unavailable, notify readers and release the attachment without destroying its stream.
    void close() noexcept;

    /// Invalidate previous completion and start a new attachment/progression epoch.
    void reset() noexcept;

    /// Publish busy before channel reads, qualifying only a genuine new stable input identity.
    void begin(
        const IMAGE *input /**< [in] Combined command stream consumed by this callback. */,
        bool genuine = true /**< [in] A real successful monitor semaphore receipt preceded this callback. */ ) noexcept;

    /// Publish after all callback reads, qualifying success only when the input identity stayed unchanged.
    void finish( const IMAGE              *input /**< [in] Combined command stream consumed by this callback. */,
                 int                       callbackResult /**< [in] Software callback result; zero is successful. */,
                 MagAOX::dmTiming::Outcome outcome = MagAOX::dmTiming::Outcome::unknown
                 /**< [in] Explicit backend submission result, independent of telemetry enable. */ ) noexcept;

    /// Return whether this fixed attachment is publishing completion records.
    bool enabled() const noexcept;

    /// Return a bounded coherent snapshot, or unavailable when a concurrent publication prevents it.
    MagAOX::dmCompletion::Record snapshot() const noexcept;

    /// Read an attached completion stream in at most three attempts, without consuming its semaphore.
    static bool
    read( const IMAGE                  *image /**< [in] Reader's dedicated completion-stream attachment. */,
          MagAOX::dmCompletion::Record &record /**< [out] Coherent recognized record when true. */ ) noexcept;
};

inline dmCompletion::~dmCompletion() noexcept
{
    close();
}

inline bool dmCompletion::compatible( const IMAGE *image ) noexcept
{
    return image != nullptr && image->md != nullptr && image->array.UI64 != nullptr &&
           image->md->datatype == _DATATYPE_UINT64 && image->md->naxis == 1 && image->md->location == -1 &&
           image->md->size[0] == MagAOX::dmCompletion::words && image->md->sem > 0 && image->semptr != nullptr;
}

inline bool dmCompletion::equal( const Identity &first, const Identity &second ) noexcept
{
    return first.inode == second.inode && first.counter == second.counter && first.sec == second.sec &&
           first.nsec == second.nsec;
}

inline bool dmCompletion::identity( const IMAGE *image, Identity &value ) noexcept
{
    if( image == nullptr || image->md == nullptr || __atomic_load_n( &image->md->write, __ATOMIC_ACQUIRE ) != 0 )
    {
        return false;
    }

    const auto *metadata = image->md;
    Identity    first{ __atomic_load_n( &metadata->inode, __ATOMIC_ACQUIRE ),
                    __atomic_load_n( &metadata->cnt0, __ATOMIC_ACQUIRE ),
                    __atomic_load_n( &metadata->writetime.tv_sec, __ATOMIC_ACQUIRE ),
                    __atomic_load_n( &metadata->writetime.tv_nsec, __ATOMIC_ACQUIRE ) };
    Identity    second{ __atomic_load_n( &metadata->inode, __ATOMIC_ACQUIRE ),
                     __atomic_load_n( &metadata->cnt0, __ATOMIC_ACQUIRE ),
                     __atomic_load_n( &metadata->writetime.tv_sec, __ATOMIC_ACQUIRE ),
                     __atomic_load_n( &metadata->writetime.tv_nsec, __ATOMIC_ACQUIRE ) };

    if( __atomic_load_n( &metadata->write, __ATOMIC_ACQUIRE ) != 0 || !equal( first, second ) || first.inode == 0 ||
        first.counter == 0 || first.sec <= 0 || first.nsec < 0 || first.nsec >= 1000000000 )
    {
        return false;
    }

    value = first;
    return true;
}

inline bool dmCompletion::read( const IMAGE *image, MagAOX::dmCompletion::Record &record ) noexcept
{
    if( !compatible( image ) )
    {
        return false;
    }

    for( size_t attempt = 0; attempt < 3; ++attempt )
    {
        if( __atomic_load_n( &image->md->write, __ATOMIC_ACQUIRE ) != 0 )
        {
            continue;
        }

        const auto                 counter = __atomic_load_n( &image->md->cnt0, __ATOMIC_ACQUIRE );
        MagAOX::dmCompletion::Wire wire{};

        for( size_t index = 0; index < wire.size(); ++index )
        {
            wire[index] = __atomic_load_n( image->array.UI64 + index, __ATOMIC_ACQUIRE );
        }

        if( __atomic_load_n( &image->md->write, __ATOMIC_ACQUIRE ) == 0 &&
            counter == __atomic_load_n( &image->md->cnt0, __ATOMIC_ACQUIRE ) )
        {
            return MagAOX::dmCompletion::decode( wire, record );
        }
    }

    return false;
}

inline void dmCompletion::publish( bool notify ) noexcept
{
    const int  savedErrno = errno;
    const auto wire       = MagAOX::dmCompletion::encode( m_record );
    __atomic_store_n( &m_stream.md->write, uint8_t( 1 ), __ATOMIC_SEQ_CST );

    for( size_t index = 0; index < wire.size(); ++index )
    {
        __atomic_store_n( m_stream.array.UI64 + index, wire[index], __ATOMIC_RELEASE );
    }

    const auto counter = __atomic_load_n( &m_stream.md->cnt0, __ATOMIC_RELAXED );
    __atomic_store_n( &m_stream.md->cnt0, counter + 1, __ATOMIC_RELEASE );
    __atomic_store_n( &m_stream.md->write, uint8_t( 0 ), __ATOMIC_RELEASE );

    if( notify )
    {
        ImageStreamIO_sempost( &m_stream, -1 );
    }

    errno = savedErrno;
}

inline int dmCompletion::open( const std::string &name )
{
    close();

    if( name.empty() )
    {
        return 0;
    }

    if( name.size() >= STRINGMAXLEN_IMAGE_NAME ||
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
        ( std::string( filename ) + ".dmCompletion.lock" ).c_str(), O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0660 );

    if( m_lock < 0 || flock( m_lock, LOCK_EX | LOCK_NB ) != 0 )
    {
        close();
        return -1;
    }

    const auto now = dmTiming::monotonicNowNs();

    if( now <= 0 )
    {
        close();
        return -1;
    }

    m_record              = {};
    m_record.session      = static_cast<uint64_t>( now );
    m_record.epoch        = 1;
    m_record.publisherPid = static_cast<uint64_t>( getpid() );

    struct stat status{};
    IMAGE       image{};

    if( lstat( filename, &status ) == 0 )
    {
        if( !S_ISREG( status.st_mode ) || ImageStreamIO_openIm( &image, name.c_str() ) != IMAGESTREAMIO_SUCCESS )
        {
            close();
            return -1;
        }

        m_stream = image;
        MagAOX::dmCompletion::Record previous;
        bool                         recognized = read( &m_stream, previous );

        if( !recognized && compatible( &m_stream ) )
        {
            // The exclusive lifetime lock permits recovery after a publisher died with write still set.
            MagAOX::dmCompletion::Wire wire{};

            for( size_t index = 0; index < wire.size(); ++index )
            {
                wire[index] = __atomic_load_n( m_stream.array.UI64 + index, __ATOMIC_ACQUIRE );
            }

            recognized = MagAOX::dmCompletion::decode( wire, previous );
        }

        if( !recognized || previous.session == 0 || previous.session == std::numeric_limits<uint64_t>::max() )
        {
            close();
            return -1;
        }

        if( m_record.session <= previous.session )
        {
            m_record.session = previous.session + 1;
        }
    }
    else
    {
        if( errno != ENOENT )
        {
            close();
            return -1;
        }

        // ImageStreamIO replaces existing names: publish a new inode with a non-replacing hard link instead.
        const auto temporary =
            ".dmCompletion_" + std::to_string( m_record.publisherPid ) + "_" + std::to_string( m_record.session );
        char     temporaryFilename[4096];
        uint32_t size = MagAOX::dmCompletion::words;

        if( ImageStreamIO_filename( temporaryFilename, sizeof( temporaryFilename ), temporary.c_str() ) !=
                IMAGESTREAMIO_SUCCESS ||
            ImageStreamIO_createIm_gpu(
                &image, temporary.c_str(), 1, &size, _DATATYPE_UINT64, -1, 1, IMAGE_NB_SEMAPHORE, 0, 0, 0 ) !=
                IMAGESTREAMIO_SUCCESS )
        {
            close();
            return -1;
        }

        m_stream = image;
        std::strncpy( m_stream.name, name.c_str(), sizeof( m_stream.name ) - 1 );
        std::strncpy( m_stream.md->name, name.c_str(), sizeof( m_stream.md->name ) - 1 );
        publish( false );

        if( link( temporaryFilename, filename ) != 0 )
        {
            unlink( temporaryFilename );
            close();
            return -1;
        }

        unlink( temporaryFilename );
    }

    m_enabled   = true;
    m_active    = false;
    m_qualified = false;
    m_haveInput = false;
    publish( true );
    return 0;
}

inline void dmCompletion::close() noexcept
{
    const int savedErrno = errno;

    if( m_enabled )
    {
        reset();
    }

    m_enabled   = false;
    m_active    = false;
    m_qualified = false;
    m_haveInput = false;

    if( m_stream.md != nullptr )
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

inline void dmCompletion::reset() noexcept
{
    if( !m_enabled )
    {
        return;
    }

    const auto session    = m_record.session;
    const auto epoch      = m_record.epoch + 1;
    const auto sequence   = m_record.sequence;
    const auto pid        = m_record.publisherPid;
    m_record              = {};
    m_record.session      = session;
    m_record.epoch        = epoch == 0 ? 1 : epoch;
    m_record.sequence     = sequence;
    m_record.publisherPid = pid;
    m_active              = false;
    m_qualified           = false;
    m_haveInput           = false;
    publish( true );
}

inline void dmCompletion::begin( const IMAGE *input, bool genuine ) noexcept
{
    if( !m_enabled )
    {
        return;
    }

    Identity   value;
    const bool valid = identity( input, value );

    if( m_active || ( valid && genuine && m_haveInput &&
                      ( value.inode != m_previous.inode || value.counter < m_previous.counter ) ) )
    {
        reset();
    }

    ++m_record.sequence;
    m_record.state           = MagAOX::dmCompletion::State::unqualified;
    m_record.inputInode      = valid ? value.inode : 0;
    m_record.inputCounter    = valid ? value.counter : 0;
    m_record.publicationSec  = valid ? value.sec : 0;
    m_record.publicationNsec = valid ? value.nsec : 0;
    m_record.startedNs       = dmTiming::monotonicNowNs();
    m_record.finishedNs      = 0;
    m_record.callbackResult  = 0;
    m_record.outcome         = MagAOX::dmTiming::Outcome::unknown;
    m_qualified = genuine && valid && m_record.startedNs > 0 && ( !m_haveInput || value.counter > m_previous.counter );

    if( m_qualified )
    {
        m_record.state = MagAOX::dmCompletion::State::busy;
    }

    if( genuine && valid )
    {
        m_previous  = value;
        m_haveInput = true;
    }

    m_active = true;
    publish( false );
}

inline void dmCompletion::finish( const IMAGE *input, int callbackResult, MagAOX::dmTiming::Outcome outcome ) noexcept
{
    if( !m_enabled || !m_active )
    {
        return;
    }

    Identity   value;
    const bool stable = identity( input, value ) && value.inode == m_record.inputInode &&
                        value.counter == m_record.inputCounter && value.sec == m_record.publicationSec &&
                        value.nsec == m_record.publicationNsec;
    m_record.finishedNs     = dmTiming::monotonicNowNs();
    m_record.callbackResult = callbackResult;
    m_record.outcome        = outcome;
    m_record.state          = MagAOX::dmCompletion::State::unqualified;

    if( callbackResult != 0 || outcome == MagAOX::dmTiming::Outcome::failed ||
        outcome == MagAOX::dmTiming::Outcome::partial || outcome == MagAOX::dmTiming::Outcome::rejected )
    {
        m_record.state = MagAOX::dmCompletion::State::error;
    }
    else if( m_qualified && stable && outcome == MagAOX::dmTiming::Outcome::success &&
             m_record.finishedNs >= m_record.startedNs )
    {
        m_record.state = MagAOX::dmCompletion::State::complete;
    }

    m_active = false;
    publish( true );
}

inline bool dmCompletion::enabled() const noexcept
{
    return m_enabled;
}

inline MagAOX::dmCompletion::Record dmCompletion::snapshot() const noexcept
{
    MagAOX::dmCompletion::Record record;

    if( m_enabled )
    {
        read( &m_stream, record );
    }

    return record;
}

} // namespace dev
} // namespace app
} // namespace MagAOX
#endif
