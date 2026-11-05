/** \file dmSyncopationProducer.hpp
 * \brief Same-host tagged syncopation publishing and source-buffer ownership for CRTP applications.
 */
#ifndef dev_dmSyncopationProducer_hpp
#define dev_dmSyncopationProducer_hpp

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <ImageStreamIO/ImageStreamIO.h>
#include <mx/app/appConfigurator.hpp>
#include <mx/math/floatUtils.hpp>

#include "../dmSyncopationTiming.hpp"

namespace MagAOX
{
namespace app
{
namespace synco
{

/// Copy a fixed signed-int64 image only when its entire publication remains coherent.
template <size_t count>
bool readRecord( const IMAGE                &image /**< [in] Read-only attached protocol stream. */,
                 std::array<int64_t, count> &record /**< [out] Coherent payload on success. */ );

/// Publish one complete fixed signed-int64 protocol record.
template <size_t count>
bool writeRecord( IMAGE &image /**< [in,out] Exclusively owned compatible protocol stream. */,
                  const std::array<int64_t, count> &record /**< [in] Complete protocol payload. */ );

/// Open a compatible fixed record image, creating it only when explicitly permitted.
bool openRecord( IMAGE             &image /**< [out] Open image on success. */,
                 const std::string &name /**< [in] Same-host ImageStreamIO name. */,
                 uint32_t           count /**< [in] Required signed-int64 element count. */,
                 bool               create /**< [in] This process owns publication and may create a missing image. */ );

template <size_t count>
bool readRecord( const IMAGE &image, std::array<int64_t, count> &record )
{
    if( !image.md || !image.array.raw || image.md->datatype != _DATATYPE_INT64 || image.md->naxis != 1 ||
        image.md->size[0] != count || image.md->nelement != count ||
        image.md->imdatamemsize != count * sizeof( int64_t ) || image.md->location != -1 || image.md->sem == 0 ||
        image.md->write )
    {
        return false;
    }

    const auto counter = image.md->cnt0;
    std::atomic_thread_fence( std::memory_order_acquire );
    std::memcpy( record.data(), image.array.raw, sizeof( record ) );
    std::atomic_thread_fence( std::memory_order_acquire );
    return !image.md->write && image.md->cnt0 == counter;
}

template <size_t count>
bool writeRecord( IMAGE &image, const std::array<int64_t, count> &record )
{
    if( !image.md || !image.array.raw || image.md->datatype != _DATATYPE_INT64 || image.md->naxis != 1 ||
        image.md->size[0] != count || image.md->nelement != count ||
        image.md->imdatamemsize != count * sizeof( int64_t ) || image.md->location != -1 || image.md->sem == 0 )
    {
        return false;
    }

    image.md->write = 1;
    std::memcpy( image.array.raw, record.data(), sizeof( record ) );
    std::atomic_thread_fence( std::memory_order_release );
    return ImageStreamIO_UpdateIm( &image ) == 0;
}

inline bool openRecord( IMAGE &image, const std::string &name, uint32_t count, bool create )
{
    char        filename[4096];
    struct stat info{};

    if( ImageStreamIO_filename( filename, sizeof( filename ), name.c_str() ) != 0 )
    {
        return false;
    }

    if( stat( filename, &info ) == 0 )
    {
        if( ImageStreamIO_openIm( &image, name.c_str() ) != 0 )
        {
            return false;
        }
    }
    else if( !create || errno != ENOENT ||
             ImageStreamIO_createIm_gpu(
                 &image, name.c_str(), 1, &count, _DATATYPE_INT64, -1, 1, IMAGE_NB_SEMAPHORE, 0, 0, 0 ) != 0 )
    {
        return false;
    }

    if( !image.array.raw || image.md->datatype != _DATATYPE_INT64 || image.md->naxis != 1 ||
        image.md->size[0] != count || image.md->nelement != count ||
        image.md->imdatamemsize != count * sizeof( int64_t ) || image.md->location != -1 || image.md->sem == 0 )
    {
        ImageStreamIO_closeIm( &image );
        image = {};
        return false;
    }

    return true;
}

} // namespace synco

namespace dev
{

/// Handle schedule discovery, exact slot tagging and one-buffer transport ownership for local producers.
/** The application owns the CPU-backed 2-D float32 source IMAGE and the thread
 * calling poll/publish/close. Add this helper as a CRTP base, alias it as
 * dmSyncopationProducerT, and use its configuration/shutdown interface macros.
 * The derived application supplies only pixels for Request::targetNs. It must
 * not modify the managed source directly, including while disabling modulation.
 *
 * A complete _ready commit transfers source ownership to synco. The matching
 * _ack releases it after a private snapshot or discard, independently of DM
 * triggering and physical consumption. Failed/late publishes do not commit.
 * A restarted producer adopts an intact predecessor commit and preserves its
 * source ownership until an exact acknowledgement or valid scheduler reset.
 * Schedule replacement/reset invalidates old pending requests. Notifications
 * only prompt rereading; callers may poll, or sleep between polls. All clocks
 * and shared images are local to the same host. No instrument commands occur.
 * \ingroup appdev
 */
template <class derivedT>
class dmSyncopationProducer
{
  public:
    using Request = synco::Request; ///< One immutable scheduler request passed to the pixel producer.

  protected:
    /** \name Configuration - Data
     * @{ */
    std::string m_syncopationTiming; ///< Empty preserves the application's unmanaged operation.

    double m_writeLeadUs{ 50 }; ///< Desired payload-completion lead before its advertised cutoff.
    /// @}

    IMAGE m_syncopationSchedule{}; ///< Read-only current scheduler timing image.

    IMAGE m_syncopationReady{}; ///< Producer-owned commit sidecar for its single payload buffer.

    IMAGE m_syncopationAck{}; ///< Read-only scheduler transport acknowledgement sidecar.

    synco::CommitRecord m_syncopationPending{}; ///< Exact publication whose payload is still consumer-owned.

    Request m_syncopationLast{}; ///< Last returned request, suppressing repeated processing of one notification.

    int64_t m_syncopationSession{ 0 }; ///< Unique producer attachment session, independent of source counters.

    int64_t m_syncopationSequence{ 0 }; ///< Positive publication sequence within the producer attachment.

    int64_t m_syncopationFileCheck{ 0 }; ///< Next backing-file identity check, outside final payload publication.

    uint64_t m_syncopationSourceInode{ 0 }; ///< Source identity frozen while protocol streams are attached.

    uint64_t m_syncopationPixels{ 0 }; ///< Validated source extent and required private payload length.

    uint32_t m_syncopationWidth{ 0 }; ///< Frozen source width; shape changes require reconnection.

    uint32_t m_syncopationHeight{ 0 }; ///< Frozen source height; shape changes require reconnection.

    int m_syncopationLock{ -1 }; ///< Lifetime-exclusive cooperative source producer lock.

    bool m_syncopationOutstanding{ false }; ///< Consumer owns the single source payload until acknowledgement.

    /// Return the CRTP application to observe shutdown while waiting.
    derivedT &derived();

    /// Attach compatible schedule/transport streams without changing the source pixels.
    bool syncopationOpen( IMAGE &source /**< [in] Application-owned writable source image. */ );

    /// Confirm that an attachment still names its original backing file.
    static bool syncopationIdentity( const IMAGE &image /**< [in] Attached protocol or source image. */ );

    /// Validate a source's shape, element count and byte extent before any pixel access.
    static bool syncopationGeometry( const IMAGE &image /**< [in] Attached source image. */ );

    /// Confirm the caller still supplies the originally validated source extent and identity.
    bool syncopationSource( const IMAGE &image /**< [in] Source supplied to poll or publish. */ ) const;

    /// Return whether two advertised requests identify exactly the same command.
    static bool syncopationSame( const Request &first /**< [in] Previous immutable request. */,
                                 const Request &second /**< [in] Current immutable request. */ );

  public:
    /// Default construction performs no shared-memory or instrument access.
    dmSyncopationProducer() = default;

    /// Source ownership and attached stream mappings cannot be copied.
    dmSyncopationProducer( const dmSyncopationProducer &other /**< [in] Noncopyable transport owner. */ ) = delete;

    /// Source ownership and attached stream mappings cannot be assigned.
    dmSyncopationProducer &
    operator=( const dmSyncopationProducer &other /**< [in] Noncopyable transport owner. */ ) = delete;

    /// Release owned attachments after the application's publishing thread has stopped.
    ~dmSyncopationProducer();

    /// Register the optional local provider timing configuration.
    int setupConfig( mx::app::appConfigurator &config /**< [in,out] Application configuration registry. */ );

    /// Load and validate optional timing configuration without opening streams.
    int loadConfig( mx::app::appConfigurator &config /**< [in] Parsed application configuration. */ );

    /// Return 1 for a new usable slot, 0 while waiting, or -1 for unavailable/incompatible transport.
    int syncopationPoll( Request &slot /**< [out] Immutable target and source-ready deadline. */,
                         IMAGE   &source /**< [in] Existing CPU-backed 2-D float32 source. */ );

    /// Wait to the configured lead, revalidate, and publish a complete tagged payload before its cutoff.
    bool syncopationPublish( const float   *pixels /**< [in] Complete private source image. */,
                             const Request &slot /**< [in] Request previously returned by poll. */,
                             IMAGE         &source /**< [in,out] Managed source; never modify elsewhere. */ );

    /// Release transport attachments without clearing source values or synthesizing DM triggers.
    void syncopationClose();
};

template <class derivedT>
derivedT &dmSyncopationProducer<derivedT>::derived()
{
    return *static_cast<derivedT *>( this );
}

template <class derivedT>
dmSyncopationProducer<derivedT>::~dmSyncopationProducer()
{
    syncopationClose();
}

template <class derivedT>
int dmSyncopationProducer<derivedT>::setupConfig( mx::app::appConfigurator &config )
{
    config.add( "syncopation.timingStream",
                "",
                "syncopation.timingStream",
                mx::app::argType::Required,
                "syncopation",
                "timingStream",
                false,
                "string",
                "Same-host tagged dmSyncopation schedule." );
    config.add( "syncopation.writeLeadUs",
                "",
                "syncopation.writeLeadUs",
                mx::app::argType::Required,
                "syncopation",
                "writeLeadUs",
                false,
                "real",
                "Producer completion lead before its input cutoff, us." );
    return 0;
}

template <class derivedT>
int dmSyncopationProducer<derivedT>::loadConfig( mx::app::appConfigurator &config )
{
    config( m_syncopationTiming, "syncopation.timingStream" );
    config( m_writeLeadUs, "syncopation.writeLeadUs" );

    if( !mx::math::isFinite( m_writeLeadUs ) || m_writeLeadUs <= 0 || m_writeLeadUs > 1000000 ||
        m_syncopationTiming.find_first_of( "/\\ \t\n" ) != std::string::npos )
    {
        return -1;
    }

    return 0;
}

template <class derivedT>
bool dmSyncopationProducer<derivedT>::syncopationIdentity( const IMAGE &image )
{
    char        filename[4096];
    struct stat info{};
    return image.md && image.md->sem > 0 &&
           ImageStreamIO_filename( filename, sizeof( filename ), image.md->name ) == 0 &&
           stat( filename, &info ) == 0 && static_cast<uint64_t>( info.st_ino ) == image.md->inode;
}

template <class derivedT>
bool dmSyncopationProducer<derivedT>::syncopationGeometry( const IMAGE &image )
{
    if( !image.md || !image.array.F || image.md->datatype != _DATATYPE_FLOAT || image.md->naxis != 2 ||
        image.md->location != -1 || image.md->size[0] == 0 || image.md->size[1] == 0 || image.md->sem == 0 ||
        image.md->inode == 0 )
    {
        return false;
    }

    const uint64_t pixels = uint64_t( image.md->size[0] ) * uint64_t( image.md->size[1] );
    return pixels <= 1048576 && image.md->nelement == pixels && image.md->imdatamemsize == pixels * sizeof( float );
}

template <class derivedT>
bool dmSyncopationProducer<derivedT>::syncopationSource( const IMAGE &image ) const
{
    return syncopationGeometry( image ) && image.md->inode == m_syncopationSourceInode &&
           image.md->nelement == m_syncopationPixels && image.md->size[0] == m_syncopationWidth &&
           image.md->size[1] == m_syncopationHeight;
}

template <class derivedT>
bool dmSyncopationProducer<derivedT>::syncopationSame( const Request &first, const Request &second )
{
    return first.epoch == second.epoch && first.generation == second.generation && first.cycle == second.cycle &&
           first.step == second.step && first.targetNs == second.targetNs && first.cutoffNs == second.cutoffNs;
}

template <class derivedT>
bool dmSyncopationProducer<derivedT>::syncopationOpen( IMAGE &source )
{
    if( m_syncopationTiming.empty() || !syncopationGeometry( source ) )
    {
        return false;
    }

    const std::string sourceName( source.md->name );

    if( sourceName.size() < 4 || sourceName.substr( sourceName.size() - 4 ) != "_aux" ||
        sourceName == m_syncopationTiming )
    {
        return false;
    }

    char filename[4096];

    if( ImageStreamIO_filename( filename, sizeof( filename ), sourceName.c_str() ) != 0 )
    {
        return false;
    }

    const std::string lockName = std::string( filename ) + ".producer";
    m_syncopationLock          = open( lockName.c_str(), O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0660 );

    if( m_syncopationLock < 0 || flock( m_syncopationLock, LOCK_EX | LOCK_NB ) != 0 ||
        !synco::openRecord( m_syncopationSchedule, m_syncopationTiming, synco::words, false ) ||
        !synco::openRecord( m_syncopationAck, sourceName + "_ack", synco::ackWords, false ) ||
        !synco::openRecord( m_syncopationReady, sourceName + "_ready", synco::commitWords, true ) )
    {
        syncopationClose();
        return false;
    }

    m_syncopationSession     = synco::monotonicNs();
    m_syncopationSequence    = 0;
    m_syncopationFileCheck   = 0;
    m_syncopationSourceInode = source.md->inode;
    m_syncopationPixels      = source.md->nelement;
    m_syncopationWidth       = source.md->size[0];
    m_syncopationHeight      = source.md->size[1];

    if( m_syncopationSession <= 0 )
    {
        syncopationClose();
        return false;
    }

    synco::CommitRecord prior{};
    synco::AckRecord    ack{};

    if( synco::readRecord( m_syncopationReady, prior ) && synco::validCommand( prior, m_syncopationSession ) &&
        !source.md->write && source.md->cnt0 == static_cast<uint64_t>( prior[synco::cmdPayloadCounter] ) &&
        source.md->inode == static_cast<uint64_t>( prior[synco::cmdPayloadInode] ) &&
        ( !synco::readRecord( m_syncopationAck, ack ) || !synco::acknowledges( ack, prior ) ) )
    {
        // Reacquiring the producer lock does not revoke a consumer's outstanding payload ownership.
        m_syncopationPending     = prior;
        m_syncopationOutstanding = true;
        m_syncopationLast        = { prior[synco::cmdEpoch],
                                     prior[synco::cmdGeneration],
                                     prior[synco::cmdCycle],
                                     int( prior[synco::cmdStep] ),
                                     prior[synco::cmdTarget],
                                     prior[synco::cmdCutoff] };
    }

    return true;
}

template <class derivedT>
int dmSyncopationProducer<derivedT>::syncopationPoll( Request &slot, IMAGE &source )
{
    if( m_syncopationSchedule.md && !syncopationSource( source ) )
    {
        syncopationClose();
        return -1;
    }

    if( !m_syncopationSchedule.md && !syncopationOpen( source ) )
    {
        return -1;
    }

    const int64_t now = synco::monotonicNs();

    if( now >= m_syncopationFileCheck )
    {
        m_syncopationFileCheck = now + 1000000000;

        if( !syncopationIdentity( source ) || !syncopationIdentity( m_syncopationSchedule ) ||
            !syncopationIdentity( m_syncopationReady ) || !syncopationIdentity( m_syncopationAck ) )
        {
            syncopationClose();
            return -1;
        }
    }

    synco::Record record{};

    if( !synco::readRecord( m_syncopationSchedule, record ) || !synco::validProductionRecord( record, now ) )
    {
        return 0;
    }

    if( m_syncopationOutstanding )
    {
        synco::AckRecord ack{};
        const bool       reset = record[synco::epoch] != m_syncopationPending[synco::cmdEpoch] ||
                           record[synco::generation] != m_syncopationPending[synco::cmdGeneration];

        if( !reset &&
            ( !synco::readRecord( m_syncopationAck, ack ) || !synco::acknowledges( ack, m_syncopationPending ) ) )
        {
            return 0;
        }

        m_syncopationOutstanding = false;
    }

    const int step = synco::nextProductionStep( record, now );

    if( step < 0 )
    {
        return 0;
    }

    slot = synco::request( record, step );

    if( syncopationSame( slot, m_syncopationLast ) )
    {
        return 0;
    }

    m_syncopationLast = slot;
    return 1;
}

template <class derivedT>
bool dmSyncopationProducer<derivedT>::syncopationPublish( const float *pixels, const Request &slot, IMAGE &source )
{
    if( !pixels || !syncopationSource( source ) || !m_syncopationSchedule.md || !m_syncopationReady.md ||
        m_syncopationOutstanding || !syncopationSame( slot, m_syncopationLast ) || derived().m_shutdown )
    {
        return false;
    }

    const int64_t publishAt = slot.cutoffNs - static_cast<int64_t>( std::ceil( m_writeLeadUs * 1000 ) );
    int64_t       now       = synco::monotonicNs();

    while( now < publishAt && !derived().m_shutdown )
    {
        const auto wake   = synco::toTimespec( std::min( publishAt, now + 1000000 ) );
        const int  result = clock_nanosleep( CLOCK_MONOTONIC, TIMER_ABSTIME, &wake, nullptr );

        if( result != 0 && result != EINTR )
        {
            return false;
        }

        now = synco::monotonicNs();
    }

    synco::Record current{};

    if( derived().m_shutdown || now <= 0 || now >= slot.cutoffNs || !syncopationSource( source ) || source.md->write ||
        !synco::readRecord( m_syncopationSchedule, current ) || !synco::validProductionRecord( current, now ) ||
        slot.step < 0 || slot.step >= current[synco::steps] ||
        !syncopationSame( slot, synco::request( current, slot.step ) ) ||
        source.md->cnt0 >= static_cast<uint64_t>( std::numeric_limits<int64_t>::max() ) ||
        source.md->inode > static_cast<uint64_t>( std::numeric_limits<int64_t>::max() ) )
    {
        return false;
    }

    for( uint64_t j = 0; j < m_syncopationPixels; ++j )
    {
        if( !mx::math::isFinite( pixels[j] ) )
        {
            return false;
        }
    }

    const auto expected = source.md->cnt0 + 1;
    source.md->write    = 1;
    std::memcpy( source.array.F, pixels, m_syncopationPixels * sizeof( float ) );
    std::atomic_thread_fence( std::memory_order_release );

    if( ImageStreamIO_UpdateIm( &source ) != 0 || source.md->write || source.md->cnt0 != expected )
    {
        return false;
    }

    const int64_t completed = synco::monotonicNs();

    if( completed <= 0 || completed > slot.cutoffNs || m_syncopationSequence == std::numeric_limits<int64_t>::max() )
    {
        return false;
    }

    m_syncopationPending = { synco::productionVersion,
                             slot.epoch,
                             slot.generation,
                             slot.cycle,
                             slot.step,
                             slot.targetNs,
                             slot.cutoffNs,
                             static_cast<int64_t>( expected ),
                             completed,
                             m_syncopationSession,
                             ++m_syncopationSequence,
                             static_cast<int64_t>( getpid() ),
                             static_cast<int64_t>( source.md->inode ) };

    if( !synco::writeRecord( m_syncopationReady, m_syncopationPending ) )
    {
        return false;
    }

    m_syncopationOutstanding = true;
    return true;
}

template <class derivedT>
void dmSyncopationProducer<derivedT>::syncopationClose()
{
    for( auto *image : { &m_syncopationSchedule, &m_syncopationReady, &m_syncopationAck } )
    {
        if( image->md )
        {
            ImageStreamIO_closeIm( image );
            *image = {};
        }
    }

    if( m_syncopationLock >= 0 )
    {
        close( m_syncopationLock );
        m_syncopationLock = -1;
    }

    m_syncopationPending     = {};
    m_syncopationLast        = {};
    m_syncopationOutstanding = false;
    m_syncopationSourceInode = 0;
    m_syncopationPixels      = 0;
    m_syncopationWidth       = 0;
    m_syncopationHeight      = 0;
}

/// Register producer configuration through the app's dmSyncopationProducerT alias.
#define DMSYNCOPATIONPRODUCER_SETUP_CONFIG( cfig )                                                                     \
    if( dmSyncopationProducerT::setupConfig( cfig ) < 0 )                                                              \
    {                                                                                                                  \
        m_shutdown = true;                                                                                             \
    }

/// Load producer configuration from an int-returning application loadConfigImpl.
#define DMSYNCOPATIONPRODUCER_LOAD_CONFIG( cfig )                                                                      \
    if( dmSyncopationProducerT::loadConfig( cfig ) < 0 )                                                               \
    {                                                                                                                  \
        return -1;                                                                                                     \
    }

/// Close transport attachments after the application's producer thread is joined.
#define DMSYNCOPATIONPRODUCER_APP_SHUTDOWN dmSyncopationProducerT::syncopationClose();

} // namespace dev
} // namespace app
} // namespace MagAOX
#endif
