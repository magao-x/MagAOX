/** \file streamCircBuff.hpp
 * \brief The MagAO-X streamCircBuff app header file
 *
 * \ingroup streamCircBuff_files
 */

#ifndef streamCircBuff_hpp
#define streamCircBuff_hpp

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

#ifdef __SSE2__
    #include <immintrin.h>
#endif

/** \defgroup streamCircBuff
 * \brief An application to keep a circular buffer of a stream
 *
 * <a href="../handbook/operating/software/apps/streamCircBuff.html">Application Documentation</a>
 *
 * \ingroup apps
 *
 */

/** \defgroup streamCircBuff_files
 * \ingroup streamCircBuff
 */

namespace MagAOX
{
namespace app
{

/// Class for application to keep a circular buffer of a stream and publish it to another stream
/** Each source frame is copied as it arrives, and then published to the circular buffer in the source data
 * type with the source acquisition time, write time and frame counter.  Readers of the circular buffer
 * therefore see the same timing as readers of the source.  The published counter is offset if needed so
 * that it keeps increasing when the source restarts.  A single-frame source frame which is rewritten during
 * the copy is dropped, and the number of such frames is logged.  Each frame's timing is also kept in the
 * per-frame arrays, so a reader using dev::shmimMonitor gets the timing of the frame it is processing from
 * `currAtime()`, `currWritetime()` and `currCnt0()`, even after newer frames are published.
 *
 * The output keeps the source data type unless `circBuff.toFloat` is set, e.g. for modalPSDs, which reads
 * float.  The frame rate is read from the `circBuff.fpsDevice` property if one is configured, and otherwise
 * measured from the source acquisition times and counters.  It changes only by more than `circBuff.fpsTol`,
 * and is published as `fps.current`, so it can be used as the fps source of readers such as modalPSDs.
 *
 * \ingroup streamCircBuff
 */
class streamCircBuff : public MagAOXApp<true>,
                       public dev::shmimMonitor<streamCircBuff>,
                       public dev::frameGrabber<streamCircBuff>,
                       public dev::telemeter<streamCircBuff>
{
    friend class dev::shmimMonitor<streamCircBuff>;
    friend class dev::frameGrabber<streamCircBuff>;
    friend class dev::telemeter<streamCircBuff>;

  public:
    /** \name app::dev Configurations
     *@{
     */

    static constexpr bool c_frameGrabber_flippable =
        false; ///< app:dev config to tell framegrabber these images can not be flipped

    ///@}

    /// The base shmimMonitor type
    typedef dev::shmimMonitor<streamCircBuff> shmimMonitorT;

    /// The base frameGrabber type
    typedef dev::frameGrabber<streamCircBuff> frameGrabberT;

    /// The telemeter type
    typedef dev::telemeter<streamCircBuff> telemeterT;

  protected:
    /** \name Configurable Parameters
     *@{
     */

    bool m_toFloat{ false }; ///< Convert the output to float, for readers such as modalPSDs which require it.

    std::string m_fpsDevice; ///< Device providing the frame rate.  If empty, the source frame rate is measured.

    std::string m_fpsProperty{ "fps" }; ///< Property providing the frame rate.

    std::string m_fpsElement{ "current" }; ///< Element providing the frame rate.

    float m_fpsTol{ 0.01 }; ///< Fractional change in the frame rate needed to update fps.

    ///@}

    sem_t m_smSemaphore{ 0 }; ///< Semaphore used to synchronize the fg thread and the sm thread.

    std::mutex
        m_stageMutex; ///< Protects the staged frame and its timing, which the sm thread writes and the fg thread reads.

    std::vector<char> m_staged; ///< Copy of the latest source frame, written by the sm thread.

    timespec m_stagedAtime{ 0, 0 }; ///< Source acquisition time of the staged frame.

    timespec m_stagedWritetime{ 0, 0 }; ///< Source write time of the staged frame.

    uint64_t m_stagedCnt0{ 0 }; ///< Source frame counter of the staged frame.

    bool m_stagedFresh{ false }; ///< True when the staged frame has not been published yet.

    timespec m_relayWritetime{ 0, 0 }; ///< Source write time of the frame being published, used only by the fg thread.

    uint64_t m_relayCnt0{ 0 }; ///< Source frame counter of the frame being published, used only by the fg thread.

    uint64_t m_cnt0Offset{
        0 }; ///< Added to the source frame counter so the published counter keeps increasing after a source restart.

    std::atomic<uint64_t> m_rewritten{
        0 }; ///< Number of source frames dropped because the source was rewritten during the copy.

    uint64_t m_rewrittenLogged{ 0 }; ///< Value of m_rewritten when last logged.

    float ( *pixget )( void *, size_t ){ nullptr }; ///< Converts a staged source pixel to float when m_toFloat is set.

    std::atomic<float> m_fps{ 0 }; ///< Source frame rate, from the fps device or measured, 0 until known.

    timespec m_fpsAtime{ 0, 0 }; ///< Source acquisition time at the start of the current rate measurement (sm thread).

    uint64_t m_fpsCnt0{ 0 }; ///< Source frame counter at the start of the current rate measurement (sm thread).

    pcf::IndiProperty m_indiP_fps; ///< The source frame rate, as fps.current.

    pcf::IndiProperty m_indiP_fpsSource; ///< The frame rate property of the fps device.

  public:
    INDI_SETCALLBACK_DECL( streamCircBuff, m_indiP_fpsSource );

  protected:
  public:
    /// Default c'tor.
    streamCircBuff();

    /// D'tor, declared and defined for noexcept.
    ~streamCircBuff() noexcept
    {
    }

    /// Setup the configuration of the shmimMonitor, frameGrabber and telemeter
    virtual void setupConfig();

    /// Implementation of loadConfig logic, separated for testing.
    /** This is called by loadConfig().
     */
    int loadConfigImpl(
        mx::app::appConfigurator &_config /**< [in] an application configuration from which to load values*/ );

    /// Load the configuration, calling loadConfigImpl
    virtual void loadConfig();

    /// Startup function
    /**
     * \returns 0 on success
     * \returns -1 on error
     */
    virtual int appStartup();

    /// Implementation of the FSM for streamCircBuff.
    /** Also logs the number of source frames dropped since the last call.
     *
     * \returns 0 on no critical error
     * \returns -1 on an error requiring shutdown
     */
    virtual int appLogic();

    /// Shutdown the app.
    /**
     * \returns 0
     */
    virtual int appShutdown();

  protected:
    /// Size the stage for a new source and request a framegrabber reconfiguration
    /**
     * \returns 0
     */
    int allocate( const dev::shmimT &dummy /**< [in] tag to differentiate shmimMonitor parents.*/ );

    /// Copy the current source frame and its timing to the stage, and wake the fg thread
    /** The frame is dropped if the source was written during the copy.
     *
     * \returns 0 on success
     * \returns -1 on error
     */
    int processImage( void              *curr_src, /**< [in] pointer to start of current frame. */
                      const dev::shmimT &dummy     /**< [in] tag to differentiate shmimMonitor parents. */
    );

    /// Update the measured source frame rate with a staged frame, when no fps device is configured
    /** The rate is measured from the source acquisition times and counters over at least 1 second, so dropped
     * frames do not bias it, and m_fps is only changed when the rate changes by more than m_fpsTol.
     */
    void measureFPS( const timespec &atime, /**< [in] source acquisition time of the frame */
                     uint64_t        cnt0   /**< [in] source frame counter of the frame */
    );

    /// Copy a frame into the output stream with non-temporal stores
    /** The output is a long circular buffer which is rarely read back, so the copy bypasses the cache rather than
     * evicting the data of processes sharing the L3 cache, such as the loop reading the source.  Ends with a store
     * fence so the frame is visible before it is published.  Uses memcpy where SSE2 is not available.
     */
    static void streamCopy( void       *dest,  /**< [out] start of the output frame */
                            const void *src,   /**< [in] start of the staged frame */
                            size_t      nbytes /**< [in] number of bytes to copy */
    );

    /** \name dev::frameGrabber interface
     *
     * @{
     */

    /// Implementation of the framegrabber configureAcquisition interface
    /** Sets the output size to that of the source, and the data type to that of the source or to float.
     *
     * \returns 0 on success
     * \returns -1 on error
     */
    int configureAcquisition();

    /// Implementation of the framegrabber fps interface
    /**
     * \returns the source frame rate, from the fps device or measured, or 0 before it is known
     */
    float fps()
    {
        return m_fps;
    }

    /// Implementation of the framegrabber startAcquisition interface
    /**
     * \returns 0 on success
     * \returns -1 on error
     */
    int startAcquisition();

    /// Implementation of the framegrabber acquireAndCheckValid interface
    /** Waits up to 1 second for a staged frame which has not been published yet.
     *
     * \returns 0 if a frame is ready
     * \returns 1 if no frame is ready
     */
    int acquireAndCheckValid();

    /// Implementation of the framegrabber loadImageIntoStream interface
    /** Copies the staged frame, converting it to float if configured, and sets the acquisition time, write time
     * and counter to publish.
     *
     * \returns 0 on success
     * \returns -1 if the stage no longer matches the output size or type, so the framegrabber reconfigures
     */
    int loadImageIntoStream( void *dest /**< [out] the next frame of the output stream */ );

    /// Implementation of the framegrabber reconfig interface
    /**
     * \returns 0 on success
     * \returns -1 on error
     */
    int reconfig();

    /// Supply the source write time and frame counter of the frame being published
    /** Called by the framegrabber after loadImageIntoStream.  The published counter is offset so it keeps
     * increasing if the source counter restarts.
     *
     * \returns 0
     */
    int
    frameGrabberRelayTiming( timespec &writetime, /**< [out] the source write time */
                             uint64_t &cnt0 /**< [in/out] on entry the next counter, on exit the counter to publish */
    );

    ///@}

    /** \name Telemeter Interface
     *
     * @{
     */
    /// Check whether the framegrabber timing telemetry is due
    /**
     * \returns 0 on success
     * \returns -1 on error
     */
    int checkRecordTimes();

    /// Record the framegrabber timing telemetry
    /**
     * \returns 0 on success
     * \returns -1 on error
     */
    int recordTelem( const telem_fgtimings * /**< [in] tag to select the telemetry type */ );

    ///@}
};

streamCircBuff::streamCircBuff() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{

    return;
}

void streamCircBuff::setupConfig()
{
    SHMIMMONITOR_SETUP_CONFIG( config );

    FRAMEGRABBER_SETUP_CONFIG( config );

    TELEMETER_SETUP_CONFIG( config );

    config.add( "circBuff.toFloat",
                "",
                "circBuff.toFloat",
                argType::Required,
                "circBuff",
                "toFloat",
                false,
                "bool",
                "Convert the output to float, e.g. for modalPSDs.  Default is false, keeping the source data type." );

    config.add( "circBuff.fpsDevice",
                "",
                "circBuff.fpsDevice",
                argType::Required,
                "circBuff",
                "fpsDevice",
                false,
                "string",
                "Device providing the frame rate.  If not set, the source frame rate is measured." );

    config.add( "circBuff.fpsProperty",
                "",
                "circBuff.fpsProperty",
                argType::Required,
                "circBuff",
                "fpsProperty",
                false,
                "string",
                "Property providing the frame rate.  Default is 'fps'." );

    config.add( "circBuff.fpsElement",
                "",
                "circBuff.fpsElement",
                argType::Required,
                "circBuff",
                "fpsElement",
                false,
                "string",
                "Element providing the frame rate.  Default is 'current'." );

    config.add( "circBuff.fpsTol",
                "",
                "circBuff.fpsTol",
                argType::Required,
                "circBuff",
                "fpsTol",
                false,
                "float",
                "Fractional change in the frame rate needed to update fps.  Default is 0.01." );
}

int streamCircBuff::loadConfigImpl( mx::app::appConfigurator &_config )
{
    SHMIMMONITOR_LOAD_CONFIG( _config );

    FRAMEGRABBER_LOAD_CONFIG( _config );

    TELEMETER_LOAD_CONFIG( config );

    _config( m_toFloat, "circBuff.toFloat" );
    _config( m_fpsDevice, "circBuff.fpsDevice" );
    _config( m_fpsProperty, "circBuff.fpsProperty" );
    _config( m_fpsElement, "circBuff.fpsElement" );
    _config( m_fpsTol, "circBuff.fpsTol" );

    return 0;
}

void streamCircBuff::loadConfig()
{
    loadConfigImpl( config );
}

int streamCircBuff::appStartup()
{
    if( sem_init( &m_smSemaphore, 0, 0 ) < 0 )
    {
        log<software_critical>( { __FILE__, __LINE__, errno, 0, "Initializing S.M. semaphore" } );
        return -1;
    }

    SHMIMMONITOR_APP_STARTUP;

    FRAMEGRABBER_APP_STARTUP;

    TELEMETER_APP_STARTUP;

    CREATE_REG_INDI_RO_NUMBER( m_indiP_fps, "fps", "Source Frame Rate", "Circular Buffer" );
    m_indiP_fps.add( pcf::IndiElement( "current" ) );
    m_indiP_fps["current"] = 0;

    if( m_fpsDevice != "" )
    {
        REG_INDI_SETPROP( m_indiP_fpsSource, m_fpsDevice, m_fpsProperty );
    }

    state( stateCodes::OPERATING );

    return 0;
}

int streamCircBuff::appLogic()
{
    SHMIMMONITOR_APP_LOGIC;

    FRAMEGRABBER_APP_LOGIC;

    TELEMETER_APP_LOGIC;

    uint64_t rewritten = m_rewritten;
    if( rewritten != m_rewrittenLogged )
    {
        log<text_log>( "Source frames dropped because they were rewritten during the copy: " +
                           std::to_string( rewritten - m_rewrittenLogged ),
                       logPrio::LOG_WARNING );
        m_rewrittenLogged = rewritten;
    }

    std::unique_lock<std::mutex> lock( m_indiMutex );

    SHMIMMONITOR_UPDATE_INDI;

    FRAMEGRABBER_UPDATE_INDI;

    updateIfChanged( m_indiP_fps, "current", m_fps.load() );

    return 0;
}

int streamCircBuff::appShutdown()
{
    SHMIMMONITOR_APP_SHUTDOWN;
    FRAMEGRABBER_APP_SHUTDOWN;
    TELEMETER_APP_SHUTDOWN;

    return 0;
}

int streamCircBuff::allocate( const dev::shmimT &dummy )
{
    static_cast<void>( dummy );

    { //mutex scope
        std::lock_guard<std::mutex> lock( m_stageMutex );
        m_staged.resize( shmimMonitorT::m_width * shmimMonitorT::m_height * shmimMonitorT::m_typeSize );
        m_stagedFresh = false;
        pixget        = getPixPointer<float>( shmimMonitorT::m_dataType );
    }

    m_reconfig = true;

    return 0;
}

int streamCircBuff::processImage( void *curr_src, const dev::shmimT &dummy )
{
    static_cast<void>( dummy );

    IMAGE &src = shmimMonitorT::m_imageStream;

    { //mutex scope
        std::lock_guard<std::mutex> lock( m_stageMutex );

        size_t frameBytes = m_staged.size();

        if( shmimMonitorT::m_depth > 1 && src.cntarray != nullptr )
        {
            // A multi-frame source does not rewrite this frame until it wraps around
            size_t slot = ( static_cast<char *>( curr_src ) - static_cast<char *>( src.array.raw ) ) / frameBytes;

            memcpy( m_staged.data(), curr_src, frameBytes );
            m_stagedAtime     = src.atimearray[slot];
            m_stagedWritetime = src.writetimearray[slot];
            m_stagedCnt0      = src.cntarray[slot];
        }
        else
        {
            // The source sets write before it writes, and increments cnt0 before clearing write
            uint64_t cnt0  = src.md->cnt0;
            int      write = src.md->write;
            std::atomic_thread_fence( std::memory_order_acquire );

            memcpy( m_staged.data(), curr_src, frameBytes );
            m_stagedAtime     = src.md->atime;
            m_stagedWritetime = src.md->writetime;

            std::atomic_thread_fence( std::memory_order_acquire );
            if( write != 0 || src.md->write != 0 || src.md->cnt0 != cnt0 )
            {
                m_stagedFresh = false;
                ++m_rewritten;
                return 0;
            }

            m_stagedCnt0 = cnt0;
        }

        m_stagedFresh = true;

        if( m_fpsDevice == "" )
        {
            measureFPS( m_stagedAtime, m_stagedCnt0 );
        }
    }

    // Now tell the f.g. to get going
    if( sem_post( &m_smSemaphore ) < 0 )
    {
        log<software_critical>( { __FILE__, __LINE__, errno, 0, "Error posting to semaphore" } );
        return -1;
    }

    return 0;
}

void streamCircBuff::measureFPS( const timespec &atime, uint64_t cnt0 )
{
    double dt = ( atime.tv_sec - m_fpsAtime.tv_sec ) + ( atime.tv_nsec - m_fpsAtime.tv_nsec ) / 1e9;

    if( m_fpsCnt0 == 0 || cnt0 <= m_fpsCnt0 || dt <= 0 )
    {
        // start a measurement, or restart it after a source restart
        m_fpsAtime = atime;
        m_fpsCnt0  = cnt0;
        return;
    }

    if( dt < 1 )
    {
        return;
    }

    float fps = ( cnt0 - m_fpsCnt0 ) / dt;
    if( std::fabs( fps - m_fps ) > m_fpsTol * fps )
    {
        m_fps = fps;
    }

    m_fpsAtime = atime;
    m_fpsCnt0  = cnt0;
}

void streamCircBuff::streamCopy( void *dest, const void *src, size_t nbytes )
{
#ifdef __SSE2__
    char       *d = static_cast<char *>( dest );
    const char *s = static_cast<const char *>( src );

    // regular stores up to the first 16 byte boundary of the destination
    size_t head = std::min( ( 16 - reinterpret_cast<uintptr_t>( d ) % 16 ) % 16, nbytes );

    memcpy( d, s, head );
    d += head;
    s += head;
    nbytes -= head;

    for( ; nbytes >= 16; nbytes -= 16, d += 16, s += 16 )
    {
        _mm_stream_si128( reinterpret_cast<__m128i *>( d ), _mm_loadu_si128( reinterpret_cast<const __m128i *>( s ) ) );
    }

    memcpy( d, s, nbytes );

    _mm_sfence();
#else
    memcpy( dest, src, nbytes );
#endif
}

int streamCircBuff::configureAcquisition()
{
    std::unique_lock<std::mutex> lock( m_indiMutex );

    ///\todo potential but verrrrry unlikely bug: shmimMonitorT could change these before allocate sets the lock above.
    /// Should use a local set of w/h instead.
    if( shmimMonitorT::m_width == 0 || shmimMonitorT::m_height == 0 )
    {
        // This means we haven't connected to the stream to accumulate. so wait.
        lock.unlock(); // don't hold the lock for a whole second.
        sleep( 1 );
        return -1;
    }

    frameGrabberT::m_width    = shmimMonitorT::m_width;
    frameGrabberT::m_height   = shmimMonitorT::m_height;
    frameGrabberT::m_dataType = m_toFloat ? _DATATYPE_FLOAT : shmimMonitorT::m_dataType;

    return 0;
}

int streamCircBuff::startAcquisition()
{
    return 0;
}

int streamCircBuff::acquireAndCheckValid()
{
    timespec ts;

    if( clock_gettime( CLOCK_REALTIME, &ts ) < 0 )
    {
        log<software_critical>( { __FILE__, __LINE__, errno, 0, "clock_gettime" } );
        return -1;
    }

    ts.tv_sec += 1;

    if( sem_timedwait( &m_smSemaphore, &ts ) != 0 )
    {
        return 1;
    }

    // A frame already published, e.g. after several posts, is not published again
    std::lock_guard<std::mutex> lock( m_stageMutex );
    return m_stagedFresh ? 0 : 1;
}

int streamCircBuff::loadImageIntoStream( void *dest )
{
    std::lock_guard<std::mutex> lock( m_stageMutex );

    size_t npix = frameGrabberT::m_width * frameGrabberT::m_height;

    if( m_staged.size() != npix * shmimMonitorT::m_typeSize )
    {
        return -1; // the source changed, so reconfigure
    }

    if( frameGrabberT::m_dataType == shmimMonitorT::m_dataType )
    {
        streamCopy( dest, m_staged.data(), m_staged.size() );
    }
    else if( frameGrabberT::m_dataType == _DATATYPE_FLOAT && pixget != nullptr )
    {
        float *fdest = static_cast<float *>( dest );
        for( size_t n = 0; n < npix; ++n )
        {
            fdest[n] = pixget( m_staged.data(), n );
        }
    }
    else
    {
        return -1; // the source type changed, so reconfigure
    }

    m_currImageTimestamp = m_stagedAtime;
    m_relayWritetime     = m_stagedWritetime;
    m_relayCnt0          = m_stagedCnt0;
    m_stagedFresh        = false;

    return 0;
}

int streamCircBuff::reconfig()
{
    return 0;
}

int streamCircBuff::frameGrabberRelayTiming( timespec &writetime, uint64_t &cnt0 )
{
    writetime = m_relayWritetime;

    if( m_relayCnt0 + m_cnt0Offset < cnt0 )
    {
        m_cnt0Offset = cnt0 - m_relayCnt0; // the source counter restarted, or the output stream is newer than the
                                           // source
    }

    cnt0 = m_relayCnt0 + m_cnt0Offset;

    return 0;
}

INDI_SETCALLBACK_DEFN( streamCircBuff, m_indiP_fpsSource )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_fpsSource, ipRecv );

    if( ipRecv.find( m_fpsElement ) != true ) // this isn't valid
    {
        return 0;
    }

    float fps = ipRecv[m_fpsElement].get<float>();

    if( std::fabs( fps - m_fps ) > m_fpsTol * fps )
    {
        m_fps = fps;
    }

    return 0;
}

int streamCircBuff::checkRecordTimes()
{
    return telemeterT::checkRecordTimes( telem_fgtimings() );
}

int streamCircBuff::recordTelem( const telem_fgtimings * )
{
    return recordFGTimings( true );
}

} // namespace app
} // namespace MagAOX

#endif // streamCircBuff_hpp
