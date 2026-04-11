/** \file mcp3208Ctrl.hpp
 * \brief The MagAO-X mcp3208 Controller header file
 *
 * \ingroup mcp3208Ctrl_files
 */

#ifndef mcp3208Ctrl_hpp
#define mcp3208Ctrl_hpp

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"
#include "dependencies/MCP3208.h" // Included for the MCP3208 device interface

/** \defgroup mcp3208Ctrl
 * \brief The MagAO-X application to readout a mcp3208 A/D on a raspberry Pi.
 *
 * <a href="../handbook/operating/software/apps/XXXXXX.html">Application Documentation</a>
 *
 * \ingroup apps
 *
 */

/** \defgroup mcp3208Ctrl_files
 * \ingroup mcp3208Ctrl
 */

namespace MagAOX
{
namespace app
{

/** MagAO-X application to read MCP3208 channels on a Raspberry Pi.
 *
 * The controller can either acquire on its internal timer loop or synchronize reads to an
 * ImageStreamIO semaphore.
 *
 * \ingroup mcp3208Ctrl
 */
class mcp3208Ctrl : public MagAOXApp<true>, public dev::frameGrabber<mcp3208Ctrl>, public dev::telemeter<mcp3208Ctrl>
{

    // Give the test harness access.
    friend class mcp3208Ctrl_test;
    friend class dev::frameGrabber<mcp3208Ctrl>;
    friend class dev::telemeter<mcp3208Ctrl>;

    typedef dev::frameGrabber<mcp3208Ctrl> frameGrabberT;
    typedef dev::telemeter<mcp3208Ctrl>    telemeterT;

    /// The active MCP3208 device interface used for hardware access.
    MCP3208Lib::MCP3208 m_adc;

    static constexpr bool c_frameGrabber_flippable = false; /**< app:dev config indicating these images can not be
                                                                 flipped */

  protected:
    /** \name Configurable Parameters - Data
     * @{
     */

    int m_numChannels{ 4 }; ///< The number of MCP3208 channels read into each output frame.

    std::string m_fpsDevice;               ///< Device name providing external fps metadata for framegrabber sizing.
    std::string m_fpsProperty{ "fps" };    ///< Property name providing external fps metadata.
    std::string m_fpsElement{ "current" }; ///< Property element containing the fps value.

    float m_fpsTol{ 0 }; ///< The tolerance used when monitoring fps metadata changes.

    std::string m_synchroShmimName;      ///< The synchronization ImageStreamIO stream name; empty selects timer mode.
    int         m_synchroPostDelay{ 0 }; ///< Requested delay between semaphore wake and A/D read in microseconds.

    ///@}

    /** \name Runtime State - Data
     * @{
     */

    /// INDI property exposing the local fps target.
    pcf::IndiProperty m_indiP_fps;

    /// Handle updates to the local fps target property.
    INDI_NEWCALLBACK_DECL( mcp3208Ctrl, m_indiP_fps );

    float m_fps{ 2000 }; ///< The target acquisition rate in frames per second.

    /// INDI property subscription used to follow an external fps source.
    pcf::IndiProperty m_indiP_fpsSource;

    /// Handle updates from the configured external fps source.
    INDI_SETCALLBACK_DECL( mcp3208Ctrl, m_indiP_fpsSource );

    float m_trigger{ 1e9f / m_fps };       ///< The timer-mode read interval in nanoseconds.
    float m_gain{ .1 };                    ///< The simple integrator gain used for timer and synchro delay control.
    float nano_sec_target{ 1e9f / m_fps }; ///< The timer-mode target interval in nanoseconds.
    float m_synchroDelay{ 0 };             ///< The controlled pre-read delay in synchronized mode, in nanoseconds.
    float m_synchroDelayTarget{ 0 };       ///< The target pre-read delay in synchronized mode, in nanoseconds.

    /// Secondary MCP3208 handle retained with the legacy class state.
    MCP3208Lib::MCP3208 adc;

    /// The timer-mode reference point for the next internal acquisition cycle.
    std::chrono::time_point<std::chrono::high_resolution_clock> m_time_start;

    /// The most recently read MCP3208 channel values published to the output stream.
    std::vector<uint16_t> m_values;

    IMAGE  m_synchroStream{};             ///< The opened synchronization stream used for semaphore-triggered reads.
    bool   m_synchroStreamOpen{ false };  ///< Tracks whether the synchronization stream is currently open.
    ino_t  m_synchroStreamInode{ 0 };     ///< Cached inode used to detect synchronization stream recreation.
    int    m_synchroSemaphoreNumber{ 5 }; ///< The claimed semaphore slot for synchronization waits.
    sem_t *m_synchroSemaphore{ nullptr }; ///< Cached pointer to the claimed synchronization semaphore.

    ///@}

    /** \name Synchronization Helpers
     * @{
     */

    /// Open the synchronization stream when semaphore-driven acquisition is enabled.
    /**
     * \returns 0 on success.
     * \returns 1 when the synchronization stream is not yet ready.
     */
    int openSynchroStream();

    /// Claim a semaphore slot from the synchronization stream.
    /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
    int claimSynchroSemaphore();

    /// Check whether the synchronization stream has disappeared or been recreated.
    /**
     * \returns true when the current synchronization stream handle is stale.
     * \returns false when the synchronization stream still matches the cached inode.
     */
    bool synchroStreamStale();

    /// Release the synchronization semaphore claim and close the synchronization stream.
    void closeSynchroStream();

    /// Acquire one frame using the internal timer loop.
    /**
     * \returns 0 when a new sample is ready.
     * \returns 1 when the loop should continue without publishing.
     */
    int acquireTimerAndCheckValid();

    /// Acquire one frame using the synchronization semaphore.
    /**
     * \returns 0 when a new sample is ready.
     * \returns 1 when the loop should continue without publishing.
     * \returns -1 on a critical timing error.
     */
    int acquireSynchroAndCheckValid();

    /// Read the current realtime clock value.
    /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
    int getRealtime( timespec &ts /**< [out] the current realtime clock value */ );

    /// Wait on the claimed synchronization semaphore until the supplied timeout.
    /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
    int waitOnSemaphore( sem_t    *sem /**< [in] the semaphore to wait on */,
                         timespec &ts /**< [in] the absolute timeout for the wait */ );

    /// Read one MCP3208 channel value.
    /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
    int readChannelValue( int       channel /**< [in] the MCP3208 channel index to read */,
                          uint16_t &value /**< [out] the sampled channel value */ );

    /// Apply the current controlled delay between semaphore wake and ADC read.
    void delayBeforeRead();

    ///@}

  public:
    /** \name Application Lifecycle
     * @{
     */

    /// Construct the application with the current repository version metadata.
    mcp3208Ctrl();

    /// Destroy the application.
    ~mcp3208Ctrl() noexcept
    {
    }

    /// Register configuration entries for the application and helper devices.
    virtual void setupConfig();

    /// Load configuration values into the application state.
    /** This helper is separated from `loadConfig()` to support unit testing.
     */
    int loadConfigImpl( mx::app::appConfigurator &_config /**< [in] the populated application configurator */ );

    /// Load the configured application state.
    virtual void loadConfig();

    /// Start the application and initialize its INDI and hardware state.
    /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
    virtual int appStartup();

    /// Execute one iteration of the application FSM.
    /**
     * \returns 0 on no critical error
     * \returns -1 on an error requiring shutdown
     */
    virtual int appLogic();

    /// Shutdown the application and release synchronization resources.
    /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
    virtual int appShutdown();

    ///@}

    /** \name Framegrabber Interface
     * @{
     */

    /// Configure the output image geometry and acquisition mode.
    /**
     * \returns 0 on success
     * \returns 1 when configuration should be retried
     */
    int configureAcquisition();

    /// Report the current acquisition rate metadata to the framegrabber.
    /**
     * \returns the current fps value.
     */
    float fps();

    /// Prepare the selected acquisition mode to begin producing samples.
    /**
     * \returns 0 on success
     * \returns -1 on error
     */
    int startAcquisition();

    /// Acquire one sample and indicate whether it should be published.
    /**
     * \returns 0 when a new sample is ready.
     * \returns 1 when no new sample should be published.
     * \returns -1 on a critical error.
     */
    int acquireAndCheckValid();

    /// Copy the current MCP3208 values into the output image buffer.
    /**
     * \returns 0 on success
     * \returns -1 on error
     */
    int loadImageIntoStream( void *dest /**< [out] the destination image buffer */ );

    /// Reset synchronization resources before the next acquisition configuration.
    /**
     * \returns 0 on success
     * \returns -1 on error
     */
    int reconfig();

    ///@}

    /** \name Telemeter Interface
     *
     * @{
     */
    /// Check whether framegrabber timing telemetry should be recorded this cycle.
    /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
    int checkRecordTimes();

    /// Record framegrabber timing telemetry.
    /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
    int recordTelem( const telem_fgtimings *telem /**< [in] the telemeter tag requested by the interface */ );

    ///@}
};

mcp3208Ctrl::mcp3208Ctrl() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    return;
}

void mcp3208Ctrl::setupConfig()
{
    FRAMEGRABBER_SETUP_CONFIG( config );
    TELEMETER_SETUP_CONFIG( config );

    config.add( "fps.device",
                "",
                "fps.device",
                argType::Required,
                "fps",
                "device",
                false,
                "string",
                "Device name for getting fps to set circular buffer length." );

    config.add( "fps.property",
                "",
                "fps.property",
                argType::Required,
                "fps",
                "property",
                false,
                "string",
                "Property name for getting fps to set circular buffer length. Default is 'fps'." );

    config.add( "fps.element",
                "",
                "fps.element",
                argType::Required,
                "fps",
                "element",
                false,
                "string",
                "Property name for getting fps to set circular buffer length. Default is 'current'." );

    config.add( "fps.tol",
                "",
                "fps.tol",
                argType::Required,
                "fps",
                "tol",
                false,
                "float",
                "Tolerance for detecting a change in FPS.  Default is 0." );

    config.add( "synchro.shmimName",
                "",
                "synchro.shmimName",
                argType::Required,
                "synchro",
                "shmimName",
                false,
                "string",
                "The ImageStreamIO stream used to synchronize acquisition. Default is timer-driven operation." );

    config.add( "synchro.postDelay",
                "",
                "synchro.postDelay",
                argType::Required,
                "synchro",
                "postDelay",
                false,
                "int",
                "Delay between a synchronization semaphore and the A/D read in microseconds. Default is 0." );

    config.add( "accel.numChannels",
                "",
                "accel.numChannels",
                argType::Required,
                "accel",
                "numChannels",
                false,
                "int",
                "Setting the number of channels needed to readout accelerometers" );
}

int mcp3208Ctrl::loadConfigImpl( mx::app::appConfigurator &_config )
{

    FRAMEGRABBER_LOAD_CONFIG( _config );
    TELEMETER_LOAD_CONFIG( _config );

    _config( m_fpsDevice, "fps.device" );
    _config( m_fpsProperty, "fps.property" );
    _config( m_fpsElement, "fps.element" );
    _config( m_fpsTol, "fps.tol" );
    _config( m_synchroShmimName, "synchro.shmimName" );
    _config( m_synchroPostDelay, "synchro.postDelay" );

    _config( m_numChannels, "accel.numChannels" ); // making number of mcp3208 channels we read out configurable

    if( m_synchroPostDelay < 0 )
    {
        m_synchroPostDelay = 0;
    }

    m_synchroDelayTarget = 1e3f * m_synchroPostDelay;
    m_synchroDelay       = m_synchroDelayTarget;

    return 0;
}

void mcp3208Ctrl::loadConfig()
{
    loadConfigImpl( config );
}

int mcp3208Ctrl::appStartup()
{
    FRAMEGRABBER_APP_STARTUP;
    TELEMETER_APP_STARTUP;

    // INDI prop for user to set fps
    CREATE_REG_INDI_NEW_NUMBERF( m_indiP_fps, "fps", 0, 10000, 1, "%d", "", "" );
    m_indiP_fps["current"].setValue( m_fps );
    m_indiP_fps["target"].setValue( m_fps );

    if( m_fpsDevice != "" )
    {
        REG_INDI_SETPROP( m_indiP_fpsSource, m_fpsDevice, m_fpsProperty );
    }

    {
        // Get the maximum privileges available
        elevatedPrivileges elPriv( this );

        m_adc.connect();
    }

    state( stateCodes::OPERATING );
    return 0;
}

int mcp3208Ctrl::appLogic()
{
    FRAMEGRABBER_APP_LOGIC;
    TELEMETER_APP_LOGIC;

    FRAMEGRABBER_UPDATE_INDI;

    updatesIfChanged<float>( m_indiP_fps, { "current", "target" }, { m_fps, m_fps } );

    return 0;
}

int mcp3208Ctrl::appShutdown()
{
    FRAMEGRABBER_APP_SHUTDOWN;
    TELEMETER_APP_SHUTDOWN;

    closeSynchroStream();

    return 0;
}

int mcp3208Ctrl::configureAcquisition()
{
    m_values.resize( m_numChannels );

    m_width    = m_numChannels;
    m_height   = 1;
    m_dataType = _DATATYPE_UINT16;

    if( !m_synchroShmimName.empty() )
    {
        log<text_log>( "Configuring semaphore-synchronized acquisition from " + m_synchroShmimName +
                           " with target delay " + std::to_string( m_synchroPostDelay ) + " us.",
                       logPrio::LOG_INFO );

        if( openSynchroStream() != 0 )
        {
            return 1;
        }

        if( claimSynchroSemaphore() != 0 )
        {
            closeSynchroStream();
            return 1;
        }
    }
    else
    {
        log<text_log>( "Configuring timer-driven acquisition.", logPrio::LOG_INFO );
    }

    return 0;
}

float mcp3208Ctrl::fps()
{
    return m_fps;
}

int mcp3208Ctrl::startAcquisition()
{
    if( !m_synchroShmimName.empty() )
    {
        if( !m_synchroStreamOpen )
        {
            return -1;
        }

        if( m_synchroSemaphore == nullptr && claimSynchroSemaphore() != 0 )
        {
            return -1;
        }

        ImageStreamIO_semflush( &m_synchroStream, m_synchroSemaphoreNumber );
        m_synchroDelay = m_synchroDelayTarget;
    }

    m_time_start = std::chrono::high_resolution_clock::now();

    return 0;
}

int mcp3208Ctrl::acquireAndCheckValid()
{
    if( !m_synchroShmimName.empty() )
    {
        return acquireSynchroAndCheckValid();
    }

    return acquireTimerAndCheckValid();
}

int mcp3208Ctrl::loadImageIntoStream( void *dest )
{
    memcpy( dest, m_values.data(), m_values.size() * sizeof( uint16_t ) );
    return 0;
}

int mcp3208Ctrl::reconfig()
{
    closeSynchroStream();
    return 0;
}

int mcp3208Ctrl::openSynchroStream()
{
    char        shmimFilename[1024];
    struct stat buffer;

    if( m_synchroShmimName.empty() || m_synchroStreamOpen )
    {
        return 0;
    }

    if( ImageStreamIO_openIm( &m_synchroStream, m_synchroShmimName.c_str() ) != 0 )
    {
        return 1;
    }

    if( m_synchroStream.md[0].sem < SEMAPHORE_MAXVAL )
    {
        ImageStreamIO_closeIm( &m_synchroStream );
        memset( &m_synchroStream, 0, sizeof( m_synchroStream ) );
        return 1;
    }

    ImageStreamIO_filename( shmimFilename, sizeof( shmimFilename ), m_synchroShmimName.c_str() );
    if( stat( shmimFilename, &buffer ) != 0 )
    {
        ImageStreamIO_closeIm( &m_synchroStream );
        memset( &m_synchroStream, 0, sizeof( m_synchroStream ) );
        return log<software_error, 1>( { __FILE__, __LINE__, errno, "stat" } );
    }

    m_synchroStreamInode = buffer.st_ino;
    m_synchroStreamOpen  = true;
    return 0;
}

int mcp3208Ctrl::claimSynchroSemaphore()
{
    if( !m_synchroStreamOpen )
    {
        return -1;
    }

    if( m_synchroSemaphore != nullptr )
    {
        return 0;
    }

    m_synchroSemaphoreNumber = ImageStreamIO_getsemwaitindex( &m_synchroStream, m_synchroSemaphoreNumber );
    if( m_synchroSemaphoreNumber < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "No valid semaphore found for " + m_synchroShmimName } );
    }

    m_synchroSemaphore = m_synchroStream.semptr[m_synchroSemaphoreNumber];

    if( m_synchroSemaphore == nullptr )
    {
        return log<software_error, -1>(
            { __FILE__, __LINE__, "No valid semaphore pointer found for " + m_synchroShmimName } );
    }

    return 0;
}

bool mcp3208Ctrl::synchroStreamStale()
{
    int         shmimFd;
    char        shmimFilename[1024];
    struct stat buffer;

    if( !m_synchroStreamOpen || m_synchroStream.md[0].sem <= 0 )
    {
        return true;
    }

    ImageStreamIO_filename( shmimFilename, sizeof( shmimFilename ), m_synchroShmimName.c_str() );

    shmimFd = open( shmimFilename, O_RDWR );
    if( shmimFd == -1 )
    {
        return true;
    }

    close( shmimFd );

    if( stat( shmimFilename, &buffer ) != 0 )
    {
        return true;
    }

    return buffer.st_ino != m_synchroStreamInode;
}

void mcp3208Ctrl::closeSynchroStream()
{
    if( m_synchroStreamOpen )
    {
        if( m_synchroSemaphore != nullptr && m_synchroSemaphoreNumber >= 0 )
        {
            m_synchroStream.semReadPID[m_synchroSemaphoreNumber] = 0;
        }

        ImageStreamIO_closeIm( &m_synchroStream );
    }

    memset( &m_synchroStream, 0, sizeof( m_synchroStream ) );
    m_synchroSemaphore       = nullptr;
    m_synchroSemaphoreNumber = 5;
    m_synchroStreamInode     = 0;
    m_synchroStreamOpen      = false;
}

int mcp3208Ctrl::acquireTimerAndCheckValid()
{
    while( !m_shutdown && !m_reconfig )
    {
        // Get current time
        auto now     = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>( now - m_time_start );

        // Read every 500 microseconds
        if( elapsed.count() >= m_trigger )
        {
            m_time_start = now; // Reset start time

            for( int i = 0; i < m_numChannels; ++i )
            {
                if( readChannelValue( i, m_values[i] ) < 0 )
                {
                    return 1;
                }
            }

            m_trigger = m_trigger - m_gain * ( elapsed.count() - nano_sec_target );

            return 0;
        }
        else
        {
            mx::sys::nanoSleep( 10000 );
        }
    }

    return 0;
}

int mcp3208Ctrl::acquireSynchroAndCheckValid()
{
    timespec ts;
    auto     synchroWake = std::chrono::high_resolution_clock::time_point();

    if( m_synchroSemaphore == nullptr )
    {
        m_reconfig = true;
        return 1;
    }

    errno = 0;
    if( getRealtime( ts ) < 0 )
    {
        return log<software_critical, -1>( { __FILE__, __LINE__, errno, 0, "clock_gettime" } );
    }

    ts.tv_sec += 1;

    errno = 0;
    if( waitOnSemaphore( m_synchroSemaphore, ts ) != 0 )
    {
        if( errno == EINTR )
        {
            return 1;
        }

        if( errno == ETIMEDOUT )
        {
            if( synchroStreamStale() )
            {
                log<text_log>( "Synchronized trigger stream changed, reconfiguring.", logPrio::LOG_NOTICE );
                m_reconfig = true;
            }

            return 1;
        }

        log<software_error>( { __FILE__, __LINE__, errno, "sem_timedwait" } );
        m_reconfig = true;
        return 1;
    }

    synchroWake = std::chrono::high_resolution_clock::now();
    delayBeforeRead();

    if( getRealtime( m_currImageTimestamp ) < 0 )
    {
        return log<software_critical, -1>( { __FILE__, __LINE__, errno, 0, "clock_gettime" } );
    }

    auto readStart = std::chrono::high_resolution_clock::now();
    auto elapsed   = std::chrono::duration_cast<std::chrono::nanoseconds>( readStart - synchroWake );

    m_synchroDelay = m_synchroDelay - m_gain * ( elapsed.count() - m_synchroDelayTarget );
    if( m_synchroDelay < 0 )
    {
        m_synchroDelay = 0;
    }

    for( int i = 0; i < m_numChannels; ++i )
    {
        if( readChannelValue( i, m_values[i] ) < 0 )
        {
            m_reconfig = true;
            return 1;
        }
    }

    return 0;
}

int mcp3208Ctrl::getRealtime( timespec &ts )
{
    return clock_gettime( CLOCK_REALTIME, &ts );
}

int mcp3208Ctrl::waitOnSemaphore( sem_t *sem, timespec &ts )
{
    return sem_timedwait( sem, &ts );
}

int mcp3208Ctrl::readChannelValue( int channel, uint16_t &value )
{
    value = m_adc.read( channel );
    return 0;
}

void mcp3208Ctrl::delayBeforeRead()
{
    if( m_synchroDelay > 0 )
    {
        mx::sys::nanoSleep( static_cast<unsigned>( m_synchroDelay ) );
    }
}

int mcp3208Ctrl::checkRecordTimes()
{
    return telemeter<mcp3208Ctrl>::checkRecordTimes( telem_fgtimings() );
}

int mcp3208Ctrl::recordTelem( const telem_fgtimings * )
{
    return recordFGTimings( true );
}

// INDI callback handling for fps configuration.
INDI_NEWCALLBACK_DEFN( mcp3208Ctrl, m_indiP_fps )( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getName() != m_indiP_fps.getName() )
    {
        log<software_error>( { __FILE__, __LINE__, "wrong INDI property received." } );
        return -1;
    }

    float target;

    if( indiTargetUpdate( m_indiP_fps, target, ipRecv, true ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        return -1;
    }

    m_fps           = target;
    m_trigger       = 1e9f / m_fps; // Update trigger value based off new fps
    nano_sec_target = 1e9f / m_fps;

    log<text_log>( "set fps = " + std::to_string( m_fps ) );
    return 0;
}

INDI_SETCALLBACK_DEFN( mcp3208Ctrl, m_indiP_fpsSource )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_fpsSource, ipRecv );

    if( ipRecv.find( m_fpsElement ) != true ) // this isn't valid
    {
        log<software_error>( { __FILE__, __LINE__, "No current property in fps source." } );
        return 0;
    }

    float target = ipRecv[m_fpsElement].get<float>();

    m_fps           = target;
    m_trigger       = 1e9f / m_fps; // Update trigger value based off new fps
    nano_sec_target = 1e9f / m_fps;

    log<text_log>( "set fps from " + m_fpsDevice + " = " + std::to_string( m_fps ) );
    return 0;

} // INDI_SETCALLBACK_DEFN(mcp3208Ctrl, m_indiP_fpsSource)

} // namespace app
} // namespace MagAOX

#endif // mcp3208Ctrl_hpp
