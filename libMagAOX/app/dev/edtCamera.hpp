/** \file edtCamera.hpp
 * \brief EDT framegrabber interface
 *
 * \ingroup app_files
 */

#ifndef edtCamera_hpp
#define edtCamera_hpp

#ifndef MAGAOX_NOEDT

    #include <edtinc.h>

    #include <cerrno>
    #include <ctime>

    #include "../../common/frameGrabberTiming.hpp"
    #include "../../common/paths.hpp"

    #include "ioDevice.hpp"

    #include "stdCamera.hpp"

namespace MagAOX
{
namespace app
{
namespace dev
{

/// MagAO-X EDT framegrabber interface
/** Implements an interface to the EDT PDV SDK
 *
 * The derived class `derivedT` must be a MagAOXApp\<true\>, and must declare this class a friend like so:
 * \code
 * friend class dev::edtCamera<derivedT>;
 * \endcode
 *
 * A static configuration variable must be defined in derivedT as
 * \code
 * static constexpr bool c_edtCamera_relativeConfigPath = true; //or: false
 * \endcode
 * which determines whether or not the EDT config path is relative to the MagAO-X config path (true) or absolute
 * (false).
 *
 * In addition, `derivedT` should be `dev::frameGrabber` or equivalent, with an `m_reconfig` member, and calls
 * to this class's `pdvStartAcquisition`, `pdvAcquire`, and `pdvReconfig` in the relevant `dev::frameGrabber`
 * implementation functions.
 *
 * Calls to this class's `setupConfig`, `loadConfig`, `appStartup`, `appLogic`, `appShutdown`
 * `onPowerOff`, and `whilePowerOff`,  must be placed in the derived class's functions of the same name.
 *
 * \ingroup appdev
 */
template <class derivedT>
class edtCamera : public ioDevice
{

  public:
  protected:
    /** \name Configurable Parameters - Data
     * @{
     */
    // Framegrabber:
    int m_unit{ 0 };     ///< EDT PDV board unit number
    int m_channel{ 0 };  ///< EDT PDV board channel number
    int m_numBuffs{ 4 }; ///< EDT PDV DMA buffer size, indicating number of images.

    // cameraConfigMap derived().m_cameraModes; ///< Map holding the possible camera mode configurations

    // std::string derived().m_startupMode; ///< The camera mode to load during first init after a power-on.

    ///@}

    PdvDev *m_pdv{ nullptr }; ///< The EDT PDV device handle

    u_char *m_image_p{ nullptr }; ///< The image data grabbed

    int m_pdvTimeouts{ 0 }; ///< Last cumulative SDK timeout count, reset when acquisition starts.

    bool m_deferredRearmAllowed{ false }; ///< Configured ring can replenish a request after publication.

    bool m_rearmPending{ false }; ///< Frame-worker obligation to replenish one consumed request or discard on exit.

    // std::string m_modeName; ///< The current mode name

    // std::string derived().m_nextMode; ///< The mode to be set by the next reconfiguration

    int         m_raw_height{ 0 }; ///< The height of the frame, according to the framegrabber
    int         m_raw_width{ 0 };  ///< The width of the frame, according to the framegrabber
    int         m_raw_depth{ 0 };  ///< The bit-depth of the frame, according to the framegrabber
    std::string m_cameraType;      ///< The camera type according to the framegrabber

  public:
    /// C'tor, sets up stdCamera
    edtCamera();

    /// Destructor, destroys the PdvDev structure
    ~edtCamera() noexcept;

    /// Send a serial command over cameralink and retrieve the response
    int pdvSerialWriteRead( std::string       &response, /**< [out] the response to the command from the device */
                            const std::string &command,  /**< [in] the command to send to the device */
                            bool               logErrors = true /**< [in] log serial transport failures when true */
    );

    /// Configure the EDT framegrabber
    int pdvConfig( std::string &cfgname /**< [in] The configuration name for the mode to set */ );

    /// Setup the configuration system
    /**
      * This should be called in `derivedT::setupConfig` as
      * \code
        edtCamera<derivedT>::setupConfig(config);
        \endcode
      * with appropriate error checking.
      */
    void setupConfig( mx::app::appConfigurator &config /**< [out] the derived classes configurator*/ );

    /// load the configuration system results
    /**
      * This should be called in `derivedT::loadConfig` as
      * \code
        edtCamera<derivedT>::loadConfig(config);
        \endcode
      * with appropriate error checking.
      */
    void loadConfig( mx::app::appConfigurator &config /**< [in] the derived classes configurator*/ );

    /// Startup function
    /**
      * This should be called in `derivedT::appStartup` as
      * \code
        edtCamera<derivedT>::appStartup();
        \endcode
      * with appropriate error checking.
      *
      * \returns 0 on success
      * \returns -1 on error, which is logged.
      */
    int appStartup();

    /// Application logic
    /** Checks the edtCamera thread
      *
      * This should be called from the derived's appLogic() as in
      * \code
        edtCamera<derivedT>::appLogic();
        \endcode
      * with appropriate error checking.
      *
      * \returns 0 on success
      * \returns -1 on error, which is logged.
      */
    int appLogic();

    /// Actions on power off
    /**
      * This should be called from the derived's onPowerOff() as in
      * \code
        edtCamera<derivedT>::onPowerOff();
        \endcode
      * with appropriate error checking.
      *
      * \returns 0 on success
      * \returns -1 on error, which is logged.
      */
    int onPowerOff();

    /// Actions while powered off
    /**
      * This should be called from the derived's whilePowerOff() as in
      * \code
        edtCamera<derivedT>::whilePowerOff();
        \endcode
      * with appropriate error checking.
      *
      * \returns 0 on success
      * \returns -1 on error, which is logged.
      */
    int whilePowerOff();

    /// Application shutdown.
    /** Shuts down the edtCamera thread
      *
      * \code
        edtCamera<derivedT>::appShutdown();
        \endcode
      * with appropriate error checking.
      *
      * \returns 0 on success
      * \returns -1 on error, which is logged.
      */
    int appShutdown();

    /// Start the configured ring and establish the cumulative timeout baseline.
    int pdvStartAcquisition();

    /// Return a complete image and SDK timestamp, or clear both outputs and fail.
    /** A changed SDK timeout count rejects even a non-null partial-image buffer.
     * Deferred rearm requires an eligible multibuffer ring and an eventual pdvFinishAcquisition() call
     * from the same frame worker. Other callers retain immediate rearm.
     * \returns zero on success, or -1 for a failed/partial acquisition or invalid timestamp.
     */
    int
    pdvAcquire( timespec &currImageTimestamp /**< [out] SDK timestamp, zeroed on failure. */,
                bool deferRearm = false /**< [in] Defer eligible ring replenishment until the attempt finishes. */ );

    /// Replenish one deferred request when continuing, or discard its obligation before recovery or exit.
    void pdvFinishAcquisition( bool continuing /**< [in] The same worker will continue with the current ring. */ );

    /// Reopen the requested mode using the existing recovery path.
    int pdvReconfig();

  protected:
    /** \name INDI
     *
     *@{
     */
  protected:
    // declare our properties

  public:
    /// The static callback function to be registered for the channel properties.
    /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
    // static int st_newCallBack_mode( void * app, ///< [in] a pointer to this, will be static_cast-ed to derivedT.
    //                                   const pcf::IndiProperty &ipRecv ///< [in] the INDI property sent with the the
    //                                   new property request.
    //                               );

    /// The callback called by the static version, to actually process the new request.
    /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
    // int newCallBack_mode( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property
    // request.*/);

    /// Update the INDI properties for this device controller
    /** You should call this once per main loop.
     * It is not called automatically.
     *
     * \returns 0 on success.
     * \returns -1 on error.
     */
    int updateINDI();

    ///@}

  private:
    /// Access the CRTP owner.
    derivedT &derived();

    /// Forward a segment marker when the owner supports optional frame timing.
    template <class hookT>
    static auto timingMark( hookT                   &owner /**< [in,out] Derived camera. */,
                            MagAOX::fgTiming::Marker marker /**< [in] Segment endpoint. */,
                            int /**< [in] Optional-hook overload priority. */ )
        -> decltype( owner.frameGrabberTimingMark( marker ), void() );

    /// Keep EDT-only owners source compatible when segment timing is absent.
    static void timingMark( derivedT                &owner /**< [in,out] Derived camera. */,
                            MagAOX::fgTiming::Marker marker /**< [in] Unused endpoint. */,
                            long /**< [in] Fallback overload priority. */ );

    /// Ask whether optional serial segment timing is enabled.
    template <class hookT>
    static auto timingEnabled( const hookT &owner /**< [in] Derived camera. */,
                               int /**< [in] Optional-hook overload priority. */ )
        -> decltype( owner.frameGrabberTimingEnabled() );

    /// Avoid diagnostic clocks when the owner has no stage timing interface.
    static bool timingEnabled( const derivedT &owner /**< [in] Derived camera. */,
                               long /**< [in] Fallback overload priority. */ );

    /// Forward one ordinary-thread serial transaction to the optional control channel.
    template <class hookT>
    static auto serialTiming( hookT  &owner /**< [in,out] Derived camera. */,
                              int64_t entered /**< [in] Monotonic transaction start, ns. */,
                              int64_t completed /**< [in] Monotonic transaction completion, ns. */,
                              bool    success /**< [in] Transaction completed successfully. */,
                              int /**< [in] Optional-hook overload priority. */ )
        -> decltype( owner.frameGrabberSerialTiming( entered, completed, success ), void() );

    /// Keep EDT-only owners compatible when the serial timing hook is absent.
    static void serialTiming( derivedT &owner /**< [in,out] Derived camera. */,
                              int64_t   entered /**< [in] Unused transaction start. */,
                              int64_t   completed /**< [in] Unused transaction completion. */,
                              bool      success /**< [in] Unused result. */,
                              long /**< [in] Fallback overload priority. */ );

    /// Read a diagnostic monotonic timestamp without changing the caller's errno.
    static int64_t timingNow();

    /// Queue one image on the frame worker and record its independent timing endpoints.
    void rearmImage();
};

template <class derivedT>
derivedT &edtCamera<derivedT>::derived()
{
    return *static_cast<derivedT *>( this );
}

template <class derivedT>
template <class hookT>
auto edtCamera<derivedT>::timingMark( hookT &owner, MagAOX::fgTiming::Marker marker, int )
    -> decltype( owner.frameGrabberTimingMark( marker ), void() )
{
    owner.frameGrabberTimingMark( marker );
}

template <class derivedT>
void edtCamera<derivedT>::timingMark( derivedT &, MagAOX::fgTiming::Marker, long )
{
}

template <class derivedT>
template <class hookT>
auto edtCamera<derivedT>::timingEnabled( const hookT &owner, int ) -> decltype( owner.frameGrabberTimingEnabled() )
{
    return owner.frameGrabberTimingEnabled();
}

template <class derivedT>
bool edtCamera<derivedT>::timingEnabled( const derivedT &, long )
{
    return false;
}

template <class derivedT>
template <class hookT>
auto edtCamera<derivedT>::serialTiming( hookT &owner, int64_t entered, int64_t completed, bool success, int )
    -> decltype( owner.frameGrabberSerialTiming( entered, completed, success ), void() )
{
    owner.frameGrabberSerialTiming( entered, completed, success );
}

template <class derivedT>
void edtCamera<derivedT>::serialTiming( derivedT &, int64_t, int64_t, bool, long )
{
}

template <class derivedT>
int64_t edtCamera<derivedT>::timingNow()
{
    const int saved = errno;
    timespec  stamp{};
    const int result = clock_gettime( CLOCK_MONOTONIC, &stamp );
    errno            = saved;

    return result == 0 ? int64_t( stamp.tv_sec ) * 1000000000 + stamp.tv_nsec : -1;
}

template <class derivedT>
edtCamera<derivedT>::edtCamera()
{
}

template <class derivedT>
edtCamera<derivedT>::~edtCamera() noexcept
{
    if( m_pdv )
    {
        pdv_close( m_pdv );
    }

    return;
}

    #define MAGAOX_PDV_SERBUFSIZE 512

template <class derivedT>
int edtCamera<derivedT>::pdvSerialWriteRead( std::string &response, const std::string &command, bool logErrors )
{
    const bool    capture = timingEnabled( derived(), 0 );
    const int64_t entered = capture ? timingNow() : -1;
    auto          finish  = [&]( int result )
    {
        if( capture )
        {
            const int saved = errno;
            serialTiming( derived(), entered, timingNow(), result == 0, 0 );
            errno = saved;
        }

        return result;
    };
    char buf[MAGAOX_PDV_SERBUFSIZE + 1];

    // Flush the channel first.
    // This does not indicate errors, so no checks possible.
    pdv_serial_read( m_pdv, buf, MAGAOX_PDV_SERBUFSIZE );

    if( pdv_serial_command( m_pdv, command.c_str() ) < 0 )
    {
        if( logErrors )
        {
            derivedT::template log<software_error>( { __FILE__, __LINE__, "PDV: error sending serial command" } );
        }
        return finish( -1 );
    }

    int ret;

    ret = pdv_serial_wait( m_pdv, m_readTimeout, 1 );

    if( ret == 0 )
    {
        if( derived().powerState() != 1 || derived().powerStateTarget() != 1 )
        {
            return finish( -1 );
        }

        if( logErrors )
        {
            derivedT::template log<software_error>( { __FILE__, __LINE__, "PDV: timeout, no serial response" } );
        }
        return finish( -1 );
    }

    u_char lastbyte = 0;
    u_char waitc    = 0;

    response.clear();

    do
    {
        ret = pdv_serial_read( m_pdv, buf, MAGAOX_PDV_SERBUFSIZE );

        if( ret > 0 )
        {
            response.append( buf, ret );
            lastbyte = static_cast<u_char>( buf[ret - 1] );
        }
        else
        {
            lastbyte = 0;
        }

        if( pdv_get_waitchar( m_pdv, &waitc ) && ( lastbyte == waitc ) )
        {
            break;
        }
        else
        {
            ret = pdv_serial_wait( m_pdv, m_readTimeout / 2, 1 );
        }
    } while( ret > 0 );

    if( ret == 0 && response.empty() && pdv_get_waitchar( m_pdv, &waitc ) )
    {
        if( logErrors )
        {
            derivedT::template log<software_error>( { __FILE__, __LINE__, "PDV: timeout in serial response" } );
        }
        return finish( -1 );
    }

    // Drain any immediate trailing bytes so CR/LF endings or prompt suffixes
    // do not become the next command's apparent response.
    ret = pdv_serial_wait( m_pdv, 10, 1 );
    while( ret > 0 )
    {
        ret = pdv_serial_read( m_pdv, buf, MAGAOX_PDV_SERBUFSIZE );

        if( ret > 0 )
        {
            ret = pdv_serial_wait( m_pdv, 10, 1 );
        }
    }

    return finish( 0 );
}

template <class derivedT>
int edtCamera<derivedT>::pdvConfig( std::string &modeName )
{
    Dependent *dd_p;
    EdtDev    *edt_p = NULL;
    Edtinfo    edtinfo;

    m_rearmPending         = false;
    m_deferredRearmAllowed = false;

    // Preliminaries
    if( m_pdv )
    {
        pdv_close( m_pdv );
        m_pdv = nullptr;
    }

    derived().m_modeName = modeName;

    if( modeName == "" )
    {
        return derivedT::template log<text_log, -1>( "Empty modeName passed to pdvConfig", logPrio::LOG_ERROR );
    }

    if( derived().m_cameraModes.count( modeName ) != 1 )
    {
        return derivedT::template log<text_log, -1>( "No mode named " + modeName + " found.", logPrio::LOG_ERROR );
    }

    // Construct config file, adding relative path if configured that way
    std::string configFile;

    if( derivedT::c_edtCamera_relativeConfigPath )
    {
        configFile = derived().configDir() + "/";
    }

    configFile += derived().m_cameraModes[modeName].m_configFile;

    derivedT::template log<text_log>( "Loading EDT PDV config file: " + configFile );

    if( ( dd_p = pdv_alloc_dependent() ) == NULL )
    {
        return derivedT::template log<software_error, -1>( { __FILE__, __LINE__, "EDT PDV alloc_dependent FAILED" } );
    }

    if( pdv_readcfg( configFile.c_str(), dd_p, &edtinfo ) != 0 )
    {
        free( dd_p );
        return derivedT::template log<software_error, -1>( { __FILE__, __LINE__, "EDT PDV readcfg FAILED" } );
    }

    char edt_devname[128];
    strncpy( edt_devname, EDT_INTERFACE, sizeof( edt_devname ) );

    if( ( edt_p = edt_open_channel( edt_devname, m_unit, m_channel ) ) == NULL )
    {
        char errstr[256];
        edt_perror( errstr );
        free( dd_p );
        return derivedT::template log<software_error, -1>(
            { __FILE__, __LINE__, std::string( "EDT PDV edt_open_channel FAILED: " ) + errstr } );
    }

    char bitdir[1];
    bitdir[0] = '\0';

    int pdv_debug = 0;

    if( derived().m_log.logLevel() > logPrio::LOG_INFO )
    {
        pdv_debug = 2;
    }

    if( pdv_initcam( edt_p, dd_p, m_unit, &edtinfo, configFile.c_str(), bitdir, pdv_debug ) != 0 )
    {
        edt_close( edt_p );
        free( dd_p );
        return derivedT::template log<software_error, -1>(
            { __FILE__, __LINE__, "initcam failed. Run with '--logLevel=DBG' to see complete debugging output." } );
    }

    edt_close( edt_p );
    free( dd_p );

    // Now open the PDV device handle for talking to the camera via the EDT board.
    if( ( m_pdv = pdv_open_channel( edt_devname, m_unit, m_channel ) ) == NULL )
    {
        std::string errstr = std::string( "pdv_open_channel(" ) + edt_devname + std::to_string( m_unit ) + "_" +
                             std::to_string( m_channel ) + ")";

        derivedT::template log<software_error>( { __FILE__, __LINE__, errstr } );
        derivedT::template log<software_error>( { __FILE__, __LINE__, errno } );

        return -1;
    }

    pdv_flush_fifo( m_pdv );

    pdv_serial_read_enable( m_pdv ); // This is undocumented, don't know if it's really needed.

    m_raw_width  = pdv_get_width( m_pdv );
    m_raw_height = pdv_get_height( m_pdv );
    m_raw_depth  = pdv_get_depth( m_pdv );
    m_cameraType = pdv_get_cameratype( m_pdv );

    derivedT::template log<text_log>( "Initialized framegrabber: " + m_cameraType );
    derivedT::template log<text_log>( "WxHxD: " + std::to_string( m_raw_width ) + " X " +
                                      std::to_string( m_raw_height ) + " X " + std::to_string( m_raw_depth ) );

    /*
     * allocate four buffers for optimal pdv ring buffer pipeline (reduce if
     * memory is at a premium)
     */
    pdv_multibuf( m_pdv, m_numBuffs );
    derivedT::template log<text_log>( "allocated " + std::to_string( m_numBuffs ) + " buffers" );

    return 0;
}

template <class derivedT>
void edtCamera<derivedT>::setupConfig( mx::app::appConfigurator &config )
{
    config.add( "framegrabber.pdv_unit",
                "",
                "framegrabber.pdv_unit",
                argType::Required,
                "framegrabber",
                "pdv_unit",
                false,
                "int",
                "The EDT PDV framegrabber unit number.  Default is 0." );
    config.add( "framegrabber.pdv_channel",
                "",
                "framegrabber.pdv_channel",
                argType::Required,
                "framegrabber",
                "pdv_channel",
                false,
                "int",
                "The EDT PDV framegrabber channel number.  Default is 0." );
    config.add( "framegrabber.numBuffs",
                "",
                "framegrabber.numBuffs",
                argType::Required,
                "framegrabber",
                "numBuffs",
                false,
                "int",
                "The EDT PDV framegrabber DMA buffer size [images].  Default is 4." );

    dev::ioDevice::setupConfig( config );
}

template <class derivedT>
void edtCamera<derivedT>::loadConfig( mx::app::appConfigurator &config )
{
    config( m_unit, "framegrabber.pdv_unit" );
    config( m_channel, "framegrabber.pdv_channel" );
    config( m_numBuffs, "framegrabber.numBuffs" );

    m_readTimeout  = 1000;
    m_writeTimeout = 1000;
    dev::ioDevice::loadConfig( config );
}

template <class derivedT>
int edtCamera<derivedT>::appStartup()
{
    if( pdvConfig( derived().m_startupMode ) < 0 )
    {
        derivedT::template log<software_error>( { __FILE__, __LINE__ } );
        return -1;
    }

    return 0;
}

template <class derivedT>
int edtCamera<derivedT>::appLogic()
{
    return 0;
}

template <class derivedT>
int edtCamera<derivedT>::onPowerOff()
{
    return 0;
}

template <class derivedT>
int edtCamera<derivedT>::whilePowerOff()
{
    return 0;
}

template <class derivedT>
int edtCamera<derivedT>::appShutdown()
{
    return 0;
}

template <class derivedT>
int edtCamera<derivedT>::pdvStartAcquisition()
{
    m_rearmPending         = false;
    m_deferredRearmAllowed = false;
    m_pdvTimeouts          = pdv_timeouts( m_pdv );

    if( m_pdvTimeouts < 0 )
    {
        return -1;
    }

    m_deferredRearmAllowed = m_pdv != nullptr && m_pdv->dd_p != nullptr && m_numBuffs > 1 &&
                             m_pdv->ring_buffer_numbufs > 1 && m_pdv->dd_p->force_single == 0 &&
                             m_pdv->dd_p->serial_trigger[0] == '\0';

    pdv_start_images( m_pdv, m_numBuffs );

    return 0;
}

template <class derivedT>
int edtCamera<derivedT>::pdvAcquire( timespec &currImageTimestamp, bool deferRearm )
{
    currImageTimestamp = {};

    if( m_rearmPending )
    {
        m_image_p = nullptr;
        return -1;
    }

    uint dmaTimeStamp[2]{};
    timingMark( derived(), MagAOX::fgTiming::Marker::waitEnter, 0 );
    m_image_p = pdv_wait_last_image_timed( m_pdv, dmaTimeStamp );
    timingMark( derived(), MagAOX::fgTiming::Marker::waitReturn, 0 );

    if( !m_image_p )
    {
        return -1;
    }

    if( deferRearm && m_deferredRearmAllowed )
    {
        m_rearmPending = true;
    }
    else
    {
        rearmImage();
    }

    timingMark( derived(), MagAOX::fgTiming::Marker::timeoutEnter, 0 );
    const int timeouts = pdv_timeouts( m_pdv );
    timingMark( derived(), MagAOX::fgTiming::Marker::timeoutReturn, 0 );
    const bool complete = timeouts >= 0 && timeouts == m_pdvTimeouts;
    m_pdvTimeouts       = timeouts;

    if( !complete || dmaTimeStamp[1] >= 1000000000 || ( dmaTimeStamp[0] == 0 && dmaTimeStamp[1] == 0 ) )
    {
        m_image_p = nullptr;
        return -1;
    }

    currImageTimestamp.tv_sec  = dmaTimeStamp[0];
    currImageTimestamp.tv_nsec = dmaTimeStamp[1];

    return 0;
}

template <class derivedT>
void edtCamera<derivedT>::pdvFinishAcquisition( bool continuing )
{
    if( !m_rearmPending )
    {
        return;
    }

    m_rearmPending = false;

    if( continuing && m_pdv != nullptr )
    {
        rearmImage();
    }
}

template <class derivedT>
void edtCamera<derivedT>::rearmImage()
{
    timingMark( derived(), MagAOX::fgTiming::Marker::rearmEnter, 0 );
    pdv_start_image( m_pdv );
    timingMark( derived(), MagAOX::fgTiming::Marker::rearmReturn, 0 );
}

template <class derivedT>
int edtCamera<derivedT>::pdvReconfig()
{

    if( pdvConfig( derived().m_nextMode ) < 0 )
    {
        derivedT::template log<text_log>( "error trying to re-configure with " + derived().m_nextMode,
                                          logPrio::LOG_ERROR );
        return -1;
    }
    else
    {
        derived().m_nextMode = "";
    }

    return 0;
}

template <class derivedT>
int edtCamera<derivedT>::updateINDI()
{
    return 0;
}

} // namespace dev
} // namespace app
} // namespace MagAOX

#endif // MAGAOX_NOEDT
#endif // edtCamera_hpp
