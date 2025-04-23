/** \file mcp3008Ctrl.hpp
 * \brief The MagAO-X MCP3008 Controller header file
 *
 * \ingroup mcp3008Ctrl_files
 */

#ifndef mcp3008Ctrl_hpp
#define mcp3008Ctrl_hpp

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"
#include "dependencies/MCP3008.h" // Included for adc.connect()

/** \defgroup mcp3008Ctrl
 * \brief The MagAO-X application to readout a MCP3008 A/D on a raspberry Pi.
 *
 * <a href="../handbook/operating/software/apps/XXXXXX.html">Application Documentation</a>
 *
 * \ingroup apps
 *
 */

/** \defgroup mcp3008Ctrl_files
 * \ingroup mcp3008Ctrl
 */

namespace MagAOX
{
namespace app
{

/// The MagAO-X MCP3008 Controller
/**
 * \ingroup mcp3008Ctrl
 */
class mcp3008Ctrl : public MagAOXApp<true>, public dev::frameGrabber<mcp3008Ctrl>, public dev::telemeter<mcp3008Ctrl>
{

    // Give the test harness access.
    friend class mcp3008Ctrl_test;
    friend class dev::frameGrabber<mcp3008Ctrl>;
    friend class dev::telemeter<mcp3008Ctrl>;

    typedef dev::frameGrabber<mcp3008Ctrl> frameGrabberT;
    typedef dev::telemeter<mcp3008Ctrl>    telemeterT;

    MCP3008Lib::MCP3008 m_adc;

    static constexpr bool c_frameGrabber_flippable = false; /**< app:dev config to tell framegrabber these images
                                                                 can not be flipped*/

  protected:
    /** \name Configurable Parameters
     *@{
     */

    int m_numChannels{ 8 }; ///< The number of channels being read out.

    std::string m_fpsDevice;               ///< Device name for getting fps to set circular buffer length.
    std::string m_fpsProperty{ "fps" };    ///< Property name for getting fps to set circular buffer length.
    std::string m_fpsElement{ "current" }; ///< Element name for getting fps to set circular buffer length.

    float m_fpsTol{ 0 }; ///< The tolerance for detecting a change in FPS.

    ///@}

    // Creating INDI property for desired fps
    pcf::IndiProperty m_indiP_fps;
    INDI_NEWCALLBACK_DECL( mcp3008Ctrl, m_indiP_fps );
    float m_fps{ 2000 }; ///< The target FPS

    pcf::IndiProperty m_indiP_fpsSource;
    INDI_SETCALLBACK_DECL( mcp3008Ctrl, m_indiP_fpsSource );

    float m_trigger{ 1e9f / m_fps }; ///< The trigger time to readout.  Adjusts to match desired FPS.
    float m_gain{ .1 };              // Gain used to adjust trigger to keep at correct fps
    float nano_sec_target{ 1e9f / m_fps };

    MCP3008Lib::MCP3008 adc;

    std::chrono::time_point<std::chrono::high_resolution_clock> m_time_start;

    std::vector<uint16_t> m_values; ///< The values read out from the chip

  public:
    /// Default c'tor.
    mcp3008Ctrl();

    /// D'tor, declared and defined for noexcept.
    ~mcp3008Ctrl() noexcept
    {
    }

    virtual void setupConfig();

    /// Implementation of loadConfig logic, separated for testing.
    /** This is called by loadConfig().
     */
    int loadConfigImpl(
        mx::app::appConfigurator &_config /**< [in] an application configuration
                                                    from which to load values*/ );

    virtual void loadConfig();

    /// Startup function
    /**
     *
     */
    virtual int appStartup();

    /// Implementation of the FSM for mcp3008Ctrl.
    /**
     * \returns 0 on no critical error
     * \returns -1 on an error requiring shutdown
     */
    virtual int appLogic();

    /// Shutdown the app.
    /**
     *
     */
    virtual int appShutdown();

    /// Implementation of the framegrabber configureAcquisition interface
    /**
     *
     * \returns 0 on success
     * \returns -1 on error
     */
    int configureAcquisition();

    /// Implementation of the frameGrabber fps interface
    /** Just returns the value of m_fps
     */
    float fps();

    /// Implementation of the framegrabber startAcquisition interface
    /**
     *
     * \returns 0 on success
     * \returns -1 on error
     */
    int startAcquisition();

    /// Implementation of the framegrabber acquireAndCheckValid interface
    /**
     *
     * \returns 0 on success
     * \returns -1 on error
     */
    int acquireAndCheckValid();

    /// Implementation of the framegrabber loadImageIntoStream interface
    /**
     *
     * \returns 0 on success
     * \returns -1 on error
     */
    int loadImageIntoStream( void *dest /**< [in] */ );

    /// Implementation of the framegrabber reconfig interface
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
    int checkRecordTimes();

    int recordTelem( const telem_fgtimings * );
};

mcp3008Ctrl::mcp3008Ctrl() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    return;
}

void mcp3008Ctrl::setupConfig()
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

int mcp3008Ctrl::loadConfigImpl( mx::app::appConfigurator &_config )
{

    FRAMEGRABBER_LOAD_CONFIG( _config );
    TELEMETER_LOAD_CONFIG( _config );

    _config( m_fpsDevice, "fps.device" );
    _config( m_fpsProperty, "fps.property" );
    _config( m_fpsElement, "fps.element" );
    _config( m_fpsTol, "fps.tol" );

    _config( m_numChannels, "accel.numChannels" ); // making number of mcp3008 channels we read out configurable

    return 0;
}

void mcp3008Ctrl::loadConfig()
{
    loadConfigImpl( config );
}

int mcp3008Ctrl::appStartup()
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

int mcp3008Ctrl::appLogic()
{
    FRAMEGRABBER_APP_LOGIC;
    TELEMETER_APP_LOGIC;

    FRAMEGRABBER_UPDATE_INDI;

    updatesIfChanged<float>( m_indiP_fps, { "current", "target" }, { m_fps, m_fps } );

    return 0;
}

int mcp3008Ctrl::appShutdown()
{
    FRAMEGRABBER_APP_SHUTDOWN;
    TELEMETER_APP_SHUTDOWN;

    return 0;
}

int mcp3008Ctrl::configureAcquisition()
{
    /** \todo @PARKER do anything needed to setup the MPC3008*/

    m_values.resize( m_numChannels );

    m_width    = m_numChannels;
    m_height   = 1;
    m_dataType = _DATATYPE_UINT16;

    return 0;
}

float mcp3008Ctrl::fps()
{
    return m_fps;
}

int mcp3008Ctrl::startAcquisition()
{
    /** \todo @PARKER Do anything needed to start the MPC3008 reading out ... probably nothing*/

    m_time_start = std::chrono::high_resolution_clock::now();

    return 0;
}

int mcp3008Ctrl::acquireAndCheckValid()
{
    while( !m_shutdown && !m_reconfig )
    {
        // Get current time
        auto now     = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>( now - m_time_start );

        // Read every 500 microseconds
        if( elapsed.count() >= m_trigger ) /** \todo @PARKER make m_trigger adjust */
        {
            m_time_start = now; // Reset start time

            for( int i = 0; i < m_numChannels; ++i )
            {
                m_values[i] = m_adc.read( i );
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

int mcp3008Ctrl::loadImageIntoStream( void *dest )
{
    memcpy( dest, m_values.data(), m_values.size() * sizeof( uint16_t ) );
    return 0;
}

int mcp3008Ctrl::reconfig()
{
    return 0;
}

int mcp3008Ctrl::checkRecordTimes()
{
    return telemeter<mcp3008Ctrl>::checkRecordTimes( telem_fgtimings() );
}

int mcp3008Ctrl::recordTelem( const telem_fgtimings * )
{
    return recordFGTimings( true );
}

// Testing for user to select star number
INDI_NEWCALLBACK_DEFN( mcp3008Ctrl, m_indiP_fps )( const pcf::IndiProperty &ipRecv )
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

    log<text_log>( "set fps = " + std::to_string( m_fps ));
    return 0;
}

INDI_SETCALLBACK_DEFN( mcp3008Ctrl, m_indiP_fpsSource )( const pcf::IndiProperty &ipRecv )
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

    log<text_log>( "set fps from " + m_fpsDevice + " = " + std::to_string( m_fps ));
    return 0;

} // INDI_SETCALLBACK_DEFN(mcp3008Ctrl, m_indiP_fpsSource)

} // namespace app
} // namespace MagAOX

#endif // mcp3008Ctrl_hpp
