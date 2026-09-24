/** \file orcaCtrl.hpp
 * \brief The MagAO-X Hamamatsu Orca-Quest2 camera controller.
 *
 * \author Joshua Liberman (jliberman54@gmail.com)
 *
 * \ingroup orcaCtrl_files
 */

#ifndef orcaCtrl_hpp
#define orcaCtrl_hpp

// #include <ImageStruct.h>
#include <ImageStreamIO/ImageStreamIO.h>

#include <dcamapi4.h>
#include <dcamprop.h>

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

#define DEBUG

#ifdef DEBUG
    #define BREADCRUMB std::cerr << __FILE__ << " " << __LINE__ << "\n";
#else
    #define BREADCRUMB
#endif

// DCAM error string helper function (don't need to convert enum2string)
inline std::string dcamErrorString( HDCAM hdcam, DCAMERR error )
{
    char text[256]{};

    DCAMDEV_STRING info{};
    info.size      = sizeof( info );
    info.iString   = static_cast<int32>( error );
    info.text      = text;
    info.textbytes = sizeof( text );

    if( failed( dcamdev_getstring( hdcam, &info ) ) )
    {
        return "DCAM error " + std::to_string( static_cast<int32>( error ) );
    }

    return text;
}

// Helper function to get device ID string from DCAM
inline std::string dcamDeviceString( HDCAM hdcam, DCAM_IDSTR stringID )
{
    char text[256]{};

    DCAMDEV_STRING info{};
    info.size      = sizeof( info );
    info.iString   = static_cast<int32>( stringID );
    info.text      = text;
    info.textbytes = sizeof( text );

    if( failed( dcamdev_getstring( hdcam, &info ) ) )
    {
        return {};
    }

    return text;
}

namespace MagAOX
{
namespace app
{

/** \defgroup orcaCtrl Hamamatsu Orca Quest 2 Camera
 * \brief Control of a Hamamatsu Orca Quest 2 Camera.
 *
 * <a href="../handbook/operating/software/apps/orcaCtrl.html">Application Documentation</a>
 *
 * \ingroup apps
 *
 */

/** \defgroup orcaCtrl_files Hamamatsu Orca Quest 2 Camera Files
 * \ingroup orcaCtrl
 */

/** MagAO-X application to control a Hamamatsu Orca Quest 2
 *
 * \ingroup orcaCtrl
 *
 * \todo Config item for ImageStreamIO name filename
 * \todo implement ImageStreamIO circular buffer, with config setting
 */
class orcaCtrl : public MagAOXApp<>,
                 public dev::stdCamera<orcaCtrl>,
                 public dev::frameGrabber<orcaCtrl>,
                 //   public dev::dssShutter<orcaCtrl>,
                 public dev::telemeter<orcaCtrl>
{

    friend class dev::stdCamera<orcaCtrl>;
    friend class dev::frameGrabber<orcaCtrl>;
    // friend class dev::dssShutter<orcaCtrl>;
    friend class dev::telemeter<orcaCtrl>;

    typedef MagAOXApp<> MagAOXAppT;

  public:
    /** \name app::dev Configurations
     *@{
     */
    static constexpr bool c_stdCamera_tempControl =
        true; ///< app::dev config to tell stdCamera to expose temperature controls

    static constexpr bool c_stdCamera_temp = true; ///< app::dev config to tell stdCamera to expose temperature

    static constexpr bool c_stdCamera_readoutSpeed =
        true; ///< app::dev config to tell stdCamera to expose readout speed controls

    static constexpr bool c_stdCamera_vShiftSpeed =
        false; ///< app:dev config to tell stdCamera not to expose vertical shift speed control
    static constexpr bool c_stdCamera_fanSpeed =
        true; ///< app::dev config to tell stdCamera to expose fan-speed control

    static constexpr bool c_stdCamera_emGain =
        false; ///< app::dev config to tell stdCamera to not expose EM gain controls

    static constexpr bool c_stdCamera_exptimeCtrl =
        true; ///< app::dev config to tell stdCamera to expose exposure time controls

    static constexpr bool c_stdCamera_fpsCtrl = false; ///< app::dev config to tell stdCamera not to expose FPS controls

    static constexpr bool c_stdCamera_fps = true; ///< app::dev config to tell stdCamera not to expose FPS status

    static constexpr bool c_stdCamera_synchro =
        false; ///< app::dev config to tell stdCamera to not expose synchro mode controls

    static constexpr bool c_stdCamera_usesModes =
        false; ///< app:dev config to tell stdCamera not to expose mode controls

    static constexpr bool c_stdCamera_usesROI = true; ///< app:dev config to tell stdCamera to expose ROI controls

    static constexpr bool c_stdCamera_cropMode =
        false; ///< app:dev config to tell stdCamera to not expose Crop Mode controls

    static constexpr bool c_stdCamera_hasShutter =
        false; ///< app:dev config to tell stdCamera to not expose shutter controls

    static constexpr bool c_stdCamera_hasFocus =
        false; ///< app:dev config to tell stdCamera to not expose focus-state reporting and goto-focus control

    static constexpr bool c_stdCamera_usesStateString =
        false; ///< app::dev confg to tell stdCamera to expose the state string property

    static constexpr bool c_frameGrabber_flippable =
        true; ///< app:dev config to tell framegrabber this camera can be flipped
              ///@}

  protected:
    /** \name configurable parameters
     *@{
     */
    std::string m_serialNumber; ///< The camera's identifying serial number

    ///@}

    int m_depth{ 0 };

    int32  m_frameSize;
    int32  m_frameCount; ///< number of frames in the circular buffer
    double m_camera_timestamp{ 0.0 };
    double m_FrameRateCalculation;
    double m_ReadOutTimeCalculation;

    // std::string m_fxngenName{ "fxngensync" }; ///< Default fxngen device name
    // std::string m_fxngenCh{ "C2" };           ///< Default fxngen channel

    std::string m_otherCamName;

    HDCAM         m_cameraHandle{ nullptr };
    HDCAM         m_modelHandle{ nullptr };
    HDCAMWAIT     m_waitHandle{ nullptr };
    DCAMBUF_FRAME m_currentFrame{}; ///< most recently locked DCAM acq frame

    std::string m_cameraName;
    std::string m_cameraModel;

    bool m_dcamBuffersAllocated{ false }; ///< True when the DCAM buffers have been allocated

    bool m_fanControlSupported{ false }; ///< True when the camera exposes the DisableCoolingFan control parameter.
    bool m_fanStatusSupported{ false };  ///< True when the camera exposes readable cooling-fan status.
    bool m_fanForcedOn{ false };         ///< True while the camera reports the cooling fan is forced on for protection.
    bool m_fanSpeedLogPending{
        false }; ///< True when the next successful fan apply should emit a notice even without a state change.

  public:
    /// Default c'tor
    orcaCtrl();

    /// Destructor
    ~orcaCtrl() noexcept;

    /// Setup the configuration system (called by MagAOXApp::setup())
    virtual void setupConfig();

    /// load the configuration system results (called by MagAOXApp::setup())
    virtual void loadConfig();

    /// Startup functions
    /** Sets up the INDI vars.
     *
     */
    virtual int appStartup();

    /// Implementation of the FSM for the Siglent SDG
    virtual int appLogic();

    /// Implementation of the on-power-off FSM logic
    virtual int onPowerOff();

    /// Implementation of the while-powered-off FSM
    virtual int whilePowerOff();

    /// Do any needed shutdown tasks.  Currently nothing in this app.
    virtual int appShutdown();

  protected:
    int getorcaParameter( int32 &value, int32 parameter );

    int getorcaParameter( double &value, int32 parameter );

    // int setorcaParameter( int32 parameter, double value, bool commit = true );

    int setorcaParameter( int32 parameter, int32 value, bool commit = true );

    int setorcaParameter( HDCAM handle, int32 parameter, double value, bool commit = true );

    int setorcaParameter( HDCAM handle, int32 parameter, int32 value, bool commit = true );

    int setorcaParameter( int32 parameter, double value, bool commit = true );

    int setorcaParameterOnline( HDCAM handle, int32 parameter, double value );

    int setorcaParameterOnline( int32 parameter, double value );

    int setorcaParameterOnline( HDCAM handle, int32 parameter, int32 value );

    int setorcaParameterOnline( int32 parameter, int32 value );

    int connect();

    int getAcquisitionState();

    /// Get the current cooling-fan state from the camera.
    int getFanSpeed();

    int getTemps();

    // stdCamera interface:

    // This must set the power-on default values of
    /* -- m_ccdTempSetpt
     * -- m_currentROI
     */
    int powerOnDefaults();

    int setTempControl();
    int setTempSetPt();
    int setReadoutSpeed();
    /// Request a cooling-fan state change through the next reconfiguration.
    int setFanSpeed();
    int setExpTime();
    int capExpTime( double &exptime );
    int setFPS();

    // Apply MagAO-X ROI w/ center ref via DCAM subarray props
    int setDcamRoi( int32 xCen /** pix units */,
                    int32 yCen /** pix units */,
                    int32 width /** pix units */,
                    int32 height /** pix units */,
                    int32 binX /** binning factor */,
                    int32 binY /** binning factor */ );
    /// Check the next ROI
    /** Checks if the target values are valid and adjusts them to the closest valid values if needed.
     *
     * \returns 0 if successful
     * \returns -1 otherwise
     */
    int checkNextROI();

    /// Reports whether the camera is currently in focus. [stdCamera interface]
    /**
     * \returns `true` when the configured external focus switch indicates the camera is in focus.
     * \returns `false` otherwise
     */
    // bool checkFocus();

    int setNextROI();

    /// Requests the configured focus preset. [stdCamera interface]
    /**
     * \returns 0 on success
     * \returns -1 if the goto-focus helper is not fully configured
     */
    // int gotoFocus();

    /// Sets the shutter state, via call to dssShutter::setShutterState(int) [stdCamera interface]
    /**
     * \returns 0 always
     */
    // int setShutter( int sh );

    // Framegrabber interface:
    int   configureAcquisition();
    float fps();
    int   startAcquisition();
    int   acquireAndCheckValid();
    int   loadImageIntoStream( void *dest );
    int   reconfig();

    // INDI:
  protected:
    pcf::IndiProperty m_indiP_readouttime;

  public:
    /** \name Telemeter Interface
     *
     * @{
     */
    int checkRecordTimes();

    int recordTelem( const telem_stdcam * );

    ///@}
};

inline orcaCtrl::orcaCtrl() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    m_powerMgtEnabled = true;

    // m_acqBuff.memory_size = 0;
    // m_acqBuff.memory      = 0;

    m_defaultReadoutSpeed    = "Standard";
    m_readoutSpeedNames      = { "Ultra-quiet", "Standard" };
    m_readoutSpeedNameLabels = { "Standard", "Ultra-quiet" };

    // m_defaultVShiftSpeed    = "1_2us";
    // m_vShiftSpeedNames      = { "0_7us", "1_2us", "2_0us", "5_0us" };
    // m_vShiftSpeedNameLabels = { "0.7 us", "1.2 us", "2.0 us", "5.0 us" };

    m_defaultFanSpeed    = "on";
    m_fanSpeedNames      = { "on", "off" };
    m_fanSpeedNameLabels = { "On", "Off" };
    m_fanSpeedName       = m_defaultFanSpeed;
    m_fanSpeedNameSet    = m_defaultFanSpeed;

    m_full_x = 2047.5;
    m_full_y = 1151.5;
    m_full_w = 4096;
    m_full_h = 2304;

    return;
}

inline orcaCtrl::~orcaCtrl() noexcept
{
    // Clear the buffers if they were allocated.  This is done here because the destructor is called after the
    // framegrabber thread has exited, so we can safely free the buffers.
    if( m_dcamBuffersAllocated )
    {
        dcamcap_stop( m_cameraHandle );
        dcambuf_release( m_cameraHandle );
        m_dcamBuffersAllocated = false;
    }
}

inline void orcaCtrl::setupConfig()
{
    config.add( "camera.serialNumber",
                "",
                "camera.serialNumber",
                argType::Required,
                "camera",
                "serialNumber",
                false,
                "int",
                "The identifying serial number of the camera." );

    dev::stdCamera<orcaCtrl>::setupConfig( config );
    dev::frameGrabber<orcaCtrl>::setupConfig( config );
    // dev::dssShutter<orcaCtrl>::setupConfig( config );
    dev::telemeter<orcaCtrl>::setupConfig( config );
}

inline void orcaCtrl::loadConfig()
{

    config( m_serialNumber, "camera.serialNumber" );
    dev::stdCamera<orcaCtrl>::loadConfig( config );
    dev::frameGrabber<orcaCtrl>::loadConfig( config );
    // dev::dssShutter<orcaCtrl>::loadConfig( config );
    dev::telemeter<orcaCtrl>::loadConfig( config );
}

inline int orcaCtrl::appStartup()
{

    // DELETE ME
    // m_outfile = fopen("/home/xsup/test2.txt", "w");

    createROIndiNumber( m_indiP_readouttime, "readout_time", "Readout Time (s)" );
    indi::addNumberElement<float>(
        m_indiP_readouttime, "value", 0.0, std::numeric_limits<float>::max(), 0.0, "%0.1f", "readout time" );
    registerIndiPropertyReadOnly( m_indiP_readouttime );

    m_minTemp  = -35;
    m_maxTemp  = 25;
    m_stepTemp = 0;

    m_minROIx  = 0;
    m_maxROIx  = 4096;
    m_stepROIx = 0;

    m_minROIy  = 0;
    m_maxROIy  = 2304;
    m_stepROIy = 0;

    m_minROIWidth  = 1;
    m_maxROIWidth  = 4096;
    m_stepROIWidth = 4;

    m_minROIHeight  = 1;
    m_maxROIHeight  = 2304;
    m_stepROIHeight = 1;

    m_minROIBinning_x  = 1;
    m_maxROIBinning_x  = 4;
    m_stepROIBinning_x = 1;

    m_minROIBinning_y  = 1;
    m_maxROIBinning_y  = 4;
    m_stepROIBinning_y = 1;

    if( dev::stdCamera<orcaCtrl>::appStartup() < 0 )
    {
        return log<software_critical, -1>( { __FILE__, __LINE__ } );
    }

    if( dev::frameGrabber<orcaCtrl>::appStartup() < 0 )
    {
        return log<software_critical, -1>( { __FILE__, __LINE__ } );
    }

    if( dev::telemeter<orcaCtrl>::appStartup() < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__ } );
    }

    return 0;
}

inline int orcaCtrl::appLogic()
{
    // and run stdCamera's appLogic
    if( dev::stdCamera<orcaCtrl>::appLogic() < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__ } );
    }

    // first run frameGrabber's appLogic to see if the f.g. thread has exited.
    if( dev::frameGrabber<orcaCtrl>::appLogic() < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__ } );
    }

    if( state() == stateCodes::NOTCONNECTED || state() == stateCodes::NODEVICE || state() == stateCodes::ERROR )
    {
        m_reconfig = true; // Trigger a f.g. thread reconfig.

        // Might have gotten here because of a power off.
        if( powerState() != 1 || powerStateTarget() != 1 )
            return 0;

        std::cerr << __LINE__ << '\n';
        std::unique_lock<std::mutex> lock( m_indiMutex );
        if( connect() < 0 )
        {
            if( powerState() != 1 || powerStateTarget() != 1 )
                return 0;
            log<software_error>( { __FILE__, __LINE__ } );
        }

        if( state() != stateCodes::CONNECTED )
            return 0;
    }

    if( state() == stateCodes::CONNECTED )
    {
        // Get a lock
        std::unique_lock<std::mutex> lock( m_indiMutex );

        if( getAcquisitionState() < 0 )
        {
            if( powerState() != 1 || powerStateTarget() != 1 )
                return 0;
            return log<software_error, 0>( { __FILE__, __LINE__ } );
        }

        if( setTempSetPt() < 0 ) // m_ccdTempSetpt already set on power on
        {
            if( powerState() != 1 || powerStateTarget() != 1 )
                return 0;
            return log<software_error, 0>( { __FILE__, __LINE__ } );
        }

        if( m_fanSpeedControlEnabled && m_fanStatusSupported && getFanSpeed() < 0 )
        {
            if( powerState() != 1 || powerStateTarget() != 1 )
                return 0;
            return log<software_error, 0>( { __FILE__, __LINE__ } );
        }

        if( frameGrabber<orcaCtrl>::updateINDI() < 0 )
        {
            return log<software_error, 0>( { __FILE__, __LINE__ } );
        }
    }

    if( state() == stateCodes::READY || state() == stateCodes::OPERATING )
    {
        // Get a lock if we can
        std::unique_lock<std::mutex> lock( m_indiMutex, std::try_to_lock );

        // but don't wait for it, just go back around.
        if( !lock.owns_lock() )
            return 0;

        if( getAcquisitionState() < 0 )
        {
            if( powerState() != 1 || powerStateTarget() != 1 )
                return 0;

            state( stateCodes::ERROR );
            return 0;
        }

        if( getTemps() < 0 )
        {
            if( powerState() != 1 || powerStateTarget() != 1 )
                return 0;

            state( stateCodes::ERROR );
            return 0;
        }

        if( m_fanSpeedControlEnabled && m_fanStatusSupported && getFanSpeed() < 0 )
        {
            if( powerState() != 1 || powerStateTarget() != 1 )
                return 0;

            state( stateCodes::ERROR );
            return 0;
        }

        if( stdCamera<orcaCtrl>::updateINDI() < 0 )
        {
            return log<software_error, 0>( { __FILE__, __LINE__ } );
        }

        if( frameGrabber<orcaCtrl>::updateINDI() < 0 )
        {
            return log<software_error, 0>( { __FILE__, __LINE__ } );
        }

        if( telemeter<orcaCtrl>::appLogic() < 0 )
        {
            log<software_error>( { __FILE__, __LINE__ } );
            return 0;
        }
    }

    // Fall through check?
    return 0;
}

inline int orcaCtrl::onPowerOff()
{
    std::lock_guard<std::mutex> lock( m_indiMutex );

    if( m_cameraHandle )
    {
        dcamdev_close( m_cameraHandle );
        m_cameraHandle = nullptr;
    }

    dcamapi_uninit();

    // if( dssShutter<orcaCtrl>::onPowerOff() < 0 )
    // {
    //     log<software_error>( { __FILE__, __LINE__ } );
    // }

    if( stdCamera<orcaCtrl>::onPowerOff() < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
    }

    return 0;
}

inline int orcaCtrl::whilePowerOff()
{
    // if( dssShutter<orcaCtrl>::whilePowerOff() < 0 )
    // {
    //     log<software_error>( { __FILE__, __LINE__ } );
    // }

    if( stdCamera<orcaCtrl>::onPowerOff() < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
    }

    return 0;
}

inline int orcaCtrl::appShutdown()
{
    dev::frameGrabber<orcaCtrl>::appShutdown();

    if( m_cameraHandle )
    {
        dcamwait_close( m_waitHandle );
        m_waitHandle = nullptr;

        dcamdev_close( m_cameraHandle );
        m_cameraHandle = nullptr;
    }

    dcamapi_uninit();

    ///\todo error check these base class fxns.
    dev::frameGrabber<orcaCtrl>::appShutdown();
    // dev::dssShutter<orcaCtrl>::appShutdown();

    return 0;
}

inline int orcaCtrl::getorcaParameter( double &value, int32 property )
{
    const DCAMERR error = dcamprop_getvalue( m_cameraHandle, property, &value );

    if( failed( error ) )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
        {
            return -1;
        }
        log<software_error>( { __FILE__, __LINE__, 0, error, dcamErrorString( m_cameraHandle, error ) } );
        return -1;
    }

    return 0;
}

inline int orcaCtrl::getorcaParameter( int32 &value, int32 property )
{
    double rawValue = 0.0;

    if( getorcaParameter( rawValue, property ) < 0 )
    {
        return -1;
    }

    value = static_cast<int32>( rawValue );
    return 0;
}

inline int orcaCtrl::setorcaParameter( int32 parameter, double value, bool commit )
{
    DCAMERR error = dcamprop_setgetvalue( m_cameraHandle, parameter, &value );
    if( error != DCAMERR_NONE )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;
        log<software_error>( { __FILE__, __LINE__, 0, error, dcamErrorString( m_cameraHandle, error ) } );
        return -1;
    }

    return 0;
}

inline int orcaCtrl::setorcaParameter( HDCAM handle, int32 parameter, double value, bool commit )
{
    DCAMERR error = dcamprop_setgetvalue( handle, parameter, &value );
    if( error != DCAMERR_NONE )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;
        log<software_error>( { __FILE__, __LINE__, 0, error, dcamErrorString( m_cameraHandle, error ) } );
        return -1;
    }

    return 0;
}

inline int orcaCtrl::setorcaParameterOnline( HDCAM handle, int32 parameter, double value )
{
    DCAMERR error = dcamprop_setvalue( handle, parameter, value );
    if( error != DCAMERR_NONE )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;
        log<software_error>( { __FILE__, __LINE__, 0, error, dcamErrorString( m_cameraHandle, error ) } );
        return -1;
    }

    return 0;
}

inline int orcaCtrl::setorcaParameterOnline( int32 parameter, double value )
{
    return setorcaParameterOnline( m_cameraHandle, parameter, value );
}

inline int orcaCtrl::connect()
{
    std::cerr << __LINE__ << '\n';
    DCAMAPI_INIT apiInit{};
    apiInit.size = sizeof( apiInit );

    DCAMERR error = dcamapi_init( &apiInit );

    if( !failed( dcambuf_alloc( m_cameraHandle, m_frameCount ) ) )
    {
        m_dcamBuffersAllocated = true;
    }

    std::cerr << __LINE__ << '\n';

    dcamdev_close( m_cameraHandle );

    std::cerr << __LINE__ << '\n';

    dcamapi_uninit();

    std::cerr << __LINE__ << '\n';

    // Have to initialize the library every time.  Otherwise we won't catch a newly booted camera.
    dcamapi_init( &apiInit );

    std::cerr << __LINE__ << '\n';

    if( m_cameraHandle )
    {
        dcamdev_close( m_cameraHandle );
        m_cameraHandle = nullptr;
    }

    std::cerr << __LINE__ << '\n';

    error = dcamapi_init( &apiInit );

    if( failed( error ) )
    {
        return -1;
    }

    const int32 deviceCount = apiInit.iDeviceCount;

    std::cerr << __LINE__ << '\n';

    if( powerState() != 1 || powerStateTarget() != 1 )
        return 0;

    std::cerr << __LINE__ << '\n';

    if( deviceCount == 0 )
    {
        dcamapi_uninit();

        state( stateCodes::NODEVICE );
        if( !stateLogged() )
        {
            log<text_log>( "no Hamamatsu available.", logPrio::LOG_NOTICE );
        }
        return 0;
    }
    else
    {
        std::cerr << "found " << deviceCount << " Hamamatsu.\n";
    }

    // Loop over device idxs and open each device until cam is detected
    for( int32 index = 0; index < deviceCount; ++index )
    {
        DCAMDEV_OPEN deviceOpen{};
        deviceOpen.size  = sizeof( deviceOpen );
        deviceOpen.index = index;

        if( failed( dcamdev_open( &deviceOpen ) ) )
        {
            continue;
        }
        // Query camera id / model

        const std::string cameraID = dcamDeviceString( deviceOpen.hdcam, DCAM_IDSTR_CAMERAID );

        if( cameraID == m_serialNumber )
        {
            log<text_log>( "Found camera with ID " + m_serialNumber );
            m_cameraName   = dcamDeviceString( deviceOpen.hdcam, DCAM_IDSTR_VENDOR );
            m_cameraModel  = dcamDeviceString( deviceOpen.hdcam, DCAM_IDSTR_MODEL );
            m_cameraHandle = deviceOpen.hdcam;

            // open a wait handle to the camera
            DCAMWAIT_OPEN waitOpen{};
            waitOpen.size  = sizeof( waitOpen );
            waitOpen.hdcam = m_cameraHandle;

            DCAMERR error = dcamwait_open( &waitOpen );

            if( failed( error ) )
            {
                log<software_error>( { __FILE__, __LINE__, 0, error, dcamErrorString( m_cameraHandle, error ) } );
                dcamdev_close( m_cameraHandle );
                m_cameraHandle = nullptr;
                return -1;
            }

            m_waitHandle = waitOpen.hwait;

            m_fanControlSupported = false;
            m_fanStatusSupported  = false;

            // Check for camera cooling fan support
            DCAMPROP_ATTR fanAttr{};
            fanAttr.cbSize = sizeof( fanAttr );
            fanAttr.iProp  = DCAM_IDPROP_SENSORCOOLERFAN;

            if( m_fanSpeedControlEnabled )
            {

                error                 = dcamprop_getattr( deviceOpen.hdcam, &fanAttr );
                m_fanControlSupported = error == DCAMERR_NONE && ( fanAttr.attribute & DCAMPROP_ATTR_WRITABLE );

                if( failed( error ) && error != DCAMERR_NOTSUPPORT )
                {
                    if( powerState() != 1 || powerStateTarget() != 1 )
                        return 0;

                    state( stateCodes::ERROR );
                    log<software_error>( { __FILE__, __LINE__, 0, error, "Error checking CoolingFan support." } );
                    dcamdev_close( deviceOpen.hdcam );
                    return -1;
                }

                // m_fanControlSupported = exists;

                if( !m_fanControlSupported )
                {
                    if( powerState() != 1 || powerStateTarget() != 1 )
                        return 0;

                    state( stateCodes::ERROR );
                    log<software_error>(
                        { __FILE__,
                          __LINE__,
                          "Fan control enabled in config, but DisableCoolingFan is not supported by this camera." } );
                    dcamdev_close( deviceOpen.hdcam );
                    return -1;
                }

                // Check that the cooling fan status exists
                // TODO: revisit this

                DCAMPROP_ATTR coolerAttr{};
                coolerAttr.cbSize = sizeof( coolerAttr );
                coolerAttr.iProp  = DCAM_IDPROP_SENSORCOOLER;

                error                = dcamprop_getattr( deviceOpen.hdcam, &coolerAttr );
                m_fanStatusSupported = error == DCAMERR_NONE && ( coolerAttr.attribute & DCAMPROP_ATTR_READABLE );

                if( failed( error ) && error != DCAMERR_NOTSUPPORT )
                {
                    if( powerState() != 1 || powerStateTarget() != 1 )
                        return 0;

                    state( stateCodes::ERROR );
                    log<software_error>( { __FILE__, __LINE__, 0, error, "Error checking CoolingFanStatus support." } );
                    dcamdev_close( deviceOpen.hdcam );
                    return -1;
                }

                if( m_fanStatusSupported )
                {
                    int32      readableStatus = 0;
                    const bool readable       = !failed( error ) && readableStatus == DCAMPROP_ATTR_READABLE;

                    error = dcamprop_getattr( deviceOpen.hdcam, &coolerAttr );
                    if( failed( error ) && error != DCAMERR_NOTSUPPORT )
                    {
                        if( powerState() != 1 || powerStateTarget() != 1 )
                            return 0;

                        state( stateCodes::ERROR );
                        log<software_error>(
                            { __FILE__, __LINE__, 0, error, "Error checking CoolingFanStatus readability." } );
                        dcamdev_close( deviceOpen.hdcam );
                        return -1;
                    }

                    m_fanStatusSupported = readable;
                }

                if( !m_fanStatusSupported )
                {
                    log<text_log>(
                        "cooling-fan status parameter unavailable; using commanded state without hardware readback",
                        logPrio::LOG_NOTICE );
                }
            }

            state( stateCodes::CONNECTED );
            log<text_log>( "Connected to " + m_cameraName + " [S/N " + m_serialNumber + "]" );

            dcamdev_close( deviceOpen.hdcam );

            m_readoutSpeedNameSet = m_defaultReadoutSpeed;
            // m_vShiftSpeedNameSet  = m_defaultVShiftSpeed;
            if( m_fanSpeedControlEnabled )
            {
                m_fanSpeedNameSet    = m_defaultFanSpeed;
                m_fanSpeedLogPending = true;
            }

            return 0;
        }
        else
        {
            if( powerState() != 1 || powerStateTarget() != 1 )
                return 0;

            state( stateCodes::ERROR );
            if( !stateLogged() )
            {
                log<software_error>( { __FILE__, __LINE__, 0, error, "Error connecting to camera." } );
            }

            dcamdev_close( deviceOpen.hdcam );

            dcamapi_uninit();
            return -1;
        }
        state( stateCodes::NODEVICE );
        if( !stateLogged() )
        {
            log<text_log>( "Camera not found in available ids." );
        }

        dcamdev_close( deviceOpen.hdcam );

        dcamapi_uninit();

        return 0;
    }
}

inline int orcaCtrl::getAcquisitionState()
{
    int32 captureStatus = 0;

    DCAMERR error = dcamcap_status( m_cameraHandle, &captureStatus );

    const bool acquisitionRunning = !failed( error ) && captureStatus == DCAMCAP_STATUS_BUSY;

    if( MagAOXAppT::m_powerState == 0 )
        return 0;

    if( error != DCAMERR_NONE )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;
        log<software_error>( { __FILE__, __LINE__, 0, error, dcamErrorString( m_cameraHandle, error ) } );
        state( stateCodes::ERROR );
        return -1;
    }

    if( acquisitionRunning )
        state( stateCodes::OPERATING );
    else
        state( stateCodes::READY );

    if( !acquisitionRunning )
    {
        log<text_log>( "acqusition stopped. restarting", logPrio::LOG_ERROR );
        m_reconfig = true;
    }

    return 0;
}

inline int orcaCtrl::getTemps()
{
    double currTemperature;

    if( getorcaParameter( currTemperature, DCAM_IDPROP_SENSORTEMPERATURE ) < 0 )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;

        log<software_error>( { __FILE__, __LINE__ } );
        state( stateCodes::ERROR );
        return -1;
    }

    m_ccdTemp = currTemperature;

    // orcaSensorTemperatureStatus
    int32 status;

    if( getorcaParameter( status, DCAM_IDPROP_SENSORCOOLERSTATUS ) < 0 )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;

        log<software_error>( { __FILE__, __LINE__ } );
        state( stateCodes::ERROR );
        return -1;
    }

    if( status == DCAMPROP_SENSORCOOLERSTATUS__BUSY )
    {
        m_tempControlStatus    = true;
        m_tempControlOnTarget  = false;
        m_tempControlStatusStr = "UNLOCKED";
    }
    else if( status == DCAMPROP_SENSORCOOLERSTATUS__READY )
    {
        m_tempControlStatus    = true;
        m_tempControlOnTarget  = true;
        m_tempControlStatusStr = "LOCKED";
    }
    else if( status == DCAMPROP_SENSORCOOLERSTATUS__WARNING )
    {
        m_tempControlStatus    = false;
        m_tempControlOnTarget  = false;
        m_tempControlStatusStr = "FAULTED";
        log<text_log>( "temperature control faulted", logPrio::LOG_ALERT );
    }
    else
    {
        m_tempControlStatus    = false;
        m_tempControlOnTarget  = false;
        m_tempControlStatusStr = "UNKNOWN";
    }

    recordCamera();

    return 0;
}

inline int orcaCtrl::getFanSpeed()
{
    if( !m_fanStatusSupported )
        return 0;

    int32 status;

    if( getorcaParameter( status, DCAM_IDPROP_SENSORCOOLER ) < 0 )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;

        log<software_error>( { __FILE__, __LINE__ } );
        state( stateCodes::ERROR );
        return -1;
    }

    // bool fanForcedOn = false;

    if( status == DCAMPROP_SENSORCOOLER__OFF )
    {
        m_fanSpeedName = "off";
    }
    else if( status == DCAMPROP_SENSORCOOLER__ON )
    {
        m_fanSpeedName = "on";
    }
    // else if( status == orcaCoolingFanStatus_ForcedOn )
    // {
    //     m_fanSpeedName = "on";
    //     fanForcedOn    = true;
    // }
    else
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "Unknown cooling-fan status returned by orca." } );
    }

    // if( fanForcedOn && !m_fanForcedOn )
    // {
    //     log<text_log>( "cooling fan forced on by camera", logPrio::LOG_NOTICE );
    // }

    // m_fanForcedOn   = fanForcedOn;
    m_fanSpeedValid = true;
    recordCamera();

    return 0;
}

inline int orcaCtrl::setFPS()
{
    return 0;
}

inline int orcaCtrl::powerOnDefaults()
{
    m_ccdTempSetpt = -35; // This is the power on setpoint

    /*m_currentROI.x = 511.5;
    m_currentROI.y = 511.5;
    m_currentROI.w = 1024;
    m_currentROI.h = 1024;
    m_currentROI.bin_x = 1;
    m_currentROI.bin_y = 1;*/

    m_readoutSpeedName = "Standard";
    // m_vShiftSpeedName  = "1_2us";

    if( m_fanSpeedControlEnabled )
    {
        m_fanSpeedName    = m_defaultFanSpeed;
        m_fanSpeedNameSet = m_defaultFanSpeed;
    }
    else
    {
        m_fanSpeedName.clear();
        m_fanSpeedNameSet.clear();
    }

    m_fanForcedOn         = false;
    m_fanControlSupported = false;
    m_fanStatusSupported  = false;
    m_fanSpeedValid       = false;
    m_fanSpeedLogPending  = m_fanSpeedControlEnabled;

    return 0;
}

inline int orcaCtrl::setTempControl()
{
    // Always on
    m_tempControlStatus    = true;
    m_tempControlStatusSet = true;
    updateSwitchIfChanged( m_indiP_tempcont, "toggle", pcf::IndiElement::On, INDI_IDLE );
    recordCamera( true );
    return 0;
}

inline int orcaCtrl::setTempSetPt()
{
    ///\todo bounds check here.
    recordCamera( true );
    m_reconfig = true;
    return 0;
}

inline int orcaCtrl::setReadoutSpeed()
{
    recordCamera( true );
    m_reconfig = true;
    return 0;
}

inline int orcaCtrl::setFanSpeed()
{
    m_fanSpeedLogPending = true;
    recordCamera( true );
    m_reconfig = true;
    return 0;
}

inline int orcaCtrl::setExpTime()
{
    long   intexptime = m_expTimeSet * 1000 * 10000 + 0.5;
    double exptime    = ( (double)intexptime ) / 10000;
    capExpTime( exptime );

    int rv;

    recordCamera( true );

    if( state() == stateCodes::OPERATING )
    {
        rv = setorcaParameterOnline( m_modelHandle, DCAM_IDPROP_EXPOSURETIME, exptime );
    }
    else
    {
        rv = setorcaParameter( m_modelHandle, DCAM_IDPROP_EXPOSURETIME, exptime );
    }

    if( rv < 0 )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;
        log<software_error>( { __FILE__, __LINE__, "Error setting exposure time" } );
        return -1;
    }

    m_expTime = exptime / 1000.0;

    recordCamera( true );

    updateIfChanged( m_indiP_exptime, "current", m_expTime, INDI_IDLE );

    if( getorcaParameter( m_FrameRateCalculation, DCAM_IDPROP_INTERNALFRAMERATE ) < 0 )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;
        log<software_error>( { __FILE__, __LINE__, "could not get FrameRateCalculation" } );
    }
    m_fps = m_FrameRateCalculation;

    recordCamera( true );

    return 0;
}

inline int orcaCtrl::capExpTime( double &exptime )
{
    // cap at minimum possible value
    if( exptime < m_ReadOutTimeCalculation )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;
        log<text_log>( "Got exposure time " + std::to_string( exptime ) + " ms but min value is " +
                       std::to_string( m_ReadOutTimeCalculation ) + " ms" );
        long intexptime = m_ReadOutTimeCalculation * 10000 + 0.5;
        exptime         = ( (double)intexptime ) / 10000;
    }

    return 0;
}

// helper func to set DCAM ROI
inline int orcaCtrl::setDcamRoi( int32 xCen, int32 yCen, int32 width, int32 height, int32 binX, int32 binY )
{
    // Define sensor dim constants
    constexpr int32 sensorW = 4096;
    constexpr int32 sensorH = 2304;

    if( width <= 0 || height <= 0 || binX <= 0 || binY <= 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "Invalid ROI parameters or binning." } );
    }

    int32 x = xCen - ( width - 1 ) / 2;
    int32 y = yCen - ( height - 1 ) / 2;

    if( m_defaultFlip == fgFlipLR || m_defaultFlip == fgFlipUDLR )
    {
        x = sensorW - x - width;
    }

    if( m_defaultFlip == fgFlipUD || m_defaultFlip == fgFlipUDLR )
    {
        y = sensorH - y - height;
    }

    if( x < 0 || y < 0 || ( x + width ) > sensorW || ( y + height ) > sensorH )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "ROI out of bounds." } );
    }

    if( setorcaParameter( DCAM_IDPROP_SUBARRAYMODE, DCAMPROP_MODE__OFF ) < 0 ||
        setorcaParameter( DCAM_IDPROP_BINNING_HORZ, binX ) < 0 ||
        setorcaParameter( DCAM_IDPROP_BINNING_VERT, binY ) < 0 || setorcaParameter( DCAM_IDPROP_SUBARRAYHPOS, x ) < 0 ||
        setorcaParameter( DCAM_IDPROP_SUBARRAYVPOS, y ) < 0 ||
        setorcaParameter( DCAM_IDPROP_SUBARRAYHSIZE, width ) < 0 ||
        setorcaParameter( DCAM_IDPROP_SUBARRAYVSIZE, height ) < 0 ||
        setorcaParameter( DCAM_IDPROP_SUBARRAYMODE, DCAMPROP_MODE__ON ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "Error setting ROI parameters." } );
    }
    return 0;
}

inline int orcaCtrl::checkNextROI()
{
    return 0;
}

// Set ROI property to busy if accepted, set toggle to Off and Idlw either way.
// Set ROI actual
// Update current values (including struct and indiP) and set to OK when done
inline int orcaCtrl::setNextROI()
{
    updateSwitchIfChanged( m_indiP_roi_set, "request", pcf::IndiElement::Off, INDI_IDLE );
    m_reconfig = true;

    return 0;
}

inline int orcaCtrl::configureAcquisition()
{

    // int32 readoutStride;
    int32 framesPerReadout;
    int32 frameStride;
    // int32 frameSize;
    int32 pixelBitDepth;

    m_camera_timestamp = 0; // reset tracked timestamp

    std::unique_lock<std::mutex> lock( m_indiMutex );

    //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
    //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
    // Check Frame Transfer
    //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
    //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

    int32 cmode;
    if( getorcaParameter( cmode, DCAM_IDPROP_READOUTSPEED ) < 0 )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;
        log<software_error>( { __FILE__, __LINE__, "could not get Readout Control Mode" } );
        return -1;
    }

    //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
    //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
    // Cooling Fan
    //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
    //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

    if( m_fanSpeedControlEnabled )
    {
        static constexpr int32 c_enableCoolingFan  = 0;
        static constexpr int32 c_disableCoolingFan = 1;

        std::string priorFanSpeed     = m_fanSpeedName;
        int32       disableCoolingFan = c_enableCoolingFan;

        if( m_fanSpeedNameSet == "on" )
        {
            disableCoolingFan = c_enableCoolingFan;
        }
        else if( m_fanSpeedNameSet == "off" )
        {
            disableCoolingFan = c_disableCoolingFan;
        }
        else
        {
            if( powerState() != 1 || powerStateTarget() != 1 )
                return -1;
            log<software_error>( { __FILE__, __LINE__, "Invalid fan speed: " + m_fanSpeedNameSet } );
            state( stateCodes::ERROR );
            return -1;
        }

        if( setorcaParameter( m_modelHandle, DCAM_IDPROP_SENSORCOOLER, disableCoolingFan ) < 0 )
        {
            if( powerState() != 1 || powerStateTarget() != 1 )
                return -1;
            log<software_error>( { __FILE__, __LINE__, "Error setting cooling-fan state" } );
            state( stateCodes::ERROR );
            return -1;
        }

        m_fanSpeedName  = m_fanSpeedNameSet;
        m_fanSpeedValid = true;
        m_fanForcedOn   = false;

        if( m_fanSpeedName != priorFanSpeed )
        {
            log<text_log>( "fan speed changed from '" + priorFanSpeed + "' to '" + m_fanSpeedName + "'",
                           logPrio::LOG_NOTICE );
        }
        else if( m_fanSpeedLogPending )
        {
            log<text_log>( "fan speed set to '" + m_fanSpeedName + "'", logPrio::LOG_NOTICE );
        }

        m_fanSpeedLogPending = false;
    }

    //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
    //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
    // Temperature
    //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
    //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

    if( setorcaParameter( DCAM_IDPROP_SENSORTEMPERATURETARGET, m_ccdTempSetpt ) < 0 )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;
        log<software_error>( { __FILE__, __LINE__, "Error setting temperature setpoint" } );
        state( stateCodes::ERROR );
        return -1;
    }

    log<text_log>( "Set temperature set point: " + std::to_string( m_ccdTempSetpt ) + " C" );

    //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
    //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
    // Dimensions
    //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
    //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

    if( setDcamRoi( m_nextROI.x, m_nextROI.y, m_nextROI.w, m_nextROI.h, m_nextROI.bin_x, m_nextROI.bin_y ) < 0 )
    {
        state( stateCodes::ERROR );
        return -1;
    }

    if( getorcaParameter( frameStride, DCAM_IDPROP_IMAGE_ROWBYTES ) < 0 )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;
        log<software_error>( { __FILE__, __LINE__, "Error getting frame stride" } );
        state( stateCodes::ERROR );

        return -1;
    }

    if( getorcaParameter( framesPerReadout, DCAM_IDPROP_FRAMEBUNDLE_NUMBER ) < 0 )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;
        log<software_error>( { __FILE__, __LINE__, "Error getting frames per readout" } );
        state( stateCodes::ERROR );
        return -1;
    }

    if( getorcaParameter( m_frameSize, DCAM_IDPROP_IMAGE_FRAMEBYTES ) < 0 )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;
        log<software_error>( { __FILE__, __LINE__, "Error getting frame size" } );
        state( stateCodes::ERROR );
        return -1;
    }

    if( getorcaParameter( pixelBitDepth, DCAM_IDPROP_BITSPERCHANNEL ) < 0 )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;
        log<software_error>( { __FILE__, __LINE__, "Error getting pixel bit depth" } );
        state( stateCodes::ERROR );
        return -1;
    }
    m_depth = pixelBitDepth;

    int32 x           = 0;
    int32 y           = 0;
    int32 imageWidth  = 0;
    int32 imageHeight = 0;

    // get DCAM ROI position and size to update current values
    if( getorcaParameter( x, DCAM_IDPROP_SUBARRAYHPOS ) < 0 || getorcaParameter( y, DCAM_IDPROP_SUBARRAYVPOS ) < 0 ||
        getorcaParameter( m_currentROI.w, DCAM_IDPROP_SUBARRAYHSIZE ) < 0 ||
        getorcaParameter( m_currentROI.h, DCAM_IDPROP_SUBARRAYVSIZE ) < 0 ||
        getorcaParameter( m_currentROI.bin_x, DCAM_IDPROP_BINNING_HORZ ) < 0 ||
        getorcaParameter( m_currentROI.bin_y, DCAM_IDPROP_BINNING_VERT ) < 0 ||
        getorcaParameter( imageWidth, DCAM_IDPROP_IMAGE_WIDTH ) < 0 || imageWidth < 0 ||
        getorcaParameter( imageHeight, DCAM_IDPROP_IMAGE_HEIGHT ) < 0 || imageHeight < 0 ||
        getorcaParameter( m_frameSize, DCAM_IDPROP_IMAGE_FRAMEBYTES ) < 0 ||
        getorcaParameter( m_depth, DCAM_IDPROP_BITSPERCHANNEL ) < 0 )
    {
        log<software_error, -1>( { __FILE__, __LINE__, "Error getting ROI parameters" } );
        state( stateCodes::ERROR );
        return -1;
    }

    m_width  = static_cast<uint32_t>( imageWidth );
    m_height = static_cast<uint32_t>( imageHeight );

    // update current ROI values
    m_xbinning = m_currentROI.bin_x;
    m_ybinning = m_currentROI.bin_y;

    m_currentROI.x = x + ( m_currentROI.w - 1 ) / 2;
    m_currentROI.y = y + ( m_currentROI.h - 1 ) / 2;

    updateIfChanged( m_indiP_roi_x, "current", m_currentROI.x, INDI_OK );
    updateIfChanged( m_indiP_roi_y, "current", m_currentROI.y, INDI_OK );
    updateIfChanged( m_indiP_roi_w, "current", m_currentROI.w, INDI_OK );
    updateIfChanged( m_indiP_roi_h, "current", m_currentROI.h, INDI_OK );
    updateIfChanged( m_indiP_roi_bin_x, "current", m_currentROI.bin_x, INDI_OK );
    updateIfChanged( m_indiP_roi_bin_y, "current", m_currentROI.bin_y, INDI_OK );

    // We also update target to the settable values
    m_nextROI.x     = m_currentROI.x;
    m_nextROI.y     = m_currentROI.y;
    m_nextROI.w     = m_currentROI.w;
    m_nextROI.h     = m_currentROI.h;
    m_nextROI.bin_x = m_currentROI.bin_x;
    m_nextROI.bin_y = m_currentROI.bin_y;

    updateIfChanged( m_indiP_roi_x, "target", m_currentROI.x, INDI_OK );
    updateIfChanged( m_indiP_roi_y, "target", m_currentROI.y, INDI_OK );
    updateIfChanged( m_indiP_roi_w, "target", m_currentROI.w, INDI_OK );
    updateIfChanged( m_indiP_roi_h, "target", m_currentROI.h, INDI_OK );
    updateIfChanged( m_indiP_roi_bin_x, "target", m_currentROI.bin_x, INDI_OK );
    updateIfChanged( m_indiP_roi_bin_y, "target", m_currentROI.bin_y, INDI_OK );

    //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
    //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
    // Exposure Time and Frame Rate
    //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
    //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

    if( getorcaParameter( m_ReadOutTimeCalculation, DCAM_IDPROP_TIMING_READOUTTIME ) < 0 )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;
        return log<software_error, -1>( { __FILE__, __LINE__, "could not get ReadOutTimeCalculation" } );
    }

    std::cerr << "Readout time is: " << m_ReadOutTimeCalculation << "\n";

    updateIfChanged(
        m_indiP_readouttime, "value", m_ReadOutTimeCalculation / 1000.0, INDI_OK ); // convert from msec to sec

    DCAMPROP_ATTR attr{};
    attr.cbSize = sizeof( attr );
    attr.iProp  = DCAM_IDPROP_EXPOSURETIME;

    DCAMERR error = dcamprop_getattr( m_modelHandle, &attr );

    if( error != DCAMERR_NONE )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;
        log<text_log>( "Constraint count is not 1: " + std::to_string( attr.cbSize ) + " constraints",
                       logPrio::LOG_ERROR );
    }
    else
    {
        m_minExpTime  = attr.valuemin;
        m_maxExpTime  = attr.valuemax;
        m_stepExpTime = attr.valuestep;

        m_indiP_exptime["current"].setMin( m_minExpTime );
        m_indiP_exptime["current"].setMax( m_maxExpTime );
        m_indiP_exptime["current"].setStep( m_stepExpTime );

        m_indiP_exptime["target"].setMin( m_minExpTime );
        m_indiP_exptime["target"].setMax( m_maxExpTime );
        m_indiP_exptime["target"].setStep( m_stepExpTime );
    }

    if( m_expTimeSet > 0 )
    {
        long   intexptime = m_expTimeSet * 1000 * 10000 + 0.5;
        double exptime    = ( (double)intexptime ) / 10000;
        capExpTime( exptime );
        std::cerr << "Setting exposure time to " << m_expTimeSet << "\n";
        int rv = setorcaParameter( m_modelHandle, DCAM_IDPROP_EXPOSURETIME, exptime );

        if( rv < 0 )
        {
            if( powerState() != 1 || powerStateTarget() != 1 )
                return -1;
            return log<software_error, -1>( { __FILE__, __LINE__, "Error setting exposure time" } );
        }
    }

    double exptime;
    if( getorcaParameter( exptime, DCAM_IDPROP_EXPOSURETIME ) < 0 )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;
        return log<software_error, -1>( { __FILE__, __LINE__, "Error getting exposure time" } );
    }
    else
    {
        capExpTime( exptime );
        m_expTime    = exptime / 1000.0;
        m_expTimeSet = m_expTime; // At this point it must be true.
        updateIfChanged( m_indiP_exptime, "current", m_expTime, INDI_IDLE );
        updateIfChanged( m_indiP_exptime, "target", m_expTimeSet, INDI_IDLE );
    }

    if( getorcaParameter( m_FrameRateCalculation, DCAM_IDPROP_INTERNALFRAMERATE ) < 0 )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;
        return log<software_error, -1>( { __FILE__, __LINE__, "Error getting frame rate" } );
    }
    else
    {
        m_fps = m_FrameRateCalculation;
        updateIfChanged( m_indiP_fps, "current", m_fps, INDI_IDLE );
    }
    std::cerr << "FrameRate is: " << m_FrameRateCalculation << "\n";

    // Start continuous acquisition

    recordCamera();

    error = dcamcap_start( m_cameraHandle, DCAMCAP_START_SEQUENCE );
    if( error != DCAMERR_NONE )
    {
        log<software_error>( { __FILE__, __LINE__, 0, error, dcamErrorString( m_cameraHandle, error ) } );
        state( stateCodes::ERROR );

        return -1;
    }

    m_dataType = _DATATYPE_UINT16; // Where does this go?

    return 0;
}

inline float orcaCtrl::fps()
{
    return m_fps;
}

inline int orcaCtrl::startAcquisition()
{
    return 0;
}

inline int orcaCtrl::acquireAndCheckValid()
{
    int32 camTimeOut = 1000; // 1 second keeps us responsive without busy-waiting too much

    // Create DCAMWait struct for acquisition updates
    DCAMWAIT_START waitStart{};
    waitStart.size      = sizeof( waitStart );
    waitStart.eventmask = DCAMWAIT_CAPEVENT_FRAMEREADY | DCAMWAIT_CAPEVENT_STOPPED;
    waitStart.timeout   = camTimeOut;

    DCAMERR error = dcamwait_start( m_waitHandle, &waitStart );

    if( error == DCAMERR_TIMEOUT )
    {
        return 1; // This sends it back to framegrabber to check for reconfig, etc.
    }

    if( failed( error ) )
    {
        log<software_error>( { __FILE__, __LINE__, 0, error, dcamErrorString( m_cameraHandle, error ) } );
        state( stateCodes::ERROR );

        return -1;
    }

    // check if acq completed
    if( waitStart.eventhappened & DCAMWAIT_CAPEVENT_STOPPED )
    {
        return 1;
    }

    // check if frames transferred to computer

    // define transfer info struct for dcamcap_transferinfo
    DCAMCAP_TRANSFERINFO transferInfo{};
    transferInfo.size  = sizeof( transferInfo );
    transferInfo.iKind = DCAMCAP_TRANSFERKIND_FRAME;

    error = dcamcap_transferinfo( m_cameraHandle, &transferInfo );

    if( failed( error ) )
    {
        log<software_error>( { __FILE__, __LINE__, 0, error, dcamErrorString( m_cameraHandle, error ) } );
        state( stateCodes::ERROR );

        return -1;
    }

    // lock the retrieved data for reading
    DCAMBUF_FRAME frame{};
    frame.size   = sizeof( frame );
    frame.iFrame = transferInfo.nNewestFrameIndex; // get idx of most recently transferred frame

    error = dcambuf_lockframe( m_cameraHandle, &frame );

    if( failed( error ) )
    {
        log<software_error>( { __FILE__, __LINE__, 0, error, dcamErrorString( m_cameraHandle, error ) } );
        state( stateCodes::ERROR );

        return -1;
    }

    m_currentFrame = frame;

    clock_gettime( CLOCK_REALTIME, &m_currImageTimestamp );

    if( m_currentFrame.buf == 0 )
    {
        return 1;
    }

    // std::cerr << "readout: " << frame.buf << " " << transferInfo.nFrameCount << "\n";

    // camera time stamp
    const double cameraTimestamp =
        static_cast<double>( frame.timestamp.sec ) + 1.0e-6 * static_cast<double>( frame.timestamp.microsec );

    if( m_camera_timestamp > 0.0 )
    {
        const double delta_ts = cameraTimestamp - m_camera_timestamp;
        // check for a frame skip
        if( delta_ts > 1.5 / m_FrameRateCalculation )
        {
            std::cerr << "Skipped frame(s)! (Expected a " << 1000. / m_FrameRateCalculation << " ms gap but got "
                      << 1000 * delta_ts << " ms)\n";
        }
    }
    // print

    m_camera_timestamp = cameraTimestamp; // update to latest

    return 0;
}

inline int orcaCtrl::loadImageIntoStream( void *dest )
{
    if( frameGrabber<orcaCtrl>::loadImageIntoStreamCopy( dest, m_currentFrame.buf, m_width, m_height, m_typeSize ) ==
        nullptr )
        return -1;

    return 0;
}

inline int orcaCtrl::reconfig()
{

    int32 captureStatus = 0;

    DCAMERR error = dcamcap_stop( m_cameraHandle );
    if( error != DCAMERR_NONE )
    {
        log<software_error>( { __FILE__, __LINE__, 0, error, dcamErrorString( m_cameraHandle, error ) } );
        state( stateCodes::ERROR );

        return -1;
    }

    error                         = dcamcap_status( m_cameraHandle, &captureStatus );
    const bool acquisitionRunning = !failed( error ) && captureStatus == DCAMCAP_STATUS_BUSY;

    while( acquisitionRunning )
    {
        if( MagAOXAppT::m_powerState == 0 )
            return 0;
        sleep( 1 );

        error = dcamcap_stop( m_cameraHandle );

        if( error != DCAMERR_NONE )
        {
            log<software_error>( { __FILE__, __LINE__, 0, error, dcamErrorString( m_cameraHandle, error ) } );
            state( stateCodes::ERROR );
            return -1;
        }

        error = dcamcap_status( m_cameraHandle, &captureStatus );
        if( error != DCAMERR_NONE )
        {
            log<software_error>( { __FILE__, __LINE__, 0, error, dcamErrorString( m_cameraHandle, error ) } );
            state( stateCodes::ERROR );
            return -1;
        }
    }

    return 0;
}

int orcaCtrl::checkRecordTimes()
{
    return telemeter<orcaCtrl>::checkRecordTimes( telem_stdcam() );
}

int orcaCtrl::recordTelem( const telem_stdcam * )
{
    return recordCamera( true );
}

} // namespace app
} // namespace MagAOX
#endif
