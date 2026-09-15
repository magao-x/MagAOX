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

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

#define DEBUG

#ifdef DEBUG
    #define BREADCRUMB std::cerr << __FILE__ << " " << __LINE__ << "\n";
#else
    #define BREADCRUMB
#endif

inline std::string orcaEnum2String( orcaEnumeratedType type, int value )
{
    const pichar *string;
    orca_GetEnumerationString( type, value, &string );
    std::string str( string );
    orca_DestroyString( string );

    return str;
}

namespace MagAOX
{
namespace app
{

// int readoutParams( int &adcQual, double &adcSpeed, const std::string &rosn )
// {
//     if( rosn == "ccd_00_1MHz" )
//     {
//         adcQual  = orcaAdcQuality_LowNoise;
//         adcSpeed = 0.1;
//     }
//     else if( rosn == "ccd_01MHz" )
//     {
//         adcQual  = orcaAdcQuality_LowNoise;
//         adcSpeed = 1;
//     }
//     else if( rosn == "emccd_05MHz" )
//     {
//         adcQual  = orcaAdcQuality_ElectronMultiplied;
//         adcSpeed = 5;
//     }
//     else if( rosn == "emccd_10MHz" )
//     {
//         adcQual  = orcaAdcQuality_ElectronMultiplied;
//         adcSpeed = 10;
//     }
//     else if( rosn == "emccd_20MHz" )
//     {
//         adcQual  = orcaAdcQuality_ElectronMultiplied;
//         adcSpeed = 20;
//     }
//     else if( rosn == "emccd_30MHz" )
//     {
//         adcQual  = orcaAdcQuality_ElectronMultiplied;
//         adcSpeed = 30;
//     }
//     else
//     {
//         return -1;
//     }

//     return 0;
// }

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

    static constexpr bool c_stdCamera_emGain = false; ///< app::dev config to tell stdCamera to not expose EM gain controls

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

    // TODO: Revisit this section because DCAM has no equivalent const to Picam's record timestamp
    // int  m_timeStampMask{ DCAM_TIMESTAMP }; // time stamp at end of exposure
    int32  m_tsRes;                                                // time stamp resolution
    int32  m_frameSize;
    double m_camera_timestamp{ 0.0 };
    float  m_FrameRateCalculation;
    float  m_ReadOutTimeCalculation;

    // std::string m_fxngenName{ "fxngensync" }; ///< Default fxngen device name
    // std::string m_fxngenCh{ "C2" };           ///< Default fxngen channel

    std::string m_otherCamName;

    orcaHandle m_cameraHandle{ 0 };
    orcaHandle m_modelHandle{ 0 };

    orcaAcquisitionBuffer m_acqBuff;
    orcaAvailableData     m_available;

    std::string m_cameraName;
    std::string m_cameraModel;
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
    int getorcaParameter( int32 &value, orcaParameter parameter );

    int getorcaParameter( double &value, orcaParameter parameter );

    // int setorcaParameter( orcaParameter parameter, pi64s value, bool commit = true );

    int setorcaParameter( orcaParameter parameter, int32 value, bool commit = true );

    int setorcaParameter( orcaHandle handle, orcaParameter parameter, double value, bool commit = true );

    int setorcaParameter( orcaHandle handle, orcaParameter parameter, int32 value, bool commit = true );

    int setorcaParameter( orcaParameter parameter, double value, bool commit = true );

    int setorcaParameterOnline( orcaHandle handle, orcaParameter parameter, double value );

    int setorcaParameterOnline( orcaParameter parameter, double value );

    int setorcaParameterOnline( orcaHandle handle, orcaParameter parameter, int32 value );

    int setorcaParameterOnline( orcaParameter parameter, int32 value );

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
    int setVShiftSpeed();
    /// Request a cooling-fan state change through the next reconfiguration.
    int  setFanSpeed();
    int  setEMGain();
    int  setExpTime();
    int  capExpTime( double &exptime );
    int  setFPS();
    int  setSynchro();
    void updateFxnGenSync();

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
    // pcf::IndiProperty m_indiP_fxngensync_freq;   ///< Property for setting fxngensync frequency
    // pcf::IndiProperty m_indiP_fxngensync_output; ///< Proprety for turning on fxngensync

    // pcf::IndiProperty m_indiP_receiveSynchro; ///< Synchro that can only be triggered from the otherCam
    // pcf::IndiProperty m_indiP_receiveExptime; ///< Exptime that can only be triggered from the otherCam

    // pcf::IndiProperty m_indiP_otherCamExptime; ///< Property for setting otherCam exptime
    // pcf::IndiProperty m_indiP_otherCamSynchro; ///< Property for setting otherCam synchro

  public:
    // INDI_NEWCALLBACK_DECL( orcaCtrl, m_indiP_adcquality );

    // INDI_NEWCALLBACK_DECL( orcaCtrl, m_indiP_receiveSynchro );

    // INDI_NEWCALLBACK_DECL( orcaCtrl, m_indiP_receiveExptime );

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

    m_acqBuff.memory_size = 0;
    m_acqBuff.memory      = 0;

    m_defaultReadoutSpeed = "Standard";
    m_readoutSpeedNames   = { "Ultra-quiet", "Standard" };
    m_readoutSpeedNameLabels = {
        "Standard", "Ultra-quiet" };

    // m_defaultVShiftSpeed    = "1_2us";
    // m_vShiftSpeedNames      = { "0_7us", "1_2us", "2_0us", "5_0us" };
    // m_vShiftSpeedNameLabels = { "0.7 us", "1.2 us", "2.0 us", "5.0 us" };

    m_defaultFanSpeed    = "on";
    m_fanSpeedNames      = { "on", "off" };
    m_fanSpeedNameLabels = { "On", "Off" };
    m_fanSpeedName       = m_defaultFanSpeed;
    m_fanSpeedNameSet    = m_defaultFanSpeed;

    m_full_x = 511.5;
    m_full_y = 511.5;
    m_full_w = 1024;
    m_full_h = 1024;

    m_maxEMGain = 1000;

    return;
}

inline orcaCtrl::~orcaCtrl() noexcept
{
    if( m_acqBuff.memory )
    {
        free( m_acqBuff.memory );
    }

    return;
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
    // config( m_fxngenName, "synchro.deviceName" );
    // config( m_fxngenCh, "synchro.channel" );
    // config( m_otherCamName, "synchro.otherCamName" );

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

    // if( dev::dssShutter<orcaCtrl>::appStartup() < 0 )
    // {
    //     return log<software_critical, -1>( { __FILE__, __LINE__ } );
    // }

    if( dev::telemeter<orcaCtrl>::appStartup() < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__ } );
    }

    // m_indiP_fxngensync_freq = pcf::IndiProperty( pcf::IndiProperty::Number );
    // m_indiP_fxngensync_freq.setDevice( m_fxngenName );
    // m_indiP_fxngensync_freq.setName( m_fxngenCh + "freq" );
    // m_indiP_fxngensync_freq.add( pcf::IndiElement( "target" ) );

    // m_indiP_fxngensync_output = pcf::IndiProperty( pcf::IndiProperty::Text );
    // m_indiP_fxngensync_output.setDevice( m_fxngenName );
    // m_indiP_fxngensync_output.setName( m_fxngenCh + "outp" );
    // m_indiP_fxngensync_output.add( pcf::IndiElement( "value" ) );

    // CREATE_REG_INDI_NEW_NUMBERD( m_indiP_receiveExptime,
    //                              "receiveExptime",
    //                              m_minExpTime,
    //                              m_maxExpTime,
    //                              m_stepExpTime,
    //                              "%0.3f",
    //                              "Exptime",
    //                              "Other cam" );
    // CREATE_REG_INDI_NEW_TOGGLESWITCH( m_indiP_receiveSynchro, "receiveSynchro" );

    // m_indiP_otherCamExptime = pcf::IndiProperty( pcf::IndiProperty::Number );
    // m_indiP_otherCamExptime.setDevice( m_otherCamName );
    // m_indiP_otherCamExptime.setName( "receiveExptime" );
    // m_indiP_otherCamExptime.add( pcf::IndiElement( "target" ) );

    // m_indiP_otherCamSynchro = pcf::IndiProperty( pcf::IndiProperty::Switch );
    // m_indiP_otherCamSynchro.setDevice( m_otherCamName );
    // m_indiP_otherCamSynchro.setName( "receiveSynchro" );
    // m_indiP_otherCamSynchro.add( pcf::IndiElement( "toggle" ) );

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

    // and run dssShutter's appLogic
    // if( dev::dssShutter<orcaCtrl>::appLogic() < 0 )
    // {
    //     return log<software_error, -1>( { __FILE__, __LINE__ } );
    // }

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
        m_cameraHandle = 0;
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
        dcamdev_close( m_cameraHandle );
        m_cameraHandle = 0;
    }

    dcamapi_uninit();

    ///\todo error check these base class fxns.
    dev::frameGrabber<orcaCtrl>::appShutdown();
    dev::dssShutter<orcaCtrl>::appShutdown();

    return 0;
}

inline int orcaCtrl::getorcaParameter( int32 &value, orcaParameter parameter )
{
    orcaError error = orca_GetParameterIntegerValue( m_cameraHandle, parameter, &value );

    if( MagAOXAppT::m_powerState == 0 )
        return -1; // Flag error but don't log

    if( error != orcaError_None )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;
        log<software_error>( { __FILE__, __LINE__, 0, error, orcaEnum2String( orcaEnumeratedType_Error, error ) } );
        return -1;
    }

    return 0;
}

inline int orcaCtrl::getorcaParameter( double &value, orcaParameter parameter )
{
    orcaError error = orca_GetParameterFloatingPointValue( m_cameraHandle, parameter, &value );

    if( MagAOXAppT::m_powerState == 0 )
        return -1; // Flag error but don't log

    if( error != orcaError_None )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;
        log<software_error>( { __FILE__, __LINE__, 0, error, orcaEnum2String( orcaEnumeratedType_Error, error ) } );
        return -1;
    }

    return 0;
}

inline int orcaCtrl::setorcaParameter( orcaParameter parameter, pi64s value, bool commit )
{
    orcaError error = orca_SetParameterLargeIntegerValue( m_cameraHandle, parameter, value );
    if( error != orcaError_None )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;
        log<software_error>( { __FILE__, __LINE__, 0, error, orcaEnum2String( orcaEnumeratedType_Error, error ) } );
        return -1;
    }

    if( !commit )
        return 0;

    const orcaParameter *failed_parameters;
    int32                 failed_parameters_count;

    error = orca_CommitParameters( m_cameraHandle, &failed_parameters, &failed_parameters_count );
    if( error != orcaError_None )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;
        log<software_error>( { __FILE__, __LINE__, 0, error, orcaEnum2String( orcaEnumeratedType_Error, error ) } );
        return -1;
    }

    for( int i = 0; i < failed_parameters_count; ++i )
    {
        if( failed_parameters[i] == parameter )
        {
            orca_DestroyParameters( failed_parameters );
            return log<text_log, -1>( "Parameter not committed" );
        }
    }

    orca_DestroyParameters( failed_parameters );

    return 0;
}

inline int orcaCtrl::setorcaParameter( orcaHandle handle, orcaParameter parameter, double value, bool commit )
{
    orcaError error = orca_SetParameterFloatingPointValue( handle, parameter, value );
    if( error != orcaError_None )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;
        log<software_error>( { __FILE__, __LINE__, 0, error, orcaEnum2String( orcaEnumeratedType_Error, error ) } );
        return -1;
    }

    if( !commit )
        return 0;

    const orcaParameter *failed_parameters;
    int32                 failed_parameters_count;

    error = orca_CommitParameters( handle, &failed_parameters, &failed_parameters_count );
    if( error != orcaError_None )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;
        log<software_error>( { __FILE__, __LINE__, 0, error, orcaEnum2String( orcaEnumeratedType_Error, error ) } );
        return -1;
    }

    for( int i = 0; i < failed_parameters_count; ++i )
    {
        if( failed_parameters[i] == parameter )
        {
            orca_DestroyParameters( failed_parameters );
            return log<text_log, -1>( "Parameter not committed" );
        }
    }

    orca_DestroyParameters( failed_parameters );

    return 0;
}

inline int orcaCtrl::setorcaParameter( orcaHandle handle, orcaParameter parameter, int32 value, bool commit )
{
    orcaError error = orca_SetParameterIntegerValue( handle, parameter, value );
    if( error != orcaError_None )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;
        log<software_error>( { __FILE__, __LINE__, 0, error, orcaEnum2String( orcaEnumeratedType_Error, error ) } );
        return -1;
    }

    if( !commit )
        return 0;

    const orcaParameter *failed_parameters;
    int32                 failed_parameters_count;

    error = orca_CommitParameters( handle, &failed_parameters, &failed_parameters_count );
    if( error != orcaError_None )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;
        log<software_error>( { __FILE__, __LINE__, 0, error, orcaEnum2String( orcaEnumeratedType_Error, error ) } );
        return -1;
    }

    for( int i = 0; i < failed_parameters_count; ++i )
    {
        if( failed_parameters[i] == parameter )
        {
            orca_DestroyParameters( failed_parameters );
            return log<text_log, -1>( "Parameter not committed" );
        }
    }

    orca_DestroyParameters( failed_parameters );

    return 0;
}

inline int orcaCtrl::setorcaParameter( orcaParameter parameter, double value, bool commit )
{
    return setorcaParameter( m_cameraHandle, parameter, value, commit );
}

inline int orcaCtrl::setorcaParameter( orcaParameter parameter, int32 value, bool commit )
{
    return setorcaParameter( m_cameraHandle, parameter, value, commit );
}

inline int orcaCtrl::setorcaParameterOnline( orcaHandle handle, orcaParameter parameter, double value )
{
    orcaError error = orca_SetParameterFloatingPointValueOnline( handle, parameter, value );
    if( error != orcaError_None )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;
        log<software_error>( { __FILE__, __LINE__, 0, error, orcaEnum2String( orcaEnumeratedType_Error, error ) } );
        return -1;
    }

    return 0;
}

inline int orcaCtrl::setorcaParameterOnline( orcaParameter parameter, double value )
{
    return setorcaParameterOnline( m_cameraHandle, parameter, value );
}

inline int orcaCtrl::setorcaParameterOnline( orcaHandle handle, orcaParameter parameter, int32 value )
{
    orcaError error = orca_SetParameterIntegerValueOnline( handle, parameter, value );
    if( error != orcaError_None )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;
        log<software_error>( { __FILE__, __LINE__, 0, error, orcaEnum2String( orcaEnumeratedType_Error, error ) } );
        return -1;
    }

    return 0;
}

inline int orcaCtrl::setorcaParameterOnline( orcaParameter parameter, int32 value )
{
    return setorcaParameterOnline( m_cameraHandle, parameter, value );
}

inline int orcaCtrl::connect()
{
    std::cerr << __LINE__ << '\n';
    orcaError     error;
    orcaCameraID *id_array;
    int32          id_count;

    if( m_acqBuff.memory )
    {
        free( m_acqBuff.memory );
        m_acqBuff.memory      = NULL;
        m_acqBuff.memory_size = 0;
    }

    std::cerr << __LINE__ << '\n';

    dcamapi_uninit();

    std::cerr << __LINE__ << '\n';

    // Have to initialize the library every time.  Otherwise we won't catch a newly booted camera.
    orca_InitializeLibrary();

    std::cerr << __LINE__ << '\n';

    if( m_cameraHandle )
    {
        dcamdev_close( m_cameraHandle );
        m_cameraHandle = 0;
    }

    std::cerr << __LINE__ << '\n';

    orca_GetAvailableCameraIDs( const_cast<const orcaCameraID **>( &id_array ), &id_count );

    std::cerr << __LINE__ << '\n';

    if( powerState() != 1 || powerStateTarget() != 1 )
        return 0;

    std::cerr << __LINE__ << '\n';

    if( id_count == 0 )
    {
        orca_DestroyCameraIDs( id_array );

        dcamapi_uninit();

        state( stateCodes::NODEVICE );
        if( !stateLogged() )
        {
            log<text_log>( "no P.I. Cameras available.", logPrio::LOG_NOTICE );
        }
        return 0;
    }
    else
    {
        std::cerr << "found " << id_count << " PI cameras.\n";
    }

    for( int i = 0; i < id_count; ++i )
    {
        if( std::string( id_array[i].serial_number ) == m_serialNumber )
        {
            log<text_log>( "Camera was found.  Now connecting." );

            error = orcaAdvanced_OpenCameraDevice( &id_array[i], &m_cameraHandle );
            if( error == orcaError_None )
            {
                m_cameraName  = id_array[i].sensor_name;
                m_cameraModel = orcaEnum2String( orcaEnumeratedType_Model, id_array[i].model );

                error = orcaAdvanced_GetCameraModel( m_cameraHandle, &m_modelHandle );
                if( error != orcaError_None )
                {
                    log<software_error>( { __FILE__, __LINE__, "failed to get camera model" } );
                }

                m_fanControlSupported = false;
                m_fanStatusSupported  = false;

                if( m_fanSpeedControlEnabled )
                {
                    pibln exists = false;

                    error = orca_DoesParameterExist( m_cameraHandle, orcaParameter_DisableCoolingFan, &exists );
                    if( error != orcaError_None )
                    {
                        if( powerState() != 1 || powerStateTarget() != 1 )
                            return 0;

                        state( stateCodes::ERROR );
                        log<software_error>(
                            { __FILE__, __LINE__, 0, error, "Error checking DisableCoolingFan support." } );
                        orca_DestroyCameraIDs( id_array );
                        return -1;
                    }

                    m_fanControlSupported = exists;

                    if( !m_fanControlSupported )
                    {
                        if( powerState() != 1 || powerStateTarget() != 1 )
                            return 0;

                        state( stateCodes::ERROR );
                        log<software_error>( { __FILE__,
                                               __LINE__,
                                               "Fan control enabled in config, but DisableCoolingFan is not supported "
                                               "by this camera." } );
                        orca_DestroyCameraIDs( id_array );
                        return -1;
                    }

                    exists = false;
                    error  = orca_DoesParameterExist( m_cameraHandle, orcaParameter_CoolingFanStatus, &exists );
                    if( error != orcaError_None )
                    {
                        if( powerState() != 1 || powerStateTarget() != 1 )
                            return 0;

                        state( stateCodes::ERROR );
                        log<software_error>(
                            { __FILE__, __LINE__, 0, error, "Error checking CoolingFanStatus support." } );
                        orca_DestroyCameraIDs( id_array );
                        return -1;
                    }

                    if( exists )
                    {
                        pibln readable = false;

                        error = orca_CanReadParameter( m_cameraHandle, orcaParameter_CoolingFanStatus, &readable );
                        if( error != orcaError_None )
                        {
                            if( powerState() != 1 || powerStateTarget() != 1 )
                                return 0;

                            state( stateCodes::ERROR );
                            log<software_error>(
                                { __FILE__, __LINE__, 0, error, "Error checking CoolingFanStatus readability." } );
                            orca_DestroyCameraIDs( id_array );
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

                orca_DestroyCameraIDs( id_array );

                m_readoutSpeedNameSet = m_defaultReadoutSpeed;
                m_vShiftSpeedNameSet  = m_defaultVShiftSpeed;
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

                orca_DestroyCameraIDs( id_array );

                dcamapi_uninit();
                return -1;
            }
        }
    }

    state( stateCodes::NODEVICE );
    if( !stateLogged() )
    {
        log<text_log>( "Camera not found in available ids." );
    }

    orca_DestroyCameraIDs( id_array );

    dcamapi_uninit();

    return 0;
}

inline int orcaCtrl::getAcquisitionState()
{
    pibln running = false;

    orcaError error = orca_IsAcquisitionRunning( m_cameraHandle, &running );

    if( MagAOXAppT::m_powerState == 0 )
        return 0;

    if( error != orcaError_None )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;
        log<software_error>( { __FILE__, __LINE__, 0, error, orcaEnum2String( orcaEnumeratedType_Error, error ) } );
        state( stateCodes::ERROR );
        return -1;
    }

    if( running )
        state( stateCodes::OPERATING );
    else
        state( stateCodes::READY );

    if( !running )
    {
        log<text_log>( "acqusition stopped. restarting", logPrio::LOG_ERROR );
        m_reconfig = true;
    }

    return 0;
}

inline int orcaCtrl::getTemps()
{
    double currTemperature;

    if( getorcaParameter( currTemperature, orcaParameter_SensorTemperatureReading ) < 0 )
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

    if( getorcaParameter( status, orcaParameter_SensorTemperatureStatus ) < 0 )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;

        log<software_error>( { __FILE__, __LINE__ } );
        state( stateCodes::ERROR );
        return -1;
    }

    if( status == orcaSensorTemperatureStatus_Unlocked )
    {
        m_tempControlStatus    = true;
        m_tempControlOnTarget  = false;
        m_tempControlStatusStr = "UNLOCKED";
    }
    else if( status == orcaSensorTemperatureStatus_Locked )
    {
        m_tempControlStatus    = true;
        m_tempControlOnTarget  = true;
        m_tempControlStatusStr = "LOCKED";
    }
    else if( status == orcaSensorTemperatureStatus_Faulted )
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

    if( getorcaParameter( status, orcaParameter_CoolingFanStatus ) < 0 )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;

        log<software_error>( { __FILE__, __LINE__ } );
        state( stateCodes::ERROR );
        return -1;
    }

    bool fanForcedOn = false;

    if( status == orcaCoolingFanStatus_Off )
    {
        m_fanSpeedName = "off";
    }
    else if( status == orcaCoolingFanStatus_On )
    {
        m_fanSpeedName = "on";
    }
    else if( status == orcaCoolingFanStatus_ForcedOn )
    {
        m_fanSpeedName = "on";
        fanForcedOn    = true;
    }
    else
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "Unknown cooling-fan status returned by orca." } );
    }

    if( fanForcedOn && !m_fanForcedOn )
    {
        log<text_log>( "cooling fan forced on by camera", logPrio::LOG_NOTICE );
    }

    m_fanForcedOn   = fanForcedOn;
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
    m_ccdTempSetpt = -55; // This is the power on setpoint

    /*m_currentROI.x = 511.5;
    m_currentROI.y = 511.5;
    m_currentROI.w = 1024;
    m_currentROI.h = 1024;
    m_currentROI.bin_x = 1;
    m_currentROI.bin_y = 1;*/

    m_readoutSpeedName = "emccd_05MHz";
    m_vShiftSpeedName  = "1_2us";

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

inline int orcaCtrl::setVShiftSpeed()
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

inline int orcaCtrl::setEMGain()
{
    int32 adcQual;
    double adcSpeed;

    if( readoutParams( adcQual, adcSpeed, m_readoutSpeedName ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__, "Invalid readout speed: " + m_readoutSpeedNameSet } );
        state( stateCodes::ERROR );
        return -1;
    }

    if( adcQual != orcaAdcQuality_ElectronMultiplied )
    {
        m_emGain   = 1;
        m_adcSpeed = adcSpeed;
        recordCamera( true );
        log<text_log>( "Attempt to set EM gain while in conventional amplifier.", logPrio::LOG_NOTICE );
        return 0;
    }

    int32 emg = m_emGainSet;
    if( emg < 0 )
    {
        emg = 0;
        log<text_log>( "EM gain limited to 0", logPrio::LOG_WARNING );
    }

    if( emg > m_maxEMGain )
    {
        emg = m_maxEMGain;
        log<text_log>( "EM gain limited to maxEMGain = " + std::to_string( emg ), logPrio::LOG_WARNING );
    }

    recordCamera( true );
    if( setorcaParameterOnline( m_modelHandle, orcaParameter_AdcEMGain, emg ) < 0 )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;
        log<software_error>( { __FILE__, __LINE__, "Error setting EM gain" } );
        return -1;
    }

    int32 AdcEMGain;
    if( getorcaParameter( AdcEMGain, orcaParameter_AdcEMGain ) < 0 )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;
        return log<software_error, -1>( { __FILE__, __LINE__, "could not get AdcEMGain" } );
    }
    m_emGain   = AdcEMGain;
    m_adcSpeed = adcSpeed;
    recordCamera( true );
    return 0;
}

void orcaCtrl::updateFxnGenSync()
{

    std::cerr << "Setting fxngen frequency to " << std::to_string( m_fps ) << " Hz" << std::endl;
    m_indiP_fxngensync_freq["target"] = m_fps;
    sendNewProperty( m_indiP_fxngensync_freq );

    // make sure fxngen is on!
    m_indiP_fxngensync_output["value"] = "On";
    sendNewProperty( m_indiP_fxngensync_output );
}

inline int orcaCtrl::setExpTime()
{
    long  intexptime = m_expTimeSet * 1000 * 10000 + 0.5;
    double exptime    = ( (double)intexptime ) / 10000;
    capExpTime( exptime );

    int rv;

    recordCamera( true );

    if( state() == stateCodes::OPERATING )
    {
        rv = setorcaParameterOnline( m_modelHandle, orcaParameter_ExposureTime, exptime );
    }
    else
    {
        rv = setorcaParameter( m_modelHandle, orcaParameter_ExposureTime, exptime );
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

    if( getorcaParameter( m_FrameRateCalculation, orcaParameter_FrameRateCalculation ) < 0 )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;
        log<software_error>( { __FILE__, __LINE__, "could not get FrameRateCalculation" } );
    }
    m_fps = m_FrameRateCalculation;

    if( m_synchro && !m_otherCamName.empty() )
    {
        std::cerr << "Setting " << m_otherCamName << " exptime to " << std::to_string( m_expTime ) << std::endl;
        m_indiP_otherCamExptime["target"] = m_expTime;
        sendNewProperty( m_indiP_otherCamExptime );

        updateFxnGenSync();
    }

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

inline bool orcaCtrl::checkFocus()
{
    return checkFocusSwitchState();
}

inline int orcaCtrl::gotoFocus()
{
    return sendGotoFocusCommand();
}

inline int orcaCtrl::setShutter( int sh )
{
    return dssShutter<orcaCtrl>::setShutterState( sh );
}

inline int orcaCtrl::configureAcquisition()
{

    int32 readoutStride;
    int32 framesPerReadout;
    int32 frameStride;
    // int32 frameSize;
    int32 pixelBitDepth;

    m_camera_timestamp = 0; // reset tracked timestamp

    std::unique_lock<std::mutex> lock( m_indiMutex );

    // Time stamp handling
    if( orca_SetParameterIntegerValue( m_modelHandle, orcaParameter_TimeStamps, m_timeStampMask ) < 0 )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;
        log<software_error>( { __FILE__, __LINE__, "Could not set time stamp mask" } );
    }
    if( orca_GetParameterLargeIntegerValue( m_modelHandle, orcaParameter_TimeStampResolution, &m_tsRes ) < 0 )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;
        log<software_error>( { __FILE__, __LINE__, "Could not get timestamp resolution" } );
    }

    //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
    //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
    // Check Frame Transfer
    //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
    //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

    int32 cmode;
    if( getorcaParameter( cmode, orcaParameter_ReadoutControlMode ) < 0 )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;
        log<software_error>( { __FILE__, __LINE__, "could not get Readout Control Mode" } );
        return -1;
    }

    if( cmode != orcaReadoutControlMode_FrameTransfer )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;
        log<software_error>( { __FILE__, __LINE__, "Readout Control Mode not configured for frame transfer" } );
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

        if( setorcaParameter( m_modelHandle, orcaParameter_DisableCoolingFan, disableCoolingFan ) < 0 )
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

    if( setorcaParameter( orcaParameter_SensorTemperatureSetPoint, m_ccdTempSetpt ) < 0 )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;
        log<software_error>( { __FILE__, __LINE__, "Error setting temperature setpoint" } );
        state( stateCodes::ERROR );
        return -1;
    }

    // log<text_log>( "Set temperature set point: " + std::to_string(m_ccdTempSetpt) + " C");

    //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
    //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
    // ADC Speed and Quality
    //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
    //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

    int32 adcQual;
    double adcSpeed;

    if( readoutParams( adcQual, adcSpeed, m_readoutSpeedNameSet ) < 0 )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;
        log<software_error>( { __FILE__, __LINE__, "Invalid readout speed: " + m_readoutSpeedNameSet } );
        state( stateCodes::ERROR );
        return -1;
    }

    if( setorcaParameter( m_modelHandle, orcaParameter_AdcSpeed, adcSpeed, false ) <
        0 ) // don't commit b/c it will error if quality mismatched
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;
        log<software_error>( { __FILE__, __LINE__, "Error setting ADC Speed" } );
        // state(stateCodes::ERROR);
        // return -1;
    }

    if( setorcaParameter( m_modelHandle, orcaParameter_AdcQuality, adcQual ) < 0 )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;
        log<software_error>( { __FILE__, __LINE__, "Error setting ADC Quality" } );
        state( stateCodes::ERROR );
        return -1;
    }
    m_adcSpeed         = adcSpeed;
    m_readoutSpeedName = m_readoutSpeedNameSet;
    log<text_log>( "Readout speed set to: " + m_readoutSpeedNameSet );

    if( adcQual == orcaAdcQuality_LowNoise )
    {
        m_emGain    = 1.0;
        m_emGainSet = 1.0;
    }

    //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
    //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
    // Vertical Shift Rate
    //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
    //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

    double vss;
    if( vshiftParams( vss, m_vShiftSpeedNameSet ) < 0 )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;
        log<software_error>( { __FILE__, __LINE__, "Invalid vertical shift speed: " + m_vShiftSpeedNameSet } );
        state( stateCodes::ERROR );
        return -1;
    }

    if( setorcaParameter( m_modelHandle, orcaParameter_VerticalShiftRate, vss ) < 0 )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;
        log<software_error>( { __FILE__, __LINE__, "Error setting Vertical Shift Rate" } );
        state( stateCodes::ERROR );
        return -1;
    }

    m_vShiftSpeedName = m_vShiftSpeedNameSet;
    m_vshiftSpeed     = vss;
    log<text_log>( "Vertical Shift Rate set to: " + m_vShiftSpeedName );

    //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
    //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
    // Dimensions
    //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
    //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

    orcaRois nextrois;
    orcaRoi  nextroi;

    nextrois.roi_array = &nextroi;
    nextrois.roi_count = 1;

    int roi_err = false;
    if( m_defaultFlip == fgFlipLR || m_defaultFlip == fgFlipUDLR )
    {
        nextroi.x = ( ( 1023 - m_nextROI.x ) - 0.5 * ( (float)m_nextROI.w - 1.0 ) );
    }
    else
    {
        nextroi.x = ( m_nextROI.x - 0.5 * ( (float)m_nextROI.w - 1.0 ) );
    }

    if( nextroi.x < 0 )
    {
        log<software_error>( { __FILE__, __LINE__, "can't set ROI to x center < 0" } );
        roi_err = true;
    }

    if( nextroi.x > 1023 )
    {
        log<software_error>( { __FILE__, __LINE__, "can't set ROI to x center > 1023" } );
        roi_err = true;
    }

    if( m_defaultFlip == fgFlipUD || m_defaultFlip == fgFlipUDLR )
    {
        nextroi.y = ( ( 1023 - m_nextROI.y ) - 0.5 * ( (float)m_nextROI.h - 1.0 ) );
    }
    else
    {
        nextroi.y = ( m_nextROI.y - 0.5 * ( (float)m_nextROI.h - 1.0 ) );
    }

    if( nextroi.y < 0 )
    {
        log<software_error>( { __FILE__, __LINE__, "can't set ROI to y center < 0" } );
        roi_err = true;
    }

    if( nextroi.y > 1023 )
    {
        log<software_error>( { __FILE__, __LINE__, "can't set ROI to y center > 1023" } );
        roi_err = true;
    }

    nextroi.width = m_nextROI.w;

    if( nextroi.width < 0 )
    {
        log<software_error>( { __FILE__, __LINE__, "can't set ROI to width to be < 0" } );
        roi_err = true;
    }

    if( nextroi.x + nextroi.width > 1024 )
    {
        log<software_error>( { __FILE__, __LINE__, "can't set ROI to width such that edge is > 1023" } );
        roi_err = true;
    }

    nextroi.height = m_nextROI.h;

    if( nextroi.y + nextroi.height > 1024 )
    {
        log<software_error>( { __FILE__, __LINE__, "can't set ROI to height such that edge is > 1023" } );
        roi_err = true;
    }

    if( nextroi.height < 0 )
    {
        log<software_error>( { __FILE__, __LINE__, "can't set ROI to height to be < 0" } );
        roi_err = true;
    }

    nextroi.x_binning = m_nextROI.bin_x;

    if( nextroi.x_binning < 0 )
    {
        log<software_error>( { __FILE__, __LINE__, "can't set ROI x binning < 0" } );
        roi_err = true;
    }

    nextroi.y_binning = m_nextROI.bin_y;

    if( nextroi.y_binning < 0 )
    {
        log<software_error>( { __FILE__, __LINE__, "can't set ROI y binning < 0" } );
        roi_err = true;
    }

    orcaError error;

    if( !roi_err )
    {
        error = orca_SetParameterRoisValue( m_cameraHandle, orcaParameter_Rois, &nextrois );
        if( error != orcaError_None )
        {
            if( powerState() != 1 || powerStateTarget() != 1 )
                return -1;
            std::cerr << orcaEnum2String( orcaEnumeratedType_Error, error ) << "\n";
            log<software_error>(
                { __FILE__, __LINE__, 0, error, orcaEnum2String( orcaEnumeratedType_Error, error ) } );
            state( stateCodes::ERROR );
            return -1;
        }
    }

    if( getorcaParameter( readoutStride, orcaParameter_ReadoutStride ) < 0 )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;
        log<software_error>( { __FILE__, __LINE__, "Error getting readout stride" } );
        state( stateCodes::ERROR );
        return -1;
    }

    if( getorcaParameter( frameStride, orcaParameter_FrameStride ) < 0 )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;
        log<software_error>( { __FILE__, __LINE__, "Error getting frame stride" } );
        state( stateCodes::ERROR );

        return -1;
    }

    if( getorcaParameter( framesPerReadout, orcaParameter_FramesPerReadout ) < 0 )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;
        log<software_error>( { __FILE__, __LINE__, "Error getting frames per readout" } );
        state( stateCodes::ERROR );
        return -1;
    }

    if( getorcaParameter( m_frameSize, orcaParameter_FrameSize ) < 0 )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;
        log<software_error>( { __FILE__, __LINE__, "Error getting frame size" } );
        state( stateCodes::ERROR );
        return -1;
    }

    if( getorcaParameter( pixelBitDepth, orcaParameter_PixelBitDepth ) < 0 )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;
        log<software_error>( { __FILE__, __LINE__, "Error getting pixel bit depth" } );
        state( stateCodes::ERROR );
        return -1;
    }
    m_depth = pixelBitDepth;

    const orcaRois *rois;
    error = orca_GetParameterRoisValue( m_cameraHandle, orcaParameter_Rois, &rois );
    if( error != orcaError_None )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;
        log<software_error>( { __FILE__, __LINE__, 0, error, orcaEnum2String( orcaEnumeratedType_Error, error ) } );
        state( stateCodes::ERROR );
        return -1;
    }
    m_xbinning         = rois->roi_array[0].x_binning;
    m_currentROI.bin_x = m_xbinning;
    m_ybinning         = rois->roi_array[0].y_binning;
    m_currentROI.bin_y = m_ybinning;

    std::cerr << rois->roi_array[0].x << "\n";
    std::cerr << ( rois->roi_array[0].x - 1 ) << "\n";
    std::cerr << rois->roi_array[0].width << "\n";
    std::cerr << 0.5 * ( (float)( rois->roi_array[0].width - 1.0 ) ) << "\n";

    if( m_defaultFlip == fgFlipLR || m_defaultFlip == fgFlipUDLR )
    {
        m_currentROI.x = ( 1023.0 - rois->roi_array[0].x ) - 0.5 * ( (float)( rois->roi_array[0].width - 1.0 ) );
        // nextroi.x = ((1023-m_nextROI.x) - 0.5*( (float) m_nextROI.w - 1.0));
    }
    else
    {
        m_currentROI.x = ( rois->roi_array[0].x ) + 0.5 * ( (float)( rois->roi_array[0].width - 1.0 ) );
    }

    if( m_defaultFlip == fgFlipUD || m_defaultFlip == fgFlipUDLR )
    {
        m_currentROI.y = ( 1023.0 - rois->roi_array[0].y ) - 0.5 * ( (float)( rois->roi_array[0].height - 1.0 ) );
        // nextroi.y = ((1023 - m_nextROI.y) - 0.5*( (float) m_nextROI.h - 1.0));
    }
    else
    {
        m_currentROI.y = ( rois->roi_array[0].y ) + 0.5 * ( (float)( rois->roi_array[0].height - 1.0 ) );
    }

    m_currentROI.w = rois->roi_array[0].width;
    m_currentROI.h = rois->roi_array[0].height;

    m_width  = rois->roi_array[0].width / rois->roi_array[0].x_binning;
    m_height = rois->roi_array[0].height / rois->roi_array[0].y_binning;
    orca_DestroyRois( rois );

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

    if( getorcaParameter( m_ReadOutTimeCalculation, orcaParameter_ReadoutTimeCalculation ) < 0 )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;
        return log<software_error, -1>( { __FILE__, __LINE__, "could not get ReadOutTimeCalculation" } );
    }
    std::cerr << "Readout time is: " << m_ReadOutTimeCalculation << "\n";
    updateIfChanged(
        m_indiP_readouttime, "value", m_ReadOutTimeCalculation / 1000.0, INDI_OK ); // convert from msec to sec

    const orcaRangeConstraint *constraint_array;
    int32                       constraint_count;
    orcaAdvanced_GetParameterRangeConstraints(
        m_modelHandle, orcaParameter_ExposureTime, &constraint_array, &constraint_count );

    if( constraint_count != 1 )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;
        log<text_log>( "Constraint count is not 1: " + std::to_string( constraint_count ) + " constraints",
                       logPrio::LOG_ERROR );
    }
    else
    {
        m_minExpTime  = constraint_array[0].minimum;
        m_maxExpTime  = constraint_array[0].maximum;
        m_stepExpTime = constraint_array[0].increment;

        m_indiP_exptime["current"].setMin( m_minExpTime );
        m_indiP_exptime["current"].setMax( m_maxExpTime );
        m_indiP_exptime["current"].setStep( m_stepExpTime );

        m_indiP_exptime["target"].setMin( m_minExpTime );
        m_indiP_exptime["target"].setMax( m_maxExpTime );
        m_indiP_exptime["target"].setStep( m_stepExpTime );
    }

    if( m_expTimeSet > 0 )
    {
        long  intexptime = m_expTimeSet * 1000 * 10000 + 0.5;
        double exptime    = ( (double)intexptime ) / 10000;
        capExpTime( exptime );
        std::cerr << "Setting exposure time to " << m_expTimeSet << "\n";
        int rv = setorcaParameter( m_modelHandle, orcaParameter_ExposureTime, exptime );

        if( rv < 0 )
        {
            if( powerState() != 1 || powerStateTarget() != 1 )
                return -1;
            return log<software_error, -1>( { __FILE__, __LINE__, "Error setting exposure time" } );
        }
    }

    double exptime;
    if( getorcaParameter( exptime, orcaParameter_ExposureTime ) < 0 )
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

    if( getorcaParameter( m_FrameRateCalculation, orcaParameter_FrameRateCalculation ) < 0 )
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

    int32 AdcQuality;
    if( getorcaParameter( AdcQuality, orcaParameter_AdcQuality ) < 0 )
    {
        std::cerr << "could not get AdcQuality\n";
    }
    std::string adcqStr = orcaEnum2String( orcaEnumeratedType_AdcQuality, AdcQuality );
    std::cerr << "AdcQuality is: " << adcqStr << "\n";

    double verticalShiftRate;
    if( getorcaParameter( verticalShiftRate, orcaParameter_VerticalShiftRate ) < 0 )
    {
        std::cerr << "could not get VerticalShiftRate\n";
    }
    std::cerr << "VerticalShiftRate is: " << verticalShiftRate << "\n";

    double AdcSpeed;
    if( getorcaParameter( AdcSpeed, orcaParameter_AdcSpeed ) < 0 )
    {
        std::cerr << "could not get AdcSpeed\n";
    }
    std::cerr << "AdcSpeed is: " << AdcSpeed << "\n";

    std::cerr << "************************************************************\n";

    int32 AdcAnalogGain;
    if( getorcaParameter( AdcAnalogGain, orcaParameter_AdcAnalogGain ) < 0 )
    {
        std::cerr << "could not get AdcAnalogGain\n";
    }
    std::string adcgStr = orcaEnum2String( orcaEnumeratedType_AdcAnalogGain, AdcAnalogGain );
    std::cerr << "AdcAnalogGain is: " << adcgStr << "\n";

    if( m_readoutSpeedName == "ccd_00_1MHz" || m_readoutSpeedName == "ccd_01MHz" )
    {
        m_emGain = 1;
    }
    else
    {
        int32 AdcEMGain;
        if( getorcaParameter( AdcEMGain, orcaParameter_AdcEMGain ) < 0 )
        {
            std::cerr << "could not get AdcEMGain\n";
        }
        m_emGain = AdcEMGain;
    }

    /*
       std::cerr << "Onlineable:\n";
       pibln onlineable;
       orca_CanSetParameterOnline(m_modelHandle, orcaParameter_ReadoutControlMode,&onlineable);
       std::cerr << "ReadoutControlMode: " << onlineable << "\n"; //0

       orca_CanSetParameterOnline(m_modelHandle, orcaParameter_AdcQuality,&onlineable);
       std::cerr << "AdcQuality: " << onlineable << "\n"; //0

       orca_CanSetParameterOnline(m_modelHandle, orcaParameter_AdcAnalogGain,&onlineable);
       std::cerr << "AdcAnalogGain: " << onlineable << "\n"; //1

       orca_CanSetParameterOnline(m_modelHandle, orcaParameter_DisableCoolingFan,&onlineable);
       std::cerr << "DisableCoolingFan: " << onlineable << "\n";//0

       orca_CanSetParameterOnline(m_modelHandle, orcaParameter_SensorTemperatureSetPoint,&onlineable);
       std::cerr << "SensorTemperatureSetPoint: " << onlineable << "\n"; //0

       orca_CanSetParameterOnline(m_modelHandle, orcaParameter_AdcEMGain,&onlineable);
       std::cerr << "AdcEMGain: " << onlineable << "\n"; //1

       orca_CanSetParameterOnline(m_modelHandle, orcaParameter_FrameRateCalculation,&onlineable);
       std::cerr << "FrameRateCalculation: " << onlineable << "\n"; //0

       std::cerr << "************************************************************\n";
    */

    // If not previously allocated, allocate a nice big buffer to play with
    pi64s newbuffsz = framesPerReadout * readoutStride * 10; // Save room for 10 frames
    if( newbuffsz > m_acqBuff.memory_size )
    {
        if( m_acqBuff.memory )
        {
            std::cerr << "Clearing\n";
            free( m_acqBuff.memory );
            m_acqBuff.memory = NULL;
            orcaAdvanced_SetAcquisitionBuffer( m_cameraHandle, NULL );
        }

        m_acqBuff.memory_size = newbuffsz;
        std::cerr << "m_acqBuff.memory_size: " << m_acqBuff.memory_size << "\n";
        m_acqBuff.memory = malloc( m_acqBuff.memory_size );

        error = orcaAdvanced_SetAcquisitionBuffer( m_cameraHandle, &m_acqBuff );
        if( error != orcaError_None )
        {
            log<software_error>(
                { __FILE__, __LINE__, 0, error, orcaEnum2String( orcaEnumeratedType_Error, error ) } );
            state( stateCodes::ERROR );

            std::cerr << "-->" << orcaEnum2String( orcaEnumeratedType_Error, error ) << "\n";
        }
    }

    // Hardware trigger
    if( m_synchroSet )
    {
        updateFxnGenSync();

        std::cerr << "Turning synchro on" << std::endl;
        setorcaParameter( m_cameraHandle, orcaParameter_TriggerDetermination, orcaTriggerDetermination_RisingEdge );
        setorcaParameter( m_cameraHandle, orcaParameter_TriggerResponse, orcaTriggerResponse_ReadoutPerTrigger );
        m_synchro = true;
        updateSwitchIfChanged( m_indiP_synchro, "toggle", pcf::IndiElement::On, INDI_IDLE );
    }
    else
    {
        std::cerr << "Turning synchro off" << std::endl;
        setorcaParameter( m_cameraHandle, orcaParameter_TriggerResponse, orcaTriggerResponse_NoResponse );
        m_synchro = false;
        updateSwitchIfChanged( m_indiP_synchro, "toggle", pcf::IndiElement::Off, INDI_IDLE );
    }

    // Start continuous acquisition
    if( setorcaParameter( orcaParameter_ReadoutCount, (pi64s)0 ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__, "Error setting readouts=0" } );
        state( stateCodes::ERROR );
        return -1;
    }

    recordCamera();

    error = orca_StartAcquisition( m_cameraHandle );
    if( error != orcaError_None )
    {
        log<software_error>( { __FILE__, __LINE__, 0, error, orcaEnum2String( orcaEnumeratedType_Error, error ) } );
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

    orcaAcquisitionStatus status;

    orcaAvailableData available;

    orcaError error;
    error = orca_WaitForAcquisitionUpdate( m_cameraHandle, camTimeOut, &available, &status );

    if( error == orcaError_TimeOutOccurred )
    {
        return 1; // This sends it back to framegrabber to check for reconfig, etc.
    }

    clock_gettime( CLOCK_REALTIME, &m_currImageTimestamp );

    if( error != orcaError_None )
    {
        log<software_error>( { __FILE__, __LINE__, 0, error, orcaEnum2String( orcaEnumeratedType_Error, error ) } );
        state( stateCodes::ERROR );

        return -1;
    }

    m_available.initial_readout = available.initial_readout;
    m_available.readout_count   = available.readout_count;

    if( m_available.initial_readout == 0 )
    {
        return 1;
    }

    // std::cerr << "readout: " << m_available.initial_readout << " " << m_available.readout_count << "\n";

    // camera time stamp
    pibyte *frame = NULL;
    pi64s   metadataOffset;

    frame          = static_cast<pibyte *>( m_available.initial_readout );
    metadataOffset = (pi64s)frame + m_frameSize;

    pi64s *tmpPtr = reinterpret_cast<pi64s *>( metadataOffset );

    double cam_ts   = (double)*tmpPtr / (double)m_tsRes;
    double delta_ts = cam_ts - m_camera_timestamp;

    // check for a frame skip
    if( delta_ts > 1.5 / m_FrameRateCalculation )
    {
        std::cerr << "Skipped frame(s)! (Expected a " << 1000. / m_FrameRateCalculation << " ms gap but got "
                  << 1000 * delta_ts << " ms)\n";
    }
    // print

    m_camera_timestamp = cam_ts; // update to latest

    // fprintf(m_outfile, "%d %-15.8f\n", m_imageStream->md->cnt0+1, (double)*tmpPtr/(double)m_tsRes);

    return 0;
}

inline int orcaCtrl::loadImageIntoStream( void *dest )
{
    if( frameGrabber<orcaCtrl>::loadImageIntoStreamCopy(
            dest, m_available.initial_readout, m_width, m_height, m_typeSize ) == nullptr )
        return -1;

    return 0;
}

inline int orcaCtrl::reconfig()
{
    ///\todo clean this up.  Just need to wait on acquisition update the first time probably.

    orcaError error = orca_StopAcquisition( m_cameraHandle );
    if( error != orcaError_None )
    {
        log<software_error>( { __FILE__, __LINE__, 0, error, orcaEnum2String( orcaEnumeratedType_Error, error ) } );
        state( stateCodes::ERROR );

        return -1;
    }

    pibln running = true;

    error = orca_IsAcquisitionRunning( m_cameraHandle, &running );

    while( running )
    {
        if( MagAOXAppT::m_powerState == 0 )
            return 0;
        sleep( 1 );

        error = orca_StopAcquisition( m_cameraHandle );

        if( error != orcaError_None )
        {
            log<software_error>(
                { __FILE__, __LINE__, 0, error, orcaEnum2String( orcaEnumeratedType_Error, error ) } );
            state( stateCodes::ERROR );
            return -1;
        }

        int32 camTimeOut = 1000;

        orcaAcquisitionStatus status;

        orcaAvailableData available;

        error = orca_WaitForAcquisitionUpdate( m_cameraHandle, camTimeOut, &available, &status );
        if( error != orcaError_None )
        {
            log<software_error>(
                { __FILE__, __LINE__, 0, error, orcaEnum2String( orcaEnumeratedType_Error, error ) } );
            state( stateCodes::ERROR );
            return -1;
        }

        //       if(! status.running )
        //       {
        //          std::cerr << "Not running \n";
        //
        //          std::cerr << "status.running: " << status.running << "\n";
        //          std::cerr << "status.errors: " << status.errors << "\n";
        //          std::cerr << "CameraFaulted: " << (int)(status.errors & orcaAcquisitionErrorsMask_CameraFaulted) <<
        //          "\n"; std::cerr << "CannectionLost: " << (int)(status.errors &
        //          orcaAcquisitionErrorsMask_ConnectionLost) << "\n"; std::cerr << "DataLost: " << (int)(status.errors
        //          & orcaAcquisitionErrorsMask_DataLost) << "\n"; std::cerr << "DataNotArriving: " <<
        //          (int)(status.errors & orcaAcquisitionErrorsMask_DataNotArriving) << "\n"; std::cerr << "None: " <<
        //          (int)(status.errors & orcaAcquisitionErrorsMask_None) << "\n"; std::cerr << "ShutterOverheated: "
        //          << (int)(status.errors & orcaAcquisitionErrorsMask_ShutterOverheated) << "\n"; std::cerr <<
        //          "status.readout_rate: " << status.readout_rate << "\n";
        //       }

        error = orca_IsAcquisitionRunning( m_cameraHandle, &running );
        if( error != orcaError_None )
        {
            log<software_error>(
                { __FILE__, __LINE__, 0, error, orcaEnum2String( orcaEnumeratedType_Error, error ) } );
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

INDI_NEWCALLBACK_DEFN( orcaCtrl, m_indiP_receiveSynchro )( const pcf::IndiProperty &ipRecv )
{

    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_receiveSynchro, ipRecv );

    if( ipRecv.getName() != m_indiP_receiveSynchro.getName() )
    {
        log<software_error>( { __FILE__, __LINE__, "wrong INDI property received" } );

        return -1;
    }

    if( !ipRecv.find( "toggle" ) )
    {
        return 0;
    }

    if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On )
    {
        updateSwitchIfChanged( m_indiP_receiveSynchro, "toggle", pcf::IndiElement::On, INDI_IDLE );

        m_synchroSet = true;
    }
    else
    {
        updateSwitchIfChanged( m_indiP_receiveSynchro, "toggle", pcf::IndiElement::Off, INDI_IDLE );

        m_synchroSet = false;
    }

    m_reconfig = true;

    return 0;
}

INDI_NEWCALLBACK_DEFN( orcaCtrl, m_indiP_receiveExptime )( const pcf::IndiProperty &ipRecv )
{

    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_receiveExptime, ipRecv );

    if( ipRecv.getName() != m_indiP_receiveExptime.getName() )
    {
        log<software_error>( { __FILE__, __LINE__, "wrong INDI property received" } );

        return -1;
    }

    if( !ipRecv.find( "target" ) )
    {
        return 0;
    }

    m_expTimeSet = ipRecv["target"].get<double>();

    updatesIfChanged<double>( m_indiP_receiveExptime, { "current" }, { m_expTimeSet } );

    // we don't need to strictly reconfig because setExpTime works online, but this
    // eludes an infinite loop of triggering the 'otherCam' in setExptime
    m_reconfig = 1;

    return 0;
}

} // namespace app
} // namespace MagAOX
#endif
