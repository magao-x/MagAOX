/** \file cred2Ctrl.hpp
 * \brief The MagAO-X C-RED 2 camera controller.
 *
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup cred2Ctrl_files
 */

#ifndef cred2Ctrl_hpp
#define cred2Ctrl_hpp

#include <algorithm>
#include <cmath>
#include <dlfcn.h>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>

#include "../../libMagAOX/libMagAOX.hpp" // Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

#include "cred2Utils.hpp"

namespace MagAOX
{
namespace app
{

/** \defgroup cred2Ctrl C-RED 2 Camera
 * \brief Control of the First Light Imaging C-RED 2 camera.
 *
 * <a href="../handbook/operating/software/apps/cred2Ctrl.html">Application Documentation</a>
 *
 * \ingroup apps
 */

/** \defgroup cred2Ctrl_files C-RED 2 Camera Files
 * \ingroup cred2Ctrl
 */

/// MagAO-X application to control the C-RED 2 camera.
/**
 * \ingroup cred2Ctrl
 */
class cred2Ctrl : public MagAOXApp<>,
                  public dev::stdCamera<cred2Ctrl>,
                  public dev::edtCamera<cred2Ctrl>,
                  public dev::frameGrabber<cred2Ctrl>,
                  public dev::telemeter<cred2Ctrl>
{
    friend class dev::stdCamera<cred2Ctrl>;
    friend class dev::edtCamera<cred2Ctrl>;
    friend class dev::frameGrabber<cred2Ctrl>;
    friend class dev::telemeter<cred2Ctrl>;

    typedef MagAOXApp<>                  MagAOXAppT;
    typedef dev::stdCamera<cred2Ctrl>    stdCameraT;
    typedef dev::frameGrabber<cred2Ctrl> frameGrabberT;
    typedef dev::telemeter<cred2Ctrl>    telemeterT;

  public:
    /** \name app::dev Configurations
     * @{
     */
    static constexpr bool c_stdCamera_tempControl  = true;  ///< Expose temperature setpoint control.
    static constexpr bool c_stdCamera_temp         = true;  ///< Expose detector temperature status.
    static constexpr bool c_stdCamera_readoutSpeed = false; ///< Do not expose readout-speed controls.
    static constexpr bool c_stdCamera_vShiftSpeed  = false; ///< Do not expose vertical-shift controls.
    static constexpr bool c_stdCamera_emGain       = false; ///< Do not expose EM-gain controls.
    static constexpr bool c_stdCamera_exptimeCtrl  = false; ///< Do not expose exposure-time controls.
    static constexpr bool c_stdCamera_fpsCtrl      = true;  ///< Expose FPS controls.
    static constexpr bool c_stdCamera_fps          = true;  ///< Expose FPS status.
    static constexpr bool c_stdCamera_fanSpeed     = true;  ///< Expose fan-speed controls.
    static constexpr bool c_stdCamera_analogGain   = true;  ///< Expose discrete analog-gain controls.
    static constexpr bool c_stdCamera_led          = true;  ///< Expose status LED controls.
    static constexpr bool c_stdCamera_synchro      = false; ///< Do not expose synchro controls in the first pass.
    static constexpr bool c_stdCamera_usesModes    = false; ///< Use one synthetic runtime mode rather than INDI modes.
    static constexpr bool c_stdCamera_usesROI      = true;  ///< Expose ROI controls.
    static constexpr bool c_stdCamera_cropMode     = false; ///< Do not expose crop-mode controls separately.
    static constexpr bool c_stdCamera_hasShutter   = false; ///< Do not expose shutter controls.
    static constexpr bool c_stdCamera_usesStateString    = false; ///< Do not expose a dark-management state string.
    static constexpr bool c_edtCamera_relativeConfigPath = false; ///< Use an absolute temporary EDT config path.
    static constexpr bool c_frameGrabber_flippable = true; ///< Expose image flip controls through the framegrabber.

    ///@}

  protected:
    /** \name Configurable Parameters - Data
     * @{
     */
    std::string m_configFile; ///< Absolute path to the temporary EDT configuration file.

    int m_serialBaud{ 115200 }; ///< Camera Link serial baud rate used for C-RED 2 CLI access.
    ///@}

    /** \name C-RED 2 State - Data
     * @{
     */
    cred2Temps m_temps; ///< Cached camera temperature values used for INDI and telemetry updates.

    bool m_cameraCropEnabled{ false }; ///< Tracks whether this controller has enabled camera-side cropping.
    int  m_roiSettleCounter{ 0 };      ///< Number of main-loop cycles to skip serial status polling after ROI changes.

    std::recursive_mutex m_cameraMutex; ///< Protects serial command traffic and EDT reconfiguration.
    ///@}

    /** \name INDI - Data
     * @{
     */
    pcf::IndiProperty m_indiP_temps;     ///< Property reporting the detailed C-RED 2 temperature channels.
    pcf::IndiProperty m_indiP_fpsLimits; ///< Property reporting the current C-RED 2 minimum and maximum FPS.

    ///@}

  public:
    /// Default c'tor.
    cred2Ctrl();

    /// D'tor, declared and defined for noexcept.
    ~cred2Ctrl() noexcept;

    /// Setup the configuration system.
    virtual void setupConfig();

    /// Load the configuration system results.
    virtual void loadConfig();

    /// Implementation of loadConfig logic with standard helper-macro error handling.
    int loadConfigImpl( mx::app::appConfigurator &config /**< [in] application configurator with loaded values */ );

    /// Startup function.
    virtual int appStartup();

    /// Main FSM logic.
    virtual int appLogic();

    /// Actions required when the camera power turns off.
    virtual int onPowerOff();

    /// Actions required while the camera remains powered off.
    virtual int whilePowerOff();

    /// Shutdown function.
    virtual int appShutdown();

    /// Query and update the camera temperature channels.
    int getTemps();

    /// Query and update the current camera frame rate.
    int getFPS();

    /// Query and update the current camera FPS limits.
    int updateFPSLimits();

    /// Query and update the current fan-control state.
    int getFanSpeed();

    /// Query and update the current analog-gain state.
    int getAnalogGain();

    /// Query and update the current LED state.
    int getLEDState();

    /// Query the camera for its current ROI and synchronize local state.
    int syncROIFromCamera();

    /** \name stdCamera Interface
     * @{
     */

    /// Set defaults for a power-on state.
    int powerOnDefaults();

    /// Implement the C-RED 2 temperature-controller toggle semantics.
    int setTempControl();

    /// Send the current target detector temperature setpoint to the camera.
    int setTempSetPt();

    /// Send the requested frame rate to the camera.
    int setFPS();

    /// Send the requested fan-control mode to the camera.
    int setFanSpeed();

    /// Send the requested analog-gain mode to the camera.
    int setAnalogGain();

    /// Send the requested LED state to the camera.
    int setLED();

    /// Required by `stdCamera`, but unused for C-RED 2.
    int setExpTime();

    /// Validate and normalize the requested ROI.
    int checkNextROI();

    /// Request that the next valid ROI be applied through reconfiguration.
    int setNextROI();

    ///@}

    /** \name Framegrabber Interface
     * @{
     */

    /// Write the temporary EDT configuration file for the pending ROI.
    int writeConfig();

    /// Configure camera-side ROI settings before acquisition starts.
    int configureAcquisition();

    /// Return the currently measured frame rate.
    float fps();

    /// Start frame acquisition on the EDT board.
    int startAcquisition();

    /// Wait for and validate the next acquired image.
    int acquireAndCheckValid();

    /// Copy the current EDT image into the output stream.
    int loadImageIntoStream( void *dest /**< [in] destination frame buffer */ );

    /// Reconfigure the EDT board for the pending ROI.
    int reconfig();

    ///@}

    /** \name Telemeter Interface
     * @{
     */

    /// Check the telemetry record timers.
    int checkRecordTimes();

    /// Record the detailed C-RED 2 temperature telemetry.
    int recordTelem( const cred2_temps * /**< [in] type-dispatch tag */ );

    /// Record standard camera telemetry.
    int recordTelem( const telem_stdcam * /**< [in] type-dispatch tag */ );

    /// Record framegrabber timing telemetry.
    int recordTelem( const telem_fgtimings * /**< [in] type-dispatch tag */ );

    /// Record the detailed C-RED 2 temperature telemetry when values change.
    int recordTemps( bool force = false /**< [in] force a telemetry record even if the cached values match */ );

    ///@}

  protected:
    /// Apply and verify the configured Camera Link serial baud rate.
    int setSerialBaud();

    /// Send a command over Camera Link serial and clean the response.
    int sendCommand( std::string       &response,         ///< [out] cleaned command response
                     const std::string &command,          /**< [in] CLI command to send */
                     bool               logFailure = true /**< [in] log transport failures when true */
    );

    /// Send a command that should return a success acknowledgement.
    int issueCommand( const std::string &command,                /**< [in] CLI command to send */
                      bool               allowNoResponse = false /**< [in] treat a missing response as acceptable */
    );
};

namespace
{

/// Normalize a C-RED 2 text response for tolerant string parsing.
inline std::string cred2LowerResponse( const std::string &response /**< [in] raw or cleaned CLI response */ )
{
    std::string clean = cred2CleanResponse( response );
    std::transform( clean.begin(),
                    clean.end(),
                    clean.begin(),
                    []( unsigned char c ) { return static_cast<char>( std::tolower( c ) ); } );

    return clean;
}

/// Convert a C-RED 2 fan percentage into the nearest exposed preset name.
inline std::string cred2FanPresetName( float fanPercent /**< [in] current or requested fan percentage */ )
{
    if( fanPercent <= 0.5f )
    {
        return "off";
    }

    if( fanPercent < 37.5f )
    {
        return "p25";
    }

    if( fanPercent < 62.5f )
    {
        return "p50";
    }

    if( fanPercent < 87.5f )
    {
        return "p75";
    }

    return "p100";
}

/// Convert an exposed fan preset name into the corresponding manual fan percentage.
inline int cred2FanPresetPercent( int               &fanPercent, ///< [out] mapped manual fan percentage
                                  const std::string &fanPreset   /**< [in] exposed fan preset name */
)
{
    if( fanPreset == "off" )
    {
        fanPercent = 0;
        return 0;
    }

    if( fanPreset == "p25" )
    {
        fanPercent = 25;
        return 0;
    }

    if( fanPreset == "p50" )
    {
        fanPercent = 50;
        return 0;
    }

    if( fanPreset == "p75" )
    {
        fanPercent = 75;
        return 0;
    }

    if( fanPreset == "p100" )
    {
        fanPercent = 100;
        return 0;
    }

    return -1;
}

/// Parse a C-RED 2 sensibility response into the exposed analog-gain preset name.
inline int cred2AnalogGainName( std::string       &gainName, ///< [out] exposed analog-gain preset name
                                const std::string &response  /**< [in] raw or cleaned sensibility response */
)
{
    std::string clean = cred2LowerResponse( response );

    if( clean.find( "medium" ) != std::string::npos || clean == "med" )
    {
        gainName = "med";
        return 0;
    }

    if( clean.find( "high" ) != std::string::npos )
    {
        gainName = "high";
        return 0;
    }

    if( clean.find( "low" ) != std::string::npos )
    {
        gainName = "low";
        return 0;
    }

    return -1;
}

/// Convert an exposed analog-gain preset name into the C-RED 2 command argument.
inline int cred2AnalogGainCommand( std::string       &commandGain, ///< [out] C-RED 2 sensibility command argument
                                   const std::string &gainName     /**< [in] exposed analog-gain preset name */
)
{
    if( gainName == "low" )
    {
        commandGain = "low";
        return 0;
    }

    if( gainName == "med" )
    {
        commandGain = "medium";
        return 0;
    }

    if( gainName == "high" )
    {
        commandGain = "high";
        return 0;
    }

    return -1;
}

} // namespace

inline cred2Ctrl::cred2Ctrl() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    m_powerMgtEnabled = true;

    m_startupTemp = 20;
    m_minTemp     = -40;
    m_maxTemp     = 20;
    m_stepTemp    = 1;

    m_stepFPS = 0.001;

    m_full_x     = 319.5;
    m_full_y     = 255.5;
    m_full_w     = 640;
    m_full_h     = 512;
    m_full_bin_x = 1;
    m_full_bin_y = 1;

    m_default_x     = m_full_x;
    m_default_y     = m_full_y;
    m_default_w     = m_full_w;
    m_default_h     = m_full_h;
    m_default_bin_x = m_full_bin_x;
    m_default_bin_y = m_full_bin_y;

    m_full_currbin_x = m_full_x;
    m_full_currbin_y = m_full_y;
    m_full_currbin_w = m_full_w;
    m_full_currbin_h = m_full_h;

    m_minROIx  = 0;
    m_maxROIx  = 639;
    m_stepROIx = 0.5;

    m_minROIy  = 0;
    m_maxROIy  = 511;
    m_stepROIy = 0.5;

    m_minROIWidth  = 32;
    m_maxROIWidth  = 640;
    m_stepROIWidth = 32;

    m_minROIHeight  = 4;
    m_maxROIHeight  = 512;
    m_stepROIHeight = 4;

    m_minROIBinning_x  = 1;
    m_maxROIBinning_x  = 1;
    m_stepROIBinning_x = 1;

    m_minROIBinning_y  = 1;
    m_maxROIBinning_y  = 1;
    m_stepROIBinning_y = 1;

    m_fanSpeedNames      = { "off", "p25", "p50", "p75", "p100", "auto" };
    m_fanSpeedNameLabels = { "Off", "25", "50", "75", "100", "Auto" };
    m_defaultFanSpeed    = "auto";
    m_fanSpeedNameSet    = m_defaultFanSpeed;

    m_analogGainNames      = { "low", "med", "high" };
    m_analogGainNameLabels = { "Low", "Med", "High" };
    m_analogGainNameSet    = "med";

    m_defaultLEDState = true;
    m_ledStateSet     = m_defaultLEDState;

    m_temps.setInvalid();
}

inline cred2Ctrl::~cred2Ctrl() noexcept
{
}

inline void cred2Ctrl::setupConfig()
{
    STDCAMERA_SETUP_CONFIG( config );

    dev::edtCamera<cred2Ctrl>::setupConfig( config );

    FRAMEGRABBER_SETUP_CONFIG( config );

    TELEMETER_SETUP_CONFIG( config );

    config.add( "camera.serialBaud",
                "",
                "camera.serialBaud",
                argType::Required,
                "camera",
                "serialBaud",
                false,
                "int",
                "The Camera Link serial baud rate for C-RED 2 CLI commands. Default is 115200." );
}

inline int cred2Ctrl::loadConfigImpl( mx::app::appConfigurator &config )
{
    STDCAMERA_LOAD_CONFIG( config );

    config( m_serialBaud, "camera.serialBaud" );

    m_configFile = "/tmp/cred2_" + configName() + ".cfg";

    m_cameraModes["runtime"] = dev::cameraConfig( { m_configFile,
                                                    "",
                                                    static_cast<unsigned>( m_nextROI.x ),
                                                    static_cast<unsigned>( m_nextROI.y ),
                                                    static_cast<unsigned>( m_nextROI.w ),
                                                    static_cast<unsigned>( m_nextROI.h ),
                                                    static_cast<unsigned>( m_nextROI.bin_x ),
                                                    static_cast<unsigned>( m_nextROI.bin_y ),
                                                    1,
                                                    1,
                                                    0 } );
    m_startupMode            = "runtime";

    if( writeConfig() < 0 )
    {
        return log<software_critical, -1>( { __FILE__, __LINE__, "could not write initial C-RED 2 EDT config" } );
    }

    dev::edtCamera<cred2Ctrl>::loadConfig( config );

    FRAMEGRABBER_LOAD_CONFIG( config );

    TELEMETER_LOAD_CONFIG( config );

    return 0;
}

inline void cred2Ctrl::loadConfig()
{
    if( loadConfigImpl( config ) < 0 )
    {
        log<software_critical>( { __FILE__, __LINE__, "error loading config" } );
        m_shutdown = true;
    }
}

inline int cred2Ctrl::appStartup()
{
    REG_INDI_NEWPROP_NOCB( m_indiP_temps, "temps", pcf::IndiProperty::Number );
    m_indiP_temps.add( pcf::IndiElement( "motherboard" ) );
    m_indiP_temps["motherboard"].set( 0 );
    m_indiP_temps.add( pcf::IndiElement( "frontend" ) );
    m_indiP_temps["frontend"].set( 0 );
    m_indiP_temps.add( pcf::IndiElement( "powerboard" ) );
    m_indiP_temps["powerboard"].set( 0 );
    m_indiP_temps.add( pcf::IndiElement( "snake" ) );
    m_indiP_temps["snake"].set( 0 );
    m_indiP_temps.add( pcf::IndiElement( "setpoint" ) );
    m_indiP_temps["setpoint"].set( 0 );
    m_indiP_temps.add( pcf::IndiElement( "peltier" ) );
    m_indiP_temps["peltier"].set( 0 );
    m_indiP_temps.add( pcf::IndiElement( "heatsink" ) );
    m_indiP_temps["heatsink"].set( 0 );

    createROIndiNumber( m_indiP_fpsLimits, "fps_limits" );
    m_indiP_fpsLimits.add( pcf::IndiElement( "min" ) );
    m_indiP_fpsLimits["min"].set( m_minFPS );
    m_indiP_fpsLimits["min"].setFormat( "%0.6f" );
    m_indiP_fpsLimits.add( pcf::IndiElement( "max" ) );
    m_indiP_fpsLimits["max"].set( m_maxFPS );
    m_indiP_fpsLimits["max"].setFormat( "%0.6f" );

    if( registerIndiPropertyReadOnly( m_indiP_fpsLimits ) < 0 )
    {
        return log<software_critical, -1>( { __FILE__, __LINE__ } );
    }

    STDCAMERA_APP_STARTUP;

    if( dev::edtCamera<cred2Ctrl>::appStartup() < 0 )
    {
        return log<software_critical, -1>( { __FILE__, __LINE__ } );
    }

    if( setSerialBaud() < 0 )
    {
        return log<software_critical, -1>( { __FILE__, __LINE__ } );
    }

    FRAMEGRABBER_APP_STARTUP;

    TELEMETER_APP_STARTUP;

    return 0;
}

inline int cred2Ctrl::appLogic()
{
    STDCAMERA_APP_LOGIC;

    if( dev::edtCamera<cred2Ctrl>::appLogic() < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__ } );
    }

    FRAMEGRABBER_APP_LOGIC;

    if( state() == stateCodes::POWERON )
    {
        return 0;
    }

    if( state() == stateCodes::NOTCONNECTED || state() == stateCodes::NODEVICE || state() == stateCodes::ERROR )
    {
        if( powerState() == 0 )
        {
            return 0;
        }

        std::string response;
        if( sendCommand( response, "fps raw" ) == 0 )
        {
            float fpsValue = 0;
            if( cred2ParseFloat( fpsValue, response ) == 0 )
            {
                state( stateCodes::CONNECTED );
            }
            else
            {
                state( stateCodes::NODEVICE );
                sleep( 1 );
                return 0;
            }
        }
        else
        {
            state( stateCodes::NODEVICE );
            sleep( 1 );
            return 0;
        }
    }

    if( state() == stateCodes::CONNECTED )
    {
        std::unique_lock<std::mutex> lock( m_indiMutex );

        if( syncROIFromCamera() < 0 )
        {
            if( powerState() != 1 || powerStateTarget() != 1 )
            {
                return 0;
            }

            state( stateCodes::ERROR );
            return 0;
        }

        if( updateFPSLimits() < 0 || getTemps() < 0 || getFPS() < 0 || getFanSpeed() < 0 || getAnalogGain() < 0 ||
            getLEDState() < 0 )
        {
            if( powerState() != 1 || powerStateTarget() != 1 )
            {
                return 0;
            }

            state( stateCodes::ERROR );
            return 0;
        }

        state( stateCodes::READY );

        m_fanSpeedNameSet = m_defaultFanSpeed;
        if( m_fanSpeedName != m_fanSpeedNameSet )
        {
            if( setFanSpeed() < 0 )
            {
                if( powerState() != 1 || powerStateTarget() != 1 )
                {
                    return 0;
                }

                return log<software_error, 0>( { __FILE__, __LINE__ } );
            }
        }

        m_ledStateSet = m_defaultLEDState;
        if( m_ledState != m_ledStateSet )
        {
            if( setLED() < 0 )
            {
                if( powerState() != 1 || powerStateTarget() != 1 )
                {
                    return 0;
                }

                return log<software_error, 0>( { __FILE__, __LINE__ } );
            }
        }

        if( m_ccdTempSetpt > -999 )
        {
            if( setTempSetPt() < 0 )
            {
                if( powerState() != 1 || powerStateTarget() != 1 )
                {
                    return 0;
                }

                return log<software_error, 0>( { __FILE__, __LINE__ } );
            }
        }
    }

    if( state() == stateCodes::READY || state() == stateCodes::OPERATING )
    {
        std::unique_lock<std::mutex> lock( m_indiMutex, std::try_to_lock );
        if( !lock.owns_lock() )
        {
            return 0;
        }

        if( m_roiSettleCounter > 0 )
        {
            --m_roiSettleCounter;
        }
        else
        {
            if( updateFPSLimits() < 0 )
            {
                if( powerState() != 1 || powerStateTarget() != 1 )
                {
                    return 0;
                }

                state( stateCodes::ERROR );
                return 0;
            }

            if( getTemps() < 0 )
            {
                if( powerState() != 1 || powerStateTarget() != 1 )
                {
                    return 0;
                }

                state( stateCodes::ERROR );
                return 0;
            }

            if( getFPS() < 0 )
            {
                if( powerState() != 1 || powerStateTarget() != 1 )
                {
                    return 0;
                }

                state( stateCodes::ERROR );
                return 0;
            }

            if( getFanSpeed() < 0 )
            {
                if( powerState() != 1 || powerStateTarget() != 1 )
                {
                    return 0;
                }

                state( stateCodes::ERROR );
                return 0;
            }

            if( getAnalogGain() < 0 )
            {
                if( powerState() != 1 || powerStateTarget() != 1 )
                {
                    return 0;
                }

                state( stateCodes::ERROR );
                return 0;
            }

            if( getLEDState() < 0 )
            {
                if( powerState() != 1 || powerStateTarget() != 1 )
                {
                    return 0;
                }

                state( stateCodes::ERROR );
                return 0;
            }
        }

        if( frameGrabber<cred2Ctrl>::updateINDI() < 0 )
        {
            log<software_error>( { __FILE__, __LINE__ } );
            state( stateCodes::ERROR );
            return 0;
        }

        if( stdCamera<cred2Ctrl>::updateINDI() < 0 )
        {
            log<software_error>( { __FILE__, __LINE__ } );
            state( stateCodes::ERROR );
            return 0;
        }

        if( edtCamera<cred2Ctrl>::updateINDI() < 0 )
        {
            log<software_error>( { __FILE__, __LINE__ } );
            state( stateCodes::ERROR );
            return 0;
        }

        if( telemeter<cred2Ctrl>::appLogic() < 0 )
        {
            log<software_error>( { __FILE__, __LINE__ } );
            return 0;
        }
    }

    return 0;
}

inline int cred2Ctrl::onPowerOff()
{
    m_powerOnCounter = 0;

    std::lock_guard<std::mutex> lock( m_indiMutex );

    m_temps.setInvalid();
    m_ccdTemp              = -999;
    m_tempControlStatus    = false;
    m_tempControlOnTarget  = false;
    m_tempControlStatusStr = "UNKNOWN";

    updateIfChanged( m_indiP_temps, "motherboard", m_temps.motherboard );
    updateIfChanged( m_indiP_temps, "frontend", m_temps.frontend );
    updateIfChanged( m_indiP_temps, "powerboard", m_temps.powerboard );
    updateIfChanged( m_indiP_temps, "snake", m_temps.snake );
    updateIfChanged( m_indiP_temps, "setpoint", m_temps.setpoint );
    updateIfChanged( m_indiP_temps, "peltier", m_temps.peltier );
    updateIfChanged( m_indiP_temps, "heatsink", m_temps.heatsink );

    if( stdCamera<cred2Ctrl>::onPowerOff() < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
    }

    if( edtCamera<cred2Ctrl>::onPowerOff() < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
    }

    if( frameGrabber<cred2Ctrl>::onPowerOff() < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
    }

    return 0;
}

inline int cred2Ctrl::whilePowerOff()
{
    std::lock_guard<std::mutex> lock( m_indiMutex );

    if( stdCamera<cred2Ctrl>::whilePowerOff() < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
    }

    if( edtCamera<cred2Ctrl>::whilePowerOff() < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
    }

    return 0;
}

inline int cred2Ctrl::appShutdown()
{
    STDCAMERA_APP_SHUTDOWN;

    dev::edtCamera<cred2Ctrl>::appShutdown();

    FRAMEGRABBER_APP_SHUTDOWN;

    TELEMETER_APP_SHUTDOWN;

    return 0;
}

inline int cred2Ctrl::sendCommand( std::string &response, const std::string &command, bool logFailure )
{
    std::string rawResponse;

    response.clear();

    { // mutex scope
        std::lock_guard<std::recursive_mutex> guard( m_cameraMutex );
        if( pdvSerialWriteRead( rawResponse, command, logFailure ) != 0 )
        {
            if( powerState() != 1 || powerStateTarget() != 1 )
            {
                return -1;
            }

            if( !logFailure )
            {
                return -1;
            }

            return log<software_error, -1>( { __FILE__, __LINE__, "error sending C-RED 2 command: " + command } );
        }
    }

    response = cred2CleanResponse( rawResponse );

    return 0;
}

inline int cred2Ctrl::issueCommand( const std::string &command, bool allowNoResponse )
{
    std::string response;
    if( sendCommand( response, command, !allowNoResponse ) < 0 )
    {
        if( allowNoResponse )
        {
            return 0;
        }

        return -1;
    }

    if( !cred2ResponseOK( response ) )
    {
        return log<text_log, -1>( "C-RED 2 rejected command '" + command + "' with response: " + response,
                                  logPrio::LOG_ERROR );
    }

    return 0;
}

inline int cred2Ctrl::syncROIFromCamera()
{
    std::string response;
    bool        cropEnabled = false;
    int         startColumn = 0;
    int         endColumn   = 0;
    int         startRow    = 0;
    int         endRow      = 0;

    if( sendCommand( response, "cropping raw", false ) < 0 ||
        cred2ParseCropState( cropEnabled, startColumn, endColumn, startRow, endRow, response ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "failed to query current cropping mode: " + response } );
    }

    if( !cropEnabled )
    {
        m_cameraCropEnabled = false;
        m_currentROI.x      = m_full_x;
        m_currentROI.y      = m_full_y;
        m_currentROI.w      = m_full_w;
        m_currentROI.h      = m_full_h;
        m_currentROI.bin_x  = m_full_bin_x;
        m_currentROI.bin_y  = m_full_bin_y;
    }
    else
    {
        cred2Roi cameraROI;

        if( startColumn == 0 && endColumn == 0 && startRow == 0 && endRow == 0 )
        {
            if( sendCommand( response, "cropping columns raw", false ) < 0 ||
                cred2ParseRange( cameraROI.startColumn, cameraROI.endColumn, response ) < 0 )
            {
                return log<software_error, -1>(
                    { __FILE__, __LINE__, "failed to query current cropping columns: " + response } );
            }

            if( sendCommand( response, "cropping rows raw", false ) < 0 ||
                cred2ParseRange( cameraROI.startRow, cameraROI.endRow, response ) < 0 )
            {
                return log<software_error, -1>(
                    { __FILE__, __LINE__, "failed to query current cropping rows: " + response } );
            }
        }
        else
        {
            cameraROI.startColumn = startColumn;
            cameraROI.endColumn   = endColumn;
            cameraROI.startRow    = startRow;
            cameraROI.endRow      = endRow;
        }

        cameraROI.fullFrame = false;

        if( cred2RoiToCenter(
                m_currentROI.x, m_currentROI.y, m_currentROI.w, m_currentROI.h, cameraROI, m_full_w, m_full_h ) < 0 )
        {
            return log<software_error, -1>( { __FILE__, __LINE__, "camera reported an invalid ROI" } );
        }

        m_currentROI.bin_x  = 1;
        m_currentROI.bin_y  = 1;
        m_cameraCropEnabled = true;
    }

    m_nextROI  = m_currentROI;
    m_width    = m_currentROI.w;
    m_height   = m_currentROI.h;
    m_dataType = _DATATYPE_INT16;

    updateIfChanged( m_indiP_roi_x, "current", m_currentROI.x, INDI_OK );
    updateIfChanged( m_indiP_roi_y, "current", m_currentROI.y, INDI_OK );
    updateIfChanged( m_indiP_roi_w, "current", m_currentROI.w, INDI_OK );
    updateIfChanged( m_indiP_roi_h, "current", m_currentROI.h, INDI_OK );
    updateIfChanged( m_indiP_roi_bin_x, "current", m_currentROI.bin_x, INDI_OK );
    updateIfChanged( m_indiP_roi_bin_y, "current", m_currentROI.bin_y, INDI_OK );

    updateIfChanged( m_indiP_roi_x, "target", m_nextROI.x, INDI_OK );
    updateIfChanged( m_indiP_roi_y, "target", m_nextROI.y, INDI_OK );
    updateIfChanged( m_indiP_roi_w, "target", m_nextROI.w, INDI_OK );
    updateIfChanged( m_indiP_roi_h, "target", m_nextROI.h, INDI_OK );
    updateIfChanged( m_indiP_roi_bin_x, "target", m_nextROI.bin_x, INDI_OK );
    updateIfChanged( m_indiP_roi_bin_y, "target", m_nextROI.bin_y, INDI_OK );

    if( m_currentROI.w != m_raw_width || m_currentROI.h != m_raw_height )
    {
        if( writeConfig() < 0 )
        {
            return log<software_error, -1>( { __FILE__, __LINE__ } );
        }

        m_nextMode = m_modeName.empty() ? m_startupMode : m_modeName;

        if( dev::edtCamera<cred2Ctrl>::pdvReconfig() < 0 )
        {
            return log<software_error, -1>( { __FILE__, __LINE__ } );
        }

        if( setSerialBaud() < 0 )
        {
            return log<software_error, -1>( { __FILE__, __LINE__ } );
        }
    }

    return 0;
}

inline int cred2Ctrl::setSerialBaud()
{
    typedef int ( *setBaudFnT )( PdvDev *, int );
    typedef int ( *getBaudFnT )( PdvDev * );

    if( m_pdv == nullptr )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "cannot set serial baud with null PDV handle" } );
    }

    static setBaudFnT setBaudFn = reinterpret_cast<setBaudFnT>( dlsym( RTLD_DEFAULT, "pdv_serial_set_baud" ) );
    static getBaudFnT getBaudFn = reinterpret_cast<getBaudFnT>( dlsym( RTLD_DEFAULT, "pdv_serial_get_baud" ) );

    if( setBaudFn == nullptr || getBaudFn == nullptr )
    {
        return 0;
    }

    if( setBaudFn( m_pdv, m_serialBaud ) < 0 )
    {
        return log<software_error, -1>(
            { __FILE__, __LINE__, "failed to set C-RED 2 serial baud to " + std::to_string( m_serialBaud ) } );
    }

    int actualBaud = getBaudFn( m_pdv );
    if( actualBaud != m_serialBaud )
    {
        return log<software_error, -1>( { __FILE__,
                                          __LINE__,
                                          "EDT serial baud verification failed: expected " +
                                              std::to_string( m_serialBaud ) + ", got " +
                                              std::to_string( actualBaud ) } );
    }

    return 0;
}

inline int cred2Ctrl::getTemps()
{
    cred2Temps         temps;
    std::string        response;
    std::vector<float> bundledTemps;
    const double       diffLimit = 1.0;
    const std::string  bundledCommand( "temperatures raw" );
    const std::string  setpointCommand( "temperatures snake setpoint raw" );
    const auto         keepLastTemps = [this]( const std::string &detail )
    {
        return log<text_log, 0>( "transient C-RED 2 temperature refresh failure; keeping previous cached values: " +
                                     detail,
                                 logPrio::LOG_WARNING );
    };

    if( sendCommand( response, bundledCommand, false ) < 0 )
    {
        return keepLastTemps( bundledCommand );
    }

    if( cred2ParseFloatVector( bundledTemps, response, 6 ) < 0 )
    {
        return keepLastTemps( bundledCommand + " -> " + response );
    }

    temps.motherboard = bundledTemps[0];
    temps.frontend    = bundledTemps[1];
    temps.powerboard  = bundledTemps[2];
    temps.snake       = bundledTemps[3];
    temps.peltier     = bundledTemps[4];
    temps.heatsink    = bundledTemps[5];

    if( sendCommand( response, setpointCommand, false ) < 0 )
    {
        return keepLastTemps( setpointCommand );
    }

    if( cred2ParseFloat( temps.setpoint, response ) < 0 )
    {
        return keepLastTemps( setpointCommand + " -> " + response );
    }

    m_temps        = temps;
    m_ccdTemp      = temps.snake;
    m_ccdTempSetpt = temps.setpoint;

    if( m_ccdTempSetpt < 19.5 )
    {
        m_tempControlStatus = true;
        if( std::fabs( m_ccdTemp - m_ccdTempSetpt ) < diffLimit )
        {
            m_tempControlStatusStr = "ON TARGET";
            m_tempControlOnTarget  = true;
        }
        else
        {
            m_tempControlStatusStr = "OFF TARGET";
            m_tempControlOnTarget  = false;
        }
    }
    else
    {
        m_tempControlStatus   = false;
        m_tempControlOnTarget = false;
        if( std::fabs( m_ccdTemp - m_ccdTempSetpt ) < diffLimit )
        {
            m_tempControlStatusStr = "TEMP OFF";
        }
        else
        {
            m_tempControlStatusStr = "WARMING";
        }
    }

    updateIfChanged( m_indiP_temps, "motherboard", m_temps.motherboard );
    updateIfChanged( m_indiP_temps, "frontend", m_temps.frontend );
    updateIfChanged( m_indiP_temps, "powerboard", m_temps.powerboard );
    updateIfChanged( m_indiP_temps, "snake", m_temps.snake );
    updateIfChanged( m_indiP_temps, "setpoint", m_temps.setpoint );
    updateIfChanged( m_indiP_temps, "peltier", m_temps.peltier );
    updateIfChanged( m_indiP_temps, "heatsink", m_temps.heatsink );

    recordTemps();
    recordCamera();

    return 0;
}

inline int cred2Ctrl::getFPS()
{
    std::string response;
    float       fpsValue = 0;

    if( sendCommand( response, "fps raw" ) < 0 )
    {
        return -1;
    }

    if( cred2ParseFloat( fpsValue, response ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "failed to parse fps response: " + response } );
    }

    m_fps = fpsValue;
    recordCamera();

    return 0;
}

inline int cred2Ctrl::getFanSpeed()
{
    std::string response;
    std::string fanMode;
    float       fanPercent = 0;

    if( sendCommand( response, "fan mode raw" ) < 0 )
    {
        return -1;
    }

    fanMode = cred2LowerResponse( response );
    if( fanMode.find( "auto" ) == std::string::npos && fanMode.find( "manual" ) == std::string::npos )
    {
        if( sendCommand( response, "fan mode" ) < 0 )
        {
            return -1;
        }

        fanMode = cred2LowerResponse( response );
    }

    if( fanMode.find( "auto" ) != std::string::npos )
    {
        m_fanSpeedName    = "auto";
        m_fanSpeedNameSet = m_fanSpeedName;
        m_fanSpeedValid   = true;
        return 0;
    }

    if( fanMode.find( "manual" ) == std::string::npos )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "failed to parse fan mode response: " + response } );
    }

    if( sendCommand( response, "fan speed raw" ) < 0 )
    {
        return -1;
    }

    if( cred2ParseFloat( fanPercent, response ) < 0 )
    {
        if( sendCommand( response, "fan speed" ) < 0 || cred2ParseFloat( fanPercent, response ) < 0 )
        {
            return log<software_error, -1>( { __FILE__, __LINE__, "failed to parse fan speed response: " + response } );
        }
    }

    m_fanSpeedName    = cred2FanPresetName( fanPercent );
    m_fanSpeedNameSet = m_fanSpeedName;
    m_fanSpeedValid   = true;
    recordCamera();

    return 0;
}

inline int cred2Ctrl::getAnalogGain()
{
    std::string response;
    std::string analogGain;

    if( sendCommand( response, "sensibility" ) < 0 )
    {
        return -1;
    }

    if( cred2AnalogGainName( analogGain, response ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "failed to parse sensibility response: " + response } );
    }

    m_analogGainName    = analogGain;
    m_analogGainNameSet = m_analogGainName;
    m_analogGainValid   = true;
    recordCamera();

    return 0;
}

inline int cred2Ctrl::getLEDState()
{
    std::string response;
    bool        ledState = false;

    if( sendCommand( response, "led raw" ) < 0 )
    {
        if( sendCommand( response, "led" ) < 0 )
        {
            return -1;
        }
    }

    if( cred2ParseBool( ledState, response ) < 0 )
    {
        std::string clean = cred2LowerResponse( response );

        if( clean.find( "off" ) != std::string::npos )
        {
            ledState = false;
        }
        else if( clean.find( "on" ) != std::string::npos )
        {
            ledState = true;
        }
        else
        {
            if( sendCommand( response, "led" ) < 0 )
            {
                return -1;
            }

            clean = cred2LowerResponse( response );
            if( clean.find( "off" ) != std::string::npos )
            {
                ledState = false;
            }
            else if( clean.find( "on" ) != std::string::npos )
            {
                ledState = true;
            }
            else
            {
                return log<software_error, -1>( { __FILE__, __LINE__, "failed to parse led response: " + response } );
            }
        }
    }

    m_ledState      = ledState;
    m_ledStateSet   = ledState;
    m_ledStateValid = true;
    recordCamera();

    return 0;
}

inline int cred2Ctrl::updateFPSLimits()
{
    std::string response;
    float       minFPS = 0;
    float       maxFPS = 0;

    if( sendCommand( response, "minfps raw" ) < 0 || cred2ParseFloat( minFPS, response ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "failed to parse minfps response: " + response } );
    }

    if( sendCommand( response, "maxfps raw" ) < 0 || cred2ParseFloat( maxFPS, response ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "failed to parse maxfps response: " + response } );
    }

    m_minFPS = minFPS;
    m_maxFPS = maxFPS;
    m_fpsSet = std::clamp( m_fpsSet, m_minFPS, m_maxFPS );

    updateIfChanged( m_indiP_fpsLimits, "min", m_minFPS, INDI_IDLE );
    updateIfChanged( m_indiP_fpsLimits, "max", m_maxFPS, INDI_IDLE );

    recordCamera();

    return 0;
}

inline int cred2Ctrl::powerOnDefaults()
{
    m_tempControlStatus    = false;
    m_tempControlStatusSet = false;
    m_tempControlStatusStr = "TEMP OFF";
    m_tempControlOnTarget  = false;
    m_cameraCropEnabled    = false;
    m_currentROI.x         = m_default_x;
    m_currentROI.y         = m_default_y;
    m_currentROI.w         = m_default_w;
    m_currentROI.h         = m_default_h;
    m_currentROI.bin_x     = m_default_bin_x;
    m_currentROI.bin_y     = m_default_bin_y;

    m_fanSpeedValid     = false;
    m_analogGainValid   = false;
    m_ledStateValid     = false;
    m_fanSpeedNameSet   = m_defaultFanSpeed;
    m_analogGainNameSet = "med";
    m_ledStateSet       = m_defaultLEDState;

    m_nextROI = m_currentROI;

    return 0;
}

inline int cred2Ctrl::setTempControl()
{
    if( m_tempControlStatusSet )
    {
        if( m_ccdTempSetpt >= 19.5 )
        {
            return log<text_log, 0>(
                "temperature control is setpoint-driven for C-RED 2; choose a target below 20 C to cool",
                logPrio::LOG_NOTICE );
        }

        return setTempSetPt();
    }

    m_ccdTempSetpt = 20;
    return setTempSetPt();
}

inline int cred2Ctrl::setTempSetPt()
{
    if( m_ccdTempSetpt < m_minTemp || m_ccdTempSetpt > m_maxTemp )
    {
        return log<text_log, -1>( "attempt to set temperature outside valid range: " + std::to_string( m_ccdTempSetpt ),
                                  logPrio::LOG_ERROR );
    }

    std::ostringstream command;
    command << "set temperatures snake " << m_ccdTempSetpt;

    if( issueCommand( command.str() ) < 0 )
    {
        return -1;
    }

    m_tempControlStatusSet = ( m_ccdTempSetpt < 19.5 );
    m_tempControlStatus    = m_tempControlStatusSet;
    m_tempControlOnTarget  = false;
    m_tempControlStatusStr = m_tempControlStatusSet ? "OFF TARGET" : "WARMING";

    recordCamera();

    return 0;
}

inline int cred2Ctrl::setFPS()
{
    if( m_fpsSet < m_minFPS || m_fpsSet > m_maxFPS )
    {
        return log<text_log, -1>( "attempt to set fps outside valid range: " + std::to_string( m_fpsSet ),
                                  logPrio::LOG_ERROR );
    }

    std::ostringstream command;
    command << "set fps " << m_fpsSet;

    if( issueCommand( command.str() ) < 0 )
    {
        return -1;
    }

    log<text_log>( "set fps: " + std::to_string( m_fpsSet ) );

    return getFPS();
}

inline int cred2Ctrl::setFanSpeed()
{
    if( m_fanSpeedNameSet == "auto" )
    {
        if( issueCommand( "set fan mode automatic" ) < 0 )
        {
            return -1;
        }
    }
    else
    {
        int fanPercent = 0;
        if( cred2FanPresetPercent( fanPercent, m_fanSpeedNameSet ) < 0 )
        {
            return log<software_error, -1>( { __FILE__, __LINE__, "unknown fan speed preset: " + m_fanSpeedNameSet } );
        }

        if( issueCommand( "set fan mode manual" ) < 0 )
        {
            return -1;
        }

        if( issueCommand( "set fan speed " + std::to_string( fanPercent ) ) < 0 )
        {
            return -1;
        }
    }

    return getFanSpeed();
}

inline int cred2Ctrl::setAnalogGain()
{
    std::string commandGain;

    if( cred2AnalogGainCommand( commandGain, m_analogGainNameSet ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "unknown analog gain preset: " + m_analogGainNameSet } );
    }

    if( issueCommand( "set sensibility " + commandGain ) < 0 )
    {
        return -1;
    }

    return getAnalogGain();
}

inline int cred2Ctrl::setLED()
{
    if( m_ledStateSet )
    {
        if( issueCommand( "set led on" ) < 0 )
        {
            return -1;
        }
    }
    else
    {
        if( issueCommand( "set led off" ) < 0 )
        {
            return -1;
        }
    }

    return getLEDState();
}

inline int cred2Ctrl::setExpTime()
{
    return 0;
}

inline int cred2Ctrl::checkNextROI()
{
    auto roundToStep = []( int value, int step )
    { return static_cast<int>( std::lround( static_cast<double>( value ) / static_cast<double>( step ) ) ) * step; };

    m_nextROI.bin_x = 1;
    m_nextROI.bin_y = 1;

    int width = roundToStep( m_nextROI.w, 32 );
    width     = std::clamp( width, 32, m_full_w );

    int height = roundToStep( m_nextROI.h, 4 );
    height     = std::clamp( height, 4, m_full_h );

    int startColumn = static_cast<int>( std::lround( m_nextROI.x - 0.5f * ( static_cast<float>( width ) - 1.0f ) ) );
    int startRow    = static_cast<int>( std::lround( m_nextROI.y - 0.5f * ( static_cast<float>( height ) - 1.0f ) ) );

    startColumn = roundToStep( startColumn, 32 );
    startRow    = roundToStep( startRow, 4 );

    startColumn = std::clamp( startColumn, 0, m_full_w - width );
    startRow    = std::clamp( startRow, 0, m_full_h - height );

    m_nextROI.w = width;
    m_nextROI.h = height;
    m_nextROI.x = startColumn + 0.5f * ( static_cast<float>( width ) - 1.0f );
    m_nextROI.y = startRow + 0.5f * ( static_cast<float>( height ) - 1.0f );

    updateIfChanged( m_indiP_roi_x, "target", m_nextROI.x, INDI_OK );
    updateIfChanged( m_indiP_roi_y, "target", m_nextROI.y, INDI_OK );
    updateIfChanged( m_indiP_roi_w, "target", m_nextROI.w, INDI_OK );
    updateIfChanged( m_indiP_roi_h, "target", m_nextROI.h, INDI_OK );
    updateIfChanged( m_indiP_roi_bin_x, "target", m_nextROI.bin_x, INDI_OK );
    updateIfChanged( m_indiP_roi_bin_y, "target", m_nextROI.bin_y, INDI_OK );

    return 0;
}

inline int cred2Ctrl::setNextROI()
{
    if( checkNextROI() < 0 )
    {
        return -1;
    }

    recordCamera( true );
    state( stateCodes::CONFIGURING );

    m_nextMode = m_modeName.empty() ? m_startupMode : m_modeName;
    m_reconfig = true;

    updateSwitchIfChanged( m_indiP_roi_set, "request", pcf::IndiElement::Off, INDI_IDLE );
    updateSwitchIfChanged( m_indiP_roi_full, "request", pcf::IndiElement::Off, INDI_IDLE );
    updateSwitchIfChanged( m_indiP_roi_last, "request", pcf::IndiElement::Off, INDI_IDLE );
    updateSwitchIfChanged( m_indiP_roi_default, "request", pcf::IndiElement::Off, INDI_IDLE );

    return 0;
}

inline int cred2Ctrl::writeConfig()
{
    std::ofstream fout( m_configFile );
    if( fout.fail() )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "error opening C-RED 2 config file for writing" } );
    }

    const int width  = m_nextROI.w / m_nextROI.bin_x;
    const int height = m_nextROI.h / m_nextROI.bin_y;

    fout << "camera_class:                  \"FirstLightImaging\"\n";
    fout << "camera_model:                  \"C-RED 2\"\n";
    fout << "camera_info:                   \"" << width << "x" << height << " (4-tap, freerun)\"\n";
    fout << "width:                         " << width << "\n";
    fout << "height:                        " << height << "\n";
    fout << "depth:                         16\n";
    fout << "extdepth:                      16\n";
    fout << "rbtfile:                       aiagcl.bit\n";
    fout << "CL_DATA_PATH_NORM:             3f       # four tap\n";
    fout << "CL_CFG_NORM:                   02\n";
    fout << "CL_CFG2_NORM:                  40\n";
    fout << "method_framesync:              EMULATE_TIMEOUT\n";
    fout << "htaps:                         4\n";
    fout << "serial_baud:                  " << m_serialBaud << "\n";
    fout << "serial_term:                   <0A>\n";
    fout << "serial_waitc:                  0D\n";

    fout.close();

    return 0;
}

inline int cred2Ctrl::configureAcquisition()
{
    std::unique_lock<std::mutex>          lock( m_indiMutex );
    std::lock_guard<std::recursive_mutex> cameraGuard( m_cameraMutex );

    cred2Roi roi;
    if( cred2RoiFromCenter( roi, m_nextROI.x, m_nextROI.y, m_nextROI.w, m_nextROI.h, m_full_w, m_full_h ) < 0 )
    {
        state( stateCodes::ERROR );
        return log<software_error, -1>( { __FILE__, __LINE__, "invalid ROI specified for C-RED 2 configure" } );
    }

    if( roi.fullFrame )
    {
        if( m_cameraCropEnabled && issueCommand( "set cropping off", true ) < 0 )
        {
            state( stateCodes::ERROR );
            return -1;
        }

        m_cameraCropEnabled = false;
    }
    else
    {
        if( issueCommand( "set cropping columns " + cred2ColumnsSpec( roi ), true ) < 0 ||
            issueCommand( "set cropping rows " + cred2RowsSpec( roi ), true ) < 0 ||
            issueCommand( "set cropping on", true ) < 0 )
        {
            state( stateCodes::ERROR );
            return -1;
        }

        m_cameraCropEnabled = true;
    }

    m_currentROI = m_nextROI;

    updateIfChanged( m_indiP_roi_x, "current", m_currentROI.x, INDI_OK );
    updateIfChanged( m_indiP_roi_y, "current", m_currentROI.y, INDI_OK );
    updateIfChanged( m_indiP_roi_w, "current", m_currentROI.w, INDI_OK );
    updateIfChanged( m_indiP_roi_h, "current", m_currentROI.h, INDI_OK );
    updateIfChanged( m_indiP_roi_bin_x, "current", m_currentROI.bin_x, INDI_OK );
    updateIfChanged( m_indiP_roi_bin_y, "current", m_currentROI.bin_y, INDI_OK );

    m_nextROI = m_currentROI;

    updateIfChanged( m_indiP_roi_x, "target", m_nextROI.x, INDI_OK );
    updateIfChanged( m_indiP_roi_y, "target", m_nextROI.y, INDI_OK );
    updateIfChanged( m_indiP_roi_w, "target", m_nextROI.w, INDI_OK );
    updateIfChanged( m_indiP_roi_h, "target", m_nextROI.h, INDI_OK );
    updateIfChanged( m_indiP_roi_bin_x, "target", m_nextROI.bin_x, INDI_OK );
    updateIfChanged( m_indiP_roi_bin_y, "target", m_nextROI.bin_y, INDI_OK );

    m_width    = m_currentROI.w;
    m_height   = m_currentROI.h;
    m_dataType = _DATATYPE_INT16;

    // Use the current FPS target while the camera settles so the framegrabber
    // does not keep reconfiguring latency buffers on a stale pre-ROI value.
    if( m_fpsSet > 0 )
    {
        m_fps = m_fpsSet;
    }

    // Give the camera a few app-logic cycles to settle after crop changes
    // before resuming serial status polls such as fps, temperatures, and
    // refreshed FPS limits.
    m_roiSettleCounter = 5;

    recordCamera( true );
    state( stateCodes::READY );

    return 0;
}

inline float cred2Ctrl::fps()
{
    return m_fps;
}

inline int cred2Ctrl::startAcquisition()
{
    state( stateCodes::OPERATING );
    recordCamera();
    return edtCamera<cred2Ctrl>::pdvStartAcquisition();
}

inline int cred2Ctrl::acquireAndCheckValid()
{
    return edtCamera<cred2Ctrl>::pdvAcquire( m_currImageTimestamp );
}

inline int cred2Ctrl::loadImageIntoStream( void *dest )
{
    if( frameGrabber<cred2Ctrl>::loadImageIntoStreamCopy( dest, m_image_p, m_width, m_height, m_typeSize ) == nullptr )
    {
        return -1;
    }

    return 0;
}

inline int cred2Ctrl::reconfig()
{
    recordCamera( true );
    state( stateCodes::CONFIGURING );

    if( writeConfig() < 0 )
    {
        return -1;
    }

    std::lock_guard<std::recursive_mutex> guard( m_cameraMutex );
    int                                   rv = edtCamera<cred2Ctrl>::pdvReconfig();
    if( rv < 0 )
    {
        return rv;
    }

    if( setSerialBaud() < 0 )
    {
        return -1;
    }

    state( stateCodes::READY );
    m_nextMode = m_modeName;

    return 0;
}

inline int cred2Ctrl::checkRecordTimes()
{
    return telemeter<cred2Ctrl>::checkRecordTimes( cred2_temps(), telem_stdcam(), telem_fgtimings() );
}

inline int cred2Ctrl::recordTelem( const cred2_temps * )
{
    return recordTemps( true );
}

inline int cred2Ctrl::recordTelem( const telem_stdcam * )
{
    return recordCamera( true );
}

inline int cred2Ctrl::recordTelem( const telem_fgtimings * )
{
    return recordFGTimings( true );
}

inline int cred2Ctrl::recordTemps( bool force )
{
    static cred2Temps lastTemps;

    if( !( lastTemps == m_temps ) || force )
    {
        telem<cred2_temps>( { m_temps.motherboard,
                              m_temps.frontend,
                              m_temps.powerboard,
                              m_temps.snake,
                              m_temps.setpoint,
                              m_temps.peltier,
                              m_temps.heatsink } );
        lastTemps = m_temps;
    }

    return 0;
}

} // namespace app
} // namespace MagAOX

#endif // cred2Ctrl_hpp
