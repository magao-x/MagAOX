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

    typedef MagAOXApp<> MagAOXAppT;

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
    static constexpr bool c_stdCamera_synchro      = false; ///< Do not expose synchro controls in the first pass.
    static constexpr bool c_stdCamera_usesModes    = false; ///< Use one synthetic runtime mode rather than INDI modes.
    static constexpr bool c_stdCamera_usesROI      = true;  ///< Expose ROI controls.
    static constexpr bool c_stdCamera_cropMode     = false; ///< Do not expose crop-mode controls separately.
    static constexpr bool c_stdCamera_hasShutter   = false; ///< Do not expose shutter controls.
    static constexpr bool c_stdCamera_usesStateString    = false; ///< Do not expose a dark-management state string.
    static constexpr bool c_edtCamera_relativeConfigPath = false; ///< Use an absolute temporary EDT config path.
    static constexpr bool c_frameGrabber_flippable       = false; ///< Do not expose image flip controls.

    ///@}

  protected:
    /** \name Configurable Parameters - Data
     * @{
     */
    std::string m_configFile; ///< Absolute path to the temporary EDT configuration file.
    ///@}

    /** \name C-RED 2 State - Data
     * @{
     */
    cred2Temps m_temps; ///< Cached camera temperature values used for INDI and telemetry updates.

    bool m_poweredOn{ false }; ///< True after a power cycle until the startup setpoint has been re-applied.

    std::recursive_mutex m_cameraMutex; ///< Protects serial command traffic and EDT reconfiguration.
    ///@}

    /** \name INDI - Data
     * @{
     */
    pcf::IndiProperty m_indiP_temps; ///< Property reporting the detailed C-RED 2 temperature channels.

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

    /// Record standard camera telemetry.
    int recordTelem( const telem_stdcam * /**< [in] type-dispatch tag */ );

    /// Record framegrabber timing telemetry.
    int recordTelem( const telem_fgtimings * /**< [in] type-dispatch tag */ );

    ///@}

  protected:
    /// Send a command over Camera Link serial and clean the response.
    int sendCommand( std::string       &response, ///< [out] cleaned command response
                     const std::string &command   /**< [in] CLI command to send */
    );

    /// Send a command that should return a success acknowledgement.
    int issueCommand( const std::string &command /**< [in] CLI command to send */ );
};

inline cred2Ctrl::cred2Ctrl() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    m_powerMgtEnabled = true;
    m_powerOnWait     = 10;

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

    m_temps.setInvalid();
}

inline cred2Ctrl::~cred2Ctrl() noexcept
{
}

inline void cred2Ctrl::setupConfig()
{
    dev::stdCamera<cred2Ctrl>::setupConfig( config );
    dev::edtCamera<cred2Ctrl>::setupConfig( config );
    dev::frameGrabber<cred2Ctrl>::setupConfig( config );
    dev::telemeter<cred2Ctrl>::setupConfig( config );
}

inline void cred2Ctrl::loadConfig()
{
    dev::stdCamera<cred2Ctrl>::loadConfig( config );

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
        log<software_critical>( { __FILE__, __LINE__, "could not write initial C-RED 2 EDT config" } );
        m_shutdown = true;
        return;
    }

    dev::edtCamera<cred2Ctrl>::loadConfig( config );
    dev::frameGrabber<cred2Ctrl>::loadConfig( config );
    dev::telemeter<cred2Ctrl>::loadConfig( config );
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

    if( dev::stdCamera<cred2Ctrl>::appStartup() < 0 )
    {
        return log<software_critical, -1>( { __FILE__, __LINE__ } );
    }

    if( dev::edtCamera<cred2Ctrl>::appStartup() < 0 )
    {
        return log<software_critical, -1>( { __FILE__, __LINE__ } );
    }

    if( dev::frameGrabber<cred2Ctrl>::appStartup() < 0 )
    {
        return log<software_critical, -1>( { __FILE__, __LINE__ } );
    }

    if( dev::telemeter<cred2Ctrl>::appStartup() < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__ } );
    }

    return 0;
}

inline int cred2Ctrl::appLogic()
{
    if( dev::stdCamera<cred2Ctrl>::appLogic() < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__ } );
    }

    if( dev::edtCamera<cred2Ctrl>::appLogic() < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__ } );
    }

    if( dev::frameGrabber<cred2Ctrl>::appLogic() < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__ } );
    }

    if( state() == stateCodes::POWERON )
    {
        return 0;
    }

    if( state() == stateCodes::NOTCONNECTED || state() == stateCodes::ERROR )
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
        }
        else
        {
            sleep( 1 );
            return 0;
        }
    }

    if( state() == stateCodes::CONNECTED )
    {
        std::unique_lock<std::mutex> lock( m_indiMutex );

        if( updateFPSLimits() < 0 || getTemps() < 0 || getFPS() < 0 )
        {
            if( powerState() != 1 || powerStateTarget() != 1 )
            {
                return 0;
            }

            state( stateCodes::ERROR );
            return 0;
        }

        state( stateCodes::READY );

        if( m_poweredOn && m_ccdTempSetpt > -999 )
        {
            m_poweredOn = false;
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

    m_poweredOn = true;

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
    dev::stdCamera<cred2Ctrl>::appShutdown();
    dev::edtCamera<cred2Ctrl>::appShutdown();
    dev::frameGrabber<cred2Ctrl>::appShutdown();
    dev::telemeter<cred2Ctrl>::appShutdown();

    return 0;
}

inline int cred2Ctrl::sendCommand( std::string &response, const std::string &command )
{
    std::string rawResponse;

    { // mutex scope
        std::lock_guard<std::recursive_mutex> guard( m_cameraMutex );
        if( pdvSerialWriteRead( rawResponse, command ) != 0 )
        {
            if( powerState() != 1 || powerStateTarget() != 1 )
            {
                return -1;
            }

            return log<software_error, -1>( { __FILE__, __LINE__, "error sending C-RED 2 command: " + command } );
        }
    }

    response = cred2CleanResponse( rawResponse );

    return 0;
}

inline int cred2Ctrl::issueCommand( const std::string &command )
{
    std::string response;
    if( sendCommand( response, command ) < 0 )
    {
        return -1;
    }

    if( !cred2ResponseOK( response ) )
    {
        return log<text_log, -1>( "C-RED 2 rejected command '" + command + "' with response: " + response,
                                  logPrio::LOG_ERROR );
    }

    return 0;
}

inline int cred2Ctrl::getTemps()
{
    cred2Temps   temps;
    std::string  response;
    const double diffLimit = 1.0;

    if( sendCommand( response, "temperatures motherboard raw" ) < 0 ||
        cred2ParseFloat( temps.motherboard, response ) < 0 )
    {
        return log<software_error, -1>(
            { __FILE__, __LINE__, "failed to parse motherboard temperature: " + response } );
    }

    if( sendCommand( response, "temperatures frontend raw" ) < 0 || cred2ParseFloat( temps.frontend, response ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "failed to parse frontend temperature: " + response } );
    }

    if( sendCommand( response, "temperatures powerboard raw" ) < 0 ||
        cred2ParseFloat( temps.powerboard, response ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "failed to parse powerboard temperature: " + response } );
    }

    if( sendCommand( response, "temperatures snake raw" ) < 0 || cred2ParseFloat( temps.snake, response ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "failed to parse detector temperature: " + response } );
    }

    if( sendCommand( response, "temperatures snake setpoint raw" ) < 0 ||
        cred2ParseFloat( temps.setpoint, response ) < 0 )
    {
        return log<software_error, -1>(
            { __FILE__, __LINE__, "failed to parse detector setpoint temperature: " + response } );
    }

    if( sendCommand( response, "temperatures peltier raw" ) < 0 || cred2ParseFloat( temps.peltier, response ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "failed to parse peltier temperature: " + response } );
    }

    if( sendCommand( response, "temperatures heatsink raw" ) < 0 || cred2ParseFloat( temps.heatsink, response ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "failed to parse heatsink temperature: " + response } );
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

    recordCamera();

    return 0;
}

inline int cred2Ctrl::powerOnDefaults()
{
    m_tempControlStatus    = false;
    m_tempControlStatusSet = false;
    m_tempControlStatusStr = "TEMP OFF";
    m_tempControlOnTarget  = false;

    m_currentROI.x     = m_default_x;
    m_currentROI.y     = m_default_y;
    m_currentROI.w     = m_default_w;
    m_currentROI.h     = m_default_h;
    m_currentROI.bin_x = m_default_bin_x;
    m_currentROI.bin_y = m_default_bin_y;

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
        if( issueCommand( "set cropping off" ) < 0 )
        {
            state( stateCodes::ERROR );
            return -1;
        }
    }
    else
    {
        if( issueCommand( "set cropping columns " + cred2ColumnsSpec( roi ) ) < 0 ||
            issueCommand( "set cropping rows " + cred2RowsSpec( roi ) ) < 0 || issueCommand( "set cropping on" ) < 0 )
        {
            state( stateCodes::ERROR );
            return -1;
        }
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

    if( updateFPSLimits() < 0 )
    {
        state( stateCodes::ERROR );
        return -1;
    }

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

    state( stateCodes::READY );
    m_nextMode = m_modeName;

    return 0;
}

inline int cred2Ctrl::checkRecordTimes()
{
    return telemeter<cred2Ctrl>::checkRecordTimes( telem_stdcam(), telem_fgtimings() );
}

inline int cred2Ctrl::recordTelem( const telem_stdcam * )
{
    return recordCamera( true );
}

inline int cred2Ctrl::recordTelem( const telem_fgtimings * )
{
    return recordFGTimings( true );
}

} // namespace app
} // namespace MagAOX

#endif // cred2Ctrl_hpp
