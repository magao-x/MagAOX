/** \file smc100ccCtrl.hpp
 * \brief Newport SMC100CC stage controller application.
 *
 * \ingroup smc100ccCtrl_files
 *
 * History:
 * - 2019-01-10 created by CJB
 *
 */
#ifndef smc100ccCtrl_hpp
#define smc100ccCtrl_hpp

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <bitset>

/** \defgroup smc100ccCtrl SMC100CC Stage Controller
 * \brief Control an SMC100CC stage and assign its configured presets.
 * \ingroup apps
 */

/** \defgroup smc100ccCtrl_files SMC100CC Stage Controller Files
 * \ingroup smc100ccCtrl
 */

namespace MagAOX
{
namespace app
{

/// Control one SMC100CC stage and publish its position, presets, and motion state.
/** The controller is addressed as axis 1 over a USB serial connection. Homing
 * moves the stage to its configured offset before normal READY operation.
 * Preset assignment compares measured positions against configured positions.
 * \ingroup smc100ccCtrl
 */
class smc100ccCtrl : public MagAOXApp<>,
                     public tty::usbDevice,
                     public dev::ioDevice,
                     public dev::stdMotionStage<smc100ccCtrl>,
                     public dev::telemeter<smc100ccCtrl>
{

    friend class dev::stdMotionStage<smc100ccCtrl>;

    friend class dev::telemeter<smc100ccCtrl>;

  protected:
    /** \name Configurable Parameters - Data
     *
     *@{
     */
    double m_homingOffset{ 0 }; ///< Absolute position requested after the controller finishes homing.

    double m_opDelta{ 0 }; ///< Move-size threshold above which the application reports OPERATING.

    double m_presetTolerance{ 1e-3 }; ///< Absolute position tolerance for assigning a preset, in stage position units.

    ///@}

    pcf::IndiProperty m_indiP_position; ///< Current and requested stage positions exposed through INDI.

    std::vector<std::string> m_validStateCodes{}; ///< Legacy controller-state list, currently unused.

    double m_position{ 0 }; ///< Most recently measured stage position used for telemetry and preset assignment.

    double m_target{ 0 }; ///< Last position target received through the position INDI property.

    bool m_wasHoming{ 0 }; ///< Whether homing was observed and still requires the configured offset move.

    bool m_powerOnHomed{ false }; ///< Whether automatic homing has been requested during this power cycle.

    bool m_moveOp{ true }; ///< Whether the current move is large enough to report OPERATING.
  public:
    /// Handle a requested position update after validating its device and property names.
    INDI_NEWCALLBACK_DECL( smc100ccCtrl, m_indiP_position );

    /// Initialize power management and disable default preset positions.
    smc100ccCtrl();

    /// Destroy the controller application.
    ~smc100ccCtrl() noexcept
    {
    }

    /// Register controller, USB, motion-stage, and telemetry configuration options.
    virtual void setupConfig();

    /// Load controller options and delegated device configuration, requesting shutdown on errors.
    virtual void loadConfig();

    /// Register the position property and initialize presets and telemetry.
    virtual int appStartup();

    /// Discover and connect the controller, then monitor homing, motion, positions, and presets.
    virtual int appLogic();

    /// Complete application shutdown; no controller-specific work is required.
    virtual int appShutdown();

    /// Reset the power-cycle homing flag and record stage and position telemetry.
    virtual int onPowerOff();

    /// Build an axis-1 serial command terminated by CR/LF.
    int makeCom( std::string       &str, /**< [out] Complete serial command. */
                 const std::string &com  /**< [in] Controller command without axis or terminator. */
    );

    /// Split a controller reply into its axis, command, and value fields.
    /** \returns 0 on a parsed response, including an unprefixed textual response.
     * \returns -1 if the response is too short to parse.
     */
    int splitResponse( int         &axis, /**< [out] Returned axis, or zero for an unprefixed textual response. */
                       std::string &com,  /**< [out] Two-character command, or empty for a textual response. */
                       std::string &val,  /**< [out] Response value with trailing CR/LF removed. */
                       std::string &resp  /**< [in] Controller response to parse. */
    );

    /// Query and validate the controller's TS reply, extracting its two-character state code.
    int getCtrlState( std::string &state /**< [out] Controller state code on a valid reply. */ );

    /// Test serial communication by issuing a TS query without parsing its response.
    /** \returns 0 if the serial exchange succeeds, or in simulation mode.
     * \returns -1 if the serial exchange fails.
     */
    int testConnection();

    /// Query and parse the controller's measured position.
    /** \returns 0 if the position was read and parsed.
     * \returns -1 on a serial or parsing error.
     */
    int getPosition( double &pos /**< [out] Measured stage position on success. */ );

    /// Query the controller's last command error and translate it to a readable message.
    /** \returns 0 if the controller reports no error.
     * \returns -1 on a controller error, malformed reply, or serial failure.
     */
    int getLastError( std::string &errStr /**< [out] Error description when available; unchanged on success. */ );

    /** \name Standard Motion Stage Interface
     * @{
     *
     */

    /// Record telemetry and send the immediate stop command without taking the INDI mutex.
    int stop();

    /// Reset the home-request switch, record telemetry, and send the homing command.
    int startHoming();

    /// Return the first preset within the configured strict position tolerance, or zero for no match.
    double presetNumber();

    /// Command an absolute position and report OPERATING when the move exceeds the configured threshold.
    int moveTo( double position /**< [in] Requested absolute position in stage position units. */ );

    ///@}

    /** \name Telemeter Interface
     *
     * @{
     */
    /// Check whether stage or position telemetry is due for recording.
    int checkRecordTimes();

    /// Force a stage telemetry record when requested by the telemeter interface.
    int recordTelem( const telem_stage *type /**< [in] Unused pointer selecting the stage telemetry type. */ );

    /// Force a position telemetry record when requested by the telemeter interface.
    int recordTelem( const telem_position *type /**< [in] Unused pointer selecting the position telemetry type. */ );

    /// Delegate stage telemetry recording to the standard motion-stage interface.
    int recordStage( bool force = false /**< [in] Record even if the stage telemetry is unchanged. */ );

    /// Record the measured position when it changes or recording is forced.
    int recordPosition( bool force = false /**< [in] Record even if the position is unchanged. */ );

    ///@}
};

inline smc100ccCtrl::smc100ccCtrl() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    m_powerMgtEnabled = true;
    m_powerOnWait     = 5; // default to 5 seconds for controller boot up.

    m_defaultPositions = false;

    return;
}

inline int smc100ccCtrl::onPowerOff()
{
    m_powerOnHomed = false;
    recordStage( true );
    recordPosition( true );
    return 0;
}

void smc100ccCtrl::setupConfig()
{
    config.add( "stage.homingOffset",
                "",
                "stage.homingOffset",
                argType::Required,
                "stage",
                "homingOffset",
                false,
                "float",
                "Homing offset, a.k.a. default starting position." );
    config.add( "stage.opDelta",
                "",
                "stage.opDelta",
                argType::Required,
                "stage",
                "opDelta",
                false,
                "float",
                "Threshold move size for switching to OPERATING." );
    config.add( "stage.presetTolerance",
                "",
                "stage.presetTolerance",
                argType::Required,
                "stage",
                "presetTolerance",
                false,
                "float",
                "Absolute position tolerance for assigning a preset, in stage position units. Default is 1e-3; the "
                "difference must be strictly smaller." );

    tty::usbDevice::setupConfig( config );
    dev::ioDevice::setupConfig( config );
    dev::stdMotionStage<smc100ccCtrl>::setupConfig( config );
    dev::telemeter<smc100ccCtrl>::setupConfig( config );
}

void smc100ccCtrl::loadConfig()
{

    config( m_homingOffset, "stage.homingOffset" );

    config( m_opDelta, "stage.opDelta" );
    config( m_presetTolerance, "stage.presetTolerance" );
    this->m_baudRate = B57600; // default for SMC100CC controller.  Will be overridden by any config setting.

    int rv = tty::usbDevice::loadConfig( config );

    if( rv != 0 && rv != TTY_E_NODEVNAMES && rv != TTY_E_DEVNOTFOUND ) // Ignore error if not plugged in
    {
        log<software_error>( { __FILE__, __LINE__, "error loading USB device configs" } );
        log<software_error>( { __FILE__, __LINE__, rv, tty::ttyErrorString( rv ) } );
        m_shutdown = 1;
    }

    rv = dev::ioDevice::loadConfig( config );
    if( rv != 0 )
    {
        log<software_error>( { __FILE__, __LINE__, "error loading io device configs" } );
        m_shutdown = 1;
    }

    rv = dev::stdMotionStage<smc100ccCtrl>::loadConfig( config );
    if( rv != 0 )
    {
        log<software_error>( { __FILE__, __LINE__, "error loading stdMotionStage configs" } );
        m_shutdown = 1;
    }

    rv = dev::telemeter<smc100ccCtrl>::loadConfig( config );
    if( rv != 0 )
    {
        log<software_error>( { __FILE__, __LINE__, "error loading telemeter configs" } );
        m_shutdown = 1;
    }
}

int smc100ccCtrl::appStartup()
{

    REG_INDI_NEWPROP( m_indiP_position, "position", pcf::IndiProperty::Number );
    m_indiP_position.add( pcf::IndiElement( "current" ) );
    m_indiP_position["current"].set( 0 );
    m_indiP_position.add( pcf::IndiElement( "target" ) );

    if( state() == stateCodes::UNINITIALIZED )
    {
        log<text_log>( "In appStartup but in state UNINITIALIZED.", logPrio::LOG_CRITICAL );
        return -1;
    }

    if( m_presetNames.size() != m_presetPositions.size() )
    {
        return log<text_log, -1>( "must set a position for each preset", logPrio::LOG_CRITICAL );
    }

    m_presetNames.insert( m_presetNames.begin(), "none" );
    m_presetPositions.insert( m_presetPositions.begin(), -1 );

    int rv = dev::stdMotionStage<smc100ccCtrl>::appStartup();
    if( rv != 0 )
    {
        log<software_error>( { __FILE__, __LINE__, "error starting up stdMotionStage" } );
        m_shutdown = 1;
    }

    rv = dev::telemeter<smc100ccCtrl>::appStartup();
    if( rv != 0 )
    {
        log<software_error>( { __FILE__, __LINE__, "error starting up telemeter" } );
        m_shutdown = 1;
    }

    return 0;
}

int smc100ccCtrl::appLogic()
{
    if( state() == stateCodes::INITIALIZED )
    {
        log<text_log>( "In appLogic but in state INITIALIZED.", logPrio::LOG_CRITICAL );
        return -1;
    }

    if( stdMotionStage<smc100ccCtrl>::appLogic() < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        return -1;
    }

    if( state() == stateCodes::POWERON )
    {
        if( !powerOnWaitElapsed() )
            return 0;

        if( m_deviceName == "" )
        {
            state( stateCodes::NODEVICE );
        }
        else
        {
            state( stateCodes::NOTCONNECTED );
            if( !stateLogged() )
            {
                std::stringstream logs;
                logs << "USB Device " << m_idVendor << ":" << m_idProduct << ":" << m_serial << " found in udev as "
                     << m_deviceName;
                log<text_log>( logs.str() );
            }
        }
    }

    if( state() == stateCodes::NODEVICE )
    {
#ifdef XWC_SIM_MODE
        m_deviceName = "/dev/simTTY";
        int rv       = TTY_E_NOERROR;
#else
        int rv = tty::usbDevice::getDeviceName();
#endif

        if( rv < 0 && rv != TTY_E_DEVNOTFOUND && rv != TTY_E_NODEVNAMES )
        {
            state( stateCodes::FAILURE );
            if( !stateLogged() )
            {
                log<software_critical>( { __FILE__, __LINE__, rv, tty::ttyErrorString( rv ) } );
            }
            return -1;
        }

        if( rv == TTY_E_DEVNOTFOUND || rv == TTY_E_NODEVNAMES )
        {
            state( stateCodes::NODEVICE );
            if( !stateLogged() )
            {
                std::stringstream logs;
                logs << "USB Device " << m_idVendor << ":" << m_idProduct << ":" << m_serial << " not found in udev";
                log<text_log>( logs.str() );
            }
            return 0;
        }
        else
        {
            state( stateCodes::NOTCONNECTED );
            if( !stateLogged() )
            {
                std::stringstream logs;
                logs << "USB Device " << m_idVendor << ":" << m_idProduct << ":" << m_serial << " found in udev as "
                     << m_deviceName;
                log<text_log>( logs.str() );
            }
        }
    }

    if( state() == stateCodes::NOTCONNECTED )
    {
        int rv;
#ifdef XWC_SIM_MODE
        rv = 0;
#else
        { // scope for elPriv
            elevatedPrivileges elPriv( this );
            rv = connect();
        }
#endif

        if( rv < 0 )
        {
            int nrv = tty::usbDevice::getDeviceName();
            if( nrv < 0 && nrv != TTY_E_DEVNOTFOUND && nrv != TTY_E_NODEVNAMES )
            {
                state( stateCodes::FAILURE );
                if( !stateLogged() )
                    log<software_critical>( { __FILE__, __LINE__, nrv, tty::ttyErrorString( nrv ) } );
                return -1;
            }

            if( nrv == TTY_E_DEVNOTFOUND || nrv == TTY_E_NODEVNAMES )
            {
                state( stateCodes::NODEVICE );

                if( !stateLogged() )
                {
                    std::stringstream logs;
                    logs << "USB Device " << m_idVendor << ":" << m_idProduct << ":" << m_serial
                         << " no longer found in udev";
                    log<text_log>( logs.str() );
                }
                return 0;
            }

            // if connect failed, and there is a device, then we have some other problem.
            state( stateCodes::FAILURE );
            if( !stateLogged() )
                log<software_error>( { __FILE__, __LINE__, rv, tty::ttyErrorString( rv ) } );
            return -1;
        }

        if( testConnection() == 0 )
        {
            state( stateCodes::CONNECTED );
        }
        else
        {
            std::string errorString;
            if( getLastError( errorString ) != 0 )
            {
                log<software_error>( { __FILE__, __LINE__, errorString } );
            }

            return 0;
        }

        if( state() == stateCodes::CONNECTED && !stateLogged() )
        {
            std::stringstream logs;
            logs << "Connected to stage(s) on " << m_deviceName;
            log<text_log>( logs.str() );
        }
    }

    if( state() == stateCodes::HOMING || state() == stateCodes::READY || state() == stateCodes::OPERATING )
    {
        if( telemeter<smc100ccCtrl>::appLogic() < 0 )
        {
            log<software_error>( { __FILE__, __LINE__ } );
            return -1;
        }
    }

    // If we're here, we can get state from controller...
    std::string axState;
    { // mutex scope
        std::unique_lock<std::mutex> lock( m_indiMutex );
        if( getCtrlState( axState ) < 0 )
        {
            if( m_powerTargetState == 0 )
                return 0;
            return log<software_error, 0>( { __FILE__, __LINE__ } );
        }
    }

    if( axState[0] == '0' )
    {
        state( stateCodes::NOTHOMED ); // This always means this.
    }
    else if( axState[0] == '1' && axState[1] == '0' )
    {
        // Need to download stage info
        log<text_log>( "getting stage information" );
        std::string com;
        if( makeCom( com, "PW1" ) < 0 )
        {
            log<software_error>( { __FILE__, __LINE__, "Error making command PW1" } );
            return 0;
        }

        int rv = MagAOX::tty::ttyWrite( com, m_fileDescrip, m_writeTimeout );
        if( rv != TTY_E_NOERROR )
        {
            if( m_powerTargetState == 0 )
                return 0;
            log<software_error>( { __FILE__, __LINE__, MagAOX::tty::ttyErrorString( rv ) } );
            return -1;
        }

        sleep( 5 );
        if( makeCom( com, "ZX2" ) < 0 )
        {
            log<software_error>( { __FILE__, __LINE__, "Error making command ZX2" } );
            return 0;
        }

        rv = MagAOX::tty::ttyWrite( com, m_fileDescrip, m_writeTimeout );
        if( rv != TTY_E_NOERROR )
        {
            if( m_powerTargetState == 0 )
                return 0;
            log<software_error>( { __FILE__, __LINE__, MagAOX::tty::ttyErrorString( rv ) } );
            return -1;
        }

        sleep( 5 );
        if( makeCom( com, "PW0" ) < 0 )
        {
            log<software_error>( { __FILE__, __LINE__, "Error making command PW0" } );
            return 0;
        }

        rv = MagAOX::tty::ttyWrite( com, m_fileDescrip, m_writeTimeout );
        if( rv != TTY_E_NOERROR )
        {
            if( m_powerTargetState == 0 )
                return 0;
            log<software_error>( { __FILE__, __LINE__, MagAOX::tty::ttyErrorString( rv ) } );
            return -1;
        }

        sleep( 5 );
        log<text_log>( "stage information loaded" );
        return 0;
    }
    else if( axState[0] == '1' && ( axState[1] == 'E' || axState[1] == 'F' ) )
    {
        state( stateCodes::HOMING );
        m_moving    = 1;
        m_wasHoming = 1;
    }
    else if( axState[0] == '2' )
    {
        m_moving = 1;
        if( m_moveOp )
        {
            state( stateCodes::OPERATING );
        }
    }
    else if( axState[0] == '3' && isdigit( axState[1] ) )
    {
        if( m_wasHoming )
        {
            std::unique_lock<std::mutex> lock( m_indiMutex );
            moveTo( m_homingOffset );
            m_wasHoming = 0;
        }
        else
        {
            m_moving = 0;
            state( stateCodes::READY );
            m_moveOp = true;
        }
    }
    else if( axState[0] == '3' )
    {
        log<text_log>( "Stage disabled.  Enabling" );
        std::string com;
        if( makeCom( com, "MM1" ) < 0 )
        {
            log<software_error>( { __FILE__, __LINE__, "Error making command PW1" } );
            return 0;
        }
        int rv = MagAOX::tty::ttyWrite( com, m_fileDescrip, m_writeTimeout );
        if( rv != TTY_E_NOERROR )
        {
            if( m_powerTargetState == 0 )
                return 0;
            log<software_error>( { __FILE__, __LINE__, MagAOX::tty::ttyErrorString( rv ) } );
            return -1;
        }
    }
    else if( axState[0] == '\0' && axState[1] == '\0' )
    {
        // a non-response, this means we should go back around and ask again
        // this doesn't seem to be an error, but isn't documented.
        // Occurs after a power off, but also sometimes after homing completes.
        return 0;
    }
    else
    {
        sleep( 1 );
        if( m_powerState == 0 )
            return 0;

        log<software_error>(
            { __FILE__,
              __LINE__,
              "Invalid state: |" + std::to_string( axState[0] ) + "|" + std::to_string( axState[1] ) + "|" } );
        state( stateCodes::ERROR );
    }

    if( state() == stateCodes::NOTHOMED )
    {
        if( m_powerOnHome && !m_powerOnHomed )
        {
            std::unique_lock<std::mutex> lock( m_indiMutex );
            startHoming();
            m_powerOnHomed = true;
        }
        return 0;
    }

    if( state() == stateCodes::READY || state() == stateCodes::OPERATING )
    {
        std::unique_lock<std::mutex> lock( m_indiMutex );

        int rv = getPosition( m_position );

        if( rv < 0 )
        {
            sleep( 1 );
            if( m_powerState == 0 )
                return 0;

            std::string errorString;

            if( getLastError( errorString ) != 0 && errorString.size() != 0 )
            {
                log<software_error>( { __FILE__, __LINE__, errorString } );
            }

            log<software_error>(
                { __FILE__, __LINE__, "There's been an error with getting current controller position." } );
        }

        recordPosition();

        updateIfChanged( m_indiP_position, "current", m_position );

        static int last_moving = -1;

        bool changed = false;
        if( last_moving != m_moving )
        {
            changed     = true;
            last_moving = m_moving;
        }

        if( changed )
        {
            if( m_moving )
            {
                m_indiP_position.setState( INDI_BUSY );
            }
            else
            {
                m_indiP_position.setState( INDI_IDLE );
                m_indiP_position["target"] = m_position;
            }
            m_indiDriver->sendSetProperty( m_indiP_position );
        }

        int n = presetNumber();
        if( n == -1 )
        {
            m_preset        = 0;
            m_preset_target = 0;
        }
        else
        {
            m_preset        = n;
            m_preset_target = n;
        }

        dev::stdMotionStage<smc100ccCtrl>::updateINDI();
        recordStage();

        return 0;
    }

    if( state() == stateCodes::ERROR )
    {
        if( m_powerTargetState == 0 )
            return 0;
        sleep( 1 );
        if( m_powerState == 0 )
            return 0;

        int rv = tty::usbDevice::getDeviceName();
        if( rv < 0 && rv != TTY_E_DEVNOTFOUND && rv != TTY_E_NODEVNAMES )
        {
            state( stateCodes::FAILURE );
            if( !stateLogged() )
            {
                log<software_critical>( { __FILE__, __LINE__, rv, tty::ttyErrorString( rv ) } );
            }
            return rv;
        }

        if( rv == TTY_E_DEVNOTFOUND || rv == TTY_E_NODEVNAMES )
        {
            state( stateCodes::NODEVICE );

            if( !stateLogged() )
            {
                std::stringstream logs;
                logs << "USB Device " << m_idVendor << ":" << m_idProduct << ":" << m_serial << " not found in udev";
                log<text_log>( logs.str() );
            }
            return 0;
        }

        sleep( 1 );
        if( m_powerState == 0 )
            return 0;

        state( stateCodes::FAILURE );
        if( !stateLogged() )
        {
            log<text_log>( "Error NOT due to loss of USB connection.  I can't fix it myself.", logPrio::LOG_CRITICAL );
        }
        return -1;
    }

    return 0;
}

int smc100ccCtrl::testConnection()
{
#ifdef XWC_SIM_MODE
    return 0;
#endif

    std::string buffer{ "1TS\r\n" };
    std::string output;

    int rv = MagAOX::tty::ttyWriteRead( output, buffer, "\r\n", false, m_fileDescrip, m_writeTimeout, m_readTimeout );

    if( rv != TTY_E_NOERROR )
    {
        if( m_powerTargetState == 0 )
            return -1;
        log<software_error>( { __FILE__, __LINE__, MagAOX::tty::ttyErrorString( rv ) } );
        return -1;
    }

    /*int axis;
    std::string com;
    std::string val;

    splitResponse( axis, com, val, output);*/

    return 0;
}

int smc100ccCtrl::appShutdown()
{
    return 0;
}

int smc100ccCtrl::makeCom( std::string &str, const std::string &com )
{
    char tmp[10];

    int axis = 1;

    snprintf( tmp, 10, "%i", axis );

    str = tmp;

    str += com;

    str += "\r\n";

    return 0;
}
int smc100ccCtrl::splitResponse( int &axis, std::string &com, std::string &val, std::string &resp )
{
    if( resp.length() < 3 )
    {
        log<software_error>( { __FILE__, __LINE__, "Invalid response" } );
        return -1;
    }

    if( isalpha( resp[0] ) )
    {
        log<software_error>( { __FILE__, __LINE__, "Invalid response" } );
        axis = 0;
        com  = "";
        val  = resp;
        return 0;
    }

    if( isalpha( resp[1] ) )
    {
        axis = resp[0] - '0';
    }
    else
    {
        axis = atoi( resp.substr( 0, 2 ).c_str() );
    }

    if( axis < 10 )
    {

        com = resp.substr( 1, 2 );
        if( resp.length() < 4 )
            val = "";
        else
            val = resp.substr( 3, resp.length() - 3 );
        if( val.size() > 1 )
        {
            while( val[val.size() - 1] == '\r' || val[val.size() - 1] == '\n' )
            {
                val.erase( val.size() - 1 );
                if( val.size() < 1 )
                    break;
            }
        }
    }
    else
    {
        if( resp.length() < 4 )
        {
            log<software_error>( { __FILE__, __LINE__, "Invalid response" } );
            com = "";
            val = "";
            return -1;
        }
        com = resp.substr( 2, 2 );
        if( resp.length() < 5 )
            val = "";
        else
            val = resp.substr( 4, resp.length() - 4 );

        if( val.size() > 1 )
        {
            while( val[val.size() - 1] == '\r' || val[val.size() - 1] == '\n' )
            {
                val.erase( val.size() - 1 );
                if( val.size() < 1 )
                    break;
            }
        }
    }

    return 0;
}

int smc100ccCtrl::getCtrlState( std::string &state )
{
    std::string com, resp;

    if( makeCom( com, "TS" ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__, "Error making command TS" } );
        return 0;
    }

    int rv = MagAOX::tty::ttyWriteRead( resp, com, "\r\n", false, m_fileDescrip, m_readTimeout, m_writeTimeout );
    if( rv != TTY_E_NOERROR )
    {
        if( m_powerTargetState == 0 )
            return -1;
        log<software_error>( { __FILE__, __LINE__, MagAOX::tty::ttyErrorString( rv ) } );
        return -1;
    }

    // std::cerr << "TS Response: " << resp << "\n";
    int         raxis;
    std::string rcom, rval;

    splitResponse( raxis, rcom, rval, resp );

    if( rcom == "" )
    {
        log<software_error>( { __FILE__, __LINE__, "An Error occurred" } );
        return -1;
    }

    if( raxis != 1 )
    {
        log<software_error>( { __FILE__, __LINE__, "Wrong axis returned" } );
        return -1;
    }

    if( rcom != "TS" )
    {
        log<software_error>( { __FILE__, __LINE__, "Wrong command returned" } );
        return -1;
    }

    if( rval.length() != 6 )
    {
        log<software_error>( { __FILE__, __LINE__, "Incorrect response length" } );
        return -1;
    }

    state = rval.substr( 4, 2 );

    return 0;
}

int smc100ccCtrl::getPosition( double &current )
{
    std::string buffer{ "1TP\r\n" };
    std::string output;
    int rv = MagAOX::tty::ttyWriteRead( output, buffer, "\r\n", false, m_fileDescrip, m_writeTimeout, m_readTimeout );

    if( rv != TTY_E_NOERROR )
    {
        if( m_powerTargetState == 0 )
            return -1;
        log<software_error>( { __FILE__, __LINE__, MagAOX::tty::ttyErrorString( rv ) } );
        return -1;
    }

    // Parse current and place into argument
    try
    {
        current = std::stod( output.substr( 3 ) );
    }
    catch( ... )
    {
        log<software_error>( { __FILE__, __LINE__, "Error occured: Unexpected output in getPosition()" } );
        return -1;
    }
    return 0;
}

int smc100ccCtrl::getLastError( std::string &errorString )
{
    std::string buffer{ "1TE\r\n" };
    std::string output;
    int rv = MagAOX::tty::ttyWriteRead( output, buffer, "\r\n", false, m_fileDescrip, m_writeTimeout, m_readTimeout );

    if( rv != TTY_E_NOERROR )
    {
        if( m_powerTargetState == 0 )
            return -1;
        log<software_error>( { __FILE__, __LINE__ } );
        std::cerr << __FILE__ << " " << __LINE__ << " " << rv << "\n";

        errorString = MagAOX::tty::ttyErrorString( rv );
        return -1;
    }

    char status;
    try
    {
        status = output.at( 3 );
    }
    catch( const std::out_of_range &oor )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        errorString = "Unknown output; controller not responding correctly.";
        return -1;
    }

    if( status == '@' )
    {
        return 0;
    }
    else
    {
        switch( status )
        {
        case 'A':
            errorString = "Unknown message code or floating point controller address.";
            break;
        case 'B':
            errorString = "Controller address not correct.";
            break;
        case 'C':
            errorString = "Parameter missing or out of range.";
            break;
        case 'D':
            errorString = "Command not allowed.";
            break;
        case 'E':
            errorString = "Home sequence already started.";
            break;
        case 'F':
            errorString = "ESP stage name unknown.";
            break;
        case 'G':
            errorString = "Displacement out of limits.";
            break;
        case 'H':
            errorString = "Command not allowed in NOT REFERENCED state.";
            break;
        case 'I':
            errorString = "Command not allowed in CONFIGURATION state.";
            break;
        case 'J':
            errorString = "Command not allowed in DISABLE state.";
            break;
        case 'K':
            errorString = "Command not allowed in READY state.";
            break;
        case 'L':
            errorString = "Command not allowed in HOMING state.";
            break;
        case 'M':
            errorString = "UCommand not allowed in MOVING state.";
            break;
        case 'N':
            errorString = "Current position out of software limit.";
            break;
        case 'S':
            errorString = "Communication Time Out.";
            break;
        case 'U':
            errorString = "Error during EEPROM access.";
            break;
        case 'V':
            errorString = "Error during command execution.";
            break;
        case 'W':
            errorString = "Command not allowed for PP version.";
            break;
        case 'X':
            errorString = "Command not allowed for CC version.";
            break;
        default:
            errorString = "unknown status";
            std::cerr << "unkown status: " << status << "\n";
        }

        log<software_error>( { __FILE__, __LINE__ } );
        return -1;
    }
}

INDI_NEWCALLBACK_DEFN( smc100ccCtrl, m_indiP_position )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_position, ipRecv );

    if( !( state() == stateCodes::READY || state() == stateCodes::OPERATING ) )
    {
        log<text_log>( "can not command position in current state" );
        return 0;
    }

    float current = -1e55, target = -1e55;

    try
    {
        current = ipRecv["current"].get<float>();
    }
    catch( ... )
    {
    }

    try
    {
        target = ipRecv["target"].get<float>();
    }
    catch( ... )
    {
    }

    if( target == -1e55 )
        target = current;

    if( target == -1e55 )
        return 0;

    // Lock the mutex, waiting if necessary
    std::unique_lock<std::mutex> lock( m_indiMutex );

    updateIfChanged( m_indiP_position, "target", target, INDI_BUSY );
    m_target = target;

    return moveTo( target );
}

int smc100ccCtrl::stop()
{
    recordStage( true );
    recordPosition( true );

    // don't lock mutex -- this must go through

    std::string buffer{ "1ST\r\n" };
    int         rv = MagAOX::tty::ttyWrite( buffer, m_fileDescrip, m_writeTimeout );

    updateSwitchIfChanged( m_indiP_stop, "request", pcf::IndiElement::Off, INDI_IDLE );

    if( rv != TTY_E_NOERROR )
    {
        if( m_powerTargetState == 0 )
            return -1;
        log<software_error>( { __FILE__, __LINE__, MagAOX::tty::ttyErrorString( rv ) } );
        return -1;
    }
    return 0;
}

int smc100ccCtrl::startHoming()
{
    updateSwitchIfChanged( m_indiP_home, "request", pcf::IndiElement::Off, INDI_IDLE );

    recordStage( true );
    recordPosition( true );

    std::string buffer{ "1OR\r\n" };
    int         rv = MagAOX::tty::ttyWrite( buffer, m_fileDescrip, m_writeTimeout );

    if( rv != TTY_E_NOERROR )
    {
        if( m_powerTargetState == 0 )
            return -1;
        log<software_error>( { __FILE__, __LINE__, MagAOX::tty::ttyErrorString( rv ) } );
        return -1;
    }
    return 0;
}

double smc100ccCtrl::presetNumber()
{
    for( size_t n = 1; n < m_presetPositions.size(); ++n )
    {
        if( fabs( m_position - m_presetPositions[n] ) < m_presetTolerance )
            return n;
    }

    return 0;
}

int smc100ccCtrl::moveTo( double position )
{
    recordStage( true );
    recordPosition( true );

    if( fabs( position - m_position ) > m_opDelta )
    {
        m_moveOp = true;
    }
    else
    {
        m_moveOp = false;
    }

    std::string buffer{ "1PA" };
    buffer = buffer + std::to_string( position ) + "\r\n";

    int rv = MagAOX::tty::ttyWrite( buffer, m_fileDescrip, m_writeTimeout );

    if( rv != TTY_E_NOERROR )
    {
        if( m_powerTargetState == 0 )
            return -1;
        log<software_error>( { __FILE__, __LINE__, MagAOX::tty::ttyErrorString( rv ) } );
        return -1;
    }

    std::string errorString;
    if( getLastError( errorString ) == 0 )
    {
        if( m_moveOp )
        {
            state( stateCodes::OPERATING );
        }
        updateIfChanged( m_indiP_position, "target", position );
        return 0;
    }
    else
    {
        log<software_error>( { __FILE__, __LINE__, errorString } );
        return -1;
    }
}

int smc100ccCtrl::checkRecordTimes()
{
    return telemeter<smc100ccCtrl>::checkRecordTimes( telem_stage(), telem_position() );
}

int smc100ccCtrl::recordTelem( const telem_stage * )
{
    return recordStage( true );
}

int smc100ccCtrl::recordTelem( const telem_position * )
{
    return recordPosition( true );
}

int smc100ccCtrl::recordStage( bool force )
{
    return stdMotionStage<smc100ccCtrl>::recordStage( force );
}

int smc100ccCtrl::recordPosition( bool force )
{
    static float last_position = 1e30;

    float fpos = m_position;

    if( fpos != last_position || force )
    {
        telem<telem_position>( fpos );
        last_position = fpos;
    }

    return 0;
}

} // namespace app
} // namespace MagAOX

#endif // smc100ccCtrl_hpp
