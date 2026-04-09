/** \file elliptecCtrl.hpp
 * \brief The MagAO-X Elliptec stage controller header.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup elliptecCtrl_files
 */

#ifndef elliptecCtrl_hpp
#define elliptecCtrl_hpp

#include "../../libMagAOX/libMagAOX.hpp"
#include "../../magaox_git_version.h"

/** \defgroup elliptecCtrl
 * \brief The MagAO-X application to control a Thorlabs Elliptec rotation stage.
 *
 * \ingroup apps
 */

/** \defgroup elliptecCtrl_files
 * \ingroup elliptecCtrl
 */

namespace MagAOX
{
namespace app
{

/// The MagAO-X Elliptec stage controller.
/** Uses the Elliptec serial protocol while exposing the standard MagAO-X motion-stage interface.
 *
 * Elliptec protocol reference:
 * - TX: `<ADDR><cmd>[args]` with no CRLF
 * - RX: a single CRLF-terminated line
 * - commands used here: `in`, `gs`, `gp`, `hoX`, `st`, `us`, `om`, `svHH`, `maXXXXXXXX`, `mrXXXXXXXX`
 *
 * \ingroup elliptecCtrl
 */
class elliptecCtrl : public MagAOXApp<>, public dev::stdMotionStage<elliptecCtrl>, public dev::telemeter<elliptecCtrl>
{
    friend class dev::stdMotionStage<elliptecCtrl>;

    friend class dev::telemeter<elliptecCtrl>;

  public:
    /// The telemeter base type.
    typedef dev::telemeter<elliptecCtrl> telemeterT;

    /// Construct the controller.
    elliptecCtrl();

    /// Destroy the controller.
    ~elliptecCtrl() noexcept
    {
    }

    /// Set up the application configuration.
    virtual void setupConfig() override;

    /// Implementation of configuration loading, separated for helper-macro use.
    int loadConfigImpl( mx::app::appConfigurator &_config /**< [in] application configuration from which to load */ );

    /// Load the application configuration.
    virtual void loadConfig() override;

    /// Start the application.
    virtual int appStartup() override;

    /// Execute one application loop iteration.
    virtual int appLogic() override;

    /// Shut the application down.
    virtual int appShutdown() override;

    /// Handle a transition to powered-off operation.
    virtual int onPowerOff() override;

    /// Handle continued operation while powered off.
    virtual int whilePowerOff() override;

    /// Stop the current motion request.
    int stop();

    /// Begin a homing sequence.
    int startHoming();

    /// Resolve the active preset index for the current stage angle.
    float presetNumber();

    /// Move to a preset target supplied by `stdMotionStage`.
    int moveTo( float target /**< [in] preset number or preset position supplied by the base helper */ );

    /// Check whether staged telemetry records need to be forced.
    int checkRecordTimes();

    /// Record stage-state telemetry on a telemeter request.
    int recordTelem( const telem_stage * /**< [in] telemetry tag used for overload resolution */ );

    /// Record native position telemetry on a telemeter request.
    int recordTelem( const telem_position * /**< [in] telemetry tag used for overload resolution */ );

    /// Record the standard stage-state telemetry.
    int recordStage( bool force = false /**< [in] force the stage-state telemetry record */ );

    /// Record the native position telemetry.
    int recordPosition( bool force = false /**< [in] force the position telemetry record */ );

    /// Handle a new absolute-angle request.
    INDI_NEWCALLBACK_DECL( elliptecCtrl, m_ipAbsDeg );

    /// Handle a new relative-step update.
    INDI_NEWCALLBACK_DECL( elliptecCtrl, m_ipRelDeg );

    /// Handle a new relative-move request.
    INDI_NEWCALLBACK_DECL( elliptecCtrl, m_ipRelMove );

    /// Handle a new velocity request.
    INDI_NEWCALLBACK_DECL( elliptecCtrl, m_ipVelPct );

    /// Handle a new optimize request.
    INDI_NEWCALLBACK_DECL( elliptecCtrl, m_ipOptimize );

    /// Handle a new save request.
    INDI_NEWCALLBACK_DECL( elliptecCtrl, m_ipSave );

  protected:
    /** \name Configurable Parameters - Data
     *
     * @{
     */

    /// Serial device path used to reach the Elliptec controller.
    std::string m_port;

    /// Serial baud rate.
    int m_baud{ 9600 };

    /// Delay after power-on before the first connection attempt.
    unsigned m_startupDelayMs{ 200 };

    /// Elliptec device address nibble encoded as ASCII hex.
    char m_addr{ '0' };

    /// Default read timeout for serial transactions in milliseconds.
    int m_readTimeoutMs{ 3000 };

    /// Read timeout while a command is pending or the device reports BUSY.
    int m_busyReadTimeoutMs{ 6000 };

    /// Optional post-write delay in milliseconds.
    int m_postWriteSleepMs{ 0 };

    /// Velocity percentage command value.
    int m_velPercent{ 40 };

    /// Relative move to apply after homing completes.
    double m_homeOffsetDeg{ 0.0 };

    /// Whether absolute moves may span two turns instead of wrapping into one turn.
    bool m_allowMultiturn{ false };

    /// Pulses per 360-degree revolution, from configuration or device query.
    uint32_t m_pulsesPerRev{ 0 };

    /// Device command used to run the optimize routine.
    std::string m_cmdOptimize{ "om" };

    /// Device command used to save parameters.
    std::string m_cmdSave{ "us" };

    /// Consecutive soft poll misses tolerated before forcing a reconnect.
    int m_commMaxMisses{ 3 };

    ///@}

    /** \name Runtime State - Data
     *
     * @{
     */

    /// Open file descriptor for the serial port, or `-1` when closed.
    int m_fd{ -1 };

    /// Whether the serial link is currently open and usable.
    bool m_connected{ false };

    /// Whether the stage was last seen in a powered-on state.
    bool m_wasPowered{ false };

    /// Last known absolute position in degrees.
    double m_posDeg{ 0.0 };

    /// Last known absolute position in native pulses.
    int32_t m_posPulses{ 0 };

    /// Whether this process has completed a homing sequence since power-up.
    bool m_homed{ false };

    /// Last status byte returned by the `gs` query.
    uint8_t m_gs{ 0x00 };

    /// Relative step size in degrees used by the `relMove` request.
    double m_relStepDeg{ 1.0 };

    /// Pending Elliptec command type tracked until the next successful poll resolves it.
    enum class Pending
    {
        None,
        MoveAbs,
        MoveRel,
        Home,
        OffsetRel,
        Optimize,
        Stop,
        Velocity,
        Save
    };

    /// Command currently in flight against the controller.
    Pending m_pending{ Pending::None };

    /// Operator-facing status text supplement for the current in-flight action.
    std::string m_statusHint;

    /// Count of consecutive soft communication misses.
    int m_commMisses{ 0 };

    ///@}

    /** \name INDI Properties - Data
     *
     * @{
     */

    /// Absolute-angle property in degrees.
    pcf::IndiProperty m_ipAbsDeg;

    /// Relative-step property in degrees.
    pcf::IndiProperty m_ipRelDeg;

    /// Momentary request property that applies `m_ipRelDeg`.
    pcf::IndiProperty m_ipRelMove;

    /// Velocity percentage property.
    pcf::IndiProperty m_ipVelPct;

    /// Read-only text status property.
    pcf::IndiProperty m_ipStatus;

    /// Momentary optimize request property.
    pcf::IndiProperty m_ipOptimize;

    /// Momentary save request property.
    pcf::IndiProperty m_ipSave;

    /// Read-only text property listing configured preset names and angles.
    pcf::IndiProperty m_ipStageNamePos;

    ///@}

    /** \name Serial Helpers
     *
     * @{
     */

    /// Convert an integer baud rate into the corresponding termios constant.
    static speed_t to_termios_baud_( int baud /**< [in] baud rate to convert */ );

    /// Open and configure the serial port.
    int openPort_();

    /// Close the serial port if it is open.
    void closePort_();

    /// Drain unread input bytes from the serial port.
    int drainInput_();

    /// Write a full command frame to the serial port.
    int writeAll_( const std::string &s /**< [in] bytes to write */ );

    /// Read a CRLF-terminated frame from the serial port.
    int readFrame_( std::string &out,       /**< [out] received frame including CRLF */
                    int          timeout_ms /**< [in] timeout in milliseconds */
    );

    /// Construct a device frame from the command payload.
    std::string frame_( const std::string &cmd /**< [in] command payload without address */ ) const;

    /// Transmit a command and optionally receive its reply.
    int txrx_( const std::string &cmd,       /**< [in] command payload without address */
               std::string       *reply,     /**< [out] reply string, or null for fire-and-forget */
               int                timeout_ms /**< [in] default timeout in milliseconds */
    );

    ///@}

    /** \name Protocol Access
     *
     * @{
     */

    /// Query static device information.
    int q_info_();

    /// Query the device status byte.
    int q_status_();

    /// Query the current device position.
    int q_position_();

    /// Issue a home command.
    int cmd_home_( uint8_t dir = 0 /**< [in] Elliptec home-direction nibble */ );

    /// Issue a stop command.
    int cmd_stop_();

    /// Issue the optimize command and wait for its first response.
    int cmd_optimize_wait_();

    /// Issue the save command and wait for its first response.
    int cmd_save_();

    /// Set the device velocity percentage.
    int cmd_setvel_( int pct /**< [in] velocity percentage from 0 to 100 */ );

    /// Command an absolute move in native pulses.
    int cmd_moveAbs_pulses_( int32_t pulses /**< [in] absolute target in native pulses */ );

    /// Command a relative move in native pulses.
    int cmd_moveRel_pulses_( int32_t pulses /**< [in] relative delta in native pulses */ );

    ///@}

    /** \name Motion Helpers
     *
     * @{
     */

    /// Move to an absolute angle in degrees.
    int moveAbsDeg_( double deg /**< [in] absolute stage angle in degrees */ );

    /// Execute the current relative move request.
    int moveRelDegCmdFromRelMove_();

    /// Move by a relative angular offset in degrees.
    int moveRelDeg_( double ddeg /**< [in] relative angular offset in degrees */ );

    /// Start an absolute move in degrees after the caller has updated target bookkeeping.
    int startMoveToDeg_( double deg /**< [in] absolute stage angle in degrees */ );

    /// Convert degrees into native pulses.
    int32_t degToPulses_( double deg /**< [in] angle in degrees */ ) const;

    /// Convert native pulses into degrees.
    double pulsesToDeg_( int32_t pulses /**< [in] position in native pulses */ ) const;

    /// Resolve the preset index matching the supplied angle.
    int presetIndexForPosition_( double deg /**< [in] angle to compare against preset positions */ ) const;

    /// Return the matching tolerance used for preset comparisons.
    double presetToleranceDeg_() const;

    /// Update `m_preset` and `m_preset_target` from the current motion state.
    void syncPresetState_();

    /// Update the high-level FSM state from the current connection and motion state.
    void syncControllerState_();

    /// Poll the device and resolve the current pending command.
    int pollDevice_();

    /// Push the current operator-facing status properties.
    void updateStatus_();

    /// Build the read-only preset-name mapping string.
    std::string buildStageNamePosText_() const;

    ///@}
};

/* ========================= impl ========================= */

inline elliptecCtrl::elliptecCtrl() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    m_presetNotation   = "preset";
    m_powerMgtEnabled  = true;
    m_defaultPositions = false;
    m_moving           = -1;
}

/* ---------- Config ---------- */

inline void elliptecCtrl::setupConfig()
{
    // Serial
    config.add(
        "stage.port", "", "stage.port", argType::Required, "stage", "port", false, "string", "Serial device path" );

    config.add( "stage.baud", "9600", "stage.baud", argType::Required, "stage", "baud", false, "int", "Baud rate" );

    config.add( "stage.address",
                "0",
                "stage.address",
                argType::Optional,
                "stage",
                "address",
                false,
                "string",
                "Elliptec address nibble (0-F)" );

    config.add( "serial.readTimeoutMs",
                "3000",
                "serial.readTimeoutMs",
                argType::Optional,
                "serial",
                "readTimeoutMs",
                false,
                "int",
                "Read timeout (ms)" );

    config.add( "serial.busyReadTimeoutMs",
                "6000",
                "serial.busyReadTimeoutMs",
                argType::Optional,
                "serial",
                "busyReadTimeoutMs",
                false,
                "int",
                "Read timeout (ms) while device is BUSY/pending" );
    config.add( "serial.postWriteSleepMs",
                "0",
                "serial.postWriteSleepMs",
                argType::Optional,
                "serial",
                "postWriteSleepMs",
                false,
                "int",
                "Sleep after write (ms)" );

    // Behavior
    config.add( "stage.homeOffset",
                "0.0",
                "stage.homeOffset",
                argType::Optional,
                "stage",
                "homeOffset",
                false,
                "double",
                "Relative offset (deg) to move after homing" );
    config.add( "stage.allowMultiturn",
                "false",
                "stage.allowMultiturn",
                argType::Optional,
                "stage",
                "allowMultiturn",
                false,
                "bool",
                "If false, wrap abs to [0,720)" );

    // Conversion
    config.add( "stage.pulsesPerRev",
                "0",
                "stage.pulsesPerRev",
                argType::Optional,
                "stage",
                "pulsesPerRev",
                false,
                "uint",
                "Override pulses per 360deg (0=auto)" );

    // Motion UI
    config.add( "motion.velPercent",
                "40",
                "motion.velPercent",
                argType::Optional,
                "motion",
                "velPercent",
                false,
                "int",
                "Velocity percent 0..100" );

    // Command aliases
    config.add( "device.optimizeCmd",
                "om",
                "device.optimizeCmd",
                argType::Optional,
                "device",
                "optimizeCmd",
                false,
                "string",
                "Optimize command" );
    config.add( "device.saveCmd",
                "us",
                "device.saveCmd",
                argType::Optional,
                "device",
                "saveCmd",
                false,
                "string",
                "Save command" );

    // Comms robustness
    config.add( "comm.maxPollMisses",
                "3",
                "comm.maxPollMisses",
                argType::Optional,
                "comm",
                "maxPollMisses",
                false,
                "int",
                "Consecutive poll timeouts tolerated before reconnect" );

    if( dev::stdMotionStage<elliptecCtrl>::setupConfig( config ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__, "Error from stdMotionStage::setupConfig" } );
        m_shutdown = true;
    }

    TELEMETER_SETUP_CONFIG( config );
}

inline int elliptecCtrl::loadConfigImpl( mx::app::appConfigurator &_config )
{
    _config( m_port, "stage.port" );
    _config( m_baud, "stage.baud" );

    std::string a;
    _config( a, "stage.address" );
    if( !a.empty() )
    {
        char c = (char)std::toupper( (unsigned char)a[0] );
        if( std::isxdigit( (unsigned char)c ) )
        {
            m_addr = c;
        }
    }

    _config( m_readTimeoutMs, "serial.readTimeoutMs" );
    _config( m_busyReadTimeoutMs, "serial.busyReadTimeoutMs" );
    if( m_busyReadTimeoutMs < m_readTimeoutMs )
    {
        m_busyReadTimeoutMs = m_readTimeoutMs;
    }
    _config( m_postWriteSleepMs, "serial.postWriteSleepMs" );

    _config( m_homeOffsetDeg, "stage.homeOffset" );
    _config( m_allowMultiturn, "stage.allowMultiturn" );

    uint32_t ppr = 0;
    _config( ppr, "stage.pulsesPerRev" );
    if( ppr )
    {
        m_pulsesPerRev = ppr;
    }

    _config( m_velPercent, "motion.velPercent" );
    if( m_velPercent < 0 )
    {
        m_velPercent = 0;
    }
    if( m_velPercent > 100 )
    {
        m_velPercent = 100;
    }

    _config( m_cmdOptimize, "device.optimizeCmd" );
    _config( m_cmdSave, "device.saveCmd" );

    _config( m_commMaxMisses, "comm.maxPollMisses" );
    if( m_commMaxMisses < 1 )
    {
        m_commMaxMisses = 1;
    }

    if( dev::stdMotionStage<elliptecCtrl>::loadConfig( _config ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "Error from stdMotionStage::loadConfig" } );
    }

    TELEMETER_LOAD_CONFIG( _config );

    return 0;
}

inline void elliptecCtrl::loadConfig()
{
    if( loadConfigImpl( config ) < 0 )
    {
        m_shutdown = true;
    }
}

/* ---------- Startup/Logic ---------- */

inline int elliptecCtrl::appStartup()
{
    if( state() == stateCodes::UNINITIALIZED )
    {
        return log<text_log, -1>( "UNINITIALIZED in appStartup", logPrio::LOG_CRITICAL );
    }

    // absDeg
    CREATE_REG_INDI_NEW_NUMBERD( m_ipAbsDeg, "absDeg", 0.0, 720.0, 0.001, "", "Absolute (deg)", "rotation" );
    indi::updateIfChanged( m_ipAbsDeg, "current", m_posDeg, m_indiDriver, INDI_IDLE );

    // relDeg (step holder)
    CREATE_REG_INDI_NEW_NUMBERD( m_ipRelDeg, "relDeg", -720.0, 720.0, 0.001, "", "Relative step (deg)", "rotation" );
    indi::updateIfChanged( m_ipRelDeg, "current", m_relStepDeg, m_indiDriver, INDI_IDLE );
    indi::updateIfChanged( m_ipRelDeg, "target", m_relStepDeg, m_indiDriver, INDI_IDLE );

    // relMove: request switch
    CREATE_REG_INDI_NEW_REQUESTSWITCH( m_ipRelMove, "relMove" );

    // velocity
    CREATE_REG_INDI_NEW_NUMBERI( m_ipVelPct, "velocity", 0, 100, 1, "", "Velocity (%)", "rotation" );
    indi::updateIfChanged( m_ipVelPct, "current", m_velPercent, m_indiDriver, INDI_IDLE );
    indi::updateIfChanged( m_ipVelPct, "target", m_velPercent, m_indiDriver, INDI_IDLE );

    if( dev::stdMotionStage<elliptecCtrl>::appStartup() < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__ } );
    }

    // STATUS text
    m_ipStatus = pcf::IndiProperty( pcf::IndiProperty::Text );
    m_ipStatus.setName( "status" );
    m_ipStatus.setLabel( "Status" );
    m_ipStatus.setGroup( "rotation" );
    m_ipStatus.addIfNoExist( pcf::IndiElement( "current", "INITIALIZED" ) );
    REG_INDI_NEWPROP_NOCB( m_ipStatus, "status", pcf::IndiProperty::Text );

    // Stage-name position mapping (text)
    m_ipStageNamePos = pcf::IndiProperty( pcf::IndiProperty::Text );
    m_ipStageNamePos.setName( "stageNamePos" );
    m_ipStageNamePos.setLabel( "Stage Name Pos (deg)" );
    m_ipStageNamePos.setGroup( "rotation" );
    m_ipStageNamePos.addIfNoExist( pcf::IndiElement( "current", buildStageNamePosText_() ) );
    REG_INDI_NEWPROP_NOCB( m_ipStageNamePos, "stageNamePos", pcf::IndiProperty::Text );

    CREATE_REG_INDI_NEW_REQUESTSWITCH( m_ipOptimize, "optimize" );
    CREATE_REG_INDI_NEW_REQUESTSWITCH( m_ipSave, "save" );

    TELEMETER_APP_STARTUP;

    syncPresetState_();
    syncControllerState_();
    updateStatus_();
    return 0;
}

inline int elliptecCtrl::appLogic()
{
    if( state() == stateCodes::INITIALIZED || state() == stateCodes::POWERON )
    {
        state( stateCodes::NOTCONNECTED );
    }

    if( dev::stdMotionStage<elliptecCtrl>::appLogic() < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__ } );
    }

    if( powerState() != 1 || powerStateTarget() != 1 )
    {
        state( stateCodes::POWEROFF );

        if( m_wasPowered )
        {
            if( onPowerOff() < 0 )
            {
                return log<software_error, -1>( { __FILE__, __LINE__ } );
            }

            m_wasPowered = false;
        }
        else
        {
            if( whilePowerOff() < 0 )
            {
                return log<software_error, -1>( { __FILE__, __LINE__ } );
            }
        }

        if( dev::stdMotionStage<elliptecCtrl>::updateINDI() < 0 )
        {
            return log<software_error, -1>( { __FILE__, __LINE__ } );
        }

        TELEMETER_APP_LOGIC;
        return 0;
    }

    if( state() == stateCodes::POWEROFF )
    {
        state( stateCodes::NOTCONNECTED );
    }

    m_wasPowered = true;

    if( !m_connected )
    {
        if( m_startupDelayMs )
        {
            std::this_thread::sleep_for( std::chrono::milliseconds( m_startupDelayMs ) );
        }

        if( openPort_() == 0 )
        {
            m_connected = true;
            state( stateCodes::CONNECTED );
            log<text_log>( "elliptecCtrl connected on " + m_port );

            drainInput_();
            (void)q_info_();
            (void)q_position_();
            (void)q_status_();
            (void)cmd_setvel_( m_velPercent );
            (void)q_status_();

            syncPresetState_();
            syncControllerState_();
            recordPosition( true );
            recordStage( true );
            updateStatus_();

            if( m_indiDriver )
            {
                m_ipStageNamePos.setTimeStamp( pcf::TimeStamp() );
                m_indiDriver->sendSetProperty( m_ipStageNamePos );
            }
        }
        else
        {
            state( stateCodes::NOTCONNECTED );
            m_moving  = -2;
            m_pending = Pending::None;
            updateStatus_();

            if( dev::stdMotionStage<elliptecCtrl>::updateINDI() < 0 )
            {
                return log<software_error, -1>( { __FILE__, __LINE__ } );
            }

            TELEMETER_APP_LOGIC;
            return 0;
        }
    }

    int prc = pollDevice_();
    if( prc < 0 )
    {
        closePort_();
        m_connected = false;
        state( stateCodes::NOTCONNECTED );
        m_moving     = -2;
        m_pending    = Pending::None;
        m_commMisses = 0;
        syncPresetState_();
        syncControllerState_();
        recordPosition( true );
        updateStatus_();

        if( dev::stdMotionStage<elliptecCtrl>::updateINDI() < 0 )
        {
            return log<software_error, -1>( { __FILE__, __LINE__ } );
        }

        TELEMETER_APP_LOGIC;
        return 0;
    }
    // prc == 0 (fresh ok) or prc == +1 (soft miss tolerated) -> continue

    // Push abs position & velocity UI
    indi::updateIfChanged( m_ipAbsDeg, "current", m_posDeg, m_indiDriver, ( m_gs == 0x09 ? INDI_BUSY : INDI_IDLE ) );
    indi::updateIfChanged( m_ipVelPct, "current", m_velPercent, m_indiDriver, INDI_IDLE );

    if( prc == 0 )
    {
        recordPosition();
    }

    if( dev::stdMotionStage<elliptecCtrl>::updateINDI() < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__ } );
    }

    TELEMETER_APP_LOGIC;
    return 0;
}

inline int elliptecCtrl::appShutdown()
{
    if( dev::stdMotionStage<elliptecCtrl>::appShutdown() < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
    }

    TELEMETER_APP_SHUTDOWN;
    closePort_();
    return 0;
}

inline int elliptecCtrl::onPowerOff()
{
    if( dev::stdMotionStage<elliptecCtrl>::onPowerOff() < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__ } );
    }

    closePort_();
    m_connected  = false;
    m_pending    = Pending::None;
    m_commMisses = 0;
    m_statusHint.clear();
    m_homed = false;
    clearPresetNameTracking();
    m_preset_target = 0;
    syncPresetState_();
    syncControllerState_();
    updateStatus_();
    recordStage( true );
    recordPosition( true );

    return 0;
}

inline int elliptecCtrl::whilePowerOff()
{
    if( dev::stdMotionStage<elliptecCtrl>::whilePowerOff() < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__ } );
    }

    return 0;
}

/* ---------- stdMotionStage surface ---------- */

inline int elliptecCtrl::stop()
{
    if( powerState() != 1 || powerStateTarget() != 1 )
    {
        return 0;
    }

    int rc = cmd_stop_();
    if( rc == 0 )
    {
        clearPresetNameTracking();
        m_preset_target = 0;
        m_statusHint.clear();
        syncPresetState_();
        syncControllerState_();
        indi::updateSwitchIfChanged( m_indiP_stop, "request", pcf::IndiElement::Off, m_indiDriver, INDI_IDLE );
        updateStatus_();
        recordStage( true );
        recordPosition( true );
    }
    return rc;
}

inline int elliptecCtrl::startHoming()
{
    if( powerState() != 1 || powerStateTarget() != 1 )
    {
        return 0;
    }

    // Non-blocking: issue command, mark pending; main loop will poll and update position
    int rc = cmd_home_( 0 );
    if( rc < 0 )
    {
        return rc;
    }

    clearPresetNameTracking();
    m_preset_target = 0;
    syncPresetState_();
    syncControllerState_();
    indi::updateSwitchIfChanged( m_indiP_home, "request", pcf::IndiElement::Off, m_indiDriver, INDI_IDLE );
    updateStatus_();
    recordStage( true );
    recordPosition( true );

    // opportunistic position read
    (void)q_position_();
    return 0;
}

inline float elliptecCtrl::presetNumber()
{
    return static_cast<float>( presetIndexForPosition_( m_posDeg ) );
}

inline int elliptecCtrl::moveTo( float target )
{
    double deg = target;
    int    idx = -1;

    if( m_movingState == 0 )
    {
        idx = static_cast<int>( std::lround( target ) ) - 1;
        if( idx < 0 )
        {
            idx = 0;
        }
        if( idx >= static_cast<int>( m_presetPositions.size() ) )
        {
            idx = static_cast<int>( m_presetPositions.size() ) - 1;
        }
        if( idx < 0 )
        {
            return -1;
        }

        deg = m_presetPositions[static_cast<size_t>( idx )];
    }
    else
    {
        idx = presetIndexForPosition_( deg );
    }

    if( idx >= 0 )
    {
        m_preset_target = static_cast<float>( idx + 1 );
    }

    return startMoveToDeg_( deg );
}

inline int elliptecCtrl::startMoveToDeg_( double deg )
{
    if( powerState() != 1 || powerStateTarget() != 1 )
    {
        return 0;
    }

    if( moveAbsDeg_( deg ) < 0 )
    {
        return -1;
    }

    m_statusHint = ( m_movingState == 1 ) ? "Moving to preset" : "Moving";
    syncPresetState_();
    syncControllerState_();
    updateStatus_();
    recordStage( true );
    recordPosition( true );
    return 0;
}

/* ---------- telemeter wrappers ---------- */

inline int elliptecCtrl::checkRecordTimes()
{
    return telemeterT::checkRecordTimes( telem_stage(), telem_position() );
}

inline int elliptecCtrl::recordTelem( const telem_stage * )
{
    return recordStage( true );
}

inline int elliptecCtrl::recordTelem( const telem_position * )
{
    return recordPosition( true );
}

inline int elliptecCtrl::recordStage( bool force )
{
    return dev::stdMotionStage<elliptecCtrl>::recordStage( force );
}

inline int elliptecCtrl::recordPosition( bool force )
{
    static double last_posDeg = 0.0;
    static bool   first       = true;

    if( first || m_posDeg != last_posDeg || force )
    {
        telem<telem_position>( static_cast<float>( m_posDeg ) );
        last_posDeg = m_posDeg;
        first       = false;
    }

    return 0;
}

/* ---------- INDI callbacks (non-blocking; main loop polls) ---------- */

// Absolute move
INDI_NEWCALLBACK_DEFN( elliptecCtrl, m_ipAbsDeg )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_ipAbsDeg, ipRecv );
    if( powerState() != 1 || powerStateTarget() != 1 )
    {
        return 0;
    }

    double tgt = 0.0;
    if( indiTargetUpdate( m_ipAbsDeg, tgt, ipRecv, true ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__ } );
    }

    indi::updateIfChanged( m_ipAbsDeg, "target", tgt, m_indiDriver, INDI_BUSY );

    clearPresetNameTracking();
    m_movingState   = 0;
    m_preset_target = 0;
    if( startMoveToDeg_( tgt ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "moveAbsDeg failed" } );
    }

    (void)q_position_();
    return 0;
}

// relDeg: updates m_relStepDeg only
INDI_NEWCALLBACK_DEFN( elliptecCtrl, m_ipRelDeg )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_ipRelDeg, ipRecv );
    double step = m_relStepDeg;
    if( indiTargetUpdate( m_ipRelDeg, step, ipRecv, true ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__ } );
    }
    if( step < -720.0 )
    {
        step = -720.0;
    }
    if( step > 720.0 )
    {
        step = 720.0;
    }
    m_relStepDeg = step;
    indi::updateIfChanged( m_ipRelDeg, "current", m_relStepDeg, m_indiDriver, INDI_IDLE );
    indi::updateIfChanged( m_ipRelDeg, "target", m_relStepDeg, m_indiDriver, INDI_IDLE );
    return 0;
}

// relMove
INDI_NEWCALLBACK_DEFN( elliptecCtrl, m_ipRelMove )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_ipRelMove, ipRecv );
    if( powerState() != 1 || powerStateTarget() != 1 )
    {
        return 0;
    }

    if( !ipRecv.find( "request" ) )
    {
        return 0;
    }

    if( ipRecv.at( "request" ).getSwitchState() == pcf::IndiElement::On )
    {
        clearPresetNameTracking();
        m_movingState   = 0;
        m_preset_target = 0;
        if( moveRelDegCmdFromRelMove_() < 0 )
        {
            return log<software_error, -1>( { __FILE__, __LINE__, "moveRelDeg failed" } );
        }

        m_statusHint = "Moving";
        syncPresetState_();
        syncControllerState_();
        updateStatus_();
        recordStage( true );
        recordPosition( true );

        (void)q_position_();

        indi::updateSwitchIfChanged( m_ipRelMove, "request", pcf::IndiElement::Off, m_indiDriver, INDI_IDLE );
    }
    return 0;
}

INDI_NEWCALLBACK_DEFN( elliptecCtrl, m_ipVelPct )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_ipVelPct, ipRecv );
    if( powerState() != 1 || powerStateTarget() != 1 )
    {
        return 0;
    }

    int pct = 0;
    if( indiTargetUpdate( m_ipVelPct, pct, ipRecv, true ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__ } );
    }
    if( pct < 0 )
    {
        pct = 0;
    }
    if( pct > 100 )
    {
        pct = 100;
    }
    if( cmd_setvel_( pct ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "setvel failed" } );
    }
    m_velPercent = pct;
    indi::updateIfChanged( m_ipVelPct, "current", m_velPercent, m_indiDriver, INDI_IDLE );
    indi::updateIfChanged( m_ipVelPct, "target", m_velPercent, m_indiDriver, INDI_IDLE );
    recordStage( true );
    recordPosition( true );
    return 0;
}

// optimize routine
INDI_NEWCALLBACK_DEFN( elliptecCtrl, m_ipOptimize )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_ipOptimize, ipRecv );
    if( powerState() != 1 || powerStateTarget() != 1 )
    {
        return 0;
    }

    if( !ipRecv.find( "request" ) )
    {
        return 0;
    }
    if( ipRecv.at( "request" ).getSwitchState() == pcf::IndiElement::On )
    {
        (void)cmd_optimize_wait_();
        m_statusHint = "Running Optimization Routine";
        syncPresetState_();
        syncControllerState_();
        updateStatus_();
        recordStage( true );
        recordPosition( true );
        indi::updateSwitchIfChanged( m_ipOptimize, "request", pcf::IndiElement::Off, m_indiDriver, INDI_IDLE );
    }
    return 0;
}

// save routine (saves device internal parameters)
INDI_NEWCALLBACK_DEFN( elliptecCtrl, m_ipSave )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_ipSave, ipRecv );
    if( powerState() != 1 || powerStateTarget() != 1 )
    {
        return 0;
    }

    if( !ipRecv.find( "request" ) )
    {
        return 0;
    }
    if( ipRecv.at( "request" ).getSwitchState() == pcf::IndiElement::On )
    {
        (void)cmd_save_();
        m_statusHint = "Saving Tuning Parameters";
        syncPresetState_();
        syncControllerState_();
        updateStatus_();
        recordStage( true );
        recordPosition( true );
        indi::updateSwitchIfChanged( m_ipSave, "request", pcf::IndiElement::Off, m_indiDriver, INDI_IDLE );
    }
    return 0;
}

/* ---------- Serial ---------- */
inline speed_t elliptecCtrl::to_termios_baud_( int b )
{
    switch( b )
    {
    case 9600:
        return B9600;
    case 19200:
        return B19200;
    case 38400:
        return B38400;
    case 57600:
        return B57600;
    case 115200:
        return B115200;
    default:
        return B9600;
    }
}

inline int elliptecCtrl::openPort_()
{
    closePort_();
    m_fd = ::open( m_port.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK );
    if( m_fd < 0 )
    {
        return -1;
    }

    termios tio{};
    if( tcgetattr( m_fd, &tio ) < 0 )
    {
        closePort_();
        return -1;
    }
    cfmakeraw( &tio );
    speed_t sp = to_termios_baud_( m_baud );
    cfsetispeed( &tio, sp );
    cfsetospeed( &tio, sp );
    tio.c_cflag |= ( CLOCAL | CREAD );
    tio.c_cflag &= ~CRTSCTS; // no HW flow
    tio.c_cc[VMIN]  = 0;
    tio.c_cc[VTIME] = 0;
    if( tcsetattr( m_fd, TCSANOW, &tio ) < 0 )
    {
        closePort_();
        return -1;
    }
    tcflush( m_fd, TCIOFLUSH );
    return 0;
}

inline void elliptecCtrl::closePort_()
{
    if( m_fd >= 0 )
    {
        ::close( m_fd );
        m_fd = -1;
    }
}

inline int elliptecCtrl::drainInput_()
{
    if( m_fd < 0 )
    {
        return -1;
    }
    char tmp[256];
    for( ;; )
    {
        ssize_t n = ::read( m_fd, tmp, sizeof( tmp ) );
        if( n <= 0 )
        {
            break;
        }
    }
    return 0;
}

inline int elliptecCtrl::writeAll_( const std::string &s )
{
    if( m_fd < 0 )
    {
        return -1;
    }
    size_t off = 0;
    while( off < s.size() )
    {
        ssize_t w = ::write( m_fd, s.data() + off, s.size() - off );
        if( w < 0 )
        {
            if( errno == EAGAIN || errno == EWOULDBLOCK )
            {
                std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
                continue;
            }
            return -1;
        }
        off += (size_t)w;
    }
    if( m_postWriteSleepMs > 0 )
    {
        std::this_thread::sleep_for( std::chrono::milliseconds( m_postWriteSleepMs ) );
    }
    return 0;
}

inline int elliptecCtrl::readFrame_( std::string &out, int timeout_ms )
{
    out.clear();
    if( m_fd < 0 )
    {
        return -1;
    }
    auto start = std::chrono::steady_clock::now();
    char ch;
    while( true )
    {
        if( std::chrono::duration_cast<std::chrono::milliseconds>( std::chrono::steady_clock::now() - start ).count() >
            timeout_ms )
        {
            return 1; // soft timeout
        }
        struct pollfd pfd
        {
            m_fd, POLLIN, 0
        };
        int pr = ::poll( &pfd, 1, 25 );
        if( pr <= 0 )
        {
            continue;
        }
        ssize_t r = ::read( m_fd, &ch, 1 );
        if( r == 1 )
        {
            out.push_back( ch );
            size_t n = out.size();
            if( n >= 2 && out[n - 2] == '\r' && out[n - 1] == '\n' )
            {
                return 0;
            }
            continue;
        }
    }
}

inline std::string elliptecCtrl::frame_( const std::string &cmd ) const
{
    std::string f;
    f.reserve( 1 + cmd.size() );
    f.push_back( m_addr );
    f += cmd;
    return f;
}

inline int elliptecCtrl::txrx_( const std::string &cmd, std::string *reply, int timeout_ms )
{
    std::string f = frame_( cmd );
    if( writeAll_( f ) < 0 )
    {
        return -1;
    }
    if( !reply )
    {
        return 0;
    }
    const int tmo = ( m_pending != Pending::None || m_gs == 0x09 ) ? m_busyReadTimeoutMs : timeout_ms;
    int       rc  = readFrame_( *reply, tmo );
    // rc: 0 ok, 1 timeout (soft), -1 hard error
    return rc;
}

/* ---------- Protocol queries ---------- */

inline int elliptecCtrl::q_info_()
{
    std::string r;
    int         rc = txrx_( "in", &r, m_readTimeoutMs );
    if( rc < 0 )
    {
        return -1;
    }
    if( rc > 0 )
    {
        return +1;
    }
    if( r.size() >= 2 && r[r.size() - 2] == '\r' )
    {
        r.resize( r.size() - 2 );
    }

    if( m_pulsesPerRev == 0 && r.size() >= 8 )
    {
        bool hex = true;
        for( size_t i = r.size() - 8; i < r.size(); ++i )
        {
            if( !std::isxdigit( (unsigned char)r[i] ) )
            {
                hex = false;
                break;
            }
        }
        if( hex )
        {
            uint32_t v = 0;
            for( size_t i = r.size() - 8; i < r.size(); ++i )
            {
                char c = (char)std::toupper( (unsigned char)r[i] );
                v      = ( v << 4 ) | (uint32_t)( ( c <= '9' ) ? ( c - '0' ) : ( 10 + ( c - 'A' ) ) );
            }
            if( v )
            {
                m_pulsesPerRev = v;
            }
        }
    }
    return 0;
}

inline int elliptecCtrl::q_status_()
{
    std::string r;
    const int   tmo = ( m_gs == 0x09 || m_pending != Pending::None ) ? m_busyReadTimeoutMs : m_readTimeoutMs;
    int         rc  = txrx_( "gs", &r, tmo );
    if( rc < 0 )
    {
        return -1;
    }
    if( rc > 0 )
    {
        return +1;
    }
    if( r.size() >= 6 && ( r[1] == 'G' || r[1] == 'g' ) && ( r[2] == 'S' || r[2] == 's' ) &&
        std::isxdigit( (unsigned char)r[3] ) && std::isxdigit( (unsigned char)r[4] ) )
    {
        auto nyb = []( char c ) -> uint8_t
        {
            c = (char)std::toupper( (unsigned char)c );
            return ( c <= '9' ) ? ( c - '0' ) : ( 10 + ( c - 'A' ) );
        };
        m_gs = (uint8_t)( ( nyb( r[3] ) << 4 ) | nyb( r[4] ) );
    }
    return 0;
}

inline int elliptecCtrl::q_position_()
{
    std::string r;
    const int   tmo = ( m_gs == 0x09 || m_pending != Pending::None ) ? m_busyReadTimeoutMs : m_readTimeoutMs;
    int         rc  = txrx_( "gp", &r, tmo );
    if( rc < 0 )
    {
        return -1;
    }
    if( rc > 0 )
    {
        return +1;
    }
    if( r.size() >= 11 && ( r[1] == 'P' || r[1] == 'p' ) && ( r[2] == 'O' || r[2] == 'o' ) )
    {
        int32_t pulses = 0;
        for( int i = 0; i < 8; i++ )
        {
            char c = (char)std::toupper( (unsigned char)r[3 + i] );
            if( !std::isxdigit( (unsigned char)c ) )
            {
                pulses = m_posPulses;
                break;
            }
            pulses = ( pulses << 4 ) | (int32_t)( ( c <= '9' ) ? ( c - '0' ) : ( 10 + ( c - 'A' ) ) );
        }
        m_posPulses = pulses;
        m_posDeg    = pulsesToDeg_( m_posPulses );
    }
    return 0;
}

/* ---------- Commands ---------- */

inline int elliptecCtrl::cmd_home_( uint8_t dir )
{
    char        nib = "0123456789ABCDEF"[dir & 0xF];
    std::string r;
    m_pending    = Pending::Home;
    m_moving     = 2;
    m_statusHint = "HOMING";
    if( txrx_( std::string( "ho" ) + nib, &r, m_readTimeoutMs ) < 0 )
    {
        return -1;
    }
    return 0;
}

inline int elliptecCtrl::cmd_stop_()
{
    std::string r;
    m_pending = Pending::Stop;
    if( txrx_( "st", &r, m_readTimeoutMs ) < 0 )
    {
        return -1;
    }
    return 0;
}

inline int elliptecCtrl::cmd_optimize_wait_()
{
    std::string r;
    m_pending    = Pending::Optimize;
    m_moving     = 1;
    m_statusHint = "Running Optimization Routine";
    if( txrx_( m_cmdOptimize, &r, m_readTimeoutMs ) < 0 )
    {
        return -1;
    }
    return 0;
}

inline int elliptecCtrl::cmd_save_()
{
    std::string r;
    m_pending    = Pending::Save;
    m_moving     = 1;
    m_statusHint = "Saving...";
    int rc       = txrx_( m_cmdSave, &r, m_readTimeoutMs );
    return rc;
}

inline int elliptecCtrl::cmd_setvel_( int pct )
{
    if( pct < 0 )
    {
        pct = 0;
    }
    if( pct > 100 )
    {
        pct = 100;
    }
    char buf[3];
    std::snprintf( buf, sizeof( buf ), "%02X", pct );
    std::string r;
    m_pending = Pending::Velocity;
    int rc    = txrx_( std::string( "sv" ) + buf, &r, m_readTimeoutMs );
    return rc;
}

inline int elliptecCtrl::cmd_moveAbs_pulses_( int32_t pulses )
{
    char hex[9];
    std::snprintf( hex, sizeof( hex ), "%08X", (uint32_t)pulses );
    std::string r;
    m_pending    = Pending::MoveAbs;
    m_moving     = 1;
    m_statusHint = "Moving...";
    if( txrx_( std::string( "ma" ) + hex, &r, m_readTimeoutMs ) < 0 )
    {
        return -1;
    }
    return 0;
}

inline int elliptecCtrl::cmd_moveRel_pulses_( int32_t pulses )
{
    char hex[9];
    std::snprintf( hex, sizeof( hex ), "%08X", (uint32_t)pulses );
    std::string r;
    m_pending    = Pending::MoveRel;
    m_moving     = 1;
    m_statusHint = "Moving...";
    if( txrx_( std::string( "mr" ) + hex, &r, m_readTimeoutMs ) < 0 )
    {
        return -1;
    }
    return 0;
}

/* ---------- Degree wrappers ---------- */

inline int32_t elliptecCtrl::degToPulses_( double deg ) const
{
    if( m_pulsesPerRev == 0 )
    {
        return 0;
    }
    return (int32_t)std::llround( ( deg / 360.0 ) * (double)m_pulsesPerRev );
}

inline double elliptecCtrl::pulsesToDeg_( int32_t pulses ) const
{
    if( m_pulsesPerRev == 0 )
    {
        return 0.0;
    }
    return (double)pulses * 360.0 / (double)m_pulsesPerRev;
}

inline int elliptecCtrl::moveAbsDeg_( double deg )
{
    if( !m_allowMultiturn )
    {
        while( deg < 0.0 )
        {
            deg += 720.0;
        }
        while( deg >= 720. )
        {
            deg -= 720.0;
        }
    }
    int32_t p = degToPulses_( deg );
    return cmd_moveAbs_pulses_( p );
}

inline int elliptecCtrl::moveRelDeg_( double ddeg )
{
    int32_t p = degToPulses_( ddeg );
    return cmd_moveRel_pulses_( p );
}

inline int elliptecCtrl::moveRelDegCmdFromRelMove_()
{
    return moveRelDeg_( m_relStepDeg );
}

inline int elliptecCtrl::presetIndexForPosition_( double deg ) const
{
    const double tol = presetToleranceDeg_();

    for( size_t i = 0; i < m_presetPositions.size(); ++i )
    {
        if( std::abs( static_cast<double>( m_presetPositions[i] ) - deg ) <= tol )
        {
            return static_cast<int>( i );
        }
    }

    return -1;
}

inline double elliptecCtrl::presetToleranceDeg_() const
{
    if( m_pulsesPerRev == 0 )
    {
        return 1e-3;
    }

    return std::max( 1e-6, std::abs( pulsesToDeg_( 1 ) ) / 2.0 );
}

inline void elliptecCtrl::syncPresetState_()
{
    const int presetIndex = presetIndexForPosition_( m_posDeg );

    if( presetIndex < 0 )
    {
        m_preset = 0;

        if( m_moving <= 0 )
        {
            m_preset_target = 0;
        }

        return;
    }

    m_preset = static_cast<float>( presetIndex + 1 );

    if( m_moving <= 0 )
    {
        m_preset_target = m_preset;
    }
}

inline void elliptecCtrl::syncControllerState_()
{
    if( state() == stateCodes::POWEROFF )
    {
        return;
    }

    if( !m_connected )
    {
        state( stateCodes::NOTCONNECTED );
        return;
    }

    if( m_moving == 2 )
    {
        state( stateCodes::HOMING );
        return;
    }

    if( m_moving == 1 )
    {
        state( stateCodes::OPERATING );
        return;
    }

    if( !m_homed )
    {
        state( stateCodes::NOTHOMED );
        return;
    }

    state( stateCodes::READY );
}

/* ---------- Poll/resolve ---------- */

inline int elliptecCtrl::pollDevice_()
{
    int rcP = q_position_();
    int rcS = q_status_();

    // Hard errors => reconnect
    if( rcP < 0 || rcS < 0 )
    {
        return -1;
    }

    // Soft timeouts => debounce; reconnect if too many misses in a row
    if( rcP > 0 || rcS > 0 )
    {
        if( ++m_commMisses <= m_commMaxMisses )
        {
            updateStatus_();
            return +1; // tolerated soft miss
        }
        else
        {
            m_commMisses = 0;
            return -1; // exceeded budget -> force reconnect
        }
    }

    // Success path
    m_commMisses = 0;

    if( m_gs == 0x09 )
    { // BUSY
        switch( m_pending )
        {
        case Pending::Home:
            m_moving = 2;
            break;
        case Pending::MoveAbs:
        case Pending::MoveRel:
        case Pending::OffsetRel:
        case Pending::Optimize:
        case Pending::Velocity:
        case Pending::Save:
        case Pending::Stop:
            m_moving = 1;
            break;
        case Pending::None:
            if( m_moving <= 0 )
            {
                m_moving = 1; // external motion
            }
            break;
        }
    }
    else
    { // OK (idle)
        switch( m_pending )
        {
        case Pending::Home:
            m_homed   = true;
            m_pending = Pending::None;
            m_moving  = 0;
            m_statusHint.clear();
            if( std::abs( m_homeOffsetDeg ) > 0.0 )
            {
                if( moveRelDeg_( m_homeOffsetDeg ) == 0 )
                {
                    m_pending    = Pending::OffsetRel;
                    m_moving     = 1;
                    m_statusHint = "Homed. Now applying offset...";
                    recordStage( true );
                    recordPosition( true );
                }
            }
            break;

        case Pending::OffsetRel:
        case Pending::MoveAbs:
        case Pending::MoveRel:
        case Pending::Velocity:
        case Pending::Save:
        case Pending::Stop:
        case Pending::Optimize:
            m_pending = Pending::None;
            m_moving  = 0;
            m_statusHint.clear();
            break;

        case Pending::None:
            m_moving = ( m_homed ? 0 : -1 );
            break;
        }
    }

    syncPresetState_();
    syncControllerState_();
    updateStatus_();
    return 0;
}

inline void elliptecCtrl::updateStatus_()
{
    if( !m_indiDriver )
    {
        return;
    }

    std::string s;
    if( state() == stateCodes::POWEROFF )
    {
        s = "Powered Off";
    }
    else if( !m_connected )
    {
        s = "Not Connected";
    }
    else if( m_moving == 2 )
    {
        s = "Homing";
    }
    else if( m_moving == 1 )
    {
        s = ( !m_statusHint.empty() ? m_statusHint : "Busy" );
    }
    else if( !m_homed )
    {
        s = "Not Homed";
    }
    else
    {
        s = "OK";
    }

    bool changed = false;
    if( !m_ipStatus.find( "current" ) )
    {
        m_ipStatus.addIfNoExist( pcf::IndiElement( "current", s ) );
        changed = true;
    }
    else
    {
        const std::string cur = m_ipStatus.at( "current" ).getValue<std::string>();
        if( cur != s )
        {
            m_ipStatus.at( "current" ).setValue( s );
            changed = true;
        }
    }
    if( changed )
    {
        m_ipStatus.setState( ( m_moving > 0 ) ? INDI_BUSY : INDI_IDLE );
        m_ipStatus.setTimeStamp( pcf::TimeStamp() );
        m_indiDriver->sendSetProperty( m_ipStatus );
    }

    // keep absDeg flowing with appropriate state
    indi::updateIfChanged( m_ipAbsDeg, "current", m_posDeg, m_indiDriver, ( m_moving > 0 ? INDI_BUSY : INDI_IDLE ) );
}

/* ---------- Stage name/position text ---------- */

inline std::string elliptecCtrl::buildStageNamePosText_() const
{
    if( m_presetNames.empty() || m_presetPositions.empty() )
    {
        return "(none)";
    }
    const size_t n   = std::min( m_presetNames.size(), m_presetPositions.size() );
    auto         fmt = []( double v ) -> std::string
    {
        std::ostringstream os;
        double             iv;
        if( std::modf( v, &iv ) == 0.0 )
        {
            os << (long long)std::llround( v );
        }
        else
        {
            os << std::fixed << std::setprecision( 6 ) << v;
        }
        std::string s = os.str();
        if( s.find( '.' ) != std::string::npos )
        {
            while( !s.empty() && s.back() == '0' )
            {
                s.pop_back();
            }
            if( !s.empty() && s.back() == '.' )
            {
                s.pop_back();
            }
        }
        return s;
    };

    std::ostringstream out;
    for( size_t i = 0; i < n; ++i )
    {
        if( i )
        {
            out << ", ";
        }
        out << m_presetNames[i] << ":" << fmt( m_presetPositions[i] );
    }
    return out.str();
}

} // namespace app
} // namespace MagAOX

#endif // elliptecCtrl_hpp
