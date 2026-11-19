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
 * Elliptec protocol reference (Thorlabs ELLx OEM/Bare modules protocol manual, Issue 12):
 * - TX: `<ADDR><cmd>[args]` with no terminator; command lengths are fixed per command
 * - RX: CRLF-terminated frames `<ADDR><HEADER><payload>`
 * - motion commands (`ho`, `ma`, `mr`, `om`) reply `GS09` while busy (or an error `GS`) and report `PO` or `GS00`
 *   on completion, so they are sent without waiting and their progress is tracked by polling `gs`
 * - commands used here: `in`, `gs`, `gp`, `hoX`, `st`, `us`, `om`, `svHH`, `maXXXXXXXX`, `mrXXXXXXXX`
 *
 * Threading: all serial I/O is serialized by `m_indiMutex`. `appLogic()` holds it while talking to the
 * device, the app's INDI callbacks acquire it, and the `stdMotionStage` callbacks acquire it before calling
 * `moveTo()` / `startHoming()`. `stop()` is called without the mutex and acquires it itself.
 *
 * The stage is modeled as a single 360 degree circle: positions are reported wrapped into [0, 360).
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
    /**
     * \returns 0 on success
     * \returns -1 on an invalid configuration
     */
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

    /// Request a stop of the current motion.
    /** Called by `stdMotionStage` without `m_indiMutex` held; acquires it itself. The Elliptec `st` command only
     * stops continuous motion, so the request is sent but the pending operation is left for the poll to resolve.
     */
    int stop();

    /// Begin a homing sequence. Caller holds `m_indiMutex`.
    int startHoming();

    /// Resolve the active 0-based preset index for the current stage angle, or -1 if not at a preset.
    float presetNumber();

    /// Move to a target supplied by `stdMotionStage`. Caller holds `m_indiMutex`.
    int moveTo( float target /**< [in] preset number (m_movingState == 0) or preset position in degrees */ );

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

    /// Human-readable meaning of an Elliptec `GS` status code.
    static std::string statusMeaning( uint8_t code /**< [in] status/error code from a `GS` frame */ );

    /// Minimum acceptable velocity percentage for an Elliptec model, or 0 if unrestricted.
    static int minVelocityForModel( uint8_t model /**< [in] model number from the `in` reply, e.g. 21 */ );

    /// Wrap an angle into [0, 360).
    static double wrapDeg( double deg /**< [in] angle in degrees */ );

    /// Shortest angular distance between two angles on a circle, in [0, 180].
    static double circularDistanceDeg( double a, /**< [in] first angle in degrees */
                                       double b /**< [in] second angle in degrees */ );

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

    /// Read timeout for a query reply in milliseconds.
    int m_readTimeoutMs{ 1000 };

    /// Optional post-write delay in milliseconds.
    int m_postWriteSleepMs{ 0 };

    /// Velocity percentage command value.
    int m_velPercent{ 80 };

    /// Absolute angle in degrees to move to after a home completes; 0 leaves the stage at home.
    double m_homeOffsetDeg{ 0.0 };

    /// Pulses per 360-degree revolution, from configuration or the `in` reply.
    uint32_t m_pulsesPerRev{ 0 };

    /// Whether `m_pulsesPerRev` was set by configuration and must not be overridden by the device.
    bool m_pulsesPerRevConfigured{ false };

    /// Device command used to run the optimize routine.
    std::string m_cmdOptimize{ "om" };

    /// Device command used to save parameters.
    std::string m_cmdSave{ "us" };

    /// Consecutive missed replies tolerated before forcing a reconnect.
    int m_commMaxMisses{ 3 };

    /// Whether every transmitted and received frame is logged at `LOG_DEBUG2`.
    bool m_serialTrace{ false };

    /// Angular tolerance in degrees for deciding the stage is at a preset.
    double m_presetToleranceDeg{ 0.5 };

    /// Whether the maintenance commands (`optimize`, `save`) are enabled.
    bool m_enableMaintenance{ false };

    /// Maximum duration of an optimize routine before it is treated as failed, in seconds.
    double m_optimizeTimeoutSec{ 600 };

    ///@}

    /** \name Runtime State - Data
     *
     * @{
     */

    /// Open file descriptor for the serial port, or `-1` when closed.
    int m_fd{ -1 };

    /// Received bytes not yet assembled into a complete frame.
    std::string m_rxBuf;

    /// Whether the serial link is currently open and usable.
    bool m_connected{ false };

    /// Last known absolute position in degrees, wrapped into [0, 360).
    double m_posDeg{ 0.0 };

    /// Last known absolute position in native pulses, as reported by the device.
    int32_t m_posPulses{ 0 };

    /// Whether the stage has completed a homing sequence since power-on. Preserved across reconnects.
    bool m_homed{ false };

    /// Last status byte received in a `GS` frame.
    uint8_t m_gs{ 0x00 };

    /// Model number from the `in` reply (e.g. 21 for an ELL21), or 0 if unknown.
    uint8_t m_model{ 0 };

    /// Relative step size in degrees used by the `relMove` request.
    double m_relStepDeg{ 1.0 };

    /// Long-running device operation tracked until the status poll resolves it.
    enum class Pending
    {
        None,       ///< No operation in progress.
        Move,       ///< An absolute or relative move.
        Home,       ///< A home command.
        HomeOffset, ///< The absolute move to `m_homeOffsetDeg` that completes a home.
        Optimize    ///< The motor optimization routine.
    };

    /// Operation currently in progress on the device.
    Pending m_pending{ Pending::None };

    /// Time the current pending operation was commanded.
    std::chrono::steady_clock::time_point m_pendingStart;

    /// Whether a busy status has been seen since the current pending operation was commanded.
    bool m_sawBusy{ false };

    /// First error status code reported during the current pending operation, or 0.
    uint8_t m_latchedError{ 0 };

    /// Whether a stop was requested during the current pending operation.
    bool m_stopRequested{ false };

    /// Whether the next connection should perform the power-on home (when `stage.powerOnHome` is set).
    bool m_powerOnHomePending{ true };

    /// Operator-facing status text supplement for the current in-flight action.
    std::string m_statusHint;

    /// Operator-facing error text from the most recent failed operation, cleared by the next command.
    std::string m_errorText;

    /// Count of consecutive missed replies.
    int m_commMisses{ 0 };

    /// Whether a failed connection attempt has been logged since the last successful connection.
    bool m_connectRetryLogged{ false };

    /// Whether position telemetry has been recorded since startup.
    bool m_positionRecorded{ false };

    /// Position in degrees at the last position telemetry record.
    double m_lastRecordedPosDeg{ 0.0 };

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

    /// A parsed Elliptec reply frame.
    struct frameT
    {
        char        addr{ 0 }; ///< Device address character.
        std::string header;    ///< Upper-case two-letter reply header, e.g. "GS", "PO", "IN".
        std::string payload;   ///< Data following the header, without CRLF.
    };

    /** \name Serial Helpers
     *
     * All serial helpers require the caller to hold `m_indiMutex`.
     *
     * @{
     */

    /// Convert an integer baud rate into the corresponding termios constant.
    static speed_t to_termios_baud_( int baud /**< [in] baud rate to convert */ );

    /// Open and configure the serial port.
    int openPort_();

    /// Close the serial port if it is open and discard any partial input.
    void closePort_();

    /// Write raw bytes to the serial port.
    int writeAll_( const std::string &s /**< [in] bytes to write */ );

    /// Read one complete, well-formed frame.
    /** Malformed lines are logged at debug level and skipped.
     * \returns 0 when a frame was read
     * \returns 1 on timeout
     * \returns -1 on a hard error
     */
    int readFrame_( frameT &frame,     /**< [out] the parsed frame */
                    int     timeout_ms /**< [in] timeout in milliseconds, 0 for non-blocking */
    );

    /// Construct a device command from the payload by prefixing the address.
    std::string frame_( const std::string &cmd /**< [in] command payload without address */ ) const;

    /// Send a command without waiting for a reply.
    int send_( const std::string &cmd /**< [in] command payload without address */ );

    /// Dispatch every complete frame already received, without blocking.
    /**
     * \returns 0 on success
     * \returns -1 on a hard error
     */
    int drainFrames_();

    /// Send a command and wait for a reply with the given header.
    /** Frames that arrive first with other headers or addresses are dispatched to `handleFrame_()`.
     * \returns 0 when the expected frame arrived
     * \returns 1 on timeout
     * \returns -1 on a hard error
     */
    int query_( const std::string &cmd,       /**< [in] command payload without address */
                const std::string &header,    /**< [in] expected reply header */
                frameT            &reply,     /**< [out] the expected reply */
                int                timeout_ms /**< [in] timeout in milliseconds */
    );

    ///@}

    /** \name Protocol Access
     *
     * All protocol helpers require the caller to hold `m_indiMutex`.
     *
     * @{
     */

    /// Dispatch an unsolicited or unexpected frame by header.
    void handleFrame_( const frameT &frame /**< [in] received frame */ );

    /// Apply a status code from a `GS` frame.
    void handleStatusCode_( uint8_t code /**< [in] status/error code */ );

    /// Parse a `PO` payload into the current position.
    void parsePosition_( const std::string &payload /**< [in] 8 hex digit position payload */ );

    /// Parse an `IN` payload into the model and pulses per revolution.
    void parseInfo_( const std::string &payload /**< [in] info payload following "IN" */ );

    /// Query static device information.
    int q_info_();

    /// Query the device status byte.
    int q_status_();

    /// Query the current device position.
    int q_position_();

    /// Send a home command and begin tracking it.
    int cmd_home_( uint8_t dir = 0 /**< [in] Elliptec home-direction nibble, 0 clockwise */ );

    /// Send a stop command and report the device's reply.
    /**
     * \returns the GS code replied, or -1 on error / no reply
     */
    int cmd_stop_();

    /// Send the optimize command and begin tracking it.
    int cmd_optimize_();

    /// Send the save command and wait for its status reply.
    int cmd_save_();

    /// Set the device velocity percentage and wait for its status reply.
    int cmd_setvel_( int pct /**< [in] velocity percentage from 0 to 100 */ );

    /// Send an absolute move in native pulses.
    int cmd_moveAbs_pulses_( int32_t pulses /**< [in] absolute target in native pulses */ );

    /// Send a relative move in native pulses.
    int cmd_moveRel_pulses_( int32_t pulses /**< [in] relative delta in native pulses */ );

    ///@}

    /** \name Motion Helpers
     *
     * All motion helpers require the caller to hold `m_indiMutex`.
     *
     * @{
     */

    /// Whether a new motion may be commanded now; logs a warning naming the request when not.
    bool motionAllowed_( const std::string &what /**< [in] name of the requested action for the log */ );

    /// Record that a long-running operation has been commanded.
    void beginPending_( Pending p /**< [in] the operation now in progress */ );

    /// Move to an absolute angle in degrees, wrapped into [0, 360).
    int moveAbsDeg_( double deg /**< [in] absolute stage angle in degrees */ );

    /// Execute the current relative move request.
    int moveRelDegCmdFromRelMove_();

    /// Move by a relative angular offset in degrees.
    int moveRelDeg_( double ddeg /**< [in] relative angular offset in degrees */ );

    /// Start an absolute move in degrees after the caller has updated target bookkeeping.
    int startMoveToDeg_( double deg /**< [in] absolute stage angle in degrees */ );

    /// Clamp a velocity request to [model minimum, 100], logging a warning when raised.
    int applyVelocityFloor_( int pct /**< [in] requested velocity percentage */ );

    /// Convert degrees into native pulses.
    int32_t degToPulses_( double deg /**< [in] angle in degrees */ ) const;

    /// Convert native pulses into degrees, without wrapping.
    double pulsesToDeg_( int32_t pulses /**< [in] position in native pulses */ ) const;

    /// Resolve the preset index matching the supplied angle on the circle.
    int presetIndexForPosition_( double deg /**< [in] angle to compare against preset positions */ ) const;

    /// Return the matching tolerance used for preset comparisons.
    double presetToleranceDeg_() const;

    /// Check that distinct preset positions are separable by the preset tolerance.
    /**
     * \returns 0 if valid
     * \returns -1 if two distinct presets are closer than twice the tolerance
     */
    int validatePresets_();

    /// Update `m_preset` and `m_preset_target` from the current motion state.
    void syncPresetState_();

    /// Update the high-level FSM state from the current connection and motion state.
    void syncControllerState_();

    /// Poll the device and resolve the current pending operation.
    /**
     * \returns 0 on a successful poll
     * \returns 1 on a tolerated missed reply
     * \returns -1 when the connection must be re-established
     */
    int pollDevice_();

    /// Resolve the pending operation from the latest status.
    void resolvePending_();

    /// Complete the pending operation successfully.
    void completePending_();

    /// Fail the pending operation, logging and surfacing the reason.
    void failPending_( const std::string &reason /**< [in] operator-facing reason */ );

    /// Connect to the device and query its initial state.
    /** The connection only succeeds once the device answers the info query and the pulses per revolution are known;
     * otherwise the port is closed and the caller retries on the next loop.
     *
     * \returns 0 on success
     * \returns -1 if the port could not be opened or the device did not identify itself
     */
    int connect_();

    /// Drop the connection after communication failure.
    void disconnect_();

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

/* ---------- Static helpers ---------- */

inline std::string elliptecCtrl::statusMeaning( uint8_t code )
{
    switch( code )
    {
    case 0:
        return "OK";
    case 1:
        return "communication time out";
    case 2:
        return "mechanical time out";
    case 3:
        return "command error or not supported";
    case 4:
        return "value out of range";
    case 5:
        return "module isolated";
    case 6:
        return "module out of isolation";
    case 7:
        return "initializing error";
    case 8:
        return "thermal error";
    case 9:
        return "busy";
    case 10:
        return "sensor error";
    case 11:
        return "motor error";
    case 12:
        return "out of range";
    case 13:
        return "over current error";
    default:
        return "reserved status " + std::to_string( static_cast<int>( code ) );
    }
}

inline int elliptecCtrl::minVelocityForModel( uint8_t model )
{
    // Protocol manual: "Lowest acceptable velocity to ELL16 and ELL21 is 50%". ELL22 shares the ELL21 drive.
    if( model == 16 || model == 21 || model == 22 )
    {
        return 50;
    }

    return 0;
}

inline double elliptecCtrl::wrapDeg( double deg )
{
    double w = std::fmod( deg, 360.0 );
    if( w < 0 )
    {
        w += 360.0;
    }
    if( w >= 360.0 )
    {
        w -= 360.0;
    }
    return w;
}

inline double elliptecCtrl::circularDistanceDeg( double a, double b )
{
    double d = std::abs( wrapDeg( a ) - wrapDeg( b ) );
    return ( d > 180.0 ) ? 360.0 - d : d;
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
                "1000",
                "serial.readTimeoutMs",
                argType::Optional,
                "serial",
                "readTimeoutMs",
                false,
                "int",
                "Timeout (ms) waiting for a query reply" );

    config.add( "serial.postWriteSleepMs",
                "0",
                "serial.postWriteSleepMs",
                argType::Optional,
                "serial",
                "postWriteSleepMs",
                false,
                "int",
                "Sleep after write (ms)" );

    config.add( "serial.trace",
                "false",
                "serial.trace",
                argType::Optional,
                "serial",
                "trace",
                false,
                "bool",
                "Log every transmitted and received frame at LOG_DEBUG2" );

    // Behavior
    config.add( "stage.homeOffset",
                "0.0",
                "stage.homeOffset",
                argType::Optional,
                "stage",
                "homeOffset",
                false,
                "double",
                "Absolute angle (deg) to move to after homing; 0 leaves the stage at home" );

    config.add( "stage.presetTolerance",
                "0.5",
                "stage.presetTolerance",
                argType::Optional,
                "stage",
                "presetTolerance",
                false,
                "double",
                "Angular tolerance (deg) for being at a preset" );

    config.add( "stage.enableMaintenance",
                "false",
                "stage.enableMaintenance",
                argType::Optional,
                "stage",
                "enableMaintenance",
                false,
                "bool",
                "Enable the optimize and save maintenance commands" );

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
                "80",
                "motion.velPercent",
                argType::Optional,
                "motion",
                "velPercent",
                false,
                "int",
                "Velocity percent 0..100 (raised to the model minimum)" );

    config.add( "motion.optimizeTimeoutSec",
                "600",
                "motion.optimizeTimeoutSec",
                argType::Optional,
                "motion",
                "optimizeTimeoutSec",
                false,
                "double",
                "Maximum duration (s) of the optimize routine" );

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
                "Consecutive missed replies tolerated before reconnect" );

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
    if( m_readTimeoutMs < 1 )
    {
        m_readTimeoutMs = 1;
    }
    _config( m_postWriteSleepMs, "serial.postWriteSleepMs" );
    _config( m_serialTrace, "serial.trace" );

    _config( m_homeOffsetDeg, "stage.homeOffset" );
    m_homeOffsetDeg = wrapDeg( m_homeOffsetDeg );

    _config( m_presetToleranceDeg, "stage.presetTolerance" );
    if( m_presetToleranceDeg <= 0 )
    {
        return log<text_log, -1>( "stage.presetTolerance must be positive", logPrio::LOG_CRITICAL );
    }

    _config( m_enableMaintenance, "stage.enableMaintenance" );

    uint32_t ppr = 0;
    _config( ppr, "stage.pulsesPerRev" );
    if( ppr )
    {
        m_pulsesPerRev           = ppr;
        m_pulsesPerRevConfigured = true;
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

    _config( m_optimizeTimeoutSec, "motion.optimizeTimeoutSec" );

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

    if( validatePresets_() < 0 )
    {
        return -1;
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
    CREATE_REG_INDI_NEW_NUMBERD( m_ipAbsDeg, "absDeg", 0.0, 360.0, 0.001, "", "Absolute (deg)", "rotation" );
    indi::updateIfChanged( m_ipAbsDeg, "current", m_posDeg, m_indiDriver, INDI_IDLE );

    // relDeg (step holder)
    CREATE_REG_INDI_NEW_NUMBERD( m_ipRelDeg, "relDeg", -360.0, 360.0, 0.001, "", "Relative step (deg)", "rotation" );
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

    // Power is being switched off: leave the device alone; onPowerOff() handles the transition.
    if( powerState() != 1 || powerStateTarget() != 1 )
    {
        TELEMETER_APP_LOGIC;
        return 0;
    }

    { // mutex scope
        std::lock_guard<std::mutex> guard( m_indiMutex );

        if( !m_connected )
        {
            if( m_startupDelayMs )
            {
                std::this_thread::sleep_for( std::chrono::milliseconds( m_startupDelayMs ) );
            }

            if( connect_() < 0 )
            {
                state( stateCodes::NOTCONNECTED );
                m_moving = -2;
                updateStatus_();
            }
        }
        else
        {
            int prc = pollDevice_();
            if( prc < 0 )
            {
                disconnect_();
            }
            else
            {
                indi::updateIfChanged( m_ipVelPct, "current", m_velPercent, m_indiDriver, INDI_IDLE );

                if( prc == 0 )
                {
                    recordPosition();
                }
            }
        }

        if( dev::stdMotionStage<elliptecCtrl>::updateINDI() < 0 )
        {
            return log<software_error, -1>( { __FILE__, __LINE__ } );
        }
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

    std::lock_guard<std::mutex> guard( m_indiMutex );
    closePort_();
    return 0;
}

inline int elliptecCtrl::onPowerOff()
{
    if( dev::stdMotionStage<elliptecCtrl>::onPowerOff() < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__ } );
    }

    std::lock_guard<std::mutex> guard( m_indiMutex );

    closePort_();
    m_connected          = false;
    m_pending            = Pending::None;
    m_stopRequested      = false;
    m_latchedError       = 0;
    m_commMisses         = 0;
    m_homed              = false;
    m_powerOnHomePending = true;
    m_statusHint.clear();
    m_errorText.clear();
    clearPresetNameTracking();
    m_preset_target = 0;
    syncPresetState_();
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

    TELEMETER_APP_LOGIC;

    return 0;
}

/* ---------- Connection ---------- */

inline int elliptecCtrl::connect_()
{
    if( openPort_() < 0 )
    {
        return -1;
    }

    // The device does not answer for a few seconds after power-on. Without the info reply the angle scale is
    // unknown, so treat that as a failed connection and retry on the next loop.
    int irc = q_info_();
    if( irc != 0 || m_pulsesPerRev == 0 )
    {
        closePort_();

        if( !m_connectRetryLogged )
        {
            log<text_log>( ( irc != 0 ) ? "device on " + m_port + " is not answering yet; retrying"
                                        : "device did not report pulses per revolution; retrying",
                           logPrio::LOG_WARNING );
            m_connectRetryLogged = true;
        }

        return -1;
    }

    m_connectRetryLogged = false;
    m_connected          = true;
    m_commMisses         = 0;
    state( stateCodes::CONNECTED );
    log<text_log>( "elliptecCtrl connected on " + m_port + " to an ELL" + std::to_string( (int)m_model ) + " with " +
                   std::to_string( m_pulsesPerRev ) + " pulses/rev" );

    m_velPercent = applyVelocityFloor_( m_velPercent );
    if( cmd_setvel_( m_velPercent ) != 0 )
    {
        log<text_log>( "velocity was not acknowledged", logPrio::LOG_WARNING );
    }
    indi::updateIfChanged( m_ipVelPct, "target", m_velPercent, m_indiDriver, INDI_IDLE );

    (void)q_position_();
    (void)q_status_();

    // If the device is already moving (e.g. after a USB reconnect) this reports an unknown motion.
    m_pending = Pending::None;
    resolvePending_();

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

    if( m_powerOnHomePending )
    {
        m_powerOnHomePending = false;

        if( m_powerOnHome )
        {
            log<text_log>( "homing after power-on" );
            if( startHoming() < 0 )
            {
                log<text_log>( "power-on home could not be started", logPrio::LOG_ERROR );
            }
        }
    }

    return 0;
}

inline void elliptecCtrl::disconnect_()
{
    log<text_log>( "lost communication with the device; reconnecting", logPrio::LOG_WARNING );

    closePort_();
    m_connected     = false;
    m_pending       = Pending::None;
    m_stopRequested = false;
    m_latchedError  = 0;
    m_commMisses    = 0;
    m_moving        = -2;
    m_statusHint.clear();
    state( stateCodes::NOTCONNECTED );
    syncPresetState_();
    recordStage( true );
    recordPosition( true );
    updateStatus_();
}

/* ---------- stdMotionStage surface ---------- */

inline int elliptecCtrl::stop()
{
    std::lock_guard<std::mutex> guard( m_indiMutex );

    indi::updateSwitchIfChanged( m_indiP_stop, "request", pcf::IndiElement::Off, m_indiDriver, INDI_IDLE );

    if( powerState() != 1 || powerStateTarget() != 1 || !m_connected )
    {
        return 0;
    }

    int code = cmd_stop_();

    if( m_pending != Pending::None )
    {
        m_stopRequested = true;
    }

    if( code == 3 )
    {
        log<text_log>( "stop is not supported by the device during this motion; the move will complete",
                       logPrio::LOG_WARNING );
        if( m_pending != Pending::None )
        {
            m_statusHint = "Stop not supported; move will complete";
        }
    }
    else if( code < 0 )
    {
        log<text_log>( "no reply to stop request", logPrio::LOG_WARNING );
    }
    else
    {
        log<text_log>( "stop requested" );
    }

    updateStatus_();
    return 0;
}

inline int elliptecCtrl::startHoming()
{
    if( powerState() != 1 || powerStateTarget() != 1 )
    {
        return 0;
    }

    indi::updateSwitchIfChanged( m_indiP_home, "request", pcf::IndiElement::Off, m_indiDriver, INDI_IDLE );

    if( !motionAllowed_( "home" ) )
    {
        return 0;
    }

    if( cmd_home_( 0 ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "home command failed" } );
    }

    clearPresetNameTracking();
    m_preset_target = 0;
    m_homed         = false;
    syncPresetState_();
    syncControllerState_();
    updateStatus_();
    recordStage( true );
    recordPosition( true );

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

    if( !motionAllowed_( "preset move" ) )
    {
        return 0;
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
    if( !m_positionRecorded || m_posDeg != m_lastRecordedPosDeg || force )
    {
        telem<telem_position>( static_cast<float>( m_posDeg ) );
        m_lastRecordedPosDeg = m_posDeg;
        m_positionRecorded   = true;
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

    std::lock_guard<std::mutex> guard( m_indiMutex );

    if( !motionAllowed_( "absolute move" ) )
    {
        return 0;
    }

    tgt = wrapDeg( tgt );
    indi::updateIfChanged( m_ipAbsDeg, "target", tgt, m_indiDriver, INDI_BUSY );

    clearPresetNameTracking();
    m_movingState   = 0;
    m_preset_target = 0;
    if( startMoveToDeg_( tgt ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "moveAbsDeg failed" } );
    }

    return 0;
}

// relDeg: updates m_relStepDeg only
INDI_NEWCALLBACK_DEFN( elliptecCtrl, m_ipRelDeg )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_ipRelDeg, ipRecv );

    std::lock_guard<std::mutex> guard( m_indiMutex );

    double step = m_relStepDeg;
    if( indiTargetUpdate( m_ipRelDeg, step, ipRecv, true ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__ } );
    }
    if( step < -360.0 )
    {
        step = -360.0;
    }
    if( step > 360.0 )
    {
        step = 360.0;
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
        std::lock_guard<std::mutex> guard( m_indiMutex );

        indi::updateSwitchIfChanged( m_ipRelMove, "request", pcf::IndiElement::Off, m_indiDriver, INDI_IDLE );

        if( !motionAllowed_( "relative move" ) )
        {
            return 0;
        }

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

    std::lock_guard<std::mutex> guard( m_indiMutex );

    pct = applyVelocityFloor_( pct );

    if( m_connected && cmd_setvel_( pct ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "setvel failed" } );
    }
    m_velPercent = pct;
    indi::updateIfChanged( m_ipVelPct, "current", m_velPercent, m_indiDriver, INDI_IDLE );
    indi::updateIfChanged( m_ipVelPct, "target", m_velPercent, m_indiDriver, INDI_IDLE );
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
        std::lock_guard<std::mutex> guard( m_indiMutex );

        indi::updateSwitchIfChanged( m_ipOptimize, "request", pcf::IndiElement::Off, m_indiDriver, INDI_IDLE );

        if( !m_enableMaintenance )
        {
            log<text_log>( "optimize rejected: maintenance commands are disabled (stage.enableMaintenance)",
                           logPrio::LOG_WARNING );
            return 0;
        }

        if( !motionAllowed_( "optimize" ) )
        {
            return 0;
        }

        if( cmd_optimize_() < 0 )
        {
            return log<software_error, -1>( { __FILE__, __LINE__, "optimize command failed" } );
        }

        log<text_log>( "motor optimization started; this may take several minutes", logPrio::LOG_NOTICE );
        syncPresetState_();
        syncControllerState_();
        updateStatus_();
        recordStage( true );
        recordPosition( true );
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
        std::lock_guard<std::mutex> guard( m_indiMutex );

        indi::updateSwitchIfChanged( m_ipSave, "request", pcf::IndiElement::Off, m_indiDriver, INDI_IDLE );

        if( !m_enableMaintenance )
        {
            log<text_log>( "save rejected: maintenance commands are disabled (stage.enableMaintenance)",
                           logPrio::LOG_WARNING );
            return 0;
        }

        if( !m_connected || m_pending != Pending::None )
        {
            log<text_log>( "save rejected: the stage must be connected and idle", logPrio::LOG_WARNING );
            return 0;
        }

        log<text_log>( "saving device parameters to non-volatile memory", logPrio::LOG_NOTICE );

        if( cmd_save_() != 0 )
        {
            log<text_log>( "save was not acknowledged", logPrio::LOG_ERROR );
        }

        updateStatus_();
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

    elevatedPrivileges elPriv( this );

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
    m_rxBuf.clear();
    return 0;
}

inline void elliptecCtrl::closePort_()
{
    if( m_fd >= 0 )
    {
        ::close( m_fd );
        m_fd = -1;
    }
    m_rxBuf.clear();
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

inline int elliptecCtrl::readFrame_( frameT &frame, int timeout_ms )
{
    if( m_fd < 0 )
    {
        return -1;
    }

    auto start = std::chrono::steady_clock::now();

    while( true )
    {
        // Extract a complete line if one is buffered.
        size_t eol = m_rxBuf.find( "\r\n" );
        while( eol != std::string::npos )
        {
            std::string line = m_rxBuf.substr( 0, eol );
            m_rxBuf.erase( 0, eol + 2 );

            if( m_serialTrace )
            {
                log<text_log>( "RX: " + line, logPrio::LOG_DEBUG2 );
            }

            if( line.size() >= 3 && std::isxdigit( (unsigned char)line[0] ) && std::isalpha( (unsigned char)line[1] ) &&
                std::isalpha( (unsigned char)line[2] ) )
            {
                frame.addr      = (char)std::toupper( (unsigned char)line[0] );
                frame.header    = line.substr( 1, 2 );
                frame.header[0] = (char)std::toupper( (unsigned char)frame.header[0] );
                frame.header[1] = (char)std::toupper( (unsigned char)frame.header[1] );
                frame.payload   = line.substr( 3 );
                return 0;
            }

            log<text_log>( "discarding malformed frame: '" + line + "'", logPrio::LOG_DEBUG );
            eol = m_rxBuf.find( "\r\n" );
        }

        int elapsed = static_cast<int>(
            std::chrono::duration_cast<std::chrono::milliseconds>( std::chrono::steady_clock::now() - start ).count() );
        int remaining = timeout_ms - elapsed;

        struct pollfd pfd{ m_fd, POLLIN, 0 };

        int pr = ::poll( &pfd, 1, ( remaining > 0 ) ? std::min( remaining, 25 ) : 0 );
        if( pr < 0 )
        {
            if( errno == EINTR )
            {
                continue;
            }
            return -1;
        }

        if( pr > 0 )
        {
            if( pfd.revents & ( POLLERR | POLLHUP | POLLNVAL ) )
            {
                return -1;
            }

            char    buf[256];
            ssize_t r = ::read( m_fd, buf, sizeof( buf ) );
            if( r > 0 )
            {
                m_rxBuf.append( buf, buf + r );
                continue;
            }
            if( r < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR )
            {
                return -1;
            }
        }

        if( remaining <= 0 )
        {
            return 1; // timeout
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

inline int elliptecCtrl::send_( const std::string &cmd )
{
    std::string f = frame_( cmd );

    if( m_serialTrace )
    {
        log<text_log>( "TX: " + f, logPrio::LOG_DEBUG2 );
    }

    return writeAll_( f );
}

inline int elliptecCtrl::drainFrames_()
{
    frameT f;
    int    rc;
    while( ( rc = readFrame_( f, 0 ) ) == 0 )
    {
        handleFrame_( f );
    }

    return ( rc < 0 ) ? -1 : 0;
}

inline int elliptecCtrl::query_( const std::string &cmd, const std::string &header, frameT &reply, int timeout_ms )
{
    // Dispatch anything already received (e.g. a motion acknowledgement) so it is not taken as this reply.
    if( drainFrames_() < 0 )
    {
        return -1;
    }

    if( send_( cmd ) < 0 )
    {
        return -1;
    }

    auto start = std::chrono::steady_clock::now();

    while( true )
    {
        int elapsed = static_cast<int>(
            std::chrono::duration_cast<std::chrono::milliseconds>( std::chrono::steady_clock::now() - start ).count() );
        int remaining = timeout_ms - elapsed;
        if( remaining < 0 )
        {
            remaining = 0;
        }

        frameT f;
        int    rc = readFrame_( f, remaining );
        if( rc != 0 )
        {
            return rc;
        }

        if( f.addr == m_addr && f.header == header )
        {
            reply = f;
            return 0;
        }

        handleFrame_( f );
    }
}

/* ---------- Protocol queries ---------- */

inline void elliptecCtrl::handleFrame_( const frameT &frame )
{
    if( frame.addr != m_addr )
    {
        return;
    }

    if( frame.header == "GS" )
    {
        if( frame.payload.size() >= 2 && std::isxdigit( (unsigned char)frame.payload[0] ) &&
            std::isxdigit( (unsigned char)frame.payload[1] ) )
        {
            handleStatusCode_( static_cast<uint8_t>( std::stoul( frame.payload.substr( 0, 2 ), nullptr, 16 ) ) );
        }
    }
    else if( frame.header == "PO" )
    {
        parsePosition_( frame.payload );
    }
    else if( frame.header == "IN" )
    {
        parseInfo_( frame.payload );
    }
}

inline void elliptecCtrl::handleStatusCode_( uint8_t code )
{
    m_gs = code;

    if( code == 0x09 )
    {
        if( m_pending != Pending::None )
        {
            m_sawBusy = true;
        }
        return;
    }

    if( code == 0x00 )
    {
        return;
    }

    if( m_pending != Pending::None )
    {
        if( m_latchedError == 0 )
        {
            m_latchedError = code;
        }
        return;
    }

    const std::string text = "Error: " + statusMeaning( code );
    if( m_errorText != text )
    {
        log<text_log>( "device reported status " + std::to_string( (int)code ) + " (" + statusMeaning( code ) + ")",
                       logPrio::LOG_WARNING );
        m_errorText = text;
    }
}

inline void elliptecCtrl::parsePosition_( const std::string &payload )
{
    if( payload.size() < 8 )
    {
        return;
    }

    uint32_t v = 0;
    for( size_t i = 0; i < 8; ++i )
    {
        char c = (char)std::toupper( (unsigned char)payload[i] );
        if( !std::isxdigit( (unsigned char)c ) )
        {
            return;
        }
        v = ( v << 4 ) | (uint32_t)( ( c <= '9' ) ? ( c - '0' ) : ( 10 + ( c - 'A' ) ) );
    }

    m_posPulses = static_cast<int32_t>( v );
    m_posDeg    = wrapDeg( pulsesToDeg_( m_posPulses ) );
}

inline void elliptecCtrl::parseInfo_( const std::string &payload )
{
    if( payload.size() >= 2 && std::isxdigit( (unsigned char)payload[0] ) &&
        std::isxdigit( (unsigned char)payload[1] ) )
    {
        uint8_t model = static_cast<uint8_t>( std::stoul( payload.substr( 0, 2 ), nullptr, 16 ) );
        m_model       = model;
    }

    if( m_pulsesPerRevConfigured || payload.size() < 8 )
    {
        return;
    }

    const std::string tail = payload.substr( payload.size() - 8 );
    for( char c : tail )
    {
        if( !std::isxdigit( (unsigned char)c ) )
        {
            return;
        }
    }

    uint32_t v = static_cast<uint32_t>( std::stoul( tail, nullptr, 16 ) );
    if( v )
    {
        m_pulsesPerRev = v;
    }
}

inline int elliptecCtrl::q_info_()
{
    frameT r;
    int    rc = query_( "in", "IN", r, m_readTimeoutMs );
    if( rc != 0 )
    {
        return rc;
    }

    parseInfo_( r.payload );
    return 0;
}

inline int elliptecCtrl::q_status_()
{
    frameT r;
    int    rc = query_( "gs", "GS", r, m_readTimeoutMs );
    if( rc != 0 )
    {
        return rc;
    }

    handleFrame_( r );
    return 0;
}

inline int elliptecCtrl::q_position_()
{
    frameT r;
    int    rc = query_( "gp", "PO", r, m_readTimeoutMs );
    if( rc != 0 )
    {
        return rc;
    }

    parsePosition_( r.payload );
    return 0;
}

/* ---------- Commands ---------- */

inline int elliptecCtrl::cmd_home_( uint8_t dir )
{
    char nib = "0123456789ABCDEF"[dir & 0xF];
    if( send_( std::string( "ho" ) + nib ) < 0 )
    {
        return -1;
    }

    beginPending_( Pending::Home );
    m_statusHint = "Homing";
    return 0;
}

inline int elliptecCtrl::cmd_stop_()
{
    frameT r;
    if( query_( "st", "GS", r, m_readTimeoutMs ) != 0 )
    {
        return -1;
    }

    if( r.payload.size() < 2 || !std::isxdigit( (unsigned char)r.payload[0] ) ||
        !std::isxdigit( (unsigned char)r.payload[1] ) )
    {
        return -1;
    }

    int code = static_cast<int>( std::stoul( r.payload.substr( 0, 2 ), nullptr, 16 ) );
    if( code == 0x09 && m_pending != Pending::None )
    {
        m_sawBusy = true;
    }
    return code;
}

inline int elliptecCtrl::cmd_optimize_()
{
    if( send_( m_cmdOptimize ) < 0 )
    {
        return -1;
    }

    beginPending_( Pending::Optimize );
    m_statusHint = "Optimizing motors (may take minutes)";
    return 0;
}

inline int elliptecCtrl::cmd_save_()
{
    frameT r;
    if( query_( m_cmdSave, "GS", r, m_readTimeoutMs ) != 0 )
    {
        return -1;
    }

    return ( r.payload.compare( 0, 2, "00" ) == 0 ) ? 0 : -1;
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

    frameT r;
    int    rc = query_( std::string( "sv" ) + buf, "GS", r, m_readTimeoutMs );
    if( rc != 0 )
    {
        return rc;
    }

    return ( r.payload.compare( 0, 2, "00" ) == 0 ) ? 0 : 1;
}

inline int elliptecCtrl::cmd_moveAbs_pulses_( int32_t pulses )
{
    char hex[9];
    std::snprintf( hex, sizeof( hex ), "%08X", (uint32_t)pulses );
    if( send_( std::string( "ma" ) + hex ) < 0 )
    {
        return -1;
    }

    beginPending_( Pending::Move );
    return 0;
}

inline int elliptecCtrl::cmd_moveRel_pulses_( int32_t pulses )
{
    char hex[9];
    std::snprintf( hex, sizeof( hex ), "%08X", (uint32_t)pulses );
    if( send_( std::string( "mr" ) + hex ) < 0 )
    {
        return -1;
    }

    beginPending_( Pending::Move );
    return 0;
}

/* ---------- Motion helpers ---------- */

inline bool elliptecCtrl::motionAllowed_( const std::string &what )
{
    if( !m_connected )
    {
        log<text_log>( what + " rejected: not connected", logPrio::LOG_WARNING );
        return false;
    }

    if( m_pulsesPerRev == 0 )
    {
        log<text_log>( what + " rejected: pulses per revolution unknown", logPrio::LOG_ERROR );
        return false;
    }

    if( m_pending == Pending::Optimize )
    {
        log<text_log>( what + " rejected: motor optimization in progress", logPrio::LOG_WARNING );
        return false;
    }

    if( m_pending == Pending::Home || m_pending == Pending::HomeOffset )
    {
        log<text_log>( what + " rejected: homing in progress", logPrio::LOG_WARNING );
        return false;
    }

    return true;
}

inline void elliptecCtrl::beginPending_( Pending p )
{
    m_pending       = p;
    m_pendingStart  = std::chrono::steady_clock::now();
    m_sawBusy       = false;
    m_latchedError  = 0;
    m_stopRequested = false;
    m_errorText.clear();
    m_moving = ( p == Pending::Home || p == Pending::HomeOffset ) ? 2 : 1;
}

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
    int32_t p = degToPulses_( wrapDeg( deg ) );
    if( m_pulsesPerRev > 0 && p >= static_cast<int32_t>( m_pulsesPerRev ) )
    {
        p -= static_cast<int32_t>( m_pulsesPerRev );
    }
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

inline int elliptecCtrl::applyVelocityFloor_( int pct )
{
    if( pct > 100 )
    {
        pct = 100;
    }

    if( pct < 0 )
    {
        pct = 0;
    }

    int minVel = minVelocityForModel( m_model );
    if( pct < minVel )
    {
        log<text_log>( "velocity " + std::to_string( pct ) + "% is below the ELL" + std::to_string( (int)m_model ) +
                           " minimum; using " + std::to_string( minVel ) + "%",
                       logPrio::LOG_WARNING );
        pct = minVel;
    }

    return pct;
}

inline int elliptecCtrl::presetIndexForPosition_( double deg ) const
{
    const double tol = presetToleranceDeg_();

    for( size_t i = 0; i < m_presetPositions.size(); ++i )
    {
        if( circularDistanceDeg( static_cast<double>( m_presetPositions[i] ), deg ) <= tol )
        {
            return static_cast<int>( i );
        }
    }

    return -1;
}

inline double elliptecCtrl::presetToleranceDeg_() const
{
    return m_presetToleranceDeg;
}

inline int elliptecCtrl::validatePresets_()
{
    for( size_t i = 0; i < m_presetPositions.size(); ++i )
    {
        for( size_t j = i + 1; j < m_presetPositions.size(); ++j )
        {
            double d = circularDistanceDeg( m_presetPositions[i], m_presetPositions[j] );
            if( d > 0 && d < 2 * m_presetToleranceDeg )
            {
                std::string n1 = ( i < m_presetNames.size() ) ? m_presetNames[i] : std::to_string( i );
                std::string n2 = ( j < m_presetNames.size() ) ? m_presetNames[j] : std::to_string( j );
                return log<text_log, -1>( "presets " + n1 + " and " + n2 +
                                              " are closer than twice stage.presetTolerance",
                                          logPrio::LOG_CRITICAL );
            }
        }
    }

    return 0;
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
    if( drainFrames_() < 0 )
    {
        return -1;
    }

    int rcP = q_position_();
    int rcS = q_status_();

    // Hard errors => reconnect
    if( rcP < 0 || rcS < 0 )
    {
        return -1;
    }

    // Missed replies => debounce; reconnect if too many in a row
    if( rcP > 0 || rcS > 0 )
    {
        if( m_pending == Pending::Optimize )
        {
            // The bus is occupied during optimization; silence is expected until the routine times out.
            double elapsed = std::chrono::duration<double>( std::chrono::steady_clock::now() - m_pendingStart ).count();
            if( elapsed < m_optimizeTimeoutSec )
            {
                updateStatus_();
                return +1;
            }

            failPending_( "motor optimization timed out" );
        }

        if( ++m_commMisses <= m_commMaxMisses )
        {
            updateStatus_();
            return +1;
        }

        m_commMisses = 0;
        return -1;
    }

    m_commMisses = 0;

    resolvePending_();

    syncPresetState_();
    syncControllerState_();
    updateStatus_();
    return 0;
}

inline void elliptecCtrl::resolvePending_()
{
    if( m_pending == Pending::None )
    {
        // Busy with nothing commanded: external motion, or a move still running after a reconnect.
        m_moving = ( m_gs == 0x09 ) ? 1 : ( m_homed ? 0 : -1 );
        return;
    }

    if( m_latchedError != 0 )
    {
        failPending_( statusMeaning( m_latchedError ) );
        return;
    }

    if( m_gs == 0x09 )
    {
        m_moving = ( m_pending == Pending::Home || m_pending == Pending::HomeOffset ) ? 2 : 1;
        return;
    }

    if( m_gs != 0x00 )
    {
        return;
    }

    // Idle reported. Guard against a status read taken before the device started the command.
    double elapsed = std::chrono::duration<double>( std::chrono::steady_clock::now() - m_pendingStart ).count();
    if( !m_sawBusy && elapsed < 0.2 )
    {
        return;
    }

    completePending_();
}

inline void elliptecCtrl::completePending_()
{
    switch( m_pending )
    {
    case Pending::Home:
        if( m_homeOffsetDeg != 0.0 )
        {
            int32_t p = degToPulses_( m_homeOffsetDeg );
            if( cmd_moveAbs_pulses_( p ) < 0 )
            {
                failPending_( "post-home offset move could not be sent" );
                return;
            }

            beginPending_( Pending::HomeOffset );
            m_statusHint = "Homed. Applying offset";
            recordStage( true );
            recordPosition( true );
            return;
        }

        m_homed = true;
        log<text_log>( "homing complete" );
        break;

    case Pending::HomeOffset:
        m_homed = true;
        log<text_log>( "homing complete" );
        break;

    case Pending::Optimize:
        m_homed = false;
        log<text_log>( "motor optimization complete; the stage must be re-homed", logPrio::LOG_NOTICE );
        break;

    case Pending::Move:
    case Pending::None:
        break;
    }

    m_pending       = Pending::None;
    m_stopRequested = false;
    m_moving        = m_homed ? 0 : -1;
    m_statusHint.clear();
    recordStage( true );
    recordPosition( true );
}

inline void elliptecCtrl::failPending_( const std::string &reason )
{
    std::string what = "move";
    if( m_pending == Pending::Home || m_pending == Pending::HomeOffset )
    {
        what    = "home";
        m_homed = false;
    }
    else if( m_pending == Pending::Optimize )
    {
        what    = "optimize";
        m_homed = false;
    }

    log<text_log>( what + " failed: " + reason, logPrio::LOG_ERROR );

    m_pending       = Pending::None;
    m_latchedError  = 0;
    m_stopRequested = false;
    m_moving        = m_homed ? 0 : -1;
    m_statusHint.clear();
    m_errorText = "Error: " + what + " failed (" + reason + ")";
    clearPresetNameTracking();
    m_preset_target = 0;
    recordStage( true );
    recordPosition( true );
}

inline void elliptecCtrl::updateStatus_()
{
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
        s = ( !m_statusHint.empty() ? m_statusHint : "Homing" );
    }
    else if( m_moving == 1 )
    {
        s = ( !m_statusHint.empty() ? m_statusHint : "Busy" );
    }
    else if( !m_errorText.empty() )
    {
        s = m_errorText;
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

    if( !m_indiDriver )
    {
        return;
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
