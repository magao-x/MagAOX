/** \file elliptecLowLevel.hpp
 * \brief The MagAO-X low-level controller for a bus of Thorlabs Elliptec devices.
 * \author Miles Lucas
 *
 * \ingroup elliptecLowLevel_files
 */

#ifndef elliptecLowLevel_hpp
#define elliptecLowLevel_hpp

#include <sys/file.h>

#include <map>

#include "../../libMagAOX/libMagAOX.hpp"
#include "../../magaox_git_version.h"

#include "elliptecDevice.hpp"

/** \defgroup elliptecLowLevel low-level Elliptec controller
 * \brief The single owner of a serial bus of Thorlabs Elliptec devices.
 *
 * \ingroup apps
 */

/** \defgroup elliptecLowLevel_files low-level Elliptec controller files
 * \ingroup elliptecLowLevel
 */

namespace MagAOX
{
namespace app
{

/// The low-level controller for a bus of Thorlabs Elliptec devices.
/** Several Elliptec devices can share one serial bus behind one USB adapter, distinguished by an address nibble.
 * Only one process may use the port, so this app owns it and the PDU power channel, and exposes every device to
 * `elliptecCtrl` clients through shared INDI properties with one element per stage, in the style of `zaberLowLevel`:
 *
 * | Property | Direction | Meaning |
 * |---|---|---|
 * | `curr_state` | published | POWEROFF, POWERON, NODEVICE, NOTCONNECTED, NOTHOMED, HOMING, OPERATING, READY |
 * | `curr_pos` | published | position in degrees, wrapped into [0, 360) |
 * | `last_error` | published | why the last operation failed, empty when OK |
 * | `tgt_pos` | new | absolute target in degrees |
 * | `req_home` | new | home the stage |
 * | `req_halt` | new | send `st` (only stops continuous motion on most models) |
 * | `velocity` | new | velocity percent, raised to the model minimum |
 * | `req_optimize`, `req_save` | new | maintenance, only with `stage.enableMaintenance` |
 *
 * Each loop polls `gp` and `gs` for every configured device in turn. Per-device state, completion by position, and
 * failure isolation live in `elliptecDevice`.
 *
 * Only one device on the bus moves at a time, so the mounts never draw motor current together. A move, home, or
 * optimize request is held by its device, replacing any request it already holds, and is sent when no device is
 * moving, including the device itself, which the ELL21 does not reliably retarget mid-move. While its request is held
 * a device reports OPERATING (HOMING for a home). When several devices hold requests they are served round-robin, so
 * a device that keeps receiving targets cannot keep the others waiting. A request is checked against the device's
 * own state when it arrives and again when it is sent; a halt drops it. Halt, velocity, and save are sent at once.
 *
 * Threading: all serial I/O is serialized by `m_indiMutex`.
 *
 * \ingroup elliptecLowLevel
 */
class elliptecLowLevel : public MagAOXApp<>, public tty::usbDevice
{
  public:
    /// Construct the controller with power management enabled.
    elliptecLowLevel();

    /// Destroy the controller.
    ~elliptecLowLevel() noexcept
    {
    }

    /// Set up the application configuration.
    virtual void setupConfig() override;

    /// Implementation of configuration loading.
    /**
     * \returns 0 on success
     * \returns -1 on an invalid configuration
     */
    int loadConfigImpl( mx::app::appConfigurator &_config /**< [in] application configuration from which to load */ );

    /// Load the application configuration.
    virtual void loadConfig() override;

    /// Register the shared INDI properties.
    virtual int appStartup() override;

    /// Execute one loop: find and open the port, identify devices, and poll them.
    virtual int appLogic() override;

    /// Close the port.
    virtual int appShutdown() override;

    /// Close the port and reset every device on power-off.
    virtual int onPowerOff() override;

    /// Keep publishing POWEROFF while powered off.
    virtual int whilePowerOff() override;

    /// Handle a new absolute target.
    INDI_NEWCALLBACK_DECL( elliptecLowLevel, m_indiP_tgt_pos );

    /// Handle a home request.
    INDI_NEWCALLBACK_DECL( elliptecLowLevel, m_indiP_req_home );

    /// Handle a halt request.
    INDI_NEWCALLBACK_DECL( elliptecLowLevel, m_indiP_req_halt );

    /// Handle a new velocity.
    INDI_NEWCALLBACK_DECL( elliptecLowLevel, m_indiP_velocity );

    /// Handle an optimize request.
    INDI_NEWCALLBACK_DECL( elliptecLowLevel, m_indiP_req_optimize );

    /// Handle a save request.
    INDI_NEWCALLBACK_DECL( elliptecLowLevel, m_indiP_req_save );

  protected:
    /** \name Configurable Parameters - Data
     * @{
     */

    /// Timeout for a query reply from an idle device, in milliseconds.
    int m_readTimeoutMs{ 1000 };

    /// Timeout for a query reply from a device with an operation in progress, in milliseconds.
    /** Moving ELL21 devices mostly do not answer, so a short timeout keeps a moving device from stalling the bus. */
    int m_busyTimeoutMs{ 200 };

    /// Consecutive missed polls tolerated from an idle device before it is re-identified.
    int m_commMaxMisses{ 3 };

    /// Whether every transmitted and received frame is logged at `LOG_DEBUG2`.
    bool m_serialTrace{ false };

    /// Maximum duration of a move or home before it fails, in seconds.
    double m_moveTimeoutSec{ 30 };

    /// Maximum duration of an optimize routine before it fails, in seconds.
    double m_optimizeTimeoutSec{ 600 };

    /// How close to its target, in degrees, a stopped device must be for a move to count as arrived.
    double m_targetToleranceDeg{ 0.5 };

    /// Whether the maintenance commands (optimize, save) are enabled.
    bool m_enableMaintenance{ false };

    /// Device command used to run the optimize routine.
    std::string m_cmdOptimize{ "om" };

    /// Device command used to save parameters to non-volatile memory.
    std::string m_cmdSave{ "us" };

    ///@}

    /** \name Runtime State - Data
     * @{
     */

    /// Open file descriptor for the serial port, or -1.
    int m_fd{ -1 };

    /// Received bytes not yet assembled into a frame.
    std::string m_rxBuf;

    /// The configured devices, in configuration order.
    std::vector<elliptecDevice> m_devices;

    /// Map from stage name to index in `m_devices`.
    std::map<std::string, size_t> m_deviceByName;

    /// Whether a failure to open the port has been logged since the last successful open.
    bool m_openFailLogged{ false };

    /// Consecutive loops with the port open but no device identified.
    int m_quietLoops{ 0 };

    /// Loops with no device identified before the port is closed and reopened.
    static constexpr int s_maxQuietLoops{ 30 };

    /// Index in `m_devices` of the device whose held request is considered first by `dispatch_()`.
    size_t m_nextDispatch{ 0 };

    ///@}

    /** \name INDI Properties - Data
     * @{
     */

    /// Published state string per stage.
    pcf::IndiProperty m_indiP_curr_state;

    /// Published position in degrees per stage.
    pcf::IndiProperty m_indiP_curr_pos;

    /// Published reason the last operation failed per stage.
    pcf::IndiProperty m_indiP_last_error;

    /// Absolute target in degrees per stage.
    pcf::IndiProperty m_indiP_tgt_pos;

    /// Home request per stage.
    pcf::IndiProperty m_indiP_req_home;

    /// Halt request per stage.
    pcf::IndiProperty m_indiP_req_halt;

    /// Velocity percent per stage.
    pcf::IndiProperty m_indiP_velocity;

    /// Optimize request per stage.
    pcf::IndiProperty m_indiP_req_optimize;

    /// Save request per stage.
    pcf::IndiProperty m_indiP_req_save;

    ///@}

    /// A parsed reply frame.
    struct frameT
    {
        char        addr{ 0 }; ///< Device address character.
        std::string header;    ///< Upper-case two-letter reply header.
        std::string payload;   ///< Data following the header, without CRLF.
    };

    /** \name Serial Helpers
     *
     * All require the caller to hold `m_indiMutex`.
     *
     * @{
     */

    /// Open the port read/write, raw, non-blocking, and lock it against other processes.
    int openPort_();

    /// Close the port and discard partial input.
    void closePort_();

    /// Write raw bytes.
    int writeAll_( const std::string &s /**< [in] bytes to write */ );

    /// Read one well-formed frame; malformed lines are logged at debug level and skipped.
    /**
     * \returns 0 when a frame was read, 1 on timeout, -1 on a hard error
     */
    int readFrame_( frameT &frame,     /**< [out] the parsed frame */
                    int     timeout_ms /**< [in] timeout in milliseconds, 0 for non-blocking */
    );

    /// Send a command to one device without waiting.
    int send_( char               addr, /**< [in] device address */
               const std::string &cmd   /**< [in] command without address */
    );

    /// Dispatch every complete frame already received, without blocking.
    int drainFrames_();

    /// Send a command and wait for a reply from the same address with the given header.
    /** Other frames that arrive first are dispatched to `handleFrame_()`.
     * \returns 0 when the expected frame arrived, 1 on timeout, -1 on a hard error
     */
    int query_( char               addr,      /**< [in] device address */
                const std::string &cmd,       /**< [in] command without address */
                const std::string &header,    /**< [in] expected reply header */
                frameT            &reply,     /**< [out] the expected reply */
                int                timeout_ms /**< [in] timeout in milliseconds */
    );

    /// Route an unsolicited or unexpected frame to its device.
    void handleFrame_( const frameT &frame /**< [in] received frame */ );

    ///@}

    /** \name Device Helpers
     *
     * All require the caller to hold `m_indiMutex`.
     *
     * @{
     */

    /// Device configured at an address, or null.
    elliptecDevice *deviceAt_( char addr /**< [in] device address */ );

    /// Device configured with a stage name, or null.
    elliptecDevice *deviceNamed_( const std::string &name /**< [in] stage name */ );

    /// Identify a device with `in` and set its velocity.
    /**
     * \returns 0 on success or a retryable failure, -1 on a hard port error
     */
    int identify_( elliptecDevice &dev /**< [in] the device */ );

    /// Poll one device's position and status and resolve its operation.
    /**
     * \returns 0 on success or a tolerated miss, -1 on a hard port error
     */
    int poll_( elliptecDevice &dev /**< [in] the device */ );

    /// Log a device event.
    void logEvent_( const elliptecDevice &dev, /**< [in] the device */
                    elliptecDevice::Event e    /**< [in] what happened */
    );

    /// Hold a motion request for a device and send it if the bus is free.
    /** Rejected, with a warning, if the device does not allow motion now.
     * \returns 0 on success or rejection, -1 on a hard port error
     */
    int requestMotion_( elliptecDevice         &dev,    /**< [in] the device */
                        elliptecDevice::Pending op,     /**< [in] the requested operation */
                        double                  deg = 0 /**< [in] target angle, for a move */
    );

    /// If no device is moving, send the next held request, taking devices round-robin.
    /**
     * \returns 0 on success or nothing to send, -1 on a hard port error
     */
    int dispatch_();

    /// Send an absolute move in degrees.
    int moveTo_( elliptecDevice &dev, /**< [in] the device */
                 double          deg  /**< [in] target angle */
    );

    /// Send a home.
    int home_( elliptecDevice &dev /**< [in] the device */ );

    /// Send the optimize routine.
    int optimize_( elliptecDevice &dev /**< [in] the device */ );

    /// Drop any held request and send a stop, recording the reply.
    int halt_( elliptecDevice &dev /**< [in] the device */ );

    /// Set a device's velocity.
    int setVelocity_( elliptecDevice &dev, /**< [in] the device */
                      int             pct  /**< [in] requested velocity percent */
    );

    /// Close the port and mark every device not connected.
    void disconnectAll_();

    /// Publish every device's state, position, error, and velocity.
    void publish_();

    /// Publish one state string for every device.
    void publishAll_( const std::string &st /**< [in] state string */ );

    /// Set an element and send it if an INDI driver is running; the local copy is kept either way.
    template <typename T>
    void setElement_( pcf::IndiProperty &p,  /**< [in/out] property */
                      const std::string &el, /**< [in] element name */
                      const T           &v   /**< [in] new value */
    );

    ///@}
};

/* ========================= impl ========================= */

inline elliptecLowLevel::elliptecLowLevel() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    m_powerMgtEnabled = true;
    m_baudRate        = B9600; // Elliptec default; overridden by usb.baud
}

/* ---------- Config ---------- */

inline void elliptecLowLevel::setupConfig()
{
    tty::usbDevice::setupConfig( config );

    config.add( "serial.readTimeoutMs",
                "1000",
                "serial.readTimeoutMs",
                argType::Optional,
                "serial",
                "readTimeoutMs",
                false,
                "int",
                "Timeout (ms) for a reply from an idle device" );

    config.add( "serial.busyTimeoutMs",
                "200",
                "serial.busyTimeoutMs",
                argType::Optional,
                "serial",
                "busyTimeoutMs",
                false,
                "int",
                "Timeout (ms) for a reply from a moving device" );

    config.add( "serial.trace",
                "false",
                "serial.trace",
                argType::Optional,
                "serial",
                "trace",
                false,
                "bool",
                "Log every frame at LOG_DEBUG2 (also set logger.logLevel=D2)" );

    config.add( "comm.maxPollMisses",
                "3",
                "comm.maxPollMisses",
                argType::Optional,
                "comm",
                "maxPollMisses",
                false,
                "int",
                "Consecutive missed polls from an idle device before it is re-identified" );

    config.add( "motion.moveTimeoutSec",
                "30",
                "motion.moveTimeoutSec",
                argType::Optional,
                "motion",
                "moveTimeoutSec",
                false,
                "double",
                "Maximum duration (s) of a move or home" );

    config.add( "motion.optimizeTimeoutSec",
                "600",
                "motion.optimizeTimeoutSec",
                argType::Optional,
                "motion",
                "optimizeTimeoutSec",
                false,
                "double",
                "Maximum duration (s) of the optimize routine" );

    config.add( "motion.targetTolerance",
                "0.5",
                "motion.targetTolerance",
                argType::Optional,
                "motion",
                "targetTolerance",
                false,
                "double",
                "How close (deg) to its target a stopped device must be for a move to count as arrived" );

    config.add( "stage.enableMaintenance",
                "false",
                "stage.enableMaintenance",
                argType::Optional,
                "stage",
                "enableMaintenance",
                false,
                "bool",
                "Enable the optimize and save maintenance commands" );

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
}

inline int elliptecLowLevel::loadConfigImpl( mx::app::appConfigurator &_config )
{
    // The device may be unpowered at startup, so a missing tty is not an error here; it is looked up again later.
    int rv = tty::usbDevice::loadConfig( _config );
    if( rv == TTY_E_BADBAUDRATE )
    {
        return log<text_log, -1>( "invalid usb.baud", logPrio::LOG_CRITICAL );
    }

    _config( m_readTimeoutMs, "serial.readTimeoutMs" );
    _config( m_busyTimeoutMs, "serial.busyTimeoutMs" );
    _config( m_serialTrace, "serial.trace" );
    _config( m_commMaxMisses, "comm.maxPollMisses" );
    _config( m_moveTimeoutSec, "motion.moveTimeoutSec" );
    _config( m_optimizeTimeoutSec, "motion.optimizeTimeoutSec" );
    _config( m_targetToleranceDeg, "motion.targetTolerance" );
    _config( m_enableMaintenance, "stage.enableMaintenance" );
    _config( m_cmdOptimize, "device.optimizeCmd" );
    _config( m_cmdSave, "device.saveCmd" );

    m_readTimeoutMs = std::max( 1, m_readTimeoutMs );
    m_busyTimeoutMs = std::max( 1, m_busyTimeoutMs );
    m_commMaxMisses = std::max( 1, m_commMaxMisses );

    // Every section with an `address` key is a device on this bus, named by the section.
    std::vector<std::string> sections;
    _config.unusedSections( sections );

    for( const auto &section : sections )
    {
        if( !_config.isSetUnused( mx::app::iniFile::makeKey( section, "address" ) ) )
        {
            continue;
        }

        elliptecDevice dev;
        dev.name( section );

        std::string addr;
        _config.configUnused( addr, mx::app::iniFile::makeKey( section, "address" ) );
        addr.erase( std::remove( addr.begin(), addr.end(), '"' ), addr.end() );
        addr.erase( std::remove_if( addr.begin(), addr.end(), ::isspace ), addr.end() );
        if( addr.size() != 1 || dev.address( addr[0] ) < 0 )
        {
            return log<text_log, -1>( "stage " + section + ": address must be one hex digit 0-F",
                                      logPrio::LOG_CRITICAL );
        }

        std::string serial;
        _config.configUnused( serial, mx::app::iniFile::makeKey( section, "serial" ) );
        serial.erase( std::remove( serial.begin(), serial.end(), '"' ), serial.end() );
        dev.expectedSerial( serial );

        int vel = dev.velPercent();
        _config.configUnused( vel, mx::app::iniFile::makeKey( section, "velPercent" ) );
        dev.velPercent( vel );

        if( deviceAt_( dev.address() ) )
        {
            return log<text_log, -1>( "stage " + section + ": address " + std::string( 1, dev.address() ) +
                                          " is already used",
                                      logPrio::LOG_CRITICAL );
        }

        m_deviceByName[section] = m_devices.size();
        m_devices.push_back( dev );
    }

    if( m_devices.empty() )
    {
        return log<text_log, -1>( "no stages configured (sections with an address key)", logPrio::LOG_CRITICAL );
    }

    return 0;
}

inline void elliptecLowLevel::loadConfig()
{
    if( loadConfigImpl( config ) < 0 )
    {
        m_shutdown = true;
    }
}

/* ---------- Startup/Logic ---------- */

inline int elliptecLowLevel::appStartup()
{
    REG_INDI_NEWPROP_NOCB( m_indiP_curr_state, "curr_state", pcf::IndiProperty::Text );
    REG_INDI_NEWPROP_NOCB( m_indiP_curr_pos, "curr_pos", pcf::IndiProperty::Number );
    REG_INDI_NEWPROP_NOCB( m_indiP_last_error, "last_error", pcf::IndiProperty::Text );

    REG_INDI_NEWPROP( m_indiP_tgt_pos, "tgt_pos", pcf::IndiProperty::Number );

    REG_INDI_NEWPROP( m_indiP_req_home, "req_home", pcf::IndiProperty::Switch );
    m_indiP_req_home.setRule( pcf::IndiProperty::AnyOfMany );

    REG_INDI_NEWPROP( m_indiP_req_halt, "req_halt", pcf::IndiProperty::Switch );
    m_indiP_req_halt.setRule( pcf::IndiProperty::AnyOfMany );

    REG_INDI_NEWPROP( m_indiP_velocity, "velocity", pcf::IndiProperty::Number );

    REG_INDI_NEWPROP( m_indiP_req_optimize, "req_optimize", pcf::IndiProperty::Switch );
    m_indiP_req_optimize.setRule( pcf::IndiProperty::AnyOfMany );

    REG_INDI_NEWPROP( m_indiP_req_save, "req_save", pcf::IndiProperty::Switch );
    m_indiP_req_save.setRule( pcf::IndiProperty::AnyOfMany );

    for( const auto &dev : m_devices )
    {
        const std::string &n = dev.name();

        m_indiP_curr_state.add( pcf::IndiElement( n ) );
        m_indiP_curr_state[n].set( std::string( "NOTCONNECTED" ) );

        m_indiP_curr_pos.add( pcf::IndiElement( n ) );
        m_indiP_curr_pos[n].set( 0.0 );

        m_indiP_last_error.add( pcf::IndiElement( n ) );
        m_indiP_last_error[n].set( std::string( "" ) );

        m_indiP_tgt_pos.add( pcf::IndiElement( n ) );
        m_indiP_tgt_pos[n].set( 0.0 );

        m_indiP_velocity.add( pcf::IndiElement( n ) );
        m_indiP_velocity[n].set( dev.velPercent() );

        for( pcf::IndiProperty *sw :
             { &m_indiP_req_home, &m_indiP_req_halt, &m_indiP_req_optimize, &m_indiP_req_save } )
        {
            sw->add( pcf::IndiElement( n ) );
            ( *sw )[n].setSwitchState( pcf::IndiElement::Off );
        }
    }

    return 0;
}

inline int elliptecLowLevel::appLogic()
{
    if( state() == stateCodes::INITIALIZED )
    {
        state( stateCodes::NODEVICE );
    }

    if( powerState() != 1 || powerStateTarget() != 1 )
    {
        return 0;
    }

    std::lock_guard<std::mutex> guard( m_indiMutex );

    if( state() == stateCodes::POWERON )
    {
        publishAll_( "POWERON" );
        if( !powerOnWaitElapsed() )
        {
            return 0;
        }
        state( stateCodes::NODEVICE );
    }

    if( m_fd < 0 )
    {
        // Find the tty by USB ids, unless a device name was set directly (tests).
        if( !m_idVendor.empty() )
        {
            int rv = tty::usbDevice::getDeviceName();
            if( rv < 0 )
            {
                state( stateCodes::NODEVICE );
                publishAll_( "NODEVICE" );
                return 0;
            }
        }

        if( openPort_() < 0 )
        {
            state( stateCodes::NOTCONNECTED );
            publishAll_( "NOTCONNECTED" );
            return 0;
        }

        state( stateCodes::CONNECTED );
        log<text_log>( "bus opened on " + m_deviceName );
    }

    int rv = 0;

    for( auto &dev : m_devices )
    {
        if( dev.wrongDevice() )
        {
            continue;
        }

        if( !dev.identified() )
        {
            rv = identify_( dev );
        }
        else
        {
            rv = poll_( dev );
        }

        if( rv < 0 )
        {
            break;
        }
    }

    // A device that finished this loop frees the bus for the next held request.
    if( rv >= 0 )
    {
        rv = dispatch_();
    }

    if( rv < 0 )
    {
        log<text_log>( "serial port error; reopening " + m_deviceName, logPrio::LOG_ERROR );
        disconnectAll_();
        state( stateCodes::NOTCONNECTED );
        publish_();
        return 0;
    }

    bool anyIdentified = false;
    for( const auto &dev : m_devices )
    {
        anyIdentified = anyIdentified || dev.identified();
    }
    state( anyIdentified ? stateCodes::READY : stateCodes::CONNECTED );

    // If no device has answered for a long time, reopen the port in case the adapter needs it.
    m_quietLoops = anyIdentified ? 0 : m_quietLoops + 1;
    if( m_quietLoops >= s_maxQuietLoops )
    {
        log<text_log>( "no device on " + m_deviceName + " is answering; reopening the port", logPrio::LOG_WARNING );
        m_quietLoops = 0;
        disconnectAll_();
        state( stateCodes::NOTCONNECTED );
    }

    publish_();
    return 0;
}

inline int elliptecLowLevel::appShutdown()
{
    std::lock_guard<std::mutex> guard( m_indiMutex );
    closePort_();
    return 0;
}

inline int elliptecLowLevel::onPowerOff()
{
    std::lock_guard<std::mutex> guard( m_indiMutex );

    closePort_();
    for( auto &dev : m_devices )
    {
        dev.powerOff();
    }

    publish_();
    publishAll_( "POWEROFF" );
    return 0;
}

inline int elliptecLowLevel::whilePowerOff()
{
    std::lock_guard<std::mutex> guard( m_indiMutex );
    publishAll_( "POWEROFF" );
    return 0;
}

/* ---------- INDI callbacks ---------- */

INDI_NEWCALLBACK_DEFN( elliptecLowLevel, m_indiP_tgt_pos )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_tgt_pos, ipRecv );

    std::lock_guard<std::mutex> guard( m_indiMutex );

    for( auto &dev : m_devices )
    {
        if( ipRecv.find( dev.name() ) )
        {
            if( requestMotion_( dev, elliptecDevice::Pending::Move, ipRecv[dev.name()].get<double>() ) < 0 )
            {
                log<software_error>( { __FILE__, __LINE__, "move failed for " + dev.name() } );
            }
        }
    }

    publish_();
    return 0;
}

INDI_NEWCALLBACK_DEFN( elliptecLowLevel, m_indiP_req_home )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_req_home, ipRecv );

    std::lock_guard<std::mutex> guard( m_indiMutex );

    for( auto &dev : m_devices )
    {
        if( ipRecv.find( dev.name() ) && ipRecv[dev.name()].getSwitchState() == pcf::IndiElement::On )
        {
            if( requestMotion_( dev, elliptecDevice::Pending::Home ) < 0 )
            {
                log<software_error>( { __FILE__, __LINE__, "home failed for " + dev.name() } );
            }
        }
    }

    publish_();
    return 0;
}

INDI_NEWCALLBACK_DEFN( elliptecLowLevel, m_indiP_req_halt )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_req_halt, ipRecv );

    std::lock_guard<std::mutex> guard( m_indiMutex );

    for( auto &dev : m_devices )
    {
        if( ipRecv.find( dev.name() ) && ipRecv[dev.name()].getSwitchState() == pcf::IndiElement::On )
        {
            halt_( dev );
        }
    }

    publish_();
    return 0;
}

INDI_NEWCALLBACK_DEFN( elliptecLowLevel, m_indiP_velocity )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_velocity, ipRecv );

    std::lock_guard<std::mutex> guard( m_indiMutex );

    for( auto &dev : m_devices )
    {
        if( ipRecv.find( dev.name() ) )
        {
            setVelocity_( dev, ipRecv[dev.name()].get<int>() );
        }
    }

    publish_();
    return 0;
}

INDI_NEWCALLBACK_DEFN( elliptecLowLevel, m_indiP_req_optimize )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_req_optimize, ipRecv );

    std::lock_guard<std::mutex> guard( m_indiMutex );

    for( auto &dev : m_devices )
    {
        if( !ipRecv.find( dev.name() ) || ipRecv[dev.name()].getSwitchState() != pcf::IndiElement::On )
        {
            continue;
        }

        if( !m_enableMaintenance )
        {
            log<text_log>( "optimize rejected: maintenance commands are disabled (stage.enableMaintenance)",
                           logPrio::LOG_WARNING );
            continue;
        }

        if( requestMotion_( dev, elliptecDevice::Pending::Optimize ) < 0 )
        {
            log<software_error>( { __FILE__, __LINE__, "optimize failed for " + dev.name() } );
        }
    }

    publish_();
    return 0;
}

INDI_NEWCALLBACK_DEFN( elliptecLowLevel, m_indiP_req_save )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_req_save, ipRecv );

    std::lock_guard<std::mutex> guard( m_indiMutex );

    for( auto &dev : m_devices )
    {
        if( !ipRecv.find( dev.name() ) || ipRecv[dev.name()].getSwitchState() != pcf::IndiElement::On )
        {
            continue;
        }

        if( !m_enableMaintenance )
        {
            log<text_log>( "save rejected: maintenance commands are disabled (stage.enableMaintenance)",
                           logPrio::LOG_WARNING );
            continue;
        }

        if( !dev.identified() || dev.pending() != elliptecDevice::Pending::None )
        {
            log<text_log>( dev.name() + ": save rejected: the stage must be connected and idle", logPrio::LOG_WARNING );
            continue;
        }

        log<text_log>( dev.name() + ": saving device parameters to non-volatile memory", logPrio::LOG_NOTICE );

        frameT r;
        if( query_( dev.address(), m_cmdSave, "GS", r, m_readTimeoutMs ) != 0 || r.payload.compare( 0, 2, "00" ) != 0 )
        {
            log<text_log>( dev.name() + ": save was not acknowledged", logPrio::LOG_ERROR );
        }
    }

    publish_();
    return 0;
}

/* ---------- Serial ---------- */

inline int elliptecLowLevel::openPort_()
{
    closePort_();

    { // privilege scope
        elevatedPrivileges elPriv( this );
        m_fd = ::open( m_deviceName.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK );
    }

    if( m_fd < 0 )
    {
        if( !m_openFailLogged )
        {
            log<text_log>( "could not open " + m_deviceName + ": " + strerror( errno ), logPrio::LOG_ERROR );
            m_openFailLogged = true;
        }
        return -1;
    }

    // Only one process may use the bus: replies would otherwise be split between readers.
    if( ::flock( m_fd, LOCK_EX | LOCK_NB ) < 0 )
    {
        if( !m_openFailLogged )
        {
            log<text_log>( m_deviceName + " is in use by another process", logPrio::LOG_ERROR );
            m_openFailLogged = true;
        }
        ::close( m_fd );
        m_fd = -1;
        return -1;
    }

    termios tio{};
    if( tcgetattr( m_fd, &tio ) < 0 )
    {
        closePort_();
        return -1;
    }
    cfmakeraw( &tio );
    cfsetispeed( &tio, m_baudRate );
    cfsetospeed( &tio, m_baudRate );
    tio.c_cflag |= ( CLOCAL | CREAD );
    tio.c_cflag &= ~CRTSCTS;
    tio.c_cc[VMIN]  = 0;
    tio.c_cc[VTIME] = 0;
    if( tcsetattr( m_fd, TCSANOW, &tio ) < 0 )
    {
        closePort_();
        return -1;
    }
    tcflush( m_fd, TCIOFLUSH );

    m_rxBuf.clear();
    m_openFailLogged = false;
    return 0;
}

inline void elliptecLowLevel::closePort_()
{
    if( m_fd >= 0 )
    {
        ::close( m_fd );
        m_fd = -1;
    }
    m_rxBuf.clear();
}

inline int elliptecLowLevel::writeAll_( const std::string &s )
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
            if( errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR )
            {
                std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
                continue;
            }
            return -1;
        }
        off += static_cast<size_t>( w );
    }

    return 0;
}

inline int elliptecLowLevel::readFrame_( frameT &frame, int timeout_ms )
{
    if( m_fd < 0 )
    {
        return -1;
    }

    auto start = std::chrono::steady_clock::now();

    while( true )
    {
        size_t eol = m_rxBuf.find( "\r\n" );
        while( eol != std::string::npos )
        {
            std::string line = m_rxBuf.substr( 0, eol );
            m_rxBuf.erase( 0, eol + 2 );

            if( m_serialTrace )
            {
                log<text_log>( "RX: " + line, logPrio::LOG_DEBUG2 );
            }

            if( line.size() >= 3 && std::isxdigit( static_cast<unsigned char>( line[0] ) ) &&
                std::isalpha( static_cast<unsigned char>( line[1] ) ) &&
                std::isalpha( static_cast<unsigned char>( line[2] ) ) )
            {
                frame.addr      = static_cast<char>( std::toupper( static_cast<unsigned char>( line[0] ) ) );
                frame.header    = line.substr( 1, 2 );
                frame.header[0] = static_cast<char>( std::toupper( static_cast<unsigned char>( frame.header[0] ) ) );
                frame.header[1] = static_cast<char>( std::toupper( static_cast<unsigned char>( frame.header[1] ) ) );
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
            return 1;
        }
    }
}

inline int elliptecLowLevel::send_( char addr, const std::string &cmd )
{
    std::string f = std::string( 1, addr ) + cmd;

    if( m_serialTrace )
    {
        log<text_log>( "TX: " + f, logPrio::LOG_DEBUG2 );
    }

    return writeAll_( f );
}

inline int elliptecLowLevel::drainFrames_()
{
    frameT f;
    int    rc;
    while( ( rc = readFrame_( f, 0 ) ) == 0 )
    {
        handleFrame_( f );
    }

    return ( rc < 0 ) ? -1 : 0;
}

inline int
elliptecLowLevel::query_( char addr, const std::string &cmd, const std::string &header, frameT &reply, int timeout_ms )
{
    if( drainFrames_() < 0 )
    {
        return -1;
    }

    if( send_( addr, cmd ) < 0 )
    {
        return -1;
    }

    auto start = std::chrono::steady_clock::now();

    while( true )
    {
        int elapsed = static_cast<int>(
            std::chrono::duration_cast<std::chrono::milliseconds>( std::chrono::steady_clock::now() - start ).count() );
        int remaining = std::max( 0, timeout_ms - elapsed );

        frameT f;
        int    rc = readFrame_( f, remaining );
        if( rc != 0 )
        {
            return rc;
        }

        if( f.addr == addr && f.header == header )
        {
            reply = f;
            return 0;
        }

        handleFrame_( f );
    }
}

inline void elliptecLowLevel::handleFrame_( const frameT &frame )
{
    elliptecDevice *dev = deviceAt_( frame.addr );
    if( !dev )
    {
        return;
    }

    uint32_t code = 0;
    if( frame.header == "GS" && elliptec::parseHex( frame.payload, 0, 2, code ) )
    {
        dev->applyStatus( static_cast<uint8_t>( code ) );
    }
    else if( frame.header == "PO" )
    {
        dev->applyPosition( frame.payload );
    }
}

/* ---------- Devices ---------- */

inline elliptecDevice *elliptecLowLevel::deviceAt_( char addr )
{
    char a = static_cast<char>( std::toupper( static_cast<unsigned char>( addr ) ) );
    for( auto &dev : m_devices )
    {
        if( dev.address() == a )
        {
            return &dev;
        }
    }
    return nullptr;
}

inline elliptecDevice *elliptecLowLevel::deviceNamed_( const std::string &name )
{
    auto it = m_deviceByName.find( name );
    return ( it == m_deviceByName.end() ) ? nullptr : &m_devices[it->second];
}

inline int elliptecLowLevel::identify_( elliptecDevice &dev )
{
    frameT r;
    int    rc = query_( dev.address(), "in", "IN", r, m_readTimeoutMs );
    if( rc < 0 )
    {
        return -1;
    }

    if( rc > 0 )
    {
        return 0; // not answering yet (booting, or not present); retry next loop
    }

    switch( dev.applyInfo( r.payload ) )
    {
    case elliptecDevice::InfoResult::Malformed:
        log<text_log>( dev.name() + ": discarding malformed info reply", logPrio::LOG_DEBUG );
        return 0;

    case elliptecDevice::InfoResult::SerialMismatch:
        log<text_log>( dev.name() + ": device at address " + std::string( 1, dev.address() ) + " has serial " +
                           dev.serial() + ", expected " + dev.expectedSerial() + "; it will not be commanded",
                       logPrio::LOG_CRITICAL );
        return 0;

    case elliptecDevice::InfoResult::Ok:
        break;
    }

    log<text_log>( dev.name() + ": connected to ELL" + std::to_string( dev.model() ) + " serial " + dev.serial() +
                   " at address " + std::string( 1, dev.address() ) + " with " + std::to_string( dev.pulsesPerRev() ) +
                   " pulses/rev" );

    setVelocity_( dev, dev.velPercent() );

    // Learn the current position and whether it is already moving.
    return poll_( dev );
}

inline int elliptecLowLevel::poll_( elliptecDevice &dev )
{
    const int tmo = ( dev.pending() != elliptecDevice::Pending::None ) ? m_busyTimeoutMs : m_readTimeoutMs;

    frameT pr;
    int    rcP = query_( dev.address(), "gp", "PO", pr, tmo );
    if( rcP < 0 )
    {
        return -1;
    }
    if( rcP == 0 )
    {
        dev.applyPosition( pr.payload );
    }

    frameT sr;
    int    rcS = query_( dev.address(), "gs", "GS", sr, tmo );
    if( rcS < 0 )
    {
        return -1;
    }
    if( rcS == 0 )
    {
        handleFrame_( sr );
    }

    elliptecDevice::Event e;
    if( rcP == 0 && rcS == 0 )
    {
        e = dev.pollAnswered( m_targetToleranceDeg );
    }
    else
    {
        e = dev.pollMissed( m_moveTimeoutSec, m_optimizeTimeoutSec, m_commMaxMisses );
    }

    logEvent_( dev, e );
    return 0;
}

inline void elliptecLowLevel::logEvent_( const elliptecDevice &dev, elliptecDevice::Event e )
{
    switch( e )
    {
    case elliptecDevice::Event::None:
        return;
    case elliptecDevice::Event::Completed:
        log<text_log>( dev.name() + ": " + dev.eventText() );
        return;
    case elliptecDevice::Event::Failed:
        log<text_log>( dev.name() + ": " + dev.eventText(), logPrio::LOG_ERROR );
        return;
    case elliptecDevice::Event::Lost:
        log<text_log>( dev.name() + ": " + dev.eventText() + "; re-identifying", logPrio::LOG_WARNING );
        return;
    }
}

inline int elliptecLowLevel::moveTo_( elliptecDevice &dev, double deg )
{
    if( !dev.motionAllowed() )
    {
        log<text_log>( dev.name() + ": move rejected: " + dev.eventText(), logPrio::LOG_WARNING );
        return 0;
    }

    int32_t p = dev.degToPulses( deg );
    char    hex[9];
    std::snprintf( hex, sizeof( hex ), "%08X", static_cast<uint32_t>( p ) );

    if( send_( dev.address(), std::string( "ma" ) + hex ) < 0 )
    {
        return -1;
    }

    dev.beginMove( p );
    return 0;
}

inline int elliptecLowLevel::optimize_( elliptecDevice &dev )
{
    if( !dev.motionAllowed() )
    {
        log<text_log>( dev.name() + ": optimize rejected: " + dev.eventText(), logPrio::LOG_WARNING );
        return 0;
    }

    if( send_( dev.address(), m_cmdOptimize ) < 0 )
    {
        return -1;
    }

    dev.beginOptimize();
    log<text_log>( dev.name() + ": motor optimization started; this may take several minutes", logPrio::LOG_NOTICE );
    return 0;
}

inline int elliptecLowLevel::requestMotion_( elliptecDevice &dev, elliptecDevice::Pending op, double deg )
{
    if( !dev.motionAllowed() )
    {
        const char *what = ( op == elliptecDevice::Pending::Home )       ? "home"
                           : ( op == elliptecDevice::Pending::Optimize ) ? "optimize"
                                                                         : "move";
        log<text_log>( dev.name() + ": " + what + " rejected: " + dev.eventText(), logPrio::LOG_WARNING );
        return 0;
    }

    dev.queue( op, deg );

    if( op == elliptecDevice::Pending::Move )
    {
        setElement_( m_indiP_tgt_pos, dev.name(), elliptec::wrapDeg( deg ) );
    }

    return dispatch_();
}

inline int elliptecLowLevel::dispatch_()
{
    for( const auto &dev : m_devices )
    {
        if( dev.moving() != 0 )
        {
            return 0;
        }
    }

    for( size_t i = 0; i < m_devices.size(); ++i )
    {
        size_t          n   = ( m_nextDispatch + i ) % m_devices.size();
        elliptecDevice &dev = m_devices[n];

        elliptecDevice::Pending op = dev.queued();
        if( op == elliptecDevice::Pending::None )
        {
            continue;
        }

        dev.clearQueued();

        int rv = 0;
        switch( op )
        {
        case elliptecDevice::Pending::Move:
            rv = moveTo_( dev, dev.queuedDeg() );
            break;
        case elliptecDevice::Pending::Home:
            rv = home_( dev );
            break;
        default:
            rv = optimize_( dev );
            break;
        }

        if( rv < 0 )
        {
            return -1;
        }

        // A request rejected now (the device changed state while it was held) lets the next device go.
        if( dev.pending() != elliptecDevice::Pending::None )
        {
            m_nextDispatch = ( n + 1 ) % m_devices.size();
            return 0;
        }
    }

    return 0;
}

inline int elliptecLowLevel::home_( elliptecDevice &dev )
{
    if( !dev.motionAllowed() )
    {
        log<text_log>( dev.name() + ": home rejected: " + dev.eventText(), logPrio::LOG_WARNING );
        return 0;
    }

    if( send_( dev.address(), "ho0" ) < 0 )
    {
        return -1;
    }

    dev.beginHome();
    log<text_log>( dev.name() + ": homing" );
    return 0;
}

inline int elliptecLowLevel::halt_( elliptecDevice &dev )
{
    dev.clearQueued();

    if( !dev.identified() )
    {
        return 0;
    }

    frameT   r;
    int      code = -1;
    uint32_t v    = 0;
    if( query_( dev.address(), "st", "GS", r, m_readTimeoutMs ) == 0 && elliptec::parseHex( r.payload, 0, 2, v ) )
    {
        code = static_cast<int>( v );
    }

    if( dev.stopRequested( code ) )
    {
        log<text_log>( dev.name() + ": stop is not supported by the device during this motion; the move will complete",
                       logPrio::LOG_WARNING );
    }
    else
    {
        log<text_log>( dev.name() + ": stop requested" );
    }

    return 0;
}

inline int elliptecLowLevel::setVelocity_( elliptecDevice &dev, int pct )
{
    int applied = dev.velPercent( pct );
    if( applied != pct )
    {
        log<text_log>( dev.name() + ": velocity " + std::to_string( pct ) + "% adjusted to " +
                           std::to_string( applied ) + "% (ELL" + std::to_string( dev.model() ) + " limits)",
                       logPrio::LOG_WARNING );
    }

    if( !dev.identified() )
    {
        return 0;
    }

    char buf[3];
    std::snprintf( buf, sizeof( buf ), "%02X", applied );

    frameT r;
    if( query_( dev.address(), std::string( "sv" ) + buf, "GS", r, m_readTimeoutMs ) != 0 ||
        r.payload.compare( 0, 2, "00" ) != 0 )
    {
        log<text_log>( dev.name() + ": velocity was not acknowledged", logPrio::LOG_WARNING );
        return -1;
    }

    return 0;
}

inline void elliptecLowLevel::disconnectAll_()
{
    closePort_();
    for( auto &dev : m_devices )
    {
        dev.disconnect();
    }
}

inline void elliptecLowLevel::publish_()
{
    for( auto &dev : m_devices )
    {
        setElement_( m_indiP_curr_state, dev.name(), dev.stateString() );
        setElement_( m_indiP_curr_pos, dev.name(), dev.positionDeg() );
        setElement_( m_indiP_last_error, dev.name(), dev.lastError() );
        setElement_( m_indiP_velocity, dev.name(), dev.velPercent() );
    }
}

inline void elliptecLowLevel::publishAll_( const std::string &st )
{
    for( auto &dev : m_devices )
    {
        setElement_( m_indiP_curr_state, dev.name(), st );
    }
}

template <typename T>
void elliptecLowLevel::setElement_( pcf::IndiProperty &p, const std::string &el, const T &v )
{
    if( !m_indiDriver )
    {
        p[el].set( v );
        return;
    }

    indi::updateIfChanged( p, el, v, m_indiDriver, INDI_IDLE );
}

} // namespace app
} // namespace MagAOX

#endif // elliptecLowLevel_hpp
