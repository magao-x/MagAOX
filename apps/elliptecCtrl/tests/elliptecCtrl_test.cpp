/** \file elliptecCtrl_test.cpp
 * \brief Catch2 tests for the elliptecCtrl app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup elliptecCtrl_files
 */

/** \defgroup elliptecCtrl_unit_test elliptecCtrl Unit Tests
 * \brief Unit tests for the elliptecCtrl application.
 *
 * \ingroup app_unit_test
 */

#include "../../../tests/catch2/catch.hpp"
#include "../../../tests/testXWC.hpp"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../elliptecCtrl.hpp"

namespace libXWCTest
{
namespace elliptecCtrlTest
{

namespace
{

/// Build a per-test scratch directory under `/tmp`.
std::string testRoot( const std::string &name )
{
    std::string root = "/tmp/elliptecCtrl_test/" + name;
    std::filesystem::remove_all( root );
    std::filesystem::create_directories( root + "/telem" );
    std::filesystem::create_directories( root + "/calib" );
    return root;
}

/// Minimal fake Elliptec controller backed by a PTY master.
class fakeElliptecDevice
{
  public:
    /// Response handler type keyed on the raw command bytes written by the app.
    typedef std::function<std::string( const std::string & )> handlerT;

    /// Construct the fake device and start its responder thread.
    explicit fakeElliptecDevice( handlerT handler = handlerT{} ) : m_handler( handler )
    {
        m_masterFd = ::posix_openpt( O_RDWR | O_NOCTTY | O_NONBLOCK );
        if( m_masterFd < 0 )
        {
            throw std::runtime_error( "posix_openpt failed" );
        }

        if( ::grantpt( m_masterFd ) != 0 )
        {
            throw std::runtime_error( "grantpt failed" );
        }

        if( ::unlockpt( m_masterFd ) != 0 )
        {
            throw std::runtime_error( "unlockpt failed" );
        }

        char slaveName[256];
        if( ::ptsname_r( m_masterFd, slaveName, sizeof( slaveName ) ) != 0 )
        {
            throw std::runtime_error( "ptsname_r failed" );
        }

        m_slavePath = slaveName;
        m_thread    = std::thread( [this]() { run(); } );
    }

    /// Stop the responder thread and close the PTY master.
    ~fakeElliptecDevice()
    {
        m_stop = true;

        if( m_thread.joinable() )
        {
            m_thread.join();
        }

        if( m_masterFd >= 0 )
        {
            ::close( m_masterFd );
        }
    }

    /// Return the PTY slave path the app should open.
    const std::string &slavePath() const
    {
        return m_slavePath;
    }

    /// Replace the current response handler.
    void handler( handlerT newHandler /**< [in] replacement command handler */ )
    {
        std::lock_guard<std::mutex> lock( m_mutex );
        m_handler = newHandler;
    }

    /// Return the commands observed so far.
    std::vector<std::string> commands() const
    {
        std::lock_guard<std::mutex> lock( m_mutex );
        return m_commands;
    }

    /// Clear the recorded command history.
    void clearCommands()
    {
        std::lock_guard<std::mutex> lock( m_mutex );
        m_commands.clear();
    }

  private:
    int                      m_masterFd{ -1 }; ///< PTY master file descriptor.
    std::string              m_slavePath;      ///< PTY slave path to open from the app.
    std::thread              m_thread;         ///< Responder thread.
    mutable std::mutex       m_mutex;          ///< Guards handler and command history.
    std::atomic<bool>        m_stop{ false };  ///< Signals responder shutdown.
    handlerT                 m_handler;        ///< Current command-response handler.
    std::vector<std::string> m_commands;       ///< Raw commands observed from the app.

    /// Write the full reply back to the PTY master.
    void writeReply( const std::string &reply )
    {
        size_t off = 0;

        while( off < reply.size() )
        {
            ssize_t written = ::write( m_masterFd, reply.data() + off, reply.size() - off );
            if( written > 0 )
            {
                off += static_cast<size_t>( written );
                continue;
            }

            if( written < 0 && ( errno == EAGAIN || errno == EWOULDBLOCK ) )
            {
                std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
                continue;
            }

            break;
        }
    }

    /// Responder loop servicing one raw Elliptec frame at a time.
    void run()
    {
        while( !m_stop )
        {
            struct pollfd pfd
            {
                m_masterFd, POLLIN, 0
            };

            int pr = ::poll( &pfd, 1, 20 );
            if( pr <= 0 )
            {
                continue;
            }

            char        buf[256];
            std::string cmd;

            while( true )
            {
                ssize_t readCount = ::read( m_masterFd, buf, sizeof( buf ) );
                if( readCount > 0 )
                {
                    cmd.append( buf, buf + readCount );
                    continue;
                }

                if( readCount < 0 && ( errno == EAGAIN || errno == EWOULDBLOCK ) )
                {
                    break;
                }

                break;
            }

            if( cmd.empty() )
            {
                continue;
            }

            handlerT handler;
            {
                std::lock_guard<std::mutex> lock( m_mutex );
                m_commands.push_back( cmd );
                handler = m_handler;
            }

            if( handler )
            {
                std::string reply = handler( cmd );
                if( !reply.empty() )
                {
                    writeReply( reply );
                }
            }
        }
    }
};

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
/// Test harness exposing the protected helpers needed for `elliptecCtrl` unit tests.
class elliptecCtrl_test : public MagAOX::app::elliptecCtrl
{
  public:
    /// Re-export the protected pending-command enum for assertions.
    using pendingT = Pending;

    /// Construct a testable controller instance.
    explicit elliptecCtrl_test( const std::string &device ) : elliptecCtrl_test( device, testRoot( device ) )
    {
    }

    /// Construct a testable controller instance in a caller-supplied scratch root.
    elliptecCtrl_test( const std::string &device, const std::string &root )
    {
        m_configName = device;
        m_basePath   = root;
        m_calibDir   = root + "/calib";
        m_sysPath    = root + "/sys";

        m_tel.logPath( root + "/telem" );
        m_tel.logExt( "bintel" );
        m_tel.logName( m_configName );
        m_tel.m_logLevel = logPrio::LOG_TELEM;
        m_maxInterval    = 3600.0;

        m_startupDelayMs    = 0;
        m_readTimeoutMs     = 25;
        m_busyReadTimeoutMs = 25;
        m_powerState        = 1;
        m_powerTargetState  = 1;

        m_ipAbsDeg.setDevice( m_configName );
        m_ipAbsDeg.setName( "absDeg" );

        m_ipRelDeg.setDevice( m_configName );
        m_ipRelDeg.setName( "relDeg" );

        m_ipRelMove.setDevice( m_configName );
        m_ipRelMove.setName( "relMove" );

        m_ipVelPct.setDevice( m_configName );
        m_ipVelPct.setName( "velocity" );

        m_ipOptimize.setDevice( m_configName );
        m_ipOptimize.setName( "optimize" );

        m_ipSave.setDevice( m_configName );
        m_ipSave.setName( "save" );

        m_indiP_preset.setDevice( m_configName );
        m_indiP_preset.setName( "preset" );

        m_indiP_presetName.setDevice( m_configName );
        m_indiP_presetName.setName( "presetName" );

        m_indiP_home.setDevice( m_configName );
        m_indiP_home.setName( "home" );

        m_indiP_stop.setDevice( m_configName );
        m_indiP_stop.setName( "stop" );
    }

    /// Shut down any open serial or telemetry resources.
    ~elliptecCtrl_test()
    {
        m_tel.logShutdown( true );
        closePort_();
    }

    /// Set up the local INDI FIFO transport and driver.
    int setupINDITransport()
    {
        std::filesystem::create_directories( std::filesystem::path( m_basePath ) / MAGAOX_driverFIFORelPath );
        std::filesystem::create_directories( std::filesystem::path( m_sysPath ) / m_configName );

        if( createINDIFIFOS() < 0 )
        {
            return -1;
        }

        using indiDriverT = std::remove_pointer_t<decltype( m_indiDriver )>;

        m_indiDriver = new indiDriverT( this, m_configName, "0", "0" );
        if( !m_indiDriver || !m_indiDriver->good() )
        {
            return -1;
        }

        return 0;
    }

    /// Prepare preset names and positions for a test.
    void setPresets( const std::vector<float> &positions, const std::vector<std::string> &names )
    {
        m_presetPositions = positions;
        m_presetNames     = names;
    }

    /// Force the app FSM state before calling lifecycle helpers.
    void setFsmState( MagAOX::app::stateCodes::stateCodeT newState /**< [in] new FSM state */ )
    {
        state( newState );
    }

    /// Force the power-management state seen by the app.
    void setPower( int current /**< [in] current power state */, int target /**< [in] target power state */ )
    {
        m_powerState       = current;
        m_powerTargetState = target;
    }

    /// Force the current connected flag.
    void setConnected( bool connected /**< [in] new connected state */ )
    {
        m_connected = connected;
    }

    /// Force the current homed flag.
    void setHomed( bool homed /**< [in] new homed state */ )
    {
        m_homed = homed;
    }

    /// Force the current stage angle in degrees.
    void setPositionDeg( double posDeg /**< [in] stage angle */ )
    {
        m_posDeg = posDeg;
    }

    /// Force the current pulse count.
    void setPositionPulses( int32_t pulses /**< [in] stage pulse count */ )
    {
        m_posPulses = pulses;
    }

    /// Force the device geometry used for degree conversions.
    void setPulsesPerRev( uint32_t pulsesPerRev /**< [in] pulses per revolution */ )
    {
        m_pulsesPerRev = pulsesPerRev;
    }

    /// Force the current pending command.
    void setPending( pendingT pending /**< [in] pending command */ )
    {
        m_pending = pending;
    }

    /// Force the current moving-state code.
    void setMoving( int8_t moving /**< [in] moving code */ )
    {
        m_moving = moving;
    }

    /// Force the current stdMotionStage command classification.
    void setMovingState( int8_t movingState /**< [in] command classification */ )
    {
        m_movingState = movingState;
    }

    /// Force the current preset telemetry fields.
    void setPresetTelemetry( float preset /**< [in] current preset */, float presetTarget /**< [in] target preset */ )
    {
        m_preset        = preset;
        m_preset_target = presetTarget;
    }

    /// Force the configured home-offset move.
    void setHomeOffset( double offsetDeg /**< [in] home offset in degrees */ )
    {
        m_homeOffsetDeg = offsetDeg;
    }

    /// Force the configured relative step size.
    void setRelStepDeg( double relStepDeg /**< [in] relative step in degrees */ )
    {
        m_relStepDeg = relStepDeg;
    }

    /// Force the connection retry counter.
    void setCommMisses( int commMisses /**< [in] number of misses */ )
    {
        m_commMisses = commMisses;
    }

    /// Force the communication miss threshold.
    void setCommMaxMisses( int commMaxMisses /**< [in] miss threshold */ )
    {
        m_commMaxMisses = commMaxMisses;
    }

    /// Force the controller status byte.
    void setGs( uint8_t gs /**< [in] new status byte */ )
    {
        m_gs = gs;
    }

    /// Force the cached status hint.
    void setStatusHint( const std::string &hint /**< [in] new status hint */ )
    {
        m_statusHint = hint;
    }

    /// Force the serial post-write sleep.
    void setPostWriteSleepMs( int postWriteSleepMs /**< [in] post-write delay in milliseconds */ )
    {
        m_postWriteSleepMs = postWriteSleepMs;
    }

    /// Force the startup delay used before the first connection attempt.
    void setStartupDelayMs( unsigned startupDelayMs /**< [in] startup delay in milliseconds */ )
    {
        m_startupDelayMs = startupDelayMs;
    }

    /// Force whether stdMotionStage should use fractional presets.
    void setFractionalPresets( bool fractionalPresets /**< [in] new fractional-preset flag */ )
    {
        m_fractionalPresets = fractionalPresets;
    }

    /// Force whether stdMotionStage should synthesize default preset positions.
    void setDefaultPositions( bool defaultPositions /**< [in] new default-position flag */ )
    {
        m_defaultPositions = defaultPositions;
    }

    /// Force whether the app has seen power on before.
    void setWasPowered( bool wasPowered /**< [in] new flag */ )
    {
        m_wasPowered = wasPowered;
    }

    /// Set the serial-port path that `openPort_()` should open.
    void setPortPath( const std::string &port /**< [in] PTY slave path */ )
    {
        m_port = port;
    }

    /// Set the serial timeouts used by protocol helpers.
    void setTimeouts( int readTimeoutMs /**< [in] default timeout */, int busyTimeoutMs /**< [in] BUSY timeout */ )
    {
        m_readTimeoutMs     = readTimeoutMs;
        m_busyReadTimeoutMs = busyTimeoutMs;
    }

    /// Start the app from the normal `INITIALIZED` state.
    int startup()
    {
        state( MagAOX::app::stateCodes::INITIALIZED );
        return appStartup();
    }

    /// Run one app logic iteration.
    int logic()
    {
        return appLogic();
    }

    /// Expose `moveTo()` for direct test use.
    int moveToDirect( float target /**< [in] preset target */ )
    {
        return moveTo( target );
    }

    /// Run app shutdown.
    int shutdown()
    {
        return appShutdown();
    }

    /// Expose `openPort_()` for test use.
    int openPortDirect()
    {
        return openPort_();
    }

    /// Expose `closePort_()` for test use.
    void closePortDirect()
    {
        closePort_();
    }

    /// Expose `drainInput_()` for test use.
    int drainInputDirect()
    {
        return drainInput_();
    }

    /// Expose `writeAll_()` for test use.
    int writeAllDirect( const std::string &payload /**< [in] raw payload to write */ )
    {
        return writeAll_( payload );
    }

    /// Expose `readFrame_()` for test use.
    int readFrameDirect( std::string &frame, int timeoutMs /**< [in] timeout in milliseconds */ )
    {
        return readFrame_( frame, timeoutMs );
    }

    /// Expose `frame_()` for test use.
    std::string frameDirect( const std::string &cmd /**< [in] payload */ ) const
    {
        return frame_( cmd );
    }

    /// Expose `txrx_()` for test use.
    int txrxDirect( const std::string &cmd, std::string *reply, int timeoutMs /**< [in] timeout in milliseconds */ )
    {
        return txrx_( cmd, reply, timeoutMs );
    }

    /// Expose `q_info_()` for test use.
    int qInfoDirect()
    {
        return q_info_();
    }

    /// Expose `q_status_()` for test use.
    int qStatusDirect()
    {
        return q_status_();
    }

    /// Expose `q_position_()` for test use.
    int qPositionDirect()
    {
        return q_position_();
    }

    /// Expose `cmd_home_()` for test use.
    int cmdHomeDirect( uint8_t dir /**< [in] Elliptec home direction nibble */ )
    {
        return cmd_home_( dir );
    }

    /// Expose `cmd_stop_()` for test use.
    int cmdStopDirect()
    {
        return cmd_stop_();
    }

    /// Expose `cmd_optimize_wait_()` for test use.
    int cmdOptimizeDirect()
    {
        return cmd_optimize_wait_();
    }

    /// Expose `cmd_save_()` for test use.
    int cmdSaveDirect()
    {
        return cmd_save_();
    }

    /// Expose `cmd_setvel_()` for test use.
    int cmdSetVelDirect( int pct /**< [in] velocity percentage */ )
    {
        return cmd_setvel_( pct );
    }

    /// Expose `moveAbsDeg_()` for test use.
    int moveAbsDegDirect( double deg /**< [in] absolute angle */ )
    {
        return moveAbsDeg_( deg );
    }

    /// Expose `moveRelDeg_()` for test use.
    int moveRelDegDirect( double ddeg /**< [in] relative angle */ )
    {
        return moveRelDeg_( ddeg );
    }

    /// Expose `moveRelDegCmdFromRelMove_()` for test use.
    int moveRelDegCmdDirect()
    {
        return moveRelDegCmdFromRelMove_();
    }

    /// Expose `startMoveToDeg_()` for test use.
    int startMoveToDegDirect( double deg /**< [in] absolute angle */ )
    {
        return startMoveToDeg_( deg );
    }

    /// Expose `to_termios_baud_()` for test use.
    speed_t toTermiosBaudDirect( int baud /**< [in] requested baud */ )
    {
        return to_termios_baud_( baud );
    }

    /// Expose `degToPulses_()` for test use.
    int32_t degToPulsesDirect( double deg /**< [in] angle */ ) const
    {
        return degToPulses_( deg );
    }

    /// Expose `pulsesToDeg_()` for test use.
    double pulsesToDegDirect( int32_t pulses /**< [in] native pulses */ ) const
    {
        return pulsesToDeg_( pulses );
    }

    /// Expose `presetIndexForPosition_()` for test use.
    int presetIndexDirect( double deg /**< [in] angle to match */ ) const
    {
        return presetIndexForPosition_( deg );
    }

    /// Expose `presetToleranceDeg_()` for test use.
    double presetToleranceDirect() const
    {
        return presetToleranceDeg_();
    }

    /// Expose `syncPresetState_()` for test use.
    void syncPresetStateDirect()
    {
        syncPresetState_();
    }

    /// Expose `syncControllerState_()` for test use.
    void syncControllerStateDirect()
    {
        syncControllerState_();
    }

    /// Expose `stdMotionStage::updateINDI()` for test use.
    int updateStageINDIDirect()
    {
        return MagAOX::app::dev::stdMotionStage<MagAOX::app::elliptecCtrl>::updateINDI();
    }

    /// Expose `pollDevice_()` for test use.
    int pollDeviceDirect()
    {
        return pollDevice_();
    }

    /// Expose `updateStatus_()` for test use.
    void updateStatusDirect()
    {
        updateStatus_();
    }

    /// Expose `buildStageNamePosText_()` for test use.
    std::string stageNamePosText() const
    {
        return buildStageNamePosText_();
    }

    /// Expose base-class preset-name alias tracking.
    int setPresetAliasIndex( int presetNameIndex /**< [in] alias index */ )
    {
        return setPresetNameTracking( presetNameIndex );
    }

    /// Expose base-class preset-name alias tracking reset.
    void clearPresetAliasIndex()
    {
        clearPresetNameTracking();
    }

    /// Expose the preset name recorded by stage telemetry.
    std::string telemetryPresetNameDirect()
    {
        return telemetryPresetName();
    }

    /// Create an absolute-angle INDI request.
    pcf::IndiProperty absDegRequest( double target /**< [in] target angle */ ) const
    {
        pcf::IndiProperty ip = m_ipAbsDeg;
        ip["target"].set( target );
        return ip;
    }

    /// Create an absolute-angle request with the correct key but no value elements.
    pcf::IndiProperty absDegNoValueRequest() const
    {
        return emptyMatchingProperty_( m_ipAbsDeg );
    }

    /// Create a relative-step INDI request.
    pcf::IndiProperty relDegRequest( double target /**< [in] relative step */ ) const
    {
        pcf::IndiProperty ip = m_ipRelDeg;
        ip["target"].set( target );
        return ip;
    }

    /// Create a relative-step request with the correct key but no value elements.
    pcf::IndiProperty relDegNoValueRequest() const
    {
        return emptyMatchingProperty_( m_ipRelDeg );
    }

    /// Create a velocity INDI request.
    pcf::IndiProperty velPctRequest( int target /**< [in] target velocity percentage */ ) const
    {
        pcf::IndiProperty ip = m_ipVelPct;
        ip["target"].set( target );
        return ip;
    }

    /// Create a velocity request with the correct key but no value elements.
    pcf::IndiProperty velPctNoValueRequest() const
    {
        return emptyMatchingProperty_( m_ipVelPct );
    }

    /// Create a stdMotionStage preset request.
    pcf::IndiProperty presetRequest( float target /**< [in] preset target */ ) const
    {
        pcf::IndiProperty ip = m_indiP_preset;
        ip["target"].set( target );
        return ip;
    }

    /// Create a one-shot request property with `request=On`.
    pcf::IndiProperty requestOn( const pcf::IndiProperty &tmpl /**< [in] template property */ ) const
    {
        pcf::IndiProperty ip = tmpl;
        ip["request"].setSwitchState( pcf::IndiElement::On );
        return ip;
    }

    /// Create a preset-name selection request with exactly one element on.
    pcf::IndiProperty presetNameRequest( const std::string &name /**< [in] selected alias */ ) const
    {
        pcf::IndiProperty ip = m_indiP_presetName;

        for( const auto &presetName : m_presetNames )
        {
            ip[presetName].setSwitchState( pcf::IndiElement::Off );
        }

        ip[name].setSwitchState( pcf::IndiElement::On );
        return ip;
    }

    /// Create a `relMove` request with `request=On`.
    pcf::IndiProperty relMoveRequest() const
    {
        return requestOn( m_ipRelMove );
    }

    /// Create a `relMove` property with the correct key but no request element.
    pcf::IndiProperty relMoveNoRequest() const
    {
        return emptyMatchingProperty_( m_ipRelMove );
    }

    /// Create an `optimize` request with `request=On`.
    pcf::IndiProperty optimizeRequest() const
    {
        return requestOn( m_ipOptimize );
    }

    /// Create an `optimize` property with the correct key but no request element.
    pcf::IndiProperty optimizeNoRequest() const
    {
        return emptyMatchingProperty_( m_ipOptimize );
    }

    /// Create a `save` request with `request=On`.
    pcf::IndiProperty saveRequest() const
    {
        return requestOn( m_ipSave );
    }

    /// Create a `save` property with the correct key but no request element.
    pcf::IndiProperty saveNoRequest() const
    {
        return emptyMatchingProperty_( m_ipSave );
    }

    /// Create a `home` request with `request=On`.
    pcf::IndiProperty homeRequest() const
    {
        return requestOn( m_indiP_home );
    }

    /// Create a `stop` request with `request=On`.
    pcf::IndiProperty stopRequest() const
    {
        return requestOn( m_indiP_stop );
    }

    /// Return the configured serial port path.
    const std::string &port() const
    {
        return m_port;
    }

    /// Read a configuration file into the inherited application configurator.
    void readConfigFile( const std::string &path /**< [in] configuration file path */ )
    {
        config.readConfig( path );
    }

    /// Return the configured baud rate.
    int baud() const
    {
        return m_baud;
    }

    /// Return the configured pulses-per-revolution value.
    uint32_t pulsesPerRev() const
    {
        return m_pulsesPerRev;
    }

    /// Return the configured Elliptec address nibble.
    char addr() const
    {
        return m_addr;
    }

    /// Return the configured default read timeout.
    int readTimeoutMs() const
    {
        return m_readTimeoutMs;
    }

    /// Return the configured BUSY read timeout.
    int busyReadTimeoutMs() const
    {
        return m_busyReadTimeoutMs;
    }

    /// Return the configured velocity percentage.
    int velPercent() const
    {
        return m_velPercent;
    }

    /// Return the configured optimize command.
    const std::string &optimizeCmd() const
    {
        return m_cmdOptimize;
    }

    /// Return the configured save command.
    const std::string &saveCmd() const
    {
        return m_cmdSave;
    }

    /// Return the configured communication miss threshold.
    int commMaxMisses() const
    {
        return m_commMaxMisses;
    }

    /// Return the current shutdown flag.
    int shutdownRequested() const
    {
        return m_shutdown;
    }

    /// Return the current open file descriptor.
    int fd() const
    {
        return m_fd;
    }

    /// Return the current connected state.
    bool connected() const
    {
        return m_connected;
    }

    /// Return the current homed state.
    bool homed() const
    {
        return m_homed;
    }

    /// Return the current pulse count.
    int32_t positionPulses() const
    {
        return m_posPulses;
    }

    /// Return the current position in degrees.
    double positionDeg() const
    {
        return m_posDeg;
    }

    /// Return the current status byte.
    uint8_t gs() const
    {
        return m_gs;
    }

    /// Return the current pending command.
    pendingT pending() const
    {
        return m_pending;
    }

    /// Return the current moving code.
    int8_t moving() const
    {
        return m_moving;
    }

    /// Return the current moving-state classification.
    int8_t movingState() const
    {
        return m_movingState;
    }

    /// Return the cached status hint.
    const std::string &statusHint() const
    {
        return m_statusHint;
    }

    /// Return the current status text property value.
    std::string statusText() const
    {
        return m_ipStatus["current"].getValue();
    }

    /// Return the current status property state.
    pcf::IndiProperty::PropertyStateType statusState() const
    {
        return m_ipStatus.getState();
    }

    /// Return the current preset telemetry value.
    float presetValue() const
    {
        return m_preset;
    }

    /// Return the current preset target telemetry value.
    float presetTargetValue() const
    {
        return m_preset_target;
    }

    /// Return the current communication miss counter.
    int commMisses() const
    {
        return m_commMisses;
    }

    /// Return the currently configured relative move step.
    double relStepDeg() const
    {
        return m_relStepDeg;
    }

    /// Return whether the app has seen power on before.
    bool wasPowered() const
    {
        return m_wasPowered;
    }

    /// Return the numeric preset current element value.
    double presetCurrentElement() const
    {
        return std::stod( m_indiP_preset["current"].getValue() );
    }

    /// Return the numeric preset target element value.
    double presetTargetElement() const
    {
        return std::stod( m_indiP_preset["target"].getValue() );
    }

    /// Return the preset-name property state.
    pcf::IndiProperty::PropertyStateType presetNameState() const
    {
        return m_indiP_presetName.getState();
    }

    /// Return whether a preset-name switch is on.
    bool presetNameOn( const std::string &name /**< [in] preset alias to inspect */ ) const
    {
        return m_indiP_presetName[name].getSwitchState() == pcf::IndiElement::On;
    }

  private:
    /// Create a property with the same unique key as a template but no elements.
    pcf::IndiProperty emptyMatchingProperty_( const pcf::IndiProperty &tmpl /**< [in] template property */ ) const
    {
        pcf::IndiProperty ip( tmpl.getType() );
        ip.setDevice( m_configName );
        ip.setName( tmpl.getName() );
        return ip;
    }
};

/// \endcond

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
/// Test-only CRTP probe used to exercise `stdMotionStage` branches that are difficult to reach through `elliptecCtrl`.
class stdMotionStage_probe : public MagAOX::app::dev::stdMotionStage<stdMotionStage_probe>
{
    friend class MagAOX::app::dev::stdMotionStage<stdMotionStage_probe>;

  public:
    /// Local configurator used by the probe harness.
    mx::app::appConfigurator config;

    /// Null test INDI driver pointer matching the type expected by `stdMotionStage`.
    MagAOX::app::indiDriver<stdMotionStage_probe> *m_indiDriver{ nullptr };

    /// Mutex matching the INDI interface expected by `stdMotionStage`.
    std::mutex m_indiMutex;

    /// Construct a probe instance with a deterministic device name.
    explicit stdMotionStage_probe( const std::string &device ) : m_configName( device )
    {
    }

    /// Return the configured device name.
    const std::string &configName() const
    {
        return m_configName;
    }

    /// Minimal logging shim matching the interface used by `stdMotionStage`.
    template <typename logT, int retval = 0>
    static int log( const typename logT::messageT & )
    {
        return retval;
    }

    /// Minimal logging shim matching the interface used by `stdMotionStage`.
    template <typename logT, int retval = 0, typename... argTs>
    static int log( argTs &&...args )
    {
        static_cast<void>( sizeof...( args ) );
        return retval;
    }

    /// Force the default-position synthesis flag.
    void setDefaultPositions( bool defaultPositions /**< [in] new default-position flag */ )
    {
        m_defaultPositions = defaultPositions;
    }

    /// Force whether preset requests should be integer-valued.
    void setFractionalPresets( bool fractionalPresets /**< [in] new fractional-preset flag */ )
    {
        m_fractionalPresets = fractionalPresets;
    }

    /// Force the configured preset table.
    void setPresets( const std::vector<float> &positions, const std::vector<std::string> &names )
    {
        m_presetPositions = positions;
        m_presetNames     = names;
    }

    /// Force the reported preset index returned to `stdMotionStage`.
    void setReportedPresetNumber( int presetNumber /**< [in] reported preset index */ )
    {
        m_reportedPresetNumber = presetNumber;
    }

    /// Force the current motion bookkeeping fields.
    void setStageTelemetry( int8_t moving /**< [in] moving state */,
                            int8_t movingState /**< [in] command class */,
                            float  preset /**< [in] current preset */ )
    {
        m_moving      = moving;
        m_movingState = movingState;
        m_preset      = preset;
    }

    /// Force the preset-name alias tracking index.
    int setPresetAliasIndex( int presetNameIndex /**< [in] alias index */ )
    {
        return setPresetNameTracking( presetNameIndex );
    }

    /// Read a configuration file into the probe configurator.
    void readConfigFile( const std::string &path /**< [in] configuration file path */ )
    {
        config.readConfig( path );
    }

    /// Run the `stdMotionStage` configuration registration.
    int setupStageConfig()
    {
        return MagAOX::app::dev::stdMotionStage<stdMotionStage_probe>::setupConfig( config );
    }

    /// Run the `stdMotionStage` configuration loader.
    int loadStageConfig()
    {
        return MagAOX::app::dev::stdMotionStage<stdMotionStage_probe>::loadConfig( config );
    }

    /// Run the `stdMotionStage` startup helper.
    int startupStage()
    {
        return MagAOX::app::dev::stdMotionStage<stdMotionStage_probe>::appStartup();
    }

    /// Force the nth property registration to fail.
    void failRegisterOnCall( int callIndex /**< [in] registration call index to fail */ )
    {
        m_failRegisterCall = callIndex;
        m_registerCalls    = 0;
    }

    /// Return the preset positions after loading configuration.
    const std::vector<float> &presetPositions() const
    {
        return m_presetPositions;
    }

    /// Return the captured step size passed to `createStandardIndiNumber`.
    double capturedNumberStep() const
    {
        return m_capturedNumberStep;
    }

    /// Return the captured format passed to `createStandardIndiNumber`.
    const std::string &capturedNumberFormat() const
    {
        return m_capturedNumberFormat;
    }

    /// Return how many times `moveTo()` has been called.
    int moveToCalls() const
    {
        return m_moveToCalls;
    }

    /// Return the most recent `moveTo()` target.
    float lastMoveTarget() const
    {
        return m_lastMoveTarget;
    }

    /// Return how many times `startHoming()` has been called.
    int homeCalls() const
    {
        return m_homeCalls;
    }

    /// Return how many times `stop()` has been called.
    int stopCalls() const
    {
        return m_stopCalls;
    }

    /// Expose the currently tracked preset target.
    float presetTargetValue() const
    {
        return m_preset_target;
    }

    /// Expose the captured active preset name.
    std::string activePresetNameDirect( int presetIndex /**< [in] preset index */ ) const
    {
        return activePresetName( presetIndex );
    }

    /// Expose the preset name used for stage telemetry.
    std::string telemetryPresetNameDirect()
    {
        return telemetryPresetName();
    }

    /// Create a valid numeric preset request.
    pcf::IndiProperty presetRequest( float target /**< [in] preset target */ ) const
    {
        pcf::IndiProperty ip = m_indiP_preset;
        ip["target"].set( target );
        return ip;
    }

    /// Create a preset request with the correct key but no target/current element.
    pcf::IndiProperty presetNoValueRequest() const
    {
        return emptyMatchingProperty_( m_indiP_preset );
    }

    /// Create a partial preset-name request to exercise missing-element handling.
    pcf::IndiProperty presetNamePartialRequest( const std::vector<std::string> &presentNames,
                                                const std::string              &onName ) const
    {
        pcf::IndiProperty ip = emptyMatchingProperty_( m_indiP_presetName );
        ip.setRule( pcf::IndiProperty::OneOfMany );

        for( const auto &name : presentNames )
        {
            ip.add( pcf::IndiElement( name, name == onName ? pcf::IndiElement::On : pcf::IndiElement::Off ) );
        }

        return ip;
    }

    /// Create a home request property with the correct key but no request element.
    pcf::IndiProperty homeNoRequest() const
    {
        return emptyMatchingProperty_( m_indiP_home );
    }

    /// Create a home request property with `request=Off`.
    pcf::IndiProperty homeOffRequest() const
    {
        pcf::IndiProperty ip = m_indiP_home;
        ip["request"].setSwitchState( pcf::IndiElement::Off );
        return ip;
    }

    /// Create a stop request property with the correct key but no request element.
    pcf::IndiProperty stopNoRequest() const
    {
        return emptyMatchingProperty_( m_indiP_stop );
    }

    /// Create a stop request property with `request=Off`.
    pcf::IndiProperty stopOffRequest() const
    {
        pcf::IndiProperty ip = m_indiP_stop;
        ip["request"].setSwitchState( pcf::IndiElement::Off );
        return ip;
    }

    /// Implement the required stop interface.
    int stop()
    {
        ++m_stopCalls;
        return 0;
    }

    /// Implement the required homing interface.
    int startHoming()
    {
        ++m_homeCalls;
        return 0;
    }

    /// Implement the required preset-number interface.
    float presetNumber()
    {
        return static_cast<float>( m_reportedPresetNumber );
    }

    /// Implement the required move interface.
    int moveTo( float target /**< [in] requested preset target */ )
    {
        ++m_moveToCalls;
        m_lastMoveTarget = target;
        return 0;
    }

    /// Minimal numeric-property helper matching the MagAOXApp interface.
    template <typename T>
    int createStandardIndiNumber( pcf::IndiProperty &prop,
                                  const std::string &name,
                                  const T           &min,
                                  const T           &max,
                                  const T           &step,
                                  const std::string &format,
                                  const std::string &label = "",
                                  const std::string &group = "" )
    {
        m_capturedNumberStep   = static_cast<double>( step );
        m_capturedNumberFormat = format;

        prop = pcf::IndiProperty( pcf::IndiProperty::Number );
        prop.setDevice( m_configName );
        prop.setName( name );
        prop.setPerm( pcf::IndiProperty::ReadWrite );
        prop.setState( pcf::IndiProperty::Idle );

        prop.add( pcf::IndiElement( "current" ) );
        prop["current"].setMin( min );
        prop["current"].setMax( max );
        prop["current"].setStep( step );
        if( !format.empty() )
        {
            prop["current"].setFormat( format );
        }

        prop.add( pcf::IndiElement( "target" ) );
        prop["target"].setMin( min );
        prop["target"].setMax( max );
        prop["target"].setStep( step );
        if( !format.empty() )
        {
            prop["target"].setFormat( format );
        }

        if( !label.empty() )
        {
            prop.setLabel( label );
        }

        if( !group.empty() )
        {
            prop.setGroup( group );
        }

        return 0;
    }

    /// Minimal request-switch helper matching the MagAOXApp interface.
    int createStandardIndiRequestSw( pcf::IndiProperty &prop,
                                     const std::string &name,
                                     const std::string &label = "",
                                     const std::string &group = "" )
    {
        prop = pcf::IndiProperty( pcf::IndiProperty::Switch );
        prop.setDevice( m_configName );
        prop.setName( name );
        prop.setPerm( pcf::IndiProperty::ReadWrite );
        prop.setState( pcf::IndiProperty::Idle );
        prop.setRule( pcf::IndiProperty::AtMostOne );
        prop.add( pcf::IndiElement( "request", pcf::IndiElement::Off ) );

        if( !label.empty() )
        {
            prop.setLabel( label );
        }

        if( !group.empty() )
        {
            prop.setGroup( group );
        }

        return 0;
    }

    /// Minimal selection-switch helper matching the MagAOXApp interface.
    int createStandardIndiSelectionSw( pcf::IndiProperty              &prop,
                                       const std::string              &name,
                                       const std::vector<std::string> &elements,
                                       const std::string              &label = "",
                                       const std::string              &group = "" )
    {
        if( elements.empty() )
        {
            return -1;
        }

        prop = pcf::IndiProperty( pcf::IndiProperty::Switch );
        prop.setDevice( m_configName );
        prop.setName( name );
        prop.setPerm( pcf::IndiProperty::ReadWrite );
        prop.setState( pcf::IndiProperty::Idle );
        prop.setRule( pcf::IndiProperty::OneOfMany );

        for( const auto &element : elements )
        {
            prop.add( pcf::IndiElement( element, pcf::IndiElement::Off ) );
        }

        if( !label.empty() )
        {
            prop.setLabel( label );
        }

        if( !group.empty() )
        {
            prop.setGroup( group );
        }

        return 0;
    }

    /// Allow tests to inject registration failures without setting up a real INDI driver.
    int registerIndiPropertyNew( pcf::IndiProperty &prop, int ( *callBack )( void *, const pcf::IndiProperty &ipRecv ) )
    {
        static_cast<void>( prop );
        static_cast<void>( callBack );

        ++m_registerCalls;
        if( m_failRegisterCall > 0 && m_registerCalls == m_failRegisterCall )
        {
            return -1;
        }

        return 0;
    }

    /// Minimal `updateIfChanged` shim used by `stdMotionStage`.
    template <typename T>
    int updateIfChanged( pcf::IndiProperty                          &prop,
                         const std::string                          &element,
                         const T                                    &value,
                         const pcf::IndiProperty::PropertyStateType &state = pcf::IndiProperty::Idle )
    {
        if( !prop.find( element ) )
        {
            prop.add( pcf::IndiElement( element ) );
        }

        prop[element].set( value );
        prop.setState( state );
        return 0;
    }

    /// Minimal `indiTargetUpdate` shim used by the numeric preset callback.
    template <typename T>
    int indiTargetUpdate( pcf::IndiProperty       &localProperty,
                          T                       &localTarget,
                          const pcf::IndiProperty &remoteProperty,
                          bool                     setBusy )
    {
        if( remoteProperty.createUniqueKey() != localProperty.createUniqueKey() )
        {
            return -1;
        }

        if( !( remoteProperty.find( "target" ) || remoteProperty.find( "current" ) ) )
        {
            return -1;
        }

        if( remoteProperty.find( "target" ) )
        {
            localTarget = remoteProperty["target"].get<T>();
        }
        else if( remoteProperty.find( "current" ) )
        {
            localTarget = remoteProperty["current"].get<T>();
        }
        else
        {
            return -1;
        }

        return updateIfChanged( localProperty, "target", localTarget, setBusy ? INDI_BUSY : INDI_IDLE );
    }

    /// Minimal telemetry shim used by `recordStage`.
    template <typename telT>
    int telem( const telT & )
    {
        return 0;
    }

  private:
    std::string m_configName; ///< Probe device name used in generated INDI properties.

    int         m_reportedPresetNumber{ 0 }; ///< Preset index reported back to `stdMotionStage`.
    int         m_moveToCalls{ 0 };          ///< Number of times `moveTo()` was called.
    float       m_lastMoveTarget{ 0.0F };    ///< Most recent move target.
    int         m_homeCalls{ 0 };            ///< Number of times `startHoming()` was called.
    int         m_stopCalls{ 0 };            ///< Number of times `stop()` was called.
    int         m_failRegisterCall{ 0 };     ///< Registration call index that should fail.
    int         m_registerCalls{ 0 };        ///< Count of registration calls observed during startup.
    double      m_capturedNumberStep{ 0.0 }; ///< Step size captured from `createStandardIndiNumber`.
    std::string m_capturedNumberFormat;      ///< Format captured from `createStandardIndiNumber`.

    /// Create a property with the same unique key as a template but no elements.
    pcf::IndiProperty emptyMatchingProperty_( const pcf::IndiProperty &tmpl /**< [in] template property */ ) const
    {
        pcf::IndiProperty ip( tmpl.getType() );
        ip.setDevice( m_configName );
        ip.setName( tmpl.getName() );
        return ip;
    }
};
/// \endcond

/// Build a reply frame for the configured Elliptec address.
std::string elliptecReply( char addr, const std::string &payload )
{
    return std::string( 1, addr ) + payload + "\r\n";
}

} // namespace

/// Verify `elliptecCtrl` configuration loading and helper formatting behavior.
/**
 * \ingroup elliptecCtrl_unit_test
 */
TEST_CASE( "elliptecCtrl loads configuration and formats helper state correctly", "[elliptecCtrl]" )
{
    SECTION( "loadConfig clamps values and helper formatting covers integer and fractional text" )
    {
        elliptecCtrl_test app( "elliptec-config", testRoot( "config" ) );

        app.setupConfig();

        mx::app::writeConfigFile(
            "/tmp/elliptecCtrl_test.conf",
            { "stage", "stage", "stage", "serial", "serial", "stage", "stage", "motion", "device", "device", "comm" },
            { "port",
              "baud",
              "address",
              "readTimeoutMs",
              "busyReadTimeoutMs",
              "homeOffset",
              "allowMultiturn",
              "velPercent",
              "optimizeCmd",
              "saveCmd",
              "maxPollMisses" },
            { "/tmp/elliptec", "115200", "f", "25", "10", "12.5", "true", "140", "xy", "zz", "0" } );

        app.readConfigFile( "/tmp/elliptecCtrl_test.conf" );
        app.loadConfig();

        // clang-format off
#ifdef ELLIPTECCTRL_TEST_DOXYGEN_REF
        MagAOX::app::elliptecCtrl::setupConfig();
        MagAOX::app::elliptecCtrl::loadConfig();
        MagAOX::app::elliptecCtrl::loadConfigImpl( app.config );
        MagAOX::app::elliptecCtrl::frame_( "gs" );
        MagAOX::app::elliptecCtrl::degToPulses_( 90.0 );
        MagAOX::app::elliptecCtrl::pulsesToDeg_( 1024 );
        MagAOX::app::elliptecCtrl::presetToleranceDeg_();
        MagAOX::app::elliptecCtrl::presetIndexForPosition_( 20.0 );
        MagAOX::app::elliptecCtrl::buildStageNamePosText_();
#endif
        // clang-format on

        REQUIRE( app.shutdownRequested() == 0 );
        REQUIRE( app.port() == "/tmp/elliptec" );
        REQUIRE( app.baud() == 115200 );
        REQUIRE( app.addr() == 'F' );
        REQUIRE( app.readTimeoutMs() == 25 );
        REQUIRE( app.busyReadTimeoutMs() == 25 );
        REQUIRE( app.velPercent() == 100 );
        REQUIRE( app.optimizeCmd() == "xy" );
        REQUIRE( app.saveCmd() == "zz" );
        REQUIRE( app.commMaxMisses() == 1 );
        REQUIRE( app.frameDirect( "gs" ) == "Fgs" );

        app.setPulsesPerRev( 4096 );
        app.setPresets( { 10.0F, 20.0F, 20.0F }, { "open", "science", "focus" } );

        REQUIRE( app.degToPulsesDirect( 180.0 ) == 2048 );
        REQUIRE( app.pulsesToDegDirect( 1024 ) == Approx( 90.0 ) );
        REQUIRE( app.presetToleranceDirect() == Approx( 360.0 / 4096.0 / 2.0 ) );
        REQUIRE( app.presetIndexDirect( 20.0 ) == 1 );
        REQUIRE( app.stageNamePosText() == "open:10, science:20, focus:20" );
    }

    SECTION( "loadConfig accepts configured pulses per revolution and negative velocity clamps to zero" )
    {
        elliptecCtrl_test appClamp( "elliptec-config-clamp", testRoot( "config_clamp" ) );

        appClamp.setupConfig();

        mx::app::writeConfigFile( "/tmp/elliptecCtrl_test_clamp.conf",
                                  { "stage", "stage", "serial", "motion", "comm" },
                                  { "port", "pulsesPerRev", "readTimeoutMs", "velPercent", "maxPollMisses" },
                                  { "/tmp/elliptec-clamp", "8192", "25", "-10", "4" } );

        appClamp.readConfigFile( "/tmp/elliptecCtrl_test_clamp.conf" );
        appClamp.loadConfig();

        REQUIRE( appClamp.pulsesPerRev() == 8192 );
        REQUIRE( appClamp.velPercent() == 0 );
    }

    SECTION( "stageNamePos text reports empty and fractional preset tables cleanly" )
    {
        elliptecCtrl_test appStageText( "elliptec-stage-text", testRoot( "stage_text" ) );
        appStageText.setPresets( {}, {} );
        REQUIRE( appStageText.stageNamePosText() == "(none)" );

        appStageText.setPresets( { 12.25F, 20.5F }, { "science", "offset" } );
        REQUIRE( appStageText.stageNamePosText() == "science:12.25, offset:20.5" );
    }
}

/// Verify the shared `stdMotionStage` helper covers its default-position, callback, and startup-failure branches.
/**
 * \ingroup elliptecCtrl_unit_test
 */
TEST_CASE( "stdMotionStage helper probe covers default positions and callback edge cases", "[elliptecCtrl]" )
{
    SECTION( "default preset positions are synthesized and zero entries fall back to their order" )
    {
        stdMotionStage_probe app( "stdmotion-config" );
        app.setDefaultPositions( true );
        REQUIRE( app.setupStageConfig() == 0 );

        mx::app::writeConfigFile( "/tmp/stdMotionStage_probe.conf",
                                  { "stage", "presets", "presets" },
                                  { "powerOnHome", "names", "positions" },
                                  { "true", "open, science, focus", "0, 5, 0" } );

        app.readConfigFile( "/tmp/stdMotionStage_probe.conf" );
        REQUIRE( app.loadStageConfig() == 0 );
        REQUIRE( app.presetPositions().size() == 3 );
        REQUIRE( app.presetPositions()[0] == Approx( 1.0F ) );
        REQUIRE( app.presetPositions()[1] == Approx( 5.0F ) );
        REQUIRE( app.presetPositions()[2] == Approx( 3.0F ) );
    }

    SECTION( "non-fractional preset startup uses integer formatting" )
    {
        stdMotionStage_probe app( "stdmotion-format" );
        app.setFractionalPresets( false );
        app.setPresets( { 1.0F, 2.0F }, { "open", "science" } );

        REQUIRE( app.startupStage() == 0 );
        REQUIRE( app.capturedNumberStep() == Approx( 1.0 ) );
        REQUIRE( app.capturedNumberFormat() == "%d" );
    }

    SECTION( "startup failure branches return errors at each registration point" )
    {
        for( int callIndex = 1; callIndex <= 4; ++callIndex )
        {
            stdMotionStage_probe app( "stdmotion-register-fail-" + std::to_string( callIndex ) );
            app.setPresets( { 1.0F, 2.0F }, { "open", "science" } );
            app.failRegisterOnCall( callIndex );

            REQUIRE( app.startupStage() == -1 );
        }

        stdMotionStage_probe appEmpty( "stdmotion-empty-names" );
        REQUIRE( appEmpty.startupStage() == -1 );
    }

    SECTION( "callbacks handle missing elements, missing values, and alias fallbacks" )
    {
        stdMotionStage_probe app( "stdmotion-callbacks" );
        app.setPresets( { 10.0F, 20.0F, 20.0F }, { "open", "science", "focus" } );
        REQUIRE( app.startupStage() == 0 );

        REQUIRE( app.newCallBack_m_indiP_preset( app.presetNoValueRequest() ) == -1 );
        REQUIRE( app.moveToCalls() == 0 );

        REQUIRE( app.newCallBack_m_indiP_presetName(
                     app.presetNamePartialRequest( { "science", "focus" }, "focus" ) ) == 0 );
        REQUIRE( app.moveToCalls() == 1 );
        REQUIRE( app.lastMoveTarget() == Approx( 20.0F ) );
        REQUIRE( app.presetTargetValue() == Approx( 20.0F ) );

        REQUIRE( app.newCallBack_m_indiP_home( app.homeNoRequest() ) == 0 );
        REQUIRE( app.homeCalls() == 0 );
        REQUIRE( app.newCallBack_m_indiP_home( app.homeOffRequest() ) == 0 );
        REQUIRE( app.homeCalls() == 0 );

        REQUIRE( app.newCallBack_m_indiP_stop( app.stopNoRequest() ) == 0 );
        REQUIRE( app.stopCalls() == 0 );
        REQUIRE( app.newCallBack_m_indiP_stop( app.stopOffRequest() ) == 0 );
        REQUIRE( app.stopCalls() == 0 );

        app.setStageTelemetry( 0, 0, 0.0F );
        REQUIRE( app.telemetryPresetNameDirect().empty() == true );

        app.setStageTelemetry( 1, 0, 1.0F );
        app.setReportedPresetNumber( 9 );
        REQUIRE( app.activePresetNameDirect( 9 ).empty() == true );
        REQUIRE( app.telemetryPresetNameDirect().empty() == true );

        REQUIRE( app.setPresetAliasIndex( 2 ) == 0 );
        app.setReportedPresetNumber( 1 );
        REQUIRE( app.activePresetNameDirect( 1 ) == "focus" );
    }
}

/// Verify the lifecycle helpers integrate with power-off handling and forced telemetry hooks.
/**
 * \ingroup elliptecCtrl_unit_test
 */
TEST_CASE( "elliptecCtrl lifecycle helpers handle startup, power off, and shutdown", "[elliptecCtrl]" )
{
    elliptecCtrl_test app( "elliptec-lifecycle", testRoot( "lifecycle" ) );
    app.setPresets( { 10.0F, 20.0F }, { "open", "science" } );

    REQUIRE( app.startup() == 0 );

    // clang-format off
#ifdef ELLIPTECCTRL_TEST_DOXYGEN_REF
    MagAOX::app::elliptecCtrl::appStartup();
    MagAOX::app::elliptecCtrl::appLogic();
    MagAOX::app::elliptecCtrl::onPowerOff();
    MagAOX::app::elliptecCtrl::whilePowerOff();
    MagAOX::app::elliptecCtrl::checkRecordTimes();
    MagAOX::app::elliptecCtrl::recordTelem( static_cast<const telem_stage *>( nullptr ) );
    MagAOX::app::elliptecCtrl::recordTelem( static_cast<const telem_position *>( nullptr ) );
    MagAOX::app::elliptecCtrl::recordStage( true );
    MagAOX::app::elliptecCtrl::recordPosition( true );
    MagAOX::app::elliptecCtrl::appShutdown();
#endif
    // clang-format on

    REQUIRE( app.checkRecordTimes() == 0 );
    REQUIRE( app.recordTelem( static_cast<const telem_stage *>( nullptr ) ) == 0 );
    REQUIRE( app.recordTelem( static_cast<const telem_position *>( nullptr ) ) == 0 );
    REQUIRE( app.recordStage( true ) == 0 );
    REQUIRE( app.recordPosition( true ) == 0 );

    app.setConnected( true );
    app.setHomed( true );
    app.setPending( elliptecCtrl_test::pendingT::MoveAbs );
    app.setMoving( 1 );
    app.setWasPowered( true );
    app.setPower( 0, 0 );

    REQUIRE( app.logic() == 0 );
    REQUIRE( app.state() == MagAOX::app::stateCodes::POWEROFF );
    REQUIRE( app.connected() == false );
    REQUIRE( app.homed() == false );
    REQUIRE( app.pending() == elliptecCtrl_test::pendingT::None );
    REQUIRE( app.moving() == -2 );
    REQUIRE( app.wasPowered() == false );

    REQUIRE( app.logic() == 0 );
    REQUIRE( app.state() == MagAOX::app::stateCodes::POWEROFF );

    REQUIRE( app.shutdown() == 0 );
    REQUIRE( app.fd() == -1 );
}

/// Verify the serial/protocol helpers talk the Elliptec wire format and cover success and timeout branches.
/**
 * \ingroup elliptecCtrl_unit_test
 */
TEST_CASE( "elliptecCtrl serial helpers and appLogic connection path speak Elliptec frames", "[elliptecCtrl]" )
{
    SECTION( "invalid serial ports fail cleanly" )
    {
        elliptecCtrl_test app( "elliptec-open-fail", testRoot( "open_fail" ) );
        app.setPortPath( "/tmp/does-not-exist" );

        REQUIRE( app.openPortDirect() == -1 );
    }

    SECTION( "the connection path queries the device and records the latest state" )
    {
        fakeElliptecDevice dev(
            []( const std::string &cmd ) -> std::string
            {
                if( cmd == "0in" )
                {
                    return elliptecReply( '0', "IN00001000" );
                }
                if( cmd == "0gp" )
                {
                    return elliptecReply( '0', "PO00000800" );
                }
                if( cmd == "0gs" )
                {
                    return elliptecReply( '0', "GS00" );
                }
                if( cmd == "0sv28" )
                {
                    return elliptecReply( '0', "GS00" );
                }

                return elliptecReply( '0', "GS00" );
            } );

        elliptecCtrl_test app( "elliptec-connect", testRoot( "connect" ) );
        app.setPresets( { 180.0F, 270.0F }, { "half", "threeQuarter" } );
        app.setPortPath( dev.slavePath() );

        REQUIRE( app.startup() == 0 );
        REQUIRE( app.logic() == 0 );

        REQUIRE( app.connected() == true );
        REQUIRE( app.positionPulses() == 0x800 );
        REQUIRE( app.positionDeg() == Approx( 180.0 ) );
        REQUIRE( app.gs() == 0x00 );
        REQUIRE( app.state() == MagAOX::app::stateCodes::NOTHOMED );

        const auto commands = dev.commands();
        REQUIRE( commands.size() >= 7 );
        REQUIRE( commands[0] == "0in" );
        REQUIRE( commands[1] == "0gp" );
        REQUIRE( commands[2] == "0gs" );
        REQUIRE( commands[3] == "0sv28" );
    }

    SECTION( "protocol helpers return soft timeouts when no frame arrives" )
    {
        fakeElliptecDevice dev(
            []( const std::string &cmd ) -> std::string
            {
                static_cast<void>( cmd );
                return "";
            } );

        elliptecCtrl_test app( "elliptec-timeout", testRoot( "timeout" ) );
        app.setPortPath( dev.slavePath() );
        app.setTimeouts( 5, 5 );

        REQUIRE( app.openPortDirect() == 0 );
        REQUIRE( app.qInfoDirect() == 1 );
        REQUIRE( app.qStatusDirect() == 1 );
        REQUIRE( app.qPositionDirect() == 1 );
    }

    SECTION( "serial helpers cover empty-drain, delayed writes, null replies, and hard query errors" )
    {
        fakeElliptecDevice dev( []( const std::string &cmd ) -> std::string
                                { return elliptecReply( cmd[0], "GS00" ); } );
        elliptecCtrl_test  app( "elliptec-serial-cold-paths", testRoot( "serial_cold_paths" ) );

        app.setPortPath( dev.slavePath() );
        app.setPostWriteSleepMs( 1 );

        REQUIRE( app.openPortDirect() == 0 );
        REQUIRE( app.drainInputDirect() == 0 );
        REQUIRE( app.writeAllDirect( "0gs" ) == 0 );
        REQUIRE( app.txrxDirect( "gs", nullptr, 5 ) == 0 );

        app.closePortDirect();
        REQUIRE( app.qInfoDirect() == -1 );
    }
}

/// Verify the INDI-facing status and callback glue covers the remaining dispatch paths.
/**
 * \ingroup elliptecCtrl_unit_test
 */
TEST_CASE( "elliptecCtrl status publishing and callback shims cover the INDI glue paths", "[elliptecCtrl]" )
{
    auto prepareMotionApp = []( elliptecCtrl_test &app, fakeElliptecDevice &dev )
    {
        app.setPresets( { 10.0F, 20.0F, 20.0F }, { "open", "science", "focus" } );
        app.setPulsesPerRev( 4096 );
        app.setPortPath( dev.slavePath() );
        REQUIRE( app.startup() == 0 );
        REQUIRE( app.openPortDirect() == 0 );
        app.setConnected( true );
        app.setPower( 1, 1 );
    };

    SECTION( "status publishing and stdMotionStage INDI updates reflect the current state" )
    {
        elliptecCtrl_test app( "elliptec-indi-status", testRoot( "indi_status" ) );
        app.setPresets( { 10.0F, 20.0F, 20.0F }, { "open", "science", "focus" } );

        REQUIRE( app.startup() == 0 );
        REQUIRE( app.setupINDITransport() == 0 );

        app.setFsmState( MagAOX::app::stateCodes::POWEROFF );
        app.setConnected( false );
        app.setMoving( -2 );
        app.updateStatusDirect();
        REQUIRE( app.statusText() == "Powered Off" );
        REQUIRE( app.statusState() == pcf::IndiProperty::Idle );

        app.setFsmState( MagAOX::app::stateCodes::NOTCONNECTED );
        app.updateStatusDirect();
        REQUIRE( app.statusText() == "Not Connected" );

        app.setConnected( true );
        app.setMoving( 2 );
        app.updateStatusDirect();
        REQUIRE( app.statusText() == "Homing" );
        REQUIRE( app.statusState() == pcf::IndiProperty::Busy );

        app.setMoving( 1 );
        app.setStatusHint( "" );
        app.updateStatusDirect();
        REQUIRE( app.statusText() == "Busy" );

        app.setStatusHint( "Applying Custom Move" );
        app.updateStatusDirect();
        REQUIRE( app.statusText() == "Applying Custom Move" );

        app.setMoving( 0 );
        app.setHomed( false );
        app.updateStatusDirect();
        REQUIRE( app.statusText() == "Not Homed" );

        app.setHomed( true );
        app.setPositionDeg( 20.0 );
        app.setMovingState( 1 );
        app.setMoving( 1 );
        app.setPresetTelemetry( 2.0F, 2.0F );
        REQUIRE( app.setPresetAliasIndex( 2 ) == 0 );
        REQUIRE( app.updateStageINDIDirect() == 0 );
        REQUIRE( app.presetNameOn( "focus" ) == true );
        REQUIRE( app.presetNameState() == pcf::IndiProperty::Busy );
        REQUIRE( app.presetCurrentElement() == Approx( 2.0 ) );
        REQUIRE( app.presetTargetElement() == Approx( 2.0 ) );

        app.setMoving( 0 );
        REQUIRE( app.updateStageINDIDirect() == 0 );
        REQUIRE( app.presetNameOn( "focus" ) == true );

        app.clearPresetAliasIndex();
        REQUIRE( app.updateStageINDIDirect() == 0 );
        REQUIRE( app.presetNameOn( "science" ) == true );

        app.setFsmState( MagAOX::app::stateCodes::READY );
        app.updateStatusDirect();
        REQUIRE( app.statusText() == "OK" );
        REQUIRE( app.statusState() == pcf::IndiProperty::Idle );

        app.updateStatusDirect();
        REQUIRE( app.statusText() == "OK" );
    }

    SECTION( "static callback shims forward to the app and stdMotionStage handlers" )
    {
        fakeElliptecDevice dev( []( const std::string &cmd ) -> std::string
                                { return elliptecReply( cmd[0], "GS00" ); } );
        elliptecCtrl_test  app( "elliptec-static-shims", testRoot( "static_shims" ) );
        prepareMotionApp( app, dev );

        // clang-format off
#ifdef ELLIPTECCTRL_TEST_DOXYGEN_REF
        MagAOX::app::elliptecCtrl::st_newCallBack_m_ipAbsDeg( &app, app.absDegRequest( 45.0 ) );
        MagAOX::app::elliptecCtrl::st_newCallBack_m_ipRelDeg( &app, app.relDegRequest( -720.0 ) );
        MagAOX::app::elliptecCtrl::st_newCallBack_m_ipRelMove( &app, app.relMoveRequest() );
        MagAOX::app::elliptecCtrl::st_newCallBack_m_ipVelPct( &app, app.velPctRequest( 10 ) );
        MagAOX::app::elliptecCtrl::st_newCallBack_m_ipOptimize( &app, app.optimizeRequest() );
        MagAOX::app::elliptecCtrl::st_newCallBack_m_ipSave( &app, app.saveRequest() );
        MagAOX::app::dev::stdMotionStage<MagAOX::app::elliptecCtrl>::st_newCallBack_stdMotionStage(
            &app, app.presetRequest( 1.0F ) );
#endif
        // clang-format on

        dev.clearCommands();
        REQUIRE( MagAOX::app::elliptecCtrl::st_newCallBack_m_ipAbsDeg( &app, app.absDegRequest( 45.0 ) ) == 0 );
        REQUIRE( dev.commands().size() == 2 );
        REQUIRE( dev.commands()[0] == "0ma00000200" );
        REQUIRE( dev.commands()[1] == "0gp" );

        REQUIRE( MagAOX::app::elliptecCtrl::st_newCallBack_m_ipRelDeg( &app, app.relDegRequest( -900.0 ) ) == 0 );
        REQUIRE( app.relStepDeg() == Approx( -720.0 ) );

        dev.clearCommands();
        REQUIRE( MagAOX::app::elliptecCtrl::st_newCallBack_m_ipRelMove( &app, app.relMoveRequest() ) == 0 );
        REQUIRE( dev.commands().size() == 2 );
        REQUIRE( dev.commands()[0] == "0mrFFFFE000" );
        REQUIRE( dev.commands()[1] == "0gp" );

        dev.clearCommands();
        REQUIRE( MagAOX::app::elliptecCtrl::st_newCallBack_m_ipVelPct( &app, app.velPctRequest( 10 ) ) == 0 );
        REQUIRE( dev.commands().back() == "0sv0A" );

        dev.clearCommands();
        REQUIRE( MagAOX::app::elliptecCtrl::st_newCallBack_m_ipOptimize( &app, app.optimizeRequest() ) == 0 );
        REQUIRE( dev.commands().back() == "0om" );

        dev.clearCommands();
        REQUIRE( MagAOX::app::elliptecCtrl::st_newCallBack_m_ipSave( &app, app.saveRequest() ) == 0 );
        REQUIRE( dev.commands().back() == "0us" );

        dev.clearCommands();
        REQUIRE( MagAOX::app::dev::stdMotionStage<MagAOX::app::elliptecCtrl>::st_newCallBack_stdMotionStage(
                     &app, app.presetRequest( 1.0F ) ) == 0 );
        REQUIRE( dev.commands().back() == "0ma00000072" );

        dev.clearCommands();
        REQUIRE( MagAOX::app::dev::stdMotionStage<MagAOX::app::elliptecCtrl>::st_newCallBack_stdMotionStage(
                     &app, app.presetNameRequest( "focus" ) ) == 0 );
        REQUIRE( dev.commands().back() == "0ma000000E4" );

        dev.clearCommands();
        REQUIRE( MagAOX::app::dev::stdMotionStage<MagAOX::app::elliptecCtrl>::st_newCallBack_stdMotionStage(
                     &app, app.homeRequest() ) == 0 );
        REQUIRE( dev.commands().size() == 2 );
        REQUIRE( dev.commands()[0] == "0ho0" );
        REQUIRE( dev.commands()[1] == "0gp" );

        dev.clearCommands();
        REQUIRE( MagAOX::app::dev::stdMotionStage<MagAOX::app::elliptecCtrl>::st_newCallBack_stdMotionStage(
                     &app, app.stopRequest() ) == 0 );
        REQUIRE( dev.commands().back() == "0st" );

        pcf::IndiProperty bogus( pcf::IndiProperty::Number );
        bogus.setDevice( app.port() );
        bogus.setName( "bogus" );
        bogus.add( pcf::IndiElement( "target" ) );
        bogus["target"].set( 1.0 );
        REQUIRE( MagAOX::app::dev::stdMotionStage<MagAOX::app::elliptecCtrl>::st_newCallBack_stdMotionStage(
                     &app, bogus ) == -1 );
    }
}

/// Verify motion helpers and both standard and Elliptec-specific callbacks command the expected frames.
/**
 * \ingroup elliptecCtrl_unit_test
 */
TEST_CASE( "elliptecCtrl motion callbacks and helpers issue the expected controller commands", "[elliptecCtrl]" )
{
    auto prepareMotionApp = []( elliptecCtrl_test &app, fakeElliptecDevice &dev )
    {
        app.setPresets( { 10.0F, 20.0F, 20.0F }, { "open", "science", "focus" } );
        app.setPulsesPerRev( 4096 );
        app.setPortPath( dev.slavePath() );
        REQUIRE( app.startup() == 0 );
        REQUIRE( app.openPortDirect() == 0 );
        app.setConnected( true );
        app.setPower( 1, 1 );
    };

    SECTION( "absolute and relative move helpers encode pulse commands correctly" )
    {
        fakeElliptecDevice dev( []( const std::string &cmd ) -> std::string
                                { return elliptecReply( cmd[0], "GS00" ); } );
        elliptecCtrl_test  app( "elliptec-helpers", testRoot( "helpers" ) );
        prepareMotionApp( app, dev );

        // clang-format off
#ifdef ELLIPTECCTRL_TEST_DOXYGEN_REF
        MagAOX::app::elliptecCtrl::moveAbsDeg_( 810.0 );
        MagAOX::app::elliptecCtrl::moveRelDeg_( 45.0 );
        MagAOX::app::elliptecCtrl::moveRelDegCmdFromRelMove_();
        MagAOX::app::elliptecCtrl::startMoveToDeg_( 90.0 );
        MagAOX::app::elliptecCtrl::cmd_setvel_( 140 );
        MagAOX::app::elliptecCtrl::cmd_optimize_wait_();
        MagAOX::app::elliptecCtrl::cmd_save_();
#endif
        // clang-format on

        dev.clearCommands();
        REQUIRE( app.moveAbsDegDirect( 810.0 ) == 0 );
        REQUIRE( dev.commands().back() == "0ma00000400" );

        dev.clearCommands();
        REQUIRE( app.moveRelDegDirect( 45.0 ) == 0 );
        REQUIRE( dev.commands().back() == "0mr00000200" );

        app.setRelStepDeg( -45.0 );
        dev.clearCommands();
        REQUIRE( app.moveRelDegCmdDirect() == 0 );
        REQUIRE( dev.commands().back() == "0mrFFFFFE00" );

        dev.clearCommands();
        REQUIRE( app.startMoveToDegDirect( 90.0 ) == 0 );
        REQUIRE( dev.commands().back() == "0ma00000400" );

        dev.clearCommands();
        REQUIRE( app.cmdSetVelDirect( 140 ) == 0 );
        REQUIRE( dev.commands().back() == "0sv64" );

        dev.clearCommands();
        app.setMovingState( 0 );
        REQUIRE( app.moveToDirect( 0.0F ) == 0 );
        REQUIRE( dev.commands().back() == "0ma00000072" );

        dev.clearCommands();
        REQUIRE( app.moveToDirect( 99.0F ) == 0 );
        REQUIRE( dev.commands().back() == "0ma000000E4" );
    }

    SECTION( "Elliptec-specific callbacks update state and send the expected commands" )
    {
        fakeElliptecDevice dev( []( const std::string &cmd ) -> std::string
                                { return elliptecReply( cmd[0], "GS00" ); } );
        elliptecCtrl_test  app( "elliptec-callbacks", testRoot( "callbacks" ) );
        prepareMotionApp( app, dev );

        // clang-format off
#ifdef ELLIPTECCTRL_TEST_DOXYGEN_REF
        MagAOX::app::elliptecCtrl::newCallBack_m_ipAbsDeg( app.absDegRequest( 45.0 ) );
        MagAOX::app::elliptecCtrl::newCallBack_m_ipRelDeg( app.relDegRequest( 90.0 ) );
        MagAOX::app::elliptecCtrl::newCallBack_m_ipRelMove( app.relMoveRequest() );
        MagAOX::app::elliptecCtrl::newCallBack_m_ipVelPct( app.velPctRequest( 100 ) );
        MagAOX::app::elliptecCtrl::newCallBack_m_ipOptimize( app.optimizeRequest() );
        MagAOX::app::elliptecCtrl::newCallBack_m_ipSave( app.saveRequest() );
#endif
        // clang-format on

        dev.clearCommands();
        REQUIRE( app.newCallBack_m_ipAbsDeg( app.absDegRequest( 45.0 ) ) == 0 );
        REQUIRE( dev.commands().size() == 2 );
        REQUIRE( dev.commands()[0] == "0ma00000200" );
        REQUIRE( dev.commands()[1] == "0gp" );

        REQUIRE( app.newCallBack_m_ipRelDeg( app.relDegRequest( 900.0 ) ) == 0 );
        REQUIRE( app.relStepDeg() == Approx( 720.0 ) );

        dev.clearCommands();
        REQUIRE( app.newCallBack_m_ipRelMove( app.relMoveRequest() ) == 0 );
        REQUIRE( dev.commands().size() == 2 );
        REQUIRE( dev.commands()[0] == "0mr00002000" );
        REQUIRE( dev.commands()[1] == "0gp" );

        dev.clearCommands();
        REQUIRE( app.newCallBack_m_ipVelPct( app.velPctRequest( 140 ) ) == 0 );
        REQUIRE( app.velPercent() == 100 );
        REQUIRE( dev.commands().back() == "0sv64" );

        dev.clearCommands();
        REQUIRE( app.newCallBack_m_ipOptimize( app.optimizeRequest() ) == 0 );
        REQUIRE( dev.commands().back() == "0om" );
        REQUIRE( app.pending() == elliptecCtrl_test::pendingT::Optimize );

        dev.clearCommands();
        REQUIRE( app.newCallBack_m_ipSave( app.saveRequest() ) == 0 );
        REQUIRE( dev.commands().back() == "0us" );
        REQUIRE( app.pending() == elliptecCtrl_test::pendingT::Save );
    }

    SECTION( "stdMotionStage callbacks preserve preset tracking and command the expected moves" )
    {
        fakeElliptecDevice dev( []( const std::string &cmd ) -> std::string
                                { return elliptecReply( cmd[0], "GS00" ); } );
        elliptecCtrl_test  app( "elliptec-stdmotion", testRoot( "stdmotion" ) );
        prepareMotionApp( app, dev );

        // clang-format off
#ifdef ELLIPTECCTRL_TEST_DOXYGEN_REF
        MagAOX::app::dev::stdMotionStage<MagAOX::app::elliptecCtrl>::newCallBack_m_indiP_preset( app.presetRequest( 2.0F ) );
        MagAOX::app::dev::stdMotionStage<MagAOX::app::elliptecCtrl>::newCallBack_m_indiP_presetName( app.presetNameRequest( "focus" ) );
        MagAOX::app::dev::stdMotionStage<MagAOX::app::elliptecCtrl>::newCallBack_m_indiP_home( app.homeRequest() );
        MagAOX::app::dev::stdMotionStage<MagAOX::app::elliptecCtrl>::newCallBack_m_indiP_stop( app.stopRequest() );
#endif
        // clang-format on

        dev.clearCommands();
        REQUIRE( app.newCallBack_m_indiP_preset( app.presetRequest( 2.0F ) ) == 0 );
        REQUIRE( dev.commands().back() == "0ma000000E4" );
        REQUIRE( app.presetTargetValue() == Approx( 2.0F ) );

        dev.clearCommands();
        REQUIRE( app.newCallBack_m_indiP_presetName( app.presetNameRequest( "focus" ) ) == 0 );
        REQUIRE( dev.commands().back() == "0ma000000E4" );
        app.setPositionDeg( 20.0 );
        app.syncPresetStateDirect();
        REQUIRE( app.telemetryPresetNameDirect() == "focus" );

        dev.clearCommands();
        REQUIRE( app.newCallBack_m_indiP_home( app.homeRequest() ) == 0 );
        REQUIRE( dev.commands().size() == 2 );
        REQUIRE( dev.commands()[0] == "0ho0" );
        REQUIRE( dev.commands()[1] == "0gp" );
        REQUIRE( app.pending() == elliptecCtrl_test::pendingT::Home );
        REQUIRE( app.moving() == 2 );

        dev.clearCommands();
        REQUIRE( app.newCallBack_m_indiP_stop( app.stopRequest() ) == 0 );
        REQUIRE( dev.commands().back() == "0st" );
        REQUIRE( app.pending() == elliptecCtrl_test::pendingT::Stop );
    }
}

/// Verify helper edge cases cover the remaining serial, parser, and appLogic branches.
/**
 * \ingroup elliptecCtrl_unit_test
 */
TEST_CASE( "elliptecCtrl helper edge cases cover the remaining serial and FSM branches", "[elliptecCtrl]" )
{
    SECTION( "serial helpers cover closed-port errors and the baud-rate mapper" )
    {
        elliptecCtrl_test app( "elliptec-helper-edges", testRoot( "helper_edges" ) );
        std::string       frame;

        REQUIRE( app.toTermiosBaudDirect( 9600 ) == B9600 );
        REQUIRE( app.toTermiosBaudDirect( 19200 ) == B19200 );
        REQUIRE( app.toTermiosBaudDirect( 38400 ) == B38400 );
        REQUIRE( app.toTermiosBaudDirect( 57600 ) == B57600 );
        REQUIRE( app.toTermiosBaudDirect( 115200 ) == B115200 );
        REQUIRE( app.toTermiosBaudDirect( 230400 ) == B9600 );

        REQUIRE( app.drainInputDirect() == -1 );
        REQUIRE( app.writeAllDirect( "0gs" ) == -1 );
        REQUIRE( app.readFrameDirect( frame, 5 ) == -1 );
        REQUIRE( app.txrxDirect( "gs", &frame, 5 ) == -1 );

        app.setPulsesPerRev( 0 );
        REQUIRE( app.degToPulsesDirect( 90.0 ) == 0 );
        REQUIRE( app.pulsesToDegDirect( 256 ) == Approx( 0.0 ) );
    }

    SECTION( "direct motion helpers return hard errors when the port is unavailable" )
    {
        elliptecCtrl_test app( "elliptec-command-errors", testRoot( "command_errors" ) );
        app.setPresets( { 10.0F, 20.0F }, { "open", "science" } );
        app.setPulsesPerRev( 4096 );
        app.setPower( 1, 1 );

        REQUIRE( app.stop() == -1 );
        REQUIRE( app.startHoming() == -1 );
        REQUIRE( app.startMoveToDegDirect( 45.0 ) == -1 );
        REQUIRE( app.moveAbsDegDirect( 45.0 ) == -1 );
        REQUIRE( app.moveRelDegDirect( 45.0 ) == -1 );
        REQUIRE( app.cmdHomeDirect( 0 ) == -1 );
        REQUIRE( app.cmdStopDirect() == -1 );
        REQUIRE( app.cmdOptimizeDirect() == -1 );
        REQUIRE( app.cmdSaveDirect() == -1 );
    }

    SECTION( "callbacks no-op while powered off and stdMotionStage guards invalid preset-name requests" )
    {
        fakeElliptecDevice dev( []( const std::string &cmd ) -> std::string
                                { return elliptecReply( cmd[0], "GS00" ); } );
        elliptecCtrl_test  app( "elliptec-callback-guards", testRoot( "callback_guards" ) );

        app.setPresets( { 10.0F, 20.0F, 20.0F }, { "open", "science", "focus" } );
        app.setPulsesPerRev( 4096 );
        app.setPortPath( dev.slavePath() );
        REQUIRE( app.startup() == 0 );
        REQUIRE( app.openPortDirect() == 0 );
        app.setConnected( true );
        app.setPower( 0, 0 );

        pcf::IndiProperty relMoveOff = app.relMoveRequest();
        relMoveOff["request"].setSwitchState( pcf::IndiElement::Off );

        pcf::IndiProperty optimizeOff = app.optimizeRequest();
        optimizeOff["request"].setSwitchState( pcf::IndiElement::Off );

        pcf::IndiProperty saveOff = app.saveRequest();
        saveOff["request"].setSwitchState( pcf::IndiElement::Off );

        REQUIRE( app.newCallBack_m_ipAbsDeg( app.absDegRequest( 45.0 ) ) == 0 );
        REQUIRE( app.newCallBack_m_ipRelMove( relMoveOff ) == 0 );
        REQUIRE( app.newCallBack_m_ipVelPct( app.velPctRequest( 33 ) ) == 0 );
        REQUIRE( app.newCallBack_m_ipOptimize( optimizeOff ) == 0 );
        REQUIRE( app.newCallBack_m_ipSave( saveOff ) == 0 );
        REQUIRE( MagAOX::app::dev::stdMotionStage<MagAOX::app::elliptecCtrl>::st_newCallBack_stdMotionStage(
                     &app, app.homeRequest() ) == 0 );
        REQUIRE( MagAOX::app::dev::stdMotionStage<MagAOX::app::elliptecCtrl>::st_newCallBack_stdMotionStage(
                     &app, app.stopRequest() ) == 0 );
        REQUIRE( dev.commands().empty() == true );

        app.setPower( 1, 1 );
        REQUIRE( app.setPresetAliasIndex( -1 ) == -1 );

        pcf::IndiProperty noPresetName = app.presetNameRequest( "open" );
        noPresetName["open"].setSwitchState( pcf::IndiElement::Off );

        pcf::IndiProperty twoPresetNames = app.presetNameRequest( "open" );
        twoPresetNames["science"].setSwitchState( pcf::IndiElement::On );

        REQUIRE( app.newCallBack_m_indiP_presetName( noPresetName ) == 0 );
        REQUIRE( app.newCallBack_m_indiP_presetName( twoPresetNames ) == -1 );
    }

    SECTION( "parser helpers preserve prior state on malformed replies and command helpers clamp edge cases" )
    {
        fakeElliptecDevice dev(
            []( const std::string &cmd ) -> std::string
            {
                if( cmd == "0in" )
                {
                    return elliptecReply( '0', "INBADHEX!!" );
                }
                if( cmd == "0gs" )
                {
                    return elliptecReply( '0', "GSZZ" );
                }
                if( cmd == "0gp" )
                {
                    return elliptecReply( '0', "PO00000Z00" );
                }

                return elliptecReply( '0', "GS00" );
            } );

        elliptecCtrl_test app( "elliptec-parser-edges", testRoot( "parser_edges" ) );
        app.setPresets( { 10.0F, 20.0F }, { "open", "science" } );
        app.setPulsesPerRev( 4096 );
        app.setPortPath( dev.slavePath() );
        REQUIRE( app.startup() == 0 );
        REQUIRE( app.openPortDirect() == 0 );

        app.setGs( 0xAA );
        app.setPositionPulses( 123 );

        REQUIRE( app.qInfoDirect() == 0 );
        REQUIRE( app.degToPulsesDirect( 180.0 ) == 2048 );

        REQUIRE( app.qStatusDirect() == 0 );
        REQUIRE( app.gs() == 0xAA );

        REQUIRE( app.qPositionDirect() == 0 );
        REQUIRE( app.positionPulses() == 123 );

        dev.clearCommands();
        REQUIRE( app.cmdSetVelDirect( -10 ) == 0 );
        REQUIRE( dev.commands().back() == "0sv00" );

        dev.clearCommands();
        REQUIRE( app.moveAbsDegDirect( -45.0 ) == 0 );
        REQUIRE( dev.commands().back() == "0ma00001E00" );

        dev.clearCommands();
        REQUIRE( app.cmdHomeDirect( 0x0F ) == 0 );
        REQUIRE( dev.commands().back() == "0hoF" );

        dev.clearCommands();
        REQUIRE( app.cmdStopDirect() == 0 );
        REQUIRE( dev.commands().back() == "0st" );

        dev.clearCommands();
        REQUIRE( app.cmdOptimizeDirect() == 0 );
        REQUIRE( dev.commands().back() == "0om" );

        dev.clearCommands();
        REQUIRE( app.cmdSaveDirect() == 0 );
        REQUIRE( dev.commands().back() == "0us" );
    }

    SECTION( "callbacks and direct helpers cover the remaining no-value, no-request, clamp, and power-off paths" )
    {
        elliptecCtrl_test app( "elliptec-callback-cold-paths", testRoot( "callback_cold_paths" ) );
        app.setPresets( { 10.0F, 20.0F }, { "open", "science" } );
        app.setPulsesPerRev( 4096 );
        app.setPower( 1, 1 );

        REQUIRE( app.startup() == 0 );

        REQUIRE( app.newCallBack_m_ipAbsDeg( app.absDegNoValueRequest() ) == -1 );
        REQUIRE( app.newCallBack_m_ipRelDeg( app.relDegNoValueRequest() ) == -1 );
        REQUIRE( app.newCallBack_m_ipRelMove( app.relMoveNoRequest() ) == 0 );
        REQUIRE( app.newCallBack_m_ipRelMove( app.relMoveRequest() ) == -1 );
        REQUIRE( app.newCallBack_m_ipVelPct( app.velPctNoValueRequest() ) == -1 );
        REQUIRE( app.newCallBack_m_ipOptimize( app.optimizeNoRequest() ) == 0 );
        REQUIRE( app.newCallBack_m_ipSave( app.saveNoRequest() ) == 0 );

        app.setPower( 0, 0 );
        REQUIRE( app.startMoveToDegDirect( 15.0 ) == 0 );

        app.setPower( 1, 1 );
        REQUIRE( app.newCallBack_m_ipVelPct( app.velPctRequest( -5 ) ) == -1 );
    }

    SECTION( "startup without presets fails cleanly and moveTo rejects an empty preset table" )
    {
        elliptecCtrl_test noPresetApp( "elliptec-no-presets", testRoot( "no_presets" ) );
        noPresetApp.setPower( 1, 1 );
        REQUIRE( noPresetApp.startup() == -1 );
        REQUIRE( noPresetApp.moveToDirect( 1.0F ) == -1 );
    }

    SECTION( "appLogic exercises the powered-off and reconnect path with INDI transport enabled" )
    {
        fakeElliptecDevice dev(
            []( const std::string &cmd ) -> std::string
            {
                if( cmd == "0in" )
                {
                    return elliptecReply( '0', "IN00001000" );
                }
                if( cmd == "0gp" )
                {
                    return elliptecReply( '0', "PO00000800" );
                }
                if( cmd == "0gs" )
                {
                    return elliptecReply( '0', "GS00" );
                }
                if( cmd == "0sv28" )
                {
                    return elliptecReply( '0', "GS00" );
                }

                return elliptecReply( '0', "GS00" );
            } );

        elliptecCtrl_test app( "elliptec-applogic-indi", testRoot( "applogic_indi" ) );
        app.setPresets( { 180.0F, 270.0F }, { "half", "threeQuarter" } );

        REQUIRE( app.startup() == 0 );
        REQUIRE( app.setupINDITransport() == 0 );

        app.setPower( 0, 0 );
        app.setWasPowered( false );
        REQUIRE( app.logic() == 0 );
        REQUIRE( app.state() == MagAOX::app::stateCodes::POWEROFF );

        app.setPower( 1, 1 );
        app.setPortPath( dev.slavePath() );
        REQUIRE( app.logic() == 0 );
        REQUIRE( app.connected() == true );
        REQUIRE( app.positionDeg() == Approx( 180.0 ) );
        REQUIRE( app.state() == MagAOX::app::stateCodes::NOTHOMED );
    }

    SECTION( "appStartup and appLogic cover the remaining connection and reconnect branches" )
    {
        {
            elliptecCtrl_test uninitialized( "elliptec-uninitialized", testRoot( "uninitialized" ) );
            REQUIRE( uninitialized.appStartup() == -1 );
        }

        {
            elliptecCtrl_test connectFail( "elliptec-connect-fail", testRoot( "connect_fail_logic" ) );
            connectFail.setPortPath( "/tmp/does-not-exist" );
            connectFail.setPresets( { 10.0F, 20.0F }, { "open", "science" } );
            REQUIRE( connectFail.startup() == 0 );
            connectFail.setFsmState( MagAOX::app::stateCodes::POWERON );
            REQUIRE( connectFail.logic() == 0 );
            REQUIRE( connectFail.connected() == false );
            REQUIRE( connectFail.state() == MagAOX::app::stateCodes::NOTCONNECTED );
            REQUIRE( connectFail.moving() == -2 );
        }

        {
            fakeElliptecDevice dev(
                []( const std::string &cmd ) -> std::string
                {
                    if( cmd == "0in" )
                    {
                        return elliptecReply( '0', "IN00001000" );
                    }
                    if( cmd == "0gp" )
                    {
                        return elliptecReply( '0', "PO00000000" );
                    }
                    if( cmd == "0gs" )
                    {
                        return elliptecReply( '0', "GS00" );
                    }
                    if( cmd == "0sv28" )
                    {
                        return elliptecReply( '0', "GS00" );
                    }

                    return elliptecReply( '0', "GS00" );
                } );

            elliptecCtrl_test app( "elliptec-applogic-edges", testRoot( "applogic_edges" ) );
            app.setPresets( { 10.0F, 20.0F }, { "open", "science" } );
            app.setPortPath( dev.slavePath() );
            app.setStartupDelayMs( 1 );
            REQUIRE( app.startup() == 0 );
            REQUIRE( app.logic() == 0 );
            REQUIRE( app.connected() == true );

            dev.handler(
                []( const std::string &cmd ) -> std::string
                {
                    static_cast<void>( cmd );
                    return "";
                } );
            app.setTimeouts( 5, 5 );
            app.setCommMaxMisses( 2 );
            REQUIRE( app.logic() == 0 );
            REQUIRE( app.connected() == true );
            REQUIRE( app.commMisses() == 1 );

            app.closePortDirect();
            REQUIRE( app.logic() == 0 );
            REQUIRE( app.connected() == false );
            REQUIRE( app.state() == MagAOX::app::stateCodes::NOTCONNECTED );
            REQUIRE( app.moving() == -2 );
            REQUIRE( app.commMisses() == 0 );
        }
    }
}

/// Verify `pollDevice_()` resolves the motion state machine and alias bookkeeping correctly.
/**
 * \ingroup elliptecCtrl_unit_test
 */
TEST_CASE( "elliptecCtrl poll resolution updates stage state across busy, idle, timeout, and offset paths",
           "[elliptecCtrl]" )
{
    auto preparePollingApp = []( elliptecCtrl_test &app, fakeElliptecDevice &dev )
    {
        app.setPresets( { 10.0F, 20.0F, 20.0F }, { "open", "science", "focus" } );
        app.setPulsesPerRev( 4096 );
        app.setPortPath( dev.slavePath() );
        REQUIRE( app.startup() == 0 );
        REQUIRE( app.openPortDirect() == 0 );
        app.setConnected( true );
        app.setPower( 1, 1 );
    };

    SECTION( "soft timeouts are tolerated until the miss budget is exceeded" )
    {
        fakeElliptecDevice dev(
            []( const std::string &cmd ) -> std::string
            {
                static_cast<void>( cmd );
                return "";
            } );
        elliptecCtrl_test app( "elliptec-soft-timeout", testRoot( "soft_timeout" ) );
        preparePollingApp( app, dev );
        app.setTimeouts( 5, 5 );
        app.setCommMaxMisses( 1 );

        REQUIRE( app.pollDeviceDirect() == 1 );
        REQUIRE( app.commMisses() == 1 );

        REQUIRE( app.pollDeviceDirect() == -1 );
        REQUIRE( app.commMisses() == 0 );
    }

    SECTION( "busy and idle status frames update the motion bookkeeping" )
    {
        fakeElliptecDevice dev;
        elliptecCtrl_test  app( "elliptec-polling", testRoot( "polling" ) );
        preparePollingApp( app, dev );

        dev.handler(
            []( const std::string &cmd ) -> std::string
            {
                if( cmd == "0gp" )
                {
                    return elliptecReply( '0', "PO000000E4" );
                }
                if( cmd == "0gs" )
                {
                    return elliptecReply( '0', "GS09" );
                }

                return elliptecReply( '0', "GS00" );
            } );

        app.setPending( elliptecCtrl_test::pendingT::None );
        app.setMoving( -1 );
        REQUIRE( app.pollDeviceDirect() == 0 );
        REQUIRE( app.moving() == 1 );
        REQUIRE( app.positionPulses() == 0xE4 );

        dev.handler(
            []( const std::string &cmd ) -> std::string
            {
                if( cmd == "0gp" )
                {
                    return elliptecReply( '0', "PO000000E4" );
                }
                if( cmd == "0gs" )
                {
                    return elliptecReply( '0', "GS00" );
                }

                return elliptecReply( '0', "GS00" );
            } );

        app.setPending( elliptecCtrl_test::pendingT::Home );
        app.setHomeOffset( 0.0 );
        REQUIRE( app.pollDeviceDirect() == 0 );
        REQUIRE( app.homed() == true );
        REQUIRE( app.pending() == elliptecCtrl_test::pendingT::None );
        REQUIRE( app.moving() == 0 );
        REQUIRE( app.state() == MagAOX::app::stateCodes::READY );
        REQUIRE( app.presetValue() == Approx( 2.0F ) );

        dev.handler(
            []( const std::string &cmd ) -> std::string
            {
                if( cmd == "0gp" )
                {
                    return elliptecReply( '0', "PO000000E4" );
                }
                if( cmd == "0gs" )
                {
                    return elliptecReply( '0', "GS09" );
                }

                return elliptecReply( '0', "GS00" );
            } );

        app.setPending( elliptecCtrl_test::pendingT::Home );
        app.setMoving( 0 );
        REQUIRE( app.pollDeviceDirect() == 0 );
        REQUIRE( app.moving() == 2 );

        app.setPending( elliptecCtrl_test::pendingT::MoveAbs );
        app.setMoving( 0 );
        REQUIRE( app.pollDeviceDirect() == 0 );
        REQUIRE( app.moving() == 1 );

        dev.handler(
            []( const std::string &cmd ) -> std::string
            {
                if( cmd == "0gp" )
                {
                    return elliptecReply( '0', "PO000000E4" );
                }
                if( cmd == "0gs" )
                {
                    return elliptecReply( '0', "GS00" );
                }

                return elliptecReply( '0', "GS00" );
            } );

        app.setPending( elliptecCtrl_test::pendingT::None );
        app.setHomed( false );
        REQUIRE( app.pollDeviceDirect() == 0 );
        REQUIRE( app.moving() == -1 );

        app.setHomed( true );
        REQUIRE( app.pollDeviceDirect() == 0 );
        REQUIRE( app.moving() == 0 );
    }

    SECTION( "home completion can trigger the configured post-home offset move" )
    {
        fakeElliptecDevice dev(
            []( const std::string &cmd ) -> std::string
            {
                if( cmd == "0gp" )
                {
                    return elliptecReply( '0', "PO00000000" );
                }
                if( cmd == "0gs" )
                {
                    return elliptecReply( '0', "GS00" );
                }
                if( cmd == "0mr00000200" )
                {
                    return elliptecReply( '0', "GS00" );
                }

                return elliptecReply( '0', "GS00" );
            } );

        elliptecCtrl_test app( "elliptec-home-offset", testRoot( "home_offset" ) );
        preparePollingApp( app, dev );
        app.setPending( elliptecCtrl_test::pendingT::Home );
        app.setHomeOffset( 45.0 );

        REQUIRE( app.pollDeviceDirect() == 0 );
        REQUIRE( app.pending() == elliptecCtrl_test::pendingT::OffsetRel );
        REQUIRE( app.moving() == 1 );
        REQUIRE( app.statusHint() == "Homed. Now applying offset..." );
        REQUIRE( dev.commands().back() == "0mr00000200" );
    }
}

} // namespace elliptecCtrlTest
} // namespace libXWCTest
