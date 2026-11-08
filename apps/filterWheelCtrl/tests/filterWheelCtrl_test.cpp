/** \file filterWheelCtrl_test.cpp
 * \brief Offline recovery, homing, and power-transition tests for the filter wheel controller.
 * \ingroup filterWheelCtrl_files
 */

#include "../../../tests/testXWC.hpp"
#include "../../../tests/outletAppTest.hpp"

#include <functional>

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
namespace filterWheelHarness
{
/// Scripted USB and serial outcomes with captured production traffic.
struct Script
{
    /// Result of hardware-free USB discovery.
    int m_discoveryResult{ 0 };

    /// Result of opening a harmless local descriptor.
    int m_connectResult{ 0 };

    /// Device name returned by successful discovery.
    std::string m_deviceName{ "/dev/filterwheel-test-reconnected" };

    /// USB discovery call count.
    unsigned m_discoveries{ 0 };

    /// Device names used for connection attempts.
    std::vector<std::string> m_connections;

    /// Actual production commands, including queries.
    std::vector<std::string> m_commands;

    /// Write command selected to fail; empty disables failure.
    std::string m_failedWrite;

    /// Matching write invocation to fail, allowing either stop write to be tested.
    unsigned m_failureCall{ 1 };

    /// Number of writes matching the selected failure command.
    unsigned m_matchingWrites{ 0 };

    /// Query selected to fail; empty disables failure.
    std::string m_failedQuery;

    /// Position reply passed to the production parser.
    std::string m_position{ "12345\r\n" };

    /// Speed reply passed to the production parser.
    std::string m_speed{ "0\r\n" };

    /// Drain result; timeout is normal, while a transport error should restart recovery.
    int m_drainResult{ TTY_E_TIMEOUTONREAD };

    /// Hook delivering power updates during a command or drain operation.
    std::function<void( const std::string & )> m_beforeIo;
};

/// Current sequential test's transport script.
Script g_script;

/// Detect home-start or move commands in captured traffic.
bool commandedMotion();

bool commandedMotion()
{
    return std::any_of( g_script.m_commands.begin(),
                        g_script.m_commands.end(),
                        []( const std::string &command )
                        {
                            return command == "GOHOSEQ\r" || command == "M\r" || command.starts_with( "LA" ) ||
                                   command.starts_with( "LR" );
                        } );
}
} // namespace filterWheelHarness

namespace MagAOX
{
namespace tty
{
/// Scripted USB boundary that never inspects hardware or opens a device.
class filterWheelTestUSBDevice : public usbDevice
{
  public:
    /// Supply scripted discovery instead of loading a real device.
    int loadConfig( mx::app::appConfigurator &config /**< [in] Unused configuration. */ );

    /// Return a scripted discovery result and fresh device path.
    int getDeviceName();

    /// Own a /dev/null descriptor on a successful connection attempt.
    int connect();
};

int filterWheelTestUSBDevice::loadConfig( mx::app::appConfigurator & )
{
    return getDeviceName();
}

int filterWheelTestUSBDevice::getDeviceName()
{
    auto &script = filterWheelHarness::g_script;
    ++script.m_discoveries;
    if( script.m_discoveryResult == 0 )
        m_deviceName = script.m_deviceName;
    return script.m_discoveryResult;
}

int filterWheelTestUSBDevice::connect()
{
    auto &script = filterWheelHarness::g_script;
    script.m_connections.push_back( m_deviceName );
    if( m_fileDescrip > 0 )
        ::close( m_fileDescrip );
    m_fileDescrip = 0;
    if( script.m_connectResult != 0 )
        return script.m_connectResult;
    m_fileDescrip = ::open( "/dev/null", O_RDWR | O_CLOEXEC );
    REQUIRE( m_fileDescrip > 0 );
    return 0;
}

/// Record a write and return the selected transport error.
int filterWheelTestWrite( const std::string &command /**< [in] Actual wire command. */,
                          int                fd /**< [in] Harmless open descriptor. */,
                          int                timeout /**< [in] Ignored timeout. */ );

/// Record a query and supply a valid, malformed, or failed response.
int filterWheelTestWriteRead( std::string       &response /**< [out] Scripted response. */,
                              const std::string &command /**< [in] Actual query. */,
                              const std::string &terminator /**< [in] Reply terminator. */,
                              bool               echo /**< [in] Echo policy. */,
                              int                fd /**< [in] Harmless open descriptor. */,
                              int                writeTimeout /**< [in] Ignored write timeout. */,
                              int                readTimeout /**< [in] Ignored read timeout. */ );

/// Simulate the expected connection-time drain timeout without waiting.
int filterWheelTestRead( std::string       &response /**< [out] Empty response. */,
                         const std::string &terminator /**< [in] Ignored terminator. */,
                         int                fd /**< [in] Harmless open descriptor. */,
                         int                timeout /**< [in] Ignored timeout. */ );

int filterWheelTestWrite( const std::string &command, int fd, int )
{
    REQUIRE( fd > 0 );
    auto &script = filterWheelHarness::g_script;
    script.m_commands.push_back( command );
    if( script.m_beforeIo )
        script.m_beforeIo( command );
    if( command == script.m_failedWrite && ++script.m_matchingWrites == script.m_failureCall )
        return TTY_E_ERRORONWRITE;
    return 0;
}

int filterWheelTestWriteRead(
    std::string &response, const std::string &command, const std::string &terminator, bool echo, int fd, int, int )
{
    REQUIRE( fd > 0 );
    REQUIRE( terminator == "\r\n" );
    REQUIRE_FALSE( echo );
    auto &script = filterWheelHarness::g_script;
    script.m_commands.push_back( command );
    if( script.m_beforeIo )
        script.m_beforeIo( command );
    if( command == script.m_failedQuery )
        return TTY_E_TIMEOUTONREAD;
    if( command == "POS\r" )
        response = script.m_position;
    else if( command == "GN\r" )
        response = script.m_speed;
    else
    {
        REQUIRE( command == "GAST\r" );
        response = "0\r\n";
    }
    return 0;
}

int filterWheelTestRead( std::string &response, const std::string &, int fd, int )
{
    REQUIRE( fd > 0 );
    response.clear();
    if( filterWheelHarness::g_script.m_beforeIo )
        filterWheelHarness::g_script.m_beforeIo( "drain" );
    return filterWheelHarness::g_script.m_drainResult;
}
} // namespace tty
} // namespace MagAOX

// Substitute application boundaries only; keep the real FSM, callbacks, and shared stage implementation.
#define MagAOXApp outletTestApp
#define telemeter outletTestTelemeter
#define usbDevice filterWheelTestUSBDevice
#define ttyWrite filterWheelTestWrite
#define ttyWriteRead filterWheelTestWriteRead
#define ttyRead filterWheelTestRead
#include "../filterWheelCtrl.hpp"
#undef ttyRead
#undef ttyWriteRead
#undef ttyWrite
#undef usbDevice
#undef telemeter
#undef MagAOXApp

namespace filterWheelHarness
{
using namespace MagAOX::app;

/// Production app with a private INDI driver and exposed recovery state.
struct Fixture : outletHarness::Controller<filterWheelCtrl>
{
    using filterWheelCtrl::m_deviceName;
    using filterWheelCtrl::m_fileDescrip;
    using filterWheelCtrl::m_homingState;
    using filterWheelCtrl::m_indiP_counts;
    using filterWheelCtrl::m_indiP_preset;
    using filterWheelCtrl::m_moving;
    using filterWheelCtrl::m_movingState;
    using filterWheelCtrl::m_powerOnHome;
    using filterWheelCtrl::m_powerOnHomePending;
    using filterWheelCtrl::m_preset_target;
    using filterWheelCtrl::m_rawPos;

    /// Set up real stage properties without telemetry threads or instrument connections.
    Fixture();

    /// Release the harmless serial descriptor before shared fixture destruction.
    ~Fixture();

    /// Deliver power updates through the registered subscription callback.
    void power( const std::string &observed /**< [in] Observed power state; empty omits this element. */,
                const std::string &target /**< [in] Requested power state; empty omits this element. */ );

    /// Confirm a stationary READY position through the production polling path.
    void ready();

    /// Fail one production poll, then clear the fault and captured traffic.
    void disconnect( const std::string &query = "GAST\r" /**< [in] Poll query to fail. */ );

    /// Dispatch raw, numeric preset, named preset, home, or stop through registered callbacks.
    int command( unsigned operation /**< [in] Command selector from zero through four. */ );
};

Fixture::Fixture()
{
    g_script                = {};
    outletHarness::g_faults = {};
    m_configName            = "test-filterwheel";
    m_circleSteps           = 360000;
    m_homeOffset            = 100;
    m_homePreset            = 0;
    m_presetNames           = { "one", "two", "three", "four", "five", "six" };
    m_presetPositions       = { 1, 2, 3, 4, 5, 6 };
    m_powerDevice           = "test-pdu";
    m_powerChannel          = "wheel";
    REQUIRE( registerIndiPropertySet( m_indiP_powerChannel,
                                      m_powerDevice,
                                      m_powerChannel,
                                      MagAOXApp<>::st_setCallBack_m_indiP_powerChannel ) == 0 );
    power( "On", "On" );
    driver();
    state( stateCodes::INITIALIZED );
    REQUIRE( appStartup() == 0 );
    state( stateCodes::READY );
    REQUIRE( connect() == 0 );
    m_rawPos = 12345;
    g_script.m_connections.clear();
    outletHarness::g_faults.m_logs.clear();
}

Fixture::~Fixture()
{
    appShutdown();
}

void Fixture::power( const std::string &observed, const std::string &target )
{
    pcf::IndiProperty received( pcf::IndiProperty::Text, m_powerDevice, m_powerChannel );
    if( !observed.empty() )
        received.add( pcf::IndiElement( "state", observed ) );
    if( !target.empty() )
        received.add( pcf::IndiElement( "target", target ) );
    auto callback = m_indiSetCallBacks.at( received.createUniqueKey() ).callBack;
    REQUIRE( callback != nullptr );
    REQUIRE( callback( this, received ) == 0 );
}

void Fixture::ready()
{
    REQUIRE( appLogic() == 0 );
    REQUIRE( state() == stateCodes::READY );
    REQUIRE( m_moving == 0 );
    g_script.m_commands.clear();
}

void Fixture::disconnect( const std::string &query )
{
    g_script.m_failedQuery = query;
    REQUIRE( appLogic() == 0 );
    REQUIRE( state() == stateCodes::NODEVICE );
    g_script.m_failedQuery.clear();
    g_script.m_commands.clear();
}

int Fixture::command( unsigned operation )
{
    const std::array<pcf::IndiProperty *, 5> properties{
        &m_indiP_counts, &m_indiP_preset, &m_indiP_presetName, &m_indiP_home, &m_indiP_stop };
    auto received = *properties.at( operation );
    if( operation < 2 )
        received["target"].set( operation == 0 ? 20000 : 2 );
    else
    {
        if( operation == 2 )
        {
            for( const auto &name : m_presetNames )
                received[name].setSwitchState( pcf::IndiElement::Off );
        }
        received[operation == 2 ? "two" : "request"].setSwitchState( pcf::IndiElement::On );
    }
    auto callback = m_indiNewCallBacks.at( received.createUniqueKey() ).callBack;
    REQUIRE( callback != nullptr );
    return callback( this, received );
}
} // namespace filterWheelHarness
/// \endcond

namespace libXWCTest
{
/** \addtogroup filterWheelCtrl_unit_test */
namespace filterWheelCtrlTest
{
using namespace filterWheelHarness;
using namespace MagAOX::app;

/// Poll errors without a confirmed READY position recover to NOTHOMED without a motion command.
/** \ingroup filterWheelCtrl_unit_test */
TEST_CASE( "filterWheelCtrl recovers polling failures without homing", "[filterWheelCtrl]" )
{
    // clang-format off
    #ifdef FILTERWHEELCTRL_TEST_DOXYGEN_REF
    filterWheelCtrl::appLogic(); filterWheelCtrl::getSwitch(); filterWheelCtrl::getMoving();
    filterWheelCtrl::getPos(); filterWheelCtrl::recoverFromError(); filterWheelCtrl::resetConnection();
    #endif
    // clang-format on
    const std::string query = GENERATE( "GAST\r", "GN\r", "POS\r" );
    auto    priorState = GENERATE( stateCodes::READY, stateCodes::OPERATING, stateCodes::HOMING, stateCodes::NOTHOMED );
    Fixture f;
    f.state( priorState );
    f.m_powerOnHome        = true;
    f.m_homingState        = priorState == stateCodes::HOMING ? 3 : 0;
    f.m_movingState        = 2;
    int previousFd         = f.m_fileDescrip;
    g_script.m_failedQuery = query;
    REQUIRE( f.appLogic() == 0 );
    REQUIRE( f.state() == stateCodes::NODEVICE );
    REQUIRE( f.m_fileDescrip == 0 );
    REQUIRE( ::fcntl( previousFd, F_GETFD ) == -1 );
    REQUIRE( errno == EBADF );
    REQUIRE( f.m_homingState == 0 );
    REQUIRE( f.m_movingState == 0 );
    REQUIRE( f.m_rawPos == 12345 );
    REQUIRE( f.m_indiP_counts.getState() == INDI_ALERT );
    g_script.m_failedQuery.clear();
    REQUIRE( f.appLogic() == 0 );
    REQUIRE( f.state() == stateCodes::NOTHOMED );
    REQUIRE( f.m_moving == 0 );
    REQUIRE( f.m_indiP_preset.getState() == INDI_IDLE );
    REQUIRE( g_script.m_discoveries == 1 );
    REQUIRE( g_script.m_connections == std::vector<std::string>{ g_script.m_deviceName } );
    REQUIRE_FALSE( commandedMotion() );
    REQUIRE( f.appLogic() == 0 );
    REQUIRE( f.state() == stateCodes::NOTHOMED );
    REQUIRE_FALSE( commandedMotion() );
}

/// ERROR, missing USB devices, and repeated open failures remain recoverable.
/** \ingroup filterWheelCtrl_unit_test */
TEST_CASE( "filterWheelCtrl retries error recovery and USB discovery", "[filterWheelCtrl]" )
{
    // clang-format off
    #ifdef FILTERWHEELCTRL_TEST_DOXYGEN_REF
    filterWheelCtrl::appLogic(); filterWheelCtrl::recoverFromError();
    #endif
    // clang-format on
    Fixture f;
    f.m_powerOnHome        = true;
    f.m_powerOnHomePending = true;
    f.m_homingState        = 4;
    f.state( stateCodes::ERROR );
    REQUIRE( f.appLogic() == 0 );
    REQUIRE( f.state() == stateCodes::NODEVICE );
    REQUIRE_FALSE( f.m_powerOnHomePending );
    for( int absent : { TTY_E_DEVNOTFOUND, TTY_E_NODEVNAMES } )
    {
        g_script.m_discoveryResult = absent;
        REQUIRE( f.appLogic() == 0 );
        REQUIRE( f.state() == stateCodes::NODEVICE );
        REQUIRE( g_script.m_connections.empty() );
    }
    g_script.m_discoveryResult = 0;
    g_script.m_connectResult   = TTY_E_TCGETATTR;
    for( unsigned retry = 0; retry < 3; ++retry )
    {
        REQUIRE( f.appLogic() == 0 );
        REQUIRE( f.state() == stateCodes::NODEVICE );
    }
    g_script.m_connectResult = 0;
    g_script.m_deviceName    = "/dev/filterwheel-new-path";
    REQUIRE( f.appLogic() == 0 );
    REQUIRE( f.state() == stateCodes::NOTHOMED );
    REQUIRE( g_script.m_connections.back() == g_script.m_deviceName );
    REQUIRE_FALSE( commandedMotion() );
}

/// Power-on honors configuration, while failed setup or homing suppresses automatic retries.
/** \ingroup filterWheelCtrl_unit_test */
TEST_CASE( "filterWheelCtrl restricts automatic homing to power-on", "[filterWheelCtrl]" )
{
    // clang-format off
    #ifdef FILTERWHEELCTRL_TEST_DOXYGEN_REF
    filterWheelCtrl::appLogic(); filterWheelCtrl::onPowerOnConnect(); filterWheelCtrl::startHoming();
    filterWheelCtrl::home(); filterWheelCtrl::writeCommand();
    #endif
    // clang-format on
    Fixture f;
    f.state( stateCodes::POWERON );
    SECTION( "automatic homing disabled" )
    {
        f.m_powerOnHome = false;
        REQUIRE( f.appLogic() == 0 );
        REQUIRE( f.state() == stateCodes::NOTHOMED );
        REQUIRE_FALSE( commandedMotion() );
    }
    SECTION( "automatic homing enabled" )
    {
        f.m_powerOnHome  = true;
        g_script.m_speed = "3000\r\n";
        REQUIRE( f.appLogic() == 0 );
        REQUIRE( f.state() == stateCodes::HOMING );
        REQUIRE( f.m_homingState == 1 );
        REQUIRE( std::count( g_script.m_commands.begin(), g_script.m_commands.end(), "GOHOSEQ\r" ) == 1 );
    }
    SECTION( "setup and home write failures suppress automatic retry" )
    {
        const std::string failed = GENERATE( "ANSW0\r",
                                             "MOTTYP2\r",
                                             "AC100.000000\r",
                                             "DEC50.000000\r",
                                             "SP3000.000000\r",
                                             "EN\r",
                                             "HA4\r",
                                             "HL4\r",
                                             "CAHOSEQ\r",
                                             "HP0\r",
                                             "HOSP3000.000000\r",
                                             "GOHOSEQ\r" );
        f.m_powerOnHome          = true;
        g_script.m_failedWrite   = failed;
        REQUIRE( f.appLogic() == 0 );
        REQUIRE( f.state() == stateCodes::NODEVICE );
        REQUIRE_FALSE( f.m_powerOnHomePending );
        REQUIRE( f.m_homingState == 0 );
        g_script.m_failedWrite.clear();
        g_script.m_commands.clear();
        REQUIRE( f.appLogic() == 0 );
        REQUIRE( f.state() == stateCodes::NOTHOMED );
        REQUIRE_FALSE( commandedMotion() );
    }
    SECTION( "failed initial position query suppresses automatic retry" )
    {
        f.m_powerOnHome        = true;
        g_script.m_failedQuery = "POS\r";
        REQUIRE( f.appLogic() == 0 );
        REQUIRE( f.state() == stateCodes::NODEVICE );
        g_script.m_failedQuery.clear();
        REQUIRE( f.appLogic() == 0 );
        REQUIRE( f.state() == stateCodes::NOTHOMED );
        REQUIRE_FALSE( commandedMotion() );
    }
    SECTION( "drain transport errors suppress automatic retry" )
    {
        f.m_powerOnHome        = true;
        g_script.m_drainResult = TTY_E_ERRORONREAD;
        REQUIRE( f.appLogic() == 0 );
        REQUIRE( f.state() == stateCodes::NODEVICE );
        REQUIRE_FALSE( f.m_powerOnHomePending );
        g_script.m_drainResult = TTY_E_TIMEOUTONREADPOLL;
        REQUIRE( f.appLogic() == 0 );
        REQUIRE( f.state() == stateCodes::NOTHOMED );
        REQUIRE_FALSE( commandedMotion() );
    }
}

/// Registered callbacks recover failed writes and reject commands while disconnected.
/** \ingroup filterWheelCtrl_unit_test */
TEST_CASE( "filterWheelCtrl recovers callback write failures", "[filterWheelCtrl]" )
{
    // clang-format off
    #ifdef FILTERWHEELCTRL_TEST_DOXYGEN_REF
    filterWheelCtrl::newCallBack_m_indiP_counts(); filterWheelCtrl::st_newCallBack_m_indiP_counts();
    filterWheelCtrl::moveToRaw(); filterWheelCtrl::moveTo(); filterWheelCtrl::startHoming(); filterWheelCtrl::stop();
    #endif
    // clang-format on
    unsigned operation = GENERATE( 0u, 1u, 2u, 3u, 4u );
    Fixture  f;
    f.m_powerOnHome        = true;
    g_script.m_failedWrite = operation == 4 ? "DI\r" : "EN\r";
    g_script.m_failureCall = operation == 4 ? 2 : 1;
    REQUIRE( f.command( operation ) == -1 );
    REQUIRE( f.state() == stateCodes::NODEVICE );
    REQUIRE( f.m_fileDescrip == 0 );
    REQUIRE( f.m_homingState == 0 );
    REQUIRE( f.m_movingState == 0 );
    g_script.m_commands.clear();
    REQUIRE( f.command( operation ) == -1 );
    REQUIRE( g_script.m_commands.empty() );
    g_script.m_failedWrite.clear();
    REQUIRE( f.appLogic() == 0 );
    REQUIRE( f.state() == stateCodes::NOTHOMED );
    REQUIRE_FALSE( commandedMotion() );
}

/// Every homing and preset step abandons its sequence after a failed motion command.
/** \ingroup filterWheelCtrl_unit_test */
TEST_CASE( "filterWheelCtrl abandons interrupted motion sequences", "[filterWheelCtrl]" )
{
    // clang-format off
    #ifdef FILTERWHEELCTRL_TEST_DOXYGEN_REF
    filterWheelCtrl::appLogic(); filterWheelCtrl::moveToRawRelative(); filterWheelCtrl::home();
    filterWheelCtrl::moveToRaw(); filterWheelCtrl::moveTo();
    #endif
    // clang-format on
    unsigned sequence = GENERATE( 0u, 1u, 2u, 3u, 4u, 5u );
    Fixture  f;
    f.m_powerOnHome   = true;
    f.m_preset_target = 2;
    f.state( sequence < 4 ? stateCodes::HOMING : stateCodes::OPERATING );
    if( sequence < 4 )
        f.m_homingState = sequence + 1;
    else
        f.m_movingState = sequence - 3;
    g_script.m_failedWrite = "EN\r";
    REQUIRE( f.appLogic() == 0 );
    REQUIRE( f.state() == stateCodes::NODEVICE );
    REQUIRE( f.m_homingState == 0 );
    REQUIRE( f.m_movingState == 0 );
    g_script.m_failedWrite.clear();
    g_script.m_commands.clear();
    REQUIRE( f.appLogic() == 0 );
    REQUIRE( f.state() == stateCodes::NOTHOMED );
    REQUIRE_FALSE( commandedMotion() );
}

/// Invalid integer responses cannot advance a sequence or overwrite the last valid position.
/** \ingroup filterWheelCtrl_unit_test */
TEST_CASE( "filterWheelCtrl rejects malformed integer replies", "[filterWheelCtrl]" )
{
    // clang-format off
    #ifdef FILTERWHEELCTRL_TEST_DOXYGEN_REF
    filterWheelCtrl::getPos(); filterWheelCtrl::getMoving(); filterWheelCtrl::readInteger();
    #endif
    // clang-format on
    const std::string invalid = GENERATE(
        "", "\r\n", "invalid\r\n", "123oops\r\n", "0\r\n123\r\n", "999999999999999999999999999999999999\r\n" );
    bool    position = GENERATE( false, true );
    Fixture f;
    f.state( stateCodes::HOMING );
    f.m_homingState = 3;
    if( position )
        g_script.m_position = invalid;
    else
        g_script.m_speed = invalid;
    REQUIRE( f.appLogic() == 0 );
    REQUIRE( f.state() == stateCodes::NODEVICE );
    REQUIRE( f.m_rawPos == 12345 );
    REQUIRE( f.m_homingState == 0 );
    REQUIRE_FALSE( commandedMotion() );
}

/// Recovery respects requested and observed power-off, while actual power-on can re-arm homing.
/** \ingroup filterWheelCtrl_unit_test */
TEST_CASE( "filterWheelCtrl respects power changes during recovery", "[filterWheelCtrl]" )
{
    // clang-format off
    #ifdef FILTERWHEELCTRL_TEST_DOXYGEN_REF
    filterWheelCtrl::appLogic(); filterWheelCtrl::onPowerOff(); filterWheelCtrl::deviceAvailable();
    #endif
    // clang-format on
    Fixture f;
    f.m_powerOnHome = true;
    SECTION( "requested power-off suppresses reconnect and callbacks" )
    {
        f.state( stateCodes::ERROR );
        REQUIRE( f.appLogic() == 0 );
        f.power( "On", "Off" );
        REQUIRE( f.appLogic() == 0 );
        REQUIRE( g_script.m_connections.empty() );
        for( unsigned operation = 0; operation < 5; ++operation )
            REQUIRE( f.command( operation ) == -1 );
        REQUIRE( g_script.m_commands.empty() );
    }
    SECTION( "power-off during a poll prevents later queries" )
    {
        g_script.m_beforeIo = [&]( const std::string &command )
        {
            if( command == "GAST\r" )
                f.power( "On", "Off" );
        };
        REQUIRE( f.appLogic() == 0 );
        REQUIRE( g_script.m_commands == std::vector<std::string>{ "GAST\r" } );
        REQUIRE( f.m_fileDescrip == 0 );
    }
    SECTION( "power-off during connection drain prevents homing" )
    {
        f.state( stateCodes::POWERON );
        g_script.m_beforeIo = [&]( const std::string &command )
        {
            if( command == "drain" )
                f.power( "On", "Off" );
        };
        REQUIRE( f.appLogic() == 0 );
        REQUIRE_FALSE( commandedMotion() );
        REQUIRE_FALSE( f.m_powerOnHomePending );
        g_script.m_beforeIo = nullptr;
        f.power( "On", "On" );
        REQUIRE( f.appLogic() == 0 );
        REQUIRE( f.state() == stateCodes::NOTHOMED );
        REQUIRE_FALSE( commandedMotion() );
    }
    SECTION( "power-off between homing writes abandons the sequence" )
    {
        f.state( stateCodes::POWERON );
        g_script.m_beforeIo = [&]( const std::string &command )
        {
            if( command == "EN\r" )
                f.power( "On", "Off" );
        };
        REQUIRE( f.appLogic() == 0 );
        REQUIRE( f.m_homingState == 0 );
        REQUIRE_FALSE( commandedMotion() );
    }
    SECTION( "observed power-off resets state and power-on re-arms configured homing" )
    {
        f.m_homingState = 3;
        f.m_movingState = 2;
        f.power( "Off", "Off" );
        f.state( stateCodes::POWEROFF );
        REQUIRE( f.onPowerOff() == 0 );
        REQUIRE( f.m_fileDescrip == 0 );
        REQUIRE( f.m_homingState == 0 );
        REQUIRE( f.m_movingState == 0 );
        REQUIRE( f.m_moving == -2 );
        REQUIRE_FALSE( f.m_powerOnHomePending );
        REQUIRE( f.appLogic() == 0 );
        REQUIRE( g_script.m_commands.empty() );
        f.power( "On", "On" );
        f.state( stateCodes::POWERON );
        g_script.m_speed = "3000\r\n";
        REQUIRE( f.appLogic() == 0 );
        REQUIRE( f.state() == stateCodes::HOMING );
        REQUIRE( f.m_homingState == 1 );
    }
}

/// Manual homing after recovery still completes the original dither, offset, and home-preset sequence.
/** \ingroup filterWheelCtrl_unit_test */
TEST_CASE( "filterWheelCtrl supports manual homing after recovery", "[filterWheelCtrl]" )
{
    // clang-format off
    #ifdef FILTERWHEELCTRL_TEST_DOXYGEN_REF
    filterWheelCtrl::startHoming(); filterWheelCtrl::home(); filterWheelCtrl::appLogic();
    #endif
    // clang-format on
    Fixture f;
    f.state( stateCodes::ERROR );
    REQUIRE( f.appLogic() == 0 );
    REQUIRE( f.appLogic() == 0 );
    REQUIRE( f.state() == stateCodes::NOTHOMED );
    REQUIRE_FALSE( commandedMotion() );
    REQUIRE( f.command( 3 ) == 0 );
    REQUIRE( f.state() == stateCodes::HOMING );
    REQUIRE( f.m_homingState == 1 );
    for( int step = 2; step <= 5; ++step )
    {
        REQUIRE( f.appLogic() == 0 );
        REQUIRE( f.state() == stateCodes::HOMING );
        REQUIRE( f.m_homingState == step );
    }
    REQUIRE( f.appLogic() == 0 );
    REQUIRE( f.state() == stateCodes::READY );
    REQUIRE( f.m_homingState == 0 );
    REQUIRE( std::count( g_script.m_commands.begin(), g_script.m_commands.end(), "GOHOSEQ\r" ) == 2 );
    REQUIRE( std::count( g_script.m_commands.begin(), g_script.m_commands.end(), "LR-50000\r" ) == 1 );
    REQUIRE( std::count( g_script.m_commands.begin(), g_script.m_commands.end(), "LA100\r" ) == 2 );
}

/// Valid signed replies and manual position commands retain their existing behavior in NOTHOMED.
/** \ingroup filterWheelCtrl_unit_test */
TEST_CASE( "filterWheelCtrl retains manual movement in NOTHOMED", "[filterWheelCtrl]" )
{
    // clang-format off
    #ifdef FILTERWHEELCTRL_TEST_DOXYGEN_REF
    filterWheelCtrl::getPos(); filterWheelCtrl::getMoving(); filterWheelCtrl::appLogic();
    filterWheelCtrl::newCallBack_m_indiP_counts(); filterWheelCtrl::moveTo(); filterWheelCtrl::moveToRaw();
    #endif
    // clang-format on
    unsigned operation = GENERATE( 0u, 1u, 2u );
    Fixture  f;
    f.state( stateCodes::NOTHOMED );
    g_script.m_position = "-12345\r\n";
    REQUIRE( f.appLogic() == 0 );
    REQUIRE( f.m_rawPos == -12345 );
    REQUIRE( f.state() == stateCodes::NOTHOMED );
    g_script.m_commands.clear();
    REQUIRE( f.command( operation ) == 0 );
    REQUIRE( g_script.m_commands ==
             std::vector<std::string>{ "EN\r", operation == 0 ? "LA20000\r" : "LA60100\r", "M\r" } );
    g_script.m_speed = "-3000\r\n";
    REQUIRE( f.appLogic() == 0 );
    REQUIRE( f.m_moving == 1 );
    REQUIRE( f.state() == stateCodes::NOTHOMED );
}
/// A confirmed stationary READY position is restored after each polling error only with fresh matching counts.
/** \ingroup filterWheelCtrl_unit_test */
TEST_CASE( "filterWheelCtrl restores confirmed stationary READY after reconnect", "[filterWheelCtrl]" )
{
    // clang-format off
    #ifdef FILTERWHEELCTRL_TEST_DOXYGEN_REF
    filterWheelCtrl::appLogic(); filterWheelCtrl::recoverFromError(); filterWheelCtrl::getPos();
    filterWheelCtrl::getMoving(); filterWheelCtrl::resetConnection();
    #endif
    // clang-format on
    const std::string query    = GENERATE( "GAST\r", "GN\r", "POS\r" );
    long              position = GENERATE( -12345L, 0L, 12345L );
    Fixture           f;
    f.m_powerOnHome     = true;
    g_script.m_position = std::to_string( position ) + "\r\n";
    f.ready();
    for( unsigned recovery = 0; recovery < 2; ++recovery )
    {
        f.disconnect( query );
        g_script.m_beforeIo = [&]( const std::string &command )
        {
            if( command == "GN\r" )
                REQUIRE( f.state() == stateCodes::CONNECTED );
        };
        REQUIRE( f.appLogic() == 0 );
        REQUIRE( f.state() == stateCodes::READY );
        REQUIRE( f.m_rawPos == position );
        REQUIRE( f.m_moving == 0 );
        REQUIRE( f.m_homingState == 0 );
        REQUIRE( f.m_movingState == 0 );
        REQUIRE( f.m_indiP_counts.getState() == INDI_IDLE );
        REQUIRE( f.m_indiP_preset.getState() == INDI_IDLE );
        REQUIRE_FALSE( commandedMotion() );
        g_script.m_beforeIo = nullptr;
    }
}

/// Changed counts or any nonzero motor speed invalidate READY recovery permanently for that interruption.
/** \ingroup filterWheelCtrl_unit_test */
TEST_CASE( "filterWheelCtrl rejects changed or moving READY recovery", "[filterWheelCtrl]" )
{
    // clang-format off
    #ifdef FILTERWHEELCTRL_TEST_DOXYGEN_REF
    filterWheelCtrl::appLogic(); filterWheelCtrl::getMoving(); filterWheelCtrl::getPos();
    #endif
    // clang-format on
    Fixture f;
    f.ready();
    SECTION( "changed raw counts during reconnect" )
    {
        long position = GENERATE( 12344L, 12346L, 372345L );
        f.disconnect();
        g_script.m_position = std::to_string( position ) + "\r\n";
    }
    SECTION( "moving during reconnect, including speeds below the normal motion threshold" )
    {
        long speed = GENERATE( -3000L, -1L, 1L, 300L, 3000L );
        f.disconnect();
        g_script.m_speed = std::to_string( speed ) + "\r\n";
    }
    SECTION( "motion observed before the position query fails" )
    {
        g_script.m_speed = "1\r\n";
        f.disconnect( "POS\r" );
    }
    REQUIRE( f.appLogic() == 0 );
    REQUIRE( f.state() == stateCodes::NOTHOMED );
    REQUIRE_FALSE( commandedMotion() );
    g_script.m_position = "12345\r\n";
    g_script.m_speed    = "0\r\n";
    REQUIRE( f.appLogic() == 0 );
    REQUIRE( f.state() == stateCodes::NOTHOMED );
    REQUIRE_FALSE( commandedMotion() );
}

/// Repeated transport, setup, parsing, and discovery failures retain the original READY snapshot until validation.
/** \ingroup filterWheelCtrl_unit_test */
TEST_CASE( "filterWheelCtrl retains READY recovery across failed retries", "[filterWheelCtrl]" )
{
    // clang-format off
    #ifdef FILTERWHEELCTRL_TEST_DOXYGEN_REF
    filterWheelCtrl::appLogic(); filterWheelCtrl::readInteger(); filterWheelCtrl::onPowerOnConnect();
    filterWheelCtrl::recoverFromError();
    #endif
    // clang-format on
    unsigned failure = GENERATE( 0u, 1u, 2u, 3u, 4u, 5u );
    Fixture  f;
    f.ready();
    f.disconnect();
    for( unsigned retry = 0; retry < 3; ++retry )
    {
        if( failure == 0 )
            g_script.m_discoveryResult = TTY_E_DEVNOTFOUND;
        else if( failure == 1 )
            g_script.m_connectResult = TTY_E_TCGETATTR;
        else if( failure == 2 )
        {
            g_script.m_failedWrite    = "ANSW0\r";
            g_script.m_matchingWrites = 0;
        }
        else if( failure == 3 )
            g_script.m_failedQuery = "GN\r";
        else if( failure == 4 )
            g_script.m_position = "invalid\r\n";
        else
            g_script.m_drainResult = TTY_E_ERRORONREAD;
        REQUIRE( f.appLogic() == 0 );
        REQUIRE( f.state() == stateCodes::NODEVICE );
        REQUIRE_FALSE( commandedMotion() );
    }
    g_script.m_discoveryResult = g_script.m_connectResult = 0;
    g_script.m_failedWrite.clear();
    g_script.m_failedQuery.clear();
    g_script.m_position    = "12345\r\n";
    g_script.m_drainResult = TTY_E_TIMEOUTONREAD;
    REQUIRE( f.appLogic() == 0 );
    REQUIRE( f.state() == stateCodes::READY );
    REQUIRE( f.m_rawPos == 12345 );
    REQUIRE_FALSE( commandedMotion() );
}

/// Accepted commands before failure and rejected requests during recovery both invalidate the READY snapshot.
/** \ingroup filterWheelCtrl_unit_test */
TEST_CASE( "filterWheelCtrl invalidates READY recovery on motion requests", "[filterWheelCtrl]" )
{
    // clang-format off
    #ifdef FILTERWHEELCTRL_TEST_DOXYGEN_REF
    filterWheelCtrl::newCallBack_m_indiP_counts(); filterWheelCtrl::moveToRaw(); filterWheelCtrl::moveTo();
    filterWheelCtrl::startHoming(); filterWheelCtrl::stop(); filterWheelCtrl::appLogic();
    #endif
    // clang-format on
    unsigned operation   = GENERATE( 0u, 1u, 2u, 3u, 4u );
    bool     beforeError = GENERATE( false, true );
    Fixture  f;
    f.ready();
    if( beforeError )
    {
        REQUIRE( f.command( operation ) == 0 );
        f.disconnect();
    }
    else
    {
        f.disconnect();
        REQUIRE( f.command( operation ) == -1 );
        REQUIRE( g_script.m_commands.empty() );
    }
    REQUIRE( f.appLogic() == 0 );
    REQUIRE( f.state() == stateCodes::NOTHOMED );
    REQUIRE_FALSE( commandedMotion() );
}

/// Every observed or requested interruption invalidates READY even when power returns before the next FSM loop.
/** \ingroup filterWheelCtrl_unit_test */
TEST_CASE( "filterWheelCtrl remembers power interruptions during READY recovery", "[filterWheelCtrl]" )
{
    // clang-format off
    #ifdef FILTERWHEELCTRL_TEST_DOXYGEN_REF
    filterWheelCtrl::setCallBack_m_indiP_powerChannel(); filterWheelCtrl::st_setCallBack_m_indiP_powerChannel();
    filterWheelCtrl::appStartup(); filterWheelCtrl::appLogic();
    #endif
    // clang-format on
    unsigned update      = GENERATE( 0u, 1u, 2u, 3u, 4u, 5u );
    bool     beforeError = GENERATE( false, true );
    Fixture  f;
    f.ready();
    if( !beforeError )
        f.disconnect();
    if( update == 0 )
        f.power( "Off", "Off" );
    else if( update == 1 )
        f.power( "On", "Off" );
    else if( update == 2 )
        f.power( "Unknown", "On" );
    else if( update == 3 )
        f.power( "On", "Unknown" );
    else if( update == 4 )
        f.power( "", "Off" );
    else
        f.power( "Off", "" );
    f.power( "On", "On" );
    if( beforeError )
        f.disconnect();
    REQUIRE( f.appLogic() == 0 );
    REQUIRE( f.state() == stateCodes::NOTHOMED );
    REQUIRE_FALSE( commandedMotion() );
}

/// Power interruptions during snapshot capture or final verification cannot be hidden by a subsequent On update.
/** \ingroup filterWheelCtrl_unit_test */
TEST_CASE( "filterWheelCtrl validates power history across READY polling", "[filterWheelCtrl]" )
{
    // clang-format off
    #ifdef FILTERWHEELCTRL_TEST_DOXYGEN_REF
    filterWheelCtrl::setCallBack_m_indiP_powerChannel(); filterWheelCtrl::appLogic();
    #endif
    // clang-format on
    Fixture f;
    SECTION( "repeated On updates retain eligibility" )
    {
        f.ready();
        f.disconnect();
        f.power( "On", "On" );
        f.power( "", "On" );
        f.power( "On", "" );
        REQUIRE( f.appLogic() == 0 );
        REQUIRE( f.state() == stateCodes::READY );
    }
    SECTION( "power interruption during reconnect verification" )
    {
        f.ready();
        f.disconnect();
        g_script.m_beforeIo = [&]( const std::string &command )
        {
            if( command == "GAST\r" )
            {
                f.power( "Off", "Off" );
                f.power( "On", "On" );
            }
        };
        REQUIRE( f.appLogic() == 0 );
        REQUIRE( f.state() == stateCodes::NOTHOMED );
    }
    SECTION( "power interruption during the initial READY observation" )
    {
        g_script.m_beforeIo = [&]( const std::string &command )
        {
            if( command == "GAST\r" )
            {
                f.power( "Off", "Off" );
                f.power( "On", "On" );
            }
        };
        f.ready();
        g_script.m_beforeIo = nullptr;
        f.disconnect();
        REQUIRE( f.appLogic() == 0 );
        REQUIRE( f.state() == stateCodes::NOTHOMED );
    }
    SECTION( "ordinary READY polling cannot erase a power interruption" )
    {
        f.ready();
        f.power( "Off", "Off" );
        f.power( "On", "On" );
        f.ready();
        f.disconnect();
        REQUIRE( f.appLogic() == 0 );
        REQUIRE( f.state() == stateCodes::NOTHOMED );
    }
    SECTION( "a rejected home request cannot erase a power interruption" )
    {
        f.ready();
        f.power( "On", "Off" );
        REQUIRE( f.command( 3 ) == -1 );
        f.power( "On", "On" );
        f.ready();
        f.disconnect();
        REQUIRE( f.appLogic() == 0 );
        REQUIRE( f.state() == stateCodes::NOTHOMED );
    }
    REQUIRE_FALSE( commandedMotion() );
}

/// Completed manual homing establishes a fresh READY reference after a power interruption.
/** \ingroup filterWheelCtrl_unit_test */
TEST_CASE( "filterWheelCtrl renews READY recovery after manual homing", "[filterWheelCtrl]" )
{
    // clang-format off
    #ifdef FILTERWHEELCTRL_TEST_DOXYGEN_REF
    filterWheelCtrl::startHoming(); filterWheelCtrl::appLogic(); filterWheelCtrl::setCallBack_m_indiP_powerChannel();
    #endif
    // clang-format on
    Fixture f;
    f.ready();
    f.disconnect();
    f.power( "Off", "Off" );
    f.power( "On", "On" );
    REQUIRE( f.appLogic() == 0 );
    REQUIRE( f.state() == stateCodes::NOTHOMED );
    REQUIRE( f.command( 3 ) == 0 );
    for( unsigned step = 0; step < 5; ++step )
        REQUIRE( f.appLogic() == 0 );
    REQUIRE( f.state() == stateCodes::READY );
    f.disconnect();
    REQUIRE( f.appLogic() == 0 );
    REQUIRE( f.state() == stateCodes::READY );
    REQUIRE_FALSE( commandedMotion() );
}

/// READY recovery requires a completed stationary READY poll and an error originating from that state.
/** \ingroup filterWheelCtrl_unit_test */
TEST_CASE( "filterWheelCtrl requires confirmed READY before recovery", "[filterWheelCtrl]" )
{
    // clang-format off
    #ifdef FILTERWHEELCTRL_TEST_DOXYGEN_REF
    filterWheelCtrl::appLogic(); filterWheelCtrl::onPowerOff(); filterWheelCtrl::resetConnection();
    #endif
    // clang-format on
    Fixture f;
    SECTION( "READY has not yet been polled" )
    {
        f.disconnect();
    }
    SECTION( "a prior READY snapshot cannot promote a different state" )
    {
        auto state = GENERATE( stateCodes::OPERATING, stateCodes::HOMING, stateCodes::NOTHOMED );
        f.ready();
        f.state( state );
        f.disconnect();
    }
    SECTION( "power-on with automatic homing disabled clears the previous snapshot" )
    {
        f.ready();
        f.m_powerOnHome = false;
        f.state( stateCodes::POWERON );
        REQUIRE( f.appLogic() == 0 );
        REQUIRE( f.state() == stateCodes::NOTHOMED );
        f.disconnect();
    }
    SECTION( "an observed power-off hook clears the previous snapshot" )
    {
        f.ready();
        f.power( "Off", "Off" );
        f.state( stateCodes::POWEROFF );
        REQUIRE( f.onPowerOff() == 0 );
        f.power( "On", "On" );
        f.state( stateCodes::POWERON );
        f.m_powerOnHome = false;
        REQUIRE( f.appLogic() == 0 );
        f.disconnect();
    }
    REQUIRE( f.appLogic() == 0 );
    REQUIRE( f.state() == stateCodes::NOTHOMED );
    REQUIRE_FALSE( commandedMotion() );
}

} // namespace filterWheelCtrlTest
} // namespace libXWCTest
