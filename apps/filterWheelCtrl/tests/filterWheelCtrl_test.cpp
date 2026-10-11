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

    /// Deliver observed and requested power through the real subscription callback.
    void power( const std::string &observed /**< [in] Observed power state. */,
                const std::string &target /**< [in] Requested power state. */ );

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
    received.add( pcf::IndiElement( "state", observed ) );
    received.add( pcf::IndiElement( "target", target ) );
    REQUIRE( setCallBack_m_indiP_powerChannel( received ) == 0 );
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

/// Poll errors release the port and recover to NOTHOMED without a motion command.
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
} // namespace filterWheelCtrlTest
} // namespace libXWCTest
