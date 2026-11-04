/** \file kcubeCtrl_test.cpp
 * \brief Offline power-transition, ownership, and recovery tests using the real TMC protocol.
 * \ingroup kcubeCtrl_files
 */
#include "../../../tests/testXWC.hpp"
#include "../../../tests/outletAppTest.hpp"
#include <ftdiHarness.hpp>
#include <future>

#define MagAOXApp outletTestApp
#include "../kcubeCtrl.hpp"
#undef MagAOXApp

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
namespace kcubeHarness
{
using namespace MagAOX::app;
using ftdiHarness::Call;
using ftdiHarness::script;

/// Expose power configuration for startup failures before any device or driver exists.
struct StartupFixture : outletHarness::Controller<kcubeCtrl>
{
    using kcubeCtrl::m_powerMgtEnabled;
};

/// Real controller and private INDI transport with no hardware access.
struct Fixture : outletHarness::Controller<kcubeCtrl>
{
    using kcubeCtrl::m_axis1Enabled;
    using kcubeCtrl::m_axis2Enabled;
    using kcubeCtrl::m_ioTimeout;
    using kcubeCtrl::m_isSet;
    using kcubeCtrl::m_kAxis1;
    using kcubeCtrl::m_kAxis2;
    using kcubeCtrl::m_nextAttempt;
    using kcubeCtrl::m_powerMgtEnabled;
    using kcubeCtrl::m_powerOnWait;
    using kcubeCtrl::m_retryInterval;

    /// Initialize both fake serials and actual INDI registrations.
    Fixture();
    /// Close hardware-free connections before the driver fixture disappears.
    ~Fixture();
    /// Deliver partial power updates through the registered production callback.
    void power( const std::string &observed /**< [in] State, or empty to omit. */,
                const std::string &target /**< [in] Target, or empty to omit. */ );
    /// Run a successful discovery/initialization after expiring the retry gate.
    void ready();
    /// Dispatch a real command property by index: identify, enable, voltage for each axis, then set.
    int command( unsigned index /**< [in] Command selector. */, bool on = true /**< [in] Switch state. */ );
    /// Select a published property by command index.
    pcf::IndiProperty &property( unsigned index /**< [in] Command selector. */ );
    /// Dispatch a supplied request through its registered production callback.
    int dispatch( const pcf::IndiProperty &request /**< [in] Command update. */ );
};

Fixture::Fixture()
{
    outletHarness::g_faults = {};
    m_configName            = "test-kcube";
    m_powerDevice           = "test-pdu";
    m_powerChannel          = "ttmperi";
    m_kAxis1.serial( "axis1" );
    m_kAxis2.serial( "axis2" );
    for( auto *axis : { &m_kAxis1, &m_kAxis2 } )
    {
        axis->preFlushSleep( 0 );
        axis->postFlushSleep( 0 );
        axis->postChanEnableSleep( 0 );
    }
    REQUIRE( registerIndiPropertySet( m_indiP_powerChannel,
                                      m_powerDevice,
                                      m_powerChannel,
                                      MagAOXApp<>::st_setCallBack_m_indiP_powerChannel ) == 0 );
    power( "On", "On" );
    driver();
    state( stateCodes::INITIALIZED );
    REQUIRE( appStartup() == 0 );
}

Fixture::~Fixture()
{
    script.beforeIo = {};
    appShutdown();
    CHECK_FALSE( script.invalidAccess );
}

void Fixture::power( const std::string &observed, const std::string &target )
{
    pcf::IndiProperty request( pcf::IndiProperty::Text, m_powerDevice, m_powerChannel );
    if( !observed.empty() )
        request.add( pcf::IndiElement( "state", observed ) );
    if( !target.empty() )
        request.add( pcf::IndiElement( "target", target ) );
    auto callback = m_indiSetCallBacks.at( request.createUniqueKey() ).callBack;
    REQUIRE( callback( this, request ) == 0 );
}

void Fixture::ready()
{
    m_nextAttempt = {};
    REQUIRE( appLogic() == 0 );
    REQUIRE( state() == stateCodes::READY );
    REQUIRE( m_kAxis1.connected() );
    REQUIRE( m_kAxis2.connected() );
}

pcf::IndiProperty &Fixture::property( unsigned index )
{
    const std::array<pcf::IndiProperty *, 7> properties{ &m_indiP_axis1_identify,
                                                         &m_indiP_axis1_enable,
                                                         &m_indiP_axis1_voltage,
                                                         &m_indiP_axis2_identify,
                                                         &m_indiP_axis2_enable,
                                                         &m_indiP_axis2_voltage,
                                                         &m_indiP_set };
    return *properties.at( index );
}

int Fixture::dispatch( const pcf::IndiProperty &request )
{
    return m_indiNewCallBacks.at( request.createUniqueKey() ).callBack( this, request );
}

int Fixture::command( unsigned index, bool on )
{
    auto request = property( index );
    if( index == 2 || index == 5 )
        request["target"].set( 80.0 );
    else
        request[index == 0 || index == 3 ? "request" : "toggle"].setSwitchState( on ? pcf::IndiElement::On
                                                                                    : pcf::IndiElement::Off );
    return dispatch( request );
}

/// Count actual FTDI calls, optionally selecting a serial and APT command.
unsigned count( const std::string &operation /**< [in] FTDI operation. */,
                const std::string &serial  = "" /**< [in] Optional serial. */,
                unsigned           command = 0 /**< [in] Optional APT command. */ )
{
    return std::count_if( script.calls.begin(),
                          script.calls.end(),
                          [&]( const Call &call )
                          {
                              return call.operation == operation && ( serial.empty() || call.serial == serial ) &&
                                     ( !command || command == call.command );
                          } );
}

/// Remove prior traffic and error capture while retaining fake context ownership.
void clearTraffic()
{
    script.calls.clear();
    outletHarness::g_faults.m_logs.clear();
}
} // namespace kcubeHarness
/// \endcond

namespace libXWCTest
{
/** \addtogroup kcubeCtrl_unit_test */
namespace kcubeCtrlTest
{
using namespace kcubeHarness;

/// Initialization keeps outputs disabled and manual commands preserve their established behavior.
/** \ingroup kcubeCtrl_unit_test */
TEST_CASE( "kcubeCtrl initializes and executes its seven real commands", "[kcubeCtrl]" )
{
    // clang-format off
    #ifdef KCUBECTRL_TEST_DOXYGEN_REF
    kcubeCtrl::appStartup(); kcubeCtrl::appLogic(); kcubeCtrl::axis1Initialize(); kcubeCtrl::axis2Initialize();
    kcubeCtrl::set(); kcubeCtrl::rest(); kcubeCtrl::axis1Enable(); kcubeCtrl::axis2Enable();
    kcubeCtrl::axis1Disable(); kcubeCtrl::axis2Disable(); kcubeCtrl::axis1Voltage(); kcubeCtrl::axis2Voltage();
    #endif
    // clang-format on
    script = {};
    Fixture app;
    app.ready();
    CHECK_FALSE( app.m_axis1Enabled );
    CHECK_FALSE( app.m_axis2Enabled );
    CHECK_FALSE( app.m_isSet );
    CHECK( count( "write", "", 0x0643 ) == 0 );
    for( unsigned index = 0; index < 7; ++index )
        CHECK( app.command( index ) == 0 );
    CHECK( app.m_isSet );
    CHECK( app.m_axis1Enabled );
    CHECK( app.m_axis2Enabled );
    CHECK( app.appLogic() == 0 );
    CHECK( app.state() == stateCodes::OPERATING );
    CHECK( app.command( 6, false ) == 0 );
    CHECK_FALSE( app.m_isSet );
    CHECK_FALSE( app.m_axis1Enabled );
    CHECK_FALSE( app.m_axis2Enabled );
    CHECK( app.appLogic() == 0 );
    CHECK( app.state() == stateCodes::READY );
}

/// The incident's failed voltage poll is nonfatal whether power remains On or changes during I/O.
/** \ingroup kcubeCtrl_unit_test */
TEST_CASE( "kcubeCtrl recovers voltage polling failures and power loss", "[kcubeCtrl][power]" )
{
    // clang-format off
    #ifdef KCUBECTRL_TEST_DOXYGEN_REF
    kcubeCtrl::appLogic(); kcubeCtrl::recover(); kcubeCtrl::resetConnection(); kcubeCtrl::transact();
    kcubeCtrl::setCallBack_m_indiP_powerChannel();
    #endif
    // clang-format on
    const auto serial     = GENERATE( std::string( "axis1" ), std::string( "axis2" ) );
    const auto operation  = GENERATE( std::string( "write" ), std::string( "read" ) );
    int        transition = GENERATE( 0, 1, 2, 3 );
    script                = {};
    Fixture app;
    app.ready();
    float before = app.m_indiP_axis1_voltage["current"].get<float>();
    clearTraffic();
    script.failure  = operation;
    script.serial   = serial;
    bool injected   = false;
    script.beforeIo = [&]( const Call &call )
    {
        if( !injected && call.operation == operation && call.serial == serial )
        {
            injected = true;
            if( transition == 1 )
                app.power( "Off", "Off" );
            if( transition == 2 )
                app.power( "", "Off" );
            if( transition == 3 )
                app.power( "Unk", "" );
        }
    };
    CHECK( app.appLogic() == 0 );
    CHECK( app.m_shutdown == 0 );
    CHECK_FALSE( app.m_kAxis1.opened() );
    CHECK_FALSE( app.m_kAxis2.opened() );
    CHECK( app.m_indiP_axis1_voltage["current"].get<float>() == before );
    CHECK( outletHarness::g_faults.m_logs.empty() == ( transition != 0 ) );
    if( serial == "axis1" )
        CHECK( count( "write", "axis2" ) == 0 );
    auto calls = script.calls.size();
    CHECK( app.appLogic() == 0 );
    CHECK( script.calls.size() == calls );
    script.beforeIo = {};
    script.failure.clear();
    app.power( "On", "On" );
    app.ready();
    CHECK_FALSE( app.m_isSet );
}

/// Target-only shutdown suppresses all commands and connection work while preserving actual power.
/** \ingroup kcubeCtrl_unit_test */
TEST_CASE( "kcubeCtrl gates every state and command on requested power", "[kcubeCtrl][power]" )
{
    // clang-format off
    #ifdef KCUBECTRL_TEST_DOXYGEN_REF
    kcubeCtrl::powerOnExpected(); kcubeCtrl::powerOffRequested(); kcubeCtrl::commandAllowed();
    kcubeCtrl::newCallBack_m_indiP_axis1_identify(); kcubeCtrl::newCallBack_m_indiP_axis2_identify();
    kcubeCtrl::newCallBack_m_indiP_axis1_enable(); kcubeCtrl::newCallBack_m_indiP_axis2_enable();
    kcubeCtrl::newCallBack_m_indiP_axis1_voltage(); kcubeCtrl::newCallBack_m_indiP_axis2_voltage();
    kcubeCtrl::newCallBack_m_indiP_set();
    #endif
    // clang-format on
    script = {};
    Fixture app;
    app.ready();
    clearTraffic();
    app.power( "", "Off" );
    CHECK( app.powerState() == 1 );
    CHECK( app.powerStateTarget() == 0 );
    for( auto state : { stateCodes::READY,
                        stateCodes::OPERATING,
                        stateCodes::POWERON,
                        stateCodes::NOTCONNECTED,
                        stateCodes::CONNECTED,
                        stateCodes::ERROR,
                        stateCodes::NODEVICE } )
    {
        app.state( state );
        for( unsigned index = 0; index < 7; ++index )
            CHECK( app.command( index ) < 0 );
        CHECK( app.appLogic() == 0 );
    }
    CHECK( count( "open" ) == 0 );
    CHECK( count( "write" ) == 0 );
    CHECK( count( "read" ) == 0 );
    CHECK( outletHarness::g_faults.m_logs.empty() );
}

/// Every startup transport failure is recoverable without retaining the peer's handle or replaying a set command.
/** \ingroup kcubeCtrl_unit_test */
TEST_CASE( "kcubeCtrl retries each failed startup operation", "[kcubeCtrl]" )
{
    // clang-format off
    #ifdef KCUBECTRL_TEST_DOXYGEN_REF
    kcubeCtrl::appLogic(); kcubeCtrl::initialize(); kcubeCtrl::recover();
    #endif
    // clang-format on
    int      mode = GENERATE( 0, 1, 2 ); // Transport failure, target Off, rapid Off/On.
    unsigned steps;
    script = {};
    {
        Fixture app;
        app.ready();
        steps = script.calls.size();
    }
    for( unsigned step = 1; step <= steps; ++step )
    {
        INFO( "startup boundary " << step );
        script = {};
        Fixture  app;
        unsigned seen   = 0;
        script.beforeIo = [&]( const Call &call )
        {
            if( ++seen == step )
            {
                if( mode == 0 )
                {
                    script.failure = call.operation;
                    script.serial  = call.serial;
                    script.matches = 0;
                }
                else
                {
                    app.power( "", "Off" );
                    if( mode == 2 )
                        app.power( "", "On" );
                }
            }
        };
        CHECK( app.appLogic() == 0 );
        CHECK( app.state() == stateCodes::NODEVICE );
        CHECK_FALSE( app.m_kAxis1.opened() );
        CHECK_FALSE( app.m_kAxis2.opened() );
        CHECK( app.m_shutdown == 0 );
        script.beforeIo = {};
        script.failure.clear();
        app.power( "On", "On" );
        app.ready();
        CHECK( count( "write", "", 0x0643 ) == 0 );
    }
}

/// Missing axes are retried at a bounded rate with balanced context ownership in either enumeration order.
/** \ingroup kcubeCtrl_unit_test */
TEST_CASE( "kcubeCtrl handles staggered enumeration and persistent absence", "[kcubeCtrl]" )
{
    // clang-format off
    #ifdef KCUBECTRL_TEST_DOXYGEN_REF
    kcubeCtrl::appLogic(); kcubeCtrl::resetConnection();
    #endif
    // clang-format on
    auto serial = GENERATE( std::string( "axis1" ), std::string( "axis2" ) );
    script      = {};
    Fixture app;
    script.failure    = "open";
    script.serial     = serial;
    script.result     = -3;
    script.persistent = true;
    CHECK( app.appLogic() == 0 );
    auto calls = script.calls.size();
    CHECK( app.appLogic() == 0 );
    CHECK( script.calls.size() == calls );
    for( unsigned n = 0; n < 5; ++n )
    {
        app.m_nextAttempt = {};
        CHECK( app.appLogic() == 0 );
        CHECK_FALSE( app.m_kAxis1.opened() );
        CHECK_FALSE( app.m_kAxis2.opened() );
    }
    CHECK( outletHarness::g_faults.m_logs.size() == 1 );
    script.failure.clear();
    app.ready();
}

/// Power loss during successful I/O and rapid Off/On invalidate the complete initialization generation.
/** \ingroup kcubeCtrl_unit_test */
TEST_CASE( "kcubeCtrl remembers power interruptions even after power returns", "[kcubeCtrl][power]" )
{
    // clang-format off
    #ifdef KCUBECTRL_TEST_DOXYGEN_REF
    kcubeCtrl::setCallBack_m_indiP_powerChannel(); kcubeCtrl::communicationAllowed(); kcubeCtrl::appLogic();
    #endif
    // clang-format on
    int mode = GENERATE( 0, 1, 2 );
    script   = {};
    Fixture app;
    app.ready();
    clearTraffic();
    if( mode == 0 )
    {
        app.power( "", "Off" );
        app.power( "", "On" );
        CHECK( app.command( 1 ) < 0 );
    }
    else
    {
        bool injected   = false;
        script.beforeIo = [&]( const Call &call )
        {
            if( !injected && call.operation == "write" )
            {
                injected = true;
                app.power( "", "Off" );
                if( mode == 2 )
                    app.power( "", "On" );
            }
        };
        CHECK( app.appLogic() == 0 );
        CHECK_FALSE( app.m_kAxis1.opened() );
        CHECK( count( "write", "axis2" ) == 0 );
        script.beforeIo = {};
        app.power( "On", "On" );
    }
    app.ready();
    CHECK( count( "open" ) == 2 );
    CHECK( count( "write", "", 0x0643 ) == 0 );
    clearTraffic();
    app.power( "On", "On" );
    app.ready();
    CHECK( count( "open" ) == 0 );
}

/// A queued callback must revalidate admission after a polling failure releases the transport lock.
/** \ingroup kcubeCtrl_unit_test */
TEST_CASE( "kcubeCtrl serializes callbacks with recovery and shutdown", "[kcubeCtrl][threads]" )
{
    // clang-format off
    #ifdef KCUBECTRL_TEST_DOXYGEN_REF
    kcubeCtrl::appLogic(); kcubeCtrl::commandAllowed(); kcubeCtrl::appShutdown(); kcubeCtrl::enableRequest();
    #endif
    // clang-format on
    script = {};
    Fixture app;
    app.ready();
    clearTraffic();
    auto request = app.property( 1 );
    request["toggle"].setSwitchState( pcf::IndiElement::On );
    std::promise<void> entered;
    std::future<int>   queued;
    script.failure  = "write";
    script.beforeIo = [&]( const Call &call )
    {
        if( call.operation != "write" || queued.valid() )
            return;
        queued = std::async( std::launch::async,
                             [&]
                             {
                                 entered.set_value();
                                 return app.dispatch( request );
                             } );
        entered.get_future().wait();
    };
    CHECK( app.appLogic() == 0 );
    REQUIRE( queued.valid() );
    CHECK( queued.get() < 0 );
    CHECK( count( "write" ) == 1 );
    CHECK( app.state() == stateCodes::NODEVICE );
    script.beforeIo = {};
    script.failure.clear();
    app.ready();
    CHECK( app.appShutdown() == 0 );
    clearTraffic();
    for( unsigned index = 0; index < 7; ++index )
        CHECK( app.command( index ) < 0 );
    CHECK( app.appLogic() == 0 );
    CHECK( script.calls.empty() );
}

/// No remaining commands are issued after any partial set/rest failure, and recovery leaves outputs disabled.
/** \ingroup kcubeCtrl_unit_test */
TEST_CASE( "kcubeCtrl abandons partial set and rest sequences", "[kcubeCtrl]" )
{
    // clang-format off
    #ifdef KCUBECTRL_TEST_DOXYGEN_REF
    kcubeCtrl::set(); kcubeCtrl::rest(); kcubeCtrl::enable(); kcubeCtrl::voltage(); kcubeCtrl::recover();
    #endif
    // clang-format on
    bool     rest    = GENERATE( false, true );
    unsigned failure = GENERATE( 1u, 2u, 3u, 4u );
    script           = {};
    Fixture app;
    app.ready();
    if( rest )
        REQUIRE( app.command( 6 ) == 0 );
    clearTraffic();
    script.failure     = "write";
    script.failureCall = failure;
    CHECK( app.command( 6, !rest ) < 0 );
    CHECK( count( "write" ) == failure );
    CHECK_FALSE( app.m_isSet );
    CHECK( app.state() == stateCodes::NODEVICE );
    script.failure.clear();
    clearTraffic();
    app.ready();
    CHECK( count( "write", "", 0x0643 ) == 0 );
    CHECK_FALSE( app.m_axis1Enabled );
    CHECK_FALSE( app.m_axis2Enabled );
}

/// Every command enters recovery on a transport error, including callbacks whose return is ignored by INDI.
/** \ingroup kcubeCtrl_unit_test */
TEST_CASE( "kcubeCtrl recovers every command transport failure", "[kcubeCtrl]" )
{
    // clang-format off
    #ifdef KCUBECTRL_TEST_DOXYGEN_REF
    kcubeCtrl::identifyRequest(); kcubeCtrl::enableRequest(); kcubeCtrl::voltageRequest();
    kcubeCtrl::newCallBack_m_indiP_set(); kcubeCtrl::recover();
    #endif
    // clang-format on
    unsigned index = GENERATE( 0u, 1u, 2u, 3u, 4u, 5u, 6u );
    script         = {};
    Fixture app;
    app.ready();
    clearTraffic();
    script.failure = "write";
    CHECK( app.command( index ) < 0 );
    CHECK( app.state() == stateCodes::NODEVICE );
    CHECK_FALSE( app.m_kAxis1.opened() );
    CHECK_FALSE( app.m_kAxis2.opened() );
    CHECK( count( "write" ) == 1 );
    script.failure.clear();
    app.ready();
    CHECK_FALSE( app.m_isSet );
}

/// Failure before or during property registration still permits idempotent shutdown.
/** \ingroup kcubeCtrl_unit_test */
TEST_CASE( "kcubeCtrl cleans up incomplete startup", "[kcubeCtrl]" )
{
    // clang-format off
    #ifdef KCUBECTRL_TEST_DOXYGEN_REF
    kcubeCtrl::appStartup(); kcubeCtrl::appShutdown(); kcubeCtrl::resetConnection();
    #endif
    // clang-format on
    unsigned failure        = GENERATE( 0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u );
    script                  = {};
    outletHarness::g_faults = {};
    StartupFixture app;
    if( failure )
    {
        app.m_powerMgtEnabled                      = false;
        outletHarness::g_faults.m_failRegistration = failure;
    }
    CHECK( app.appStartup() < 0 );
    CHECK( app.appShutdown() == 0 );
    CHECK( app.appShutdown() == 0 );
    CHECK( script.calls.empty() );
}

/// Invalid identities and unbounded timing settings stop startup before USB access.
/** \ingroup kcubeCtrl_unit_test */
TEST_CASE( "kcubeCtrl validates connection configuration", "[kcubeCtrl]" )
{
    // clang-format off
    #ifdef KCUBECTRL_TEST_DOXYGEN_REF
    kcubeCtrl::setupConfig(); kcubeCtrl::loadConfig(); kcubeCtrl::loadConfigImpl();
    #endif
    // clang-format on
    unsigned mode           = GENERATE( 0u, 1u, 2u, 3u, 4u, 5u, 6u );
    script                  = {};
    outletHarness::g_faults = {};
    StartupFixture app;
    std::string    text = "[axis1]\nserial=axis1\n[axis2]\nserial=";
    text += mode == 1 ? "" : mode == 2 ? "axis1" : "axis2";
    text += "\n[connection]\nretryInterval=";
    text += mode == 3 ? "0" : mode == 4 ? "60001" : "1000";
    text += "\n[timeouts]\nio=";
    text += mode == 5 ? "0" : mode == 6 ? "60001" : "1000";
    app.configText( text + "\n" );
    CHECK( ( app.loadConfigImpl( app.config ) == 0 ) == ( mode == 0 ) );
    app.loadConfig();
    CHECK( bool( app.m_shutdown ) == ( mode != 0 ) );
    CHECK( script.calls.empty() );
}

/// Cleanup is idempotent even when the underlying release reports a disconnected USB interface.
/** \ingroup kcubeCtrl_unit_test */
TEST_CASE( "kcubeCtrl survives close errors and repeated power-off cleanup", "[kcubeCtrl][power]" )
{
    // clang-format off
    #ifdef KCUBECTRL_TEST_DOXYGEN_REF
    kcubeCtrl::onPowerOff(); kcubeCtrl::whilePowerOff(); kcubeCtrl::appShutdown(); kcubeCtrl::resetConnection();
    #endif
    // clang-format on
    script = {};
    Fixture app;
    app.ready();
    script.failure    = "close";
    script.persistent = true;
    app.power( "Off", "Off" );
    app.state( stateCodes::POWEROFF );
    CHECK( app.onPowerOff() == 0 );
    CHECK( app.onPowerOff() == 0 );
    CHECK( app.whilePowerOff() == 0 );
    CHECK_FALSE( app.m_kAxis1.connected() );
    CHECK_FALSE( app.m_kAxis2.opened() );
    CHECK( count( "close" ) == 2 );
    script.failure.clear();
    app.power( "On", "On" );
    app.state( stateCodes::POWERON );
    app.ready();
}

/// Malformed callback input never reaches the transport, while finite voltages retain clamping behavior.
/** \ingroup kcubeCtrl_unit_test */
TEST_CASE( "kcubeCtrl validates callback content and property identity", "[kcubeCtrl]" )
{
    // clang-format off
    #ifdef KCUBECTRL_TEST_DOXYGEN_REF
    kcubeCtrl::voltageRequest(); kcubeCtrl::enableRequest(); kcubeCtrl::identifyRequest();
    kcubeCtrl::newCallBack_m_indiP_set();
    #endif
    // clang-format on
    script = {};
    Fixture app;
    app.ready();
    clearTraffic();
    for( unsigned index = 0; index < 7; ++index )
    {
        auto &property = app.property( index );
        auto  callback = app.m_indiNewCallBacks.at( property.createUniqueKey() ).callBack;
        auto  request  = property;
        request.setDevice( "wrong" );
        CHECK( callback( &app, request ) < 0 );
        request = property;
        request.setName( "wrong" );
        CHECK( callback( &app, request ) < 0 );
        pcf::IndiProperty missing( property.getType(), property.getDevice(), property.getName() );
        CHECK( callback( &app, missing ) < 0 );
    }
    for( unsigned axis : { 2u, 5u } )
    {
        for( auto value : { "nan", "inf", "-inf", "garbage" } )
        {
            auto request = app.property( axis );
            request["target"].set( std::string( value ) );
            CHECK( app.dispatch( request ) < 0 );
        }
    }
    CHECK( script.calls.empty() );
    auto request = app.property( 2 );
    request["target"].set( 200.0 );
    CHECK( app.dispatch( request ) == 0 );
    CHECK( app.m_indiP_axis1_voltage["target"].get<float>() == 150 );
}

/// Unknown power gates communication, unmanaged power remains supported, and boot/retry delays are explicit.
/** \ingroup kcubeCtrl_unit_test */
TEST_CASE( "kcubeCtrl supports partial unknown power and configurable startup delay", "[kcubeCtrl][power]" )
{
    // clang-format off
    #ifdef KCUBECTRL_TEST_DOXYGEN_REF
    kcubeCtrl::appLogic(); kcubeCtrl::powerOnExpected(); kcubeCtrl::powerOffRequested();
    #endif
    // clang-format on
    script = {};
    Fixture app;
    app.power( "", "Unk" );
    CHECK_FALSE( app.powerOnExpected() );
    CHECK_FALSE( app.powerOffRequested() );
    CHECK( app.appLogic() == 0 );
    CHECK( script.calls.empty() );
    app.m_powerOnWait = 20;
    app.power( "On", "On" );
    CHECK( app.appLogic() == 0 );
    CHECK( script.calls.empty() );
    app.ready();
    CHECK( app.onPowerOff() == 0 );
    app.power( "Off", "Off" );
    app.m_powerMgtEnabled = false;
    app.state( stateCodes::POWERON );
    clearTraffic();
    app.ready();
    CHECK( count( "open" ) == 2 );
}

/// The real dependency's deadline and parser errors return to the application recovery path.
/** \ingroup kcubeCtrl_unit_test */
TEST_CASE( "kcubeCtrl recovers incomplete and malformed transport responses", "[kcubeCtrl]" )
{
    // clang-format off
    #ifdef KCUBECTRL_TEST_DOXYGEN_REF
    kcubeCtrl::appLogic(); kcubeCtrl::transact(); kcubeCtrl::recover();
    #endif
    // clang-format on
    int mode = GENERATE( 0, 1, 2, 3 );
    script   = {};
    Fixture app;
    app.ready();
    clearTraffic();
    app.m_kAxis1.ioTimeout( 5 );
    if( mode == 0 )
        script.readChunk = 0;
    if( mode == 1 )
    {
        script.failure = "write";
        script.result  = 3;
    }
    if( mode == 2 )
        script.replyHook = []( auto &reply ) { reply[0] ^= 1; };
    if( mode == 3 )
        script.replyHook = []( auto &reply ) { reply.resize( 7 ); };
    auto start = std::chrono::steady_clock::now();
    CHECK( app.appLogic() == 0 );
    CHECK( std::chrono::steady_clock::now() - start < std::chrono::seconds( 1 ) );
    CHECK( app.state() == stateCodes::NODEVICE );
    CHECK( count( "write", "axis2" ) == 0 );
}
} // namespace kcubeCtrlTest
} // namespace libXWCTest
