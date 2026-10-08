/** \file zaberLowLevel_power_test.cpp
 * \brief Offline power-target and communication-error regression tests for ASCII Zaber control.
 * \ingroup zaberLowLevel_files
 */
#include "../../../tests/testXWC.hpp"
#include "../../../tests/outletAppTest.hpp"

extern "C"
{
#include "../za_serial.c"
}

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
namespace zaberPowerHarness
{
/// Scripted transport operation selected to fail.
enum class Operation { None, Connect, Disconnect, Drain, Send, Receive };

/// Hardware-free transport state and one-shot power-target injection.
struct Script
{
    /// Operation to fail, or None for the normal scripted transport.
    Operation m_failure{ Operation::None };

    /// Invocation of the selected operation to fail.
    unsigned m_failureCall{ 1 };

    /// Counts of connect, disconnect, drain, send, and receive operations.
    std::array<unsigned, 6> m_calls{};

    /// Actual ASCII commands passed by the production app and stage helpers.
    std::vector<std::string> m_commands;

    /// Queued valid replies, followed by a normal discovery timeout when empty.
    std::deque<std::string> m_replies;

    /// Hook injecting the real power callback immediately before a transport failure returns.
    std::function<void()> m_beforeFailure;

    /// Whether the hook has already been invoked.
    bool m_injected{ false };
};

/// Current sequential fixture's script; no transport can open a device.
inline Script g_script;

/// Count an operation and invoke its one-shot failure hook when selected.
bool fail( Operation operation /**< [in] Operation being attempted. */ );

/// Simulate opening an ASCII port without opening a file or changing hardware.
int connect( z_port *port, /**< [out] Scripted port marker. */
             const char *name /**< [in] Ignored device path. */ );

/// Simulate closing a port without closing a real descriptor.
int disconnect( z_port port /**< [in] Ignored scripted port marker. */ );

/// Simulate draining a serial buffer without reading a descriptor.
int drain( z_port port /**< [in] Ignored scripted port marker. */ );

/// Record a command and return its byte count or a scripted error.
int send( z_port port, /**< [in] Ignored scripted port marker. */
          const char *command, /**< [in] Actual ASCII command. */
          size_t length /**< [in] Command byte count. */ );

/// Return a queued reply, scripted error, or timeout without reading a device.
int receive( z_port port, /**< [in] Ignored scripted port marker. */
             char *buffer, /**< [out] Buffer receiving the queued reply. */
             int length /**< [in] Response buffer capacity. */ );

bool fail( Operation operation )
{
    unsigned call = ++g_script.m_calls[static_cast<size_t>( operation )];
    if( operation != g_script.m_failure || call != g_script.m_failureCall )
        return false;
    if( g_script.m_beforeFailure && !g_script.m_injected )
    {
        g_script.m_injected = true;
        g_script.m_beforeFailure();
    }
    return true;
}

int connect( z_port *port, const char * )
{
    *port = fail( Operation::Connect ) ? 0 : 99;
    return *port ? Z_SUCCESS : Z_ERROR_SYSTEM_ERROR;
}

int disconnect( z_port )
{
    return fail( Operation::Disconnect ) ? Z_ERROR_SYSTEM_ERROR : Z_SUCCESS;
}

int drain( z_port )
{
    return fail( Operation::Drain ) ? Z_ERROR_SYSTEM_ERROR : Z_SUCCESS;
}

int send( z_port, const char *command, size_t length )
{
    g_script.m_commands.emplace_back( command, length );
    return fail( Operation::Send ) ? Z_ERROR_SYSTEM_ERROR : static_cast<int>( length );
}

int receive( z_port, char *buffer, int length )
{
    if( fail( Operation::Receive ) )
        return Z_ERROR_SYSTEM_ERROR;
    if( g_script.m_replies.empty() )
        return Z_ERROR_TIMEOUT;
    auto reply = g_script.m_replies.front();
    g_script.m_replies.pop_front();
    size_t count = std::min( reply.size(), static_cast<size_t>( length - 1 ) );
    std::copy_n( reply.data(), count, buffer );
    return count;
}
} // namespace zaberPowerHarness

#define za_connect zaberPowerHarness::connect
#define za_disconnect zaberPowerHarness::disconnect
#define za_drain zaberPowerHarness::drain
#define za_send zaberPowerHarness::send
#define za_receive zaberPowerHarness::receive
#define MagAOXApp outletTestApp
#include "../zaberLowLevel.hpp"
#undef MagAOXApp
#undef za_receive
#undef za_send
#undef za_drain
#undef za_disconnect
#undef za_connect
/// \endcond

namespace libXWCTest
{
/** \addtogroup zaberLowLevel_unit_test
 * \ingroup application_unit_test
 */
namespace zaberLowLevelTest
{
using namespace MagAOX::app;
using namespace outletHarness;
using namespace zaberPowerHarness;

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
/// Real app and stage methods with captured logging, isolated state, and scripted serial operations.
struct PowerFixture : Controller<zaberLowLevel>
{
    /// Expose only the scripted port marker.
    using zaberLowLevel::m_port;

    /// Initialize one retained stage and its real INDI command properties.
    PowerFixture();

    /// Deliver observed and target states through the production power callback.
    void power( const std::string &observed, /**< [in] Observed power state. */
                const std::string &target /**< [in] Requested power state. */ );

    /// Deliver a target-only Off update without changing the observed On state.
    void targetOff();

    /// Dispatch one of the seven commands through its actual registered callback.
    int command( unsigned operation /**< [in] Move, home, home-all, halt, emergency halt, knob, or LED index. */ );
};

PowerFixture::PowerFixture()
{
    m_sysPath = m_directory.m_path + "/sys";
    std::filesystem::create_directories( m_sysPath + "/" + m_configName );
    m_stages.emplace_back( this );
    m_stages.back().name( "stageA" );
    m_stages.back().serial( "49820" );
    m_stages.back().deviceAddress( 1 );
    std::ofstream saved( m_sysPath + "/" + m_configName + "/stageA" );
    saved << "12345\n0\n54321\n77\n";
    saved.close();
    state( stateCodes::INITIALIZED );
    REQUIRE( appStartup() == 0 );
    state( stateCodes::READY );
    m_port = 99;
    power( "On", "On" );
    outletHarness::g_faults.m_logs.clear();
}

void PowerFixture::power( const std::string &observed, const std::string &target )
{
    auto update = property( "pdu2", "stagezaber", "state", observed );
    update.add( pcf::IndiElement( "target", target ) );
    REQUIRE( setCallBack_m_indiP_powerChannel( update ) == 0 );
}

void PowerFixture::targetOff()
{
    REQUIRE( setCallBack_m_indiP_powerChannel( property( "pdu2", "stagezaber", "target", "Off" ) ) == 0 );
    REQUIRE( powerState() == 1 );
    REQUIRE( powerStateTarget() == 0 );
}

int PowerFixture::command( unsigned operation )
{
    const std::array<pcf::IndiProperty *, 7> properties{ &m_indiP_tgt_pos,
                                                       &m_indiP_req_home,
                                                       &m_indiP_req_home_all,
                                                       &m_indiP_req_halt,
                                                       &m_indiP_req_ehalt,
                                                       &m_indiP_knob_enable,
                                                       &m_indiP_led_enable };
    auto request = *properties.at( operation );
    if( operation == 0 )
        request["stageA"].set( 200 );
    else
        request[operation == 2 ? "request" : "stageA"].setSwitchState( pcf::IndiElement::On );
    auto callback = m_indiNewCallBacks.at( request.createUniqueKey() ).callBack;
    REQUIRE( callback != nullptr );
    return callback( this, request );
}
/// \endcond

/// A known Off target prevents new connection, discovery, polling, or stage commands while observed power is On.
/** \ingroup zaberLowLevel_unit_test */
TEST_CASE( "Zaber waits for power-off without starting more communication", "[zaberLowLevel][power]" )
{
    // clang-format off
    #ifdef ZABERLOWLEVEL_TEST_DOXYGEN_REF
    zaberLowLevel::powerOffRequested(); zaberLowLevel::appLogic(); zaberLowLevel::connect();
    zaberLowLevel::refreshStageDiscovery(); zaberLowLevel::loadStages();
    MagAOX::app::MagAOXApp<true>::setCallBack_m_indiP_powerChannel();
    #endif
    // clang-format on
    outletHarness::g_faults = {};
    g_script = {};
    PowerFixture app;
    app.targetOff();
    for( auto stateCode : { stateCodes::READY, stateCodes::NOTCONNECTED, stateCodes::CONNECTED, stateCodes::ERROR } )
    {
        app.state( stateCode );
        CHECK( app.appLogic() == 0 );
        CHECK( app.state() == stateCode );
    }
    CHECK( app.connect() == ZC_NOT_CONNECTED );
    CHECK( app.refreshStageDiscovery() == ZC_ERROR );
    std::string snapshot = "@01 0 OK IDLE -- 49820\n";
    CHECK( app.loadStages( snapshot ) == ZC_ERROR );
    for( unsigned operation = 0; operation < 7; ++operation )
        CHECK( app.command( operation ) < 0 );
    CHECK( g_script.m_calls == std::array<unsigned, 6>{} );
    CHECK( outletHarness::g_faults.m_logs.empty() );
}

/// Discovery failures arriving with an Off target are silent; On/On failures retain error reporting and FSM recovery.
/** \ingroup zaberLowLevel_unit_test */
TEST_CASE( "Zaber discovery checks power after transport failures", "[zaberLowLevel][power]" )
{
    // clang-format off
    #ifdef ZABERLOWLEVEL_TEST_DOXYGEN_REF
    zaberLowLevel::refreshStageDiscovery(); zaberLowLevel::powerOnExpected(); zaberLowLevel::appLogic();
    #endif
    // clang-format on
    for( auto failure : { Operation::Drain, Operation::Send, Operation::Receive } )
    {
        for( bool turningOff : { false, true } )
        {
            outletHarness::g_faults = {};
            g_script = {};
            PowerFixture app;
            g_script.m_failure = failure;
            if( turningOff )
                g_script.m_beforeFailure = [&] { app.targetOff(); };
            REQUIRE( app.appLogic() == 0 );
            CHECK( app.powerState() == 1 );
            CHECK( app.state() == ( turningOff ? stateCodes::READY : stateCodes::ERROR ) );
            CHECK( outletHarness::g_faults.m_logs.empty() == turningOff );
        }
    }
}

/// Connection failures during target changes retain their return codes and suppress expected power-loss errors.
/** \ingroup zaberLowLevel_unit_test */
TEST_CASE( "Zaber connection checks power after each failing transport phase", "[zaberLowLevel][power]" )
{
    // clang-format off
    #ifdef ZABERLOWLEVEL_TEST_DOXYGEN_REF
    zaberLowLevel::connect();
    #endif
    // clang-format on
    for( auto [failure, call] : { std::pair{ Operation::Connect, 1u }, { Operation::Drain, 1u },
                                 { Operation::Drain, 2u }, { Operation::Send, 1u }, { Operation::Send, 2u },
                                 { Operation::Receive, 1u } } )
    {
        for( bool turningOff : { false, true } )
        {
            outletHarness::g_faults = {};
            g_script = {};
            PowerFixture app;
            app.state( stateCodes::NOTCONNECTED );
            app.m_port = 0;
            g_script.m_failure = failure;
            g_script.m_failureCall = call;
            if( turningOff )
                g_script.m_beforeFailure = [&] { app.targetOff(); };
            CHECK( app.connect() == ( failure == Operation::Connect ? ZC_NOT_CONNECTED : ZC_ERROR ) );
            CHECK( outletHarness::g_faults.m_logs.empty() == turningOff );
            CHECK( app.state() == ( turningOff || failure == Operation::Connect ? stateCodes::NOTCONNECTED :
                                   stateCodes::ERROR ) );
        }
    }
}

/// Parent callbacks preserve stage-level suppression instead of reporting the same expected failure again.
/** \ingroup zaberLowLevel_unit_test */
TEST_CASE( "Zaber parent commands preserve stage power-loss suppression", "[zaberLowLevel][power]" )
{
    // clang-format off
    #ifdef ZABERLOWLEVEL_TEST_DOXYGEN_REF
    zaberLowLevel::newCallBack_m_indiP_tgt_pos(); zaberLowLevel::newCallBack_m_indiP_req_home();
    zaberLowLevel::newCallBack_m_indiP_req_home_all(); zaberLowLevel::newCallBack_m_indiP_req_halt();
    zaberLowLevel::newCallBack_m_indiP_req_ehalt(); zaberLowLevel::newCallBack_m_indiP_knob_enable();
    zaberLowLevel::newCallBack_m_indiP_led_enable();
    zaberLowLevel::st_newCallBack_m_indiP_tgt_pos(); zaberLowLevel::st_newCallBack_m_indiP_req_home();
    zaberLowLevel::st_newCallBack_m_indiP_req_home_all(); zaberLowLevel::st_newCallBack_m_indiP_req_halt();
    zaberLowLevel::st_newCallBack_m_indiP_req_ehalt(); zaberLowLevel::st_newCallBack_m_indiP_knob_enable();
    zaberLowLevel::st_newCallBack_m_indiP_led_enable(); zaberStage<zaberLowLevel>::sendCommand();
    #endif
    // clang-format on
    for( unsigned operation = 0; operation < 7; ++operation )
    {
        for( bool turningOff : { false, true } )
        {
            outletHarness::g_faults = {};
            g_script = {};
            PowerFixture app;
            g_script.m_failure = Operation::Send;
            if( turningOff )
                g_script.m_beforeFailure = [&] { app.targetOff(); };
            CHECK( app.command( operation ) == ( operation == 4 ? 0 : -1 ) );
            CHECK( g_script.m_commands.size() == 1 );
            CHECK( outletHarness::g_faults.m_logs.empty() == turningOff );
        }
    }
}

/// Cleanup after a power cut does not emit a new disconnect error; powered-on failures remain visible.
/** \ingroup zaberLowLevel_unit_test */
TEST_CASE( "Zaber cleanup suppresses expected disconnect errors", "[zaberLowLevel][power]" )
{
    // clang-format off
    #ifdef ZABERLOWLEVEL_TEST_DOXYGEN_REF
    zaberLowLevel::resetConnection(); zaberLowLevel::onPowerOff();
    #endif
    // clang-format on
    for( bool turningOff : { false, true } )
    {
        outletHarness::g_faults = {};
        g_script = {};
        PowerFixture app;
        g_script.m_failure = Operation::Disconnect;
        if( turningOff )
            app.targetOff();
        REQUIRE( app.resetConnection() == 0 );
        CHECK( app.m_port == 0 );
        CHECK( outletHarness::g_faults.m_logs.empty() == turningOff );
    }
}

/// An unknown initial target still permits initial connection, avoiding a startup wait for a first power command.
/** \ingroup zaberLowLevel_unit_test */
TEST_CASE( "Zaber permits initial communication with an unknown target", "[zaberLowLevel][power]" )
{
    // clang-format off
    #ifdef ZABERLOWLEVEL_TEST_DOXYGEN_REF
    zaberLowLevel::powerOffRequested(); zaberLowLevel::connect();
    #endif
    // clang-format on
    outletHarness::g_faults = {};
    g_script = {};
    PowerFixture app;
    app.power( "On", "Unk" );
    app.m_port = 0;
    g_script.m_replies.push_back( "@01 0 OK IDLE -- 49820\n" );
    CHECK( app.connect() == ZC_CONNECTED );
    CHECK( g_script.m_calls[static_cast<size_t>( Operation::Connect )] == 1 );
    CHECK( g_script.m_commands.size() == 2 );
}

} // namespace zaberLowLevelTest
} // namespace libXWCTest
