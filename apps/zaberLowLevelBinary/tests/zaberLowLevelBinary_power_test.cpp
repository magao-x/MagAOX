/** \file zaberLowLevelBinary_power_test.cpp
 * \brief Offline power-target and communication-error regressions for binary Zaber control.
 * \ingroup zaberLowLevelBinary_files
 */
#include "../../../tests/testXWC.hpp"
#include "../../../tests/outletAppTest.hpp"

extern "C"
{
#include "../zb_serial.c"
}

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
namespace zaberBinaryPowerHarness
{
/// Transport phase selected for a one-shot failure.
enum class Operation
{
    None,
    Connect,
    Disconnect,
    Timeout,
    Drain,
    Send,
    Receive
};

/// Sequential fixture's hardware-free transport and target-only injection.
struct Script
{
    /// Operation to fail, or None for normal replies.
    Operation m_failure{ Operation::None };

    /// Invocation of the selected operation to fail.
    unsigned m_failureCall{ 1 };

    /// Counts of all scripted transport calls.
    std::array<unsigned, 7> m_calls{};

    /// Actual binary commands sent by the production app and helper.
    std::vector<std::array<uint8_t, 6>> m_commands;

    /// Device mode retained by the fake device.
    int32_t m_mode{ 128 };

    /// Stored target speed retained by the fake device.
    int32_t m_speed{ 1000 };

    /// Stored hold current retained by the fake device.
    int32_t m_holdCurrent{ 0 };

    /// Corrupt a selected reply instead of returning a transport error.
    bool m_corruptReply{ false };

    /// Real power callback invoked before the selected failure returns.
    std::function<void()> m_beforeFailure;

    /// Whether the one-shot callback has run.
    bool m_injected{ false };
};

/// Current sequential fixture's script; no function opens a device.
inline Script g_script;

/// Count an operation and inject the one-shot power change if selected.
bool fail( Operation operation /**< [in] Transport phase being attempted. */ );

/// Return a scripted marker without opening a serial port.
int connect( z_port *port, /**< [out] Scripted port marker. */
             const char *name /**< [in] Ignored device path. */ );

/// Simulate closing the marker without closing a real descriptor.
int disconnect( z_port port /**< [in] Ignored port marker. */ );

/// Simulate configuring a timeout without accessing a descriptor.
int timeout( z_port port, /**< [in] Ignored port marker. */
             int milliseconds /**< [in] Ignored timeout. */ );

/// Simulate draining the marker without accessing a descriptor.
int drain( z_port port /**< [in] Ignored port marker. */ );

/// Record a real six-byte command and simulate acceptance or failure.
int send( z_port port, /**< [in] Ignored port marker. */
          const uint8_t *command /**< [in] Six-byte binary command. */ );

/// Encode a valid reply or return a scripted failure without reading a device.
int receive( z_port port, /**< [in] Ignored port marker. */
             uint8_t *reply /**< [out] Six-byte reply. */ );

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

int timeout( z_port, int )
{
    return fail( Operation::Timeout ) ? Z_ERROR_SYSTEM_ERROR : Z_SUCCESS;
}

int drain( z_port )
{
    return fail( Operation::Drain ) ? Z_ERROR_SYSTEM_ERROR : Z_SUCCESS;
}

int send( z_port, const uint8_t *command )
{
    std::array<uint8_t, 6> bytes;
    std::copy_n( command, bytes.size(), bytes.begin() );
    g_script.m_commands.push_back( bytes );
    if( fail( Operation::Send ) )
        return Z_ERROR_SYSTEM_ERROR;
    int32_t data;
    REQUIRE( zb_decode( &data, command ) == Z_SUCCESS );
    if( command[1] == 40 )
        g_script.m_mode = data;
    if( command[1] == 42 )
        g_script.m_speed = data;
    if( command[1] == 39 )
        g_script.m_holdCurrent = data;
    return 6;
}

int receive( z_port, uint8_t *reply )
{
    bool failure = fail( Operation::Receive );
    if( failure && !g_script.m_corruptReply )
        return Z_ERROR_SYSTEM_ERROR;
    REQUIRE( !g_script.m_commands.empty() );
    auto command = g_script.m_commands.back();
    int32_t data;
    REQUIRE( zb_decode( &data, command.data() ) == Z_SUCCESS );
    uint8_t replyCommand = command[1];
    int32_t response = 0;
    if( command[1] == 63 )
        response = 49820;
    else if( command[1] == 60 || command[1] == 17 )
        response = 12345;
    else if( command[1] == 53 )
    {
        replyCommand = data;
        if( data == 40 )
            response = g_script.m_mode;
        else if( data == 42 )
            response = g_script.m_speed;
        else if( data == 39 )
            response = g_script.m_holdCurrent;
        else if( data == 44 )
            response = 54321;
    }
    REQUIRE( zb_encode( reply, failure ? 2 : command[0], replyCommand, response ) == Z_SUCCESS );
    return 6;
}
} // namespace zaberBinaryPowerHarness

#define zb_connect zaberBinaryPowerHarness::connect
#define zb_disconnect zaberBinaryPowerHarness::disconnect
#define zb_set_timeout zaberBinaryPowerHarness::timeout
#define zb_drain zaberBinaryPowerHarness::drain
#define zb_send zaberBinaryPowerHarness::send
#define zb_receive zaberBinaryPowerHarness::receive
#define MagAOXApp outletTestApp
#include "../zaberLowLevelBinary.hpp"
#undef MagAOXApp
#undef zb_receive
#undef zb_send
#undef zb_drain
#undef zb_set_timeout
#undef zb_disconnect
#undef zb_connect
/// \endcond

namespace libXWCTest
{
/** \addtogroup zaberLowLevelBinary_unit_test
 * \ingroup application_unit_test
 */
namespace zaberLowLevelBinaryTest
{
using namespace MagAOX::app;
using namespace outletHarness;
using namespace zaberBinaryPowerHarness;

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
/// Production controller and stage methods with captured logs and fake binary transport.
struct PowerFixture : Controller<zaberLowLevelBinary>
{
    /// Expose the scripted marker for connection and cleanup checks.
    using zaberLowLevelBinary::m_port;

    /// Expose the discovery limit so shutdown cancellation is tested across multiple addresses.
    using zaberLowLevelBinary::m_maxDiscoveryAddress;

    /// Initialize one retained stage and its actual registered callbacks.
    PowerFixture();

    /// Deliver observed and target states through the production power callback.
    void power( const std::string &observed, /**< [in] Observed state. */
                const std::string &target /**< [in] Requested target. */ );

    /// Deliver target Off while retaining observed On.
    void targetOff();

    /// Dispatch a stage command through its actual registered callback.
    int command( unsigned operation /**< [in] Move, home, home-all, halt, emergency halt, or knob index. */ );

    /// Invoke the real stage helper's query or no-reply method.
    int stageCommand( bool query /**< [in] Whether to wait for a binary reply. */ );

    /// Return the last known address without accessing hardware.
    int stageAddress();
};

PowerFixture::PowerFixture()
{
    m_sysPath = m_directory.m_path + "/sys";
    std::filesystem::create_directories( m_sysPath + "/" + m_configName );
    m_stages.emplace_back( this );
    m_stages.back().name( "stageA" );
    m_stages.back().serial( "49820" );
    m_stages.back().deviceAddress( 1 );
    m_stageName.emplace( "stageA", 0 );
    m_stageSerial.emplace( "49820", 0 );
    std::ofstream saved( m_sysPath + "/" + m_configName + "/stageA" );
    saved << "12345\n1\n54321\n77\n";
    saved.close();
    m_maxDiscoveryAddress = 1;
    m_renumberPauseMs = 0;
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
    const std::array<pcf::IndiProperty *, 6> properties{ &m_indiP_tgt_pos, &m_indiP_req_home,
        &m_indiP_req_home_all, &m_indiP_req_halt, &m_indiP_req_ehalt, &m_indiP_knob_enable };
    auto request = *properties.at( operation );
    if( operation == 0 )
        request["stageA"].set( 200 );
    else
        request[operation == 2 ? "request" : "stageA"].setSwitchState( pcf::IndiElement::On );
    auto callback = m_indiNewCallBacks.at( request.createUniqueKey() ).callBack;
    REQUIRE( callback != nullptr );
    return callback( this, request );
}

int PowerFixture::stageCommand( bool query )
{
    int32_t response;
    return query ? m_stages[0].queryCommand( response, m_port, 54, 0, 54 )
                 : m_stages[0].sendCommandNoReply( m_port, 23, 0 );
}

int PowerFixture::stageAddress()
{
    return m_stages[0].deviceAddress();
}
/// \endcond

/// A target-only Off update prevents new serial work before observed power changes.
/** \ingroup zaberLowLevelBinary_unit_test */
TEST_CASE( "Binary Zaber waits for power-off without starting communication", "[zaberLowLevelBinary][power]" )
{
    // clang-format off
    #ifdef ZABERLOWLEVELBINARY_TEST_DOXYGEN_REF
    zaberLowLevelBinary::powerOffRequested(); zaberLowLevelBinary::powerOnExpected();
    zaberLowLevelBinary::appLogic(); zaberLowLevelBinary::connect(); zaberLowLevelBinary::loadStages();
    zaberLowLevelBinary::refreshStageDiscovery(); zaberLowLevelBinary::queryDevice();
    zaberLowLevelBinary::sendCommandNoReply();
    MagAOX::app::MagAOXApp<true>::setCallBack_m_indiP_powerChannel();
    #endif
    // clang-format on
    outletHarness::g_faults = {};
    g_script = {};
    PowerFixture app;
    app.targetOff();
    for( auto code : { stateCodes::POWERON, stateCodes::NODEVICE, stateCodes::NOTCONNECTED,
                      stateCodes::CONNECTED, stateCodes::READY, stateCodes::ERROR } )
    {
        app.state( code );
        CHECK( app.appLogic() == 0 );
        CHECK( app.state() == code );
    }
    CHECK( app.connect() == ZBC_NOT_CONNECTED );
    CHECK( app.loadStages() == ZBC_ERROR );
    CHECK( app.loadStages( { 1 }, { "49820" } ) == ZBC_ERROR );
    CHECK( app.refreshStageDiscovery() == ZBC_ERROR );
    int32_t response;
    CHECK( app.queryDevice( response, 1, 63, 0, 63 ) < 0 );
    CHECK( app.sendCommandNoReply( 0, 2, 0 ) < 0 );
    CHECK( app.stageCommand( false ) < 0 );
    CHECK( app.stageCommand( true ) < 0 );
    for( unsigned operation = 0; operation < 6; ++operation )
        CHECK( app.command( operation ) < 0 );
    CHECK( g_script.m_calls == std::array<unsigned, 7>{} );
    CHECK( app.m_port == 99 );
    CHECK( app.stageAddress() == 1 );
    CHECK( outletHarness::g_faults.m_logs.empty() );
    app.power( "On", "On" );
    app.state( stateCodes::CONNECTED );
    CHECK( app.appLogic() == 0 );
    CHECK( app.state() == stateCodes::READY );
}

/// Each failing connection phase rechecks the target before logging or entering ERROR.
/** \ingroup zaberLowLevelBinary_unit_test */
TEST_CASE( "Binary Zaber connection suppresses expected power-loss failures", "[zaberLowLevelBinary][power]" )
{
    // clang-format off
    #ifdef ZABERLOWLEVELBINARY_TEST_DOXYGEN_REF
    zaberLowLevelBinary::connect(); zaberLowLevelBinary::loadStages();
    #endif
    // clang-format on
    for( auto [failure, call] : { std::pair{ Operation::Connect, 1u }, { Operation::Timeout, 1u },
                                 { Operation::Drain, 1u }, { Operation::Drain, 3u }, { Operation::Send, 1u } } )
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
            CHECK( app.connect() == ( failure == Operation::Connect ? ZBC_NOT_CONNECTED : ZBC_ERROR ) );
            CHECK( app.state() == ( turningOff || failure == Operation::Connect ? stateCodes::NOTCONNECTED : stateCodes::ERROR ) );
            CHECK( outletHarness::g_faults.m_logs.empty() == turningOff );
        }
    }
}

/// Discovery cancels on Off without replacing the last known stage mapping or reporting a missing device.
/** \ingroup zaberLowLevelBinary_unit_test */
TEST_CASE( "Binary Zaber discovery stops during a power-target change", "[zaberLowLevelBinary][power]" )
{
    // clang-format off
    #ifdef ZABERLOWLEVELBINARY_TEST_DOXYGEN_REF
    zaberLowLevelBinary::refreshStageDiscovery(); zaberLowLevelBinary::queryDevice();
    #endif
    // clang-format on
    for( auto failure : { Operation::Drain, Operation::Send, Operation::Receive } )
    {
        outletHarness::g_faults = {};
        g_script = {};
        PowerFixture app;
        app.m_maxDiscoveryAddress = 4;
        g_script.m_failure = failure;
        g_script.m_beforeFailure = [&] { app.targetOff(); };
        CHECK( app.appLogic() == 0 );
        CHECK( app.state() == stateCodes::READY );
        CHECK( app.stageAddress() == 1 );
        CHECK( outletHarness::g_faults.m_logs.empty() );
        CHECK( g_script.m_calls[static_cast<size_t>( Operation::Send )] <= 1 );
    }
}

/// Both binary helper paths suppress send/read/protocol errors during shutdown and retain On/On diagnostics.
/** \ingroup zaberLowLevelBinary_unit_test */
TEST_CASE( "Binary Zaber stage commands preserve power-loss suppression", "[zaberLowLevelBinary][power]" )
{
    // clang-format off
    #ifdef ZABERLOWLEVELBINARY_TEST_DOXYGEN_REF
    zaberBinaryStage<zaberLowLevelBinary>::powerOffRequested(); zaberBinaryStage<zaberLowLevelBinary>::powerOnExpected();
    zaberBinaryStage<zaberLowLevelBinary>::queryCommand(); zaberBinaryStage<zaberLowLevelBinary>::sendCommandNoReply();
    #endif
    // clang-format on
    for( auto [query, failure, corrupt] : { std::tuple{ false, Operation::Send, false },
                                           { true, Operation::Send, false }, { true, Operation::Receive, false },
                                           { true, Operation::Receive, true } } )
    {
        for( bool turningOff : { false, true } )
        {
            outletHarness::g_faults = {};
            g_script = {};
            PowerFixture app;
            g_script.m_failure = failure;
            g_script.m_corruptReply = corrupt;
            if( turningOff )
                g_script.m_beforeFailure = [&] { app.targetOff(); };
            CHECK( app.stageCommand( query ) < 0 );
            CHECK( outletHarness::g_faults.m_logs.empty() == turningOff );
        }
    }
}

/// Registered parent callbacks retain stage-level suppression when shutdown begins during a command.
/** \ingroup zaberLowLevelBinary_unit_test */
TEST_CASE( "Binary Zaber registered callbacks suppress shutdown errors", "[zaberLowLevelBinary][power]" )
{
    // clang-format off
    #ifdef ZABERLOWLEVELBINARY_TEST_DOXYGEN_REF
    zaberLowLevelBinary::newCallBack_m_indiP_tgt_pos(); zaberLowLevelBinary::newCallBack_m_indiP_req_home();
    zaberLowLevelBinary::newCallBack_m_indiP_req_home_all(); zaberLowLevelBinary::newCallBack_m_indiP_req_halt();
    zaberLowLevelBinary::newCallBack_m_indiP_req_ehalt(); zaberLowLevelBinary::newCallBack_m_indiP_knob_enable();
    zaberLowLevelBinary::st_newCallBack_m_indiP_tgt_pos(); zaberLowLevelBinary::st_newCallBack_m_indiP_req_home();
    zaberLowLevelBinary::st_newCallBack_m_indiP_req_home_all(); zaberLowLevelBinary::st_newCallBack_m_indiP_req_halt();
    zaberLowLevelBinary::st_newCallBack_m_indiP_req_ehalt(); zaberLowLevelBinary::st_newCallBack_m_indiP_knob_enable();
    #endif
    // clang-format on
    for( unsigned operation = 0; operation < 6; ++operation )
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

/// Polling and connection initialization preserve their FSM when power changes during any serial phase.
/** \ingroup zaberLowLevelBinary_unit_test */
TEST_CASE( "Binary Zaber polling suppresses failures at every serial phase", "[zaberLowLevelBinary][power]" )
{
    // clang-format off
    #ifdef ZABERLOWLEVELBINARY_TEST_DOXYGEN_REF
    zaberLowLevelBinary::appLogic();
    zaberBinaryStage<zaberLowLevelBinary>::enableKnob(); zaberBinaryStage<zaberLowLevelBinary>::getMaxPos();
    zaberBinaryStage<zaberLowLevelBinary>::setTargetSpeed(); zaberBinaryStage<zaberLowLevelBinary>::updatePos();
    zaberBinaryStage<zaberLowLevelBinary>::recallParkPosition(); zaberBinaryStage<zaberLowLevelBinary>::restoreParkedState();
    zaberBinaryStage<zaberLowLevelBinary>::getWarnings(); zaberBinaryStage<zaberLowLevelBinary>::getKnob();
    #endif
    // clang-format on
    for( auto code : { stateCodes::CONNECTED, stateCodes::READY } )
    {
        std::array<unsigned, 7> counts;
        {
            outletHarness::g_faults = {};
            g_script = {};
            PowerFixture app;
            app.state( code );
            REQUIRE( app.appLogic() == 0 );
            REQUIRE( app.state() == stateCodes::READY );
            counts = g_script.m_calls;
        }
        for( auto failure : { Operation::Send, Operation::Receive } )
        {
            for( unsigned call = 1; call <= counts[static_cast<size_t>( failure )]; ++call )
            {
                outletHarness::g_faults = {};
                g_script = {};
                PowerFixture app;
                app.state( code );
                g_script.m_failure = failure;
                g_script.m_failureCall = call;
                g_script.m_beforeFailure = [&] { app.targetOff(); };
                CHECK( app.appLogic() == 0 );
                CHECK( app.state() == code );
                CHECK( std::none_of( outletHarness::g_faults.m_logs.begin(), outletHarness::g_faults.m_logs.end(),
                                    []( const Log &entry ) { return entry.m_priority <= logPrio::LOG_ERROR; } ) );
                CHECK( g_script.m_calls[static_cast<size_t>( failure )] == call );
            }
        }
    }
}

/// Unexpected polling failures while power remains On still enter the recoverable ERROR state.
/** \ingroup zaberLowLevelBinary_unit_test */
TEST_CASE( "Binary Zaber reports unexpected powered-on polling failures", "[zaberLowLevelBinary][power]" )
{
    // clang-format off
    #ifdef ZABERLOWLEVELBINARY_TEST_DOXYGEN_REF
    zaberLowLevelBinary::appLogic(); zaberBinaryStage<zaberLowLevelBinary>::queryCommand();
    #endif
    // clang-format on
    for( auto code : { stateCodes::CONNECTED, stateCodes::READY } )
    {
        for( auto failure : { Operation::Send, Operation::Receive } )
        {
            outletHarness::g_faults = {};
            g_script = {};
            PowerFixture app;
            app.state( code );
            g_script.m_failure = failure;
            g_script.m_failureCall = code == stateCodes::READY ? 2 : 1;
            CHECK( app.appLogic() == 0 );
            CHECK( app.state() == stateCodes::ERROR );
            CHECK( std::any_of( outletHarness::g_faults.m_logs.begin(), outletHarness::g_faults.m_logs.end(),
                               []( const Log &entry ) { return entry.m_priority <= logPrio::LOG_ERROR; } ) );
        }
    }
}

/// Observed-off cleanup closes bookkeeping and publishes the retained snapshot; On/On disconnect failures remain visible.
/** \ingroup zaberLowLevelBinary_unit_test */
TEST_CASE( "Binary Zaber power-off cleanup retains the snapshot", "[zaberLowLevelBinary][power]" )
{
    // clang-format off
    #ifdef ZABERLOWLEVELBINARY_TEST_DOXYGEN_REF
    zaberLowLevelBinary::resetConnection(); zaberLowLevelBinary::onPowerOff();
    zaberBinaryStage<zaberLowLevelBinary>::onPowerOff();
    #endif
    // clang-format on
    for( bool turningOff : { false, true } )
    {
        outletHarness::g_faults = {};
        g_script = {};
        PowerFixture app;
        g_script.m_failure = Operation::Disconnect;
        if( turningOff )
        {
            app.power( "Off", "Off" );
            REQUIRE( app.onPowerOff() == 0 );
            auto snapshot = app.m_indiNewCallBacks.at( "test-pdu.curr_pos" ).property;
            CHECK( (*snapshot)["stageA"].get<long>() == 12345 );
        }
        else
            REQUIRE( app.resetConnection() == 0 );
        CHECK( app.m_port == 0 );
        CHECK( outletHarness::g_faults.m_logs.empty() == turningOff );
    }
}

/// An unknown initial target permits initial connection before the first operator request.
/** \ingroup zaberLowLevelBinary_unit_test */
TEST_CASE( "Binary Zaber permits startup with an unknown target", "[zaberLowLevelBinary][power]" )
{
    // clang-format off
    #ifdef ZABERLOWLEVELBINARY_TEST_DOXYGEN_REF
    zaberLowLevelBinary::connect();
    #endif
    // clang-format on
    outletHarness::g_faults = {};
    g_script = {};
    PowerFixture app;
    app.power( "On", "Unk" );
    app.m_port = 0;
    CHECK( app.connect() == ZBC_CONNECTED );
    CHECK( g_script.m_calls[static_cast<size_t>( Operation::Connect )] == 1 );
}

} // namespace zaberLowLevelBinaryTest
} // namespace libXWCTest
