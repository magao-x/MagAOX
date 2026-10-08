/** \file virtualPDU_test.cpp
 * \brief Offline behavioral, traffic, and failure-contract tests for virtual PDU control.
 * \ingroup virtualPDU_files
 */
#include "../../../tests/testXWC.hpp"
#include "../../../tests/outletAppTest.hpp"

#define MagAOXApp outletTestApp
#define telemeter outletTestTelemeter
#include "../virtualPDU.hpp"
#undef telemeter
#undef MagAOXApp

namespace libXWCTest
{
/** \defgroup virtualPDU_unit_test virtualPDU Unit Tests
 * \ingroup application_unit_test
 */
namespace virtualPDUTest
{
using namespace MagAOX::app;
using namespace outletHarness;

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
/// Expose source observations and inject the inherited update failure.
struct Fixture : Controller<virtualPDU>
{
    /// Expose source state for deterministic expiry tests.
    using virtualPDU::m_endpoints;
    /// Expose shared FSM observations.
    using virtualPDU::m_sources;
    /// Expose polling deadline.
    using virtualPDU::m_lastPoll;
    /// Expose refresh configuration.
    using virtualPDU::m_pollInterval;
    /// Expose stale limit configuration.
    using virtualPDU::m_staleTimeout;
    /// Fail the inherited all-outlet update only when selected.
    bool m_failUpdate{ false };
    /// Exercise normal updates unless their return needs fault injection.
    int updateOutletStates() override;
    /// Load the normal three-endpoint/two-channel example.
    void configure();
    /// Inject valid channel observations and READY FSM reports.
    void observed();
};
int Fixture::updateOutletStates()
{
    if( m_failUpdate )
        return -1;
    return dev::outletController<virtualPDU>::updateOutletStates();
}
void Fixture::configure()
{
    configText( "[outlet1]\ndevice=ac\nchannel=power\n[outlet2]\ndevice=usb\nchannel=power\n"
                "[outlet3]\ndevice=ac\nchannel=aux\n[combined]\noutlets=1,2\nonOrder=0,1\noffOrder=1,0\n"
                "onDelays=0,3\noffDelays=0,3\n[independent]\noutlet=3\n" );
    loadConfig();
    REQUIRE( !m_shutdown );
    REQUIRE( appStartup() == 0 );
}
void Fixture::observed()
{
    REQUIRE( setCallBack_source( property( "ac", "fsm", "state", "READY" ) ) == 0 );
    REQUIRE( setCallBack_source( property( "usb", "fsm", "state", "READY" ) ) == 0 );
    REQUIRE( setCallBack_source( property( "ac", "power", "state", "Off" ) ) == 0 );
    REQUIRE( setCallBack_source( property( "usb", "power", "state", "Off" ) ) == 0 );
    REQUIRE( setCallBack_source( property( "ac", "aux", "state", "Off" ) ) == 0 );
}
/// \endcond

/// Verify one-based configuration and stable unique subscriptions.
/** \ingroup virtualPDU_unit_test */
TEST_CASE( "virtual PDU maps remote channels and publishes the standard interface", "[virtualPDU]" )
{
    // clang-format off
    #ifdef VIRTUALPDU_TEST_DOXYGEN_REF
    virtualPDU::virtualPDU(); virtualPDU::~virtualPDU(); virtualPDU::setupConfig();
    virtualPDU::loadConfigImpl(); virtualPDU::appStartup();
    #endif
    // clang-format on
    g_faults = {};
    Fixture app;
    app.configure();
    REQUIRE( app.state() == stateCodes::READY );
    REQUIRE( app.m_endpoints.size() == 3 );
    REQUIRE( app.m_sources.size() == 2 );
    REQUIRE( app.channelOutlets( "combined" ) == std::vector<size_t>{ 0, 1 } );
    REQUIRE( app.channelOffOrder( "combined" ) == std::vector<size_t>{ 1, 0 } );
    REQUIRE( app.m_indiNewCallBacks.size() == 7 );
    REQUIRE( app.m_indiSetCallBacks.size() == 5 );
    REQUIRE( app.m_indiP_chOnDelays["combined"].get<int>() == 3 );
    REQUIRE( app.m_indiP_chOutlets["combined"].get<std::string>() == "0,1" );
    for( auto &[key, registration] : app.m_indiSetCallBacks )
        REQUIRE( registration.property->createUniqueKey() == key );
}

/// Reject malformed, cyclic, aliased, shared, or out-of-range virtual configurations.
/** \ingroup virtualPDU_unit_test */
TEST_CASE( "virtual PDU rejects invalid mappings and channel sequences", "[virtualPDU]" )
{
    // clang-format off
    #ifdef VIRTUALPDU_TEST_DOXYGEN_REF
    virtualPDU::loadConfigImpl(); virtualPDU::loadConfig();
    #endif
    // clang-format on
    const std::string endpoint = "[outlet1]\ndevice=ac\nchannel=power\n";
    const std::string two = endpoint + "[outlet2]\ndevice=usb\nchannel=power\n";
    for( const auto &[text, details] :
         std::vector<std::pair<std::string, std::vector<std::string>>>{
             { "", { "No [outletN]", "[outlet1]", "required" } },
             { "[device]\npollInterval=0\n", { "[device]", "pollInterval=0", "greater than 0" } },
             { "[device]\npollInterval=-1\n", { "pollInterval=-1", "greater than 0" } },
             { "[device]\nstaleTimeout=3\n", { "staleTimeout=3", "exceed pollInterval=5" } },
             { "[device]\nstaleTimeout=5\n", { "staleTimeout=5", "exceed pollInterval=5" } },
             { "[outlet0]\ndevice=ac\nchannel=x\n", { "[outlet0]", "positive integer" } },
             { "[outletx]\ndevice=ac\nchannel=x\n", { "[outletx]", "positive integer" } },
             { "[outlet01]\ndevice=ac\nchannel=x\n", { "[outlet01]", "leading zeros" } },
             { "[outlet2]\ndevice=ac\nchannel=x\n", { "[outlet1]", "missing", "consecutive" } },
             { endpoint + "[outlet3]\ndevice=usb\nchannel=x\n", { "[outlet2]", "missing", "consecutive" } },
             { "[outlet1]\nchannel=x\n", { "[outlet1]", "device", "required" } },
             { "[outlet1]\ndevice=ac\n", { "[outlet1]", "channel", "required" } },
             { "[outlet1]\ndevice=test-pdu\nchannel=x\n", { "[outlet1]", "test-pdu", "this virtual PDU" } },
             { "[outlet1]\ndevice=ac\nchannel=fsm\n", { "[outlet1]", "fsm", "reserved" } },
             { endpoint + "[outlet2]\ndevice=ac\nchannel=power\n[x]\noutlets=1,2\n",
               { "[outlet2]", "[outlet1]", "ac.power", "duplicates" } },
             { endpoint + "[x]\noutlet=1\n[y]\noutlet=1\n", { "[x]", "[y]", "outlet 1", "shared" } },
             { two + "[camera]\noutlets=1,2\n[lamp]\noutlets=1,2\n",
               { "[camera]", "[lamp]", "outlet 1", "shared" } },
             { endpoint + "[fsm]\noutlet=1\n", { "[fsm]", "reserved INDI property" } },
             { endpoint + "[x]\noutlets=0\n", { "[x]", "outlets", "outlet 0", "range 1..1" } },
             { endpoint + "[x]\noutlets=2\n", { "[x]", "outlets", "outlet 2", "range 1..1" } },
             { endpoint + "[x]\noutlets=1,1\n", { "[x]", "outlet 1", "repeated" } },
             { endpoint + "[x]\noutlet=1\nonOrder=1\n", { "[x]", "onOrder", "permutation", "0..0" } },
             { endpoint + "[x]\noutlet=1\noffOrder=1\n", { "[x]", "offOrder", "permutation", "0..0" } },
             { endpoint + "[x]\noutlet=1\nonDelays=0,2\n", { "[x]", "onDelays", "2 entries", "expected 1" } },
             { endpoint + "[x]\noutlet=1\noffDelays=0,2\n", { "[x]", "offDelays", "2 entries", "expected 1" } },
             { endpoint + "[x]\noutlet=1\nonOrder=0,0\n", { "[x]", "onOrder", "2 entries", "expected 1" } },
             { endpoint + "[x]\noutlet=1\noffOrder=0,0\n", { "[x]", "offOrder", "2 entries", "expected 1" } },
             { endpoint + "[x]\noutlet=1\nonOrder=bad\n", { "[x]", "onOrder", "'bad'", "nonnegative" } },
             { endpoint + "[x]\noutlet=1\nonDelays=-2\n", { "[x]", "onDelays", "'-2'", "nonnegative" } },
             { endpoint + "[x]\noutlet=1\nonOrder= \n", { "[x]", "onOrder" } },
             { endpoint + "[x]\noutlet=1\nonOrder=0,,1\n", { "[x]", "onOrder", "empty value" } },
             { endpoint + "[x]\noutlet=1\nonDelays=0, ,1\n", { "[x]", "onDelays", "empty value" } },
             { endpoint + "[telem_rotate]\noutlet=1\n", { "[telem_rotate]", "reserved INDI property" } },
             { endpoint + "[x]\noutlet=2147483648\n", { "[x]", "outlet 2147483648", "range 1..1" } },
             { endpoint + "[x]\noutlet=1\nonDelays=4294967296\n",
               { "[x]", "onDelays", "4294967296", "maximum 4294967295", "milliseconds" } },
             { endpoint + "[x]\noutlet=1\noffDelays=4294967296\n",
               { "[x]", "offDelays", "4294967296", "maximum 4294967295", "milliseconds" } },
             { two + "[x]\noutlets=1,2\nonOrder=0,0\n", { "[x]", "onOrder", "permutation", "0..1" } },
             { two + "[x]\noutlets=1,2\noffOrder=0,0\n", { "[x]", "offOrder", "permutation", "0..1" } },
             { endpoint, { "outlet channel", "outlet=", "outlets=" } },
             { endpoint + "[unused]\nvalue=1\n", { "outlet channel", "outlet=", "outlets=" } },
             { endpoint + "[x]\noutlets=\n", { "[x]", "no outlets", "at least one" } } } )
    {
        g_faults = {};
        Fixture app;
        app.configText( text );
        INFO( text );
        REQUIRE( app.loadConfigImpl( app.config ) < 0 );
        REQUIRE( !g_faults.m_logs.empty() );
        std::string diagnostic;
        for( const auto &entry : g_faults.m_logs )
            diagnostic += entry.m_message + '\n';
        INFO( diagnostic );
        for( const auto &detail : details )
            CHECK( diagnostic.find( detail ) != std::string::npos );
    }
}

/// Report the copied channel names and the one-based outlet causing startup rejection.
/** \ingroup virtualPDU_unit_test */
TEST_CASE( "virtual PDU explains shared outlet failures during configuration loading", "[virtualPDU]" )
{
    // clang-format off
    #ifdef VIRTUALPDU_TEST_DOXYGEN_REF
    virtualPDU::loadConfig(); virtualPDU::loadConfigImpl(); virtualPDU::configError();
    #endif
    // clang-format on
    g_faults = {};
    Fixture app;
    app.configText( "[outlet1]\ndevice=ac\nchannel=power\n[outlet2]\ndevice=usb\nchannel=power\n"
                    "[camera]\noutlets=1,2\n[lamp]\noutlets=1,2\n" );
    app.loadConfig();
    REQUIRE( app.m_shutdown );
    REQUIRE( g_faults.m_logs.size() == 1 );
    CHECK( g_faults.m_logs[0].m_priority == flatlogs::logPrio::LOG_CRITICAL );
    CHECK( g_faults.m_logs[0].m_message ==
           "Invalid virtual PDU configuration: Virtual outlet 1 is shared by channels [camera] and [lamp]; "
           "sharing outlets is not supported" );
}

/// Exercise registration and telemetry configuration/startup/shutdown failures.
/** \ingroup virtualPDU_unit_test */
TEST_CASE( "virtual PDU propagates lifecycle failures", "[virtualPDU]" )
{
    // clang-format off
    #ifdef VIRTUALPDU_TEST_DOXYGEN_REF
    virtualPDU::setupConfig(); virtualPDU::loadConfigImpl(); virtualPDU::appStartup(); virtualPDU::appShutdown();
    #endif
    // clang-format on
    g_faults = {};
    {
        Fixture app;
        app.configText( "[outlet1]\ndevice=ac\nchannel=power\n[x]\noutlet=1\n" );
        REQUIRE( app.loadConfigImpl( app.config ) == 0 );
        REQUIRE( app.updateOutletState( 0 ) == 0 );
        REQUIRE( app.outletState( 0 ) == OUTLET_STATE_UNKNOWN );
        REQUIRE( app.turnOutletOn( 0 ) < 0 );
        REQUIRE( app.newCallBack_channels( property( "test-pdu", "x", "target", "On" ) ) < 0 );
    }
    for( unsigned failure = 1; failure <= 8; ++failure )
    {
        g_faults = {};
        Fixture app;
        app.configText( "[outlet1]\ndevice=ac\nchannel=power\n[x]\noutlet=1\n" );
        REQUIRE( app.loadConfigImpl( app.config ) == 0 );
        g_faults.m_failRegistration = failure;
        REQUIRE( app.appStartup() < 0 );
    }
    for( size_t call : { 0, 1, 2, 4 } )
    {
        g_faults = {};
        Fixture app;
        g_faults.m_telemResults[call] = -1;
        app.configText( "[outlet1]\ndevice=ac\nchannel=power\n[x]\noutlet=1\n" );
        if( call == 0 )
            REQUIRE( app.m_shutdown );
        else if( call == 1 )
            REQUIRE( app.loadConfigImpl( app.config ) < 0 );
        else
        {
            REQUIRE( app.loadConfigImpl( app.config ) == 0 );
            if( call == 2 )
                REQUIRE( app.appStartup() < 0 );
            else
                REQUIRE( app.appShutdown() == 0 );
        }
    }
}

/// Verify exact outgoing traffic, sequencing, and observed-state independence.
/** \ingroup virtualPDU_unit_test */
TEST_CASE( "virtual PDU dispatches correct ordered INDI targets", "[virtualPDU]" )
{
    // clang-format off
    #ifdef VIRTUALPDU_TEST_DOXYGEN_REF
    virtualPDU::newCallBack_channels(); virtualPDU::turnOutletOn(); virtualPDU::turnOutletOff(); virtualPDU::sendOutlet();
    #endif
    // clang-format on
    g_faults = {};
    Fixture app;
    app.configure();
    app.driver();
    app.observed();
    auto started = std::chrono::steady_clock::now();
    REQUIRE( app.newCallBack_channels( property( "test-pdu", "combined", "target", "on" ) ) == 0 );
    REQUIRE( std::chrono::steady_clock::now() - started >= std::chrono::milliseconds( 3 ) );
    std::vector<pcf::IndiProperty> commands;
    for( auto &message : app.messages() )
        if( message.getType() == pcf::IndiMessage::NewProperty )
            commands.push_back( message.getProperty() );
    REQUIRE( commands.size() == 2 );
    REQUIRE( commands[0].getDevice() == "ac" );
    REQUIRE( commands[1].getDevice() == "usb" );
    for( auto &command : commands )
    {
        REQUIRE( command.getName() == "power" );
        REQUIRE( command.getType() == pcf::IndiProperty::Text );
        REQUIRE( command.getNumElements() == 1 );
        REQUIRE( command["target"].get<std::string>() == "On" );
    }
    REQUIRE( app.channelState( "combined" ) == OUTLET_STATE_OFF );
    REQUIRE( app.setCallBack_source( property( "ac", "power", "state", "On" ) ) == 0 );
    REQUIRE( app.channelState( "combined" ) == OUTLET_STATE_INTERMEDIATE );
    REQUIRE( app.setCallBack_source( property( "usb", "power", "state", "On" ) ) == 0 );
    REQUIRE( app.channelState( "combined" ) == OUTLET_STATE_ON );
    REQUIRE( app.newCallBack_channels( property( "test-pdu", "combined", "state", "off" ) ) == 0 );
    commands.clear();
    for( auto &message : app.messages() )
        if( message.getType() == pcf::IndiMessage::NewProperty )
            commands.push_back( message.getProperty() );
    REQUIRE( commands.size() == 2 );
    REQUIRE( commands[0].getDevice() == "usb" );
    REQUIRE( commands[1].getDevice() == "ac" );
    REQUIRE( commands[0]["target"].get<std::string>() == "Off" );
    REQUIRE( app.channelState( "combined" ) == OUTLET_STATE_ON );
    REQUIRE( app.newCallBack_channels( property( "test-pdu", "combined", "target", "On" ) ) == 0 );
    REQUIRE( g_faults.m_sends == 4 ); // Already observed On: no further dispatch.
    REQUIRE( app.newCallBack_channels( property( "other", "combined", "target", "Off" ) ) < 0 );
    REQUIRE( app.newCallBack_channels( property( "test-pdu", "absent", "target", "Off" ) ) < 0 );
    app.state( stateCodes::NOTCONNECTED );
    REQUIRE( app.newCallBack_channels( property( "test-pdu", "combined", "target", "Off" ) ) < 0 );
}

/// Stop dispatch on send errors without rollback, and block only unavailable channels.
/** \ingroup virtualPDU_unit_test */
TEST_CASE( "virtual PDU handles partial dispatch and per-channel source loss", "[virtualPDU]" )
{
    // clang-format off
    #ifdef VIRTUALPDU_TEST_DOXYGEN_REF
    virtualPDU::available(); virtualPDU::sendOutlet(); virtualPDU::newCallBack_channels(); virtualPDU::updateOutletState();
    #endif
    // clang-format on
    for( unsigned failure : { 1, 2 } )
    {
        g_faults = {};
        Fixture app;
        app.configure();
        app.driver();
        app.observed();
        g_faults.m_failSend = failure;
        REQUIRE( app.newCallBack_channels( property( "test-pdu", "combined", "target", "On" ) ) < 0 );
        REQUIRE( g_faults.m_sends == failure );
        REQUIRE( app.channelState( "combined" ) == OUTLET_STATE_OFF );
    }
    g_faults = {};
    Fixture app;
    app.configure();
    app.driver();
    app.observed();
    REQUIRE( app.setCallBack_source( property( "usb", "fsm", "state", "NOTCONNECTED" ) ) == 0 );
    REQUIRE( app.channelState( "combined" ) == OUTLET_STATE_INTERMEDIATE );
    REQUIRE( app.newCallBack_channels( property( "test-pdu", "combined", "target", "On" ) ) < 0 );
    REQUIRE( g_faults.m_sends == 0 );
    REQUIRE( app.newCallBack_channels( property( "test-pdu", "independent", "target", "On" ) ) == 0 );
    REQUIRE( g_faults.m_sends == 1 );
    REQUIRE( app.state() == stateCodes::READY );
    REQUIRE( app.turnOutletOn( -1 ) < 0 );
    REQUIRE( app.turnOutletOff( 3 ) < 0 );
    REQUIRE( app.updateOutletState( -1 ) < 0 );
    REQUIRE( app.updateOutletState( 3 ) < 0 );
    REQUIRE( app.turnOutletOn( 1 ) < 0 );
}

/// Merge only valid observed state elements and recover after fresh FSM/channel reports.
/** \ingroup virtualPDU_unit_test */
TEST_CASE( "virtual PDU distinguishes observed state from targets and stale reports", "[virtualPDU]" )
{
    // clang-format off
    #ifdef VIRTUALPDU_TEST_DOXYGEN_REF
    virtualPDU::st_setCallBack_source(); virtualPDU::setCallBack_source(); virtualPDU::appLogic();
    #endif
    // clang-format on
    g_faults = {};
    Fixture app;
    app.configure();
    app.driver();
    app.observed();
    REQUIRE( virtualPDU::st_setCallBack_source( &app, property( "ac", "power", "target", "On" ) ) == 0 );
    REQUIRE( app.outletState( 0 ) == OUTLET_STATE_OFF );
    REQUIRE( app.setCallBack_source( property( "other", "power", "state", "On" ) ) < 0 );
    REQUIRE( app.setCallBack_source( property( "other", "fsm", "state", "READY" ) ) < 0 );
    REQUIRE( app.setCallBack_source( property( "ac", "fsm", "target", "READY" ) ) < 0 );
    pcf::IndiProperty wrong( pcf::IndiProperty::Number, "ac", "fsm" );
    wrong.add( pcf::IndiElement( "state", "READY" ) );
    REQUIRE( app.setCallBack_source( wrong ) < 0 );
    app.observed();
    for( auto value : { "Int", "Unk", "garbage" } )
    {
        REQUIRE( app.setCallBack_source( property( "ac", "power", "state", value ) ) == 0 );
        REQUIRE( app.outletState( 0 ) ==
                 ( std::string( value ) == "Int" ? OUTLET_STATE_INTERMEDIATE : OUTLET_STATE_UNKNOWN ) );
    }
    REQUIRE( app.setCallBack_source( property( "ac", "power", "state", "Unk" ) ) == 0 );
    REQUIRE( app.turnOutletOn( 0 ) == 0 );
    REQUIRE( app.outletState( 0 ) == OUTLET_STATE_UNKNOWN );
    REQUIRE( app.setCallBack_source( property( "ac", "power", "state", "garbage" ) ) == 0 );
    REQUIRE( app.turnOutletOn( 0 ) < 0 );
    wrong = pcf::IndiProperty( pcf::IndiProperty::Number, "ac", "power" );
    wrong.add( pcf::IndiElement( "state", "On" ) );
    REQUIRE( app.setCallBack_source( wrong ) == 0 );
    REQUIRE( app.outletState( 0 ) == OUTLET_STATE_UNKNOWN );
    wrong = pcf::IndiProperty( pcf::IndiProperty::Number, "ac", "power" );
    wrong.add( pcf::IndiElement( "target", 1 ) );
    REQUIRE( app.setCallBack_source( wrong ) == 0 );
    REQUIRE( app.outletState( 0 ) == OUTLET_STATE_UNKNOWN );
    app.observed();
    app.m_endpoints[0].m_received -= std::chrono::seconds( 20 );
    REQUIRE( app.appLogic() == 0 );
    REQUIRE( app.outletState( 0 ) == OUTLET_STATE_UNKNOWN );
    REQUIRE( app.outletState( 1 ) == OUTLET_STATE_OFF );
    REQUIRE( app.setCallBack_source( property( "ac", "power", "target", "Off" ) ) == 0 );
    REQUIRE( app.outletState( 0 ) == OUTLET_STATE_UNKNOWN );
    REQUIRE( app.setCallBack_source( property( "ac", "power", "state", "Off" ) ) == 0 );
    REQUIRE( app.outletState( 0 ) == OUTLET_STATE_OFF );
    app.m_sources[0].m_received -= std::chrono::seconds( 20 );
    REQUIRE( app.appLogic() == 0 );
    REQUIRE( app.outletState( 0 ) == OUTLET_STATE_UNKNOWN );
    REQUIRE( app.outletState( 2 ) == OUTLET_STATE_UNKNOWN );
    REQUIRE( app.setCallBack_source( property( "ac", "fsm", "state", "READY" ) ) == 0 );
    REQUIRE( app.appLogic() == 0 );
    REQUIRE( app.outletState( 0 ) == OUTLET_STATE_OFF );
    app.m_lastPoll -= std::chrono::seconds( 10 );
    REQUIRE( app.appLogic() == 0 );
    REQUIRE( !app.messages().empty() );
}

/// Record initial/change/forced observations and propagate scheduling/update failures.
/** \ingroup virtualPDU_unit_test */
TEST_CASE( "virtual PDU telemetry tracks observations independently of commands", "[virtualPDU]" )
{
    // clang-format off
    #ifdef VIRTUALPDU_TEST_DOXYGEN_REF
    virtualPDU::appLogic(); virtualPDU::checkRecordTimes(); virtualPDU::recordTelem(); virtualPDU::loadConfig();
    #endif
    // clang-format on
    g_faults = {};
    Fixture app;
    app.configure();
    REQUIRE( !app.m_shutdown );
    REQUIRE( app.appLogic() == 0 );
    REQUIRE( g_faults.m_records.size() == 1 );
    REQUIRE( app.appLogic() == 0 );
    REQUIRE( g_faults.m_records.size() == 1 );
    g_faults.m_due = true;
    REQUIRE( app.appLogic() == 0 );
    REQUIRE( g_faults.m_records.size() == 2 );
    g_faults.m_due = false;
    app.observed();
    REQUIRE( app.appLogic() == 0 );
    auto &record = g_faults.m_records.back();
    REQUIRE( record.m_code == MagAOX::logger::telem_outlet::eventCode );
    REQUIRE( MagAOX::logger::telem_outlet::states( record.m_payload.data() ) == std::vector<int8_t>{ 0, 0, 0 } );
    g_faults.m_recordResult = -1;
    REQUIRE( app.setCallBack_source( property( "ac", "power", "state", "On" ) ) < 0 );
    REQUIRE( app.recordTelem( static_cast<const MagAOX::logger::telem_outlet *>( nullptr ) ) < 0 );
    app.m_outletTelemRecorded = false;
    REQUIRE( app.appLogic() < 0 );
    g_faults.m_recordResult    = 0;
    g_faults.m_telemResults[3] = -1;
    REQUIRE( app.appLogic() < 0 );
    g_faults.m_telemResults[3] = 0;
    app.m_failUpdate           = true;
    REQUIRE( app.appLogic() < 0 );
    app.m_failUpdate = false;
    contended( app.m_indiMutex, [&] { REQUIRE( app.appLogic() == 0 ); } );
    REQUIRE( app.appShutdown() == 0 );
}
/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
struct PowerConsumer : outletTestApp<false>
{
    /// Enable the real power-monitoring callback on a sequential offline consumer.
    PowerConsumer();
    /// No process setup is needed for an isolated callback consumer.
    void setupConfig() override
    {
    }
    /// No configuration is loaded for an isolated callback consumer.
    void loadConfig() override
    {
    }
    /// Keep the fixture lifecycle offline.
    int appStartup() override
    {
        return 0;
    }
    /// Keep the fixture lifecycle offline.
    int appLogic() override
    {
        return 0;
    }
    /// Keep the fixture lifecycle offline.
    int appShutdown() override
    {
        return 0;
    }
};
PowerConsumer::PowerConsumer() : outletTestApp( "test", false )
{
    m_powerMgtEnabled = true;
}
/// \endcond

/// Verify that a fresh app records its initial snapshot even if a previous instance recorded identical states.
/** \ingroup virtualPDU_unit_test */
TEST_CASE( "Virtual PDU telemetry suppression belongs to each app instance", "[virtualPDU]" )
{
    // clang-format off
    #ifdef VIRTUALPDU_TEST_DOXYGEN_REF
    virtualPDU::appLogic(); virtualPDU::recordTelem();
    #endif
    // clang-format on
    g_faults = {};
    for( int instance = 0; instance < 2; ++instance )
    {
        Fixture app;
        app.configure();
        REQUIRE( app.appLogic() == 0 );
        REQUIRE( g_faults.m_records.size() == static_cast<size_t>( instance + 1 ) );
    }
}

/// Verify the real power consumer interprets the published aggregate state and target.
/** \ingroup virtualPDU_unit_test */
TEST_CASE( "Virtual PDU properties drive the real MagAOXApp power callback", "[virtualPDU][power]" )
{
    // clang-format off
    #ifdef VIRTUALPDU_TEST_DOXYGEN_REF
    virtualPDU::setCallBack_source(); virtualPDU::newCallBack_channels();
    MagAOX::app::MagAOXApp<false>::setCallBack_m_indiP_powerChannel();
    #endif
    // clang-format on
    g_faults = {};
    std::vector<pcf::IndiProperty> snapshots;
    {
        Fixture app;
        app.configure();
        app.driver();
        app.observed();
        for( auto value : { "On", "Off", "Int", "Unk" } )
        {
            REQUIRE( app.setCallBack_source( property( "ac", "power", "state", value ) ) == 0 );
            REQUIRE( app.setCallBack_source( property( "usb", "power", "state", value ) ) == 0 );
            if( std::string( value ) == "On" || std::string( value ) == "Off" )
                REQUIRE( app.newCallBack_channels( property( "test-pdu", "combined", "target", value ) ) == 0 );
            REQUIRE( app.updateINDI() == 0 );
            snapshots.push_back( app.m_channels["combined"].m_indiP_prop );
        }
    }
    PowerConsumer consumer;
    for( size_t index = 0; index < snapshots.size(); ++index )
    {
        REQUIRE( consumer.setCallBack_m_indiP_powerChannel( snapshots[index] ) == 0 );
        REQUIRE( consumer.powerState() == ( index == 0 ? 1 : index == 1 ? 0 : -1 ) );
        REQUIRE( consumer.powerStateTarget() == ( index == 0 ? 1 : 0 ) );
    }
}
} // namespace virtualPDUTest
} // namespace libXWCTest
