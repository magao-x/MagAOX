/** \file acronameUsbHub_test.cpp
 * \brief Offline USB-hub configuration, FSM, BrainStem failure, and telemetry tests.
 * \ingroup acronameUsbHub_files
 */
#include "../../../tests/testXWC.hpp"
#include "../../../tests/outletAppTest.hpp"
#include "../../../libs/BrainStem2/BrainStem2/BrainStem-all.h"

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
namespace hubHarness
{
/// Scripted BrainStem connection and USB-port outcomes, with no USB enumeration.
struct Faults
{
    /// Next connection result.
    aErr m_connect{ aErrNone };
    /// Port-read result.
    aErr m_read{ aErrNone };
    /// Enable/disable result.
    aErr m_write{ aErrNone };
    /// Connected flag returned by the fake hub.
    bool m_connected{ true };
    /// Number of disconnect operations, including destructor cleanup.
    unsigned m_disconnects{ 0 };
    /// Per-port enable state.
    std::array<uint32_t, 8> m_ports{};
    /// Serial used by the production connect operation.
    uint32_t m_serial{ 0 };
};
/// Suite-local scripted state.
inline Faults g_faults;
/// Fake port-control entity matching only the API used by the real app.
struct Ports
{
    /// Return the selected actual port state and a scripted acquisition error.
    aErr getPortState( int port /**< [in] zero-based port */, uint32_t *state /**< [out] scripted state */ );
    /// Enable a port and return the scripted outcome.
    aErr setPortEnable( int port /**< [in] zero-based port */ );
    /// Disable a port and return the scripted outcome.
    aErr setPortDisable( int port /**< [in] zero-based port */ );
};
aErr Ports::getPortState( int port, uint32_t *state )
{
    *state = g_faults.m_ports.at( port );
    return g_faults.m_read;
}
aErr Ports::setPortEnable( int port )
{
    g_faults.m_ports.at( port ) = 1;
    return g_faults.m_write;
}
aErr Ports::setPortDisable( int port )
{
    g_faults.m_ports.at( port ) = 0;
    return g_faults.m_write;
}
/// Fake hub which cannot perform hardware I/O.
struct Hub
{
    /// Port entity used by the actual app.
    Ports usb;
    /// Capture the requested serial and return a scripted connection result.
    aErr connect( int transport /**< [in] ignored transport */, uint32_t serial /**< [in] serial */ );
    /// Count cleanup and disconnect requests.
    void disconnect();
    /// Return the scripted connection state.
    bool isConnected();
};
aErr Hub::connect( int, uint32_t serial )
{
    g_faults.m_serial = serial;
    return g_faults.m_connect;
}
void Hub::disconnect()
{
    ++g_faults.m_disconnects;
}
bool Hub::isConnected()
{
    return g_faults.m_connected;
}
/// Fake model/version/serial entity for the app's informational connection log.
struct System
{
    /// Initialize without accessing a module.
    void init( Hub *hub /**< [in] ignored hub */, int index /**< [in] ignored entity */ );
    /// Return a fixed test model.
    void getModel( uint8_t *model /**< [out] model */ );
    /// Return a fixed test firmware version.
    void getVersion( uint32_t *version /**< [out] firmware version */ );
    /// Return the connection's captured serial.
    void getSerialNumber( uint32_t *serial /**< [out] serial */ );
};
void System::init( Hub *, int )
{
}
void System::getModel( uint8_t *model )
{
    *model = 1;
}
void System::getVersion( uint32_t *version )
{
    *version = 2;
}
void System::getSerialNumber( uint32_t *serial )
{
    *serial = g_faults.m_serial;
}
/// Resolve the scripted model without calling BrainStem.
const char *modelName( uint8_t model /**< [in] ignored model */ );
/// Format the scripted version without calling BrainStem.
void        versionString( uint32_t version /**< [in] ignored version */,
                           char    *text /**< [out] text */,
                           size_t   size /**< [in] buffer capacity */ );
const char *modelName( uint8_t )
{
    return "TestHub";
}
void versionString( uint32_t, char *text, size_t size )
{
    std::snprintf( text, size, "2.0" );
}
} // namespace hubHarness
/// \endcond
#define MagAOXApp outletTestApp
#define telemeter outletTestTelemeter
#define aUSBHub3p hubHarness::Hub
#define SystemClass hubHarness::System
#define aDefs_GetModelName hubHarness::modelName
#define aVersion_ParseString hubHarness::versionString
#include "../acronameUsbHub.hpp"
#undef aVersion_ParseString
#undef aDefs_GetModelName
#undef SystemClass
#undef aUSBHub3p
#undef telemeter
#undef MagAOXApp

namespace libXWCTest
{
/** \defgroup acronameUsbHub_unit_test acronameUsbHub Unit Tests
 * \ingroup application_unit_test
 */
namespace acronameUsbHubTest
{
using namespace MagAOX::app;
using namespace outletHarness;
/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
/// Expose state while exercising the actual controller methods.
struct Fixture : Controller<acronameUsbHub>
{
    /// Expose the configured device serial.
    using acronameUsbHub::m_serialNumber;
    /// Expose connection state for cleanup/connection-loss contracts.
    using acronameUsbHub::m_connected;
    /// Load the real serial/channel configuration.
    void configure();
};
void Fixture::configure()
{
    configText( "[device]\nserialNumber=1234\n[camera]\noutlets=0,1\n" );
    loadConfig();
    REQUIRE( !m_shutdown );
}
/// \endcond

/// Cover configuration, standard interface startup, and telemetry helper failures.
/** \ingroup acronameUsbHub_unit_test */
TEST_CASE( "USB hub configuration and lifecycle", "[acronameUsbHub]" )
{
    // clang-format off
    #ifdef ACRONAMEUSBHUB_TEST_DOXYGEN_REF
    acronameUsbHub::acronameUsbHub(); acronameUsbHub::~acronameUsbHub(); acronameUsbHub::setupConfig();
    acronameUsbHub::loadConfig(); acronameUsbHub::loadConfigImpl(); acronameUsbHub::appStartup(); acronameUsbHub::appShutdown();
    #endif
    // clang-format on
    g_faults             = {};
    hubHarness::g_faults = {};
    {
        Fixture app;
        app.configure();
        REQUIRE( app.m_serialNumber == 1234 );
        REQUIRE( app.channelOutlets( "camera" ) == std::vector<size_t>{ 0, 1 } );
        REQUIRE( app.appStartup() == 0 );
        REQUIRE( app.state() == stateCodes::NOTCONNECTED );
        REQUIRE( app.appShutdown() == 0 );
    }
    REQUIRE( hubHarness::g_faults.m_disconnects == 1 );
    for( size_t call : { 0, 1, 2, 4 } )
    {
        g_faults             = {};
        hubHarness::g_faults = {};
        Fixture app;
        g_faults.m_telemResults[call] = -1;
        app.configText( "[camera]\noutlet=0\n" );
        if( call == 0 )
            REQUIRE( app.m_shutdown );
        else if( call == 1 )
        {
            app.loadConfig();
            REQUIRE( app.m_shutdown );
        }
        else
        {
            REQUIRE( app.loadConfigImpl( app.config ) == 0 );
            if( call == 2 )
                REQUIRE( app.appStartup() < 0 );
            else
                REQUIRE( app.appShutdown() == 0 );
        }
    }
    for( unsigned call = 1; call <= 6; ++call )
    {
        g_faults             = {};
        hubHarness::g_faults = {};
        Fixture app;
        app.configure();
        g_faults.m_failRegistration = call;
        REQUIRE( app.appStartup() < 0 );
    }
    g_faults             = {};
    hubHarness::g_faults = {};
    Fixture invalid;
    invalid.configText( "[device]\nserialNumber=1234\n" );
    invalid.loadConfig();
    REQUIRE( invalid.m_shutdown );
}

/// Verify retry, connection logs, dropped-link invalidation, and power-off cleanup.
/** \ingroup acronameUsbHub_unit_test */
TEST_CASE( "USB hub connection FSM and power-off state", "[acronameUsbHub]" )
{
    // clang-format off
    #ifdef ACRONAMEUSBHUB_TEST_DOXYGEN_REF
    acronameUsbHub::appLogic(); acronameUsbHub::onPowerOff(); acronameUsbHub::whilePowerOff();
    #endif
    // clang-format on
    g_faults             = {};
    hubHarness::g_faults = {};
    Fixture app;
    app.configure();
    REQUIRE( app.appStartup() == 0 );
    hubHarness::g_faults.m_connect = aErrConnection;
    app.state( stateCodes::POWERON );
    REQUIRE( app.appLogic() == 0 );
    REQUIRE( app.state() == stateCodes::NOTCONNECTED );
    app.m_connected = true;
    REQUIRE( app.appLogic() == 0 );
    REQUIRE( !app.m_connected );
    hubHarness::g_faults.m_connect = aErrNone;
    REQUIRE( app.appLogic() == 0 );
    REQUIRE( app.m_connected );
    REQUIRE( app.state() == stateCodes::READY );
    REQUIRE( hubHarness::g_faults.m_serial == 1234 );
    REQUIRE( std::any_of( g_faults.m_logs.begin(),
                          g_faults.m_logs.end(),
                          []( const Log &log )
                          { return log.m_message.find( "TestHub #1234" ) != std::string::npos; } ) );
    hubHarness::g_faults.m_connected = false;
    REQUIRE( app.appLogic() == 0 );
    REQUIRE( app.state() == stateCodes::NOTCONNECTED );
    REQUIRE( !app.m_connected );
    for( int n = 0; n < 8; ++n )
        REQUIRE( app.outletState( n ) == OUTLET_STATE_UNKNOWN );
    app.m_connected = true;
    REQUIRE( app.onPowerOff() == 0 );
    REQUIRE( !app.m_connected );
    REQUIRE( app.onPowerOff() == 0 );
    for( int n = 0; n < 8; ++n )
        REQUIRE( app.outletState( n ) == OUTLET_STATE_OFF );
    REQUIRE( app.whilePowerOff() == 0 );
    app.state( stateCodes::FAILURE );
    REQUIRE( app.appLogic() == 0 );
    g_faults.m_telemResults[3] = -1;
    REQUIRE( app.whilePowerOff() < 0 );
}

/// Exercise actual port reads/enables/disables and every handled BrainStem error.
/** \ingroup acronameUsbHub_unit_test */
TEST_CASE( "USB hub port state and control error contracts", "[acronameUsbHub]" )
{
    // clang-format off
    #ifdef ACRONAMEUSBHUB_TEST_DOXYGEN_REF
    acronameUsbHub::updateOutletState(); acronameUsbHub::turnOutletOn(); acronameUsbHub::turnOutletOff();
    #endif
    // clang-format on
    g_faults             = {};
    hubHarness::g_faults = {};
    Fixture app;
    app.configure();
    for( int n = 0; n < 8; ++n )
    {
        REQUIRE( app.turnOutletOn( n ) == 0 );
        REQUIRE( app.updateOutletState( n ) == 0 );
        REQUIRE( app.outletState( n ) == OUTLET_STATE_ON );
        REQUIRE( app.turnOutletOff( n ) == 0 );
        REQUIRE( app.updateOutletState( n ) == 0 );
        REQUIRE( app.outletState( n ) == OUTLET_STATE_OFF );
    }
    for( aErr error : { aErrTimeout, aErrConnection, aErrParam } )
    {
        hubHarness::g_faults.m_read = error;
        REQUIRE( app.updateOutletState( 0 ) == ( error == aErrParam ? 0 : -1 ) );
        hubHarness::g_faults.m_write = error;
        REQUIRE( app.turnOutletOn( 0 ) == ( error == aErrParam ? 0 : -1 ) );
        REQUIRE( app.turnOutletOff( 0 ) == ( error == aErrParam ? 0 : -1 ) );
    }
}

/// Observe initial/change/forced telemetry and its lifecycle failure propagation.
/** \ingroup acronameUsbHub_unit_test */
TEST_CASE( "USB hub telemetry captures observed states", "[acronameUsbHub]" )
{
    // clang-format off
    #ifdef ACRONAMEUSBHUB_TEST_DOXYGEN_REF
    acronameUsbHub::appLogic(); acronameUsbHub::checkRecordTimes(); acronameUsbHub::recordTelem(); acronameUsbHub::onPowerOff();
    #endif
    // clang-format on
    g_faults             = {};
    hubHarness::g_faults = {};
    Fixture app;
    app.configure();
    REQUIRE( app.appStartup() == 0 );
    REQUIRE( app.appLogic() == 0 );
    REQUIRE( g_faults.m_records.size() == 1 );
    REQUIRE( app.appLogic() == 0 );
    REQUIRE( g_faults.m_records.size() == 1 );
    hubHarness::g_faults.m_ports[0] = 1;
    REQUIRE( app.appLogic() == 0 );
    REQUIRE( MagAOX::logger::telem_outlet::states( g_faults.m_records.back().m_payload.data() )[0] == OUTLET_STATE_ON );
    g_faults.m_due = true;
    REQUIRE( app.appLogic() == 0 );
    REQUIRE( app.whilePowerOff() == 0 );
    g_faults.m_due          = false;
    g_faults.m_recordResult = -1;
    REQUIRE( app.recordTelem( static_cast<const MagAOX::logger::telem_outlet *>( nullptr ) ) < 0 );
    app.m_outletTelemRecorded = false;
    REQUIRE( app.appLogic() < 0 );
    g_faults.m_recordResult    = 0;
    g_faults.m_telemResults[3] = -1;
    REQUIRE( app.appLogic() < 0 );
}
/// Reject the one-past-end physical outlet configuration before control or telemetry can index it.
/** \ingroup acronameUsbHub_unit_test */
TEST_CASE( "acronameUsbHub rejects a one-past-end configured outlet", "[acronameUsbHub]" )
{
    // clang-format off
    #ifdef ACRONAMEUSBHUB_TEST_DOXYGEN_REF
    acronameUsbHub::loadConfig(); acronameUsbHub::loadConfigImpl();
    #endif
    // clang-format on
    outletHarness::g_faults = {};
    Fixture app;
    app.configText( "[invalid]\noutlet=8\n" );
    app.loadConfig();
    REQUIRE( app.m_shutdown );
}
} // namespace acronameUsbHubTest
} // namespace libXWCTest
