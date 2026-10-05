/** \file zaberLowLevel_test.cpp
 * \brief Catch2 tests for the zaberLowLevel app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup zaberLowLevel_files
 */

#include <filesystem>
#include <fstream>

// Direct include to avoid having to link separately
extern "C"
{
#include "../za_serial.c"
}

#include "../../../tests/testXWC.hpp"
#include "../../../tests/testMacrosINDI.hpp"

#include "../zaberLowLevel.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup zaberLowLevel_unit_test zaberLowLevel Unit Tests
 * \brief Unit tests for the zaberLowLevel application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `zaberLowLevel` unit tests.
/** \ingroup zaberLowLevel_unit_test
 */
namespace zaberLowLevelTest
{

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
class zaberLowLevel_test : public zaberLowLevel
{
  public:
    /// Construct a testable low-level controller instance.
    zaberLowLevel_test( const std::string &device )
    {
        m_configName = device;

        XWCTEST_SETUP_INDI_NEW_PROP( tgt_pos );
        XWCTEST_SETUP_INDI_NEW_PROP( req_home );
        XWCTEST_SETUP_INDI_NEW_PROP( req_home_all );
        XWCTEST_SETUP_INDI_NEW_PROP( req_halt );
        XWCTEST_SETUP_INDI_NEW_PROP( req_ehalt );
        XWCTEST_SETUP_INDI_NEW_PROP( knob_enable );
        XWCTEST_SETUP_INDI_NEW_PROP( led_enable );
    }

    /// Set up a single staged snapshot and INDI transport for power-off tests.
    int setupPowerOffSnapshot( const std::string &stageName, long rawPos, bool parked, long maxPos, time_t lastHomed )
    {
        std::error_code ec;

        m_testRoot = std::filesystem::temp_directory_path() / ( "zaberLowLevel_test_" + m_configName );
        std::filesystem::remove_all( m_testRoot, ec );

        m_basePath = m_testRoot.string();
        m_sysPath  = ( m_testRoot / "sys" ).string();

        std::filesystem::create_directories( m_testRoot / MAGAOX_driverFIFORelPath );
        std::filesystem::create_directories( std::filesystem::path( m_sysPath ) / m_configName );

        m_stages.emplace_back( this );
        m_stages.back().name( stageName );
        m_stages.back().serial( "serial0" );

        {
            std::ofstream stateOut( std::filesystem::path( m_sysPath ) / m_configName / stageName );
            stateOut << rawPos << '\n' << parked << '\n' << maxPos << '\n' << lastHomed << '\n';
        }

        if( appStartup() < 0 )
        {
            return -1;
        }

        if( createINDIFIFOS() < 0 )
        {
            return -1;
        }

        m_indiDriver = new indiDriver<MagAOXAppT>( this, m_configName, "0", "0" );

        return ( m_indiDriver && m_indiDriver->good() ) ? 0 : -1;
    }

    /// Configure a stage entry for discovery tests.
    int addConfiguredStage( const std::string &stageName, const std::string &serial, int deviceAddress = -1 )
    {
        m_stages.emplace_back( this );
        m_stages.back().name( stageName );
        m_stages.back().serial( serial );
        m_stages.back().deviceAddress( deviceAddress );

        const size_t idx = m_stages.size() - 1;

        m_stageName.insert( { stageName, idx } );
        m_stageSerial.insert( { serial, idx } );

        return 0;
    }

    /// Load the parsed system-serial snapshot through the production discovery code.
    int loadParsedStages( std::string serialResponse )
    {
        return loadStages( serialResponse );
    }

    /// Set the cached device address for a configured stage.
    int setDeviceAddressFor( size_t stageIndex, int deviceAddress )
    {
        m_stages.at( stageIndex ).deviceAddress( deviceAddress );
        return 0;
    }

    /// Get the cached device address for a configured stage.
    int deviceAddressFor( size_t stageIndex ) const
    {
        return m_stages.at( stageIndex ).deviceAddress();
    }

    /// Drive the recoverable error handler under test.
    int recoverTransportError( bool devicePresent )
    {
        return recoverFromError( devicePresent );
    }

    /// Set the FSM state for recovery tests.
    int setAppState( stateCodes::stateCodeT newState )
    {
        state( newState );
        return 0;
    }

    /// Get the FSM state for recovery tests.
    stateCodes::stateCodeT appState() const
    {
        return state();
    }

    /// Read the value of a text or number element from a test property.
    std::string propertyValue( const pcf::IndiProperty &property, const std::string &element ) const
    {
        return property[element].getValue();
    }

    /// Get the current-position property value for a stage.
    std::string currPosValue( const std::string &stageName ) const
    {
        return propertyValue( m_indiP_curr_pos, stageName );
    }

    /// Get the target-position property value for a stage.
    std::string tgtPosValue( const std::string &stageName ) const
    {
        return propertyValue( m_indiP_tgt_pos, stageName );
    }

    /// Get the parked-state property value for a stage.
    std::string parkedValue( const std::string &stageName ) const
    {
        return propertyValue( m_indiP_parked, stageName );
    }

    /// Get the last-homed property value for a stage.
    std::string lastHomedValue( const std::string &stageName ) const
    {
        return propertyValue( m_indiP_lastHomed, stageName );
    }

    /// Get the max-position property value for a stage.
    std::string maxPosValue( const std::string &stageName ) const
    {
        return propertyValue( m_indiP_max_pos, stageName );
    }

    /// Get the current-state property value for a stage.
    std::string currStateValue( const std::string &stageName ) const
    {
        return propertyValue( m_indiP_curr_state, stageName );
    }

    /// Get the warning-switch property value for a stage.
    std::string warnValue( const std::string &stageName ) const
    {
        return propertyValue( m_indiP_warn, stageName );
    }

    /// Invoke the power-off handling under test.
    int doOnPowerOff()
    {
        return onPowerOff();
    }

    ~zaberLowLevel_test() noexcept
    {
        std::error_code ec;

        delete m_indiDriver;
        m_indiDriver = nullptr;
        std::filesystem::remove_all( m_testRoot, ec );
    }

  private:
    std::filesystem::path m_testRoot; ///< Temporary directory backing the test FIFOs and state snapshot.
};
/// \endcond

/// Verify zaberLowLevel callback validation and power-off snapshots preserve stage state.
/**
 * \ingroup zaberLowLevel_unit_test
 */
SCENARIO( "INDI Callbacks", "[zaberLowLevel]" )
{
    // clang-format off
    #ifdef ZABERLOWLEVEL_TEST_DOXYGEN_REF
    zaberLowLevel::newCallBack_m_indiP_tgt_pos( pcf::IndiProperty() );
    zaberLowLevel::newCallBack_m_indiP_req_home( pcf::IndiProperty() );
    zaberLowLevel::onPowerOff();
    #endif
    // clang-format on

    XWCTEST_INDI_NEW_CALLBACK( zaberLowLevel, tgt_pos );
    XWCTEST_INDI_NEW_CALLBACK( zaberLowLevel, req_home );
    XWCTEST_INDI_NEW_CALLBACK( zaberLowLevel, req_home_all );
    XWCTEST_INDI_NEW_CALLBACK( zaberLowLevel, req_halt );
    XWCTEST_INDI_NEW_CALLBACK( zaberLowLevel, req_ehalt );
    XWCTEST_INDI_NEW_CALLBACK( zaberLowLevel, knob_enable );
    XWCTEST_INDI_NEW_CALLBACK( zaberLowLevel, led_enable );
}

SCENARIO( "Power-off INDI snapshot retains stage state", "[zaberLowLevel]" )
{
    zaberLowLevel_test zllt( "zlltest" );

    REQUIRE( zllt.setupPowerOffSnapshot( "stageA", 12345, true, 54321, 77 ) == 0 );

    REQUIRE( zllt.doOnPowerOff() == 0 );

    REQUIRE( zllt.currPosValue( "stageA" ) == "12345" );
    REQUIRE( zllt.tgtPosValue( "stageA" ) == "12345" );
    REQUIRE( zllt.parkedValue( "stageA" ) == "1" );
    REQUIRE( zllt.lastHomedValue( "stageA" ) == "77" );
    REQUIRE( zllt.maxPosValue( "stageA" ) == "54321" );
    REQUIRE( zllt.currStateValue( "stageA" ) == "POWEROFF" );
    REQUIRE( zllt.warnValue( "stageA" ) == "Off" );
}

/// Verify discovery clears stale addresses and reports missing configured stages safely.
/**
 * \ingroup zaberLowLevel_unit_test
 */
SCENARIO( "Stage discovery resets stale device addresses", "[zaberLowLevel]" )
{
    // clang-format off
    #ifdef ZABERLOWLEVEL_TEST_DOXYGEN_REF
    zaberLowLevel::loadStages( std::declval<std::string &>() );
    #endif
    // clang-format on

    zaberLowLevel_test zllt( "zlltest" );

    REQUIRE( zllt.addConfiguredStage( "stagebs", "49820", 1 ) == 0 );
    REQUIRE( zllt.addConfiguredStage( "stageirf", "49821", 2 ) == 0 );

    std::string serialResponse = "@01 0 OK IDLE WR 49820\n";

    REQUIRE( zllt.loadParsedStages( serialResponse ) == ZC_CONNECTED );
    REQUIRE( zllt.deviceAddressFor( 0 ) == 1 );
    REQUIRE( zllt.deviceAddressFor( 1 ) < 1 );
}

/// Verify a later discovery pass can find a stage that was missing initially.
/**
 * \ingroup zaberLowLevel_unit_test
 */
SCENARIO( "Stage discovery can find devices that appear later", "[zaberLowLevel]" )
{
    zaberLowLevel_test zllt( "zlltest_rediscover" );

    REQUIRE( zllt.addConfiguredStage( "stagebs", "49820" ) == 0 );
    REQUIRE( zllt.addConfiguredStage( "stageirf", "49821" ) == 0 );

    std::string serialResponse1 = "@01 0 OK IDLE WR 49820\n";
    std::string serialResponse2 = "@01 0 OK IDLE WR 49820\n@02 0 OK IDLE WR 49821\n";

    REQUIRE( zllt.loadParsedStages( serialResponse1 ) == ZC_CONNECTED );
    REQUIRE( zllt.deviceAddressFor( 0 ) == 1 );
    REQUIRE( zllt.deviceAddressFor( 1 ) < 1 );

    REQUIRE( zllt.loadParsedStages( serialResponse2 ) == ZC_CONNECTED );
    REQUIRE( zllt.deviceAddressFor( 0 ) == 1 );
    REQUIRE( zllt.deviceAddressFor( 1 ) == 2 );
}

/// Verify communication failures drop the app back into reconnectable states.
/**
 * \ingroup zaberLowLevel_unit_test
 */
SCENARIO( "Recoverable transport errors transition to reconnect states", "[zaberLowLevel]" )
{
    // clang-format off
    #ifdef ZABERLOWLEVEL_TEST_DOXYGEN_REF
    zaberLowLevel::resetConnection();
    zaberLowLevel::recoverFromError( true );
    #endif
    // clang-format on

    SECTION( "A present tty returns the app to NOTCONNECTED" )
    {
        zaberLowLevel_test zllt( "zlltest_present" );

        REQUIRE( zllt.setupPowerOffSnapshot( "stageA", 12345, true, 54321, 77 ) == 0 );
        REQUIRE( zllt.setDeviceAddressFor( 0, 1 ) == 0 );
        REQUIRE( zllt.setAppState( stateCodes::ERROR ) == 0 );

        REQUIRE( zllt.recoverTransportError( true ) == 0 );
        REQUIRE( zllt.appState() == stateCodes::NOTCONNECTED );
        REQUIRE( zllt.currStateValue( "stageA" ) == "NOTCONNECTED" );
    }

    SECTION( "A missing tty returns the app to NODEVICE" )
    {
        zaberLowLevel_test zllt( "zlltest_missing" );

        REQUIRE( zllt.setupPowerOffSnapshot( "stageA", 12345, true, 54321, 77 ) == 0 );
        REQUIRE( zllt.setDeviceAddressFor( 0, 1 ) == 0 );
        REQUIRE( zllt.setAppState( stateCodes::ERROR ) == 0 );

        REQUIRE( zllt.recoverTransportError( false ) == 0 );
        REQUIRE( zllt.appState() == stateCodes::NODEVICE );
        REQUIRE( zllt.currStateValue( "stageA" ) == "NODEVICE" );
    }
}

} // namespace zaberLowLevelTest

} // namespace libXWCTest
