/** \file zaberLowLevel_test.cpp
 * \brief Catch2 tests for the zaberLowLevel app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * History:
 */

#include <filesystem>
#include <fstream>

// Direct include to avoid having to link separately
extern "C"
{
#include "../za_serial.c"
}

#include "../../../tests/catch2/catch.hpp"
#include "../../tests/testMacrosINDI.hpp"

#include "../zaberLowLevel.hpp"

using namespace MagAOX::app;

namespace ZLLTEST
{

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

SCENARIO( "INDI Callbacks", "[zaberLowLevel]" )
{
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

} // namespace ZLLTEST
