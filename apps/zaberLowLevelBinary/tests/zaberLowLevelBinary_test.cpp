/** \file zaberLowLevelBinary_test.cpp
 * \brief Catch2 tests for the zaberLowLevelBinary app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup zaberLowLevelBinary_files
 */

#include <filesystem>
#include <fstream>

extern "C"
{
#include "../zb_serial.c"
}

#include "../../../tests/catch2/catch.hpp"
#include "../../tests/testMacrosINDI.hpp"

#include "../zaberLowLevelBinary.hpp"

using namespace MagAOX::app;

namespace ZLLBTEST
{

class zaberLowLevelBinary_test : public zaberLowLevelBinary
{
  public:
    /// Construct the test harness and set up INDI callback fixtures.
    zaberLowLevelBinary_test( const std::string &device )
    {
        m_configName = device;

        XWCTEST_SETUP_INDI_NEW_PROP( tgt_pos );
        XWCTEST_SETUP_INDI_NEW_PROP( req_home );
        XWCTEST_SETUP_INDI_NEW_PROP( req_home_all );
        XWCTEST_SETUP_INDI_NEW_PROP( req_halt );
        XWCTEST_SETUP_INDI_NEW_PROP( req_ehalt );
        XWCTEST_SETUP_INDI_NEW_PROP( knob_enable );
    }

    /// Set up a single staged snapshot and INDI transport for power-off tests.
    int setupPowerOffSnapshot( const std::string &stageName, long rawPos, bool parked, long maxPos, time_t lastHomed )
    {
        std::error_code ec;

        m_testRoot = std::filesystem::temp_directory_path() / ( "zaberLowLevelBinary_test_" + m_configName );
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

    /// Read the value of a text, number, or switch element from a test property.
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

    ~zaberLowLevelBinary_test() noexcept
    {
        std::error_code ec;

        delete m_indiDriver;
        m_indiDriver = nullptr;
        std::filesystem::remove_all( m_testRoot, ec );
    }

  private:
    std::filesystem::path m_testRoot; ///< Temporary directory backing the test FIFOs and state snapshot.
};

SCENARIO( "INDI Callbacks", "[zaberLowLevelBinary]" )
{
    XWCTEST_INDI_NEW_CALLBACK( zaberLowLevelBinary, tgt_pos );
    XWCTEST_INDI_NEW_CALLBACK( zaberLowLevelBinary, req_home );
    XWCTEST_INDI_NEW_CALLBACK( zaberLowLevelBinary, req_home_all );
    XWCTEST_INDI_NEW_CALLBACK( zaberLowLevelBinary, req_halt );
    XWCTEST_INDI_NEW_CALLBACK( zaberLowLevelBinary, req_ehalt );
    XWCTEST_INDI_NEW_CALLBACK( zaberLowLevelBinary, knob_enable );
}

SCENARIO( "Power-off INDI snapshot retains stage state", "[zaberLowLevelBinary]" )
{
    zaberLowLevelBinary_test zllbt( "zllbtest" );

    REQUIRE( zllbt.setupPowerOffSnapshot( "stageA", 12345, true, 54321, 77 ) == 0 );

    REQUIRE( zllbt.doOnPowerOff() == 0 );

    REQUIRE( zllbt.currPosValue( "stageA" ) == "12345" );
    REQUIRE( zllbt.tgtPosValue( "stageA" ) == "12345" );
    REQUIRE( zllbt.parkedValue( "stageA" ) == "1" );
    REQUIRE( zllbt.lastHomedValue( "stageA" ) == "77" );
    REQUIRE( zllbt.maxPosValue( "stageA" ) == "54321" );
    REQUIRE( zllbt.currStateValue( "stageA" ) == "POWEROFF" );
    REQUIRE( zllbt.warnValue( "stageA" ) == "Off" );
}

} // namespace ZLLBTEST
