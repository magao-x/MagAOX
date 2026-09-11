// #define CATCH_CONFIG_MAIN
#include "../../../../tests/catch2/catch.hpp"

#include <mx/sys/timeUtils.hpp>

#include "telemeter_test.hpp"

#undef app_telemeter_hpp
#undef MAPPNS
#define XWCTEST_NAMESPACE XWCTEST_TELEMETER_LOGSTART_ns
#define XWCTEST_TELEMETER_LOGSTART
#include "telemeter_test.hpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_TELEMETER_LOGSTART

/** \defgroup app_dev_unit_tests libXWC::app::dev Unit Tests
 * \ingroup app_unit_test
*/

/** \defgroup telemeter_tests libXWC::app::dev::telemeter Unit Tests
 * \ingroup app_dev_unit_tests
 */

/// Test telemeter Configuration
/**
 * \ingroup telemeter_tests
 */
TEST_CASE( "Test telemeter Configuration", "[dev::telemeter]" )
{
    SECTION( "a config file with no [telemeter] section, loading defaults" )
    {
        // Just a dummy config setting
        mx::app::writeConfigFile( "/tmp/telemeter_test.conf", { "none" }, { "nada" }, { "0" } );

        mx::app::appConfigurator config;

        telemeter_tests::telemeterTest pdt;

        int rv;
        rv = pdt.setupConfig( config );
        REQUIRE( rv == 0 );

        config.readConfig( "/tmp/telemeter_test.conf" );

        rv = pdt.loadConfig( config );
        REQUIRE( rv == 0 );

        REQUIRE( pdt.m_tel.logPath().find( "telem" ) != std::string::npos );
        REQUIRE( pdt.m_tel.logExt() == "bintel" );
        REQUIRE( pdt.m_tel.logName() == pdt.configName() );
        REQUIRE( pdt.m_maxInterval == 10.0 );
    }

    SECTION( "a config file with a [telemeter] section changing everything" )
    {
        // Just a dummy config setting
        mx::app::writeConfigFile( "/tmp/telemeter_test.conf",
                                  { "telemeter", "telemeter", "telemeter" },
                                  { "logDir", "logExt", "maxInterval" },
                                  { "/new/log/path", "txt", "25" } );

        mx::app::appConfigurator config;

        telemeter_tests::telemeterTest pdt;

        int rv;
        rv = pdt.setupConfig( config );
        REQUIRE( rv == 0 );

        config.readConfig( "/tmp/telemeter_test.conf" );

        rv = pdt.loadConfig( config );
        REQUIRE( rv == 0 );

        REQUIRE( pdt.m_tel.logPath() == "/new/log/path" );
        REQUIRE( pdt.m_tel.logExt() == "txt" );
        REQUIRE( pdt.m_maxInterval == 25 );
    }

     #ifdef XWCTEST_DOX_REF
    MagAOX::app::dev::telemeter::setupConfig();
    MagAOX::app::dev::telemeter::loadConfig();
    #endif
}

/// Test telemeter app logic
/**
 * \ingroup telemeter_tests
 */
TEST_CASE( "Test telemeter app logic", "[dev::telemeter]" )
{
    SECTION( "no errors" )
    {
        // Just a dummy config setting
        mx::app::writeConfigFile( "/tmp/telemeter_test.conf", { "none" }, { "nada" }, { "0" } );

        mx::app::appConfigurator config;

        telemeter_tests::telemeterTest pdt;

        int rv;
        rv = pdt.setupConfig( config );
        REQUIRE( rv == 0 );

        config.readConfig( "/tmp/telemeter_test.conf" );

        rv = pdt.loadConfig( config );
        REQUIRE( rv == 0 );

        pdt.m_tel.logPath( "/tmp/telems" );

        rv = pdt.appStartup();
        REQUIRE( rv == 0 );

        rv = pdt.appLogic();
        REQUIRE( rv == 0 );

        rv = pdt.appShutdown();
        REQUIRE( rv == 0 );
    }

    SECTION( "log thread shutsdown" )
    {
        // Just a dummy config setting
        mx::app::writeConfigFile( "/tmp/telemeter_test.conf", { "none" }, { "nada" }, { "0" } );

        mx::app::appConfigurator config;

        telemeter_tests::telemeterTest pdt;

        int rv;
        rv = pdt.setupConfig( config );
        REQUIRE( rv == 0 );

        config.readConfig( "/tmp/telemeter_test.conf" );

        rv = pdt.loadConfig( config );
        REQUIRE( rv == 0 );

        pdt.m_tel.logPath( "/tmp/telems" );

        rv = pdt.appStartup();
        REQUIRE( rv == 0 );

        rv = pdt.appLogic();
        REQUIRE( rv == 0 );

        pdt.m_tel.logShutdown( true );
        sleep( 1 );

        rv = pdt.appLogic();
        REQUIRE( rv == -1 );

        rv = pdt.appShutdown();
        REQUIRE( rv == 0 );
    }

    #ifdef XWCTEST_DOX_REF
    MagAOX::app::dev::telemeter::setupConfig();
    MagAOX::app::dev::telemeter::loadConfig();
    MagAOX::app::dev::telemeter::appStartup();
    MagAOX::app::dev::telemeter::appLogic();
    MagAOX::app::dev::telemeter::appShutdown();
    #endif
}

/// Test telemeter telem-logger fails to start
/**
 * \ingroup telemeter_tests
 */
TEST_CASE( "Test telemeter telem-logger fails to start", "[dev::telemeter]" )
{
    // Just a dummy config setting
    mx::app::writeConfigFile( "/tmp/telemeter_test.conf", { "none" }, { "nada" }, { "0" } );

    mx::app::appConfigurator config;

    telemeter_tests::XWCTEST_TELEMETER_LOGSTART_ns::telemeterTest pdt;

    int rv;
    rv = pdt.setupConfig( config );
    REQUIRE( rv == 0 );

    config.readConfig( "/tmp/telemeter_test.conf" );

    rv = pdt.loadConfig( config );
    REQUIRE( rv == 0 );

    pdt.m_tel.logPath( "/tmp/telems" );

    rv = pdt.appStartup();
    REQUIRE( rv == -1 );

    #ifdef XWCTEST_DOX_REF
    MagAOX::app::dev::telemeter::setupConfig();
    MagAOX::app::dev::telemeter::loadConfig();
    MagAOX::app::dev::telemeter::appStartup();
    #endif
}
