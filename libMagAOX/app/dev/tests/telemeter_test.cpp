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

        telemeter_tests::telemeterTest pdt( "xx", false );

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

        telemeter_tests::telemeterTest pdt( "xx", false );

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

/// Test telemeter maxLogTime Configuration
/**
 * \ingroup telemeter_tests
 */
TEST_CASE( "Test telemeter maxLogTime Configuration", "[dev::telemeter]" )
{
    SECTION( "a valid maxLogTime is applied" )
    {
        mx::app::writeConfigFile( "/tmp/telemeter_test.conf", { "telemeter" }, { "maxLogTime" }, { "15" } );

        mx::app::appConfigurator config;

        telemeter_tests::telemeterTest pdt( "xx", false );

        REQUIRE( pdt.setupConfig( config ) == 0 );

        config.readConfig( "/tmp/telemeter_test.conf" );

        REQUIRE( pdt.loadConfig( config ) == 0 );
        REQUIRE( pdt.m_tel.maxLogTime() == 15 );
        REQUIRE( pdt.shutdown() == 0 );
    }

    SECTION( "a decimal maxLogTime, as written by Python apps, is accepted" )
    {
        mx::app::writeConfigFile( "/tmp/telemeter_test.conf", { "telemeter" }, { "maxLogTime" }, { "15.0" } );

        mx::app::appConfigurator config;

        telemeter_tests::telemeterTest pdt( "xx", false );

        REQUIRE( pdt.setupConfig( config ) == 0 );

        config.readConfig( "/tmp/telemeter_test.conf" );

        REQUIRE( pdt.loadConfig( config ) == 0 );
        REQUIRE( pdt.m_tel.maxLogTime() == 15 );
        REQUIRE( pdt.shutdown() == 0 );
    }

    SECTION( "no maxLogTime keeps the default" )
    {
        mx::app::writeConfigFile( "/tmp/telemeter_test.conf", { "none" }, { "nada" }, { "0" } );

        mx::app::appConfigurator config;

        telemeter_tests::telemeterTest pdt( "xx", false );

        REQUIRE( pdt.setupConfig( config ) == 0 );

        config.readConfig( "/tmp/telemeter_test.conf" );

        REQUIRE( pdt.loadConfig( config ) == 0 );
        REQUIRE( pdt.m_tel.maxLogTime() == MAGAOX_default_maxLogTime );
        REQUIRE( pdt.shutdown() == 0 );
    }

    SECTION( "an invalid maxLogTime is not applied, the default is kept, and the app keeps running" )
    {
        for( std::string bad : { "-5", "abc", "600000", "0.3" } )
        {
            INFO( "maxLogTime = \"" << bad << "\"" );

            mx::app::writeConfigFile( "/tmp/telemeter_test.conf", { "telemeter" }, { "maxLogTime" }, { bad } );

            mx::app::appConfigurator config;

            telemeter_tests::telemeterTest pdt( "xx", false );

            REQUIRE( pdt.setupConfig( config ) == 0 );

            config.readConfig( "/tmp/telemeter_test.conf" );

            REQUIRE( pdt.loadConfig( config ) == 0 );
            REQUIRE( pdt.m_tel.maxLogTime() == MAGAOX_default_maxLogTime );
            REQUIRE( pdt.shutdown() == 0 );
        }
    }

    SECTION( "logManager::loadConfig reports an invalid maxLogTime to the caller" )
    {
        mx::app::writeConfigFile( "/tmp/telemeter_test.conf", { "telemeter" }, { "maxLogTime" }, { "abc" } );

        mx::app::appConfigurator config;

        telemeter_tests::telemeterTest pdt( "xx", false );

        REQUIRE( pdt.setupConfig( config ) == 0 );

        config.readConfig( "/tmp/telemeter_test.conf" );

        std::string err;
        REQUIRE( pdt.m_tel.loadConfig( config, &err ) == -1 );

        INFO( "errMsg: " << err );
        REQUIRE( err.find( "telemeter.maxLogTime" ) != std::string::npos );
        REQUIRE( err.find( "\"abc\"" ) != std::string::npos );
        REQUIRE( err.find( "Using " + std::to_string( MAGAOX_default_maxLogTime ) ) != std::string::npos );
        REQUIRE( pdt.m_tel.maxLogTime() == MAGAOX_default_maxLogTime );
    }

    SECTION( "a valid maxLogTime leaves the error message empty" )
    {
        mx::app::writeConfigFile( "/tmp/telemeter_test.conf", { "telemeter" }, { "maxLogTime" }, { "15" } );

        mx::app::appConfigurator config;

        telemeter_tests::telemeterTest pdt( "xx", false );

        REQUIRE( pdt.setupConfig( config ) == 0 );

        config.readConfig( "/tmp/telemeter_test.conf" );

        std::string err;
        REQUIRE( pdt.m_tel.loadConfig( config, &err ) == 0 );
        REQUIRE( err.empty() );
        REQUIRE( pdt.m_tel.maxLogTime() == 15 );
    }

    SECTION( "an invalid maxLogTime does not prevent the remaining settings loading" )
    {
        mx::app::writeConfigFile( "/tmp/telemeter_test.conf",
                                  { "telemeter", "telemeter", "telemeter" },
                                  { "maxLogTime", "writePause", "maxInterval" },
                                  { "abc", "12345", "25" } );

        mx::app::appConfigurator config;

        telemeter_tests::telemeterTest pdt( "xx", false );

        REQUIRE( pdt.setupConfig( config ) == 0 );

        config.readConfig( "/tmp/telemeter_test.conf" );

        REQUIRE( pdt.loadConfig( config ) == 0 );
        REQUIRE( pdt.m_tel.maxLogTime() == MAGAOX_default_maxLogTime );
        REQUIRE( pdt.m_tel.writePause() == 12345 );
        REQUIRE( pdt.m_maxInterval == 25 );
        REQUIRE( pdt.shutdown() == 0 );
    }
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

        telemeter_tests::telemeterTest pdt( "xx", false );

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

        telemeter_tests::telemeterTest pdt( "xx", false );

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

    telemeter_tests::XWCTEST_TELEMETER_LOGSTART_ns::telemeterTest pdt( "xx", false );

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
