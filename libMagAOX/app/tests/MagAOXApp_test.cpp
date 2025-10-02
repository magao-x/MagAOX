// #define CATCH_CONFIG_MAIN
#include "../../../tests/catch2/catch.hpp"

#include <filesystem>

#include <mx/sys/timeUtils.hpp>

#include "../MagAOXApp.hpp"
#include "MagAOXApp_test.hpp"

#undef app_MagAOXApp_hpp
#undef app_tests_MagAOXApp_test_hpp
#define XWCTEST_NAMESPACE XWCTEST_MAGAOXAPP_PID_LOCKED_ns
#define XWCTEST_MAGAOXAPP_PID_LOCKED
#include "../MagAOXApp.hpp"
#include "MagAOXApp_test.hpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_MAGAOXAPP_PID_LOCKED

#undef app_MagAOXApp_hpp
#undef app_tests_MagAOXApp_test_hpp
#define XWCTEST_NAMESPACE XWCTEST_MAGAOXAPP_PID_WRITE_FAIL_ns
#define XWCTEST_MAGAOXAPP_PID_WRITE_FAIL
#include "../MagAOXApp.hpp"
#include "MagAOXApp_test.hpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_MAGAOXAPP_PID_WRITE_FAIL

#undef app_MagAOXApp_hpp
#undef app_tests_MagAOXApp_test_hpp
#define XWCTEST_NAMESPACE XWCTEST_MAGAOXAPP_EXEC_NORM_ns
#define XWCTEST_MAGAOXAPP_EXEC_NORM
#include "../MagAOXApp.hpp"
#include "MagAOXApp_test.hpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_MAGAOXAPP_EXEC_NORM


namespace libXWCTest
{
namespace appTest
{

/** \defgroup app_unit_test libXWC::app Unit Tests
 * \ingroup unit_test
*/

/** \defgroup MagAOXApp_unit_test MagAOXApp Unit Tests
 * \ingroup app_unit_test
 */

/// Namespace for XWC::app::MagAOXApp tests
/** \ingroup MagAOXApp_unit_test
 *
 */
namespace MagAOXAppTest
{

/// MagAOXApp 2nd instance
/**
 * \ingroup MagAOXApp_unit_test
 */
TEST_CASE( "MagAOXApp 2nd instance", "[app::MagAOXApp]" )
{
    SECTION( "test 2nd app" )
    {
        bool caught = false;

        MagAOXApp_test app1;

        try
        {
            MagAOXApp_test app2;
        }
        catch( const std::logic_error &e )
        {
            caught = true;
        }

        REQUIRE( caught == true );
    }

    #ifdef XWCTEST_DOXYGEN_REF_PROTECTED
    MagAOX::app::MagAOXApp<true> app("", true);
    #endif

}

/// MagAOXApp INDI NewProperty
/**
 * \ingroup MagAOXApp_unit_test
 */
SCENARIO( "MagAOXApp INDI NewProperty", "[app::MagAOXApp]" )
{
    GIVEN( "a new property request" )
    {
        WHEN( "a wrong device name" )
        {
            MagAOXApp_test app;

            app.setConfigName( "test" );

            REQUIRE( app.configName() == "test" );

            pcf::IndiProperty prop;
            app.registerIndiPropertyNew( prop,
                                         "nprop",
                                         pcf::IndiProperty::Number,
                                         pcf::IndiProperty::ReadWrite,
                                         pcf::IndiProperty::Idle,
                                         callback );

            pcf::IndiProperty nprop;

            // First test the right device name
            nprop.setDevice( "test" );
            nprop.setName( "nprop" );

            app.handleNewProperty( nprop );

            REQUIRE( app.called_back == 1 );

            app.called_back = 0;

            // Now test the wrong device name
            nprop.setDevice( "wrong" );

            app.handleNewProperty( nprop );

            REQUIRE( app.called_back == 0 );
        }
    }

    #ifdef XWCTEST_DOXYGEN_REF_PROTECTED
    MagAOX::app::MagAOXApp<true> app("", true);
    app.configName();
    pcf::IndiProperty prop;
    app.registerIndiPropertyNew( prop,
                                 "nprop",
                                         pcf::IndiProperty::Number,
                                         pcf::IndiProperty::ReadWrite,
                                         pcf::IndiProperty::Idle,
                                         callback );
    app.handleNewProperty(prop);
    #endif
}

/// Setting defaults
/**
 * \ingroup MagAOXApp_unit_test
 */
TEST_CASE( "Setting defaults", "[app::MagAOXApp]" )
{
    SECTION( "using default paths, configname is invoked name" )
    {
        std::vector<std::string> argvstr( { "./execname" } );

        std::vector<const char *> argv( argvstr.size() + 1, NULL );
        for( size_t index = 0; index < argvstr.size(); ++index )
        {
            argv[index] = argvstr[index].c_str();
        }

        MagAOXApp_test app;

        app.invokedName() = argv[0];
        REQUIRE( app.doHelp() == false );
        app.setDefaults( argv.size() - 1, const_cast<char **>( argv.data() ) );
        REQUIRE( app.doHelp() == true );

        app.basePath(); // make lcov records this call
        REQUIRE( app.basePath() == MAGAOX_path );
        app.configDir(); // make lcov records this call
        REQUIRE( app.configDir() == app.basePath() + '/' + MAGAOX_configRelPath );
        REQUIRE( app.configPathGlobal() == app.configDir() + "/magaox.conf" );
        app.calibDir(); // make lcov records this call
        REQUIRE( app.calibDir() == app.basePath() + '/' + MAGAOX_calibRelPath );
        REQUIRE( app.m_log.logPath() == app.basePath() + '/' + MAGAOX_logRelPath );
        app.sysPath(); // make lcov records this call
        REQUIRE( app.sysPath() == app.basePath() + '/' + MAGAOX_sysRelPath );
        app.secretsPath(); // make lcov records this call
        REQUIRE( app.secretsPath() == app.basePath() + '/' + MAGAOX_secretsRelPath );
        app.cpusetPath(); // make lcov records this call
        REQUIRE( app.cpusetPath() == MAGAOX_cpusetPath );
        app.configBase(); // make lcov records this call
        REQUIRE( app.configBase() == "" );
        REQUIRE( app.configPathUser() == "" );
        REQUIRE( app.configName() == "execname" );
        REQUIRE( app.configPathLocal() == app.configDir() + "/execname.conf" );

        REQUIRE( app.doHelp() == true );
    }

    SECTION( "using default paths, with config-ed name" )
    {
        std::vector<std::string> argvstr( { "./execname", "-n", "testapp" } );

        std::vector<const char *> argv( argvstr.size() + 1, NULL );
        for( size_t index = 0; index < argvstr.size(); ++index )
        {
            argv[index] = argvstr[index].c_str();
        }

        MagAOXApp_test app;
        app.invokedName() = argv[0];

        app.setDefaults( argv.size() - 1, const_cast<char **>( argv.data() ) );

        REQUIRE( app.basePath() == MAGAOX_path );
        REQUIRE( app.configDir() == app.basePath() + '/' + MAGAOX_configRelPath );
        REQUIRE( app.configPathGlobal() == app.configDir() + "/magaox.conf" );
        REQUIRE( app.calibDir() == app.basePath() + '/' + MAGAOX_calibRelPath );
        REQUIRE( app.m_log.logPath() == app.basePath() + '/' + MAGAOX_logRelPath );

        REQUIRE( app.sysPath() == app.basePath() + '/' + MAGAOX_sysRelPath );
        REQUIRE( app.secretsPath() == app.basePath() + '/' + MAGAOX_secretsRelPath );
        REQUIRE( app.cpusetPath() == MAGAOX_cpusetPath );
        REQUIRE( app.configBase() == "" );
        REQUIRE( app.configPathUser() == "" );
        REQUIRE( app.configName() == "testapp" );
        REQUIRE( app.configPathLocal() == app.configDir() + "/testapp.conf" );
        REQUIRE( app.doHelp() == false );
    }

    // Something goes wrong here, third time is the charm.
    //  Hangs on config.parseCommandLine
    SECTION( "using environment paths, with config-ed name" )
    {
        std::vector<const char *> argv;
        std::vector<std::string>  argvstr( { "./execname", "--name", "testapp2" } );

        argv.resize( argvstr.size() + 1, NULL );
        for( size_t index = 0; index < argvstr.size(); ++index )
        {
            argv[index] = argvstr[index].c_str();
        }

        char ppath[1024];
        snprintf( ppath, sizeof( ppath ), "%s=/tmp/MagAOXApp_test", MAGAOX_env_path );
        putenv( ppath );

        char cpath[1024];
        snprintf( cpath, sizeof( cpath ), "%s=config2", MAGAOX_env_config );
        putenv( cpath );

        char cbpath[1024];
        snprintf( cbpath, sizeof( cbpath ), "%s=calib2", MAGAOX_env_calib );
        putenv( cbpath );

        char lpath[1024];
        snprintf( lpath, sizeof( lpath ), "%s=logs2", MAGAOX_env_log );
        putenv( lpath );

        char syspath[1024];
        snprintf( syspath, sizeof( syspath ), "%s=sys2", MAGAOX_env_sys );
        putenv( syspath );

        char secretspath[1024];
        snprintf( secretspath, sizeof( secretspath ), "%s=secrets2", MAGAOX_env_secrets );
        putenv( secretspath );

        char cpupath[1024];
        snprintf( cpupath, sizeof( cpupath ), "%s=/tmp/MagAOX/cpuset", MAGAOX_env_cpuset );
        putenv( cpupath );

        MagAOXApp_test app;

        app.invokedName() = argv[0];
        app.setConfigBase( "cbase" );

        app.setDefaults( argv.size() - 1, const_cast<char **>( argv.data() ) );

        REQUIRE( app.basePath() == "/tmp/MagAOXApp_test" );
        REQUIRE( app.configDir() == app.basePath() + '/' + "config2" );
        REQUIRE( app.configPathGlobal() == app.configDir() + "/magaox.conf" );
        REQUIRE( app.calibDir() == app.basePath() + '/' + "calib2" );
        REQUIRE( app.m_log.logPath() == app.basePath() + '/' + "logs2" );
        REQUIRE( app.sysPath() == app.basePath() + '/' + "sys2" );
        REQUIRE( app.secretsPath() == app.basePath() + '/' + "secrets2" );
        REQUIRE( app.cpusetPath() == "/tmp/MagAOX/cpuset" );
        REQUIRE( app.configBase() == "cbase" );
        REQUIRE( app.configPathUser() == app.configDir() + "/cbase.conf" );
        REQUIRE( app.configName() == "testapp2" );
        REQUIRE( app.configPathLocal() == app.configDir() + "/testapp2.conf" );
        REQUIRE( app.doHelp() == false );
    }
}

/// Configuring MagAOXApp
/**
 * \ingroup MagAOXApp_unit_test
 */
TEST_CASE( "Configuring MagAOXApp", "[app::MagAOXApp]" )
{
    SECTION( "setup basic config" )
    {
        MagAOXApp_test app;
        app.setPowerMgtEnabled( true );

        app.setupBasicConfig();

        REQUIRE( app.shutdown() == false );
    }

    SECTION( "load basic config w all defaults w/out pwr management" )
    {
        MagAOXApp_test app;
        app.setPowerMgtEnabled( false );

        app.setupBasicConfig();

        app.loadBasicConfig();

        app.checkConfig();

        REQUIRE( app.stateAlert() == false );
        REQUIRE( app.gitAlert() == false );
        REQUIRE( app.shutdown() == false );
    }

    SECTION( "load basic config w all defaults w/out pwr management, setting state and clearing alerts" )
    {
        std::vector<const char *> argv;
        std::vector<std::string>  argvstr( { "./execname", "--name", "testapp" } );

        argv.resize( argvstr.size() + 1, NULL );
        for( size_t index = 0; index < argvstr.size(); ++index )
        {
            argv[index] = argvstr[index].c_str();
        }

        MagAOXApp_test app( true );
        app.setPowerMgtEnabled( false );

        app.setupBasicConfig();

        app.setDefaults( argv.size() - 1, const_cast<char **>( argv.data() ) );

        app.loadBasicConfig();

        app.checkConfig();

        REQUIRE( app.stateAlert() == true );
        REQUIRE( app.gitAlert() == true );
        REQUIRE( app.shutdown() == false );

        app.doFSMClearAlert();
        REQUIRE( app.stateAlert() == false );
        REQUIRE( app.gitAlert() == true );
        REQUIRE( app.shutdown() == false );

        app.doFSMClearAlert(); // calls an immediate return of clearFSMAlert

        // Now test each path out of clearFSMAlert
        app.state( MagAOX::app::stateCodes::READY );
        REQUIRE( app.state() == MagAOX::app::stateCodes::READY );

        REQUIRE( app.stateLogged() == 0 );
        REQUIRE( app.stateLogged() == 1 );

        app.setAlert();
        REQUIRE( app.stateAlert() == true );

        app.doFSMClearAlert();
        REQUIRE( app.stateAlert() == false );

        app.state( MagAOX::app::stateCodes::HOMING );
        REQUIRE( app.state() == MagAOX::app::stateCodes::HOMING );

        app.setAlert();
        REQUIRE( app.stateAlert() == true );

        app.doFSMClearAlert();
        REQUIRE( app.stateAlert() == false );

        app.state( MagAOX::app::stateCodes::NODEVICE );
        REQUIRE( app.state() == MagAOX::app::stateCodes::NODEVICE );

        app.setAlert();
        REQUIRE( app.stateAlert() == true );

        app.doFSMClearAlert();
        REQUIRE( app.stateAlert() == false );

        app.state( MagAOX::app::stateCodes::LOGGEDIN );
        REQUIRE( app.state() == MagAOX::app::stateCodes::LOGGEDIN );

        app.setAlert();
        REQUIRE( app.stateAlert() == true );

        app.doFSMClearAlert();
        REQUIRE( app.stateAlert() == false );

        app.state( MagAOX::app::stateCodes::NODEVICE );
        REQUIRE( app.state() == MagAOX::app::stateCodes::NODEVICE );

        app.setAlert();
        REQUIRE( app.stateAlert() == true );

        app.doFSMClearAlert();
        REQUIRE( app.stateAlert() == false );

        app.state( MagAOX::app::stateCodes::NOTHOMED );
        REQUIRE( app.state() == MagAOX::app::stateCodes::NOTHOMED );

        app.setAlert();
        REQUIRE( app.stateAlert() == true );

        app.doFSMClearAlert();
        REQUIRE( app.stateAlert() == false );
    }

    SECTION( "load basic config w all defaults w unconfigured pwr management" )
    {
        MagAOXApp_test app;
        app.setPowerMgtEnabled( true );

        app.setupBasicConfig();

        app.loadBasicConfig();

        REQUIRE( app.shutdown() == true );

        app.checkConfig();

        REQUIRE( app.stateAlert() == false );
        REQUIRE( app.gitAlert() == false );
        REQUIRE( app.shutdown() == true );
    }

    SECTION( "load a full config" )
    {
        std::vector<const char *> argv;
        std::vector<std::string>  argvstr( { "./execname", "-n", "testapp", "--config.validate" } );

        argv.resize( argvstr.size() + 1, NULL );
        for( size_t index = 0; index < argvstr.size(); ++index )
        {
            argv[index] = argvstr[index].c_str();
        }

        char ppath[1024];
        snprintf( ppath, sizeof( ppath ), "%s=/tmp/MagAOXApp_test", MAGAOX_env_path );
        putenv( ppath );

        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/config" );

        std::ofstream fout;
        fout.open( "/tmp/MagAOXApp_test/config/magaox.conf" );
        fout.close();

        mx::app::writeConfigFile( "/tmp/MagAOXApp_test/config/testapp.conf",
                                  { "", "power", "power", "power", "power", "power" },
                                  { "loopPause", "device", "channel", "element", "targetElement", "powerOnWait" },
                                  { "2500", "pdu9", "thisch", "thisel", "thistgtel", "500000" } );

        MagAOXApp_test app( true );
        app.setPowerMgtEnabled( true );

        app.setup( argv.size() - 1, const_cast<char **>( argv.data() ) );

        REQUIRE( app.stateAlert() == true ); // due to git
        REQUIRE( app.configOnly() == true );
        REQUIRE( app.loopPause() == 2500 );
        REQUIRE( app.powerDevice() == "pdu9" );
        REQUIRE( app.powerChannel() == "thisch" );
        REQUIRE( app.powerElement() == "thisel" );
        REQUIRE( app.powerTargetElement() == "thistgtel" );
        REQUIRE( app.powerOnWait() == 0 );

        REQUIRE( app.shutdown() == false );
    }

    SECTION( "load a full config w unknown config in file, do help" )
    {
        std::vector<const char *> argv;
        std::vector<std::string>  argvstr( { "./execname", "-n", "testapp" } );

        argv.resize( argvstr.size() + 1, NULL );
        for( size_t index = 0; index < argvstr.size(); ++index )
        {
            argv[index] = argvstr[index].c_str();
        }

        char ppath[1024];
        snprintf( ppath, sizeof( ppath ), "%s=/tmp/MagAOXApp_test", MAGAOX_env_path );
        putenv( ppath );

        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/config" );

        std::ofstream fout;
        fout.open( "/tmp/MagAOXApp_test/config/magaox.conf" );
        fout.close();

        // this adds unknown=value
        mx::app::writeConfigFile(
            "/tmp/MagAOXApp_test/config/testapp.conf",
            { "", "power", "power", "power", "power", "power", "" },
            { "loopPause", "device", "channel", "element", "targetElement", "powerOnWait", "unknown" },
            { "2500", "pdu9", "thisch", "thisel", "thistgtel", "500", "value" } );

        MagAOXApp_test app( false );
        app.setPowerMgtEnabled( true );

        app.setup( argv.size() - 1, const_cast<char **>( argv.data() ) );

        REQUIRE( app.stateAlert() == false );
        REQUIRE( app.configOnly() == false );
        REQUIRE( app.loopPause() == 2500 );
        REQUIRE( app.powerDevice() == "pdu9" );
        REQUIRE( app.powerChannel() == "thisch" );
        REQUIRE( app.powerElement() == "thisel" );
        REQUIRE( app.powerTargetElement() == "thistgtel" );
        REQUIRE( app.powerOnWait() == 500 );

        REQUIRE( app.doHelp() == true );
        REQUIRE( app.shutdown() == true );
    }

    SECTION( "load a full config w unknown config in file, validate" )
    {
        std::vector<const char *> argv;
        std::vector<std::string>  argvstr( { "./execname", "-n", "testapp", "--config.validate" } );

        argv.resize( argvstr.size() + 1, NULL );
        for( size_t index = 0; index < argvstr.size(); ++index )
        {
            argv[index] = argvstr[index].c_str();
        }

        char ppath[1024];
        snprintf( ppath, sizeof( ppath ), "%s=/tmp/MagAOXApp_test", MAGAOX_env_path );
        putenv( ppath );

        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/config" );

        std::ofstream fout;
        fout.open( "/tmp/MagAOXApp_test/config/magaox.conf" );
        fout.close();

        // this adds unknown=value
        mx::app::writeConfigFile(
            "/tmp/MagAOXApp_test/config/testapp.conf",
            { "", "power", "power", "power", "power", "power", "" },
            { "loopPause", "device", "channel", "element", "targetElement", "powerOnWait", "unknown" },
            { "2500", "pdu9", "thisch", "thisel", "thistgtel", "500000", "value" } );

        MagAOXApp_test app( true );
        app.setPowerMgtEnabled( true );

        app.setup( argv.size() - 1, const_cast<char **>( argv.data() ) );

        REQUIRE( app.stateAlert() == true ); // due to git
        REQUIRE( app.configOnly() == true );
        REQUIRE( app.loopPause() == 2500 );
        REQUIRE( app.powerDevice() == "pdu9" );
        REQUIRE( app.powerChannel() == "thisch" );
        REQUIRE( app.powerElement() == "thisel" );
        REQUIRE( app.powerTargetElement() == "thistgtel" );
        REQUIRE( app.powerOnWait() == 0 );

        REQUIRE( app.doHelp() == false );
        REQUIRE( app.shutdown() == true );
    }

    SECTION( "load a full config w non-option clopt" )
    {
        std::vector<const char *> argv;
        std::vector<std::string>  argvstr( { "./execname", "-n", "testapp", "straylight" } );

        argv.resize( argvstr.size() + 1, NULL );
        for( size_t index = 0; index < argvstr.size(); ++index )
        {
            argv[index] = argvstr[index].c_str();
        }

        char ppath[1024];
        snprintf( ppath, sizeof( ppath ), "%s=/tmp/MagAOXApp_test", MAGAOX_env_path );
        putenv( ppath );

        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/config" );

        std::ofstream fout;
        fout.open( "/tmp/MagAOXApp_test/config/magaox.conf" );
        fout.close();

        mx::app::writeConfigFile( "/tmp/MagAOXApp_test/config/testapp.conf",
                                  { "", "power", "power", "power", "power", "power" },
                                  { "loopPause", "device", "channel", "element", "targetElement", "powerOnWait" },
                                  { "2500", "pdu9", "thisch", "thisel", "thistgtel", "500000" } );

        MagAOXApp_test app( false );
        app.setPowerMgtEnabled( true );

        app.setup( argv.size() - 1, const_cast<char **>( argv.data() ) );

        REQUIRE( app.stateAlert() == false ); // due to git
        REQUIRE( app.configOnly() == false );
        REQUIRE( app.loopPause() == 2500 );
        REQUIRE( app.powerDevice() == "pdu9" );
        REQUIRE( app.powerChannel() == "thisch" );
        REQUIRE( app.powerElement() == "thisel" );
        REQUIRE( app.powerTargetElement() == "thistgtel" );
        REQUIRE( app.powerOnWait() == 0 );

        REQUIRE( app.doHelp() == true );
        REQUIRE( app.shutdown() == true );
    }

    SECTION( "load a full config w no power mgt opts" )
    {
        std::vector<const char *> argv;
        std::vector<std::string>  argvstr( { "./execname", "-n", "testapp" } );

        argv.resize( argvstr.size() + 1, NULL );
        for( size_t index = 0; index < argvstr.size(); ++index )
        {
            argv[index] = argvstr[index].c_str();
        }

        char ppath[1024];
        snprintf( ppath, sizeof( ppath ), "%s=/tmp/MagAOXApp_test", MAGAOX_env_path );
        putenv( ppath );

        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/config" );

        std::ofstream fout;
        fout.open( "/tmp/MagAOXApp_test/config/magaox.conf" );
        fout.close();

        mx::app::writeConfigFile( "/tmp/MagAOXApp_test/config/testapp.conf", { "" }, { "loopPause" }, { "2500" } );

        MagAOXApp_test app( false );
        app.setPowerMgtEnabled( true );

        app.setup( argv.size() - 1, const_cast<char **>( argv.data() ) );

        REQUIRE( app.stateAlert() == false ); // due to git
        REQUIRE( app.configOnly() == false );
        REQUIRE( app.loopPause() == 2500 );
        REQUIRE( app.powerDevice() == "" );
        REQUIRE( app.powerChannel() == "" );
        REQUIRE( app.powerElement() == "state" );
        REQUIRE( app.powerTargetElement() == "target" );
        REQUIRE( app.powerOnWait() == 0 );

        REQUIRE( app.doHelp() == true );
        REQUIRE( app.shutdown() == true );
    }

    SECTION( "load a full config w unused config options" )
    {
        std::vector<const char *> argv;
        std::vector<std::string>  argvstr( { "./execname", "-n", "testapp", "--config.validate" } );

        argv.resize( argvstr.size() + 1, NULL );
        for( size_t index = 0; index < argvstr.size(); ++index )
        {
            argv[index] = argvstr[index].c_str();
        }

        char ppath[1024];
        snprintf( ppath, sizeof( ppath ), "%s=/tmp/MagAOXApp_test", MAGAOX_env_path );
        putenv( ppath );

        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/config" );

        std::ofstream fout;
        fout.open( "/tmp/MagAOXApp_test/config/magaox.conf" );
        fout.close();

        // this adds unknown=value
        mx::app::writeConfigFile( "/tmp/MagAOXApp_test/config/testapp.conf",
                                  { "", "power", "power", "power", "power", "power" },
                                  { "loopPause", "device", "channel", "element", "targetElement", "powerOnWait" },
                                  { "2500", "pdu9", "thisch", "thisel", "thistgtel", "500000" } );

        MagAOXApp_test app( true );
        app.setPowerMgtEnabled( true );

        app.addUnusedConfig();

        app.setup( argv.size() - 1, const_cast<char **>( argv.data() ) );

        REQUIRE( app.stateAlert() == true ); // due to git
        REQUIRE( app.configOnly() == true );
        REQUIRE( app.loopPause() == 2500 );
        REQUIRE( app.powerDevice() == "pdu9" );
        REQUIRE( app.powerChannel() == "thisch" );
        REQUIRE( app.powerElement() == "thisel" );
        REQUIRE( app.powerTargetElement() == "thistgtel" );
        REQUIRE( app.powerOnWait() == 0 );

        REQUIRE( app.doHelp() == false );
        REQUIRE( app.shutdown() == false );
    }
}

/// PID Locking
/**
 * \ingroup MagAOXApp_unit_test
 */
TEST_CASE( "PID Locking", "[app::MagAOXApp]" )
{
    SECTION( "Basic PID Lock" )
    {
        std::vector<const char *> argv;
        std::vector<std::string>  argvstr( { "./execname", "-n", "testapp" } );

        argv.resize( argvstr.size() + 1, NULL );
        for( size_t index = 0; index < argvstr.size(); ++index )
        {
            argv[index] = argvstr[index].c_str();
        }

        char ppath[1024];
        snprintf( ppath, sizeof( ppath ), "%s=/tmp/MagAOXApp_test", MAGAOX_env_path );
        putenv( ppath );

        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/sys/" );

        // First delete the directory and files in case this is a repeat call
        std::filesystem::remove_all( "/tmp/MagAOXApp_test/sys/testapp" );

        MagAOXApp_test app;
        app.invokedName() = argv[0];
        app.setDefaults( argv.size() - 1, const_cast<char **>( argv.data() ) );

        REQUIRE( !std::filesystem::exists( "/tmp/MagAOXApp_test/sys/testapp/pid" ) );

        int rv = app.lockPID();
        REQUIRE( rv == 0 );
        REQUIRE( std::filesystem::exists( "/tmp/MagAOXApp_test/sys/testapp/pid" ) );

        rv = app.unlockPID();
        REQUIRE( rv == 0 );
        REQUIRE( !std::filesystem::exists( "/tmp/MagAOXApp_test/sys/testapp/pid" ) );
    }

    SECTION( "PID Lock, app directory creation error" )
    {
        std::vector<const char *> argv;
        std::vector<std::string>  argvstr( { "./execname", "-n", "testapp" } );

        argv.resize( argvstr.size() + 1, NULL );
        for( size_t index = 0; index < argvstr.size(); ++index )
        {
            argv[index] = argvstr[index].c_str();
        }

        char ppath[1024];
        snprintf( ppath, sizeof( ppath ), "%s=/tmp/MagAOXApp_test", MAGAOX_env_path );
        putenv( ppath );

        // First delete the directory and files in case this is a repeat call
        std::filesystem::remove_all( "/tmp/MagAOXApp_test" );

        MagAOXApp_test app;
        app.invokedName() = argv[0];
        app.setDefaults( argv.size() - 1, const_cast<char **>( argv.data() ) );

        int rv = app.lockPID();
        REQUIRE( rv == -1 );

        rv = app.unlockPID();
        REQUIRE( rv == -1 );
    }

    SECTION( "Stale lock" )
    {
        std::vector<const char *> argv;
        std::vector<std::string>  argvstr( { "./execname", "-n", "testapp" } );

        argv.resize( argvstr.size() + 1, NULL );
        for( size_t index = 0; index < argvstr.size(); ++index )
        {
            argv[index] = argvstr[index].c_str();
        }

        char ppath[1024];
        snprintf( ppath, sizeof( ppath ), "%s=/tmp/MagAOXApp_test", MAGAOX_env_path );
        putenv( ppath );

        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/sys/testapp" );

        std::ofstream fout;
        fout.open( "/tmp/MagAOXApp_test/sys/testapp/pid" );
        fout << 1;
        fout.close();

        MagAOXApp_test app;
        app.invokedName() = argv[0];
        app.setDefaults( argv.size() - 1, const_cast<char **>( argv.data() ) );

        int rv = app.lockPID();
        REQUIRE( rv == 0 );
        REQUIRE( std::filesystem::exists( "/tmp/MagAOXApp_test/sys/testapp/pid" ) );

        rv = app.unlockPID();
        REQUIRE( rv == 0 );
        REQUIRE( !std::filesystem::exists( "/tmp/MagAOXApp_test/sys/testapp/pid" ) );
    }

    SECTION( "already locked" )
    {
        std::vector<const char *> argv;
        std::vector<std::string>  argvstr( { "./execname", "-n", "testapp" } );

        argv.resize( argvstr.size() + 1, NULL );
        for( size_t index = 0; index < argvstr.size(); ++index )
        {
            argv[index] = argvstr[index].c_str();
        }

        char ppath[1024];
        snprintf( ppath, sizeof( ppath ), "%s=/tmp/MagAOXApp_test", MAGAOX_env_path );
        putenv( ppath );

        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/sys/testapp" );

        std::ofstream fout;
        fout.open( "/tmp/MagAOXApp_test/sys/testapp/pid" );
        fout << 1;
        fout.close();

        XWCTEST_MAGAOXAPP_PID_LOCKED_ns::MagAOXApp_test app;
        app.invokedName() = argv[0];
        app.setDefaults( argv.size() - 1, const_cast<char **>( argv.data() ) );

        int rv = app.lockPID();
        REQUIRE( rv == -1 );
    }

    SECTION( "write fails" )
    {
        std::vector<const char *> argv;
        std::vector<std::string>  argvstr( { "./execname", "-n", "testapp" } );

        argv.resize( argvstr.size() + 1, NULL );
        for( size_t index = 0; index < argvstr.size(); ++index )
        {
            argv[index] = argvstr[index].c_str();
        }

        char ppath[1024];
        snprintf( ppath, sizeof( ppath ), "%s=/tmp/MagAOXApp_test", MAGAOX_env_path );
        putenv( ppath );

        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/sys/testapp" );

        std::ofstream fout;
        fout.open( "/tmp/MagAOXApp_test/sys/testapp/pid" );
        fout << 1;
        fout.close();

        XWCTEST_MAGAOXAPP_PID_WRITE_FAIL_ns::MagAOXApp_test app;
        app.invokedName() = argv[0];
        app.setDefaults( argv.size() - 1, const_cast<char **>( argv.data() ) );

        int rv = app.lockPID();
        REQUIRE( rv == -1 );
    }
}

/// MagAOXApp Power Management Logic Outside of Execute
/**
 * \ingroup MagAOXApp_unit_test
 */
TEST_CASE( "MagAOXApp Power Management Logic Outside of Execute", "[app::MagAOXApp]" )
{
    SECTION( "Power Management Not Configured" )
    {
        MagAOXApp_test app( true );
        app.setPowerMgtEnabled( false );

        REQUIRE( app.onPowerOff() == 0 );
        REQUIRE( app.whilePowerOff() == 0 );
        REQUIRE( app.powerOnWaitElapsed() == true );
        REQUIRE( app.powerState() == 1 );
        REQUIRE( app.powerStateTarget() == 1 );
    }

    SECTION( "Power Management Configured" )
    {
        MagAOXApp_test app( true );
        app.setPowerMgtEnabled( true );
        app.configurePowerManagement( "pdu", "test" );

        REQUIRE( app.onPowerOff() == 0 );
        REQUIRE( app.whilePowerOff() == 0 );
        REQUIRE( app.powerOnWaitElapsed() == true );

        // Comes up unknown
        REQUIRE( app.powerState() == -1 );
        REQUIRE( app.powerStateTarget() == -1 );

        app.setPowerState( "Off", "Off" );
        REQUIRE( app.powerState() == 0 );
        REQUIRE( app.powerStateTarget() == 0 );

        app.setPowerState( "Int", "Int" );
        REQUIRE( app.powerState() == -1 );
        REQUIRE( app.powerStateTarget() == -1 );

        app.setPowerState( "Off", "On" );
        REQUIRE( app.powerState() == 0 );
        REQUIRE( app.powerStateTarget() == 1 );

        app.configurePowerOnWait( 10, 0, 1e9 );
        REQUIRE( app.loopPause() == 1e9 );

        // 10 checks, then true on 11th
        REQUIRE( app.powerOnWaitElapsed() == false );
        REQUIRE( app.powerOnWaitElapsed() == false );
        REQUIRE( app.powerOnWaitElapsed() == false );
        REQUIRE( app.powerOnWaitElapsed() == false );
        REQUIRE( app.powerOnWaitElapsed() == false );
        REQUIRE( app.powerOnWaitElapsed() == false );
        REQUIRE( app.powerOnWaitElapsed() == false );
        REQUIRE( app.powerOnWaitElapsed() == false );
        REQUIRE( app.powerOnWaitElapsed() == false );
        REQUIRE( app.powerOnWaitElapsed() == false );
        REQUIRE( app.powerOnWaitElapsed() == false );
        REQUIRE( app.powerOnWaitElapsed() == true );

        app.setPowerState( "On", "On" );
        REQUIRE( app.powerState() == 1 );
        REQUIRE( app.powerStateTarget() == 1 );

        app.setPowerState( "On", "Off" );
        REQUIRE( app.powerState() == 1 );
        REQUIRE( app.powerStateTarget() == 0 );

        app.setPowerState( "Off", "Off" );
        REQUIRE( app.powerState() == 0 );
        REQUIRE( app.powerStateTarget() == 0 );
    }
}

/// INDI preperty creation utilities
/**
 * \ingroup MagAOXApp_unit_test
 */
TEST_CASE( "INDI preperty creation utilities", "[app::MagAOXApp]" )
{
    SECTION( "createStandardIndiText" )
    {
        MagAOXApp_test app;
        app.setConfigName( "test" );

        pcf::IndiProperty ip;

        app.createStandardIndiText( ip, "tprop", "tlabel", "tgroup" );

        REQUIRE( ip.getType() == pcf::IndiProperty::Text );
        REQUIRE( ip.getDevice() == "test" );
        REQUIRE( ip.getName() == "tprop" );
        REQUIRE( ip.getPerm() == pcf::IndiProperty::ReadWrite );
        REQUIRE( ip.getState() == pcf::IndiProperty::Idle );
        REQUIRE( ip.find( "current" ) == true );
        REQUIRE( ip.find( "target" ) == true );
        REQUIRE( ip.getLabel() == "tlabel" );
        REQUIRE( ip.getGroup() == "tgroup" );
    }

    SECTION( "createROIndiText" )
    {
        MagAOXApp_test app;
        app.setConfigName( "test" );

        pcf::IndiProperty ip;

        app.createROIndiText( ip, "tprop", "tel", "tlabel", "tgroup", "ellabel" );

        REQUIRE( ip.getType() == pcf::IndiProperty::Text );
        REQUIRE( ip.getDevice() == "test" );
        REQUIRE( ip.getName() == "tprop" );
        REQUIRE( ip.getPerm() == pcf::IndiProperty::ReadOnly );
        REQUIRE( ip.getState() == pcf::IndiProperty::Idle );
        REQUIRE( ip.getLabel() == "tlabel" );
        REQUIRE( ip.getGroup() == "tgroup" );

        REQUIRE( ip.find( "tel" ) == true );
        REQUIRE( ip["tel"].getLabel() == "ellabel" );
    }

    SECTION( "createStandardIndiNumber" )
    {
        MagAOXApp_test app;
        app.setConfigName( "test" );

        pcf::IndiProperty ip;

        app.createStandardIndiNumber<double>( ip, "tprop", 0.001, 1, 0.002, "%0.23g", "tlabel", "tgroup" );

        REQUIRE( ip.getType() == pcf::IndiProperty::Number );
        REQUIRE( ip.getDevice() == "test" );
        REQUIRE( ip.getName() == "tprop" );
        REQUIRE( ip.getPerm() == pcf::IndiProperty::ReadWrite );
        REQUIRE( ip.getState() == pcf::IndiProperty::Idle );

        REQUIRE( ip.find( "current" ) == true );
        REQUIRE( ip["current"].getMin() == "0.001" );
        REQUIRE( ip["current"].getMax() == "1" );
        REQUIRE( ip["current"].getStep() == "0.002" );
        REQUIRE( ip["current"].getFormat() == "%0.23g" );

        REQUIRE( ip.find( "target" ) == true );
        REQUIRE( ip["target"].getMin() == "0.001" );
        REQUIRE( ip["target"].getMax() == "1" );
        REQUIRE( ip["target"].getStep() == "0.002" );
        REQUIRE( ip["target"].getFormat() == "%0.23g" );

        REQUIRE( ip.getLabel() == "tlabel" );
        REQUIRE( ip.getGroup() == "tgroup" );
    }

    SECTION( "createROIndiNumber" )
    {
        MagAOXApp_test app;
        app.setConfigName( "test" );

        pcf::IndiProperty ip;

        app.createROIndiNumber( ip, "tprop", "tlabel", "tgroup" );

        REQUIRE( ip.getType() == pcf::IndiProperty::Number );
        REQUIRE( ip.getDevice() == "test" );
        REQUIRE( ip.getName() == "tprop" );
        REQUIRE( ip.getPerm() == pcf::IndiProperty::ReadOnly );
        REQUIRE( ip.getState() == pcf::IndiProperty::Idle );
        REQUIRE( ip.getLabel() == "tlabel" );
        REQUIRE( ip.getGroup() == "tgroup" );
    }

    SECTION( "createStandardIndiToggleSw" )
    {
        MagAOXApp_test app;
        app.setConfigName( "testz" );

        pcf::IndiProperty ip;

        app.createStandardIndiToggleSw( ip, "tpropz", "tlabelz", "tgroupz" );

        REQUIRE( ip.getType() == pcf::IndiProperty::Switch );
        REQUIRE( ip.getDevice() == "testz" );
        REQUIRE( ip.getName() == "tpropz" );
        REQUIRE( ip.getPerm() == pcf::IndiProperty::ReadWrite );
        REQUIRE( ip.getState() == pcf::IndiProperty::Idle );
        REQUIRE( ip.getRule() == pcf::IndiProperty::AtMostOne );

        REQUIRE( ip.getNumElements() == 1 );
        REQUIRE( ip.find( "toggle" ) == true );
        REQUIRE( ip["toggle"].getSwitchState() == pcf::IndiElement::Off );
        REQUIRE( ip.getLabel() == "tlabelz" );
        REQUIRE( ip.getGroup() == "tgroupz" );
    }

    SECTION( "createStandardIndiRequestSw" )
    {
        MagAOXApp_test app;
        app.setConfigName( "testz" );

        pcf::IndiProperty ip;

        app.createStandardIndiRequestSw( ip, "tpropz", "tlabelz", "tgroupz" );

        REQUIRE( ip.getType() == pcf::IndiProperty::Switch );
        REQUIRE( ip.getDevice() == "testz" );
        REQUIRE( ip.getName() == "tpropz" );
        REQUIRE( ip.getPerm() == pcf::IndiProperty::ReadWrite );
        REQUIRE( ip.getState() == pcf::IndiProperty::Idle );
        REQUIRE( ip.getRule() == pcf::IndiProperty::AtMostOne );

        REQUIRE( ip.getNumElements() == 1 );
        REQUIRE( ip.find( "request" ) == true );
        REQUIRE( ip["request"].getSwitchState() == pcf::IndiElement::Off );
        REQUIRE( ip.getLabel() == "tlabelz" );
        REQUIRE( ip.getGroup() == "tgroupz" );
    }

    SECTION( "createStandardIndiSelectionSw, w/ labels" )
    {
        MagAOXApp_test app;
        app.setConfigName( "testy" );

        pcf::IndiProperty ip;

        std::vector<std::string> els( { "el1", "el2", "el3" } );
        std::vector<std::string> labs( { "l1", "", "l3" } );

        app.createStandardIndiSelectionSw( ip, "tpropy", els, labs, "tlabely", "tgroupy" );

        REQUIRE( ip.getType() == pcf::IndiProperty::Switch );
        REQUIRE( ip.getDevice() == "testy" );
        REQUIRE( ip.getName() == "tpropy" );
        REQUIRE( ip.getPerm() == pcf::IndiProperty::ReadWrite );
        REQUIRE( ip.getState() == pcf::IndiProperty::Idle );
        REQUIRE( ip.getRule() == pcf::IndiProperty::OneOfMany );

        REQUIRE( ip.getNumElements() == 3 );
        REQUIRE( ip.find( "el1" ) == true );
        REQUIRE( ip["el1"].getSwitchState() == pcf::IndiElement::Off );
        REQUIRE( ip["el1"].getLabel() == "l1" );

        REQUIRE( ip.find( "el2" ) == true );
        REQUIRE( ip["el2"].getSwitchState() == pcf::IndiElement::Off );
        REQUIRE( ip["el2"].getLabel() == "" );

        REQUIRE( ip.find( "el3" ) == true );
        REQUIRE( ip["el3"].getSwitchState() == pcf::IndiElement::Off );
        REQUIRE( ip["el3"].getLabel() == "l3" );

        REQUIRE( ip.getLabel() == "tlabely" );
        REQUIRE( ip.getGroup() == "tgroupy" );
    }

    SECTION( "createStandardIndiSelectionSw, no labels" )
    {
        MagAOXApp_test app;
        app.setConfigName( "testy" );

        pcf::IndiProperty ip;

        std::vector<std::string> els( { "el1", "el2", "el3" } );

        app.createStandardIndiSelectionSw( ip, "tpropy", els, "tlabely", "tgroupy" );

        REQUIRE( ip.getType() == pcf::IndiProperty::Switch );
        REQUIRE( ip.getDevice() == "testy" );
        REQUIRE( ip.getName() == "tpropy" );
        REQUIRE( ip.getPerm() == pcf::IndiProperty::ReadWrite );
        REQUIRE( ip.getState() == pcf::IndiProperty::Idle );
        REQUIRE( ip.getRule() == pcf::IndiProperty::OneOfMany );

        REQUIRE( ip.getNumElements() == 3 );
        REQUIRE( ip.find( "el1" ) == true );
        REQUIRE( ip["el1"].getSwitchState() == pcf::IndiElement::Off );
        REQUIRE( ip["el1"].getLabel() == "el1" );

        REQUIRE( ip.find( "el2" ) == true );
        REQUIRE( ip["el2"].getSwitchState() == pcf::IndiElement::Off );
        REQUIRE( ip["el2"].getLabel() == "el2" );

        REQUIRE( ip.find( "el3" ) == true );
        REQUIRE( ip["el3"].getSwitchState() == pcf::IndiElement::Off );
        REQUIRE( ip["el3"].getLabel() == "el3" );

        REQUIRE( ip.getLabel() == "tlabely" );
        REQUIRE( ip.getGroup() == "tgroupy" );
    }
}

/// Signal Handlers
/**
 * \ingroup MagAOXApp_unit_test
 */
TEST_CASE( "Signal Handlers", "[app::MagAOXApp]" )
{
    SECTION( "Setting and calling signal handler: SIGTERM" )
    {

        MagAOXApp_test app;

        // this is just to touch this function
        REQUIRE( app.setSigTermHandler() == 0 );

        REQUIRE( app.shutdown() == 0 );
        app._handlerSigTerm( SIGTERM, nullptr, nullptr );
        REQUIRE( app.shutdown() == 1 );
    }

    SECTION( "Setting and calling signal handler: SIGINT" )
    {

        MagAOXApp_test app;

        // this is just to touch this function
        REQUIRE( app.setSigTermHandler() == 0 );

        REQUIRE( app.shutdown() == 0 );
        app._handlerSigTerm( SIGINT, nullptr, nullptr );
        REQUIRE( app.shutdown() == 1 );
    }

    SECTION( "Setting and calling signal handler: SIGQUIT" )
    {

        MagAOXApp_test app;

        // this is just to touch this function
        REQUIRE( app.setSigTermHandler() == 0 );

        REQUIRE( app.shutdown() == 0 );
        app._handlerSigTerm( SIGQUIT, nullptr, nullptr );
        REQUIRE( app.shutdown() == 1 );
    }

    SECTION( "Setting and calling signal handler: SIGHUP" )
    {

        MagAOXApp_test app;

        // this is just to touch this function
        REQUIRE( app.setSigTermHandler() == 0 );

        REQUIRE( app.shutdown() == 0 );
        app._handlerSigTerm( SIGHUP, nullptr, nullptr );
        REQUIRE( app.shutdown() == 1 );
    }
}

/// Setting Euid
/**
 * \ingroup MagAOXApp_unit_test
 */
TEST_CASE( "Setting Euid", "[app::MagAOXApp]" )
{

    MagAOXApp_test app;

    REQUIRE( app.setEuidReal() == 0 );

    REQUIRE( app.setEuidCalled() == 0 );

    REQUIRE( app.setEuidReal( 0 ) == -1 );
    REQUIRE( app.setEuidCalled( 0 ) == -1 );
}



/// Tests of utilities in cpp
/**
 * \ingroup MagAOXApp_unit_test
 */
TEST_CASE( "Tests of utilities in cpp", "[app::MagAOXApp]" )
{
    SECTION( "sigusr1 handler" )
    {
        // this is just to touch this function
        MagAOX::app::sigUsr1Handler( 0, nullptr, nullptr );

        REQUIRE( true );
    }
}

} // namespace MagAOXAppTest
} // namespace appTest
} // namespace libXWCTest
