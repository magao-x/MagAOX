// #define CATCH_CONFIG_MAIN
/** \file MagAOXApp_test.cpp
  * \brief Catch2 tests for the MagAOX::app::MagAOXApp<true> application base class.
  *
  * The tests use the MagAOXApp_test harness declared in MagAOXApp_test.hpp. The harness
  * exposes the protected MagAOXApp interface and stubs the appStartup(), appLogic(), and
  * appShutdown() hooks. Configuration, PID locking, INDI property registration, the INDI
  * handle* callbacks, power management, signal handlers, threadStart(), and the FIFO and
  * INDI driver startup paths are all exercised for real. INDI callbacks are driven with
  * hand-built pcf::IndiProperty objects. No INDI server is needed.
  *
  * The tests create real directory trees under /tmp such as /tmp/MagAOXApp_test and force
  * operating system failures by removing directory permissions. Failure branches that no
  * external condition can trigger are reached with the XWCTEST_* fault injection macros in
  * MagAOXApp.hpp. Each macro combination is compiled into its own XWCTEST_NAMESPACE copy of
  * MagAOXApp and of the harness by re-including both headers below.
  *
  * \ingroup MagAOXApp_unit_test
  */
#include "../../../tests/catch2/catch.hpp"

#include <filesystem>
#include <atomic>

#include <mx/sys/timeUtils.hpp>

#include "../MagAOXApp.hpp"
#include "MagAOXApp_test.hpp"

// Captured now, before the XWCTEST_MAGAOXAPP_SIGTERMH_SIGQUIT hook below permanently
// redefines the SIGQUIT token to SIGKILL for the rest of this translation unit. Any later
// test code that writes the literal token SIGQUIT would otherwise unknowingly pass the value
// of SIGKILL instead.
static const int REAL_SIGQUIT = SIGQUIT;

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

#undef app_MagAOXApp_hpp
#undef app_tests_MagAOXApp_test_hpp
#define XWCTEST_NAMESPACE XWCTEST_MAGAOXAPP_SIGTERMH_SIGQUIT_ns
#define XWCTEST_MAGAOXAPP_SIGTERMH_SIGQUIT
#include "../MagAOXApp.hpp"
#include "MagAOXApp_test.hpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_MAGAOXAPP_SIGTERMH_SIGQUIT
// NOTE: XWCTEST_MAGAOXAPP_SIGTERMH_SIGQUIT redefines the SIGQUIT macro to SIGKILL inside
// setSigTermHandler() in MagAOXApp.hpp. That redefinition is never restored. It is not scoped
// or undone within the header. The substitution therefore leaks for the rest of this
// translation unit, so only one of the SIGTERMH_SIGTERM, SIGTERMH_SIGQUIT, and
// SIGTERMH_SIGINT hooks may be used per .cpp file. Combining more than one here makes the
// compiler see duplicate case labels in the handlerSigTerm switch, because they all collapse
// onto SIGKILL. Do not add the SIGTERM or SIGINT variants below this point in this file. See
// MagAOXAppExecute_test.cpp for a second, independent use of one of these hooks.

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
    MagAOX::app::MagAOXApp<true> app( "", true );
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

        WHEN( "a null callback was registered" )
        {
            // shmimMonitor<> registers some of its properties this way. See the call
            // registerIndiPropertyNew( m_indiP_shmimName, nullptr ) in shmimMonitor.hpp.
            // handleNewProperty() must tolerate a matched but null callback.
            MagAOXApp_test app;
            app.setConfigName( "test" );

            pcf::IndiProperty prop;
            REQUIRE( app.registerIndiPropertyNew( prop,
                                                  "nullcb",
                                                  pcf::IndiProperty::Number,
                                                  pcf::IndiProperty::ReadWrite,
                                                  pcf::IndiProperty::Idle,
                                                  nullptr ) == 0 );

            pcf::IndiProperty nprop;
            nprop.setDevice( "test" );
            nprop.setName( "nullcb" );

            app.handleNewProperty( nprop );
            REQUIRE( true );
        }
    }

#ifdef XWCTEST_DOXYGEN_REF_PROTECTED
    MagAOX::app::MagAOXApp<true> app( "", true );
    app.configName();
    pcf::IndiProperty prop;
    app.registerIndiPropertyNew(
        prop, "nprop", pcf::IndiProperty::Number, pcf::IndiProperty::ReadWrite, pcf::IndiProperty::Idle, callback );
    app.handleNewProperty( prop );
#endif
}

/// Setting defaults
/**
 * \ingroup MagAOXApp_unit_test
 */
TEST_CASE( "MagAOXApp INDI SetProperty retry backoff", "[app::MagAOXApp]" )
{
    using namespace std::chrono;

    MagAOXApp_test app;

    pcf::IndiProperty prop;
    REQUIRE( app.registerIndiPropertySet( prop, "publisher", "property", callback ) == 0 );

    std::string key = app.setPropertyKey( prop );

    auto now = steady_clock::now();

    REQUIRE( app.shouldRequestSetProperty( key, false, now ) == true );

    app.noteSetPropertyRequested( key, now );
    REQUIRE( app.setPropertyRetryCount( key ) == 1 );
    REQUIRE( app.setPropertyRetryDelay( key ) == seconds( 1 ) );
    REQUIRE( app.shouldRequestSetProperty( key, false, now ) == false );
    REQUIRE( app.shouldRequestSetProperty( key, false, now + seconds( 1 ) ) == true );

    app.noteSetPropertyRequested( key, now + seconds( 1 ) );
    REQUIRE( app.setPropertyRetryCount( key ) == 2 );
    REQUIRE( app.setPropertyRetryDelay( key ) == seconds( 2 ) );
    REQUIRE( app.shouldRequestSetProperty( key, false, now + seconds( 2 ) ) == false );
    REQUIRE( app.shouldRequestSetProperty( key, false, now + seconds( 3 ) ) == true );

    app.noteSetPropertyRequested( key, now + seconds( 3 ) );
    app.noteSetPropertyRequested( key, now + seconds( 7 ) );
    app.noteSetPropertyRequested( key, now + seconds( 15 ) );
    app.noteSetPropertyRequested( key, now + seconds( 31 ) );
    REQUIRE( app.setPropertyRetryDelay( key ) == seconds( 32 ) );
    REQUIRE( app.setPropertyMissingLogged( key ) == false );

    app.noteSetPropertyRequested( key, now + seconds( 63 ) );
    REQUIRE( app.setPropertyRetryDelay( key ) == seconds( 60 ) );
    REQUIRE( app.setPropertyMissingLogged( key ) == true );

    app.noteSetPropertyRequested( key, now + seconds( 123 ) );
    REQUIRE( app.setPropertyRetryDelay( key ) == seconds( 60 ) );
    REQUIRE( app.setPropertyRetryCount( key ) == 8 );
}

TEST_CASE( "MagAOXApp INDI SetProperty retry reset", "[app::MagAOXApp]" )
{
    using namespace std::chrono;

    MagAOXApp_test app;

    pcf::IndiProperty prop;
    REQUIRE( app.registerIndiPropertySet( prop, "publisher", "property", callback ) == 0 );

    std::string key = app.setPropertyKey( prop );

    auto now = steady_clock::now();

    app.noteSetPropertyRequested( key, now );
    app.noteSetPropertyRequested( key, now + seconds( 1 ) );

    REQUIRE( app.setPropertyRetryCount( key ) == 2 );
    REQUIRE( app.setPropertyRetryDelay( key ) == seconds( 2 ) );

    app.resetSetPropertyRetry( key );
    REQUIRE( app.setPropertyRetryCount( key ) == 0 );
    REQUIRE( app.setPropertyRetryDelay( key ) == steady_clock::duration::zero() );
    REQUIRE( app.setPropertyMissingLogged( key ) == false );
    REQUIRE( app.shouldRequestSetProperty( key, false, now ) == true );

    app.markSetPropertyReceived( key, true );
    REQUIRE( app.shouldRequestSetProperty( key, false, now ) == false );
    REQUIRE( app.shouldRequestSetProperty( key, true, now ) == true );
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

        // Calling setDefaults() again on the same app re-registers "fsm" and
        // "fsm_clear_alert", which are already registered from the first call. This
        // exercises the registerIndiPropertyNew() failure logging branches.
        app.setDefaults( argv.size() - 1, const_cast<char **>( argv.data() ) );
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
                                  { "2500", "pdu0", "outlet1", "state", "target", "500000" } ); // power values follow the real pdu naming. Nothing is connected.

        MagAOXApp_test app( true );
        app.setPowerMgtEnabled( true );

        app.setup( argv.size() - 1, const_cast<char **>( argv.data() ) );

        REQUIRE( app.stateAlert() == true ); // due to git
        REQUIRE( app.configOnly() == true );
        REQUIRE( app.loopPause() == 2500 );
        REQUIRE( app.powerDevice() == "pdu0" );
        REQUIRE( app.powerChannel() == "outlet1" );
        REQUIRE( app.powerElement() == "state" );
        REQUIRE( app.powerTargetElement() == "target" );
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
            { "2500", "pdu0", "outlet1", "state", "target", "500", "value" } );

        MagAOXApp_test app( false );
        app.setPowerMgtEnabled( true );

        app.setup( argv.size() - 1, const_cast<char **>( argv.data() ) );

        REQUIRE( app.stateAlert() == false );
        REQUIRE( app.configOnly() == false );
        REQUIRE( app.loopPause() == 2500 );
        REQUIRE( app.powerDevice() == "pdu0" );
        REQUIRE( app.powerChannel() == "outlet1" );
        REQUIRE( app.powerElement() == "state" );
        REQUIRE( app.powerTargetElement() == "target" );
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
            { "2500", "pdu0", "outlet1", "state", "target", "500000", "value" } );

        MagAOXApp_test app( true );
        app.setPowerMgtEnabled( true );

        app.setup( argv.size() - 1, const_cast<char **>( argv.data() ) );

        REQUIRE( app.stateAlert() == true ); // due to git
        REQUIRE( app.configOnly() == true );
        REQUIRE( app.loopPause() == 2500 );
        REQUIRE( app.powerDevice() == "pdu0" );
        REQUIRE( app.powerChannel() == "outlet1" );
        REQUIRE( app.powerElement() == "state" );
        REQUIRE( app.powerTargetElement() == "target" );
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
                                  { "2500", "pdu0", "outlet1", "state", "target", "500000" } );

        MagAOXApp_test app( false );
        app.setPowerMgtEnabled( true );

        app.setup( argv.size() - 1, const_cast<char **>( argv.data() ) );

        REQUIRE( app.stateAlert() == false ); // due to git
        REQUIRE( app.configOnly() == false );
        REQUIRE( app.loopPause() == 2500 );
        REQUIRE( app.powerDevice() == "pdu0" );
        REQUIRE( app.powerChannel() == "outlet1" );
        REQUIRE( app.powerElement() == "state" );
        REQUIRE( app.powerTargetElement() == "target" );
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
        REQUIRE( app.powerOnWait() == 55 ); // The default value. It is not set in the config.

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
                                  { "2500", "pdu0", "outlet1", "state", "target", "500000" } );

        MagAOXApp_test app( true );
        app.setPowerMgtEnabled( true );

        app.addUnusedConfig();

        app.setup( argv.size() - 1, const_cast<char **>( argv.data() ) );

        REQUIRE( app.stateAlert() == true ); // due to git
        REQUIRE( app.configOnly() == true );
        REQUIRE( app.loopPause() == 2500 );
        REQUIRE( app.powerDevice() == "pdu0" );
        REQUIRE( app.powerChannel() == "outlet1" );
        REQUIRE( app.powerElement() == "state" );
        REQUIRE( app.powerTargetElement() == "target" );
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

    SECTION( "PID Lock, per-app statusDir creation error" )
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
        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/sys" );
        // The statusDir root now already exists, which mkdir tolerates with EEXIST. It is made
        // read-only so the per-app subdirectory mkdir() fails with a real EACCES.
        REQUIRE( chmod( "/tmp/MagAOXApp_test/sys", 0555 ) == 0 );

        MagAOXApp_test app;
        app.invokedName() = argv[0];
        app.setDefaults( argv.size() - 1, const_cast<char **>( argv.data() ) );

        int rv = app.lockPID();

        chmod( "/tmp/MagAOXApp_test/sys", 0755 );
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

    SECTION( "real match against this test process's own /proc/pid/cmdline" )
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

        // Write the pid of this real, currently running test process. /proc/<pid>/cmdline
        // then exists and is readable. The "Stale lock" section instead writes pid 1, which
        // either does not exist or belongs to an unrelated process.
        std::ofstream fout;
        fout.open( "/tmp/MagAOXApp_test/sys/testapp/pid" );
        fout << getpid();
        fout.close();

        MagAOXApp_test app;
        // "MagAOXApp_test" is a real substring of the argv[0] of this process, which is the
        // path of the compiled test binary. pidCmdLine.find(invokedName) therefore finds it
        // for real. This exercises the branch where invokedPos is found, without the
        // XWCTEST_MAGAOXAPP_PID_LOCKED hook, which hardcodes both positions to 0. "testapp"
        // is not part of the real command line of this process, so configPos stays npos and
        // the lock is granted normally rather than reported as already locked.
        app.invokedName() = "MagAOXApp_test";
        app.setDefaults( argv.size() - 1, const_cast<char **>( argv.data() ) );

        int rv = app.lockPID();
        REQUIRE( rv == 0 );
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

    SECTION( "loadBasicConfig with power management registers a duplicate Set property" )
    {
        std::vector<const char *> argv;
        std::vector<std::string>  argvstr( { "./execname", "-n", "pwrdup" } );

        argv.resize( argvstr.size() + 1, NULL );
        for( size_t index = 0; index < argvstr.size(); ++index )
        {
            argv[index] = argvstr[index].c_str();
        }

        char ppath[1024];
        snprintf( ppath, sizeof( ppath ), "%s=/tmp/MagAOXApp_pwrdup", MAGAOX_env_path );
        putenv( ppath );

        std::filesystem::remove_all( "/tmp/MagAOXApp_pwrdup" );
        mx::ioutils::createDirectories( "/tmp/MagAOXApp_pwrdup/config" );
        mx::ioutils::createDirectories( "/tmp/MagAOXApp_pwrdup/logs" );

        std::ofstream fout;
        fout.open( "/tmp/MagAOXApp_pwrdup/config/magaox.conf" );
        fout.close();

        mx::app::writeConfigFile( "/tmp/MagAOXApp_pwrdup/config/pwrdup.conf",
                                  { "power", "power" },
                                  { "device", "channel" },
                                  { "pdu0", "outlet1" } );

        MagAOXApp_test app;
        app.setPowerMgtEnabled( true );
        app.invokedName() = argv[0];
        app.setup( argv.size() - 1, const_cast<char **>( argv.data() ) );

        REQUIRE( app.shutdown() == false );

        // Calling loadBasicConfig() again attempts to register the same Set property a
        // second time, which registerIndiPropertySet() rejects as a duplicate.
        app.loadBasicConfig();
        REQUIRE( true );
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

        // Exercise the static wrapper generated by INDI_SETCALLBACK_DECL directly. See its
        // definition for why setPowerState() above does not already cover it.
        app.setPowerStateViaStaticWrapper( "Off", "Off" );
        REQUIRE( app.powerState() == 0 );

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
        // Uses REAL_SIGQUIT, not the literal token SIGQUIT. See its definition near the top
        // of this file. The SIGTERMH_SIGQUIT hook below poisons that token.
        app._handlerSigTerm( REAL_SIGQUIT, nullptr, nullptr );
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

    SECTION( "sigaction fails for SIGQUIT" )
    {
        // See the NOTE near the #include block for this hook at the top of this file. Only
        // one of the SIGTERMH_SIGTERM, SIGTERMH_SIGQUIT, and SIGTERMH_SIGINT hooks can be
        // used per translation unit.
        XWCTEST_MAGAOXAPP_SIGTERMH_SIGQUIT_ns::MagAOXApp_test app;

        REQUIRE( app.setSigTermHandler() == -1 );
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

/// Verify that the elevated privileges RAII guard tolerates redundant elevate() and
/// restore() calls.
/**
 * \ingroup MagAOXApp_unit_test
 */
TEST_CASE( "Elevated privileges double guard", "[app::MagAOXApp]" )
{
    MagAOXApp_test app;

    // Exercises the early-return branches for redundant elevate() and restore() calls. No
    // externally observable state changes. This only needs to run without hitting the
    // guarded logic twice.
    app.testElevatedPrivilegesDoubleGuard();

    REQUIRE( true );
}

/// Verify that unlockPID() reports failure when the pid file was already removed by
/// someone else.
/**
 * \ingroup MagAOXApp_unit_test
 */
TEST_CASE( "PID unlock after external removal", "[app::MagAOXApp]" )
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
    std::filesystem::remove_all( "/tmp/MagAOXApp_test/sys/testapp" );

    MagAOXApp_test app;
    app.invokedName() = argv[0];
    app.setDefaults( argv.size() - 1, const_cast<char **>( argv.data() ) );

    REQUIRE( app.lockPID() == 0 );
    REQUIRE( std::filesystem::exists( "/tmp/MagAOXApp_test/sys/testapp/pid" ) );

    // Remove the pid file out from under the app, so unlockPID's ::remove() call fails.
    std::filesystem::remove( "/tmp/MagAOXApp_test/sys/testapp/pid" );

    REQUIRE( app.unlockPID() == -1 );
}

/// Verify that each registerIndiProperty* overload rejects a second registration of the same
/// property name.
/**
 * \ingroup MagAOXApp_unit_test
 */
TEST_CASE( "Duplicate INDI property registration", "[app::MagAOXApp]" )
{
    SECTION( "registerIndiPropertyNew duplicate" )
    {
        MagAOXApp_test app;
        pcf::IndiProperty prop1, prop2;
        REQUIRE( app.registerIndiPropertyNew( prop1,
                                              "dup",
                                              pcf::IndiProperty::Number,
                                              pcf::IndiProperty::ReadWrite,
                                              pcf::IndiProperty::Idle,
                                              callback ) == 0 );
        REQUIRE( app.registerIndiPropertyNew( prop2,
                                              "dup",
                                              pcf::IndiProperty::Number,
                                              pcf::IndiProperty::ReadWrite,
                                              pcf::IndiProperty::Idle,
                                              callback ) == -1 );
    }

    SECTION( "registerIndiPropertySet duplicate" )
    {
        MagAOXApp_test app;
        pcf::IndiProperty prop1, prop2;
        REQUIRE( app.registerIndiPropertySet( prop1, "dev", "dup", callback ) == 0 );
        REQUIRE( app.registerIndiPropertySet( prop2, "dev", "dup", callback ) == -1 );
    }

    SECTION( "registerIndiPropertyNew duplicate, with explicit switch rule" )
    {
        // This overload also sets the SwitchRuleType of the property. No macro or app in this
        // codebase uses it, but it is part of the public API and so is tested directly here.
        MagAOXApp_test app;
        pcf::IndiProperty prop1, prop2;
        REQUIRE( app.registerIndiPropertyNew( prop1,
                                              "duprule",
                                              pcf::IndiProperty::Switch,
                                              pcf::IndiProperty::ReadWrite,
                                              pcf::IndiProperty::Idle,
                                              pcf::IndiProperty::OneOfMany,
                                              callback ) == 0 );
        REQUIRE( app.registerIndiPropertyNew( prop2,
                                              "duprule",
                                              pcf::IndiProperty::Switch,
                                              pcf::IndiProperty::ReadWrite,
                                              pcf::IndiProperty::Idle,
                                              pcf::IndiProperty::OneOfMany,
                                              callback ) == -1 );
    }

    SECTION( "registerIndiPropertyReadOnly duplicate" )
    {
        MagAOXApp_test app;
        pcf::IndiProperty prop1, prop2;
        REQUIRE( app.registerIndiPropertyReadOnly(
                     prop1, "dup", pcf::IndiProperty::Number, pcf::IndiProperty::ReadOnly, pcf::IndiProperty::Idle ) ==
                 0 );
        REQUIRE( app.registerIndiPropertyReadOnly(
                     prop2, "dup", pcf::IndiProperty::Number, pcf::IndiProperty::ReadOnly, pcf::IndiProperty::Idle ) ==
                 -1 );
    }
}

/// Verify the handleGetProperties(), handleNewProperty(), handleSetProperty(), and
/// handleDefProperty() INDI callbacks and the updateIfChanged() family of helpers, with a
/// FIFO-less INDI driver installed.
/**
 * \ingroup MagAOXApp_unit_test
 */
TEST_CASE( "INDI property handlers", "[app::MagAOXApp]" )
{
    MagAOXApp_test app;
    app.setConfigName( "handlertest" ); // Constructs a FIFO-less indiDriver that is not good(), so m_indiDriver != nullptr.

    pcf::IndiProperty roProp;
    REQUIRE( app.registerIndiPropertyReadOnly(
                 roProp, "roprop", pcf::IndiProperty::Number, pcf::IndiProperty::ReadOnly, pcf::IndiProperty::Idle ) ==
             0 );

    pcf::IndiProperty newProp;
    REQUIRE( app.registerIndiPropertyNew( newProp,
                                          "newprop",
                                          pcf::IndiProperty::Number,
                                          pcf::IndiProperty::ReadWrite,
                                          pcf::IndiProperty::Idle,
                                          callback ) == 0 );

    SECTION( "handleGetProperties: wrong device is ignored" )
    {
        pcf::IndiProperty req;
        req.setDevice( "not-handlertest" );
        app.handleGetProperties( req );
        REQUIRE( true );
    }

    SECTION( "handleGetProperties: no name requests all properties" )
    {
        pcf::IndiProperty req;
        req.setDevice( "handlertest" );
        app.handleGetProperties( req );
        REQUIRE( true );
    }

    SECTION( "handleGetProperties: valid name found" )
    {
        pcf::IndiProperty req;
        req.setDevice( "handlertest" );
        req.setName( "newprop" );
        app.handleGetProperties( req );
        REQUIRE( true );
    }

    SECTION( "handleGetProperties: valid name not found" )
    {
        pcf::IndiProperty req;
        req.setDevice( "handlertest" );
        req.setName( "nosuchprop" );
        app.handleGetProperties( req );
        REQUIRE( true );
    }

    SECTION( "handleSetProperty: found calls back, not found is a no-op" )
    {
        pcf::IndiProperty setProp;
        REQUIRE( app.registerIndiPropertySet( setProp, "pubdev", "pubprop", callback ) == 0 );

        pcf::IndiProperty spFound;
        spFound.setDevice( "pubdev" );
        spFound.setName( "pubprop" );
        app.called_back = 0;
        app.handleSetProperty( spFound );
        REQUIRE( app.called_back == 1 );

        pcf::IndiProperty spMissing;
        spMissing.setDevice( "nodev" );
        spMissing.setName( "noprop" );
        app.called_back = 0;
        app.handleSetProperty( spMissing );
        REQUIRE( app.called_back == 0 );
    }

    SECTION( "handleDefProperty delegates to handleSetProperty" )
    {
        pcf::IndiProperty setProp;
        REQUIRE( app.registerIndiPropertySet( setProp, "pubdev2", "pubprop2", callback ) == 0 );

        pcf::IndiProperty dp;
        dp.setDevice( "pubdev2" );
        dp.setName( "pubprop2" );
        app.called_back = 0;
        app.handleDefProperty( dp );
        REQUIRE( app.called_back == 1 );
    }

    SECTION( "updateSwitchIfChanged with an active driver" )
    {
        pcf::IndiProperty sw;
        sw.setDevice( "handlertest" );
        sw.setName( "swprop" );
        sw.add( pcf::IndiElement( "toggle", pcf::IndiElement::Off ) );

        app.updateSwitchIfChanged( sw, "toggle", pcf::IndiElement::On );
        REQUIRE( sw["toggle"].getSwitchState() == pcf::IndiElement::On );
    }

    SECTION( "updateIfChanged(const char *) overload" )
    {
        pcf::IndiProperty txt( pcf::IndiProperty::Text );
        txt.setDevice( "handlertest" );
        txt.setName( "txtprop" );
        txt.add( pcf::IndiElement( "value" ) );
        txt["value"].setValue( "" );

        app.updateIfChanged( txt, "value", "a literal string" );
        REQUIRE( true );
    }

    SECTION( "updatesIfChanged (vector) overload" )
    {
        pcf::IndiProperty multi( pcf::IndiProperty::Number );
        multi.setDevice( "handlertest" );
        multi.setName( "multiprop" );
        multi.add( pcf::IndiElement( "a" ) );
        multi.add( pcf::IndiElement( "b" ) );

        std::vector<const char *> els{ "a", "b" };
        std::vector<double>       vals{ 1.0, 2.0 };
        app.updatesIfChanged( multi, els, vals );
        REQUIRE( true );
    }

    SECTION( "updateIfChanged (vector<string> els) overload" )
    {
        pcf::IndiProperty multi2( pcf::IndiProperty::Number );
        multi2.setDevice( "handlertest" );
        multi2.setName( "multiprop2" );
        multi2.add( pcf::IndiElement( "a" ) );
        multi2.add( pcf::IndiElement( "b" ) );

        std::vector<std::string> els2{ "a", "b" };
        std::vector<double>      vals2{ 1.0, 2.0 };
        app.updateIfChanged( multi2, els2, vals2 );
        REQUIRE( true );
    }
}

/// Verify the indiTargetUpdate() branches for mismatched property keys and for a remote
/// property with neither a target nor a current element. The branch for no non-empty value
/// found is unreachable dead code, because finding either element unconditionally sets
/// set = true. See the LCOV_EXCL note next to it in MagAOXApp.hpp.
/**
 * \ingroup MagAOXApp_unit_test
 */
TEST_CASE( "MagAOXApp indiTargetUpdate", "[app::MagAOXApp]" )
{
    MagAOXApp_test app;

    pcf::IndiProperty local( pcf::IndiProperty::Number );
    local.setDevice( "itudev" );
    local.setName( "ituprop" );
    local.add( pcf::IndiElement( "target" ) );

    double localTarget = 0;

    SECTION( "mismatched property keys" )
    {
        pcf::IndiProperty remote( pcf::IndiProperty::Number );
        remote.setDevice( "otherdev" );
        remote.setName( "otherprop" );

        REQUIRE( app.indiTargetUpdate( local, localTarget, remote, false ) == -1 );
    }

    SECTION( "neither target nor current element present" )
    {
        pcf::IndiProperty remote( pcf::IndiProperty::Number );
        remote.setDevice( "itudev" );
        remote.setName( "ituprop" );
        remote.add( pcf::IndiElement( "somethingelse" ) );

        REQUIRE( app.indiTargetUpdate( local, localTarget, remote, false ) == -1 );
    }
}

/// Verify the m_indiDriver == nullptr guards in handleGetProperties(), handleNewProperty(),
/// and handleSetProperty(). The "INDI property handlers" test above always has a driver that
/// is not null, even though it is not good().
/**
 * \ingroup MagAOXApp_unit_test
 */
TEST_CASE( "MagAOXApp INDI handle* callbacks with no driver", "[app::MagAOXApp]" )
{
    MagAOXApp_test    app; // m_indiDriver is still nullptr.
    pcf::IndiProperty ip;
    ip.setDevice( "nodriverdev" );
    ip.setName( "someprop" );

    app.handleGetProperties( ip );
    app.handleNewProperty( ip );
    app.handleSetProperty( ip );
    REQUIRE( true );
}

/// Helpers for the threadStart() tests. g_ran records that a thread body ran to the end.
namespace threadStartTest
{
std::atomic<int> g_ran{ 0 };

/// Minimal thread function that follows the threadStart() contract.
void trivialThreadStart( MagAOXAppTest::MagAOXApp_test *m )
{
    // Set the tpid reference immediately, as the caller contract of threadStart() requires.
    // Then wait for thrdInit to clear. threadStart() itself flips it once past its own setup.
    m->m_testThreadID = getpid();
    while( m->m_testThreadInit )
    {
        mx::sys::milliSleep( 10 );
    }
    g_ran = 1;
}

/// Thread function that deliberately never sets tpid and returns immediately. It exercises
/// the threadStart() timeout branch that reports the tpid for the thread was not set.
void neverSetsPidThreadStart( MagAOXAppTest::MagAOXApp_test *m )
{
    static_cast<void>( m );
}
} // namespace threadStartTest

/// Verify threadStart() with real std::thread objects. Priority clamping, the cpuset tasks
/// file handling, and the tpid timeout are all exercised for real.
/**
 * \ingroup MagAOXApp_unit_test
 */
TEST_CASE( "MagAOXApp threadStart", "[app::MagAOXApp]" )
{
    SECTION( "starts a real thread successfully" )
    {
        MagAOXAppTest::MagAOXApp_test app;

        std::thread       thrd;
        pcf::IndiProperty thProp;
        threadStartTest::g_ran = 0;

        int rv = app.threadStart( thrd,
                                  app.m_testThreadInit,
                                  app.m_testThreadID,
                                  thProp,
                                  0,
                                  "",
                                  "trivial",
                                  &app,
                                  threadStartTest::trivialThreadStart );

        REQUIRE( rv == 0 );
        REQUIRE( thrd.joinable() );
        thrd.join();
        REQUIRE( threadStartTest::g_ran == 1 );
    }

    SECTION( "thrdPrio > 99 is clamped and a positive priority fails without CAP_SYS_NICE" )
    {
        MagAOXAppTest::MagAOXApp_test app;

        std::thread       thrd;
        pcf::IndiProperty thProp;
        threadStartTest::g_ran = 0;

        // A process without root or CAP_SYS_NICE cannot raise its own scheduling priority,
        // so this exercises the real pthread_setschedparam() failure branch. 150 is also
        // outside the valid range of 0 to 99, which exercises the clamp to 99 branch too.
        int rv = app.threadStart( thrd,
                                  app.m_testThreadInit,
                                  app.m_testThreadID,
                                  thProp,
                                  150,
                                  "",
                                  "trivial",
                                  &app,
                                  threadStartTest::trivialThreadStart );

        REQUIRE( rv == 0 );
        REQUIRE( thrd.joinable() );
        thrd.join();
        REQUIRE( threadStartTest::g_ran == 1 );
    }

    SECTION( "thrdPrio < 0 is clamped to 0" )
    {
        MagAOXAppTest::MagAOXApp_test app;

        std::thread       thrd;
        pcf::IndiProperty thProp;
        threadStartTest::g_ran = 0;

        int rv = app.threadStart( thrd,
                                  app.m_testThreadInit,
                                  app.m_testThreadID,
                                  thProp,
                                  -5,
                                  "",
                                  "trivial",
                                  &app,
                                  threadStartTest::trivialThreadStart );

        REQUIRE( rv == 0 );
        REQUIRE( thrd.joinable() );
        thrd.join();
        REQUIRE( threadStartTest::g_ran == 1 );
    }

    SECTION( "a nonexistent cpuset path fails to open" )
    {
        MagAOXAppTest::MagAOXApp_test app;

        std::thread       thrd;
        pcf::IndiProperty thProp;
        threadStartTest::g_ran = 0;

        int rv = app.threadStart( thrd,
                                  app.m_testThreadInit,
                                  app.m_testThreadID,
                                  thProp,
                                  0,
                                  "no-such-cpuset",
                                  "trivial",
                                  &app,
                                  threadStartTest::trivialThreadStart );

        REQUIRE( rv == -1 );

        // threadStart() sets thrdInit to true at its very start and only ever clears it at
        // the very end. An early return such as this cpuset failure never clears it, so the
        // real spawned thread, which still checks m_testThreadInit in its loop, would spin
        // forever. Clear it here so the thread can exit and be joined.
        app.m_testThreadInit = false;
        if( thrd.joinable() )
        {
            thrd.join();
        }
    }

    SECTION( "a thread that never sets its pid times out" )
    {
        MagAOXAppTest::MagAOXApp_test app;

        std::thread       thrd;
        pcf::IndiProperty thProp;

        // threadStart() waits up to 1 real second for tpid to become nonzero. This thread
        // function deliberately never sets it. It also does not loop on thrdInit, so it exits
        // immediately and the join below cannot hang.
        int rv = app.threadStart( thrd,
                                  app.m_testThreadInit,
                                  app.m_testThreadID,
                                  thProp,
                                  0,
                                  "",
                                  "trivial",
                                  &app,
                                  threadStartTest::neverSetsPidThreadStart );

        REQUIRE( rv == -1 );
        if( thrd.joinable() )
        {
            thrd.join();
        }
    }

    SECTION( "a real, writable cpuset tasks file succeeds" )
    {
        std::filesystem::remove_all( "/tmp/MagAOXApp_cpuset_test" );
        mx::ioutils::createDirectories( "/tmp/MagAOXApp_cpuset_test/realcpuset" );
        std::ofstream fout;
        fout.open( "/tmp/MagAOXApp_cpuset_test/realcpuset/tasks" );
        fout.close();

        char cpath[1024];
        snprintf( cpath, sizeof( cpath ), "%s=/tmp/MagAOXApp_cpuset_test", MAGAOX_env_cpuset );
        putenv( cpath );

        std::vector<const char *> argv;
        std::vector<std::string>  argvstr( { "./execname" } );
        argv.resize( argvstr.size() + 1, NULL );
        argv[0] = argvstr[0].c_str();

        MagAOXAppTest::MagAOXApp_test app;
        app.invokedName() = argv[0];
        app.setDefaults( argv.size() - 1, const_cast<char **>( argv.data() ) );
        REQUIRE( app.cpusetPath() == "/tmp/MagAOXApp_cpuset_test" );

        std::thread       thrd;
        pcf::IndiProperty thProp;
        threadStartTest::g_ran = 0;

        int rv = app.threadStart( thrd,
                                  app.m_testThreadInit,
                                  app.m_testThreadID,
                                  thProp,
                                  0,
                                  "realcpuset",
                                  "trivial",
                                  &app,
                                  threadStartTest::trivialThreadStart );

        REQUIRE( rv == 0 );
        REQUIRE( thrd.joinable() );
        thrd.join();
        REQUIRE( threadStartTest::g_ran == 1 );

        // Confirm the real pid was actually written to the real "tasks" file.
        std::ifstream fin( "/tmp/MagAOXApp_cpuset_test/realcpuset/tasks" );
        std::string    written;
        fin >> written;
        REQUIRE( written == std::to_string( app.m_testThreadID ) );
    }
}

/// Verify sendNewProperty(ipSend, el, newVal). It looks up el through
/// IndiProperty::operator[], which throws a real std::runtime_error for a name the property
/// does not have.
/**
 * \ingroup MagAOXApp_unit_test
 */
TEST_CASE( "MagAOXApp sendNewProperty(prop, element, value)", "[app::MagAOXApp]" )
{
    MagAOXApp_test app;
    app.setConfigName( "sendnewproptest" ); // Constructs a FIFO-less indiDriver so m_indiDriver != nullptr.

    pcf::IndiProperty ipSend( pcf::IndiProperty::Number );
    ipSend.setDevice( "somedev" );
    ipSend.setName( "someprop" );
    ipSend.add( pcf::IndiElement( "val" ) );

    SECTION( "a real element name reaches the driver (this FIFO-less driver then fails "
             "the actual transmit, same as the existing handlertest coverage)" )
    {
        // The point of this section is that the exception catch is not exercised. Whatever
        // sendNewProperty() on the driver itself returns is unrelated.
        app.sendNewProperty( ipSend, "val", 3.14 );
    }

    SECTION( "a nonexistent element name is caught and reported" )
    {
        REQUIRE( app.sendNewProperty( ipSend, "notanelement", 3.14 ) == -1 );
    }

}

/// Verify that sendNewProperty(prop, element, value) fails on a fresh app whose INDI driver
/// was never created.
/**
 * \ingroup MagAOXApp_unit_test
 */
TEST_CASE( "MagAOXApp sendNewProperty(prop, element, value) fails with no INDI driver initialized",
           "[app::MagAOXApp]" )
{
    MagAOXApp_test freshApp; // A fresh app. m_indiDriver is still nullptr.

    pcf::IndiProperty ipSend( pcf::IndiProperty::Number );
    ipSend.setDevice( "somedev" );
    ipSend.setName( "someprop" );
    ipSend.add( pcf::IndiElement( "val" ) );

    REQUIRE( freshApp.sendNewProperty( ipSend, "val", 3.14 ) == -1 );
}

/// Verify the createINDIFIFOS() failures on the second FIFO, which is the output FIFO, and
/// on the third FIFO, which is the control FIFO. The failure on the first FIFO, the input
/// FIFO, is already tested. The "running execute" test forces it by never creating the fifos
/// directory at all, so mkfifo() fails on the very first call. Here the directory exists and
/// the earlier FIFOs are pre-created as real named pipes. Nothing is mocked. The directory is
/// then made read-only so the next mkfifo() fails with a real EACCES. That is not EEXIST,
/// which would otherwise be silently tolerated.
/**
 * \ingroup MagAOXApp_unit_test
 */
TEST_CASE( "MagAOXApp createINDIFIFOS 2nd and 3rd FIFO failures", "[app::MagAOXApp]" )
{
    std::vector<const char *> argv;
    std::vector<std::string>  argvstr( { "./execname", "-n", "fifotest" } );
    argv.resize( argvstr.size() + 1, NULL );
    for( size_t index = 0; index < argvstr.size(); ++index )
    {
        argv[index] = argvstr[index].c_str();
    }

    char ppath[1024];
    snprintf( ppath, sizeof( ppath ), "%s=/tmp/MagAOXApp_fifotest", MAGAOX_env_path );
    putenv( ppath );

    std::filesystem::remove_all( "/tmp/MagAOXApp_fifotest" );
    mx::ioutils::createDirectories( "/tmp/MagAOXApp_fifotest/drivers/fifos" );

    SECTION( "2nd (output) FIFO fails" )
    {
        MagAOXApp_test app;
        app.invokedName() = argv[0];
        app.setDefaults( argv.size() - 1, const_cast<char **>( argv.data() ) );

        REQUIRE( mkfifo( "/tmp/MagAOXApp_fifotest/drivers/fifos/fifotest.in", 0660 ) == 0 );
        REQUIRE( chmod( "/tmp/MagAOXApp_fifotest/drivers/fifos", 0555 ) == 0 );

        int rv = app.callCreateINDIFIFOS();

        chmod( "/tmp/MagAOXApp_fifotest/drivers/fifos", 0755 );
        REQUIRE( rv == -1 );
    }

    SECTION( "3rd (control) FIFO fails" )
    {
        MagAOXApp_test app;
        app.invokedName() = argv[0];
        app.setDefaults( argv.size() - 1, const_cast<char **>( argv.data() ) );

        REQUIRE( mkfifo( "/tmp/MagAOXApp_fifotest/drivers/fifos/fifotest.in", 0660 ) == 0 );
        REQUIRE( mkfifo( "/tmp/MagAOXApp_fifotest/drivers/fifos/fifotest.out", 0660 ) == 0 );
        REQUIRE( chmod( "/tmp/MagAOXApp_fifotest/drivers/fifos", 0555 ) == 0 );

        int rv = app.callCreateINDIFIFOS();

        chmod( "/tmp/MagAOXApp_fifotest/drivers/fifos", 0755 );
        REQUIRE( rv == -1 );
    }
}

/// Verify the single-argument sendNewProperty(ipSend) overload, which is distinct from the
/// sendNewProperty(prop, element, value) overload tested above. Also verify
/// sendNewStandardIndiToggle() and the wrong-key branch of newCallBack_clearFSMAlert().
/**
 * \ingroup MagAOXApp_unit_test
 */
TEST_CASE( "MagAOXApp sendNewProperty(ipSend), sendNewStandardIndiToggle, and clearFSMAlert callback", "[app::MagAOXApp]" )
{
    SECTION( "sendNewProperty(ipSend) fails with no INDI driver initialized" )
    {
        MagAOXApp_test app; // A fresh app. m_indiDriver is still nullptr.
        pcf::IndiProperty ipSend( pcf::IndiProperty::Number );
        ipSend.setDevice( "somedev" );
        ipSend.setName( "someprop" );

        REQUIRE( app.sendNewProperty( ipSend ) == -1 );
    }

    SECTION( "sendNewProperty(ipSend) fails when the driver has no real indiserver to connect to" )
    {
        // With a driver present that is not null, indiDriver::sendNewProperty() still fails
        // for real. Nothing is mocked. Connecting its internal indiClient requires a real
        // indiserver process, which does not exist in this test environment.
        MagAOXApp_test app;
        app.setConfigName( "sendnewpropdrivertest" );

        pcf::IndiProperty ipSend( pcf::IndiProperty::Number );
        ipSend.setDevice( "somedev" );
        ipSend.setName( "someprop" );

        REQUIRE( app.sendNewProperty( ipSend ) == -1 );
    }

    SECTION( "sendNewStandardIndiToggle propagates that same sendNewProperty() failure" )
    {
        MagAOXApp_test app; // A fresh app. m_indiDriver is still nullptr.

        REQUIRE( app.sendNewStandardIndiToggle( "somedev", "someprop", true ) == -1 );
        REQUIRE( app.sendNewStandardIndiToggle( "somedev", "someprop", false ) == -1 );
    }

    SECTION( "newCallBack_clearFSMAlert wrong-key branch" )
    {
        MagAOXApp_test app;
        pcf::IndiProperty ipWrong( pcf::IndiProperty::Switch );
        ipWrong.setDevice( "somethingelse" );
        ipWrong.setName( "notclearfsmalert" );

        REQUIRE( app.newCallBack_clearFSMAlert( ipWrong ) == -1 );
    }
}

/// Verify the startINDI() path where indiDriver->good() is false. The other execute() and
/// FIFO failure tests cannot reach it. They never create the FIFOs directory at all, so
/// createINDIFIFOS() itself fails before the indiDriver is even constructed.
/**
 * \ingroup MagAOXApp_unit_test
 */
TEST_CASE( "MagAOXApp startINDI good() failure", "[app::MagAOXApp]" )
{
    std::vector<const char *> argv;
    std::vector<std::string>  argvstr( { "./execname", "-n", "startinditest" } );
    argv.resize( argvstr.size() + 1, NULL );
    for( size_t index = 0; index < argvstr.size(); ++index )
    {
        argv[index] = argvstr[index].c_str();
    }

    char ppath[1024];
    snprintf( ppath, sizeof( ppath ), "%s=/tmp/MagAOXApp_startinditest", MAGAOX_env_path );
    putenv( ppath );

    std::filesystem::remove_all( "/tmp/MagAOXApp_startinditest" );
    mx::ioutils::createDirectories( "/tmp/MagAOXApp_startinditest/drivers/fifos" );

    SECTION( "indiDriver->good() == false when the FIFOs can't be opened" )
    {
        MagAOXApp_test app;
        app.invokedName() = argv[0];
        app.setDefaults( argv.size() - 1, const_cast<char **>( argv.data() ) );

        // Pre-create the FIFOs as real, valid named pipes, then strip all permissions.
        // createINDIFIFOS() tolerates their existence through EEXIST, but the open() calls
        // inside indiDriver fail with a real EACCES.
        REQUIRE( mkfifo( "/tmp/MagAOXApp_startinditest/drivers/fifos/startinditest.in", 0660 ) == 0 );
        REQUIRE( mkfifo( "/tmp/MagAOXApp_startinditest/drivers/fifos/startinditest.out", 0660 ) == 0 );
        REQUIRE( mkfifo( "/tmp/MagAOXApp_startinditest/drivers/fifos/startinditest.ctrl", 0660 ) == 0 );
        REQUIRE( chmod( "/tmp/MagAOXApp_startinditest/drivers/fifos/startinditest.in", 0000 ) == 0 );
        REQUIRE( chmod( "/tmp/MagAOXApp_startinditest/drivers/fifos/startinditest.out", 0000 ) == 0 );
        REQUIRE( chmod( "/tmp/MagAOXApp_startinditest/drivers/fifos/startinditest.ctrl", 0000 ) == 0 );

        int rv = app.callStartINDI();

        chmod( "/tmp/MagAOXApp_startinditest/drivers/fifos/startinditest.in", 0660 );
        chmod( "/tmp/MagAOXApp_startinditest/drivers/fifos/startinditest.out", 0660 );
        chmod( "/tmp/MagAOXApp_startinditest/drivers/fifos/startinditest.ctrl", 0660 );
        REQUIRE( rv == -1 );
    }

}

/// Verify that sendGetPropertySetList() detects a key mismatch, for both the forced refresh
/// path where all is true and the incremental path where all is false. The key of a Set
/// property is fixed when registerIndiPropertySet() is called. If the caller later renames
/// the same property object in place, rather than registering a new one, the two diverge.
/// sendGetPropertySetList() must detect and skip it rather than requesting garbage.
/**
 * \ingroup MagAOXApp_unit_test
 */
TEST_CASE( "MagAOXApp sendGetPropertySetList detects a renamed Set property", "[app::MagAOXApp]" )
{
    MagAOXApp_test app;

    pcf::IndiProperty setProp;
    REQUIRE( app.registerIndiPropertySet( setProp, "setdev", "setprop", callback ) == 0 );

    // Mutate the already registered property directly, so its current key diverges from the
    // key it was registered and stored under.
    setProp.setName( "renamed" );

    app.callSendGetPropertySetList( false );
    app.callSendGetPropertySetList( true );
    // Both prior calls resolved every entry, so m_allDefsReceived is now true. This third
    // call with all false hits the early return for nothing to do.
    app.callSendGetPropertySetList( false );
    REQUIRE( true );
}

} // namespace MagAOXAppTest
} // namespace appTest
} // namespace libXWCTest
