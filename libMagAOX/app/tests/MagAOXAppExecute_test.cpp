// #define CATCH_CONFIG_MAIN
/** \file MagAOXAppExecute_test.cpp
  * \brief Catch2 tests for MagAOXApp<true>::execute() and its failure paths.
  *
  * Each section runs the real execute() main loop on the MagAOXApp_test harness declared in
  * MagAOXApp_test.hpp. The tests create a real MagAO-X directory tree under
  * /tmp/MagAOXApp_test with a config directory, a logs directory, and a FIFO directory, and
  * point the MagAO-X path environment variable at it.
  *
  * Failure branches that no external condition can trigger are reached with the XWCTEST_*
  * fault injection macros in MagAOXApp.hpp. Each combination of macros is compiled into its
  * own XWCTEST_NAMESPACE copy of MagAOXApp and of the harness by re-including both headers.
  * The NOTE comments below explain why every test that starts the log thread needs its own
  * namespace, and why only one signal handler hook may be used in this file.
  *
  * \ingroup MagAOXApp_unit_test
  */
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

#undef app_MagAOXApp_hpp
#undef app_tests_MagAOXApp_test_hpp
#define XWCTEST_NAMESPACE XWCTEST_MAGAOXAPP_EXEC_WRONG_USER_ns
#define XWCTEST_MAGAOXAPP_EXEC_WRONG_USER
#include "../MagAOXApp.hpp"
#include "MagAOXApp_test.hpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_MAGAOXAPP_EXEC_WRONG_USER

#undef app_MagAOXApp_hpp
#undef app_tests_MagAOXApp_test_hpp
#define XWCTEST_NAMESPACE XWCTEST_MAGAOXAPP_EXEC_LOG_START_ns
#define XWCTEST_MAGAOXAPP_EXEC_LOG_START
#define XWCTEST_MAGAOXAPP_PID_UNLOCK_ERR
#include "../MagAOXApp.hpp"
#include "MagAOXApp_test.hpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_MAGAOXAPP_EXEC_LOG_START
#undef XWCTEST_MAGAOXAPP_PID_UNLOCK_ERR

#undef app_MagAOXApp_hpp
#undef app_tests_MagAOXApp_test_hpp
#define XWCTEST_NAMESPACE XWCTEST_MAGAOXAPP_EXEC_LOG_DEATH_ns
#define XWCTEST_MAGAOXAPP_EXEC_NORM
#define XWCTEST_MAGAOXAPP_EXEC_LOG_DEATH
#include "../MagAOXApp.hpp"
#include "MagAOXApp_test.hpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_MAGAOXAPP_EXEC_NORM
#undef XWCTEST_MAGAOXAPP_EXEC_LOG_DEATH


#undef app_MagAOXApp_hpp
#undef app_tests_MagAOXApp_test_hpp
#define XWCTEST_NAMESPACE XWCTEST_MAGAOXAPP_EXEC_SIGTERM_ns
#define XWCTEST_MAGAOXAPP_SIGTERMH_ERR
#define XWCTEST_MAGAOXAPP_PID_UNLOCK_ERR
#include "../MagAOXApp.hpp"
#include "MagAOXApp_test.hpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_MAGAOXAPP_SIGTERMH_ERR
#undef XWCTEST_MAGAOXAPP_PID_UNLOCK_ERR


#undef app_MagAOXApp_hpp
#undef app_tests_MagAOXApp_test_hpp
#define XWCTEST_NAMESPACE XWCTEST_MAGAOXAPP_EXEC_NORM_PID_UNLOCK_ns
#define XWCTEST_MAGAOXAPP_EXEC_NORM
#define XWCTEST_MAGAOXAPP_PID_UNLOCK_ERR
#include "../MagAOXApp.hpp"
#include "MagAOXApp_test.hpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_MAGAOXAPP_EXEC_NORM
#undef XWCTEST_MAGAOXAPP_PID_UNLOCK_ERR

#undef app_MagAOXApp_hpp
#undef app_tests_MagAOXApp_test_hpp
#define XWCTEST_NAMESPACE XWCTEST_MAGAOXAPP_EXEC_NORM_APPSTARTUP_ns
#define XWCTEST_MAGAOXAPP_EXEC_NORM
#define XWCTEST_MAGAOXAPP_PID_UNLOCK_ERR
#include "../MagAOXApp.hpp"
#include "MagAOXApp_test.hpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_MAGAOXAPP_EXEC_NORM
#undef XWCTEST_MAGAOXAPP_PID_UNLOCK_ERR

#undef app_MagAOXApp_hpp
#undef app_tests_MagAOXApp_test_hpp
#define XWCTEST_NAMESPACE XWCTEST_MAGAOXAPP_EXEC_NORM_APPLOGIC_ns
#define XWCTEST_MAGAOXAPP_EXEC_NORM
#define XWCTEST_MAGAOXAPP_PID_UNLOCK_ERR
#include "../MagAOXApp.hpp"
#include "MagAOXApp_test.hpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_MAGAOXAPP_EXEC_NORM
#undef XWCTEST_MAGAOXAPP_PID_UNLOCK_ERR

#undef app_MagAOXApp_hpp
#undef app_tests_MagAOXApp_test_hpp
#define XWCTEST_NAMESPACE XWCTEST_MAGAOXAPP_EXEC_NORM_APPSHUTDOWN_ns
#define XWCTEST_MAGAOXAPP_EXEC_NORM
#define XWCTEST_MAGAOXAPP_PID_UNLOCK_ERR
#include "../MagAOXApp.hpp"
#include "MagAOXApp_test.hpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_MAGAOXAPP_EXEC_NORM
#undef XWCTEST_MAGAOXAPP_PID_UNLOCK_ERR

#undef app_MagAOXApp_hpp
#undef app_tests_MagAOXApp_test_hpp
#define XWCTEST_NAMESPACE XWCTEST_MAGAOXAPP_EXEC_POWEROFF_AGAIN_ns
#define XWCTEST_MAGAOXAPP_EXEC_NORM
#include "../MagAOXApp.hpp"
#include "MagAOXApp_test.hpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_MAGAOXAPP_EXEC_NORM

// NOTE: the MagAOXApp log manager m_log is a static singleton per type. Once its background
// thread is started through logThreadStart() it is never joined or detached. That is by
// design, because the log thread of a real app runs for the life of the process. The
// std::thread move assignment operator calls std::terminate() if the target is already
// joinable. Any XWCTEST_NAMESPACE type whose execute() call gets far enough to call
// logThreadStart() successfully may therefore only be used once, in one test, for the life
// of this binary. That is why so many near-identical namespaces exist below. Each test that
// reaches the log thread start needs its own dedicated type.

#undef app_MagAOXApp_hpp
#undef app_tests_MagAOXApp_test_hpp
#define XWCTEST_NAMESPACE XWCTEST_MAGAOXAPP_EXEC_INDIFAIL_ns
#include "../MagAOXApp.hpp"
#include "MagAOXApp_test.hpp"
#undef XWCTEST_NAMESPACE

#undef app_MagAOXApp_hpp
#undef app_tests_MagAOXApp_test_hpp
#define XWCTEST_NAMESPACE XWCTEST_MAGAOXAPP_EXEC_INDIFAIL_UNLOCK_ERR_ns
#define XWCTEST_MAGAOXAPP_PID_UNLOCK_ERR
#include "../MagAOXApp.hpp"
#include "MagAOXApp_test.hpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_MAGAOXAPP_PID_UNLOCK_ERR

#undef app_MagAOXApp_hpp
#undef app_tests_MagAOXApp_test_hpp
#define XWCTEST_NAMESPACE XWCTEST_MAGAOXAPP_EXEC_WHILEPOWEROFF_ns
#define XWCTEST_MAGAOXAPP_EXEC_NORM
#include "../MagAOXApp.hpp"
#include "MagAOXApp_test.hpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_MAGAOXAPP_EXEC_NORM

#undef app_MagAOXApp_hpp
#undef app_tests_MagAOXApp_test_hpp
#define XWCTEST_NAMESPACE XWCTEST_MAGAOXAPP_EXEC_STALLED_ns
#include "../MagAOXApp.hpp"
#include "MagAOXApp_test.hpp"
#undef XWCTEST_NAMESPACE

#undef app_MagAOXApp_hpp
#undef app_tests_MagAOXApp_test_hpp
#define XWCTEST_NAMESPACE XWCTEST_MAGAOXAPP_SIGTERMH_SIGINT_ns
#define XWCTEST_MAGAOXAPP_SIGTERMH_SIGINT
#include "../MagAOXApp.hpp"
#include "MagAOXApp_test.hpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_MAGAOXAPP_SIGTERMH_SIGINT
// NOTE: as in MagAOXApp_test.cpp, XWCTEST_MAGAOXAPP_SIGTERMH_SIGINT redefines the SIGINT macro
// to SIGKILL inside setSigTermHandler(), and the header never restores it. That leaks for the
// rest of this translation unit. Only one of the SIGTERMH_SIGTERM, SIGTERMH_SIGQUIT, and
// SIGTERMH_SIGINT hooks may therefore be used per .cpp file. MagAOXApp_test.cpp already
// exercises the SIGQUIT branch this way. This file is a separate translation unit, so it
// independently exercises the SIGINT branch.


namespace libXWCTest
{
namespace appTest
{
namespace MagAOXAppTest
{

/// running execute
/**
 * \ingroup MagAOXApp_unit_test
 */
TEST_CASE( "running execute", "[app::MagAOXApp]" )
{
    SECTION( "complete run-through" )
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

        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/config" );
        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/logs" );
        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/sys/testapp" );
        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/drivers/fifos" );

        std::ofstream fout;
        fout.open( "/tmp/MagAOXApp_test/config/magaox.conf" );
        fout.close();

        mx::app::writeConfigFile( "/tmp/MagAOXApp_test/config/testapp.conf",
                                  { "", "power", "power", "power", "power", "power" },
                                  { "loopPause", "device", "channel", "element", "targetElement", "powerOnWait" },
                                  { "2500", "pdu0", "outlet1", "state", "target", "5" } ); // power values follow the real pdu naming. Nothing is connected.

        XWCTEST_MAGAOXAPP_EXEC_NORM_ns::MagAOXApp_test app( false );
        app.setPowerMgtEnabled( true );
        app.invokedName() = argv[0];

        app.setup( argv.size() - 1, const_cast<char **>( argv.data() ) );

        int rv = app.execute();
        REQUIRE( rv == 0 );
    }

    SECTION( "power goes off again after coming on" )
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

        std::filesystem::remove_all( "/tmp/MagAOXApp_test" );

        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/config" );
        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/logs" );
        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/sys/testapp" );
        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/drivers/fifos" );

        std::ofstream fout;
        fout.open( "/tmp/MagAOXApp_test/config/magaox.conf" );
        fout.close();

        mx::app::writeConfigFile( "/tmp/MagAOXApp_test/config/testapp.conf",
                                  { "", "power", "power", "power", "power", "power" },
                                  { "loopPause", "device", "channel", "element", "targetElement", "powerOnWait" },
                                  { "2500", "pdu0", "outlet1", "state", "target", "5" } );

        // The main loop makes four passes. Pass A: whilePowerOff() runs, then the
        // XWCTEST_MAGAOXAPP_EXEC_NORM macro forces m_powerState to 1. Pass B: execute() sees
        // state() == POWEROFF with m_powerState == 1, transitions to POWERON, and calls
        // appLogic() for the first time. Pass C: appLogic() runs a second time. The harness
        // hook flips m_powerState back to 0 right there and arms onPowerOffFail. Pass D needs
        // the raised testTimesThrough > 2 guard. It sees state() still POWERON with
        // m_powerState == 0, and calls the now failing onPowerOff() from inside the main
        // loop. This is distinct from the one-time onPowerOff() call before the loop, which
        // the "stalled" section already covers.
        XWCTEST_MAGAOXAPP_EXEC_POWEROFF_AGAIN_ns::MagAOXApp_test app( false );
        app.setPowerMgtEnabled( true );
        app.invokedName() = argv[0];
        app.m_flipPowerOffOnCall = 2;

        app.setup( argv.size() - 1, const_cast<char **>( argv.data() ) );

        int rv = app.execute();
        REQUIRE( rv == 0 );
    }

    SECTION( "No log directory" )
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

        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/config" );
        //don't create logs so the user check fails
        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/sys/testapp" );
        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/drivers/fifos" );

        std::ofstream fout;
        fout.open( "/tmp/MagAOXApp_test/config/magaox.conf" );
        fout.close();

        mx::app::writeConfigFile( "/tmp/MagAOXApp_test/config/testapp.conf",
                                  { "", "power", "power", "power", "power", "power" },
                                  { "loopPause", "device", "channel", "element", "targetElement", "powerOnWait" },
                                  { "2500", "pdu0", "outlet1", "state", "target", "5" } );

        MagAOXApp_test app( false );
        app.setPowerMgtEnabled( true );
        app.invokedName() = argv[0];

        app.setup( argv.size() - 1, const_cast<char **>( argv.data() ) );

        int rv = app.execute();
        REQUIRE( rv == -1 );
    }

    SECTION( "log thread dies during the main loop" )
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

        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/config" );
        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/logs" );
        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/sys/testapp" );
        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/drivers/fifos" );

        std::ofstream fout;
        fout.open( "/tmp/MagAOXApp_test/config/magaox.conf" );
        fout.close();

        mx::app::writeConfigFile( "/tmp/MagAOXApp_test/config/testapp.conf",
                                  { "", "power", "power", "power", "power", "power" },
                                  { "loopPause", "device", "channel", "element", "targetElement", "powerOnWait" },
                                  { "2500", "pdu0", "outlet1", "state", "target", "5" } );

        // The XWCTEST_MAGAOXAPP_EXEC_LOG_DEATH macro in this namespace's execute() forces the
        // log thread to stop right after it is confirmed running. The first iteration of the
        // main loop therefore finds it already dead. This exercises the check for a log thread
        // that is not running inside the loop, which is distinct from the check before
        // appStartup().
        XWCTEST_MAGAOXAPP_EXEC_LOG_DEATH_ns::MagAOXApp_test app( false );
        app.setPowerMgtEnabled( true );
        app.invokedName() = argv[0];

        app.setup( argv.size() - 1, const_cast<char **>( argv.data() ) );

        int rv = app.execute();
        REQUIRE( rv == 0 );
    }

    SECTION( "wrong user" )
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

        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/config" );
        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/logs" );
        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/sys/testapp" );
        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/drivers/fifos" );

        std::ofstream fout;
        fout.open( "/tmp/MagAOXApp_test/config/magaox.conf" );
        fout.close();

        mx::app::writeConfigFile( "/tmp/MagAOXApp_test/config/testapp.conf",
                                  { "", "power", "power", "power", "power", "power" },
                                  { "loopPause", "device", "channel", "element", "targetElement", "powerOnWait" },
                                  { "2500", "pdu0", "outlet1", "state", "target", "5" } );

        XWCTEST_MAGAOXAPP_EXEC_WRONG_USER_ns::MagAOXApp_test app( false );
        app.setPowerMgtEnabled( true );
        app.invokedName() = argv[0];

        app.setup( argv.size() - 1, const_cast<char **>( argv.data() ) );

        int rv = app.execute();
        REQUIRE( rv == -1 );
    }

    SECTION( "PID Lock Error" )
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

        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/config" );
        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/logs" );
        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/sys/testapp" );
        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/drivers/fifos" );

        std::ofstream fout;
        fout.open( "/tmp/MagAOXApp_test/config/magaox.conf" );
        fout.close();

        mx::app::writeConfigFile( "/tmp/MagAOXApp_test/config/testapp.conf",
                                  { "", "power", "power", "power", "power", "power" },
                                  { "loopPause", "device", "channel", "element", "targetElement", "powerOnWait" },
                                  { "2500", "pdu0", "outlet1", "state", "target", "5" } );

        XWCTEST_MAGAOXAPP_PID_WRITE_FAIL_ns::MagAOXApp_test app( false );
        app.setPowerMgtEnabled( true );
        app.invokedName() = argv[0];

        app.setup( argv.size() - 1, const_cast<char **>( argv.data() ) );

        int rv = app.execute();
        REQUIRE( rv == -1 );
    }

    SECTION( "Log fails to start" )
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

        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/config" );
        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/logs" );
        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/sys/testapp" );
        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/drivers/fifos" );

        std::ofstream fout;
        fout.open( "/tmp/MagAOXApp_test/config/magaox.conf" );
        fout.close();

        mx::app::writeConfigFile( "/tmp/MagAOXApp_test/config/testapp.conf",
                                  { "", "power", "power", "power", "power", "power" },
                                  { "loopPause", "device", "channel", "element", "targetElement", "powerOnWait" },
                                  { "2500", "pdu0", "outlet1", "state", "target", "5" } );

        XWCTEST_MAGAOXAPP_EXEC_LOG_START_ns::MagAOXApp_test app( false );
        app.setPowerMgtEnabled( true );
        app.invokedName() = argv[0];

        app.setup( argv.size() - 1, const_cast<char **>( argv.data() ) );

        int rv = app.execute();
        REQUIRE( rv == -1 );
    }

    SECTION( "Setting sigterm handler fails" )
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

        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/config" );
        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/logs" );
        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/sys/testapp" );
        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/drivers/fifos" );

        std::ofstream fout;
        fout.open( "/tmp/MagAOXApp_test/config/magaox.conf" );
        fout.close();

        mx::app::writeConfigFile( "/tmp/MagAOXApp_test/config/testapp.conf",
                                  { "", "power", "power", "power", "power", "power" },
                                  { "loopPause", "device", "channel", "element", "targetElement", "powerOnWait" },
                                  { "2500", "pdu0", "outlet1", "state", "target", "5" } );

        XWCTEST_MAGAOXAPP_EXEC_SIGTERM_ns::MagAOXApp_test app( false );
        app.setPowerMgtEnabled( true );
        app.invokedName() = argv[0];

        app.setup( argv.size() - 1, const_cast<char **>( argv.data() ) );

        int rv = app.execute();
        REQUIRE( rv == -1 );
    }

    SECTION( "appStartup failure" )
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

        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/config" );
        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/logs" );
        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/sys/testapp" );
        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/drivers/fifos" );

        std::ofstream fout;
        fout.open( "/tmp/MagAOXApp_test/config/magaox.conf" );
        fout.close();

        mx::app::writeConfigFile( "/tmp/MagAOXApp_test/config/testapp.conf",
                                  { "", "power", "power", "power", "power", "power" },
                                  { "loopPause", "device", "channel", "element", "targetElement", "powerOnWait" },
                                  { "2500", "pdu0", "outlet1", "state", "target", "5" } );

        XWCTEST_MAGAOXAPP_EXEC_NORM_APPSTARTUP_ns::MagAOXApp_test app( false );
        app.setPowerMgtEnabled( true );
        app.invokedName() = argv[0];

        app.setup( argv.size() - 1, const_cast<char **>( argv.data() ) );

        app.appStartupFail = true;

        int rv = app.execute();
        REQUIRE( rv == -1 );
    }

    SECTION( "appLogic failure" )
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

        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/config" );
        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/logs" );
        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/sys/testapp" );
        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/drivers/fifos" );

        std::ofstream fout;
        fout.open( "/tmp/MagAOXApp_test/config/magaox.conf" );
        fout.close();

        mx::app::writeConfigFile( "/tmp/MagAOXApp_test/config/testapp.conf",
                                  { "", "power", "power", "power", "power", "power" },
                                  { "loopPause", "device", "channel", "element", "targetElement", "powerOnWait" },
                                  { "2500", "pdu0", "outlet1", "state", "target", "5" } );

        XWCTEST_MAGAOXAPP_EXEC_NORM_APPLOGIC_ns::MagAOXApp_test app( false );
        app.setPowerMgtEnabled( true );
        app.invokedName() = argv[0];

        app.setup( argv.size() - 1, const_cast<char **>( argv.data() ) );
        app.appLogicFail = true;
        int rv = app.execute(); // An appLogic() failure only triggers shutdown. execute() still returns 0.
        REQUIRE( rv == 0 );
    }

    SECTION( "appShutdown failure" )
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

        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/config" );
        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/logs" );
        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/sys/testapp" );
        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/drivers/fifos" );

        std::ofstream fout;
        fout.open( "/tmp/MagAOXApp_test/config/magaox.conf" );
        fout.close();

        mx::app::writeConfigFile( "/tmp/MagAOXApp_test/config/testapp.conf",
                                  { "", "power", "power", "power", "power", "power" },
                                  { "loopPause", "device", "channel", "element", "targetElement", "powerOnWait" },
                                  { "2500", "pdu0", "outlet1", "state", "target", "5" } );

        XWCTEST_MAGAOXAPP_EXEC_NORM_APPSHUTDOWN_ns::MagAOXApp_test app( false );
        app.setPowerMgtEnabled( true );
        app.invokedName() = argv[0];

        app.setup( argv.size() - 1, const_cast<char **>( argv.data() ) );
        app.appShutdownFail = true;
        int rv = app.execute();//this still returns 0
        REQUIRE( rv == 0 );
    }

    SECTION( "INDI fails to start" )
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

        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/config" );
        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/logs" );
        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/sys/testapp" );
        // Deliberately do not create drivers/fifos. mkfifo() inside createINDIFIFOS() then fails
        // with ENOENT because the parent directory is missing. That makes startINDI() fail, and
        // therefore execute() fails too.

        std::ofstream fout;
        fout.open( "/tmp/MagAOXApp_test/config/magaox.conf" );
        fout.close();

        mx::app::writeConfigFile( "/tmp/MagAOXApp_test/config/testapp.conf",
                                  { "", "power", "power", "power", "power", "power" },
                                  { "loopPause", "device", "channel", "element", "targetElement", "powerOnWait" },
                                  { "2500", "pdu0", "outlet1", "state", "target", "5" } );

        XWCTEST_MAGAOXAPP_EXEC_INDIFAIL_ns::MagAOXApp_test app( false );
        app.setPowerMgtEnabled( true );
        app.invokedName() = argv[0];
        app.appShutdownFail = true; // Also exercises the appShutdown() error branch taken after the INDI failure.

        app.setup( argv.size() - 1, const_cast<char **>( argv.data() ) );

        int rv = app.execute();
        REQUIRE( rv == -1 );
    }

    SECTION( "INDI fails to start and unlockPID() also fails" )
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

        std::filesystem::remove_all( "/tmp/MagAOXApp_test" );

        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/config" );
        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/logs" );
        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/sys/testapp" );
        // Deliberately do not create drivers/fifos, so startINDI() fails as in the "INDI fails
        // to start" section above. This namespace also forces unlockPID() to fail
        // unconditionally through XWCTEST_MAGAOXAPP_PID_UNLOCK_ERR. That exercises the
        // unlockPID() error branch taken right after the INDI startup failure.

        std::ofstream fout;
        fout.open( "/tmp/MagAOXApp_test/config/magaox.conf" );
        fout.close();

        mx::app::writeConfigFile( "/tmp/MagAOXApp_test/config/testapp.conf",
                                  { "", "power", "power", "power", "power", "power" },
                                  { "loopPause", "device", "channel", "element", "targetElement", "powerOnWait" },
                                  { "2500", "pdu0", "outlet1", "state", "target", "5" } );

        XWCTEST_MAGAOXAPP_EXEC_INDIFAIL_UNLOCK_ERR_ns::MagAOXApp_test app( false );
        app.setPowerMgtEnabled( true );
        app.invokedName() = argv[0];

        app.setup( argv.size() - 1, const_cast<char **>( argv.data() ) );

        int rv = app.execute();
        REQUIRE( rv == -1 );
    }

    SECTION( "whilePowerOff failure in main loop" )
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

        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/config" );
        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/logs" );
        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/sys/testapp" );
        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/drivers/fifos" );

        std::ofstream fout;
        fout.open( "/tmp/MagAOXApp_test/config/magaox.conf" );
        fout.close();

        mx::app::writeConfigFile( "/tmp/MagAOXApp_test/config/testapp.conf",
                                  { "", "power", "power", "power", "power", "power" },
                                  { "loopPause", "device", "channel", "element", "targetElement", "powerOnWait" },
                                  { "2500", "pdu0", "outlet1", "state", "target", "5" } );

        // EXEC_NORM forces the power state to 0, which is off, at startup and skips the real
        // wait. It also limits the main loop to a couple of iterations. whilePowerOff()
        // therefore runs for real while powered off.
        XWCTEST_MAGAOXAPP_EXEC_WHILEPOWEROFF_ns::MagAOXApp_test app( false );
        app.setPowerMgtEnabled( true );
        app.invokedName() = argv[0];
        app.whilePowerOffFail = true;

        app.setup( argv.size() - 1, const_cast<char **>( argv.data() ) );

        int rv = app.execute(); // A whilePowerOff() failure only triggers shutdown. execute() still returns 0.
        REQUIRE( rv == 0 );
    }

    SECTION( "stalled waiting for power state, then onPowerOff fails" )
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

        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/config" );
        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/logs" );
        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/sys/testapp" );
        mx::ioutils::createDirectories( "/tmp/MagAOXApp_test/drivers/fifos" );

        std::ofstream fout;
        fout.open( "/tmp/MagAOXApp_test/config/magaox.conf" );
        fout.close();

        mx::app::writeConfigFile( "/tmp/MagAOXApp_test/config/testapp.conf",
                                  { "", "power", "power", "power", "power", "power" },
                                  { "loopPause", "device", "channel", "element", "targetElement", "powerOnWait" },
                                  { "2500", "pdu0", "outlet1", "state", "target", "5" } );

        // No fault injection toggle forces m_powerState here, so it stays at its startup
        // value of -1, which means unknown. execute() blocks in a real sleep(1) loop for 30
        // real seconds until it gives up with "stalled waiting for power state". It then
        // falls through to the onPowerOff() call, which the test makes fail too.
        XWCTEST_MAGAOXAPP_EXEC_STALLED_ns::MagAOXApp_test app( false );
        app.setPowerMgtEnabled( true );
        app.invokedName() = argv[0];
        app.onPowerOffFail = true;

        app.setup( argv.size() - 1, const_cast<char **>( argv.data() ) );

        int rv = app.execute();
        REQUIRE( rv == 0 );
    }
}

/// Verify that setSigTermHandler() reports failure when the sigaction() call for SIGINT fails.
/// The XWCTEST_MAGAOXAPP_SIGTERMH_SIGINT hook forces exactly that call to fail.
/**
 * \ingroup MagAOXApp_unit_test
 */
TEST_CASE( "Signal handler setup failure for SIGINT", "[app::MagAOXApp]" )
{
    // NOTE: see the comment near the #include block for this hook at the top of this file.
    // The XWCTEST_MAGAOXAPP_SIGTERMH_SIGINT redefinition of SIGINT to SIGKILL leaks for the
    // rest of this translation unit, so it must be the only SIGTERMH_SIGTERM, SIGTERMH_SIGQUIT,
    // or SIGTERMH_SIGINT hook used here.
    XWCTEST_MAGAOXAPP_SIGTERMH_SIGINT_ns::MagAOXApp_test app;

    REQUIRE( app.setSigTermHandler() == -1 );
}


} // namespace MagAOXAppTest
} // namespace appTest
} // namespace libXWCTest
