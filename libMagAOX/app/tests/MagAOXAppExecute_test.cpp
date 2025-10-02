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
                                  { "2500", "pdu9", "thisch", "thisel", "thistgtel", "5" } );

        XWCTEST_MAGAOXAPP_EXEC_NORM_ns::MagAOXApp_test app( false );
        app.setPowerMgtEnabled( true );
        app.invokedName() = argv[0];

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
                                  { "2500", "pdu9", "thisch", "thisel", "thistgtel", "5" } );

        MagAOXApp_test app( false );
        app.setPowerMgtEnabled( true );
        app.invokedName() = argv[0];

        app.setup( argv.size() - 1, const_cast<char **>( argv.data() ) );

        int rv = app.execute();
        REQUIRE( rv == -1 );
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
                                  { "2500", "pdu9", "thisch", "thisel", "thistgtel", "5" } );

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
                                  { "2500", "pdu9", "thisch", "thisel", "thistgtel", "5" } );

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
                                  { "2500", "pdu9", "thisch", "thisel", "thistgtel", "5" } );

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
                                  { "2500", "pdu9", "thisch", "thisel", "thistgtel", "5" } );

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
                                  { "2500", "pdu9", "thisch", "thisel", "thistgtel", "5" } );

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
                                  { "2500", "pdu9", "thisch", "thisel", "thistgtel", "5" } );

        XWCTEST_MAGAOXAPP_EXEC_NORM_APPLOGIC_ns::MagAOXApp_test app( false );
        app.setPowerMgtEnabled( true );
        app.invokedName() = argv[0];

        app.setup( argv.size() - 1, const_cast<char **>( argv.data() ) );
        app.appLogicFail = true;
        int rv = app.execute();
        REQUIRE( rv == -1 );
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
                                  { "2500", "pdu9", "thisch", "thisel", "thistgtel", "5" } );

        XWCTEST_MAGAOXAPP_EXEC_NORM_APPSHUTDOWN_ns::MagAOXApp_test app( false );
        app.setPowerMgtEnabled( true );
        app.invokedName() = argv[0];

        app.setup( argv.size() - 1, const_cast<char **>( argv.data() ) );
        app.appShutdownFail = true;
        int rv = app.execute();//this still returns 0
        REQUIRE( rv == 0 );
    }
}


} // namespace MagAOXAppTest
} // namespace appTest
} // namespace libXWCTest
