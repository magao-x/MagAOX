/** \file logManager_test.cpp
 * \brief Tests for the logManager class
 *
 * Technique: a mock parent records delivered messages, and a subclass exposes the
 * protected log queue. Most tests call logThreadExec() directly on the test thread so
 * they control exactly what is processed. The thread lifecycle tests start a real
 * std::thread. Fault-injected builds of logManager and logFileRaw, compiled under test
 * namespaces with XWCTEST_ macros, force the thread start and write failure branches.
 * Log files and config files are written under /tmp.
 * \ingroup logger_files
 */

#include "../../../tests/testXWC.hpp"

#include "../logManager.hpp"
#include "../logFileRaw.hpp"

#include <filesystem>
#include <chrono>
#include <vector>

// Fault injection. Each block below compiles a production header a second time inside a
// test namespace with one XWCTEST_ fault macro defined. The include guard is undefined
// first so the header is really re-read.

// Forces logThreadStart() to throw a std::runtime_error before constructing the thread.
#undef logger_logManager_hpp
#define XWCTEST_NAMESPACE XWCTEST_LOGMANAGER_LOGTHREADSTART_STD_EXCEPTION_ns
#define XWCTEST_LOGMANAGER_LOGTHREADSTART_STD_EXCEPTION
#include "../logManager.hpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_LOGMANAGER_LOGTHREADSTART_STD_EXCEPTION

// Forces logThreadStart() to throw a value that is not a std::exception before
// constructing the thread.
#undef logger_logManager_hpp
#define XWCTEST_NAMESPACE XWCTEST_LOGMANAGER_LOGTHREADSTART_UNKNOWN_EXCEPTION_ns
#define XWCTEST_LOGMANAGER_LOGTHREADSTART_UNKNOWN_EXCEPTION
#include "../logManager.hpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_LOGMANAGER_LOGTHREADSTART_UNKNOWN_EXCEPTION

// Removes the std::thread construction from logThreadStart(), so its joinable() check
// fails for real.
#undef logger_logManager_hpp
#define XWCTEST_NAMESPACE XWCTEST_LOGMANAGER_LOGTHREADSTART_NOT_JOINABLE_ns
#define XWCTEST_LOGMANAGER_LOGTHREADSTART_NOT_JOINABLE
#include "../logManager.hpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_LOGMANAGER_LOGTHREADSTART_NOT_JOINABLE

// Makes logFileRaw::writeLog() see fwrite report zero bytes written. This build is used
// as the log file type of a logManager below, so the manager sees a write failure.
#undef logger_logFileRaw_hpp
#define XWCTEST_NAMESPACE XWCTEST_LOGFILERAW_WRITELOG_FWRITE_FAIL_ns
#define XWCTEST_LOGFILERAW_WRITELOG_FWRITE_FAIL
#include "../logFileRaw.hpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_LOGFILERAW_WRITELOG_FWRITE_FAIL

namespace libXWCTest
{
namespace loggerTest
{

/** \defgroup logManager_unit_test logManager Unit Tests
 * \ingroup logger_unit_test
 */

/// Namespace for XWC::logger::logManager tests
/** \ingroup logManager_unit_test
 *
 */
namespace logManagerTest_ns
{

/// A minimal stand-in for a MagAOXApp-like parent. It records delivered messages instead
/// of presenting them.
struct mockParent
{
    std::vector<flatlogs::bufferPtrT> messages;

    void logMessage( flatlogs::bufferPtrT &buf )
    {
        messages.push_back( buf );
    }
};

/// Exposes the protected log queue for verification.
template <class parentT, class logFileT>
struct logManagerTest : public MagAOX::logger::logManager<parentT, logFileT>
{
    size_t test_queueSize()
    {
        return this->m_logQueue.size();
    }
};

typedef logManagerTest<mockParent, MagAOX::logger::logFileRaw<XWC_DEFAULT_VERBOSITY>> logManagerTestT;

typedef logManagerTest<mockParent,
                       MagAOX::logger::XWCTEST_LOGFILERAW_WRITELOG_FWRITE_FAIL_ns::logFileRaw<XWC_DEFAULT_VERBOSITY>>
    logManagerFwriteFailT;

typedef MagAOX::logger::XWCTEST_LOGMANAGER_LOGTHREADSTART_STD_EXCEPTION_ns::
    logManager<mockParent, MagAOX::logger::logFileRaw<XWC_DEFAULT_VERBOSITY>>
        logManagerStdExceptionT;

typedef MagAOX::logger::XWCTEST_LOGMANAGER_LOGTHREADSTART_UNKNOWN_EXCEPTION_ns::
    logManager<mockParent, MagAOX::logger::logFileRaw<XWC_DEFAULT_VERBOSITY>>
        logManagerUnknownExceptionT;

typedef MagAOX::logger::XWCTEST_LOGMANAGER_LOGTHREADSTART_NOT_JOINABLE_ns::
    logManager<mockParent, MagAOX::logger::logFileRaw<XWC_DEFAULT_VERBOSITY>>
        logManagerNotJoinableT;

/// Construction, getters, and setters. Each section sets one value and reads it back,
/// including the rejected and clamped cases.
/**
 * \ingroup logManager_unit_test
 */
TEST_CASE( "logManager getters and setters", "[libMagAOX::logger::logManager]" )
{
    SECTION( "default construction" )
    {
        logManagerTestT lm;

        REQUIRE( lm.parent() == nullptr );
        REQUIRE( lm.logShutdown() == false );
        REQUIRE( lm.writePause() == MAGAOX_default_writePause );
        REQUIRE( lm.logLevel() == flatlogs::logPrio::LOG_INFO );
        REQUIRE( lm.logThreadPrio() == 0 );
        REQUIRE( lm.logThreadRunning() == false );
    }

    SECTION( "parent" )
    {
        logManagerTestT lm;
        mockParent      mp;

        lm.parent( &mp );

        REQUIRE( lm.parent() == &mp );
    }

    SECTION( "logShutdown" )
    {
        logManagerTestT lm;

        REQUIRE( lm.logShutdown( true ) == 0 );
        REQUIRE( lm.logShutdown() == true );

        REQUIRE( lm.logShutdown( false ) == 0 );
        REQUIRE( lm.logShutdown() == false );
    }

    SECTION( "writePause rejects 0 and accepts a valid value" )
    {
        logManagerTestT lm;

        REQUIRE( lm.writePause( 0 ) == -1 );
        REQUIRE( lm.writePause() == MAGAOX_default_writePause ); // The value is unchanged.

        REQUIRE( lm.writePause( 12345 ) == 0 ); // arbitrary value
        REQUIRE( lm.writePause() == 12345 );
    }

    SECTION( "logLevel" )
    {
        logManagerTestT lm;

        REQUIRE( lm.logLevel( flatlogs::logPrio::LOG_WARNING ) == 0 );
        REQUIRE( lm.logLevel() == flatlogs::logPrio::LOG_WARNING );
    }

    SECTION( "logThreadPrio accepts a valid value, clamps negative values, and rejects values over 98" )
    {
        logManagerTestT lm;

        REQUIRE( lm.logThreadPrio( 50 ) == 0 );
        REQUIRE( lm.logThreadPrio() == 50 );

        REQUIRE( lm.logThreadPrio( -5 ) == 0 );
        REQUIRE( lm.logThreadPrio() == 0 );

        REQUIRE( lm.logThreadPrio( 99 ) == -1 );
        REQUIRE( lm.logThreadPrio() == 0 ); // The value is unchanged from the clamp above.
    }
}

/// setupConfig and loadConfig. The sections write real config files under /tmp and read
/// them back through an appConfigurator.
/**
 * \ingroup logManager_unit_test
 */
TEST_CASE( "logManager configuration", "[libMagAOX::logger::logManager]" )
{
    SECTION( "setupConfig registers the logger options" )
    {
        mx::app::appConfigurator config;

        logManagerTestT lm;

        REQUIRE( lm.setupConfig( config ) == 0 );
    }

    SECTION( "loadConfig with no logger section present leaves the defaults" )
    {
        mx::app::appConfigurator config;

        logManagerTestT lm;

        REQUIRE( lm.setupConfig( config ) == 0 );
        REQUIRE( lm.loadConfig( config ) == 0 );

        REQUIRE( lm.logPath() == "." );
        REQUIRE( lm.logLevel() == flatlogs::logPrio::LOG_INFO );
    }

    SECTION( "loadConfig applies logDir, logExt, maxLogSize, writePause, and logThreadPrio" )
    {
        mx::app::writeConfigFile( "/tmp/logManager_test.conf",
                                  { "logger", "logger", "logger", "logger", "logger" },
                                  { "logDir", "logExt", "maxLogSize", "writePause", "logThreadPrio" },
                                  { "/tmp/logManager_test_dir", "binlog", "1000", "2000", "5" } );

        mx::app::appConfigurator config;

        logManagerTestT lm;

        REQUIRE( lm.setupConfig( config ) == 0 );

        config.readConfig( "/tmp/logManager_test.conf" );

        REQUIRE( lm.loadConfig( config ) == 0 );

        REQUIRE( lm.logPath() == "/tmp/logManager_test_dir" );
        REQUIRE( lm.logExt() == "binlog" );
        REQUIRE( lm.maxLogSize() == 1000 );
        REQUIRE( lm.writePause() == 2000 );
        REQUIRE( lm.logThreadPrio() == 5 );
    }

    SECTION( "loadConfig applies a recognized logLevel string" )
    {
        mx::app::writeConfigFile( "/tmp/logManager_test.conf", { "logger" }, { "logLevel" }, { "WARNING" } );

        mx::app::appConfigurator config;

        logManagerTestT lm;

        REQUIRE( lm.setupConfig( config ) == 0 );

        config.readConfig( "/tmp/logManager_test.conf" );

        REQUIRE( lm.loadConfig( config ) == 0 );

        REQUIRE( lm.logLevel() == flatlogs::logPrio::LOG_WARNING );
    }

    SECTION( "loadConfig maps a DEFAULT logLevel string to INFO" )
    {
        mx::app::writeConfigFile( "/tmp/logManager_test.conf", { "logger" }, { "logLevel" }, { "DEFAULT" } );

        mx::app::appConfigurator config;

        logManagerTestT lm;

        REQUIRE( lm.setupConfig( config ) == 0 );

        config.readConfig( "/tmp/logManager_test.conf" );

        REQUIRE( lm.loadConfig( config ) == 0 );

        REQUIRE( lm.logLevel() == flatlogs::logPrio::LOG_INFO );
    }

    SECTION( "loadConfig maps an unrecognized logLevel string to INFO" )
    {
        mx::app::writeConfigFile( "/tmp/logManager_test.conf", { "logger" }, { "logLevel" }, { "bogus-level" } ); // starts with a letter no level name starts with, so it is unrecognized

        mx::app::appConfigurator config;

        logManagerTestT lm;

        REQUIRE( lm.setupConfig( config ) == 0 );

        config.readConfig( "/tmp/logManager_test.conf" );

        REQUIRE( lm.loadConfig( config ) == 0 );

        REQUIRE( lm.logLevel() == flatlogs::logPrio::LOG_INFO );
    }
}

/// The log<logT>() overloads and the level filter. Each section queues one entry and
/// checks the queue size.
/**
 * \ingroup logManager_unit_test
 */
TEST_CASE( "logManager creates and queues log entries", "[libMagAOX::logger::logManager]" )
{
    logManagerTestT lm;
    lm.logPath( "/tmp" );

    std::string testPath = lm.logPath() + '/' + lm.logName();
    std::filesystem::remove_all( testPath );

    SECTION( "a log above the current level is filtered and not queued" )
    {
        lm.log<MagAOX::logger::software_error>( { __FILE__, __LINE__, 0, 0, std::string( "filtered" ) },
                                                flatlogs::logPrio::LOG_DEBUG );

        REQUIRE( lm.test_queueSize() == 0 );
    }

    SECTION( "a log with a message and the default timestamp is queued" )
    {
        lm.log<MagAOX::logger::software_error>( { __FILE__, __LINE__, 0, 0, std::string( "queued" ) } );

        REQUIRE( lm.test_queueSize() == 1 );
    }

    SECTION( "a log with a message and an explicit timestamp is queued" )
    {
        flatlogs::timespecX ts;
        ts.gettime();

        lm.log<MagAOX::logger::software_error>( ts, { __FILE__, __LINE__, 0, 0, std::string( "queued" ) } );

        REQUIRE( lm.test_queueSize() == 1 );
    }

    SECTION( "a log with no message and the default timestamp is queued" )
    {
        lm.log<MagAOX::logger::loop_closed>();

        REQUIRE( lm.test_queueSize() == 1 );
    }

    SECTION( "a log with no message and an explicit timestamp is queued" )
    {
        flatlogs::timespecX ts;
        ts.gettime();

        lm.log<MagAOX::logger::loop_closed>( ts );

        REQUIRE( lm.test_queueSize() == 1 );
    }

    // The destructor drains whatever ended up queued. See the "drains any remaining
    // queued entries" test case below for a check that this actually happens.
}

/// logThreadExec is called directly on the test thread, so the test controls exactly what
/// gets processed.
/**
 * \ingroup logManager_unit_test
 */
TEST_CASE( "logManager processes queued entries via logThreadExec", "[libMagAOX::logger::logManager]" )
{
    SECTION( "drains and writes a queued entry, printing to stderr when there is no parent" )
    {
        logManagerTestT lm;
        lm.logPath( "/tmp" );

        std::string testPath = lm.logPath() + '/' + lm.logName();
        std::filesystem::remove_all( testPath );

        // software_error defaults to LOG_ERROR, which is at or below LOG_NOTICE. No parent
        // is set, so this exercises the stderr print branch.
        lm.log<MagAOX::logger::software_error>( { __FILE__, __LINE__, 0, 0, std::string( "printed" ) } );

        REQUIRE( lm.test_queueSize() == 1 );

        lm.logShutdown( true );
        lm.logThreadExec();

        REQUIRE( lm.test_queueSize() == 0 );
        REQUIRE( lm.logThreadRunning() == false );
    }

    SECTION( "drains and writes a queued entry without printing when the level is above NOTICE" )
    {
        logManagerTestT lm;
        lm.logPath( "/tmp" );

        std::string testPath = lm.logPath() + '/' + lm.logName();
        std::filesystem::remove_all( testPath );

        // text_log defaults to LOG_INFO, which is above LOG_NOTICE. So the stderr print
        // branch is skipped even though there is no parent.
        lm.log<MagAOX::logger::text_log>( std::string( "not printed" ) );

        REQUIRE( lm.test_queueSize() == 1 );

        lm.logShutdown( true );
        lm.logThreadExec();

        REQUIRE( lm.test_queueSize() == 0 );
        REQUIRE( lm.logThreadRunning() == false );
    }

    SECTION( "drains and delivers a queued entry to the parent instead of stderr" )
    {
        mockParent mp;

        logManagerTestT lm;
        lm.logPath( "/tmp" );
        lm.parent( &mp );

        std::string testPath = lm.logPath() + '/' + lm.logName();
        std::filesystem::remove_all( testPath );

        lm.log<MagAOX::logger::software_error>( { __FILE__, __LINE__, 0, 0, std::string( "to parent" ) } );

        lm.logShutdown( true );
        lm.logThreadExec();

        REQUIRE( mp.messages.size() == 1 );
        REQUIRE( lm.test_queueSize() == 0 );
    }

    SECTION( "an empty queue with shutdown already set never enters the processing loop" )
    {
        // With shutdown already true and the queue empty, the while condition
        // !m_logShutdown || !m_logQueue.empty() is false from the very first check. So
        // the loop body never runs, and neither does its flush() call. This is a distinct
        // path from the sections in this test case that drain a queued entry. There the
        // queue is non-empty on entry, so the loop runs exactly once. The empty queue
        // loop body and its flush() call are covered by the "pauses between empty-queue
        // checks" thread lifecycle test. That test runs with shutdown initially false, so
        // the loop is actually entered.
        logManagerTestT lm;
        lm.logPath( "/tmp" );

        std::string testPath = lm.logPath() + '/' + lm.logName();
        std::filesystem::remove_all( testPath );

        lm.logShutdown( true );
        lm.logThreadExec();

        REQUIRE( lm.test_queueSize() == 0 );
        REQUIRE( lm.logThreadRunning() == false );
    }

    SECTION( "a writeLog failure stops the loop and leaves the entry queued" )
    {
        logManagerFwriteFailT lm;
        lm.logPath( "/tmp" );

        std::string testPath = lm.logPath() + '/' + lm.logName();
        std::filesystem::remove_all( testPath );

        lm.log<MagAOX::logger::software_error>( { __FILE__, __LINE__, 0, 0, std::string( "will fail" ) } );

        REQUIRE( lm.test_queueSize() == 1 );

        lm.logShutdown( true );
        lm.logThreadExec();

        // The failing entry is never erased, since writeLog's error causes an early return.
        REQUIRE( lm.test_queueSize() == 1 );
        REQUIRE( lm.logThreadRunning() == false );
    }
}

/// logThreadStart, _logThreadStart, and the lifecycle of the background thread. These
/// sections start a real std::thread and rely on the destructor to join it.
/**
 * \ingroup logManager_unit_test
 */
TEST_CASE( "logManager thread lifecycle", "[libMagAOX::logger::logManager]" )
{
    SECTION( "starts and stops immediately when shutdown is already set" )
    {
        logManagerTestT lm;
        lm.logPath( "/tmp" );

        std::string testPath = lm.logPath() + '/' + lm.logName();
        std::filesystem::remove_all( testPath );

        // Shutting down before starting means the first check of the loop condition by
        // the thread is false, because the queue is empty. So the thread exits almost
        // immediately.
        lm.logShutdown( true );

        REQUIRE( lm.logThreadStart() == 0 );

        // The destructor of lm joins the thread, which is already exiting.
    }

    SECTION( "reports an error when the scheduling priority can't be set" )
    {
        logManagerTestT lm;
        lm.logPath( "/tmp" );

        std::string testPath = lm.logPath() + '/' + lm.logName();
        std::filesystem::remove_all( testPath );

        // SCHED_OTHER only accepts priority 0 on Linux, so requesting a nonzero priority
        // makes the real pthread_setschedparam call fail with EINVAL.
        lm.logThreadPrio( 50 );
        lm.logShutdown( true );

        REQUIRE( lm.logThreadStart() == -1 );

        // The destructor of lm joins the thread, which is already exiting, and drains the
        // software_error log that logThreadStart() queued to report the failure.
    }

    SECTION( "pauses between empty-queue checks while running" )
    {
        logManagerTestT lm;
        lm.logPath( "/tmp" );

        std::string testPath = lm.logPath() + '/' + lm.logName();
        std::filesystem::remove_all( testPath );

        lm.writePause( 1000 ); // This is 1 microsecond, so many empty queue and sleep cycles happen quickly.

        REQUIRE( lm.logThreadStart() == 0 );

        // Give the thread a chance to spin through several empty queue and sleep cycles.
        std::this_thread::sleep_for( std::chrono::milliseconds( 100 ) );

        lm.logShutdown( true );

        // The destructor of lm joins the thread once the thread observes the shutdown
        // flag.
    }

    SECTION( "reports an error when std::thread construction throws a std::exception" )
    {
        logManagerStdExceptionT lm;
        lm.logPath( "/tmp" );

        std::string testPath = lm.logPath() + '/' + lm.logName();
        std::filesystem::remove_all( testPath );

        REQUIRE( lm.logThreadStart() == -1 );

        // The thread was never created because the throw happens before construction. So
        // the destructor of lm has nothing to join. It does drain the software_error log
        // that logThreadStart() queued to report the exception.
    }

    SECTION( "reports an error when std::thread construction throws a non-std::exception" )
    {
        logManagerUnknownExceptionT lm;
        lm.logPath( "/tmp" );

        std::string testPath = lm.logPath() + '/' + lm.logName();
        std::filesystem::remove_all( testPath );

        REQUIRE( lm.logThreadStart() == -1 );
    }

    SECTION( "reports an error when the thread doesn't start" )
    {
        logManagerNotJoinableT lm;
        lm.logPath( "/tmp" );

        std::string testPath = lm.logPath() + '/' + lm.logName();
        std::filesystem::remove_all( testPath );

        REQUIRE( lm.logThreadStart() == -1 );
    }
}

/// The destructor drains any remaining queued entries before the object goes away.
/**
 * \ingroup logManager_unit_test
 */
TEST_CASE( "logManager destructor drains any remaining queued entries", "[libMagAOX::logger::logManager]" )
{
    SECTION( "an empty queue at destruction does nothing extra" )
    {
        logManagerTestT lm;
        lm.logPath( "/tmp" );

        // lm destructs immediately. The thread was never started, so it is not joinable.
        // The queue is empty, so logThreadExec() is not called again.
    }

    SECTION( "a non-empty queue at destruction is drained by the destructor" )
    {
        logManagerTestT lm;
        lm.logPath( "/tmp" );

        std::string testPath = lm.logPath() + '/' + lm.logName();
        std::filesystem::remove_all( testPath );

        lm.log<MagAOX::logger::software_error>( { __FILE__, __LINE__, 0, 0, std::string( "drained by dtor" ) } );

        REQUIRE( lm.test_queueSize() == 1 );

        // lm goes out of scope here. The thread was never started, so joinable() is false.
        // But the queue is non-empty, so the destructor calls logThreadExec() once to
        // drain it.
    }
}

} // namespace logManagerTest_ns
} // namespace loggerTest
} // namespace libXWCTest
