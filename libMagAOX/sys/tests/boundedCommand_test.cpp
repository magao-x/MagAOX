/** \file boundedCommand_test.cpp
 * \brief Offline command capture, timeout, process ownership, and cleanup tests.
 * \ingroup boundedCommand_unit_test
 */

#include "../../../tests/testXWC.hpp"
#include "../boundedCommand.hpp"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <limits>
#include <pthread.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

namespace libXWCTest
{
namespace boundedCommandTest
{

using MagAOX::sys::boundedCommand;

/// \cond TEST_HARNESS

/// Wait a bounded time for the owner to reap a normally interruptible fake child.
bool reapSoon( boundedCommand &command /**< [in,out] owner of a killed fake process */ )
{
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds( 1 );

    while( command.pending() && std::chrono::steady_clock::now() < deadline )
    {
        std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
    }

    return !command.pending();
}

/// Receive an interrupt without requesting restart of an interrupted blocking call.
void interruptWait( int signal /**< [in] unused test signal */ )
{
    static_cast<void>( signal );
}

/// Restore the test process signal handler even when an assertion fails.
class signalScope
{
  public:
    /// Install a harmless SIGUSR1 handler for interrupted-poll coverage.
    signalScope();

    /// Restore the prior signal handler.
    ~signalScope();

  private:
    struct sigaction m_previous{}; ///< Handler restored when the interrupt fixture leaves scope.
};

signalScope::signalScope()
{
    struct sigaction action{};
    action.sa_handler = interruptWait;
    sigemptyset( &action.sa_mask );
    REQUIRE( sigaction( SIGUSR1, &action, &m_previous ) == 0 );
}

signalScope::~signalScope()
{
    sigaction( SIGUSR1, &m_previous, nullptr );
}

/// \endcond

/// Preserve argv boundaries, embedded newlines, and both independent output streams.
/** \ingroup boundedCommand_unit_test */
TEST_CASE( "Bounded command preserves arguments and both output streams", "[boundedCommand]" )
{
    boundedCommand command;
    auto           result =
        command.run( { "/bin/sh", "-c", "printf '%s' \"$1\"; printf 'error\\n' >&2", "fake", "a b\n$(false)" }, 1 );
    REQUIRE( result.m_status == boundedCommand::status::success );
    REQUIRE( result.m_exitCode == 0 );
    REQUIRE( result.m_output == "a b\n$(false)" );
    REQUIRE( result.m_error == "error\n" );
    REQUIRE_FALSE( command.pending() );
}

/// Distinguish failed spawn, invalid input, child exit, and signal termination.
/** \ingroup boundedCommand_unit_test */
TEST_CASE( "Bounded command reports child and spawn failures", "[boundedCommand]" )
{
    boundedCommand command;
    REQUIRE( command.run( { "/no/such/bounded-command-executable" }, 1 ).m_status ==
             boundedCommand::status::spawnError );
    REQUIRE( command.run( {}, 1 ).m_status == boundedCommand::status::spawnError );
    REQUIRE( command.run( { "sh" }, 1 ).m_status == boundedCommand::status::spawnError );
    REQUIRE( command.run( { "/bin/true" }, 0 ).m_status == boundedCommand::status::spawnError );
    REQUIRE( command.run( { "/bin/true" }, std::numeric_limits<double>::infinity() ).m_status ==
             boundedCommand::status::spawnError );
    REQUIRE( command.run( { "/bin/true" }, std::numeric_limits<double>::quiet_NaN() ).m_status ==
             boundedCommand::status::spawnError );
    REQUIRE( command.run( { "/bin/true", std::string( "a\0b", 3 ) }, 1 ).m_status ==
             boundedCommand::status::spawnError );

    auto exited = command.run( { "/bin/sh", "-c", "printf failed >&2; exit 17" }, 1 );
    REQUIRE( exited.m_status == boundedCommand::status::exitError );
    REQUIRE( exited.m_exitCode == 17 );
    REQUIRE( exited.m_error == "failed" );

    auto signaled = command.run( { "/bin/sh", "-c", "kill -TERM $$" }, 1 );
    REQUIRE( signaled.m_status == boundedCommand::status::signalError );
    REQUIRE( signaled.m_exitCode == SIGTERM );
}

/// Drain both pipes before waiting even when their combined output greatly exceeds pipe capacity.
/** \ingroup boundedCommand_unit_test */
TEST_CASE( "Bounded command drains large output without pipe deadlock", "[boundedCommand]" )
{
    boundedCommand command;
    auto           result = command.run( { "/bin/sh",
                                           "-c",
                                           "i=0; while [ $i -lt 4000 ]; do printf '12345678901234567890123456789012'; printf "
                                                     "'abcdefghijklmnopqrstuvwxyzABCDEF' >&2; i=$((i+1)); done" },
                               3,
                               256000 );
    REQUIRE( result.m_status == boundedCommand::status::success );
    REQUIRE( result.m_output.size() == 128000 );
    REQUIRE( result.m_error.size() == 128000 );
    REQUIRE( result.m_output.substr( 127968 ) == "12345678901234567890123456789012" );
    REQUIRE( result.m_error.substr( 127968 ) == "abcdefghijklmnopqrstuvwxyzABCDEF" );
}

/// Enforce a combined stream limit, including exact-limit and empty-output boundary cases.
/** \ingroup boundedCommand_unit_test */
TEST_CASE( "Bounded command limits combined output", "[boundedCommand]" )
{
    boundedCommand command;
    auto           exact = command.run( { "/bin/sh", "-c", "printf abc; printf def >&2" }, 1, 6 );
    REQUIRE( exact.m_status == boundedCommand::status::success );

    auto exceeded = command.run( { "/bin/sh", "-c", "printf abc; printf def >&2" }, 1, 5 );
    REQUIRE( exceeded.m_status == boundedCommand::status::outputLimit );
    REQUIRE( exceeded.m_output.size() + exceeded.m_error.size() == 5 );
    REQUIRE( reapSoon( command ) );
    REQUIRE( command.run( { "/bin/true" }, 1, 0 ).m_status == boundedCommand::status::success );
    REQUIRE( command.run( { "/bin/sh", "-c", "printf x" }, 1, 0 ).m_status == boundedCommand::status::outputLimit );
    REQUIRE( reapSoon( command ) );
}

/// Bound sleeping commands, continuous output, and descendants that retain their parent's pipes.
/** \ingroup boundedCommand_unit_test */
TEST_CASE( "Bounded command deadlines survive continuous output and held pipes", "[boundedCommand]" )
{
    for( const std::string script :
         { "sleep 10", "while :; do printf data; done", "sleep 10 & exit 0", "exec 1>&- 2>&-; sleep 10" } )
    {
        boundedCommand command;
        auto           started = std::chrono::steady_clock::now();
        auto           result  = command.run( { "/bin/sh", "-c", script }, 0.05, 64 * 1024 * 1024 );
        double         elapsed = std::chrono::duration<double>( std::chrono::steady_clock::now() - started ).count();
        INFO( script );
        REQUIRE( result.m_status == boundedCommand::status::timeout );
        REQUIRE( elapsed < 1 );
        REQUIRE( reapSoon( command ) );
        REQUIRE( command.run( { "/bin/true" }, 1 ).m_status == boundedCommand::status::success );
    }
}

/// Repeated timeouts leave no waitable direct children and never reap another caller's child.
/** \ingroup boundedCommand_unit_test */
TEST_CASE( "Bounded command retains exclusive ownership of its children", "[boundedCommand]" )
{
    pid_t unrelated = fork();
    REQUIRE( unrelated >= 0 );

    if( unrelated == 0 )
    {
        _exit( 23 );
    }

    boundedCommand command;

    for( unsigned iteration = 0; iteration < 10; ++iteration )
    {
        auto timedOut = command.run( { "/bin/sh", "-c", "sleep 10" }, 0.01 );
        REQUIRE( timedOut.m_status == boundedCommand::status::timeout );
        REQUIRE( reapSoon( command ) );
    }

    int status = 0;
    REQUIRE( waitpid( unrelated, &status, 0 ) == unrelated );
    REQUIRE( WIFEXITED( status ) );
    REQUIRE( WEXITSTATUS( status ) == 23 );
    REQUIRE( waitpid( -1, nullptr, WNOHANG ) == -1 );
    REQUIRE( errno == ECHILD );
}

/// Interrupted polling continues until command completion rather than discarding a valid result.
/** \ingroup boundedCommand_unit_test */
TEST_CASE( "Bounded command tolerates interrupted waits", "[boundedCommand]" )
{
    signalScope       signals;
    pthread_t         mainThread = pthread_self();
    std::atomic<bool> stop{ false };
    std::thread       interrupter(
        [&]()
        {
            while( !stop )
            {
                pthread_kill( mainThread, SIGUSR1 );
                std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
            }
        } );
    boundedCommand command;
    auto           result = command.run( { "/bin/sh", "-c", "sleep 0.05; printf done" }, 1 );
    stop                  = true;
    interrupter.join();
    REQUIRE( result.m_status == boundedCommand::status::success );
    REQUIRE( result.m_output == "done" );
}

/// Destruction after a timeout promptly reaps an ordinary interruptible child.
/** \ingroup boundedCommand_unit_test */
TEST_CASE( "Bounded command destruction cleans up a timed-out child", "[boundedCommand]" )
{
    auto started = std::chrono::steady_clock::now();

    {
        boundedCommand command;
        REQUIRE( command.run( { "/bin/sh", "-c", "sleep 10" }, 0.01 ).m_status == boundedCommand::status::timeout );
    }

    REQUIRE( std::chrono::duration<double>( std::chrono::steady_clock::now() - started ).count() < 1 );
    REQUIRE( waitpid( -1, nullptr, WNOHANG ) == -1 );
    REQUIRE( errno == ECHILD );
}

} // namespace boundedCommandTest
} // namespace libXWCTest
