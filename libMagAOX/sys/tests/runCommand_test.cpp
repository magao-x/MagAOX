/** \file runCommand_test.cpp
 * \brief Verify legacy command capture coexists with other owned child processes.
 * \ingroup runCommand_unit_test
 */

#include "../../../tests/testXWC.hpp"
#include "../runCommand.hpp"
#include "../boundedCommand.hpp"

#include <cerrno>
#include <chrono>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

namespace libXWCTest
{
namespace runCommandTest
{

/// \cond TEST_HARNESS
/// Own an exited child that command helpers must leave available to its actual owner.
class unrelatedChild
{
  public:
    pid_t m_pid{ -1 }; ///< Sole child whose exit status belongs to this fixture.

    /// Leave one known exited child unreaped before invoking a command helper.
    unrelatedChild();

    /// Reap the fixture's child if the test did not already consume its status.
    ~unrelatedChild();
};

unrelatedChild::unrelatedChild()
{
    m_pid = fork();

    if( m_pid == 0 )
    {
        _exit( 23 );
    }

    REQUIRE( m_pid > 0 );
    siginfo_t child{};
    int       result;

    do
    {
        result = waitid( P_PID, m_pid, &child, WEXITED | WNOWAIT );
    } while( result < 0 && errno == EINTR );

    REQUIRE( result == 0 );
    REQUIRE( child.si_pid == m_pid );
}

unrelatedChild::~unrelatedChild()
{
    while( waitpid( m_pid, nullptr, 0 ) < 0 && errno == EINTR )
    {
    }
}
/// \endcond

/// A command preserves its output without consuming another owner's already available exit status.
/** \ingroup runCommand_unit_test */
TEST_CASE( "Legacy commands only reap their own child", "[runCommand]" )
{
    unrelatedChild           other;
    std::vector<std::string> output, errors;
    std::vector<std::string> command{ "/bin/sh", "-c", "printf 'output\\n'; printf 'error\\n' >&2" };
    REQUIRE( MagAOX::sys::runCommand( output, errors, command ) == 0 );
    REQUIRE( output == std::vector<std::string>{ "output" } );
    REQUIRE( errors == std::vector<std::string>{ "error" } );
    int status = 0;
    REQUIRE( waitpid( other.m_pid, &status, WNOHANG ) == other.m_pid );
    REQUIRE( WIFEXITED( status ) );
    REQUIRE( WEXITSTATUS( status ) == 23 );
}

/// Legacy collection after a bounded timeout cannot reap a different caller's child or disrupt cleanup.
/** \ingroup runCommand_unit_test */
TEST_CASE( "Legacy commands coexist with timed-out bounded commands", "[runCommand]" )
{
    unrelatedChild              other;
    MagAOX::sys::boundedCommand bounded;
    auto                        result = bounded.run( { "/bin/sh", "-c", "sleep 10" }, 0.01 );
    REQUIRE( result.m_status == MagAOX::sys::boundedCommand::status::timeout );

    std::vector<std::string> output, errors;
    std::vector<std::string> command{ "/bin/echo", "after-timeout" };
    REQUIRE( MagAOX::sys::runCommand( output, errors, command ) == 0 );
    REQUIRE( output == std::vector<std::string>{ "after-timeout" } );
    REQUIRE( errors.empty() );

    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds( 1 );

    while( bounded.pending() && std::chrono::steady_clock::now() < deadline )
    {
        std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
    }

    REQUIRE_FALSE( bounded.pending() );
    int status = 0;
    REQUIRE( waitpid( other.m_pid, &status, WNOHANG ) == other.m_pid );
    REQUIRE( WIFEXITED( status ) );
    REQUIRE( WEXITSTATUS( status ) == 23 );
}

} // namespace runCommandTest
} // namespace libXWCTest
