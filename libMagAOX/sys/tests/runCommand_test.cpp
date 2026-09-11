/** \file runCommand_test.cpp
  * \brief Catch2 tests for MagAOX::sys::runCommand().
  *
  * No mocks are used. The success tests fork and exec real programs such as /bin/echo and
  * read their output back through real pipes. The failure tests lower the process resource
  * limits with setrlimit() so that pipe() and fork() fail for real inside runCommand(). The
  * original limits are restored before each assertion. The fork failure test relies on
  * RLIMIT_NPROC, which the kernel does not enforce for the root user, so that test is only
  * meaningful when run as an unprivileged user.
  */
#include "../../../tests/catch2/catch.hpp"

#include <sys/resource.h>

#include "../runCommand.hpp"

namespace libXWCTest
{
namespace sysTest
{

// Verifies that runCommand() runs a real child process and splits its standard output and
// standard error into separate line vectors.
TEST_CASE( "runCommand runs a command and captures its output", "[libMagAOX::sys::runCommand]" )
{
    SECTION( "captures stdout" )
    {
        std::vector<std::string> out, err;
        std::vector<std::string> cmd{ "/bin/echo", "hello", "world" };

        int rv = MagAOX::sys::runCommand( out, err, cmd );

        REQUIRE( rv == 0 );
        REQUIRE( out.size() == 1 );
        REQUIRE( out[0] == "hello world" );
        REQUIRE( err.size() == 0 );
    }

    SECTION( "captures stderr" )
    {
        std::vector<std::string> out, err;
        // ls on a nonexistent path execs fine but writes its complaint to stderr.
        std::vector<std::string> cmd{ "/bin/ls", "/no/such/path/xwctest" };

        int rv = MagAOX::sys::runCommand( out, err, cmd );

        REQUIRE( rv == 0 );
        REQUIRE( out.size() == 0 );
        REQUIRE( err.size() == 1 );
    }

    SECTION( "captures multiple lines of stdout" )
    {
        std::vector<std::string> out, err;
        std::vector<std::string> cmd{ "/bin/printf", "one\ntwo\nthree\n" };

        int rv = MagAOX::sys::runCommand( out, err, cmd );

        REQUIRE( rv == 0 );
        REQUIRE( out.size() == 3 );
        REQUIRE( out[0] == "one" );
        REQUIRE( out[1] == "two" );
        REQUIRE( out[2] == "three" );
    }
}

// Verifies the pipe creation error paths by lowering RLIMIT_NOFILE so that the first or the
// second pipe() call inside runCommand() fails for real. The limit is restored before the
// assertions so Catch2 can still write its report.
TEST_CASE( "runCommand reports an error when it can't create its pipes", "[libMagAOX::sys::runCommand]" )
{
    struct rlimit orig;
    REQUIRE( getrlimit( RLIMIT_NOFILE, &orig ) == 0 );

    SECTION( "the stdout pipe fails" )
    {
        struct rlimit tiny;
        tiny.rlim_cur = 3; // Only stdin, stdout, and stderr fit. The first pipe cannot open.
        tiny.rlim_max = orig.rlim_max;
        REQUIRE( setrlimit( RLIMIT_NOFILE, &tiny ) == 0 );

        std::vector<std::string> out, err;
        std::vector<std::string> cmd{ "/bin/echo", "unreachable" };
        int                      rv = MagAOX::sys::runCommand( out, err, cmd );

        REQUIRE( setrlimit( RLIMIT_NOFILE, &orig ) == 0 );

        REQUIRE( rv == -1 );
        REQUIRE( out.size() == 1 );
    }

    SECTION( "the stderr pipe fails" )
    {
        struct rlimit tiny;
        tiny.rlim_cur = 5; // Exactly the first pipe's two descriptors fit. The second pipe cannot open.
        tiny.rlim_max = orig.rlim_max;
        REQUIRE( setrlimit( RLIMIT_NOFILE, &tiny ) == 0 );

        std::vector<std::string> out, err;
        std::vector<std::string> cmd{ "/bin/echo", "unreachable" };
        int                      rv = MagAOX::sys::runCommand( out, err, cmd );

        REQUIRE( setrlimit( RLIMIT_NOFILE, &orig ) == 0 );

        REQUIRE( rv == -1 );
        REQUIRE( out.size() == 1 );
    }
}

// Verifies the fork() error path by lowering RLIMIT_NPROC below the number of processes this
// user already owns, so the fork inside runCommand() fails for real. The limit is restored
// before the assertions.
TEST_CASE( "runCommand reports an error when fork() fails", "[libMagAOX::sys::runCommand]" )
{
    struct rlimit orig;
    REQUIRE( getrlimit( RLIMIT_NPROC, &orig ) == 0 );

    struct rlimit tiny;
    tiny.rlim_cur = 1; // This process alone already reaches the limit, so the next fork() fails.
    tiny.rlim_max = orig.rlim_max;
    REQUIRE( setrlimit( RLIMIT_NPROC, &tiny ) == 0 );

    std::vector<std::string> out, err;
    std::vector<std::string> cmd{ "/bin/echo", "unreachable" };
    int                      rv = MagAOX::sys::runCommand( out, err, cmd );

    REQUIRE( setrlimit( RLIMIT_NPROC, &orig ) == 0 );

    REQUIRE( rv == -1 );
    REQUIRE( out.size() == 1 );
}

} // namespace sysTest
} // namespace libXWCTest
