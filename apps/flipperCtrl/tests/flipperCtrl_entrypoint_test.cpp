/** \file flipperCtrl_entrypoint_test.cpp
 * \brief Bounded smoke test for the real flipperCtrl executable entrypoint.
 * \author MagAO-X developers
 * \ingroup flipperCtrl_unit_test
 */

#include "../../../tests/testXWC.hpp"

#include <cerrno>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>

#include <fcntl.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

namespace libXWCTest
{
namespace flipperCtrlTest
{
/** \addtogroup flipperCtrl_unit_test
 * @{ */

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
/// Own an isolated working directory and MagAO-X runtime paths.
class HelpDirectory
{
  public:
    /// Create a private directory with all runtime subdirectories.
    HelpDirectory();

    /// Remove the private tree after the child has been reaped.
    ~HelpDirectory();

    /// Private working/base directory owned by this fixture.
    std::filesystem::path m_path;
};

HelpDirectory::HelpDirectory()
{
    char  pattern[] = "/tmp/flipperCtrl-help-XXXXXX";
    char *created   = ::mkdtemp( pattern );
    REQUIRE( created != nullptr );
    m_path = created;
    for( const auto &relative : { "config", "calib", "logs", "telem", "sys", "secrets", "rawimages" } )
        std::filesystem::create_directory( m_path / relative );
}

HelpDirectory::~HelpDirectory()
{
    std::error_code error;
    std::filesystem::remove_all( m_path, error );
}

/// Run --help in a child, preserving exit status and cleaning up even after a failed assertion.
class HelpProcess
{
  public:
    /// Launch the actual executable with private paths and captured standard streams.
    HelpProcess( const std::filesystem::path &executable /**< [in] real app executable */,
                 const std::filesystem::path &directory /**< [in] child working/base directory */ );

    /// Kill and reap a child that has not completed before fixture destruction.
    ~HelpProcess();

    /// Reap the child within ten seconds, rejecting a hang rather than skipping the test.
    /** \returns The waitpid status, suitable for WIFEXITED and WEXITSTATUS. */
    int wait();

  private:
    /// Child process owned until a successful wait; -1 means already reaped.
    pid_t m_pid{ -1 };
};

HelpProcess::HelpProcess( const std::filesystem::path &executable, const std::filesystem::path &directory )
{
    int output = ::open( ( directory / "help.txt" ).c_str(), O_CREAT | O_WRONLY | O_TRUNC | O_CLOEXEC, 0600 );
    REQUIRE( output >= 0 );
    m_pid = ::fork();
    if( m_pid == 0 )
    {
        if( ::dup2( output, STDOUT_FILENO ) < 0 || ::dup2( output, STDERR_FILENO ) < 0 ||
            ::chdir( directory.c_str() ) < 0 || ::setenv( "MAGAOX_PATH", directory.c_str(), 1 ) < 0 ||
            ::setenv( "MAGAOX_CONFIG_RPATH", "config", 1 ) < 0 || ::setenv( "MAGAOX_CALIB_RPATH", "calib", 1 ) < 0 ||
            ::setenv( "MAGAOX_LOG_RPATH", "logs", 1 ) < 0 || ::setenv( "MAGAOX_TELEM_RPATH", "telem", 1 ) < 0 ||
            ::setenv( "MAGAOX_SYS_RPATH", "sys", 1 ) < 0 || ::setenv( "MAGAOX_SECRETS_RPATH", "secrets", 1 ) < 0 ||
            ::setenv( "MAGAOX_RAWIMAGE_RPATH", "rawimages", 1 ) < 0 )
            ::_exit( 126 );
        ::close( output );
        ::execl( executable.c_str(), executable.c_str(), "--help", static_cast<char *>( nullptr ) );
        ::_exit( 127 );
    }
    ::close( output );
    REQUIRE( m_pid > 0 );
}

HelpProcess::~HelpProcess()
{
    if( m_pid <= 0 )
        return;
    ::kill( m_pid, SIGKILL );
    while( ::waitpid( m_pid, nullptr, 0 ) < 0 && errno == EINTR )
    {
    }
}

int HelpProcess::wait()
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds( 10 );
    do
    {
        int   status = 0;
        pid_t result = ::waitpid( m_pid, &status, WNOHANG );
        if( result == m_pid )
        {
            m_pid = -1;
            return status;
        }
        if( result < 0 && errno != EINTR )
            FAIL( "waitpid failed while waiting for flipperCtrl --help" );
        std::this_thread::sleep_for( std::chrono::milliseconds( 10 ) );
    } while( std::chrono::steady_clock::now() < deadline );
    FAIL( "flipperCtrl --help exceeded the ten-second deadline" );
    return -1;
}
/// \endcond

/** \brief Exercise \ref flipperCtrl.cpp through the real executable's help-only exit.
 * \details Covers the constructor, main delegation, and destruction without starting the device loop.
 * \ingroup flipperCtrl_unit_test
 */
TEST_CASE( "flipper executable help exits before device startup", "[flipperCtrl][entrypoint]" )
{
    // clang-format off
#ifdef FLIPPERCTRL_TEST_DOXYGEN_REF
    XWCTEST_DOXYGEN_REF(MagAOX::app::flipperCtrl::flipperCtrl());
    XWCTEST_DOXYGEN_REF(::main(0, nullptr));
#endif
    // clang-format on
    const auto executable =
        std::filesystem::read_symlink( "/proc/self/exe" ).parent_path().parent_path() / "flipperCtrl";
    REQUIRE( std::filesystem::is_regular_file( executable ) );
    REQUIRE( ::access( executable.c_str(), X_OK ) == 0 );
    HelpDirectory directory;
    HelpProcess   child( executable, directory.m_path );
    const int     status = child.wait();
    REQUIRE( WIFEXITED( status ) );
    REQUIRE( WEXITSTATUS( status ) == 1 );
    std::ifstream     stream( directory.m_path / "help.txt" );
    const std::string help{ std::istreambuf_iterator<char>( stream ), std::istreambuf_iterator<char>() };
    REQUIRE( help.find( "flipper.reverse" ) != std::string::npos );
    REQUIRE( help.find( "device.readTimeout" ) != std::string::npos );
    REQUIRE( help.find( "device.writeTimeout" ) != std::string::npos );
    for( const auto &entry : std::filesystem::recursive_directory_iterator( directory.m_path ) )
        REQUIRE( entry.path().filename() != "position" );
}

/** @} */
} // namespace flipperCtrlTest
} // namespace libXWCTest
