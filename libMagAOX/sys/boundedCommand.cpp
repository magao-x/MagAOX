/** \file boundedCommand.cpp
 * \brief Bounded command capture using spawn, nonblocking pipes, and owned-child cleanup.
 * \ingroup sys_files
 */

#include "boundedCommand.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

extern char **environ;

namespace MagAOX
{
namespace sys
{
namespace
{

/// Close one owned descriptor and invalidate it without retrying a potentially reused descriptor.
void closeDescriptor( int &descriptor /**< [in,out] descriptor owned by this scope */ ) noexcept
{
    if( descriptor >= 0 )
    {
        close( descriptor );
        descriptor = -1;
    }
}

/// Own both capture pipes through all error paths.
struct capturePipes
{
    int m_output[2]{ -1, -1 }; ///< Parent read and child write ends of standard output.

    int m_error[2]{ -1, -1 }; ///< Parent read and child write ends of standard error.

    /// Close remaining capture descriptors.
    ~capturePipes();
};

capturePipes::~capturePipes()
{
    for( int *pipe : { m_output, m_error } )
    {
        closeDescriptor( pipe[0] );
        closeDescriptor( pipe[1] );
    }
}

/// Make close-on-exec pipes with nonblocking readers and descriptors outside the standard stream range.
bool makePipe( int ( &pipe )[2] /**< [out] owned read and write descriptors */ )
{
    if( pipe2( pipe, O_CLOEXEC ) < 0 )
    {
        return false;
    }

    for( int &descriptor : pipe )
    {
        if( descriptor <= STDERR_FILENO )
        {
            int replacement = fcntl( descriptor, F_DUPFD_CLOEXEC, STDERR_FILENO + 1 );
            closeDescriptor( descriptor );
            descriptor = replacement;

            if( descriptor < 0 )
            {
                return false;
            }
        }
    }

    return fcntl( pipe[0], F_SETFL, O_NONBLOCK ) == 0;
}

/// Reap a specific child after ownership has left a destroyed command object.
void reapDetached( pid_t pid /**< [in] sole child still owned by this cleanup thread */ ) noexcept
{
    while( waitpid( pid, nullptr, 0 ) < 0 && errno == EINTR )
    {
    }
}

} // namespace

boundedCommand::~boundedCommand() noexcept
{
    shutdown();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds( 20 );

    while( pending() && std::chrono::steady_clock::now() < deadline )
    {
        std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
    }

    if( m_pid > 0 )
    {
        try
        {
            std::thread( reapDetached, m_pid ).detach();
        }
        catch( ... )
        {
            // Under thread-resource exhaustion, leave final reaping to process exit instead of blocking.
        }
    }
}

bool boundedCommand::pending() noexcept
{
    if( m_pid <= 0 )
    {
        return false;
    }

    siginfo_t child{};

    if( waitid( P_PID, m_pid, &child, WEXITED | WNOHANG | WNOWAIT ) < 0 )
    {
        if( errno == ECHILD )
        {
            m_pid = -1;
        }

        return m_pid > 0;
    }

    kill( -m_pid, SIGKILL );
    kill( m_pid, SIGKILL );
    pid_t reaped = waitpid( m_pid, nullptr, WNOHANG );

    if( reaped == m_pid || ( reaped < 0 && errno == ECHILD ) )
    {
        m_pid = -1;
    }

    return m_pid > 0;
}

void boundedCommand::shutdown() noexcept
{
    pending();
}

boundedCommand::result boundedCommand::run( const std::vector<std::string> &argv, double timeoutSec, size_t maxOutput )
{
    result collected;

    if( pending() )
    {
        collected.m_status = status::busy;
        return collected;
    }

    // Production builds use -ffast-math, so reject nonfinite deadlines using their IEEE exponent bits.
    static_assert( sizeof( double ) == sizeof( uint64_t ) && std::numeric_limits<double>::is_iec559 );
    uint64_t timeoutBits;
    std::memcpy( &timeoutBits, &timeoutSec, sizeof( timeoutBits ) );

    if( argv.empty() || argv[0].empty() || argv[0][0] != '/' ||
        ( timeoutBits & UINT64_C( 0x7ff0000000000000 ) ) == UINT64_C( 0x7ff0000000000000 ) || timeoutSec <= 0 )
    {
        return collected;
    }

    std::vector<char *> arguments;
    arguments.reserve( argv.size() + 1 );

    for( const auto &argument : argv )
    {
        if( argument.find( '\0' ) != std::string::npos )
        {
            return collected;
        }

        arguments.push_back( const_cast<char *>( argument.c_str() ) );
    }

    arguments.push_back( nullptr );

    capturePipes pipes;

    if( !makePipe( pipes.m_output ) || !makePipe( pipes.m_error ) )
    {
        return collected;
    }

    posix_spawn_file_actions_t actions;
    posix_spawnattr_t          attributes;

    if( posix_spawn_file_actions_init( &actions ) != 0 )
    {
        return collected;
    }

    if( posix_spawnattr_init( &attributes ) != 0 )
    {
        posix_spawn_file_actions_destroy( &actions );
        return collected;
    }

    sigset_t emptySignals;
    sigset_t defaultSignals;
    sigemptyset( &emptySignals );
    sigfillset( &defaultSignals );

    int setup = posix_spawn_file_actions_addopen( &actions, STDIN_FILENO, "/dev/null", O_RDONLY, 0 );
    setup |= posix_spawn_file_actions_adddup2( &actions, pipes.m_output[1], STDOUT_FILENO );
    setup |= posix_spawn_file_actions_adddup2( &actions, pipes.m_error[1], STDERR_FILENO );

    for( int descriptor : { pipes.m_output[0], pipes.m_output[1], pipes.m_error[0], pipes.m_error[1] } )
    {
        setup |= posix_spawn_file_actions_addclose( &actions, descriptor );
    }

    setup |= posix_spawnattr_setpgroup( &attributes, 0 );
    setup |= posix_spawnattr_setsigmask( &attributes, &emptySignals );
    setup |= posix_spawnattr_setsigdefault( &attributes, &defaultSignals );
    setup |= posix_spawnattr_setflags(
        &attributes, POSIX_SPAWN_SETPGROUP | POSIX_SPAWN_RESETIDS | POSIX_SPAWN_SETSIGMASK | POSIX_SPAWN_SETSIGDEF );

    const auto started = std::chrono::steady_clock::now();
    int        spawned =
        setup != 0 ? setup : posix_spawn( &m_pid, argv[0].c_str(), &actions, &attributes, arguments.data(), environ );
    posix_spawnattr_destroy( &attributes );
    posix_spawn_file_actions_destroy( &actions );
    closeDescriptor( pipes.m_output[1] );
    closeDescriptor( pipes.m_error[1] );

    if( spawned != 0 )
    {
        m_pid = -1;
        return collected;
    }

    collected.m_status = status::success;
    size_t captured    = 0;

    while( true )
    {
        double elapsed = std::chrono::duration<double>( std::chrono::steady_clock::now() - started ).count();

        if( elapsed >= timeoutSec )
        {
            collected.m_status = status::timeout;
            break;
        }

        // Read at most one chunk per stream per pass so a continuous writer cannot starve the deadline or stderr.
        for( unsigned stream = 0; stream < 2; ++stream )
        {
            int         &descriptor = stream == 0 ? pipes.m_output[0] : pipes.m_error[0];
            std::string &output     = stream == 0 ? collected.m_output : collected.m_error;

            if( descriptor < 0 )
            {
                continue;
            }

            char    buffer[4096];
            ssize_t count = read( descriptor, buffer, sizeof( buffer ) );

            if( count > 0 )
            {
                size_t accepted = std::min( size_t( count ), maxOutput - captured );
                output.append( buffer, accepted );
                captured += accepted;

                if( accepted != size_t( count ) )
                {
                    collected.m_status = status::outputLimit;
                    break;
                }
            }
            else if( count == 0 )
            {
                closeDescriptor( descriptor );
            }
            else if( errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR )
            {
                collected.m_status = status::ioError;
                break;
            }
        }

        if( collected.m_status != status::success )
        {
            break;
        }

        if( pipes.m_output[0] < 0 && pipes.m_error[0] < 0 )
        {
            // Leave an exited leader unreaped until group cleanup, preserving its PID against reuse.
            siginfo_t child{};

            if( waitid( P_PID, m_pid, &child, WEXITED | WNOHANG | WNOWAIT ) < 0 )
            {
                if( errno == EINTR )
                {
                    continue;
                }

                collected.m_status = status::ioError;
                break;
            }

            if( child.si_pid == m_pid )
            {
                collected.m_exitCode = child.si_status;
                collected.m_status   = child.si_code == CLD_EXITED
                                           ? ( child.si_status == 0 ? status::success : status::exitError )
                                           : status::signalError;
                break;
            }
        }

        elapsed          = std::chrono::duration<double>( std::chrono::steady_clock::now() - started ).count();
        double remaining = timeoutSec - elapsed;

        if( remaining <= 0 )
        {
            continue;
        }

        pollfd descriptors[2]{ { pipes.m_output[0], POLLIN, 0 }, { pipes.m_error[0], POLLIN, 0 } };
        int    milliseconds = int( std::ceil( std::min( remaining, 0.01 ) * 1000 ) );

        if( poll( descriptors, 2, milliseconds ) < 0 && errno != EINTR )
        {
            collected.m_status = status::ioError;
            break;
        }
    }

    closeDescriptor( pipes.m_output[0] );
    closeDescriptor( pipes.m_error[0] );
    shutdown();
    return collected;
}

} // namespace sys
} // namespace MagAOX
