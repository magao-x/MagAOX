/** \file cacaoWriter.cpp
 * \brief Observe the installed milk FPS ABI without a Python trainer or writable mapping.
 */
#include "cacaoWriter.hpp"

#include <cerrno>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <fcntl.h>
#include <signal.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef NNRECONSTRUCTOR_TEST
    #include "tests/cacaoWriterTestAbi.hpp"
#else
extern "C"
{
    #include <CLIcore.h>
}
#endif

namespace MagAOX::app
{

cacaoWriter::cacaoWriter( const std::string &shmDirectory, const std::string &procDirectory )
    : m_shmDirectory( shmDirectory ), m_procDirectory( procDirectory )
{
}

bool cacaoWriter::running()
{
    FUNCTION_PARAMETER_STRUCT_MD md{};
    struct stat                  info{};
    const auto                   path = m_shmDirectory + "/" + m_name + ".fps.shm";
    const int                    fd   = open( path.c_str(), O_RDONLY | O_CLOEXEC );
    if( fd < 0 )
        throw std::runtime_error( "Cannot read CACAO FPS: " + path );
    const bool valid =
        fstat( fd, &info ) == 0 && S_ISREG( info.st_mode ) && pread( fd, &md, sizeof( md ), 0 ) == sizeof( md );
    close( fd );
    const auto metadataSize  = static_cast<off_t>( sizeof( md ) );
    const auto parameterSize = static_cast<off_t>( sizeof( FUNCTION_PARAMETER ) );
    if( !valid || std::string( md.name, strnlen( md.name, sizeof( md.name ) ) ) != m_name || md.NBparamMAX <= 0 ||
        info.st_size < metadataSize || md.NBparamMAX != ( info.st_size - metadataSize ) / parameterSize ||
        ( info.st_size - metadataSize ) % parameterSize != 0 )
        throw std::runtime_error( "CACAO FPS layout does not match installed milk headers" );

    // Match milk RUNrunning: a missing process is stopped only after CMDRUN clears.
    if( md.runpid <= 0 || ( getpgid( md.runpid ) < 0 && errno == ESRCH ) )
    {
        if( md.status & FUNCTION_PARAMETER_STRUCT_STATUS_CMDRUN )
            throw std::runtime_error( "CACAO RUN start/stop is unresolved" );
        m_identity.clear();
        return false;
    }
    const auto    proc = m_procDirectory + "/" + std::to_string( md.runpid );
    std::ifstream statFile( proc + "/stat" ), cmdFile( proc + "/cmdline" );
    std::string   line, command;
    std::getline( statFile, line );
    std::getline( cmdFile, command, '\xff' );
    const auto end = line.rfind( ')' );
    if( !statFile || !cmdFile || end == std::string::npos || m_processMatch.empty() ||
        command.find( m_processMatch ) == std::string::npos )
        throw std::runtime_error( "CACAO RUN PID does not identify the configured writer" );
    std::istringstream fields( line.substr( end + 1 ) );
    std::string        state, ticks;
    fields >> state;
    for( int field = 4; field <= 22; ++field )
        fields >> ticks;
    if( !fields || state == "Z" || ticks.empty() )
        throw std::runtime_error( "CACAO RUN process state is unavailable" );
    const auto identity = std::to_string( info.st_dev ) + ":" + std::to_string( info.st_ino ) + ":" +
                          std::to_string( md.runpid ) + ":" + std::to_string( md.runpidstarttime.tv_sec ) + ":" +
                          std::to_string( md.runpidstarttime.tv_nsec );
    if( identity == m_identity && ticks != m_startTicks )
        throw std::runtime_error( "CACAO RUN PID was reused" );
    m_identity   = identity;
    m_startTicks = ticks;
    return true;
}

void cacaoWriter::command( bool enabled )
{
    const int fd = open( m_fifo.c_str(), O_WRONLY | O_NONBLOCK | O_CLOEXEC );
    if( fd < 0 )
        throw std::runtime_error( "CACAO controller FIFO is unavailable" );
    struct stat info{};
    if( fstat( fd, &info ) != 0 || !S_ISFIFO( info.st_mode ) )
    {
        close( fd );
        throw std::runtime_error( "CACAO controller path is not a FIFO" );
    }
    const auto request = std::string( enabled ? "runstart " : "runstop " ) + m_name + "\n";
    // A disappearing reader must report a failed command rather than kill inference.
    sigset_t blocked, previous, pending;
    sigemptyset( &blocked );
    sigaddset( &blocked, SIGPIPE );
    if( pthread_sigmask( SIG_BLOCK, &blocked, &previous ) != 0 )
    {
        close( fd );
        throw std::runtime_error( "Cannot protect FIFO delivery from SIGPIPE" );
    }
    sigpending( &pending );
    const auto written = write( fd, request.data(), request.size() );
    if( written < 0 && errno == EPIPE && !sigismember( &pending, SIGPIPE ) )
    {
        sigpending( &pending );
        if( sigismember( &pending, SIGPIPE ) )
        {
            int signal;
            sigwait( &blocked, &signal );
        }
    }
    pthread_sigmask( SIG_SETMASK, &previous, nullptr );
    close( fd );
    if( written != static_cast<ssize_t>( request.size() ) )
        throw std::runtime_error( "CACAO FIFO request was not delivered" );
}

} // namespace MagAOX::app
