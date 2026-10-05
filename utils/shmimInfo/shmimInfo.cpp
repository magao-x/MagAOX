/** \file shmimInfo.cpp
 * \brief The shmimInfo main program and implementation.
 *
 * \ingroup shmimInfo_files
 *
 * \author Codex
 */

#include "shmimInfo.hpp"

#include <cerrno>
#include <csignal>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <semaphore.h>
#include <sys/stat.h>
#include <unistd.h>

namespace
{

volatile sig_atomic_t g_timeToDie = 0;

/// Handle termination signals by requesting a clean shutdown.
void sigTermHandler( int        signum /**< [in] received signal number. */,
                     siginfo_t *siginf /**< [in] unused signal metadata. */,
                     void      *ucont /**< [in] unused signal context. */ )
{
    static_cast<void>( signum );
    static_cast<void>( siginf );
    static_cast<void>( ucont );

    std::cerr << "\n";

    g_timeToDie = 1;
}

/// Install the signal handlers used by this utility.
int installSignalHandlers( const std::string &invokedName /**< [in] utility name for error reporting. */ )
{
    struct sigaction act;
    sigset_t         set;

    act.sa_sigaction = sigTermHandler;
    act.sa_flags     = SA_SIGINFO;
    sigemptyset( &set );
    act.sa_mask = set;

    errno = 0;
    if( sigaction( SIGTERM, &act, nullptr ) < 0 )
    {
        std::cerr << " (" << invokedName << "): error setting SIGTERM handler: " << strerror( errno ) << "\n";
        return -1;
    }

    errno = 0;
    if( sigaction( SIGQUIT, &act, nullptr ) < 0 )
    {
        std::cerr << " (" << invokedName << "): error setting SIGQUIT handler: " << strerror( errno ) << "\n";
        return -1;
    }

    errno = 0;
    if( sigaction( SIGINT, &act, nullptr ) < 0 )
    {
        std::cerr << " (" << invokedName << "): error setting SIGINT handler: " << strerror( errno ) << "\n";
        return -1;
    }

    return 0;
}

} // namespace

shmimInfo::shmimInfo() = default;

shmimInfo::~shmimInfo()
{
    closeStream();
}

void shmimInfo::setupConfig()
{
    config.add( "shmimName",
                "n",
                "shmimName",
                argType::Required,
                "",
                "shmimName",
                false,
                "string",
                "The name of the shared memory image stream to inspect." );

    config.add( "nFrames",
                "N",
                "nFrames",
                argType::Required,
                "",
                "nFrames",
                false,
                "int",
                "The number of frame arrivals to time when measuring FPS. Default is 100." );

    config.add( "timeout",
                "t",
                "timeout",
                argType::Required,
                "",
                "timeout",
                false,
                "float",
                "The per-frame wait timeout in seconds. Default is 1.0 second." );
}

void shmimInfo::loadConfig()
{
    config( m_shmimName, "shmimName" );
    config( m_nFrames, "nFrames" );
    config( m_timeoutSec, "timeout" );

    if( m_shmimName.empty() )
    {
        std::cerr << "shmim name not specified with -n\n";
        doHelp = true;
        return;
    }

    if( m_nFrames < 2 )
    {
        std::cerr << "nFrames must be at least 2 to measure frame rate\n";
        doHelp = true;
        return;
    }

    if( m_timeoutSec <= 0 )
    {
        std::cerr << "timeout must be greater than 0 seconds\n";
        doHelp = true;
        return;
    }
}

int shmimInfo::execute()
{
    if( installSignalHandlers( invokedName ) < 0 )
    {
        return -1;
    }

    if( openStream() < 0 )
    {
        closeStream();
        return -1;
    }

    std::cout << "shmim: " << m_shmimName << "\n";
    std::cout << "size: " << m_width << " " << m_height << " " << m_depth << "\n";

    int rv = measureFrames();
    closeStream();

    return rv;
}

int shmimInfo::openStream()
{
    int logged = 0;

    while( !g_timeToDie )
    {
        int  SM_fd;
        char SM_fname[200];
        ImageStreamIO_filename( SM_fname, sizeof( SM_fname ), m_shmimName.c_str() );

        SM_fd = open( SM_fname, O_RDWR );
        if( SM_fd == -1 )
        {
            if( !logged )
            {
                std::cerr << "ImageStream " << m_shmimName << " not found (yet). Retrying . . .\n";
            }

            logged = 1;
            sleep( 1 );
            continue;
        }

        logged = 0;
        close( SM_fd );

        if( ImageStreamIO_openIm( &m_imageStream, m_shmimName.c_str() ) != 0 )
        {
            mx::sys::sleep( 1 );
            continue;
        }

        if( m_imageStream.md[0].sem <= m_semaphoreNumber )
        {
            ImageStreamIO_closeIm( &m_imageStream );
            mx::sys::sleep( 1 );
            continue;
        }

        struct stat buffer;
        if( stat( SM_fname, &buffer ) != 0 )
        {
            std::cerr << "Could not get inode for " << m_shmimName << ".\n";
            ImageStreamIO_closeIm( &m_imageStream );
            return -1;
        }

        m_inode  = buffer.st_ino;
        m_opened = true;

        m_width  = m_imageStream.md[0].size[0];
        m_height = 1;
        m_depth  = 1;

        if( m_imageStream.md[0].naxis > 1 )
        {
            m_height = m_imageStream.md[0].size[1];
        }

        if( m_imageStream.md[0].naxis > 2 )
        {
            m_depth = m_imageStream.md[0].size[2];
        }

        return 0;
    }

    return -1;
}

void shmimInfo::closeStream()
{
    if( !m_opened )
    {
        return;
    }

    if( m_semaphoreNumber >= 0 )
    {
        m_imageStream.semReadPID[m_semaphoreNumber] = 0;
    }

    ImageStreamIO_closeIm( &m_imageStream );
    m_opened = false;
}

int shmimInfo::measureFrames()
{
    m_semaphoreNumber = ImageStreamIO_getsemwaitindex( &m_imageStream, m_semaphoreNumber );
    if( m_semaphoreNumber < 0 )
    {
        std::cerr << "No valid semaphore found for " << m_shmimName << ".\n";
        return -1;
    }

    ImageStreamIO_semflush( &m_imageStream, m_semaphoreNumber );

    sem_t *sem = m_imageStream.semptr[m_semaphoreNumber];

    timespec t0{};
    timespec t1{};

    for( size_t n = 0; n < m_nFrames; ++n )
    {
        timespec ts;

        if( clock_gettime( CLOCK_REALTIME, &ts ) < 0 )
        {
            std::cerr << "error from clock_gettime\n";
            return -1;
        }

        ts.tv_sec += static_cast<time_t>( m_timeoutSec );
        ts.tv_nsec += static_cast<long>( ( m_timeoutSec - static_cast<time_t>( m_timeoutSec ) ) * 1e9 );
        if( ts.tv_nsec >= 1000000000L )
        {
            ++ts.tv_sec;
            ts.tv_nsec -= 1000000000L;
        }

        if( sem_timedwait( sem, &ts ) != 0 )
        {
            if( errno == ETIMEDOUT )
            {
                std::cerr << "Timed out waiting for frame " << ( n + 1 ) << " of " << m_nFrames << ".\n";
            }
            else if( errno == EINTR && g_timeToDie )
            {
                std::cerr << "Interrupted while waiting for frames.\n";
            }
            else
            {
                std::cerr << "error from sem_timedwait: " << strerror( errno ) << "\n";
            }

            char SM_fname[200];
            ImageStreamIO_filename( SM_fname, sizeof( SM_fname ), m_shmimName.c_str() );

            struct stat buffer;
            if( stat( SM_fname, &buffer ) != 0 || buffer.st_ino != m_inode )
            {
                std::cerr << "The shmim changed while waiting for frames.\n";
            }

            return -1;
        }

        if( clock_gettime( CLOCK_MONOTONIC, n == 0 ? &t0 : &t1 ) < 0 )
        {
            std::cerr << "error from clock_gettime\n";
            return -1;
        }
    }

    double elapsed = elapsedSeconds( t0, t1 );
    if( elapsed <= 0 )
    {
        std::cerr << "Measured non-positive elapsed time.\n";
        return -1;
    }

    double fps = static_cast<double>( m_nFrames - 1 ) / elapsed;

    std::cout << std::fixed << std::setprecision( 6 );
    std::cout << "timed_frames: " << m_nFrames << "\n";
    std::cout << "elapsed_sec: " << elapsed << "\n";
    std::cout << "fps: " << fps << "\n";

    return 0;
}

double shmimInfo::elapsedSeconds( const timespec &t0, const timespec &t1 )
{
    return static_cast<double>( t1.tv_sec - t0.tv_sec ) + 1e-9 * static_cast<double>( t1.tv_nsec - t0.tv_nsec );
}

int main( int argc, char **argv )
{
    shmimInfo si;

    return si.main( argc, argv );
}
