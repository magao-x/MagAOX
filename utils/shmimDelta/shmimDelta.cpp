/** \file shmimDelta.cpp
 * \brief The shmimDelta main program and implementation.
 *
 * \ingroup shmimDelta_files
 *
 * \author Codex
 */

#include "shmimDelta.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <csignal>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sys/stat.h>
#include <thread>
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

shmimDelta::shmimDelta() = default;

shmimDelta::~shmimDelta()
{
    closeStreams();
}

void shmimDelta::setupConfig()
{
    config.add( "shmimName1",
                "1",
                "shmimName1",
                argType::Required,
                "",
                "shmimName1",
                false,
                "string",
                "The name of the first shared memory image stream. Delta is shmimName2 minus shmimName1." );

    config.add( "shmimName2",
                "2",
                "shmimName2",
                argType::Required,
                "",
                "shmimName2",
                false,
                "string",
                "The name of the second shared memory image stream. Delta is shmimName2 minus shmimName1." );

    config.add( "nFrames",
                "N",
                "nFrames",
                argType::Required,
                "",
                "nFrames",
                false,
                "int",
                "The number of paired semaphore arrivals to measure. Default is 100." );

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

void shmimDelta::loadConfig()
{
    config( m_shmimName1, "shmimName1" );
    config( m_shmimName2, "shmimName2" );
    config( m_nFrames, "nFrames" );
    config( m_timeoutSec, "timeout" );

    if( m_shmimName1.empty() )
    {
        std::cerr << "first shmim name not specified with --shmimName1\n";
        doHelp = true;
        return;
    }

    if( m_shmimName2.empty() )
    {
        std::cerr << "second shmim name not specified with --shmimName2\n";
        doHelp = true;
        return;
    }

    if( m_nFrames < 2 )
    {
        std::cerr << "nFrames must be at least 2 to measure delta rms\n";
        doHelp = true;
        return;
    }

    if( m_timeoutSec <= 0 )
    {
        std::cerr << "timeout must be greater than 0 seconds\n";
        doHelp = true;
        return;
    }

    m_stream1.m_name = m_shmimName1;
    m_stream2.m_name = m_shmimName2;
}

int shmimDelta::execute()
{
    if( installSignalHandlers( invokedName ) < 0 )
    {
        return -1;
    }

    if( openStream( m_stream1 ) < 0 || openStream( m_stream2 ) < 0 )
    {
        closeStreams();
        return -1;
    }

    std::cout << "shmim1: " << m_stream1.m_name << "\n";
    std::cout << "size1: " << m_stream1.m_width << " " << m_stream1.m_height << " " << m_stream1.m_depth << "\n";
    std::cout << "shmim2: " << m_stream2.m_name << "\n";
    std::cout << "size2: " << m_stream2.m_width << " " << m_stream2.m_height << " " << m_stream2.m_depth << "\n";

    int rv = measureDeltas();
    closeStreams();

    return rv;
}

double shmimDelta::elapsedUsec( const timespec &t0, const timespec &t1 )
{
    return 1e6 * static_cast<double>( t1.tv_sec - t0.tv_sec ) + 1e-3 * static_cast<double>( t1.tv_nsec - t0.tv_nsec );
}

double shmimDelta::mean( const std::vector<double> &values )
{
    if( values.empty() )
    {
        return 0;
    }

    return std::accumulate( values.begin(), values.end(), 0.0 ) / static_cast<double>( values.size() );
}

double shmimDelta::rms( const std::vector<double> &values, double meanValue )
{
    if( values.empty() )
    {
        return 0;
    }

    double variance = 0;

    for( const double value : values )
    {
        const double residual = value - meanValue;
        variance += residual * residual;
    }

    return std::sqrt( variance / static_cast<double>( values.size() ) );
}

int shmimDelta::openStream( streamState &stream )
{
    int logged = 0;

    while( !g_timeToDie )
    {
        int  SM_fd;
        char SM_fname[200];
        ImageStreamIO_filename( SM_fname, sizeof( SM_fname ), stream.m_name.c_str() );

        SM_fd = open( SM_fname, O_RDWR );
        if( SM_fd == -1 )
        {
            if( !logged )
            {
                std::cerr << "ImageStream " << stream.m_name << " not found (yet). Retrying . . .\n";
            }

            logged = 1;
            sleep( 1 );
            continue;
        }

        logged = 0;
        close( SM_fd );

        if( ImageStreamIO_openIm( &stream.m_imageStream, stream.m_name.c_str() ) != 0 )
        {
            mx::sys::sleep( 1 );
            continue;
        }

        if( stream.m_imageStream.md[0].sem <= stream.m_semaphoreNumber )
        {
            ImageStreamIO_closeIm( &stream.m_imageStream );
            mx::sys::sleep( 1 );
            continue;
        }

        struct stat buffer;
        if( stat( SM_fname, &buffer ) != 0 )
        {
            std::cerr << "Could not get inode for " << stream.m_name << ".\n";
            ImageStreamIO_closeIm( &stream.m_imageStream );
            return -1;
        }

        stream.m_inode  = buffer.st_ino;
        stream.m_opened = true;

        stream.m_width  = stream.m_imageStream.md[0].size[0];
        stream.m_height = 1;
        stream.m_depth  = 1;

        if( stream.m_imageStream.md[0].naxis > 1 )
        {
            stream.m_height = stream.m_imageStream.md[0].size[1];
        }

        if( stream.m_imageStream.md[0].naxis > 2 )
        {
            stream.m_depth = stream.m_imageStream.md[0].size[2];
        }

        return 0;
    }

    return -1;
}

void shmimDelta::closeStream( streamState &stream )
{
    if( !stream.m_opened )
    {
        return;
    }

    if( stream.m_semaphoreNumber >= 0 )
    {
        stream.m_imageStream.semReadPID[stream.m_semaphoreNumber] = 0;
    }

    ImageStreamIO_closeIm( &stream.m_imageStream );
    stream.m_opened    = false;
    stream.m_semaphore = nullptr;
}

void shmimDelta::closeStreams()
{
    closeStream( m_stream1 );
    closeStream( m_stream2 );
}

int shmimDelta::prepareSemaphore( streamState &stream )
{
    stream.m_semaphoreNumber = ImageStreamIO_getsemwaitindex( &stream.m_imageStream, stream.m_semaphoreNumber );
    if( stream.m_semaphoreNumber < 0 )
    {
        std::cerr << "No valid semaphore found for " << stream.m_name << ".\n";
        return -1;
    }

    ImageStreamIO_semflush( &stream.m_imageStream, stream.m_semaphoreNumber );
    stream.m_semaphore = stream.m_imageStream.semptr[stream.m_semaphoreNumber];

    return 0;
}

int shmimDelta::measureDeltas()
{
    if( prepareSemaphore( m_stream1 ) < 0 || prepareSemaphore( m_stream2 ) < 0 )
    {
        return -1;
    }

    if( synchronizeStreams() < 0 )
    {
        return -1;
    }

    std::thread thread1;
    std::thread thread2;

    try
    {
        thread1 = std::thread( waitThreadStart, this, &m_stream1 );
        thread2 = std::thread( waitThreadStart, this, &m_stream2 );
    }
    catch( const std::exception &e )
    {
        std::cerr << "exception starting wait threads: " << e.what() << "\n";
        g_timeToDie = 1;
        if( thread1.joinable() )
        {
            thread1.join();
        }

        if( thread2.joinable() )
        {
            thread2.join();
        }

        return -1;
    }

    thread1.join();
    thread2.join();

    if( m_stream1.m_status < 0 )
    {
        std::cerr << m_stream1.m_errorMessage << "\n";
        return -1;
    }

    if( m_stream2.m_status < 0 )
    {
        std::cerr << m_stream2.m_errorMessage << "\n";
        return -1;
    }

    const size_t pairedFrames = pairByReferenceCounter();

    if( pairedFrames < 2 )
    {
        std::cerr << "Need at least 2 paired arrivals with matching synchronized counter advances to calculate delta "
                     "rms.\n";
        return -1;
    }

    const double deltaMean = mean( m_deltaUsec );
    const double deltaRms  = rms( m_deltaUsec, deltaMean );

    std::vector<double> stream1FrameDeltaUsec;
    std::vector<double> stream2FrameDeltaUsec;

    calculateFrameDeltas( m_stream1, stream1FrameDeltaUsec );
    calculateFrameDeltas( m_stream2, stream2FrameDeltaUsec );

    const double stream1DeltaMean = mean( stream1FrameDeltaUsec );
    const double stream1DeltaRms  = rms( stream1FrameDeltaUsec, stream1DeltaMean );
    const double stream2DeltaMean = mean( stream2FrameDeltaUsec );
    const double stream2DeltaRms  = rms( stream2FrameDeltaUsec, stream2DeltaMean );

    std::cout << std::fixed << std::setprecision( 3 );
    std::cout << "timed_pairs: " << pairedFrames << "\n";
    std::cout << "pairing: reference_cnt0\n";
    std::cout << "reference1_cnt0: " << m_stream1.m_referenceSample.m_cnt0 << "\n";
    std::cout << "reference2_cnt0: " << m_stream2.m_referenceSample.m_cnt0 << "\n";
    std::cout << "reference_delta_usec: "
              << elapsedUsec( m_stream1.m_referenceSample.m_eventTime, m_stream2.m_referenceSample.m_eventTime )
              << "\n";
    std::cout << "delta_mean_usec: " << deltaMean << "\n";
    std::cout << "delta_rms_usec: " << deltaRms << "\n";
    std::cout << "stream1_delta_mean_usec: " << stream1DeltaMean << "\n";
    std::cout << "stream1_delta_rms_usec: " << stream1DeltaRms << "\n";
    std::cout << "stream2_delta_mean_usec: " << stream2DeltaMean << "\n";
    std::cout << "stream2_delta_rms_usec: " << stream2DeltaRms << "\n";

    return 0;
}

int shmimDelta::synchronizeStreams()
{
    m_stream1.m_haveReference = false;
    m_stream2.m_haveReference = false;

    m_stream1.m_samples.clear();
    m_stream2.m_samples.clear();

    ImageStreamIO_semflush( &m_stream1.m_imageStream, m_stream1.m_semaphoreNumber );
    ImageStreamIO_semflush( &m_stream2.m_imageStream, m_stream2.m_semaphoreNumber );

    if( waitForFrame( m_stream1, 1 ) < 0 )
    {
        std::cerr << "Error synchronizing reference from " << m_stream1.m_name << ": " << m_stream1.m_errorMessage
                  << "\n";
        return -1;
    }

    if( m_stream1.m_samples.empty() )
    {
        std::cerr << "No reference frame recorded from " << m_stream1.m_name << ".\n";
        return -1;
    }

    m_stream1.m_referenceSample = m_stream1.m_samples.back();
    m_stream1.m_haveReference   = true;

    size_t stream2Attempts = 0;

    // Do not flush stream 2 here; a low-latency post after the stream-1 reference may already be queued.
    while( !g_timeToDie )
    {
        ++stream2Attempts;

        if( waitForFrame( m_stream2, stream2Attempts ) < 0 )
        {
            std::cerr << "Error synchronizing reference from " << m_stream2.m_name << ": " << m_stream2.m_errorMessage
                      << "\n";
            return -1;
        }

        if( m_stream2.m_samples.empty() )
        {
            continue;
        }

        const auto &candidate = m_stream2.m_samples.back();
        if( timeAtOrAfter( candidate.m_eventTime, m_stream1.m_referenceSample.m_eventTime ) )
        {
            m_stream2.m_referenceSample = candidate;
            m_stream2.m_haveReference   = true;
            break;
        }
    }

    if( !m_stream2.m_haveReference )
    {
        std::cerr << "No reference frame recorded from " << m_stream2.m_name << " after " << m_stream1.m_name << ".\n";
        return -1;
    }

    m_stream2.m_samples.clear();
    m_stream2.m_samples.push_back( m_stream2.m_referenceSample );

    return 0;
}

size_t shmimDelta::pairByReferenceCounter()
{
    m_deltaUsec.clear();

    if( !m_stream1.m_haveReference || !m_stream2.m_haveReference )
    {
        return 0;
    }

    size_t n1 = 0;
    size_t n2 = 0;

    while( n1 < m_stream1.m_samples.size() && n2 < m_stream2.m_samples.size() )
    {
        const auto &sample1 = m_stream1.m_samples[n1];
        const auto &sample2 = m_stream2.m_samples[n2];

        uint64_t advance1 = 0;
        uint64_t advance2 = 0;

        if( !counterAdvance( advance1, sample1, m_stream1.m_referenceSample ) )
        {
            ++n1;
            continue;
        }

        if( !counterAdvance( advance2, sample2, m_stream2.m_referenceSample ) )
        {
            ++n2;
            continue;
        }

        if( advance1 == advance2 )
        {
            m_deltaUsec.push_back( elapsedUsec( sample1.m_eventTime, sample2.m_eventTime ) );
            ++n1;
            ++n2;
        }
        else if( advance1 < advance2 )
        {
            ++n1;
        }
        else
        {
            ++n2;
        }
    }

    return m_deltaUsec.size();
}

size_t shmimDelta::calculateFrameDeltas( const streamState &stream, std::vector<double> &frameDeltas ) const
{
    frameDeltas.clear();

    if( stream.m_samples.size() < 2 )
    {
        return 0;
    }

    frameDeltas.reserve( stream.m_samples.size() - 1 );

    for( size_t n = 1; n < stream.m_samples.size(); ++n )
    {
        frameDeltas.push_back( elapsedUsec( stream.m_samples[n - 1].m_eventTime, stream.m_samples[n].m_eventTime ) );
    }

    return frameDeltas.size();
}

bool shmimDelta::counterAdvance( uint64_t                       &advance,
                                 const streamState::frameSample &sample,
                                 const streamState::frameSample &reference ) const
{
    if( sample.m_cnt0 < reference.m_cnt0 )
    {
        return false;
    }

    advance = sample.m_cnt0 - reference.m_cnt0;

    return true;
}

void shmimDelta::waitThreadStart( shmimDelta *app, streamState *stream )
{
    stream->m_status = app->waitFrames( *stream );
}

int shmimDelta::waitFrames( streamState &stream )
{
    stream.m_samples.reserve( m_nFrames );
    stream.m_errorMessage.clear();

    while( stream.m_samples.size() < m_nFrames && !g_timeToDie )
    {
        if( waitForFrame( stream, stream.m_samples.size() + 1 ) < 0 )
        {
            return -1;
        }
    }

    if( g_timeToDie )
    {
        stream.m_errorMessage = "Interrupted while waiting for frames from " + stream.m_name + ".";
        return -1;
    }

    return 0;
}

int shmimDelta::waitForFrame( streamState &stream, size_t frameNumber )
{
    timespec deadline;
    if( setWaitDeadline( deadline ) < 0 )
    {
        stream.m_errorMessage = "error from clock_gettime while waiting for " + stream.m_name;
        return -1;
    }

    if( sem_timedwait( stream.m_semaphore, &deadline ) != 0 )
    {
        if( errno == ETIMEDOUT )
        {
            stream.m_errorMessage = "Timed out waiting for frame " + std::to_string( frameNumber ) + " of " +
                                    std::to_string( m_nFrames ) + " from " + stream.m_name + ".";
        }
        else if( errno == EINTR && g_timeToDie )
        {
            stream.m_errorMessage = "Interrupted while waiting for frames from " + stream.m_name + ".";
        }
        else
        {
            stream.m_errorMessage = "error from sem_timedwait for " + stream.m_name + ": " + strerror( errno );
        }

        if( streamChanged( stream ) )
        {
            stream.m_errorMessage += " The shmim changed while waiting for frames.";
        }

        return -1;
    }

    while( sem_trywait( stream.m_semaphore ) == 0 )
    {
    }

    if( errno != EAGAIN && errno != EINTR )
    {
        stream.m_errorMessage = "error from sem_trywait for " + stream.m_name + ": " + strerror( errno );
        return -1;
    }

    return recordLatestSample( stream );
}

int shmimDelta::recordLatestSample( streamState &stream )
{
    streamState::frameSample sample = latestSample( stream );

    if( clock_gettime( CLOCK_MONOTONIC, &sample.m_arrivalTime ) < 0 )
    {
        stream.m_errorMessage = "error from clock_gettime after semaphore wake for " + stream.m_name;
        return -1;
    }

    if( !validTime( sample.m_eventTime ) )
    {
        sample.m_eventTime = sample.m_arrivalTime;
    }

    if( !stream.m_samples.empty() && sample.m_cnt0 == stream.m_samples.back().m_cnt0 )
    {
        return 0;
    }

    stream.m_samples.push_back( sample );

    return 0;
}

shmimDelta::streamState::frameSample shmimDelta::latestSample( const streamState &stream ) const
{
    streamState::frameSample sample;

    const IMAGE &image = stream.m_imageStream;

    uint64_t currImage = ImageStreamIO_readLastWroteIndex( &image );
    if( currImage >= ImageStreamIO_nbSlices( &image ) )
    {
        currImage = 0;
    }

    if( image.cntarray != nullptr )
    {
        sample.m_cnt0 = image.cntarray[currImage];
    }
    else
    {
        sample.m_cnt0 = image.md[0].cnt0;
    }

    if( image.writetimearray != nullptr && validTime( image.writetimearray[currImage] ) )
    {
        sample.m_eventTime        = image.writetimearray[currImage];
        sample.m_usedMetadataTime = true;
    }
    else if( validTime( image.md[0].writetime ) )
    {
        sample.m_eventTime        = image.md[0].writetime;
        sample.m_usedMetadataTime = true;
    }

    return sample;
}

bool shmimDelta::validTime( const timespec &ts )
{
    return ts.tv_sec != 0 || ts.tv_nsec != 0;
}

bool shmimDelta::timeAtOrAfter( const timespec &ts, const timespec &reference )
{
    if( ts.tv_sec != reference.tv_sec )
    {
        return ts.tv_sec > reference.tv_sec;
    }

    return ts.tv_nsec >= reference.tv_nsec;
}

bool shmimDelta::streamChanged( const streamState &stream ) const
{
    char SM_fname[200];
    ImageStreamIO_filename( SM_fname, sizeof( SM_fname ), stream.m_name.c_str() );

    struct stat buffer;
    if( stat( SM_fname, &buffer ) != 0 )
    {
        return true;
    }

    return buffer.st_ino != stream.m_inode;
}

int shmimDelta::setWaitDeadline( timespec &deadline ) const
{
    if( clock_gettime( CLOCK_REALTIME, &deadline ) < 0 )
    {
        return -1;
    }

    const auto seconds = static_cast<time_t>( m_timeoutSec );
    deadline.tv_sec += seconds;
    deadline.tv_nsec += static_cast<long>( ( m_timeoutSec - static_cast<double>( seconds ) ) * 1e9 );

    if( deadline.tv_nsec >= 1000000000L )
    {
        ++deadline.tv_sec;
        deadline.tv_nsec -= 1000000000L;
    }

    return 0;
}

int main( int argc, char **argv )
{
    shmimDelta sd;

    return sd.main( argc, argv );
}
