/** \file po4aoRouting.cpp
 * \brief Bounded HOWFS utility execution and installed milk ABI verification.
 */
#include "po4aoRouting.hpp"
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <thread>
#include <fcntl.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#ifdef PO4AO_ROUTE_TEST
    #include "tests/po4aoRouteTestAbi.hpp"
#else
extern "C"
{
    #include <CLIcore.h>
    #include <processinfo.h>
}
#endif

namespace MagAOX::app
{
namespace
{
#ifdef PO4AO_ROUTE_TEST
constexpr const char *root               = ".build/routing/root";
constexpr const char *shm                = ".build/routing/shm/";
constexpr const char *proc               = ".build/routing/proc";
constexpr const char *switchUtility      = ".build/routing/root/switch.sh";
constexpr const char *calibrationUtility = ".build/routing/root/calibrate.sh";
#else
constexpr const char *root               = "/opt/MagAOX/cacao/tweeter-vispyr/tweeter-vispyr-rootdir";
constexpr const char *shm                = "/milk/shm/";
constexpr const char *proc               = "/proc";
constexpr const char *switchUtility      = "/usr/local/bin/howfs_mfilt_switch";
constexpr const char *calibrationUtility = "/usr/local/milk/bin/cacao-calib-apply";
#endif

/// Read a required completion/identity file without executing a control utility.
std::string line( const std::string &path /**< [in] Required text file. */ )
{
    std::ifstream input( path );
    std::string   result;
    if( !std::getline( input, result ) || result.empty() )
    {
        throw std::runtime_error( "Missing readback: " + path );
    }
    return result;
}

/// Read one installed-ABI record from a regular file.
template <class T>
T readRecord( const std::string &path /**< [in] Shared record file. */,
              off_t              offset = 0 /**< [in] Byte offset in the record file. */ )
{
    T         result{};
    const int fd = open( path.c_str(), O_RDONLY | O_CLOEXEC );
    if( fd < 0 )
    {
        throw std::runtime_error( "Cannot read " + path );
    }
    struct stat info{};
    const bool  valid = fstat( fd, &info ) == 0 && S_ISREG( info.st_mode ) &&
                        pread( fd, &result, sizeof( result ), offset ) == sizeof( result );
    close( fd );
    if( !valid )
    {
        throw std::runtime_error( "Incomplete readback: " + path );
    }
    return result;
}

/// Read matching FPS metadata and enforce the installed parameter ABI.
FUNCTION_PARAMETER_STRUCT_MD metadata()
{
    const std::string path = std::string( shm ) + "mfilt-1.fps.shm";
    const auto        md   = readRecord<FUNCTION_PARAMETER_STRUCT_MD>( path );
    const auto        size = std::filesystem::file_size( path );
    if( std::string( md.name, strnlen( md.name, sizeof( md.name ) ) ) != "mfilt-1" || md.NBparamMAX <= 0 ||
        size != sizeof( md ) + static_cast<uint64_t>( md.NBparamMAX ) * sizeof( FUNCTION_PARAMETER ) )
    {
        throw std::runtime_error( "mfilt FPS layout differs from installed headers" );
    }
    return md;
}

/// Read one exact FPS parameter; no writable mapping is created.
FUNCTION_PARAMETER parameter( const std::string &name /**< [in] Full FPS parameter key. */ )
{
    const auto md = metadata();
    for( long index = 0; index < md.NBparamMAX; ++index )
    {
        const auto p = readRecord<FUNCTION_PARAMETER>( std::string( shm ) + "mfilt-1.fps.shm",
                                                       sizeof( md ) + index * sizeof( FUNCTION_PARAMETER ) );
        if( std::string( p.keywordfull, strnlen( p.keywordfull, sizeof( p.keywordfull ) ) ) == name )
        {
            return p;
        }
    }
    throw std::runtime_error( "Missing FPS parameter: " + name );
}

/// Enforce the existing operator-managed loop prerequisite; never open or close it here.
void requireOpenLoop()
{
    const auto p = parameter( "mfilt-1.loopON" );
    if( p.type != FPTYPE_ONOFF || ( p.fpflag & FPFLAG_ONOFF ) || p.val.ui64[0] != 0 )
    {
        throw std::runtime_error( "Open the HOWFS loop before changing the PO4AO route" );
    }
}

/// Check the shared deadline while waiting outside all application locks.
void waitStep( Po4aoRouteBackend::Deadline deadline /**< [in] Operation deadline. */,
               const std::atomic<bool>    &cancelled /**< [in] Shutdown cancellation. */ )
{
    if( cancelled.load() || std::chrono::steady_clock::now() >= deadline )
    {
        throw std::runtime_error( "Routing readback timed out" );
    }
    std::this_thread::sleep_for( std::chrono::milliseconds( 20 ) );
}

/// Execute a fixed utility in its required directory and settle its process group on timeout.
void command( const char                 *executable /**< [in] Verified absolute executable. */,
              const char                 *argument /**< [in] Single fixed argument. */,
              Po4aoRouteBackend::Deadline deadline /**< [in] Operation deadline. */,
              const std::atomic<bool>    &cancelled /**< [in] Shutdown cancellation. */ )
{
    if( cancelled.load() || std::chrono::steady_clock::now() >= deadline )
    {
        throw std::runtime_error( "Routing cancelled before command delivery" );
    }
    const auto  executablePath = std::filesystem::absolute( executable ).string();
    const pid_t child          = fork();
    if( child < 0 )
    {
        throw std::runtime_error( "Cannot launch routing utility" );
    }
    if( child == 0 )
    {
        setpgid( 0, 0 );
        if( chdir( root ) != 0 )
        {
            _exit( 126 );
        }
        char *const argv[] = { const_cast<char *>( executablePath.c_str() ), const_cast<char *>( argument ), nullptr };
        execv( executablePath.c_str(), argv );
        _exit( 127 );
    }
    setpgid( child, child );
    int status = 0;
    for( ;; )
    {
        const auto result = waitpid( child, &status, WNOHANG );
        if( result == child )
        {
            break;
        }
        if( result < 0 && errno != EINTR )
        {
            throw std::runtime_error( "Routing child status is unknown" );
        }
        if( cancelled.load() || std::chrono::steady_clock::now() >= deadline )
        {
            kill( -child, SIGKILL );
            while( waitpid( child, &status, 0 ) < 0 && errno == EINTR )
            {
            }
            throw std::runtime_error( "Routing utility timed out; queued FIFO requests may still be pending" );
        }
        std::this_thread::sleep_for( std::chrono::milliseconds( 20 ) );
    }
    if( !WIFEXITED( status ) || WEXITSTATUS( status ) != 0 )
    {
        throw std::runtime_error( "Routing utility failed: " + std::string( executable ) );
    }
}
} // namespace

Po4aoRouteIO::Po4aoRouteIO() : m_writer( shm, proc )
{
}

std::string Po4aoRouteIO::route() const
{
    const auto target = std::filesystem::canonical( std::string( shm ) + "aol1_mfiltINPUT.im.shm" );
    for( const auto *name : { "aol1_modevalWFS", "po4ao_output" } )
    {
        const auto path = std::string( shm ) + name + ".im.shm";
        if( std::filesystem::exists( path ) && target == std::filesystem::canonical( path ) )
        {
            return name;
        }
    }
    throw std::runtime_error( "Unrecognized mfilt input route" );
}

Po4aoRouteSnapshot Po4aoRouteIO::capture()
{
    requireOpenLoop();
    Po4aoRouteSnapshot result;
    result.route        = route();
    result.calibration  = line( std::string( shm ) + "aol1_calib_source.txt" );
    m_beforeCalibration = line( std::string( shm ) + "aol1_calib_loaded.txt" );
    m_beforeWriter      = m_writer.runningIdentity();
    if( !std::filesystem::is_directory( std::string( root ) + "/../tweeter-vispyr-calibs/default" ) )
    {
        throw std::runtime_error( "Default calibration is unavailable" );
    }
    return result;
}

void Po4aoRouteIO::switchInput( const std::string &target, Po4aoRouteBackend::Deadline deadline )
{
    if( target != "aol1_modevalWFS" && target != "po4ao_output" )
    {
        throw std::runtime_error( "Invalid routing input" );
    }
    requireOpenLoop();
    // The worker now holds NN's reservation. A writer change between preflight and
    // that hold must not be mistaken for the calibration-driven restart below.
    m_beforeWriter = m_writer.runningIdentity();
    command( switchUtility, target.c_str(), deadline, m_cancelled );
    if( route() != target )
    {
        throw std::runtime_error( "Input switch readback mismatch" );
    }
}

void Po4aoRouteIO::applyDefault( Po4aoRouteBackend::Deadline deadline )
{
    command( calibrationUtility, "default", deadline, m_cancelled );
    const auto expected = std::filesystem::canonical( std::string( root ) + "/../tweeter-vispyr-calibs/default" );
    while( line( std::string( shm ) + "aol1_calib_loaded.txt" ) == m_beforeCalibration ||
           std::filesystem::canonical( line( std::string( shm ) + "aol1_calib_source.txt" ) ) != expected )
    {
        waitStep( deadline, m_cancelled );
    }
    // The inspected utility queues one WFS runstart before writing its marker. Await that
    // new process, not an old still-running process or a stopped writer with work queued.
    for( ;; )
    {
        if( m_cancelled.load() || std::chrono::steady_clock::now() >= deadline )
        {
            throw std::runtime_error( "CACAO calibration restart not acknowledged" );
        }
        try
        {
            const auto identity = m_writer.runningIdentity();
            if( !identity.empty() && identity != m_beforeWriter )
            {
                break;
            }
        }
        catch( const std::exception & )
        {
            // Transient/incomplete FPS or process evidence cannot acknowledge a restart.
        }
        std::this_thread::sleep_for( std::chrono::milliseconds( 20 ) );
    }
    requireOpenLoop();
}

void Po4aoRouteIO::verifyConsumption( const std::string &target, Po4aoRouteBackend::Deadline deadline )
{
    const auto input = parameter( "mfilt-1.inmval" );
    if( std::string( input.val.string[0], strnlen( input.val.string[0], sizeof( input.val.string[0] ) ) ) !=
        "aol1_mfiltINPUT" )
    {
        throw std::runtime_error( "mfilt does not use the configured input alias" );
    }
#ifndef PO4AO_ROUTE_TEST
    cacaoWriter consumer;
    consumer.m_name         = "mfilt-1";
    consumer.m_processMatch = "mfilt-1";
#endif
    pid_t    previousPid     = 0;
    uint64_t previousCounter = 0;
    long     previousLoop    = 0;
    for( ;; )
    {
        waitStep( deadline, m_cancelled );
        if( route() != target )
        {
            throw std::runtime_error( "Downstream route changed during verification" );
        }
#ifndef PO4AO_ROUTE_TEST
        try
        {
            if( !consumer.running() )
            {
                continue;
            }
        }
        catch( const std::exception & )
        {
            continue;
        } // Pending FPS starts are not completion.
#endif
        const auto md = metadata();
        if( md.runpid <= 0 || kill( md.runpid, 0 ) != 0 )
        {
            continue;
        }
        const auto path = std::string( shm ) + "proc.mfilt-1." + std::to_string( md.runpid ) + ".shm";
        if( !std::filesystem::exists( path ) )
        {
            continue;
        }
        if( std::filesystem::file_size( path ) != sizeof( PROCESSINFO ) )
        {
            throw std::runtime_error( "process-info ABI mismatch" );
        }
        const auto  p = readRecord<PROCESSINFO>( path );
        struct stat info{};
        if( stat( ( std::string( shm ) + target + ".im.shm" ).c_str(), &info ) != 0 )
        {
            throw std::runtime_error( "Requested input disappeared" );
        }
        if( p.PID != md.runpid || p.loopstat != PROCESSINFO_LOOPSTAT_ACTIVE || p.triggerstreaminode != info.st_ino ||
            p.triggerstreamID < 0 )
        {
            continue;
        }
        if( previousPid == md.runpid && p.triggerstreamcnt > previousCounter && p.loopcnt > previousLoop )
        {
            return;
        }
        previousPid     = md.runpid;
        previousCounter = p.triggerstreamcnt;
        previousLoop    = p.loopcnt;
    }
}
} // namespace MagAOX::app
