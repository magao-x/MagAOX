/** \file streamWriter_fault_test.cpp
 * \brief Deterministic dependency failures and worker-loop contracts for streamWriter.
 * \ingroup streamWriter_files
 */
#include "../../../tests/testXWC.hpp"

#include <deque>
#include <functional>
#include <filesystem>
#include <fstream>
#include <memory>

#define protected public
#include "../../../libMagAOX/libMagAOX.hpp"
#undef protected
#include <ImageStreamIO/ImageStreamIO.h>
#include <xrif/xrif.h>

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
namespace streamWriterHarness
{
/// One captured application diagnostic.
struct Log
{
    flatlogs::logPrioT m_priority; ///< Actual severity after default resolution.
    std::string        m_text;     ///< Formatted production log message.
};

/// Selected XRIF call failure, optionally after performing the real operation.
struct XrifFault
{
    unsigned m_calls{ 0 };     ///< Number of operation calls.
    unsigned m_fail{ 0 };      ///< One-based call to fail; zero disables the fault.
    bool     m_after{ false }; ///< Report an error after producing real encoder output.
};

/// Faults apply only to dependency calls in the included application.
struct Faults
{
    bool     m_throwTelem{ false }; ///< Simulate a telemetry sink exception after serialization.
    unsigned m_waitCalls{ 0 }, m_sleepCalls{ 0 },
        m_tryWaitCalls{ 0 }; ///< Bound synchronous loop events so regressions fail instead of hanging.
    XrifFault m_configure, m_setSize, m_allocateRaw, m_allocateReordered, m_lz4, m_encode,
        m_header;                                      ///< Encoder operations.
    bool                  m_throwConstructor{ false }; ///< Simulate failure while constructing the telemetry base.
    unsigned              m_writeCalls{ 0 }, m_failWrite{ 0 }; ///< Selected short archive write.
    unsigned              m_clockCalls{ 0 };                   ///< Number of worker clock calls.
    std::function<void()> m_afterClock;                        ///< Synchronous shutdown while timestamping a frame.
    int                   m_telemSetup{ 0 }, m_telemLoad{ 0 }, m_telemStartup{ 0 }, m_telemLogic{ 0 },
        m_telemShutdown{ 0 };                                      ///< Helper results.
    unsigned              m_schedules{ 0 };                        ///< Number of telemetry scheduler calls.
    bool                  m_due{ false };                          ///< Force the next scheduled record.
    unsigned              m_mallocCalls{ 0 }, m_failMalloc{ 0 };   ///< Selected application allocation failure.
    unsigned              m_newCalls{ 0 }, m_failNew{ 0 };         ///< Selected XRIF handle failure.
    unsigned              m_sigCalls{ 0 }, m_failSig{ 0 };         ///< Selected signal installation failure.
    unsigned              m_threadCalls{ 0 }, m_failThread{ 0 };   ///< Selected worker startup failure.
    unsigned              m_tryJoinCalls{ 0 }, m_failTryJoin{ 0 }; ///< Selected exited-worker check.
    bool                  m_throwTryJoin{ false };                 ///< Throw instead of reporting an exited worker.
    unsigned              m_joinCalls{ 0 }, m_failJoin{ 0 };       ///< Selected join exception.
    bool                  m_failSemInit{ false };                  ///< Fail writer semaphore initialization.
    bool                  m_failPost{ false };                     ///< Fail a queued write notification.
    bool                  m_failClock{ false };                    ///< Fail realtime clock acquisition.
    bool                  m_failDirectories{ false };              ///< Throw a generic directory creation exception.
    bool                  m_failXrifAllocate{ false };             ///< Fail encoder raw-buffer allocation.
    unsigned              m_closeCalls{ 0 }; ///< Image closes observed by the synchronous fixture.
    std::function<int()>  m_openFile, m_stat, m_openImage, m_getIndex, m_tryWait; ///< Selected stream dependencies.
    std::function<int()>  m_wait;                  ///< One synchronous ingest or writer-loop event.
    std::function<void()> m_sleep;                 ///< Synchronous startup retry or initialization event.
    IMAGE                 m_image{};               ///< Borrowed synthetic source supplied to the app.
    std::vector<Log>      m_logs;                  ///< Captured real log payloads.
    std::vector<std::vector<uint8_t>> m_telemetry; ///< Captured serialized telemetry records.
};
Faults g_faults; ///< Process-local state, reset before each fixture.

/// Reset dependency faults and captured output.
void reset();

/// Count captured diagnostics containing a substring.
size_t logs( const std::string &text /**< [in] diagnostic substring */ );

void reset()
{
    g_faults = Faults{};
}
size_t logs( const std::string &text )
{
    return std::count_if( g_faults.m_logs.begin(),
                          g_faults.m_logs.end(),
                          [&]( const Log &entry ) { return entry.m_text.find( text ) != std::string::npos; } );
}
} // namespace streamWriterHarness

// Overlay only the app's standard-library lookups; never add declarations to namespace std.
namespace streamWriterTestStd
{
using namespace std;
/// Real thread storage with controllable join exceptions and no background test worker.
class thread : public std::thread
{
  public:
    /// Treat the selected dependency fault as a joinable worker.
    bool joinable() const;
    /// Inject the selected exception or join a real worker.
    void join();
};
bool thread::joinable() const
{
    return streamWriterHarness::g_faults.m_failJoin != 0 || std::thread::joinable();
}
void thread::join()
{
    auto &f = streamWriterHarness::g_faults;
    if( ++f.m_joinCalls == f.m_failJoin )
        throw std::system_error( std::make_error_code( std::errc::resource_deadlock_would_occur ) );
    if( std::thread::joinable() )
        std::thread::join();
}
namespace filesystem
{
using namespace std::filesystem;
/// Delegate actual directory creation except for the selected generic exception.
bool create_directories( const std::filesystem::path &path /**< [in] archive directory */ );
bool create_directories( const std::filesystem::path &path )
{
    if( streamWriterHarness::g_faults.m_failDirectories )
        throw std::runtime_error( "injected directory failure" );
    return std::filesystem::create_directories( path );
}
} // namespace filesystem
} // namespace streamWriterTestStd

namespace MagAOX::app
{
/// Preserve framework configuration and INDI while capturing logs and avoiding worker startup.
template <bool useINDI = true>
class streamWriterTestApp : public MagAOXApp<useINDI>
{
  public:
    using MagAOXApp<useINDI>::m_configName;
    /// Initialize the real app after silencing its shared logger.
    streamWriterTestApp( const std::string &sha /**< [in] revision */, bool modified /**< [in] dirty flag */ );
    /// Silence the shared logger before construction.
    static const std::string &quiet( const std::string &sha /**< [in] revision */ );
    /// Capture the real log message and preserve the requested return value.
    template <class logT, int retval = 0>
    static int log( const typename logT::messageT &msg /**< [in] payload */,
                    flatlogs::logPrioT             priority = flatlogs::logPrio::LOG_DEFAULT /**< [in] severity */ );
    /// Inject worker-start results without constructing background threads.
    template <class App, class Function>
    int threadStart( std::thread       &worker /**< [out] worker slot */,
                     bool              &init /**< [out] startup synchronizer */,
                     pid_t             &id /**< [out] worker ID */,
                     pcf::IndiProperty &property /**< [out] worker property */,
                     int                priority /**< [in] priority */,
                     const std::string &cpuset /**< [in] CPU set */,
                     const std::string &name /**< [in] worker name */,
                     App               *app /**< [in] app instance */,
                     Function         &&start /**< [in] entrypoint */ );
};
template <bool useINDI>
streamWriterTestApp<useINDI>::streamWriterTestApp( const std::string &sha, bool modified )
    : MagAOXApp<useINDI>( quiet( sha ), modified )
{
}
template <bool useINDI>
const std::string &streamWriterTestApp<useINDI>::quiet( const std::string &sha )
{
    MagAOXApp<useINDI>::m_log.m_logLevel = flatlogs::logPrio::LOG_EMERGENCY;
    return sha;
}
template <bool useINDI>
template <class logT, int retval>
int streamWriterTestApp<useINDI>::log( const typename logT::messageT &msg, flatlogs::logPrioT priority )
{
    if( priority == flatlogs::logPrio::LOG_DEFAULT )
        priority = logT::defaultLevel;
    streamWriterHarness::g_faults.m_logs.push_back(
        { priority, logT::msgString( msg.builder.GetBufferPointer(), msg.builder.GetSize() ) } );
    return retval;
}
template <bool useINDI>
template <class App, class Function>
int streamWriterTestApp<useINDI>::threadStart( std::thread &,
                                               bool &,
                                               pid_t &,
                                               pcf::IndiProperty &,
                                               int,
                                               const std::string &,
                                               const std::string &,
                                               App *,
                                               Function && )
{
    auto &f = streamWriterHarness::g_faults;
    return ++f.m_threadCalls == f.m_failThread ? -1 : 0;
}
namespace dev
{
/// Threadless sink retaining real telemetry configuration and production record dispatch.
template <class App>
class streamWriterTestTelemeter : public telemeter<App>
{
  public:
    /// Simulate a telemetry-base construction failure for app resource unwinding.
    streamWriterTestTelemeter();
    /// Register the real helper options or inject failure.
    int setupConfig( mx::app::appConfigurator &config /**< [in/out] configurator */ );
    /// Load the real helper options or inject failure.
    int loadConfig( mx::app::appConfigurator &config /**< [in] configurator */ );
    /// Return the selected startup result.
    int appStartup();
    /// Check the production record dispatcher unless scheduling fails.
    int appLogic();
    /// Return the selected shutdown result.
    int appShutdown();
    /// Force the next record when its injected deadline is due.
    int checkRecordTimes( const logger::telem_saving_state &type /**< [in] record selector */ );
    /// Capture the production FlatBuffer payload.
    template <class T>
    int telem( const typename T::messageT &msg /**< [in] telemetry payload */ );
};
template <class App>
streamWriterTestTelemeter<App>::streamWriterTestTelemeter()
{
    if( streamWriterHarness::g_faults.m_throwConstructor )
        throw std::runtime_error( "injected telemetry constructor" );
}
template <class App>
int streamWriterTestTelemeter<App>::setupConfig( mx::app::appConfigurator &config )
{
    return streamWriterHarness::g_faults.m_telemSetup < 0 ? -1 : telemeter<App>::setupConfig( config );
}
template <class App>
int streamWriterTestTelemeter<App>::loadConfig( mx::app::appConfigurator &config )
{
    return streamWriterHarness::g_faults.m_telemLoad < 0 ? -1 : telemeter<App>::loadConfig( config );
}
template <class App>
int streamWriterTestTelemeter<App>::appStartup()
{
    return streamWriterHarness::g_faults.m_telemStartup;
}
template <class App>
int streamWriterTestTelemeter<App>::appLogic()
{
    auto &f = streamWriterHarness::g_faults;
    ++f.m_schedules;
    return f.m_telemLogic < 0 ? -1 : static_cast<App *>( this )->checkRecordTimes();
}
template <class App>
int streamWriterTestTelemeter<App>::appShutdown()
{
    return streamWriterHarness::g_faults.m_telemShutdown;
}
template <class App>
int streamWriterTestTelemeter<App>::checkRecordTimes( const logger::telem_saving_state &type )
{
    if( !streamWriterHarness::g_faults.m_due )
        return 0;
    streamWriterHarness::g_faults.m_due = false;
    return static_cast<App *>( this )->recordTelem( &type );
}
template <class App>
template <class T>
int streamWriterTestTelemeter<App>::telem( const typename T::messageT &msg )
{
    auto *begin = msg.builder.GetBufferPointer();
    streamWriterHarness::g_faults.m_telemetry.emplace_back( begin, begin + msg.builder.GetSize() );
    if( streamWriterHarness::g_faults.m_throwTelem )
        throw std::runtime_error( "injected telemetry sink" );
    return 0;
}
} // namespace dev
} // namespace MagAOX::app
/// Allocate real memory except at the selected application allocation.
void *swMalloc( size_t size /**< [in] requested bytes */ );
void *swMalloc( size_t size )
{
    auto &f = streamWriterHarness::g_faults;
    if( ++f.m_mallocCalls == f.m_failMalloc )
    {
        errno = ENOMEM;
        return nullptr;
    }
    return ::malloc( size );
}
/// Allocate a real XRIF handle except at the selected call.
xrif_error_t swNew( xrif_t *handle /**< [out] encoder handle */ );
xrif_error_t swNew( xrif_t *handle )
{
    auto &f = streamWriterHarness::g_faults;
    return ++f.m_newCalls == f.m_failNew ? XRIF_ERROR_MALLOC : ::xrif_new( handle );
}
/// Invoke the real encoder operation or report the selected error.
xrif_error_t swXrifCall( streamWriterHarness::XrifFault      &fault /**< [in/out] operation fault */,
                         const std::function<xrif_error_t()> &operation /**< [in] real dependency call */ );
xrif_error_t swXrifCall( streamWriterHarness::XrifFault &fault, const std::function<xrif_error_t()> &operation )
{
    bool fail = ++fault.m_calls == fault.m_fail;
    if( !fail || fault.m_after )
    {
        auto result = operation();
        if( result != XRIF_NOERROR )
            return result;
    }
    return fail ? XRIF_ERROR_BADARG : XRIF_NOERROR;
}
/// Apply the selected fault to xrif_configure while retaining real encoding when requested.
xrif_error_t swConfigure( xrif_t handle /**< [in/out] encoder handle */,
                          int    difference /**< [in] encoder option */,
                          int    reorder /**< [in] encoder option */,
                          int    compress /**< [in] encoder option */ );
xrif_error_t swConfigure( xrif_t handle, int difference, int reorder, int compress )
{
    return swXrifCall( streamWriterHarness::g_faults.m_configure,
                       [&] { return ::xrif_configure( handle, difference, reorder, compress ); } );
}
/// Apply the selected fault to xrif_set_size while retaining real encoding when requested.
xrif_error_t swSetSize( xrif_t           handle /**< [in/out] encoder handle */,
                        xrif_dimension_t width /**< [in] encoder option */,
                        xrif_dimension_t height /**< [in] encoder option */,
                        xrif_dimension_t depth /**< [in] encoder option */,
                        xrif_dimension_t frames /**< [in] encoder option */,
                        xrif_typecode_t  type /**< [in] encoder option */ );
xrif_error_t swSetSize( xrif_t           handle,
                        xrif_dimension_t width,
                        xrif_dimension_t height,
                        xrif_dimension_t depth,
                        xrif_dimension_t frames,
                        xrif_typecode_t  type )
{
    return swXrifCall( streamWriterHarness::g_faults.m_setSize,
                       [&] { return ::xrif_set_size( handle, width, height, depth, frames, type ); } );
}
/// Apply the selected fault to xrif_allocate_raw while retaining real encoding when requested.
xrif_error_t swAllocateRaw( xrif_t handle /**< [in/out] encoder handle */ );
xrif_error_t swAllocateRaw( xrif_t handle )
{
    if( streamWriterHarness::g_faults.m_failXrifAllocate )
        return XRIF_ERROR_MALLOC;
    return swXrifCall( streamWriterHarness::g_faults.m_allocateRaw, [&] { return ::xrif_allocate_raw( handle ); } );
}
/// Apply the selected fault to xrif_allocate_reordered while retaining real encoding when requested.
xrif_error_t swAllocateReordered( xrif_t handle /**< [in/out] encoder handle */ );
xrif_error_t swAllocateReordered( xrif_t handle )
{
    return swXrifCall( streamWriterHarness::g_faults.m_allocateReordered,
                       [&] { return ::xrif_allocate_reordered( handle ); } );
}
/// Apply the selected fault to xrif_set_lz4_acceleration while retaining real encoding when requested.
xrif_error_t swLz4( xrif_t handle /**< [in/out] encoder handle */, int32_t acceleration /**< [in] encoder option */ );
xrif_error_t swLz4( xrif_t handle, int32_t acceleration )
{
    return swXrifCall( streamWriterHarness::g_faults.m_lz4,
                       [&] { return ::xrif_set_lz4_acceleration( handle, acceleration ); } );
}
/// Apply the selected fault to xrif_encode while retaining real encoding when requested.
xrif_error_t swEncode( xrif_t handle /**< [in/out] encoder handle */ );
xrif_error_t swEncode( xrif_t handle )
{
    return swXrifCall( streamWriterHarness::g_faults.m_encode, [&] { return ::xrif_encode( handle ); } );
}
/// Generate a real header before reporting a selected warning.
xrif_error_t swHeader( char *header /**< [out] header bytes */, xrif_t handle /**< [in] encoder */ );
xrif_error_t swHeader( char *header, xrif_t handle )
{
    return swXrifCall( streamWriterHarness::g_faults.m_header, [&] { return ::xrif_write_header( header, handle ); } );
}
/// Produce a selected short write while retaining real archive output.
size_t swWrite( const void *data /**< [in] bytes */,
                size_t      size /**< [in] item size */,
                size_t      count /**< [in] item count */,
                FILE       *file /**< [in] archive */ );
size_t swWrite( const void *data, size_t size, size_t count, FILE *file )
{
    auto &f = streamWriterHarness::g_faults;
    if( ++f.m_writeCalls == f.m_failWrite )
        count = 1;
    return ::fwrite( data, size, count, file );
}
/// Simulate signal registration without replacing the process's signal handlers.
int swSigaction( int                     signal /**< [in] signal */,
                 const struct sigaction *action /**< [in] action */,
                 struct sigaction       *old /**< [out] old action */ );
int swSigaction( int, const struct sigaction *, struct sigaction * )
{
    auto &f = streamWriterHarness::g_faults;
    if( ++f.m_sigCalls == f.m_failSig )
    {
        errno = EINVAL;
        return -1;
    }
    return 0;
}
/// Inject an exited worker or an exception without joining native threads.
int swTryJoin( pthread_t thread /**< [in] native thread */, void **result /**< [out] thread result */ );
int swTryJoin( pthread_t, void ** )
{
    auto &f = streamWriterHarness::g_faults;
    if( ++f.m_tryJoinCalls != f.m_failTryJoin )
        return EBUSY;
    if( f.m_throwTryJoin )
        throw std::runtime_error( "injected join check" );
    return 0;
}
/// Initialize a real local writer semaphore unless its failure is requested.
int swSemInit( sem_t   *sem /**< [out] semaphore */,
               int      shared /**< [in] process sharing */,
               unsigned value /**< [in] initial count */ );
int swSemInit( sem_t *sem, int shared, unsigned value )
{
    if( streamWriterHarness::g_faults.m_failSemInit )
    {
        errno = ENOSPC;
        return -1;
    }
    return ::sem_init( sem, shared, value );
}
/// Record a write notification without queuing asynchronous work.
int swPost( sem_t *sem /**< [in/out] writer semaphore */ );
int swPost( sem_t * )
{
    if( streamWriterHarness::g_faults.m_failPost )
    {
        errno = EINVAL;
        return -1;
    }
    return 0;
}
/// Supply one deterministic semaphore event.
int swWait( sem_t *sem /**< [in] source or writer semaphore */, const timespec *deadline /**< [in] deadline */ );
int swWait( sem_t *, const timespec * )
{
    if( ++streamWriterHarness::g_faults.m_waitCalls > 64 )
        throw std::runtime_error( "worker loop exceeded event limit" );
    return streamWriterHarness::g_faults.m_wait();
}
/// Drain the synthetic stream or inject a drain failure.
int swTryWait( sem_t *sem /**< [in] source semaphore */ );
int swTryWait( sem_t * )
{
    auto &f = streamWriterHarness::g_faults;
    if( ++f.m_tryWaitCalls > 64 )
        throw std::runtime_error( "semaphore drain exceeded event limit" );
    if( f.m_tryWait )
        return f.m_tryWait();
    errno = EAGAIN;
    return -1;
}
/// Acquire real realtime unless a clock failure is selected.
int swClock( clockid_t clock /**< [in] clock ID */, timespec *time /**< [out] current time */ );
int swClock( clockid_t clock, timespec *time )
{
    if( streamWriterHarness::g_faults.m_failClock )
    {
        errno = EINVAL;
        return -1;
    }
    int result = ::clock_gettime( clock, time );
    ++streamWriterHarness::g_faults.m_clockCalls;
    if( streamWriterHarness::g_faults.m_afterClock )
        streamWriterHarness::g_faults.m_afterClock();
    return result;
}
/// Run a synchronous initialization or retry event instead of sleeping.
unsigned swSleep( unsigned seconds /**< [in] requested wait */ );
unsigned swSleep( unsigned )
{
    if( ++streamWriterHarness::g_faults.m_sleepCalls > 16 )
        throw std::runtime_error( "discovery loop exceeded event limit" );
    streamWriterHarness::g_faults.m_sleep();
    return 0;
}
namespace mx::sys
{
/// Run the same synchronous retry event for the mxlib sleep overload.
int swSleep( double seconds /**< [in] requested wait */ );
int swSleep( double )
{
    ::swSleep( 0 );
    return 0;
}
} // namespace mx::sys
/// Simulate opening the private stream file.
int swOpen( const char *path /**< [in] stream path */, int flags /**< [in] access flags */ );
int swOpen( const char *, int )
{
    auto &f = streamWriterHarness::g_faults;
    return f.m_openFile ? f.m_openFile() : 42;
}
/// Close the synthetic descriptor without touching a process descriptor.
int swClose( int descriptor /**< [in] synthetic descriptor */ );
int swClose( int )
{
    return 0;
}
/// Supply a stable inode unless the test models replacement or disappearance.
int swStat( const char *path /**< [in] stream path */, struct stat *info /**< [out] file metadata */ );
int swStat( const char *, struct stat *info )
{
    info->st_ino = 42;
    auto &f      = streamWriterHarness::g_faults;
    int   result = f.m_stat ? f.m_stat() : 0;
    if( result > 0 )
    {
        info->st_ino = result;
        return 0;
    }
    return result;
}
/// Borrow the fixture's synthetic ImageStreamIO image.
int swOpenImage( IMAGE *image /**< [out] borrowed image */, const char *name /**< [in] stream name */ );
int swOpenImage( IMAGE *image, const char * )
{
    auto &f = streamWriterHarness::g_faults;
    *image  = f.m_image;
    return f.m_openImage ? f.m_openImage() : 0;
}
/// Count image closes without releasing borrowed fixture data.
int swCloseImage( IMAGE *image /**< [in/out] borrowed image */ );
int swCloseImage( IMAGE * )
{
    ++streamWriterHarness::g_faults.m_closeCalls;
    return 0;
}
/// Return an available synthetic semaphore index or the selected failure.
int swGetIndex( IMAGE *image /**< [in] borrowed image */, int preferred /**< [in] preferred semaphore */ );
int swGetIndex( IMAGE *, int )
{
    auto &f = streamWriterHarness::g_faults;
    return f.m_getIndex ? f.m_getIndex() : 0;
}
/// No stale semaphore posts are retained in the synchronous fixture.
int swFlush( IMAGE *image /**< [in] borrowed image */, long index /**< [in] semaphore index */ );
int swFlush( IMAGE *, long )
{
    return 0;
}
/// \endcond

#define std streamWriterTestStd
#define MagAOXApp streamWriterTestApp
#define telemeter streamWriterTestTelemeter
#define malloc swMalloc
#define xrif_configure swConfigure
#define xrif_set_size swSetSize
#define xrif_allocate_reordered swAllocateReordered
#define xrif_set_lz4_acceleration swLz4
#define xrif_encode swEncode
#define xrif_write_header swHeader
#define fwrite swWrite
#define xrif_new swNew
#define xrif_allocate_raw swAllocateRaw
#define sigaction( ... ) swSigaction( __VA_ARGS__ )
#define pthread_tryjoin_np swTryJoin
#define sem_init swSemInit
#define sem_post swPost
#define sem_timedwait swWait
#define sem_trywait swTryWait
#define clock_gettime swClock
#define sleep swSleep
#define open swOpen
#define close swClose
#define stat( ... ) swStat( __VA_ARGS__ )
#define ImageStreamIO_openIm swOpenImage
#define ImageStreamIO_closeIm swCloseImage
#define ImageStreamIO_getsemwaitindex swGetIndex
#define ImageStreamIO_semflush swFlush
#define protected public
#include "../streamWriter.hpp"
#undef protected
#undef ImageStreamIO_semflush
#undef ImageStreamIO_getsemwaitindex
#undef ImageStreamIO_closeIm
#undef ImageStreamIO_openIm
#undef stat
#undef close
#undef open
#undef sleep
#undef clock_gettime
#undef sem_trywait
#undef sem_timedwait
#undef sem_post
#undef sem_init
#undef pthread_tryjoin_np
#undef sigaction
#undef xrif_configure
#undef xrif_set_size
#undef xrif_allocate_reordered
#undef xrif_set_lz4_acceleration
#undef xrif_encode
#undef xrif_write_header
#undef fwrite
#undef xrif_allocate_raw
#undef xrif_new
#undef malloc
#undef telemeter
#undef MagAOXApp
#undef std

namespace libXWCTest::streamWriterTest
{
/** \addtogroup streamWriter_unit_test
 * @{ */
using MagAOX::app::streamWriter;
using namespace streamWriterHarness;
using namespace MagAOX::logger;
using MagAOX::app::stateCodes;

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
/// Own an isolated output tree and a synthetic source with synchronous worker events.
class Fixture
{
  public:
    /// Set up a private directory, real encoder handles, and borrowed stream metadata.
    Fixture();
    /// Free the private tree after app-owned resources have been released.
    ~Fixture();
    /// Supply one new frame and clear completed write ownership.
    void frame( uint64_t count /**< [in] frame counter */, long seconds = 100 /**< [in] acquisition time */ );
    /// Configure the synthetic source and real circular buffers for direct encoding.
    void buffers();
    /// Encode one frame using the production output path.
    int encode();
    /// Initialize the INDI status properties without launching workers.
    void properties();

    streamWriter          m_app;                                ///< Production application under test.
    std::filesystem::path m_path;                               ///< Owned unique output directory.
    IMAGE_METADATA        m_metadata{};                         ///< Borrowed source metadata.
    uint16_t              m_pixels[2]{ 10, 20 };                ///< Two synthetic one-pixel frames.
    uint64_t              m_counts[2]{ 0, 0 };                  ///< Frame-array counters.
    timespec              m_times[2]{ { 100, 1 }, { 100, 2 } }; ///< Frame acquisition and write times.
    sem_t                 m_sem{};             ///< Harmless semaphore identity; synchronous wrappers handle waits.
    sem_t                *m_sems[1]{ &m_sem }; ///< Borrowed semaphore pointer table.
    pid_t                 m_readers[1]{ 0 };   ///< Borrowed source reader ownership slot.
};
Fixture::Fixture()
{
    reset();
    char  pattern[] = "/tmp/streamWriter-fault-XXXXXX";
    char *path      = ::mkdtemp( pattern );
    REQUIRE( path != nullptr );
    m_path                         = path;
    m_app.m_configName             = "streamWriter-fault";
    m_app.m_outName                = "streamWriter-fault";
    m_app.m_rawimageDir            = m_path.string();
    m_app.m_fgThreadInit           = false;
    m_app.m_swThreadInit           = false;
    m_app.m_maxCircBuffLength      = 4;
    m_app.m_maxWriteChunkLength    = 2;
    m_app.m_maxCircBuffSize        = 1;
    m_app.m_semWaitNSec            = 1000;
    m_app.m_writeCompletionTimeout = 0;
    m_metadata.naxis               = 3;
    m_metadata.size[0] = m_metadata.size[1] = 1;
    m_metadata.size[2]                      = 2;
    m_metadata.datatype                     = _DATATYPE_UINT16;
    m_metadata.sem                          = SEMAPHORE_MAXVAL;
    g_faults.m_image.md                     = &m_metadata;
    g_faults.m_image.array.raw              = m_pixels;
    g_faults.m_image.cntarray               = m_counts;
    g_faults.m_image.atimearray             = m_times;
    g_faults.m_image.writetimearray         = m_times;
    g_faults.m_image.semptr                 = m_sems;
    g_faults.m_image.semReadPID             = m_readers;
    g_faults.m_wait                         = [this]
    {
        m_app.m_shutdown = 1;
        errno            = EINTR;
        return -1;
    };
    g_faults.m_sleep = [this] { m_app.m_shutdown = 1; };
}
Fixture::~Fixture()
{
    std::error_code error;
    std::filesystem::remove_all( m_path, error );
}
void Fixture::frame( uint64_t count, long seconds )
{
    m_app.m_writePending = false;
    m_metadata.cnt0 = m_counts[0] = count;
    m_times[0]                    = { seconds, 1 };
    m_metadata.atime = m_metadata.writetime = m_times[0];
}
void Fixture::buffers()
{
    m_app.m_width = m_app.m_height = 1;
    m_app.m_typeSize               = 2;
    m_app.m_dataType               = _DATATYPE_UINT16;
    REQUIRE( m_app.initialize_xrif() == 0 );
    REQUIRE( m_app.allocate_circbufs() == 0 );
    REQUIRE( m_app.allocate_xrif() == 0 );
    reinterpret_cast<uint16_t *>( m_app.m_rawImageCircBuff )[0] = 23;
    auto *t                                                     = m_app.m_timingCircBuff;
    t[0]                                                        = 1;
    t[1]                                                        = 100;
    t[2]                                                        = 1;
    t[3]                                                        = 100;
    t[4]                                                        = 1;
}
int Fixture::encode()
{
    m_app.m_writing             = STOP_WRITING;
    m_app.m_currSaveStart       = 0;
    m_app.m_currSaveStop        = 1;
    m_app.m_currSaveStopFrameNo = 1;
    m_app.m_writePending        = true;
    return m_app.doEncode();
}
void Fixture::properties()
{
    m_app.createStandardIndiToggleSw( m_app.m_indiP_writing, "writing" );
    m_app.m_indiP_xrifStats = pcf::IndiProperty( pcf::IndiProperty::Number );
    for( const auto *name : { "ratio",
                              "differenceMBsec",
                              "reorderMBsec",
                              "compressMBsec",
                              "encodeMBsec",
                              "differenceFPS",
                              "reorderFPS",
                              "compressFPS",
                              "encodeFPS" } )
        m_app.m_indiP_xrifStats.add( pcf::IndiElement( name ) );
}
/// \endcond

/** \brief Schedule idle and active telemetry and propagate all helper failures.
 * \ingroup streamWriter_unit_test */
TEST_CASE( "streamWriter schedules telemetry in every writing state and honors helper failures", "[streamWriter]" )
{
    // clang-format off
#ifdef STREAMWRITER_TEST_DOXYGEN_REF
    XWCTEST_DOXYGEN_REF(streamWriter::setupConfig());
    XWCTEST_DOXYGEN_REF(streamWriter::loadConfigImpl(mx::app::appConfigurator()));
    XWCTEST_DOXYGEN_REF(streamWriter::loadConfig());
    XWCTEST_DOXYGEN_REF(streamWriter::appLogic());
    XWCTEST_DOXYGEN_REF(streamWriter::appShutdown());
    XWCTEST_DOXYGEN_REF(streamWriter::checkRecordTimes());
    XWCTEST_DOXYGEN_REF(streamWriter::recordTelem(nullptr));
#endif
    // clang-format on
    Fixture f;
    auto   &app = f.m_app;
    f.properties();
    SECTION( "due idle records are emitted, with active records in all writing phases" )
    {
        for( int writing : { NOT_WRITING, START_WRITING, WRITING, STOP_WRITING } )
        {
            app.m_writing  = writing;
            g_faults.m_due = true;
            REQUIRE( app.appLogic() == 0 );
            REQUIRE( app.state() == ( writing == NOT_WRITING ? stateCodes::READY : stateCodes::OPERATING ) );
            const auto &bytes  = g_faults.m_telemetry.back();
            auto       *record = MagAOX::logger::GetSaving_state_change_fb( bytes.data() );
            REQUIRE( record->state() == ( writing == NOT_WRITING ? 0 : 1 ) );
        }
        REQUIRE( g_faults.m_schedules == 4 );
        REQUIRE( g_faults.m_telemetry.size() == 4 );
    }
    SECTION( "setup error requests shutdown" )
    {
        g_faults.m_telemSetup = -1;
        app.setupConfig();
        REQUIRE( app.shutdown() );
        REQUIRE( logs( "setupConfig" ) == 1 );
    }
    SECTION( "load error requests shutdown" )
    {
        app.setupConfig();
        g_faults.m_telemLoad = -1;
        app.loadConfig();
        REQUIRE( app.shutdown() );
        REQUIRE( logs( "loadConfig" ) == 1 );
    }
    SECTION( "negative stop timeout is clamped" )
    {
        app.setupConfig();
        app.m_writeStopTimeout = -1;
        REQUIRE( app.loadConfigImpl( app.config ) == 0 );
        REQUIRE( app.m_writeStopTimeout == 0 );
    }
    SECTION( "logic errors are fatal while idle and active" )
    {
        g_faults.m_telemLogic = -1;
        for( int writing : { NOT_WRITING, WRITING } )
        {
            app.m_writing = writing;
            REQUIRE( app.appLogic() == -1 );
        }
        REQUIRE( g_faults.m_schedules == 2 );
    }
    SECTION( "shutdown records helper failure and still succeeds" )
    {
        g_faults.m_telemShutdown = -1;
        REQUIRE( app.appShutdown() == 0 );
        REQUIRE( logs( "appShutdown" ) == 1 );
    }
}

/** \brief Exercise startup and worker checks at every failing dependency boundary.
 * \ingroup streamWriter_unit_test */
TEST_CASE( "streamWriter startup and worker health report dependency failures", "[streamWriter]" )
{
    // clang-format off
#ifdef STREAMWRITER_TEST_DOXYGEN_REF
    XWCTEST_DOXYGEN_REF(streamWriter::appStartup());
    XWCTEST_DOXYGEN_REF(streamWriter::appLogic());
    XWCTEST_DOXYGEN_REF(streamWriter::appShutdown());
    XWCTEST_DOXYGEN_REF(streamWriter::setSigSegvHandler());
    XWCTEST_DOXYGEN_REF(streamWriter::_handlerSigSegv(0,nullptr,nullptr));
#endif
    // clang-format on
    Fixture f;
    auto   &app = f.m_app;
    SECTION( "both signal installation failures abort startup" )
    {
        auto call          = GENERATE( 1u, 2u );
        g_faults.m_failSig = call;
        REQUIRE( app.appStartup() == -1 );
        REQUIRE( g_faults.m_sigCalls == call );
        REQUIRE( g_faults.m_threadCalls == 0 );
    }
    SECTION( "semaphore initialization failure aborts startup" )
    {
        g_faults.m_failSemInit = true;
        REQUIRE( app.appStartup() == -1 );
        REQUIRE( g_faults.m_threadCalls == 0 );
    }
    SECTION( "worker start failures preserve existing startup returns" )
    {
        auto call             = GENERATE( 1u, 2u );
        g_faults.m_failThread = call;
        REQUIRE( app.appStartup() == ( call == 1 ? -1 : 0 ) );
        REQUIRE( logs( "SW FILE:" ) >= 1 );
        REQUIRE( ::sem_destroy( &app.m_swSemaphore ) == 0 );
    }
    SECTION( "encoder initialization failure is diagnosed during startup" )
    {
        g_faults.m_failNew = 1;
        REQUIRE( app.appStartup() == 0 );
        REQUIRE( logs( "allocation or initialization" ) == 1 );
        REQUIRE( ::sem_destroy( &app.m_swSemaphore ) == 0 );
    }
    SECTION( "telemetry startup failure aborts startup" )
    {
        g_faults.m_telemStartup = -1;
        REQUIRE( app.appStartup() == -1 );
        REQUIRE( ::sem_destroy( &app.m_swSemaphore ) == 0 );
    }
    SECTION( "worker exit and exception both fail appLogic before scheduling" )
    {
        auto call               = GENERATE( 1u, 2u );
        auto throwing           = GENERATE( false, true );
        g_faults.m_failTryJoin  = call;
        g_faults.m_throwTryJoin = throwing;
        REQUIRE( app.appLogic() == -1 );
        REQUIRE( g_faults.m_schedules == 0 );
        REQUIRE( logs( "thread has exited" ) == 1 );
    }
    SECTION( "join exceptions do not prevent resource cleanup" )
    {
        f.buffers();
        g_faults.m_failJoin = GENERATE( 1u, 2u );
        REQUIRE( app.appShutdown() == 0 );
        REQUIRE( app.m_rawImageCircBuff == nullptr );
        REQUIRE( app.m_xrif == nullptr );
        REQUIRE( app.m_xrif_timing == nullptr );
    }
    SECTION( "signal trampoline requests source restart" )
    {
        streamWriter::_handlerSigSegv( SIGBUS, nullptr, nullptr );
        REQUIRE( app.m_restart );
    }
    SECTION( "summary warning backoff saturates and resets when quiet" )
    {
        f.properties();
        app.m_nextSkipSummaryTime    = 1;
        app.m_skipSummaryIntervalSec = 40;
        app.m_repeatSemaphoreCount   = 2;
        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.m_skipSummaryIntervalSec == 60 );
        REQUIRE( logs( "2 repeated semaphore wakes" ) == 1 );
        REQUIRE( g_faults.m_logs.back().m_priority == flatlogs::logPrio::LOG_WARNING );
        app.m_nextSkipSummaryTime = 1;
        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.m_skipSummaryIntervalSec == 10 );
    }
}

/** \brief Check failed allocations, resizing, callback dispatch, and encoder cleanup.
 * \ingroup streamWriter_unit_test */
TEST_CASE( "streamWriter allocations and output failures release write ownership", "[streamWriter]" )
{
    // clang-format off
#ifdef STREAMWRITER_TEST_DOXYGEN_REF
    XWCTEST_DOXYGEN_REF(streamWriter::initialize_xrif());
    XWCTEST_DOXYGEN_REF(streamWriter::allocate_circbufs());
    XWCTEST_DOXYGEN_REF(streamWriter::allocate_xrif());
    XWCTEST_DOXYGEN_REF(streamWriter::doEncode());
    XWCTEST_DOXYGEN_REF(streamWriter::appShutdown());
    XWCTEST_DOXYGEN_REF(streamWriter::st_newCallBack_m_indiP_writing(nullptr,pcf::IndiProperty()));
#endif
    // clang-format on
    Fixture f;
    auto   &app = f.m_app;
    SECTION( "either XRIF handle allocation can fail" )
    {
        g_faults.m_failNew = GENERATE( 1u, 2u );
        REQUIRE( app.initialize_xrif() == -1 );
        REQUIRE( logs( "allocation or initialization" ) == 1 );
    }
    SECTION( "either XRIF header allocation can fail" )
    {
        g_faults.m_failMalloc = GENERATE( 1u, 2u );
        REQUIRE( app.initialize_xrif() == -1 );
        REQUIRE( logs( "header allocation failed" ) == 1 );
    }
    SECTION( "either circular buffer allocation can fail" )
    {
        f.buffers();
        g_faults.m_failMalloc = g_faults.m_mallocCalls + GENERATE( 1u, 2u );
        REQUIRE( app.allocate_circbufs() == -1 );
        REQUIRE( logs( "buffer allocation failure" ) == 1 );
    }
    SECTION( "reallocation frees old storage" )
    {
        f.buffers();
        REQUIRE( app.allocate_circbufs() == 0 );
        REQUIRE( app.m_rawImageCircBuff != nullptr );
        REQUIRE( app.m_timingCircBuff != nullptr );
    }
    SECTION( "invalid chunk sizes fail circular buffer validation" )
    {
        app.m_width = app.m_height = app.m_typeSize = 1;
        app.m_maxWriteChunkLength                   = GENERATE( 4u, 3u );
        REQUIRE( app.allocate_circbufs() == -1 );
        REQUIRE( app.m_rawImageCircBuff == nullptr );
    }
    SECTION( "generic filesystem exception clears write ownership" )
    {
        f.buffers();
        g_faults.m_failDirectories = true;
        REQUIRE( f.encode() == -1 );
        REQUIRE_FALSE( app.m_writePending );
        REQUIRE( logs( "injected directory failure" ) == 1 );
    }
    SECTION( "both empty and nonempty reconnect flushes resume writing" )
    {
        auto count = GENERATE( 0u, 1u );
        f.buffers();
        app.m_resumeAfterReconnect = true;
        if( count == 0 )
        {
            app.m_currSaveStop = app.m_currSaveStart = 0;
            app.m_writing                            = STOP_WRITING;
            REQUIRE( app.doEncode() == 0 );
        }
        else
            REQUIRE( f.encode() == 0 );
        REQUIRE( app.m_writing == START_WRITING );
        REQUIRE_FALSE( app.m_writePending );
    }
    SECTION( "real static callback validates and changes the writer state" )
    {
        f.properties();
        pcf::IndiProperty request = app.m_indiP_writing;
        request["toggle"].setSwitchState( pcf::IndiElement::On );
        REQUIRE( streamWriter::st_newCallBack_m_indiP_writing( &app, request ) == 0 );
        REQUIRE( app.m_writing == START_WRITING );
    }
    SECTION( "shutdown-before-start releases preallocated circular buffers" )
    {
        f.buffers();
        app.m_shutdown = 1;
        streamWriter::fgThreadStart( &app );
        REQUIRE( app.m_rawImageCircBuff == nullptr );
        REQUIRE( app.m_timingCircBuff == nullptr );
    }
}

/** \brief Cover discovery retries and all failures before source ingestion begins.
 * \ingroup streamWriter_unit_test */
TEST_CASE( "streamWriter discovery and ingest initialization have bounded exits", "[streamWriter]" )
{
    // clang-format off
#ifdef STREAMWRITER_TEST_DOXYGEN_REF
    XWCTEST_DOXYGEN_REF(streamWriter::fgThreadExec());
    XWCTEST_DOXYGEN_REF(streamWriter::fgThreadStart(nullptr));
    XWCTEST_DOXYGEN_REF(streamWriter::allocate_circbufs());
    XWCTEST_DOXYGEN_REF(streamWriter::allocate_xrif());
#endif
    // clang-format on
    Fixture f;
    auto   &app = f.m_app;
    REQUIRE( app.initialize_xrif() == 0 );
    SECTION( "initialization wait respects shutdown" )
    {
        app.m_fgThreadInit = true;
        app.fgThreadExec();
        REQUIRE( app.shutdown() );
    }
    SECTION( "missing stream retry respects shutdown" )
    {
        g_faults.m_openFile = []
        {
            errno = ENOENT;
            return -1;
        };
        app.fgThreadExec();
        REQUIRE( logs( "not found (yet)" ) == 1 );
    }
    SECTION( "source not ready closes before retry" )
    {
        f.m_metadata.sem = 0;
        app.fgThreadExec();
        REQUIRE( g_faults.m_closeCalls == 1 );
    }
    SECTION( "openImage failure retries without closing an unopened image" )
    {
        g_faults.m_openImage = [] { return -1; };
        app.fgThreadExec();
        REQUIRE( g_faults.m_closeCalls == 0 );
    }
    SECTION( "restart during discovery retries from the beginning" )
    {
        g_faults.m_openFile = [&]
        {
            app.m_restart = true;
            return -1;
        };
        g_faults.m_sleep = [&] { app.m_shutdown = 1; };
        app.fgThreadExec();
        REQUIRE( app.shutdown() );
    }
    SECTION( "shutdown after successful open closes the source" )
    {
        g_faults.m_openImage = [&]
        {
            app.m_shutdown = 1;
            return 0;
        };
        app.fgThreadExec();
        REQUIRE( g_faults.m_closeCalls == 1 );
    }
    SECTION( "initial inode failure closes the source and exits" )
    {
        g_faults.m_stat = []
        {
            errno = ENOENT;
            return -1;
        };
        app.fgThreadExec();
        REQUIRE( g_faults.m_closeCalls == 1 );
        REQUIRE( logs( "Could not get inode" ) == 1 );
    }
    SECTION( "semaphore selection failure exits with a diagnostic" )
    {
        g_faults.m_getIndex = [] { return -1; };
        app.fgThreadExec();
        REQUIRE( logs( "No valid semaphore" ) == 1 );
    }
    SECTION( "circular buffer allocation failure exits" )
    {
        g_faults.m_failMalloc = g_faults.m_mallocCalls + 1;
        app.fgThreadExec();
        REQUIRE( logs( "buffer allocation failure" ) == 1 );
    }
    SECTION( "XRIF allocation failure exits" )
    {
        g_faults.m_failXrifAllocate = true;
        app.fgThreadExec();
        REQUIRE( logs( "xrif_allocate_raw error" ) == 1 );
    }
}

/** \brief Check semaphore errors, timestamps, source changes, and every write-notification path.
 * \ingroup streamWriter_unit_test */
TEST_CASE( "streamWriter ingest loop handles stream events and notification failures", "[streamWriter]" )
{
    // clang-format off
#ifdef STREAMWRITER_TEST_DOXYGEN_REF
    XWCTEST_DOXYGEN_REF(streamWriter::fgThreadExec());
    XWCTEST_DOXYGEN_REF(streamWriter::waitForWriteCompletion(0));
#endif
    // clang-format on
    Fixture f;
    auto   &app = f.m_app;
    REQUIRE( app.initialize_xrif() == 0 );
    unsigned events = 0;
    SECTION( "two dimensional streams use metadata and wrap circular buffers" )
    {
        f.m_metadata.naxis   = 2;
        f.m_metadata.size[2] = 1;
        g_faults.m_wait      = [&]
        {
            if( ++events <= 5 )
            {
                f.frame( events );
                return 0;
            }
            app.m_shutdown = 1;
            errno          = EINTR;
            return -1;
        };
        app.fgThreadExec();
        REQUIRE( events == 6 );
        REQUIRE( g_faults.m_closeCalls == 1 );
        REQUIRE( f.m_readers[0] == 0 );
    }
    SECTION( "repeated unchanged frames force reopen after eleven events" )
    {
        unsigned opens       = 0;
        g_faults.m_openImage = [&]
        {
            if( ++opens == 2 )
                app.m_shutdown = 1;
            return 0;
        };
        g_faults.m_wait = [&]
        {
            ++events;
            return 0;
        };
        app.fgThreadExec();
        REQUIRE( app.m_repeatSemaphoreCount == 11 );
        REQUIRE( opens == 2 );
    }
    SECTION( "drain failure exits and diagnoses only while running" )
    {
        auto shutdown      = GENERATE( false, true );
        g_faults.m_wait    = [] { return 0; };
        g_faults.m_tryWait = [&]
        {
            app.m_shutdown = shutdown;
            errno          = EINVAL;
            return -1;
        };
        unsigned opens       = 0;
        g_faults.m_openImage = [&]
        {
            if( ++opens == 2 )
                app.m_shutdown = 1;
            return 0;
        };
        app.fgThreadExec();
        REQUIRE( logs( "sem_trywait" ) == ( shutdown ? 0 : 1 ) );
    }
    SECTION( "timestamp clock failure exits before a frame can be scheduled" )
    {
        g_faults.m_wait = [&]
        {
            f.frame( 1, 0 );
            g_faults.m_failClock = true;
            return 0;
        };
        app.fgThreadExec();
        REQUIRE( logs( "clock_gettime" ) == 1 );
        REQUIRE_FALSE( app.m_writePending );
    }
    SECTION( "worker wait clock failure exits before waiting" )
    {
        g_faults.m_failClock = true;
        app.fgThreadExec();
        REQUIRE( logs( "clock_gettime" ) == 1 );
    }
    SECTION( "shutdown or restart after semaphore drain prevents frame ingestion" )
    {
        auto     restart     = GENERATE( false, true );
        unsigned opens       = 0;
        g_faults.m_openImage = [&]
        {
            if( ++opens == 2 )
                app.m_shutdown = 1;
            return 0;
        };
        g_faults.m_wait = [&]
        {
            f.frame( 1 );
            app.m_shutdown = !restart;
            app.m_restart  = restart;
            return 0;
        };
        app.fgThreadExec();
        REQUIRE( app.m_currImage == 0 );
    }
    SECTION( "every timed wait error releases the image before reopening" )
    {
        int      error       = GENERATE( EINTR, EINVAL, ETIMEDOUT );
        auto     reason      = GENERATE( 0, 1, 2, 3 );
        unsigned opens       = 0;
        g_faults.m_openImage = [&]
        {
            if( ++opens == 2 )
                app.m_shutdown = 1;
            return 0;
        };
        g_faults.m_wait = [&]
        {
            ++events;
            if( reason == 1 )
                f.m_metadata.sem = 0;
            errno = error;
            return -1;
        };
        if( reason == 2 )
            g_faults.m_openFile = [&] { return events ? -1 : 42; };
        if( reason == 3 )
            g_faults.m_stat = [&] { return events ? -1 : 0; };
        if( reason == 0 && error == ETIMEDOUT )
            g_faults.m_stat = [&] { return events ? 43 : 0; };
        app.fgThreadExec();
        REQUIRE( events == 1 );
        REQUIRE( g_faults.m_closeCalls == ( reason == 2 ? 1 : 2 ) );
        REQUIRE( logs( "sem_timedwait" ) == ( reason != 1 && error == EINVAL ? 1 : 0 ) );
    }
    SECTION( "all six notification failures clear queued buffer ownership" )
    {
        int mode           = GENERATE( 0, 1, 2, 3, 4, 5 );
        app.m_writing      = START_WRITING;
        app.m_maxChunkTime = mode == 1 ? 1 : ( mode == 3 ? 0 : 10000000000.0 );
        g_faults.m_wait    = [&]
        {
            ++events;
            if( events == 1 )
            {
                f.frame( 1 );
                return 0;
            }
            g_faults.m_failPost = true;
            if( mode == 0 )
            {
                f.frame( 2 );
                return 0;
            }
            if( mode == 1 )
            {
                app.m_writeChunkLength = 4;
                f.frame( 2, 102 );
                return 0;
            }
            if( mode == 2 )
            {
                app.m_writing = STOP_WRITING;
                f.frame( 2 );
                return 0;
            }
            if( mode == 3 )
            {
                app.m_maxChunkTime = 0;
                app.m_shutdown     = 1;
                errno              = ETIMEDOUT;
                return -1;
            }
            if( mode == 4 )
            {
                app.m_writing           = STOP_WRITING;
                app.m_stopWriteDeadline = 0;
                errno                   = ETIMEDOUT;
                return -1;
            }
            f.frame( 2 );
            app.m_restart = true;
            return 0;
        };
        app.fgThreadExec();
        REQUIRE_FALSE( app.m_writePending );
        REQUIRE( logs( "Error posting to semaphore" ) == 1 );
    }
    SECTION( "image-time flush re-arms writing and aligned chunks wrap" )
    {
        app.m_writing      = START_WRITING;
        app.m_maxChunkTime = 1;
        g_faults.m_wait    = [&]
        {
            if( ++events <= 7 )
            {
                f.frame( events, 100 + events * 2 );
                return 0;
            }
            app.m_writePending = false;
            app.m_shutdown     = 1;
            errno              = EINTR;
            return -1;
        };
        app.fgThreadExec();
        REQUIRE( events == 8 );
        REQUIRE_FALSE( app.m_writePending );
        REQUIRE( g_faults.m_closeCalls == 1 );
    }
    SECTION( "pending writer timeout shuts down before releasing source buffers" )
    {
        app.m_writing   = START_WRITING;
        g_faults.m_wait = [&]
        {
            ++events;
            if( events <= 2 )
            {
                f.frame( events );
                return 0;
            }
            app.m_restart = true;
            errno         = EINTR;
            return -1;
        };
        app.fgThreadExec();
        REQUIRE( app.shutdown() );
        REQUIRE( app.m_writePending );
        REQUIRE( app.m_rawImageCircBuff != nullptr );
    }
    SECTION( "reconnect with no remainder preserves writing or completes stopping" )
    {
        bool stopping           = GENERATE( false, true );
        app.m_stopWriteDeadline = 1e20;
        unsigned opens          = 0;
        g_faults.m_openImage    = [&]
        {
            if( ++opens == 2 )
                app.m_shutdown = 1;
            return 0;
        };
        g_faults.m_wait = [&]
        {
            app.m_writing = stopping ? STOP_WRITING : WRITING;
            app.m_restart = true;
            errno         = EINTR;
            return -1;
        };
        app.fgThreadExec();
        REQUIRE( app.m_writing == ( stopping ? NOT_WRITING : START_WRITING ) );
        REQUIRE( app.m_resumeAfterReconnect == !stopping );
    }
}

/** \brief Exercise writer initialization, shutdown, timeouts, interrupts, and encode failure.
 * \ingroup streamWriter_unit_test */
TEST_CASE( "streamWriter writer loop reports clock semaphore and encoding failures", "[streamWriter]" )
{
    // clang-format off
#ifdef STREAMWRITER_TEST_DOXYGEN_REF
    XWCTEST_DOXYGEN_REF(streamWriter::swThreadStart(nullptr));
    XWCTEST_DOXYGEN_REF(streamWriter::swThreadExec());
    XWCTEST_DOXYGEN_REF(streamWriter::doEncode());
#endif
    // clang-format on
    Fixture f;
    auto   &app = f.m_app;
    SECTION( "initialization and readiness waits honor shutdown" )
    {
        bool initializing  = GENERATE( false, true );
        app.m_swThreadInit = initializing;
        streamWriter::swThreadStart( &app );
        REQUIRE( app.shutdown() );
    }
    SECTION( "realtime failure returns before semaphore wait" )
    {
        app.state( stateCodes::READY );
        g_faults.m_failClock = true;
        app.swThreadExec();
        REQUIRE( logs( "clock_gettime" ) == 1 );
    }
    SECTION( "timeout and interrupt retry while other errors stop the worker" )
    {
        app.state( stateCodes::READY );
        int      error  = GENERATE( EINTR, ETIMEDOUT, EINVAL );
        unsigned events = 0;
        g_faults.m_wait = [&]
        {
            if( ++events == 2 )
                app.m_shutdown = 1;
            errno = error;
            return -1;
        };
        app.swThreadExec();
        REQUIRE( events == ( error == EINVAL ? 1 : 2 ) );
        REQUIRE( logs( "sem_timedwait" ) == ( error == EINVAL ? 1 : 0 ) );
    }
    SECTION( "successful no-op encode clears pending ownership" )
    {
        app.state( stateCodes::OPERATING );
        app.m_writePending = true;
        unsigned events    = 0;
        g_faults.m_wait    = [&]
        {
            if( ++events == 2 )
            {
                app.m_shutdown = 1;
                errno          = EINTR;
                return -1;
            }
            return 0;
        };
        app.swThreadExec();
        REQUIRE_FALSE( app.m_writePending );
        REQUIRE( logs( "error encoding data" ) == 0 );
    }
    SECTION( "encode error returns after clearing pending ownership" )
    {
        f.buffers();
        app.state( stateCodes::READY );
        app.m_writing              = WRITING;
        app.m_currSaveStop         = 1;
        app.m_writePending         = true;
        g_faults.m_failDirectories = true;
        g_faults.m_wait            = [] { return 0; };
        app.swThreadExec();
        REQUIRE_FALSE( app.m_writePending );
        REQUIRE( logs( "error encoding data" ) == 1 );
    }
}

/** \brief Preserve encoder setup failures and validate real archives when XRIF reports warnings.
 * \ingroup streamWriter_unit_test */
TEST_CASE( "streamWriter XRIF errors and short headers retain their documented outcomes", "[streamWriter]" )
{
    // clang-format off
#ifdef STREAMWRITER_TEST_DOXYGEN_REF
    XWCTEST_DOXYGEN_REF(streamWriter::initialize_xrif());
    XWCTEST_DOXYGEN_REF(streamWriter::allocate_xrif());
    XWCTEST_DOXYGEN_REF(streamWriter::doEncode());
    XWCTEST_DOXYGEN_REF(streamWriter::recordSavingStats(true));
#endif
    // clang-format on
    Fixture f;
    auto   &app = f.m_app;
    SECTION( "initial XRIF configuration failures abort for either compression policy and handle" )
    {
        app.m_compress              = GENERATE( false, true );
        g_faults.m_configure.m_fail = GENERATE( 1u, 2u );
        REQUIRE( app.initialize_xrif() == -1 );
        REQUIRE( logs( "configuration error" ) == 1 );
    }
    SECTION( "each buffer setup operation reports a selected error" )
    {
        int  operation = GENERATE( 0, 1, 2, 3 );
        auto call      = GENERATE( 1u, 2u );
        app.m_compress = GENERATE( false, true );
        REQUIRE( app.initialize_xrif() == 0 );
        app.m_width = app.m_height = 1;
        app.m_dataType             = _DATATYPE_UINT16;
        auto *fault                = operation == 0   ? &g_faults.m_configure
                                     : operation == 1 ? &g_faults.m_setSize
                                     : operation == 2 ? &g_faults.m_allocateRaw
                                                      : &g_faults.m_allocateReordered;
        *fault                     = XrifFault{};
        fault->m_fail              = call;
        REQUIRE( app.allocate_xrif() == -1 );
        REQUIRE( logs( "error." ) == 1 );
    }
    SECTION( "encode warnings retain a decodable image and timing archive" )
    {
        int  operation = GENERATE( 0, 1, 2, 3 );
        auto call      = GENERATE( 1u, 2u );
        f.buffers();
        auto *fault    = operation == 0   ? &g_faults.m_setSize
                         : operation == 1 ? &g_faults.m_lz4
                         : operation == 2 ? &g_faults.m_encode
                                          : &g_faults.m_header;
        *fault         = XrifFault{};
        fault->m_fail  = call;
        fault->m_after = true;
        REQUIRE( f.encode() == 0 );
        REQUIRE_FALSE( app.m_writePending );
        REQUIRE( logs( "error." ) >= 1 );
        std::ifstream     file( app.m_outFilePath, std::ios::binary );
        std::vector<char> bytes{ std::istreambuf_iterator<char>( file ), std::istreambuf_iterator<char>() };
        REQUIRE( bytes.size() > XRIF_HEADER_SIZE * 2 );
        xrif_t decoder = nullptr;
        REQUIRE( ::xrif_new( &decoder ) == XRIF_NOERROR );
        std::unique_ptr<std::remove_pointer_t<xrif_t>, decltype( &::xrif_delete )> decoderOwner( decoder,
                                                                                                 ::xrif_delete );
        uint32_t                                                                   header = 0;
        REQUIRE( ::xrif_read_header( decoder, &header, bytes.data() ) == XRIF_NOERROR );
        REQUIRE( ::xrif_allocate( decoder ) == XRIF_NOERROR );
        memcpy( decoder->raw_buffer, bytes.data() + header, decoder->compressed_size );
        REQUIRE( ::xrif_decode( decoder ) == XRIF_NOERROR );
        REQUIRE( reinterpret_cast<uint16_t *>( decoder->raw_buffer )[0] == 23 );
        const auto timingOffset = header + decoder->compressed_size;
        REQUIRE( ::xrif_read_header( decoder, &header, bytes.data() + timingOffset ) == XRIF_NOERROR );
        REQUIRE( ::xrif_allocate( decoder ) == XRIF_NOERROR );
        memcpy( decoder->raw_buffer, bytes.data() + timingOffset + header, decoder->compressed_size );
        REQUIRE( ::xrif_decode( decoder ) == XRIF_NOERROR );
        REQUIRE( reinterpret_cast<uint64_t *>( decoder->raw_buffer )[0] == 1 );
        REQUIRE( reinterpret_cast<uint64_t *>( decoder->raw_buffer )[1] == 100 );
    }
    SECTION( "short image and timing headers produce alerts and a truncated archive" )
    {
        f.buffers();
        auto call            = GENERATE( 1u, 3u );
        g_faults.m_failWrite = call;
        REQUIRE( f.encode() == 0 );
        REQUIRE_FALSE( app.m_writePending );
        REQUIRE( logs( call == 1 ? "failure writing header" : "failure writing timing header" ) == 1 );
        REQUIRE( std::filesystem::file_size( app.m_outFilePath ) ==
                 XRIF_HEADER_SIZE + 1 + app.m_xrif->compressed_size + app.m_xrif_timing->compressed_size );
    }
    SECTION( "telemetry exception unwinds the serialized statistics payload" )
    {
        f.buffers();
        g_faults.m_throwTelem = true;
        REQUIRE_THROWS_AS( app.recordSavingStats( true ), std::runtime_error );
        REQUIRE( g_faults.m_telemetry.size() == 1 );
        g_faults.m_throwTelem = false;
        REQUIRE( app.recordSavingStats( true ) == 0 );
        REQUIRE( g_faults.m_telemetry.size() == 2 );
    }
    SECTION( "each statistics change triggers a real telemetry payload" )
    {
        f.buffers();
        REQUIRE( app.recordSavingStats( true ) == 0 );
        g_faults.m_telemetry.clear();
        app.m_xrif->reorder_rate = 3;
        REQUIRE( app.recordSavingStats() == 0 );
        app.m_xrif->compress_rate = 4;
        REQUIRE( app.recordSavingStats() == 0 );
        REQUIRE( g_faults.m_telemetry.size() == 2 );
        auto *stats = Gettelem_saving_fb( g_faults.m_telemetry.back().data() );
        REQUIRE( stats->reorder_rate() == 3 );
        REQUIRE( stats->compress_rate() == 4 );
        REQUIRE( app.recordSavingStats() == 0 );
        REQUIRE( g_faults.m_telemetry.size() == 2 );
    }
}

/** \brief Unwind a failed constructor and honor shutdown arriving during legacy timestamp acquisition.
 * \ingroup streamWriter_unit_test */
TEST_CASE( "streamWriter construction and timestamp shutdown release their resources", "[streamWriter]" )
{
    // clang-format off
#ifdef STREAMWRITER_TEST_DOXYGEN_REF
    XWCTEST_DOXYGEN_REF(streamWriter::streamWriter());
    XWCTEST_DOXYGEN_REF(streamWriter::fgThreadExec());
#endif
    // clang-format on
    SECTION( "failed telemetry construction unwinds the app base" )
    {
        reset();
        g_faults.m_throwConstructor = true;
        REQUIRE_THROWS_AS( streamWriter(), std::runtime_error );
        g_faults.m_throwConstructor = false;
        streamWriter app;
        REQUIRE_FALSE( app.shutdown() );
    }
    SECTION( "shutdown after legacy timestamping requests a stop flush" )
    {
        Fixture f;
        auto   &app = f.m_app;
        REQUIRE( app.initialize_xrif() == 0 );
        app.m_writing         = WRITING;
        g_faults.m_afterClock = [&]
        {
            if( g_faults.m_clockCalls == 2 )
                app.m_shutdown = 1;
        };
        g_faults.m_wait = [&]
        {
            f.frame( 1, 0 );
            return 0;
        };
        app.fgThreadExec();
        REQUIRE( app.m_writing == STOP_WRITING );
        REQUIRE( app.m_stopWriteDeadline == 0 );
        REQUIRE( app.shutdown() );
        REQUIRE( app.m_writePending );
    }
}
/** @} */
} // namespace libXWCTest::streamWriterTest
