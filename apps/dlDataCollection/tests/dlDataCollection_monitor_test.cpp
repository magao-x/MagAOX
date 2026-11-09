/** \file dlDataCollection_monitor_test.cpp
 * \brief Exercise the real monitor loop and POSIX semaphores with an ImageStreamIO mapping double.
 * \ingroup dlDataCollection_unit_test
 *
 * This is not an installed ImageStreamIO ABI or Linux performance test. It measures
 * the collector Start/Stop path when the monitor misses an intervening READY state.
 */
#define CATCH_CONFIG_MAIN
#include "../../../tests/testXWC.hpp"
#include <atomic>
#include <condition_variable>
#include <sstream>
#include <vector>
#include "../../../INDI/libcommon/IndiProperty.hpp"
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <semaphore.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <thread>
#include <fcntl.h>
#include <unistd.h>

/// \cond MONITOR_BOUNDARIES
#define ImageStruct_hpp
#ifdef __APPLE__
    #undef SYS_gettid
    #define SYS_gettid SYS_getpid
/// Unused platform boundary; the fixture joins its actual C++ thread directly.
inline int pthread_tryjoin_np( pthread_t /**< [in] Thread. */, void ** /**< [out] Unused result. */ )
{
    return EBUSY;
}
#endif
namespace mx::app
{
/// Accept fixture-supplied configuration without reading host configuration files.
struct appConfigurator
{
    /// Ignore option registration at the offline boundary.
    template <class... T>
    void add( T &&.../**< [in] Unused registration arguments. */ )
    {
    }
    /// Leave fixture-supplied values unchanged.
    template <class... T>
    void operator()( T &&.../**< [in] Unused configuration arguments. */ )
    {
    }
};
} // namespace mx::app
namespace mx::sys
{
/// Preserve bounded production monitor waits.
inline void sleep( int s /**< [in] Wait duration in seconds. */ )
{
    std::this_thread::sleep_for( std::chrono::seconds( s ) );
}
} // namespace mx::sys
namespace argType
{
constexpr int Required = 0;
}
namespace indi
{
template <class... T>
/// Discard transport-only updates.
void updateIfChanged( T &&.../**< [in] Unused property update. */ )
{
}
} // namespace indi
namespace MagAOX::app
{
namespace stateCodes
{
using stateCodeT    = int;
constexpr int READY = 0, OPERATING = 1;
} // namespace stateCodes
struct software_error
{
};
struct software_critical
{
};
struct text_log
{
};
namespace logPrio
{
constexpr int LOG_NOTICE = 0, LOG_INFO = 0;
}
/// No signal-driven teardown is needed; the fixture joins the monitor thread.
inline void
sigUsr1Handler( int /**< [in] Signal. */, siginfo_t * /**< [in] Signal metadata. */, void * /**< [in] Context. */ )
{
}
} // namespace MagAOX::app
struct Message
{
    template <class... T>
    /// Ignore production log payloads.
    Message( T &&.../**< [in] Unused message fields. */ )
    {
    }
};
/// Minimal image metadata layout consumed by the production monitor and collector.
struct Metadata
{
    int      sem{ 10 }, naxis{ 2 }, location{ -1 }, shared{ 1 }, write{ 0 };
    uint64_t nelement{ 16 }, imdatamemsize{ 64 };
    uint8_t  datatype{ 9 };
    uint32_t size[3]{ 4, 4, 0 };
    uint64_t cnt0{ 0 }, cnt1{ 0 }, imagetype{ 0 };
    timespec atime{}, writetime{};
};
/// Non-owning mapping view; all backing storage belongs to the fixture.
struct IMAGE
{
    timespec *atimearray{ nullptr }, *writetimearray{ nullptr };
    uint64_t *cntarray{ nullptr };
    Metadata *md{ nullptr };
    sem_t   **semptr{ nullptr };
    pid_t    *semReadPID{ nullptr };
    union
    {
        void  *raw{ nullptr };
        float *F;
    } array;
};
constexpr int _DATATYPE_FLOAT  = 9;
constexpr int SEMAPHORE_MAXVAL = 10, IMAGE_NB_SEMAPHORE = 10, IMAGESTREAMIO_SUCCESS = 0, CIRCULAR_BUFFER = 1,
              ZAXIS_TEMPORAL = 2;
namespace boundary
{
/// Observed monitor activity; source attachment must survive the missed READY transition.
inline std::atomic<unsigned> waits{ 0 }, opens{ 0 }, closes{ 0 };
/// Source and command storage outlives the joined monitor thread.
inline Metadata    md;
inline sem_t      *semaphores[10]{};
inline pid_t       readers[10]{};
inline float       pixels[16]{}, commands[2500]{};
inline Metadata    commandMd;
inline std::string commandPath;
/// Hold the actual monitor wait across the quick Stop/Start pair.
inline std::mutex              gate;
inline std::condition_variable changed;
inline bool                    blocked{ false }, release{ false };
inline std::string             path;
} // namespace boundary
/// Resolve only private fixture mappings.
inline int ImageStreamIO_filename( char       *out /**< [out] Path buffer. */,
                                   size_t      count /**< [in] Buffer length. */,
                                   const char *name /**< [in] Synthetic stream. */ )
{
    std::snprintf( out,
                   count,
                   "%s",
                   ( std::string( name ) == "collector-command" ? boundary::commandPath : boundary::path ).c_str() );
    return 0;
}
/// Attach a synthetic source or command mapping.
inline int ImageStreamIO_openIm( IMAGE      *image /**< [out] Mapping handle. */,
                                 const char *name /**< [in] Synthetic stream. */ )
{
    if( std::string( name ) == "collector-command" )
    {
        image->md        = &boundary::commandMd;
        image->array.raw = boundary::commands;
        image->semptr    = boundary::semaphores;
        return 0;
    }
    image->md         = &boundary::md;
    image->semptr     = boundary::semaphores;
    image->semReadPID = boundary::readers;
    image->array.raw  = boundary::pixels;
    ++boundary::opens;
    return 0;
}
/// Track source detachments without destroying fixture storage.
inline int ImageStreamIO_closeIm( IMAGE *image /**< [in,out] Mapping handle. */ )
{
    if( image->md == &boundary::md )
    {
        ++boundary::closes;
    }
    image->md = nullptr;
    return 0;
}
/// A synthetic command channel has no downstream consumer in this attachment-only test.
inline int ImageStreamIO_sempost( IMAGE * /**< [in] Unused mapping. */, int /**< [in] Semaphore slot. */ )
{
    return 0;
}
/// Reserve the monitor reader slot in private metadata.
inline int ImageStreamIO_getsemwaitindex( IMAGE * /**< [in] Mapping. */, int slot /**< [in] Reader slot. */ )
{
    boundary::readers[slot] = getpid();
    return slot;
}
/// Drain stale notifications from the private semaphore.
inline int ImageStreamIO_semflush( IMAGE * /**< [in] Mapping. */, int slot /**< [in] Reader slot. */ )
{
    while( sem_trywait( boundary::semaphores[slot] ) == 0 )
    {
    }
    return 0;
}
/// The fixture exposes only float32 mappings.
inline size_t ImageStreamIO_typesize( uint8_t /**< [in] Fixture datatype. */ )
{
    return sizeof( float );
}
/// External stream destruction is outside this attachment-only fixture.
inline int ImageStreamIO_destroyIm( IMAGE * /**< [in] Unused mapping. */ )
{
    return 0;
}
/// Reject unexpected stream creation; the fixture only attaches owned storage.
inline int ImageStreamIO_createIm_gpu( IMAGE * /**< [out] Mapping. */,
                                       const char * /**< [in] Name. */,
                                       int /**< [in] Axes. */,
                                       uint32_t * /**< [in] Sizes. */,
                                       uint8_t /**< [in] Type. */,
                                       int /**< [in] Location. */,
                                       int /**< [in] Shared. */,
                                       int /**< [in] Semaphores. */,
                                       int /**< [in] Keywords. */,
                                       int /**< [in] Flags. */,
                                       int /**< [in] Circular-buffer depth. */ )
{
    return -1;
}
/// Instrument every actual monitor wait; Darwin needs a bounded POSIX polling fallback.
inline int countedWait( sem_t          *sem /**< [in] Private semaphore. */,
                        const timespec *until /**< [in] Absolute deadline. */ )
{
    ++boundary::waits;
    { //mutex scope
        std::unique_lock<std::mutex> lock( boundary::gate );
        boundary::blocked = true;
        boundary::changed.notify_all();
        boundary::changed.wait( lock,
                                []
                                {
                                    return boundary::release;
                                } );
    }
#ifdef __APPLE__
    for( ;; )
    {
        if( sem_trywait( sem ) == 0 )
        {
            return 0;
        }
        if( errno != EAGAIN )
        {
            return -1;
        }
        timespec now{};
        clock_gettime( CLOCK_REALTIME, &now );
        if( now.tv_sec > until->tv_sec || ( now.tv_sec == until->tv_sec && now.tv_nsec >= until->tv_nsec ) )
        {
            errno = ETIMEDOUT;
            return -1;
        }
        std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
    }
#else
    return sem_timedwait( sem, until );
#endif
}

// Compile the production collector against the real INDI value types and monitor;
// only registration, configuration, logging and image mapping cross fake boundaries.
namespace MagAOX::app
{
/// Offline application framework; no driver or external process is created.
template <bool>
class MagAOXApp
{
  public:
    /// Socket-free property discovery endpoint, absent throughout this fixture.
    struct Driver
    {
        /// No external discovery connection is opened.
        void sendGetProperties( const pcf::IndiProperty & /**< [in] Unused request. */ )
        {
        }
    };
    Driver                  *m_indiDriver{ nullptr };      ///< No live connection.
    std::atomic<bool>        m_shutdown{ false };          ///< Read by the real monitor.
    std::atomic<int>         m_state{ stateCodes::READY }; ///< Cross-thread FSM state.
    std::mutex               m_indiMutex;                  ///< Real app readback/configuration serialization.
    mx::app::appConfigurator config;                       ///< Configuration supplied by the fixture.
    /// Accept version metadata without starting the instrument framework.
    MagAOXApp( const char * /**< [in] Revision. */, int /**< [in] Modified flag. */ )
    {
    }
    /// Preserve polymorphic production app destruction.
    virtual ~MagAOXApp() = default;
    /// Production lifecycle overrides remain virtual in the offline framework.
    virtual void setupConfig()
    {
    }
    /// Production configuration entry point.
    virtual void loadConfig()
    {
    }
    /// Production startup entry point.
    virtual int appStartup()
    {
        return 0;
    }
    /// Production application loop entry point.
    virtual int appLogic()
    {
        return 0;
    }
    /// Production shutdown entry point.
    virtual int appShutdown()
    {
        return 0;
    }
    /// Read the real fixture shutdown flag.
    int shutdown() const
    {
        return m_shutdown.load();
    }
    /// Read the state sampled by the production monitor.
    int state() const
    {
        return m_state.load();
    }
    /// Publish a state transition to the production monitor.
    void state( int value /**< [in] New FSM state. */ )
    {
        m_state = value;
    }
    /// Private fixture identity.
    std::string configName()
    {
        return "collector-test";
    }
    /// Discard logs while preserving the production return code.
    template <class T, int result = 0>
    static int log( Message /**< [in] Unused payload. */, int /**< [in] Unused priority. */ = 0 )
    {
        return result;
    }
    /// Registration has no external side effects.
    template <class... T>
    int registerIndiPropertyNew( T &&.../**< [in] Unused offline framework arguments. */ )
    {
        return 0;
    }
    /// Registration has no external side effects.
    template <class... T>
    int registerIndiPropertyReadOnly( T &&.../**< [in] Unused offline framework arguments. */ )
    {
        return 0;
    }
    /// The fixture starts and joins its monitor explicitly.
    template <class... T>
    int threadStart( T &&.../**< [in] Unused offline framework arguments. */ )
    {
        return 0;
    }
    /// Unused startup boundary; tests drive appLogic and real callbacks.
    template <class... T>
    int createStandardIndiRequestSw( T &&.../**< [in] Unused offline framework arguments. */ )
    {
        return 0;
    }
    /// Unused startup boundary.
    template <class... T>
    int createStandardIndiText( T &&.../**< [in] Unused offline framework arguments. */ )
    {
        return 0;
    }
    /// Unused startup boundary.
    template <class... T>
    int createROIndiText( T &&.../**< [in] Unused offline framework arguments. */ )
    {
        return 0;
    }
    /// Suppress transport-only readback updates.
    template <class... T>
    void updateIfChanged( T &&.../**< [in] Unused offline framework arguments. */ )
    {
    }
    /// Suppress transport-only switch updates.
    template <class... T>
    void updateSwitchIfChanged( T &&.../**< [in] Unused offline framework arguments. */ )
    {
    }
};
} //namespace MagAOX::app
#define REG_INDI_SETPROP( ... )
#define INDI_NEWCALLBACK_DECL( cls, prop ) int newCallBack_##prop( const pcf::IndiProperty & )
#define INDI_NEWCALLBACK_DEFN( cls, prop ) int cls::newCallBack_##prop
#define INDI_SETCALLBACK_DECL( cls, prop ) int setCallBack_##prop( const pcf::IndiProperty & )
#define INDI_SETCALLBACK_DEFN( cls, prop ) int cls::setCallBack_##prop
#define INDI_NEWCALLBACK( prop ) nullptr
#define INDI_VALIDATE_CALLBACK_PROPS( prop, recv )
#define DLDATACOLLECTION_TEST
#define MAGAOX_CURRENT_SHA1 "collector-monitor-test"
#define MAGAOX_REPO_MODIFIED 0
#define sem_timedwait countedWait
#include "../../../libMagAOX/app/dev/shmimMonitor.hpp"
#undef sem_timedwait

#include "../dlDataCollection.hpp"
/// \endcond

namespace libXWCTest::dlDataCollectionTest
{
/// \cond COLLECTOR_MONITOR_FIXTURE
/// Run the production collector and monitor using private in-memory image mappings.
class MonitorFixture : public MagAOX::app::dlDataCollection
{
  public:
    std::string m_semName; ///< Owned private semaphore name.
    std::thread m_thread;  ///< Actual shmimMonitor loop, joined on all assertion paths.
    /// Configure a small valid collector geometry and launch its real monitor loop.
    MonitorFixture();
    /// Release the test barrier and join before destroying image/semaphore storage.
    ~MonitorFixture();
    /// Run an actual Stop followed immediately by Start while the monitor is gated.
    void restart();
    /// Deliver fresh coherent observation values without an INDI connection.
    void observations();
    /// Release a fresh frame after the Start/Stop race has been established.
    void release();
    /// Read the source/command attachment admission flag.
    bool connected() const
    {
        return m_connected.load();
    }
    /// Read the session phase protected by the production Session mutex.
    std::string phase()
    {
        return m_session.progress().m_phase;
    }
    /// Retain an earlier run's observation-change latch until fresh Start preflight succeeds.
    void previousObservationFailure()
    {
        m_observationChanged = true;
    }
    /// Force the existing attachment deadline without a five-second test delay.
    void expire()
    {
        m_observationDeadline = std::chrono::steady_clock::now();
    }
    /// Inspect the actual app-visible start error.
    std::string error() const
    {
        return m_controlError;
    }
};

MonitorFixture::MonitorFixture()
{
    boundary::opens = boundary::closes = boundary::waits = 0;
    boundary::blocked = boundary::release = false;
    boundary::md                          = Metadata{};
    boundary::commandMd                   = Metadata{};
    boundary::commandMd.size[0] = boundary::commandMd.size[1] = 50;
    boundary::commandMd.nelement                              = 2500;
    boundary::commandMd.imdatamemsize                         = 10000;
    std::fill( std::begin( boundary::commands ), std::end( boundary::commands ), 0.F );
    std::fill( std::begin( boundary::readers ), std::end( boundary::readers ), 0 );
    m_semName = "/collector-monitor-" + std::to_string( getpid() );
    sem_unlink( m_semName.c_str() );
    auto *sem = sem_open( m_semName.c_str(), O_CREAT | O_EXCL, 0600, 0 );
    if( sem == SEM_FAILED )
    {
        throw std::runtime_error( "private semaphore creation failed" );
    }
    for( auto &entry : boundary::semaphores )
    {
        entry = sem;
    }
    boundary::path        = "/tmp/collector-source-" + std::to_string( getpid() );
    boundary::commandPath = "/tmp/collector-command-" + std::to_string( getpid() );
    std::ofstream( boundary::path ) << "private mapping";
    std::ofstream( boundary::commandPath ) << "private command mapping";
    m_bank    = "/unused-synthetic-bank";
    m_output  = "/unused-synthetic-output";
    m_channel = "collector-command";
    m_pupil   = 6;
    m_x = m_y      = { 0, 2 };
    m_shmimName    = "collector-source";
    m_smThreadInit = false;
    state( MagAOX::app::stateCodes::OPERATING );
    m_thread = std::thread(
        [this]
        {
            smThreadExec();
        } );
}

void MonitorFixture::release()
{
    { //mutex scope
        std::lock_guard<std::mutex> lock( boundary::gate );
        boundary::release = true;
    }
    boundary::changed.notify_all();
    sem_post( boundary::semaphores[5] );
}

MonitorFixture::~MonitorFixture()
{
    m_shutdown = true;
    release();
    if( m_thread.joinable() )
    {
        m_thread.join();
    }
    appShutdown();
    sem_close( boundary::semaphores[5] );
    sem_unlink( m_semName.c_str() );
    std::filesystem::remove( boundary::path );
    std::filesystem::remove( boundary::commandPath );
}

void MonitorFixture::restart()
{
    m_stopRequested = true;
    appLogic();
    REQUIRE( state() == MagAOX::app::stateCodes::READY );
    m_startRequested = true;
    appLogic();
    REQUIRE( state() == MagAOX::app::stateCodes::OPERATING );
    REQUIRE( m_startPending.load() );
    REQUIRE_FALSE( m_connected.load() );
}

void MonitorFixture::observations()
{
    std::lock_guard<std::mutex> lock( m_indiMutex );
    const auto                  now = std::chrono::steady_clock::now();
    m_nd.m_value                    = "nd1";
    m_flipwfsf.m_value              = "in";
    m_nd.m_state = m_flipwfsf.m_state = "READY";
    m_nd.m_received = m_flipwfsf.m_received = now;
    m_nd.m_stateReceived = m_flipwfsf.m_stateReceived = now;
    m_fps.m_value                                     = 1000;
    m_emgain.m_value                                  = 1;
    m_fps.m_received = m_emgain.m_received = now;
}
/// \endcond

/** \brief Start revalidates a retained attachment when READY was never sampled by the real monitor loop. */
TEST_CASE( "Collector quick restart revalidates the still-attached monitor", "[dlDataCollection][monitor]" )
{
    // clang-format off
#ifdef DLDATACOLLECTION_TEST_DOXYGEN_REF
    MagAOX::app::dlDataCollection::appLogic;
    MagAOX::app::dlDataCollection::allocate;
    MagAOX::app::dlDataCollection::processImage;
    MagAOX::app::dlDataCollection::checkObservation;
#endif
    // clang-format on
    MonitorFixture app;
    { //mutex scope
        std::unique_lock<std::mutex> lock( boundary::gate );
        REQUIRE( boundary::changed.wait_for( lock,
                                             std::chrono::seconds( 3 ),
                                             []
                                             {
                                                 return boundary::blocked;
                                             } ) );
    }
    REQUIRE( boundary::opens == 1 );
    app.restart();
    app.observations();
    bool expectStart = true;
    SECTION( "zero channel and fresh readbacks" )
    {
    }
    SECTION( "previous observation failure does not prevent attachment revalidation" )
    {
        app.previousObservationFailure();
    }
    SECTION( "a nonzero command channel still rejects the new session" )
    {
        boundary::commands[0] = 1.F;
        expectStart           = false;
    }
    app.release();
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds( 2 );
    while( !app.connected() && std::chrono::steady_clock::now() < until )
    {
        std::this_thread::sleep_for( std::chrono::milliseconds( 2 ) );
    }
    CHECK( app.connected() );
    REQUIRE( boundary::opens == 1 );
    REQUIRE( boundary::closes == 0 );
    app.expire();
    app.appLogic();
    if( expectStart )
    {
        CHECK( app.error().empty() );
        CHECK( app.phase() == "preparing" );
    }
    else
    {
        CHECK( app.error() == "dedicated command channel is not zero" );
        CHECK( app.phase() == "idle" );
        CHECK( boundary::commands[0] == 1.F );
    }
    // No I/O worker is started: reaching preparing verifies app admission, without issuing a command.
    REQUIRE( boundary::commandMd.cnt0 == 0 );
}
} //namespace libXWCTest::dlDataCollectionTest
