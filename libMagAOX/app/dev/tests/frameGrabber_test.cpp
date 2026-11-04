/** \file frameGrabber_test.cpp
 * \brief Offline worker regressions for concurrent framegrabber reconfiguration requests.
 * \ingroup frameGrabber_unit_test
 */

#include "../../../../tests/testXWC.hpp"
#include "../../../libMagAOX.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
namespace frameGrabberHarness
{
using MagAOX::app::stateCodes;
using namespace std::chrono_literals;

/// Own a private directory for all ImageStreamIO mappings in this test process.
class Streams
{
  public:
    /// Directory containing only this suite's temporary streams.
    std::string m_directory;

    /// Previous process stream-directory setting.
    std::string m_previous;

    /// Whether the stream-directory environment variable was previously set.
    bool m_hadPrevious{ false };

    /// Select a private directory before the first ImageStreamIO operation.
    Streams();

    /// Restore the environment after all workers and mappings have been destroyed.
    ~Streams();
};

Streams::Streams()
{
    char directory[] = "/tmp/framegrabber-test-XXXXXX";
    REQUIRE( mkdtemp( directory ) != nullptr );
    m_directory = directory;
    if( const char *previous = getenv( "MILK_SHM_DIR" ) )
    {
        m_hadPrevious = true;
        m_previous    = previous;
    }
    REQUIRE( setenv( "MILK_SHM_DIR", m_directory.c_str(), 1 ) == 0 );
}

Streams::~Streams()
{
    if( m_hadPrevious )
        setenv( "MILK_SHM_DIR", m_previous.c_str(), 1 );
    else
        unsetenv( "MILK_SHM_DIR" );
    std::error_code error;
    std::filesystem::remove_all( m_directory, error );
}

/// Minimal fake camera running the real framegrabber loop with local shared memory.
class Camera : public MagAOX::app::dev::frameGrabber<Camera>
{
  public:
    /// Select a deterministic pause in the real worker's derived-camera interface.
    enum class Pause
    {
        none,
        configure,
        fps,
        reconfig,
        ready
    };

    /// Disable pixel flipping for this fixture.
    static constexpr bool c_frameGrabber_flippable = false;

    /// Serialize requested settings, observations, and test barriers.
    std::mutex m_mutex;

    /// Notify the test and worker when the next transition is available.
    std::condition_variable m_condition;

    /// Thread-safe simulated application shutdown.
    std::atomic<bool> m_stop{ false };

    /// Thread-safe simulated application state.
    std::atomic<stateCodes::stateCodeT> m_state{ stateCodes::OPERATING };

    /// Thread-safe simulated camera power.
    std::atomic<int> m_power{ 1 };

    /// Owned thread invoking the production worker directly.
    std::thread m_worker;

    /// Latest requested camera setting, protected by m_mutex.
    int m_requested{ 1 };

    /// Whether only reconfig applies the requested setting, matching mode-based cameras.
    bool m_applyInReconfig{ false };

    /// Setting selected by the last reconfig callback.
    int m_selected{ 1 };

    /// Settings captured at entry to each configureAcquisition call.
    std::vector<int> m_snapshots;

    /// Settings actually applied by successful configureAcquisition calls.
    std::vector<int> m_applied;

    /// Number of acquisition-start attempts.
    unsigned m_starts{ 0 };

    /// Number of acquisition callbacks entered.
    unsigned m_acquisitions{ 0 };

    /// Number of reconfiguration cleanup callbacks.
    unsigned m_restarts{ 0 };

    /// Number of readiness checks, including checks while waiting for readiness.
    unsigned m_stateChecks{ 0 };

    /// Number of power checks, including checks while waiting for power.
    unsigned m_powerChecks{ 0 };

    /// Number of upcoming configuration attempts to fail.
    unsigned m_configFailures{ 0 };

    /// Number of upcoming acquisition-start attempts to fail.
    unsigned m_startFailures{ 0 };

    /// Number of upcoming reconfiguration cleanup attempts to fail before applying settings.
    unsigned m_reconfigFailures{ 0 };

    /// One-shot pause selected before launching the worker.
    Pause m_pauseAt{ Pause::none };

    /// Whether the worker has reached the selected pause.
    bool m_paused{ false };

    /// Whether the test has released the selected pause.
    bool m_release{ false };

    /// Whether a bounded worker wait expired.
    bool m_timedOut{ false };

    /// Exception captured from the worker for assertion on the test thread.
    std::exception_ptr m_exception;

    /// Initialize a unique local output stream and disable latency accumulation.
    Camera();

    /// Release barriers, stop the worker, and join before fixture members are destroyed.
    ~Camera();

    /// Launch the actual framegrabber worker without any device or INDI connection.
    void start();

    /// Stop and join the worker, including when a test assertion failed.
    void stop();

    /// Queue a camera setting while holding the same mutex used for configuration snapshots.
    void request( int target /**< [in] New camera setting. */ );

    /// Release the one-shot worker pause.
    void release();

    /// Wait a bounded time for a test observation protected by m_mutex.
    template <class Predicate>
    bool wait( Predicate predicate /**< [in] Condition evaluated while holding m_mutex. */ );

    /// Capture and apply the requested setting, optionally pausing after the snapshot.
    int configureAcquisition();

    /// Pause after configuration inside the real configCircBuffs call.
    float fps();

    /// Record acquisition startup, optionally injecting a transient failure.
    int startAcquisition();

    /// Block until a request, power transition, or shutdown wakes acquisition.
    int acquireAndCheckValid();

    /// Provide the unused pixel-copy interface; acquisition never returns a frame.
    int loadImageIntoStream( void *destination /**< [out] Frame destination. */ );

    /// Apply deferred settings during cleanup, optionally injecting a transient failure.
    int reconfig();

    /// Report the thread-safe shutdown flag.
    bool shutdown();

    /// Report and count readiness observations.
    stateCodes::stateCodeT state();

    /// Report and count power observations.
    int powerState();

    /// Discard worker diagnostics without starting application logging.
    template <class Log>
    static int log( const typename Log::messageT &message /**< [in] Diagnostic payload. */,
                    flatlogs::logPrioT priority = flatlogs::logPrio::LOG_DEFAULT /**< [in] Diagnostic priority. */ );

  private:
    /// Hold a one-shot pause with a timeout and unconditional shutdown escape.
    void pause( Pause                         point /**< [in] Current callback boundary. */,
                std::unique_lock<std::mutex> &lock /**< [in/out] Lock released while waiting. */ );
};

Camera::Camera()
{
    static Streams  streams;
    static unsigned sequence   = 0;
    m_shmimName                = "framegrabber_" + std::to_string( getpid() ) + "_" + std::to_string( ++sequence );
    m_fgThreadInit             = false;
    m_width                    = 2;
    m_height                   = 2;
    m_dataType                 = _DATATYPE_FLOAT;
    m_latencyCircBuffMaxLength = 0;
}

Camera::~Camera()
{
    stop();
}

void Camera::start()
{
    m_worker = std::thread(
        [this]
        {
            try
            {
                fgThreadExec();
            }
            catch( ... )
            {
                m_exception = std::current_exception();
            }
        } );
}

void Camera::stop()
{
    { //mutex scope
        std::lock_guard<std::mutex> lock( m_mutex );
        m_stop = true;
    }
    m_condition.notify_all();
    if( m_worker.joinable() )
        m_worker.join();
}

void Camera::request( int target )
{
    { //mutex scope
        std::lock_guard<std::mutex> lock( m_mutex );
        m_requested = target;
        m_reconfig  = true;
    }
    m_condition.notify_all();
}

void Camera::release()
{
    { //mutex scope
        std::lock_guard<std::mutex> lock( m_mutex );
        m_release = true;
    }
    m_condition.notify_all();
}

template <class Predicate>
bool Camera::wait( Predicate predicate )
{
    std::unique_lock<std::mutex> lock( m_mutex );
    return m_condition.wait_for( lock, 4s, predicate );
}

void Camera::pause( Pause point, std::unique_lock<std::mutex> &lock )
{
    if( m_pauseAt != point )
        return;
    m_pauseAt = Pause::none;
    m_paused  = true;
    m_condition.notify_all();
    if( !m_condition.wait_for( lock, 5s, [this] { return m_release || m_stop; } ) )
    {
        m_timedOut = true;
        m_stop     = true;
    }
}

int Camera::configureAcquisition()
{
    std::unique_lock<std::mutex> lock( m_mutex );
    int                          target = m_applyInReconfig ? m_selected : m_requested;
    m_snapshots.push_back( target );
    m_condition.notify_all();
    pause( Pause::configure, lock );
    if( m_configFailures > 0 )
    {
        --m_configFailures;
        return -1;
    }
    m_applied.push_back( target );
    m_condition.notify_all();
    return 0;
}

float Camera::fps()
{
    std::unique_lock<std::mutex> lock( m_mutex );
    pause( Pause::fps, lock );
    return 0;
}

int Camera::startAcquisition()
{
    std::lock_guard<std::mutex> lock( m_mutex );
    ++m_starts;
    m_condition.notify_all();
    if( m_startFailures > 0 )
    {
        --m_startFailures;
        return -1;
    }
    return 0;
}

int Camera::acquireAndCheckValid()
{
    std::unique_lock<std::mutex> lock( m_mutex );
    ++m_acquisitions;
    m_condition.notify_all();
    if( !m_condition.wait_for( lock, 5s, [this] { return m_stop || m_reconfig || m_power <= 0; } ) )
    {
        m_timedOut = true;
        m_stop     = true;
    }
    return 1;
}

int Camera::loadImageIntoStream( void *destination )
{
    static_cast<void>( destination );
    return 0;
}

int Camera::reconfig()
{
    std::unique_lock<std::mutex> lock( m_mutex );
    ++m_restarts;
    if( m_reconfigFailures > 0 )
    {
        --m_reconfigFailures;
        m_condition.notify_all();
        return -1;
    }
    m_selected = m_requested;
    m_condition.notify_all();
    pause( Pause::reconfig, lock );
    return 0;
}

bool Camera::shutdown()
{
    return m_stop;
}

stateCodes::stateCodeT Camera::state()
{
    std::unique_lock<std::mutex> lock( m_mutex );
    ++m_stateChecks;
    m_condition.notify_all();
    if( m_restarts > 0 )
        pause( Pause::ready, lock );
    return m_state;
}

int Camera::powerState()
{
    std::lock_guard<std::mutex> lock( m_mutex );
    ++m_powerChecks;
    m_condition.notify_all();
    return m_power;
}

template <class Log>
int Camera::log( const typename Log::messageT &message, flatlogs::logPrioT priority )
{
    static_cast<void>( message );
    static_cast<void>( priority );
    return 0;
}
} // namespace frameGrabberHarness
/// \endcond

namespace libXWCTest
{
/** \addtogroup frameGrabber_unit_test */
namespace frameGrabberTest
{
using frameGrabberHarness::Camera;
using MagAOX::app::stateCodes;

/// Requests arriving after the worker's configuration snapshot must configure the latest setting on the next pass.
/** \ingroup frameGrabber_unit_test */
TEST_CASE( "frameGrabber preserves requests arriving during configuration", "[frameGrabber]" )
{
    // clang-format off
    #ifdef FRAMEGRABBER_TEST_DOXYGEN_REF
    MagAOX::app::dev::frameGrabber::fgThreadExec;
    MagAOX::app::dev::frameGrabber::configCircBuffs;
    #endif
    // clang-format on
    Camera camera;
    SECTION( "during the derived configuration callback" )
    {
        camera.m_pauseAt = Camera::Pause::configure;
    }
    SECTION( "after configuration while preparing the latency buffers" )
    {
        camera.m_pauseAt = Camera::Pause::fps;
    }
    camera.start();
    REQUIRE( camera.wait( [&] { return camera.m_paused; } ) );
    camera.request( 2 );
    camera.request( 3 );
    camera.release();
    REQUIRE( camera.wait( [&] { return camera.m_applied.size() >= 2 && camera.m_acquisitions >= 1; } ) );
    camera.stop();
    REQUIRE_FALSE( camera.m_timedOut );
    REQUIRE_FALSE( camera.m_exception );
    REQUIRE( camera.m_snapshots == std::vector<int>{ 1, 3 } );
    REQUIRE( camera.m_applied == std::vector<int>{ 1, 3 } );
    REQUIRE( camera.m_starts == 2 );
    REQUIRE( camera.m_restarts == 1 );
}

/// Mode selection performed by reconfig must retain a request queued during or just after that callback.
/** \ingroup frameGrabber_unit_test */
TEST_CASE( "frameGrabber preserves requests around deferred mode selection", "[frameGrabber]" )
{
    // clang-format off
    #ifdef FRAMEGRABBER_TEST_DOXYGEN_REF
    MagAOX::app::dev::frameGrabber::fgThreadExec;
    #endif
    // clang-format on
    Camera camera;
    camera.m_applyInReconfig = true;
    SECTION( "during cleanup after the mode snapshot" )
    {
        camera.m_pauseAt = Camera::Pause::reconfig;
    }
    SECTION( "after cleanup before the next configuration" )
    {
        camera.m_pauseAt = Camera::Pause::ready;
    }
    camera.request( 2 );
    camera.start();
    REQUIRE( camera.wait( [&] { return camera.m_paused; } ) );
    camera.request( 3 );
    camera.release();
    REQUIRE( camera.wait( [&] { return camera.m_applied.size() >= 3 && camera.m_acquisitions >= 1; } ) );
    camera.stop();
    REQUIRE_FALSE( camera.m_timedOut );
    REQUIRE_FALSE( camera.m_exception );
    REQUIRE( camera.m_snapshots == std::vector<int>{ 1, 2, 3 } );
    REQUIRE( camera.m_applied == std::vector<int>{ 1, 2, 3 } );
    REQUIRE( camera.m_starts == 3 );
    REQUIRE( camera.m_restarts == 2 );
}

/// A failed configuration or start retries the latest requested setting instead of losing the pending update.
/** \ingroup frameGrabber_unit_test */
TEST_CASE( "frameGrabber retries requests after configuration and acquisition-start failures", "[frameGrabber]" )
{
    // clang-format off
    #ifdef FRAMEGRABBER_TEST_DOXYGEN_REF
    MagAOX::app::dev::frameGrabber::fgThreadExec;
    #endif
    // clang-format on
    Camera camera;
    camera.m_pauseAt          = Camera::Pause::configure;
    bool configurationFailure = false;
    SECTION( "configuration fails once" )
    {
        camera.m_configFailures = 1;
        configurationFailure    = true;
    }
    SECTION( "acquisition startup fails once" )
    {
        camera.m_startFailures = 1;
    }
    camera.start();
    REQUIRE( camera.wait( [&] { return camera.m_paused; } ) );
    camera.request( 2 );
    camera.release();
    REQUIRE( camera.wait( [&] { return camera.m_snapshots.size() >= 2 && camera.m_acquisitions >= 1; } ) );
    camera.stop();
    REQUIRE_FALSE( camera.m_timedOut );
    REQUIRE_FALSE( camera.m_exception );
    REQUIRE( camera.m_snapshots == std::vector<int>{ 1, 2, 2 } );
    REQUIRE( camera.m_restarts == 1 );
    if( configurationFailure )
    {
        REQUIRE( camera.m_applied == std::vector<int>{ 2, 2 } );
        REQUIRE( camera.m_starts == 2 );
    }
    else
    {
        REQUIRE( camera.m_applied == std::vector<int>{ 1, 2, 2 } );
        REQUIRE( camera.m_starts == 3 );
    }
}

/// Failed cleanup must leave a retry request until deferred camera settings are successfully selected.
/** \ingroup frameGrabber_unit_test */
TEST_CASE( "frameGrabber retries a failed reconfiguration cleanup", "[frameGrabber]" )
{
    // clang-format off
    #ifdef FRAMEGRABBER_TEST_DOXYGEN_REF
    MagAOX::app::dev::frameGrabber::fgThreadExec;
    #endif
    // clang-format on
    Camera camera;
    camera.m_applyInReconfig  = true;
    camera.m_reconfigFailures = 1;
    camera.request( 2 );
    camera.request( 3 );
    camera.start();
    REQUIRE( camera.wait( [&] { return camera.m_applied.size() >= 3 && camera.m_acquisitions >= 1; } ) );
    camera.stop();
    REQUIRE_FALSE( camera.m_timedOut );
    REQUIRE_FALSE( camera.m_exception );
    REQUIRE( camera.m_snapshots == std::vector<int>{ 1, 1, 3 } );
    REQUIRE( camera.m_applied == std::vector<int>{ 1, 1, 3 } );
    REQUIRE( camera.m_starts == 3 );
    REQUIRE( camera.m_restarts == 2 );
}

/// Requests queued before startup coalesce, and a later request wakes acquisition and is applied.
/** \ingroup frameGrabber_unit_test */
TEST_CASE( "frameGrabber consumes queued requests and handles requests during acquisition", "[frameGrabber]" )
{
    // clang-format off
    #ifdef FRAMEGRABBER_TEST_DOXYGEN_REF
    MagAOX::app::dev::frameGrabber::fgThreadExec;
    #endif
    // clang-format on
    Camera camera;
    camera.request( 2 );
    camera.request( 3 );
    camera.start();
    REQUIRE( camera.wait( [&] { return camera.m_acquisitions >= 1; } ) );
    camera.request( 4 );
    REQUIRE( camera.wait( [&] { return camera.m_acquisitions >= 2; } ) );
    camera.stop();
    REQUIRE_FALSE( camera.m_timedOut );
    REQUIRE_FALSE( camera.m_exception );
    REQUIRE( camera.m_applied == std::vector<int>{ 3, 3, 4 } );
    REQUIRE( camera.m_snapshots == std::vector<int>{ 3, 3, 4 } );
    REQUIRE( camera.m_restarts == 2 );
}

/// Pending requests survive readiness and power waits and configure only after both prerequisites hold.
/** \ingroup frameGrabber_unit_test */
TEST_CASE( "frameGrabber retains requests while waiting for readiness and power", "[frameGrabber]" )
{
    // clang-format off
    #ifdef FRAMEGRABBER_TEST_DOXYGEN_REF
    MagAOX::app::dev::frameGrabber::fgThreadExec;
    #endif
    // clang-format on
    Camera camera;
    camera.m_state = stateCodes::NOTCONNECTED;
    camera.m_power = 0;
    camera.request( 2 );
    camera.start();
    REQUIRE( camera.wait( [&] { return camera.m_stateChecks > 0; } ) );
    REQUIRE( camera.wait( [&] { return camera.m_snapshots.empty(); } ) );
    camera.m_state = stateCodes::READY;
    REQUIRE( camera.wait( [&] { return camera.m_powerChecks > 0; } ) );
    REQUIRE( camera.wait( [&] { return camera.m_snapshots.empty(); } ) );
    camera.request( 3 );
    camera.m_power = 1;
    REQUIRE( camera.wait( [&] { return camera.m_acquisitions >= 1; } ) );
    camera.stop();
    REQUIRE_FALSE( camera.m_timedOut );
    REQUIRE_FALSE( camera.m_exception );
    REQUIRE( camera.m_applied == std::vector<int>{ 3, 3 } );
    REQUIRE( camera.m_starts == 2 );
    REQUIRE( camera.m_restarts == 1 );
}
} // namespace frameGrabberTest
} // namespace libXWCTest
