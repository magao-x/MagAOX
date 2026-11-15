/** \file loopTimingMonitor_integration_test.cpp
 * \brief Offline polling-worker integration with private synthetic timestamp streams.
 * \ingroup loopTimingMonitor_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "../loopTimingMonitor.hpp"

#include <chrono>
#include <filesystem>
#include <functional>
#include <stdexcept>

namespace libXWCTest
{
namespace loopTimingMonitorTest
{

/// \cond LOOP_TIMING_WORKER_FIXTURES
namespace
{

/// Keep synthetic ImageStreamIO sources outside all production shared-memory directories.
class WorkerDirectory
{
  public:
    std::string m_path; ///< Temporary directory retained until all fixture mappings close.

    std::string m_previous; ///< Original ImageStreamIO directory environment value.

    bool m_hadPrevious{ false }; ///< Whether the original environment value must be restored.

    /// Select a private directory before ImageStreamIO caches its first directory lookup.
    WorkerDirectory();

    /// Restore the environment and remove only this executable's temporary files.
    ~WorkerDirectory();
};

WorkerDirectory::WorkerDirectory()
{
    char       pattern[] = "/tmp/loop-timing-worker-XXXXXX";
    const auto path      = mkdtemp( pattern );

    if( !path )
    {
        throw std::runtime_error( "cannot create private loop timing worker directory" );
    }

    m_path              = path;
    const auto previous = getenv( "MILK_SHM_DIR" );
    m_hadPrevious       = previous != nullptr;

    if( previous )
    {
        m_previous = previous;
    }

    REQUIRE( setenv( "MILK_SHM_DIR", m_path.c_str(), 1 ) == 0 );
}

WorkerDirectory::~WorkerDirectory()
{
    if( m_hadPrevious )
    {
        setenv( "MILK_SHM_DIR", m_previous.c_str(), 1 );
    }
    else
    {
        unsetenv( "MILK_SHM_DIR" );
    }

    std::filesystem::remove_all( m_path );
}

/// Return the directory whose lifetime exceeds every stream fixture.
WorkerDirectory &workerDirectory();

WorkerDirectory &workerDirectory()
{
    static WorkerDirectory value;
    return value;
}

/// Produce a unique name within the private workstation directory.
std::string workerName();

std::string workerName()
{
    static_cast<void>( workerDirectory() );
    static unsigned sequence = 0;
    return "loopTimingWorker" + std::to_string( getpid() ) + "_" + std::to_string( ++sequence );
}

/// Wait for a concurrent condition with a finite deadline and no exact wakeup assumption.
bool waitUntil( const std::function<bool()> &condition /**< [in] Condition checked on this test thread. */,
                std::chrono::milliseconds    limit = std::chrono::seconds( 3 ) /**< [in] Maximum wait duration. */ );

bool waitUntil( const std::function<bool()> &condition, std::chrono::milliseconds limit )
{
    const auto end = std::chrono::steady_clock::now() + limit;

    do
    {
        if( condition() )
        {
            return true;
        }

        std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
    } while( std::chrono::steady_clock::now() < end );

    return condition();
}

/// Own a synthetic camera and an actual metadata-only SDK timestamp publisher.
class WorkerSources
{
  public:
    std::string m_cameraName; ///< Unique camera stream name.

    std::string m_sdkName; ///< Unique SDK diagnostic stream name.

    IMAGE m_camera{}; ///< Test-owned camera metadata written without semaphore notifications.

    MagAOX::app::dev::dmSdkTimingStream m_sdk; ///< Real diagnostic publisher; no SDK or hardware is invoked.

    uint64_t m_counter{ 0 }; ///< Camera publication sequence.

    /// Create both private sources before starting the independent monitor.
    WorkerSources();

    /// Close source mappings after the application fixture has joined its worker.
    ~WorkerSources();

    /// Publish a clock-paired camera/SDK event with a known 100 us software-endpoint difference.
    bool publish();

    /// Restart only the camera publication count while both endpoint timestamps keep advancing.
    bool regressCamera();
};

WorkerSources::WorkerSources() : m_cameraName( workerName() ), m_sdkName( workerName() )
{
    uint32_t size[]{ 1, 1 };
    REQUIRE( ImageStreamIO_createIm_gpu(
                 &m_camera, m_cameraName.c_str(), 2, size, _DATATYPE_UINT8, -1, 1, 0, 0, 0, 0 ) == 0 );
    REQUIRE( m_sdk.open( m_sdkName ) == 0 );
    REQUIRE( publish() );
}

WorkerSources::~WorkerSources()
{
    m_sdk.close();
    ImageStreamIO_closeIm( &m_camera );
}

bool WorkerSources::publish()
{
    timespec before{}, external{}, after{};
    int64_t  beforeNs = 0;
    int64_t  afterNs  = 0;
    bool     paired   = false;

    for( unsigned attempt = 0; attempt < 100; ++attempt )
    {
        if( clock_gettime( CLOCK_MONOTONIC, &before ) != 0 || clock_gettime( CLOCK_REALTIME, &external ) != 0 ||
            clock_gettime( CLOCK_MONOTONIC, &after ) != 0 )
        {
            return false;
        }

        beforeNs = int64_t( before.tv_sec ) * 1000000000 + before.tv_nsec;
        afterNs  = int64_t( after.tv_sec ) * 1000000000 + after.tv_nsec;

        if( afterNs >= beforeNs && afterNs - beforeNs <= 20000 )
        {
            paired = true;
            break;
        }
    }

    if( !paired )
    {
        return false;
    }

    const auto externalNs  = int64_t( external.tv_sec ) * 1000000000 + external.tv_nsec;
    const auto acquisition = externalNs - 1000000;
    const auto sdkStart    = beforeNs + ( afterNs - beforeNs ) / 2 - 900000;
    __atomic_store_n( &m_camera.md->write, uint8_t( 1 ), __ATOMIC_SEQ_CST );
    __atomic_store_n( &m_camera.md->atime.tv_sec, time_t( acquisition / 1000000000 ), __ATOMIC_RELEASE );
    __atomic_store_n( &m_camera.md->atime.tv_nsec, long( acquisition % 1000000000 ), __ATOMIC_RELEASE );
    __atomic_store_n( &m_camera.md->writetime.tv_sec, external.tv_sec, __ATOMIC_RELEASE );
    __atomic_store_n( &m_camera.md->writetime.tv_nsec, external.tv_nsec, __ATOMIC_RELEASE );
    __atomic_store_n( &m_camera.md->cnt0, ++m_counter, __ATOMIC_RELEASE );
    __atomic_store_n( &m_camera.md->write, uint8_t( 0 ), __ATOMIC_RELEASE );
    return m_sdk.publish( sdkStart );
}

bool WorkerSources::regressCamera()
{
    m_counter = 0;
    return publish();
}

/// Run the real polling owner without starting INDI, telemetry services or any control application.
class WorkerHarness : public MagAOX::app::loopTimingMonitor
{
  public:
    using loopTimingMonitor::consume;
    using loopTimingMonitor::m_captureEpoch;
    using loopTimingMonitor::m_counterRegressions;
    using loopTimingMonitor::m_monitor;
    using loopTimingMonitor::m_pollSleepUs;
    using loopTimingMonitor::m_queue;
    using loopTimingMonitor::m_streamsReady;
    using loopTimingMonitor::m_workerFailed;
    using loopTimingMonitor::monotonicNow;
    using loopTimingMonitor::stopWorker;
    using loopTimingMonitor::timingValues;

    /// Configure bounded state using private source names and generous test clock brackets.
    WorkerHarness( const std::string &camera /**< [in] Private camera source name. */,
                   const std::string &sdk /**< [in] Private SDK source name. */ );

    /// Deliver one local synthetic state packet to the application's real callback.
    int source( const std::string &device /**< [in] Configured source device. */,
                bool               operating /**< [in] Affirmative state, or an inactive state. */ );

    /// Send affirmative states for the configured camera, DM and loop devices.
    void enable();

    /// Deliver a loop gate transition through the actual callback.
    void loop( bool on /**< [in] Whether the synthetic loop is enabled. */ );

    /// Start the real worker with scheduler initialization already satisfied by the test.
    void start();

    /// Publish slowly until the real worker and consumer provide a qualified population.
    bool collect( WorkerSources         &sources /**< [in,out] Synthetic local stream producers. */,
                  std::array<double, 6> &values /**< [out] Qualified fgtimes values. */ );
};

WorkerHarness::WorkerHarness( const std::string &camera, const std::string &sdk )
{
    m_configName           = "loopTimingMonitor-worker-test";
    m_cameraStream         = camera;
    m_sdkStream            = sdk;
    m_capacity             = 512;
    m_clockBracketUs       = 100;
    m_cameraQuantizationUs = 0;
    m_queue.configure( m_capacity );
    REQUIRE( m_monitor.configure( m_maxFPS, m_windowSec, m_capacity ) );
}

int WorkerHarness::source( const std::string &device, bool operating )
{
    const bool        isLoop = device == m_loopDevice;
    pcf::IndiProperty property( isLoop ? pcf::IndiProperty::Switch : pcf::IndiProperty::Text );
    property.setDevice( device );
    property.setName( isLoop ? "loop_state" : "fsm" );
    property.add( pcf::IndiElement( isLoop ? "toggle" : "state" ) );

    if( isLoop )
    {
        property["toggle"].setSwitchState( operating ? pcf::IndiElement::On : pcf::IndiElement::Off );
    }
    else
    {
        property["state"] = operating ? "OPERATING" : "READY";
    }

    return setCallBack_source( property );
}

void WorkerHarness::enable()
{
    REQUIRE( source( m_cameraDevice, true ) == 0 );
    REQUIRE( source( m_dmDevice, true ) == 0 );
    REQUIRE( source( m_loopDevice, true ) == 0 );
}

void WorkerHarness::loop( bool on )
{
    REQUIRE( source( m_loopDevice, on ) == 0 );
}

void WorkerHarness::start()
{
    m_stop        = false;
    m_workerReady = true;
    m_thread      = std::thread(
        [this]()
        {
            threadExec();
        } );
}

bool WorkerHarness::collect( WorkerSources &sources, std::array<double, 6> &values )
{
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds( 3 );

    do
    {
        if( !sources.publish() )
        {
            return false;
        }

        std::this_thread::sleep_for( std::chrono::milliseconds( 5 ) );
        consume( monotonicNow() );

        if( timingValues( values ) && m_monitor.summary().latency.count >= 4 )
        {
            return true;
        }
    } while( std::chrono::steady_clock::now() < end && !m_workerFailed.load() );

    return false;
}

} // namespace
/// \endcond

/** \brief Exercise real polling, clock conversion, matching, source resets and gate transitions together.
 * \ingroup loopTimingMonitor_unit_test
 */
TEST_CASE( "loop timing worker measures private paired streams and resets with the gate",
           "[loopTimingMonitor][integration]" )
{
    // clang-format off
    #ifdef LOOPTIMINGMONITOR_TEST_DOXYGEN_REF
    MagAOX::app::loopTimingMonitor::threadExec();
    MagAOX::app::loopTimingMonitor::setCallBack_source( {} );
    MagAOX::app::loopTimingMonitor::consume( 1 );
    MagAOX::app::loopTimingMonitor::timingValues( {} );
    MagAOX::app::loopTimingMonitor::stopWorker();
    #endif
    // clang-format on
    WorkerSources sources;
    WorkerHarness app( sources.m_cameraName, sources.m_sdkName );
    app.enable();
    app.start();
    REQUIRE( waitUntil(
        [&]()
        {
            return app.m_streamsReady.load() || app.m_workerFailed.load();
        } ) );
    REQUIRE_FALSE( app.m_workerFailed.load() );
    std::array<double, 6> values{};
    REQUIRE( app.collect( sources, values ) );
    CHECK( values[0] > 0.001 );
    CHECK( values[2] > 0.001 );
    CHECK( values[4] == Approx( 100e-6 ).margin( 20e-6 ) );
    CHECK( values[5] < 20e-6 );

    const auto beforeRegression = app.m_counterRegressions.load();
    const auto beforeReset      = app.m_captureEpoch.load();
    REQUIRE( sources.publish() );
    REQUIRE( waitUntil(
        [&]()
        {
            return app.m_queue.queued() >= 2;
        } ) );
    REQUIRE( sources.regressCamera() );
    REQUIRE( waitUntil(
        [&]()
        {
            return app.m_counterRegressions.load() > beforeRegression && app.m_captureEpoch.load() > beforeReset;
        } ) );
    CHECK( app.m_counterRegressions.load() == beforeRegression + 1 );
    CHECK_FALSE( app.timingValues( values ) );
    app.consume( app.monotonicNow() );
    CHECK( app.m_monitor.summary().latency.count == 0 );
    CHECK_FALSE( app.timingValues( values ) );
    REQUIRE( app.collect( sources, values ) );
    CHECK( values[4] == Approx( 100e-6 ).margin( 20e-6 ) );
    CHECK( values[5] < 20e-6 );

    const auto epoch = app.m_captureEpoch.load();
    app.loop( false );
    CHECK_FALSE( app.timingValues( values ) );
    app.consume( app.monotonicNow() );
    REQUIRE( app.m_monitor.summary().latency.count == 0 );
    REQUIRE( waitUntil(
        [&]()
        {
            return !app.m_streamsReady.load();
        } ) );
    REQUIRE( sources.publish() );
    app.consume( app.monotonicNow() );
    CHECK( app.m_monitor.summary().latency.count == 0 );

    app.loop( true );
    app.consume( app.monotonicNow() );
    CHECK_FALSE( app.timingValues( values ) );
    REQUIRE( waitUntil(
        [&]()
        {
            return app.m_streamsReady.load() || app.m_workerFailed.load();
        } ) );
    REQUIRE_FALSE( app.m_workerFailed.load() );
    CHECK( app.m_captureEpoch.load() > epoch );
    REQUIRE( app.collect( sources, values ) );
    CHECK( values[4] == Approx( 100e-6 ).margin( 20e-6 ) );
    CHECK( values[5] < 20e-6 );
    app.stopWorker();
    CHECK_FALSE( app.m_workerFailed.load() );
    CHECK_FALSE( app.timingValues( values ) );
}

/** \brief Missing sources never prevent the actual polling worker from stopping promptly.
 * \ingroup loopTimingMonitor_unit_test
 */
TEST_CASE( "loop timing worker stops while its sources are absent", "[loopTimingMonitor][integration]" )
{
    // clang-format off
    #ifdef LOOPTIMINGMONITOR_TEST_DOXYGEN_REF
    MagAOX::app::loopTimingMonitor::threadExec();
    MagAOX::app::loopTimingMonitor::sleepFor( 1 );
    MagAOX::app::loopTimingMonitor::stopWorker();
    #endif
    // clang-format on
    WorkerHarness app( workerName(), workerName() );
    app.m_pollSleepUs = 1000000;
    app.enable();
    app.start();
    REQUIRE( waitUntil(
        [&]()
        {
            return app.m_captureEpoch.load() > 1;
        } ) );
    REQUIRE_FALSE( app.m_streamsReady.load() );
    const auto before = std::chrono::steady_clock::now();
    app.stopWorker();
    CHECK( std::chrono::steady_clock::now() - before < std::chrono::seconds( 1 ) );
    CHECK_FALSE( app.m_workerFailed.load() );
}

} // namespace loopTimingMonitorTest
} // namespace libXWCTest
