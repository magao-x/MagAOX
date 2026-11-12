/** \file loopTimingMonitor_synco_test.cpp
 * \brief Offline automatic-mode polling with qualified main labels and actual event histories.
 * \ingroup loopTimingMonitor_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "../loopTimingMonitor.hpp"

#include <chrono>
#include <filesystem>
#include <functional>
#include <stdexcept>
#include <vector>

namespace libXWCTest
{
namespace loopTimingMonitorTest
{

/// \cond LOOP_TIMING_SYNCO_FIXTURES
namespace
{

/// Keep every synthetic stream in a private directory for this executable's lifetime.
class SyncoDirectory
{
    std::string m_path; ///< Test-owned directory removed after every source mapping closes.

    std::string m_previous; ///< Saved ImageStreamIO directory environment value.

    bool m_hadPrevious{ false }; ///< Whether to restore rather than unset the original value.

  public:
    /// Select a private directory before the first ImageStreamIO lookup.
    SyncoDirectory();

    /// Restore the environment and remove only this fixture's files.
    ~SyncoDirectory();
};

SyncoDirectory::SyncoDirectory()
{
    char       pattern[] = "/tmp/loop-timing-synco-XXXXXX";
    const auto path      = mkdtemp( pattern );

    if( !path )
    {
        throw std::runtime_error( "cannot create private loop timing synco directory" );
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

SyncoDirectory::~SyncoDirectory()
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

/// Generate a unique name after establishing the private ImageStreamIO directory.
std::string syncoName();

std::string syncoName()
{
    static SyncoDirectory directory;
    static unsigned       sequence = 0;
    return "loopTimingSynco" + std::to_string( getpid() ) + "_" + std::to_string( ++sequence );
}

/// Wait for a concurrent worker condition without assuming an exact scheduling delay.
bool waitSynco( const std::function<bool()> &condition /**< [in] Condition checked on the test thread. */ );

bool waitSynco( const std::function<bool()> &condition )
{
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds( 3 );

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

/// Own a camera and two real diagnostic histories without invoking any control application.
class SyncoSources
{
  public:
    std::string m_cameraName; ///< Unique camera metadata name.

    std::string m_sdkName; ///< Unique native endpoint history name.

    std::string m_mainName; ///< Unique qualified-main history name.

    IMAGE m_camera{}; ///< Test-owned metadata source with no semaphore notifications.

    MagAOX::app::dev::dmTimingEventStream m_sdk; ///< Actual bounded SDK event publisher.

    MagAOX::app::dev::dmTimingEventStream m_main; ///< Actual bounded main/mode event publisher.

    uint64_t m_generation{ 1 }; ///< Synthetic producer mode generation.

    bool m_enabled{ false }; ///< Mode frozen into each positively qualified main label.

    bool m_holdMain{ false }; ///< Defer qualified-main publication while camera and SDK capture continue.

    /// Create private sources before the actual worker is started.
    explicit SyncoSources( size_t capacity = 128 /**< [in] Retained history records per publisher. */ );

    /// Close mappings after the application fixture has joined its worker.
    ~SyncoSources();

    /// Publish one primary camera/main pair plus optional auxiliary native entries.
    bool frame( unsigned auxiliary /**< [in] Number of extra native commands lacking main labels. */,
                unsigned missed = 0 /**< [in] Camera publications skipped before this observed frame. */ );

    /// Publish every held main in its original capture order without producing another camera or SDK event.
    bool flushMain();

    /// Publish a native entry without a main label, including after a disable transition.
    bool auxiliary();

    /// Publish a diagnostic mode transition independently of the passive INDI callback.
    bool mode( bool enabled /**< [in] New frozen producer enable state. */ );

    /// Replace the qualified-main history with a new inode and fresh capture sequence.
    bool restartMain();

  private:
    size_t m_capacity; ///< Fixed stream capacity reused on publisher restart.

    uint64_t m_primary{ 0 }; ///< Consecutive genuine main cycles, independent of auxiliary commands.

    uint64_t m_cameraCounter{ 0 }; ///< Camera counter, including deliberately unobserved publications.

    uint64_t m_combined{ 0 }; ///< Exact combined identity counter for every native command.

    uint64_t m_sdkSequence{ 0 }; ///< Native publisher capture sequence.

    uint64_t m_mainSequence{ 0 }; ///< Main/mode publisher capture sequence.

    int64_t m_lastStart{ 0 }; ///< Last native timestamp, including unclassified auxiliary commands.

    std::vector<MagAOX::dmTimingEvents::Record>
        m_pendingMain; ///< Immutable labels awaiting ordinary-thread publication.

    /// Publish one exact native endpoint and optionally its main classification.
    bool native( int64_t startNs /**< [in] Existing monotonic SDK endpoint. */,
                 int64_t publicationNs /**< [in] Raw combined publication identity timestamp. */,
                 bool    main /**< [in] Whether this command has a positive main classification. */ );
};

SyncoSources::SyncoSources( size_t capacity )
    : m_cameraName( syncoName() ), m_sdkName( syncoName() ), m_mainName( syncoName() ), m_capacity( capacity )
{
    uint32_t size[]{ 1, 1 };
    REQUIRE( ImageStreamIO_createIm_gpu(
                 &m_camera, m_cameraName.c_str(), 2, size, _DATATYPE_UINT8, -1, 1, 0, 0, 0, 0 ) == 0 );
    REQUIRE( m_sdk.open( m_sdkName, capacity ) == 0 );
    REQUIRE( m_main.open( m_mainName, capacity ) == 0 );
    REQUIRE( frame( 0 ) );
}

SyncoSources::~SyncoSources()
{
    m_main.close();
    m_sdk.close();
    ImageStreamIO_closeIm( &m_camera );
}

bool SyncoSources::native( int64_t startNs, int64_t publicationNs, bool main )
{
    MagAOX::dmTimingEvents::Record event;
    event.kind     = MagAOX::dmTimingEvents::Kind::sdk;
    event.key      = { 71, ++m_combined, publicationNs / 1000000000, publicationNs % 1000000000 };
    event.sequence = ++m_sdkSequence;
    event.startNs = m_lastStart = startNs;

    if( !m_sdk.publish( event ) )
    {
        return false;
    }

    if( main )
    {
        event.kind           = MagAOX::dmTimingEvents::Kind::main;
        event.sequence       = ++m_mainSequence;
        event.generation     = m_generation;
        event.primaryInode   = 93;
        event.primaryCounter = m_primary;
        event.steps          = 4;
        event.enabled        = m_enabled;
        event.startNs        = 0;

        if( m_holdMain )
        {
            m_pendingMain.push_back( event );
            return true;
        }

        return m_main.publish( event );
    }

    return true;
}

bool SyncoSources::frame( unsigned auxiliary, unsigned missed )
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

    const auto publication = int64_t( external.tv_sec ) * 1000000000 + external.tv_nsec;
    const auto acquisition = publication - 1000000;
    const auto start       = beforeNs + ( afterNs - beforeNs ) / 2 - 900000;
    ++m_primary;
    m_cameraCounter += 1 + missed;
    __atomic_store_n( &m_camera.md->write, uint8_t( 1 ), __ATOMIC_SEQ_CST );
    __atomic_store_n( &m_camera.md->atime.tv_sec, time_t( acquisition / 1000000000 ), __ATOMIC_RELEASE );
    __atomic_store_n( &m_camera.md->atime.tv_nsec, long( acquisition % 1000000000 ), __ATOMIC_RELEASE );
    __atomic_store_n( &m_camera.md->writetime.tv_sec, external.tv_sec, __ATOMIC_RELEASE );
    __atomic_store_n( &m_camera.md->writetime.tv_nsec, external.tv_nsec, __ATOMIC_RELEASE );
    __atomic_store_n( &m_camera.md->cnt0, m_cameraCounter, __ATOMIC_RELEASE );
    __atomic_store_n( &m_camera.md->write, uint8_t( 0 ), __ATOMIC_RELEASE );

    if( !native( start, publication, true ) )
    {
        return false;
    }

    for( unsigned step = 0; step < auxiliary; ++step )
    {
        if( !native( m_lastStart + 30000, publication, false ) )
        {
            return false;
        }
    }

    return true;
}

bool SyncoSources::flushMain()
{
    for( const auto &record : m_pendingMain )
    {
        if( !m_main.publish( record ) )
        {
            return false;
        }
    }

    m_pendingMain.clear();
    m_holdMain = false;
    return true;
}

bool SyncoSources::auxiliary()
{
    timespec publication{};

    if( clock_gettime( CLOCK_REALTIME, &publication ) != 0 )
    {
        return false;
    }

    return native( m_lastStart + 1, int64_t( publication.tv_sec ) * 1000000000 + publication.tv_nsec, false );
}

bool SyncoSources::mode( bool enabled )
{
    m_enabled = enabled;
    ++m_generation;
    MagAOX::dmTimingEvents::Record event;
    event.kind       = MagAOX::dmTimingEvents::Kind::mode;
    event.sequence   = ++m_mainSequence;
    event.generation = m_generation;
    event.steps      = 4;
    event.enabled    = enabled;
    return m_main.publish( event );
}

bool SyncoSources::restartMain()
{
    m_main.close();
    m_mainSequence = 0;
    ++m_generation;
    return m_main.open( m_mainName, m_capacity ) == 0;
}

/// Run actual polling and ordinary consumption without activating INDI or telemetry services.
class SyncoHarness : public MagAOX::app::loopTimingMonitor
{
  public:
    using loopTimingMonitor::consume;
    using loopTimingMonitor::m_cameraMissed;
    using loopTimingMonitor::m_captureEpoch;
    using loopTimingMonitor::m_gateEnabled;
    using loopTimingMonitor::m_mainMissed;
    using loopTimingMonitor::m_monitor;
    using loopTimingMonitor::m_queue;
    using loopTimingMonitor::m_sdkMissed;
    using loopTimingMonitor::m_streamsReady;
    using loopTimingMonitor::m_synco;
    using loopTimingMonitor::m_workerFailed;
    using loopTimingMonitor::monotonicNow;
    using loopTimingMonitor::stopWorker;
    using loopTimingMonitor::timingValues;

    /// Configure a real automatic-mode worker using only private sources.
    explicit SyncoHarness( const SyncoSources &sources /**< [in] Private stream names. */ );

    /// Deliver synthetic camera, DM and loop operating states through the real callback.
    void enable();

    /// Deliver one passive synco enable notification through the real callback.
    void synco( bool enabled /**< [in] Synthetic enable state. */ );

    /// Start the worker with ordinary scheduling already accepted by the fixture.
    void start();

    /// Collect a qualified main-only population using paced synthetic frames.
    bool collect( SyncoSources          &sources /**< [in,out] Private event producers. */,
                  unsigned               auxiliary /**< [in] Auxiliary native commands per camera/main frame. */,
                  std::array<double, 6> &values /**< [out] Qualified interval and latency statistics. */ );
};

SyncoHarness::SyncoHarness( const SyncoSources &sources )
{
    m_configName           = "loopTimingMonitor-synco-test";
    m_cameraStream         = sources.m_cameraName;
    m_sdkEventStream       = sources.m_sdkName;
    m_mainStream           = sources.m_mainName;
    m_sdkStream            = syncoName();
    m_syncoDevice          = "privateSynco";
    m_capacity             = 128;
    m_pollSleepUs          = 1000;
    m_clockBracketUs       = 100;
    m_cameraQuantizationUs = 0;
    m_queue.configure( m_capacity );
    REQUIRE( m_monitor.configure( m_maxFPS, m_windowSec, m_capacity ) );
    m_synco.configure( m_capacity, int64_t( m_windowSec * 1e9 ) );
}

void SyncoHarness::enable()
{
    for( const auto &device : { m_cameraDevice, m_dmDevice, m_loopDevice } )
    {
        const bool        loop = device == m_loopDevice;
        pcf::IndiProperty property( loop ? pcf::IndiProperty::Switch : pcf::IndiProperty::Text );
        property.setDevice( device );
        property.setName( loop ? "loop_state" : "fsm" );
        property.add( pcf::IndiElement( loop ? "toggle" : "state" ) );

        if( loop )
        {
            property["toggle"].setSwitchState( pcf::IndiElement::On );
        }
        else
        {
            property["state"] = "OPERATING";
        }

        REQUIRE( setCallBack_source( property ) == 0 );
    }
}

void SyncoHarness::synco( bool enabled )
{
    pcf::IndiProperty property( pcf::IndiProperty::Switch );
    property.setDevice( m_syncoDevice );
    property.setName( "enable" );
    property.add( pcf::IndiElement( "toggle" ) );
    property["toggle"].setSwitchState( enabled ? pcf::IndiElement::On : pcf::IndiElement::Off );
    REQUIRE( setCallBack_source( property ) == 0 );
}

void SyncoHarness::start()
{
    m_stop        = false;
    m_workerReady = true;
    m_thread      = std::thread(
        [this]()
        {
            threadExec();
        } );
}

bool SyncoHarness::collect( SyncoSources &sources, unsigned auxiliary, std::array<double, 6> &values )
{
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds( 3 );

    do
    {
        if( !sources.frame( auxiliary ) )
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

/** \brief Passive enable updates switch modes while only positively labelled mains contribute timing.
 * \ingroup loopTimingMonitor_unit_test
 */
TEST_CASE( "loop timing automatic worker follows Off On Off without measuring auxiliary commands",
           "[loopTimingMonitor][integration][synco]" )
{
    // clang-format off
    #ifdef LOOPTIMINGMONITOR_TEST_DOXYGEN_REF
    MagAOX::app::loopTimingMonitor::threadExec();
    MagAOX::app::loopTimingMonitor::setCallBack_source( {} );
    MagAOX::app::loopTimingMonitor::consume( 1 );
    MagAOX::app::loopTimingMonitor::timingValues( {} );
    #endif
    // clang-format on
    SyncoSources sources;
    SyncoHarness app( sources );
    app.enable();
    app.start();
    REQUIRE( waitSynco(
        [&]()
        {
            return app.m_captureEpoch.load() > 1;
        } ) );
    REQUIRE_FALSE( app.m_gateEnabled.load() );
    REQUIRE_FALSE( app.m_streamsReady.load() );
    std::array<double, 6> values{};
    REQUIRE_FALSE( app.timingValues( values ) );

    for( const bool enabled : { false, true, false } )
    {
        const auto epoch = app.m_captureEpoch.load();
        app.synco( enabled );
        REQUIRE_FALSE( app.timingValues( values ) );
        REQUIRE( waitSynco(
            [&]()
            {
                return app.m_captureEpoch.load() > epoch && app.m_streamsReady.load();
            } ) );
        app.consume( app.monotonicNow() );
        REQUIRE( app.m_monitor.summary().latency.count == 0 );
        REQUIRE( sources.mode( enabled ) );
        REQUIRE( app.collect( sources, enabled ? 3 : 0, values ) );
        REQUIRE( app.m_synco.known() );
        CHECK( app.m_synco.enabled() == enabled );
        CHECK( values[0] > 0.001 );
        CHECK( values[2] == Approx( values[0] ).epsilon( 0.2 ) );
        CHECK( values[4] == Approx( 100e-6 ).margin( 20e-6 ) );
        CHECK( values[5] < 20e-6 );
        CHECK( app.m_synco.quality().invalid == 0 );
    }

    const auto sdkCount     = app.m_monitor.summary().sdkIntervals.count;
    const auto latencyCount = app.m_monitor.summary().latency.count;
    REQUIRE( sources.auxiliary() );
    REQUIRE( waitSynco(
        [&]()
        {
            return app.m_queue.queued() > 0;
        } ) );
    app.consume( app.monotonicNow() );
    CHECK( app.m_monitor.summary().sdkIntervals.count == sdkCount );
    CHECK( app.m_monitor.summary().latency.count == latencyCount );
    REQUIRE( app.timingValues( values ) );
    CHECK( values[4] == Approx( 100e-6 ).margin( 20e-6 ) );
    app.stopWorker();
    REQUIRE_FALSE( app.m_workerFailed.load() );
}

/** \brief Publisher replacement and overwritten event history revoke old measurements before recovery.
 * \ingroup loopTimingMonitor_unit_test
 */
TEST_CASE( "loop timing automatic worker clears timing after source restart and event loss",
           "[loopTimingMonitor][integration][synco]" )
{
    // clang-format off
    #ifdef LOOPTIMINGMONITOR_TEST_DOXYGEN_REF
    MagAOX::app::loopTimingMonitor::threadExec();
    MagAOX::app::loopTimingMonitor::consume( 1 );
    MagAOX::app::loopTimingMonitor::timingValues( {} );
    #endif
    // clang-format on
    SyncoSources sources( 4 );
    SyncoHarness app( sources );
    app.enable();
    app.synco( true );
    app.start();
    REQUIRE( waitSynco(
        [&]()
        {
            return app.m_streamsReady.load();
        } ) );
    REQUIRE( sources.mode( true ) );
    std::array<double, 6> values{};
    REQUIRE( app.collect( sources, 0, values ) );
    const auto beforeRestart = app.m_captureEpoch.load();
    REQUIRE( sources.restartMain() );
    REQUIRE( waitSynco(
        [&]()
        {
            return app.m_captureEpoch.load() > beforeRestart;
        } ) );
    REQUIRE_FALSE( app.timingValues( values ) );
    app.consume( app.monotonicNow() );
    REQUIRE( app.m_monitor.summary().latency.count == 0 );
    REQUIRE( app.collect( sources, 0, values ) );
    CHECK( values[4] == Approx( 100e-6 ).margin( 20e-6 ) );
    const auto beforeGap = app.m_captureEpoch.load();
    const auto missed    = app.m_sdkMissed.load();

    for( unsigned index = 0; index < 1024; ++index )
    {
        REQUIRE( sources.auxiliary() );
    }

    REQUIRE( waitSynco(
        [&]()
        {
            return app.m_sdkMissed.load() > missed && app.m_captureEpoch.load() > beforeGap;
        } ) );
    REQUIRE_FALSE( app.timingValues( values ) );
    app.consume( app.monotonicNow() );
    REQUIRE( app.m_monitor.summary().latency.count == 0 );
    REQUIRE( app.collect( sources, 0, values ) );
    CHECK( values[2] == Approx( values[0] ).epsilon( 0.2 ) );
    CHECK( values[4] == Approx( 100e-6 ).margin( 20e-6 ) );
    app.stopWorker();
    REQUIRE_FALSE( app.m_workerFailed.load() );
}

/** \brief A camera gap preserves native history until delayed main labels permit safe timing recovery.
 * \ingroup loopTimingMonitor_unit_test
 */
TEST_CASE( "loop timing automatic worker retains delayed main history across a camera gap",
           "[loopTimingMonitor][integration][synco]" )
{
    // clang-format off
    #ifdef LOOPTIMINGMONITOR_TEST_DOXYGEN_REF
    MagAOX::app::loopTimingMonitor::threadExec();
    MagAOX::app::loopTimingMonitor::consume( 1 );
    MagAOX::app::loopTimingMonitor::timingValues( {} );
    #endif
    // clang-format on
    SyncoSources sources;
    SyncoHarness app( sources );
    app.enable();
    app.synco( true );
    app.start();
    REQUIRE( waitSynco(
        [&]()
        {
            return app.m_streamsReady.load();
        } ) );
    REQUIRE( sources.mode( true ) );
    REQUIRE( waitSynco(
        [&]()
        {
            return app.m_queue.queued() > 0;
        } ) );
    app.consume( app.monotonicNow() );
    REQUIRE( app.m_synco.known() );
    REQUIRE( app.m_synco.enabled() );
    const auto epoch   = app.m_captureEpoch.load();
    sources.m_holdMain = true;
    std::array<double, 6> values{};

    for( unsigned frame = 0; frame < 9; ++frame )
    {
        std::this_thread::sleep_for( std::chrono::milliseconds( 5 ) );
        REQUIRE( sources.frame( 3, frame == 3 ? 2 : 0 ) );
        REQUIRE( waitSynco(
            [&]()
            {
                return app.m_queue.queued() >= 5 || app.m_captureEpoch.load() != epoch;
            } ) );
        REQUIRE( app.m_captureEpoch.load() == epoch );
        app.consume( app.monotonicNow() );
        REQUIRE_FALSE( app.timingValues( values ) );
        CHECK( app.m_cameraMissed.load() == ( frame < 3 ? 0 : 2 ) );
        CHECK( app.m_streamsReady.load() );
    }

    REQUIRE( sources.flushMain() );
    REQUIRE( waitSynco(
        [&]()
        {
            return app.m_queue.queued() >= 9 || app.m_captureEpoch.load() != epoch;
        } ) );
    REQUIRE( app.m_captureEpoch.load() == epoch );
    app.consume( app.monotonicNow() );
    REQUIRE( app.timingValues( values ) );
    CHECK( app.m_monitor.summary().latency.count >= 3 );
    CHECK( values[0] > 0.001 );
    CHECK( values[2] == Approx( values[0] ).epsilon( 0.2 ) );
    CHECK( values[4] == Approx( 100e-6 ).margin( 20e-6 ) );
    CHECK( app.m_cameraMissed.load() == 2 );
    CHECK( app.m_sdkMissed.load() == 0 );
    CHECK( app.m_mainMissed.load() == 0 );
    CHECK( app.m_synco.quality().sdkMissed == 0 );
    CHECK( app.m_synco.quality().mainMissed == 0 );
    CHECK( app.m_synco.quality().unmatched == 0 );
    app.stopWorker();
    REQUIRE_FALSE( app.m_workerFailed.load() );
}

} // namespace loopTimingMonitorTest
} // namespace libXWCTest
