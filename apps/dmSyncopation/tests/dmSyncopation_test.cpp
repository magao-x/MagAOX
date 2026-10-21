/** \file dmSyncopation_test.cpp
 * \brief Deterministic timing/control tests with isolated workstation-local ImageStreamIO streams.
 * \ingroup dmSyncopation_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "../dmSyncopation.hpp"

#include <atomic>
#include <filesystem>
#include <fstream>
#include <future>
#include <limits>

namespace libXWCTest
{
namespace dmSyncopationTest
{
using namespace MagAOX::app;

/// \cond TEST_HARNESS
/// Scheduler access and cleanup without starting app, hardware, INDI or monitor threads.
class Harness : public dmSyncopation
{
  public:
    using dmSyncopation::allocate;
    using dmSyncopation::consume;
    using dmSyncopation::copyBatch;
    using dmSyncopation::executeStep;
    using dmSyncopation::Fault;
    using dmSyncopation::m_advances;
    using dmSyncopation::m_bounds;
    using dmSyncopation::m_cameraElement;
    using dmSyncopation::m_cameraFps;
    using dmSyncopation::m_cameraJitterUs;
    using dmSyncopation::m_cameraMaxUs;
    using dmSyncopation::m_cameraMinUs;
    using dmSyncopation::m_cameraMutex;
    using dmSyncopation::m_cameraProperty;
    using dmSyncopation::m_cameraReport;
    using dmSyncopation::m_cameraTime;
    using dmSyncopation::m_clean;
    using dmSyncopation::m_combined;
    using dmSyncopation::m_consumed;
    using dmSyncopation::m_copiedLate;
    using dmSyncopation::m_copyMax;
    using dmSyncopation::m_copyStarted;
    using dmSyncopation::m_cycleMutex;
    using dmSyncopation::m_delivered;
    using dmSyncopation::m_destinations;
    using dmSyncopation::m_enabled;
    using dmSyncopation::m_epoch;
    using dmSyncopation::m_faultCounts;
    using dmSyncopation::m_faultInterval;
    using dmSyncopation::m_faultPeriod;
    using dmSyncopation::m_faults;
    using dmSyncopation::m_faultTiming;
    using dmSyncopation::m_indiP_camera;
    using dmSyncopation::m_indiP_enable;
    using dmSyncopation::m_indiP_reacquire;
    using dmSyncopation::m_indiP_settings;
    using dmSyncopation::m_lastFaultNs;
    using dmSyncopation::m_lastFaultReason;
    using dmSyncopation::m_lastMain;
    using dmSyncopation::m_lockFrames;
    using dmSyncopation::m_mappings;
    using dmSyncopation::m_measuredPeriod;
    using dmSyncopation::m_missed;
    using dmSyncopation::m_notStarted;
    using dmSyncopation::m_now;
    using dmSyncopation::m_period;
    using dmSyncopation::m_periodSource;
    using dmSyncopation::m_previousCallbackEnd;
    using dmSyncopation::m_previousCallbackRuntime;
    using dmSyncopation::m_previousPrimaryPublication;
    using dmSyncopation::m_primaryCounter;
    using dmSyncopation::m_primaryTiming;
    using dmSyncopation::m_primed;
    using dmSyncopation::m_rateFrames;
    using dmSyncopation::m_reason;
    using dmSyncopation::m_record;
    using dmSyncopation::m_requested;
    using dmSyncopation::m_resources;
    using dmSyncopation::m_scheduleState;
    using dmSyncopation::m_submission;
    using dmSyncopation::m_timing;
    using dmSyncopation::m_wait;
    using dmSyncopation::missedStep;
    using dmSyncopation::readCamera;
    using dmSyncopation::readPrimaryTiming;
    using dmSyncopation::snapshot;
    using dmSyncopation::updatePeriod;
    using dmSyncopation::waitUntil;
    using MagAOXApp<true>::config;
    using dev::shmimMonitor<dmSyncopation>::m_imageStream;

    /// Set a stable test identity and an unopened primary handle.
    Harness();

    /// Close only test attachments; the fixture destroys its isolated streams.
    ~Harness() noexcept;

    /// Attach the primary using the same metadata the monitor supplies before allocate.
    void attach();

    /// Initialize callback properties without starting any threads or INDI connections.
    void properties();
};

Harness::Harness()
{
    m_configName  = "synco_test";
    m_imageStream = {};
}

Harness::~Harness() noexcept
{
    closeResources();
    if( m_imageStream.md )
        ImageStreamIO_closeIm( &m_imageStream );
}

void Harness::attach()
{
    m_shmimName    = "dm97disp03";
    m_combinedName = "dm97disp";
    m_destinations = { "dm97disp07", "dm97disp08" };
    m_timingName   = "synco_test_timing";
    m_epoch        = 12345;
    REQUIRE( ImageStreamIO_openIm( &m_imageStream, m_shmimName.c_str() ) == 0 );
    m_width           = m_imageStream.md->size[0];
    m_height          = m_imageStream.md->size[1];
    m_depth           = 1;
    m_dataType        = _DATATYPE_FLOAT;
    m_inode           = inode( m_shmimName );
    m_semaphoreNumber = 5;
    REQUIRE( allocate( dev::shmimT{} ) == 0 );
    REQUIRE( m_resources );
}

void Harness::properties()
{
    createStandardIndiToggleSw( m_indiP_enable, "enable" );
    createStandardIndiRequestSw( m_indiP_reacquire, "reacquire" );
    m_indiP_settings = pcf::IndiProperty( pcf::IndiProperty::Number );
    m_indiP_settings.setDevice( configName() );
    m_indiP_settings.setName( "settings" );
}

/// Owned files and images under a unique MILK_SHM_DIR; never attaches an instrument stream.
struct Streams
{
    std::string          directory;         ///< Unique temporary directory.
    std::string          prior;             ///< Saved process stream-directory environment.
    bool                 hadPrior{ false }; ///< Whether the old environment value existed.
    std::array<IMAGE, 4> images{};          ///< Primary, combined and two numbered destinations.

    /// Create local-only float images and initialize their pixels.
    Streams();

    /// Destroy test-owned streams and restore the environment.
    ~Streams();
};

Streams::Streams()
{
    if( const char *value = getenv( "MILK_SHM_DIR" ) )
    {
        prior    = value;
        hadPrior = true;
    }
    // ImageStreamIO caches its directory on first filename lookup. Keep one directory
    // for this test executable; individual fixtures still destroy all of their images.
    static const std::string processDirectory = []()
    {
        char        pattern[] = "/tmp/dm-synco-test-XXXXXX";
        const char *path      = mkdtemp( pattern );
        if( !path )
            throw std::runtime_error( "cannot create isolated ImageStreamIO directory" );
        return std::string( path );
    }();
    directory = processDirectory;
    setenv( "MILK_SHM_DIR", directory.c_str(), 1 );
    const std::array<const char *, 4> names{ "dm97disp03", "dm97disp", "dm97disp07", "dm97disp08" };
    uint32_t                          sizes[]{ 3, 3 };
    for( size_t j = 0; j < images.size(); ++j )
    {
        REQUIRE( ImageStreamIO_createIm_gpu(
                     &images[j], names[j], 2, sizes, _DATATYPE_FLOAT, -1, 1, IMAGE_NB_SEMAPHORE, 0, 0, 0 ) == 0 );
        std::fill_n( images[j].array.F, 9, float( j ) );
    }
}

Streams::~Streams()
{
    for( const char *name : { "dm97disp07_aux", "dm97disp08_aux", "synco_test_timing" } )
    {
        IMAGE image{};
        if( ImageStreamIO_openIm( &image, name ) == 0 )
            ImageStreamIO_destroyIm( &image );
    }
    for( auto &image : images )
        if( image.md )
            ImageStreamIO_destroyIm( &image );
    if( hadPrior )
        setenv( "MILK_SHM_DIR", prior.c_str(), 1 );
    else
        unsetenv( "MILK_SHM_DIR" );
}

/// Fake monotonic execution clock, with controlled output acknowledgements during waits.
struct Clock
{
    static inline int64_t  time          = 1000000000; ///< Current virtual nanoseconds.
    static inline Harness *app           = nullptr;    ///< Active isolated app fixture.
    static inline bool     acknowledge   = true;       ///< Whether a synthetic combiner consumes auxiliary triggers.
    static inline bool     earlyMain     = false;      ///< Inject a primary event during the next wait.
    static inline bool     lateCopy      = false;      ///< Inject an overrun only after a destination copy starts.
    static inline bool     disableInCopy = false;      ///< Cancel between passive mapping copies.
    static inline int64_t  disableAt     = 0;          ///< Cancel in the final pre-trigger interval.
    static inline int64_t  wakeDelayNs   = 0;          ///< Deterministic scheduling delay after every requested wake.
    static inline bool     finishOutput  = false; ///< Complete an expected output whose counter is already visible.
    static inline uint64_t trigger       = 0;     ///< Last designated destination counter observed.

    /// Read deterministic elapsed work rather than host scheduler latency.
    static int64_t now();

    /// Advance to the absolute target plus wake delay and emulate one combiner publication.
    static int wait( const timespec *target /**< [in] absolute wait target */ );

    /// Connect the fake clock and choose feasible synthetic timing bounds.
    static void connect( Harness &scheduler /**< [in,out] isolated scheduler */ );

    /// Publish synthetic primary/combined events before invoking the monitor callback.
    static void primary( int64_t  observed /**< [in] Absolute virtual observation time. */,
                         uint64_t count = 1 /**< [in] Genuine publications represented by this callback. */ );
};

int64_t Clock::now()
{
    time += 100;
    if( app && ( ( disableInCopy && app->m_copyStarted ) || ( disableAt > 0 && time >= disableAt ) ) )
    {
        app->m_enabled = false;
        disableInCopy  = false;
        disableAt      = 0;
    }
    if( lateCopy && app && app->m_copyStarted )
    {
        time += 50000;
        lateCopy = false;
    }
    return time;
}

int Clock::wait( const timespec *target )
{
    time = std::max( time, int64_t( target->tv_sec ) * 1000000000 + target->tv_nsec ) + wakeDelayNs;
    if( finishOutput )
    {
        app->m_combined->image()->md->write = 0;
        finishOutput                        = false;
    }
    if( earlyMain )
    {
        ++app->m_imageStream.md->cnt0;
        earlyMain = false;
    }
    if( acknowledge && !app->m_mappings.empty() )
    {
        const auto value = app->m_mappings.front().m_destination->image()->md->cnt0;
        if( trigger != value )
        {
            trigger = value;
            ++app->m_combined->image()->md->cnt0;
        }
    }
    return 0;
}

void Clock::connect( Harness &scheduler )
{
    app                            = &scheduler;
    time                           = 1000000000;
    trigger                        = scheduler.m_mappings.front().m_counter;
    acknowledge                    = true;
    earlyMain                      = false;
    lateCopy                       = false;
    disableInCopy                  = false;
    disableAt                      = 0;
    wakeDelayNs                    = 0;
    finishOutput                   = false;
    scheduler.m_now                = now;
    scheduler.m_wait               = wait;
    scheduler.m_enabled            = true;
    scheduler.m_primaryCounter     = scheduler.m_imageStream.md->cnt0;
    scheduler.m_consumed           = scheduler.m_combined->image()->md->cnt0;
    scheduler.m_bounds.guard       = 1000;
    scheduler.m_bounds.delivery    = 1000;
    scheduler.m_bounds.observation = 1000;
    scheduler.m_bounds.early       = 5000;
    scheduler.m_bounds.lead        = 10000;
    scheduler.m_bounds.snapshot    = 3000;
    scheduler.m_bounds.copy        = 3000;
    scheduler.m_bounds.reserve     = 1000;
    scheduler.m_lockFrames         = 2;
}

void Clock::primary( int64_t observed, uint64_t count )
{
    time = observed;
    app->m_imageStream.md->cnt0 += count;
    app->m_combined->image()->md->cnt0 += count;
    REQUIRE( app->processImage( nullptr, dev::shmimT{} ) == 0 );
}

/// Publish independent wall-clock metadata before a monotonic callback without rewinding elapsed execution time.
void primaryObservation( Harness &app /**< [in,out] isolated scheduler */,
                         int64_t  observed /**< [in] monotonic time before callback entry */,
                         int64_t  publication /**< [in] independent wall-clock publication time */,
                         uint64_t count = 1 /**< [in] publications represented by this observation */ )
{
    REQUIRE( observed >= Clock::time );
    auto *metadata      = app.m_imageStream.md;
    metadata->write     = 1;
    metadata->writetime = synco::toTimespec( publication );
    metadata->cnt0 += count;
    if( app.m_combined )
        app.m_combined->image()->md->cnt0 += count;
    metadata->write = 0;
    Clock::time     = observed;
    REQUIRE( app.processImage( nullptr, dev::shmimT{} ) == 0 );
}

/// Atomic clock and metadata mutation at the existing sampling read, without nondeterministic writer races.
struct PrimaryProbe
{
    static inline std::atomic<int64_t> time{ 1000000000 }; ///< Monotonic execution shared with a lock-wait worker.
    static inline IMAGE *image{ nullptr }; ///< Isolated primary whose publication bracket may be interrupted.
    static inline int    tear{ 0 };        ///< One-shot write-flag or counter change during metadata sampling.
    static inline std::promise<void> *entered{ nullptr }; ///< Optional notification that callback entry was captured.

    /// Read elapsed work, then inject one metadata change or notify the lock-wait coordinator.
    static int64_t now();

    /// Clear all test-only hooks before adopting this clock.
    static void reset();
};

int64_t PrimaryProbe::now()
{
    const int64_t value = time.fetch_add( 100 ) + 100;
    if( image && tear )
    {
        if( tear == 1 )
            image->md->write = 1;
        else
            ++image->md->cnt0;
        tear = 0;
    }
    if( entered )
    {
        auto *signal = entered;
        entered      = nullptr;
        signal->set_value();
    }
    return value;
}

void PrimaryProbe::reset()
{
    time    = 1000000000;
    image   = nullptr;
    tear    = 0;
    entered = nullptr;
}

/// Create a numeric camera acquisition report using seconds for interval statistics.
pcf::IndiProperty
cameraTiming( double fps /**< [in] Measured acquisition frequency in Hz. */,
              bool   statistics = true /**< [in] Include jitter/minimum/maximum acquisition intervals. */ )
{
    pcf::IndiProperty packet( pcf::IndiProperty::Number );
    packet.setDevice( "camwfs" );
    packet.setName( "fg_timing" );
    packet.add( pcf::IndiElement( "acq_fps" ) );
    packet["acq_fps"] = fps;
    if( statistics )
    {
        for( const auto &name : { "acq_jitter", "acq_min", "acq_max" } )
            packet.add( pcf::IndiElement( name ) );
        packet["acq_jitter"] = 1.234e-6;
        packet["acq_min"]    = 490e-6;
        packet["acq_max"]    = 510e-6;
    }
    return packet;
}
/// \endcond

/// Verify fixed count feasibility, rounding, passive-only operation and narrow copy-window rejection.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation count remains fixed at the 500 us boundary", "[dmSyncopation]" )
{
    synco::Bounds bounds;
    bounds.delivery = 1000;
    bounds.guard    = 1000;
    bounds.early    = 5000;
    bounds.lead     = 10000;
    for( int64_t period : { 499980, 500000, 500020 } )
    {
        REQUIRE( synco::feasible( period, 3, bounds ) );
        REQUIRE( synco::maximumSteps( period, bounds ) >= 3 );
    }
    REQUIRE( synco::feasible( 500000, 1, bounds ) );
    REQUIRE_FALSE( synco::feasible( 500000, 0, bounds ) );
    REQUIRE_FALSE( synco::feasible( 500000, 17, bounds ) );
    REQUIRE_FALSE( synco::feasible( 500000, 3, synco::Bounds{} ) );
    REQUIRE( synco::copyCutoff( 1000000, synco::Bounds{} ) == 974000 );
    bounds.copy = 100000;
    REQUIRE_FALSE( synco::feasible( 500000, 3, bounds ) );
    REQUIRE( synco::classifyStep( false, 0, 100 ) == synco::StepOutcome::notStarted );
    REQUIRE( synco::classifyStep( true, 100, 100 ) == synco::StepOutcome::copiedOnTime );
    REQUIRE( synco::classifyStep( true, 101, 100 ) == synco::StepOutcome::copiedLate );
}

/// Verify independently clamped feedback, no missed-sample learning and restart/expiry record semantics.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation feedback and future record validity", "[dmSyncopation]" )
{
    int64_t advance = 0;
    for( int j = 0; j < 100; ++j )
        advance = synco::feedback( advance, 2000 - advance, 0.2, 20000, true );
    REQUIRE( advance == Approx( 2000 ).margin( 5 ) );
    REQUIRE( synco::feedback( advance, 1000000, 0.2, 20000, false ) == advance );
    REQUIRE( synco::feedback( 0, 1000000, 1, 20000, true ) == 20000 );
    REQUIRE( synco::feedback( 0, -1000000, 1, 20000, true ) == -20000 );
    REQUIRE( synco::feedback( 17, 1000000, 0, 20000, true ) == 17 );
    synco::Record record{};
    record[synco::version]     = 1;
    record[synco::epoch]       = 123;
    record[synco::generation]  = 1;
    record[synco::state]       = synco::acquiring;
    record[synco::steps]       = 3;
    record[synco::period]      = 500000;
    record[synco::reference]   = 1000000;
    record[synco::expires]     = 2000000;
    record[synco::cutoffs]     = 1400000;
    record[synco::targets]     = 1500000;
    record[synco::cutoffs + 1] = 1100000;
    record[synco::targets + 1] = 1166666;
    REQUIRE( synco::validRecord( record, 1000001 ) );
    REQUIRE( synco::nextStep( record, 1000001 ) == 1 );
    REQUIRE( synco::nextStep( record, 1100001 ) == 0 );
    REQUIRE_FALSE( synco::validRecord( record, 2000000 ) );
    record[synco::epoch] = 0;
    REQUIRE_FALSE( synco::validRecord( record, 1000001 ) );
}

/// Verify the aggregate survives the ordinary log-entry/schema verification path.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation telemetry is registered with log verification", "[dmSyncopation]" )
{
    using namespace MagAOX::logger;
    telem_dmsynco::messageT message(
        synco::armed, 3, 500000, 100, 2, 1, 7, 1000, 2000, { -500, 800, 100 }, { 0, 10, 20 } );
    flatlogs::bufferPtrT buffer;
    REQUIRE( flatlogs::logHeader::createLog<telem_dmsynco>(
                 buffer, flatlogs::timespecX( 1, 2 ), message, flatlogs::logPrio::LOG_DEFAULT ) == 0 );
    REQUIRE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
    const auto *record = GetTelem_dmsynco_fb( flatlogs::logHeader::messageBuffer( buffer ) );
    REQUIRE( record->residuals_ns()->Get( 1 ) == 800 );
    REQUIRE( record->advances_ns()->Get( 2 ) == 20 );
    REQUIRE( record->delivered() == 100 );
    REQUIRE_FALSE( verifyLogEntry( telem_dmsynco::eventCode, nullptr ) );
}

/// Exercise registered configuration, invalid ranges and same-DM/alias restrictions without shared-memory writes.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation validates configuration before attaching streams", "[dmSyncopation]" )
{
    // clang-format off
    #ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::setupConfig();
    dmSyncopation::loadConfigImpl(mx::app::appConfigurator());
    #endif
    // clang-format on
    Harness app;
    app.setupConfig();
    REQUIRE( app.loadConfigImpl( app.config ) == 0 );
    REQUIRE( app.m_requested == 3 );
    REQUIRE( app.m_cameraProperty == "fg_timing" );
    REQUIRE( app.m_cameraElement == "acq_fps" );
    REQUIRE_FALSE( app.m_enabled );
    app.m_destinations = { "dm01disp07", "dm02disp08" };
    REQUIRE( app.loadConfigImpl( app.config ) < 0 );
    app.m_destinations = { "dm01disp07", "dm01disp07" };
    REQUIRE( app.loadConfigImpl( app.config ) < 0 );
    app.m_destinations = { "dm01disp07", "dm01disp08" };
    app.m_requested    = 17;
    REQUIRE( app.loadConfigImpl( app.config ) < 0 );
}

/// Differentiate passive deadline outcomes, dependent cancellations and controller-reader holds.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation classifies misses and preserves reader lifetime", "[dmSyncopation]" )
{
    // clang-format off
    #ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::executeStep(0, 0, 0);
    dmSyncopation::missedStep();
    #endif
    // clang-format on
    Streams streams;
    Harness app;
    app.attach();
    Clock::connect( app );
    app.m_scheduleState = synco::armed;
    app.m_lastMain      = Clock::time;

    SECTION( "reader hold prevents even passive destination replacement" )
    {
        auto &mapping = app.m_mappings.front();
        std::fill_n( mapping.m_destination->data(), 9, 6.0f );
        app.m_submission     = Clock::time + 100000;
        app.m_requested      = 2;
        const int64_t cutoff = Clock::time + 100000;
        REQUIRE_FALSE( app.executeStep( 0, cutoff - 10000, cutoff ) );
        REQUIRE( Clock::time < cutoff ); // Reject the impossible hold before waiting into a future main event.
        REQUIRE( app.m_notStarted == 1 );
        REQUIRE( mapping.m_destination->data()[0] == 6.0f );
        REQUIRE( mapping.m_destination->image()->md->cnt0 == mapping.m_counter );
        REQUIRE( app.m_missed == 1 );
        REQUIRE( app.m_faults == 0 );
    }
    SECTION( "a late copy is detected even below a historical maximum" )
    {
        app.m_copyMax   = 1000000;
        Clock::lateCopy = true;
        REQUIRE_FALSE( app.executeStep( 0, Clock::time + 90000, Clock::time + 100000 ) );
        REQUIRE( app.m_copiedLate == 1 );
        REQUIRE( app.m_notStarted == 0 );
        REQUIRE( app.m_missed == 1 );
    }
    SECTION( "three original misses reacquire; a single miss does not count cancelled dependents" )
    {
        app.missedStep();
        REQUIRE( app.m_scheduleState == synco::armed );
        REQUIRE( app.m_missed == 1 );
        app.missedStep();
        REQUIRE( app.m_scheduleState == synco::armed );
        app.missedStep();
        REQUIRE( app.m_scheduleState == synco::invalid );
        REQUIRE( app.m_missed == 3 );
    }
}

/// Finish an in-flight passive batch on disable, suppress its trigger and wait for complete output publication.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation cancellation preserves complete batches without deadline misses", "[dmSyncopation]" )
{
    // clang-format off
    #ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::executeStep(0, 0, 0);
    dmSyncopation::consume();
    #endif
    // clang-format on
    Streams streams;
    Harness app;
    app.attach();
    Clock::connect( app );
    app.m_scheduleState = synco::armed;
    app.m_lastMain      = Clock::time;
    SECTION( "disable inside a passive batch retains the complete staged values" )
    {
        for( auto &mapping : app.m_mappings )
        {
            std::fill_n( mapping.m_source->data(), 9, 7.0f );
            mapping.m_source->post();
        }
        Clock::disableInCopy = true;
        REQUIRE_FALSE( app.executeStep( 0, Clock::time + 90000, Clock::time + 100000 ) );
        for( auto &mapping : app.m_mappings )
        {
            REQUIRE( mapping.m_destination->data()[0] == 7.0f );
            REQUIRE( mapping.m_destination->image()->md->cnt0 == mapping.m_counter );
        }
        REQUIRE( app.m_missed == 0 );
        REQUIRE( app.m_notStarted == 0 );
    }
    SECTION( "disable in the final publication interval suppresses the counter" )
    {
        const int64_t target = Clock::time + 166666;
        Clock::disableAt     = target + 200;
        REQUIRE_FALSE( app.executeStep( 1, target, Clock::time + 300000 ) );
        REQUIRE_FALSE( app.m_enabled );
        REQUIRE( app.m_mappings.front().m_destination->image()->md->cnt0 == 0 );
        REQUIRE( app.m_missed == 0 );
    }
    SECTION( "the expected counter can precede write-flag clearing" )
    {
        ++app.m_combined->image()->md->cnt0;
        app.m_combined->image()->md->write = 1;
        Clock::finishOutput                = true;
        REQUIRE( app.consume() );
        REQUIRE( app.m_faults == 0 );
        REQUIRE( app.m_consumed == app.m_combined->image()->md->cnt0 );
    }
}

/// Reject replaced/malformed resources without publishing into a conflicting timing or DM stream.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation invalidates stream replacement and malformed timing", "[dmSyncopation]" )
{
    // clang-format off
    #ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::processImage(nullptr, dev::shmimT());
    #endif
    // clang-format on
    Streams streams;
    Harness app;
    app.attach();
    Clock::connect( app );
    SECTION( "timing inode replacement invalidates the old publisher" )
    {
        char filename[PATH_MAX];
        ImageStreamIO_filename( filename, sizeof( filename ), "synco_test_timing" );
        const std::string backup = std::string( filename ) + ".old";
        REQUIRE( rename( filename, backup.c_str() ) == 0 );
        std::ofstream replacement( filename );
        replacement << "replacement";
        replacement.close();
        app.processImage( nullptr, dev::shmimT{} );
        REQUIRE_FALSE( app.m_resources );
        REQUIRE( app.m_scheduleState == synco::invalid );
        std::remove( filename );
        REQUIRE( rename( backup.c_str(), filename ) == 0 );
    }
    SECTION( "changed timing layout is rejected before a record write" )
    {
        const auto count         = app.m_timing.md->cnt0;
        app.m_timing.md->size[0] = 1;
        app.processImage( nullptr, dev::shmimT{} );
        REQUIRE_FALSE( app.m_resources );
        // Restore via the original stream handle so cleanup sees the allocated extent.
        IMAGE old{};
        REQUIRE( ImageStreamIO_openIm( &old, "synco_test_timing" ) == 0 );
        REQUIRE( old.md->cnt0 == count );
        old.md->size[0] = synco::words;
        ImageStreamIO_closeIm( &old );
    }
}

/// Verify persistent source fallback and always-copy behavior preserve primary and passive destination counters.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation copies cached and unchanged commands passively", "[dmSyncopation]" )
{
    // clang-format off
    #ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::snapshot();
    dmSyncopation::copyBatch(0);
    #endif
    // clang-format on
    Streams streams;
    Harness app;
    app.attach();
    Clock::connect( app );
    const auto primary     = streams.images[0].md->cnt0;
    const auto primaryTime = streams.images[0].md->writetime;
    auto      &mapping     = app.m_mappings.front();
    std::fill_n( mapping.m_source->data(), 9, 7.0f );
    mapping.m_source->post();
    REQUIRE( app.snapshot() );
    REQUIRE( app.copyBatch( Clock::time + 10000 ) );
    REQUIRE( mapping.m_destination->data()[0] == 7.0f );
    std::fill_n( mapping.m_destination->data(), 9, -1.0f );
    REQUIRE( app.copyBatch( Clock::time + 10000 ) ); // Same staged/counter values still cause every memcpy.
    REQUIRE( mapping.m_destination->data()[0] == 7.0f );
    mapping.m_source->setWrite( true );
    std::fill_n( mapping.m_source->data(), 9, 99.0f );
    REQUIRE( app.snapshot() );
    REQUIRE( app.copyBatch( Clock::time + 10000 ) );
    REQUIRE( mapping.m_destination->data()[0] == 7.0f );
    mapping.m_source->setWrite( false );
    REQUIRE( mapping.m_destination->image()->md->cnt0 == mapping.m_counter );
    REQUIRE( streams.images[0].md->cnt0 == primary );
    REQUIRE( streams.images[0].md->writetime.tv_sec == primaryTime.tv_sec );
    REQUIRE( streams.images[0].md->writetime.tv_nsec == primaryTime.tv_nsec );
}

/// Drive complete cycles with virtual waits, exact output acknowledgements and a fixed requested count.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation primes step zero before issuing later triggers", "[dmSyncopation]" )
{
    // clang-format off
    #ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::processImage(nullptr, dev::shmimT());
    dmSyncopation::executeStep(0, 0, 0);
    #endif
    // clang-format on
    Streams streams;
    Harness app;
    app.attach();
    Clock::connect( app );
    for( int cycle = 0; cycle < 6; ++cycle )
    {
        Clock::time = 1000000000 + cycle * 500000;
        ++streams.images[0].md->cnt0;
        ++streams.images[1].md->cnt0;
        REQUIRE( app.processImage( nullptr, dev::shmimT{} ) == 0 );
        if( cycle < 2 )
            REQUIRE( app.m_delivered == 0 );
    }
    REQUIRE( app.m_primed );
    REQUIRE( app.m_scheduleState == synco::armed );
    REQUIRE( app.m_requested == 3 );
    REQUIRE( app.m_delivered == 6 );
    REQUIRE( app.m_mappings.front().m_destination->image()->md->cnt0 == 6 );
    REQUIRE( app.m_mappings[1].m_destination->image()->md->cnt0 == 0 );
    REQUIRE( streams.images[0].md->cnt0 == 6 );
    REQUIRE( app.m_faults == 0 );
    REQUIRE( synco::validRecord( app.m_record, Clock::time ) );

    SECTION( "an early main cancels before any new counter publication" )
    {
        const auto count = app.m_mappings.front().m_destination->image()->md->cnt0;
        Clock::earlyMain = true;
        REQUIRE_FALSE( app.waitUntil( Clock::time + 20000 ) );
        REQUIRE( app.m_scheduleState == synco::invalid );
        REQUIRE( app.m_mappings.front().m_destination->image()->md->cnt0 == count );
    }
    SECTION( "a missing output is an immediate synchronization fault" )
    {
        Clock::acknowledge = false;
        Clock::time        = 1003000000;
        ++streams.images[0].md->cnt0;
        ++streams.images[1].md->cnt0;
        app.processImage( nullptr, dev::shmimT{} );
        REQUIRE( app.m_scheduleState == synco::invalid );
        REQUIRE( app.m_faults > 0 );
    }
    SECTION( "one late main cannot stretch the next safety window" )
    {
        const auto delivered = app.m_delivered;
        Clock::time          = 1004000000; // Last main was at 2.5 ms: a single 1.5 ms interval.
        ++streams.images[0].md->cnt0;
        ++streams.images[1].md->cnt0;
        app.processImage( nullptr, dev::shmimT{} );
        REQUIRE( app.m_delivered == delivered );
        REQUIRE_FALSE( app.m_primed );
        REQUIRE( app.m_scheduleState == synco::invalid );
    }
    SECTION( "unexpected destination writes invalidate and refresh resources" )
    {
        ++streams.images[2].md->cnt0;
        app.processImage( nullptr, dev::shmimT{} );
        REQUIRE_FALSE( app.m_resources );
        REQUIRE( app.m_scheduleState == synco::invalid );
    }
}

/// Keep conservative copy budgets from moving fast passive copies toward the hard latest start.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation passive feedback retains headroom with a grown copy budget", "[dmSyncopation][regression]" )
{
    // clang-format off
    #ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::executeStep(0, 0, 0);
    #endif
    // clang-format on
    Streams streams;
    Harness app;
    app.attach();
    Clock::connect( app );
    app.m_bounds.copy   = 21000;
    app.m_requested     = 2;
    app.m_scheduleState = synco::armed;
    for( int cycle = 0; cycle < 30; ++cycle )
    {
        INFO( "passive cycle " << cycle );
        const int64_t observed = 1000000000 + cycle * 500000;
        Clock::time            = observed;
        app.m_lastMain         = observed;
        app.m_submission       = observed;
        const int64_t cutoff   = synco::copyCutoff( observed + 500000, app.m_bounds );
        REQUIRE( app.executeStep( 0, cutoff - app.m_bounds.reserve, cutoff ) );
        // The virtual memcpy is much faster than its conservative budget. Completing
        // before the latest permissible start demonstrates retained scheduling headroom.
        REQUIRE( Clock::time < cutoff - app.m_bounds.copy );
    }
    REQUIRE( app.m_notStarted == 0 );
    REQUIRE( app.m_missed == 0 );
    REQUIRE( app.m_faults == 0 );
    REQUIRE( app.m_mappings.front().m_destination->image()->md->cnt0 == 0 );
}

/// Prime and retain a feasible two-step schedule despite repeatable five-microsecond wake latency.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation primes despite deterministic wake overshoot", "[dmSyncopation][regression]" )
{
    // clang-format off
    #ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::processImage(nullptr, dev::shmimT());
    #endif
    // clang-format on
    Streams streams;
    Harness app;
    app.attach();
    Clock::connect( app );
    app.m_bounds          = synco::Bounds{};
    app.m_bounds.copy     = 21000;
    app.m_bounds.snapshot = 7000;
    app.m_requested       = 2;
    Clock::wakeDelayNs    = 5000;
    REQUIRE( app.setCallBack_m_indiP_camera( cameraTiming( 2000, false ) ) == 0 );
    REQUIRE( synco::feasible( 500000, 2, app.m_bounds ) );
    for( int cycle = 0; cycle < 40; ++cycle )
    {
        INFO( "delayed-wake cycle " << cycle );
        Clock::time = 1000000000 + cycle * 500000;
        ++streams.images[0].md->cnt0;
        ++streams.images[1].md->cnt0;
        REQUIRE( app.processImage( nullptr, dev::shmimT{} ) == 0 );
        if( cycle >= 10 )
        {
            REQUIRE( app.m_primed );
            REQUIRE( app.m_scheduleState == synco::armed );
        }
    }
    REQUIRE( app.m_requested == 2 );
    REQUIRE( app.m_delivered >= 30 );
    REQUIRE( app.m_notStarted == 0 );
    REQUIRE( app.m_missed == 0 );
    REQUIRE( app.m_faults == 0 );
}

/// Drain duplicate callback notifications without inventing a new main observation during reacquisition.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation ignores queued primary notifications around a counter jump", "[dmSyncopation][regression]" )
{
    // clang-format off
    #ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::processImage(nullptr, dev::shmimT());
    #endif
    // clang-format on
    Streams streams;
    Harness app;
    app.attach();
    Clock::connect( app );
    app.m_requested = 2;
    ++streams.images[0].md->cnt0;
    ++streams.images[1].md->cnt0;
    REQUIRE( app.processImage( nullptr, dev::shmimT{} ) == 0 );
    const int64_t firstObservation = app.m_lastMain;
    REQUIRE( firstObservation > 0 );
    const auto firstRecord = app.m_timing.md->cnt0;
    for( int duplicate = 0; duplicate < 5; ++duplicate )
    {
        REQUIRE( app.processImage( nullptr, dev::shmimT{} ) == 0 );
        REQUIRE( app.m_lastMain == firstObservation );
        REQUIRE( app.m_clean == 0 );
        REQUIRE( app.m_faults == 0 );
        REQUIRE( app.m_delivered == 0 );
        REQUIRE( app.m_timing.md->cnt0 == firstRecord );
    }

    Clock::time = 1000500000;
    streams.images[0].md->cnt0 += 3;
    streams.images[1].md->cnt0 += 3;
    REQUIRE( app.processImage( nullptr, dev::shmimT{} ) == 0 );
    REQUIRE( app.m_primaryCounter == streams.images[0].md->cnt0 );
    REQUIRE( app.m_faults == 1 );
    REQUIRE( app.m_lastMain == 0 );
    REQUIRE_FALSE( app.m_primed );
    REQUIRE( app.m_faultCounts[size_t( Harness::Fault::primary )] == 1 );
    const std::string lastFault     = app.m_lastFaultReason;
    const int64_t     lastFaultTime = app.m_lastFaultNs;
    REQUIRE( lastFault != "none" );
    REQUIRE( lastFaultTime > 0 );
    const auto invalidRecord = app.m_timing.md->cnt0;
    for( int duplicate = 0; duplicate < 5; ++duplicate )
    {
        REQUIRE( app.processImage( nullptr, dev::shmimT{} ) == 0 );
        REQUIRE( app.m_lastMain == 0 );
        REQUIRE( app.m_clean == 0 );
        REQUIRE( app.m_faults == 1 );
        REQUIRE( app.m_delivered == 0 );
        REQUIRE( app.m_timing.md->cnt0 == invalidRecord );
    }

    for( int cycle = 2; cycle < 12; ++cycle )
    {
        Clock::time = 1000000000 + cycle * 500000;
        ++streams.images[0].md->cnt0;
        ++streams.images[1].md->cnt0;
        REQUIRE( app.processImage( nullptr, dev::shmimT{} ) == 0 );
    }
    REQUIRE( app.m_primed );
    REQUIRE( app.m_scheduleState == synco::armed );
    REQUIRE( app.m_delivered > 0 );
    REQUIRE( app.m_faults == 1 );
    REQUIRE( std::string( app.m_reason ) != lastFault );
    REQUIRE( std::string( app.m_lastFaultReason ) == lastFault );
    REQUIRE( app.m_lastFaultNs == lastFaultTime );
    REQUIRE( app.m_faultCounts[size_t( Harness::Fault::primary )] == 1 );
}

/// Validate measured acquisition reports, optional seconds-to-microseconds diagnostics and custom property names.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation consumes measured camera acquisition reports", "[dmSyncopation][camera]" )
{
    // clang-format off
    #ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::setCallBack_m_indiP_camera(pcf::IndiProperty());
    dmSyncopation::loadConfigImpl(mx::app::appConfigurator());
    #endif
    // clang-format on
    Harness app;
    Clock::app  = nullptr;
    Clock::time = 1000000000;
    app.m_now   = Clock::now;
    REQUIRE( app.m_cameraJitterUs == -1 );
    REQUIRE( app.m_cameraMinUs == -1 );
    REQUIRE( app.m_cameraMaxUs == -1 );
    auto packet = cameraTiming( 2000.08 );
    REQUIRE( app.setCallBack_m_indiP_camera( packet ) == 0 );
    REQUIRE( app.m_cameraFps.load() == Approx( 2000.08 ) );
    REQUIRE( app.m_cameraJitterUs == Approx( 1.234 ) );
    REQUIRE( app.m_cameraMinUs == Approx( 490 ) );
    REQUIRE( app.m_cameraMaxUs == Approx( 510 ) );
    const int64_t acceptedTime = app.m_cameraTime.load();
    REQUIRE( acceptedTime > 0 );

    SECTION( "a subsequent FPS-only report clears obsolete optional statistics" )
    {
        REQUIRE( app.setCallBack_m_indiP_camera( cameraTiming( 1900, false ) ) == 0 );
        REQUIRE( app.m_cameraFps.load() == 1900 );
        REQUIRE( app.m_cameraTime.load() > acceptedTime );
        REQUIRE( app.m_cameraJitterUs == -1 );
        REQUIRE( app.m_cameraMinUs == -1 );
        REQUIRE( app.m_cameraMaxUs == -1 );
    }
    SECTION( "the camera reciprocal is rounded once without callback-rate smoothing" )
    {
        app.updatePeriod( Clock::time, 0, true );
        REQUIRE( app.m_period == std::llround( 1e9 / 2000.08 ) );
        REQUIRE( std::string( app.m_periodSource ) == "camera acquisition" );
    }
    SECTION( "numeric acquisition fields allow surrounding whitespace" )
    {
        packet["acq_fps"]    = " 2000.08 \t";
        packet["acq_jitter"] = " 1.234e-6\n";
        REQUIRE( app.setCallBack_m_indiP_camera( packet ) == 0 );
        REQUIRE( app.m_cameraFps.load() == Approx( 2000.08 ) );
        REQUIRE( app.m_cameraJitterUs == Approx( 1.234 ) );
        REQUIRE( app.m_cameraTime.load() > acceptedTime );
    }
    SECTION( "invalid required and optional values reject the entire report" )
    {
        for( double bad : { 0.0,
                            -1.0,
                            100001.0,
                            std::numeric_limits<double>::quiet_NaN(),
                            std::numeric_limits<double>::infinity() } )
        {
            packet["acq_fps"] = bad;
            INFO( "acq_fps=" << packet["acq_fps"].getValue() );
            REQUIRE( app.setCallBack_m_indiP_camera( packet ) < 0 );
            REQUIRE( app.m_cameraFps.load() == Approx( 2000.08 ) );
            REQUIRE( app.m_cameraTime.load() == acceptedTime );
        }
        for( const auto &bad : { "", "invalid", "2000Hz", "nan", "inf", "1e999" } )
        {
            packet["acq_fps"] = bad;
            INFO( "acq_fps=" << bad );
            REQUIRE( app.setCallBack_m_indiP_camera( packet ) < 0 );
            REQUIRE( app.m_cameraFps.load() == Approx( 2000.08 ) );
            REQUIRE( app.m_cameraTime.load() == acceptedTime );
        }
        for( const auto &name : { "acq_jitter", "acq_min", "acq_max" } )
            for( double bad :
                 { -1.0, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity() } )
            {
                packet       = cameraTiming( 1900 );
                packet[name] = bad;
                INFO( name << '=' << packet[name].getValue() );
                REQUIRE( app.setCallBack_m_indiP_camera( packet ) < 0 );
                REQUIRE( app.m_cameraFps.load() == Approx( 2000.08 ) );
                REQUIRE( app.m_cameraTime.load() == acceptedTime );
                REQUIRE( app.m_cameraJitterUs == Approx( 1.234 ) );
                REQUIRE( app.m_cameraMinUs == Approx( 490 ) );
                REQUIRE( app.m_cameraMaxUs == Approx( 510 ) );
            }
        for( const auto &name : { "acq_jitter", "acq_min", "acq_max" } )
            for( const auto &bad : { "", "invalid", "1e-6 trailing", "nan", "inf", "1e999" } )
            {
                packet       = cameraTiming( 1900 );
                packet[name] = bad;
                INFO( name << '=' << bad );
                REQUIRE( app.setCallBack_m_indiP_camera( packet ) < 0 );
                REQUIRE( app.m_cameraFps.load() == Approx( 2000.08 ) );
                REQUIRE( app.m_cameraTime.load() == acceptedTime );
                REQUIRE( app.m_cameraJitterUs == Approx( 1.234 ) );
                REQUIRE( app.m_cameraMinUs == Approx( 490 ) );
                REQUIRE( app.m_cameraMaxUs == Approx( 510 ) );
            }
        packet            = cameraTiming( 1900 );
        packet["acq_min"] = 1e-3;
        REQUIRE( app.setCallBack_m_indiP_camera( packet ) < 0 );
        REQUIRE( app.m_cameraTime.load() == acceptedTime );
    }
    SECTION( "missing acquisition rate and mismatched property identities are rejected" )
    {
        pcf::IndiProperty missing( pcf::IndiProperty::Number );
        missing.setDevice( "camwfs" );
        missing.setName( "fg_timing" );
        missing.add( pcf::IndiElement( "acq_jitter" ) );
        missing["acq_jitter"] = 1e-6;
        REQUIRE( app.setCallBack_m_indiP_camera( missing ) < 0 );
        packet.setDevice( "other_camera" );
        REQUIRE( app.setCallBack_m_indiP_camera( packet ) < 0 );
        packet.setDevice( "camwfs" );
        packet.setName( "fps" );
        REQUIRE( app.setCallBack_m_indiP_camera( packet ) < 0 );
        pcf::IndiProperty text( pcf::IndiProperty::Text );
        text.setDevice( "camwfs" );
        text.setName( "fg_timing" );
        text.add( pcf::IndiElement( "acq_fps" ) );
        text["acq_fps"] = "2000";
        REQUIRE( app.setCallBack_m_indiP_camera( text ) < 0 );
        REQUIRE( app.m_cameraTime.load() == acceptedTime );
    }
    SECTION( "configured camera property and element select the received measurement" )
    {
        app.setupConfig();
        char path[] = "/tmp/dm-synco-camera-XXXXXX";
        int  fd     = mkstemp( path );
        REQUIRE( fd >= 0 );
        close( fd );
        {
            std::ofstream settings( path );
            settings << "[tracking]\ncameraProperty=measurement\ncameraElement=rate\n";
        }
        app.config.readConfig( path );
        std::remove( path );
        REQUIRE( app.loadConfigImpl( app.config ) == 0 );
        REQUIRE( app.m_cameraProperty == "measurement" );
        REQUIRE( app.m_cameraElement == "rate" );
        REQUIRE( app.setCallBack_m_indiP_camera( packet ) < 0 );
        packet.setName( "measurement" );
        REQUIRE( app.setCallBack_m_indiP_camera( packet ) < 0 );
        packet.add( pcf::IndiElement( "rate" ) );
        packet["rate"] = 1750.0;
        REQUIRE( app.setCallBack_m_indiP_camera( packet ) == 0 );
        REQUIRE( app.m_cameraFps.load() == 1750 );
    }
}

/// Accept an INDI acquisition report while a timed cycle owns its scheduling mutex.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation camera callback does not wait for a timed cycle", "[dmSyncopation][camera][regression]" )
{
    // clang-format off
    #ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::setCallBack_m_indiP_camera(pcf::IndiProperty());
    #endif
    // clang-format on
    Harness app;
    Clock::app                          = nullptr;
    Clock::time                         = 1000000000;
    app.m_now                           = Clock::now;
    auto                         packet = cameraTiming( 2000 );
    std::unique_lock<std::mutex> cycle( app.m_cycleMutex );
    auto       response  = std::async( std::launch::async, [&]() { return app.setCallBack_m_indiP_camera( packet ); } );
    const auto completed = response.wait_for( std::chrono::milliseconds( 500 ) );
    cycle.unlock(); // Release even with the old blocking callback before checking or joining the worker.
    const int result = response.get();
    REQUIRE( completed == std::future_status::ready );
    REQUIRE( result == 0 );
    REQUIRE( app.m_cameraFps.load() == 2000 );
    REQUIRE( app.m_cameraTime.load() == 1000000100 );
    REQUIRE( app.m_cameraJitterUs == Approx( 1.234 ) );
    REQUIRE( app.m_cameraMinUs == Approx( 490 ) );
    REQUIRE( app.m_cameraMaxUs == Approx( 510 ) );
}

/// Keep a coherent cached acquisition report if its mailbox is busy and consume the next report without refreshing its
/// age.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation timed cycle never waits for the camera mailbox", "[dmSyncopation][camera][regression]" )
{
    // clang-format off
    #ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::readCamera();
    dmSyncopation::updatePeriod(0, 0, true);
    #endif
    // clang-format on
    Harness app;
    Clock::app  = nullptr;
    Clock::time = 1000000000;
    app.m_now   = Clock::now;
    REQUIRE( app.setCallBack_m_indiP_camera( cameraTiming( 2000 ) ) == 0 );
    app.readCamera();
    const auto cached = app.m_cameraReport;
    REQUIRE( cached.m_fps == 2000 );
    REQUIRE( cached.m_received == 1000000100 );
    REQUIRE( cached.m_jitterUs == Approx( 1.234 ) );
    REQUIRE( cached.m_minUs == Approx( 490 ) );
    REQUIRE( cached.m_maxUs == Approx( 510 ) );

    Clock::time          = 2000000000;
    auto packet          = cameraTiming( 1600 );
    packet["acq_jitter"] = 2e-6;
    packet["acq_min"]    = 600e-6;
    packet["acq_max"]    = 650e-6;
    REQUIRE( app.setCallBack_m_indiP_camera( packet ) == 0 );
    const int64_t                received = app.m_cameraTime.load();
    std::unique_lock<std::mutex> mailbox( app.m_cameraMutex );
    auto                         response  = std::async( std::launch::async,
                                [&]()
                                {
                                    std::lock_guard<std::mutex> cycle( app.m_cycleMutex );
                                    app.readCamera();
                                    app.updatePeriod( 2000000200, 0, true );
                                } );
    const auto                   completed = response.wait_for( std::chrono::milliseconds( 500 ) );
    mailbox.unlock(); // Release before assertions or joining so a blocking implementation cannot hang the suite.
    response.get();
    REQUIRE( completed == std::future_status::ready );
    REQUIRE( app.m_cameraReport.m_fps == cached.m_fps );
    REQUIRE( app.m_cameraReport.m_received == cached.m_received );
    REQUIRE( app.m_cameraReport.m_jitterUs == cached.m_jitterUs );
    REQUIRE( app.m_cameraReport.m_minUs == cached.m_minUs );
    REQUIRE( app.m_cameraReport.m_maxUs == cached.m_maxUs );
    REQUIRE( app.m_period == 500000 );
    REQUIRE( std::string( app.m_periodSource ) == "camera acquisition" );

    Clock::time += 1000000;
    app.readCamera();
    REQUIRE( app.m_cameraReport.m_fps == 1600 );
    REQUIRE( app.m_cameraReport.m_received == received );
    REQUIRE( app.m_cameraReport.m_received < Clock::time );
    REQUIRE( app.m_cameraReport.m_jitterUs == Approx( 2 ) );
    REQUIRE( app.m_cameraReport.m_minUs == Approx( 600 ) );
    REQUIRE( app.m_cameraReport.m_maxUs == Approx( 650 ) );
    app.updatePeriod( Clock::time, 0, true );
    REQUIRE( app.m_period == 625000 );
    REQUIRE( std::string( app.m_periodSource ) == "camera acquisition" );
}

/// Use the acquisition period as a stable reference without learning callback-to-callback phase jitter.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation camera period survives callback jitter", "[dmSyncopation][camera]" )
{
    // clang-format off
    #ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::processImage(nullptr, dev::shmimT());
    dmSyncopation::updatePeriod(0, 0, true);
    #endif
    // clang-format on
    Streams streams;
    Harness app;
    app.attach();
    Clock::connect( app );
    app.m_bounds    = synco::Bounds{};
    app.m_requested = 2;
    REQUIRE( app.setCallBack_m_indiP_camera( cameraTiming( 2000 ) ) == 0 );
    const std::array<int64_t, 4> jitter{ 0, 8000, 0, -8000 };
    for( int cycle = 0; cycle < 80; ++cycle )
    {
        INFO( "jitter cycle " << cycle );
        Clock::primary( 1000000000 + cycle * 500000 + jitter[cycle % jitter.size()] );
        REQUIRE( app.m_period == 500000 );
        REQUIRE( std::string( app.m_periodSource ) == "camera acquisition" );
    }
    REQUIRE( app.m_faults == 0 );
    REQUIRE( app.m_primed );
    REQUIRE( app.m_delivered > 60 );
    REQUIRE( app.m_bounds.early == 20000 );
    REQUIRE( app.m_bounds.observation == 5000 );
}

/// Reject an isolated late observation without replacing the reference period or creating a follow-on fault.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation isolated phase outlier does not become the next period", "[dmSyncopation][camera]" )
{
    // clang-format off
    #ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::processImage(nullptr, dev::shmimT());
    #endif
    // clang-format on
    Streams streams;
    Harness app;
    app.attach();
    Clock::connect( app );
    app.m_bounds      = synco::Bounds{};
    app.m_requested   = 2;
    const bool camera = GENERATE( false, true );
    if( camera )
        REQUIRE( app.setCallBack_m_indiP_camera( cameraTiming( 2000 ) ) == 0 );
    for( int cycle = 0; cycle < 40; ++cycle )
        Clock::primary( 1000000000 + cycle * 500000 );
    REQUIRE( app.m_primed );
    REQUIRE( app.m_faults == 0 );
    const int64_t late = 1000000000 + 39 * 500000 + 525000;
    Clock::primary( late );
    REQUIRE( app.m_faults == 1 );
    REQUIRE( app.m_period == 500000 );
    REQUIRE( app.m_faultInterval == 525000 );
    REQUIRE( app.m_faultPeriod == 500000 );
    REQUIRE( app.m_faultInterval - app.m_faultPeriod == 25000 );
    REQUIRE_FALSE( app.m_primed );
    for( int cycle = 1; cycle <= 8; ++cycle )
    {
        Clock::primary( late + cycle * 500000 );
        REQUIRE( app.m_period == 500000 );
        REQUIRE( app.m_faults == 1 );
    }
    REQUIRE( app.m_primed );
    REQUIRE( app.m_scheduleState == synco::armed );
    REQUIRE( app.m_faultInterval == 525000 );
    REQUIRE( app.m_faultPeriod == 500000 );
}

/// Attribute a faulting callback interval to independent publication timing and preserve the same-event diagnostic
/// latch.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation separates primary publication and observation intervals", "[dmSyncopation][diagnostics]" )
{
    // clang-format off
    #ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::processImage(nullptr, dev::shmimT());
    dmSyncopation::readPrimaryTiming(0, 0, true);
    #endif
    // clang-format on
    Streams streams;
    Harness app;
    app.attach();
    Clock::connect( app );
    app.m_requested  = 2;
    app.m_lockFrames = 100;
    REQUIRE( app.setCallBack_m_indiP_camera( cameraTiming( 2000, false ) ) == 0 );
    const int64_t observed = Clock::time + 1000000;
    const int64_t wall     = 1700000000123456789LL;
    primaryObservation( app, observed, wall );
    REQUIRE( app.m_primaryTiming.m_publication == -1 );
    REQUIRE( app.m_primaryTiming.m_previousRuntime == -1 );
    REQUIRE( app.m_primaryTiming.m_gap == -1 );
    primaryObservation( app, observed + 500000, wall + 500000 );
    REQUIRE( app.m_primaryTiming.m_publication == 500000 );
    const auto previousRuntime = app.m_previousCallbackRuntime;
    const auto previousEnd     = app.m_previousCallbackEnd;

    const int64_t publicationInterval =
        GENERATE( int64_t( 500000 ), int64_t( 555000 ), int64_t( 600000 ), int64_t( 0 ) );
    const int64_t faultPublication = publicationInterval ? wall + 500000 + publicationInterval : 0;
    primaryObservation( app, observed + 1055000, faultPublication );
    REQUIRE( app.m_faults == 1 );
    REQUIRE( app.m_faultInterval == 555000 );
    REQUIRE( app.m_faultPeriod == 500000 );
    REQUIRE( app.m_faultTiming.m_counter == app.m_primaryCounter );
    REQUIRE( app.m_faultTiming.m_publication == ( publicationInterval ? publicationInterval : -1 ) );
    if( publicationInterval )
        REQUIRE( app.m_faultInterval - app.m_faultTiming.m_publication == 555000 - publicationInterval );
    REQUIRE( app.m_faultTiming.m_previousRuntime == previousRuntime );
    REQUIRE( app.m_faultTiming.m_gap == observed + 1055100 - previousEnd );
    REQUIRE( app.m_faultTiming.m_sampleDelay >= 0 );
    REQUIRE( app.m_faultTiming.m_sampleDelay < 10000 );
    const auto fault = app.m_faultTiming;
    for( int miss = 0; miss < 3; ++miss )
        app.missedStep();
    app.m_lockFrames = 2;
    for( int cycle = 1; cycle <= 6; ++cycle )
    {
        primaryObservation( app,
                            observed + 1055000 + cycle * 500000,
                            wall + 500000 + ( publicationInterval ? publicationInterval : 500000 ) + cycle * 500000 );
        REQUIRE( app.m_faultTiming.m_counter == fault.m_counter );
        REQUIRE( app.m_faultTiming.m_publication == fault.m_publication );
        REQUIRE( app.m_faultTiming.m_previousRuntime == fault.m_previousRuntime );
        REQUIRE( app.m_faultTiming.m_gap == fault.m_gap );
        REQUIRE( app.m_faultTiming.m_sampleDelay == fault.m_sampleDelay );
    }
    REQUIRE( app.m_primed );
    REQUIRE( app.m_faults == 1 );

    streams.images[2].md->write = 1;
    REQUIRE( app.processImage( nullptr, dev::shmimT{} ) == 0 );
    streams.images[2].md->write = 0;
    REQUIRE( app.m_faultCounts[size_t( Harness::Fault::resource )] == 1 );
    REQUIRE( app.m_faultTiming.m_counter == 0 );
    REQUIRE( app.m_faultTiming.m_publication == -1 );
    REQUIRE( app.m_faultTiming.m_previousRuntime == -1 );
    REQUIRE( app.m_faultTiming.m_gap == -1 );
    REQUIRE( app.m_faultTiming.m_sampleDelay == -1 );
}

/// Reject invalid timestamps and publications changed inside the metadata bracket without bridging the invalid sample.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation rejects incoherent primary publication metadata", "[dmSyncopation][diagnostics]" )
{
    // clang-format off
    #ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::readPrimaryTiming(0, 0, true);
    #endif
    // clang-format on
    Streams streams;
    Harness app;
    app.attach();
    Clock::connect( app );
    app.m_enabled          = false;
    const int64_t observed = Clock::time + 1000000;
    const int64_t wall     = 1700000000000000000LL;
    primaryObservation( app, observed, wall );
    primaryObservation( app, observed + 500000, wall + 500000 );
    REQUIRE( app.m_primaryTiming.m_publication == 500000 );

    PrimaryProbe::reset();
    PrimaryProbe::time  = Clock::time + 10000;
    PrimaryProbe::image = &app.m_imageStream;
    app.m_now           = PrimaryProbe::now;
    auto *metadata      = app.m_imageStream.md;
    metadata->cnt0 += 1;
    metadata->writetime = synco::toTimespec( wall + 1000000 );
    const auto expected = metadata->cnt0;
    SECTION( "a writer is still publishing" )
    {
        metadata->write = 1;
    }
    SECTION( "the supplied observation counter no longer matches" )
    {
        ++metadata->cnt0;
    }
    SECTION( "the timestamp is absent" )
    {
        metadata->writetime = { 0, 0 };
    }
    SECTION( "negative seconds are invalid" )
    {
        metadata->writetime.tv_sec = -1;
    }
    SECTION( "negative nanoseconds are invalid" )
    {
        metadata->writetime.tv_nsec = -1;
    }
    SECTION( "nanoseconds must be normalized" )
    {
        metadata->writetime.tv_nsec = 1000000000;
    }
    SECTION( "nanosecond conversion must fit a signed diagnostic value" )
    {
        metadata->writetime.tv_sec = INT64_MAX / 1000000000 + 1;
    }
    SECTION( "write begins after the timestamp was copied" )
    {
        PrimaryProbe::tear = 1;
    }
    SECTION( "the counter changes after the timestamp was copied" )
    {
        PrimaryProbe::tear = 2;
    }
    app.readPrimaryTiming( PrimaryProbe::time.load(), expected, true );
    REQUIRE( app.m_primaryTiming.m_counter == expected );
    REQUIRE( app.m_primaryTiming.m_publication == -1 );
    REQUIRE( app.m_previousPrimaryPublication == 0 );
    REQUIRE( app.m_primaryTiming.m_sampleDelay >= 0 );

    metadata->write      = 0;
    app.m_primaryCounter = metadata->cnt0;
    ++metadata->cnt0;
    metadata->writetime = synco::toTimespec( wall + 1500000 );
    app.readPrimaryTiming( PrimaryProbe::time.load(), metadata->cnt0, true );
    REQUIRE( app.m_primaryTiming.m_publication == -1 );
    app.m_primaryCounter = metadata->cnt0;
    ++metadata->cnt0;
    metadata->writetime = synco::toTimespec( wall + 2000000 );
    app.readPrimaryTiming( PrimaryProbe::time.load(), metadata->cnt0, true );
    REQUIRE( app.m_primaryTiming.m_publication == 500000 );
    PrimaryProbe::reset();
}

/// Start new publication baselines after counter discontinuities or a resource epoch change.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation never associates publication intervals across primary discontinuities",
           "[dmSyncopation][diagnostics]" )
{
    // clang-format off
    #ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::processImage(nullptr, dev::shmimT());
    dmSyncopation::allocate(dev::shmimT());
    #endif
    // clang-format on
    Streams streams;
    Harness app;
    app.attach();
    Clock::connect( app );
    app.m_enabled          = false;
    const int64_t observed = Clock::time + 1000000;
    const int64_t wall     = 1700000000000000000LL;
    primaryObservation( app, observed, wall );
    primaryObservation( app, observed + 500000, wall + 500000 );
    REQUIRE( app.m_primaryTiming.m_publication == 500000 );
    SECTION( "a skipped counter adopts a new stable publication baseline" )
    {
        primaryObservation( app, observed + 2000000, wall + 2000000, 3 );
        REQUIRE( app.m_primaryTiming.m_publication == -1 );
        REQUIRE( app.m_previousPrimaryPublication == wall + 2000000 );
        primaryObservation( app, observed + 2500000, wall + 2500000 );
        REQUIRE( app.m_primaryTiming.m_publication == 500000 );
    }
    SECTION( "a reset counter cannot preserve the former timestamp association" )
    {
        app.m_imageStream.md->cnt0        = 0;
        app.m_combined->image()->md->cnt0 = 0;
        primaryObservation( app, observed + 1000000, wall + 1000000 );
        REQUIRE( app.m_primaryTiming.m_publication == -1 );
        REQUIRE( app.m_previousPrimaryPublication == wall + 1000000 );
        primaryObservation( app, observed + 1500000, wall + 1500000 );
        REQUIRE( app.m_primaryTiming.m_publication == 500000 );
    }
    SECTION( "unsigned wrap is a discontinuity even though adding one wraps too" )
    {
        app.m_primaryCounter       = UINT64_MAX;
        app.m_imageStream.md->cnt0 = UINT64_MAX;
        primaryObservation( app, observed + 1000000, wall + 1000000 );
        REQUIRE( app.m_primaryTiming.m_counter == 0 );
        REQUIRE( app.m_primaryTiming.m_publication == -1 );
        primaryObservation( app, observed + 1500000, wall + 1500000 );
        REQUIRE( app.m_primaryTiming.m_publication == 500000 );
    }
    SECTION( "equal or backwards timestamps establish a new baseline without a usable interval" )
    {
        const int64_t adjustment = GENERATE( int64_t( 0 ), int64_t( -500000 ) );
        const int64_t restarted  = wall + 500000 + adjustment;
        primaryObservation( app, observed + 1000000, restarted );
        REQUIRE( app.m_primaryTiming.m_publication == -1 );
        REQUIRE( app.m_previousPrimaryPublication == restarted );
        primaryObservation( app, observed + 1500000, restarted + 500000 );
        REQUIRE( app.m_primaryTiming.m_publication == 500000 );
    }
    SECTION( "reallocation discards callback and publication baselines" )
    {
        REQUIRE( app.allocate( dev::shmimT{} ) == 0 );
        REQUIRE( app.m_previousPrimaryPublication == 0 );
        REQUIRE( app.m_previousCallbackRuntime == -1 );
        REQUIRE( app.m_previousCallbackEnd == 0 );
        primaryObservation( app, observed + 1000000, wall + 1000000 );
        REQUIRE( app.m_primaryTiming.m_publication == -1 );
        REQUIRE( app.m_primaryTiming.m_previousRuntime == -1 );
        REQUIRE( app.m_primaryTiming.m_gap == -1 );
    }
    SECTION( "a replaced resource breaks the old epoch before reattachment" )
    {
        char filename[PATH_MAX];
        ImageStreamIO_filename( filename, sizeof( filename ), "synco_test_timing" );
        const std::string backup = std::string( filename ) + ".old";
        REQUIRE( rename( filename, backup.c_str() ) == 0 );
        {
            std::ofstream replacement( filename );
            replacement << "replacement";
        }
        REQUIRE( app.processImage( nullptr, dev::shmimT{} ) == 0 );
        REQUIRE_FALSE( app.m_resources );
        REQUIRE( app.m_previousPrimaryPublication == 0 );
        REQUIRE( std::remove( filename ) == 0 );
        REQUIRE( rename( backup.c_str(), filename ) == 0 );
        primaryObservation( app, observed + 1000000, wall + 1000000 );
        REQUIRE( app.m_resources );
        REQUIRE( app.m_primaryTiming.m_publication == -1 );
        primaryObservation( app, observed + 1500000, wall + 1500000 );
        REQUIRE( app.m_primaryTiming.m_publication == 500000 );
    }
}

/// Record every completed callback body while ignoring duplicate posts as publication observations.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation records callback completion across early and scheduled returns",
           "[dmSyncopation][diagnostics]" )
{
    // clang-format off
    #ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::processImage(nullptr, dev::shmimT());
    dmSyncopation::processPrimary(0);
    #endif
    // clang-format on
    Streams streams;
    Harness app;
    app.attach();
    Clock::connect( app );
    app.m_requested        = 2;
    app.m_enabled          = false;
    const int64_t observed = Clock::time + 1000000;
    const int64_t wall     = 1700000000000000000LL;
    primaryObservation( app, observed, wall );
    REQUIRE( app.m_previousCallbackRuntime == app.m_previousCallbackEnd - observed - 100 );
    REQUIRE( app.m_previousCallbackRuntime > 0 );
    const auto sample = app.m_primaryTiming;
    const auto stamp  = app.m_previousPrimaryPublication;

    Clock::time += 10000;
    const int64_t duplicateEntry = Clock::time + 100;
    REQUIRE( app.processImage( nullptr, dev::shmimT{} ) == 0 );
    REQUIRE( app.m_previousCallbackRuntime == app.m_previousCallbackEnd - duplicateEntry );
    REQUIRE( app.m_previousCallbackRuntime > 0 );
    REQUIRE( app.m_primaryTiming.m_counter == sample.m_counter );
    REQUIRE( app.m_primaryTiming.m_publication == sample.m_publication );
    REQUIRE( app.m_primaryTiming.m_previousRuntime == sample.m_previousRuntime );
    REQUIRE( app.m_primaryTiming.m_gap == sample.m_gap );
    REQUIRE( app.m_primaryTiming.m_sampleDelay == sample.m_sampleDelay );
    REQUIRE( app.m_previousPrimaryPublication == stamp );
    auto previousRuntime = app.m_previousCallbackRuntime;
    auto previousEnd     = app.m_previousCallbackEnd;

    app.m_enabled = true;
    for( int cycle = 1; cycle <= 5; ++cycle )
    {
        const int64_t entry = observed + cycle * 500000 + 100;
        primaryObservation( app, entry - 100, wall + cycle * 500000 );
        REQUIRE( app.m_primaryTiming.m_previousRuntime == previousRuntime );
        REQUIRE( app.m_primaryTiming.m_gap == entry - previousEnd );
        REQUIRE( app.m_previousCallbackRuntime == app.m_previousCallbackEnd - entry );
        REQUIRE( app.m_previousCallbackRuntime > 0 );
        if( cycle == 1 )
            REQUIRE( app.m_previousCallbackRuntime >= 50000 );
        if( cycle >= 3 )
            REQUIRE( app.m_previousCallbackRuntime > 400000 );
        previousRuntime = app.m_previousCallbackRuntime;
        previousEnd     = app.m_previousCallbackEnd;
    }
    REQUIRE( app.m_primed );
    REQUIRE( app.m_delivered >= 2 );
    REQUIRE( app.m_faults == 0 );
}

/// Include time waiting for the cycle mutex in both callback runtime and entry-to-metadata delay.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation callback diagnostics include cycle lock delay", "[dmSyncopation][diagnostics]" )
{
    // clang-format off
    #ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::processImage(nullptr, dev::shmimT());
    #endif
    // clang-format on
    Streams streams;
    Harness app;
    app.attach();
    app.m_enabled = false;
    PrimaryProbe::reset();
    app.m_now = PrimaryProbe::now;
    ++app.m_imageStream.md->cnt0;
    app.m_imageStream.md->writetime = { 1700000000, 0 };
    std::promise<void> started;
    auto               entered = started.get_future();
    PrimaryProbe::entered      = &started;
    std::unique_lock<std::mutex> cycle( app.m_cycleMutex );
    auto response = std::async( std::launch::async, [&]() { return app.processImage( nullptr, dev::shmimT{} ); } );
    const auto captured = entered.wait_for( std::chrono::milliseconds( 500 ) );
    PrimaryProbe::time.fetch_add( 55000 );
    cycle.unlock(); // Release before checking the entry notification or joining the worker.
    const int result = response.get();
    REQUIRE( captured == std::future_status::ready );
    REQUIRE( result == 0 );
    REQUIRE( app.m_previousCallbackRuntime >= 55000 );
    REQUIRE( app.m_previousCallbackRuntime == app.m_previousCallbackEnd - 1000000100 );
    REQUIRE( app.m_primaryTiming.m_sampleDelay >= 55000 );
    REQUIRE( app.m_primaryTiming.m_previousRuntime == -1 );
    REQUIRE( app.m_primaryTiming.m_gap == -1 );
    PrimaryProbe::reset();
}

/// Qualify real primary-rate changes across phase resets while excluding missed counters and incoherent windows.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation fallback qualifies a coherent primary period", "[dmSyncopation][camera]" )
{
    // clang-format off
    #ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::processImage(nullptr, dev::shmimT());
    dmSyncopation::updatePeriod(0, 0, true);
    #endif
    // clang-format on
    Streams streams;
    Harness app;
    app.attach();
    Clock::connect( app );
    app.m_bounds    = synco::Bounds{};
    app.m_requested = 2;
    SECTION( "a sustained genuine rate change eventually rearms at its qualified mean" )
    {
        for( int cycle = 0; cycle < 40; ++cycle )
            Clock::primary( 1000000000 + cycle * 500000 );
        REQUIRE( app.m_period == 500000 );
        REQUIRE( std::string( app.m_periodSource ) == "primary window" );
        const int64_t last = 1000000000 + 39 * 500000;
        for( unsigned cycle = 1; cycle <= app.m_rateFrames + 8; ++cycle )
        {
            Clock::primary( last + cycle * 600000 );
            if( cycle < app.m_rateFrames )
                REQUIRE( app.m_period == 500000 );
        }
        REQUIRE( app.m_period == 600000 );
        REQUIRE( app.m_measuredPeriod == 600000 );
        REQUIRE( std::string( app.m_periodSource ) == "primary window" );
        REQUIRE( app.m_primed );
        REQUIRE( app.m_scheduleState == synco::armed );
        REQUIRE( app.m_bounds.early == 20000 );
    }
    SECTION( "counter gaps cannot qualify their longer observation interval as a slow loop" )
    {
        for( unsigned cycle = 0; cycle < app.m_rateFrames + 5; ++cycle )
        {
            Clock::primary( 1000000000 + cycle * 1000000, 2 );
            REQUIRE( app.m_period == 500000 );
            REQUIRE( app.m_measuredPeriod == 0 );
            REQUIRE( std::string( app.m_periodSource ) == "nominal" );
        }
        REQUIRE_FALSE( app.m_primed );
        REQUIRE( app.m_delivered == 0 );
    }
    SECTION( "a wide interval range cannot qualify until a complete coherent window follows" )
    {
        app.m_rateFrames = 4;
        int64_t observed = Clock::time;
        for( int interval : { 600000, 650000, 600000, 650000, 600000, 650000 } )
        {
            observed += interval;
            app.updatePeriod( observed, interval, true );
            REQUIRE( app.m_measuredPeriod == 0 );
            REQUIRE( app.m_period == 500000 );
        }
        for( int cycle = 0; cycle < 4; ++cycle )
        {
            observed += 600000;
            app.updatePeriod( observed, 600000, true );
        }
        REQUIRE( app.m_measuredPeriod == 600000 );
        REQUIRE( app.m_period == 600000 );
    }
}

/// Expire camera timing at five seconds and reacquire with the qualified primary or nominal reference.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation expires camera reports without relaxing timing bounds", "[dmSyncopation][camera]" )
{
    // clang-format off
    #ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::updatePeriod(0, 0, true);
    #endif
    // clang-format on
    Streams streams;
    Harness app;
    app.attach();
    Clock::connect( app );
    app.m_rateFrames = 4;
    REQUIRE( app.setCallBack_m_indiP_camera( cameraTiming( 2500 ) ) == 0 );
    const int64_t receipt = app.m_cameraTime.load();
    app.updatePeriod( receipt, 0, true );
    REQUIRE( app.m_period == 400000 );
    REQUIRE( std::string( app.m_periodSource ) == "camera acquisition" );
    const auto bounds = app.m_bounds;
    SECTION( "a future receipt is not a valid reference on this monotonic timeline" )
    {
        Clock::time = receipt - 101;
        app.updatePeriod( receipt - 1, 0, true );
        REQUIRE( app.m_period == 500000 );
        REQUIRE( std::string( app.m_periodSource ) == "nominal" );
    }
    SECTION( "a report received after callback entry is still current under the cycle lock" )
    {
        Clock::time = receipt + 1000;
        app.updatePeriod( receipt - 1000, 0, true );
        REQUIRE( app.m_period == 400000 );
        REQUIRE( std::string( app.m_periodSource ) == "camera acquisition" );
    }
    SECTION( "a current camera report wins until expiry then falls back to nominal" )
    {
        Clock::time = receipt + 4999999899;
        app.updatePeriod( receipt + 4999999999, 0, true );
        REQUIRE( app.m_period == 400000 );
        app.m_primed        = true;
        app.m_scheduleState = synco::armed;
        app.m_lastMain      = receipt + 4999999999;
        Clock::time         = receipt + 4999999900;
        app.updatePeriod( receipt + 5000000000, 0, true );
        REQUIRE( app.m_period == 500000 );
        REQUIRE( std::string( app.m_periodSource ) == "nominal" );
        REQUIRE_FALSE( app.m_primed );
        REQUIRE( app.m_scheduleState != synco::armed );
        REQUIRE( app.m_lastMain == 0 );
        REQUIRE( app.m_faults == 0 );
    }
    SECTION( "an expired report uses an already qualified primary window" )
    {
        for( int cycle = 1; cycle <= 4; ++cycle )
            app.updatePeriod( receipt + cycle * 600000, 600000, true );
        REQUIRE( app.m_measuredPeriod == 600000 );
        REQUIRE( app.m_period == 400000 );
        app.m_primed        = true;
        app.m_scheduleState = synco::armed;
        Clock::time         = receipt + 4999999901;
        app.updatePeriod( receipt + 5000000001, 600000, true );
        REQUIRE( app.m_period == 600000 );
        REQUIRE( std::string( app.m_periodSource ) == "primary window" );
        REQUIRE_FALSE( app.m_primed );
        REQUIRE( app.m_faults == 0 );
    }
    REQUIRE( app.m_bounds.early == bounds.early );
    REQUIRE( app.m_bounds.observation == bounds.observation );
    REQUIRE( app.m_bounds.delivery == bounds.delivery );
    REQUIRE( app.m_bounds.keepout == bounds.keepout );
    REQUIRE( app.m_bounds.guard == bounds.guard );
}

/// Validate actual INDI callback behavior, including atomic updates and disabling before publication.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation INDI callbacks enforce disabled settings", "[dmSyncopation]" )
{
    // clang-format off
    #ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::newCallBack_m_indiP_settings(pcf::IndiProperty());
    dmSyncopation::newCallBack_m_indiP_enable(pcf::IndiProperty());
    dmSyncopation::setCallBack_m_indiP_camera(pcf::IndiProperty());
    #endif
    // clang-format on
    Harness app;
    app.properties();
    pcf::IndiProperty settings( pcf::IndiProperty::Number );
    settings.setDevice( "synco_test" );
    settings.setName( "settings" );
    settings.add( pcf::IndiElement( "steps" ) );
    settings["steps"] = 2.0;
    REQUIRE( app.newCallBack_m_indiP_settings( settings ) == 0 );
    REQUIRE( app.m_requested == 2 );
    settings["steps"] = 2.5;
    REQUIRE( app.newCallBack_m_indiP_settings( settings ) < 0 );
    REQUIRE( app.m_requested == 2 );
    settings["steps"] = std::numeric_limits<double>::quiet_NaN();
    REQUIRE( app.newCallBack_m_indiP_settings( settings ) < 0 );
    settings["steps"] = 3.0;
    app.m_enabled     = true;
    REQUIRE( app.newCallBack_m_indiP_settings( settings ) < 0 );
    REQUIRE( app.m_requested == 2 );
    auto enable      = app.m_indiP_enable;
    enable["toggle"] = pcf::IndiElement::Off;
    REQUIRE( app.newCallBack_m_indiP_enable( enable ) == 0 );
    REQUIRE_FALSE( app.m_enabled );
    REQUIRE( app.m_scheduleState == synco::invalid );
    settings.setDevice( "different" );
    REQUIRE( app.newCallBack_m_indiP_settings( settings ) < 0 );
    pcf::IndiProperty fps( pcf::IndiProperty::Number );
    fps.setDevice( "camwfs" );
    fps.setName( "fg_timing" );
    fps.add( pcf::IndiElement( "acq_fps" ) );
    fps["acq_fps"] = 2000.08;
    REQUIRE( app.setCallBack_m_indiP_camera( fps ) == 0 );
    REQUIRE( app.m_cameraFps.load() == Approx( 2000.08 ) );
    fps["acq_fps"] = std::numeric_limits<double>::infinity();
    REQUIRE( app.setCallBack_m_indiP_camera( fps ) < 0 );
}

} // namespace dmSyncopationTest
} // namespace libXWCTest
