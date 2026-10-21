/** \file dmSyncopation_test.cpp
 * \brief Deterministic timing/control tests with isolated workstation-local ImageStreamIO streams.
 * \ingroup dmSyncopation_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "../dmSyncopation.hpp"

#include <filesystem>
#include <fstream>
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
    using dmSyncopation::m_cameraFps;
    using dmSyncopation::m_clean;
    using dmSyncopation::m_combined;
    using dmSyncopation::m_consumed;
    using dmSyncopation::m_copiedLate;
    using dmSyncopation::m_copyMax;
    using dmSyncopation::m_copyStarted;
    using dmSyncopation::m_delivered;
    using dmSyncopation::m_destinations;
    using dmSyncopation::m_enabled;
    using dmSyncopation::m_epoch;
    using dmSyncopation::m_faultCounts;
    using dmSyncopation::m_faults;
    using dmSyncopation::m_indiP_camera;
    using dmSyncopation::m_indiP_enable;
    using dmSyncopation::m_indiP_reacquire;
    using dmSyncopation::m_indiP_settings;
    using dmSyncopation::m_lastFaultNs;
    using dmSyncopation::m_lastFaultReason;
    using dmSyncopation::m_lastMain;
    using dmSyncopation::m_lockFrames;
    using dmSyncopation::m_mappings;
    using dmSyncopation::m_missed;
    using dmSyncopation::m_notStarted;
    using dmSyncopation::m_now;
    using dmSyncopation::m_period;
    using dmSyncopation::m_primaryCounter;
    using dmSyncopation::m_primed;
    using dmSyncopation::m_reason;
    using dmSyncopation::m_record;
    using dmSyncopation::m_requested;
    using dmSyncopation::m_resources;
    using dmSyncopation::m_scheduleState;
    using dmSyncopation::m_submission;
    using dmSyncopation::m_timing;
    using dmSyncopation::m_wait;
    using dmSyncopation::missedStep;
    using dmSyncopation::snapshot;
    using dmSyncopation::waitUntil;
    using MagAOXApp<true>::config;
    using dev::shmimMonitor<dmSyncopation>::m_imageStream;

    /// Set a stable test instance identity.
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
    m_configName = "synco_test";
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
    fps.setName( "fps" );
    fps.add( pcf::IndiElement( "current" ) );
    fps["current"] = 2000.08;
    REQUIRE( app.setCallBack_m_indiP_camera( fps ) == 0 );
    REQUIRE( app.m_cameraFps.load() == Approx( 2000.08 ) );
    fps["current"] = std::numeric_limits<double>::infinity();
    REQUIRE( app.setCallBack_m_indiP_camera( fps ) < 0 );
}

} // namespace dmSyncopationTest
} // namespace libXWCTest
