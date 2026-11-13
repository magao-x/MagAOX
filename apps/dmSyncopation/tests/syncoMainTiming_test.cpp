/** \file syncoMainTiming_test.cpp
 * \brief Offline qualified-main capture, passive snapshots and automatic mode-transition tests.
 * \ingroup dmSyncopation_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "../dmSyncopation.hpp"

namespace libXWCTest
{
namespace dmSyncopationTest
{

using MainTiming = MagAOX::app::synco::MainTiming;
using Kind       = MagAOX::dmTimingEvents::Kind;
using Record     = MagAOX::dmTimingEvents::Record;

/// \cond MAIN_TIMING_TEST_HELPERS
/// Plain local metadata only: no stream creation, semaphore ownership or instrument connection.
struct MainImages
{
    IMAGE_METADATA    primaryMetadata{}, combinedMetadata{}; ///< Fixture-owned borrowed metadata.
    STREAM_PROC_TRACE trace{};                               ///< Fixture-owned coherent producer trace.
    IMAGE             primary{}, combined{};                 ///< Non-owning handles passed only to snapshot readers.

    /// Bind local image handles and initialize valid distinct publisher incarnations.
    MainImages();

    /// Set one immutable primary publication.
    void main( uint64_t counter /**< [in] Primary publication counter. */ );

    /// Publish a local output snapshot linked to a selected primary publication.
    void output( uint64_t counter /**< [in] Combined publication counter. */,
                 uint64_t primaryCounter /**< [in] Primary counter sampled when publishing this output. */ );

    /// Return the exact fixture timestamp for a primary publication.
    static int64_t stamp( uint64_t counter /**< [in] Primary counter. */ );
};

MainImages::MainImages()
{
    primary.md                   = &primaryMetadata;
    combined.md                  = &combinedMetadata;
    combined.streamproctrace     = &trace;
    primaryMetadata.inode        = 11;
    combinedMetadata.inode       = 22;
    combinedMetadata.NBproctrace = 1;
    main( 1 );
    output( 1, 1 );
}

int64_t MainImages::stamp( uint64_t counter )
{
    return 1000000000 + int64_t( counter ) * 1000000;
}

void MainImages::main( uint64_t counter )
{
    primaryMetadata.cnt0      = counter;
    primaryMetadata.writetime = MagAOX::app::synco::toTimespec( stamp( counter ) );
}

void MainImages::output( uint64_t counter, uint64_t primaryCounter )
{
    combinedMetadata.cnt0      = counter;
    combinedMetadata.writetime = MagAOX::app::synco::toTimespec( stamp( primaryCounter ) + 20000 );
    trace.trigger_inode        = primaryMetadata.inode;
    trace.cnt0                 = primaryCounter;
    trace.procwrite_PID        = 123;
    trace.ts_procstart         = MagAOX::app::synco::toTimespec( stamp( primaryCounter ) + 10000 );
    trace.ts_streamupdate      = MagAOX::app::synco::toTimespec( stamp( primaryCounter ) + 15000 );
}

/// Expose actual mode callbacks and passive capture with borrowed fixture images and no started threads.
class MainHarness : public MagAOX::app::dmSyncopation
{
  public:
    using dmSyncopation::m_enabled;
    using dmSyncopation::m_generation;
    using dmSyncopation::m_mainTiming;
    using dmSyncopation::m_zeroEventStride;
    using dmSyncopation::m_zeroStats;
    using dmSyncopation::publishZeroTestStatus;

    /// Bind only local metadata and configure the preallocated capture handoff.
    explicit MainHarness( MainImages &images /**< [in,out] Metadata whose lifetime encloses this harness. */ );

    /// Unbind local metadata before production base cleanup.
    ~MainHarness() noexcept;

    /// Invoke the real enable callback with its configured property name.
    int enable( bool value /**< [in] Requested scheduler state. */ );

    /// Invoke the exact primary-entry diagnostic hook under its production mutex.
    void observe();
};

MainHarness::MainHarness( MainImages &images )
{
    m_configName        = "local-main-timing";
    m_productionEnabled = true;
    m_requested         = 4;
    m_mainTimingName    = "unpublished-local-main-timing";
    m_mainTiming.configure( 64 );
    MagAOX::app::dev::shmimMonitor<MagAOX::app::dmSyncopation>::m_imageStream = images.primary;
    m_mainTimingCombined                                                      = images.combined;
    createStandardIndiToggleSw( m_indiP_enable, "enable" );
}

MainHarness::~MainHarness() noexcept
{
    MagAOX::app::dev::shmimMonitor<MagAOX::app::dmSyncopation>::m_imageStream = {};
    m_mainTimingCombined                                                      = {};
}

int MainHarness::enable( bool value )
{
    pcf::IndiProperty property( pcf::IndiProperty::Switch );
    property.setDevice( m_configName );
    property.setName( "enable" );
    property.add( pcf::IndiElement( "toggle" ) );
    property["toggle"].setSwitchState( value ? pcf::IndiElement::On : pcf::IndiElement::Off );
    return newCallBack_m_indiP_enable( property );
}

void MainHarness::observe()
{
    std::lock_guard<std::mutex> lock( m_cycleMutex );
    capturePassiveMainTiming();
}
/// \endcond

/** \brief Export both edges and retain a frozen generation through rapid mode changes without primary callbacks.
 * \ingroup dmSyncopation_unit_test
 */
TEST_CASE( "main timing mode markers include initial off and every transition", "[dmSyncopation][mainTiming]" )
{
    MainImages  images;
    MainHarness app( images );
    app.observe();
    Record record;
    REQUIRE( app.m_mainTiming.pop( record ) );
    CHECK( record.kind == Kind::mode );
    CHECK_FALSE( record.enabled );
    CHECK( record.steps == 4 );
    auto previous = record.generation;

    for( bool enabled : { true, false, true, false } )
    {
        REQUIRE( app.enable( enabled ) == 0 );
        REQUIRE( app.m_mainTiming.pop( record ) );
        CHECK( record.kind == Kind::mode );
        CHECK( record.enabled == enabled );
        CHECK( record.generation > previous );
        CHECK( MagAOX::dmTimingEvents::valid( record ) );
        previous = record.generation;
    }

    REQUIRE( app.enable( false ) == 0 );
    CHECK_FALSE( app.m_mainTiming.pop( record ) );
    REQUIRE( app.enable( true ) == 0 );
    REQUIRE( app.m_mainTiming.pop( record ) );
    previous      = record.generation;
    app.m_enabled = false; // An internal fault/timeout, with no external INDI request.
    app.publishZeroTestStatus( 1000000000, true );
    REQUIRE( app.m_mainTiming.pop( record ) );
    CHECK_FALSE( record.enabled );
    CHECK( record.generation > previous );

    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    MagAOX::app::dmSyncopation::newCallBack_m_indiP_enable;
    MagAOX::app::dmSyncopation::publishZeroTestStatus;
    MagAOX::app::dmSyncopation::capturePassiveMainTiming;
#endif
    // clang-format on
}

/** \brief Classify stable disabled mains without scheduler resources, command writes or sampled event logging.
 * \ingroup dmSyncopation_unit_test
 */
TEST_CASE( "main timing passively classifies disabled primary publications", "[dmSyncopation][mainTiming]" )
{
    MainImages  images;
    MainHarness app( images );
    app.m_zeroEventStride = GENERATE( 1, 1024 );
    app.observe();
    Record record;
    REQUIRE( app.m_mainTiming.pop( record ) );
    REQUIRE_FALSE( app.m_mainTiming.pop( record ) );
    images.main( 2 );
    images.output( 2, 2 );
    const auto beforePrimary  = images.primaryMetadata;
    const auto beforeCombined = images.combinedMetadata;
    const auto beforeTrace    = images.trace;
    app.observe();
    REQUIRE( app.m_mainTiming.pop( record ) );
    CHECK( record.kind == Kind::main );
    CHECK_FALSE( record.enabled );
    CHECK( record.primaryInode == 11 );
    CHECK( record.primaryCounter == 2 );
    CHECK( record.key.inode == 22 );
    CHECK( record.key.counter == 2 );
    CHECK( MagAOX::dmTimingEvents::valid( record ) );
    CHECK( std::memcmp( &beforePrimary, &images.primaryMetadata, sizeof( beforePrimary ) ) == 0 );
    CHECK( std::memcmp( &beforeCombined, &images.combinedMetadata, sizeof( beforeCombined ) ) == 0 );
    CHECK( std::memcmp( &beforeTrace, &images.trace, sizeof( beforeTrace ) ) == 0 );
    CHECK( app.m_zeroStats.posted == 0 );
    CHECK( app.m_zeroStats.staged == 0 );
    app.observe();
    CHECK_FALSE( app.m_mainTiming.pop( record ) );

    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    MagAOX::app::dmSyncopation::capturePassiveMainTiming;
    MagAOX::app::synco::MainTiming::passive;
#endif
    // clang-format on
}

/** \brief A main completing after its callback can be qualified at the next observation without polling or waiting.
 * \ingroup dmSyncopation_unit_test
 */
TEST_CASE( "main timing retains one pending disabled primary", "[dmSyncopation][mainTiming]" )
{
    MainImages  images;
    MainHarness app( images );
    images.main( 10 );
    images.output( 20, 9 );
    app.observe();
    Record record;
    REQUIRE( app.m_mainTiming.pop( record ) );
    images.main( 11 );
    images.output( 21, 10 );
    app.observe();
    REQUIRE( app.m_mainTiming.pop( record ) );
    CHECK( record.kind == Kind::main );
    CHECK( record.primaryCounter == 10 );
    CHECK( record.key.counter == 21 );
    images.main( 12 );
    images.output( 22, 11 );
    app.observe();
    REQUIRE( app.m_mainTiming.pop( record ) );
    CHECK( record.primaryCounter == 11 );

    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    MagAOX::app::synco::MainTiming::passive;
#endif
    // clang-format on
}

/** \brief An older computation cannot become a main because its output trace samples a newer primary counter.
 * \ingroup dmSyncopation_unit_test
 */
TEST_CASE( "main timing rejects auxiliary remnants across off handover", "[dmSyncopation][mainTiming]" )
{
    MainImages  images;
    MainHarness app( images );
    REQUIRE( app.enable( true ) == 0 );
    const auto oldGeneration = app.m_mainTiming.generation();
    REQUIRE( app.enable( false ) == 0 );
    Record record;

    while( app.m_mainTiming.pop( record ) )
    {
        REQUIRE( record.kind == Kind::mode );
    }

    images.main( 11 );
    images.output( 50, 10 );
    app.observe();
    images.main( 12 );
    images.output( 51, GENERATE( 11, 12 ) );
    // The output updater samples the then-current primary counter, not a frozen input identity.
    // This older computation started before either candidate primary and must qualify neither.
    images.trace.ts_procstart = MagAOX::app::synco::toTimespec( MainImages::stamp( 10 ) + 10000 );
    app.observe();
    CHECK_FALSE( app.m_mainTiming.pop( record ) );
    app.m_mainTiming.capture( { 22, 51, 1, 10020000 }, 11, 10, oldGeneration );
    CHECK_FALSE( app.m_mainTiming.pop( record ) );
    images.main( 13 );
    images.output( 52, 13 );
    app.observe();
    REQUIRE( app.m_mainTiming.pop( record ) );
    CHECK( record.kind == Kind::main );
    CHECK_FALSE( record.enabled );
    CHECK( record.primaryCounter == 13 );
    CHECK( record.generation != oldGeneration );

    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    MagAOX::app::synco::MainTiming::passive;
    MagAOX::app::synco::MainTiming::capture;
#endif
    // clang-format on
}

/** \brief A coalesced update remains a main when the combiner first recomputes after that primary was published.
 * \ingroup dmSyncopation_unit_test
 */
TEST_CASE( "main timing accepts first main response coalesced with a final auxiliary", "[dmSyncopation][mainTiming]" )
{
    MainImages  images;
    MainHarness app( images );
    REQUIRE( app.enable( true ) == 0 );
    const auto enabledGeneration = app.m_mainTiming.generation();
    REQUIRE( app.enable( false ) == 0 );
    Record record;

    while( app.m_mainTiming.pop( record ) )
    {
        REQUIRE( record.kind == Kind::mode );
    }

    images.main( 10 );
    images.output( 20, 9 );
    app.observe();
    CHECK_FALSE( app.m_mainTiming.pop( record ) );

    // DELAY-mode dmcomb polls the sum of channel counters and recomputes all channels.
    // A final auxiliary change coalesced with primary10 therefore includes primary10
    // when computation starts after its publication. Exactly one output since the
    // pre-primary baseline makes this the first response containing primary10.
    images.output( 21, 10 );
    images.main( 11 );
    app.observe();
    REQUIRE( app.m_mainTiming.pop( record ) );
    CHECK( record.kind == Kind::main );
    CHECK_FALSE( record.enabled );
    CHECK( record.primaryCounter == 10 );
    CHECK( record.key.counter == 21 );
    CHECK( record.generation != enabledGeneration );
    CHECK( MagAOX::dmTimingEvents::valid( record ) );
    CHECK( app.m_zeroStats.posted == 0 );
    CHECK( app.m_zeroStats.staged == 0 );
    CHECK_FALSE( app.m_mainTiming.pop( record ) );

    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    MagAOX::app::synco::MainTiming::passive;
#endif
    // clang-format on
}

/** \brief Counter gaps and malformed or ambiguous trace evidence suppress labels and require new continuity.
 * \ingroup dmSyncopation_unit_test
 */
TEST_CASE( "main timing rejects uncertain disabled snapshots", "[dmSyncopation][mainTiming]" )
{
    MainImages  images;
    MainHarness app( images );
    app.observe();
    Record record;
    REQUIRE( app.m_mainTiming.pop( record ) );
    images.main( 2 );
    images.output( 2, 2 );

    SECTION( "output overwritten" )
    {
        images.combinedMetadata.cnt0 = 4;
    }
    SECTION( "primary gap" )
    {
        images.main( 4 );
    }
    SECTION( "output metadata changing" )
    {
        images.combinedMetadata.write = 1;
    }
    SECTION( "missing trace" )
    {
        images.combinedMetadata.NBproctrace = 0;
    }
    SECTION( "wrong primary inode" )
    {
        images.trace.trigger_inode = 99;
    }
    SECTION( "stale producer start" )
    {
        images.trace.ts_procstart = MagAOX::app::synco::toTimespec( MainImages::stamp( 1 ) );
    }
    SECTION( "timestamp-first counter publication" )
    {
        images.combinedMetadata.cnt0 = 1;
    }
    app.observe();

    while( app.m_mainTiming.pop( record ) )
    {
        CHECK( record.kind != Kind::main );
    }

    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    MagAOX::app::synco::MainTiming::passive;
    MagAOX::app::synco::MainTiming::sample;
#endif
    // clang-format on
}

/** \brief Queue loss advances the generation and keeps a logical sequence gap visible after recovery.
 * \ingroup dmSyncopation_unit_test
 */
TEST_CASE( "main timing bounded capture reports overflow and freezes modes", "[dmSyncopation][mainTiming]" )
{
    MainTiming timing;
    timing.configure( 4 );
    const auto generation = timing.synchronize( true, 4, 1 );

    for( uint64_t counter = 1; counter <= 4; ++counter )
    {
        timing.capture( { 22, counter, 1, int64_t( counter * 1000000 ) }, 11, counter, generation );
    }

    CHECK( timing.queued() == 4 );
    CHECK( timing.dropped() == 1 );
    CHECK( timing.generation() > generation );
    Record   record;
    uint64_t lastSequence = 0;

    while( timing.pop( record ) )
    {
        CHECK( record.generation == generation );
        CHECK( record.enabled );
        lastSequence = record.sequence;
    }

    const auto recovered = timing.synchronize( false, 4, 1 );
    REQUIRE( timing.pop( record ) );
    CHECK( record.kind == Kind::mode );
    CHECK_FALSE( record.enabled );
    CHECK( record.sequence > lastSequence + 1 );
    CHECK( record.generation == recovered );
    timing.capture( { 22, 5, 1, 5000000 }, 11, 5, generation );
    CHECK_FALSE( timing.pop( record ) );
    timing.capture( { 22, 6, 1, 6000000 }, 11, 6, recovered );
    REQUIRE( timing.pop( record ) );
    CHECK( record.kind == Kind::main );
    CHECK_FALSE( record.enabled );
    CHECK( record.generation == recovered );

    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    MagAOX::app::synco::MainTiming::synchronize;
    MagAOX::app::synco::MainTiming::capture;
    MagAOX::app::synco::MainTiming::pop;
#endif
    // clang-format on
}

} // namespace dmSyncopationTest
} // namespace libXWCTest
