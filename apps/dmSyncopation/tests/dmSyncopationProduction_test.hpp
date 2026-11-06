/** \file dmSyncopationProduction_test.hpp
 * \brief Private-stream production admission, independent fallback and passive-copy regressions.
 * \ingroup dmSyncopation_unit_test
 */
#ifndef dmSyncopationProduction_test_hpp
#define dmSyncopationProduction_test_hpp

// Included inside the existing dmSyncopation test namespace after its private-stream harness.

/// \cond PRODUCTION_TEST_HARNESS
/// Deterministic local clock for direct production snapshot/copy tests.
struct ProductionClock
{
    static inline int64_t  time{ 1000000000 }; ///< Private monotonic time, unrelated to instrument clocks.
    static inline unsigned delayRead{ 0 };     ///< Countdown to one delayed clock read, zero when disarmed.
    static inline int64_t  delayNs{ 0 };       ///< One-shot simulated elapsed work on the selected read.

    /// Advance private bookkeeping work and inject the selected one-shot delay without an operating-system wait.
    static int64_t now();
};

int64_t ProductionClock::now()
{
    time += 100;

    if( delayRead && --delayRead == 0 )
    {
        time += delayNs;
        delayNs = 0;
    }

    return time;
}

/// Expose production operations without starting app or observer threads.
class ProductionHarness : public Harness
{
  public:
    using dmSyncopation::closeProduction;
    using dmSyncopation::configureDiagnostics;
    using dmSyncopation::detailedTimingEnabled;
    using dmSyncopation::m_generation;
    using dmSyncopation::m_indiP_production;
    using dmSyncopation::m_passiveCopied;
    using dmSyncopation::m_productionEnabled;
    using dmSyncopation::m_productionReadyLeadUs;
    using dmSyncopation::m_productionStep0LeadUs;
    using dmSyncopation::m_snapshotMax;
    using dmSyncopation::m_zeroTrial;
    using dmSyncopation::openProduction;
    using dmSyncopation::productionCopy;
    using dmSyncopation::productionSnapshot;
    using dmSyncopation::publishProductionSchedule;
    using dmSyncopation::setupProductionINDI;
    using MagAOXApp<true>::m_indiNewCallBacks;

    /// Attach only fixture streams and initialize the independent production mappings.
    void initialize();
};

void ProductionHarness::initialize()
{
    ProductionClock::time      = 1000000000;
    ProductionClock::delayRead = 0;
    ProductionClock::delayNs   = 0;
    m_now                      = ProductionClock::now;
    attachPrimaryOnly();
    m_productionEnabled = true;
    m_zeroTestChannel   = m_destinations.front();
    m_requested         = 4;
    m_enabled           = true;
    REQUIRE( openProduction() );
    REQUIRE( m_mappings.size() == 2 );
}

/// Destroy only fixed test-owned generated streams after the app closes its attachments.
struct ProductionSidecars
{
    /// Remove the private fixture's production sources and metadata, never instrument streams.
    ~ProductionSidecars();
};

ProductionSidecars::~ProductionSidecars()
{
    for( const char *name : { "dm97disp07_aux_ready",
                              "dm97disp07_aux_ack",
                              "dm97disp08_aux_ready",
                              "dm97disp08_aux_ack",
                              "dm97disp07_aux",
                              "dm97disp08_aux",
                              "synco_test_timing" } )
    {
        IMAGE image{};

        if( ImageStreamIO_openIm( &image, name ) == 0 )
        {
            // Imported handles leave used=0; this fixture explicitly owns these private files.
            image.used = 1;
            ImageStreamIO_destroyIm( &image );
        }
    }
}

/// Make one immutable slot well ahead of this fixture's clock.
synco::Request productionRequest( const ProductionHarness &app /**< [in] Private scheduler identity. */,
                                  int                      step /**< [in] Passive zero or active subdivision. */,
                                  int64_t                  target /**< [in] Absolute private-clock target. */ )
{
    return { app.m_epoch, app.m_generation, 42, step, target, target - 50000 };
}

/// Publish one complete fixture-owned source and its exact producer commitment.
synco::CommitRecord productionCommand( ProductionHarness    &app /**< [in,out] Private mappings. */,
                                       size_t                mapping /**< [in] Independent fixture producer index. */,
                                       const synco::Request &request /**< [in] Immutable advertised slot. */,
                                       float                 value /**< [in] Uniform command value. */,
                                       int64_t sequence = 1 /**< [in] Positive private producer sequence. */ )
{
    auto &source     = *app.m_mappings[mapping].m_source->image();
    source.md->write = 1;
    std::fill_n( source.array.F, 9, value );
    REQUIRE( ImageStreamIO_UpdateIm( &source ) == 0 );
    synco::CommitRecord command{};
    command[synco::cmdVersion]        = synco::productionVersion;
    command[synco::cmdEpoch]          = request.epoch;
    command[synco::cmdGeneration]     = request.generation;
    command[synco::cmdCycle]          = request.cycle;
    command[synco::cmdStep]           = request.step;
    command[synco::cmdTarget]         = request.targetNs;
    command[synco::cmdCutoff]         = request.cutoffNs;
    command[synco::cmdPayloadCounter] = int64_t( source.md->cnt0 );
    command[synco::cmdReadyNs]        = request.cutoffNs - 1000;
    command[synco::cmdSession]        = 7;
    command[synco::cmdSequence]       = sequence;
    command[synco::cmdPublisher]      = int64_t( getpid() );
    command[synco::cmdPayloadInode]   = int64_t( source.md->inode );
    auto &ready                       = app.m_mappings[mapping].m_ready;
    REQUIRE( ready.md != nullptr );
    ready.md->write = 1;
    std::memcpy( ready.array.raw, command.data(), sizeof( command ) );
    REQUIRE( ImageStreamIO_UpdateIm( &ready ) == 0 );
    return command;
}

/// Read the private consumer echo without touching any semaphore or external service.
synco::AckRecord productionAck( const ProductionHarness &app /**< [in] Private scheduler. */,
                                size_t                   mapping /**< [in] Independent producer index. */ )
{
    synco::AckRecord result{};
    const auto      &image = app.m_mappings[mapping].m_ack;
    REQUIRE( image.md != nullptr );
    REQUIRE( image.md->write == 0 );
    std::memcpy( result.data(), image.array.raw, sizeof( result ) );
    return result;
}
/// Advance the private controller clock while emulating one cooperative changing producer.
struct ProductionCycleClock
{
    static inline ProductionHarness  *app{ nullptr };  ///< Unstarted production app on private fixture streams.
    static inline synco::CommitRecord pending{};       ///< Exact single outstanding source publication.
    static inline int64_t             lastTarget{ 0 }; ///< Last submitted target, independent of notifications.
    static inline int64_t             sequence{ 0 }; ///< Cooperative fake producer's monotonically increasing sequence.
    static inline int64_t copyDelayNs{ 0 }; ///< One-shot overrun after the first slot's destination copy completes.

    /// Publish each earliest future request only after the consumer releases the preceding payload.
    static void produce();

    /// Forward a private time read and publish newly advertised fixture-only commands.
    static int64_t now();

    /// Forward a private absolute wait while preserving fake combiner and reader-release behavior.
    static int wait( const timespec *target /**< [in] Absolute private monotonic wake. */ );
};

void ProductionCycleClock::produce()
{
    if( !app || !app->m_resources )
    {
        return;
    }

    if( pending[synco::cmdSequence] && !synco::acknowledges( productionAck( *app, 0 ), pending ) )
    {
        return;
    }

    const auto step = synco::nextProductionStep( app->m_record, ZeroClock::time );

    if( step < 0 )
    {
        return;
    }

    const auto request = synco::request( app->m_record, step );

    if( request.targetNs == lastTarget )
    {
        return;
    }

    pending    = productionCommand( *app, 0, request, step == 0 ? 20.0F : 10.0F + step, ++sequence );
    lastTarget = request.targetNs;
}

int64_t ProductionCycleClock::now()
{
    if( copyDelayNs && app && !app->m_mappings.empty() && app->m_mappings.front().m_cache.front() == 11.0F )
    {
        ZeroClock::time += copyDelayNs;
        copyDelayNs = 0;
    }

    const auto now = ZeroClock::now();
    produce();
    return now;
}

int ProductionCycleClock::wait( const timespec *target )
{
    const auto result = ZeroClock::wait( target );
    produce();
    return result;
}
/// \endcond

/// Production startup registers one complete definition in the INDI-enabled registry without driver threads.
TEST_CASE( "production INDI startup registers one serializable definition", "[dmSyncopation][production][indi]" )
{
    ProductionHarness app;
    const bool        enabled = GENERATE( false, true );
    app.m_productionEnabled   = enabled;
    REQUIRE( app.setupProductionINDI() == 0 );

    if( enabled )
    {
        const auto &property = app.m_indiP_production;
        const auto  key      = property.createUniqueKey();
        REQUIRE( app.m_indiNewCallBacks.size() == 1 );
        REQUIRE( app.m_indiNewCallBacks.count( key ) == 1 );
        CHECK( app.m_indiNewCallBacks.at( key ).property == &property );
        CHECK( app.m_indiNewCallBacks.at( key ).callBack == nullptr );
        CHECK( property.getPerm() == pcf::IndiProperty::ReadOnly );
        CHECK( property.getState() == pcf::IndiProperty::Idle );
        CHECK( property.getElements().size() == 3 + 3 * app.m_destinations.size() );
        CHECK( property["enabled"].get<double>() == 1.0 );

        for( const auto &element : property.getElements() )
        {
            CHECK( mx::math::isFinite( element.second.get<double>() ) );
        }

        const auto xml =
            pcf::IndiXmlParser( pcf::IndiMessage( pcf::IndiMessage::Define, property ), "1.7" ).createXmlString();
        CHECK( xml.find( "<defNumberVector" ) != std::string::npos );
        CHECK( xml.find( "name=\"production\"" ) != std::string::npos );
        CHECK( xml.find( "name=\"accepted0\"" ) != std::string::npos );
    }
    else
    {
        CHECK( app.m_indiNewCallBacks.empty() );
        CHECK( app.m_indiP_production.getElements().empty() );
    }

    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    MagAOX::app::dmSyncopation::setupProductionINDI;
#endif
    // clang-format on
}

/// Independent missing inputs retain only their own value while every selected mapping is copied passively.
TEST_CASE( "production inputs have independent deadlines and passive copies", "[dmSyncopation][production]" )
{
    Streams            streams;
    ProductionSidecars sidecars;
    ProductionHarness  app;
    app.initialize();
    const auto first      = productionRequest( app, 0, ProductionClock::time + 125000 );
    const auto commandA   = productionCommand( app, 0, first, 10 );
    ProductionClock::time = first.cutoffNs;
    const auto counterA   = streams.images[2].md->cnt0;
    const auto counterB   = streams.images[3].md->cnt0;
    REQUIRE( app.productionSnapshot( first, ProductionClock::time ) );
    CHECK( app.m_mappings[0].m_stage.front() == 10 );
    CHECK( app.m_mappings[1].m_stage.front() == 3 );
    CHECK( app.m_mappings[0].m_cache.front() == 2 );
    CHECK( app.m_mappings[0].m_accepted == 1 );
    CHECK( app.m_mappings[1].m_missing == 1 );
    const auto acknowledged = productionAck( app, 0 );
    CHECK( synco::acknowledges( acknowledged, commandA ) );
    CHECK( acknowledged[synco::ackStatus] == synco::copied );
    REQUIRE( app.productionCopy( streams.images[0].md->cnt0, first.targetNs - 20000 ) );

    for( size_t pixel = 0; pixel < 9; ++pixel )
    {
        CHECK( streams.images[2].array.F[pixel] == 10 );
        CHECK( streams.images[3].array.F[pixel] == 3 );
    }

    CHECK( streams.images[2].md->cnt0 == counterA );
    CHECK( streams.images[3].md->cnt0 == counterB );
    CHECK( app.m_mappings[0].m_cache.front() == 10 );

    const auto second     = productionRequest( app, 1, first.targetNs + 125000 );
    const auto commandB   = productionCommand( app, 1, second, 30 );
    ProductionClock::time = second.cutoffNs;
    REQUIRE( app.productionSnapshot( second, ProductionClock::time ) );
    REQUIRE( app.productionCopy( streams.images[0].md->cnt0, second.targetNs - 10000 ) );
    CHECK( app.m_mappings[0].m_missing == 1 );
    CHECK( app.m_mappings[1].m_accepted == 1 );
    CHECK( synco::acknowledges( productionAck( app, 1 ), commandB ) );
    CHECK( streams.images[2].array.F[0] == 10 );
    CHECK( streams.images[3].array.F[0] == 30 );
    CHECK( streams.images[2].md->cnt0 == counterA );
    CHECK( streams.images[3].md->cnt0 == counterB );

    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    MagAOX::app::dmSyncopation::productionSnapshot;
    MagAOX::app::dmSyncopation::productionCopy;
#endif
    // clang-format on
}

/// Production startup is opt-in, needs release evidence and rejects conflicting modes or impossible preparation leads.
TEST_CASE( "production startup validates mode and preparation settings", "[dmSyncopation][production][config]" )
{
    ProductionHarness app;
    app.setupConfig();
    const int  fault      = GENERATE( 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 );
    char       path[]     = "/tmp/dm-synco-production-config-XXXXXX";
    const auto descriptor = mkstemp( path );
    REQUIRE( descriptor >= 0 );
    close( descriptor );

    {
        std::ofstream config( path );
        config << "[shmimMonitor]\nshmimName=dm97disp03\n"
               << "[streams]\ncombined=dm97disp\ntiming=synco_test_timing\n"
               << "destinations=dm97disp07,dm97disp08\n"
               << "[scheduler]\nmaxSubframes="
               << ( fault == 8   ? 5
                    : fault == 9 ? 1
                                 : 4 )
               << "\nmaxCadenceHz=8000\ncopyBudgetUs=30\nsnapshotBudgetUs=10\n"
               << "[readerReleaseMonitor]\nshmimName=" << ( fault == 3 || fault == 13 ? "" : "dmtweeter_read_done" )
               << "\n"
               << "[production]\nenabled=" << ( fault == 13 ? "false" : "true" ) << "\nreadyLeadUs="
               << ( fault == 4   ? 20
                    : fault == 7 ? 100
                                 : 50 )
               << "\nstep0LeadUs=" << ( fault == 5 ? 60 : 20 ) << "\nwakeAheadUs=" << ( fault == 6 ? 50 : 10 )
               << "\nlateUs=6\n"
               << "[zeroTest]\nenabled=" << ( fault == 1 ? "true" : "false" ) << "\nchannel=dm97disp07\n"
               << "[simulation]\nenabled=" << ( fault == 2 ? "true" : "false" )
               << "\nreaderReleaseReady=false\npipelineBackend=false\n";

        if( fault >= 10 && fault <= 12 )
        {
            config << "[diagnostics]\nwindowSec=" << ( fault == 10 ? 2 : 0 )
                   << "\ncombiner=" << ( fault == 11 ? "false" : "true" )
                   << "\nmaxEvents=" << ( fault == 12 ? 0 : 20000 ) << '\n';
        }
    }

    app.config.readConfig( path );
    std::remove( path );
    const auto result = app.loadConfigImpl( app.config );
    CAPTURE( fault );

    if( fault == 0 || fault == 9 || fault == 10 || fault == 11 || fault == 13 )
    {
        CHECK( result == 0 );
        CHECK( app.m_productionEnabled == ( fault != 13 ) );
        CHECK_FALSE( app.m_zeroTestEnabled );
        CHECK_FALSE( app.m_simulationEnabled );
        CHECK( app.m_zeroLateUs == ( fault == 13 ? 5 : 6 ) );
        CHECK( app.m_zeroWakeAheadUs == ( fault == 13 ? 0 : 10 ) );
        CHECK( app.m_productionReadyLeadUs == 50 );
        CHECK( app.m_productionStep0LeadUs == 20 );
        CHECK( app.m_requested == ( fault == 9 ? 1 : 4 ) );
        CHECK( app.m_maxCadence == 8000 );
        CHECK( app.m_bounds.copy == 30000 );
        CHECK( app.m_bounds.snapshot == 10000 );
        CHECK( app.m_primaryCadenceWindow == ( fault == 10 || fault == 13 ? 2 : 0 ) );
        app.configureDiagnostics();
        CHECK( app.detailedTimingEnabled() == ( fault == 10 || fault == 13 ) );

        if( app.m_productionEnabled )
        {
            CHECK( app.readerReleaseMonitorEnabled() );
            CHECK( app.m_simulationReleaseQueue.capacity() > 0 );
        }

        if( !app.detailedTimingEnabled() )
        {
            CHECK_FALSE( app.m_primaryCadence.enabled() );
            CHECK_FALSE( app.m_combinerCadence.enabled() );
            CHECK_FALSE( app.m_cameraCadence.enabled() );
            CHECK_FALSE( app.m_cameraPublicationCadence.enabled() );
            CHECK_FALSE( app.m_pipelineCadence.enabled() );
            CHECK_FALSE( app.m_readerTiming.enabled() );
            CHECK_FALSE( app.m_readerReleaseTiming.enabled() );
            CHECK( app.m_zeroEvents.capacity() == 0 );
        }
    }
    else
    {
        CHECK( result < 0 );
    }

    CHECK_FALSE( app.m_enabled );
    CHECK_FALSE( app.m_resources );
    CHECK_FALSE( app.m_zeroAttached );

    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    MagAOX::app::dmSyncopation::setupConfig;
    MagAOX::app::dmSyncopation::loadConfigImpl;
    MagAOX::app::dmSyncopation::configureDiagnostics;
    MagAOX::app::dmSyncopation::detailedTimingEnabled;
    MagAOX::app::dmSyncopation::readerReleaseMonitorEnabled;
#endif
    // clang-format on
}

/// A malformed future commitment is discarded so it cannot hold that producer's single buffer indefinitely.
TEST_CASE( "production discards malformed future commands without blocking independent updates",
           "[dmSyncopation][production]" )
{
    Streams            streams;
    ProductionSidecars sidecars;
    ProductionHarness  app;
    app.initialize();
    const auto current         = productionRequest( app, 1, ProductionClock::time + 125000 );
    auto       future          = productionRequest( app, 2, current.targetNs + 125000 );
    auto       command         = productionCommand( app, 0, future, 10 );
    command[synco::cmdVersion] = -1;
    auto &ready                = app.m_mappings[0].m_ready;
    ready.md->write            = 1;
    std::memcpy( ready.array.raw, command.data(), sizeof( command ) );
    REQUIRE( ImageStreamIO_UpdateIm( &ready ) == 0 );
    productionCommand( app, 1, current, 30 );
    ProductionClock::time = current.cutoffNs;
    REQUIRE( app.productionSnapshot( current, ProductionClock::time ) );
    const auto acknowledgement = productionAck( app, 0 );
    CHECK( synco::acknowledges( acknowledgement, command ) );
    CHECK( acknowledgement[synco::ackStatus] == synco::discarded );
    CHECK( app.m_mappings[0].m_rejected == 1 );
    CHECK( app.m_mappings[0].m_stage.front() == 2 );
    CHECK( app.m_mappings[1].m_stage.front() == 30 );

    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    MagAOX::app::dmSyncopation::productionSnapshot;
#endif
    // clang-format on
}

/// Malformed metadata is rejected before reading pixels or command words from an incompatible view.
TEST_CASE( "production malformed input metadata is isolated to its mapping", "[dmSyncopation][production]" )
{
    Streams            streams;
    ProductionSidecars sidecars;
    ProductionHarness  app;
    app.initialize();
    const auto request = productionRequest( app, 1, ProductionClock::time + 125000 );
    productionCommand( app, 0, request, 10 );
    productionCommand( app, 1, request, 30 );
    const int  fault    = GENERATE( 0, 1, 2 );
    auto      *metadata = fault == 2 ? app.m_mappings[0].m_source->image()->md : app.m_mappings[0].m_ready.md;
    const auto datatype = metadata->datatype;
    const auto count    = metadata->size[0];

    if( fault == 0 )
    {
        metadata->datatype = _DATATYPE_FLOAT;
    }
    else
    {
        --metadata->size[0];
    }

    ProductionClock::time = request.cutoffNs;
    const bool captured   = app.productionSnapshot( request, ProductionClock::time );
    metadata->datatype    = datatype;
    metadata->size[0]     = count;
    REQUIRE( captured );
    CHECK( app.m_mappings[0].m_accepted == 0 );
    CHECK( app.m_mappings[0].m_rejected == 1 );
    CHECK( app.m_mappings[0].m_stage.front() == 2 );
    CHECK( app.m_mappings[1].m_accepted == 1 );
    CHECK( app.m_mappings[1].m_stage.front() == 30 );

    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    MagAOX::app::dmSyncopation::productionSnapshot;
#endif
    // clang-format on
}

/// Stale, late, incoherent or nonfinite commands do not replace retained values or block independent inputs.
TEST_CASE( "production rejects bad inputs independently and does not replay old slots", "[dmSyncopation][production]" )
{
    Streams            streams;
    ProductionSidecars sidecars;
    ProductionHarness  app;
    app.initialize();
    const int fault = GENERATE( 0, 1, 2, 3, 4, 5 );
    CAPTURE( fault );
    auto       request = productionRequest( app, 1, ProductionClock::time + 125000 );
    auto       invalid = productionCommand( app, 0, request, 10 );
    const auto valid   = productionCommand( app, 1, request, 30 );

    if( fault == 0 )
    {
        --invalid[synco::cmdCycle];
    }
    else if( fault == 1 )
    {
        invalid[synco::cmdReadyNs] = request.cutoffNs + 1;
    }
    else if( fault == 2 )
    {
        ++invalid[synco::cmdPayloadCounter];
    }
    else if( fault == 3 )
    {
        volatile uint32_t injected = 0x7fc00000U;
        const uint32_t    bits     = injected;
        std::memcpy( app.m_mappings[0].m_source->data(), &bits, sizeof( bits ) );
        CHECK( mx::math::isNan( app.m_mappings[0].m_source->data()[0] ) );
    }
    else if( fault == 4 )
    {
        app.m_mappings[0].m_source->image()->md->write = 1;
    }
    else if( fault == 5 )
    {
        ++invalid[synco::cmdGeneration];
    }

    auto &ready     = app.m_mappings[0].m_ready;
    ready.md->write = 1;
    std::memcpy( ready.array.raw, invalid.data(), sizeof( invalid ) );
    REQUIRE( ImageStreamIO_UpdateIm( &ready ) == 0 );
    ProductionClock::time = request.cutoffNs + 100;
    REQUIRE( app.productionSnapshot( request, ProductionClock::time ) );
    REQUIRE( app.productionCopy( streams.images[0].md->cnt0, request.targetNs - 10000 ) );
    CHECK( streams.images[2].array.F[0] == 2 );
    CHECK( streams.images[3].array.F[0] == 30 );
    CHECK( app.m_mappings[0].m_accepted == 0 );
    CHECK( app.m_mappings[1].m_accepted == 1 );
    CHECK( synco::acknowledges( productionAck( app, 1 ), valid ) );
    app.m_mappings[0].m_source->image()->md->write = 0;

    request.targetNs += 125000;
    request.cutoffNs += 125000;
    request.step          = 2;
    ProductionClock::time = request.cutoffNs;
    const auto fresh      = productionCommand( app, 0, request, 11, 2 );
    REQUIRE( app.productionSnapshot( request, ProductionClock::time ) );
    REQUIRE( app.productionCopy( streams.images[0].md->cnt0, request.targetNs - 10000 ) );
    CHECK( streams.images[2].array.F[0] == 11 );
    CHECK( streams.images[3].array.F[0] == 30 );
    CHECK( synco::acknowledges( productionAck( app, 0 ), fresh ) );

    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    MagAOX::app::dmSyncopation::productionSnapshot;
    MagAOX::app::dmSyncopation::productionCopy;
#endif
    // clang-format on
}

/// A shared copy veto keeps accepted source snapshots separate from destination retention.
TEST_CASE( "production copy safety never turns a private snapshot into an applied contribution",
           "[dmSyncopation][production]" )
{
    Streams            streams;
    ProductionSidecars sidecars;
    ProductionHarness  app;
    app.initialize();
    const auto request = productionRequest( app, 0, ProductionClock::time + 125000 );
    productionCommand( app, 0, request, 10 );
    ProductionClock::time = request.cutoffNs;
    REQUIRE( app.productionSnapshot( request, ProductionClock::time ) );
    const auto primary = streams.images[0].md->cnt0;
    ++streams.images[0].md->cnt0;
    CHECK_FALSE( app.productionCopy( primary, request.targetNs - 20000 ) );
    CHECK( streams.images[2].array.F[0] == 2 );
    CHECK( app.m_mappings[0].m_cache.front() == 2 );
    CHECK( app.m_mappings[0].m_stage.front() == 10 );

    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    MagAOX::app::dmSyncopation::productionCopy;
#endif
    // clang-format on
}

/// A configured copy allowance still vetoes a batch before any destination metadata or pixels change.
TEST_CASE( "production fixed copy budget retains the pre-copy fit guard", "[dmSyncopation][production][budget]" )
{
    Streams            streams;
    ProductionSidecars sidecars;
    ProductionHarness  app;
    app.initialize();
    app.m_bounds.copy  = 30000;
    const auto request = productionRequest( app, 1, ProductionClock::time + 125000 );
    productionCommand( app, 0, request, 10 );
    productionCommand( app, 1, request, 30 );
    ProductionClock::time = request.cutoffNs;
    REQUIRE( app.productionSnapshot( request, ProductionClock::time ) );
    const auto stampA   = streams.images[2].md->writetime;
    const auto stampB   = streams.images[3].md->writetime;
    const auto counterA = streams.images[2].md->cnt0;
    const auto counterB = streams.images[3].md->cnt0;

    CHECK_FALSE( app.productionCopy( streams.images[0].md->cnt0, ProductionClock::time + 20000 ) );
    CHECK( app.m_zeroWaitFailure == Harness::ZeroOutcome::preparationLate );
    CHECK( app.m_bounds.copy == 30000 );
    CHECK( app.m_copyMax == 0 );
    CHECK( app.m_mappings[0].m_cache.front() == 2 );
    CHECK( app.m_mappings[1].m_cache.front() == 3 );
    CHECK( streams.images[2].md->cnt0 == counterA );
    CHECK( streams.images[3].md->cnt0 == counterB );
    CHECK( streams.images[2].md->write == 0 );
    CHECK( streams.images[3].md->write == 0 );
    CHECK( streams.images[2].md->writetime.tv_sec == stampA.tv_sec );
    CHECK( streams.images[2].md->writetime.tv_nsec == stampA.tv_nsec );
    CHECK( streams.images[3].md->writetime.tv_sec == stampB.tv_sec );
    CHECK( streams.images[3].md->writetime.tv_nsec == stampB.tv_nsec );

    for( size_t pixel = 0; pixel < 9; ++pixel )
    {
        CHECK( streams.images[2].array.F[pixel] == 2 );
        CHECK( streams.images[3].array.F[pixel] == 3 );
    }

    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    MagAOX::app::dmSyncopation::productionCopy;
#endif
    // clang-format on
}

/// A slow private snapshot updates its observed maximum without changing the configured allowance or later admission.
TEST_CASE( "production snapshot excursions preserve the configured budget", "[dmSyncopation][production][budget]" )
{
    Streams            streams;
    ProductionSidecars sidecars;
    ProductionHarness  app;
    app.initialize();
    app.m_bounds.snapshot = 10000;
    const auto request    = productionRequest( app, 1, ProductionClock::time + 125000 );
    productionCommand( app, 0, request, 10 );
    ProductionClock::time      = request.cutoffNs;
    ProductionClock::delayRead = 2;
    ProductionClock::delayNs   = 90000;

    REQUIRE( app.productionSnapshot( request, ProductionClock::time ) );
    CHECK( ProductionClock::delayRead == 0 );
    CHECK( app.m_snapshotMax == 90100 );
    CHECK( app.m_bounds.snapshot == 10000 );

    const auto next = productionRequest( app, 2, request.targetNs + 125000 );
    productionCommand( app, 0, next, 11, 2 );
    ProductionClock::time = next.cutoffNs;
    REQUIRE( app.productionSnapshot( next, ProductionClock::time ) );
    REQUIRE( app.productionCopy( streams.images[0].md->cnt0, next.targetNs - 10000 ) );
    CHECK( app.m_snapshotMax == 90100 );
    CHECK( app.m_bounds.snapshot == 10000 );
    CHECK( streams.images[2].array.F[0] == 11 );

    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    MagAOX::app::dmSyncopation::productionSnapshot;
    MagAOX::app::dmSyncopation::productionCopy;
#endif
    // clang-format on
}

/// A real copy overrun suppresses its trigger while later slots and passive copies recover with the fixed allowance.
TEST_CASE( "production copy overruns do not latch later slots off", "[dmSyncopation][production][budget][integration]" )
{
    Streams            streams;
    ProductionSidecars sidecars;
    ProductionHarness  app;
    ZeroClock::connect( app, streams, 4 );
    const auto next             = ZeroClock::warm();
    app.m_productionEnabled     = true;
    app.m_zeroTestEnabled       = false;
    app.m_productionReadyLeadUs = 70;
    app.m_productionStep0LeadUs = 20;
    app.m_zeroLateUs            = 6;
    app.m_zeroWakeAheadUs       = 10;
    app.m_maxCadence            = 8000;
    app.m_bounds.copy           = 30000;
    REQUIRE( app.openProduction() );
    ProductionCycleClock::app         = &app;
    ProductionCycleClock::pending     = {};
    ProductionCycleClock::lastTarget  = 0;
    ProductionCycleClock::sequence    = 0;
    ProductionCycleClock::copyDelayNs = 90000;
    app.m_now                         = ProductionCycleClock::now;
    app.m_wait                        = ProductionCycleClock::wait;
    const auto counterBefore          = streams.images[2].md->cnt0;
    app.m_enabled                     = true;

    ZeroClock::primary( next );
    CHECK( ProductionCycleClock::copyDelayNs == 0 );
    CHECK( app.m_copyMax >= 90000 );
    CHECK( app.m_bounds.copy == 30000 );
    CHECK( app.m_zeroStats.preparationLate == 1 );
    CHECK( app.m_zeroStats.skipped == 1 );
    CHECK( app.m_zeroStats.posted == 2 );
    CHECK( app.m_zeroStats.acknowledged == 2 );
    CHECK( app.m_passiveCopied == 1 );
    CHECK( streams.images[2].md->cnt0 == counterBefore + 2 );
    REQUIRE( ZeroClock::posts.size() == 2 );
    CHECK( ZeroClock::posts[0] >= next + 250000 );
    CHECK( ZeroClock::posts[0] <= next + 256000 );
    CHECK( ZeroClock::posts[1] >= next + 375000 );
    CHECK( ZeroClock::posts[1] <= next + 381000 );
    const auto copyMaximum = app.m_copyMax;

    ZeroClock::primary( next + 500000 );
    CHECK( app.m_copyMax == copyMaximum );
    CHECK( app.m_bounds.copy == 30000 );
    CHECK( app.m_zeroStats.preparationLate == 1 );
    CHECK( app.m_zeroStats.skipped == 1 );
    CHECK( app.m_zeroStats.posted == 5 );
    CHECK( app.m_zeroStats.acknowledged == 5 );
    CHECK( app.m_passiveCopied == 2 );
    CHECK( streams.images[2].md->cnt0 == counterBefore + 5 );
    CHECK( app.m_enabled );
    ProductionCycleClock::app = nullptr;

    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    MagAOX::app::dmSyncopation::processZeroTestPrimary;
    MagAOX::app::dmSyncopation::productionCopy;
    MagAOX::app::dmSyncopation::productionStep0;
#endif
    // clang-format on
}

/// Both diagnostic modes preserve real receipt handling, tagged deliveries, passive preparation and continuous
/// lifetime.
TEST_CASE( "production common scheduler continuously publishes tagged contributions",
           "[dmSyncopation][production][integration]" )
{
    Streams            streams;
    ProductionSidecars sidecars;
    ProductionHarness  app;
    const int          steps    = GENERATE( 1, 4 );
    const bool         detailed = GENERATE( false, true );
    CAPTURE( steps, detailed );
    ZeroClock::connect( app, streams, steps );
    app.m_primaryCadenceWindow    = detailed ? 2 : 0;
    app.m_primaryCadenceMaxEvents = 256;
    app.configureDiagnostics();
    const auto next         = ZeroClock::warm();
    app.m_productionEnabled = true;
    app.m_zeroTestEnabled   = false;
    app.m_zeroLateUs        = 6;
    app.m_zeroWakeAheadUs   = 10;
    app.m_maxCadence        = 8000;
    REQUIRE( app.openProduction() );
    const auto          initialReady = app.m_mappings[1].m_ready.md;
    synco::CommitRecord initialB{};
    std::memcpy( initialB.data(), app.m_mappings[1].m_ready.array.raw, sizeof( initialB ) );
    CAPTURE( initialReady->cnt0,
             initialB[synco::cmdVersion],
             app.m_mappings[1].m_inputAvailable,
             app.m_mappings[1].m_source->image()->md->size[0],
             app.m_mappings[1].m_source->image()->md->size[1] );
    ProductionCycleClock::app         = &app;
    ProductionCycleClock::pending     = {};
    ProductionCycleClock::lastTarget  = 0;
    ProductionCycleClock::sequence    = 0;
    ProductionCycleClock::copyDelayNs = 0;
    app.m_now                         = ProductionCycleClock::now;
    app.m_wait                        = ProductionCycleClock::wait;
    const auto primaryBefore          = streams.images[0].md->cnt0;
    const auto counterBefore          = streams.images[2].md->cnt0;
    app.m_enabled                     = true;
    // ZeroClock uses the actual receipt hook followed by processImage, including when every collector is disabled.
    ZeroClock::primary( next );
    CHECK( app.m_primaryCadence.enabled() == detailed );
    REQUIRE( app.m_zeroStats.posted == uint64_t( steps - 1 ) );
    CHECK( app.m_zeroStats.acknowledged == uint64_t( steps - 1 ) );
    CHECK( app.m_zeroStats.skipped == 0 );
    CHECK( app.m_mappings[0].m_accepted == uint64_t( steps ) );
    CAPTURE( app.m_mappings[1].m_rejected );
    CHECK( app.m_mappings[1].m_inputAvailable );
    CHECK( app.m_mappings[1].m_missing == uint64_t( steps ) );
    CHECK( streams.images[2].array.F[0] == 20 );
    CHECK( streams.images[3].array.F[0] == 3 );
    CHECK( streams.images[2].md->cnt0 == counterBefore + uint64_t( steps - 1 ) );
    CHECK( streams.images[0].md->cnt0 == primaryBefore + 1 );
    CHECK( app.m_enabled );
    CHECK_FALSE( app.m_zeroStats.finished );

    // Production ignores a finite trial duration both inside the timed cycle and in wall-clock status expiry.
    app.m_zeroDurationSec = 0.000001;
    ZeroClock::primary( next + 500000 );
    CHECK( app.m_zeroStats.posted == uint64_t( 2 * ( steps - 1 ) ) );
    CHECK( app.m_zeroStats.acknowledged == uint64_t( 2 * ( steps - 1 ) ) );
    CHECK( app.m_zeroStats.skipped == 0 );
    CHECK( app.m_enabled );
    app.publishZeroTestStatus( app.m_zeroStats.startedNs + 121000000000LL, true );
    CHECK( app.m_enabled );
    CHECK_FALSE( app.m_zeroStats.finished );

    if( detailed )
    {
        CHECK( app.m_zeroEvents.queued() > 0 );
        CHECK( app.m_zeroPhase.summary().observed > 0 );
        CHECK( app.m_zeroPublicationTiming.summary().mainObserved > 0 );
    }
    else
    {
        CHECK( app.m_zeroEvents.queued() == 0 );
        CHECK( app.m_zeroEvents.dropped() == 0 );
        CHECK( app.m_zeroPhase.summary().observed == 0 );
        CHECK( app.m_zeroPublicationTiming.summary().mainObserved == 0 );
        CHECK( app.m_zeroPublicationTiming.summary().auxiliaryObserved == 0 );
        CHECK( app.m_zeroTiming.summary().classified == 0 );
    }

    ProductionCycleClock::app = nullptr;

    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    MagAOX::app::dmSyncopation::processZeroTestPrimary;
    MagAOX::app::dmSyncopation::productionStep0;
    MagAOX::app::dmSyncopation::publishZeroTestStatus;
    MagAOX::app::dmSyncopation::configureDiagnostics;
    MagAOX::app::dmSyncopation::shmimMonitorTiming;
    MagAOX::app::dmSyncopation::processImage;
#endif
    // clang-format on
}

/// Minimal diagnostics retain reader-release, final SDK-failure and publication-clock safety guards.
TEST_CASE( "minimal production diagnostics preserve safety rejections",
           "[dmSyncopation][production][integration][minimalDiagnostics]" )
{
    Streams            streams;
    CompletionStream   sdk;
    ProductionSidecars sidecars;
    ProductionHarness  app;
    const int          fault = GENERATE( 0, 1, 2 );
    CAPTURE( fault );
    ZeroClock::connect( app, streams, 4 );
    app.m_primaryCadenceWindow = 0;
    app.configureDiagnostics();
    const auto next         = ZeroClock::warm();
    app.m_productionEnabled = true;
    app.m_zeroTestEnabled   = false;
    app.m_zeroLateUs        = 6;
    app.m_zeroWakeAheadUs   = 10;
    app.m_maxCadence        = 8000;
    REQUIRE( app.openProduction() );
    app.m_enabled = true;
    ZeroClock::primary( next );
    REQUIRE( app.m_zeroStats.posted == 3 );
    REQUIRE( app.m_zeroStats.skipped == 0 );
    const auto observed = next + 500000;

    if( fault == 0 )
    {
        ZeroClock::release = false;
    }
    else if( fault == 1 )
    {
        auto record = readerCompletion( streams.images[1], ZeroClock::sequence, observed - 2000, observed - 1000 );
        record.kind = MagAOX::dmCompletion::Kind::sdk;
        record.backendResult = 0;
        sdk.store( record );
        REQUIRE( ImageStreamIO_openIm( &app.m_zeroSdk, "synco_reader_done" ) == 0 );
        synco::GridTime grid;
        uint64_t        period = 0;
        REQUIRE( app.zeroTestGrid( streams.images[0].md->cnt0 + 1, grid, period ) );
        ZeroClock::endpoint    = &sdk;
        ZeroClock::injection   = 6;
        ZeroClock::injectionAt = synco::stepGrid( grid, period, 1, 4 ).ns - 3000;
    }
    else
    {
        ZeroClock::publicationClockDelay = 2000;
    }

    const auto posted = ZeroClock::posts.size();
    ZeroClock::primary( observed );
    CHECK( ZeroClock::posts.size() == posted );
    CHECK( app.m_zeroStats.posted == 3 );

    if( fault == 2 )
    {
        CHECK( app.m_zeroStats.clockRejected == 1 );
        CHECK( app.m_zeroStats.clockExcludedCycles == 1 );
        CHECK( app.m_zeroStats.clockExcludedSlots == 3 );
        CHECK( app.m_zeroStats.warming );
    }
    else
    {
        CHECK( app.m_zeroStats.readerMissing == 3 );
        CHECK( app.m_zeroStats.skipped == 3 );
        CHECK( ZeroClock::injections == ( fault == 1 ? 1 : 0 ) );
    }

    CHECK( app.m_zeroEvents.queued() == 0 );
    CHECK( app.m_zeroTiming.summary().classified == 0 );
    CHECK( app.m_zeroPhase.summary().observed == 0 );
    CHECK( app.m_zeroPublicationTiming.summary().mainObserved == 0 );
    CHECK( app.m_zeroPublicationTiming.summary().auxiliaryObserved == 0 );
    checkZeroPartition( app );

    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    MagAOX::app::dmSyncopation::configureDiagnostics;
    MagAOX::app::dmSyncopation::processZeroTestPrimary;
    MagAOX::app::dmSyncopation::zeroTestFinalGuard;
    MagAOX::app::dmSyncopation::zeroTestClockRejected;
#endif
    // clang-format on
}

/// Qualified clock recovery refreshes health without restarting either continuous or finite scheduling.
TEST_CASE( "qualified scheduler health recovers after a rejected publication clock",
           "[dmSyncopation][production][zeroTest][clockRecovery]" )
{
    Streams            streams;
    ProductionSidecars sidecars;
    ProductionHarness  app;
    const bool         production = GENERATE( false, true );
    ZeroClock::connect( app, streams, 4 );
    const auto next         = ZeroClock::warm();
    app.m_productionEnabled = production;
    app.m_zeroTestEnabled   = !production;
    app.m_zeroLateUs        = 6;
    app.m_zeroWakeAheadUs   = 10;
    app.m_maxCadence        = 8000;

    if( production )
    {
        REQUIRE( app.openProduction() );
    }

    app.m_enabled = true;
    ZeroClock::primary( next );
    REQUIRE( app.m_zeroStats.posted == 3 );
    const auto started      = app.m_zeroStats.startedNs;
    const auto trial        = app.m_zeroTrial;
    const auto activeReason = production ? "production active" : "zero trial active";
    CHECK( std::string( app.m_reason ) == activeReason );

    ZeroClock::publicationClockDelay = 2000;
    ZeroClock::primary( next + 500000 );
    REQUIRE_FALSE( app.m_zeroStats.active );
    REQUIRE( app.m_zeroStats.warming );
    CHECK( std::string( app.m_reason ) ==
           ( production ? "production publication clock unqualified" : "zero publication clock unqualified" ) );

    ZeroClock::primary( next + 1000000 );
    CHECK( app.m_zeroStats.active );
    CHECK_FALSE( app.m_zeroStats.warming );
    CHECK( std::string( app.m_reason ) == activeReason );
    CHECK( app.m_zeroStats.startedNs == started );
    CHECK( app.m_zeroTrial == trial );

    // The combined output is already visible at callback entry, so recovery first establishes a new baseline.
    CHECK( app.m_zeroStats.mainUnmatched == 1 );
    CHECK( app.m_zeroStats.posted == 3 );
    CHECK( app.m_zeroStats.skipped == 3 );
    CHECK( app.m_zeroStats.clockExcludedCycles == 1 );

    ZeroClock::primary( next + 1500000 );
    CHECK( app.m_zeroStats.active );
    CHECK( std::string( app.m_reason ) == activeReason );
    CHECK( app.m_zeroStats.startedNs == started );
    CHECK( app.m_zeroTrial == trial );
    CHECK( app.m_zeroStats.mainMatched == 2 );
    CHECK( app.m_zeroStats.mainUnmatched == 1 );
    CHECK( app.m_zeroStats.posted == 6 );
    CHECK( app.m_zeroStats.skipped == 3 );

    unsigned           starts = 0;
    Harness::ZeroEvent event;

    while( app.m_zeroEvents.pop( event ) )
    {
        starts += event.outcome == int( Harness::ZeroOutcome::started );
    }

    CHECK( starts == 1 );

    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    MagAOX::app::dmSyncopation::processZeroTestPrimary;
    MagAOX::app::dmSyncopation::zeroTestClockRejected;
#endif
    // clang-format on
}

#endif
