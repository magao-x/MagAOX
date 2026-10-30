/** \file syncoSimulation_test.cpp
 * \brief Pure fixed-grid simulation tests; no application, stream, clock or instrument is started.
 * \ingroup dmSyncopation_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "../syncoSimulation.hpp"

#include <limits>

namespace libXWCTest
{
namespace dmSyncopationTest
{
using MagAOX::app::synco::Simulation;

/// \cond TEST_HARNESS
/// Feed a coherent main and its ordinary ten-microsecond-later combined notification.
static void feed( Simulation &simulation /**< [in,out] bounded model */,
                  uint64_t    counter /**< [in] primary and output identity */,
                  int64_t     observed /**< [in] monotonic main observation */,
                  int64_t     interval = 500000 /**< [in] producer-clock publication interval */ )
{
    simulation.primary( counter, observed, interval );
    simulation.combined( counter, counter, observed + 10000 );
}

/// Acquire a stable two-kHz grid and return the most recent actual main time.
static int64_t warm( Simulation &simulation /**< [in,out] configured model */ )
{
    for( uint64_t counter = 1; counter <= 40; ++counter )
    {
        feed( simulation, counter, 1000000000 + int64_t( counter ) * 500000, counter == 1 ? 0 : 500000 );
    }

    return 1020000000;
}

/// Make exact raw combined identity independent of the local scheduling clock.
static MagAOX::dmSyncoSimulation::Observation readyContext( uint64_t counter /**< [in] Combined counter. */ )
{
    MagAOX::dmSyncoSimulation::Observation context;
    context.streamInode   = 55;
    context.publicationNs = 1792000000000000000LL + int64_t( counter ) * 500000;
    return context;
}

/// Construct a released actual-main record; no hypothetical auxiliary command receives this signal.
static MagAOX::dmSyncoSimulation::Readiness readyRecord( uint64_t counter /**< [in] Exact combined identity. */,
                                                         int64_t  main /**< [in] Actual main observation. */,
                                                         int64_t available /**< [in] Coherent capture availability. */ )
{
    MagAOX::dmSyncoSimulation::Readiness event;
    const auto                           context = readyContext( counter );
    auto                                &record  = event.record;
    record.kind                                  = MagAOX::dmCompletion::Kind::readers;
    record.state                                 = MagAOX::dmCompletion::State::released;
    record.session                               = 123;
    record.epoch                                 = 1;
    record.sequence                              = counter;
    record.publisherPid                          = 456;
    record.inputInode                            = context.streamInode;
    record.inputCounter                          = counter;
    record.publicationSec                        = context.publicationNs / 1000000000;
    record.publicationNsec                       = context.publicationNs % 1000000000;
    record.startedNs                             = main + 11000;
    record.finishedNs                            = main + 30000;
    event.observedNs                             = available;
    event.sourceEpoch                            = 1;
    return event;
}

/// Feed only the actual-main and exact combined identities, leaving release availability explicit.
static void readyMain( Simulation &simulation /**< [in,out] Configured model. */,
                       uint64_t    counter /**< [in] Primary and combined counter. */,
                       int64_t     observed /**< [in] Actual main observation. */ )
{
    simulation.primary( counter, observed, counter == 1 ? 0 : 500000 );
    simulation.combined( counter, counter, observed + 10000, true, readyContext( counter ) );
}

/// Warm readiness mode with a forty-microsecond actual release for every exact main command.
static int64_t warmReady( Simulation &simulation /**< [in,out] Configured model. */ )
{
    for( uint64_t counter = 1; counter <= 40; ++counter )
    {
        const int64_t now = 1000000000 + int64_t( counter ) * 500000;
        readyMain( simulation, counter, now );
        simulation.release( readyRecord( counter, now, now + 40000 ) );
    }

    return 1020000000;
}
/// \endcond

/** \brief Constant cadence acquires a pre-observation grid and evaluates each fixed slot once. */
TEST_CASE( "synco simulation constant grid", "[dmSyncopation][simulation]" )
{
    Simulation simulation;
    REQUIRE( simulation.configure( {} ) );
    const auto  last    = warm( simulation );
    const auto &summary = simulation.summary();
    REQUIRE( summary.enabled );
    REQUIRE( summary.valid );
    REQUIRE( summary.feasible );
    REQUIRE( summary.gridNs == last );
    REQUIRE( summary.periodNs == 500000 );
    REQUIRE( summary.periodSamples == 32 );
    REQUIRE( summary.phaseSamples == 32 );
    REQUIRE( summary.phaseCount > 0 );
    REQUIRE( summary.phaseNs == 0 );
    REQUIRE( summary.nextGridNs == last + 500000 );
    REQUIRE( summary.nextActiveNs == last + 250000 );
    REQUIRE( summary.window == 32 );
    REQUIRE( summary.copyNs == 10000 );
    const auto active  = summary.outcomes[1].wouldRun;
    const auto passive = summary.outcomes[0].wouldRun;
    simulation.advance( last + 475000 );
    REQUIRE( summary.outcomes[1].wouldRun == active + 1 );
    REQUIRE( summary.outcomes[0].wouldRun == passive + 1 );
    const auto considered = summary.outcomes[1].considered;
    simulation.advance( last + 480000 );
    REQUIRE( summary.outcomes[1].considered == considered );
    REQUIRE( summary.outcomes[1].overlap == 0 );
}

/** \brief An early main skips passive preparation without suppressing its independently safe active slot. */
TEST_CASE( "synco simulation independent passive skip", "[dmSyncopation][simulation]" )
{
    Simulation simulation;
    REQUIRE( simulation.configure( {} ) );
    const auto last = warm( simulation );
    simulation.advance( last + 300000 );
    const auto active = simulation.summary().outcomes[1].wouldRun;
    const auto early  = simulation.summary().outcomes[0].earlyMain;
    feed( simulation, 41, last + 420000, 420000 );
    REQUIRE( simulation.summary().counter == 41 );
    REQUIRE( simulation.summary().gridNs == last + 500000 );
    REQUIRE( simulation.summary().phaseNs == -80000 );
    REQUIRE( simulation.summary().nextActiveNs == last + 750000 );
    REQUIRE( simulation.summary().outcomes[0].earlyMain == early + 1 );
    simulation.advance( last + 760000 );
    REQUIRE( simulation.summary().outcomes[1].wouldRun == active + 1 );
    feed( simulation, 42, last + 1000000, 580000 );
    REQUIRE( simulation.summary().periodNs == 500000 );
    REQUIRE( simulation.summary().gridNs == last + 1000000 );
    REQUIRE( simulation.summary().phaseNs == 0 );
}

/** \brief A late main leaves passive preparation complete but its actual reader hold can veto the fixed target. */
TEST_CASE( "synco simulation late main hold", "[dmSyncopation][simulation]" )
{
    Simulation simulation;
    REQUIRE( simulation.configure( {} ) );
    const auto last = warm( simulation );
    simulation.advance( last + 475000 );
    const auto passive = simulation.summary().outcomes[0].wouldRun;
    const auto active  = simulation.summary().outcomes[1].wouldRun;
    const auto held    = simulation.summary().outcomes[1].readerHold;
    feed( simulation, 41, last + 580000, 580000 );
    REQUIRE( simulation.summary().outcomes[0].wouldRun == passive );
    REQUIRE( simulation.summary().nextActiveNs == last + 750000 );
    simulation.advance( last + 760000 );
    REQUIRE( simulation.summary().outcomes[1].wouldRun == active );
    REQUIRE( simulation.summary().outcomes[1].readerHold == held + 1 );
}

/** \brief Large measured late/catch-up intervals retain counter labels and the nominal seed rather than nearest-slot
 * assignment. */
TEST_CASE( "synco simulation large catchup", "[dmSyncopation][simulation]" )
{
    Simulation simulation;
    REQUIRE( simulation.configure( {} ) );
    const auto last = warm( simulation );
    feed( simulation, 41, last + 885783, 885783 );
    REQUIRE( simulation.summary().counter == 41 );
    REQUIRE( simulation.summary().gridNs == last + 500000 );
    REQUIRE( simulation.summary().phaseNs == 385783 );
    REQUIRE( simulation.summary().periodNs == 500000 );
    const auto excluded = simulation.summary().excluded;
    feed( simulation, 42, last + 1087895, 202112 );
    REQUIRE( simulation.summary().counter == 42 );
    REQUIRE( simulation.summary().gridNs == last + 1000000 );
    REQUIRE( simulation.summary().phaseNs == 87895 );
    REQUIRE( simulation.summary().periodNs == 500000 );
    REQUIRE( simulation.summary().excluded == excluded + 1 );
    REQUIRE( simulation.summary().nextActiveNs == last + 1250000 );
    REQUIRE( simulation.summary().gaps == 0 );
    REQUIRE( simulation.summary().resets == 0 );
    simulation.advance( last + 1260000 );
    REQUIRE( simulation.summary().outcomes[1].readerHold > 0 );
}

/** \brief Missing real mains or coherent output acknowledgements veto independent active slots. */
TEST_CASE( "synco simulation requires actual association", "[dmSyncopation][simulation]" )
{
    Simulation simulation;
    REQUIRE( simulation.configure( {} ) );
    const auto last = warm( simulation );

    SECTION( "no main is manufactured at its grid time" )
    {
        simulation.advance( last + 760000 );
        REQUIRE( simulation.summary().outcomes[1].missingMain > 0 );
        REQUIRE( simulation.summary().counter == 40 );
    }

    SECTION( "a main without a linked combination is insufficient" )
    {
        simulation.primary( 41, last + 500000, 500000 );
        const auto missing = simulation.summary().outcomes[1].missingOutput;
        simulation.advance( last + 760000 );
        REQUIRE( simulation.summary().outcomes[1].missingOutput == missing + 1 );
    }

    SECTION( "incoherent output cannot acknowledge a main" )
    {
        simulation.primary( 41, last + 500000, 500000 );
        simulation.combined( 41, 41, last + 510000, false );
        simulation.advance( last + 760000 );
        REQUIRE( simulation.summary().outputGaps == 1 );
        REQUIRE( simulation.summary().outcomes[1].missingOutput > 0 );
    }
}

/** \brief Forward gaps retain period knowledge, duplicates do not retrain, and rollback begins a clean session. */
TEST_CASE( "synco simulation counter lifecycle", "[dmSyncopation][simulation]" )
{
    Simulation simulation;
    REQUIRE( simulation.configure( {} ) );
    const auto last = warm( simulation );
    simulation.primary( 40, last + 20000, 500000 );
    REQUIRE( simulation.summary().duplicates == 1 );
    REQUIRE( simulation.summary().sequence == 40 );
    feed( simulation, 43, last + 1500000, 1500000 );
    REQUIRE( simulation.summary().gaps == 1 );
    REQUIRE( simulation.summary().periodSamples == 32 );
    REQUIRE( simulation.summary().periodNs == 500000 );
    REQUIRE( simulation.summary().gridNs == last + 1500000 );
    feed( simulation, 44, last + 2000000 );
    REQUIRE( simulation.summary().valid );
    const auto session = simulation.summary().session;
    simulation.primary( 1, last + 3000000, 500000 );
    REQUIRE( simulation.summary().session == session + 1 );
    REQUIRE( simulation.summary().resets == 1 );
    REQUIRE( simulation.summary().sequence == 1 );
    REQUIRE( simulation.summary().periodSamples == 0 );
    REQUIRE_FALSE( simulation.summary().valid );
    REQUIRE( simulation.summary().outcomes[1].considered == 0 );
}

/** \brief Prediction commitments precede observations, and phase learning changes only newly committed future cycles.
 */
TEST_CASE( "synco simulation frozen horizon", "[dmSyncopation][simulation]" )
{
    Simulation         simulation;
    Simulation::Config config;
    config.window    = 4;
    config.phaseGain = 1;
    config.phaseSlew = 1000;
    REQUIRE( simulation.configure( config ) );
    auto last = warm( simulation );

    for( uint64_t counter = 41; counter <= 46; ++counter )
    {
        const auto committed = simulation.summary().nextGridNs;
        last += 500000;
        feed( simulation, counter, last + 20000 );
        REQUIRE( simulation.summary().gridNs == committed );
        REQUIRE( simulation.summary().nextActiveNs == committed + 250000 );
        REQUIRE( simulation.summary().nextGridNs >= committed + 500000 );
        REQUIRE( simulation.summary().nextGridNs <= committed + 501000 );
    }

    REQUIRE( simulation.summary().phaseCount > 0 );
    REQUIRE( simulation.summary().nextGridNs > last + 500000 );
}

/** \brief Future coherent outputs tolerate monitor notification order without inventing a main or double-counting a
 * link. */
TEST_CASE( "synco simulation future output buffering", "[dmSyncopation][simulation]" )
{
    Simulation simulation;
    REQUIRE( simulation.configure( {} ) );
    const auto last         = warm( simulation );
    const auto associations = simulation.summary().associations;
    simulation.combined( 41, 41, last + 500000 );
    REQUIRE( simulation.summary().associations == associations );
    REQUIRE( simulation.summary().counter == 40 );
    simulation.combined( 41, 41, last + 501000 );
    simulation.primary( 41, last + 502000, 500000 );
    REQUIRE( simulation.summary().associations == associations + 1 );
    REQUIRE( simulation.summary().outputGaps == 0 );
    const auto active = simulation.summary().outcomes[1].wouldRun;
    simulation.advance( last + 760000 );
    REQUIRE( simulation.summary().outcomes[1].wouldRun == active + 1 );
}

/** \brief A coherent future output proves its main happened before the primary monitor catches up. */
TEST_CASE( "synco simulation future output cancels passive copy", "[dmSyncopation][simulation]" )
{
    Simulation simulation;
    REQUIRE( simulation.configure( {} ) );
    const auto last    = warm( simulation );
    const auto passive = simulation.summary().outcomes[0].wouldRun;
    const auto early   = simulation.summary().outcomes[0].earlyMain;
    simulation.combined( 41, 41, last + 430000 );
    simulation.advance( last + 475000 );
    REQUIRE( simulation.summary().counter == 40 );
    REQUIRE( simulation.summary().outcomes[0].wouldRun == passive );
    REQUIRE( simulation.summary().outcomes[0].earlyMain == early + 1 );
    simulation.primary( 41, last + 480000, 500000 );
    const auto active = simulation.summary().outcomes[1].wouldRun;
    simulation.advance( last + 760000 );
    REQUIRE( simulation.summary().outcomes[1].wouldRun == active + 1 );
}

/** \brief Known future-main output suppresses an older unstarted active slot without inventing a main observation. */
TEST_CASE( "synco simulation future output expires older active work", "[dmSyncopation][simulation]" )
{
    Simulation simulation;
    REQUIRE( simulation.configure( {} ) );
    const auto last    = warm( simulation );
    const auto active  = simulation.summary().outcomes[1].wouldRun;
    const auto expired = simulation.summary().outcomes[1].expired;
    simulation.combined( 41, 41, last + 200000 );
    simulation.advance( last + 300000 );
    REQUIRE( simulation.summary().counter == 40 );
    REQUIRE( simulation.summary().outcomes[1].wouldRun == active );
    REQUIRE( simulation.summary().outcomes[1].expired == expired + 1 );
    simulation.primary( 41, last + 420000, 420000 );
    simulation.advance( last + 760000 );
    REQUIRE( simulation.summary().outcomes[1].wouldRun == active + 1 );
}

/** \brief Future-output observations preserve modeled overlap evidence even if the primary callback arrives later. */
TEST_CASE( "synco simulation future output overlap evidence", "[dmSyncopation][simulation]" )
{
    Simulation simulation;
    REQUIRE( simulation.configure( {} ) );
    const auto last = warm( simulation );

    SECTION( "future output arrives during an active reader hold" )
    {
        feed( simulation, 41, last + 500000 );
        simulation.advance( last + 760000 );
        const auto overlap = simulation.summary().outcomes[1].overlap;
        simulation.combined( 42, 42, last + 800000 );
        REQUIRE( simulation.summary().counter == 41 );
        REQUIRE( simulation.summary().outcomes[1].overlap == overlap + 1 );
        simulation.primary( 42, last + 950000, 450000 );
        REQUIRE( simulation.summary().outcomes[1].overlap == overlap + 1 );
    }

    SECTION( "future output arrives during a passive modeled copy" )
    {
        const auto overlap = simulation.summary().outcomes[0].overlap;
        simulation.combined( 41, 41, last + 465000 );
        REQUIRE( simulation.summary().counter == 40 );
        REQUIRE( simulation.summary().outcomes[0].overlap == overlap + 1 );
        simulation.primary( 41, last + 480000, 480000 );
        REQUIRE( simulation.summary().outcomes[0].overlap == overlap + 1 );
    }
}

/** \brief Ambiguous or incoherent output progression poisons buffered future acknowledgements. */
TEST_CASE( "synco simulation rejects ambiguous future links", "[dmSyncopation][simulation]" )
{
    Simulation simulation;
    REQUIRE( simulation.configure( {} ) );
    const auto last         = warm( simulation );
    const auto associations = simulation.summary().associations;
    simulation.combined( 41, 41, last + 430000 );

    SECTION( "different output counter repeats a future primary" )
    {
        simulation.combined( 42, 41, last + 435000 );
        simulation.combined( 42, 41, last + 436000 );
    }

    SECTION( "a later incoherent output invalidates earlier pending links" )
    {
        simulation.combined( 42, 42, last + 435000, false );
    }

    SECTION( "a forward output gap invalidates earlier pending links" )
    {
        simulation.combined( 43, 42, last + 435000 );
    }

    REQUIRE( simulation.summary().outputGaps == 1 );
    simulation.primary( 41, last + 480000, 500000 );
    REQUIRE( simulation.summary().associations == associations );
    const auto missing = simulation.summary().outcomes[1].missingOutput;
    simulation.advance( last + 760000 );
    REQUIRE( simulation.summary().outcomes[1].missingOutput == missing + 1 );
}

/** \brief Producer-interval noise remains diagnostic and cannot retime an advertised monotonic cycle. */
TEST_CASE( "synco simulation diagnostic intervals leave frozen cadence", "[dmSyncopation][simulation]" )
{
    Simulation simulation;
    REQUIRE( simulation.configure( {} ) );
    const auto last = warm( simulation );
    feed( simulation, 41, last + 500000, 300000 );
    feed( simulation, 42, last + 1000000, 300000 );
    REQUIRE( simulation.summary().periodNs == 500000 );
    REQUIRE( simulation.summary().feasible );
    const auto active = simulation.summary().outcomes[1].wouldRun;
    simulation.advance( last + 1260000 );
    REQUIRE( simulation.summary().outcomes[1].wouldRun == active + 1 );
}

/** \brief A cadence-rating limit can reject an otherwise structurally feasible nominal request. */
TEST_CASE( "synco simulation nominal cadence rating", "[dmSyncopation][simulation]" )
{
    Simulation         simulation;
    Simulation::Config config;
    config.cadenceMinPeriod = 600000;
    REQUIRE( MagAOX::app::synco::feasible( 500000, 2, config.bounds ) );
    REQUIRE( simulation.configure( config ) );
    const auto last = warm( simulation );
    REQUIRE( simulation.summary().valid );
    REQUIRE_FALSE( simulation.summary().feasible );
    REQUIRE( simulation.summary().cadenceMinPeriodNs == 600000 );
    simulation.advance( last + 760000 );
    REQUIRE( simulation.summary().outcomes[1].wouldRun == 0 );
    REQUIRE( simulation.summary().outcomes[1].expired > 0 );
    config.cadenceMinPeriod = -1;
    REQUIRE_FALSE( simulation.configure( config ) );
    config.cadenceMinPeriod = 1000000000001LL;
    REQUIRE_FALSE( simulation.configure( config ) );
}

/** \brief New rating-incompatible estimates do not retroactively reject an already committed admissible cycle. */
TEST_CASE( "synco simulation frozen cadence qualification", "[dmSyncopation][simulation]" )
{
    Simulation         simulation;
    Simulation::Config config;
    config.window           = 2;
    config.phaseGain        = 0;
    config.bounds.keepout   = 50000;
    config.cadenceMinPeriod = 450000;
    config.cadenceWindowNs  = 200000;
    config.cadenceUpdateNs  = 200000;
    REQUIRE( simulation.configure( config ) );
    const auto last = warm( simulation );
    simulation.advance( last + 300000 );
    const auto active = simulation.summary().outcomes[1].wouldRun;
    feed( simulation, 41, last + 400000, 400000 );
    REQUIRE( simulation.summary().periodNs == 400000 );
    REQUIRE_FALSE( simulation.summary().feasible );
    simulation.advance( last + 760000 );
    REQUIRE( simulation.summary().outcomes[1].wouldRun == active + 1 );
    feed( simulation, 42, last + 800000, 400000 );
    simulation.advance( last + 1110000 );
    REQUIRE( simulation.summary().outcomes[1].wouldRun == active + 1 );
    REQUIRE( simulation.summary().outcomes[1].expired > 0 );
}

/** \brief Expiry clears buffered outputs so old evidence cannot resurrect an acknowledgement. */
TEST_CASE( "synco simulation expiry clears future links", "[dmSyncopation][simulation]" )
{
    Simulation simulation;
    REQUIRE( simulation.configure( {} ) );
    const auto last         = warm( simulation );
    const auto associations = simulation.summary().associations;
    simulation.combined( 41, 41, last + 500000 );
    simulation.advance( last + 4000001 );
    REQUIRE_FALSE( simulation.summary().valid );
    simulation.primary( 41, last + 4500000, 0 );
    REQUIRE( simulation.summary().associations == associations );
    REQUIRE_FALSE( simulation.summary().valid );
}

/** \brief Model-bound violations remain explicit overlaps, while silence expires all pending predictions. */
TEST_CASE( "synco simulation overlap and expiry", "[dmSyncopation][simulation]" )
{
    Simulation simulation;
    REQUIRE( simulation.configure( {} ) );
    const auto last = warm( simulation );

    SECTION( "main arrives during modeled active reader lifetime" )
    {
        const auto overlap = simulation.summary().outcomes[1].overlap;
        feed( simulation, 41, last + 270000, 270000 );
        REQUIRE( simulation.summary().outcomes[1].overlap == overlap + 1 );
        REQUIRE( simulation.summary().counter == 41 );
    }

    SECTION( "main arrives inside modeled passive batch copy" )
    {
        const auto overlap = simulation.summary().outcomes[0].overlap;
        feed( simulation, 41, last + 465000, 465000 );
        REQUIRE( simulation.summary().outcomes[0].overlap == overlap + 1 );
    }

    SECTION( "silence and primary clock regression start fresh estimation" )
    {
        simulation.advance( last + 4000001 );
        REQUIRE_FALSE( simulation.summary().valid );
        REQUIRE( simulation.summary().nextActiveNs == 0 );
        REQUIRE( simulation.summary().periodSamples == 0 );
        const auto session = simulation.summary().session;
        simulation.primary( 41, last + 1000000, 500000 );
        REQUIRE( simulation.summary().invalid == 1 );
        REQUIRE( simulation.summary().session == session + 1 );
        REQUIRE( simulation.summary().counter == 41 );
        REQUIRE( simulation.summary().cadence.anchors == 1 );
        REQUIRE_FALSE( simulation.summary().valid );
    }
}

/** \brief Slow cadence refresh and unavailable producer intervals do not expire fresh primary observations. */
TEST_CASE( "synco simulation observation freshness independent of cadence refresh", "[dmSyncopation][simulation]" )
{
    Simulation simulation;
    REQUIRE( simulation.configure( {} ) );
    auto observed = warm( simulation );

    for( uint64_t counter = 41; counter <= 2050; ++counter )
    {
        observed += 500000;
        feed( simulation, counter, observed, 0 );
    }

    REQUIRE( simulation.summary().valid );
    REQUIRE( simulation.summary().periodSamples == 32 );
    REQUIRE( simulation.summary().phaseSamples == 32 );
    REQUIRE( simulation.summary().nextActiveNs > observed );
    REQUIRE( simulation.summary().cadence.updatedNs < observed - 4000000 );
    REQUIRE( simulation.summary().cadence.observationNs == observed );
    REQUIRE( simulation.summary().cadence.updates == 1 );
}

/** \brief Accepted configuration changes start a distinct session; rejected changes preserve the current one. */
TEST_CASE( "synco simulation configuration sessions", "[dmSyncopation][simulation]" )
{
    Simulation simulation;
    REQUIRE( simulation.configure( {} ) );
    warm( simulation );
    const auto         session = simulation.summary().session;
    Simulation::Config config;
    config.steps = 1;
    REQUIRE( simulation.configure( config ) );
    REQUIRE( simulation.summary().session == session + 1 );
    REQUIRE( simulation.summary().sequence == 0 );
    REQUIRE_FALSE( simulation.summary().valid );
    REQUIRE( simulation.summary().steps == 1 );
    REQUIRE( simulation.summary().outcomes[1].considered == 0 );
    config.steps = 17;
    REQUIRE_FALSE( simulation.configure( config ) );
    REQUIRE( simulation.summary().session == session + 1 );
    REQUIRE( simulation.summary().steps == 1 );
}

/** \brief Invalid models are rejected and large timestamps fail closed without signed overflow. */
TEST_CASE( "synco simulation validates bounded arithmetic", "[dmSyncopation][simulation]" )
{
    Simulation         simulation;
    Simulation::Config config;
    config.window = 257;
    REQUIRE_FALSE( simulation.configure( config ) );
    config           = {};
    config.phaseGain = std::numeric_limits<double>::quiet_NaN();
    REQUIRE_FALSE( simulation.configure( config ) );
    config             = {};
    config.bounds.copy = std::numeric_limits<int64_t>::max();
    REQUIRE_FALSE( simulation.configure( config ) );
    config                = {};
    config.bounds.keepout = -1;
    REQUIRE_FALSE( simulation.configure( config ) );
    config       = {};
    config.steps = 17;
    REQUIRE_FALSE( simulation.configure( config ) );
    config        = {};
    config.window = 2;
    REQUIRE( simulation.configure( config ) );
    const auto end = std::numeric_limits<int64_t>::max();
    feed( simulation, 1, end - 1200000, 0 );
    feed( simulation, 2, end - 700000 );
    feed( simulation, 3, end - 200000 );
    REQUIRE_FALSE( simulation.summary().valid );
    REQUIRE( simulation.summary().invalid > 0 );
    CHECK( simulation.estimatorStatus().observed == 3 );
    CHECK( simulation.estimatorStatus().history == 3 );
}

/** \brief Trace events preserve frozen copy/reader boundaries, identity and unconverted caller-clock context. */
TEST_CASE( "synco simulation overlap trace windows", "[dmSyncopation][simulation]" )
{
    using namespace MagAOX::dmSyncoSimulation;
    Simulation simulation;
    REQUIRE( simulation.configure( {} ) );
    const auto  last = warm( simulation );
    Observation context;
    context.streamInode          = 71;
    context.triggerInode         = 72;
    context.receiptNs            = last + 460000;
    context.callbackNs           = last + 461000;
    context.lockRequestNs        = last + 462000;
    context.lockAcquiredNs       = last + 463000;
    context.publicationNs        = 1800000000000000000LL;
    context.primaryPublicationNs = 1800000000000000100LL;
    context.startNs              = 1800000000000000200LL;
    context.updateNs             = 1800000000000000300LL;
    OverlapEvent event;

    SECTION( "passive copy evidence preserves both cycle identities and producer clocks" )
    {
        simulation.primary( 41, last + 465000, 465000, context );
        REQUIRE( simulation.popOverlap( event ) );
        CHECK( event.id == 1 );
        CHECK( event.session == simulation.summary().session );
        CHECK( event.mainCounter == 41 );
        CHECK( event.combinedCounter == 40 );
        CHECK( event.planCounter == 41 );
        CHECK( event.lastMainCounter == 40 );
        CHECK( event.planEstimatorId == 40 );
        CHECK( event.lastEstimatorId == 40 );
        CHECK( event.step == 0 );
        CHECK( event.origin == OverlapOrigin::primary );
        CHECK( event.window == OverlapWindow::copy );
        CHECK( event.observedNs == last + 465000 );
        CHECK( event.mainGridNs == last + 500000 );
        CHECK( event.planGridNs == event.mainGridNs );
        CHECK( event.periodNs == 500000 );
        CHECK( event.periodQ32 == uint64_t( 500000 ) * periodScale );
        CHECK( event.planFractionQ32 == 0 );
        CHECK( event.copyBeginNs == last + 462000 );
        CHECK( event.copyEndNs == last + 472000 );
        CHECK( event.readerEndNs == 0 );
        CHECK( event.lastMainNs == last );
        CHECK( event.lastOutputNs == last + 10000 );
        CHECK( event.observation.streamInode == context.streamInode );
        CHECK( event.observation.triggerInode == context.triggerInode );
        CHECK( event.observation.receiptNs == context.receiptNs );
        CHECK( event.observation.callbackNs == context.callbackNs );
        CHECK( event.observation.lockRequestNs == context.lockRequestNs );
        CHECK( event.observation.lockAcquiredNs == context.lockAcquiredNs );
        CHECK( event.observation.publicationNs == context.publicationNs );
        CHECK( event.observation.primaryPublicationNs == context.primaryPublicationNs );
        CHECK( event.observation.startNs == context.startNs );
        CHECK( event.observation.updateNs == context.updateNs );
        CHECK_FALSE( simulation.popOverlap( event ) );
    }

    SECTION( "active copy end begins the reader-hold classification" )
    {
        simulation.advance( last + 240000 );
        simulation.primary( 41, last + 250000, 250000 );
        REQUIRE( simulation.popOverlap( event ) );
        CHECK( event.step == 1 );
        CHECK( event.planCounter == 40 );
        CHECK( event.planGridNs == last );
        CHECK( event.mainGridNs == last + 500000 );
        CHECK( event.window == OverlapWindow::readerHold );
        CHECK( event.copyBeginNs == last + 240000 );
        CHECK( event.copyEndNs == last + 250000 );
        CHECK( event.readerEndNs == last + 405000 );
        CHECK( event.observation.publicationNs == 0 );
    }

    SECTION( "active copy start is included once that modeled copy has started" )
    {
        simulation.advance( last + 240000 );
        simulation.primary( 41, last + 240000, 240000 );
        REQUIRE( simulation.popOverlap( event ) );
        CHECK( event.window == OverlapWindow::copy );
        CHECK( event.step == 1 );
    }

    SECTION( "passive copy end is excluded" )
    {
        simulation.primary( 41, last + 472000, 472000 );
        CHECK_FALSE( simulation.popOverlap( event ) );
    }

    SECTION( "active reader end is excluded" )
    {
        simulation.advance( last + 240000 );
        simulation.primary( 41, last + 405000, 405000 );
        CHECK_FALSE( simulation.popOverlap( event ) );
    }
}

/** \brief Both monitor origins remain available for offline deduplication, with distinct counters and steps retained.
 */
TEST_CASE( "synco simulation overlap trace identities", "[dmSyncopation][simulation]" )
{
    using namespace MagAOX::dmSyncoSimulation;
    Simulation simulation;
    REQUIRE( simulation.configure( {} ) );
    const auto last = warm( simulation );
    simulation.combined( 41, 41, last + 463000 );
    simulation.primary( 41, last + 464000, 464000 );
    OverlapEvent combined, primary;
    REQUIRE( simulation.popOverlap( combined ) );
    REQUIRE( simulation.popOverlap( primary ) );
    CHECK( combined.origin == OverlapOrigin::combined );
    CHECK( primary.origin == OverlapOrigin::primary );
    CHECK( combined.id + 1 == primary.id );
    CHECK( combined.session == primary.session );
    CHECK( combined.planCounter == primary.planCounter );
    CHECK( combined.step == primary.step );
    CHECK( combined.mainCounter == primary.mainCounter );
    CHECK( combined.mainGridNs == primary.mainGridNs );
    CHECK( combined.combinedCounter == primary.combinedCounter );
    CHECK( combined.lastMainCounter == 40 );
    CHECK( primary.lastMainCounter == 40 );
    CHECK( combined.planEstimatorId == 40 );
    CHECK( primary.planEstimatorId == combined.planEstimatorId );
    CHECK( combined.lastEstimatorId == 40 );
    CHECK( primary.lastEstimatorId == combined.lastEstimatorId );

    simulation.advance( last + 745000 );
    const auto nextGrid = simulation.summary().nextGridNs;
    simulation.primary( 42, last + 746000, 500000 );
    simulation.primary( 43, last + 748000, 500000 );
    OverlapEvent second, third;
    REQUIRE( simulation.popOverlap( second ) );
    REQUIRE( simulation.popOverlap( third ) );
    CHECK( second.planCounter == 41 );
    CHECK( second.step == 1 );
    CHECK( second.mainCounter == 42 );
    CHECK( second.mainGridNs == nextGrid );
    CHECK( third.planCounter == second.planCounter );
    CHECK( third.step == second.step );
    CHECK( third.mainCounter == 43 );
    CHECK( third.id == second.id + 1 );
    CHECK( second.planEstimatorId == 40 );
    CHECK( second.lastEstimatorId == 41 );
    CHECK( third.planEstimatorId == second.planEstimatorId );
    CHECK( third.lastEstimatorId == 42 );
    CHECK_FALSE( simulation.popOverlap( third ) );
}

/** \brief Saturation drops new records, draining recovers capacity, and session changes preserve FIFO accounting. */
TEST_CASE( "synco simulation overlap trace lifetime", "[dmSyncopation][simulation]" )
{
    using namespace MagAOX::dmSyncoSimulation;
    Simulation   retained, drained;
    OverlapEvent event;

    for( uint64_t id = 1; id <= traceCapacity + 5; ++id )
    {
        REQUIRE( retained.configure( {} ) );
        REQUIRE( drained.configure( {} ) );
        const auto last = warm( retained );
        REQUIRE( warm( drained ) == last );
        retained.primary( 41, last + 465000, 465000 );
        drained.primary( 41, last + 465000, 465000 );
        REQUIRE( drained.popOverlap( event ) );
        CHECK( event.id == id );
        CHECK( event.session == id );
        const auto status = retained.traceStatus();
        CHECK( status.enabled );
        CHECK( status.capacity == traceCapacity );
        CHECK( status.detected == id );
        CHECK( status.detected == status.drained + status.dropped + status.queued );
        CHECK( status.logFailed == 0 );

        const auto &a = retained.summary();
        const auto &b = drained.summary();
        CHECK( a.valid == b.valid );
        CHECK( a.feasible == b.feasible );
        CHECK( a.periodNs == b.periodNs );
        CHECK( a.gridNs == b.gridNs );
        CHECK( a.nextGridNs == b.nextGridNs );
        CHECK( a.nextActiveNs == b.nextActiveNs );
        CHECK( a.phaseNs == b.phaseNs );
        CHECK( a.sequence == b.sequence );
        CHECK( a.associations == b.associations );

        for( size_t step = 0; step < maxSteps; ++step )
        {
            CHECK( a.outcomes[step].considered == b.outcomes[step].considered );
            CHECK( a.outcomes[step].wouldRun == b.outcomes[step].wouldRun );
            CHECK( a.outcomes[step].skipped == b.outcomes[step].skipped );
            CHECK( a.outcomes[step].overlap == b.outcomes[step].overlap );
        }
    }

    auto status = retained.traceStatus();
    REQUIRE( status.queued == traceCapacity );
    REQUIRE( status.dropped == 5 );
    REQUIRE( status.drained == 0 );

    for( uint64_t id = 1; id <= 3; ++id )
    {
        REQUIRE( retained.popOverlap( event ) );
        CHECK( event.id == id );
        CHECK( event.session == id );
    }

    retained.reset();
    status = retained.traceStatus();
    CHECK( status.queued == traceCapacity - 3 );
    CHECK( status.drained == 3 );
    CHECK( status.dropped == 5 );
    const auto last = warm( retained );
    retained.primary( 41, last + 465000, 465000 );

    for( uint64_t id = 4; id <= traceCapacity; ++id )
    {
        REQUIRE( retained.popOverlap( event ) );
        CHECK( event.id == id );
        CHECK( event.session == id );
    }

    REQUIRE( retained.popOverlap( event ) );
    CHECK( event.id == traceCapacity + 6 );
    CHECK( event.session == traceCapacity + 6 );
    CHECK_FALSE( retained.popOverlap( event ) );
    status = retained.traceStatus();
    CHECK( status.queued == 0 );
    CHECK( status.drained == traceCapacity + 1 );
    CHECK( status.detected == status.drained + status.dropped + status.queued );
    CHECK( drained.traceStatus().dropped == 0 );
    CHECK( drained.traceStatus().queued == 0 );
}

/** \brief Overlap-triggered estimator collection retains warming history, the committing source and exactly 64 updates.
 */
TEST_CASE( "synco simulation estimator pre and post history", "[dmSyncopation][simulation]" )
{
    using namespace MagAOX::dmSyncoSimulation;
    Simulation simulation;
    REQUIRE( simulation.configure( {} ) );
    const auto last   = warm( simulation );
    auto       status = simulation.estimatorStatus();
    CHECK( status.observed == 40 );
    CHECK( status.history == 40 );
    CHECK( status.triggers == 0 );
    CHECK( status.queued == 0 );
    EstimatorSample sample;
    CHECK_FALSE( simulation.popEstimator( sample ) );

    simulation.primary( 40, last + 20000, 500000 );
    simulation.primary( 41, 0, 500000 );
    CHECK( simulation.estimatorStatus().observed == 40 );
    feed( simulation, 41, last + 465000, 465000 );
    status = simulation.estimatorStatus();
    CHECK( status.observed == 41 );
    CHECK( status.triggers == 1 );
    CHECK( status.selected == 41 );
    CHECK( status.postRemaining == estimatorPostSamples - 1 );

    for( uint64_t id = 1; id <= 41; ++id )
    {
        REQUIRE( simulation.popEstimator( sample ) );
        CHECK( sample.id == id );
        CHECK( sample.counter == id );
        CHECK( sample.sequence == id );
        CHECK( sample.session == simulation.summary().session );
        CHECK( sample.phaseGain == 0.1 );
        CHECK( sample.phaseSlewNs == 1000 );
        CHECK( sample.periodBeforeNs == 500000 );
        CHECK( sample.periodNs == 500000 );
        CHECK( sample.sequential == ( id > 1 ) );
        CHECK( sample.intervalAccepted == ( id > 1 ) );
        CHECK( sample.phaseSamples == std::min<uint64_t>( id, 32 ) );
        CHECK( sample.periodSamples == std::min<uint64_t>( id - 1, 32 ) );

        if( id < 32 )
        {
            CHECK_FALSE( sample.committed );
            CHECK( sample.nextCounter == 0 );
            CHECK( sample.sourcePlanCounter == 0 );
            CHECK( sample.sourceEstimatorId == 0 );
            CHECK( sample.sourceGridNs == 0 );
            CHECK( sample.baseGridNs == 0 );
            CHECK( sample.candidateNs == 0 );
            CHECK( sample.rawCorrectionNs == 0 );
            CHECK( sample.appliedCorrectionNs == 0 );
            CHECK( sample.committedGridNs == 0 );
        }
        else
        {
            CHECK( sample.committed );
            CHECK( sample.nextCounter == id + 1 );
            CHECK( sample.sourcePlanCounter == id );
            CHECK( sample.sourceEstimatorId == ( id == 32 ? id : id - 1 ) );
            CHECK( sample.sourceGridNs == 1000000000 + int64_t( id ) * 500000 );
            CHECK( sample.baseGridNs == sample.sourceGridNs + sample.periodNs );
            CHECK( sample.candidateNs == sample.baseGridNs );
            CHECK( sample.rawCorrectionNs == 0 );
            CHECK( sample.appliedCorrectionNs == 0 );
            CHECK( sample.committedGridNs == sample.baseGridNs );
        }
    }

    for( uint64_t counter = 42; counter <= 105; ++counter )
    {
        feed( simulation, counter, last + int64_t( counter - 40 ) * 500000, counter == 42 ? 535000 : 500000 );
    }

    status = simulation.estimatorStatus();
    CHECK( status.observed == 105 );
    CHECK( status.selected == 104 );
    CHECK( status.drained == 41 );
    CHECK( status.queued == 63 );
    CHECK( status.postRemaining == 0 );
    CHECK( status.selected == status.drained + status.dropped + status.queued );

    for( uint64_t id = 42; id <= 104; ++id )
    {
        REQUIRE( simulation.popEstimator( sample ) );
        CHECK( sample.id == id );
    }

    CHECK_FALSE( simulation.popEstimator( sample ) );
}

/** \brief Both observer origins merge overlapping history windows without duplicating selected estimator updates. */
TEST_CASE( "synco simulation estimator trigger merging", "[dmSyncopation][simulation]" )
{
    using namespace MagAOX::dmSyncoSimulation;
    Simulation simulation;
    REQUIRE( simulation.configure( {} ) );
    const auto last = warm( simulation );
    simulation.combined( 41, 41, last + 463000 );
    CHECK( simulation.estimatorStatus().postRemaining == estimatorPostSamples );
    simulation.primary( 41, last + 464000, 464000 );
    auto status = simulation.estimatorStatus();
    CHECK( status.triggers == 2 );
    CHECK( status.selected == 41 );
    CHECK( status.postRemaining == estimatorPostSamples - 1 );

    OverlapEvent combined, primary;
    REQUIRE( simulation.popOverlap( combined ) );
    REQUIRE( simulation.popOverlap( primary ) );
    CHECK( combined.planEstimatorId == 40 );
    CHECK( primary.planEstimatorId == 40 );
    CHECK( combined.lastEstimatorId == 40 );
    CHECK( primary.lastEstimatorId == 40 );
    EstimatorSample sample;

    for( uint64_t id = 1; id <= 41; ++id )
    {
        REQUIRE( simulation.popEstimator( sample ) );
        CHECK( sample.id == id );
    }

    CHECK_FALSE( simulation.popEstimator( sample ) );
    simulation.advance( last + 740000 );
    simulation.primary( 42, last + 745000, 500000 );
    status = simulation.estimatorStatus();
    CHECK( status.triggers == 3 );
    CHECK( status.selected == 42 );
    CHECK( status.postRemaining == estimatorPostSamples - 1 );
    REQUIRE( simulation.popEstimator( sample ) );
    CHECK( sample.id == 42 );
    CHECK_FALSE( simulation.popEstimator( sample ) );
}

/** \brief Export saturation and reset preserve process IDs and losses while changing no frozen-grid outcomes. */
TEST_CASE( "synco simulation estimator export lifetime", "[dmSyncopation][simulation]" )
{
    using namespace MagAOX::dmSyncoSimulation;
    Simulation      retained, drained;
    EstimatorSample sample;
    const uint64_t  sessions = estimatorTraceCapacity / 41 + 2;

    for( uint64_t session = 1; session <= sessions; ++session )
    {
        REQUIRE( retained.configure( {} ) );
        REQUIRE( drained.configure( {} ) );
        const auto last = warm( retained );
        REQUIRE( warm( drained ) == last );
        feed( retained, 41, last + 465000, 465000 );
        feed( drained, 41, last + 465000, 465000 );

        for( uint64_t offset = 1; offset <= 41; ++offset )
        {
            REQUIRE( drained.popEstimator( sample ) );
            CHECK( sample.id == ( session - 1 ) * 41 + offset );
            CHECK( sample.session == session );
        }

        CHECK_FALSE( drained.popEstimator( sample ) );
        const auto &a = retained.summary();
        const auto &b = drained.summary();
        CHECK( a.periodNs == b.periodNs );
        CHECK( a.gridNs == b.gridNs );
        CHECK( a.nextGridNs == b.nextGridNs );
        CHECK( a.nextActiveNs == b.nextActiveNs );
        CHECK( a.phaseNs == b.phaseNs );

        for( size_t step = 0; step < maxSteps; ++step )
        {
            CHECK( a.outcomes[step].considered == b.outcomes[step].considered );
            CHECK( a.outcomes[step].wouldRun == b.outcomes[step].wouldRun );
            CHECK( a.outcomes[step].skipped == b.outcomes[step].skipped );
            CHECK( a.outcomes[step].overlap == b.outcomes[step].overlap );
        }
    }

    auto status = retained.estimatorStatus();
    CHECK( status.observed == sessions * 41 );
    CHECK( status.selected == sessions * 41 );
    CHECK( status.queued == estimatorTraceCapacity );
    CHECK( status.dropped == sessions * 41 - estimatorTraceCapacity );
    CHECK( status.drained == 0 );
    CHECK( status.selected == status.drained + status.dropped + status.queued );
    CHECK( drained.estimatorStatus().dropped == 0 );
    CHECK( drained.estimatorStatus().queued == 0 );

    for( uint64_t id = 1; id <= estimatorTraceCapacity; ++id )
    {
        REQUIRE( retained.popEstimator( sample ) );
        CHECK( sample.id == id );
    }

    const auto last = retained.summary().lastMainNs;
    feed( retained, 42, last + 535000, 535000 );
    CHECK( retained.estimatorStatus().queued == 1 );
    const auto dropped = retained.estimatorStatus().dropped;
    retained.reset();
    status = retained.estimatorStatus();
    CHECK( status.observed == sessions * 41 + 1 );
    CHECK( status.history == 0 );
    CHECK( status.postRemaining == 0 );
    CHECK( status.dropped == dropped );
    CHECK( status.queued == 1 );
    REQUIRE( retained.configure( {} ) );
    REQUIRE( retained.popEstimator( sample ) );
    CHECK( sample.id == sessions * 41 + 1 );
    CHECK( sample.session == sessions );
    CHECK_FALSE( retained.popEstimator( sample ) );
    const auto restarted = warm( retained );
    feed( retained, 41, restarted + 465000, 465000 );

    for( uint64_t offset = 1; offset <= 41; ++offset )
    {
        REQUIRE( retained.popEstimator( sample ) );
        CHECK( sample.id == sessions * 41 + 1 + offset );
        CHECK( sample.session == sessions + 2 );
    }

    CHECK_FALSE( retained.popEstimator( sample ) );
    status = retained.estimatorStatus();
    CHECK( status.selected == status.drained + status.dropped + status.queued );
}

/** \brief The rolling pre-trigger history retains the latest 256 updates even at the largest estimator window. */
TEST_CASE( "synco simulation estimator history capacity", "[dmSyncopation][simulation]" )
{
    using namespace MagAOX::dmSyncoSimulation;
    Simulation         simulation;
    Simulation::Config config;
    config.window = estimatorHistoryCapacity;
    REQUIRE( simulation.configure( config ) );

    for( uint64_t counter = 1; counter <= 300; ++counter )
    {
        feed( simulation, counter, 1000000000 + int64_t( counter ) * 500000, counter == 1 ? 0 : 500000 );
    }

    CHECK( simulation.summary().valid );
    CHECK( simulation.estimatorStatus().history == estimatorHistoryCapacity );
    CHECK( simulation.estimatorStatus().selected == 0 );
    feed( simulation, 301, 1150000000 + 465000, 465000 );
    const auto status = simulation.estimatorStatus();
    CHECK( status.observed == 301 );
    CHECK( status.history == estimatorHistoryCapacity );
    CHECK( status.selected == estimatorHistoryCapacity + 1 );
    CHECK( status.triggers == 1 );
    EstimatorSample sample;

    for( uint64_t id = 45; id <= 301; ++id )
    {
        REQUIRE( simulation.popEstimator( sample ) );
        CHECK( sample.id == id );
        CHECK( sample.counter == id );
    }

    CHECK( sample.periodSamples == estimatorHistoryCapacity );
    CHECK( sample.phaseSamples == estimatorHistoryCapacity );
    CHECK_FALSE( simulation.popEstimator( sample ) );
}

/** \brief A historical phase kick can cause later overlaps during constant cadence, and the trace preserves its cause.
 */
TEST_CASE( "synco simulation estimator retained phase kick", "[dmSyncopation][simulation]" )
{
    using namespace MagAOX::dmSyncoSimulation;
    Simulation simulation;
    REQUIRE( simulation.configure( {} ) );
    auto     time     = warm( simulation );
    uint64_t overlaps = 0;

    for( uint64_t counter = 41; counter <= 140; ++counter )
    {
        const int64_t interval = counter == 41 ? 460000 : 500000;
        time += interval;
        feed( simulation, counter, time, interval );
        OverlapEvent event;

        while( simulation.popOverlap( event ) )
        {
            CHECK( event.step == 0 );
            CHECK( event.mainCounter == 59 + overlaps );
            CHECK( event.observedNs - event.mainGridNs == -37000 + int64_t( overlaps ) * 1000 );
            CHECK( interval == 500000 );
            CHECK( event.planEstimatorId == event.mainCounter - 1 );
            CHECK( event.lastEstimatorId == event.mainCounter - 1 );
            ++overlaps;
        }
    }

    CHECK( overlaps == 9 );
    CHECK( simulation.summary().outcomes[1].overlap == 0 );
    CHECK( simulation.summary().periodNs == 500000 );
    CHECK( simulation.estimatorStatus().triggers == 9 );
    CHECK( simulation.estimatorStatus().postRemaining == 0 );
    CHECK( simulation.estimatorStatus().selected == 130 );
    EstimatorSample sample;
    bool            kick = false, clipped = false;

    while( simulation.popEstimator( sample ) )
    {
        if( sample.counter == 41 )
        {
            kick = true;
            CHECK( sample.publicationIntervalNs == 460000 );
            CHECK( sample.observedNs - sample.predictedNs == -40000 );
            CHECK( sample.periodBeforeNs == 500000 );
            CHECK( sample.periodNs == 500000 );
            CHECK( sample.rawCorrectionNs == 0 );
            CHECK( sample.appliedCorrectionNs == 0 );
        }

        if( sample.counter == 59 )
        {
            clipped = true;
            CHECK( sample.publicationIntervalNs == 500000 );
            CHECK( sample.periodNs == 500000 );
            CHECK( sample.rawCorrectionNs < -sample.phaseSlewNs );
            CHECK( sample.appliedCorrectionNs == -1000 );
            CHECK( sample.committedGridNs == sample.baseGridNs + sample.appliedCorrectionNs );
        }
    }

    CHECK( kick );
    CHECK( clipped );
}

/** \brief Accepted updates still complete diagnostics when the largest counter prevents a future commitment. */
TEST_CASE( "synco simulation estimator accepted terminal update", "[dmSyncopation][simulation]" )
{
    using namespace MagAOX::dmSyncoSimulation;
    Simulation simulation;
    REQUIRE( simulation.configure( {} ) );
    const auto last = warm( simulation );
    simulation.advance( last + 240000 );
    simulation.primary( std::numeric_limits<uint64_t>::max(), last + 245000, 245000 );
    CHECK( simulation.estimatorStatus().observed == 41 );
    CHECK_FALSE( simulation.summary().valid );
    EstimatorSample sample;

    for( uint64_t id = 1; id <= 41; ++id )
    {
        REQUIRE( simulation.popEstimator( sample ) );
    }

    CHECK( sample.id == 41 );
    CHECK( sample.counter == std::numeric_limits<uint64_t>::max() );
    CHECK_FALSE( sample.sequential );
    CHECK_FALSE( sample.intervalAccepted );
    CHECK( sample.nextCounter == 0 );
    CHECK( sample.sourcePlanCounter == 0 );
    CHECK( sample.baseGridNs == 0 );
    CHECK_FALSE( sample.committed );
    CHECK( sample.committedGridNs == 0 );
}

/** \brief Monotonic counter/time endpoints qualify after one second, grow to ten seconds, and refresh only once per
 * second. */
TEST_CASE( "synco simulation long cadence qualification", "[dmSyncopation][simulation]" )
{
    using namespace MagAOX::dmSyncoSimulation;
    Simulation simulation;
    REQUIRE( simulation.configure( {} ) );

    for( uint64_t counter = 1; counter <= 24001; ++counter )
    {
        const int64_t observed = 1000000000 + int64_t( counter ) * 500000;
        feed( simulation, counter, observed, counter % 3 == 0 ? 506000 : 497000 );

        if( counter == 32 )
        {
            CHECK( simulation.summary().valid );
            CHECK( simulation.summary().cadence.state == CadenceState::nominal );
            CHECK( simulation.summary().cadence.updates == 0 );
        }

        if( counter == 2000 )
        {
            CHECK( simulation.summary().cadence.updatedNs == 0 );
            CHECK( simulation.summary().cadence.anchors == 1 );
        }

        if( counter == 2001 )
        {
            const auto &cadence = simulation.summary().cadence;
            CHECK( cadence.state == CadenceState::growing );
            CHECK( cadence.spanNs == 1000000000 );
            CHECK( cadence.cycles == 2000 );
            CHECK( cadence.updates == 1 );
            CHECK( cadence.updatedNs == observed );
        }

        if( counter == 20001 || counter == 24001 )
        {
            const auto &cadence = simulation.summary().cadence;
            CHECK( cadence.state == CadenceState::full );
            CHECK( cadence.spanNs == 10000000000LL );
            CHECK( cadence.cycles == 20000 );
            CHECK( cadence.periodQ32 == uint64_t( 500000 ) * periodScale );
            CHECK( cadence.observationNs == observed );
            CHECK( cadence.updatedNs == observed );
            CHECK( simulation.summary().phaseNs == 0 );
            CHECK( simulation.summary().nextGridNs == observed + 500000 );
        }
    }

    CHECK( simulation.summary().cadence.updates == 12 );
    CHECK( simulation.summary().cadence.rejected == 0 );
}

/** \brief Asymmetric adjacent jitter, paired excursions and forward counter gaps cannot bias the true long-span
 * denominator. */
TEST_CASE( "synco simulation long cadence noisy endpoints", "[dmSyncopation][simulation]" )
{
    using namespace MagAOX::dmSyncoSimulation;
    Simulation simulation;
    REQUIRE( simulation.configure( {} ) );
    int64_t previous = 0;

    for( uint64_t counter = 1; counter <= 24001; ++counter )
    {
        if( counter == 1001 )
        {
            continue;
        }

        const int64_t jitter   = counter % 3 == 1 ? 0 : ( counter % 3 == 2 ? 6000 : -3000 );
        const int64_t observed = 1000000000 + int64_t( counter ) * 500000 + jitter + ( counter == 5000 ? 80000 : 0 );
        feed( simulation, counter, observed, previous ? observed - previous : 0 );
        previous = observed;
    }

    const auto &summary = simulation.summary();
    const auto &cadence = summary.cadence;
    CHECK( summary.gaps == 1 );
    CHECK( cadence.state == CadenceState::full );
    CHECK( cadence.rejected == 0 );
    CHECK( cadence.cycles >= 20000 );
    CHECK( cadence.spanNs >= cadence.windowNs );
    CHECK( cadence.periodQ32 > uint64_t( 499999 ) * periodScale );
    CHECK( cadence.periodQ32 < uint64_t( 500001 ) * periodScale );
    CHECK( summary.periodNs == 500000 );
}

/** \brief Duplicates and incoherent timestamps leave cadence unchanged; silence and positive clock reversal reset
 * epochs. */
TEST_CASE( "synco simulation cadence epoch and freshness", "[dmSyncopation][simulation]" )
{
    using namespace MagAOX::dmSyncoSimulation;
    Simulation simulation;
    REQUIRE( simulation.configure( {} ) );

    for( uint64_t counter = 1; counter <= 2001; ++counter )
    {
        feed( simulation, counter, 1000000000 + int64_t( counter ) * 500000 );
    }

    const auto cadence = simulation.summary().cadence;
    simulation.primary( 2001, cadence.observationNs + 20000 );
    CHECK( simulation.summary().cadence.observationNs == cadence.observationNs );
    CHECK( simulation.summary().cadence.anchors == cadence.anchors );
    CHECK( simulation.summary().cadence.updates == cadence.updates );
    simulation.primary( 2002, 0 );
    CHECK( simulation.summary().cadence.epoch == cadence.epoch );
    CHECK( simulation.summary().cadence.periodQ32 == cadence.periodQ32 );
    simulation.advance( cadence.observationNs + 4000001 );
    const auto expired = simulation.summary().cadence;
    CHECK( expired.epoch == cadence.epoch + 1 );
    CHECK( expired.state == CadenceState::nominal );
    CHECK( expired.anchors == 0 );
    CHECK( expired.observationNs == 0 );
    CHECK_FALSE( simulation.summary().valid );
    simulation.advance( cadence.observationNs + 8000001 );
    CHECK( simulation.summary().cadence.epoch == expired.epoch );
    feed( simulation, 2010, cadence.observationNs + 8500000 );
    CHECK( simulation.summary().cadence.epoch == expired.epoch );
    CHECK( simulation.summary().cadence.anchors == 1 );
    CHECK( simulation.summary().cadence.updates == 0 );
    simulation.reset();
    CHECK( simulation.summary().cadence.epoch == expired.epoch + 1 );
    Simulation::Config config;
    config.nominalPeriod = 600000;
    REQUIRE( simulation.configure( config ) );
    CHECK( simulation.summary().cadence.epoch == expired.epoch + 2 );
    CHECK( simulation.summary().cadence.periodQ32 == uint64_t( 600000 ) * periodScale );
    simulation.primary( 1, 1000000000 );
    simulation.primary( 2, 999999999 );
    CHECK( simulation.summary().cadence.epoch == expired.epoch + 3 );
    CHECK( simulation.summary().cadence.anchors == 1 );
    CHECK( simulation.summary().cadence.observationNs == 999999999 );
    CHECK( simulation.summary().cadence.updatedNs == 0 );
    CHECK( simulation.summary().counter == 2 );
    CHECK( simulation.summary().sequence == 1 );
    CHECK( simulation.summary().invalid == 1 );
}

/** \brief Implausible long-span rates are rejected without replacing the selected nominal period. */
TEST_CASE( "synco simulation cadence rejects implausible endpoint rate", "[dmSyncopation][simulation]" )
{
    using namespace MagAOX::dmSyncoSimulation;
    Simulation         simulation;
    Simulation::Config config;
    config.cadenceWindowNs = 1000000;
    config.cadenceUpdateNs = 1000000;
    REQUIRE( simulation.configure( config ) );
    simulation.primary( 1, 1000000000 );
    simulation.primary( 1001, 1001000000 );
    CHECK( simulation.summary().cadence.rejected == 1 );
    CHECK( simulation.summary().cadence.updates == 0 );
    CHECK( simulation.summary().cadence.periodQ32 == uint64_t( 500000 ) * periodScale );
    CHECK( simulation.summary().cadence.state == CadenceState::nominal );
    CHECK( simulation.summary().cadence.observationNs == 1001000000 );
    CHECK( simulation.summary().cadence.anchors == 2 );
}

/** \brief Fractional nanoseconds survive repeated two-minute grid extrapolation and direct subframe placement. */
TEST_CASE( "synco simulation fractional cadence arithmetic", "[dmSyncopation][simulation]" )
{
    using namespace MagAOX::dmSyncoSimulation;
    using namespace MagAOX::app::synco;
    const uint64_t period = uint64_t( 500000 ) * periodScale + periodScale / 4;
    GridTime       time{ 1000000000, 0 };

    for( uint64_t counter = 1; counter <= 240000; ++counter )
    {
        time = shiftGrid( time, counter - 1, counter, period );

        if( counter % 4000 == 0 )
        {
            CHECK( time.ns == 1000000000 + int64_t( counter ) * 500000 + int64_t( counter / 4 ) );
            CHECK( time.fractionQ32 == 0 );
        }
    }

    CHECK( shiftGrid( time, 240000, 0, period ).ns == 1000000000 );
    CHECK( stepGrid( { 1000000000, 0 }, period, 1, 2 ).ns == 1000250000 );
    CHECK( stepGrid( { 1000000000, 0 }, period, 1, 2 ).fractionQ32 == int64_t( periodScale / 8 ) );
    CHECK( shiftGrid( { std::numeric_limits<int64_t>::max(), 0 }, 0, 1, period ).ns == 0 );
    CHECK( shiftGrid( { 1, 0 }, 1, 0, period ).ns == 0 );
    CHECK( shiftGrid( { 1000000000, int64_t( periodScale / 2 ) }, 0, 1, period ).ns == 0 );
    CHECK( stepGrid( time, period, 1, 0 ).ns == 0 );
    CHECK( shiftGrid( time, 0, std::numeric_limits<uint64_t>::max(), std::numeric_limits<uint64_t>::max() ).ns == 0 );
    CHECK( stepGrid( time, std::numeric_limits<uint64_t>::max(), 1, 1 ).ns == 0 );
    CHECK( roundGrid( std::numeric_limits<__int128>::max() ).ns == 0 );
    CHECK( roundGrid( std::numeric_limits<__int128>::min() ).ns == 0 );
    const auto half = roundGrid( __int128( 1000 ) * periodScale + periodScale / 2 );
    CHECK( half.ns == 1001 );
    CHECK( half.fractionQ32 == -int64_t( periodScale / 2 ) );
}

/** \brief The complete scheduler carries a quarter-nanosecond period for two minutes without per-cycle rounding drift.
 */
TEST_CASE( "synco simulation fractional cadence long run", "[dmSyncopation][simulation]" )
{
    using namespace MagAOX::dmSyncoSimulation;
    Simulation         simulation;
    Simulation::Config config;
    config.phaseGain = 0;
    REQUIRE( simulation.configure( config ) );
    int64_t measuredOrigin = 0;

    for( uint64_t counter = 1; counter <= 240001; ++counter )
    {
        const int64_t observed = 1000000000 + int64_t( counter - 1 ) * 500000 + int64_t( ( counter - 1 + 2 ) / 4 );
        const int64_t frozen   = simulation.summary().nextGridNs;
        feed( simulation, counter, observed );

        if( counter == 2001 )
        {
            measuredOrigin = simulation.summary().nextGridNs;
            CHECK( simulation.summary().gridNs == frozen );
            CHECK( simulation.summary().cadence.periodQ32 == uint64_t( 500000 ) * periodScale + periodScale / 4 );
            CHECK( simulation.summary().nextActiveNs == frozen + 250000 );
        }

        if( counter >= 4001 && counter % 4000 == 1 )
        {
            const int64_t cycles   = int64_t( counter - 2001 );
            const int64_t expected = measuredOrigin + cycles * 500000 + cycles / 4;
            CHECK( simulation.summary().nextGridNs == expected );
            CHECK( simulation.summary().cadence.periodQ32 == uint64_t( 500000 ) * periodScale + periodScale / 4 );
        }
    }

    CHECK( simulation.summary().valid );
    CHECK( simulation.summary().outcomes[0].overlap == 0 );
    CHECK( simulation.summary().outcomes[1].overlap == 0 );
}

/** \brief Silence restores nominal feasibility together with the nominal cadence seed. */
TEST_CASE( "synco simulation stale cadence restores nominal feasibility", "[dmSyncopation][simulation]" )
{
    Simulation         simulation;
    Simulation::Config config;
    config.cadenceMinPeriod = 500001;
    REQUIRE( simulation.configure( config ) );
    REQUIRE_FALSE( simulation.summary().feasible );
    int64_t last = 0;

    for( uint64_t counter = 1; counter <= 2001; ++counter )
    {
        last = 1000000000 + int64_t( counter ) * 500002;
        feed( simulation, counter, last, 500002 );
    }

    REQUIRE( simulation.summary().cadence.updates == 1 );
    REQUIRE( simulation.summary().periodNs == 500002 );
    REQUIRE( simulation.summary().feasible );
    simulation.advance( last + config.timeout + 1 );
    CHECK( simulation.summary().periodNs == config.nominalPeriod );
    CHECK_FALSE( simulation.summary().valid );
    CHECK_FALSE( simulation.summary().feasible );
}

/** \brief Requested cadence spans must fit the fixed anchor ring and avoid invalid or overflowing durations. */
TEST_CASE( "synco simulation cadence configuration bounds", "[dmSyncopation][simulation]" )
{
    Simulation         simulation;
    Simulation::Config config;
    config.cadenceUpdateNs = 0;
    CHECK_FALSE( simulation.configure( config ) );
    config                 = {};
    config.cadenceWindowNs = config.cadenceUpdateNs - 1;
    CHECK_FALSE( simulation.configure( config ) );
    config                 = {};
    config.cadenceWindowNs = config.cadenceUpdateNs * 64;
    CHECK_FALSE( simulation.configure( config ) );
    config.cadenceWindowNs = config.cadenceUpdateNs * 63;
    CHECK( simulation.configure( config ) );
    config.cadenceWindowNs = std::numeric_limits<int64_t>::max();
    CHECK_FALSE( simulation.configure( config ) );
}

/// Keep observed readiness optional, split read and backend assumptions, and reject contradictory model settings.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "synco readiness preserves legacy defaults and validates separate assumptions",
           "[dmSyncopation][simulation]" )
{
    Simulation         simulation;
    Simulation::Config config;
    CHECK_FALSE( config.readerReleaseReady );
    CHECK( config.virtualReaderNs == 30000 );
    CHECK( config.backendNs == 155000 );

    SECTION( "Legacy three-step feasibility remains unchanged" )
    {
        config.steps = 3;
        REQUIRE( simulation.configure( config ) );
        warm( simulation );
        CHECK_FALSE( simulation.summary().feasible );
        CHECK_FALSE( simulation.summary().readerReleaseReady );
    }

    SECTION( "A whole backend cannot complete before its last modeled read" )
    {
        config.readerReleaseReady = true;
        config.backendNs          = config.virtualReaderNs - 1;
        CHECK_FALSE( simulation.configure( config ) );
    }

    SECTION( "Virtual durations require positive bounded values" )
    {
        config.readerReleaseReady = true;
        config.virtualReaderNs    = 0;
        CHECK_FALSE( simulation.configure( config ) );
        config.virtualReaderNs = 30000;
        config.backendNs       = 1000000001;
        CHECK_FALSE( simulation.configure( config ) );
    }
}

/// Use coherent availability plus guard at the immutable deadline, rather than an earlier publisher endpoint.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "synco readiness uses capture availability and skips independently", "[dmSyncopation][simulation]" )
{
    Simulation         simulation;
    Simulation::Config config;
    config.readerReleaseReady = true;
    REQUIRE( simulation.configure( config ) );
    const auto main = warmReady( simulation ) + 500000;
    readyMain( simulation, 41, main );
    const auto    before = simulation.summary().outcomes[1];
    const int64_t start  = main + 240000;
    const int     mode   = GENERATE( 0, 1, 2 );

    if( mode )
    {
        simulation.release( readyRecord( 41, main, start - config.bounds.guard + ( mode == 2 ? 1 : 0 ) ) );
    }

    simulation.advance( main + 250000 );
    const auto &summary = simulation.summary();
    CHECK( summary.outcomes[1].considered == before.considered + 1 );
    CHECK( summary.outcomes[1].wouldRun == before.wouldRun + ( mode == 1 ? 1 : 0 ) );
    CHECK( summary.outcomes[1].releaseMissing == before.releaseMissing + ( mode == 0 ? 1 : 0 ) );
    CHECK( summary.outcomes[1].releaseLate == before.releaseLate + ( mode == 2 ? 1 : 0 ) );
    CHECK( summary.nextGridNs == main + 500000 );
    CHECK( summary.periodNs == 500000 );

    if( !mode )
    {
        simulation.release( readyRecord( 41, main, main + 251000 ) );
        simulation.advance( main + 252000 );
        CHECK( summary.outcomes[1].wouldRun == before.wouldRun );
        CHECK( summary.outcomes[1].releaseMissing == before.releaseMissing + 1 );
    }
}

/// Retain an early arrival until exact combined association and refuse a mismatching raw publication identity.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "synco readiness accepts either evidence order with exact combined identity", "[dmSyncopation][simulation]" )
{
    Simulation         simulation;
    Simulation::Config config;
    config.readerReleaseReady = true;
    REQUIRE( simulation.configure( config ) );
    const auto main = warmReady( simulation ) + 500000;
    simulation.primary( 41, main, 500000 );
    const auto before   = simulation.summary().outcomes[1];
    auto       event    = readyRecord( 41, main, main + 40000 );
    const bool mismatch = GENERATE( false, true );

    if( mismatch )
    {
        ++event.record.publicationNsec;
    }

    simulation.release( event );
    simulation.combined( 41, 41, main + 60000, true, readyContext( 41 ) );
    simulation.advance( main + 250000 );
    CHECK( simulation.summary().outcomes[1].wouldRun == before.wouldRun + ( mismatch ? 0 : 1 ) );
    CHECK( simulation.summary().outcomes[1].releaseMissing == before.releaseMissing + ( mismatch ? 1 : 0 ) );
}

/// Recover from forward publisher/session epochs while revoking incoherent, dropped, regressed or failed evidence.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "synco readiness source lifecycle never reuses revoked releases", "[dmSyncopation][simulation]" )
{
    Simulation         simulation;
    Simulation::Config config;
    config.readerReleaseReady = true;
    REQUIRE( simulation.configure( config ) );
    const auto main  = warmReady( simulation );
    auto       event = readyRecord( 40, main, main + 40000 );

    SECTION( "A reset rejects duplicate release reuse" )
    {
        simulation.resetRelease();
        simulation.release( event );
        CHECK( simulation.summary().latestRelease.observedNs == 0 );
        CHECK( simulation.summary().readiness.duplicates == 1 );
    }

    SECTION( "Transport drops retain the exact loss count" )
    {
        simulation.releaseDrop( 7 );
        CHECK( simulation.summary().readiness.queueDrops == 7 );
        simulation.release( event );
        CHECK( simulation.summary().latestRelease.observedNs == 0 );
    }

    SECTION( "Incoherent transport revokes the previous qualified release" )
    {
        event.coherent = false;
        simulation.release( event );
        CHECK( simulation.summary().readiness.incoherent == 1 );
        CHECK( simulation.summary().latestRelease.observedNs == 0 );
    }

    SECTION( "Forward publisher restart on the same stream permits a fresh sequence" )
    {
        ++event.record.session;
        ++event.record.publisherPid;
        event.record.sequence = 1;
        simulation.release( event );
        REQUIRE( simulation.summary().latestRelease.record.session == event.record.session );
        --event.record.session;
        simulation.release( event );
        CHECK( simulation.summary().latestRelease.observedNs == 0 );
        CHECK( simulation.summary().readiness.invalid == 1 );
    }

    SECTION( "Forward producer epoch clears old evidence and rejects its replay" )
    {
        ++event.record.epoch;
        event.record.sequence = 1;
        simulation.release( event );
        REQUIRE( simulation.summary().latestRelease.record.epoch == event.record.epoch );
        --event.record.epoch;
        simulation.release( event );
        CHECK( simulation.summary().latestRelease.observedNs == 0 );
        CHECK( simulation.summary().readiness.invalid == 1 );
    }

    SECTION( "Old attachment epochs cannot replace a new source" )
    {
        event.sourceEpoch = 2;
        simulation.release( event );
        REQUIRE( simulation.summary().latestRelease.sourceEpoch == 2 );
        event.sourceEpoch = 1;
        simulation.release( event );
        CHECK( simulation.summary().latestRelease.sourceEpoch == 2 );
        CHECK( simulation.summary().readiness.invalid == 1 );
        return;
    }

    SECTION( "A sequence gap revokes history and does not authorize the skipped attempt" )
    {
        event = readyRecord( 42, main + 50000, main + 100000 );
        simulation.release( event );
        CHECK( simulation.summary().readiness.gaps == 1 );
        CHECK( simulation.summary().latestRelease.observedNs == 0 );
    }

    SECTION( "SDK failure revokes reads for scheduling until a producer epoch recovers" )
    {
        event.record.outcome       = MagAOX::dmTiming::Outcome::failed;
        event.record.backendResult = 7;
        simulation.release( event );
        CHECK( simulation.summary().readiness.sdkFailed == 1 );
        CHECK( simulation.summary().latestRelease.observedNs == 0 );
        auto recovery = readyRecord( 41, main + 500000, main + 540000 );
        simulation.release( recovery );
        CHECK( simulation.summary().latestRelease.observedNs == 0 );
        ++recovery.record.epoch;
        simulation.release( recovery );
        CHECK( simulation.summary().latestRelease.record.epoch == 2 );
        return;
    }

    const auto missing = simulation.summary().outcomes[1].releaseMissing;
    simulation.advance( main + 250000 );
    CHECK( simulation.summary().outcomes[1].releaseMissing == missing + 1 );
}

/// Never use an observed actual-main acknowledgement to clear a hypothetical auxiliary reader interval.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "synco readiness preserves virtual auxiliary ownership", "[dmSyncopation][simulation]" )
{
    Simulation         simulation;
    Simulation::Config config;
    config.readerReleaseReady = true;
    config.virtualReaderNs    = 200000;
    config.backendNs          = 200000;
    REQUIRE( simulation.configure( config ) );
    const auto main   = warmReady( simulation );
    const auto before = simulation.summary().outcomes;
    simulation.advance( main + 475000 );
    CHECK( simulation.summary().outcomes[1].wouldRun == before[1].wouldRun + 1 );
    CHECK( simulation.summary().outcomes[0].wouldRun == before[0].wouldRun );
    CHECK( simulation.summary().outcomes[0].readerHold == before[0].readerHold + 1 );
}

/// Report an incomplete three-step backend envelope without suppressing independently eligible earlier slots.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "synco readiness three-step geometry retains explicit backend qualification", "[dmSyncopation][simulation]" )
{
    Simulation         simulation;
    Simulation::Config config;
    config.readerReleaseReady = true;
    config.steps              = 3;
    const bool shorter        = GENERATE( false, true );

    if( shorter )
    {
        config.backendNs = 120000; // Explicit conditional model, never silently inferred from the early release.
    }

    REQUIRE( simulation.configure( config ) );
    const auto main   = warmReady( simulation );
    const auto before = simulation.summary().outcomes;
    REQUIRE( simulation.summary().nextActiveNs == main + 166667 );
    CHECK( simulation.summary().readerCycleFeasible );
    CHECK( simulation.summary().backendCycleFeasible == shorter );
    CHECK( simulation.summary().feasible == shorter );
    CHECK( simulation.summary().backendNs == ( shorter ? 120000 : 155000 ) );
    simulation.advance( main + 475000 );
    const auto &summary = simulation.summary();
    CHECK( summary.outcomes[0].wouldRun == before[0].wouldRun + 1 );
    CHECK( summary.outcomes[1].wouldRun == before[1].wouldRun + 1 );
    CHECK( summary.outcomes[2].wouldRun == before[2].wouldRun + ( shorter ? 1 : 0 ) );
    CHECK( summary.outcomes[2].backendWrap == before[2].backendWrap + ( shorter ? 0 : 1 ) );
    CHECK( summary.nextGridNs == main + 500000 );
    CHECK( summary.periodNs == 500000 );
}

/// An invalidation affects future slots without erasing a release that was already known at an earlier deadline.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "synco readiness revocation respects immutable earlier decisions", "[dmSyncopation][simulation]" )
{
    Simulation         simulation;
    Simulation::Config config;
    config.readerReleaseReady = true;
    REQUIRE( simulation.configure( config ) );
    const auto main         = warmReady( simulation );
    const auto before       = simulation.summary().outcomes[1];
    const bool late         = GENERATE( false, true );
    auto       invalidation = readyRecord( 40, main, main + ( late ? 300000 : 100000 ) );
    invalidation.coherent   = false;
    simulation.release( invalidation );
    simulation.advance( main + 400000 );
    CHECK( simulation.summary().outcomes[1].considered == before.considered + 1 );
    CHECK( simulation.summary().outcomes[1].wouldRun == before.wouldRun + ( late ? 1 : 0 ) );
    CHECK( simulation.summary().outcomes[1].releaseMissing == before.releaseMissing + ( late ? 0 : 1 ) );
    CHECK( simulation.summary().readiness.incoherent == 1 );
    CHECK( simulation.summary().nextGridNs == main + 500000 );
}

/// Use exact frozen target rounding at the backend wrap boundary without silently changing the assumption.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "synco readiness backend wrap boundary is inclusive", "[dmSyncopation][simulation]" )
{
    Simulation         simulation;
    Simulation::Config config;
    config.readerReleaseReady = true;
    config.steps              = 3;
    config.backendNs          = GENERATE( 140000LL, 141666LL, 141667LL, 141668LL );
    REQUIRE( simulation.configure( config ) );
    const auto main   = warmReady( simulation );
    const auto before = simulation.summary().outcomes[2];
    simulation.advance( main + 475000 );
    CHECK( simulation.summary().outcomes[2].wouldRun == before.wouldRun + ( config.backendNs <= 141667 ? 1 : 0 ) );
    CHECK( simulation.summary().outcomes[2].backendWrap == before.backendWrap + ( config.backendNs > 141667 ? 1 : 0 ) );
    CHECK( simulation.summary().backendNs == config.backendNs );
}

/// A later SDK result must not make success appear available at the earlier release capture.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "synco readiness preserves the first captured release through enrichment", "[dmSyncopation][simulation]" )
{
    Simulation         simulation;
    Simulation::Config config;
    config.readerReleaseReady = true;
    REQUIRE( simulation.configure( config ) );
    const auto main             = warmReady( simulation );
    const auto qualified        = simulation.summary().readiness.qualified;
    auto       result           = readyRecord( 40, main, main + 130000 );
    result.record.outcome       = MagAOX::dmTiming::Outcome::success;
    result.record.backendResult = 0;
    simulation.release( result );
    const auto &evidence = simulation.summary().latestRelease;
    CHECK( evidence.record.outcome == MagAOX::dmTiming::Outcome::unknown );
    CHECK( evidence.record.backendResult == MagAOX::dmCompletion::unavailableResult );
    CHECK( evidence.observedNs == main + 40000 );
    CHECK( simulation.summary().readiness.qualified == qualified );
    CHECK( simulation.summary().readiness.duplicates == 0 );
    const auto before = simulation.summary().outcomes[1].wouldRun;
    simulation.advance( main + 250000 );
    CHECK( simulation.summary().outcomes[1].wouldRun == before + 1 );
}

/// Reader clearance does not remove actual-main or prior virtual backend occupancy.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "synco readiness checks actual and virtual backend ownership separately", "[dmSyncopation][simulation]" )
{
    Simulation         simulation;
    Simulation::Config config;
    config.readerReleaseReady = true;
    const bool virtualJob     = GENERATE( false, true );

    if( virtualJob )
    {
        config.steps     = 3;
        config.backendNs = 200000;
    }

    REQUIRE( simulation.configure( config ) );
    warmReady( simulation );
    const int64_t grid = simulation.summary().nextGridNs;
    const int64_t main = grid + ( virtualJob ? -100000 : 120000 );
    readyMain( simulation, 41, main );
    simulation.release( readyRecord( 41, main, main + 40000 ) );
    const auto before = simulation.summary().outcomes;
    simulation.advance( grid + 350000 );
    const auto &after = simulation.summary().outcomes;

    if( virtualJob )
    {
        CHECK( after[1].wouldRun == before[1].wouldRun + 1 );
        CHECK( after[2].wouldRun == before[2].wouldRun );
        CHECK( after[2].backendBusy == before[2].backendBusy + 1 );
    }
    else
    {
        CHECK( after[1].wouldRun == before[1].wouldRun );
        CHECK( after[1].backendBusy == before[1].backendBusy + 1 );
        CHECK( after[1].releaseLate == before[1].releaseLate );
    }

    CHECK( simulation.summary().gridNs == grid );
}

/// Report main/backend conflicts after reader clearance and retain the exclusive backend-end boundary.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "synco readiness detects an early main during virtual backend ownership", "[dmSyncopation][simulation]" )
{
    Simulation         simulation;
    Simulation::Config config;
    config.readerReleaseReady = GENERATE( false, true );
    const int64_t arrival     = GENERATE( 400000LL, 405000LL );
    REQUIRE( simulation.configure( config ) );
    const auto main = config.readerReleaseReady ? warmReady( simulation ) : warm( simulation );
    simulation.advance( main + 350000 );
    REQUIRE( simulation.traceStatus().detected == 0 );
    const auto before = simulation.summary().outcomes[1].overlap;
    simulation.primary( 41, main + arrival, arrival );
    const bool conflict = arrival < 405000;
    CHECK( simulation.summary().outcomes[1].overlap == before + ( conflict ? 1 : 0 ) );
    MagAOX::dmSyncoSimulation::OverlapEvent event;
    CHECK( simulation.popOverlap( event ) == conflict );

    if( conflict )
    {
        CHECK( event.window == ( config.readerReleaseReady ? MagAOX::dmSyncoSimulation::OverlapWindow::backend
                                                           : MagAOX::dmSyncoSimulation::OverlapWindow::readerHold ) );
        CHECK( event.readerEndNs == main + ( config.readerReleaseReady ? 300000 : 405000 ) );
        CHECK( event.backendEndNs == ( config.readerReleaseReady ? main + 405000 : 0 ) );
        CHECK( event.observedNs == main + 400000 );
        CHECK( event.copyBeginNs == main + 240000 );
        CHECK( event.copyEndNs == main + 250000 );
    }

    simulation.reset();
    simulation.primary( 1, main + 410000, 0 );
    CHECK_FALSE( simulation.popOverlap( event ) );
}

/// Passive preparation must not erase the independently retained backend ownership of the preceding active step.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "synco readiness retains active backend provenance after passive copying", "[dmSyncopation][simulation]" )
{
    Simulation         simulation;
    Simulation::Config config;
    config.readerReleaseReady = true;
    config.backendNs          = 223000;
    REQUIRE( simulation.configure( config ) );
    const auto main = warmReady( simulation );
    REQUIRE( simulation.summary().feasible );
    const auto before = simulation.summary().outcomes;
    simulation.advance( main + 472000 );
    REQUIRE( simulation.summary().outcomes[0].wouldRun == before[0].wouldRun + 1 );
    const int64_t arrival = GENERATE( 472500LL, 473000LL );
    simulation.primary( 41, main + arrival, arrival );
    const bool conflict = arrival < 473000;
    CHECK( simulation.summary().outcomes[0].overlap == before[0].overlap );
    CHECK( simulation.summary().outcomes[1].overlap == before[1].overlap + ( conflict ? 1 : 0 ) );
    MagAOX::dmSyncoSimulation::OverlapEvent event;
    CHECK( simulation.popOverlap( event ) == conflict );

    if( conflict )
    {
        CHECK( event.window == MagAOX::dmSyncoSimulation::OverlapWindow::backend );
        CHECK( event.step == 1 );
        CHECK( event.planCounter == 40 );
        CHECK( event.copyBeginNs == main + 240000 );
        CHECK( event.copyEndNs == main + 250000 );
        CHECK( event.readerEndNs == main + 300000 );
        CHECK( event.backendEndNs == main + 473000 );
        CHECK( event.observedNs == main + 472500 );
    }
}

/// Keep the pipelined mode opt-in and reject contradictory stage assumptions before acquiring a grid.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "synco pipeline configuration keeps legacy bounds separate", "[dmSyncopation][simulation][pipeline]" )
{
    // clang-format off
    #ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    MagAOX::app::synco::Simulation::configure({});
    #endif
    // clang-format on
    Simulation         simulation;
    Simulation::Config config;
    REQUIRE_FALSE( config.pipelineBackend );
    config.pipelineBackend = true;
    REQUIRE_FALSE( simulation.configure( config ) );
    config.readerReleaseReady = true;
    REQUIRE_FALSE( simulation.configure( config ) );
    config.virtualReaderNs = 60000;
    REQUIRE( simulation.configure( config ) );
    REQUIRE( simulation.summary().pipeline.enabled );
    REQUIRE( simulation.summary().pipeline.capacity == 3 );

    config.backendNs = 1; // The legacy whole-backend duration is not a pipeline stage.
    REQUIRE( simulation.configure( config ) );
    config.pipelineSdkServiceNs = 0;
    REQUIRE_FALSE( simulation.configure( config ) );
}

/// A measured SDK service can fit its stage while zero added main wait still excludes the final auxiliary slot.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "synco pipeline preserves zero added main SDK wait", "[dmSyncopation][simulation][pipeline]" )
{
    // clang-format off
    #ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    MagAOX::app::synco::Simulation::advance(0);
    #endif
    // clang-format on
    Simulation         simulation;
    Simulation::Config config;
    config.readerReleaseReady = config.pipelineBackend = true;
    config.virtualReaderNs                             = 60000;
    config.steps                                       = 3;
    REQUIRE( simulation.configure( config ) );
    const auto main     = warmReady( simulation );
    const auto before   = simulation.summary().outcomes;
    const auto admitted = simulation.summary().pipeline.auxiliaryAdmitted;
    simulation.advance( main + 475000 );
    const auto &summary = simulation.summary();
    CHECK( summary.outcomes[0].wouldRun == before[0].wouldRun + 1 );
    CHECK( summary.outcomes[1].wouldRun == before[1].wouldRun + 1 );
    CHECK( summary.outcomes[1].backendBusy == before[1].backendBusy );
    CHECK( summary.outcomes[2].wouldRun == before[2].wouldRun );
    CHECK( summary.outcomes[2].backendWrap == before[2].backendWrap + 1 );
    CHECK( summary.pipeline.auxiliaryAdmitted == admitted + 1 );
    CHECK( summary.readerCycleFeasible );
    CHECK( summary.pipeline.sdkCycleFeasible );
    CHECK( summary.pipeline.postCycleFeasible );
    CHECK( summary.pipeline.creditCycleFeasible );
    CHECK_FALSE( summary.pipeline.mainProtectionFeasible );
    CHECK_FALSE( summary.backendCycleFeasible );
    CHECK_FALSE( summary.feasible );
    CHECK( summary.pipeline.mainSdkWaits == 0 );
    CHECK( summary.pipeline.mainCreditMisses == 0 );
    CHECK( summary.pipeline.mainClearanceShortfallNs >= 33333 );
    CHECK( summary.pipeline.mainAdmitted + summary.pipeline.auxiliaryAdmitted ==
           summary.pipeline.retired + summary.pipeline.discarded + summary.pipeline.inFlight );
}

/// A conditional short-prefix scenario admits three slots while earlier post work still owns a credit.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "synco pipeline overlaps post retirement without shifting trigger targets",
           "[dmSyncopation][simulation][pipeline]" )
{
    // clang-format off
    #ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    MagAOX::app::synco::Simulation::advance(0);
    #endif
    // clang-format on
    Simulation         simulation;
    Simulation::Config config;
    config.readerReleaseReady = config.pipelineBackend = true;
    config.steps                                       = 3;
    config.bounds.delivery                             = 0;
    config.pipelinePrepareNs                           = 5000;
    config.virtualReaderNs                             = 10000;
    REQUIRE( simulation.configure( config ) );
    const auto main   = warmReady( simulation );
    const auto before = simulation.summary().outcomes;
    const auto next   = simulation.summary().nextGridNs;
    simulation.advance( main + 475000 );
    const auto &summary = simulation.summary();
    CHECK( summary.feasible );
    CHECK( summary.outcomes[1].wouldRun == before[1].wouldRun + 1 );
    CHECK( summary.outcomes[2].wouldRun == before[2].wouldRun + 1 );
    CHECK( summary.nextGridNs == next );
    CHECK( summary.periodNs == 500000 );
    CHECK( summary.pipeline.sdkReadyNs < main + 475000 );
    CHECK( summary.pipeline.postReadyNs > main + 475000 );
    CHECK( summary.pipeline.inFlight > 0 );
    CHECK( summary.pipeline.maximumInFlight <= 3 );
    CHECK( summary.pipeline.mainSdkWaits == 0 );
}

/// Lost output provenance quarantines uncertain work instead of manufacturing an idle SDK owner.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "synco pipeline output loss retains a resource fence", "[dmSyncopation][simulation][pipeline]" )
{
    // clang-format off
    #ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    MagAOX::app::synco::Simulation::combined(0,0,0);
    #endif
    // clang-format on
    Simulation         simulation;
    Simulation::Config config;
    config.readerReleaseReady = config.pipelineBackend = true;
    config.virtualReaderNs                             = 60000;
    REQUIRE( simulation.configure( config ) );
    const auto main   = warmReady( simulation );
    const auto before = simulation.summary().pipeline;
    simulation.combined( 41, 40, main + 50000, false );
    CHECK( simulation.summary().outputGaps > 0 );
    CHECK( simulation.summary().pipeline.resets == before.resets + 1 );
    CHECK( simulation.summary().pipeline.discarded > before.discarded );
    CHECK( simulation.summary().pipeline.inFlight == 0 );
    CHECK( simulation.summary().pipeline.sdkReadyNs > main + 50000 );
    simulation.primary( 41, main + 500000, 500000 );
    simulation.combined( 42, 41, main + 510000, true, readyContext( 42 ) );
    CHECK( simulation.summary().pipeline.mainCreditMisses == before.mainCreditMisses + 1 );
    const auto &pipeline = simulation.summary().pipeline;
    CHECK( pipeline.mainAdmitted + pipeline.auxiliaryAdmitted ==
           pipeline.retired + pipeline.discarded + pipeline.inFlight );
}

/// Keep exact waiting-main/blocker identities and separate preparation from native-worker queueing.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "synco delayed-main trace preserves main-only burst provenance", "[dmSyncopation][simulation][mainWait]" )
{
    Simulation         simulation;
    Simulation::Config config;
    config.readerReleaseReady = config.pipelineBackend = true;
    config.virtualReaderNs                             = 60000;
    config.steps                                       = 1;
    REQUIRE( simulation.configure( config ) );
    constexpr int64_t start = 1000000000;
    auto              first = readyContext( 101 );
    first.receiptNs         = start - 200;
    first.callbackNs        = start - 150;
    first.lockRequestNs     = start - 100;
    first.lockAcquiredNs    = start - 50;
    auto second             = readyContext( 102 );
    second.receiptNs        = start + 20000;
    second.callbackNs       = start + 20100;
    second.lockRequestNs    = start + 20200;
    second.lockAcquiredNs   = start + 20800;
    simulation.primary( 1, start - 1000 );
    simulation.combined( 101, 1, start, true, first );
    simulation.primary( 2, start + 19829, 500000 );
    simulation.combined( 102, 2, start + 20829, true, second );
    CHECK( simulation.mainWaitStatus().detected == 0 ); // Not final until actual modeled SDK entry.
    simulation.advance( start + 500000 );
    CHECK( simulation.summary().pipeline.mainSdkWaits == 1 );
    CHECK( simulation.summary().pipeline.maximumMainWaitNs == 99171 );
    CHECK( simulation.summary().pipeline.auxiliaryAdmitted == 0 );
    MagAOX::dmSyncoSimulation::MainWaitEvent event;
    REQUIRE( simulation.popMainWait( event ) );
    CHECK( event.id == 1 );
    CHECK( event.session == simulation.summary().session );
    CHECK( event.ordinal == 1 );
    CHECK( event.prepWaitNs == 19171 );
    CHECK( event.sdkWaitNs == 80000 );
    CHECK( event.main.counter == 102 );
    CHECK( event.main.primaryCounter == 2 );
    CHECK( event.main.observation.receiptNs == second.receiptNs );
    CHECK( event.main.observation.lockAcquiredNs == second.lockAcquiredNs );
    CHECK( event.main.observation.publicationNs == second.publicationNs );
    REQUIRE( event.blockerCount == 1 );
    CHECK( event.blockers[0].job.counter == 101 );
    CHECK( event.blockers[0].job.primaryCounter == 1 );
    CHECK( event.blockers[0].job.step == 0 );
    CHECK( event.blockers[0].job.observation.receiptNs == first.receiptNs );
    CHECK( event.provenanceComplete );
    CHECK_FALSE( simulation.popMainWait( event ) );
    CHECK( simulation.mainWaitStatus().drained == 1 );

    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    MagAOX::app::synco::Simulation::combined(0,0,0);
    MagAOX::app::synco::Simulation::advance(0);
    MagAOX::app::synco::Simulation::popMainWait;
#endif
    // clang-format on
}

/// Diagnostic FIFO saturation and resets retain accounting without changing resource admission.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "synco delayed-main FIFO is bounded across sessions", "[dmSyncopation][simulation][mainWait]" )
{
    using namespace MagAOX::dmSyncoSimulation;
    Simulation         simulation;
    Simulation::Config config;
    config.readerReleaseReady = config.pipelineBackend = true;
    config.virtualReaderNs                             = 60000;
    config.steps                                       = 1;
    REQUIRE( simulation.configure( config ) );
    const auto session = simulation.summary().session;

    for( uint64_t index = 0; index < mainWaitTraceCapacity + 2; ++index )
    {
        const int64_t start = 1000000000 + int64_t( index ) * 1000000;
        simulation.primary( 2 * index + 1, start - 1000 );
        simulation.combined( 2 * index + 101, 2 * index + 1, start );
        simulation.primary( 2 * index + 2, start + 19829, 500000 );
        simulation.combined( 2 * index + 102, 2 * index + 2, start + 20829 );
        simulation.advance( start + 500000 );
    }

    auto status = simulation.mainWaitStatus();
    CHECK( status.enabled );
    CHECK( status.capacity == mainWaitTraceCapacity );
    CHECK( status.detected == mainWaitTraceCapacity + 2 );
    CHECK( status.queued == mainWaitTraceCapacity );
    CHECK( status.dropped == 2 );
    CHECK( status.drained == 0 );
    CHECK( simulation.summary().pipeline.mainSdkWaits == status.detected );
    CHECK( simulation.summary().pipeline.mainAdmitted == 2 * status.detected );
    CHECK( simulation.summary().pipeline.mainCreditMisses == 0 );
    CHECK( simulation.summary().pipeline.auxiliaryAdmitted == 0 );
    simulation.reset();
    config.pipelineBackend = false;
    REQUIRE( simulation.configure( config ) );
    CHECK_FALSE( simulation.mainWaitStatus().enabled );
    MainWaitEvent event;

    for( uint64_t index = 0; index < mainWaitTraceCapacity; ++index )
    {
        REQUIRE( simulation.popMainWait( event ) );
        CHECK( event.id == index + 1 );
        CHECK( event.session == session );
    }

    CHECK_FALSE( simulation.popMainWait( event ) );
    status = simulation.mainWaitStatus();
    CHECK( status.detected == status.drained + status.dropped + status.queued );
    REQUIRE( simulation.configure( config ) );
    feed( simulation, 1, 2000000000 );
    simulation.advance( 2001000000 );
    CHECK( simulation.mainWaitStatus().detected == status.detected );
    CHECK_FALSE( simulation.popMainWait( event ) );

    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    MagAOX::app::synco::Simulation::configure({});
    MagAOX::app::synco::Simulation::reset;
    MagAOX::app::synco::Simulation::mainWaitStatus;
    MagAOX::app::synco::Simulation::popMainWait;
#endif
    // clang-format on
}

} // namespace dmSyncopationTest
} // namespace libXWCTest
