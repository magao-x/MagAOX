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

/** \brief Large measured late/catch-up intervals retain counter labels and the median rather than nearest-slot
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

/** \brief An already committed cycle uses its frozen period when classifying a newly infeasible estimator value. */
TEST_CASE( "synco simulation feasibility belongs to frozen cycle", "[dmSyncopation][simulation]" )
{
    Simulation         simulation;
    Simulation::Config config;
    config.window = 2;
    REQUIRE( simulation.configure( config ) );
    const auto last = warm( simulation );
    feed( simulation, 41, last + 500000, 300000 );
    feed( simulation, 42, last + 1000000, 300000 );
    REQUIRE( simulation.summary().periodNs == 300000 );
    REQUIRE_FALSE( simulation.summary().feasible );
    const auto held = simulation.summary().outcomes[1].readerHold;
    simulation.advance( last + 1110000 );
    REQUIRE( simulation.summary().outcomes[1].readerHold == held + 1 );
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
    REQUIRE( simulation.configure( config ) );
    const auto last = warm( simulation );
    simulation.advance( last + 300000 );
    const auto active = simulation.summary().outcomes[1].wouldRun;
    feed( simulation, 41, last + 500000, 300000 );
    REQUIRE( simulation.summary().periodNs == 400000 );
    REQUIRE_FALSE( simulation.summary().feasible );
    simulation.advance( last + 760000 );
    REQUIRE( simulation.summary().outcomes[1].wouldRun == active + 1 );
    feed( simulation, 42, last + 900000, 400000 );
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

    SECTION( "silence and out-of-order inputs" )
    {
        simulation.advance( last + 4000001 );
        REQUIRE_FALSE( simulation.summary().valid );
        REQUIRE( simulation.summary().nextActiveNs == 0 );
        REQUIRE( simulation.summary().periodSamples == 0 );
        const auto invalid = simulation.summary().invalid;
        simulation.primary( 41, last + 1000000, 500000 );
        REQUIRE( simulation.summary().invalid == invalid + 1 );
        REQUIRE( simulation.summary().counter == 40 );
    }
}

/** \brief Continuing main notifications cannot keep an unavailable publication-period estimate fresh. */
TEST_CASE( "synco simulation period sample expiry", "[dmSyncopation][simulation]" )
{
    Simulation simulation;
    REQUIRE( simulation.configure( {} ) );
    auto observed = warm( simulation );

    for( uint64_t counter = 41; counter <= 50; ++counter )
    {
        observed += 500000;
        feed( simulation, counter, observed, 0 );
    }

    REQUIRE_FALSE( simulation.summary().valid );
    REQUIRE( simulation.summary().periodSamples == 0 );
    REQUIRE( simulation.summary().phaseSamples == 32 );
    REQUIRE( simulation.summary().nextActiveNs == 0 );
    REQUIRE( simulation.summary().counter == 50 );
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
}

} // namespace dmSyncopationTest
} // namespace libXWCTest
