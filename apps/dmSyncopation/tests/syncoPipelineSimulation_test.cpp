/** \file syncoPipelineSimulation_test.cpp
 * \brief Offline fixed-storage controller-resource forecasts and zero-added-main-wait admission tests.
 * \ingroup dmSyncopation_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "../syncoPipelineSimulation.hpp"

#include <limits>
#include <type_traits>

namespace libXWCTest
{
namespace dmSyncopationTest
{
using MagAOX::app::synco::PipelineSimulation;

/// \cond TEST_HARNESS
/// Verify conservation includes credits reserved before their input arrives and separately discarded work.
static void conserved( const PipelineSimulation &simulation /**< [in] Resource model under test. */ )
{
    const auto &summary = simulation.summary();
    REQUIRE( summary.mainAdmitted + summary.auxiliaryAdmitted ==
             summary.retired + summary.discarded + summary.inFlight );
    REQUIRE( summary.inFlight <= PipelineSimulation::capacity );
    REQUIRE( summary.maximumInFlight <= PipelineSimulation::capacity );
}
/// \endcond

/** \brief Conditional stages overlap while SDK and post ownership remain individually serialized. */
TEST_CASE( "synco pipeline independent stages", "[dmSyncopation][pipelineSimulation]" )
{
    static_assert( std::is_trivially_copyable<PipelineSimulation>::value, "Forecasts must use fixed-storage copies." );
    PipelineSimulation simulation;
    constexpr int64_t  base = 1000000;
    REQUIRE( simulation.configure( {} ) );
    REQUIRE( simulation.summary().enabled );
    REQUIRE( simulation.summary().capacity == 3 );
    REQUIRE( simulation.summary().prepareNs == 35000 );
    REQUIRE( simulation.summary().sdkServiceNs == 120000 );
    REQUIRE( simulation.summary().postNs == 150000 );
    REQUIRE( simulation.main( 1, base ) );
    simulation.advance( base + 54999 );
    REQUIRE( simulation.summary().sdkReadyNs == 0 );
    simulation.advance( base + 55000 );
    REQUIRE( simulation.summary().sdkReadyNs == base + 175000 );
    REQUIRE( simulation.summary().postReadyNs == 0 );
    REQUIRE( simulation.auxiliary( 1, 1, base + 100000, base + 166667, base + 475000 ) ==
             PipelineSimulation::Result::accepted );
    REQUIRE( simulation.summary().inFlight == 2 );
    REQUIRE( simulation.summary().auxiliaryAdmitted == 1 );
    REQUIRE( simulation.summary().sdkReadyNs == base + 175000 );
    REQUIRE( simulation.summary().postReadyNs == 0 );
    simulation.advance( base + 175000 );
    REQUIRE( simulation.summary().postReadyNs == base + 325000 );
    simulation.advance( base + 221667 );
    REQUIRE( simulation.summary().sdkReadyNs == base + 341667 );
    REQUIRE( simulation.summary().postReadyNs == base + 325000 );
    simulation.advance( base + 325000 );
    REQUIRE( simulation.summary().retired == 1 );
    REQUIRE( simulation.summary().inFlight == 1 );
    simulation.advance( base + 341667 );
    REQUIRE( simulation.summary().postReadyNs == base + 491667 );
    simulation.advance( base + 491666 );
    REQUIRE( simulation.summary().inFlight == 1 );
    simulation.advance( base + 491667 );
    REQUIRE( simulation.summary().retired == 2 );
    REQUIRE( simulation.summary().inFlight == 0 );
    REQUIRE( simulation.summary().mainSdkWaits == 0 );
    conserved( simulation );
}

/** \brief The final three-step slot receives no credit for preparation of its following main. */
TEST_CASE( "synco pipeline zero added main SDK wait", "[dmSyncopation][pipelineSimulation]" )
{
    PipelineSimulation simulation;
    constexpr int64_t  base = 1000000;
    REQUIRE( simulation.configure( {} ) );
    REQUIRE( simulation.main( 1, base ) );
    REQUIRE( simulation.auxiliary( 1, 1, base + 100000, base + 166667, base + 475000 ) ==
             PipelineSimulation::Result::accepted );
    REQUIRE( simulation.auxiliary( 1, 2, base + 250000, base + 333333, base + 475000 ) ==
             PipelineSimulation::Result::mainSdkBusy );
    REQUIRE( simulation.summary().mainClearanceShortfallNs == 33333 );
    REQUIRE( simulation.summary().auxiliaryAdmitted == 1 );
    REQUIRE( simulation.summary().inFlight == 2 );
    REQUIRE( simulation.summary().sdkReadyNs == base + 175000 );
    REQUIRE( simulation.summary().postReadyNs == 0 );

    // Equality clears the following main without charging its preparation against the SDK budget.
    REQUIRE( simulation.auxiliary( 1, 2, base + 250000, base + 333333, base + 508333 ) ==
             PipelineSimulation::Result::accepted );
    REQUIRE( simulation.summary().inFlight == 3 );
    REQUIRE( simulation.main( 2, base + 508333 ) );
    simulation.advance( base + 563333 );
    REQUIRE( simulation.summary().mainSdkWaits == 0 );
    REQUIRE( simulation.summary().maximumMainWaitNs == 0 );
    conserved( simulation );
}

/** \brief Forecasts reject both preparation and SDK queuing without committing any predicted transitions. */
TEST_CASE( "synco pipeline candidate contention", "[dmSyncopation][pipelineSimulation]" )
{
    constexpr int64_t  base = 1000000;
    PipelineSimulation simulation;
    REQUIRE( simulation.configure( {} ) );
    REQUIRE( simulation.main( 1, base ) );
    REQUIRE( simulation.auxiliary( 1, 1, base + 10000, base + 20000, base + 500000 ) ==
             PipelineSimulation::Result::prepareBusy );
    REQUIRE( simulation.summary().sdkReadyNs == 0 );
    REQUIRE( simulation.summary().auxiliaryAdmitted == 0 );
    REQUIRE( simulation.summary().maximumInFlight == 1 );
    REQUIRE( simulation.auxiliary( 1, 1, base + 10000, base + 60000, base + 500000 ) ==
             PipelineSimulation::Result::sdkBusy );
    REQUIRE( simulation.summary().sdkReadyNs == 0 );
    REQUIRE( simulation.summary().postReadyNs == 0 );
    REQUIRE( simulation.summary().mainSdkWaits == 0 );
    simulation.advance( base + 175000 );
    REQUIRE( simulation.summary().sdkReadyNs == base + 175000 );
    REQUIRE( simulation.summary().postReadyNs == base + 325000 );
    conserved( simulation );
}

/** \brief Credit forecasts preserve a free next-main lease even when SDK and monitor would already be idle. */
TEST_CASE( "synco pipeline next main credit protection", "[dmSyncopation][pipelineSimulation]" )
{
    PipelineSimulation         simulation;
    PipelineSimulation::Config config;
    config.postNs          = 500000;
    constexpr int64_t base = 1000000;
    REQUIRE( simulation.configure( config ) );
    REQUIRE( simulation.main( 1, base ) );
    REQUIRE( simulation.auxiliary( 1, 1, base + 100000, base + 166667, base + 600000 ) ==
             PipelineSimulation::Result::accepted );
    REQUIRE( simulation.auxiliary( 1, 2, base + 250000, base + 333333, base + 600000 ) ==
             PipelineSimulation::Result::mainCreditsFull );
    REQUIRE( simulation.summary().auxiliaryAdmitted == 1 );
    REQUIRE( simulation.summary().inFlight == 2 );
    REQUIRE( simulation.summary().mainClearanceShortfallNs == 0 );
    REQUIRE( simulation.auxiliary( 1, 2, base + 250000, base + 333333, base + 675000 ) ==
             PipelineSimulation::Result::accepted );
    REQUIRE( simulation.summary().inFlight == 3 );
    REQUIRE( simulation.main( 2, base + 675000 ) );
    REQUIRE( simulation.summary().retired == 1 );
    REQUIRE( simulation.summary().inFlight == 3 );
    simulation.advance( base + 730000 );
    REQUIRE( simulation.summary().mainSdkWaits == 0 );
    REQUIRE( simulation.summary().maximumInFlight == 3 );
    conserved( simulation );
}

/** \brief An unexpected earlier main executes before a previously reserved future auxiliary input. */
TEST_CASE( "synco pipeline causal future reservation", "[dmSyncopation][pipelineSimulation]" )
{
    PipelineSimulation simulation;
    constexpr int64_t  base = 1000000;
    REQUIRE( simulation.configure( {} ) );
    REQUIRE( simulation.main( 1, base ) );
    REQUIRE( simulation.auxiliary( 1, 1, base + 100000, base + 400000, base + 1000000 ) ==
             PipelineSimulation::Result::accepted );
    REQUIRE( simulation.summary().inFlight == 2 );
    REQUIRE( simulation.summary().sdkReadyNs == base + 175000 );
    REQUIRE( simulation.main( 2, base + 200000 ) );
    simulation.advance( base + 255000 );
    REQUIRE( simulation.summary().sdkReadyNs == base + 375000 );
    REQUIRE( simulation.summary().maximumInFlight == 3 );
    simulation.advance( base + 455000 );
    REQUIRE( simulation.summary().sdkReadyNs == base + 575000 );
    REQUIRE( simulation.summary().postReadyNs == base + 525000 );
    simulation.advance( base + 525000 );
    REQUIRE( simulation.summary().retired == 2 );
    REQUIRE( simulation.summary().inFlight == 1 );
    REQUIRE( simulation.summary().mainSdkWaits == 0 );
    simulation.advance( base + 575000 );
    REQUIRE( simulation.summary().postReadyNs == base + 725000 );
    simulation.advance( base + 725000 );
    REQUIRE( simulation.summary().retired == 3 );
    conserved( simulation );
}

/** \brief An exact input-time tie favors the genuine main rather than its earlier auxiliary reservation. */
TEST_CASE( "synco pipeline simultaneous genuine input priority", "[dmSyncopation][pipelineSimulation]" )
{
    PipelineSimulation simulation;
    constexpr int64_t  base = 1000000;
    REQUIRE( simulation.configure( {} ) );
    REQUIRE( simulation.main( 1, base ) );
    REQUIRE( simulation.auxiliary( 1, 1, base + 100000, base + 400000, base + 1000000 ) ==
             PipelineSimulation::Result::accepted );
    REQUIRE( simulation.main( 2, base + 400000 ) );
    simulation.advance( base + 575000 );
    REQUIRE( simulation.summary().sdkReadyNs == base + 695000 );
    REQUIRE( simulation.summary().mainSdkWaits == 0 );
    REQUIRE( simulation.summary().maximumMainWaitNs == 0 );
    simulation.advance( base + 1000000 );
    REQUIRE( simulation.summary().retired == 3 );
    conserved( simulation );
}

/** \brief Unexpected mains retain native nonpreemption and expose queue waits and genuine credit misses. */
TEST_CASE( "synco pipeline actual main contention", "[dmSyncopation][pipelineSimulation]" )
{
    PipelineSimulation simulation;
    constexpr int64_t  base = 1000000;
    REQUIRE( simulation.configure( {} ) );
    REQUIRE( simulation.main( 1, base ) );
    REQUIRE( simulation.main( 2, base + 100000 ) );
    simulation.advance( base + 174999 );
    REQUIRE( simulation.summary().mainSdkWaits == 0 );
    simulation.advance( base + 175000 );
    REQUIRE( simulation.summary().mainSdkWaits == 1 );
    REQUIRE( simulation.summary().maximumMainWaitNs == 20000 );
    REQUIRE( simulation.summary().sdkReadyNs == base + 295000 );
    REQUIRE( simulation.main( 3, base + 200000 ) );
    REQUIRE_FALSE( simulation.main( 4, base + 210000 ) );
    REQUIRE( simulation.summary().mainCreditMisses == 1 );
    REQUIRE( simulation.summary().mainAdmitted == 3 );
    REQUIRE( simulation.auxiliary( 1, 1, base + 220000, base + 300000, base + 1000000 ) ==
             PipelineSimulation::Result::creditsFull );
    simulation.advance( base + 295000 );
    REQUIRE( simulation.summary().mainSdkWaits == 2 );
    REQUIRE( simulation.summary().maximumMainWaitNs == 40000 );
    simulation.advance( base + 1000000 );
    REQUIRE( simulation.summary().retired == 3 );
    REQUIRE( simulation.summary().inFlight == 0 );
    conserved( simulation );
}

/** \brief A main delayed only by earlier shared readers still records its added SDK-entry latency. */
TEST_CASE( "synco pipeline actual main preparation wait", "[dmSyncopation][pipelineSimulation]" )
{
    PipelineSimulation         simulation;
    PipelineSimulation::Config config;
    config.sdkServiceNs    = 1000;
    constexpr int64_t base = 1000000;
    REQUIRE( simulation.configure( config ) );
    REQUIRE( simulation.main( 1, base ) );
    REQUIRE( simulation.main( 2, base + 10000 ) );
    simulation.advance( base + 94999 );
    REQUIRE( simulation.summary().mainSdkWaits == 0 );
    simulation.advance( base + 95000 );
    REQUIRE( simulation.summary().mainSdkWaits == 1 );
    REQUIRE( simulation.summary().maximumMainWaitNs == 30000 );
    REQUIRE( simulation.summary().sdkReadyNs == base + 96000 );
    simulation.advance( base + 1000000 );
    REQUIRE( simulation.summary().retired == 2 );
    conserved( simulation );
}

/** \brief A long final-reader stage cannot consume the following main's preparation window. */
TEST_CASE( "synco pipeline last reader main protection", "[dmSyncopation][pipelineSimulation]" )
{
    PipelineSimulation         simulation;
    PipelineSimulation::Config config;
    config.readerNs        = 300000;
    constexpr int64_t base = 1000000;
    REQUIRE( simulation.configure( config ) );
    REQUIRE( simulation.main( 1, base ) );
    REQUIRE( simulation.auxiliary( 1, 1, base + 100000, base + 350000, base + 600000 ) ==
             PipelineSimulation::Result::mainSdkBusy );
    REQUIRE( simulation.summary().mainClearanceShortfallNs == 50000 );
    REQUIRE( simulation.summary().auxiliaryAdmitted == 0 );
    REQUIRE( simulation.auxiliary( 1, 1, base + 100000, base + 350000, base + 650000 ) ==
             PipelineSimulation::Result::accepted );
    REQUIRE( simulation.main( 2, base + 650000 ) );
    simulation.advance( base + 705000 );
    REQUIRE( simulation.summary().mainSdkWaits == 0 );
    conserved( simulation );
}

/** \brief Reset discards owned jobs once and quarantines their projected drain before credit reuse. */
TEST_CASE( "synco pipeline reset resource quarantine", "[dmSyncopation][pipelineSimulation]" )
{
    PipelineSimulation simulation;
    constexpr int64_t  base = 1000000;
    REQUIRE( simulation.configure( {} ) );
    REQUIRE( simulation.main( 1, base ) );
    REQUIRE( simulation.auxiliary( 1, 1, base + 100000, base + 400000, base + 1000000 ) ==
             PipelineSimulation::Result::accepted );
    simulation.reset();
    REQUIRE( simulation.summary().inFlight == 0 );
    REQUIRE( simulation.summary().retired == 0 );
    REQUIRE( simulation.summary().discarded == 2 );
    REQUIRE( simulation.summary().resets == 1 );
    REQUIRE( simulation.summary().sdkReadyNs == base + 725000 );
    REQUIRE( simulation.summary().postReadyNs == base + 725000 );
    REQUIRE_FALSE( simulation.main( 2, base + 200000 ) );
    REQUIRE( simulation.summary().mainCreditMisses == 1 );
    REQUIRE( simulation.auxiliary( 1, 2, base + 250000, base + 333333, base + 1000000 ) ==
             PipelineSimulation::Result::creditsFull );
    simulation.reset();
    REQUIRE( simulation.summary().discarded == 2 );
    REQUIRE( simulation.summary().resets == 2 );
    REQUIRE( simulation.summary().postReadyNs == base + 725000 );
    REQUIRE( simulation.main( 2, base + 725000 ) );
    simulation.advance( base + 1050000 );
    REQUIRE( simulation.summary().retired == 1 );
    REQUIRE( simulation.summary().discarded == 2 );
    REQUIRE( simulation.summary().inFlight == 0 );
    conserved( simulation );

    REQUIRE( simulation.configure( {} ) );
    REQUIRE( simulation.summary().mainAdmitted == 0 );
    REQUIRE( simulation.summary().auxiliaryAdmitted == 0 );
    REQUIRE( simulation.summary().discarded == 0 );
    REQUIRE( simulation.summary().mainCreditMisses == 0 );
    REQUIRE( simulation.summary().resets == 0 );
    REQUIRE( simulation.main( 1, base ) );
    conserved( simulation );
}

/** \brief Unknown source activity retains a conservative fence even without any previously owned command. */
TEST_CASE( "synco pipeline unknown command quarantine", "[dmSyncopation][pipelineSimulation]" )
{
    PipelineSimulation simulation;
    constexpr int64_t  base = 1000000;
    REQUIRE( simulation.configure( {} ) );
    simulation.invalidate( base );
    const int64_t fence = base + 3 * ( 20000 + 60000 + 120000 + 150000 );
    REQUIRE( simulation.summary().resets == 1 );
    REQUIRE( simulation.summary().discarded == 0 );
    REQUIRE( simulation.summary().sdkReadyNs == fence );
    REQUIRE( simulation.summary().postReadyNs == fence );
    REQUIRE_FALSE( simulation.main( 1, fence - 1 ) );
    REQUIRE( simulation.auxiliary( 1, 1, fence - 1, fence, fence + 500000 ) ==
             PipelineSimulation::Result::creditsFull );
    REQUIRE( simulation.main( 1, fence ) );
    REQUIRE( simulation.summary().mainCreditMisses == 1 );
    REQUIRE( simulation.summary().mainAdmitted == 1 );
    conserved( simulation );
}

/** \brief Invalid configurations and unrepresentable times never allocate or overwrite a command credit. */
TEST_CASE( "synco pipeline checked arithmetic", "[dmSyncopation][pipelineSimulation]" )
{
    PipelineSimulation         simulation;
    PipelineSimulation::Config config;
    constexpr int64_t          base    = 1000000;
    const auto                 maximum = std::numeric_limits<int64_t>::max();
    REQUIRE_FALSE( simulation.summary().enabled );
    REQUIRE_FALSE( simulation.main( 1, base ) );
    REQUIRE( simulation.auxiliary( 1, 1, base, base + 100000, base + 500000 ) == PipelineSimulation::Result::invalid );
    simulation.reset();
    simulation.invalidate( base );
    REQUIRE( simulation.summary().resets == 0 );
    config.deliveryNs = -1;
    REQUIRE_FALSE( simulation.configure( config ) );
    config           = {};
    config.prepareNs = 40001;
    REQUIRE_FALSE( simulation.configure( config ) );
    config              = {};
    config.sdkServiceNs = 0;
    REQUIRE_FALSE( simulation.configure( config ) );
    config        = {};
    config.postNs = maximum;
    REQUIRE_FALSE( simulation.configure( config ) );
    config            = {};
    config.deliveryNs = 1000000001;
    REQUIRE_FALSE( simulation.configure( config ) );
    config           = {};
    config.prepareNs = 1000000001;
    REQUIRE_FALSE( simulation.configure( config ) );
    config          = {};
    config.readerNs = 1000000001;
    REQUIRE_FALSE( simulation.configure( config ) );
    config              = {};
    config.sdkServiceNs = 1000000001;
    REQUIRE_FALSE( simulation.configure( config ) );
    config        = {};
    config.postNs = 1000000001;
    REQUIRE_FALSE( simulation.configure( config ) );
    config.deliveryNs   = 0;
    config.prepareNs    = 1000000000;
    config.readerNs     = 1000000000;
    config.sdkServiceNs = 1000000000;
    config.postNs       = 1000000000;
    REQUIRE( simulation.configure( config ) );
    REQUIRE( simulation.configure( {} ) );
    REQUIRE( simulation.main( 1, base ) );
    simulation.advance( base + 100000 );
    REQUIRE_FALSE( simulation.main( 2, base + 99999 ) );
    REQUIRE( simulation.auxiliary( 1, 0, base + 100000, base + 200000, base + 500000 ) ==
             PipelineSimulation::Result::invalid );
    REQUIRE( simulation.auxiliary( 1, 1, base + 99999, base + 200000, base + 500000 ) ==
             PipelineSimulation::Result::invalid );
    REQUIRE( simulation.auxiliary( 1, 1, base + 200000, base + 100000, base + 500000 ) ==
             PipelineSimulation::Result::invalid );
    REQUIRE( simulation.auxiliary( 1, 1, base + 200000, base + 300000, base + 250000 ) ==
             PipelineSimulation::Result::invalid );
    REQUIRE( simulation.auxiliary( 1, 1, maximum - 2, maximum - 1, maximum ) == PipelineSimulation::Result::invalid );
    REQUIRE( simulation.summary().inFlight == 1 );
    REQUIRE( simulation.summary().mainAdmitted == 1 );
    REQUIRE( simulation.summary().auxiliaryAdmitted == 0 );
    simulation.advance( -1 );
    simulation.advance( base + 99999 );
    REQUIRE( simulation.summary().inFlight == 1 );
    REQUIRE_FALSE( simulation.main( 2, maximum - 1 ) );
    REQUIRE( simulation.summary().retired == 1 );
    REQUIRE( simulation.summary().inFlight == 0 );
    conserved( simulation );

    REQUIRE( simulation.configure( {} ) );
    simulation.invalidate( maximum - 1 );
    REQUIRE( simulation.summary().sdkReadyNs == maximum );
    REQUIRE_FALSE( simulation.main( 1, maximum - 1 ) );
    conserved( simulation );
}

} // namespace dmSyncopationTest
} // namespace libXWCTest
