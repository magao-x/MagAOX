/** \file dmReaderTiming_test.cpp
 * \brief Offline qualification, retention and lifecycle tests for DM reader-protection measurements.
 */
#include "../../../../tests/testXWC.hpp"
#include "../dmReaderTiming.hpp"

#include <algorithm>
#include <atomic>
#include <limits>
#include <thread>
#include <vector>

namespace libXWCTest
{
namespace dmReaderTimingTest
{

using Collector = MagAOX::app::dev::dmReaderTiming;
using namespace MagAOX::dmReaderTiming;
using MagAOX::dmCompletion::State;
using MagAOX::dmTiming::Outcome;

/// \cond DM_READER_TIMING_TEST_HELPERS
namespace
{

/// Create exact identities and independently clocked source observations without any hardware or shared streams.
struct Fixture
{
    Primary   m_primary;   ///< Named primary observation.
    Combined  m_combined;  ///< Combined command with its primary trace link.
    Completed m_completed; ///< Exact successful controller completion.

    /// Create one deterministic command at a 2 kHz cadence.
    explicit Fixture( uint64_t index = 0 /**< [in] Counter and local-time offset from the first command. */ );

    /// Enqueue every source observation without performing any consumer work.
    void capture( Collector &collector /**< [in,out] Collector receiving the three independent handoffs. */ ) const;

    /// Return a local update time after every source capture in this fixture.
    int64_t now() const;
};

Fixture::Fixture( uint64_t index )
{
    const int64_t local     = 1000000000 + static_cast<int64_t>( index ) * 500000;
    const int64_t external  = 1800000000000000000 + static_cast<int64_t>( index ) * 500000;
    m_primary.input         = { 11, 100 + index, external / 1000000000, external % 1000000000 };
    m_primary.observedNs    = local;
    m_combined.input        = { 22, 200 + index, ( external + 20000 ) / 1000000000, ( external + 20000 ) % 1000000000 };
    m_combined.primaryInode = m_primary.input.inode;
    m_combined.primaryCounter = m_primary.input.counter;
    m_combined.startNs        = external + 10000;
    m_combined.observedNs     = local + 30000;
    m_combined.budgetNs       = 155000;
    auto &record              = m_completed.record;
    record.session            = 900;
    record.epoch              = 3;
    record.sequence           = index + 1;
    record.state              = State::complete;
    record.inputInode         = m_combined.input.inode;
    record.inputCounter       = m_combined.input.counter;
    record.publicationSec     = m_combined.input.sec;
    record.publicationNsec    = m_combined.input.nsec;
    record.startedNs          = local + 35000;
    record.finishedNs         = local + 135000;
    record.outcome            = Outcome::success;
    record.publisherPid       = 1234;
    m_completed.observedNs    = local + 140000;
}

void Fixture::capture( Collector &collector ) const
{
    collector.primary( m_primary );
    collector.combined( m_combined );
    collector.completion( m_completed );
}

int64_t Fixture::now() const
{
    return std::max( { m_primary.observedNs, m_combined.observedNs, m_completed.observedNs } ) + 10000;
}

} // namespace
/// \endcond

/** \brief Qualify exact producer identities and the six reader-protection duration populations.
 * \ingroup dmReaderTiming_unit_test
 */
TEST_CASE( "DM reader timing joins qualified identities and retains exact worst evidence", "[dmReaderTiming]" )
{
    // clang-format off
#ifdef DM_READER_TIMING_TEST_DOXYGEN_REF
    MagAOX::app::dev::dmReaderTiming::configure;
    MagAOX::app::dev::dmReaderTiming::primary;
    MagAOX::app::dev::dmReaderTiming::combined;
    MagAOX::app::dev::dmReaderTiming::completion;
    MagAOX::app::dev::dmReaderTiming::update;
    MagAOX::app::dev::dmReaderTiming::summary;
#endif
    // clang-format on

    Collector collector;
    collector.configure( 2, 32 );
    const Fixture first;
    first.capture( collector );
    REQUIRE( collector.summary().total.events == 0 );
    collector.update( first.now() );
    const auto &summary = collector.summary();
    REQUIRE( summary.enabled );
    REQUIRE( summary.total.events == 3 );
    REQUIRE( summary.total.matched == 1 );
    REQUIRE( summary.pendingPrimary == 1 );
    REQUIRE( summary.pendingCombined == 0 );
    REQUIRE( summary.pendingCompletion == 0 );
    REQUIRE( summary.queued == 0 );
    REQUIRE( summary.worstValid );
    REQUIRE( summary.maxRequiredHoldNs == 105000 );
    REQUIRE( summary.maxObservedHoldNs == 110000 );
    REQUIRE( summary.overBudget == 0 );
    REQUIRE( summary.worst.baseNs == first.m_combined.observedNs );
    REQUIRE( summary.worst.primary.input.counter == first.m_primary.input.counter );
    REQUIRE( summary.worst.combined.input.counter == first.m_combined.input.counter );
    REQUIRE( summary.worst.completed.record.sequence == first.m_completed.record.sequence );
    REQUIRE( summary.worst.offsetNs == 105000 );
    REQUIRE( summary.worst.requiredNs == 105000 );
    REQUIRE( summary.worst.observedHoldNs == 110000 );
    REQUIRE( summary.worst.slackNs == 50000 );
    REQUIRE( summary.metrics[index( Metric::completionOffset )].mean == Approx( 105e-6 ) );
    REQUIRE( summary.metrics[index( Metric::requiredHold )].mean == Approx( 105e-6 ) );
    REQUIRE( summary.metrics[index( Metric::observedHold )].mean == Approx( 110e-6 ) );
    REQUIRE( summary.metrics[index( Metric::processing )].mean == Approx( 100e-6 ) );
    REQUIRE( summary.metrics[index( Metric::notification )].mean == Approx( 5e-6 ) );
    REQUIRE( summary.metrics[index( Metric::modelSlack )].mean == Approx( 50e-6 ) );

    const Fixture second( 1 );
    second.capture( collector );
    collector.update( second.now() );
    REQUIRE( summary.total.matched == 2 );
    REQUIRE( summary.span == Approx( 500e-6 ) );

    for( size_t metric = 0; metric < index( Metric::count ); ++metric )
    {
        REQUIRE( summary.metrics[metric].count == 2 );
        REQUIRE( summary.metrics[metric].valid );
        REQUIRE( summary.metrics[metric].stddev == Approx( 0 ) );
        REQUIRE( summary.metrics[metric].span == Approx( 500e-6 ) );
        REQUIRE( std::string( metricName( static_cast<Metric>( metric ) ) ) != "unknown" );
    }

    REQUIRE( std::string( metricName( Metric::count ) ) == "unknown" );
}

/** \brief Preserve signed negative completion offsets when the combined observer wakes after actual completion.
 * \ingroup dmReaderTiming_unit_test
 */
TEST_CASE( "DM reader timing separates actual completion from delayed observer release", "[dmReaderTiming]" )
{
    Collector collector;
    collector.configure( 2, 16 );
    Fixture fixture;
    fixture.m_combined.observedNs = fixture.m_completed.record.finishedNs + 50000;
    fixture.capture( collector );
    collector.update( fixture.now() );
    const auto &summary = collector.summary();
    REQUIRE( summary.total.matched == 1 );
    REQUIRE( summary.worst.offsetNs == -50000 );
    REQUIRE( summary.worst.requiredNs == 0 );
    REQUIRE( summary.worst.observedHoldNs == 0 );
    REQUIRE( summary.worst.slackNs == 205000 );
    REQUIRE( summary.metrics[index( Metric::completionOffset )].mean == Approx( -50e-6 ) );
    REQUIRE( summary.metrics[index( Metric::notification )].mean == Approx( 5e-6 ) );

    Fixture later( 1 );
    later.m_completed.observedNs += 100000;
    later.capture( collector );
    collector.update( later.now() );
    REQUIRE( summary.total.matched == 2 );
    REQUIRE( summary.maxObservedHoldNs == 210000 );
    REQUIRE( summary.maxRequiredHoldNs == 105000 );
    REQUIRE( summary.metrics[index( Metric::notification )].max == Approx( 105e-6 ) );
}

/** \brief Join all source arrival orders across separate ordinary updates, then reject replayed evidence.
 * \ingroup dmReaderTiming_unit_test
 */
TEST_CASE( "DM reader timing supports out of order delivery without double matching", "[dmReaderTiming]" )
{
    std::array<int, 3> order{ 0, 1, 2 };

    do
    {
        Collector collector;
        collector.configure( 2, 16 );
        const Fixture fixture;

        for( size_t item = 0; item < order.size(); ++item )
        {
            switch( order[item] )
            {
            case 0:
                collector.primary( fixture.m_primary );
                break;
            case 1:
                collector.combined( fixture.m_combined );
                break;
            case 2:
                collector.completion( fixture.m_completed );
                break;
            }

            collector.update( fixture.now() + static_cast<int64_t>( item ) );
            REQUIRE( collector.summary().total.matched == ( item == 2 ? 1 : 0 ) );
        }

        fixture.capture( collector );
        collector.update( fixture.now() + 4 );
        REQUIRE( collector.summary().total.matched == 1 );
        REQUIRE( collector.summary().total.duplicates == 3 );
    } while( std::next_permutation( order.begin(), order.end() ) );
}

/** \brief Require exact raw combined identity and external trace ordering before forming a timing sample.
 * \ingroup dmReaderTiming_unit_test
 */
TEST_CASE( "DM reader timing rejects mismatched identity and unordered producer clocks", "[dmReaderTiming]" )
{
    for( int variant = 0; variant < 5; ++variant )
    {
        Collector collector;
        collector.configure( 1, 16 );
        Fixture fixture;

        switch( variant )
        {
        case 0:
            ++fixture.m_completed.record.publicationNsec;
            break;
        case 1:
            --fixture.m_combined.startNs;
            fixture.m_combined.startNs -= 10000;
            break;
        case 2:
            fixture.m_combined.startNs += 10001;
            break;
        case 3:
            ++fixture.m_combined.primaryCounter;
            break;
        case 4:
            ++fixture.m_completed.record.inputInode;
            break;
        }

        fixture.capture( collector );
        collector.update( fixture.now() );
        REQUIRE( collector.summary().total.matched == 0 );
        REQUIRE_FALSE( collector.summary().worstValid );
        collector.update( fixture.now() + 2000000000 );

        if( variant < 2 )
        {
            REQUIRE( collector.summary().total.identityMismatch == 1 );
            REQUIRE( collector.summary().total.invalid == 1 );
        }
        else if( variant == 3 )
        {
            REQUIRE( collector.summary().total.missingPrimary == 1 );
        }
        else
        {
            REQUIRE( collector.summary().total.missingCombined == 1 );
        }
    }
}

/** \brief Treat busy, unavailable, error and unqualified records as distinct diagnostics, never as completions.
 * \ingroup dmReaderTiming_unit_test
 */
TEST_CASE( "DM reader timing qualifies completion state independently of notification sequence", "[dmReaderTiming]" )
{
    Collector collector;
    collector.configure( 2, 32 );
    Fixture fixture;
    auto    busy           = fixture.m_completed;
    busy.record.state      = State::busy;
    busy.record.finishedNs = 0;
    busy.record.outcome    = Outcome::unknown;
    collector.completion( busy );
    collector.completion( busy );
    fixture.capture( collector );
    collector.update( fixture.now() );
    REQUIRE( collector.summary().total.busy == 1 );
    REQUIRE( collector.summary().total.duplicates == 1 );
    REQUIRE( collector.summary().total.matched == 1 );

    const State states[]{ State::unavailable, State::error, State::unqualified };

    for( size_t index = 0; index < 3; ++index )
    {
        Fixture item( index + 1 );
        item.m_completed.record.state   = states[index];
        item.m_completed.record.outcome = index == 2 ? Outcome::noSend : Outcome::failed;
        item.capture( collector );
        collector.update( item.now() );
    }

    REQUIRE( collector.summary().total.unavailable == 1 );
    REQUIRE( collector.summary().total.error == 1 );
    REQUIRE( collector.summary().total.unqualified == 1 );
    REQUIRE( collector.summary().total.matched == 1 );

    for( const auto outcome :
         { Outcome::unknown, Outcome::failed, Outcome::partial, Outcome::noSend, Outcome::rejected } )
    {
        Collector invalid;
        invalid.configure( 2, 8 );
        Fixture item;
        item.m_completed.record.outcome = outcome;
        item.capture( invalid );
        invalid.update( item.now() );
        REQUIRE( invalid.summary().total.invalid == 1 );
        REQUIRE( invalid.summary().total.matched == 0 );
    }
}

/** \brief Check capture freshness against completion time while allowing a later consumer to retain lifetime evidence.
 * \ingroup dmReaderTiming_unit_test
 */
TEST_CASE( "DM reader timing separates capture freshness from consumer scheduling delay", "[dmReaderTiming]" )
{
    Collector collector;
    collector.configure( 1, 16 );
    const Fixture fixture;
    fixture.capture( collector );
    collector.update( fixture.now() + 5000000000 );
    REQUIRE( collector.summary().total.matched == 1 );
    REQUIRE( collector.summary().total.invalid == 0 );
    REQUIRE( collector.summary().worstValid );
    REQUIRE( collector.summary().metrics[0].count == 0 );
    REQUIRE( collector.summary().age == -1 );

    SECTION( "stale at capture" )
    {
        Fixture stale( 1 );
        stale.m_completed.observedNs = stale.m_completed.record.finishedNs + 1000000001;
        stale.capture( collector );
        collector.update( stale.now() + 5000000000 );
        REQUIRE( collector.summary().total.invalid == 1 );
        REQUIRE( collector.summary().total.matched == 1 );
    }

    SECTION( "completion timestamp lies after its observer" )
    {
        Fixture invalid( 1 );
        invalid.m_completed.observedNs = invalid.m_completed.record.finishedNs - 1;
        invalid.capture( collector );
        collector.update( fixture.now() + 6000000000 );
        REQUIRE( collector.summary().total.invalid == 1 );
        REQUIRE( collector.summary().total.matched == 1 );
    }
}

/** \brief Make absent source evidence, capacity eviction and individual queue loss visible without fabricating joins.
 * \ingroup dmReaderTiming_unit_test
 */
TEST_CASE( "DM reader timing bounds independent queues and pending populations", "[dmReaderTiming]" )
{
    Collector collector;
    collector.configure( 1, 1 );
    const Fixture first;
    const Fixture second( 1 );
    first.capture( collector );
    second.capture( collector );
    collector.update( second.now() );
    REQUIRE( collector.summary().total.matched == 1 );
    REQUIRE( collector.summary().total.droppedPrimary == 1 );
    REQUIRE( collector.summary().total.droppedCombined == 1 );
    REQUIRE( collector.summary().total.droppedCompletion == 1 );
    REQUIRE( collector.summary().pendingPrimary == 1 );
    collector.completion( second.m_completed );
    collector.update( second.now() + 1 );
    REQUIRE( collector.summary().pendingCompletion == 1 );
    const Fixture third( 2 );
    collector.completion( third.m_completed );
    collector.update( third.now() );
    REQUIRE( collector.summary().pendingCompletion == 1 );
    REQUIRE( collector.summary().total.evicted == 1 );
    REQUIRE( collector.summary().total.missingCombined == 1 );
    collector.update( third.now() + 2000000000 );
    REQUIRE( collector.summary().pendingCompletion == 0 );
    REQUIRE( collector.summary().pendingPrimary == 0 );
    REQUIRE( collector.summary().total.missingCombined == 2 );
    REQUIRE( collector.summary().total.expired == 2 );
    REQUIRE( collector.summary().total.matched == 1 );

    Collector missingPrimary;
    missingPrimary.configure( 1, 8 );
    missingPrimary.combined( first.m_combined );
    missingPrimary.completion( first.m_completed );
    missingPrimary.update( first.now() );
    REQUIRE( missingPrimary.summary().pendingCombined == 1 );
    REQUIRE( missingPrimary.summary().pendingCompletion == 1 );
    missingPrimary.update( first.now() + 2000000000 );
    REQUIRE( missingPrimary.summary().total.missingPrimary == 1 );
    REQUIRE( missingPrimary.summary().total.expired == 2 );
}

/** \brief Invalidate queued and pending old-source joins without erasing lifetime extreme evidence or requalifying
 * replay.
 * \ingroup dmReaderTiming_unit_test
 */
TEST_CASE( "DM reader timing source resets retain lifetime evidence and reject old attempts", "[dmReaderTiming]" )
{
    // clang-format off
#ifdef DM_READER_TIMING_TEST_DOXYGEN_REF
    MagAOX::app::dev::dmReaderTiming::resetPrimary;
    MagAOX::app::dev::dmReaderTiming::resetCombined;
    MagAOX::app::dev::dmReaderTiming::resetCompletion;
#endif
    // clang-format on

    for( int source = 0; source < 3; ++source )
    {
        Collector collector;
        collector.configure( 2, 16 );
        const Fixture first;
        first.capture( collector );
        collector.update( first.now() );
        const auto    session = collector.summary().session;
        const Fixture second( 1 );
        second.capture( collector );

        switch( source )
        {
        case 0:
            collector.resetPrimary();
            break;
        case 1:
            collector.resetCombined();
            break;
        case 2:
            collector.resetCompletion();
            break;
        }

        collector.update( second.now() );
        REQUIRE( collector.summary().session == session + 1 );
        REQUIRE( collector.summary().total.sourceResets == 1 );
        REQUIRE( collector.summary().total.staleEpoch == 3 );
        REQUIRE( collector.summary().total.matched == 1 );
        REQUIRE( collector.summary().metrics[0].count == 0 );
        REQUIRE( collector.summary().worstValid );
        REQUIRE( collector.summary().worst.completed.record.sequence == 1 );
        first.capture( collector );
        collector.update( second.now() + 1 );
        REQUIRE( collector.summary().total.matched == 1 );
        REQUIRE( collector.summary().total.duplicates >= 1 );
        second.capture( collector );
        collector.update( second.now() + 2 );
        REQUIRE( collector.summary().total.matched == 2 );
    }
}

/** \brief Enforce monotonic completion provenance across publisher resets and reject stale-publisher replay.
 * \ingroup dmReaderTiming_unit_test
 */
TEST_CASE( "DM reader timing publisher changes invalidate peers and never roll provenance backward",
           "[dmReaderTiming]" )
{
    Collector collector;
    collector.configure( 2, 16 );
    const Fixture first;
    first.capture( collector );
    collector.update( first.now() );
    Fixture second( 1 );
    second.m_completed.record.session  = 901;
    second.m_completed.record.epoch    = 1;
    second.m_completed.record.sequence = 1;
    second.capture( collector );
    collector.update( second.now() );
    REQUIRE( collector.summary().total.sourceResets == 1 );
    REQUIRE( collector.summary().total.staleEpoch == 2 );
    REQUIRE( collector.summary().total.matched == 1 );
    const auto invalid = collector.summary().total.invalid;
    collector.completion( first.m_completed );
    collector.update( second.now() + 1 );
    REQUIRE( collector.summary().total.invalid == invalid + 1 );
    REQUIRE( collector.summary().total.sourceResets == 1 );
    Fixture third( 2 );
    third.m_completed.record.session  = 901;
    third.m_completed.record.epoch    = 1;
    third.m_completed.record.sequence = 2;
    third.capture( collector );
    collector.update( third.now() );
    REQUIRE( collector.summary().total.matched == 2 );
    REQUIRE( collector.summary().worst.completed.record.session == 900 );
}

/** \brief Expose coalesced attempt/counter progression and preserve lifetime maxima outside the trailing window.
 * \ingroup dmReaderTiming_unit_test
 */
TEST_CASE( "DM reader timing reports gaps and distinguishes worst hold from worst observer delay", "[dmReaderTiming]" )
{
    Collector collector;
    collector.configure( 1, 16 );
    const Fixture first;
    first.capture( collector );
    collector.update( first.now() );
    Fixture later( 4 );
    later.m_completed.record.finishedNs += 80000;
    later.m_completed.observedNs += 180000;
    later.capture( collector );
    collector.update( later.now() );
    REQUIRE( collector.summary().total.sequenceGaps == 3 );
    REQUIRE( collector.summary().total.counterGaps == 3 );
    REQUIRE( collector.summary().total.matched == 2 );
    REQUIRE( collector.summary().maxRequiredHoldNs == 185000 );
    REQUIRE( collector.summary().maxObservedHoldNs == 290000 );
    REQUIRE( collector.summary().overBudget == 1 );
    REQUIRE( collector.summary().worst.completed.record.sequence == 5 );
    REQUIRE( collector.summary().worst.slackNs == -30000 );
    REQUIRE( collector.summary().metrics[index( Metric::modelSlack )].min == Approx( -30e-6 ) );
    collector.update( later.now() + 2000000000 );
    REQUIRE( collector.summary().metrics[0].count == 0 );
    REQUIRE( collector.summary().maxRequiredHoldNs == 185000 );
    REQUIRE( collector.summary().overBudget == 1 );
    REQUIRE( collector.summary().worst.completed.record.sequence == 5 );
}

/** \brief Reject malformed timestamps, arithmetic overflow, incoherent captures and clocks that move backward.
 * \ingroup dmReaderTiming_unit_test
 */
TEST_CASE( "DM reader timing rejects invalid metadata and clock domains", "[dmReaderTiming]" )
{
    SECTION( "explicit incoherence" )
    {
        Collector collector;
        collector.configure( 1, 8 );
        const Fixture fixture;
        collector.primary( fixture.m_primary, false );
        collector.combined( fixture.m_combined, false );
        collector.completion( fixture.m_completed, false );
        collector.update( fixture.now() );
        REQUIRE( collector.summary().total.incoherent == 3 );
        REQUIRE( collector.summary().total.matched == 0 );
    }

    SECTION( "future capture" )
    {
        Collector collector;
        collector.configure( 1, 8 );
        const Fixture fixture;
        fixture.capture( collector );
        collector.update( fixture.m_primary.observedNs - 1 );
        REQUIRE( collector.summary().total.invalid == 3 );
        REQUIRE( collector.summary().total.matched == 0 );
    }

    SECTION( "invalid producer timestamp" )
    {
        Collector collector;
        collector.configure( 1, 8 );
        Fixture fixture;
        fixture.m_primary.input.sec = std::numeric_limits<int64_t>::max();
        fixture.capture( collector );
        collector.update( fixture.now() );
        REQUIRE( collector.summary().total.invalid == 1 );
        REQUIRE( collector.summary().total.matched == 0 );
    }

    SECTION( "slack overflow from a negative offset" )
    {
        Collector collector;
        collector.configure( 1, 8 );
        Fixture fixture;
        fixture.m_combined.observedNs = fixture.m_completed.record.finishedNs + 1;
        fixture.m_combined.budgetNs   = std::numeric_limits<int64_t>::max();
        fixture.capture( collector );
        collector.update( fixture.now() );
        REQUIRE( collector.summary().total.invalid == 1 );
        REQUIRE( collector.summary().total.matched == 0 );
    }

    SECTION( "backward update invalidates pending epochs" )
    {
        Collector collector;
        collector.configure( 1, 8 );
        const Fixture fixture;
        fixture.capture( collector );
        collector.update( fixture.now() );
        collector.update( fixture.now() - 1 );
        REQUIRE( collector.summary().total.invalid == 1 );
        REQUIRE( collector.summary().total.sourceResets == 1 );
        REQUIRE( collector.summary().total.matched == 1 );
        REQUIRE( collector.summary().metrics[0].count == 0 );
        REQUIRE( collector.summary().worstValid );
        collector.update( -1 );
        REQUIRE( collector.summary().total.invalid == 2 );
    }
}

/** \brief Disable without allocations or source effects when configuration is zero or nonfinite.
 * \ingroup dmReaderTiming_unit_test
 */
TEST_CASE( "DM reader timing validates configuration under real time compiler flags", "[dmReaderTiming]" )
{
    for( const auto window :
         { 0.0, -1.0, 1e12, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN() } )
    {
        Collector collector;
        collector.configure( window, 16 );
        REQUIRE_FALSE( collector.enabled() );
        const Fixture fixture;
        fixture.capture( collector );
        collector.resetPrimary();
        collector.resetCombined();
        collector.resetCompletion();
        collector.update( fixture.now() );
        REQUIRE_FALSE( collector.summary().enabled );
        REQUIRE( collector.summary().total.events == 0 );
        REQUIRE( collector.summary().capacity == 0 );
    }

    Collector collector;
    collector.configure( 1, 0 );
    REQUIRE_FALSE( collector.enabled() );
    collector.configure( 1, 4 );
    REQUIRE( collector.enabled() );
    const Fixture fixture;
    fixture.capture( collector );
    collector.update( fixture.now() );
    REQUIRE( collector.summary().total.matched == 1 );
    collector.configure( 1, 4 );
    REQUIRE( collector.summary().total.events == 0 );
    REQUIRE( collector.summary().total.evicted == 0 );
    REQUIRE_FALSE( collector.summary().worstValid );
    fixture.capture( collector );
    collector.update( fixture.now() );
    REQUIRE( collector.summary().total.matched == 1 );
}

/** \brief Reuse a named primary for multiple distinct combined publications without inventing a primary loss on expiry.
 * \ingroup dmReaderTiming_unit_test
 */
TEST_CASE( "DM reader timing reuses primary evidence for distinct combined commands", "[dmReaderTiming]" )
{
    Collector collector;
    collector.configure( 1, 16 );
    const Fixture first;
    first.capture( collector );
    collector.update( first.now() );
    Fixture second( 1 );
    second.m_combined.primaryCounter = first.m_primary.input.counter;
    second.m_combined.budgetNs       = 100000;
    collector.combined( second.m_combined );
    collector.completion( second.m_completed );
    collector.update( second.now() );
    REQUIRE( collector.summary().total.matched == 2 );
    REQUIRE( collector.summary().total.primary == 1 );
    REQUIRE( collector.summary().total.missingPrimary == 0 );
    REQUIRE( collector.summary().overBudget == 1 );
    collector.update( second.now() + 2000000000 );
    REQUIRE( collector.summary().total.expired == 1 );
    REQUIRE( collector.summary().total.missingPrimary == 0 );
    REQUIRE( collector.summary().total.missingCombined == 0 );
}

/** \brief Preserve phase/source epochs across implicit counter reset and reject reused completion identities.
 * \ingroup dmReaderTiming_unit_test
 */
TEST_CASE( "DM reader timing rejects reused callbacks and exposes implicit source restart", "[dmReaderTiming]" )
{
    Collector collector;
    collector.configure( 2, 16 );
    const Fixture first;
    first.capture( collector );
    collector.update( first.now() );
    auto replay = first.m_completed;
    ++replay.record.sequence;
    collector.completion( replay );
    collector.update( first.now() + 1 );
    REQUIRE( collector.summary().total.duplicates == 1 );
    REQUIRE( collector.summary().total.matched == 1 );

    Fixture reset( 1 );
    reset.m_primary.input.counter   = 1;
    reset.m_combined.primaryCounter = 1;
    reset.capture( collector );
    collector.update( reset.now() );
    REQUIRE( collector.summary().total.sourceResets == 1 );
    REQUIRE( collector.summary().total.staleEpoch == 1 );
    REQUIRE( collector.summary().total.matched == 1 );
    REQUIRE( collector.summary().worstValid );

    Fixture after( 2 );
    after.m_primary.input.counter   = 2;
    after.m_combined.primaryCounter = 2;
    after.capture( collector );
    collector.update( after.now() );
    REQUIRE( collector.summary().total.matched == 2 );
}

/** \brief Drain independent source owners concurrently without overwriting unread evidence or generating false joins.
 * \ingroup dmReaderTiming_unit_test
 */
TEST_CASE( "DM reader timing maintains ownership with three concurrent capture producers", "[dmReaderTiming]" )
{
    constexpr size_t count = 512;
    Collector        collector;
    collector.configure( 2, count * 2 );
    std::vector<Fixture> fixtures;
    fixtures.reserve( count );

    for( size_t index = 0; index < count; ++index )
    {
        fixtures.emplace_back( index );
    }

    std::atomic<unsigned> finished{ 0 };
    std::thread           primary(
        [&]
        {
            for( const auto &fixture : fixtures )
            {
                collector.primary( fixture.m_primary );
            }

            finished.fetch_add( 1, std::memory_order_release );
        } );
    std::thread combined(
        [&]
        {
            for( const auto &fixture : fixtures )
            {
                collector.combined( fixture.m_combined );
            }

            finished.fetch_add( 1, std::memory_order_release );
        } );
    std::thread completed(
        [&]
        {
            for( const auto &fixture : fixtures )
            {
                collector.completion( fixture.m_completed );
            }

            finished.fetch_add( 1, std::memory_order_release );
        } );

    while( finished.load( std::memory_order_acquire ) < 3 )
    {
        collector.update( fixtures.back().now() );
        std::this_thread::yield();
    }

    primary.join();
    combined.join();
    completed.join();
    collector.update( fixtures.back().now() );
    const auto &summary = collector.summary();
    REQUIRE( summary.total.events == count * 3 );
    REQUIRE( summary.total.matched == count );
    REQUIRE( summary.total.invalid == 0 );
    REQUIRE( summary.total.duplicates == 0 );
    REQUIRE( summary.total.sequenceGaps == 0 );
    REQUIRE( summary.total.counterGaps == 0 );
    REQUIRE( summary.total.droppedPrimary == 0 );
    REQUIRE( summary.total.droppedCombined == 0 );
    REQUIRE( summary.total.droppedCompletion == 0 );
    REQUIRE( summary.total.evicted == 0 );
    REQUIRE( summary.queued == 0 );
    REQUIRE( summary.pendingPrimary == count );
    REQUIRE( summary.pendingCombined == 0 );
    REQUIRE( summary.pendingCompletion == 0 );
    REQUIRE( summary.metrics[0].count == count );
    REQUIRE( summary.span == Approx( ( count - 1 ) * 500e-6 ) );
}

} // namespace dmReaderTimingTest
} // namespace libXWCTest
