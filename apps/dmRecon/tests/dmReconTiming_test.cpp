/** \file dmReconTiming_test.cpp
 * \brief Tests of dmRecon's nonblocking loop-trigger timing diagnostics.
 * \ingroup dmRecon_unit_test
 */

#include "../../../tests/testXWC.hpp"

#include "../dmReconTiming.hpp"

namespace libXWCTest
{
namespace dmReconTest
{

using MagAOX::app::dmReconTiming;

/// \cond DMRECON_TIMING_TEST_HELPERS

/// Make trigger metadata with a fixed attachment and a nanosecond offset.
static dmReconTiming::Trigger trigger( uint64_t counter, /**< [in] Source publication counter. */
                                       long     offset   /**< [in] Offset within the test's fixed second. */
)
{
    return { 1, 101, counter, { 100, offset } };
}

/// \endcond

/// A delayed callback is classified using source time, including an exact deadline miss.
/** \ingroup dmRecon_unit_test */
TEST_CASE( "dmRecon timing uses source timestamps", "[dmReconTiming]" )
{
    dmReconTiming timing;
    auto          update = timing.published( trigger( 10, 100 ), { 100, 200 } );
    REQUIRE( update.beforeNextTrigger == 0 );
    REQUIRE( update.late == 0 );
    REQUIRE( update.unknown == 0 );

    SECTION( "Publication precedes the next trigger" )
    {
        update = timing.observe( trigger( 11, 250 ) );
        REQUIRE( update.beforeNextTrigger == 1 );
        REQUIRE( update.late == 0 );
        REQUIRE( update.marginNs == 50 );
    }

    SECTION( "The observer runs late but its source triggered before completion" )
    {
        update = timing.observe( trigger( 11, 150 ) );
        REQUIRE( update.beforeNextTrigger == 0 );
        REQUIRE( update.late == 1 );
        REQUIRE( update.marginNs == -50 );
    }

    SECTION( "Equality is a miss" )
    {
        update = timing.observe( trigger( 11, 200 ) );
        REQUIRE( update.late == 1 );
        REQUIRE( update.marginNs == 0 );
    }

    REQUIRE( update.unknown == 0 );
    REQUIRE( timing.reset().unknown == 0 );
}

/// Trigger callbacks may precede the publication callback or arrive out of order.
/** \ingroup dmRecon_unit_test */
TEST_CASE( "dmRecon timing handles callback reordering", "[dmReconTiming]" )
{
    dmReconTiming timing;
    timing.observe( trigger( 12, 300 ) );
    timing.observe( trigger( 10, 100 ) );
    timing.observe( trigger( 11, 250 ) );

    auto update = timing.published( trigger( 10, 100 ), { 100, 200 } );
    REQUIRE( update.beforeNextTrigger == 1 );
    REQUIRE( update.marginNs == 50 );
    REQUIRE( update.unknown == 0 );

    update = timing.observe( trigger( 11, 250 ) );
    REQUIRE( update.beforeNextTrigger == 0 );
    REQUIRE( update.late == 0 );
    REQUIRE( update.unknown == 0 );
}

/// Missing exact-next observations never produce a successful timing classification.
/** \ingroup dmRecon_unit_test */
TEST_CASE( "dmRecon timing leaves gaps unknown", "[dmReconTiming]" )
{
    dmReconTiming timing;
    timing.published( trigger( 10, 100 ), { 100, 200 } );
    auto update = timing.observe( trigger( 12, 300 ) );
    REQUIRE( update.beforeNextTrigger == 0 );
    REQUIRE( update.late == 0 );

    SECTION( "The missing callback arrives later" )
    {
        update = timing.observe( trigger( 11, 150 ) );
        REQUIRE( update.late == 1 );
        REQUIRE( update.unknown == 0 );
    }

    SECTION( "A new publication retires the unresolved candidate" )
    {
        update = timing.published( trigger( 12, 300 ), { 100, 400 } );
        REQUIRE( update.unknown == 1 );
        REQUIRE( update.beforeNextTrigger == 0 );
        REQUIRE( timing.reset().unknown == 1 );
    }
}

/// Reconnects retire old publications; delayed observations from old generations are ignored.
/** \ingroup dmRecon_unit_test */
TEST_CASE( "dmRecon timing separates attachment generations", "[dmReconTiming]" )
{
    dmReconTiming timing;
    timing.published( trigger( 10, 100 ), { 100, 200 } );
    auto next       = trigger( 1, 300 );
    next.generation = 2;
    next.inode      = 102;
    REQUIRE( timing.observe( next ).unknown == 1 );
    REQUIRE( timing.published( next, { 100, 350 } ).unknown == 0 );

    auto update = timing.observe( trigger( 11, 250 ) );
    REQUIRE( update.beforeNextTrigger == 0 );
    REQUIRE( update.late == 0 );
    REQUIRE( update.unknown == 0 );

    next.counter           = 2;
    next.writeTime.tv_nsec = 400;
    REQUIRE( timing.observe( next ).beforeNextTrigger == 1 );
    REQUIRE( timing.published( trigger( 10, 100 ), { 100, 200 } ).unknown == 1 );
}

/// Counter resets, contradictory duplicates, and clock reversals invalidate the attachment.
/** \ingroup dmRecon_unit_test */
TEST_CASE( "dmRecon timing rejects inconsistent metadata", "[dmReconTiming]" )
{
    dmReconTiming timing;
    timing.published( trigger( 10, 100 ), { 100, 200 } );
    auto invalid = trigger( 11, 250 );

    SECTION( "Counter reset with later source time" )
    {
        invalid.counter = 1;
    }

    SECTION( "Source clock goes backwards" )
    {
        invalid.writeTime.tv_nsec = 50;
    }

    SECTION( "A duplicate counter has a different source time" )
    {
        invalid.counter = 10;
    }

    SECTION( "A changed inode lacks a new attachment generation" )
    {
        invalid.inode = 102;
    }

    auto update = timing.observe( invalid );
    REQUIRE( update.unknown == 1 );
    REQUIRE( update.beforeNextTrigger == 0 );
    REQUIRE( update.late == 0 );
    REQUIRE( timing.published( trigger( 11, 250 ), { 100, 275 } ).unknown == 1 );

    timing.reset();
    auto baseline       = trigger( 11, 250 );
    baseline.generation = 2;
    timing.published( baseline, { 100, 275 } );
    auto next       = trigger( 12, 300 );
    next.generation = 2;
    REQUIRE( timing.observe( next ).beforeNextTrigger == 1 );
}

/// Missing identities, malformed timestamps, and counter wrap never imply success.
/** \ingroup dmRecon_unit_test */
TEST_CASE( "dmRecon timing requires complete publication metadata", "[dmReconTiming]" )
{
    dmReconTiming timing;
    auto          baseline = trigger( 10, 100 );
    timespec      end{ 100, 200 };

    SECTION( "Missing generation" )
    {
        baseline.generation = 0;
    }

    SECTION( "Missing stream identity" )
    {
        baseline.inode = 0;
    }

    SECTION( "Missing baseline time" )
    {
        baseline.writeTime = {};
    }

    SECTION( "Missing completion time" )
    {
        end = {};
    }

    SECTION( "Unnormalized timestamp" )
    {
        end.tv_nsec = 1000000000;
    }

    SECTION( "Negative timestamp" )
    {
        end.tv_sec = -1;
    }

    SECTION( "Timestamp overflows integer nanoseconds" )
    {
        end.tv_sec = std::numeric_limits<time_t>::max();
    }

    SECTION( "Completion precedes its baseline" )
    {
        end.tv_nsec = 50;
    }

    SECTION( "Counter wrap" )
    {
        baseline.counter = std::numeric_limits<uint64_t>::max();
    }

    auto update = timing.published( baseline, end );
    REQUIRE( update.unknown == 1 );
    REQUIRE( update.beforeNextTrigger == 0 );
    REQUIRE( update.late == 0 );
    REQUIRE( timing.observe( trigger( 11, 250 ) ).beforeNextTrigger == 0 );
}

/// A trigger without a source timestamp cannot resolve a pending publication favorably.
/** \ingroup dmRecon_unit_test */
TEST_CASE( "dmRecon timing does not substitute receipt time", "[dmReconTiming]" )
{
    dmReconTiming timing;
    timing.published( trigger( 10, 100 ), { 100, 200 } );
    auto next      = trigger( 11, 250 );
    next.writeTime = {};
    auto update    = timing.observe( next );
    REQUIRE( update.unknown == 1 );
    REQUIRE( update.beforeNextTrigger == 0 );
    REQUIRE( update.late == 0 );
    REQUIRE( timing.reset().unknown == 0 );
}

/// Retiring one unknown publication does not hide an immediately classifiable replacement.
/** \ingroup dmRecon_unit_test */
TEST_CASE( "dmRecon timing reports retirement and classification together", "[dmReconTiming]" )
{
    dmReconTiming timing;
    timing.observe( trigger( 13, 400 ) );
    timing.published( trigger( 10, 100 ), { 100, 200 } );
    auto update = timing.published( trigger( 12, 300 ), { 100, 350 } );
    REQUIRE( update.unknown == 1 );
    REQUIRE( update.beforeNextTrigger == 1 );
    REQUIRE( update.marginNs == 50 );
}

/// The fixed history limit sacrifices certainty rather than guessing the missing next trigger.
/** \ingroup dmRecon_unit_test */
TEST_CASE( "dmRecon timing bounds observation history", "[dmReconTiming]" )
{
    dmReconTiming timing;

    for( uint64_t n = 10; n < 20; ++n )
    {
        timing.observe( trigger( n, static_cast<long>( n * 100 ) ) );
    }

    auto update = timing.published( trigger( 10, 1000 ), { 100, 1050 } );
    REQUIRE( update.beforeNextTrigger == 0 );
    REQUIRE( update.late == 0 );
    REQUIRE( timing.reset().unknown == 1 );
}

/// An observer lagging the source command cannot manufacture a deadline-miss warning.
/** \ingroup dmRecon_unit_test */
TEST_CASE( "dmRecon timing requires a bracket around the source publication", "[dmReconTiming]" )
{
    dmReconTiming timing;
    timing.published( trigger( 10, 100 ), { 100, 300 }, { 100, 200 } );

    SECTION( "Exact next trigger brackets the source and follows feedback" )
    {
        auto update = timing.observe( trigger( 11, 350 ) );
        REQUIRE( update.beforeNextTrigger == 1 );
        REQUIRE( update.unknown == 0 );
        REQUIRE( update.late == 0 );
    }

    SECTION( "Exact next trigger brackets the source but precedes feedback" )
    {
        auto update = timing.observe( trigger( 11, 250 ) );
        REQUIRE( update.late == 1 );
        REQUIRE( update.unknown == 0 );
        REQUIRE( update.beforeNextTrigger == 0 );
    }

    SECTION( "An old baseline's successor precedes the command" )
    {
        auto update = timing.observe( trigger( 11, 150 ) );
        REQUIRE( update.unknown == 1 );
        REQUIRE( update.late == 0 );
        REQUIRE( update.beforeNextTrigger == 0 );
    }

    SECTION( "Source publication exactly at the next trigger is unverified" )
    {
        auto update = timing.observe( trigger( 11, 200 ) );
        REQUIRE( update.unknown == 1 );
        REQUIRE( update.late == 0 );
        REQUIRE( update.beforeNextTrigger == 0 );
    }
}

/// Invalid or differently ordered source clocks cannot produce a verified timing classification.
/** \ingroup dmRecon_unit_test */
TEST_CASE( "dmRecon timing validates source publication time", "[dmReconTiming]" )
{
    dmReconTiming timing;
    timespec      source{ 100, 200 };

    SECTION( "Missing source publication time" )
    {
        source = {};
    }

    SECTION( "Source precedes the selected baseline" )
    {
        source.tv_nsec = 50;
    }

    SECTION( "Source follows completed feedback" )
    {
        source.tv_nsec = 350;
    }

    auto update = timing.published( trigger( 10, 100 ), { 100, 300 }, source );
    REQUIRE( update.unknown == 1 );
    REQUIRE( update.beforeNextTrigger == 0 );
    REQUIRE( update.late == 0 );
    REQUIRE( timing.observe( trigger( 11, 400 ) ).beforeNextTrigger == 0 );
}

} // namespace dmReconTest
} // namespace libXWCTest
