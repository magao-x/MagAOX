/** \file autoJayAcquisition_test.cpp
 * \brief Offline sequence, timing, and frame-identity tests for autoJay.
 * \ingroup autoJay_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "../autoJayAcquisition.hpp"

namespace libXWCTest
{
namespace autoJayTest
{
using namespace MagAOX::app::autoJayDetail;

/// Verify separate DM commands, removal settling, and unbiased distinct-frame statistics.
/** \ingroup autoJay_unit_test */
TEST_CASE( "autoJay sequences distinct baseline and DM stacks", "[autoJay][acquisition]" )
{
    calibration cal;
    cal.frames       = 2;
    cal.settleFrames = 1;
    cal.dmSleep      = 0.1;
    acquisition seq;
    REQUIRE( seq.start( cal, 0 ) );
    double                                 now   = 0;
    uint64_t                               frame = 0;
    std::array<uint64_t, 2>                counters{ 10, 20 };
    const std::array<acquisition::step, 5> expected{ acquisition::step::baseline,
                                                     acquisition::step::ncpc,
                                                     acquisition::step::removeNcpc,
                                                     acquisition::step::tweeter,
                                                     acquisition::step::cleanup };
    for( auto phase : expected )
    {
        REQUIRE( seq.state() == phase );
        REQUIRE( seq.commandPending() );
        auto command = seq.command();
        REQUIRE( command[0] == ( phase == acquisition::step::ncpc ? 1 : 0 ) );
        REQUIRE( command[1] == ( phase == acquisition::step::tweeter ? 1 : 0 ) );
        seq.commanded( counters, now );
        seq.poll( { counters[0] + 1, counters[1] }, now );
        seq.frame( imageT::Constant( 3, 4, 99 ), ++frame, now );
        REQUIRE( seq.state() == phase );
        ++counters[0];
        ++counters[1];
        seq.poll( counters, now );
        seq.frame( imageT::Constant( 3, 4, 99 ), ++frame, now + 0.05 );
        now += 0.2;
        seq.frame( imageT::Constant( 3, 4, 99 ), ++frame, now ); // settling frame
        seq.frame( imageT::Constant( 3, 4, 99 ), frame, now );   // duplicate does not settle or accumulate
        REQUIRE( seq.state() == phase );
        seq.frame( imageT::Constant( 3, 4, 2 ), ++frame, now );
        if( phase == acquisition::step::removeNcpc || phase == acquisition::step::cleanup )
            continue;
        seq.frame( imageT::Constant( 3, 4, 100 ), frame, now ); // duplicate must not affect the statistics
        REQUIRE( seq.state() == phase );
        seq.frame( imageT::Constant( 3, 4, 4 ), ++frame, now );
    }
    REQUIRE( seq.state() == acquisition::step::complete );
    for( const auto &stack : seq.stacks() )
    {
        REQUIRE( stack.count == 2 );
        REQUIRE( stack.lastCounter > stack.firstCounter );
        REQUIRE( ( stack.mean == 3 ).all() );
        REQUIRE( ( stack.variance == 1 ).all() ); // sample variance=2, variance of mean=1
    }
}

/// Verify bounded command waits and failure preservation at every phase.
/** \ingroup autoJay_unit_test */
TEST_CASE( "autoJay aborts timeouts resets and invalid source frames", "[autoJay][acquisition]" )
{
    calibration cal;
    cal.frames       = 2;
    cal.settleFrames = 0;
    cal.dmSleep      = 0;
    cal.phaseTimeout = 1;
    acquisition seq;
    REQUIRE( seq.start( cal, 0 ) );
    SECTION( "command deadline is bounded" )
    {
        seq.poll( { 0, 0 }, 2 );
        REQUIRE( seq.state() == acquisition::step::failed );
    }
    SECTION( "one missing DM trigger cannot proceed" )
    {
        seq.commanded( { 5, 5 }, 0 );
        seq.poll( { 6, 5 }, 0.2 );
        seq.frame( imageT::Ones( 3, 4 ), 1, 0.3 );
        REQUIRE( seq.stacks()[0].count == 0 );
        seq.poll( { 6, 5 }, 2 );
        REQUIRE( seq.state() == acquisition::step::failed );
    }
    SECTION( "combined counter reset" )
    {
        seq.commanded( { 5, 5 }, 0 );
        seq.poll( { 1, 6 }, 0.2 );
        REQUIRE( seq.error() == "combined DM counter reset" );
    }
    SECTION( "combined reset after triggering is still detected" )
    {
        seq.commanded( { 5, 5 }, 0 );
        seq.poll( { 100, 100 }, 0.2 );
        seq.frame( imageT::Ones( 3, 4 ), 1, 0.3 );
        seq.poll( { 10, 101 }, 0.4 );
        REQUIRE( seq.error() == "combined DM counter reset" );
    }
    SECTION( "unrelated combined update cannot acknowledge the poke" )
    {
        seq.commanded( { 5, 5 }, 0 );
        seq.poll( { 6, 6 }, 0.2, false );
        seq.frame( imageT::Ones( 3, 4 ), 1, 0.3 );
        REQUIRE( seq.stacks()[0].count == 0 );
        REQUIRE( seq.state() == acquisition::step::baseline );
        seq.poll( { 7, 7 }, 0.4, true );
        seq.frame( imageT::Ones( 3, 4 ), 2, 0.5 );
        REQUIRE( seq.stacks()[0].count == 1 );
    }
    SECTION( "delayed delivery of a pre-command frame is excluded with zero settling" )
    {
        seq.commanded( { 5, 5 }, 0.2 );
        seq.poll( { 6, 6 }, 0.3 );
        seq.frame( imageT::Ones( 3, 4 ), 1, 0.1 );
        REQUIRE( seq.stacks()[0].count == 0 );
        seq.frame( imageT::Ones( 3, 4 ), 2, 0.4 );
        REQUIRE( seq.stacks()[0].count == 1 );
    }
    SECTION( "a clipped source cannot be hidden by averaging" )
    {
        REQUIRE( seq.start( cal, 0, 10 ) );
        seq.commanded( { 5, 5 }, 0 );
        seq.poll( { 6, 6 }, 0.2 );
        seq.frame( imageT::Ones( 3, 4 ), 1, 0.3 );
        imageT clipped  = imageT::Ones( 3, 4 );
        clipped( 1, 1 ) = 10;
        seq.frame( clipped, 2, 0.4 );
        REQUIRE( seq.error() == "saturated camera frame" );
    }
    SECTION( "camera timeout after the trigger" )
    {
        seq.commanded( { 5, 5 }, 0 );
        seq.poll( { 6, 6 }, 0.2 );
        seq.frame( imageT::Ones( 3, 4 ), 1, 2 );
        REQUIRE( seq.state() == acquisition::step::failed );
    }
    SECTION( "camera reset" )
    {
        seq.commanded( { 5, 5 }, 0 );
        seq.poll( { 6, 6 }, 0.2 );
        seq.frame( imageT::Ones( 3, 4 ), 10, 0.3 );
        seq.frame( imageT::Ones( 3, 4 ), 1, 0.4 );
        REQUIRE( seq.error() == "camera counter reset" );
    }
    SECTION( "geometry changes" )
    {
        seq.commanded( { 5, 5 }, 0 );
        seq.poll( { 6, 6 }, 0.2 );
        seq.frame( imageT::Ones( 3, 4 ), 1, 0.3 );
        seq.frame( imageT::Ones( 4, 4 ), 2, 0.4 );
        REQUIRE( seq.error() == "camera dimensions changed during stack" );
    }
    SECTION( "nonfinite source" )
    {
        seq.commanded( { 5, 5 }, 0 );
        seq.poll( { 6, 6 }, 0.2 );
        imageT image  = imageT::Ones( 3, 4 );
        image( 0, 0 ) = std::numeric_limits<float>::infinity();
        seq.frame( image, 1, 0.3 );
        REQUIRE( seq.error() == "nonfinite camera image" );
    }
    SECTION( "cancellation keeps its reason and ignores queued frames" )
    {
        seq.fail( "cancelled" );
        seq.fail( "secondary failure" );
        seq.frame( imageT::Ones( 3, 4 ), 1, 0.2 );
        REQUIRE( seq.error() == "cancelled" );
        REQUIRE_FALSE( seq.commandPending() );
    }
}

/// Verify invalid timing cannot start a transaction and fresh starts clear prior statistics.
/** \ingroup autoJay_unit_test */
TEST_CASE( "autoJay validates acquisition settings", "[autoJay][acquisition]" )
{
    calibration cal;
    acquisition seq;
    SECTION( "one frame cannot estimate variance" )
    {
        cal.frames = 1;
    }
    SECTION( "excessive frame request" )
    {
        cal.frames = 10001;
    }
    SECTION( "negative delay" )
    {
        cal.dmSleep = -1;
    }
    SECTION( "settling exceeds deadline" )
    {
        cal.phaseTimeout = cal.dmSleep;
    }
    SECTION( "nonfinite deadline" )
    {
        cal.phaseTimeout = std::numeric_limits<double>::quiet_NaN();
    }
    REQUIRE_FALSE( seq.start( cal, 0 ) );
    REQUIRE( seq.state() == acquisition::step::failed );
    REQUIRE( seq.start( calibration{}, 0 ) );
    REQUIRE( seq.state() == acquisition::step::baseline );
    REQUIRE( seq.error().empty() );
    REQUIRE( seq.stacks()[0].count == 0 );
}

} // namespace autoJayTest
} // namespace libXWCTest
