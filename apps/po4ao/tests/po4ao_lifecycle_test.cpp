/** \file po4ao_lifecycle_test.cpp
 * \brief Route failures and exact callback telemetry without instrument access.
 * \ingroup po4ao_unit_test
 */
#include "../../../tests/catch2/catch.hpp"
#include "../po4aoRouting.hpp"
#include "../po4aoStatistics.hpp"

namespace libXWCTest::po4aoTest
{
using namespace MagAOX::app;
/** \brief Each command/readback failure retains routing identity and prevents premature detachment. */
TEST_CASE( "PO4AO routing preserves the request identity and never parks on failure", "[po4ao][routing]" )
{
    for( bool start : { false, true } )
        for( int failure = -1; failure < ( start ? 7 : 6 ); ++failure )
        {
            Po4aoRouteBackend        b;
            Po4aoRouteSnapshot       retained;
            const Po4aoRouteSnapshot before{ "po4ao_output", "previous", "nn" };
            int                      step   = 0;
            bool                     parked = false;
            std::vector<std::string> phases;
            auto                     boundary = [&]
            {
                if( step++ == failure )
                    throw std::runtime_error( "injected failure" );
            };
            b.capture = [&]( auto )
            {
                boundary();
                return before;
            };
            b.reserveWriter = [&]( const auto &s, auto )
            {
                REQUIRE( s.writer == "nn" );
                boundary();
            };
            b.freshOutput = [&]( auto ) { boundary(); };
            b.switchInput = [&]( const auto &target, auto )
            {
                REQUIRE( target == ( start ? "po4ao_output" : "aol1_modevalWFS" ) );
                boundary();
            };
            b.applyDefault  = [&]( auto ) { boundary(); };
            b.restoreWriter = [&]( const auto &s, auto )
            {
                REQUIRE( s.writer == before.writer );
                boundary();
            };
            b.verifyConsumption = [&]( const auto &, auto ) { boundary(); };
            b.park              = [&]( auto ) { parked = true; };
            auto run            = [&]
            {
                routePo4ao(
                    start,
                    b,
                    retained,
                    [&]( const auto &p ) { phases.push_back( p ); },
                    std::chrono::steady_clock::now() );
            };
            if( failure < 0 )
            {
                REQUIRE_NOTHROW( run() );
                REQUIRE( parked == !start );
                REQUIRE( phases.back() == ( start ? "running" : "stopped" ) );
            }
            else
            {
                REQUIRE_THROWS( run() );
                REQUIRE_FALSE( parked );
            }
            if( failure != 0 )
                REQUIRE( retained.route == before.route );
        }
}

/** \brief Empty, sub-microsecond, idle and mixed-model windows remain distinguishable. */
TEST_CASE( "PO4AO telemetry describes the actual rolling window", "[po4ao][telemetry]" )
{
    Po4aoSamples samples;
    auto         s = samples.snapshot( 0 );
    REQUIRE( s.count == 0 );
    REQUIRE_FALSE( s.fresh );
    samples.record( 0, 0, 0, 100 );
    s = samples.snapshot( 0 );
    REQUIRE( s.count == 1 );
    REQUIRE( s.p50 == 0 );
    REQUIRE( s.fresh );
    REQUIRE_FALSE( s.mixed );
    REQUIRE_FALSE( samples.snapshot( s.sequence ).fresh );
    constexpr uint64_t model = UINT64_MAX - 10;
    for( unsigned i = 1; i <= 1024; ++i )
        samples.record( i, model, 2, 100 + i );
    s = samples.snapshot( 1 );
    REQUIRE( s.count == 1024 );
    REQUIRE( s.sequence == 1025 );
    REQUIRE( s.model == model );
    REQUIRE( s.backend == 2 );
    REQUIRE( s.p50 == 512 );
    REQUIRE( s.p95 == 973 );
    REQUIRE( s.p99 == 1014 );
    REQUIRE( s.maximum == 1024 );
    REQUIRE_FALSE( s.mixed );
    samples.record( 0, 0, 0, 2000 );
    s = samples.snapshot( 1025 );
    REQUIRE( s.fresh );
    REQUIRE( s.mixed );
    REQUIRE( s.lastTimeNs == 2000 );
    REQUIRE( s.model == 0 );
    REQUIRE( s.backend == 0 );
}
} // namespace libXWCTest::po4aoTest
