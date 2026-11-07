/** \file loopTimingGate_test.cpp
 * \brief Offline remote-state eligibility and transition tests for loop timing.
 * \ingroup loopTimingMonitor_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "../loopTimingGate.hpp"

namespace libXWCTest
{
namespace loopTimingMonitorTest
{

using Gate = MagAOX::app::loopTiming::Gate;

/** \brief Require each configured prerequisite and preserve intervening off/on transitions.
 * \ingroup loopTimingMonitor_unit_test
 */
TEST_CASE( "Loop timing gate requires three affirmative sources", "[loopTimingMonitor][gate]" )
{
    // clang-format off
    #ifdef LOOPTIMINGMONITOR_TEST_DOXYGEN_REF
    MagAOX::app::loopTiming::Gate::update( {}, true, 1 );
    MagAOX::app::loopTiming::Gate::enabled( 1, 1 );
    MagAOX::app::loopTiming::Gate::generation();
    #endif
    // clang-format on
    Gate gate;
    REQUIRE_FALSE( gate.enabled( 100, 1000 ) );
    gate.update( Gate::Source::camera, true, 100 );
    gate.update( Gate::Source::dm, true, 100 );
    REQUIRE_FALSE( gate.enabled( 100, 1000 ) );
    gate.update( Gate::Source::loop, true, 100 );
    REQUIRE( gate.enabled( 100, 1000 ) );

    const auto ready = gate.generation();
    gate.update( Gate::Source::camera, true, 101 );
    gate.update( Gate::Source::dm, true, 101 );
    gate.update( Gate::Source::loop, true, 101 );
    REQUIRE( gate.generation() == ready );

    int64_t now = 101;

    for( auto source : { Gate::Source::camera, Gate::Source::dm, Gate::Source::loop } )
    {
        const auto before = gate.generation();
        gate.update( source, false, ++now );
        REQUIRE_FALSE( gate.enabled( now, 1000 ) );
        gate.update( source, true, ++now );
        REQUIRE( gate.enabled( now, 1000 ) );
        REQUIRE( gate.generation() == before + 2 );
    }
}

/** \brief Expire stale input and require a fresh observation after invalid state information.
 * \ingroup loopTimingMonitor_unit_test
 */
TEST_CASE( "Loop timing gate revokes stale and malformed state", "[loopTimingMonitor][gate]" )
{
    // clang-format off
    #ifdef LOOPTIMINGMONITOR_TEST_DOXYGEN_REF
    MagAOX::app::loopTiming::Gate::update( {}, true, 1 );
    MagAOX::app::loopTiming::Gate::invalidate( {} );
    MagAOX::app::loopTiming::Gate::enabled( 1, 1 );
    #endif
    // clang-format on
    Gate gate;
    gate.update( Gate::Source::camera, true, 100 );
    gate.update( Gate::Source::dm, true, 150 );
    gate.update( Gate::Source::loop, true, 150 );
    REQUIRE( gate.enabled( 199, 100 ) );

    const auto before = gate.generation();
    REQUIRE_FALSE( gate.enabled( 200, 100 ) );
    REQUIRE( gate.generation() == before + 1 );
    REQUIRE_FALSE( gate.enabled( 201, 100 ) );
    REQUIRE( gate.generation() == before + 1 );

    gate.update( Gate::Source::camera, true, 201 );
    REQUIRE( gate.enabled( 201, 100 ) );
    gate.invalidate( Gate::Source::loop );
    REQUIRE_FALSE( gate.enabled( 201, 100 ) );
    gate.update( Gate::Source::loop, true, 202 );
    REQUIRE( gate.enabled( 202, 100 ) );

    gate.update( Gate::Source::dm, true, 0 );
    REQUIRE_FALSE( gate.enabled( 202, 100 ) );
    gate.update( Gate::Source::dm, true, 203 );
    gate.update( Gate::Source::dm, true, 202 );
    REQUIRE_FALSE( gate.enabled( 203, 100 ) );
}

/** \brief Reset eligibility on clock regressions, invalid expiry settings and explicit reset.
 * \ingroup loopTimingMonitor_unit_test
 */
TEST_CASE( "Loop timing gate starts fresh after reset", "[loopTimingMonitor][gate]" )
{
    // clang-format off
    #ifdef LOOPTIMINGMONITOR_TEST_DOXYGEN_REF
    MagAOX::app::loopTiming::Gate::reset();
    MagAOX::app::loopTiming::Gate::enabled( 1, 1 );
    #endif
    // clang-format on
    Gate gate;

    for( auto source : { Gate::Source::camera, Gate::Source::dm, Gate::Source::loop } )
    {
        gate.update( source, true, 100 );
    }

    REQUIRE_FALSE( gate.enabled( 99, 100 ) );

    for( auto source : { Gate::Source::camera, Gate::Source::dm, Gate::Source::loop } )
    {
        gate.update( source, true, 200 );
    }

    REQUIRE_FALSE( gate.enabled( 200, 0 ) );

    for( auto source : { Gate::Source::camera, Gate::Source::dm, Gate::Source::loop } )
    {
        gate.update( source, true, 300 );
    }

    const auto before = gate.generation();
    gate.reset();
    REQUIRE( gate.generation() == before + 1 );
    REQUIRE_FALSE( gate.enabled( 300, 100 ) );
}

} // namespace loopTimingMonitorTest
} // namespace libXWCTest
