/** \file loopTimingGate_test.cpp
 * \brief Offline retained-subscription eligibility and transition tests for loop timing.
 * \ingroup loopTimingMonitor_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "../loopTimingGate.hpp"

namespace libXWCTest
{
namespace loopTimingMonitorTest
{

using Gate = MagAOX::app::loopTiming::Gate;

/** \brief Require an affirmative observation from every prerequisite before enabling collection.
 * \ingroup loopTimingMonitor_unit_test
 */
TEST_CASE( "Loop timing gate requires three affirmative sources", "[loopTimingMonitor][gate]" )
{
    Gate gate;
    REQUIRE_FALSE( gate.enabled() );
    gate.update( Gate::Source::camera, true );
    REQUIRE_FALSE( gate.enabled() );
    gate.update( Gate::Source::dm, true );
    REQUIRE_FALSE( gate.enabled() );
    gate.update( Gate::Source::loop, false );
    REQUIRE_FALSE( gate.enabled() );
    gate.update( Gate::Source::loop, true );
    REQUIRE( gate.enabled() );

    // clang-format off
#ifdef LOOPTIMINGMONITOR_TEST_DOXYGEN_REF
    MagAOX::app::loopTiming::Gate::update;
    MagAOX::app::loopTiming::Gate::enabled;
#endif
    // clang-format on
}

/** \brief Retain unchanged subscription states without refreshes or generation changes on read.
 * \ingroup loopTimingMonitor_unit_test
 */
TEST_CASE( "Loop timing gate retains subscription states until changed", "[loopTimingMonitor][gate]" )
{
    Gate gate;

    for( auto source : { Gate::Source::camera, Gate::Source::dm, Gate::Source::loop } )
    {
        gate.update( source, true );
    }

    const Gate &retained = gate;
    const auto  ready    = gate.generation();
    REQUIRE( retained.enabled() );
    REQUIRE( retained.generation() == ready );

    for( auto source : { Gate::Source::camera, Gate::Source::dm, Gate::Source::loop } )
    {
        gate.update( source, true );
        REQUIRE( retained.enabled() );
        REQUIRE( retained.generation() == ready );
    }

    // clang-format off
#ifdef LOOPTIMINGMONITOR_TEST_DOXYGEN_REF
    MagAOX::app::loopTiming::Gate::update;
    MagAOX::app::loopTiming::Gate::enabled;
    MagAOX::app::loopTiming::Gate::generation;
#endif
    // clang-format on
}

/** \brief Preserve an off/on transition even when the consumer does not inspect the intermediate state.
 * \ingroup loopTimingMonitor_unit_test
 */
TEST_CASE( "Loop timing gate preserves intervening source transitions", "[loopTimingMonitor][gate]" )
{
    Gate gate;

    for( auto source : { Gate::Source::camera, Gate::Source::dm, Gate::Source::loop } )
    {
        gate.update( source, true );
    }

    for( auto source : { Gate::Source::camera, Gate::Source::dm, Gate::Source::loop } )
    {
        const auto before = gate.generation();
        gate.update( source, false );
        REQUIRE_FALSE( gate.enabled() );
        REQUIRE( gate.generation() == before + 1 );
        gate.update( source, false );
        REQUIRE( gate.generation() == before + 1 );
        gate.update( source, true );
        REQUIRE( gate.enabled() );
        REQUIRE( gate.generation() == before + 2 );

        gate.update( source, false );
        gate.update( source, true );
        REQUIRE( gate.enabled() );
        REQUIRE( gate.generation() == before + 4 );
    }

    // clang-format off
#ifdef LOOPTIMINGMONITOR_TEST_DOXYGEN_REF
    MagAOX::app::loopTiming::Gate::update;
    MagAOX::app::loopTiming::Gate::enabled;
    MagAOX::app::loopTiming::Gate::generation;
#endif
    // clang-format on
}

/** \brief Explicit invalidation requires a new update from the invalidated source, without erasing other states.
 * \ingroup loopTimingMonitor_unit_test
 */
TEST_CASE( "Loop timing gate explicitly revokes an unusable subscription", "[loopTimingMonitor][gate]" )
{
    Gate gate;

    for( auto source : { Gate::Source::camera, Gate::Source::dm, Gate::Source::loop } )
    {
        gate.update( source, true );
    }

    for( auto source : { Gate::Source::camera, Gate::Source::dm, Gate::Source::loop } )
    {
        const auto before = gate.generation();
        gate.invalidate( source );
        REQUIRE_FALSE( gate.enabled() );
        REQUIRE( gate.generation() == before + 1 );
        gate.update( source, true );
        REQUIRE( gate.enabled() );
        REQUIRE( gate.generation() == before + 2 );
    }

    // clang-format off
#ifdef LOOPTIMINGMONITOR_TEST_DOXYGEN_REF
    MagAOX::app::loopTiming::Gate::update;
    MagAOX::app::loopTiming::Gate::invalidate;
    MagAOX::app::loopTiming::Gate::enabled;
    MagAOX::app::loopTiming::Gate::generation;
#endif
    // clang-format on
}

/** \brief An explicit reset forgets every prerequisite and requires all three subscriptions again.
 * \ingroup loopTimingMonitor_unit_test
 */
TEST_CASE( "Loop timing gate starts fresh after reset", "[loopTimingMonitor][gate]" )
{
    Gate gate;

    for( auto source : { Gate::Source::camera, Gate::Source::dm, Gate::Source::loop } )
    {
        gate.update( source, true );
    }

    const auto before = gate.generation();
    gate.reset();
    REQUIRE( gate.generation() == before + 1 );
    REQUIRE_FALSE( gate.enabled() );
    gate.update( Gate::Source::camera, true );
    REQUIRE_FALSE( gate.enabled() );
    gate.update( Gate::Source::dm, true );
    REQUIRE_FALSE( gate.enabled() );
    gate.update( Gate::Source::loop, true );
    REQUIRE( gate.enabled() );

    // clang-format off
#ifdef LOOPTIMINGMONITOR_TEST_DOXYGEN_REF
    MagAOX::app::loopTiming::Gate::reset;
    MagAOX::app::loopTiming::Gate::update;
    MagAOX::app::loopTiming::Gate::enabled;
    MagAOX::app::loopTiming::Gate::generation;
#endif
    // clang-format on
}

} // namespace loopTimingMonitorTest
} // namespace libXWCTest
