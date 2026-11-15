/** \file loopTiming_test.cpp
 * \brief Offline association, cadence, statistics and shared-clock regression tests.
 * \ingroup loopTimingMonitor_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "../loopTiming.hpp"
#include "../../dmSyncopation/syncoPublicationClock.hpp"

#include <type_traits>

namespace libXWCTest
{
namespace loopTimingMonitorTest
{
using MagAOX::app::loopTiming::Monitor;

/// \cond LOOP_TIMING_FIXTURE
constexpr int64_t base = 1000000000; ///< Positive MONOTONIC origin for all synthetic sequences.

/// Supply one exact camera timestamp with optional symmetric mapping uncertainty.
void camera( Monitor &monitor /**< [in,out] Pure timing consumer. */,
             uint64_t counter /**< [in] Observed camera counter. */,
             int64_t  offset /**< [in] Timestamp offset from base, ns. */,
             int64_t  uncertainty = 0 /**< [in] Symmetric mapping uncertainty, ns. */ )
{
    monitor.camera( counter, base + offset, base + offset - uncertainty, base + offset + uncertainty );
}

/// Establish SDK observation coverage before the first camera timestamp.
void baseline( Monitor &monitor /**< [in,out] Configured timing consumer. */ )
{
    monitor.sdk( 0, base - 400000 );
}
/// \endcond

/** \brief Validate bounds before accepting observations or changing an existing configuration. */
TEST_CASE( "loop timing validates configuration", "[loopTimingMonitor][loopTiming]" )
{
    Monitor monitor;
    CHECK_FALSE( monitor.configure( 0, 2, 100 ) );
    CHECK_FALSE( monitor.configure( -1, 2, 100 ) );
    CHECK_FALSE( monitor.configure( std::numeric_limits<double>::infinity(), 2, 100 ) );
    CHECK_FALSE( monitor.configure( 1e10, 2, 100 ) );
    CHECK_FALSE( monitor.configure( 3600, 0, 100 ) );
    CHECK_FALSE( monitor.configure( 3600, std::numeric_limits<double>::quiet_NaN(), 100 ) );
    CHECK_FALSE( monitor.configure( 3600, 2, 2 ) );
    REQUIRE( monitor.configure( 3600, 2, 100 ) );
    CHECK( monitor.summary().quality.resets == 0 );
    CHECK( monitor.summary().epoch == 1 );
    // clang-format off
#ifdef LOOPTIMINGMONITOR_TEST_DOXYGEN_REF
    MagAOX::app::loopTiming::Monitor::configure;
    MagAOX::app::loopTiming::Monitor::summary;
#endif
    // clang-format on
}

/** \brief Recover exact integer differences and sample deviation independently of arrival order. */
TEST_CASE( "loop timing matches delayed SDK visibility and reports sample statistics",
           "[loopTimingMonitor][loopTiming]" )
{
    Monitor monitor;
    REQUIRE( monitor.configure( 3600, 2, 100 ) );
    baseline( monitor );
    const bool camerasFirst = GENERATE( false, true );

    if( camerasFirst )
    {
        camera( monitor, 1, 0 );
        camera( monitor, 2, 500000 );
        camera( monitor, 3, 1000000 );
        monitor.sdk( 1, base + 100000 );
        monitor.sdk( 2, base + 620000 );
        monitor.sdk( 3, base + 1140000 );
    }
    else
    {
        monitor.sdk( 1, base + 100000 );
        camera( monitor, 1, 0 );
        monitor.sdk( 2, base + 620000 );
        camera( monitor, 2, 500000 );
        monitor.sdk( 3, base + 1140000 );
        camera( monitor, 3, 1000000 );
    }

    monitor.update( base + 1200000 );
    const auto &summary = monitor.summary();
    CHECK( summary.quality.accepted == 2 );
    CHECK( summary.latency.count == 2 );
    CHECK( summary.latency.mean == Approx( 110e-6 ) );
    CHECK( summary.latency.stddev == Approx( std::sqrt( 200.0 ) * 1e-6 ) );
    CHECK( summary.latency.valid );
    CHECK( summary.cameraIntervals.count == 2 );
    CHECK( summary.cameraIntervals.mean == Approx( 500e-6 ) );
    CHECK( summary.sdkIntervals.count == 3 );
    CHECK( summary.sdkIntervals.mean == Approx( 1540e-6 / 3 ) );
    CHECK( summary.newestNs == base + 620000 );
    CHECK( summary.age == Approx( 580e-6 ) );
    CHECK( summary.latency.span == Approx( 620e-6 ) );
    // clang-format off
#ifdef LOOPTIMINGMONITOR_TEST_DOXYGEN_REF
    MagAOX::app::loopTiming::Monitor::configure;
    MagAOX::app::loopTiming::Monitor::camera;
    MagAOX::app::loopTiming::Monitor::sdk;
    MagAOX::app::loopTiming::Monitor::update;
    MagAOX::app::loopTiming::Monitor::summary;
#endif
    // clang-format on
}

/** \brief A closing watermark prevents an early candidate from concealing a second SDK event. */
TEST_CASE( "loop timing rejects multiple candidates before closing an interval", "[loopTimingMonitor][loopTiming]" )
{
    Monitor monitor;
    REQUIRE( monitor.configure( 3600, 2, 100 ) );
    baseline( monitor );
    camera( monitor, 1, 0 );
    camera( monitor, 2, 500000 );
    monitor.sdk( 1, base + 100000 );
    CHECK( monitor.summary().quality.accepted == 0 );
    monitor.sdk( 2, base + 200000 );
    CHECK( monitor.summary().quality.ambiguous == 0 );
    monitor.sdk( 3, base + 600000 );
    CHECK( monitor.summary().quality.accepted == 0 );
    CHECK( monitor.summary().quality.ambiguous == 1 );
    // clang-format off
#ifdef LOOPTIMINGMONITOR_TEST_DOXYGEN_REF
    MagAOX::app::loopTiming::Monitor::configure;
    MagAOX::app::loopTiming::Monitor::camera;
    MagAOX::app::loopTiming::Monitor::sdk;
    MagAOX::app::loopTiming::Monitor::summary;
#endif
    // clang-format on
}

/** \brief Startup never treats an interval predating SDK observation coverage as complete. */
TEST_CASE( "loop timing discards partial startup intervals", "[loopTimingMonitor][loopTiming]" )
{
    Monitor monitor;
    REQUIRE( monitor.configure( 3600, 2, 100 ) );
    camera( monitor, 1, 0 );
    monitor.sdk( 1, base + 100000 );
    camera( monitor, 2, 500000 );
    monitor.sdk( 2, base + 600000 );
    CHECK( monitor.summary().quality.accepted == 0 );
    camera( monitor, 3, 1000000 );
    monitor.sdk( 3, base + 1100000 );
    monitor.update( base + 1200000 );
    CHECK( monitor.summary().latency.count == 1 );
    CHECK_FALSE( monitor.summary().latency.valid );
    CHECK( monitor.summary().latency.mean == Approx( 100e-6 ) );
    // clang-format off
#ifdef LOOPTIMINGMONITOR_TEST_DOXYGEN_REF
    MagAOX::app::loopTiming::Monitor::configure;
    MagAOX::app::loopTiming::Monitor::camera;
    MagAOX::app::loopTiming::Monitor::sdk;
    MagAOX::app::loopTiming::Monitor::update;
    MagAOX::app::loopTiming::Monitor::summary;
#endif
    // clang-format on
}

/** \brief Independent source gaps count missed publications and prevent matching across lost coverage. */
TEST_CASE( "loop timing detects source gaps and recovers continuity", "[loopTimingMonitor][loopTiming]" )
{
    Monitor monitor;
    REQUIRE( monitor.configure( 3600, 2, 100 ) );
    baseline( monitor );
    camera( monitor, 1, 0 );
    monitor.sdk( 1, base + 100000 );
    const bool cameraGap = GENERATE( false, true );

    if( cameraGap )
    {
        camera( monitor, 4, 1500000 );
        CHECK( monitor.summary().quality.cameraMissed == 2 );
        CHECK( monitor.summary().quality.sdkMissed == 0 );
        monitor.sdk( 4, base + 1600000 );
    }
    else
    {
        monitor.sdk( 4, base + 1600000 );
        CHECK( monitor.summary().quality.sdkMissed == 2 );
        CHECK( monitor.summary().quality.cameraMissed == 0 );
    }

    camera( monitor, 5, 2000000 );
    monitor.sdk( 5, base + 2100000 );
    camera( monitor, 6, 2500000 );
    monitor.sdk( 6, base + 2600000 );
    monitor.update( base + 2700000 );
    CHECK( monitor.summary().quality.accepted == 1 );
    CHECK( monitor.summary().latency.mean == Approx( 100e-6 ) );
    CHECK( monitor.summary().quality.resets == 1 );
    // clang-format off
#ifdef LOOPTIMINGMONITOR_TEST_DOXYGEN_REF
    MagAOX::app::loopTiming::Monitor::configure;
    MagAOX::app::loopTiming::Monitor::camera;
    MagAOX::app::loopTiming::Monitor::sdk;
    MagAOX::app::loopTiming::Monitor::update;
    MagAOX::app::loopTiming::Monitor::summary;
#endif
    // clang-format on
}

/** \brief Unchanged polls are harmless, while changed identities, counter wraps and backsteps reset continuity. */
TEST_CASE( "loop timing separates repeated polls from source regressions", "[loopTimingMonitor][loopTiming]" )
{
    Monitor monitor;
    REQUIRE( monitor.configure( 3600, 2, 100 ) );
    baseline( monitor );
    camera( monitor, 5, 0 );
    camera( monitor, 5, 0 );
    CHECK( monitor.summary().quality.invalid == 0 );
    CHECK( monitor.summary().quality.resets == 0 );

    SECTION( "changed timestamp with unchanged counter" )
    {
        camera( monitor, 5, 1 );
        CHECK( monitor.summary().quality.invalid == 1 );
    }

    SECTION( "counter regression" )
    {
        camera( monitor, 1, 500000 );
        CHECK( monitor.summary().quality.counterRegressions == 1 );
    }

    SECTION( "timestamp regression" )
    {
        camera( monitor, 6, -1 );
        CHECK( monitor.summary().quality.invalid == 1 );
    }

    SECTION( "timestamp regression accompanying a gap" )
    {
        camera( monitor, 7, -1 );
        CHECK( monitor.summary().quality.invalid == 1 );
        CHECK( monitor.summary().quality.cameraMissed == 1 );
    }

    SECTION( "counter wrap" )
    {
        monitor.sdk( UINT64_MAX, base + 100000 );
        monitor.sdk( 0, base + 600000 );
        CHECK( monitor.summary().quality.counterRegressions == 1 );
    }
    // clang-format off
#ifdef LOOPTIMINGMONITOR_TEST_DOXYGEN_REF
    MagAOX::app::loopTiming::Monitor::configure;
    MagAOX::app::loopTiming::Monitor::camera;
    MagAOX::app::loopTiming::Monitor::sdk;
    MagAOX::app::loopTiming::Monitor::summary;
#endif
    // clang-format on
}

/** \brief Missing commands remain unmatched and do not shift subsequent valid associations. */
TEST_CASE( "loop timing rejects absent SDK commands without reusing another frame", "[loopTimingMonitor][loopTiming]" )
{
    Monitor monitor;
    REQUIRE( monitor.configure( 3600, 2, 100 ) );
    baseline( monitor );
    camera( monitor, 1, 0 );
    camera( monitor, 2, 500000 );
    camera( monitor, 3, 1000000 );
    monitor.sdk( 1, base + 600000 );
    monitor.sdk( 2, base + 1100000 );
    monitor.update( base + 1200000 );
    CHECK( monitor.summary().quality.unmatched == 1 );
    CHECK( monitor.summary().quality.accepted == 1 );
    CHECK( monitor.summary().latency.mean == Approx( 100e-6 ) );
    // clang-format off
#ifdef LOOPTIMINGMONITOR_TEST_DOXYGEN_REF
    MagAOX::app::loopTiming::Monitor::configure;
    MagAOX::app::loopTiming::Monitor::camera;
    MagAOX::app::loopTiming::Monitor::sdk;
    MagAOX::app::loopTiming::Monitor::update;
    MagAOX::app::loopTiming::Monitor::summary;
#endif
    // clang-format on
}

/** \brief Boundary equality and mapping uncertainty exclude potentially misassociated endpoints. */
TEST_CASE( "loop timing rejects uncertain interval boundaries", "[loopTimingMonitor][loopTiming]" )
{
    Monitor monitor;
    REQUIRE( monitor.configure( 3600, 2, 100 ) );
    baseline( monitor );
    camera( monitor, 1, 0, 1000 );
    camera( monitor, 2, 500000, 1000 );
    const auto offset = GENERATE( int64_t( 0 ), int64_t( 1000 ), int64_t( 499000 ), int64_t( 500000 ) );
    monitor.sdk( 1, base + offset );
    monitor.sdk( 2, base + 600000 );
    monitor.update( base + 700000 );
    CHECK( monitor.summary().quality.accepted == 0 );
    CHECK( monitor.summary().quality.ambiguous == 1 );
    // clang-format off
#ifdef LOOPTIMINGMONITOR_TEST_DOXYGEN_REF
    MagAOX::app::loopTiming::Monitor::configure;
    MagAOX::app::loopTiming::Monitor::camera;
    MagAOX::app::loopTiming::Monitor::sdk;
    MagAOX::app::loopTiming::Monitor::update;
    MagAOX::app::loopTiming::Monitor::summary;
#endif
    // clang-format on
}

/** \brief Maximum FPS sets a strict latency bound without rejecting short camera intervals caused by jitter. */
TEST_CASE( "loop timing applies maximum FPS to latency rather than rejecting jitter",
           "[loopTimingMonitor][loopTiming]" )
{
    Monitor monitor;
    REQUIRE( monitor.configure( 3600, 2, 100 ) );
    baseline( monitor );
    camera( monitor, 1, 0 );
    const bool shortInterval = GENERATE( false, true );

    if( shortInterval )
    {
        camera( monitor, 2, 260000 );
        monitor.sdk( 1, base + 100000 );
        monitor.sdk( 2, base + 360000 );
        CHECK( monitor.summary().quality.accepted == 1 );
    }
    else
    {
        camera( monitor, 2, 500000 );
        monitor.sdk( 1, base + 300000 );
        monitor.sdk( 2, base + 600000 );
        CHECK( monitor.summary().quality.accepted == 0 );
        CHECK( monitor.summary().quality.invalid == 1 );
    }
    // clang-format off
#ifdef LOOPTIMINGMONITOR_TEST_DOXYGEN_REF
    MagAOX::app::loopTiming::Monitor::configure;
    MagAOX::app::loopTiming::Monitor::camera;
    MagAOX::app::loopTiming::Monitor::sdk;
    MagAOX::app::loopTiming::Monitor::summary;
#endif
    // clang-format on
}

/** \brief Original camera timestamps preserve native-clock cadence independently of mapping changes. */
TEST_CASE( "loop timing measures camera intervals in the source clock", "[loopTimingMonitor][loopTiming]" )
{
    Monitor monitor;
    REQUIRE( monitor.configure( 3600, 2, 100 ) );
    monitor.camera( 1, base, base - 1000, base + 1000, base + 1000000000 );
    monitor.camera( 2, base + 500200, base + 499200, base + 501200, base + 1000500000 );
    monitor.camera( 3, base + 999800, base + 998800, base + 1000800, base + 1001000000 );
    monitor.update( base + 1100000 );
    CHECK( monitor.summary().cameraIntervals.count == 2 );
    CHECK( monitor.summary().cameraIntervals.mean == Approx( 500e-6 ) );
    CHECK( monitor.summary().cameraIntervals.stddev == Approx( 0 ).margin( 1e-15 ) );
    // clang-format off
#ifdef LOOPTIMINGMONITOR_TEST_DOXYGEN_REF
    MagAOX::app::loopTiming::Monitor::configure;
    MagAOX::app::loopTiming::Monitor::camera;
    MagAOX::app::loopTiming::Monitor::update;
    MagAOX::app::loopTiming::Monitor::summary;
#endif
    // clang-format on
}

/** \brief Statistics expire by event time while freshness retains the age of the last accepted endpoint. */
TEST_CASE( "loop timing expires stopped streams and clears statistics on epoch changes",
           "[loopTimingMonitor][loopTiming]" )
{
    Monitor monitor;
    REQUIRE( monitor.configure( 3600, 0.001, 100 ) );
    baseline( monitor );
    camera( monitor, 1, 0 );
    monitor.sdk( 1, base + 100000 );
    camera( monitor, 2, 500000 );
    monitor.sdk( 2, base + 600000 );
    monitor.update( base + 700000 );
    CHECK( monitor.summary().latency.count == 1 );
    monitor.update( base + 2000000 );
    CHECK( monitor.summary().latency.count == 0 );
    CHECK( monitor.summary().cameraIntervals.count == 0 );
    CHECK( monitor.summary().sdkIntervals.count == 0 );
    CHECK( monitor.summary().age == Approx( 1900e-6 ) );
    const auto epoch = monitor.summary().epoch;
    monitor.reset();
    CHECK( monitor.summary().newestNs == 0 );
    CHECK( monitor.summary().age == -1 );
    CHECK( monitor.summary().epoch != epoch );
    CHECK( monitor.summary().quality.accepted == 1 );
    // clang-format off
#ifdef LOOPTIMINGMONITOR_TEST_DOXYGEN_REF
    MagAOX::app::loopTiming::Monitor::configure;
    MagAOX::app::loopTiming::Monitor::camera;
    MagAOX::app::loopTiming::Monitor::sdk;
    MagAOX::app::loopTiming::Monitor::update;
    MagAOX::app::loopTiming::Monitor::reset;
    MagAOX::app::loopTiming::Monitor::summary;
#endif
    // clang-format on
}

/** \brief Bounded queues count loss and discard associations spanning a pending-history overflow. */
TEST_CASE( "loop timing bounds pending history and retained populations", "[loopTimingMonitor][loopTiming]" )
{
    Monitor monitor;
    REQUIRE( monitor.configure( 3600, 2, 3 ) );
    baseline( monitor );
    camera( monitor, 1, 0 );
    camera( monitor, 2, 500000 );
    camera( monitor, 3, 1000000 );
    camera( monitor, 4, 1500000 );
    CHECK( monitor.summary().quality.overflow == 1 );
    CHECK( monitor.summary().quality.resets == 1 );
    monitor.sdk( 1, base + 1600000 );

    for( uint64_t index = 5; index < 12; ++index )
    {
        camera( monitor, index, int64_t( index - 1 ) * 500000 );
        monitor.sdk( index - 3, base + int64_t( index - 1 ) * 500000 + 100000 );
    }

    monitor.update( base + 6000000 );
    CHECK( monitor.summary().latency.count == 3 );
    CHECK( monitor.summary().cameraIntervals.count == 3 );
    CHECK( monitor.summary().sdkIntervals.count == 3 );
    CHECK( monitor.summary().quality.overflow > 1 );
    // clang-format off
#ifdef LOOPTIMINGMONITOR_TEST_DOXYGEN_REF
    MagAOX::app::loopTiming::Monitor::configure;
    MagAOX::app::loopTiming::Monitor::camera;
    MagAOX::app::loopTiming::Monitor::sdk;
    MagAOX::app::loopTiming::Monitor::update;
    MagAOX::app::loopTiming::Monitor::summary;
#endif
    // clang-format on
}

/** \brief Malformed camera bounds and a regressing summary clock cannot expose a valid old population. */
TEST_CASE( "loop timing rejects malformed bounds and summary clock regressions", "[loopTimingMonitor][loopTiming]" )
{
    Monitor monitor;
    REQUIRE( monitor.configure( 3600, 2, 100 ) );
    monitor.camera( 1, base, base + 1, base + 2 );
    CHECK( monitor.summary().quality.invalid == 1 );
    monitor.update( base );
    monitor.update( base - 1 );
    CHECK( monitor.summary().quality.invalid == 2 );
    CHECK_FALSE( monitor.summary().latency.valid );
    // clang-format off
#ifdef LOOPTIMINGMONITOR_TEST_DOXYGEN_REF
    MagAOX::app::loopTiming::Monitor::configure;
    MagAOX::app::loopTiming::Monitor::camera;
    MagAOX::app::loopTiming::Monitor::update;
    MagAOX::app::loopTiming::Monitor::summary;
#endif
    // clang-format on
}

/** \brief The shared clock retains synco's API, clock bounds, discontinuity checks and age rejection. */
TEST_CASE( "loop timing shares the existing qualified publication clock", "[loopTimingMonitor][publicationClock]" )
{
    using MagAOX::timing::PublicationClock;
    STATIC_REQUIRE( std::is_same_v<PublicationClock, MagAOX::app::synco::PublicationClock> );
    PublicationClock  clock;
    constexpr int64_t offset = 2000000000000000000LL;
    REQUIRE( clock.configure() );
    REQUIRE( clock.sample( base, offset + base + 50, base + 100 ) );
    CHECK( clock.map( offset + base - 1000 ) == base - 1000 );
    CHECK( clock.status().publicationLowerNs == base - 1051 );
    CHECK( clock.status().publicationUpperNs == base - 949 );
    CHECK( clock.project( offset + base + 125000 ) == base + 125000 );
    CHECK( clock.map( offset + base + 125000 ) == 0 );
    CHECK( clock.map( offset + base - 37000000000LL ) == 0 );
    CHECK( clock.status().stalePublication == 1 );
    const auto epoch = clock.status().epoch;
    CHECK_FALSE( clock.sample( base + 500000, offset + base + 501000, base + 502000 ) );
    CHECK( clock.status().wideBracket == 1 );
    CHECK( clock.status().epoch == epoch );
    CHECK_FALSE( clock.sample( base + 1000000, offset + base + 2000050, base + 1000100 ) );
    CHECK( clock.status().discontinuity == 1 );
    CHECK( clock.status().epoch != epoch );
    CHECK_FALSE( clock.status().valid );
    // clang-format off
#ifdef LOOPTIMINGMONITOR_TEST_DOXYGEN_REF
    MagAOX::timing::PublicationClock::configure;
    MagAOX::timing::PublicationClock::sample;
    MagAOX::timing::PublicationClock::map;
    MagAOX::timing::PublicationClock::project;
    MagAOX::timing::PublicationClock::status;
#endif
    // clang-format on
}

} // namespace loopTimingMonitorTest
} // namespace libXWCTest
