/** \file loopTimingSynco_test.cpp
 * \brief Exact-main joins, automatic mode epochs and auxiliary rejection with synthetic records.
 * \ingroup loopTimingMonitor_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "../loopTimingSynco.hpp"
#include "../loopTiming.hpp"

namespace libXWCTest
{
namespace loopTimingMonitorTest
{
/// \cond LOOP_TIMING_SYNCO_HELPERS
using Synco  = MagAOX::app::loopTiming::Synco;
using Record = MagAOX::dmTimingEvents::Record;
using Kind   = MagAOX::dmTimingEvents::Kind;

/// Construct a native endpoint with a unique combined identity.
Record native( uint64_t sequence /**< [in] Native producer sequence and combined counter. */,
               int64_t  startNs /**< [in] Original monotonic SDK entry. */ );

/// Classify an exact native identity as one genuine primary main.
Record mainLabel( const Record &sdk /**< [in] Native identity to qualify. */,
                  uint64_t      sequence /**< [in] Main producer sequence. */,
                  uint64_t      primary /**< [in] Genuine primary counter. */,
                  uint64_t      generation = 1 /**< [in] Frozen mode generation. */,
                  bool          enabled    = true /**< [in] Effective synco enable state. */ );

/// Construct a mode change independently of native endpoint arrival.
Record mode( uint64_t sequence /**< [in] Main producer sequence. */,
             uint64_t generation /**< [in] New mode generation. */,
             bool     enabled /**< [in] Effective enable state. */ );

Record native( uint64_t sequence, int64_t startNs )
{
    Record record;
    record.kind     = Kind::sdk;
    record.key      = { 42, sequence, 10, int64_t( sequence ) };
    record.sequence = sequence;
    record.startNs  = startNs;
    return record;
}

Record mainLabel( const Record &sdk, uint64_t sequence, uint64_t primary, uint64_t generation, bool enabled )
{
    Record record;
    record.kind           = Kind::main;
    record.key            = sdk.key;
    record.sequence       = sequence;
    record.generation     = generation;
    record.primaryInode   = 7;
    record.primaryCounter = primary;
    record.steps          = 4;
    record.enabled        = enabled;
    return record;
}

Record mode( uint64_t sequence, uint64_t generation, bool enabled )
{
    auto record = mainLabel( native( 1, 1 ), sequence, 1, generation, enabled );
    record.kind = Kind::mode;
    record.key  = {};
    return record;
}
/// \endcond

/** \brief Four-step traffic, including a preceding cycle's auxiliary, contributes only main latency and cadence.
 * \ingroup loopTimingMonitor_unit_test
 */
TEST_CASE( "synco join excludes auxiliaries across camera boundaries", "[loopTimingMonitor][syncoJoin]" )
{
    Synco join;
    join.configure( 128, 2000000000 );
    MagAOX::app::loopTiming::Monitor monitor;
    REQUIRE( monitor.configure( 3600, 2, 128 ) );
    constexpr int64_t base        = 1000000000;
    const bool        labelsFirst = GENERATE( false, true );

    for( uint64_t frame = 1; frame <= 10; ++frame )
    {
        const auto stamp = base + int64_t( frame - 1 ) * 500000;
        monitor.camera( frame, stamp, stamp, stamp );
    }

    // The first native endpoint is the preceding cycle's step 3, only 41 us after camera 1.
    join.observe( native( 1, base + 41000 ), 10, base );
    uint64_t matches = 0;

    for( uint64_t frame = 1; frame <= 9; ++frame )
    {
        const auto sdk   = native( 4 * frame - 2, base + int64_t( frame - 1 ) * 500000 + 166000 );
        const auto label = mainLabel( sdk, frame, frame );

        if( labelsFirst )
        {
            join.observe( label, 20, base );
        }

        join.observe( sdk, 10, base );
        join.observe( label, 20, base );
        Synco::Match match;

        while( join.pop( match, base ) )
        {
            CHECK( match.counter == frame );
            CHECK( match.startNs == sdk.startNs );
            monitor.sdk( match.counter, match.startNs );
            ++matches;
        }

        for( uint64_t step = 1; step <= 3; ++step )
        {
            join.observe( native( sdk.sequence + step, sdk.startNs + int64_t( step ) * 125000 ), 10, base );
            CHECK_FALSE( join.pop( match, base ) );
        }
    }

    CHECK( matches == 9 );
    monitor.update( base + 5000000 );
    REQUIRE( monitor.summary().latency.valid );
    CHECK( monitor.summary().latency.mean == Approx( 166e-6 ) );
    CHECK( monitor.summary().sdkIntervals.mean == Approx( 500e-6 ) );
    CHECK( join.quality().sdkMissed == 0 );

    // clang-format off
#ifdef LOOPTIMINGMONITOR_TEST_DOXYGEN_REF
    MagAOX::app::loopTiming::Synco::observe;
    MagAOX::app::loopTiming::Synco::pop;
#endif
    // clang-format on
}

/** \brief Keep native results in primary order and never replace a missing main with an auxiliary.
 * \ingroup loopTimingMonitor_unit_test
 */
TEST_CASE( "synco join orders labels and detects missing native mains", "[loopTimingMonitor][syncoJoin]" )
{
    Synco join;
    join.configure( 8, 100 );
    const auto first  = native( 1, 1000 );
    const auto second = native( 2, 1500 );
    join.observe( mainLabel( first, 1, 10 ), 20, 1 );
    join.observe( mainLabel( second, 2, 11 ), 20, 1 );
    Synco::Match match;
    CHECK_FALSE( join.pop( match, 1 ) );
    join.observe( first, 10, 2 );
    REQUIRE( join.pop( match, 2 ) );
    CHECK( match.counter == 10 );
    join.observe( second, 10, 2 );
    REQUIRE( join.pop( match, 2 ) );
    CHECK( match.counter == 11 );

    auto missing = native( 3, 2000 );
    join.observe( mainLabel( missing, 3, 12 ), 20, 3 );
    auto auxiliary     = native( 4, 2100 );
    auxiliary.sequence = 3; // A failed SDK call produces no successful native record.
    join.observe( auxiliary, 10, 3 );
    const auto epoch = join.epoch();
    CHECK_FALSE( join.pop( match, 3 ) );
    CHECK( join.quality().unmatched == 1 );
    CHECK( join.epoch() > epoch );
    join.observe( mainLabel( native( 5, 2500 ), 4, 13 ), 20, 4 );
    CHECK_FALSE( join.pop( match, 105 ) );
    CHECK( join.quality().unmatched == 2 );

    // clang-format off
#ifdef LOOPTIMINGMONITOR_TEST_DOXYGEN_REF
    MagAOX::app::loopTiming::Synco::observe;
    MagAOX::app::loopTiming::Synco::pop;
    MagAOX::app::loopTiming::Synco::epoch;
#endif
    // clang-format on
}

/** \brief Frozen label modes survive delayed endpoints, rapid toggles and publisher restarts.
 * \ingroup loopTimingMonitor_unit_test
 */
TEST_CASE( "synco join switches ordinary and step0 epochs automatically", "[loopTimingMonitor][syncoJoin]" )
{
    Synco join;
    join.configure( 16, 1000 );
    Synco::Match match;
    const auto   first = native( 1, 1000 );
    join.observe( first, 10, 1 );
    join.observe( mainLabel( first, 1, 1, 1, false ), 20, 1 );
    REQUIRE( join.pop( match, 1 ) );
    REQUIRE( join.known() );
    CHECK_FALSE( join.enabled() );
    auto epoch = join.epoch();
    join.observe( mode( 2, 2, true ), 20, 2 );
    CHECK( join.enabled() );
    CHECK( join.epoch() > epoch );
    join.observe( mainLabel( native( 2, 1500 ), 3, 2, 2, true ), 20, 2 );
    join.observe( mode( 4, 3, false ), 20, 3 );
    join.observe( native( 2, 1500 ), 10, 3 );
    CHECK_FALSE( join.pop( match, 3 ) );
    join.observe( native( 3, 1600 ), 10, 3 ); // Late auxiliary after Off.
    CHECK_FALSE( join.pop( match, 3 ) );
    join.observe( mode( 5, 4, true ), 20, 3 );
    join.observe( mode( 6, 5, false ), 20, 3 );
    CHECK( join.generation() == 5 );
    CHECK_FALSE( join.enabled() );
    join.observe( native( 4, 2000 ), 10, 3 );
    join.observe( mainLabel( native( 4, 2000 ), 7, 4, 5, false ), 20, 3 );
    REQUIRE( join.pop( match, 3 ) );
    CHECK( match.counter == 4 );
    epoch = join.epoch();
    join.observe( mode( 1, 1, false ), 21, 4 );
    CHECK( join.epoch() > epoch );
    CHECK( join.generation() == 1 );

    // clang-format off
#ifdef LOOPTIMINGMONITOR_TEST_DOXYGEN_REF
    MagAOX::app::loopTiming::Synco::observe;
    MagAOX::app::loopTiming::Synco::pop;
    MagAOX::app::loopTiming::Synco::epoch;
#endif
    // clang-format on
}

/** \brief Detect capture loss, conflicting duplicates, partial identity matches and bounded history exhaustion.
 * \ingroup loopTimingMonitor_unit_test
 */
TEST_CASE( "synco join rejects lost and conflicting identity evidence", "[loopTimingMonitor][syncoJoin]" )
{
    Synco join;
    join.configure( 4, 100 );
    const auto sdk = native( 1, 1000 );
    join.observe( sdk, 10, 1 );
    join.observe( sdk, 10, 1 );
    CHECK( join.quality().duplicates == 1 );
    auto label = mainLabel( sdk, 1, 1 );
    ++label.key.nsec;
    join.observe( label, 20, 1 );
    Synco::Match match;
    CHECK_FALSE( join.pop( match, 1 ) );
    CHECK( join.quality().unmatched == 1 );
    const auto epoch = join.epoch();
    join.observe( native( 3, 2000 ), 10, 2 );
    CHECK( join.epoch() > epoch );
    CHECK( join.quality().sdkMissed == 1 );
    join.observe( mode( 3, 2, true ), 20, 2 );
    CHECK( join.quality().mainMissed == 1 );
    auto conflict = mode( 3, 2, false );
    join.observe( conflict, 20, 2 );
    CHECK_FALSE( join.known() );
    CHECK( join.quality().invalid == 1 );

    for( uint64_t i = 0; i < 5; ++i )
    {
        join.observe( mainLabel( native( i + 10, 3000 + int64_t( i ) ), i + 4, i + 10, 3 ), 20, 3 );
    }

    CHECK( join.quality().overflow == 1 );
    CHECK_FALSE( join.pop( match, 104 ) );
    CHECK( join.quality().unmatched == 2 );

    // clang-format off
#ifdef LOOPTIMINGMONITOR_TEST_DOXYGEN_REF
    MagAOX::app::loopTiming::Synco::observe;
    MagAOX::app::loopTiming::Synco::pop;
    MagAOX::app::loopTiming::Synco::epoch;
#endif
    // clang-format on
}

} // namespace loopTimingMonitorTest
} // namespace libXWCTest
