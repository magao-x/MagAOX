/** \file wfsScanSchedule_test.cpp
 * \brief Catch2 tests for WFS-referenced ADC scan spacing, cadence changes, phase, and loss.
 *
 * Ported from ad7689Ctrl's scheduler tests when the schedule moved to libMagAOX.
 */

#include "../../../tests/testXWC.hpp"

#include "../wfsScanSchedule.hpp"

#include <limits>
#include <utility>

namespace libXWCTest
{
namespace wfsScanScheduleTest
{

using MagAOX::app::wfsScan::syncSchedule;

/** Four 125-us scans and three fractional-period scans stay aligned without rounding drift.
 * \ingroup app_unit_test
 */
TEST_CASE( "wfsScan subscan spacing at 8 and 6 kHz", "[wfsScanSchedule]" )
{
    // clang-format off
#ifdef WFSSCANSCHEDULE_TEST_DOXYGEN_REF
    MagAOX::app::wfsScan::syncSchedule::arm;
    MagAOX::app::wfsScan::syncSchedule::deadline;
    MagAOX::app::wfsScan::syncSchedule::take;
#endif
    // clang-format on
    syncSchedule  clock;
    const int64_t epoch = 1000000000;
    REQUIRE( clock.arm( epoch, 500000, 8000, 0, 56000, 20000 ) );
    REQUIRE( clock.m_count == 4 );
    REQUIRE( clock.fps() == 8000 );

    for( int64_t offset : { 0LL, 125000LL, 250000LL, 375000LL } )
    {
        REQUIRE( clock.deadline() == epoch + offset );
        REQUIRE_FALSE( clock.take( epoch + offset - 1 ) );
        REQUIRE( clock.take( epoch + offset ) );
    }

    REQUIRE_FALSE( clock.take( epoch + 500000 ) );
    REQUIRE( clock.m_skipped == 0 );
    clock = {};

    for( int64_t frame = 0; frame < 1000; ++frame )
    {
        const int64_t anchor = epoch + frame * 500000;
        REQUIRE( clock.arm( anchor, 500000, 6000, 0, 56000, 20000 ) );
        REQUIRE( clock.m_count == 3 );
        REQUIRE( clock.fps() == Approx( 6000 ) );

        for( int64_t offset : { 0LL, 166667LL, 333333LL } )
        {
            REQUIRE( clock.deadline() == anchor + offset );
            REQUIRE( clock.take( anchor + offset ) );
        }
    }

    REQUIRE( clock.m_skipped == 0 );
    REQUIRE( clock.end() == epoch + 500000000 );
}

/** Re-anchor integer scan counts and report the actual rate when the ratio is non-integral.
 * \ingroup app_unit_test
 */
TEST_CASE( "wfsScan cadence and requested rate changes", "[wfsScanSchedule]" )
{
    // clang-format off
#ifdef WFSSCANSCHEDULE_TEST_DOXYGEN_REF
    MagAOX::app::wfsScan::syncSchedule::arm;
    MagAOX::app::wfsScan::syncSchedule::fps;
#endif
    // clang-format on
    syncSchedule clock;
    REQUIRE( clock.arm( 1000000000, 500000, 8000, 0, 56000, 20000 ) );
    REQUIRE( clock.arm( 1000500000, 1000000, 8000, 0, 56000, 20000 ) );
    REQUIRE( clock.m_count == 8 );
    REQUIRE( clock.m_interval == 125000 );
    REQUIRE( clock.m_skipped == 4 ); // Previous period had not been acquired.
    REQUIRE( clock.arm( 1001500000, 1000000, 6000, 0, 56000, 20000 ) );
    REQUIRE( clock.m_count == 6 );
    REQUIRE( clock.m_interval == Approx( 1000000.0 / 6 ) );
    REQUIRE( clock.arm( 1002500000, 1e9 / 1800, 8000, 0, 56000, 20000 ) );
    REQUIRE( clock.m_count == 4 );
    REQUIRE( clock.fps() == Approx( 7200 ) );
    REQUIRE( clock.m_interval == Approx( 1e9 / 7200 ) );
}

/** The WFS phase selects a slot-equivalent first offset; leave ADC and jitter/publication headroom.
 * \ingroup app_unit_test
 */
TEST_CASE( "wfsScan phase wrapping and service cap", "[wfsScanSchedule]" )
{
    // clang-format off
#ifdef WFSSCANSCHEDULE_TEST_DOXYGEN_REF
    MagAOX::app::wfsScan::syncSchedule::arm;
    MagAOX::app::wfsScan::syncSchedule::deadline;
#endif
    // clang-format on
    syncSchedule clock;
    REQUIRE( clock.arm( 1000000000, 500000, 8000, 409100, 56000, 20000 ) );
    REQUIRE( clock.m_phase == 34100 );
    REQUIRE( clock.deadline() == 1000034100 );
    REQUIRE( clock.arm( 1000500000, 500000, 8000, -100000, 56000, 20000 ) );
    REQUIRE( clock.m_phase == 25000 );
    REQUIRE( clock.arm( 1001000000, 500000, 8000, 120000, 56000, 20000 ) );
    REQUIRE( clock.m_phase == 49000 );

    for( unsigned i = 0; i < 4; ++i )
    {
        REQUIRE( clock.take( clock.deadline() + 1000 ) ); // Guard permits finite scheduler lateness.
    }

    REQUIRE( clock.m_skipped == 8 ); // Two earlier schedules were replaced without acquisition.
}

/** Skip expired work and avoid compressed rephasing across a new WFS edge; no free-running after the period.
 * \ingroup app_unit_test
 */
TEST_CASE( "wfsScan missed subscans never become catch-up bursts", "[wfsScanSchedule]" )
{
    // clang-format off
#ifdef WFSSCANSCHEDULE_TEST_DOXYGEN_REF
    MagAOX::app::wfsScan::syncSchedule::discardLate;
    MagAOX::app::wfsScan::syncSchedule::cancel;
    MagAOX::app::wfsScan::syncSchedule::take;
#endif
    // clang-format on
    syncSchedule clock;
    REQUIRE( clock.arm( 1000000000, 500000, 8000, 0, 56000, 20000 ) );
    REQUIRE( clock.take( 1000250000 ) );
    REQUIRE( clock.m_skipped == 2 );
    REQUIRE_FALSE( clock.take( 1000250001 ) );
    REQUIRE( clock.deadline() == 1000375000 );
    REQUIRE( clock.arm( 1000300000, 500000, 8000, 0, 56000, 20000 ) );
    REQUIRE_FALSE( clock.take( 1000300000 ) ); // New edge too close to the preceding sample.
    REQUIRE( clock.deadline() == 1000425000 );
    REQUIRE( clock.take( 1000425000 ) );
    REQUIRE_FALSE( clock.take( 1000800000 ) );
    REQUIRE( clock.m_slot == clock.m_count );
    const uint64_t skipped = clock.m_skipped;
    clock.cancel();
    REQUIRE( clock.m_count == 0 );
    REQUIRE_FALSE( clock.take( 1001000000 ) );
    REQUIRE( clock.m_skipped == skipped );
}

/** The scan count is the highest integer multiple of the WFS rate not above the maximum rate.
 * \ingroup app_unit_test
 */
TEST_CASE( "wfsScan highest integer multiple of the WFS rate within the limit", "[wfsScanSchedule]" )
{
    // clang-format off
#ifdef WFSSCANSCHEDULE_TEST_DOXYGEN_REF
    MagAOX::app::wfsScan::syncSchedule::arm;
    MagAOX::app::wfsScan::syncSchedule::fps;
#endif
    // clang-format on
    syncSchedule clock;

    // WFS rate [Hz], expected scans per WFS period with an 8 kHz limit.
    for( auto [wfs, count] : { std::pair{ 2000.0, 4u },
                               std::pair{ 500.0, 16u },
                               std::pair{ 700.0, 11u },
                               std::pair{ 1700.0, 4u },
                               std::pair{ 2000.4, 4u },
                               std::pair{ 1000.0, 8u } } )
    {
        REQUIRE( clock.arm( 1000000000, 1e9 / wfs, 8000, 0, 40000, 20000 ) );
        REQUIRE( clock.m_count == count );
        REQUIRE( clock.fps() == Approx( count * wfs ) );
        REQUIRE( clock.fps() <= 8000 * ( 1 + MagAOX::app::wfsScan::c_ratioTolerance ) );
    }

    // A WFS faster than the limit still gets one scan per WFS frame.
    REQUIRE( clock.arm( 1000000000, 1e9 / 9000, 8000, 0, 40000, 20000 ) );
    REQUIRE( clock.m_count == 1 );

    // A lower limit lowers the multiple: 6 kHz gives 3 scans at 2 kHz and 8 at 700 Hz.
    REQUIRE( clock.arm( 1000000000, 500000, 6000, 0, 40000, 20000 ) );
    REQUIRE( clock.m_count == 3 );
    REQUIRE( clock.arm( 1000000000, 1e9 / 700, 6000, 0, 40000, 20000 ) );
    REQUIRE( clock.m_count == 8 );

    // Never above the limit: 2.5 scans per period at 10 kHz is 2, where rounding would give 3 (12 kHz).
    REQUIRE( clock.arm( 1000000000, 250000, 10000, 0, 10000, 1000 ) );
    REQUIRE( clock.m_count == 2 );
}

/** A scan service too long for the rate limit lowers the count to the highest that fits, rather than none.
 * \ingroup app_unit_test
 */
TEST_CASE( "wfsScan lowers the scan count to fit the scan service", "[wfsScanSchedule]" )
{
    // clang-format off
#ifdef WFSSCANSCHEDULE_TEST_DOXYGEN_REF
    MagAOX::app::wfsScan::syncSchedule::arm;
#endif
    // clang-format on
    syncSchedule clock;

    // 106.5 us reads plus a 20 us guard do not fit 125 us: 3 scans (6 kHz) at a 2 kHz WFS.
    REQUIRE( clock.arm( 1000000000, 500000, 8000, 0, 106530, 20000 ) );
    REQUIRE( clock.m_count == 3 );
    REQUIRE( clock.fps() == Approx( 6000.0 ) );
    REQUIRE( clock.m_budget < clock.m_interval );

    // A 10 us guard fits 4 scans again.
    REQUIRE( clock.arm( 1000000000, 500000, 8000, 0, 106530, 10000 ) );
    REQUIRE( clock.m_count == 4 );

    // A budget dividing the period exactly leaves no margin, so one scan fewer.
    REQUIRE( clock.arm( 1000000000, 500000, 8000, 0, 105000, 20000 ) );
    REQUIRE( clock.m_count == 3 );
}

/** Reject invalid or physically impossible schedules.
 * \ingroup app_unit_test
 */
TEST_CASE( "wfsScan rejects impossible subscan timing", "[wfsScanSchedule]" )
{
    // clang-format off
#ifdef WFSSCANSCHEDULE_TEST_DOXYGEN_REF
    MagAOX::app::wfsScan::syncSchedule::arm;
#endif
    // clang-format on
    syncSchedule clock;
    REQUIRE_FALSE( clock.arm( -1, 500000, 8000, 0, 56000, 20000 ) );
    REQUIRE_FALSE( clock.arm( 1000000000, 0, 8000, 0, 56000, 20000 ) );
    REQUIRE_FALSE( clock.arm( 1000000000, 500000, 0, 0, 56000, 20000 ) );
    REQUIRE_FALSE( clock.arm( 1000000000, 500000, 8000, 0, 480000, 20000 ) ); // Not one scan fits.
    REQUIRE_FALSE( clock.arm( std::numeric_limits<int64_t>::max(), 500000, 8000, 0, 56000, 20000 ) );

    for( double invalid : { std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN() } )
    {
        REQUIRE_FALSE( clock.arm( 1000000000, invalid, 8000, 0, 56000, 20000 ) );
        REQUIRE_FALSE( clock.arm( 1000000000, 500000, invalid, 0, 56000, 20000 ) );
        REQUIRE_FALSE( clock.arm( 1000000000, 500000, 8000, invalid, 56000, 20000 ) );
        REQUIRE_FALSE( clock.arm( 1000000000, 500000, 8000, 0, invalid, 20000 ) );
    }
}

} // namespace wfsScanScheduleTest
} // namespace libXWCTest
