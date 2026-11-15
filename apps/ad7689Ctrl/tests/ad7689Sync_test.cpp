/** \file ad7689Sync_test.cpp
 * \brief Deterministic WFS subscan spacing, cadence changes, phase, and loss tests.
 * \ingroup ad7689Ctrl_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "../ad7689Sync.hpp"

#include <array>
#include <limits>

namespace libXWCTest
{
namespace ad7689CtrlTest
{
using MagAOX::app::ad7689CtrlDetail::syncSchedule;

/** Four 125-us scans and three fractional-period scans stay aligned without rounding drift.
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "AD7689 WFS subscan spacing at 8 and 6 kHz", "[ad7689Ctrl][syncSchedule]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689CtrlDetail::syncSchedule::arm;
    MagAOX::app::ad7689CtrlDetail::syncSchedule::deadline;
    MagAOX::app::ad7689CtrlDetail::syncSchedule::take;
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
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "AD7689 WFS cadence and requested rate changes", "[ad7689Ctrl][syncSchedule]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689CtrlDetail::syncSchedule::arm;
    MagAOX::app::ad7689CtrlDetail::syncSchedule::fps;
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
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "AD7689 WFS phase wrapping and service cap", "[ad7689Ctrl][syncSchedule]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689CtrlDetail::syncSchedule::arm;
    MagAOX::app::ad7689CtrlDetail::syncSchedule::deadline;
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
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "AD7689 missed subscans never become catch-up bursts", "[ad7689Ctrl][syncSchedule]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689CtrlDetail::syncSchedule::discardLate;
    MagAOX::app::ad7689CtrlDetail::syncSchedule::cancel;
    MagAOX::app::ad7689CtrlDetail::syncSchedule::take;
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

/** Reject invalid or physically impossible schedules, including a rounded rate above the admission limit.
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "AD7689 rejects impossible subscan timing", "[ad7689Ctrl][syncSchedule]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689CtrlDetail::syncSchedule::arm;
#endif
    // clang-format on
    syncSchedule clock;
    REQUIRE_FALSE( clock.arm( -1, 500000, 8000, 0, 56000, 20000 ) );
    REQUIRE_FALSE( clock.arm( 1000000000, 0, 8000, 0, 56000, 20000 ) );
    REQUIRE_FALSE( clock.arm( 1000000000, 500000, 0, 0, 56000, 20000 ) );
    REQUIRE_FALSE( clock.arm( 1000000000, 500000, 8000, 0, 224000, 20000 ) );
    REQUIRE_FALSE( clock.arm( 1000000000, 250000, 10000, 0, 10000, 1000 ) );
    REQUIRE_FALSE( clock.arm( std::numeric_limits<int64_t>::max(), 500000, 8000, 0, 56000, 20000 ) );

    for( double invalid : { std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN() } )
    {
        REQUIRE_FALSE( clock.arm( 1000000000, invalid, 8000, 0, 56000, 20000 ) );
        REQUIRE_FALSE( clock.arm( 1000000000, 500000, invalid, 0, 56000, 20000 ) );
        REQUIRE_FALSE( clock.arm( 1000000000, 500000, 8000, invalid, 56000, 20000 ) );
        REQUIRE_FALSE( clock.arm( 1000000000, 500000, 8000, 0, invalid, 20000 ) );
    }
}

/** The generic DDSPC clock supports all requested integer ratios and preserves the full phase delay.
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "AD7689 DDSPC generic integer ratios", "[ad7689Ctrl][ddspc]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689CtrlDetail::continuousSchedule::reference;
    MagAOX::app::ad7689CtrlDetail::continuousSchedule::take;
    MagAOX::app::ad7689CtrlDetail::ddspcSettings::delay;
#endif
    // clang-format on
    using namespace MagAOX::app::ad7689CtrlDetail;
    const std::array<std::array<double, 3>, 8> cases{ { { 2000, 8000, 4 },
                                                        { 2000, 6000, 3 },
                                                        { 2000, 4000, 2 },
                                                        { 2000, 2000, 1 },
                                                        { 1000, 8000, 8 },
                                                        { 1000, 6000, 6 },
                                                        { 1000, 3000, 3 },
                                                        { 500, 8000, 16 } } };

    for( const auto &profile : cases )
    {
        continuousSchedule clock;
        const double       period = 1e9 / profile[0];
        ddspcSettings      settings;
        const double       delay = settings.delay( period );
        REQUIRE( clock.reference( 1, 1000000000, period, profile[1], delay, delay, 56000, 20000, 0.05 ) );
        REQUIRE( clock.m_count == profile[2] );
        REQUIRE( clock.fps() == Approx( profile[1] ) );
        scanSlot slot;

        for( unsigned i = 0; i < clock.m_count; ++i )
        {
            const int64_t expected = 1000000000 + std::llround( delay + i * period / profile[2] );
            REQUIRE( clock.deadline() == expected );
            REQUIRE( clock.take( expected, slot ) );
            REQUIRE( slot.m_reference.m_id == 1 );
            REQUIRE( slot.m_slot == i );
        }

        REQUIRE_FALSE( clock.pending() );
        REQUIRE_FALSE( clock.take( clock.deadline(), slot ) );
        REQUIRE( clock.m_skipped == 0 );
    }
}

/** New WFS references authorize a single continuous lattice without replacing overlapping old slots.
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "AD7689 DDSPC continuous 300 us phase across WFS boundaries", "[ad7689Ctrl][ddspc]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689CtrlDetail::continuousSchedule::reference;
    MagAOX::app::ad7689CtrlDetail::continuousSchedule::take;
#endif
    // clang-format on
    using namespace MagAOX::app::ad7689CtrlDetail;
    continuousSchedule clock;
    const int64_t      epoch = 1000000000;
    REQUIRE( clock.reference( 1000, epoch, 500000, 8000, 300000, 300000, 15000, 10000, 0.05 ) );
    uint64_t reference     = 1001;
    int64_t  nextReference = epoch + 500000;

    for( uint64_t i = 0; i < 8000; ++i )
    {
        const int64_t expected = epoch + 300000 + i * 125000;

        while( nextReference <= expected )
        {
            const int64_t frozen = clock.deadline();
            REQUIRE( clock.reference( reference++, nextReference, 500000, 8000, 300000, 300000, 15000, 10000, 0.05 ) );
            REQUIRE( clock.deadline() == frozen );
            nextReference += 500000;
        }

        scanSlot slot;
        REQUIRE( clock.deadline() == expected );
        REQUIRE( clock.take( expected, slot ) );
        REQUIRE( slot.m_reference.m_id == 1000 + i / 4 );
        REQUIRE( slot.m_slot == i % 4 );
        REQUIRE( slot.m_index == i );
    }

    REQUIRE( clock.m_epochs == 1 );
    REQUIRE( clock.m_skipped == 0 );
}

/** Fractional intervals are multiplied from an unrounded origin, not added as truncated nanoseconds.
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "AD7689 DDSPC long-term fractional clock drift", "[ad7689Ctrl][ddspc]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689CtrlDetail::continuousSchedule::offset;
    MagAOX::app::ad7689CtrlDetail::continuousSchedule::advance;
#endif
    // clang-format on
    using namespace MagAOX::app::ad7689CtrlDetail;
    continuousSchedule clock;
    const int64_t      epoch = 1000000000000LL;
    REQUIRE( clock.reference( 1, epoch, 500000, 6000, 300000, 300000, 56000, 20000, 0.05 ) );
    long double maximumError = 0;
    bool        valid        = true;

    for( uint64_t i = 0; i < 1000000; ++i )
    {
        if( i && i % 3 == 0 )
        {
            valid = clock.reference(
                        i / 3 + 1, epoch + i / 3 * 500000, 500000, 6000, 300000, 300000, 56000, 20000, 0.05 ) &&
                    valid;
        }

        const long double ideal = 300000 + static_cast<long double>( i ) * 500000 / 3;
        maximumError =
            std::max( maximumError, std::abs( static_cast<long double>( clock.deadline() - epoch ) - ideal ) );
        scanSlot slot;
        valid = clock.take( clock.deadline(), slot ) && valid;
    }

    REQUIRE( valid );
    REQUIRE( maximumError < 0.51 );
    REQUIRE( clock.m_skipped == 0 );
    REQUIRE( clock.m_epochs == 1 );
}

/** Delay adjustments slew only future intervals, bounded by both phase policy and ADC headroom.
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "AD7689 DDSPC runtime delay slews without replacing pending slots", "[ad7689Ctrl][ddspc]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689CtrlDetail::continuousSchedule::advance;
    MagAOX::app::ad7689CtrlDetail::continuousSchedule::reference;
#endif
    // clang-format on
    using namespace MagAOX::app::ad7689CtrlDetail;
    continuousSchedule clock;
    REQUIRE( clock.reference( 1, 1000000000, 500000, 8000, 300000, 300000, 56000, 20000, 0.05 ) );
    scanSlot slot;
    REQUIRE( clock.take( clock.deadline(), slot ) );
    const int64_t frozen = clock.deadline();
    REQUIRE( clock.reference( 2, 1000500000, 500000, 8000, 350000, 350000, 56000, 20000, 0.05 ) );
    REQUIRE( clock.deadline() == frozen );
    int64_t previous = clock.deadline();

    for( unsigned i = 0; i < 7; ++i )
    {
        REQUIRE( clock.take( previous, slot ) );
        const int64_t next = clock.deadline();
        REQUIRE( next - previous == 131250 );
        previous = next;
    }

    REQUIRE( clock.m_target - clock.m_applied == 6250 );
    REQUIRE( clock.m_skipped == 0 );
}

/** Missing/duplicate references and expired deadlines never create unauthorized or burst publications.
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "AD7689 DDSPC missed deadlines and reference recovery", "[ad7689Ctrl][ddspc]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689CtrlDetail::continuousSchedule::discardLate;
    MagAOX::app::ad7689CtrlDetail::continuousSchedule::cancel;
#endif
    // clang-format on
    using namespace MagAOX::app::ad7689CtrlDetail;
    continuousSchedule clock;
    REQUIRE( clock.reference( 1, 1000000000, 500000, 8000, 300000, 300000, 56000, 20000, 0.05 ) );
    REQUIRE_FALSE( clock.reference( 1, 1000000001, 500000, 8000, 300000, 300000, 56000, 20000, 0.05 ) );
    REQUIRE( clock.m_limit == 4 );
    scanSlot slot;
    REQUIRE( clock.take( 1000560000, slot ) );
    REQUIRE( slot.m_index == 2 );
    REQUIRE( clock.m_skipped == 2 );
    REQUIRE_FALSE( clock.take( 1000560001, slot ) );
    REQUIRE( clock.take( 1000675000, slot ) );
    REQUIRE_FALSE( clock.take( 1000800000, slot ) );
    REQUIRE( clock.end() == 1000800000 );
    REQUIRE( clock.reference( 3, 1001000000, 500000, 8000, 300000, 300000, 56000, 20000, 0.05 ) );
    REQUIRE( clock.m_epochs == 2 );
    REQUIRE( clock.deadline() == 1001300000 );
    REQUIRE( clock.reference( 4, 1002000000, 1000000, 8000, 550000, 550000, 56000, 20000, 0.05 ) );
    REQUIRE( clock.m_count == 8 );
    REQUIRE( clock.fps() == 8000 );
    REQUIRE( clock.deadline() == 1002550000 );
    REQUIRE( clock.m_skipped == 6 );
    clock.cancel();
    REQUIRE_FALSE( clock.pending() );
    REQUIRE( clock.m_skipped == 14 );
}

/** DDSPC settings, rounded rates and scan budgets reject nonfinite, noncausal or impossible timing.
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "AD7689 DDSPC invalid settings and physical admission", "[ad7689Ctrl][ddspc]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689CtrlDetail::ddspcSettings::valid;
    MagAOX::app::ad7689CtrlDetail::continuousSchedule::reference;
#endif
    // clang-format on
    using namespace MagAOX::app::ad7689CtrlDetail;
    ddspcSettings settings;
    REQUIRE( settings.valid() );
    REQUIRE( settings.delay( 500000 ) == 300000 );
    REQUIRE( settings.delay( 1000000 ) == 550000 );
    continuousSchedule clock;
    REQUIRE( clock.reference( 1, 1000000000, 1e9 / 1800, 8000, 300000, 300000, 56000, 20000, 0.05 ) );
    REQUIRE( clock.m_count == 4 );
    REQUIRE( clock.fps() == Approx( 7200 ) );
    REQUIRE_FALSE( clock.reference( 2, 1001000000, 500000, 8000, 300000, 300000, 110000, 20000, 0.05 ) );
    REQUIRE_FALSE( clock.m_active );
    REQUIRE_FALSE( clock.reference( 2, 1001000000, 500000, 8000, -1, -1, 56000, 20000, 0.05 ) );
    REQUIRE_FALSE( clock.reference( 2, 1001000000, 500000, 8000, 300000, 300000, 56000, 20000, 1 ) );

    for( double invalid :
         { -1.0, 1e10, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN() } )
    {
        settings.m_latency = invalid;
        REQUIRE_FALSE( settings.valid() );
        REQUIRE_FALSE( clock.reference( 2, 1001000000, 500000, 8000, invalid, invalid, 56000, 20000, 0.05 ) );
    }
}

} // namespace ad7689CtrlTest
} // namespace libXWCTest
