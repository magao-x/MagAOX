/** \file wfsClock_test.cpp
 * \brief Catch2 tests for the rolling ten-second WFS clock fit.
 *
 * Ported from ad7689Ctrl's scheduler tests when the clock moved to libMagAOX.
 */

#include "../../../tests/testXWC.hpp"

#include "../wfsClock.hpp"

#include <cmath>

namespace libXWCTest
{
namespace wfsClockTest
{

using MagAOX::app::wfsScan::rollingWfsClock;

/** Rolling histories cover ten seconds at different source rates, rather than a fixed event count.
 * \ingroup app_unit_test
 */
TEST_CASE( "wfsClock history follows elapsed time", "[wfsClock]" )
{
    // clang-format off
#ifdef WFSCLOCK_TEST_DOXYGEN_REF
    MagAOX::app::wfsScan::rollingWfsClock::observe;
    MagAOX::app::wfsScan::rollingWfsClock::prune;
#endif
    // clang-format on

    for( int64_t period : { 500000LL, 1000000LL, 2000000LL } )
    {
        rollingWfsClock clock;
        const uint64_t  frames = rollingWfsClock::c_window / period;

        for( uint64_t i = 0; i <= 3 * frames; ++i )
        {
            clock.observe( i, 1000000000000LL + i * period );
        }

        REQUIRE( clock.valid() );
        REQUIRE( clock.m_count == frames + 1 );
        REQUIRE( clock.m_history.size() == frames + 1 );
        REQUIRE( clock.span() == 10 );
        REQUIRE( clock.m_period == Approx( period ).margin( 1e-6 ) );
        REQUIRE( clock.m_jitter < 0.01 );
        REQUIRE( clock.m_resets == 0 );
    }
}

/** Independent bounded semaphore noise averages out of the fitted period and phase.
 * \ingroup app_unit_test
 */
TEST_CASE( "wfsClock rejects ten microsecond semaphore jitter", "[wfsClock]" )
{
    // clang-format off
#ifdef WFSCLOCK_TEST_DOXYGEN_REF
    MagAOX::app::wfsScan::rollingWfsClock::reference;
#endif
    // clang-format on
    rollingWfsClock fit;
    uint32_t        random = 42;
    const auto      jitter = [&]() -> int64_t
    {
        random = random * 1664525U + 1013904223U;
        return static_cast<int64_t>( random % 20001 ) - 10000;
    };
    const int64_t epoch = 9000000000000000LL;

    for( uint64_t i = 0; i <= 20000; ++i )
    {
        fit.observe( i, epoch + i * 500000 + jitter() );
    }

    REQUIRE( fit.valid() );
    REQUIRE( fit.m_period == Approx( 500000 ).margin( 0.1 ) );
    REQUIRE( std::abs( fit.reference( 20000 ) - epoch - 10000000000LL ) < 500 );
    REQUIRE( fit.m_jitter > 5000 );
    REQUIRE( fit.m_jitter < 6500 );

    bool valid = true;

    for( uint64_t i = 20001; i <= 40000; ++i )
    {
        valid = fit.observe( i, epoch + i * 500000 + jitter() ) && valid;
    }

    REQUIRE( valid );
    REQUIRE( fit.m_resets == 0 );
    REQUIRE( fit.m_rejected == 0 );
    REQUIRE( fit.span() > 9.999 );
    REQUIRE( fit.m_baselineMature );
    REQUIRE( std::abs( fit.m_drift ) < 5000 );
}

/** Centered timestamps and source indices retain sub-nanosecond regression accuracy over long uptime.
 * \ingroup app_unit_test
 */
TEST_CASE( "wfsClock fit handles fractional periods and large clocks", "[wfsClock]" )
{
    // clang-format off
#ifdef WFSCLOCK_TEST_DOXYGEN_REF
    MagAOX::app::wfsScan::rollingWfsClock::moments;
    MagAOX::app::wfsScan::rollingWfsClock::fit;
#endif
    // clang-format on
    rollingWfsClock   fit;
    const int64_t     epoch    = 9000000000000000LL;
    const uint64_t    originId = ( 1ULL << 62 );
    const long double period   = 500000.375L;

    for( uint64_t i = 0; i <= 100000; ++i )
    {
        fit.observe( originId + i, epoch + std::llround( i * period ) );
    }

    REQUIRE( fit.valid() );
    REQUIRE( fit.m_period == Approx( static_cast<double>( period ) ).margin( 1e-5 ) );
    REQUIRE( std::abs( fit.reference( originId + 100000 ) - epoch - std::llround( 100000 * period ) ) <= 1 );
    REQUIRE( fit.m_jitter < 0.5 );
    REQUIRE( fit.span() <= 10 );
    REQUIRE( fit.span() > 9.999 );
    REQUIRE( fit.m_resets == 0 );
}

/** The rolling fit follows slow WFS period drift without restarting.
 * \ingroup app_unit_test
 */
TEST_CASE( "wfsClock follows slow WFS drift continuously", "[wfsClock]" )
{
    // clang-format off
#ifdef WFSCLOCK_TEST_DOXYGEN_REF
    MagAOX::app::wfsScan::rollingWfsClock::observe;
#endif
    // clang-format on
    rollingWfsClock fit;
    const int64_t   epoch = 1000000000000LL;

    for( uint64_t i = 0; i <= 60000; ++i )
    {
        const long double drift = static_cast<long double>( i ) * i / 2400; // 100 ppm period drift over 30 s.
        fit.observe( i, epoch + i * 500000 + std::llround( drift ) );
    }

    REQUIRE( fit.valid() );
    REQUIRE( fit.m_period > 500040 );
    REQUIRE( fit.m_period < 500050 );
    REQUIRE( fit.m_drift > 900000 );
    REQUIRE( fit.m_drift < 1100000 );
    REQUIRE( fit.m_baselineMature );
    REQUIRE( fit.m_resets == 0 );
}

/** Reject individual outliers/duplicates, normalize real ID gaps, then confirm a persistent new cadence promptly.
 * \ingroup app_unit_test
 */
TEST_CASE( "wfsClock robust events and persistent cadence change", "[wfsClock]" )
{
    // clang-format off
#ifdef WFSCLOCK_TEST_DOXYGEN_REF
    MagAOX::app::wfsScan::rollingWfsClock::observe;
    MagAOX::app::wfsScan::rollingWfsClock::reset;
#endif
    // clang-format on
    rollingWfsClock fit;
    fit.m_minEvents     = 8;
    const int64_t epoch = 1000000000000LL;

    for( uint64_t i = 0; i < 40; ++i )
    {
        fit.observe( i, epoch + i * 500000 );
    }

    REQUIRE_FALSE( fit.observe( 39, epoch + 39 * 500000 + 10000 ) );
    REQUIRE( fit.m_count == 40 );
    REQUIRE( fit.observe( 42, epoch + 42 * 500000 ) );
    REQUIRE( fit.m_period == 500000 );
    REQUIRE_FALSE( fit.observe( 43, epoch + 43 * 500000 + 200000 ) );
    REQUIRE( fit.m_count == 41 );
    REQUIRE( fit.valid() );
    REQUIRE( fit.observe( 44, epoch + 44 * 500000 ) );
    REQUIRE( fit.m_period == 500000 );
    REQUIRE( fit.m_jitter < 0.01 );
    REQUIRE( fit.m_rejected == 1 );
    REQUIRE( fit.m_resets == 0 );
    const int64_t change = epoch + 44 * 500000;

    for( uint64_t i = 1; i <= 8; ++i )
    {
        const bool ready = fit.observe( 44 + i, change + i * 1000000 );
        REQUIRE( ready == ( i == 8 ) );

        if( i >= 3 && i < 8 )
        {
            REQUIRE_FALSE( fit.valid() );
        }
    }

    REQUIRE( fit.valid() );
    REQUIRE( fit.m_resets == 1 );
    REQUIRE( fit.m_count == 8 );
    REQUIRE( fit.m_period == 1000000 );
    REQUIRE( fit.m_history.front().m_id == 45 );
    REQUIRE( fit.reference( 52 ) == change + 8000000 );
    REQUIRE_FALSE( fit.observe( 1, change + 9000000 ) );
    REQUIRE_FALSE( fit.valid() );
    REQUIRE( fit.m_resets == 2 );

    for( uint64_t i = 2; i <= 8; ++i )
    {
        fit.observe( i, change + 8000000 + i * 1000000 );
    }

    REQUIRE( fit.valid() );
    REQUIRE_FALSE( fit.observe( 1ULL << 63, change + 17000000 ) );
    REQUIRE( fit.m_period == 1000000 );
    REQUIRE( fit.valid() ); // A malformed counter jump is an outlier, not a valid phase update.
}

/** The interval-consistency gate allows the full difference of successive +/-10-us arrival errors.
 * \ingroup app_unit_test
 */
TEST_CASE( "wfsClock startup tolerates jitter at the admitted rate ceiling", "[wfsClock]" )
{
    // clang-format off
#ifdef WFSCLOCK_TEST_DOXYGEN_REF
    MagAOX::app::wfsScan::rollingWfsClock::observe;
#endif
    // clang-format on
    rollingWfsClock fit;
    bool            qualified = true;

    for( uint64_t i = 0; i < 1000; ++i )
    {
        const int64_t error = i % 2 ? 10000 : -10000;
        const bool    ready = fit.observe( i, 1000000000000LL + i * 100000 + error );
        qualified           = qualified && ready == ( i >= 15 );
    }

    REQUIRE( qualified );
    REQUIRE( fit.valid() );
    REQUIRE( fit.m_period == Approx( 100000 ).margin( 0.1 ) );
    REQUIRE( fit.m_jitter == Approx( 10000 ).margin( 1 ) );
    REQUIRE( fit.m_rejected == 0 );
    REQUIRE( fit.m_resets == 0 );
}

} // namespace wfsClockTest
} // namespace libXWCTest
