/** \file dmTiming_test.cpp
 * \brief Deterministic and concurrent tests of bounded DM timing collection.
 */
#include "../../../../tests/testXWC.hpp"
#include "../dmTiming.hpp"

#include <atomic>
#include <thread>

namespace libXWCTest
{
namespace dmTimingTests
{

using Collector = MagAOX::app::dev::dmTiming;
using namespace MagAOX::dmTiming;

/// \cond DM_TIMING_TEST_HELPERS
namespace
{

/// Capture a complete synthetic controller command with a qualified successful submission.
void command( Collector &collector,   /**< [in,out] Collector receiving the event. */
              int64_t    receipt,     /**< [in] Monotonic receipt timestamp. */
              int64_t    extra   = 0, /**< [in] Additional SDK and epilogue duration. */
              uint64_t   counter = 0 /**< [in] Observed stream counter. */ );

/// Capture an event without submitting a controller command.
void unsent( Collector &collector, /**< [in,out] Collector receiving the event. */
             int64_t    receipt,   /**< [in] Monotonic receipt timestamp. */
             Outcome    result /**< [in] Known callback outcome. */ );

/// Capture an external publication observation without an SDK submission.
void publication( Collector &collector,           /**< [in,out] Collector receiving the event. */
                  int64_t    observation,         /**< [in] Local monotonic receipt or synthetic-entry baseline. */
                  int64_t    publicationNs,       /**< [in] External integer-nanosecond image writetime. */
                  uint64_t   counter,             /**< [in] Observed stream counter. */
                  bool       counterValid = true, /**< [in] Whether the counter observation is available. */
                  bool       synthetic    = false /**< [in] Use callback entry without a semaphore receipt. */ );

void command( Collector &collector, int64_t receipt, int64_t extra, uint64_t counter )
{
    collector.beginEventAt( true, receipt );
    collector.inputCounter( counter );
    collector.markAt( Marker::entry, receipt + 10 );
    collector.markAt( Marker::actuatorStart, receipt + 20 );
    collector.markAt( Marker::actuatorEnd, receipt + 30 );
    collector.markAt( Marker::sdkStart, receipt + 40 );
    collector.markAt( Marker::sdkEnd, receipt + 100 + extra );
    collector.outcome( Outcome::success );
    collector.markAt( Marker::satStart, receipt + 110 + extra );
    collector.markAt( Marker::satEnd, receipt + 120 + extra );
    collector.markAt( Marker::deltaStart, receipt + 130 + extra );
    collector.markAt( Marker::deltaEnd, receipt + 150 + extra );
    collector.markAt( Marker::postStart, receipt + 160 + extra );
    collector.markAt( Marker::postEnd, receipt + 170 + extra );
    collector.markAt( Marker::finish, receipt + 190 + extra );
    collector.finishEvent();
}

void unsent( Collector &collector, int64_t receipt, Outcome result )
{
    collector.beginEventAt( true, receipt );
    collector.markAt( Marker::entry, receipt + 10 );
    collector.outcome( result );
    collector.markAt( Marker::finish, receipt + 20 );
    collector.finishEvent();
}

void publication( Collector &collector,
                  int64_t    observation,
                  int64_t    publicationNs,
                  uint64_t   counter,
                  bool       counterValid,
                  bool       synthetic )
{
    collector.beginEventAt( !synthetic, observation );
    collector.markAt( Marker::entry, observation + 10 );

    if( counterValid )
    {
        collector.inputCounter( counter );
    }

    collector.inputPublication( publicationNs );
    collector.outcome( Outcome::noSend );
    collector.markAt( Marker::finish, observation + 20 );
    collector.finishEvent();
}

} // namespace
/// \endcond

/** \defgroup dmTiming_tests DM timing collector tests
 * \ingroup app_dev_unit_tests
 */

/** \brief Preserve nanosecond differences and independently count phase populations.
 * \ingroup dmTiming_tests
 */
TEST_CASE( "DM timing endpoint statistics preserve integer precision", "[dev::dmTiming]" )
{
    Collector collector;
    collector.configure( 2, 20 );
    constexpr int64_t base = 9000000000000000LL;
    command( collector, base, 0, 10 );
    command( collector, base + 500000, 10, 11 );
    command( collector, base + 1000002, 20, 12 );
    collector.update( base + 1000300 );
    const auto &summary = collector.summary();
    REQUIRE( summary.fgValid );
    REQUIRE( summary.state == Availability::valid );
    const auto &receipt = summary.metrics[metricIndex( Metric::receiptInterval )];
    REQUIRE( receipt.count == 2 );
    CHECK( receipt.mean == Approx( 0.000500001 ).margin( 1e-16 ) );
    CHECK( receipt.stddev == Approx( std::sqrt( 2.0 ) * 1e-9 ).margin( 1e-16 ) );
    CHECK( receipt.min == Approx( 0.0005 ) );
    CHECK( receipt.max == Approx( 0.000500002 ) );
    CHECK( receipt.span == Approx( 0.001000002 ) );
    const auto &send = summary.metrics[metricIndex( Metric::sendInterval )];
    CHECK( send.mean == Approx( 0.000500011 ) );
    CHECK( send.stddev == Approx( std::sqrt( 2.0 ) * 1e-9 ).margin( 1e-16 ) );
    const auto &latency = summary.metrics[metricIndex( Metric::receiptSend )];
    CHECK( latency.count == 3 );
    CHECK( latency.mean == Approx( 110e-9 ) );
    CHECK( latency.stddev == Approx( 10e-9 ) );
    CHECK( summary.metrics[metricIndex( Metric::dispatch )].mean == Approx( 10e-9 ) );
    CHECK( summary.metrics[metricIndex( Metric::actuator )].mean == Approx( 10e-9 ) );
    CHECK( summary.metrics[metricIndex( Metric::sdk )].mean == Approx( 70e-9 ) );
    CHECK( summary.metrics[metricIndex( Metric::saturation )].mean == Approx( 10e-9 ) );
    CHECK( summary.metrics[metricIndex( Metric::delta )].mean == Approx( 20e-9 ) );
    CHECK( summary.metrics[metricIndex( Metric::post )].mean == Approx( 10e-9 ) );
    CHECK( summary.metrics[metricIndex( Metric::throughDelta )].mean == Approx( 150e-9 ) );
    CHECK( summary.metrics[metricIndex( Metric::callback )].mean == Approx( 190e-9 ) );
    CHECK( summary.metrics[metricIndex( Metric::receiptCallback )].mean == Approx( 200e-9 ) );
    CHECK( summary.total.receipts == 3 );
    CHECK( summary.total.submitted == 3 );
    CHECK( summary.total.invalid == 0 );
    CHECK( summary.delta.events == 3 );
    CHECK( summary.age == Approx( 88e-9 ) );
    collector.update( base + 1000300 );
    CHECK( collector.summary().delta.events == 0 );
    CHECK( collector.summary().metrics[metricIndex( Metric::receiptSend )].count == 3 );
    collector.update( -1 );
    CHECK( collector.summary().state == Availability::unavailable );
    CHECK_FALSE( collector.summary().fgValid );
    collector.update( base + 1000300 );
    CHECK( collector.summary().fgValid );

    // clang-format off
#ifdef XWCTEST_DOX_REF
    MagAOX::app::dev::dmTiming::markAt;
    MagAOX::app::dev::dmTiming::calculate;
#endif
    // clang-format on
}

/** \brief Require every contributing endpoint to lie in the retained trailing population.
 * \ingroup dmTiming_tests
 */
TEST_CASE( "DM timing windows warm up, cap, and expire without fresh commands", "[dev::dmTiming]" )
{
    Collector collector;
    collector.configure( 0.001, 8 );
    command( collector, 1000000 );
    collector.update( 1000300 );
    REQUIRE_FALSE( collector.summary().fgValid );
    CHECK( collector.summary().state == Availability::warmingUp );
    CHECK( collector.summary().metrics[metricIndex( Metric::receiptSend )].count == 1 );
    command( collector, 1500000 );
    command( collector, 2000000 );
    collector.update( 2000300 );
    CHECK( collector.summary().metrics[metricIndex( Metric::receiptInterval )].count == 1 );
    CHECK( collector.summary().metrics[metricIndex( Metric::receiptSend )].count == 2 );
    CHECK_FALSE( collector.summary().fgValid );
    collector.update( 4000000 );
    CHECK( collector.summary().state == Availability::stale );
    CHECK( collector.summary().age == -1 );
    CHECK( collector.summary().newestNs == -1 );

    for( const auto &metric : collector.summary().metrics )
    {
        CHECK( metric.count == 0 );
        CHECK_FALSE( metric.valid );
    }

    collector.configure( 2, 2 );
    command( collector, 1000000 );
    collector.update( 1000300 );
    command( collector, 1500000 );
    collector.update( 1500300 );
    command( collector, 2000000 );
    collector.update( 2000300 );
    CHECK( collector.summary().capacity == 2 );
    CHECK( collector.summary().metrics[metricIndex( Metric::receiptInterval )].count == 1 );
    CHECK( collector.summary().span == Approx( 0.000500190 ) );

    // clang-format off
#ifdef XWCTEST_DOX_REF
    MagAOX::app::dev::dmTiming::configure;
    MagAOX::app::dev::dmTiming::update;
#endif
    // clang-format on
}

/** \brief Keep real successful-send spacing across known failures and preserve pre-failure endpoints.
 * \ingroup dmTiming_tests
 */
TEST_CASE( "DM timing separates submission success from callback failure", "[dev::dmTiming]" )
{
    Collector collector;
    collector.configure( 2, 20 );
    command( collector, 1000000 );
    unsent( collector, 1500000, Outcome::failed );
    unsent( collector, 2000000, Outcome::noSend );
    command( collector, 2500000 );
    collector.beginEventAt( true, 3000000 );
    collector.markAt( Marker::entry, 3000010 );
    collector.markAt( Marker::sdkStart, 3000040 );
    collector.markAt( Marker::sdkEnd, 3000100 );
    collector.outcome( Outcome::success );
    collector.outcome( Outcome::failed ); // Delta/post failure after a successful SDK return.
    collector.markAt( Marker::finish, 3000190 );
    collector.finishEvent();
    unsent( collector, 3500000, Outcome::unknown );
    command( collector, 4000000 );
    collector.update( 4000300 );
    const auto &summary = collector.summary();
    CHECK( summary.total.receipts == 7 );
    CHECK( summary.total.submitted == 4 );
    CHECK( summary.total.failed == 2 );
    CHECK( summary.total.noSend == 1 );
    CHECK( summary.metrics[metricIndex( Metric::receiptSend )].count == 4 );
    const auto &send = summary.metrics[metricIndex( Metric::sendInterval )];
    CHECK( send.count == 2 );
    CHECK( send.min == Approx( 0.0005 ) );
    CHECK( send.max == Approx( 0.0015 ) );
    CHECK( summary.metrics[metricIndex( Metric::delta )].count == 3 );

    // clang-format off
#ifdef XWCTEST_DOX_REF
    MagAOX::app::dev::dmTiming::outcome;
    MagAOX::app::dev::dmTiming::calculate;
#endif
    // clang-format on
}

/** \brief Queue loss is bounded, counted, and breaks interval association across lost events.
 * \ingroup dmTiming_tests
 */
TEST_CASE( "DM timing handoff never overwrites unread events", "[dev::dmTiming]" )
{
    Collector collector;
    collector.configure( 2, 3 );

    for( int i = 0; i < 4; ++i )
    {
        command( collector, 1000000 + i * 500000, 0, 100 + i );
    }

    collector.update( 3000300 );
    CHECK( collector.summary().total.events == 3 );
    CHECK( collector.summary().total.dropped == 1 );
    CHECK( collector.summary().metrics[metricIndex( Metric::receiptInterval )].count == 2 );
    command( collector, 3000000, 0, 104 );
    collector.update( 3000300 );
    CHECK( collector.summary().total.events == 4 );
    CHECK( collector.summary().total.dropped == 1 );
    CHECK( collector.summary().delta.dropped == 0 );
    CHECK( collector.summary().metrics[metricIndex( Metric::receiptInterval )].count == 1 );
    CHECK( collector.summary().metrics[metricIndex( Metric::sendInterval )].count == 1 );
    CHECK( collector.summary().total.counterGaps == 0 );

    // clang-format off
#ifdef XWCTEST_DOX_REF
    MagAOX::app::dev::dmTiming::finishEvent;
    MagAOX::app::dev::dmTiming::countEvent;
#endif
    // clang-format on
}

/** \brief Synthetic reads, invalid markers, rejected receipts and attachment resets remain distinguishable.
 * \ingroup dmTiming_tests
 */
TEST_CASE( "DM timing lifecycle never reuses a stale wake or timestamp", "[dev::dmTiming]" )
{
    Collector collector;
    collector.configure( 2, 20 );
    collector.beginEventAt( false, -1 );
    collector.markAt( Marker::entry, 1000000 );
    collector.markAt( Marker::sdkEnd, 1000100 );
    collector.outcome( Outcome::success );
    collector.finishEvent();
    command( collector, 1500000, 0, 10 );
    command( collector, 2000000, 0, 10 );
    command( collector, 2500000, 0, 13 );
    collector.beginEventAt( true, 3000000 );
    collector.markAt( Marker::sdkStart, 3000040 );
    collector.markAt( Marker::sdkEnd, 3000030 ); // A backward endpoint is not a zero-duration observation.
    collector.outcome( Outcome::success );
    collector.finishEvent();
    collector.update( 3000300 );
    CHECK( collector.summary().total.synthetic == 1 );
    CHECK( collector.summary().total.receipts == 4 );
    CHECK( collector.summary().total.submitted == 5 );
    CHECK( collector.summary().total.duplicates == 1 );
    CHECK( collector.summary().total.counterGaps == 1 );
    CHECK( collector.summary().total.invalid == 1 );
    CHECK( collector.summary().metrics[metricIndex( Metric::receiptSend )].count == 3 );
    CHECK( collector.summary().metrics[metricIndex( Metric::sdk )].count == 3 );
    collector.beginEventAt( true, 3500000 );
    REQUIRE( collector.active() );
    collector.newSession();
    CHECK_FALSE( collector.active() );
    collector.update( 4000300 );
    CHECK( collector.summary().session == 2 );
    CHECK( collector.summary().total.rejected == 1 );
    CHECK( collector.summary().state == Availability::warmingUp );
    CHECK( collector.summary().metrics[metricIndex( Metric::receiptInterval )].count == 0 );
    command( collector, 4500000 );
    collector.update( 4500300 );
    CHECK( collector.summary().metrics[metricIndex( Metric::receiptSend )].count == 1 );
    CHECK( collector.summary().metrics[metricIndex( Metric::receiptInterval )].count == 0 );
    collector.configure( 0, 20 );
    collector.beginEvent( true );
    collector.mark( Marker::entry );
    collector.finishEvent();
    collector.update( 5000000 );
    CHECK_FALSE( collector.enabled() );
    CHECK_FALSE( collector.active() );
    CHECK( collector.summary().state == Availability::disabled );
    CHECK( collector.summary().total.events == 0 );

    // clang-format off
#ifdef XWCTEST_DOX_REF
    MagAOX::app::dev::dmTiming::newSession;
    MagAOX::app::dev::dmTiming::beginEventAt;
    MagAOX::app::dev::dmTiming::markAt;
#endif
    // clang-format on
}

/** \brief Monotonic capture must not alter errno used by the surrounding SDK error handler.
 * \ingroup dmTiming_tests
 */
TEST_CASE( "DM timing clock capture preserves errno", "[dev::dmTiming]" )
{
    Collector collector;
    collector.configure( 2, 10 );
    errno = EBUSY;
    collector.beginEvent( true );
    REQUIRE( errno == EBUSY );
    collector.mark( Marker::entry );
    REQUIRE( errno == EBUSY );
    collector.mark( Marker::sdkEnd );
    REQUIRE( errno == EBUSY );
    collector.outcome( Outcome::success );
    collector.finishEvent();
    CHECK( errno == EBUSY );
    collector.update();
    CHECK( collector.summary().total.invalid == 0 );

    // clang-format off
#ifdef XWCTEST_DOX_REF
    MagAOX::app::dev::dmTiming::monotonicNowNs;
    MagAOX::app::dev::dmTiming::mark;
#endif
    // clang-format on
}

/** \brief Concurrent producer/consumer wraps preserve payload ownership and count every loss.
 * \ingroup dmTiming_tests
 */
TEST_CASE( "DM timing SPSC handoff remains coherent during concurrent wraps", "[dev::dmTiming][concurrency]" )
{
    Collector collector;
    collector.configure( 2, 64 );
    constexpr uint64_t count = 100000;
    std::atomic<bool>  done{ false };
    std::thread        producer(
        [&]
        {
            for( uint64_t i = 0; i < count; ++i )
            {
                if( i == count / 2 )
                {
                    collector.newSession();
                }

                command( collector, 1000000 + static_cast<int64_t>( i ) * 500000, 0, i );
            }

            done.store( true, std::memory_order_release );
        } );
    const int64_t now = 1000000 + static_cast<int64_t>( count ) * 500000;

    while( !done.load( std::memory_order_acquire ) )
    {
        collector.update( now );
    }

    producer.join();
    collector.update( now );
    const auto &summary = collector.summary();
    CHECK( summary.total.events + summary.total.dropped == count );
    CHECK( summary.total.events == summary.total.receipts );
    CHECK( summary.total.events == summary.total.submitted );
    CHECK( summary.total.invalid == 0 );
    CHECK( summary.total.duplicates == 0 );
    CHECK( summary.total.counterGaps == 0 );
    CHECK( summary.queued == 0 );
    CHECK( summary.session == 2 );

    for( const auto &metric : summary.metrics )
    {
        if( metric.count )
        {
            CHECK( metric.min == Approx( metric.max ) );
        }
    }

    // clang-format off
#ifdef XWCTEST_DOX_REF
    MagAOX::app::dev::dmTiming::finishEvent;
    MagAOX::app::dev::dmTiming::update;
#endif
    // clang-format on
}

/** \brief Publication cadence preserves epoch precision while using local observations for coverage.
 * \ingroup dmTiming_tests
 */
TEST_CASE( "DM timing publication cadence is independent of receipt cadence", "[dev::dmTiming][publication]" )
{
    Collector collector;
    collector.configure( 2, 20 );
    constexpr int64_t epoch = 1792000000000000000LL;
    publication( collector, 1000000, epoch, 10 );
    publication( collector, 1100000, epoch + 500000, 11 );
    publication( collector, 3000000, epoch + 1000002, 12 );
    collector.update( 3100000 );

    const auto &summary = collector.summary();
    const auto &metric  = summary.metrics[metricIndex( Metric::publicationInterval )];
    REQUIRE( metric.count == 2 );
    CHECK( metric.valid );
    CHECK( metric.mean == Approx( 0.000500001 ).margin( 1e-16 ) );
    CHECK( metric.stddev == Approx( std::sqrt( 2.0 ) * 1e-9 ).margin( 1e-16 ) );
    CHECK( metric.min == Approx( 0.0005 ) );
    CHECK( metric.max == Approx( 0.000500002 ) );
    CHECK( metric.span == Approx( 0.002 ) );
    CHECK( summary.metrics[metricIndex( Metric::receiptInterval )].mean == Approx( 0.001 ) );
    CHECK_FALSE( summary.fgValid );
    CHECK( std::string( metricName( Metric::publicationInterval ) ) == "publicationInterval" );
    CHECK( metricIndex( Metric::publicationInterval ) == 12 );

    collector.configure( 2, 20 );
    publication( collector, 1000000, epoch, 10, true, true );
    publication( collector, 1500000, epoch + 500000, 11 );
    collector.update( 1500020 );

    const auto &synthetic = collector.summary().metrics[metricIndex( Metric::publicationInterval )];
    CHECK( synthetic.count == 1 );
    CHECK( synthetic.mean == Approx( 0.0005 ) );
    CHECK( synthetic.span == Approx( 0.000499990 ) );
    CHECK( collector.summary().total.synthetic == 1 );

    // clang-format off
#ifdef XWCTEST_DOX_REF
    MagAOX::app::dev::dmTiming::inputPublication;
    MagAOX::app::dev::dmTiming::calculate;
#endif
    // clang-format on
}

/** \brief Missing, repeated, backward and nonconsecutive observations cannot fabricate publication intervals.
 * \ingroup dmTiming_tests
 */
TEST_CASE( "DM timing publication pairing requires consecutive valid observations", "[dev::dmTiming][publication]" )
{
    Collector collector;
    collector.configure( 2, 30 );
    constexpr int64_t epoch      = 1792000000000000000LL;
    const uint64_t    counters[] = { 10, 11, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25 };
    const int64_t     stamps[]   = { epoch,
                                     epoch + 500000,
                                     epoch + 500000,
                                     epoch + 1000000,
                                     epoch + 2000000,
                                     epoch + 2500000,
                                     0,
                                     epoch + 3500000,
                                     epoch + 4000000,
                                     epoch + 3900000,
                                     epoch + 4400000,
                                     epoch + 4400000,
                                     epoch + 4900000,
                                     epoch + 5400000,
                                     epoch + 5900000,
                                     epoch + 6400000 };

    for( size_t i = 0; i < sizeof( counters ) / sizeof( counters[0] ); ++i )
    {
        publication( collector, 1000000 + static_cast<int64_t>( i ) * 500000, stamps[i], counters[i], i != 13 );
    }

    collector.update( 9000000 );
    const auto &metric = collector.summary().metrics[metricIndex( Metric::publicationInterval )];
    REQUIRE( metric.count == 7 );
    CHECK( metric.mean == Approx( 0.0005 ) );
    CHECK( metric.stddev == 0 );
    CHECK( collector.summary().total.duplicates == 1 );
    CHECK( collector.summary().total.counterGaps == 1 );
    CHECK( collector.summary().total.invalid == 0 );

    collector.configure( 2, 10 );
    publication( collector, 1000000, epoch, 10 );
    collector.beginEventAt( true, 1500000 );
    collector.inputCounter( 11 );
    collector.inputPublication( epoch + 500000 );
    collector.outcome( Outcome::failed );
    collector.finishEvent();
    publication( collector, 2000000, epoch + 1000000, 12 );
    collector.update( 2000020 );
    CHECK( collector.summary().metrics[metricIndex( Metric::publicationInterval )].count == 2 );
    CHECK( collector.summary().metrics[metricIndex( Metric::sendInterval )].count == 0 );

    collector.configure( 2, 10 );
    collector.beginEventAt( false, -1 );
    collector.inputCounter( 10 );
    collector.inputPublication( epoch );
    collector.finishEvent(); // Missing local observation cannot contribute coverage.
    publication( collector, 1500000, epoch + 500000, 11 );
    publication( collector, 2000000, epoch + 1000000, 12 );
    collector.update( 2000020 );
    CHECK( collector.summary().metrics[metricIndex( Metric::publicationInterval )].count == 1 );

    // clang-format off
#ifdef XWCTEST_DOX_REF
    MagAOX::app::dev::dmTiming::inputCounter;
    MagAOX::app::dev::dmTiming::inputPublication;
    MagAOX::app::dev::dmTiming::calculate;
#endif
    // clang-format on
}

/** \brief Publication intervals obey local window/cap retention and never cross resets or lost events.
 * \ingroup dmTiming_tests
 */
TEST_CASE( "DM timing publication windows respect retention and loss boundaries", "[dev::dmTiming][publication]" )
{
    Collector         collector;
    constexpr int64_t epoch = 1792000000000000000LL;

    SECTION( "Both local observation anchors must be in the elapsed-time window" )
    {
        collector.configure( 0.001, 8 );
        publication( collector, 1000000, epoch, 10 );
        publication( collector, 1500000, epoch + 500000, 11 );
        publication( collector, 2000000, epoch + 1000000, 12 );
        collector.update( 2000020 );

        const auto &metric = collector.summary().metrics[metricIndex( Metric::publicationInterval )];
        CHECK( metric.count == 1 );
        CHECK( metric.span == Approx( 0.0005 ) );
        CHECK_FALSE( metric.valid );
        collector.update( 4000000 );
        CHECK( collector.summary().metrics[metricIndex( Metric::publicationInterval )].count == 0 );
    }

    SECTION( "Capacity cannot borrow a predecessor outside the retained population" )
    {
        collector.configure( 2, 2 );

        for( int i = 0; i < 3; ++i )
        {
            publication( collector, 1000000 + i * 500000, epoch + i * 500000, 10 + i );
            collector.update( 1000020 + i * 500000 );
        }

        CHECK( collector.summary().metrics[metricIndex( Metric::publicationInterval )].count == 1 );
        CHECK( collector.summary().metrics[metricIndex( Metric::publicationInterval )].span == Approx( 0.0005 ) );
    }

    SECTION( "A lost timing event breaks pairing even when the next observed counter is consecutive" )
    {
        collector.configure( 2, 3 );

        for( int i = 0; i < 4; ++i )
        {
            publication( collector, 1000000 + i * 500000, epoch + i * 500000, 10 + i );
        }

        collector.update( 2500020 );
        CHECK( collector.summary().total.dropped == 1 );
        CHECK( collector.summary().metrics[metricIndex( Metric::publicationInterval )].count == 2 );
        publication( collector, 3000000, epoch + 1500000, 13 );
        collector.update( 3000020 );
        CHECK( collector.summary().metrics[metricIndex( Metric::publicationInterval )].count == 1 );
    }

    SECTION( "A fresh monitor session cannot pair with the previous attachment" )
    {
        collector.configure( 2, 8 );
        publication( collector, 1000000, epoch, 10 );
        collector.update( 1000020 );
        collector.newSession();
        publication( collector, 1500000, epoch + 500000, 11 );
        collector.update( 1500020 );
        CHECK( collector.summary().metrics[metricIndex( Metric::publicationInterval )].count == 0 );
        publication( collector, 2000000, epoch + 1000000, 12 );
        collector.update( 2000020 );
        CHECK( collector.summary().metrics[metricIndex( Metric::publicationInterval )].count == 1 );
    }

    SECTION( "A wrapped input counter starts a new publication association" )
    {
        collector.configure( 2, 8 );
        publication( collector, 1000000, epoch, std::numeric_limits<uint64_t>::max() );
        publication( collector, 1500000, epoch + 500000, 0 );
        collector.update( 1500020 );
        CHECK( collector.summary().metrics[metricIndex( Metric::publicationInterval )].count == 0 );
        publication( collector, 2000000, epoch + 1000000, 1 );
        collector.update( 2000020 );
        CHECK( collector.summary().metrics[metricIndex( Metric::publicationInterval )].count == 1 );
    }

    // clang-format off
#ifdef XWCTEST_DOX_REF
    MagAOX::app::dev::dmTiming::newSession;
    MagAOX::app::dev::dmTiming::finishEvent;
    MagAOX::app::dev::dmTiming::calculate;
#endif
    // clang-format on
}

} // namespace dmTimingTests
} // namespace libXWCTest
