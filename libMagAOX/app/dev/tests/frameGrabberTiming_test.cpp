/** \file frameGrabberTiming_test.cpp
 * \brief Paired timestamp, stage attribution, control overlap and concurrent handoff qualification tests.
 */
#include <atomic>
#include <thread>

#include "../../../../tests/testXWC.hpp"
#include "../frameGrabberTiming.hpp"

namespace libXWCTest
{
namespace frameGrabberTimingTest
{
using Collector = MagAOX::app::dev::frameGrabberTiming;
using namespace MagAOX::fgTiming;

/// \cond FG_TIMING_TEST_HELPERS
namespace
{
/// Convert deterministic external nanoseconds to a normalized timestamp.
timespec stamp( int64_t ns /**< [in] External nanoseconds. */ )
{
    return { ns / 1000000000, ns % 1000000000 };
}

/// Publish one complete synthetic acquisition with independently controlled conversion and bracket delays.
void frame( Collector &collector /**< [in,out] Collector under test. */,
            int64_t    returned /**< [in] SDK-return endpoint, ns. */,
            uint64_t   counter /**< [in] Camera and output counter. */,
            int64_t    extra   = 0 /**< [in] Additional conversion duration, ns. */,
            int64_t    bracket = 100 /**< [in] Realtime clock-call bracket, ns. */,
            int64_t    post    = 100 /**< [in] Main post duration, ns. */ )
{
    constexpr int64_t epoch = 1700000000000000000LL;
    collector.beginAt( returned - 490000 );
    collector.markAt( Marker::waitEnter, returned - 489999 );
    collector.markAt( Marker::waitReturn, returned );
    collector.markAt( Marker::rearmReturn, returned + 100 );
    collector.markAt( Marker::validated, returned + 200 );
    collector.markAt( Marker::loadDone, returned + 300 + extra );
    collector.markAt( Marker::writeStampDone, returned + 300 + extra + bracket );
    collector.frame( counter );
    collector.published( stamp( epoch + returned ), stamp( epoch + returned + 300 + extra + bracket ), counter, 77 );
    collector.markAt( Marker::mainPostDone, returned + 300 + extra + bracket + post );
    collector.markAt( Marker::syncPostDone, returned + 500 + extra + bracket + post );
    collector.markAt( Marker::loopTailDone, returned + 700 + extra + bracket + post );
    collector.finish();
}
} // namespace
/// \endcond

/// Keep legacy paired meanings while optional stages generate no endpoints or control records.
/** \ingroup frameGrabberTiming_unit_test
 * \see MagAOX::app::dev::frameGrabberTiming
 */
TEST_CASE( "Paired framegrabber timing remains independent of optional stage clocks", "[frameGrabberTiming]" )
{
    Collector collector;
    collector.configure( 2, 8 );
    REQUIRE( collector.enabled() );
    REQUIRE_FALSE( collector.stagesEnabled() );
    constexpr int64_t epoch = 1700000000000000000LL;

    for( uint64_t index = 0; index < 4; ++index )
    {
        collector.beginAt( -1 );
        collector.markAt( Marker::waitReturn, -1 );
        collector.published( stamp( epoch + index * 500000 ), stamp( epoch + index * 500000 + 50000 ), index, 22 );
        collector.finish();
    }

    collector.control( -1, -1, false );
    collector.update( 1000000000, epoch + 1600000 );
    const auto &summary = collector.summary();
    REQUIRE( summary.fgValid );
    REQUIRE( summary.metrics[metricIndex( Metric::acquisitionInterval )].mean == Approx( 500e-6 ) );
    REQUIRE( summary.metrics[metricIndex( Metric::publicationInterval )].mean == Approx( 500e-6 ) );
    REQUIRE( summary.metrics[metricIndex( Metric::acquisitionToWrite )].mean == Approx( 50e-6 ) );
    REQUIRE( summary.metrics[metricIndex( Metric::wait )].count == 0 );
    REQUIRE( summary.total.invalid == 0 );
    REQUIRE( summary.total.controls == 0 );
    REQUIRE( summary.age == Approx( 50e-6 ) );
    REQUIRE( summary.lastCounter == 3 );
    REQUIRE( summary.streamInode == 22 );
    collector.update( 4000000000, epoch + 3001600000 );
    REQUIRE( collector.summary().state == Availability::stale );
    REQUIRE_FALSE( collector.summary().fgValid );
}

/// Attribute signed per-frame segment changes to exactly matched source and target intervals.
/** \ingroup frameGrabberTiming_unit_test
 * \see MagAOX::app::dev::frameGrabberTiming
 */
TEST_CASE( "Stage timing decomposes matched publication excursions and previous tails", "[frameGrabberTiming]" )
{
    Collector collector;
    collector.configure( 2, 16, true, 20e-6, 1e-6 );
    collector.context( 62, 1, 1 );
    frame( collector, 1000000000, 1 );
    frame( collector, 1000500000, 2 );
    frame( collector, 1001030000, 3, 30000 );
    frame( collector, 1001530000, 4 );
    frame( collector, 1002030000, 5 );
    collector.update( 1002100000, 1700000001002100000LL );
    const auto &summary = collector.summary();
    REQUIRE( summary.intervals == 4 );
    REQUIRE( summary.qualified == 4 );
    REQUIRE( summary.ambiguous == 0 );
    REQUIRE( summary.late == 1 );
    REQUIRE( summary.early == 1 );
    REQUIRE( summary.median == Approx( 500e-6 ) );
    REQUIRE( summary.lateWaitReturnInterval.mean == Approx( 530e-6 ) );
    REQUIRE( summary.lateWriteEndpointInterval.mean == Approx( 560e-6 ) );
    REQUIRE( summary.lateChange[metricIndex( Metric::waitToWrite )].mean == Approx( 30e-6 ) );
    REQUIRE( summary.lateWaitReturnInterval.mean + summary.lateChange[metricIndex( Metric::waitToWrite )].mean ==
             Approx( summary.lateWriteEndpointInterval.mean ) );
    REQUIRE( summary.earlyWaitReturnInterval.mean == Approx( 500e-6 ) );
    REQUIRE( summary.earlyWriteEndpointInterval.mean == Approx( 470e-6 ) );
    REQUIRE( summary.earlyChange[metricIndex( Metric::conversion )].mean == Approx( -30e-6 ) );
    REQUIRE( summary.latePreviousTail.count == 1 );
    REQUIRE( summary.mode == 62 );
    REQUIRE( summary.binX == 1 );
    REQUIRE( summary.total.invalid == 0 );
}

/// Retain wide-bracket outliers in raw timing while refusing precise stage attribution.
/** \ingroup frameGrabberTiming_unit_test
 * \see MagAOX::app::dev::frameGrabberTiming
 */
TEST_CASE( "Wide or undecided clock brackets remain raw observations with ambiguous attribution",
           "[frameGrabberTiming]" )
{
    Collector collector;
    collector.configure( 2, 8, true, 20e-6, 1e-6 );
    frame( collector, 1000000000, 1 );
    frame( collector, 1000500000, 2, 0, 80000 );
    frame( collector, 1001000000, 3 );
    collector.update( 1001100000 );
    REQUIRE( collector.summary().intervals == 2 );
    REQUIRE( collector.summary().qualified == 0 );
    REQUIRE( collector.summary().ambiguous == 2 );
    REQUIRE( collector.summary().total.bracketWide == 1 );
    REQUIRE( collector.summary().metrics[metricIndex( Metric::writeEndpointInterval )].max == Approx( 579900e-9 ) );
    collector.configure( 2, 8, true );
    frame( collector, 1000000000, 1 );
    frame( collector, 1000500000, 2 );
    collector.update( 1000600000 );
    REQUIRE( collector.summary().intervals == 1 );
    REQUIRE( collector.summary().ambiguous == 1 );
    REQUIRE( collector.summary().qualified == 0 );
}

/// Qualify queue losses, repeated/nonunit identities, failed captures and explicit epochs independently.
/** \ingroup frameGrabberTiming_unit_test
 * \see MagAOX::app::dev::frameGrabberTiming
 */
TEST_CASE( "Frame timing loss and epoch accounting never invents consecutive frames", "[frameGrabberTiming]" )
{
    Collector collector;
    collector.configure( 2, 2, true, 20e-6, 1e-6 );
    frame( collector, 1000000000, 1 );
    frame( collector, 1000500000, 2 );
    frame( collector, 1001000000, 3 );
    collector.update( 1001100000 );
    REQUIRE( collector.summary().total.events == 2 );
    REQUIRE( collector.summary().total.dropped == 1 );
    frame( collector, 1001500000, 4 );
    collector.update( 1001600000 );
    REQUIRE( collector.summary().intervals == 0 );
    REQUIRE( collector.summary().total.counterGaps == 0 );
    frame( collector, 1002000000, 9 );
    collector.update( 1002100000 );
    REQUIRE( collector.summary().total.counterGaps == 1 );
    REQUIRE( collector.summary().total.frameGaps == 1 );
    frame( collector, 1002500000, 9 );
    collector.update( 1002600000 );
    REQUIRE( collector.summary().total.duplicates == 1 );
    collector.beginAt( 1003000000 );
    collector.markAt( Marker::waitReturn, 1002999999 );
    collector.finish( Outcome::failed );
    collector.update( 1003100000 );
    REQUIRE( collector.summary().total.failed == 1 );
    REQUIRE( collector.summary().total.invalid == 1 );
    collector.newSession();
    collector.update( 1003200000 );
    REQUIRE( collector.summary().session == 2 );
    REQUIRE( collector.summary().intervals == 0 );
    REQUIRE( collector.summary().state == Availability::warmingUp );
}

/// Compute completed serial overlap independently of frame success and count invalid/lost control events.
/** \ingroup frameGrabberTiming_unit_test
 * \see MagAOX::app::dev::frameGrabberTiming
 */
TEST_CASE( "Serial controls retain their own quality and bounded overlapping frame context", "[frameGrabberTiming]" )
{
    Collector collector;
    collector.configure( 2, 3, true, 20e-6, 1e-6 );
    collector.control( 999999000, 1000000100, true );
    collector.control( 1000000050, 1000000300, false );
    collector.control( 5, 4, true );
    collector.control( 6, 7, true );
    frame( collector, 1000000000, 1 );
    collector.update( 1000100000 );
    const auto &summary = collector.summary();
    REQUIRE( summary.total.controls == 3 );
    REQUIRE( summary.total.controlFailed == 1 );
    REQUIRE( summary.total.controlInvalid == 1 );
    REQUIRE( summary.total.controlDropped == 1 );
    REQUIRE( summary.serialDuration.count == 2 );
    REQUIRE( summary.serialOverlaps == 1 );
    REQUIRE( summary.metrics[metricIndex( Metric::serialOverlap )].mean == Approx( 1300e-9 ) );
    collector.update( 1000200000 );
    REQUIRE( collector.summary().delta.controls == 0 );
}
/// Keep failed attempts visible while never treating a missing realtime stamp as epoch-zero camera data.
/** \ingroup frameGrabberTiming_unit_test
 * \see MagAOX::app::dev::frameGrabberTiming
 */
TEST_CASE( "Paired timing separates intentional missing endpoints from failed publication clocks",
           "[frameGrabberTiming]" )
{
    Collector collector;
    collector.configure( 2, 8 );
    collector.beginAt( -1 );
    collector.finish( Outcome::failed );
    collector.update( 1000000000, 1700000000000000000LL );
    REQUIRE( collector.summary().total.failed == 1 );
    REQUIRE( collector.summary().total.invalid == 0 );
    collector.beginAt( -1 );
    collector.published( stamp( 1700000000000000000LL ), {}, 1, 77 );
    collector.finish();
    collector.update( 1000100000, 1700000000000100000LL );
    REQUIRE( collector.summary().total.invalid == 1 );
    REQUIRE( collector.summary().metrics[metricIndex( Metric::acquisitionToWrite )].count == 0 );
    collector.beginAt( -1 );
    collector.published( stamp( 1700000000000500000LL ), stamp( 1700000000000550000LL ), 2, 77 );
    collector.finish();
    collector.update( 1001000000, 1700000000001000000LL );
    REQUIRE( collector.summary().metrics[metricIndex( Metric::publicationInterval )].count == 0 );
    REQUIRE( collector.summary().metrics[metricIndex( Metric::acquisitionToWrite )].mean == Approx( 50e-6 ) );
}

/// Preserve actual delivery intervals across discarded attempts and retain an output despite a later hook failure.
/** \ingroup frameGrabberTiming_unit_test
 * \see MagAOX::app::dev::frameGrabberTiming
 */
TEST_CASE( "Legacy paired timing preserves delivery gaps independently of attempt outcomes", "[frameGrabberTiming]" )
{
    Collector collector;
    collector.configure( 2, 16, true, 20e-6, 1e-6 );
    frame( collector, 1000000000, 1 );
    collector.beginAt( 1000300000 );
    collector.frame( 2 );
    collector.finish( Outcome::rejected );
    collector.beginAt( 1000500000 );
    collector.frame( 3 );
    collector.published( stamp( 1700000001001000000LL ), stamp( 1700000001001000400LL ), 2, 77 );
    collector.finish();
    collector.update( 1001100000, 1700000001001100000LL );
    REQUIRE( collector.summary().metrics[metricIndex( Metric::publicationInterval )].count == 1 );
    REQUIRE( collector.summary().metrics[metricIndex( Metric::publicationInterval )].mean == Approx( 1e-3 ) );
    REQUIRE( collector.summary().metrics[metricIndex( Metric::acquisitionInterval )].mean == Approx( 1e-3 ) );
    REQUIRE( collector.summary().intervals == 0 );
    REQUIRE( collector.summary().total.rejected == 1 );
    collector.beginAt( 1001200000 );
    collector.frame( 4 );
    collector.published( stamp( 1700000001001500000LL ), stamp( 1700000001001500400LL ), 3, 77 );
    collector.finish( Outcome::failed );
    collector.update( 1001600000, 1700000001001600000LL );
    REQUIRE( collector.summary().total.published == 3 );
    REQUIRE( collector.summary().total.failed == 1 );
    REQUIRE( collector.summary().metrics[metricIndex( Metric::publicationInterval )].count == 2 );
    REQUIRE( collector.summary().metrics[metricIndex( Metric::acquisitionToWrite )].count == 3 );
}

/// Retain events that race a captured evaluation time without inventing timestamp failures or dropping data.
/** \ingroup frameGrabberTiming_unit_test
 * \see MagAOX::app::dev::frameGrabberTiming
 */
TEST_CASE( "Newly arrived frame and control records survive a captured consumer-time boundary", "[frameGrabberTiming]" )
{
    Collector collector;
    collector.configure( 2, 8 );
    collector.beginAt( -1 );
    collector.published( stamp( 1700000000000000000LL ), stamp( 1700000000000050000LL ), 1, 77 );
    collector.finish();
    collector.update( 1000000000, 1700000000000040000LL );
    REQUIRE( collector.summary().total.invalid == 0 );
    REQUIRE( collector.summary().metrics[metricIndex( Metric::acquisitionToWrite )].count == 0 );
    collector.update( 1000020000, 1700000000000060000LL );
    REQUIRE( collector.summary().metrics[metricIndex( Metric::acquisitionToWrite )].count == 1 );
    collector.configure( 2, 8, true, 20e-6, 1e-6 );
    frame( collector, 1000000000, 1 );
    collector.control( 999999000, 1000000100, true );
    collector.update( 999999999 );
    REQUIRE( collector.summary().metrics[metricIndex( Metric::wait )].count == 0 );
    REQUIRE( collector.summary().serialDuration.count == 0 );
    collector.update( 1000010000 );
    REQUIRE( collector.summary().metrics[metricIndex( Metric::wait )].count == 1 );
    REQUIRE( collector.summary().serialDuration.count == 1 );
}

/// Keep acquisition and completed-control payloads bounded and fully accounted while the ordinary reader advances
/// epochs.
/** \ingroup frameGrabberTiming_unit_test
 * \see MagAOX::app::dev::frameGrabberTiming::beginAt
 * \see MagAOX::app::dev::frameGrabberTiming::finish
 * \see MagAOX::app::dev::frameGrabberTiming::control
 * \see MagAOX::app::dev::frameGrabberTiming::update
 * \see MagAOX::app::dev::frameGrabberTiming::newSession
 * \see MagAOX::app::dev::frameGrabberTiming::configure
 */
TEST_CASE( "Concurrent frame and control handoffs preserve bounded history and loss accounting",
           "[frameGrabberTiming][concurrency]" )
{
    constexpr uint64_t attempts = 10000, controls = 3000, batchSize = 500, epochSize = 997;
    constexpr size_t   capacity = 32;
    constexpr int64_t  epoch    = 1700000000000000000LL;
    Collector          collector;
    collector.configure( 0.005, capacity, true, 20e-6, 1e-6 );
    std::atomic<int64_t>  latest{ 1000000000 };
    std::atomic<bool>     run{ false }, framesDone{ false }, controlsDone{ false }, invalidSummary{ false };
    std::atomic<uint64_t> publishedBatch{ 0 }, observedBatch{ 0 };
    std::jthread          producer(
        [&]
        {
            while( !run.load( std::memory_order_acquire ) )
            {
                std::this_thread::yield();
            }

            uint64_t counter = 0;

            for( uint64_t attempt = 1; attempt <= attempts; ++attempt )
            {
                if( attempt % epochSize == 0 )
                {
                    collector.newSession();
                }

                const int64_t returned = 1000000000 + attempt * 500000;

                if( attempt % 11 == 0 )
                {
                    collector.beginAt( returned - 490000 );
                    collector.markAt( Marker::waitReturn, returned );
                    collector.finish( Outcome::rejected );
                }
                else
                {
                    frame( collector, returned, ++counter, attempt % 23 == 12 ? 30000 : 0 );
                }

                latest.store( returned + 40000, std::memory_order_release );

                if( attempt % batchSize == 0 )
                {
                    const uint64_t batch = attempt / batchSize;
                    publishedBatch.store( batch, std::memory_order_release );

                    while( observedBatch.load( std::memory_order_acquire ) < batch )
                    {
                        std::this_thread::yield();
                    }
                }
            }

            framesDone.store( true, std::memory_order_release );
        } );
    std::jthread controlProducer(
        [&]
        {
            while( !run.load( std::memory_order_acquire ) )
            {
                std::this_thread::yield();
            }

            for( uint64_t index = 0; index < controls; ++index )
            {
                const int64_t entered = latest.load( std::memory_order_acquire );
                collector.control( entered, entered + 1000, index % 17 != 0 );
                std::this_thread::yield();
            }

            controlsDone.store( true, std::memory_order_release );
        } );
    run.store( true, std::memory_order_release );
    size_t snapshots = 0;

    while( !framesDone.load( std::memory_order_acquire ) || !controlsDone.load( std::memory_order_acquire ) )
    {
        const uint64_t batch = publishedBatch.load( std::memory_order_acquire );
        const int64_t  now   = latest.load( std::memory_order_acquire );
        collector.update( now, epoch + now );
        const auto &summary = collector.summary();
        bool        bounded = summary.queued <= capacity && summary.controlQueued <= capacity &&
                       summary.intervals < capacity && summary.qualified <= summary.intervals &&
                       summary.ambiguous == summary.intervals - summary.qualified &&
                       summary.late <= summary.qualified && summary.early <= summary.qualified - summary.late;

        for( size_t index = 0; index < summary.metrics.size(); ++index )
        {
            bounded = bounded && summary.metrics[index].count <= capacity &&
                      summary.lateChange[index].count <= summary.late &&
                      summary.earlyChange[index].count <= summary.early;
        }

        if( !bounded )
        {
            invalidSummary.store( true, std::memory_order_relaxed );
        }

        ++snapshots;
        observedBatch.store( batch, std::memory_order_release );
    }

    producer.join();
    controlProducer.join();
    const int64_t final = latest.load( std::memory_order_acquire ) + 1000000;
    collector.update( final, epoch + final );
    const auto summary = collector.summary();
    REQUIRE_FALSE( invalidSummary.load( std::memory_order_relaxed ) );
    REQUIRE( snapshots >= attempts / batchSize );
    REQUIRE( summary.total.events + summary.total.dropped == attempts );
    REQUIRE( summary.total.controls + summary.total.controlDropped == controls );
    REQUIRE( summary.session == 1 + attempts / epochSize );

    // Seed the current epoch after both producers stop, even if its earlier events were all dropped.
    frame( collector, final + 500000, attempts + 1 );
    collector.update( final + 600000, epoch + final + 600000 );
    REQUIRE( collector.summary().metrics[metricIndex( Metric::acquisitionToWrite )].count > 0 );
    collector.update( final + 1000000000, epoch + final + 1000000000 );
    REQUIRE( collector.summary().state == Availability::stale );
    REQUIRE( collector.summary().metrics[metricIndex( Metric::acquisitionToWrite )].count == 0 );
    collector.configure( 0, capacity, false );
    REQUIRE_FALSE( collector.enabled() );
    REQUIRE_FALSE( collector.stagesEnabled() );
    collector.beginAt( final );
    collector.finish();
    collector.update( final, epoch + final );
    REQUIRE( collector.summary().total.events == 0 );
}

} // namespace frameGrabberTimingTest
} // namespace libXWCTest
