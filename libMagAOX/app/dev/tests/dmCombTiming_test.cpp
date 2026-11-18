/** \file dmCombTiming_test.cpp
 * \brief Offline deterministic and concurrent coverage of observational DM combiner timing.
 */
#include "../../../../tests/testXWC.hpp"
#include "../dmCombTiming.hpp"

#include <atomic>
#include <thread>

namespace libXWCTest
{
namespace dmCombTimingTests
{
using Collector = MagAOX::app::dev::dmCombTiming;
using namespace MagAOX::dmCombTiming;

/// \cond DM_COMB_TIMING_TEST_HELPERS
namespace
{

/// Own ordinary local structs, without creating or opening a shared-memory stream.
struct Stream
{
    IMAGE                            image{};    ///< Descriptor pointing only at test-owned local memory.
    IMAGE_METADATA                   metadata{}; ///< Output metadata under test.
    std::array<STREAM_PROC_TRACE, 2> traces{};   ///< Writer and upstream trace rows.

    /// Point the descriptor at its local metadata and trace storage.
    Stream();

    /// Set ordered external-clock timestamps and consecutive output/primary counters.
    void frame( uint64_t counter /**< [in] Output counter. */,
                int64_t  publicationBase /**< [in] External frame-reference nanoseconds. */,
                int64_t  extra = 0 /**< [in] Additional process execution duration, reducing publication overhead. */ );
};

/// Convert exact positive external integer nanoseconds into a normalized timespec.
timespec stamp( int64_t ns /**< [in] External-clock nanoseconds. */ );

timespec stamp( int64_t ns )
{
    return { ns / 1000000000, ns % 1000000000 };
}

Stream::Stream()
{
    image.md              = &metadata;
    image.streamproctrace = traces.data();
    metadata.NBproctrace  = traces.size();
}

void Stream::frame( uint64_t counter, int64_t publicationBase, int64_t extra )
{
    metadata.write            = 0;
    metadata.cnt0             = counter;
    metadata.writetime        = stamp( publicationBase + 350 );
    metadata.NBproctrace      = traces.size();
    traces[0].procwrite_PID   = 11;
    traces[0].trigger_inode   = 42;
    traces[0].cnt0            = counter + 100;
    traces[0].ts_procstart    = stamp( publicationBase + 100 );
    traces[0].ts_streamupdate = stamp( publicationBase + 300 + extra );
    traces[1].procwrite_PID   = 22;
    traces[1].cnt0            = counter * 3; // The upstream row's own input counter is not the primary output counter.
    traces[1].ts_streamupdate = stamp( publicationBase );
}

} // namespace
/// \endcond

/** \defgroup dmCombTiming_tests Combined DM stream timing tests
 * \ingroup app_dev_unit_tests
 */

/** \brief Preserve integer precision and keep producer-clock durations separate from local coverage.
 * \ingroup dmCombTiming_tests
 */
TEST_CASE( "DM combiner timing measures qualified trace durations", "[dev::dmCombTiming]" )
{
    Collector collector;
    collector.configure( 2, 20 );
    Stream            stream;
    constexpr int64_t epoch = 1792000000000000000LL;

    for( int i = 0; i < 4; ++i )
    {
        stream.frame( i, epoch + i * 500000, i * 10 );
        collector.observe( stream.image, 42, 1000000100 + i * 500002, 1000000000 + i * 500002 );
    }

    collector.update( 1001600000 );
    const auto &summary = collector.summary();
    REQUIRE( summary.metrics[index( Metric::publicationInterval )].count == 3 );
    CHECK( summary.metrics[index( Metric::publicationInterval )].mean == Approx( 0.0005 ) );
    CHECK( summary.metrics[index( Metric::receiptInterval )].mean == Approx( 0.000500002 ) );
    CHECK( summary.metrics[index( Metric::execution )].mean == Approx( 220e-9 ) );
    CHECK( summary.metrics[index( Metric::execution )].stddev == Approx( 10e-9 ) );
    CHECK( summary.metrics[index( Metric::publication )].mean == Approx( 30e-9 ) );
    CHECK( summary.metrics[index( Metric::publication )].stddev == Approx( 10e-9 ) );
    CHECK( summary.metrics[index( Metric::primaryStart )].mean == Approx( 100e-9 ) );
    CHECK( summary.metrics[index( Metric::primaryPublication )].mean == Approx( 350e-9 ) );
    CHECK( summary.metrics[index( Metric::execution )].span == Approx( 0.001000004 ) );
    CHECK( summary.span == Approx( 0.001500006 ) );
    CHECK( summary.age == Approx( 0.000099994 ) );
    CHECK( summary.writerPid == 11 );
    CHECK( summary.triggerInode == 42 );
    CHECK( summary.primaryCounter == 103 );
    CHECK( summary.total.events == 4 );
    CHECK( summary.total.receipts == 4 );
    CHECK( summary.total.invalidTrace == 0 );
    CHECK( summary.total.repeatedStart == 0 );

    for( const auto &metric : summary.metrics )
    {
        CHECK( metric.valid );
        CHECK( metric.count == 3 );
    }

    CHECK( std::string( metricName( Metric::primaryPublication ) ) == "primaryPublication" );
    CHECK( std::string( metricName( Metric::count ) ) == "unknown" );

    // clang-format off
#ifdef XWCTEST_DOX_REF
    MagAOX::app::dev::dmCombTiming::observe;
    MagAOX::app::dev::dmCombTiming::qualify;
    MagAOX::app::dev::dmCombTiming::calculate;
#endif
    // clang-format on
}

/** \brief Stale timing instrumentation and repeated primary associations remain unavailable independently.
 * \ingroup dmCombTiming_tests
 */
TEST_CASE( "DM combiner timing excludes repeated starts and primary associations", "[dev::dmCombTiming]" )
{
    Collector collector;
    collector.configure( 2, 20 );
    Stream            stream;
    constexpr int64_t epoch = 1792000000000000000LL;
    stream.frame( 0, epoch );
    collector.observe( stream.image, 42, 1000, 1000 );
    stream.frame( 1, epoch + 500000 );
    collector.observe( stream.image, 42, 2000, 2000 );
    collector.observe( stream.image, 42, 3000, 3000 ); // Same output; retain actual receipt but not trace work.
    stream.frame( 2, epoch + 1000000 );
    stream.traces[0].ts_procstart    = stamp( epoch + 500100 );
    stream.traces[1].ts_streamupdate = stamp( epoch + 500050 );
    collector.observe( stream.image, 42, 4000, 4000 );
    stream.frame( 3, epoch + 1500000 );
    stream.traces[0].cnt0 = 102; // Start advanced but the linked input counter did not.
    collector.observe( stream.image, 42, 5000, 5000 );
    collector.update( 6000 );

    const auto &summary = collector.summary();
    CHECK( summary.total.events == 5 );
    CHECK( summary.total.receipts == 5 );
    CHECK( summary.total.duplicates == 1 );
    CHECK( summary.total.repeatedStart == 1 );
    CHECK( summary.total.repeatedPrimary == 1 );
    CHECK( summary.metrics[index( Metric::receiptInterval )].count == 4 );
    CHECK( summary.metrics[index( Metric::publicationInterval )].count == 3 );
    CHECK( summary.metrics[index( Metric::execution )].count == 2 );
    CHECK( summary.metrics[index( Metric::primaryStart )].count == 1 );
    CHECK_FALSE( summary.metrics[index( Metric::primaryStart )].valid );

    // clang-format off
#ifdef XWCTEST_DOX_REF
    MagAOX::app::dev::dmCombTiming::qualify;
#endif
    // clang-format on
}

/** \brief Frozen or backward upstream rows and changed upstream writers cannot fabricate increasing latency.
 * \ingroup dmCombTiming_tests
 */
TEST_CASE( "DM combiner timing requires a fresh upstream publication", "[dev::dmCombTiming][regression]" )
{
    Collector collector;
    collector.configure( 2, 20 );
    Stream            stream;
    constexpr int64_t epoch = 1792000000000000000LL;

    for( int i = 0; i < 7; ++i )
    {
        stream.frame( i, epoch + i * 500000 );

        if( i == 2 )
        {
            stream.traces[1].ts_streamupdate = stamp( epoch + 500000 );
        }
        else if( i == 3 )
        {
            stream.traces[1].ts_streamupdate = stamp( epoch + 400000 );
        }
        else if( i == 4 )
        {
            stream.traces[1].ts_streamupdate = stamp( epoch + 450000 );
        }

        if( i >= 5 )
        {
            stream.traces[1].procwrite_PID = 33;
        }

        collector.observe( stream.image, 42, 1000 + i * 1000, 1000 + i * 1000 );
    }

    collector.update( 8000 );
    CHECK( collector.summary().total.repeatedPrimary == 2 );
    CHECK( collector.summary().metrics[index( Metric::execution )].count == 6 );
    CHECK( collector.summary().metrics[index( Metric::primaryStart )].count == 3 );
    CHECK( collector.summary().metrics[index( Metric::primaryPublication )].count == 3 );

    // clang-format off
#ifdef XWCTEST_DOX_REF
    MagAOX::app::dev::dmCombTiming::qualify;
#endif
    // clang-format on
}

/** \brief Missing/invalid rows and unlinked clocks are counted without inventing trace spans.
 * \ingroup dmCombTiming_tests
 */
TEST_CASE( "DM combiner timing qualifies metadata and trace availability", "[dev::dmCombTiming]" )
{
    Collector collector;
    collector.configure( 2, 20 );
    Stream            stream;
    constexpr int64_t epoch = 1792000000000000000LL;
    stream.frame( 0, epoch );
    stream.metadata.write = 1;
    collector.observe( stream.image, 42, 1000, 1000 );
    stream.frame( 1, epoch + 500000 );
    stream.metadata.writetime.tv_nsec = 1000000000;
    collector.observe( stream.image, 42, 2000, 2000 );
    stream.frame( 2, epoch + 1000000 );
    stream.metadata.NBproctrace = 0;
    collector.observe( stream.image, 42, 3000, 3000 );
    stream.frame( 3, epoch + 1500000 );
    stream.traces[0].procwrite_PID = 0;
    collector.observe( stream.image, 42, 4000, 4000 );
    stream.frame( 4, epoch + 2000000 );
    stream.traces[0].ts_procstart = stamp( epoch + 2000400 ); // Start is after update/publication.
    collector.observe( stream.image, 42, 5000, 5000 );
    stream.frame( 5, epoch + 2500000 );
    stream.traces[0].trigger_inode = 99;
    collector.observe( stream.image, 42, 6000, 6000 );
    stream.frame( 6, epoch + 3000000 );
    stream.metadata.NBproctrace = 1;
    collector.observe( stream.image, 42, 7000, 7000 );
    stream.frame( 7, epoch + 3500000 );
    stream.traces[1].ts_streamupdate = stamp( epoch + 3500200 ); // Upstream update is after process start.
    collector.observe( stream.image, 42, 8000, 8000 );
    stream.frame( 8, epoch + 4000000 );
    collector.observe( stream.image, 0, 9000, 9000 ); // An unknown primary inode is not an association.
    collector.update( 10000 );

    const auto &summary = collector.summary();
    CHECK( summary.total.events == 9 );
    CHECK( summary.total.invalidMetadata == 2 );
    CHECK( summary.total.missingTrace == 1 );
    CHECK( summary.total.invalidTrace == 2 );
    CHECK( summary.total.unlinkedPrimary == 2 );
    CHECK( summary.total.invalidPrimary == 2 );
    CHECK( summary.metrics[index( Metric::receiptInterval )].count == 8 );
    CHECK( summary.metrics[index( Metric::primaryPublication )].count == 0 );

    // clang-format off
#ifdef XWCTEST_DOX_REF
    MagAOX::app::dev::dmCombTiming::timestamp;
    MagAOX::app::dev::dmCombTiming::observe;
    MagAOX::app::dev::dmCombTiming::qualify;
#endif
    // clang-format on
}

/** \brief Loss boundaries and monitor resets cannot pair unrelated output publications or trace baselines.
 * \ingroup dmCombTiming_tests
 */
TEST_CASE( "DM combiner timing respects gaps, queue capacity and epochs", "[dev::dmCombTiming]" )
{
    Collector collector;
    collector.configure( 2, 3 );
    Stream            stream;
    constexpr int64_t epoch = 1792000000000000000LL;

    for( int i = 0; i < 4; ++i )
    {
        stream.frame( i, epoch + i * 500000 );
        collector.observe( stream.image, 42, 1000 + i * 1000, 1000 + i * 1000 );
    }

    collector.update( 5000 );
    CHECK( collector.summary().total.events == 3 );
    CHECK( collector.summary().total.dropped == 1 );
    stream.frame( 3, epoch + 1500000 );
    collector.observe( stream.image, 42, 5000, 5000 );
    collector.update( 5000 );
    CHECK( collector.summary().metrics[index( Metric::publicationInterval )].count == 1 );
    CHECK( collector.summary().metrics[index( Metric::execution )].count == 2 );
    CHECK( collector.summary().total.counterGaps == 0 );
    collector.rejected( 6000 );
    collector.update( 6000 );
    CHECK( collector.summary().total.rejected == 1 );
    CHECK( collector.summary().total.receipts == 5 );
    CHECK( collector.summary().total.invalidMetadata == 0 );
    const auto session = collector.summary().session;
    collector.newSession();
    collector.update( 7000 );
    CHECK( collector.summary().session == session + 1 );
    CHECK( collector.summary().writerPid == 0 );
    CHECK( collector.summary().metrics[index( Metric::receiptInterval )].count == 0 );
    stream.frame( 20, epoch + 10000000 );
    collector.observe( stream.image, 42, 8000 );
    collector.update( 8000 );
    CHECK( collector.summary().total.synthetic == 1 );
    CHECK( collector.summary().metrics[index( Metric::execution )].count == 0 );

    collector.configure( 2, 10 );
    stream.frame( 0, epoch );
    collector.observe( stream.image, 42, 1000, 1000 );
    stream.frame( 2, epoch + 1000000 );
    collector.observe( stream.image, 42, 2000, 2000 );
    stream.frame( 3, epoch + 1500000 );
    collector.observe( stream.image, 42, 3000, 3000 );
    collector.update( 3000 );
    CHECK( collector.summary().total.counterGaps == 1 );
    CHECK( collector.summary().metrics[index( Metric::publicationInterval )].count == 1 );
    CHECK( collector.summary().metrics[index( Metric::execution )].count == 1 );
    CHECK( collector.summary().metrics[index( Metric::primaryStart )].count == 1 );

    // clang-format off
#ifdef XWCTEST_DOX_REF
    MagAOX::app::dev::dmCombTiming::enqueue;
    MagAOX::app::dev::dmCombTiming::newSession;
    MagAOX::app::dev::dmCombTiming::rejected;
#endif
    // clang-format on
}

/** \brief Local elapsed-time windows expire without new events and never compare external epochs with local time.
 * \ingroup dmCombTiming_tests
 */
TEST_CASE( "DM combiner timing uses local observation coverage and disabled collection", "[dev::dmCombTiming]" )
{
    Collector collector;
    collector.configure( 0.001, 8 );
    Stream            stream;
    constexpr int64_t epoch = 1792000000000000000LL;

    for( int i = 0; i < 3; ++i )
    {
        stream.frame( i, epoch + i * 500000 );
        collector.observe( stream.image, 42, 1000000 + i * 500000, 1000000 + i * 500000 );
    }

    collector.update( 2000001 );
    CHECK( collector.summary().metrics[index( Metric::publicationInterval )].count == 1 );
    CHECK( collector.summary().metrics[index( Metric::receiptInterval )].count == 1 );
    CHECK( collector.summary().metrics[index( Metric::execution )].count == 2 );
    CHECK( collector.summary().metrics[index( Metric::execution )].span == Approx( 0.0005 ) );
    collector.update( 4000000 );
    CHECK( collector.summary().age == -1 );

    for( const auto &metric : collector.summary().metrics )
    {
        CHECK( metric.count == 0 );
        CHECK_FALSE( metric.valid );
    }

    collector.configure( 0, 8 );
    collector.observe( stream.image, 42, 1000, 1000 );
    collector.rejected( 2000 );
    collector.newSession();
    collector.update( 3000 );
    CHECK_FALSE( collector.enabled() );
    CHECK_FALSE( collector.summary().enabled );
    CHECK( collector.summary().total.events == 0 );
    CHECK( collector.summary().session == 1 );

    // clang-format off
#ifdef XWCTEST_DOX_REF
    MagAOX::app::dev::dmCombTiming::configure;
    MagAOX::app::dev::dmCombTiming::calculate;
#endif
    // clang-format on
}

/** \brief Concurrent queue wraps preserve copied payload ownership and account for every dropped observation.
 * \ingroup dmCombTiming_tests
 */
TEST_CASE( "DM combiner timing SPSC handoff remains coherent during wraps", "[dev::dmCombTiming][concurrency]" )
{
    Collector collector;
    collector.configure( 2, 64 );
    constexpr int     count = 100000;
    constexpr int64_t epoch = 1792000000000000000LL;
    std::atomic<bool> done{ false };
    std::thread       producer(
        [&]()
        {
            Stream stream;

            for( int i = 0; i < count; ++i )
            {
                stream.frame( i, epoch + int64_t( i ) * 500000 );
                collector.observe( stream.image, 42, 1000000 + int64_t( i ) * 500000, 1000000 + int64_t( i ) * 500000 );
            }

            done.store( true, std::memory_order_release );
        } );

    while( !done.load( std::memory_order_acquire ) )
    {
        collector.update( 1000000 + int64_t( count ) * 500000 );
    }

    producer.join();
    collector.update( 1000000 + int64_t( count ) * 500000 );
    const auto &summary = collector.summary();
    CHECK( summary.total.events + summary.total.dropped == count );
    CHECK( summary.total.events == summary.total.receipts );
    CHECK( summary.total.invalidMetadata == 0 );
    CHECK( summary.total.invalidTrace == 0 );
    CHECK( summary.total.duplicates == 0 );
    CHECK( summary.total.counterGaps == 0 );
    CHECK( summary.queued == 0 );

    // clang-format off
#ifdef XWCTEST_DOX_REF
    MagAOX::app::dev::dmCombTiming::observe;
    MagAOX::app::dev::dmCombTiming::enqueue;
    MagAOX::app::dev::dmCombTiming::update;
#endif
    // clang-format on
}

} // namespace dmCombTimingTests
} // namespace libXWCTest
