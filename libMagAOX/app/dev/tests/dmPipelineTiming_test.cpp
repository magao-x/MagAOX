/** \file dmPipelineTiming_test.cpp
 * \brief Deterministic tests of upstream timing attribution and bounded observational qualification.
 */
#include "../../../../tests/testXWC.hpp"
#include "../dmPipelineTiming.hpp"

namespace libXWCTest
{
namespace dmPipelineTimingTests
{

using Collector = MagAOX::app::dev::dmPipelineTiming;

/// \cond DM_PIPELINE_TIMING_TEST_HELPERS
namespace
{

/// Build existing trace rows with independent trigger-input counters and a zero-filled tail.
struct Stream
{
    IMAGE          image{};    ///< In-memory descriptor; no shared-memory or hardware attachment.
    IMAGE_METADATA metadata{}; ///< Writable deterministic publication metadata.
    std::array<STREAM_PROC_TRACE, MagAOX::dmPipelineTiming::maxStages> rows{}; ///< Existing process-trace fixture.

    /// Connect the descriptor to fixture-owned metadata and trace storage.
    Stream();

    /// Set a complete three-stage chain from its oldest start, adding delay at one selected endpoint.
    void frame( uint64_t number /**< [in] Frame index; each linked row has its own counter offset. */,
                int64_t  sourceNs = -1 /**< [in] Oldest trace start; negative selects a regular 500 us cadence. */,
                int64_t  executionExtra   = 0 /**< [in] Additional row-one execution time in ns. */,
                int64_t  handoffExtra     = 0 /**< [in] Additional row-zero handoff time in ns. */,
                int64_t  publicationExtra = 0 /**< [in] Additional primary writetime tail in ns. */ );

    /// Set one camera-triggered stage with independent wake, execution and publication-tail changes.
    void singleStageFrame( uint64_t number /**< [in] Consecutive frame and camera-trigger index. */,
                           int64_t  wakeExtra /**< [in] Start displacement from the regular cadence in ns. */,
                           int64_t  executionExtra /**< [in] Additional execution duration in ns. */,
                           int64_t  publicationExtra /**< [in] Additional output-publication tail in ns. */ );

    /// Capture the current fixture metadata at a regular independent local receipt timestamp.
    void capture( Collector &collector /**< [in,out] Collector receiving the event. */,
                  uint64_t   number /**< [in] Local observation index. */,
                  bool       synthetic = false /**< [in] Omit a real semaphore receipt. */ );

    /// Set a nanosecond producer timestamp without loss of integer precision.
    static timespec stamp( int64_t ns /**< [in] Positive external timestamp in ns. */ );
};

Stream::Stream()
{
    image.md              = &metadata;
    image.streamproctrace = rows.data();
    metadata.NBproctrace  = rows.size();
}

timespec Stream::stamp( int64_t ns )
{
    return { static_cast<time_t>( ns / 1000000000 ), static_cast<long>( ns % 1000000000 ) };
}

void Stream::frame(
    uint64_t number, int64_t sourceNs, int64_t executionExtra, int64_t handoffExtra, int64_t publicationExtra )
{
    constexpr int64_t externalBase = 1790000000000000000LL;
    int64_t           source       = sourceNs < 0 ? externalBase + number * 500000 : sourceNs;
    rows                           = {};
    metadata.write                 = 0;
    metadata.cnt0                  = number + 1;
    metadata.NBproctrace           = rows.size();

    for( size_t i = 0; i < 3; ++i )
    {
        rows[i].procwrite_PID = 100 + i;
        rows[i].trigger_inode = i < 2 ? 700 + i : 0;
        rows[i].triggerstatus = i < 2 ? 2 : 0;
        rows[i].cnt0          = 1000 * ( i + 1 ) + number;
    }

    rows[2].ts_procstart    = stamp( source );
    rows[2].ts_streamupdate = stamp( source + 10000 );
    rows[1].ts_procstart    = stamp( source + 13000 );
    rows[1].ts_streamupdate = stamp( source + 33000 + executionExtra );
    rows[0].ts_procstart    = stamp( source + 38000 + executionExtra + handoffExtra );
    rows[0].ts_streamupdate = stamp( source + 68000 + executionExtra + handoffExtra );
    metadata.writetime      = stamp( source + 70000 + executionExtra + handoffExtra + publicationExtra );
}

void Stream::capture( Collector &collector, uint64_t number, bool synthetic )
{
    int64_t local = 1000000 + number * 500000;
    collector.observe( image, local + 100, synthetic ? -1 : local );
}

void Stream::singleStageFrame( uint64_t number, int64_t wakeExtra, int64_t executionExtra, int64_t publicationExtra )
{
    constexpr int64_t externalBase = 1790000000000000000LL;
    int64_t           start        = externalBase + number * 500000 + wakeExtra;
    frame( number );
    rows                    = {};
    rows[0].procwrite_PID   = 100;
    rows[0].trigger_inode   = 700;
    rows[0].triggerstatus   = 2;
    rows[0].cnt0            = 1000 + number;
    rows[0].ts_procstart    = stamp( start );
    rows[0].ts_streamupdate = stamp( start + 10000 + executionExtra );
    metadata.writetime      = stamp( start + 12000 + executionExtra + publicationExtra );
}

} // namespace
/// \endcond

/** \ingroup dmPipelineTiming_unit_test */

/** \brief An isolated camera-triggered stage exposes wake cadence without downstream AO processes.
 * \ingroup dmPipelineTiming_unit_test
 */
TEST_CASE( "Pipeline timing isolates one camera-triggered stage", "[dev::dmPipelineTiming]" )
{
    Collector collector;
    collector.configure( 2, 20, 20e-6 );
    Stream  stream;
    int64_t wakeExtra = 0, executionExtra = 0, publicationExtra = 0;

    SECTION( "A wake excursion changes the source cadence" )
    {
        wakeExtra = 40000;
    }

    SECTION( "A processing excursion leaves wake cadence unchanged" )
    {
        executionExtra = 40000;
    }

    SECTION( "A publication-tail excursion leaves wake and processing cadence unchanged" )
    {
        publicationExtra = 40000;
    }

    for( uint64_t i = 0; i < 7; ++i )
    {
        bool excursion = i == 2;
        stream.singleStageFrame(
            i, excursion ? wakeExtra : 0, excursion ? executionExtra : 0, excursion ? publicationExtra : 0 );
        stream.capture( collector, i );
    }

    collector.update( 4000100 );
    const auto &summary = collector.summary();
    REQUIRE( summary.stageCount == 1 );
    CHECK( summary.stages[0].pid == 100 );
    CHECK( summary.stages[0].inputInode == 700 );
    CHECK( summary.intervals == 6 );
    CHECK( summary.median == Approx( 500e-6 ) );
    CHECK( summary.late == 1 );
    CHECK( summary.early == 1 );
    CHECK( summary.sourceInterval.count == 6 );
    CHECK( summary.sourceInterval.valid );
    CHECK( summary.sourceInterval.mean == Approx( 500e-6 ) );
    CHECK( summary.sourceInterval.stddev == Approx( std::sqrt( 0.4 ) * wakeExtra * 1e-9 ) );
    CHECK( summary.sourceInterval.min == Approx( ( 500000 - wakeExtra ) * 1e-9 ) );
    CHECK( summary.sourceInterval.max == Approx( ( 500000 + wakeExtra ) * 1e-9 ) );
    CHECK( summary.sourceInterval.span == Approx( 0.003 ) );
    CHECK( summary.lateSourceInterval.mean == Approx( ( 500000 + wakeExtra ) * 1e-9 ) );
    CHECK( summary.earlySourceInterval.mean == Approx( ( 500000 - wakeExtra ) * 1e-9 ) );
    CHECK( summary.stages[0].execution.count == 6 );
    CHECK( summary.stages[0].lateExecution.mean == Approx( ( 10000 + executionExtra ) * 1e-9 ) );
    CHECK( summary.stages[0].earlyExecution.mean == Approx( 10e-6 ) );
    CHECK( summary.stages[0].lateExecutionChange.mean == Approx( executionExtra * 1e-9 ) );
    CHECK( summary.stages[0].earlyExecutionChange.mean == Approx( -executionExtra * 1e-9 ) );
    CHECK( summary.stages[0].handoff.count == 0 );
    CHECK( summary.latePublicationChange.mean == Approx( publicationExtra * 1e-9 ) );
    CHECK( summary.earlyPublicationChange.mean == Approx( -publicationExtra * 1e-9 ) );
    CHECK( summary.lateSourceInterval.mean + summary.stages[0].lateExecutionChange.mean +
               summary.latePublicationChange.mean ==
           Approx( 540e-6 ) );
    CHECK( summary.earlySourceInterval.mean + summary.stages[0].earlyExecutionChange.mean +
               summary.earlyPublicationChange.mean ==
           Approx( 460e-6 ) );
    CHECK( summary.total.invalidTrace == 0 );
    CHECK( summary.total.missingTrace == 0 );
    CHECK( summary.total.counterGaps == 0 );

    // clang-format off
#ifdef XWCTEST_DOX_REF
    MagAOX::app::dev::dmPipelineTiming::observe;
    MagAOX::app::dev::dmPipelineTiming::calculate;
#endif
    // clang-format on
}

/** \brief Distinguish added execution from handoff delay and preserve signed catch-up changes.
 * \ingroup dmPipelineTiming_unit_test
 */
TEST_CASE( "Pipeline timing locates execution and handoff excursions", "[dev::dmPipelineTiming]" )
{
    Collector collector;
    collector.configure( 2, 20, 20e-6 );
    Stream stream;

    SECTION( "A single processing stage takes longer" )
    {
        for( uint64_t i = 0; i < 7; ++i )
        {
            stream.frame( i, -1, i == 2 ? 40000 : 0 );
            stream.capture( collector, i );
        }

        collector.update( 4000100 );
        const auto &summary = collector.summary();
        REQUIRE( summary.stageCount == 3 );
        CHECK( summary.intervals == 6 );
        CHECK( summary.median == Approx( 500e-6 ) );
        CHECK( summary.late == 1 );
        CHECK( summary.early == 1 );
        CHECK( summary.stages[0].rowIndex == 0 );
        CHECK( summary.stages[0].pid == 100 );
        CHECK( summary.stages[0].inputInode == 700 );
        CHECK( summary.stages[1].execution.count == 6 );
        CHECK( summary.stages[1].execution.mean == Approx( 80e-6 / 3 ) );
        CHECK( summary.stages[1].execution.stddev == Approx( std::sqrt( 8.0 / 3 ) * 10e-6 ) );
        CHECK( summary.stages[1].execution.span == Approx( 0.0025 ) );
        CHECK( summary.stages[1].lateExecution.mean == Approx( 60e-6 ) );
        CHECK( summary.stages[1].earlyExecution.mean == Approx( 20e-6 ) );
        CHECK( summary.stages[1].lateExecutionChange.mean == Approx( 40e-6 ) );
        CHECK( summary.stages[1].earlyExecutionChange.mean == Approx( -40e-6 ) );
        CHECK( summary.stages[0].lateHandoffChange.mean == 0 );
        CHECK( summary.stages[1].lateHandoffChange.mean == 0 );
        CHECK( summary.sourceInterval.mean == Approx( 500e-6 ) );
        CHECK( summary.lateSourceInterval.mean == Approx( 500e-6 ) );
        CHECK( summary.earlySourceInterval.mean == Approx( 500e-6 ) );
        CHECK( summary.latePublicationChange.mean == 0 );
        CHECK( summary.stages[2].execution.mean == Approx( 10e-6 ) );
        CHECK( summary.stages[2].handoff.count == 0 );
        CHECK( summary.total.invalidTrace == 0 );
        CHECK( summary.total.missingTrace == 0 );
    }

    SECTION( "A stage waits longer after its upstream update" )
    {
        for( uint64_t i = 0; i < 7; ++i )
        {
            stream.frame( i, -1, 0, i == 2 ? 45000 : 0 );
            stream.capture( collector, i );
        }

        collector.update( 4000100 );
        const auto &summary = collector.summary();
        REQUIRE( summary.stageCount == 3 );
        CHECK( summary.late == 1 );
        CHECK( summary.early == 1 );
        CHECK( summary.stages[0].lateHandoff.mean == Approx( 50e-6 ) );
        CHECK( summary.stages[0].lateHandoffChange.mean == Approx( 45e-6 ) );
        CHECK( summary.stages[0].earlyHandoffChange.mean == Approx( -45e-6 ) );
        CHECK( summary.stages[0].lateExecutionChange.mean == 0 );
        CHECK( summary.stages[1].lateExecutionChange.mean == 0 );
    }

    // clang-format off
#ifdef XWCTEST_DOX_REF
    MagAOX::app::dev::dmPipelineTiming::observe;
    MagAOX::app::dev::dmPipelineTiming::calculate;
#endif
    // clang-format on
}

/** \brief Keep source-boundary and publication-tail variation distinct from represented stage costs.
 * \ingroup dmPipelineTiming_unit_test
 */
TEST_CASE( "Pipeline timing exposes unrepresented arrival and publication changes", "[dev::dmPipelineTiming]" )
{
    Collector collector;
    collector.configure( 2, 20, 20e-6 );
    Stream            stream;
    constexpr int64_t externalBase = 1790000000000000000LL;

    SECTION( "Excursion enters before the oldest represented stage" )
    {
        for( uint64_t i = 0; i < 7; ++i )
        {
            stream.frame( i, externalBase + i * 500000 + ( i == 2 ? 40000 : 0 ) );
            stream.capture( collector, i );
        }

        collector.update( 4000100 );
        const auto &summary = collector.summary();
        CHECK( summary.lateSourceInterval.mean == Approx( 540e-6 ) );
        CHECK( summary.earlySourceInterval.mean == Approx( 460e-6 ) );
        CHECK( summary.stages[0].lateExecutionChange.mean == 0 );
        CHECK( summary.stages[0].lateHandoffChange.mean == 0 );
    }

    SECTION( "Excursion enters after the last represented output update" )
    {
        for( uint64_t i = 0; i < 7; ++i )
        {
            stream.frame( i, -1, 0, 0, i == 2 ? 40000 : 0 );
            stream.capture( collector, i );
        }

        collector.update( 4000100 );
        const auto &summary = collector.summary();
        CHECK( summary.lateSourceInterval.mean == Approx( 500e-6 ) );
        CHECK( summary.latePublicationChange.mean == Approx( 40e-6 ) );
        CHECK( summary.earlyPublicationChange.mean == Approx( -40e-6 ) );
        CHECK( summary.stages[0].lateExecutionChange.mean == 0 );
    }

    // clang-format off
#ifdef XWCTEST_DOX_REF
    MagAOX::app::dev::dmPipelineTiming::calculate;
#endif
    // clang-format on
}

/** \brief Counter gaps, rejected/synthetic callbacks and resets cannot bridge conditional associations.
 * \ingroup dmPipelineTiming_unit_test
 */
TEST_CASE( "Pipeline timing breaks associations after losses and epochs", "[dev::dmPipelineTiming]" )
{
    Collector collector;
    collector.configure( 2, 20, 20e-6 );
    Stream stream;
    stream.frame( 0 );
    stream.capture( collector, 0 );
    stream.frame( 1 );
    stream.capture( collector, 1 );

    SECTION( "A primary counter skips an observation" )
    {
        stream.frame( 4 );
        stream.capture( collector, 2 );
        stream.frame( 5 );
        stream.capture( collector, 3 );
        collector.update( 3000000 );
        CHECK( collector.summary().total.counterGaps == 1 );
        CHECK( collector.summary().intervals == 2 );
        CHECK( collector.summary().stages[0].execution.count == 2 );
        CHECK( collector.summary().late == 0 );
    }

    SECTION( "A duplicate cannot manufacture a new stage observation" )
    {
        stream.capture( collector, 2 );
        stream.frame( 2 );
        stream.capture( collector, 3 );
        collector.update( 3000000 );
        CHECK( collector.summary().total.duplicates == 1 );
        CHECK( collector.summary().stages[0].execution.count == 1 );
    }

    SECTION( "A synthetic event is excluded from consecutive classification" )
    {
        stream.frame( 2 );
        stream.capture( collector, 2, true );
        stream.frame( 3 );
        stream.capture( collector, 3 );
        collector.update( 3000000 );
        CHECK( collector.summary().total.synthetic == 1 );
        CHECK( collector.summary().intervals == 1 );
        CHECK( collector.summary().stages[0].execution.count == 1 );
    }

    SECTION( "A rejected receipt is visible and breaks the chain" )
    {
        collector.rejected( 2000000 );
        stream.frame( 3 );
        stream.capture( collector, 3 );
        collector.update( 3000000 );
        CHECK( collector.summary().total.rejected == 1 );
        CHECK( collector.summary().total.invalidMetadata == 0 );
        CHECK( collector.summary().intervals == 1 );
        CHECK( collector.summary().stages[0].execution.count == 1 );
    }

    SECTION( "An attachment change clears retained populations without clearing quality totals" )
    {
        collector.update( 1600000 );
        collector.newSession();
        collector.update( 1700000 );
        CHECK( collector.summary().session == 2 );
        CHECK( collector.summary().stageCount == 0 );
        CHECK( collector.summary().intervals == 0 );
        CHECK( collector.summary().total.events == 2 );
        stream.frame( 2 );
        stream.capture( collector, 2 );
        collector.update( 2100000 );
        CHECK( collector.summary().stages[0].execution.count == 0 );
    }

    // clang-format off
#ifdef XWCTEST_DOX_REF
    MagAOX::app::dev::dmPipelineTiming::qualify;
    MagAOX::app::dev::dmPipelineTiming::newSession;
    MagAOX::app::dev::dmPipelineTiming::rejected;
#endif
    // clang-format on
}

/** \brief Reject malformed, frozen, incomplete and changed trace identities without censoring publication tails.
 * \ingroup dmPipelineTiming_unit_test
 */
TEST_CASE( "Pipeline timing qualifies progressing coherent trace chains", "[dev::dmPipelineTiming]" )
{
    Collector collector;
    collector.configure( 2, 20, 20e-6 );
    Stream stream;
    stream.frame( 0 );
    stream.capture( collector, 0 );
    stream.frame( 1 );

    SECTION( "An unchanged process start indicates disabled or stale MILK timing" )
    {
        stream.rows[1].ts_procstart = Stream::stamp( 1790000000000013000LL );
        stream.capture( collector, 1 );
        collector.update( 1600000 );
        CHECK( collector.summary().total.invalidTrace == 1 );
        CHECK( collector.summary().intervals == 1 );
        CHECK( collector.summary().stages[0].execution.count == 0 );
    }

    SECTION( "A nonconsecutive upstream trigger counter has no adjacent-frame attribution" )
    {
        stream.rows[1].cnt0 += 3;
        stream.capture( collector, 1 );
        collector.update( 1600000 );
        CHECK( collector.summary().total.invalidTrace == 1 );
        CHECK( collector.summary().intervals == 1 );
        CHECK( collector.summary().sourceInterval.count == 0 );
    }

    SECTION( "A timeout status invalidates linked timing" )
    {
        stream.rows[0].triggerstatus = 3;
        stream.capture( collector, 1 );
        collector.update( 1600000 );
        CHECK( collector.summary().total.invalidTrace == 1 );
    }

    SECTION( "An impossible upstream update order invalidates the chain" )
    {
        stream.rows[2].ts_streamupdate = stream.rows[0].ts_streamupdate;
        stream.capture( collector, 1 );
        collector.update( 1600000 );
        CHECK( collector.summary().total.invalidTrace == 1 );
    }

    SECTION( "A zero-filled tail is ordinary, but a populated row after a hole is invalid" )
    {
        stream.rows[1].procwrite_PID = 0;
        stream.capture( collector, 1 );
        collector.update( 1600000 );
        CHECK( collector.summary().total.invalidTrace == 1 );
        CHECK( collector.summary().total.missingTrace == 0 );
    }

    SECTION( "A row identity change starts a distinct current topology" )
    {
        stream.rows[0].procwrite_PID = 999;
        stream.capture( collector, 1 );
        stream.frame( 2 );
        stream.rows[0].procwrite_PID = 999;
        stream.capture( collector, 2 );
        collector.update( 2100000 );
        CHECK( collector.summary().total.topologyChanges == 1 );
        CHECK( collector.summary().stages[0].pid == 999 );
        CHECK( collector.summary().stages[0].execution.count == 1 );
        CHECK( collector.summary().intervals == 2 );
    }

    SECTION( "A missing trace does not remove qualified primary publication intervals" )
    {
        stream.image.streamproctrace = nullptr;
        stream.capture( collector, 1 );
        collector.update( 1600000 );
        CHECK( collector.summary().total.missingTrace == 1 );
        CHECK( collector.summary().intervals == 1 );
    }

    SECTION( "Oversized advertised source capacity is bounded and explicitly reported" )
    {
        stream.metadata.NBproctrace = MagAOX::dmPipelineTiming::maxStages + 5;
        stream.capture( collector, 1 );
        collector.update( 1600000 );
        CHECK( collector.summary().total.truncated == 1 );
        CHECK( collector.summary().stageCount == 3 );
        CHECK( collector.summary().stages[0].execution.count == 1 );
    }

    SECTION( "Malformed publication metadata remains a quality failure" )
    {
        stream.metadata.writetime.tv_nsec = 1000000000;
        stream.capture( collector, 1 );
        collector.update( 1600000 );
        CHECK( collector.summary().total.invalidMetadata == 1 );
        CHECK( collector.summary().intervals == 0 );
    }

    // clang-format off
#ifdef XWCTEST_DOX_REF
    MagAOX::app::dev::dmPipelineTiming::observe;
    MagAOX::app::dev::dmPipelineTiming::qualify;
#endif
    // clang-format on
}

/** \brief Bound queue loss and window retention, and safely disable invalid configuration.
 * \ingroup dmPipelineTiming_unit_test
 */
TEST_CASE( "Pipeline timing remains bounded and expires without observations", "[dev::dmPipelineTiming]" )
{
    Collector collector;
    Stream    stream;
    collector.configure( 0.001, 2, 20e-6 );

    for( uint64_t i = 0; i < 3; ++i )
    {
        stream.frame( i );
        stream.capture( collector, i );
    }

    collector.update( 1600000 );
    CHECK( collector.summary().total.dropped == 1 );
    CHECK( collector.summary().total.events == 2 );
    CHECK( collector.summary().stages[0].execution.count == 1 );
    stream.frame( 3 );
    stream.capture( collector, 3 );
    collector.update( 2600000 );
    CHECK( collector.summary().intervals == 0 );
    CHECK( collector.summary().stages[0].execution.count == 0 );
    collector.update( 5000000 );
    CHECK( collector.summary().stageCount == 0 );
    CHECK( collector.summary().age == -1 );
    CHECK( collector.summary().sourceInterval.count == 0 );

    collector.configure( 2, 20, 0 );

    for( uint64_t i = 0; i < 4; ++i )
    {
        stream.frame( i, -1, i == 2 ? 40000 : 0 );
        stream.capture( collector, i );
    }

    collector.update( 2600000 );
    CHECK( collector.summary().threshold == 0 );
    CHECK( collector.summary().median == Approx( 500e-6 ) );
    CHECK( collector.summary().stages[0].execution.count == 3 );
    CHECK( collector.summary().late == 0 );
    CHECK( collector.summary().stages[0].lateExecution.count == 0 );
    collector.configure( std::numeric_limits<double>::quiet_NaN(), 20, 20e-6 );
    CHECK_FALSE( collector.enabled() );
    collector.configure( std::numeric_limits<double>::infinity(), 20, 20e-6 );
    CHECK_FALSE( collector.enabled() );
    collector.configure( 2, 0, 20e-6 );
    CHECK_FALSE( collector.enabled() );
    collector.configure( 2, 20, std::numeric_limits<double>::quiet_NaN() );
    CHECK( collector.enabled() );
    CHECK( collector.summary().threshold == 0 );

    // clang-format off
#ifdef XWCTEST_DOX_REF
    MagAOX::app::dev::dmPipelineTiming::configure;
    MagAOX::app::dev::dmPipelineTiming::update;
#endif
    // clang-format on
}

} // namespace dmPipelineTimingTests
} // namespace libXWCTest
