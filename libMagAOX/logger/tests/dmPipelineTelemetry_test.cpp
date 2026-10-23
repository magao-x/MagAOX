/** \file dmPipelineTelemetry_test.cpp
 * \brief Independent camera cadence and upstream-stage telemetry round trips, validation and compatibility.
 * \ingroup logger_files
 */
#include <limits>

#include "../../../tests/testXWC.hpp"
#include "../logBinarySchemata.hpp"
#include "../generated/logMemberAccessor.hpp"

namespace libXWCTest
{
namespace loggerTest
{
/// Tests of independent camera and upstream-stage telemetry.
/** \ingroup dmPipelineTelemetry_unit_test */
namespace dmPipelineTelemetryTest
{
using namespace MagAOX;
using namespace MagAOX::logger;

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
/// Populate a duration or signed change with independently identifiable units and coverage.
dmTiming::Stats sampleStats( uint64_t count /**< [in] retained population */,
                             double   mean /**< [in] seconds, possibly signed */ )
{
    return { count, mean, 1e-6, mean - 2e-6, mean + 2e-6, 1.5, count >= 2 };
}

/// Construct two named trace rows with distinct conditional and signed-change populations.
dmPipelineTiming::Summary samplePipeline()
{
    dmPipelineTiming::Summary summary;
    summary.enabled                = true;
    summary.session                = ( uint64_t( 1 ) << 54 ) + 7;
    summary.sequence               = 31;
    summary.nowNs                  = 12345678900;
    summary.windowSec              = 2;
    summary.span                   = 1.8;
    summary.age                    = 0.2;
    summary.capacity               = 32;
    summary.queued                 = 3;
    summary.median                 = 500e-6;
    summary.threshold              = 20e-6;
    summary.intervals              = 12;
    summary.late                   = 3;
    summary.early                  = 2;
    summary.stageCount             = 2;
    summary.total                  = { ( uint64_t( 1 ) << 54 ) + 1, 102, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    summary.sourceInterval         = sampleStats( 12, 500e-6 );
    summary.lateSourceInterval     = sampleStats( 3, 550e-6 );
    summary.earlySourceInterval    = sampleStats( 2, 450e-6 );
    summary.latePublicationChange  = sampleStats( 3, -5e-6 );
    summary.earlyPublicationChange = sampleStats( 2, 5e-6 );

    for( size_t index = 0; index < summary.stageCount; ++index )
    {
        auto &stage                = summary.stages[index];
        stage.rowIndex             = index;
        stage.pid                  = 12345 + index;
        stage.inputInode           = ( uint64_t( 1 ) << 54 ) + 20 + index;
        stage.execution            = sampleStats( 13, ( index + 1 ) * 10e-6 );
        stage.handoff              = sampleStats( 13, ( index + 1 ) * 20e-6 );
        stage.lateExecution        = sampleStats( 3, ( index + 1 ) * 30e-6 );
        stage.lateHandoff          = sampleStats( 3, ( index + 1 ) * 40e-6 );
        stage.earlyExecution       = sampleStats( 2, ( index + 1 ) * 50e-6 );
        stage.earlyHandoff         = sampleStats( 2, ( index + 1 ) * 60e-6 );
        stage.lateExecutionChange  = sampleStats( 3, ( index + 1 ) * 7e-6 );
        stage.lateHandoffChange    = sampleStats( 3, ( index + 1 ) * -8e-6 );
        stage.earlyExecutionChange = sampleStats( 2, ( index + 1 ) * -9e-6 );
        stage.earlyHandoffChange   = sampleStats( 2, ( index + 1 ) * 10e-6 );
    }

    return summary;
}

/// Construct camera acquisition intervals with a separate callback-observation population.
dmTiming::Summary sampleCamera()
{
    dmTiming::Summary summary;
    summary.enabled                                                                 = true;
    summary.session                                                                 = 7;
    summary.sequence                                                                = 8;
    summary.nowNs                                                                   = 12345678900;
    summary.windowSec                                                               = 2;
    summary.span                                                                    = 1.9;
    summary.age                                                                     = 0.1;
    summary.capacity                                                                = 32;
    summary.metrics[dmTiming::metricIndex( dmTiming::Metric::publicationInterval )] = sampleStats( 12, 500e-6 );
    summary.metrics[dmTiming::metricIndex( dmTiming::Metric::receiptInterval )]     = sampleStats( 12, 501e-6 );
    summary.intervalPairs.enabled                                                   = true;
    summary.intervalPairs.median                                                    = 500e-6;
    summary.intervalPairs.threshold                                                 = 20e-6;
    summary.intervalPairs.intervals                                                 = 12;
    summary.intervalPairs.pairs                                                     = 11;
    summary.intervalPairs.afterLate                                                 = sampleStats( 3, 450e-6 );
    summary.intervalPairs.afterEarly                                                = sampleStats( 2, 550e-6 );
    summary.intervalPairs.lateEarlySum                                              = sampleStats( 2, 1000e-6 );
    summary.intervalPairs.earlyLateSum                                              = sampleStats( 1, 1001e-6 );
    summary.total = { 13, 13, 0, 0, 0, 13, 0, 0, 0, 0, 0, 0 };
    return summary;
}

/// Construct a scheduler-independent diagnostic record through the public message API.
telem_dmsynco::messageT message( const dmTiming::Summary         &camera /**< [in] independent acquisition cadence */,
                                 const dmPipelineTiming::Summary &pipeline /**< [in] trace-stage populations */ )
{
    return { 0, 2, 500000, 0, 0, 0, 0, 0, 0, {}, {}, {}, false, {}, camera, pipeline };
}

/// Build a real log envelope for the standard verification dispatch.
flatlogs::bufferPtrT envelope( const telem_dmsynco::messageT &input /**< [in] serialized diagnostic message */ )
{
    flatlogs::bufferPtrT buffer;
    REQUIRE( flatlogs::logHeader::createLog<telem_dmsynco>( buffer, { 1, 2 }, input, flatlogs::logPrio::LOG_TELEM ) ==
             0 );
    return buffer;
}

/// Check all population fields without truncating counters or conflating signed changes with durations.
void sameStats( const DmSyncoTimingStats *actual /**< [in] serialized population */,
                const dmTiming::Stats    &expected /**< [in] original population */ )
{
    REQUIRE( actual != nullptr );
    REQUIRE( actual->count() == expected.count );
    REQUIRE( actual->mean() == Approx( expected.mean ) );
    REQUIRE( actual->stddev() == Approx( expected.stddev ) );
    REQUIRE( actual->min() == Approx( expected.min ) );
    REQUIRE( actual->max() == Approx( expected.max ) );
    REQUIRE( actual->span() == expected.span );
    REQUIRE( actual->valid() == expected.valid );
}
/// \endcond

/// Preserve independent camera acquisition intervals, pipeline identities, conditional populations and signed changes.
/** \ingroup dmPipelineTelemetry_unit_test
 * \see MagAOX::logger::telem_dmsynco::messageT
 * \see MagAOX::logger::telem_dmsynco::verify
 * \see MagAOX::logger::telem_dmsynco::msgString
 */
TEST_CASE( "Synco telemetry preserves camera and trace-stage timing", "[dmPipelineTelemetry]" )
{
    const auto camera   = sampleCamera();
    const auto pipeline = samplePipeline();
    auto       input    = message( camera, pipeline );
    auto       buffer   = envelope( input );
    REQUIRE( telem_dmsynco::verify( buffer, flatlogs::logHeader::msgLen( buffer ) ) );
    REQUIRE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
    auto       *payload  = input.builder.GetBufferPointer();
    const auto *record   = GetTelem_dmsynco_fb( payload );
    const auto *observed = record->camera_timing();
    REQUIRE( observed->enabled() );
    REQUIRE( observed->valid() );
    REQUIRE( observed->session() == camera.session );
    REQUIRE( observed->sequence() == camera.sequence );
    sameStats( observed->publication(),
               camera.metrics[dmTiming::metricIndex( dmTiming::Metric::publicationInterval )] );
    sameStats( observed->receipt(), camera.metrics[dmTiming::metricIndex( dmTiming::Metric::receiptInterval )] );
    REQUIRE( observed->interval_pairs()->late_early_sum()->count() == 2 );
    REQUIRE( observed->interval_pairs()->early_late_sum()->count() == 1 );
    const auto *trace = record->pipeline_timing();
    REQUIRE( trace->enabled() );
    REQUIRE( trace->session() == pipeline.session );
    REQUIRE( trace->sequence() == pipeline.sequence );
    REQUIRE( trace->now_ns() == pipeline.nowNs );
    REQUIRE( trace->window_sec() == pipeline.windowSec );
    REQUIRE( trace->span_sec() == pipeline.span );
    REQUIRE( trace->age_sec() == pipeline.age );
    REQUIRE( trace->capacity() == pipeline.capacity );
    REQUIRE( trace->queued() == pipeline.queued );
    REQUIRE( trace->median() == pipeline.median );
    REQUIRE( trace->threshold() == pipeline.threshold );
    REQUIRE( trace->intervals() == pipeline.intervals );
    REQUIRE( trace->late() == pipeline.late );
    REQUIRE( trace->early() == pipeline.early );
    REQUIRE( trace->stage_count() == 2 );
    REQUIRE( trace->stages()->size() == 2 );
    const auto *quality = trace->total();
    REQUIRE( quality->events() == pipeline.total.events );
    REQUIRE( quality->receipts() == 102 );
    REQUIRE( quality->synthetic() == 3 );
    REQUIRE( quality->rejected() == 4 );
    REQUIRE( quality->duplicates() == 5 );
    REQUIRE( quality->counter_gaps() == 6 );
    REQUIRE( quality->invalid_metadata() == 7 );
    REQUIRE( quality->missing_trace() == 8 );
    REQUIRE( quality->invalid_trace() == 9 );
    REQUIRE( quality->topology_changes() == 10 );
    REQUIRE( quality->truncated() == 11 );
    REQUIRE( quality->dropped() == 12 );
    sameStats( trace->source_interval(), pipeline.sourceInterval );
    sameStats( trace->late_source_interval(), pipeline.lateSourceInterval );
    sameStats( trace->early_source_interval(), pipeline.earlySourceInterval );
    sameStats( trace->late_publication_change(), pipeline.latePublicationChange );
    sameStats( trace->early_publication_change(), pipeline.earlyPublicationChange );

    for( size_t index = 0; index < pipeline.stageCount; ++index )
    {
        const auto *stage    = trace->stages()->Get( index );
        const auto &expected = pipeline.stages[index];
        REQUIRE( stage->row_index() == index );
        REQUIRE( stage->pid() == expected.pid );
        REQUIRE( stage->input_inode() == expected.inputInode );
        sameStats( stage->execution(), expected.execution );
        sameStats( stage->handoff(), expected.handoff );
        sameStats( stage->late_execution(), expected.lateExecution );
        sameStats( stage->late_handoff(), expected.lateHandoff );
        sameStats( stage->early_execution(), expected.earlyExecution );
        sameStats( stage->early_handoff(), expected.earlyHandoff );
        sameStats( stage->late_execution_change(), expected.lateExecutionChange );
        sameStats( stage->late_handoff_change(), expected.lateHandoffChange );
        sameStats( stage->early_execution_change(), expected.earlyExecutionChange );
        sameStats( stage->early_handoff_change(), expected.earlyHandoffChange );
    }

    const auto text = telem_dmsynco::msgString( payload, input.builder.GetSize() );
    REQUIRE( text.find( "schedule_valid=0" ) != std::string::npos );
    REQUIRE( text.find( "cameraAcquisitionInterval[n=12 valid=1] mean_s=0.0005" ) != std::string::npos );
    REQUIRE( text.find( "cameraReceiptInterval[n=12 valid=1] mean_s=0.000501" ) != std::string::npos );
    REQUIRE( text.find( "camera_pairs={enabled:1,intervals:12,pairs:11" ) != std::string::npos );
    REQUIRE( text.find( "pipeline_stage={row:0,pid:12345,inputInode:18014398509482004}" ) != std::string::npos );
    REQUIRE( text.find( "stageLateHandoffChange[n=3 valid=1] mean_s=-8e-06" ) != std::string::npos );
    REQUIRE( text.find( "pipelineLatePublicationChange[n=3 valid=1] mean_s=-5e-06" ) != std::string::npos );
    REQUIRE( text.find( "pipelineSourceInterval[n=12 valid=1] mean_s=0.0005" ) != std::string::npos );
    const auto          json = telem_dmsynco::msgJSON( payload,
                                              input.builder.GetSize(),
                                              reinterpret_cast<const uint8_t *>( telem_dmsynco_bfbs ),
                                              telem_dmsynco_bfbs_len );
    flatbuffers::Parser parser;
    REQUIRE( parser.Deserialize( reinterpret_cast<const uint8_t *>( telem_dmsynco_bfbs ), telem_dmsynco_bfbs_len ) );
    REQUIRE( parser.Parse( json.c_str() ) );
    const auto *decoded = GetTelem_dmsynco_fb( parser.builder_.GetBufferPointer() )->pipeline_timing();
    REQUIRE( decoded->session() == pipeline.session );
    REQUIRE( decoded->total()->events() == pipeline.total.events );
    sameStats( decoded->stages()->Get( 1 )->early_execution_change(), pipeline.stages[1].earlyExecutionChange );
}

/// Distinguish physically absent legacy fields, configured empty collection and disabled diagnostics.
/** \ingroup dmPipelineTelemetry_unit_test
 * \see MagAOX::logger::telem_dmsynco::messageT
 * \see MagAOX::logger::telem_dmsynco::verify
 * \see MagAOX::logger::telem_dmsynco::msgString
 */
TEST_CASE( "Synco diagnostic telemetry preserves absent and disabled availability", "[dmPipelineTelemetry]" )
{
    auto input  = message( {}, {} );
    auto buffer = envelope( input );
    REQUIRE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
    auto text = telem_dmsynco::msgString( input.builder.GetBufferPointer(), input.builder.GetSize() );
    REQUIRE( text.find( "camera_enabled=0 camera_valid=0" ) != std::string::npos );
    REQUIRE( text.find( "cameraAcquisitionInterval[n=0 valid=0]=unavailable" ) != std::string::npos );
    REQUIRE( text.find( "pipeline_enabled=0" ) != std::string::npos );
    REQUIRE( text.find( "pipeline_timing=unavailable" ) == std::string::npos );
    auto pipeline      = samplePipeline();
    pipeline.intervals = pipeline.late = pipeline.early = 0;
    pipeline.median                                     = 0;
    pipeline.stageCount                                 = 0;
    pipeline.sourceInterval = pipeline.lateSourceInterval = pipeline.earlySourceInterval = {};
    pipeline.latePublicationChange = pipeline.earlyPublicationChange = {};
    auto warming                                                     = message( {}, pipeline );
    auto warmingBuffer                                               = envelope( warming );
    REQUIRE( verifyLogEntry( telem_dmsynco::eventCode, warmingBuffer.get() ) );
    auto unclassified      = samplePipeline();
    unclassified.threshold = 0;
    unclassified.late = unclassified.early = 0;
    unclassified.lateSourceInterval = unclassified.earlySourceInterval = {};
    unclassified.latePublicationChange = unclassified.earlyPublicationChange = {};

    for( auto &stage : unclassified.stages )
    {
        stage.lateExecution = stage.lateHandoff = stage.earlyExecution = stage.earlyHandoff = {};
        stage.lateExecutionChange = stage.lateHandoffChange = {};
        stage.earlyExecutionChange = stage.earlyHandoffChange = {};
    }

    auto ordinary       = message( {}, unclassified );
    auto ordinaryBuffer = envelope( ordinary );
    REQUIRE( verifyLogEntry( telem_dmsynco::eventCode, ordinaryBuffer.get() ) );

    input.builder.Clear();
    input.builder.Finish( CreateTelem_dmsynco_fb( input.builder ) );
    auto legacy = envelope( input );
    REQUIRE( verifyLogEntry( telem_dmsynco::eventCode, legacy.get() ) );
    text = telem_dmsynco::msgString( input.builder.GetBufferPointer(), input.builder.GetSize() );
    REQUIRE( text.find( "camera_timing=unavailable camera_pairs=unavailable pipeline_timing=unavailable" ) !=
             std::string::npos );
}

/// Reject shape, topology, conditional-count, unit and finite-value corruption while accepting signed changes.
/** \ingroup dmPipelineTelemetry_unit_test
 * \see MagAOX::logger::telem_dmsynco::messageT
 * \see MagAOX::logger::telem_dmsynco::verify
 * \see MagAOX::logger::telem_dmsynco::msgString
 */
TEST_CASE( "Synco diagnostic telemetry rejects impossible trace populations", "[dmPipelineTelemetry]" )
{
    for( int variant = 0; variant < 26; ++variant )
    {
        INFO( "invalid trace variant " << variant );
        auto  pipeline = samplePipeline();
        auto &stage    = pipeline.stages[0];

        switch( variant )
        {
        case 0:
            pipeline.stageCount = dmPipelineTiming::maxStages + 1;
            break;
        case 1:
            pipeline.stages[1].rowIndex = 0;
            break;
        case 2:
            stage.rowIndex = dmPipelineTiming::maxStages;
            break;
        case 3:
            pipeline.queued = pipeline.capacity + 1;
            break;
        case 4:
            pipeline.intervals = pipeline.capacity;
            break;
        case 5:
            pipeline.late = pipeline.intervals + 1;
            break;
        case 6:
            pipeline.early = pipeline.intervals;
            break;
        case 7:
            pipeline.median = 0;
            break;
        case 8:
            pipeline.threshold = -1;
            break;
        case 9:
            pipeline.windowSec = std::numeric_limits<double>::infinity();
            break;
        case 10:
            pipeline.age = -2;
            break;
        case 11:
            pipeline.enabled = false;
            break;
        case 12:
            stage.execution = sampleStats( 13, -10e-6 );
            break;
        case 13:
            stage.lateExecution.count = pipeline.late + 1;
            break;
        case 14:
            stage.earlyHandoff.count = pipeline.early + 1;
            break;
        case 15:
            stage.earlyExecutionChange.count = stage.earlyExecution.count + 1;
            break;
        case 16:
            stage.lateHandoffChange.mean = std::numeric_limits<double>::quiet_NaN();
            break;
        case 17:
            stage.execution.count = 1;
            break;
        case 18:
            stage.handoff.span = -1;
            break;
        case 19:
            pipeline.sourceInterval.count = pipeline.intervals + 1;
            break;
        case 20:
            pipeline.lateSourceInterval.mean = -1;
            break;
        case 21:
            pipeline.earlyPublicationChange.count = pipeline.early + 1;
            break;
        case 22:
            pipeline.latePublicationChange.stddev = -1;
            break;
        case 23:
            stage.pid = 0;
            break;
        case 24:
            stage.inputInode = 0;
            break;
        case 25:
            pipeline.threshold = 0;
            break;
        }

        auto input  = message( {}, pipeline );
        auto buffer = envelope( input );
        REQUIRE_FALSE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
    }
}

/// Verify independent camera statistics and directed pairs through the same nested cadence validation.
/** \ingroup dmPipelineTelemetry_unit_test
 * \see MagAOX::logger::telem_dmsynco::messageT
 * \see MagAOX::logger::telem_dmsynco::verify
 * \see MagAOX::logger::telem_dmsynco::msgString
 */
TEST_CASE( "Synco diagnostic telemetry rejects invalid camera cadence", "[dmPipelineTelemetry]" )
{
    for( int variant = 0; variant < 4; ++variant )
    {
        auto camera = sampleCamera();

        switch( variant )
        {
        case 0:
            camera.metrics[dmTiming::metricIndex( dmTiming::Metric::publicationInterval )].mean = -1;
            break;
        case 1:
            camera.intervalPairs.afterLate.count = camera.intervalPairs.pairs + 1;
            break;
        case 2:
            camera.intervalPairs.threshold = std::numeric_limits<double>::infinity();
            break;
        case 3:
            camera.span = std::numeric_limits<double>::quiet_NaN();
            break;
        }

        auto input  = message( camera, {} );
        auto buffer = envelope( input );
        REQUIRE_FALSE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
    }

    auto input  = message( sampleCamera(), samplePipeline() );
    auto buffer = envelope( input );
    REQUIRE_FALSE( telem_dmsynco::verify( buffer, 1 ) );
    std::memset( flatlogs::logHeader::messageBuffer( buffer ), 0xff, 4 );
    REQUIRE_FALSE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
}

} // namespace dmPipelineTelemetryTest
} // namespace loggerTest
} // namespace libXWCTest
