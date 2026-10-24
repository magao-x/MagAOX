/** \file frameGrabberTelemetry_test.cpp
 * \brief Optional frame-grabber stage telemetry round trips, metadata projections and invalid-population guards.
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
/// Frame-grabber stage logger tests.
/** \ingroup frameGrabberTelemetry_unit_test */
namespace frameGrabberTelemetryTest
{
using namespace MagAOX;
using namespace MagAOX::logger;

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
/// Construct independently identifiable raw durations, signed changes and quality populations.
fgTiming::Summary sampleSummary()
{
    fgTiming::Summary summary;
    summary.session                    = ( uint64_t( 1 ) << 54 ) + 3;
    summary.sequence                   = 19;
    summary.nowNs                      = 12345678900;
    summary.newestNs                   = 12245678900;
    summary.windowSec                  = 2;
    summary.span                       = 1.8;
    summary.age                        = 0.1;
    summary.capacity                   = 32;
    summary.queued                     = 3;
    summary.controlQueued              = 2;
    summary.enabled                    = true;
    summary.stagesEnabled              = true;
    summary.fgValid                    = true;
    summary.state                      = fgTiming::Availability::valid;
    summary.median                     = 500e-6;
    summary.threshold                  = 20e-6;
    summary.bracketLimit               = 5e-6;
    summary.intervals                  = 12;
    summary.qualified                  = 10;
    summary.ambiguous                  = 2;
    summary.late                       = 3;
    summary.early                      = 2;
    summary.serialOverlaps             = 4;
    summary.streamInode                = ( uint64_t( 1 ) << 54 ) + 9;
    summary.lastFrame                  = 12345;
    summary.lastCounter                = 54321;
    summary.mode                       = 4;
    summary.binX                       = 2;
    summary.binY                       = 3;
    summary.serialDuration             = { 3, 10e-3, 1e-3, 9e-3, 11e-3, 1.4, true };
    summary.lateWaitReturnInterval     = { 3, 0.0005, 1e-6, 0.000498, 0.0005020000000000001, 1.4, true };
    summary.earlyWaitReturnInterval    = { 2, 0.0005, 1e-6, 0.000498, 0.0005020000000000001, 1.4, true };
    summary.lateWriteEndpointInterval  = { 3, 0.00055, 1e-6, 0.000548, 0.0005520000000000001, 1.4, true };
    summary.earlyWriteEndpointInterval = { 2, 0.00045, 1e-6, 0.000448, 0.000452, 1.4, true };
    summary.latePreviousTail           = { 3, 0.00012, 1e-6, 0.00011800000000000001, 0.000122, 1.4, true };
    summary.earlyPreviousTail          = { 2, 0.000125, 1e-6, 0.000123, 0.000127, 1.4, true };
    summary.total = { 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, ( uint64_t( 1 ) << 54 ) + 114 };
    summary.delta = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 };

    for( size_t index = 0; index < summary.metrics.size(); ++index )
    {
        const double mean          = ( index + 1 ) * 10e-6;
        summary.metrics[index]     = { 12, mean, 1e-6, mean - 2e-6, mean + 2e-6, 1.5, true };
        summary.lateChange[index]  = { 3, mean, 2e-6, mean - 3e-6, mean + 3e-6, 1.4, true };
        summary.earlyChange[index] = { 2, -mean, 3e-6, -mean - 4e-6, -mean + 4e-6, 1.3, true };
    }

    summary.metrics[fgTiming::metricIndex( fgTiming::Metric::acquisitionToWrite )] = {
        12, -5e-6, 1e-6, -7e-6, -3e-6, 1.5, true };
    return summary;
}

/// Build a real log envelope for generated verification dispatch.
flatlogs::bufferPtrT envelope( const telem_fgstage::messageT &input /**< [in] serialized diagnostic message */ )
{
    flatlogs::bufferPtrT buffer;
    REQUIRE( flatlogs::logHeader::createLog<telem_fgstage>( buffer, { 1, 2 }, input, flatlogs::logPrio::LOG_TELEM ) ==
             0 );
    return buffer;
}

/// Project a typed field through the real generated metadata dispatcher.
template <typename valueT>
valueT project( const std::string &name /**< [in] metadata field name */,
                void              *payload /**< [in] serialized message */ )
{
    const auto detail = logMemberAccessor( telem_fgstage::eventCode, name );
    REQUIRE( detail.accessor != nullptr );
    REQUIRE( detail.valType == mx::fits::fitsType<valueT>() );
    REQUIRE( detail.metaType == logMeta::metaTypes::State );
    return reinterpret_cast<valueT ( * )( void * )>( detail.accessor )( payload );
}

/// Compare every field of a seconds-based population without narrowing its counter.
void sameStats( const FgStageStats    *actual /**< [in] serialized independent population */,
                const fgTiming::Stats &expected /**< [in] original seconds-based population */ )
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

/// Serialize a metric vector in selected identity order for compatibility and shape checks.
flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<FgStageMetricStats>>>
rows( flatbuffers::FlatBufferBuilder &builder /**< [in/out] destination builder */,
      const std::vector<unsigned>    &identifiers /**< [in] metric IDs, including intentionally malformed sets */,
      const std::array<fgTiming::Stats, size_t( fgTiming::Metric::count )> &values /**< [in] known populations */ )
{
    std::vector<flatbuffers::Offset<FgStageMetricStats>> result;

    for( const auto identifier : identifiers )
    {
        const auto &value      = values[identifier < values.size() ? identifier : 0];
        const auto  population = CreateFgStageStats(
            builder, value.count, value.mean, value.stddev, value.min, value.max, value.span, value.valid );
        result.push_back( CreateFgStageMetricStats( builder, FgStageMetric( identifier ), population ) );
    }

    return builder.CreateVector( result );
}

/// Rebuild one record using arbitrary metric identities without invoking the production builder's fixed layout.
void replaceRows( telem_fgstage::messageT     &message /**< [in/out] serialized record to replace */,
                  const fgTiming::Summary     &summary /**< [in] coherent snapshot values */,
                  const std::vector<unsigned> &identifiers /**< [in] metric identity order */ )
{
    auto &builder = message.builder;
    builder.Clear();
    builder.ForceDefaults( true );
    const auto  raw    = rows( builder, identifiers, summary.metrics );
    const auto  late   = rows( builder, identifiers, summary.lateChange );
    const auto  early  = rows( builder, identifiers, summary.earlyChange );
    const auto &value  = summary.serialDuration;
    const auto  serial = CreateFgStageStats(
        builder, value.count, value.mean, value.stddev, value.min, value.max, value.span, value.valid );
    const auto  total                           = CreateFgStageCounters( builder, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10 );
    const auto  delta                           = CreateFgStageCounters( builder );
    const auto &lateWaitReturnIntervalValue     = summary.lateWaitReturnInterval;
    const auto  lateWaitReturnInterval          = CreateFgStageStats( builder,
                                                            lateWaitReturnIntervalValue.count,
                                                            lateWaitReturnIntervalValue.mean,
                                                            lateWaitReturnIntervalValue.stddev,
                                                            lateWaitReturnIntervalValue.min,
                                                            lateWaitReturnIntervalValue.max,
                                                            lateWaitReturnIntervalValue.span,
                                                            lateWaitReturnIntervalValue.valid );
    const auto &earlyWaitReturnIntervalValue    = summary.earlyWaitReturnInterval;
    const auto  earlyWaitReturnInterval         = CreateFgStageStats( builder,
                                                             earlyWaitReturnIntervalValue.count,
                                                             earlyWaitReturnIntervalValue.mean,
                                                             earlyWaitReturnIntervalValue.stddev,
                                                             earlyWaitReturnIntervalValue.min,
                                                             earlyWaitReturnIntervalValue.max,
                                                             earlyWaitReturnIntervalValue.span,
                                                             earlyWaitReturnIntervalValue.valid );
    const auto &lateWriteEndpointIntervalValue  = summary.lateWriteEndpointInterval;
    const auto  lateWriteEndpointInterval       = CreateFgStageStats( builder,
                                                               lateWriteEndpointIntervalValue.count,
                                                               lateWriteEndpointIntervalValue.mean,
                                                               lateWriteEndpointIntervalValue.stddev,
                                                               lateWriteEndpointIntervalValue.min,
                                                               lateWriteEndpointIntervalValue.max,
                                                               lateWriteEndpointIntervalValue.span,
                                                               lateWriteEndpointIntervalValue.valid );
    const auto &earlyWriteEndpointIntervalValue = summary.earlyWriteEndpointInterval;
    const auto  earlyWriteEndpointInterval      = CreateFgStageStats( builder,
                                                                earlyWriteEndpointIntervalValue.count,
                                                                earlyWriteEndpointIntervalValue.mean,
                                                                earlyWriteEndpointIntervalValue.stddev,
                                                                earlyWriteEndpointIntervalValue.min,
                                                                earlyWriteEndpointIntervalValue.max,
                                                                earlyWriteEndpointIntervalValue.span,
                                                                earlyWriteEndpointIntervalValue.valid );
    const auto &latePreviousTailValue           = summary.latePreviousTail;
    const auto  latePreviousTail                = CreateFgStageStats( builder,
                                                      latePreviousTailValue.count,
                                                      latePreviousTailValue.mean,
                                                      latePreviousTailValue.stddev,
                                                      latePreviousTailValue.min,
                                                      latePreviousTailValue.max,
                                                      latePreviousTailValue.span,
                                                      latePreviousTailValue.valid );
    const auto &earlyPreviousTailValue          = summary.earlyPreviousTail;
    const auto  earlyPreviousTail               = CreateFgStageStats( builder,
                                                       earlyPreviousTailValue.count,
                                                       earlyPreviousTailValue.mean,
                                                       earlyPreviousTailValue.stddev,
                                                       earlyPreviousTailValue.min,
                                                       earlyPreviousTailValue.max,
                                                       earlyPreviousTailValue.span,
                                                       earlyPreviousTailValue.valid );
    builder.Finish( CreateTelem_fgstage_fb( builder,
                                            summary.session,
                                            summary.sequence,
                                            summary.nowNs,
                                            summary.newestNs,
                                            summary.windowSec,
                                            summary.span,
                                            summary.age,
                                            summary.capacity,
                                            summary.queued,
                                            summary.controlQueued,
                                            summary.enabled,
                                            summary.stagesEnabled,
                                            summary.fgValid,
                                            FgStageAvailability( summary.state ),
                                            raw,
                                            late,
                                            early,
                                            serial,
                                            summary.median,
                                            summary.threshold,
                                            summary.bracketLimit,
                                            summary.intervals,
                                            summary.qualified,
                                            summary.ambiguous,
                                            summary.late,
                                            summary.early,
                                            summary.serialOverlaps,
                                            summary.streamInode,
                                            summary.lastFrame,
                                            summary.lastCounter,
                                            summary.mode,
                                            summary.binX,
                                            summary.binY,
                                            total,
                                            delta,
                                            lateWaitReturnInterval,
                                            earlyWaitReturnInterval,
                                            lateWriteEndpointInterval,
                                            earlyWriteEndpointInterval,
                                            latePreviousTail,
                                            earlyPreviousTail ) );
}
/// \endcond

/// Preserve every segment, conditional change, quality counter and frame context through binary and JSON paths.
/** \ingroup frameGrabberTelemetry_unit_test
 * \see MagAOX::logger::telem_fgstage::messageT
 * \see MagAOX::logger::telem_fgstage::verify
 * \see MagAOX::logger::telem_fgstage::msgString
 */
TEST_CASE( "Frame grabber stage telemetry preserves independent populations", "[frameGrabberTelemetry]" )
{
    const auto              summary = sampleSummary();
    telem_fgstage::messageT input( summary );
    auto                    buffer = envelope( input );
    REQUIRE( telem_fgstage::verify( buffer, flatlogs::logHeader::msgLen( buffer ) ) );
    REQUIRE( verifyLogEntry( telem_fgstage::eventCode, buffer.get() ) );
    REQUIRE( eventCodeName( telem_fgstage::eventCode ) == "telem_fgstage" );
    auto       *payload = input.builder.GetBufferPointer();
    const auto *record  = GetTelem_fgstage_fb( payload );
    REQUIRE( record->session() == summary.session );
    REQUIRE( record->sequence() == summary.sequence );
    REQUIRE( record->now_ns() == summary.nowNs );
    REQUIRE( record->newest_ns() == summary.newestNs );
    REQUIRE( record->window_sec() == summary.windowSec );
    REQUIRE( record->span_sec() == summary.span );
    REQUIRE( record->age_sec() == summary.age );
    REQUIRE( record->capacity() == summary.capacity );
    REQUIRE( record->queued() == summary.queued );
    REQUIRE( record->control_queued() == summary.controlQueued );
    REQUIRE( record->enabled() );
    REQUIRE( record->stages_enabled() );
    REQUIRE( record->fg_valid() );
    REQUIRE( record->state() == FgStageAvailability_valid );
    REQUIRE( record->median() == summary.median );
    REQUIRE( record->threshold() == summary.threshold );
    REQUIRE( record->bracket_limit() == summary.bracketLimit );
    REQUIRE( record->intervals() == 12 );
    REQUIRE( record->qualified() == 10 );
    REQUIRE( record->ambiguous() == 2 );
    REQUIRE( record->late() == 3 );
    REQUIRE( record->early() == 2 );
    REQUIRE( record->serial_overlaps() == 4 );
    REQUIRE( record->stream_inode() == summary.streamInode );
    REQUIRE( record->last_frame() == summary.lastFrame );
    REQUIRE( record->last_counter() == summary.lastCounter );
    REQUIRE( record->mode() == 4 );
    REQUIRE( record->bin_x() == 2 );
    REQUIRE( record->bin_y() == 3 );
    REQUIRE( record->metrics()->size() == summary.metrics.size() );
    REQUIRE( record->late_change()->size() == summary.lateChange.size() );
    REQUIRE( record->early_change()->size() == summary.earlyChange.size() );
    sameStats( record->serial_duration(), summary.serialDuration );
    sameStats( record->late_wait_return_interval(), summary.lateWaitReturnInterval );
    sameStats( record->early_wait_return_interval(), summary.earlyWaitReturnInterval );
    sameStats( record->late_write_endpoint_interval(), summary.lateWriteEndpointInterval );
    sameStats( record->early_write_endpoint_interval(), summary.earlyWriteEndpointInterval );
    sameStats( record->late_previous_tail(), summary.latePreviousTail );
    sameStats( record->early_previous_tail(), summary.earlyPreviousTail );

    for( size_t index = 0; index < summary.metrics.size(); ++index )
    {
        REQUIRE( size_t( record->metrics()->Get( index )->metric() ) == index );
        sameStats( record->metrics()->Get( index )->value(), summary.metrics[index] );
        sameStats( record->late_change()->Get( index )->value(), summary.lateChange[index] );
        sameStats( record->early_change()->Get( index )->value(), summary.earlyChange[index] );
    }

    const auto text = telem_fgstage::msgString( payload, input.builder.GetSize() );
    REQUIRE( text.starts_with( "[fgstage] session=" ) );
    REQUIRE( text.find( "state=valid enabled=1 stages_enabled=1 fg_valid=1" ) != std::string::npos );
    REQUIRE( text.find( "intervals=12 qualified=10 ambiguous=2 late=3 early=2" ) != std::string::npos );
    REQUIRE( text.find( "acquisitionToWrite[n=12 valid=1]=-5e-06+/-1e-06" ) != std::string::npos );
    REQUIRE( text.find( "earlyChange_rearm[n=2 valid=1]=-7e-05" ) != std::string::npos );
    REQUIRE( text.find( "serialDuration[n=3 valid=1]=0.01+/-0.001" ) != std::string::npos );
    REQUIRE( text.find( "controlDropped:18014398509482098" ) != std::string::npos );
    const auto          json = telem_fgstage::msgJSON( payload,
                                              input.builder.GetSize(),
                                              reinterpret_cast<const uint8_t *>( telem_fgstage_bfbs ),
                                              telem_fgstage_bfbs_len );
    flatbuffers::Parser parser;
    REQUIRE( parser.Deserialize( reinterpret_cast<const uint8_t *>( telem_fgstage_bfbs ), telem_fgstage_bfbs_len ) );
    REQUIRE( parser.Parse( json.c_str() ) );
    const auto *decoded = GetTelem_fgstage_fb( parser.builder_.GetBufferPointer() );
    REQUIRE( decoded->session() == summary.session );
    REQUIRE( decoded->total()->control_dropped() == summary.total.controlDropped );
    REQUIRE( decoded->stream_inode() == summary.streamInode );
    sameStats( decoded->early_change()->Get( size_t( fgTiming::Metric::loopTail ) )->value(),
               summary.earlyChange[size_t( fgTiming::Metric::loopTail )] );
}

/// Project exact typed metadata and duration signs while preserving raw versus qualified cohort names.
/** \ingroup frameGrabberTelemetry_unit_test
 * \see MagAOX::logger::telem_fgstage::getAccessor
 */
TEST_CASE( "Frame grabber stage metadata exposes coverage and matched changes", "[frameGrabberTelemetry]" )
{
    const auto              summary = sampleSummary();
    telem_fgstage::messageT input( summary );
    auto                   *payload = input.builder.GetBufferPointer();
    REQUIRE( project<uint64_t>( "session", payload ) == summary.session );
    REQUIRE( project<uint64_t>( "streamInode", payload ) == summary.streamInode );
    REQUIRE( project<uint32_t>( "mode", payload ) == 4 );
    REQUIRE( project<uint32_t>( "binX", payload ) == 2 );
    REQUIRE( project<uint32_t>( "binY", payload ) == 3 );
    REQUIRE( project<uint64_t>( "lastFrame", payload ) == summary.lastFrame );
    REQUIRE( project<uint64_t>( "lastCounter", payload ) == summary.lastCounter );
    REQUIRE( project<int64_t>( "nowNs", payload ) == summary.nowNs );
    REQUIRE( project<int64_t>( "newestNs", payload ) == summary.newestNs );
    REQUIRE( project<double>( "windowSec", payload ) == 2 );
    REQUIRE( project<double>( "span", payload ) == 1.8 );
    REQUIRE( project<double>( "age", payload ) == 0.1 );
    REQUIRE( project<uint64_t>( "capacity", payload ) == 32 );
    REQUIRE( project<uint64_t>( "queued", payload ) == 3 );
    REQUIRE( project<uint64_t>( "controlQueued", payload ) == 2 );
    REQUIRE( project<bool>( "enabled", payload ) );
    REQUIRE( project<bool>( "stagesEnabled", payload ) );
    REQUIRE( project<bool>( "fgValid", payload ) );
    REQUIRE( project<int>( "state", payload ) == int( fgTiming::Availability::valid ) );
    REQUIRE( project<double>( "median", payload ) == summary.median );
    REQUIRE( project<double>( "threshold", payload ) == summary.threshold );
    REQUIRE( project<double>( "bracketLimit", payload ) == summary.bracketLimit );
    REQUIRE( project<uint64_t>( "intervals", payload ) == 12 );
    REQUIRE( project<uint64_t>( "qualified", payload ) == 10 );
    REQUIRE( project<uint64_t>( "ambiguous", payload ) == 2 );
    REQUIRE( project<uint64_t>( "late", payload ) == 3 );
    REQUIRE( project<uint64_t>( "early", payload ) == 2 );
    REQUIRE( project<uint64_t>( "serialOverlaps", payload ) == 4 );
    REQUIRE( project<double>( "serialDuration_mean", payload ) == 10e-3 );
    REQUIRE( project<double>( "lateWaitReturnInterval_mean", payload ) == summary.lateWaitReturnInterval.mean );
    REQUIRE( project<double>( "earlyWaitReturnInterval_mean", payload ) == summary.earlyWaitReturnInterval.mean );
    REQUIRE( project<double>( "lateWriteEndpointInterval_mean", payload ) == summary.lateWriteEndpointInterval.mean );
    REQUIRE( project<double>( "earlyWriteEndpointInterval_mean", payload ) == summary.earlyWriteEndpointInterval.mean );
    REQUIRE( project<double>( "latePreviousTail_mean", payload ) == summary.latePreviousTail.mean );
    REQUIRE( project<double>( "earlyPreviousTail_mean", payload ) == summary.earlyPreviousTail.mean );

    for( size_t index = 0; index < summary.metrics.size(); ++index )
    {
        const auto label = std::string( fgTiming::metricName( fgTiming::Metric( index ) ) );
        const std::pair<std::string, const fgTiming::Stats *> populations[] = {
            { label, &summary.metrics[index] },
            { "lateChange_" + label, &summary.lateChange[index] },
            { "earlyChange_" + label, &summary.earlyChange[index] } };

        for( const auto &population : populations )
        {
            const auto &name     = population.first;
            const auto &expected = *population.second;
            REQUIRE( project<uint64_t>( name + "_count", payload ) == expected.count );
            REQUIRE( project<bool>( name + "_valid", payload ) == expected.valid );
            REQUIRE( project<double>( name + "_mean", payload ) == expected.mean );
            REQUIRE( project<double>( name + "_stddev", payload ) == expected.stddev );
            REQUIRE( project<double>( name + "_min", payload ) == expected.min );
            REQUIRE( project<double>( name + "_max", payload ) == expected.max );
            REQUIRE( project<double>( name + "_span", payload ) == expected.span );
        }
    }

    const auto &names  = fgTiming::counterNames();
    const auto &fields = fgTiming::counterFields();

    for( size_t index = 0; index < names.size(); ++index )
    {
        REQUIRE( project<uint64_t>( std::string( "total_" ) + names[index], payload ) == summary.total.*fields[index] );
        REQUIRE( project<uint64_t>( std::string( "delta_" ) + names[index], payload ) == summary.delta.*fields[index] );
    }

    REQUIRE( telem_fgstage::getAccessor( "unknown" ).accessor == nullptr );
    REQUIRE( telem_fgstage::getAccessor( "wait_unknown" ).accessor == nullptr );
}

/// Keep disabled, warming, stale and ambiguous raw measurements distinct from successful attribution.
/** \ingroup frameGrabberTelemetry_unit_test
 * \see MagAOX::logger::telem_fgstage::verify
 * \see MagAOX::logger::telem_fgstage::msgString
 * \see MagAOX::logger::telem_fgstage::getAccessor
 */
TEST_CASE( "Frame grabber telemetry preserves unavailable and ambiguous populations", "[frameGrabberTelemetry]" )
{
    for( const auto state : { fgTiming::Availability::disabled,
                              fgTiming::Availability::warmingUp,
                              fgTiming::Availability::stale,
                              fgTiming::Availability::unavailable } )
    {
        fgTiming::Summary summary;
        summary.state = state;
        telem_fgstage::messageT input( summary );
        auto                    buffer = envelope( input );
        REQUIRE( verifyLogEntry( telem_fgstage::eventCode, buffer.get() ) );
        REQUIRE( mx::math::isNan( project<double>( "rearm_mean", input.builder.GetBufferPointer() ) ) );
        REQUIRE( project<uint64_t>( "rearm_count", input.builder.GetBufferPointer() ) == 0 );
    }

    auto summary         = sampleSummary();
    summary.bracketLimit = 0;
    summary.qualified = summary.late = summary.early = 0;
    summary.ambiguous                                = summary.intervals;
    summary.lateChange                               = {};
    summary.lateWaitReturnInterval                   = {};
    summary.earlyWaitReturnInterval                  = {};
    summary.lateWriteEndpointInterval                = {};
    summary.earlyWriteEndpointInterval               = {};
    summary.latePreviousTail                         = {};
    summary.earlyPreviousTail                        = {};

    summary.earlyChange                                       = {};
    summary.metrics[size_t( fgTiming::Metric::writeBracket )] = { 1, 80e-6, 0, 80e-6, 80e-6, 0, false };
    telem_fgstage::messageT input( summary );
    auto                    buffer = envelope( input );
    REQUIRE( verifyLogEntry( telem_fgstage::eventCode, buffer.get() ) );
    auto *payload = input.builder.GetBufferPointer();
    REQUIRE( project<double>( "writeBracket_mean", payload ) == 80e-6 );
    REQUIRE( mx::math::isNan( project<double>( "writeBracket_stddev", payload ) ) );
    REQUIRE( project<uint64_t>( "ambiguous", payload ) == 12 );
    REQUIRE( project<uint64_t>( "lateChange_rearm_count", payload ) == 0 );
    const auto text = telem_fgstage::msgString( payload, input.builder.GetSize() );
    REQUIRE( text.find( "writeBracket[n=1 valid=0]=8e-05+/-unavailable" ) != std::string::npos );
    REQUIRE( text.find( "lateChange_rearm[n=0 valid=0]=unavailable" ) != std::string::npos );
}

/// Reject nonfinite values, impossible cohort counts, negative local durations and invalid record shapes.
/** \ingroup frameGrabberTelemetry_unit_test
 * \see MagAOX::logger::telem_fgstage::verify
 */
TEST_CASE( "Frame grabber telemetry rejects malformed snapshot populations", "[frameGrabberTelemetry]" )
{
    for( int variant = 0; variant < 25; ++variant )
    {
        INFO( "malformed stage variant " << variant );
        auto  summary = sampleSummary();
        auto &rearm   = summary.metrics[size_t( fgTiming::Metric::rearm )];

        switch( variant )
        {
        case 0:
            summary.state = fgTiming::Availability( 255 );
            break;
        case 1:
            summary.windowSec = std::numeric_limits<double>::infinity();
            break;
        case 2:
            summary.span = -1;
            break;
        case 3:
            summary.age = -2;
            break;
        case 4:
            summary.queued = summary.capacity + 1;
            break;
        case 5:
            summary.controlQueued = summary.capacity + 1;
            break;
        case 6:
            summary.intervals = summary.capacity;
            break;
        case 7:
            summary.qualified = summary.intervals + 1;
            break;
        case 8:
            summary.ambiguous = summary.intervals;
            break;
        case 9:
            summary.late = summary.qualified + 1;
            break;
        case 10:
            summary.early = summary.qualified;
            break;
        case 11:
            summary.median = 0;
            break;
        case 12:
            summary.threshold = std::numeric_limits<double>::quiet_NaN();
            break;
        case 13:
            summary.bracketLimit = 0;
            break;
        case 14:
            summary.stagesEnabled = false;
            break;
        case 15:
            rearm.count = 1;
            break;
        case 16:
            rearm.mean = std::numeric_limits<double>::quiet_NaN();
            break;
        case 17:
            rearm = { 2, -5e-6, 1e-6, -6e-6, -4e-6, 1, true };
            break;
        case 18:
            rearm.stddev = -1;
            break;
        case 19:
            summary.lateChange[0].count = summary.late + 1;
            break;
        case 20:
            summary.earlyChange[0].span = -1;
            break;
        case 21:
            summary.serialDuration.min = -1;
            break;
        case 22:
            summary.serialDuration.count = summary.total.controls + 1;
            break;
        case 23:
            summary.lateWaitReturnInterval.count = summary.late + 1;
            break;
        case 24:
            summary.earlyPreviousTail.min = -1;
            break;
        }

        telem_fgstage::messageT input( summary );
        auto                    buffer = envelope( input );
        REQUIRE_FALSE( verifyLogEntry( telem_fgstage::eventCode, buffer.get() ) );
    }

    telem_fgstage::messageT input( sampleSummary() );
    auto                    buffer = envelope( input );
    REQUIRE_FALSE( telem_fgstage::verify( buffer, 1 ) );
    std::memset( flatlogs::logHeader::messageBuffer( buffer ), 0xff, 4 );
    REQUIRE_FALSE( verifyLogEntry( telem_fgstage::eventCode, buffer.get() ) );
    flatlogs::bufferPtrT absent;
    REQUIRE_FALSE( telem_fgstage::verify( absent, 0 ) );
}

/// Resolve metrics by stable IDs after reordering and reject duplicate, unknown or incomplete metric sets.
/** \ingroup frameGrabberTelemetry_unit_test
 * \see MagAOX::logger::telem_fgstage::verify
 * \see MagAOX::logger::telem_fgstage::getAccessor
 */
TEST_CASE( "Frame grabber telemetry validates stable metric identities", "[frameGrabberTelemetry]" )
{
    const auto summary = sampleSummary();

    for( unsigned variant = 0; variant < 4; ++variant )
    {
        std::vector<unsigned> identifiers;

        for( unsigned index = 0; index < unsigned( fgTiming::Metric::count ); ++index )
        {
            identifiers.push_back( unsigned( fgTiming::Metric::count ) - index - 1 );
        }

        if( variant == 1 )
        {
            identifiers.back() = identifiers.front();
        }
        else if( variant == 2 )
        {
            identifiers.back() = 255;
        }
        else if( variant == 3 )
        {
            identifiers.pop_back();
        }

        telem_fgstage::messageT input( summary );
        replaceRows( input, summary, identifiers );
        auto buffer = envelope( input );
        REQUIRE( verifyLogEntry( telem_fgstage::eventCode, buffer.get() ) == ( variant == 0 ) );

        if( variant == 0 )
        {
            REQUIRE( project<double>( "rearm_mean", input.builder.GetBufferPointer() ) ==
                     summary.metrics[size_t( fgTiming::Metric::rearm )].mean );
        }
    }
}

} // namespace frameGrabberTelemetryTest
} // namespace loggerTest
} // namespace libXWCTest
