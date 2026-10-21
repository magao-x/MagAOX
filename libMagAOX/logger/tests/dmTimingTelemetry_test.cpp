/** \file dmTimingTelemetry_test.cpp
 * \brief Controller timing telemetry round trips, validity and metadata projections.
 * \ingroup logger_files
 */
#include "../../../tests/testXWC.hpp"
#include "../logBinarySchemata.hpp"
#include "../generated/logMemberAccessor.hpp"

namespace libXWCTest
{
namespace loggerTest
{
/** \defgroup dmTimingTelemetry_unit_test DM Timing Telemetry Tests
 * \ingroup logger_unit_test
 */
namespace dmTimingTelemetryTest
{
using namespace MagAOX;
using namespace MagAOX::logger;

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
/// Construct distinct metric populations and large counters for exact serialization checks.
dmTiming::Summary sampleSummary()
{
    dmTiming::Summary summary;
    summary.session   = ( uint64_t( 1 ) << 54 ) + 3;
    summary.sequence  = 17;
    summary.nowNs     = 9876543210;
    summary.newestNs  = 9676543210;
    summary.windowSec = 2;
    summary.span      = 1.8;
    summary.age       = 0.2;
    summary.capacity  = 20000;
    summary.queued    = 4;
    summary.enabled   = true;
    summary.fgValid   = true;
    summary.state     = dmTiming::Availability::valid;
    for( size_t index = 0; index < summary.metrics.size(); ++index )
    {
        auto &value  = summary.metrics[index];
        value.count  = index + 3;
        value.mean   = ( index + 1 ) * 1e-4;
        value.stddev = ( index + 1 ) * 1e-6;
        value.min    = value.mean - value.stddev;
        value.max    = value.mean + value.stddev;
        value.span   = 1.5;
        value.valid  = true;
    }
    summary.total = { 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112 };
    summary.delta = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    return summary;
}

/// Evaluate a real generated metadata-dispatch accessor using its declared scalar type.
template <typename valueT>
valueT project( const std::string &name /**< [in] metadata member name */,
                void              *payload /**< [in] serialized message */ )
{
    auto detail = logMemberAccessor( telem_dmtimings::eventCode, name );
    REQUIRE( detail.accessor != nullptr );
    REQUIRE( detail.valType == mx::fits::fitsType<valueT>() );
    REQUIRE( detail.metaType == logMeta::metaTypes::State );
    return reinterpret_cast<valueT ( * )( void * )>( detail.accessor )( payload );
}

/// Build a real log envelope for the ordinary verification dispatch.
flatlogs::bufferPtrT envelope( const telem_dmtimings::messageT &message /**< [in] serialized summary */ )
{
    flatlogs::bufferPtrT buffer;
    REQUIRE( flatlogs::logHeader::createLog<telem_dmtimings>(
                 buffer, { 1, 2 }, message, flatlogs::logPrio::LOG_TELEM ) == 0 );
    return buffer;
}
/// \endcond

/// Preserve all metric populations, quality counters and timestamp/coverage units through binary and JSON paths.
/** \ingroup dmTimingTelemetry_unit_test */
TEST_CASE( "DM timing telemetry preserves controller-local populations", "[dmTimingTelemetry]" )
{
    const auto                summary = sampleSummary();
    telem_dmtimings::messageT message( summary );
    auto                      buffer = envelope( message );
    REQUIRE( telem_dmtimings::verify( buffer, flatlogs::logHeader::msgLen( buffer ) ) );
    REQUIRE( verifyLogEntry( telem_dmtimings::eventCode, buffer.get() ) );
    REQUIRE( eventCodeName( telem_dmtimings::eventCode ) == "telem_dmtimings" );
    auto       *payload = message.builder.GetBufferPointer();
    const auto *record  = GetTelem_dmtimings_fb( payload );
    REQUIRE( record->session() == summary.session );
    REQUIRE( record->sequence() == summary.sequence );
    REQUIRE( record->now_ns() == summary.nowNs );
    REQUIRE( record->newest_ns() == summary.newestNs );
    REQUIRE( record->window_sec() == 2 );
    REQUIRE( record->span_sec() == Approx( 1.8 ) );
    REQUIRE( record->age_sec() == Approx( 0.2 ) );
    REQUIRE( record->capacity() == 20000 );
    REQUIRE( record->queued() == 4 );
    REQUIRE( record->enabled() );
    REQUIRE( record->fg_valid() );
    REQUIRE( record->state() == DmTimingAvailability_valid );
    REQUIRE( record->metrics()->size() == size_t( dmTiming::Metric::count ) );
    for( size_t index = 0; index < summary.metrics.size(); ++index )
    {
        const auto *row      = record->metrics()->Get( index );
        const auto &expected = summary.metrics[index];
        REQUIRE( size_t( row->metric() ) == index );
        REQUIRE( row->count() == expected.count );
        REQUIRE( row->mean() == expected.mean );
        REQUIRE( row->stddev() == expected.stddev );
        REQUIRE( row->min() == expected.min );
        REQUIRE( row->max() == expected.max );
        REQUIRE( row->span() == expected.span );
        REQUIRE( row->valid() );
    }
    const std::string text = telem_dmtimings::msgString( payload, message.builder.GetSize() );
    REQUIRE( text.starts_with( "[dmtimes] session=" ) );
    REQUIRE( text.find( "state=valid" ) != std::string::npos );
    REQUIRE( text.find( "receiptInterval[n=3]=" ) != std::string::npos );
    REQUIRE( text.find( "dropped:112" ) != std::string::npos );
    const auto          json = telem_dmtimings::msgJSON( payload,
                                                message.builder.GetSize(),
                                                reinterpret_cast<const uint8_t *>( telem_dmtimings_bfbs ),
                                                telem_dmtimings_bfbs_len );
    flatbuffers::Parser parser;
    REQUIRE(
        parser.Deserialize( reinterpret_cast<const uint8_t *>( telem_dmtimings_bfbs ), telem_dmtimings_bfbs_len ) );
    REQUIRE( parser.Parse( json.c_str() ) );
    const auto *jsonRecord = GetTelem_dmtimings_fb( parser.builder_.GetBufferPointer() );
    REQUIRE( jsonRecord->session() == summary.session );
    REQUIRE( jsonRecord->metrics()->Get( size_t( dmTiming::Metric::sdk ) )->mean() ==
             Approx( summary.metrics[size_t( dmTiming::Metric::sdk )].mean ) );
    REQUIRE( jsonRecord->total()->dropped() == 112 );
    REQUIRE( jsonRecord->delta()->dropped() == 12 );
}

/// Expose every named metric and both counter populations without interpolating summaries or truncating counters.
/** \ingroup dmTimingTelemetry_unit_test */
TEST_CASE( "DM timing metadata projects metrics and collection quality", "[dmTimingTelemetry]" )
{
    const auto                         summary = sampleSummary();
    telem_dmtimings::messageT          message( summary );
    auto                              *payload = message.builder.GetBufferPointer();
    const std::array<const char *, 12> names{ "receiptInterval",
                                              "sendInterval",
                                              "receiptSend",
                                              "dispatch",
                                              "actuator",
                                              "sdk",
                                              "saturation",
                                              "delta",
                                              "post",
                                              "throughDelta",
                                              "callback",
                                              "receiptCallback" };
    for( size_t index = 0; index < names.size(); ++index )
    {
        const std::string name     = names[index];
        const auto       &expected = summary.metrics[index];
        REQUIRE( project<uint64_t>( name + "_count", payload ) == expected.count );
        REQUIRE( project<bool>( name + "_valid", payload ) );
        REQUIRE( project<double>( name + "_mean", payload ) == expected.mean );
        REQUIRE( project<double>( name + "_stddev", payload ) == expected.stddev );
        REQUIRE( project<double>( name + "_min", payload ) == expected.min );
        REQUIRE( project<double>( name + "_max", payload ) == expected.max );
        REQUIRE( project<double>( name + "_span", payload ) == expected.span );
    }
    REQUIRE( project<uint64_t>( "session", payload ) == summary.session );
    REQUIRE( project<uint64_t>( "sequence", payload ) == summary.sequence );
    REQUIRE( project<int64_t>( "nowNs", payload ) == summary.nowNs );
    REQUIRE( project<int64_t>( "newestNs", payload ) == summary.newestNs );
    REQUIRE( project<double>( "windowSec", payload ) == summary.windowSec );
    REQUIRE( project<double>( "span", payload ) == summary.span );
    REQUIRE( project<double>( "age", payload ) == summary.age );
    REQUIRE( project<uint64_t>( "capacity", payload ) == summary.capacity );
    REQUIRE( project<uint64_t>( "queued", payload ) == summary.queued );
    REQUIRE( project<bool>( "enabled", payload ) );
    REQUIRE( project<bool>( "fgValid", payload ) );
    REQUIRE( project<int>( "state", payload ) == int( dmTiming::Availability::valid ) );
    const std::array<const char *, 12> counters{ "events",
                                                 "receipts",
                                                 "submitted",
                                                 "failed",
                                                 "partial",
                                                 "noSend",
                                                 "synthetic",
                                                 "rejected",
                                                 "duplicates",
                                                 "counterGaps",
                                                 "invalid",
                                                 "dropped" };
    for( size_t index = 0; index < counters.size(); ++index )
    {
        REQUIRE( project<uint64_t>( std::string( "total_" ) + counters[index], payload ) == index + 101 );
        REQUIRE( project<uint64_t>( std::string( "delta_" ) + counters[index], payload ) == index + 1 );
    }
    for( const auto *name : { "absent", "receiptInterval_absent", "total_absent", "delta_absent" } )
        REQUIRE( telem_dmtimings::getAccessor( name ).accessor == nullptr );
}

/// Keep unavailable populations explicit through every availability state while preserving counts and rejecting invalid
/// payloads.
/** \ingroup dmTimingTelemetry_unit_test */
TEST_CASE( "DM timing telemetry distinguishes availability and rejects malformed metrics", "[dmTimingTelemetry]" )
{
    for( const auto state : { dmTiming::Availability::disabled,
                              dmTiming::Availability::warmingUp,
                              dmTiming::Availability::valid,
                              dmTiming::Availability::stale,
                              dmTiming::Availability::unavailable } )
    {
        auto summary    = sampleSummary();
        summary.state   = state;
        summary.fgValid = false;
        auto &sdk       = summary.metrics[size_t( dmTiming::Metric::sdk )];
        sdk.valid       = false;
        sdk.count       = 1;
        telem_dmtimings::messageT message( summary );
        auto                      buffer = envelope( message );
        REQUIRE( verifyLogEntry( telem_dmtimings::eventCode, buffer.get() ) );
        auto *payload = message.builder.GetBufferPointer();
        REQUIRE( project<int>( "state", payload ) == int( state ) );
        REQUIRE_FALSE( project<bool>( "sdk_valid", payload ) );
        REQUIRE( project<uint64_t>( "sdk_count", payload ) == 1 );
        REQUIRE( mx::math::isNan( project<double>( "sdk_mean", payload ) ) );
        REQUIRE( mx::math::isNan( project<double>( "sdk_stddev", payload ) ) );
        REQUIRE( mx::math::isNan( project<double>( "sdk_min", payload ) ) );
        REQUIRE( mx::math::isNan( project<double>( "sdk_max", payload ) ) );
        REQUIRE( mx::math::isNan( project<double>( "sdk_span", payload ) ) );
        REQUIRE( telem_dmtimings::msgString( payload, message.builder.GetSize() ).find( "sdk[n=1]=unavailable" ) !=
                 std::string::npos );
    }
    for( int invalid = 0; invalid < 4; ++invalid )
    {
        INFO( "invalid case " << invalid );
        auto  summary = sampleSummary();
        auto &sdk     = summary.metrics[size_t( dmTiming::Metric::sdk )];
        if( invalid == 0 )
            sdk.count = 1;
        if( invalid == 1 )
            sdk.stddev = std::numeric_limits<double>::quiet_NaN();
        if( invalid == 2 )
            sdk.min = sdk.max + 1;
        if( invalid == 3 )
            summary.state = dmTiming::Availability( 99 );
        telem_dmtimings::messageT message( summary );
        auto                      buffer = envelope( message );
        REQUIRE_FALSE( verifyLogEntry( telem_dmtimings::eventCode, buffer.get() ) );
    }
    telem_dmtimings::messageT message( sampleSummary() );
    auto                      buffer = envelope( message );
    REQUIRE_FALSE( telem_dmtimings::verify( buffer, 1 ) );
    std::memset( flatlogs::logHeader::messageBuffer( buffer ), 0xff, 4 );
    REQUIRE_FALSE( verifyLogEntry( telem_dmtimings::eventCode, buffer.get() ) );
    flatlogs::bufferPtrT absent;
    REQUIRE_FALSE( telem_dmtimings::verify( absent, 0 ) );
    REQUIRE_FALSE( verifyLogEntry( telem_dmtimings::eventCode, nullptr ) );
}

/// Accept reordering by stable IDs while rejecting missing, duplicated and unknown metric identifiers.
/** \ingroup dmTimingTelemetry_unit_test */
TEST_CASE( "DM timing telemetry verifies the stable metric set", "[dmTimingTelemetry]" )
{
    for( int variant = 0; variant < 4; ++variant )
    {
        INFO( "metric-set variant " << variant );
        telem_dmtimings::messageT message( sampleSummary() );
        auto                     &builder = message.builder;
        builder.Clear();
        std::vector<flatbuffers::Offset<DmTimingStats>> rows;
        const size_t                                    count = variant == 1 ? 11 : 12;
        for( size_t index = 0; index < count; ++index )
        {
            size_t id = count - index - 1;
            if( variant == 2 && index == 0 )
                id = 0;
            if( variant == 3 && index == 0 )
                id = 99;
            rows.push_back( CreateDmTimingStats( builder, DmTimingMetric( id ), 3, 1e-4, 1e-6, 9e-5, 2e-4, 1, true ) );
        }
        const auto                metrics = builder.CreateVector( rows );
        const auto                total   = CreateDmTimingCounters( builder );
        const auto                delta   = CreateDmTimingCounters( builder );
        Telem_dmtimings_fbBuilder record( builder );
        record.add_metrics( metrics );
        record.add_total( total );
        record.add_delta( delta );
        builder.Finish( record.Finish() );
        auto buffer = envelope( message );
        REQUIRE( verifyLogEntry( telem_dmtimings::eventCode, buffer.get() ) == ( variant == 0 ) );
        if( variant == 0 )
            REQUIRE( project<double>( "sdk_mean", builder.GetBufferPointer() ) == 1e-4 );
    }
}

} // namespace dmTimingTelemetryTest
} // namespace loggerTest
} // namespace libXWCTest
