/** \file dmTimingTelemetry_test.cpp
 * \brief Controller timing and ordered-primary-excursion telemetry round trips and validity.
 * \ingroup logger_files
 */
#include "../../../tests/testXWC.hpp"
#include "../../app/dev/dmTiming.hpp"
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

    summary.total                   = { 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112 };
    summary.delta                   = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    summary.delivery.enabled        = true;
    summary.delivery.clock          = dmTiming::PublicationClock::realtime;
    summary.delivery.bracketLimitNs = 1000;
    summary.delivery.total          = { 101, 80, 5, 4, 3, 3, 3, 3 };
    summary.delivery.delta          = { 1, 1, 0, 0, 0, 0, 0, 0 };
    summary.metrics[size_t( dmTiming::Metric::deliveryLower )]   = { 20, 3e-6, 1e-7, 2e-6, 4e-6, 1.5, true };
    summary.metrics[size_t( dmTiming::Metric::deliveryUpper )]   = { 20, 3.2e-6, 1e-7, 2.2e-6, 4.2e-6, 1.5, true };
    summary.metrics[size_t( dmTiming::Metric::deliveryBracket )] = { 20, 2e-7, 1e-8, 1e-7, 3e-7, 1.5, true };
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

/// Build a real syncopation log envelope for optional primary pair verification.
flatlogs::bufferPtrT envelope( const telem_dmsynco::messageT &message /**< [in] serialized syncopation summary */ )
{
    flatlogs::bufferPtrT buffer;
    REQUIRE( flatlogs::logHeader::createLog<telem_dmsynco>( buffer, { 1, 2 }, message, flatlogs::logPrio::LOG_TELEM ) ==
             0 );
    return buffer;
}

/// Construct asymmetric directed transitions so serialization cannot exchange their denominators.
dmTiming::Summary pairSummary()
{
    auto  summary      = sampleSummary();
    auto &pairs        = summary.intervalPairs;
    pairs.enabled      = true;
    pairs.intervals    = 12;
    pairs.pairs        = 11;
    pairs.median       = 500e-6;
    pairs.threshold    = 20e-6;
    pairs.afterLate    = { 5, 460e-6, 15e-6, 420e-6, 500e-6, 1.5, true };
    pairs.afterEarly   = { 4, 540e-6, 20e-6, 500e-6, 580e-6, 1.4, true };
    pairs.lateEarlySum = { 3, 1001e-6, 2e-6, 999e-6, 1003e-6, 1.3, true };
    pairs.earlyLateSum = { 2, 998e-6, 3e-6, 995e-6, 1001e-6, 1.2, true };
    return summary;
}

/// Omit one optional or required field from a valid mutable test buffer without bypassing construction checks.
template <typename tableT>
void omitDeliveryField( const tableT          *table /**< [in,out] FlatBuffer table owned by the test builder. */,
                        flatbuffers::voffset_t field /**< [in] Generated vtable field identifier. */ )
{
    auto *data   = reinterpret_cast<uint8_t *>( const_cast<tableT *>( table ) );
    auto *vtable = data - flatbuffers::ReadScalar<flatbuffers::soffset_t>( data );
    REQUIRE( field < flatbuffers::ReadScalar<flatbuffers::voffset_t>( vtable ) );
    flatbuffers::WriteScalar<flatbuffers::voffset_t>( vtable + field, 0 );
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
    REQUIRE( text.find( "sdkTail[n=21]=" ) != std::string::npos );
    REQUIRE( text.find( "sdkService[n=22]=" ) != std::string::npos );
    REQUIRE( text.find( "conversion[n=23]=" ) != std::string::npos );
    REQUIRE( text.find( "sourceCheck[n=24]=" ) != std::string::npos );
    REQUIRE( text.find( "shapePublication[n=25]=" ) != std::string::npos );
    REQUIRE( text.find( "receiptSdk[n=26]=" ) != std::string::npos );
    REQUIRE( text.find( "preparation[n=27]=" ) != std::string::npos );
    REQUIRE( text.find( "deliveryLower[n=20]=" ) != std::string::npos );
    REQUIRE( text.find( "deliveryUpper[n=20]=" ) != std::string::npos );
    REQUIRE( text.find( "deliveryBracket[n=20]=" ) != std::string::npos );
    REQUIRE( text.find( "delivery_quality={enabled:1,clock:realtime,bracketLimit_ns:1000}" ) != std::string::npos );
    REQUIRE( text.find( "delivery_total={attempted:101,qualified:80,unavailable:5,invalid:4,wideBracket:3," ) !=
             std::string::npos );
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
    REQUIRE( jsonRecord->delivery_quality() != nullptr );
    REQUIRE( jsonRecord->delivery_quality()->enabled() );
    REQUIRE( jsonRecord->delivery_quality()->clock() == DmDeliveryClock_realtime );
    REQUIRE( jsonRecord->delivery_quality()->bracket_limit_ns() == 1000 );
    REQUIRE( jsonRecord->delivery_quality()->total()->qualified() == 80 );
    REQUIRE( jsonRecord->delivery_quality()->delta()->qualified() == 1 );

    const auto *publication = jsonRecord->metrics()->Get( size_t( dmTiming::Metric::publicationInterval ) );
    const auto &expected    = summary.metrics[size_t( dmTiming::Metric::publicationInterval )];
    REQUIRE( publication->metric() == DmTimingMetric_publicationInterval );
    REQUIRE( publication->count() == expected.count );
    REQUIRE( publication->mean() == Approx( expected.mean ) );
    REQUIRE( publication->stddev() == Approx( expected.stddev ) );
    REQUIRE( publication->min() == Approx( expected.min ) );
    REQUIRE( publication->max() == Approx( expected.max ) );
    REQUIRE( publication->span() == expected.span );
    REQUIRE( publication->valid() );

    for( const auto metric : { dmTiming::Metric::sdkTail,
                               dmTiming::Metric::sdkService,
                               dmTiming::Metric::conversion,
                               dmTiming::Metric::sourceCheck,
                               dmTiming::Metric::shapePublication,
                               dmTiming::Metric::receiptSdk,
                               dmTiming::Metric::preparation,
                               dmTiming::Metric::deliveryLower,
                               dmTiming::Metric::deliveryUpper,
                               dmTiming::Metric::deliveryBracket } )
    {
        const auto  index = size_t( metric );
        const auto *row   = jsonRecord->metrics()->Get( index );
        REQUIRE( size_t( row->metric() ) == index );
        REQUIRE( row->mean() == Approx( summary.metrics[index].mean ) );
        REQUIRE( row->max() == Approx( summary.metrics[index].max ) );
    }
}

/// Expose every named metric and both counter populations without interpolating summaries or truncating counters.
/** \ingroup dmTimingTelemetry_unit_test */
TEST_CASE( "DM timing metadata projects metrics and collection quality", "[dmTimingTelemetry]" )
{
    const auto                         summary = sampleSummary();
    telem_dmtimings::messageT          message( summary );
    auto                              *payload = message.builder.GetBufferPointer();
    const std::array<const char *, 28> names{ "receiptInterval",
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
                                              "receiptCallback",
                                              "publicationInterval",
                                              "sdkDispatch",
                                              "readerRelease",
                                              "staging",
                                              "publication",
                                              "sdkWait",
                                              "sdkTail",
                                              "sdkService",
                                              "conversion",
                                              "sourceCheck",
                                              "shapePublication",
                                              "receiptSdk",
                                              "preparation",
                                              "deliveryLower",
                                              "deliveryUpper",
                                              "deliveryBracket" };

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

    REQUIRE( project<bool>( "delivery_enabled", payload ) );
    REQUIRE( project<int>( "delivery_clock", payload ) == int( dmTiming::PublicationClock::realtime ) );
    REQUIRE( project<int64_t>( "delivery_bracketLimitNs", payload ) == 1000 );
    const std::array<const char *, 8> deliveryNames{ "attempted",
                                                     "qualified",
                                                     "unavailable",
                                                     "invalid",
                                                     "wideBracket",
                                                     "clockDiscontinuity",
                                                     "futurePublication",
                                                     "identityRejected" };
    const std::array<uint64_t, 8>     totals{ 101, 80, 5, 4, 3, 3, 3, 3 };
    const std::array<uint64_t, 8>     deltas{ 1, 1, 0, 0, 0, 0, 0, 0 };

    for( size_t index = 0; index < deliveryNames.size(); ++index )
    {
        REQUIRE( project<uint64_t>( std::string( "delivery_total_" ) + deliveryNames[index], payload ) ==
                 totals[index] );
        REQUIRE( project<uint64_t>( std::string( "delivery_delta_" ) + deliveryNames[index], payload ) ==
                 deltas[index] );
    }

    for( const auto *name : { "absent",
                              "receiptInterval_absent",
                              "total_absent",
                              "delta_absent",
                              "delivery_absent",
                              "delivery_total_absent",
                              "delivery_delta_absent" } )
    {
        REQUIRE( telem_dmtimings::getAccessor( name ).accessor == nullptr );
    }
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
        {
            sdk.count = 1;
        }

        if( invalid == 1 )
        {
            sdk.stddev = std::numeric_limits<double>::quiet_NaN();
        }

        if( invalid == 2 )
        {
            sdk.min = sdk.max + 1;
        }

        if( invalid == 3 )
        {
            summary.state = dmTiming::Availability( 99 );
        }

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

/// Accept complete legacy/current metric sets in any order while rejecting missing, duplicated and unknown identifiers.
/** \ingroup dmTimingTelemetry_unit_test */
TEST_CASE( "DM timing telemetry verifies the stable metric set", "[dmTimingTelemetry]" )
{
    for( int variant = 0; variant < 22; ++variant )
    {
        INFO( "metric-set variant " << variant );
        telem_dmtimings::messageT message( sampleSummary() );
        auto                     &builder = message.builder;
        builder.Clear();
        std::vector<flatbuffers::Offset<DmTimingStats>> rows;
        const size_t                                    count = variant == 21                  ? 27
                                                                : variant >= 18                ? 28
                                                                : variant == 15                ? 23
                                                                : variant >= 14                ? 24
                                                                : variant == 0 || variant == 3 ? 12
                                                                : variant == 2                 ? 11
                                                                : variant == 6                 ? 14
                                                                : variant == 11                ? 19
                                                                : variant >= 10                ? 20
                                                                : variant >= 7                 ? 18
                                                                                               : 13;

        for( size_t index = 0; index < count; ++index )
        {
            size_t id = count - index - 1;

            if( variant == 3 && index == 0 )
            {
                id = 12; // A new row cannot replace a required legacy row in a twelve-row record.
            }

            if( variant == 4 && index == 0 )
            {
                id = 0;
            }

            if( variant == 5 && index == 0 )
            {
                id = 99;
            }

            if( variant == 8 && index == 0 )
            {
                id = 0;
            }

            if( variant == 9 && index == 0 )
            {
                id = 18;
            }

            if( variant == 12 && index == 0 )
            {
                id = 0;
            }

            if( variant == 13 && index == 0 )
            {
                id = 20;
            }

            if( variant == 16 && index == 0 )
            {
                id = 0;
            }

            if( variant == 17 && index == 0 )
            {
                id = 24;
            }

            if( variant == 19 && index == 0 )
            {
                id = 0;
            }

            if( variant == 20 && index == 0 )
            {
                id = 28;
            }

            if( variant >= 18 && id >= 25 )
            {
                rows.push_back( CreateDmTimingStats( builder, DmTimingMetric( id ) ) );
            }
            else
            {
                rows.push_back(
                    CreateDmTimingStats( builder, DmTimingMetric( id ), 3, 1e-4, 1e-6, 9e-5, 2e-4, 1, true ) );
            }
        }

        const auto                             metrics = builder.CreateVector( rows );
        const auto                             total   = CreateDmTimingCounters( builder );
        const auto                             delta   = CreateDmTimingCounters( builder );
        flatbuffers::Offset<DmDeliveryQuality> quality;

        if( variant >= 18 )
        {
            const auto deliveryTotal = CreateDmDeliveryCounters( builder );
            const auto deliveryDelta = CreateDmDeliveryCounters( builder );
            quality = CreateDmDeliveryQuality( builder, false, DmDeliveryClock_none, 0, deliveryTotal, deliveryDelta );
        }

        Telem_dmtimings_fbBuilder record( builder );
        record.add_metrics( metrics );
        record.add_total( total );
        record.add_delta( delta );

        if( variant >= 18 )
        {
            record.add_delivery_quality( quality );
        }

        builder.Finish( record.Finish() );
        auto buffer = envelope( message );
        REQUIRE( verifyLogEntry( telem_dmtimings::eventCode, buffer.get() ) ==
                 ( variant <= 1 || variant == 7 || variant == 10 || variant == 14 || variant == 18 ) );

        if( variant <= 1 )
        {
            REQUIRE( project<double>( "sdk_mean", builder.GetBufferPointer() ) == 1e-4 );
        }

        if( variant == 0 )
        {
            REQUIRE( project<uint64_t>( "publicationInterval_count", builder.GetBufferPointer() ) == 0 );
            REQUIRE_FALSE( project<bool>( "publicationInterval_valid", builder.GetBufferPointer() ) );

            for( const auto *field : { "mean", "stddev", "min", "max", "span" } )
            {
                REQUIRE( mx::math::isNan(
                    project<double>( std::string( "publicationInterval_" ) + field, builder.GetBufferPointer() ) ) );
            }
        }

        if( variant == 1 )
        {
            REQUIRE( project<double>( "publicationInterval_mean", builder.GetBufferPointer() ) == 1e-4 );
        }

        if( variant <= 1 )
        {
            REQUIRE( project<uint64_t>( "readerRelease_count", builder.GetBufferPointer() ) == 0 );
            REQUIRE_FALSE( project<bool>( "sdkWait_valid", builder.GetBufferPointer() ) );
            REQUIRE( mx::math::isNan( project<double>( "staging_mean", builder.GetBufferPointer() ) ) );
        }

        if( variant == 7 )
        {
            REQUIRE( project<double>( "readerRelease_mean", builder.GetBufferPointer() ) == 1e-4 );
            REQUIRE( project<double>( "sdkWait_mean", builder.GetBufferPointer() ) == 1e-4 );
        }

        if( variant <= 1 || variant == 7 )
        {
            REQUIRE( project<uint64_t>( "sdkTail_count", builder.GetBufferPointer() ) == 0 );
            REQUIRE_FALSE( project<bool>( "sdkService_valid", builder.GetBufferPointer() ) );
            REQUIRE( mx::math::isNan( project<double>( "sdkTail_mean", builder.GetBufferPointer() ) ) );
        }

        if( variant == 10 )
        {
            REQUIRE( project<double>( "sdkTail_mean", builder.GetBufferPointer() ) == 1e-4 );
            REQUIRE( project<double>( "sdkService_mean", builder.GetBufferPointer() ) == 1e-4 );
        }

        if( variant <= 1 || variant == 7 || variant == 10 || variant == 14 )
        {
            for( const auto *name : { "conversion", "sourceCheck", "shapePublication", "receiptSdk" } )
            {
                const std::string prefix = std::string( name ) + '_';
                REQUIRE( project<uint64_t>( prefix + "count", builder.GetBufferPointer() ) ==
                         ( variant == 14 ? 3 : 0 ) );
                REQUIRE( project<bool>( prefix + "valid", builder.GetBufferPointer() ) == ( variant == 14 ) );

                for( const auto *field : { "mean", "stddev", "min", "max", "span" } )
                {
                    const double value = project<double>( prefix + field, builder.GetBufferPointer() );

                    if( variant == 14 )
                    {
                        REQUIRE( mx::math::isFinite( value ) );
                    }
                    else
                    {
                        REQUIRE( mx::math::isNan( value ) );
                    }
                }
            }
        }
    }
}

/// Distinguish disabled, sparse and declared realtime/TAI delivery captures while preserving exact units.
/** \ingroup dmTimingTelemetry_unit_test */
TEST_CASE( "DM delivery telemetry preserves sparse and clock qualification", "[dmTimingTelemetry][delivery]" )
{
    for( unsigned variant = 0; variant < 4; ++variant )
    {
        auto summary = sampleSummary();

        if( variant == 0 )
        {
            summary.delivery = {};
        }
        else
        {
            summary.delivery.clock =
                variant == 3 ? dmTiming::PublicationClock::tai : dmTiming::PublicationClock::realtime;
        }

        for( const auto metric :
             { dmTiming::Metric::deliveryLower, dmTiming::Metric::deliveryUpper, dmTiming::Metric::deliveryBracket } )
        {
            auto &value = summary.metrics[size_t( metric )];

            if( variant <= 1 )
            {
                value = {};
            }
            else if( variant == 2 )
            {
                value.count = 1;
                value.min = value.max = value.mean;
                value.stddev = value.span = 0;
                value.valid               = false;
            }
        }

        telem_dmtimings::messageT message( summary );
        auto                      buffer = envelope( message );
        REQUIRE( verifyLogEntry( telem_dmtimings::eventCode, buffer.get() ) );
        const auto *payload = message.builder.GetBufferPointer();
        const auto *record  = GetTelem_dmtimings_fb( payload );
        REQUIRE( record->delivery_quality() != nullptr );
        REQUIRE( record->delivery_quality()->enabled() == ( variant != 0 ) );
        REQUIRE( int( record->delivery_quality()->clock() ) == int( summary.delivery.clock ) );
        REQUIRE( record->metrics()->Get( 25 )->count() == ( variant < 2 ? 0 : variant == 2 ? 1 : 20 ) );
        const auto text = telem_dmtimings::msgString( const_cast<uint8_t *>( payload ), message.builder.GetSize() );
        REQUIRE( text.find( "delivery_quality=unavailable" ) == std::string::npos );

        if( variant == 2 )
        {
            REQUIRE( text.find( "deliveryLower[n=1]=unavailable" ) != std::string::npos );
            REQUIRE_FALSE( project<bool>( "deliveryLower_valid", const_cast<uint8_t *>( payload ) ) );
            REQUIRE( project<uint64_t>( "deliveryLower_count", const_cast<uint8_t *>( payload ) ) == 1 );
        }

        const auto          json = telem_dmtimings::msgJSON( const_cast<uint8_t *>( payload ),
                                                    message.builder.GetSize(),
                                                    reinterpret_cast<const uint8_t *>( telem_dmtimings_bfbs ),
                                                    telem_dmtimings_bfbs_len );
        flatbuffers::Parser parser;
        REQUIRE(
            parser.Deserialize( reinterpret_cast<const uint8_t *>( telem_dmtimings_bfbs ), telem_dmtimings_bfbs_len ) );
        REQUIRE( parser.Parse( json.c_str() ) );
        REQUIRE( GetTelem_dmtimings_fb( parser.builder_.GetBufferPointer() )->delivery_quality()->enabled() ==
                 ( variant != 0 ) );
    }
}

/// Serialize real consumer summaries so rejected captures and qualified endpoint populations remain consistent.
/** \ingroup dmTimingTelemetry_unit_test */
TEST_CASE( "DM delivery telemetry accepts real engine qualification and expiration", "[dmTimingTelemetry][delivery]" )
{
    using namespace MagAOX::dmTiming;
    MagAOX::app::dev::dmTiming timing;
    timing.configure( 2, 32, 0, PublicationClock::realtime, 1000 );
    constexpr int64_t base = 1000000000;

    for( uint64_t index = 0; index < 5; ++index )
    {
        Event event;
        event.sequence = event.counter = index + 1;
        event.session                  = 1;
        event.counterValid = event.submitted = true;
        event.outcome                        = Outcome::success;
        const int64_t receipt                = base + int64_t( index ) * 500000;
        event.publicationClock               = PublicationClock::realtime;
        event.receiptExternalNs              = 1792000000000000000LL + receipt;
        event.receiptAfterNs                 = receipt + ( index == 2 ? 2000 : 200 );
        event.publicationNs                  = event.receiptExternalNs - ( index == 1 ? 100 : 4000 );

        for( const auto &[marker, offset] :
             std::array<std::pair<Marker, int64_t>, 6>{ { { Marker::receipt, 0 },
                                                          { Marker::entry, 1000 },
                                                          { Marker::dispatch, 12000 },
                                                          { Marker::sdkStart, 16000 },
                                                          { Marker::sdkEnd, 120000 },
                                                          { Marker::finish, 140000 } } } )
        {
            MagAOX::app::dev::dmTiming::markEvent( event, marker, receipt + offset );
        }

        timing.submitEvent( event );
    }

    timing.update( base + 2500000 );
    const auto &summary = timing.summary();
    REQUIRE( summary.delivery.total.attempted == 5 );
    REQUIRE( summary.delivery.total.qualified == 3 );
    REQUIRE( summary.delivery.total.futurePublication == 1 );
    REQUIRE( summary.delivery.total.wideBracket == 1 );
    REQUIRE( summary.metrics[size_t( Metric::preparation )].count == 5 );
    REQUIRE( summary.metrics[size_t( Metric::preparation )].mean == Approx( 12e-6 ) );
    REQUIRE( summary.metrics[size_t( Metric::deliveryLower )].mean == Approx( 3.8e-6 ) );
    REQUIRE( summary.metrics[size_t( Metric::deliveryUpper )].mean == Approx( 4e-6 ) );
    REQUIRE( summary.metrics[size_t( Metric::deliveryBracket )].mean == Approx( 0.2e-6 ) );
    telem_dmtimings::messageT message( summary );
    auto                      buffer = envelope( message );
    REQUIRE( verifyLogEntry( telem_dmtimings::eventCode, buffer.get() ) );
    const auto          json = telem_dmtimings::msgJSON( message.builder.GetBufferPointer(),
                                                message.builder.GetSize(),
                                                reinterpret_cast<const uint8_t *>( telem_dmtimings_bfbs ),
                                                telem_dmtimings_bfbs_len );
    flatbuffers::Parser parser;
    REQUIRE(
        parser.Deserialize( reinterpret_cast<const uint8_t *>( telem_dmtimings_bfbs ), telem_dmtimings_bfbs_len ) );
    REQUIRE( parser.Parse( json.c_str() ) );
    REQUIRE( GetTelem_dmtimings_fb( parser.builder_.GetBufferPointer() )->delivery_quality()->total()->qualified() ==
             3 );
    timing.update( base + 3000000000LL );
    telem_dmtimings::messageT expired( timing.summary() );
    buffer = envelope( expired );
    REQUIRE( verifyLogEntry( telem_dmtimings::eventCode, buffer.get() ) );
    REQUIRE( timing.summary().metrics[size_t( Metric::deliveryLower )].count == 0 );
    REQUIRE( timing.summary().delivery.total.qualified == 3 );
    REQUIRE( timing.summary().delivery.delta.attempted == 0 );

    // clang-format off
#ifdef DM_TIMING_TEST_DOXYGEN_REF
    MagAOX::app::dev::dmTiming::configure;
    MagAOX::app::dev::dmTiming::submitEvent;
    MagAOX::app::dev::dmTiming::update;
#endif
    // clang-format on
}

/// Reject contradictory capture outcomes, unpaired delivery populations and impossible clock brackets.
/** \ingroup dmTimingTelemetry_unit_test */
TEST_CASE( "DM delivery telemetry rejects malformed qualification", "[dmTimingTelemetry][delivery]" )
{
    for( unsigned variant = 0; variant < 28; ++variant )
    {
        INFO( "delivery malformed variant " << variant );
        auto  summary  = sampleSummary();
        auto &delivery = summary.delivery;
        auto &lower    = summary.metrics[size_t( dmTiming::Metric::deliveryLower )];
        auto &upper    = summary.metrics[size_t( dmTiming::Metric::deliveryUpper )];
        auto &bracket  = summary.metrics[size_t( dmTiming::Metric::deliveryBracket )];

        switch( variant )
        {
        case 0:
            delivery.clock = dmTiming::PublicationClock( 99 );
            break;
        case 1:
            delivery.enabled = false;
            break;
        case 2:
            delivery.clock = dmTiming::PublicationClock::none;
            break;
        case 3:
            delivery.bracketLimitNs = 0;
            break;
        case 4:
            delivery.bracketLimitNs = -1;
            break;
        case 5:
            delivery.bracketLimitNs = 1000000001;
            break;
        case 6:
            ++delivery.total.attempted;
            break;
        case 7:
            ++delivery.delta.attempted;
            break;
        case 8:
            delivery.total.qualified = std::numeric_limits<uint64_t>::max();
            break;
        case 9:
            delivery.delta = delivery.total;
            break;
        case 10:
            delivery.delta                  = { 1, 0, 0, 0, 0, 0, 0, 1 };
            delivery.total.identityRejected = 0;
            ++delivery.total.qualified;
            break;
        case 11:
            --lower.count;
            break;
        case 12:
            upper.span = 1.4;
            break;
        case 13:
            bracket.valid = false;
            break;
        case 14:
            delivery.total.unavailable += delivery.total.qualified;
            delivery.total.qualified = 0;
            break;
        case 15:
            upper.mean -= 1e-6;
            upper.min -= 1e-6;
            upper.max -= 1e-6;
            break;
        case 16:
            lower.mean += 1e-6;
            lower.min += 1e-6;
            lower.max += 1e-6;
            break;
        case 17:
            bracket.span = 1.4;
            break;
        case 18:
            bracket.max = 2e-6;
            break;
        case 19:
            bracket.mean = bracket.min = 1e-7;
            break;
        case 20:
            upper.max = 5e-6;
            break;
        case 21:
            upper.min = 2.5e-6;
            break;
        case 22:
            lower.min = -1e-9;
            break;
        case 23:
            bracket.mean = std::numeric_limits<double>::quiet_NaN();
            break;
        case 24:
            lower.count = upper.count = bracket.count = 1;
            lower.valid = upper.valid = bracket.valid = false;
            break;
        case 25:
            lower.count = upper.count = bracket.count = 0;
            lower.valid = upper.valid = bracket.valid = false;
            break;
        case 26:
            delivery.total.attempted += 100;
            delivery.total.unavailable += 100;
            break;
        case 27:
            delivery                = {};
            delivery.bracketLimitNs = 1000;
            lower = upper = bracket = {};
            break;
        }

        telem_dmtimings::messageT message( summary );
        auto                      buffer = envelope( message );
        REQUIRE_FALSE( verifyLogEntry( telem_dmtimings::eventCode, buffer.get() ) );
    }
}

/// Historical metric sets omit delivery qualification; current records require all nested quality tables.
/** \ingroup dmTimingTelemetry_unit_test */
TEST_CASE( "DM delivery telemetry distinguishes missing tables from legacy records", "[dmTimingTelemetry][delivery]" )
{
    for( unsigned variant = 0; variant < 3; ++variant )
    {
        telem_dmtimings::messageT message( sampleSummary() );
        const auto               *record  = GetTelem_dmtimings_fb( message.builder.GetBufferPointer() );
        const auto               *quality = record->delivery_quality();

        if( variant == 0 )
        {
            omitDeliveryField( record, Telem_dmtimings_fb::VT_DELIVERY_QUALITY );
        }
        else
        {
            omitDeliveryField( quality, variant == 1 ? DmDeliveryQuality::VT_TOTAL : DmDeliveryQuality::VT_DELTA );
        }

        auto buffer = envelope( message );
        REQUIRE_FALSE( verifyLogEntry( telem_dmtimings::eventCode, buffer.get() ) );
    }

    for( const auto count : { 12, 13, 18, 20, 24 } )
    {
        telem_dmtimings::messageT message( sampleSummary() );
        auto                     &builder = message.builder;
        builder.Clear();
        std::vector<flatbuffers::Offset<DmTimingStats>> rows;

        for( int index = 0; index < count; ++index )
        {
            rows.push_back( CreateDmTimingStats( builder, DmTimingMetric( index ) ) );
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
        REQUIRE( verifyLogEntry( telem_dmtimings::eventCode, buffer.get() ) );
        REQUIRE( telem_dmtimings::msgString( builder.GetBufferPointer(), builder.GetSize() )
                     .find( "delivery_quality=unavailable" ) != std::string::npos );
        REQUIRE_FALSE( project<bool>( "delivery_enabled", builder.GetBufferPointer() ) );
        REQUIRE( project<uint64_t>( "delivery_total_attempted", builder.GetBufferPointer() ) == 0 );
        REQUIRE( project<uint64_t>( "preparation_count", builder.GetBufferPointer() ) == 0 );
        REQUIRE( mx::math::isNan( project<double>( "deliveryUpper_max", builder.GetBufferPointer() ) ) );
    }
}

/// Preserve directed transition counts, conditional intervals and two-interval sums through binary, text and JSON.
/** \ingroup dmTimingTelemetry_unit_test */
TEST_CASE( "Syncopation telemetry preserves ordered primary excursions", "[dmTimingTelemetry]" )
{
    const auto              summary = pairSummary();
    telem_dmsynco::messageT message( 0, 2, 500000, 0, 0, 0, 0, 0, 0, {}, {}, summary );
    auto                    buffer = envelope( message );
    REQUIRE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
    auto       *payload = message.builder.GetBufferPointer();
    const auto *primary = GetTelem_dmsynco_fb( payload )->primary_timing();
    const auto *pairs   = primary->interval_pairs();
    REQUIRE( pairs != nullptr );
    REQUIRE( pairs->enabled() );
    REQUIRE( pairs->intervals() == 12 );
    REQUIRE( pairs->pairs() == 11 );
    REQUIRE( pairs->median() == 500e-6 );
    REQUIRE( pairs->threshold() == 20e-6 );
    REQUIRE( primary->session() == summary.session );
    REQUIRE( primary->dropped() == summary.total.dropped );

    const std::array<const DmSyncoTimingStats *, 4> populations{
        pairs->after_late(), pairs->after_early(), pairs->late_early_sum(), pairs->early_late_sum() };
    const auto                                  &expected = summary.intervalPairs;
    const std::array<const dmTiming::Stats *, 4> source{
        &expected.afterLate, &expected.afterEarly, &expected.lateEarlySum, &expected.earlyLateSum };

    for( size_t index = 0; index < populations.size(); ++index )
    {
        const auto *actual = populations[index];
        const auto *value  = source[index];
        REQUIRE( actual->count() == value->count );
        REQUIRE( actual->mean() == value->mean );
        REQUIRE( actual->stddev() == value->stddev );
        REQUIRE( actual->min() == value->min );
        REQUIRE( actual->max() == value->max );
        REQUIRE( actual->span() == value->span );
        REQUIRE( actual->valid() );
    }

    const auto text = telem_dmsynco::msgString( payload, message.builder.GetSize() );
    REQUIRE( text.find( "primary_pairs={enabled:1,intervals:12,pairs:11,median_s:0.0005,threshold_s:2e-05," ) !=
             std::string::npos );
    REQUIRE( text.find( "lateEarly:3/5,earlyLate:2/4}" ) != std::string::npos );
    REQUIRE( text.find( "primaryAfterLateInterval[n=5 valid=1] mean_s=0.00046" ) != std::string::npos );
    REQUIRE( text.find( "primaryAfterEarlyInterval[n=4 valid=1] mean_s=0.00054" ) != std::string::npos );
    REQUIRE( text.find( "primaryLateEarlySum[n=3 valid=1] mean_s=0.001001" ) != std::string::npos );
    REQUIRE( text.find( "primaryEarlyLateSum[n=2 valid=1] mean_s=0.000998" ) != std::string::npos );

    const auto          json = telem_dmsynco::msgJSON( payload,
                                              message.builder.GetSize(),
                                              reinterpret_cast<const uint8_t *>( telem_dmsynco_bfbs ),
                                              telem_dmsynco_bfbs_len );
    flatbuffers::Parser parser;
    REQUIRE( parser.Deserialize( reinterpret_cast<const uint8_t *>( telem_dmsynco_bfbs ), telem_dmsynco_bfbs_len ) );
    REQUIRE( parser.Parse( json.c_str() ) );
    const auto *jsonPairs =
        GetTelem_dmsynco_fb( parser.builder_.GetBufferPointer() )->primary_timing()->interval_pairs();
    REQUIRE( jsonPairs->median() == Approx( expected.median ) );
    REQUIRE( jsonPairs->threshold() == Approx( expected.threshold ) );
    REQUIRE( jsonPairs->after_late()->count() == 5 );
    REQUIRE( jsonPairs->after_early()->count() == 4 );
    REQUIRE( jsonPairs->late_early_sum()->count() == 3 );
    REQUIRE( jsonPairs->early_late_sum()->count() == 2 );
    REQUIRE( jsonPairs->late_early_sum()->mean() == Approx( expected.lateEarlySum.mean ) );
    REQUIRE( jsonPairs->early_late_sum()->mean() == Approx( expected.earlyLateSum.mean ) );
}

/// Distinguish old absent fields, disabled analysis, enabled empty windows and single-sample conditional statistics.
/** \ingroup dmTimingTelemetry_unit_test */
TEST_CASE( "Syncopation primary pair telemetry preserves legacy and sparse availability", "[dmTimingTelemetry]" )
{
    for( int variant = 0; variant < 4; ++variant )
    {
        INFO( "availability variant " << variant );
        dmTiming::Summary summary;

        if( variant >= 1 )
        {
            summary.intervalPairs.enabled   = true;
            summary.intervalPairs.threshold = 20e-6;
        }

        if( variant >= 2 )
        {
            auto &pairs        = summary.intervalPairs;
            pairs.intervals    = 2;
            pairs.pairs        = 1;
            pairs.median       = 500e-6;
            pairs.afterLate    = { 1, 420e-6, 0, 420e-6, 420e-6, 0, false };
            pairs.lateEarlySum = { 1, 1000e-6, 0, 1000e-6, 1000e-6, 0, false };
        }

        telem_dmsynco::messageT message( 0, 2, 500000, 0, 0, 0, 0, 0, 0, {}, {}, summary );

        if( variant == 3 )
        {
            auto &builder = message.builder;
            builder.Clear();
            const auto                  publication = CreateDmSyncoTimingStats( builder );
            const auto                  receipt     = CreateDmSyncoTimingStats( builder );
            DmSyncoPrimaryTimingBuilder primary( builder );
            primary.add_publication( publication );
            primary.add_receipt( receipt );
            const auto              primaryTiming = primary.Finish();
            Telem_dmsynco_fbBuilder record( builder );
            record.add_primary_timing( primaryTiming );
            builder.Finish( record.Finish() );
        }

        auto buffer = envelope( message );
        REQUIRE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
        auto       *payload = message.builder.GetBufferPointer();
        const auto *primary = GetTelem_dmsynco_fb( payload )->primary_timing();
        const auto  text    = telem_dmsynco::msgString( payload, message.builder.GetSize() );

        if( variant == 3 )
        {
            REQUIRE( primary->interval_pairs() == nullptr );
            REQUIRE( text.find( "primary_pairs=unavailable" ) != std::string::npos );
        }
        else
        {
            REQUIRE( primary->interval_pairs() != nullptr );
            REQUIRE( primary->interval_pairs()->enabled() == ( variant != 0 ) );
            REQUIRE( text.find( "primary_pairs=unavailable" ) == std::string::npos );

            if( variant == 2 )
            {
                REQUIRE( text.find( "lateEarly:1/1,earlyLate:0/0}" ) != std::string::npos );
                REQUIRE( text.find( "primaryAfterLateInterval[n=1 valid=0] mean_s=0.00042 stddev_s=unavailable" ) !=
                         std::string::npos );
            }
            else
            {
                REQUIRE( text.find( "primaryAfterLateInterval[n=0 valid=0]=unavailable" ) != std::string::npos );
            }
        }
    }
}

/// Reject impossible directed counts and nonfinite or inconsistent classification parameters.
/** \ingroup dmTimingTelemetry_unit_test */
TEST_CASE( "Syncopation primary pair telemetry rejects malformed populations", "[dmTimingTelemetry]" )
{
    for( int variant = 0; variant < 15; ++variant )
    {
        INFO( "malformed pair variant " << variant );
        auto  summary = pairSummary();
        auto &pairs   = summary.intervalPairs;

        switch( variant )
        {
        case 0:
            pairs.pairs = pairs.intervals;
            break;
        case 1:
            pairs.intervals = 0;
            break;
        case 2:
            pairs.afterLate.count = pairs.pairs + 1;
            break;
        case 3:
            pairs.afterEarly.count = pairs.pairs;
            break;
        case 4:
            pairs.lateEarlySum.count = pairs.afterLate.count + 1;
            break;
        case 5:
            pairs.earlyLateSum.count = pairs.afterEarly.count + 1;
            break;
        case 6:
            pairs.median = std::numeric_limits<double>::quiet_NaN();
            break;
        case 7:
            pairs.threshold = std::numeric_limits<double>::infinity();
            break;
        case 8:
            pairs.median = 0;
            break;
        case 9:
            pairs.threshold = 0;
            break;
        case 10:
            pairs.enabled = false;
            break;
        case 11:
            pairs.afterLate.count = 1;
            break;
        case 12:
            pairs.earlyLateSum.stddev = -1;
            break;
        case 13:
            pairs.lateEarlySum.mean = std::numeric_limits<double>::quiet_NaN();
            break;
        case 14:
            pairs.afterEarly.min = pairs.afterEarly.max + 1;
            break;
        }

        telem_dmsynco::messageT message( 0, 2, 500000, 0, 0, 0, 0, 0, 0, {}, {}, summary );
        auto                    buffer = envelope( message );
        REQUIRE_FALSE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
    }
}

} // namespace dmTimingTelemetryTest
} // namespace loggerTest
} // namespace libXWCTest
