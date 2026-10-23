/** \file dmSyncoTelemetry_test.cpp
 * \brief Syncopation primary/combiner telemetry compatibility, statistics and availability.
 * \ingroup logger_files
 */
#include "../../../tests/testXWC.hpp"
#include "../logBinarySchemata.hpp"
#include "../generated/logMemberAccessor.hpp"

namespace libXWCTest
{
namespace loggerTest
{
/** \defgroup dmSyncoTelemetry_unit_test Syncopation Telemetry Tests
 * \ingroup logger_unit_test
 */
namespace dmSyncoTelemetryTest
{
using namespace MagAOX;
using namespace MagAOX::logger;

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
/// Construct independently valid primary cadence with no SDK-oriented compatible population.
dmTiming::Summary samplePrimary()
{
    dmTiming::Summary primary;
    primary.session                                                                 = ( uint64_t( 1 ) << 54 ) + 5;
    primary.sequence                                                                = 27;
    primary.nowNs                                                                   = 1234567890123;
    primary.windowSec                                                               = 2;
    primary.span                                                                    = 1.6;
    primary.age                                                                     = 0.125;
    primary.capacity                                                                = 20000;
    primary.queued                                                                  = 3;
    primary.enabled                                                                 = true;
    primary.fgValid                                                                 = false;
    primary.metrics[dmTiming::metricIndex( dmTiming::Metric::publicationInterval )] = {
        4, 500e-6, 1e-6, 498e-6, 502e-6, 1.4, true };
    primary.metrics[dmTiming::metricIndex( dmTiming::Metric::receiptInterval )] = {
        3, 510e-6, 2e-6, 507e-6, 513e-6, 1.2, true };
    primary.total.events      = 100;
    primary.total.receipts    = 99;
    primary.total.synthetic   = 1;
    primary.total.rejected    = 2;
    primary.total.duplicates  = 3;
    primary.total.counterGaps = 4;
    primary.total.invalid     = 5;
    primary.total.dropped     = 6;
    return primary;
}

/// Construct six independently populated combiner metrics with full-width trace provenance and quality counts.
dmCombTiming::Summary sampleCombiner()
{
    dmCombTiming::Summary combiner;
    combiner.session        = ( uint64_t( 1 ) << 54 ) + 17;
    combiner.sequence       = 31;
    combiner.nowNs          = 2345678901234;
    combiner.windowSec      = 2;
    combiner.span           = 1.75;
    combiner.age            = 0.25;
    combiner.capacity       = 20000;
    combiner.queued         = 2;
    combiner.enabled        = true;
    combiner.writerPid      = 12345;
    combiner.triggerInode   = ( uint64_t( 1 ) << 56 ) + 9;
    combiner.primaryCounter = ( uint64_t( 1 ) << 55 ) + 7;

    for( size_t index = 0; index < combiner.metrics.size(); ++index )
    {
        auto &stats  = combiner.metrics[index];
        stats.count  = index + 3;
        stats.mean   = ( index + 1 ) * 10e-6;
        stats.stddev = 1e-6;
        stats.min    = stats.mean - 1e-6;
        stats.max    = stats.mean + 1e-6;
        stats.span   = 1.25;
        stats.valid  = true;
    }

    combiner.total = { 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114 };
    return combiner;
}

/// Populate unchanged scheduler fields with independently sampled primary and combiner timing.
telem_dmsynco::messageT message( const dmTiming::Summary     &primary /**< [in] coherent primary summary */,
                                 const dmCombTiming::Summary &combiner = {} /**< [in] independent combiner summary */ )
{
    return telem_dmsynco::messageT(
        3, 2, 500000, 11, 2, 3, 4, 150, 250, { -100, 200 }, { 300, 400 }, primary, true, combiner );
}

/// Build a real log envelope for shared verification dispatch.
flatlogs::bufferPtrT envelope( const telem_dmsynco::messageT &input /**< [in] serialized aggregate */ )
{
    flatlogs::bufferPtrT buffer;
    REQUIRE( flatlogs::logHeader::createLog<telem_dmsynco>( buffer, { 1, 2 }, input, flatlogs::logPrio::LOG_TELEM ) ==
             0 );
    return buffer;
}

/// Check a typed population's count, units and sample-statistic fields after binary or JSON roundtrip.
void checkStats( const DmSyncoTimingStats *actual /**< [in] deserialized timing population */,
                 const dmTiming::Stats    &expected /**< [in] original values in seconds */ )
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

/// Return the six explicitly named combiner populations in their stable metric order.
std::array<const DmSyncoTimingStats *, 6>
combinerStats( const DmSyncoCombinerTiming *timing /**< [in] deserialized combiner summary */ )
{
    return { timing->publication_interval(),
             timing->receipt_interval(),
             timing->execution(),
             timing->publication(),
             timing->primary_start(),
             timing->primary_publication() };
}
/// \endcond

/// Preserve all old scheduler fields and independent primary cadence/quality through binary and JSON serialization.
/** \ingroup dmSyncoTelemetry_unit_test */
TEST_CASE( "Synco telemetry retains scheduler fields and primary cadence", "[dmSyncoTelemetry]" )
{
    const auto primary = samplePrimary();
    auto       input   = message( primary );
    auto       buffer  = envelope( input );
    REQUIRE( telem_dmsynco::verify( buffer, flatlogs::logHeader::msgLen( buffer ) ) );
    REQUIRE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
    auto       *payload = input.builder.GetBufferPointer();
    const auto *record  = GetTelem_dmsynco_fb( payload );
    REQUIRE( record->state() == 3 );
    REQUIRE( record->steps() == 2 );
    REQUIRE( record->period_ns() == 500000 );
    REQUIRE( record->delivered() == 11 );
    REQUIRE( record->missed() == 2 );
    REQUIRE( record->faults() == 3 );
    REQUIRE( record->cached() == 4 );
    REQUIRE( record->snapshot_ns() == 150 );
    REQUIRE( record->copy_ns() == 250 );
    REQUIRE( record->residuals_ns()->Get( 0 ) == -100 );
    REQUIRE( record->residuals_ns()->Get( 1 ) == 200 );
    REQUIRE( record->advances_ns()->Get( 0 ) == 300 );
    REQUIRE( record->advances_ns()->Get( 1 ) == 400 );
    REQUIRE( record->schedule_valid() );
    const auto *timing = record->primary_timing();
    REQUIRE( timing != nullptr );
    REQUIRE( timing->enabled() );
    REQUIRE( timing->valid() );
    REQUIRE_FALSE( primary.fgValid );
    REQUIRE( timing->session() == primary.session );
    REQUIRE( timing->sequence() == primary.sequence );
    REQUIRE( timing->now_ns() == primary.nowNs );
    REQUIRE( timing->window_sec() == primary.windowSec );
    REQUIRE( timing->span_sec() == primary.span );
    REQUIRE( timing->age_sec() == primary.age );
    REQUIRE( timing->capacity() == primary.capacity );
    REQUIRE( timing->queued() == primary.queued );
    REQUIRE( timing->events() == 100 );
    REQUIRE( timing->receipts() == 99 );
    REQUIRE( timing->synthetic() == 1 );
    REQUIRE( timing->rejected() == 2 );
    REQUIRE( timing->duplicates() == 3 );
    REQUIRE( timing->counter_gaps() == 4 );
    REQUIRE( timing->invalid() == 5 );
    REQUIRE( timing->dropped() == 6 );
    checkStats( timing->publication(),
                primary.metrics[dmTiming::metricIndex( dmTiming::Metric::publicationInterval )] );
    checkStats( timing->receipt(), primary.metrics[dmTiming::metricIndex( dmTiming::Metric::receiptInterval )] );

    const auto text = telem_dmsynco::msgString( payload, input.builder.GetSize() );
    REQUIRE( text.starts_with( "[dmsynco] state=3 steps=2 period_ns=500000 delivered=11 missed=2 faults=3" ) );
    REQUIRE( text.find( "schedule_valid=1" ) != std::string::npos );
    REQUIRE( text.find( "primary_valid=1" ) != std::string::npos );
    REQUIRE( text.find( "primaryPublicationInterval[n=4 valid=1] mean_s=" ) != std::string::npos );
    REQUIRE( text.find( "primaryReceiptInterval[n=3 valid=1] mean_s=" ) != std::string::npos );
    REQUIRE( text.find( "stddev_s=" ) != std::string::npos );
    REQUIRE( text.find( "min_s=" ) != std::string::npos );
    REQUIRE( text.find( "max_s=" ) != std::string::npos );
    REQUIRE( text.find( "span_s=" ) != std::string::npos );
    REQUIRE( text.find( "dropped:6" ) != std::string::npos );

    const auto          json = telem_dmsynco::msgJSON( payload,
                                              input.builder.GetSize(),
                                              reinterpret_cast<const uint8_t *>( telem_dmsynco_bfbs ),
                                              telem_dmsynco_bfbs_len );
    flatbuffers::Parser parser;
    REQUIRE( parser.Deserialize( reinterpret_cast<const uint8_t *>( telem_dmsynco_bfbs ), telem_dmsynco_bfbs_len ) );
    REQUIRE( parser.Parse( json.c_str() ) );
    const auto *jsonRecord = GetTelem_dmsynco_fb( parser.builder_.GetBufferPointer() );
    REQUIRE( jsonRecord->schedule_valid() );
    REQUIRE( jsonRecord->primary_timing()->session() == primary.session );
    REQUIRE( jsonRecord->primary_timing()->valid() );
    checkStats( jsonRecord->primary_timing()->publication(),
                primary.metrics[dmTiming::metricIndex( dmTiming::Metric::publicationInterval )] );
    checkStats( jsonRecord->primary_timing()->receipt(),
                primary.metrics[dmTiming::metricIndex( dmTiming::Metric::receiptInterval )] );
    REQUIRE( logMemberAccessor( telem_dmsynco::eventCode, "primary_timing" ).accessor == nullptr );
}

/// Accept genuine old payloads without the new table and keep eleven-argument construction source-compatible.
/** \ingroup dmSyncoTelemetry_unit_test */
TEST_CASE( "Synco telemetry accepts legacy records and defaults primary timing disabled", "[dmSyncoTelemetry]" )
{
    telem_dmsynco::messageT input( 3, 2, 500000, 11, 2, 3, 4, 150, 250, { -100, 200 }, { 300, 400 } );
    auto                    buffer = envelope( input );
    REQUIRE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
    const auto *record = GetTelem_dmsynco_fb( input.builder.GetBufferPointer() );
    REQUIRE( record->schedule_valid() );
    REQUIRE( record->primary_timing() != nullptr );
    REQUIRE_FALSE( record->primary_timing()->enabled() );
    REQUIRE_FALSE( record->primary_timing()->valid() );
    REQUIRE( record->primary_timing()->publication()->count() == 0 );
    REQUIRE( record->primary_timing()->receipt()->count() == 0 );
    REQUIRE( record->primary_timing()->age_sec() == -1 );
    REQUIRE( record->combiner_timing() != nullptr );
    REQUIRE_FALSE( record->combiner_timing()->enabled() );
    REQUIRE( record->combiner_timing()->execution()->count() == 0 );
    REQUIRE( telem_dmsynco::msgString( input.builder.GetBufferPointer(), input.builder.GetSize() )
                 .find( "primaryPublicationInterval[n=0 valid=0]=unavailable" ) != std::string::npos );

    input.builder.Clear();
    input.builder.ForceDefaults( false ); // Omit appended defaulted fields as the original writer did.
    const auto residuals = input.builder.CreateVector( std::vector<int64_t>{ -100, 200 } );
    const auto advances  = input.builder.CreateVector( std::vector<int64_t>{ 300, 400 } );
    input.builder.Finish(
        CreateTelem_dmsynco_fb( input.builder, 3, 2, 500000, 11, 2, 3, 4, 150, 250, residuals, advances ) );
    auto legacy = envelope( input );
    REQUIRE( verifyLogEntry( telem_dmsynco::eventCode, legacy.get() ) );
    record = GetTelem_dmsynco_fb( input.builder.GetBufferPointer() );
    REQUIRE( record->primary_timing() == nullptr );
    REQUIRE_FALSE( flatbuffers::IsFieldPresent( record, Telem_dmsynco_fb::VT_PRIMARY_TIMING ) );
    REQUIRE_FALSE( flatbuffers::IsFieldPresent( record, Telem_dmsynco_fb::VT_SCHEDULE_VALID ) );
    REQUIRE_FALSE( flatbuffers::IsFieldPresent( record, Telem_dmsynco_fb::VT_COMBINER_TIMING ) );
    REQUIRE( record->combiner_timing() == nullptr );
    REQUIRE( record->schedule_valid() );
    REQUIRE( record->residuals_ns()->Get( 0 ) == -100 );
    REQUIRE( record->advances_ns()->Get( 1 ) == 400 );
    REQUIRE( telem_dmsynco::msgString( input.builder.GetBufferPointer(), input.builder.GetSize() ) ==
             "[dmsynco] state=3 steps=2 period_ns=500000 delivered=11 missed=2 faults=3 schedule_valid=1 "
             "primary_timing=unavailable combiner_timing=unavailable" );
}

/// Keep primary availability independent of SDK populations and unavailable scheduler snapshots.
/** \ingroup dmSyncoTelemetry_unit_test */
TEST_CASE( "Synco telemetry separates primary validity from scheduler availability", "[dmSyncoTelemetry]" )
{
    const auto              primary = samplePrimary();
    telem_dmsynco::messageT input( 0, 0, 0, 0, 0, 0, 0, 0, 0, {}, {}, primary, false );
    auto                    buffer = envelope( input );
    REQUIRE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
    const auto *record = GetTelem_dmsynco_fb( input.builder.GetBufferPointer() );
    REQUIRE_FALSE( record->schedule_valid() );
    REQUIRE( record->primary_timing()->valid() );
    REQUIRE( record->primary_timing()->enabled() );
    REQUIRE( record->primary_timing()->events() == 100 );
    REQUIRE( telem_dmsynco::msgString( input.builder.GetBufferPointer(), input.builder.GetSize() )
                 .find( "schedule_valid=0 primary_enabled=1 primary_valid=1" ) != std::string::npos );

    auto incomplete    = primary;
    incomplete.fgValid = true;
    auto &receipt      = incomplete.metrics[dmTiming::metricIndex( dmTiming::Metric::receiptInterval )];
    receipt.count      = 1;
    receipt.valid      = false;
    auto warming       = message( incomplete );
    auto warmingBuffer = envelope( warming );
    REQUIRE( verifyLogEntry( telem_dmsynco::eventCode, warmingBuffer.get() ) );
    REQUIRE_FALSE( GetTelem_dmsynco_fb( warming.builder.GetBufferPointer() )->primary_timing()->valid() );
    REQUIRE( telem_dmsynco::msgString( warming.builder.GetBufferPointer(), warming.builder.GetSize() )
                 .find( "primaryReceiptInterval[n=1 valid=0] mean_s=0.00051 stddev_s=unavailable" ) !=
             std::string::npos );
}

/// Reject corrupt nested cadence populations, inconsistent availability and malformed buffers without changing old
/// logs.
/** \ingroup dmSyncoTelemetry_unit_test */
TEST_CASE( "Synco telemetry rejects invalid primary populations", "[dmSyncoTelemetry]" )
{
    for( unsigned invalid = 0; invalid < 8; ++invalid )
    {
        INFO( "invalid primary case " << invalid );
        auto  primary     = samplePrimary();
        auto &publication = primary.metrics[dmTiming::metricIndex( dmTiming::Metric::publicationInterval )];
        auto &receipt     = primary.metrics[dmTiming::metricIndex( dmTiming::Metric::receiptInterval )];

        if( invalid == 0 )
        {
            publication.count = 1;
        }

        if( invalid == 1 )
        {
            publication.stddev = std::numeric_limits<double>::quiet_NaN();
        }

        if( invalid == 2 )
        {
            publication.mean = std::numeric_limits<double>::quiet_NaN();
        }

        if( invalid == 3 )
        {
            receipt.min = receipt.max + 1;
        }

        if( invalid == 4 )
        {
            receipt.span = -1;
        }

        if( invalid == 5 )
        {
            primary.windowSec = std::numeric_limits<double>::infinity();
        }

        if( invalid == 6 )
        {
            primary.age = -2;
        }

        if( invalid == 7 )
        {
            publication.count = 1;
            publication.valid = false;
            publication.mean  = std::numeric_limits<double>::infinity();
        }

        auto input  = message( primary );
        auto buffer = envelope( input );
        REQUIRE_FALSE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
    }

    auto input = message( samplePrimary() );
    input.builder.Clear();
    const auto publication = CreateDmSyncoTimingStats( input.builder, 3, 500e-6, 1e-6, 499e-6, 501e-6, 1, true );
    const auto receipt     = CreateDmSyncoTimingStats( input.builder, 3, 500e-6, 1e-6, 499e-6, 501e-6, 1, true );
    DmSyncoPrimaryTimingBuilder primary( input.builder );
    primary.add_publication( publication );
    primary.add_receipt( receipt );
    const auto primaryTiming = primary.Finish(); // Both populations valid, but aggregate valid flag is false.
    Telem_dmsynco_fbBuilder record( input.builder );
    record.add_primary_timing( primaryTiming );
    input.builder.Finish( record.Finish() );
    auto inconsistent = envelope( input );
    REQUIRE_FALSE( verifyLogEntry( telem_dmsynco::eventCode, inconsistent.get() ) );

    auto valid  = message( samplePrimary() );
    auto buffer = envelope( valid );
    REQUIRE_FALSE( telem_dmsynco::verify( buffer, 1 ) );
    std::memset( flatlogs::logHeader::messageBuffer( buffer ), 0xff, 4 );
    REQUIRE_FALSE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
    flatlogs::bufferPtrT absent;
    REQUIRE_FALSE( telem_dmsynco::verify( absent, 0 ) );
    REQUIRE_FALSE( verifyLogEntry( telem_dmsynco::eventCode, nullptr ) );
}

/// Preserve every independently qualified combiner population, provenance identifier and quality count in binary/JSON.
/** \ingroup dmSyncoTelemetry_unit_test */
TEST_CASE( "Synco telemetry preserves combiner statistics and provenance", "[dmSyncoTelemetry]" )
{
    const auto combiner = sampleCombiner();
    auto       input    = message( samplePrimary(), combiner );
    auto       buffer   = envelope( input );
    REQUIRE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
    auto       *payload = input.builder.GetBufferPointer();
    const auto *record  = GetTelem_dmsynco_fb( payload );
    const auto *timing  = record->combiner_timing();
    REQUIRE( record->schedule_valid() );
    REQUIRE( record->primary_timing()->valid() );
    REQUIRE( timing != nullptr );
    REQUIRE( timing->enabled() );
    REQUIRE( timing->session() == combiner.session );
    REQUIRE( timing->sequence() == combiner.sequence );
    REQUIRE( timing->now_ns() == combiner.nowNs );
    REQUIRE( timing->window_sec() == combiner.windowSec );
    REQUIRE( timing->span_sec() == combiner.span );
    REQUIRE( timing->age_sec() == combiner.age );
    REQUIRE( timing->capacity() == combiner.capacity );
    REQUIRE( timing->queued() == combiner.queued );
    REQUIRE( timing->writer_pid() == combiner.writerPid );
    REQUIRE( timing->trigger_inode() == combiner.triggerInode );
    REQUIRE( timing->primary_counter() == combiner.primaryCounter );
    REQUIRE( timing->events() == 101 );
    REQUIRE( timing->receipts() == 102 );
    REQUIRE( timing->synthetic() == 103 );
    REQUIRE( timing->rejected() == 104 );
    REQUIRE( timing->duplicates() == 105 );
    REQUIRE( timing->counter_gaps() == 106 );
    REQUIRE( timing->invalid_metadata() == 107 );
    REQUIRE( timing->missing_trace() == 108 );
    REQUIRE( timing->invalid_trace() == 109 );
    REQUIRE( timing->repeated_start() == 110 );
    REQUIRE( timing->unlinked_primary() == 111 );
    REQUIRE( timing->invalid_primary() == 112 );
    REQUIRE( timing->repeated_primary() == 113 );
    REQUIRE( timing->dropped() == 114 );
    const auto populations = combinerStats( timing );

    for( size_t index = 0; index < populations.size(); ++index )
    {
        checkStats( populations[index], combiner.metrics[index] );
    }

    const auto text = telem_dmsynco::msgString( payload, input.builder.GetSize() );
    REQUIRE( text.starts_with( "[dmsynco] state=3 steps=2 period_ns=500000 delivered=11 missed=2 faults=3" ) );
    REQUIRE( text.find( "combiner_enabled=1" ) != std::string::npos );
    REQUIRE( text.find( "combiner_writer_pid=12345" ) != std::string::npos );
    REQUIRE( text.find( "invalidMetadata:107" ) != std::string::npos );
    REQUIRE( text.find( "repeatedPrimary:113" ) != std::string::npos );

    for( const auto *name : { "combinerPublicationInterval",
                              "combinerReceiptInterval",
                              "combinerExecution",
                              "combinerPublication",
                              "combinerPrimaryStart",
                              "combinerPrimaryPublication" } )
    {
        REQUIRE( text.find( std::string( name ) + "[n=" ) != std::string::npos );
    }

    const auto          json = telem_dmsynco::msgJSON( payload,
                                              input.builder.GetSize(),
                                              reinterpret_cast<const uint8_t *>( telem_dmsynco_bfbs ),
                                              telem_dmsynco_bfbs_len );
    flatbuffers::Parser parser;
    REQUIRE( parser.Deserialize( reinterpret_cast<const uint8_t *>( telem_dmsynco_bfbs ), telem_dmsynco_bfbs_len ) );
    REQUIRE( parser.Parse( json.c_str() ) );
    const auto *jsonRecord = GetTelem_dmsynco_fb( parser.builder_.GetBufferPointer() );
    const auto *jsonTiming = jsonRecord->combiner_timing();
    REQUIRE( jsonTiming != nullptr );
    REQUIRE( jsonTiming->session() == combiner.session );
    REQUIRE( jsonTiming->writer_pid() == combiner.writerPid );
    REQUIRE( jsonTiming->trigger_inode() == combiner.triggerInode );
    REQUIRE( jsonTiming->primary_counter() == combiner.primaryCounter );
    REQUIRE( jsonTiming->dropped() == 114 );
    const auto jsonPopulations = combinerStats( jsonTiming );

    for( size_t index = 0; index < jsonPopulations.size(); ++index )
    {
        checkStats( jsonPopulations[index], combiner.metrics[index] );
    }
}

/// Validate and format optional primary/combiner tables independently, including legacy primary-only records.
/** \ingroup dmSyncoTelemetry_unit_test */
TEST_CASE( "Synco telemetry handles independent optional timing tables", "[dmSyncoTelemetry]" )
{
    for( unsigned variant = 0; variant < 3; ++variant )
    {
        INFO( "optional table variant " << variant );
        auto input = message( samplePrimary() );
        input.builder.Clear();
        input.builder.ForceDefaults( false );
        const auto stats =
            CreateDmSyncoTimingStats( input.builder, variant == 2 ? 1 : 3, 500e-6, 1e-6, 499e-6, 501e-6, 1, true );
        flatbuffers::Offset<DmSyncoPrimaryTiming>  primary;
        flatbuffers::Offset<DmSyncoCombinerTiming> combiner;

        if( variant == 1 )
        {
            DmSyncoPrimaryTimingBuilder timing( input.builder );
            timing.add_enabled( true );
            timing.add_valid( true );
            timing.add_publication( stats );
            timing.add_receipt( stats );
            primary = timing.Finish();
        }
        else
        {
            DmSyncoCombinerTimingBuilder timing( input.builder );
            timing.add_enabled( true );
            timing.add_publication_interval( stats );
            timing.add_receipt_interval( stats );
            timing.add_execution( stats );
            timing.add_publication( stats );
            timing.add_primary_start( stats );
            timing.add_primary_publication( stats );
            combiner = timing.Finish();
        }

        Telem_dmsynco_fbBuilder record( input.builder );
        record.add_primary_timing( primary );
        record.add_combiner_timing( combiner );
        record.add_schedule_valid( false );
        input.builder.Finish( record.Finish() );
        auto buffer = envelope( input );
        REQUIRE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) == ( variant != 2 ) );

        if( variant < 2 )
        {
            const auto text = telem_dmsynco::msgString( input.builder.GetBufferPointer(), input.builder.GetSize() );
            REQUIRE( text.find( "schedule_valid=0" ) != std::string::npos );

            if( variant == 0 )
            {
                REQUIRE( text.find( "primary_timing=unavailable" ) != std::string::npos );
                REQUIRE( text.find( "combinerExecution[n=3 valid=1]" ) != std::string::npos );
            }
            else
            {
                REQUIRE( text.find( "primary_valid=1" ) != std::string::npos );
                REQUIRE( text.find( "combiner_timing=unavailable" ) != std::string::npos );
            }
        }
    }
}

/// Reject corrupt metrics in each combiner population and malformed coverage while allowing independent warm-up.
/** \ingroup dmSyncoTelemetry_unit_test */
TEST_CASE( "Synco telemetry validates every combiner population", "[dmSyncoTelemetry]" )
{
    for( size_t invalid = 0; invalid < 9; ++invalid )
    {
        INFO( "invalid combiner case " << invalid );
        auto combiner = sampleCombiner();

        if( invalid < combiner.metrics.size() )
        {
            combiner.metrics[invalid].stddev = std::numeric_limits<double>::quiet_NaN();
        }

        if( invalid == 6 )
        {
            combiner.windowSec = -1;
        }

        if( invalid == 7 )
        {
            combiner.span = std::numeric_limits<double>::infinity();
        }

        if( invalid == 8 )
        {
            combiner.age = -2;
        }

        auto input  = message( samplePrimary(), combiner );
        auto buffer = envelope( input );
        REQUIRE_FALSE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
    }

    auto  combiner  = sampleCombiner();
    auto &execution = combiner.metrics[dmCombTiming::index( dmCombTiming::Metric::execution )];
    execution.count = 1;
    execution.valid = false;
    auto input      = message( samplePrimary(), combiner );
    auto buffer     = envelope( input );
    REQUIRE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
    REQUIRE( telem_dmsynco::msgString( input.builder.GetBufferPointer(), input.builder.GetSize() )
                 .find( "combinerExecution[n=1 valid=0] mean_s=3e-05 stddev_s=unavailable" ) != std::string::npos );
}

} // namespace dmSyncoTelemetryTest
} // namespace loggerTest
} // namespace libXWCTest
