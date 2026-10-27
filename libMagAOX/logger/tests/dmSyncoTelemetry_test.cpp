/** \file dmSyncoTelemetry_test.cpp
 * \brief Syncopation cadence, conditional simulation and raw overlap telemetry compatibility and validation.
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
    primary.session   = ( uint64_t( 1 ) << 54 ) + 5;
    primary.sequence  = 27;
    primary.nowNs     = 1234567890123;
    primary.windowSec = 2;
    primary.span      = 1.6;
    primary.age       = 0.125;
    primary.capacity  = 20000;
    primary.queued    = 3;
    primary.enabled   = true;
    primary.fgValid   = false;

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

/// Construct a hypothetical ready grid with independently counted skips and retrospective overlaps.
dmSyncoSimulation::Summary sampleSimulation()
{
    dmSyncoSimulation::Summary summary;
    summary.enabled            = true;
    summary.valid              = true;
    summary.feasible           = true;
    summary.session            = ( uint64_t( 1 ) << 54 ) + 19;
    summary.sequence           = 100;
    summary.nowNs              = 1234567890123;
    summary.lastMainNs         = summary.nowNs - 1000;
    summary.counter            = ( uint64_t( 1 ) << 55 ) + 23;
    summary.steps              = 2;
    summary.periodNs           = 500000;
    summary.gridNs             = summary.lastMainNs - 3500;
    summary.phaseNs            = 3500;
    summary.phaseMinNs         = -2000;
    summary.phaseMaxNs         = 4000;
    summary.phaseCount         = 90;
    summary.periodSamples      = 32;
    summary.phaseSamples       = 32;
    summary.nextGridNs         = summary.gridNs + summary.periodNs;
    summary.nextActiveNs       = summary.gridNs + summary.periodNs / 2;
    summary.earlyReserveNs     = 20000;
    summary.deliveryNs         = 21000;
    summary.keepoutNs          = 115000;
    summary.guardNs            = 22000;
    summary.snapshotNs         = 10000;
    summary.copyNs             = 11000;
    summary.leadNs             = 23000;
    summary.observationNs      = 5000;
    summary.visibilityNs       = 1000;
    summary.reserveNs          = 2000;
    summary.window             = 32;
    summary.phaseGain          = 0.1;
    summary.phaseSlewNs        = 1500;
    summary.cadenceMinPeriodNs = 333334;
    summary.duplicates         = 4;
    summary.gaps               = 1;
    summary.resets             = 2;
    summary.invalid            = 3;
    summary.excluded           = 5;
    summary.outputGaps         = 6;
    summary.associations       = 70;
    summary.outcomes[0]        = { 12, 7, 5, 2, 0, 1, 0, 2, 17 };
    summary.outcomes[1]        = { 10, 6, 4, 0, 1, 1, 2, 0, 3 };
    return summary;
}

/// Construct both overlap classes with older sessions, full-width identities and separate producer clocks.
dmSyncoSimulation::TraceBatch sampleOverlapTrace()
{
    dmSyncoSimulation::TraceBatch trace;
    trace.count  = 2;
    trace.status = { true, 12, 8, 1, 3, dmSyncoSimulation::traceCapacity, 2 };

    for( size_t index = 0; index < trace.count; ++index )
    {
        auto &event           = trace.events[index];
        event.id              = index + 7;
        event.session         = ( uint64_t( 1 ) << 54 ) + 18 - index;
        event.mainCounter     = ( uint64_t( 1 ) << 55 ) + 23 + index;
        event.combinedCounter = ( uint64_t( 1 ) << 56 ) + 29 + index;
        event.planCounter     = event.mainCounter - 1;
        event.lastMainCounter = event.mainCounter - 1;
        event.step            = int( index );
        event.origin = index ? dmSyncoSimulation::OverlapOrigin::combined : dmSyncoSimulation::OverlapOrigin::primary;
        event.window = index ? dmSyncoSimulation::OverlapWindow::readerHold : dmSyncoSimulation::OverlapWindow::copy;
        event.observedNs                       = ( int64_t( 1 ) << 56 ) + 1000001 + int64_t( index ) * 500000;
        event.mainGridNs                       = event.observedNs - 3500;
        event.planGridNs                       = event.mainGridNs - 500000;
        event.periodNs                         = 500000;
        event.copyBeginNs                      = event.observedNs - ( index ? 10000 : 500 );
        event.copyEndNs                        = event.copyBeginNs + 1000;
        event.readerEndNs                      = index ? event.copyEndNs + 150000 : 0;
        event.lastMainNs                       = event.observedNs - 500000;
        event.lastOutputNs                     = event.lastMainNs + 10000;
        event.observation.streamInode          = ( uint64_t( 1 ) << 57 ) + 31 + index;
        event.observation.triggerInode         = index ? trace.events[0].observation.streamInode : 0;
        event.observation.receiptNs            = event.observedNs - 400;
        event.observation.callbackNs           = event.observedNs - 300;
        event.observation.lockRequestNs        = event.observedNs - 200;
        event.observation.lockAcquiredNs       = event.observedNs - 100;
        event.observation.publicationNs        = 1800000000000000001 + int64_t( index ) * 500000;
        event.observation.primaryPublicationNs = index ? event.observation.publicationNs - 10000 : 0;
        event.observation.startNs              = index ? event.observation.publicationNs - 5000 : 0;
        event.observation.updateNs             = index ? event.observation.publicationNs - 200 : 0;
    }

    return trace;
}

/// Populate unchanged scheduler fields with independently sampled primary and combiner timing.
telem_dmsynco::messageT
message( const dmTiming::Summary             &primary /**< [in] coherent primary summary */,
         const dmCombTiming::Summary         &combiner   = {} /**< [in] independent combiner summary */,
         const dmSyncoSimulation::Summary    &simulation = {} /**< [in] hypothetical shadow prediction */,
         const dmSyncoSimulation::TraceBatch &trace      = {} /**< [in] drained overlap evidence */ )
{
    return telem_dmsynco::messageT( 3,
                                    2,
                                    500000,
                                    11,
                                    2,
                                    3,
                                    4,
                                    150,
                                    250,
                                    { -100, 200 },
                                    { 300, 400 },
                                    primary,
                                    true,
                                    combiner,
                                    {},
                                    {},
                                    {},
                                    simulation,
                                    trace );
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

/// Check every shadow model, phase, quality and conditional outcome field after serialization.
void checkSimulation( const DmSyncoSimulation          *actual /**< [in] deserialized conditional prediction */,
                      const dmSyncoSimulation::Summary &expected /**< [in] source snapshot */ )
{
    REQUIRE( actual != nullptr );
    REQUIRE( actual->enabled() == expected.enabled );
    REQUIRE( actual->valid() == expected.valid );
    REQUIRE( actual->feasible() == expected.feasible );
    REQUIRE( actual->session() == expected.session );
    REQUIRE( actual->sequence() == expected.sequence );
    REQUIRE( actual->now_ns() == expected.nowNs );
    REQUIRE( actual->last_main_ns() == expected.lastMainNs );
    REQUIRE( actual->counter() == expected.counter );
    REQUIRE( actual->steps() == expected.steps );
    REQUIRE( actual->period_ns() == expected.periodNs );
    REQUIRE( actual->grid_ns() == expected.gridNs );
    REQUIRE( actual->phase_ns() == expected.phaseNs );
    REQUIRE( actual->phase_min_ns() == expected.phaseMinNs );
    REQUIRE( actual->phase_max_ns() == expected.phaseMaxNs );
    REQUIRE( actual->phase_count() == expected.phaseCount );
    REQUIRE( actual->period_samples() == expected.periodSamples );
    REQUIRE( actual->phase_samples() == expected.phaseSamples );
    REQUIRE( actual->next_grid_ns() == expected.nextGridNs );
    REQUIRE( actual->next_active_ns() == expected.nextActiveNs );
    REQUIRE( actual->early_reserve_ns() == expected.earlyReserveNs );
    REQUIRE( actual->delivery_ns() == expected.deliveryNs );
    REQUIRE( actual->keepout_ns() == expected.keepoutNs );
    REQUIRE( actual->guard_ns() == expected.guardNs );
    REQUIRE( actual->snapshot_ns() == expected.snapshotNs );
    REQUIRE( actual->copy_ns() == expected.copyNs );
    REQUIRE( actual->lead_ns() == expected.leadNs );
    REQUIRE( actual->observation_ns() == expected.observationNs );
    REQUIRE( actual->visibility_ns() == expected.visibilityNs );
    REQUIRE( actual->reserve_ns() == expected.reserveNs );
    REQUIRE( actual->window() == expected.window );
    REQUIRE( actual->phase_gain() == expected.phaseGain );
    REQUIRE( actual->phase_slew_ns() == expected.phaseSlewNs );
    REQUIRE( actual->cadence_min_period_ns() == expected.cadenceMinPeriodNs );
    REQUIRE( actual->duplicates() == expected.duplicates );
    REQUIRE( actual->gaps() == expected.gaps );
    REQUIRE( actual->resets() == expected.resets );
    REQUIRE( actual->invalid() == expected.invalid );
    REQUIRE( actual->excluded() == expected.excluded );
    REQUIRE( actual->output_gaps() == expected.outputGaps );
    REQUIRE( actual->associations() == expected.associations );
    REQUIRE( actual->outcomes()->size() == size_t( expected.steps ) );

    for( int index = 0; index < expected.steps; ++index )
    {
        const auto *step   = actual->outcomes()->Get( index );
        const auto &source = expected.outcomes[index];
        REQUIRE( step->considered() == source.considered );
        REQUIRE( step->would_run() == source.wouldRun );
        REQUIRE( step->skipped() == source.skipped );
        REQUIRE( step->early_main() == source.earlyMain );
        REQUIRE( step->missing_main() == source.missingMain );
        REQUIRE( step->missing_output() == source.missingOutput );
        REQUIRE( step->reader_hold() == source.readerHold );
        REQUIRE( step->expired() == source.expired );
        REQUIRE( step->overlap() == source.overlap );
    }
}

/// Preserve every raw overlap endpoint, clock domain, classification and lifetime quality count.
void checkOverlapTrace( const DmSyncoOverlapTrace           *actual /**< [in] serialized overlap batch */,
                        const dmSyncoSimulation::TraceBatch &expected /**< [in] unconverted evidence */ )
{
    REQUIRE( actual != nullptr );
    CHECK( actual->enabled() == expected.status.enabled );
    CHECK( actual->detected() == expected.status.detected );
    CHECK( actual->drained() == expected.status.drained );
    CHECK( actual->dropped() == expected.status.dropped );
    CHECK( actual->queued() == expected.status.queued );
    CHECK( actual->capacity() == expected.status.capacity );
    CHECK( actual->log_failed() == expected.status.logFailed );
    REQUIRE( actual->batch_count() == expected.count );
    REQUIRE( actual->events()->size() == expected.count );

    for( size_t index = 0; index < expected.count; ++index )
    {
        const auto *event  = actual->events()->Get( index );
        const auto &source = expected.events[index];
        CHECK( event->id() == source.id );
        CHECK( event->session() == source.session );
        CHECK( event->main_counter() == source.mainCounter );
        CHECK( event->combined_counter() == source.combinedCounter );
        CHECK( event->plan_counter() == source.planCounter );
        CHECK( event->last_main_counter() == source.lastMainCounter );
        CHECK( event->step() == source.step );
        CHECK( int( event->origin() ) == int( source.origin ) );
        CHECK( int( event->window() ) == int( source.window ) );
        CHECK( event->observed_ns() == source.observedNs );
        CHECK( event->main_grid_ns() == source.mainGridNs );
        CHECK( event->plan_grid_ns() == source.planGridNs );
        CHECK( event->period_ns() == source.periodNs );
        CHECK( event->copy_begin_ns() == source.copyBeginNs );
        CHECK( event->copy_end_ns() == source.copyEndNs );
        CHECK( event->reader_end_ns() == source.readerEndNs );
        CHECK( event->last_main_ns() == source.lastMainNs );
        CHECK( event->last_output_ns() == source.lastOutputNs );
        const auto *observation = event->observation();
        REQUIRE( observation != nullptr );
        CHECK( observation->stream_inode() == source.observation.streamInode );
        CHECK( observation->trigger_inode() == source.observation.triggerInode );
        CHECK( observation->receipt_ns() == source.observation.receiptNs );
        CHECK( observation->callback_ns() == source.observation.callbackNs );
        CHECK( observation->lock_request_ns() == source.observation.lockRequestNs );
        CHECK( observation->lock_acquired_ns() == source.observation.lockAcquiredNs );
        CHECK( observation->publication_ns() == source.observation.publicationNs );
        CHECK( observation->primary_publication_ns() == source.observation.primaryPublicationNs );
        CHECK( observation->start_ns() == source.observation.startNs );
        CHECK( observation->update_ns() == source.observation.updateNs );
    }
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
    REQUIRE( record->simulation() == nullptr );
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
    REQUIRE_FALSE( flatbuffers::IsFieldPresent( record, Telem_dmsynco_fb::VT_SIMULATION ) );
    REQUIRE( record->combiner_timing() == nullptr );
    REQUIRE( record->simulation() == nullptr );
    REQUIRE( record->schedule_valid() );
    REQUIRE( record->residuals_ns()->Get( 0 ) == -100 );
    REQUIRE( record->advances_ns()->Get( 1 ) == 400 );
    REQUIRE( telem_dmsynco::msgString( input.builder.GetBufferPointer(), input.builder.GetSize() ) ==
             "[dmsynco] state=3 steps=2 period_ns=500000 delivered=11 missed=2 faults=3 schedule_valid=1 "
             "primary_timing=unavailable primary_pairs=unavailable combiner_timing=unavailable "
             "camera_timing=unavailable camera_pairs=unavailable pipeline_timing=unavailable "
             "cameraWrite_timing=unavailable cameraWrite_pairs=unavailable" );
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

/// Preserve every optional shadow field and keep hypothetical readiness separate from delivered hardware commands.
/** \ingroup dmSyncoTelemetry_unit_test */
TEST_CASE( "Synco telemetry preserves conditional shadow predictions", "[dmSyncoTelemetry]" )
{
    const auto simulation = sampleSimulation();
    auto       input      = message( samplePrimary(), sampleCombiner(), simulation );
    auto       buffer     = envelope( input );
    REQUIRE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
    const auto *record = GetTelem_dmsynco_fb( input.builder.GetBufferPointer() );
    checkSimulation( record->simulation(), simulation );
    REQUIRE( record->state() == 3 );
    REQUIRE( record->delivered() == 11 );
    REQUIRE( record->schedule_valid() );

    const auto text = telem_dmsynco::msgString( input.builder.GetBufferPointer(), input.builder.GetSize() );
    REQUIRE( text.find( "simulation_enabled=1 simulation_valid=1 simulation_feasible=1" ) != std::string::npos );
    REQUIRE( text.find( "simulation_mode=shadow_conditional" ) != std::string::npos );
    REQUIRE( text.find( "simulation_model={delivery_ns:21000,keepout_ns:115000" ) != std::string::npos );
    REQUIRE( text.find( "window:32,phaseGain:0.1,phaseSlew_ns:1500,cadenceMinPeriod_ns:333334}" ) !=
             std::string::npos );
    REQUIRE( text.find( "simulation_phase_ns=3500 simulation_phase_min_ns=-2000" ) != std::string::npos );
    REQUIRE( text.find( "sim_step={index:0,considered:12,wouldRun:7,skipped:5" ) != std::string::npos );
    REQUIRE( text.find( "overlap:17}" ) != std::string::npos );
    REQUIRE( text.find( "simulation_armed" ) == std::string::npos );
    REQUIRE( text.find( "simulation_delivered" ) == std::string::npos );

    const auto          json = telem_dmsynco::msgJSON( input.builder.GetBufferPointer(),
                                              input.builder.GetSize(),
                                              reinterpret_cast<const uint8_t *>( telem_dmsynco_bfbs ),
                                              telem_dmsynco_bfbs_len );
    flatbuffers::Parser parser;
    REQUIRE( parser.Deserialize( reinterpret_cast<const uint8_t *>( telem_dmsynco_bfbs ), telem_dmsynco_bfbs_len ) );
    REQUIRE( parser.Parse( json.c_str() ) );
    checkSimulation( GetTelem_dmsynco_fb( parser.builder_.GetBufferPointer() )->simulation(), simulation );
}

/// Allow disabled and warming snapshots without imposing ready-grid requirements or extending legacy output.
/** \ingroup dmSyncoTelemetry_unit_test */
TEST_CASE( "Synco telemetry keeps optional shadow warmup independent", "[dmSyncoTelemetry]" )
{
    const auto primary          = samplePrimary();
    auto       disabled         = message( primary );
    auto       explicitDisabled = message( primary, {}, dmSyncoSimulation::Summary{} );
    REQUIRE( GetTelem_dmsynco_fb( disabled.builder.GetBufferPointer() )->simulation() == nullptr );
    REQUIRE(
        telem_dmsynco::msgString( disabled.builder.GetBufferPointer(), disabled.builder.GetSize() ) ==
        telem_dmsynco::msgString( explicitDisabled.builder.GetBufferPointer(), explicitDisabled.builder.GetSize() ) );
    REQUIRE( telem_dmsynco::msgString( disabled.builder.GetBufferPointer(), disabled.builder.GetSize() )
                 .find( "simulation" ) == std::string::npos );

    dmSyncoSimulation::Summary warming;
    warming.enabled = true;
    warming.steps   = GENERATE( 1, 16 );
    auto input      = message( primary, {}, warming );
    auto buffer     = envelope( input );
    REQUIRE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
    checkSimulation( GetTelem_dmsynco_fb( input.builder.GetBufferPointer() )->simulation(), warming );
    REQUIRE( telem_dmsynco::msgString( input.builder.GetBufferPointer(), input.builder.GetSize() )
                 .find( "simulation_enabled=1 simulation_valid=0 simulation_feasible=0" ) != std::string::npos );
}

/// Reject inconsistent phase, dimensions, timing and count partitions without adding retrospective overlap to skips.
/** \ingroup dmSyncoTelemetry_unit_test */
TEST_CASE( "Synco telemetry rejects inconsistent shadow summaries", "[dmSyncoTelemetry]" )
{
    for( unsigned invalid = 0; invalid < 32; ++invalid )
    {
        INFO( "invalid simulation case " << invalid );
        auto simulation = sampleSimulation();

        switch( invalid )
        {
        case 0:
            simulation.steps = 0;
            break;
        case 1:
            simulation.steps = 17;
            break;
        case 2:
            simulation.session = 0;
            break;
        case 3:
            simulation.periodNs = 0;
            break;
        case 4:
            simulation.gridNs = 0;
            break;
        case 5:
            simulation.lastMainNs = simulation.nowNs + 1;
            break;
        case 6:
            simulation.earlyReserveNs = -1;
            break;
        case 7:
            simulation.phaseMinNs = simulation.phaseNs + 1;
            break;
        case 8:
            simulation.phaseMaxNs = simulation.phaseNs - 1;
            break;
        case 9:
            simulation.phaseCount = 0;
            break;
        case 10:
            simulation.phaseCount = simulation.sequence + 1;
            break;
        case 11:
            simulation.periodSamples = simulation.sequence + 1;
            break;
        case 12:
            simulation.phaseSamples = simulation.sequence + 1;
            break;
        case 13:
            simulation.outcomes[0].wouldRun = simulation.outcomes[0].considered + 1;
            break;
        case 14:
            ++simulation.outcomes[0].skipped;
            break;
        case 15:
            simulation.outcomes[0].earlyMain = simulation.outcomes[0].skipped + 1;
            break;
        case 16:
            simulation.outcomes[1].missingMain = simulation.outcomes[1].skipped + 1;
            break;
        case 17:
            simulation.outcomes[1].readerHold = simulation.outcomes[1].skipped + 1;
            break;
        case 18:
            simulation.outcomes[0].expired = simulation.outcomes[0].skipped + 1;
            break;
        case 19:
            simulation.phaseGain = std::numeric_limits<double>::quiet_NaN();
            break;
        case 20:
            simulation.phaseGain = std::numeric_limits<double>::infinity();
            break;
        case 21:
            simulation.phaseGain = 1.1;
            break;
        case 22:
            simulation.deliveryNs = -1;
            break;
        case 23:
            simulation.phaseSlewNs = -1;
            break;
        case 24:
            simulation.window = 1;
            break;
        case 25:
            simulation.phaseCount = 1;
            break;
        case 26:
            simulation.outcomes[1].missingOutput = simulation.outcomes[1].skipped + 1;
            break;
        case 27:
            simulation.nowNs = -1;
            break;
        case 28:
            ++simulation.outcomes[0].earlyMain; // Each reason is bounded, but their sum exceeds skipped.
            break;
        case 29:
            --simulation.outcomes[0].earlyMain; // A skipped occurrence must have exactly one reason.
            break;
        case 30:
            simulation.outcomes[0] = { UINT64_MAX, 0, UINT64_MAX, UINT64_MAX, UINT64_MAX, 0, 0, 0, 0 };
            break;
        case 31:
            simulation.cadenceMinPeriodNs = -1;
            break;
        }

        auto input  = message( samplePrimary(), {}, simulation );
        auto buffer = envelope( input );
        REQUIRE_FALSE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
    }

    auto input = message( samplePrimary() );
    input.builder.Clear();
    const auto outcomes = input.builder.CreateVector( std::vector<flatbuffers::Offset<DmSyncoSimulationStep>>{} );
    DmSyncoSimulationBuilder simulation( input.builder );
    simulation.add_enabled( true );
    simulation.add_session( 1 );
    simulation.add_steps( 2 );
    simulation.add_outcomes( outcomes );
    const auto              shadow = simulation.Finish();
    Telem_dmsynco_fbBuilder record( input.builder );
    record.add_simulation( shadow );
    input.builder.Finish( record.Finish() );
    auto buffer = envelope( input );
    REQUIRE_FALSE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );

    auto        serialized  = message( samplePrimary(), {}, sampleSimulation() );
    auto        corrupt     = envelope( serialized );
    const auto *decoded     = GetTelem_dmsynco_fb( flatlogs::logHeader::messageBuffer( corrupt ) );
    auto       *stepOffsets = const_cast<uint8_t *>( decoded->simulation()->outcomes()->Data() );
    std::memset( stepOffsets, 0xff, sizeof( flatbuffers::uoffset_t ) );
    REQUIRE_FALSE( verifyLogEntry( telem_dmsynco::eventCode, corrupt.get() ) );
}

/// Preserve full-width overlap evidence and older-session batches through binary, JSON and text readers.
/** \ingroup dmSyncoTelemetry_unit_test */
TEST_CASE( "Synco telemetry preserves raw overlap evidence across sessions", "[dmSyncoTelemetry]" )
{
    // clang-format off
#ifdef DMSYNCO_TELEMETRY_TEST_DOXYGEN_REF
    MagAOX::logger::telem_dmsynco::messageT;
    MagAOX::logger::telem_dmsynco::verify;
    MagAOX::logger::telem_dmsynco::msgString;
#endif
    // clang-format on
    const auto trace  = sampleOverlapTrace();
    auto       input  = message( samplePrimary(), {}, sampleSimulation(), trace );
    auto       buffer = envelope( input );
    REQUIRE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
    const auto *simulation = GetTelem_dmsynco_fb( input.builder.GetBufferPointer() )->simulation();
    checkOverlapTrace( simulation->overlap_trace(), trace );
    REQUIRE( simulation->overlap_trace()->events()->Get( 0 )->session() < simulation->session() );
    REQUIRE( simulation->overlap_trace()->events()->Get( 0 )->main_grid_ns() !=
             simulation->overlap_trace()->events()->Get( 0 )->plan_grid_ns() );

    const auto text = telem_dmsynco::msgString( input.builder.GetBufferPointer(), input.builder.GetSize() );
    CHECK(
        text.find(
            "trace_quality={enabled:1,detected:12,drained:8,dropped:1,queued:3,capacity:256,logFailed:2,batch:2}" ) !=
        std::string::npos );
    CHECK( text.find( "origin:primary,window:copy" ) != std::string::npos );
    CHECK( text.find( "origin:combined,window:readerHold" ) != std::string::npos );
    CHECK( text.find( "mainCounter:" + std::to_string( trace.events[0].mainCounter ) ) != std::string::npos );
    CHECK( text.find( "model_clock:monotonic,observed_ns:" + std::to_string( trace.events[0].observedNs ) ) !=
           std::string::npos );
    CHECK( text.find( "producer_clock:unconverted,publication_ns:1800000000000000001" ) != std::string::npos );
    CHECK( text.find( "lockAcquired_ns:" + std::to_string( trace.events[1].observation.lockAcquiredNs ) ) !=
           std::string::npos );

    const auto          json = telem_dmsynco::msgJSON( input.builder.GetBufferPointer(),
                                              input.builder.GetSize(),
                                              reinterpret_cast<const uint8_t *>( telem_dmsynco_bfbs ),
                                              telem_dmsynco_bfbs_len );
    flatbuffers::Parser parser;
    REQUIRE( parser.Deserialize( reinterpret_cast<const uint8_t *>( telem_dmsynco_bfbs ), telem_dmsynco_bfbs_len ) );
    REQUIRE( parser.Parse( json.c_str() ) );
    checkOverlapTrace( GetTelem_dmsynco_fb( parser.builder_.GetBufferPointer() )->simulation()->overlap_trace(),
                       trace );
}

/// Accept older simulations without the optional trace, empty drains and unavailable raw endpoints.
/** \ingroup dmSyncoTelemetry_unit_test */
TEST_CASE( "Synco telemetry keeps optional overlap evidence independent", "[dmSyncoTelemetry]" )
{
    auto legacy = message( samplePrimary(), {}, sampleSimulation() );
    auto buffer = envelope( legacy );
    REQUIRE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
    const auto *simulation = GetTelem_dmsynco_fb( legacy.builder.GetBufferPointer() )->simulation();
    REQUIRE( simulation->overlap_trace() == nullptr );
    REQUIRE_FALSE( flatbuffers::IsFieldPresent( simulation, DmSyncoSimulation::VT_OVERLAP_TRACE ) );
    REQUIRE( telem_dmsynco::msgString( legacy.builder.GetBufferPointer(), legacy.builder.GetSize() )
                 .find( "trace_quality=" ) == std::string::npos );

    auto disabledTrace           = sampleOverlapTrace();
    disabledTrace.status.enabled = false;
    auto disabled                = message( samplePrimary(), {}, sampleSimulation(), disabledTrace );
    REQUIRE( GetTelem_dmsynco_fb( disabled.builder.GetBufferPointer() )->simulation()->overlap_trace() == nullptr );

    dmSyncoSimulation::TraceBatch empty;
    empty.status.enabled = true;
    auto input           = message( samplePrimary(), {}, sampleSimulation(), empty );
    buffer               = envelope( input );
    REQUIRE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
    checkOverlapTrace( GetTelem_dmsynco_fb( input.builder.GetBufferPointer() )->simulation()->overlap_trace(), empty );

    auto trace                  = sampleOverlapTrace();
    trace.events[0].observation = {};
    trace.events[0].mainGridNs  = 0; // A copy can overlap a main without a retained prediction for that main.
    input                       = message( samplePrimary(), {}, sampleSimulation(), trace );
    buffer                      = envelope( input );
    REQUIRE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
    checkOverlapTrace( GetTelem_dmsynco_fb( input.builder.GetBufferPointer() )->simulation()->overlap_trace(), trace );

    auto &active       = trace.events[1];
    active.window      = dmSyncoSimulation::OverlapWindow::copy;
    active.readerEndNs = active.copyEndNs; // Zero delivery/keepout/guard still permits an active copy interval.
    active.observedNs  = active.copyBeginNs;
    active.observation = {};
    input              = message( samplePrimary(), {}, sampleSimulation(), trace );
    buffer             = envelope( input );
    REQUIRE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
    checkOverlapTrace( GetTelem_dmsynco_fb( input.builder.GetBufferPointer() )->simulation()->overlap_trace(), trace );
}

/// Reject impossible queue accounting, unbounded batches and incorrectly classified or ordered evidence.
/** \ingroup dmSyncoTelemetry_unit_test */
TEST_CASE( "Synco telemetry rejects malformed overlap evidence", "[dmSyncoTelemetry]" )
{
    for( unsigned invalid = 0; invalid < 43; ++invalid )
    {
        INFO( "invalid overlap case " << invalid );
        auto  trace   = sampleOverlapTrace();
        auto &quality = trace.status;
        auto &event   = trace.events[0];
        auto &active  = trace.events[1];

        switch( invalid )
        {
        case 0:
            quality.capacity = 0;
            break;
        case 1:
            quality.capacity = dmSyncoSimulation::traceCapacity + 1;
            break;
        case 2:
            quality.queued   = quality.capacity + 1;
            quality.detected = quality.drained + quality.dropped + quality.queued;
            break;
        case 3:
            quality.drained = quality.detected + 1;
            break;
        case 4:
            quality.dropped = UINT64_MAX;
            break;
        case 5:
            ++quality.queued;
            break;
        case 6:
            quality.logFailed = quality.drained + 1;
            break;
        case 7:
            quality.drained  = 1;
            quality.detected = quality.drained + quality.dropped + quality.queued;
            break;
        case 8:
            trace.count = dmSyncoSimulation::traceBatchCapacity + 1;
            break;
        case 9:
            event.id = 0;
            break;
        case 10:
            event.id = quality.detected + 1;
            break;
        case 11:
            active.id = event.id;
            break;
        case 12:
            event.session = 0;
            break;
        case 13:
            event.step = -1;
            break;
        case 14:
            event.step = dmSyncoSimulation::maxSteps;
            break;
        case 15:
            event.origin = dmSyncoSimulation::OverlapOrigin( 2 );
            break;
        case 16:
            event.window = dmSyncoSimulation::OverlapWindow( 2 );
            break;
        case 17:
            event.observedNs = 0;
            break;
        case 18:
            event.mainGridNs = -1;
            break;
        case 19:
            event.planGridNs = 0;
            break;
        case 20:
            event.periodNs = 0;
            break;
        case 21:
            event.copyBeginNs = 0;
            break;
        case 22:
            event.copyEndNs = event.copyBeginNs;
            break;
        case 23:
            event.lastMainNs = event.observedNs + 1;
            break;
        case 24:
            event.lastOutputNs = event.observedNs + 1;
            break;
        case 25:
            event.readerEndNs = event.copyEndNs + 1;
            break;
        case 26:
            active.readerEndNs = active.copyEndNs;
            break;
        case 27:
            event.observedNs = event.copyEndNs;
            break;
        case 28:
            event.observedNs = event.copyBeginNs - 1;
            break;
        case 29:
            event.window = dmSyncoSimulation::OverlapWindow::readerHold;
            break;
        case 30:
            active.observedNs = active.readerEndNs;
            break;
        case 31:
            active.observedNs = active.copyEndNs - 1;
            break;
        case 32:
            event.observation.publicationNs = -1;
            break;
        case 33:
            event.observation.primaryPublicationNs = -1;
            break;
        case 34:
            event.observation.startNs = -1;
            break;
        case 35:
            event.observation.updateNs = -1;
            break;
        case 36:
            event.observation.receiptNs = event.observation.callbackNs + 1;
            break;
        case 37:
            event.observation.callbackNs = event.observation.lockRequestNs + 1;
            break;
        case 38:
            event.observation.lockRequestNs = event.observation.lockAcquiredNs + 1;
            break;
        case 39:
            event.observation.lockAcquiredNs = event.observedNs + 1;
            break;
        case 40:
            event.observation.receiptNs = -1;
            break;
        case 41:
            event.lastMainNs = -1;
            break;
        case 42:
            event.lastOutputNs = -1;
            break;
        }

        auto input  = message( samplePrimary(), {}, sampleSimulation(), trace );
        auto buffer = envelope( input );
        REQUIRE_FALSE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
    }

    auto        input   = message( samplePrimary(), {}, sampleSimulation(), sampleOverlapTrace() );
    auto        corrupt = envelope( input );
    const auto *decoded = GetTelem_dmsynco_fb( flatlogs::logHeader::messageBuffer( corrupt ) );
    auto       *events  = const_cast<uint8_t *>( decoded->simulation()->overlap_trace()->events()->Data() );
    std::memset( events, 0xff, sizeof( flatbuffers::uoffset_t ) );
    REQUIRE_FALSE( verifyLogEntry( telem_dmsynco::eventCode, corrupt.get() ) );
}

} // namespace dmSyncoTelemetryTest
} // namespace loggerTest
} // namespace libXWCTest
