/** \file dmSyncoTelemetry_test.cpp
 * \brief Syncopation timing, reader protection and shadow telemetry compatibility and validation.
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

/// Construct independently queued older-session estimator updates with exact large timestamps and signed corrections.
dmSyncoSimulation::TraceBatch sampleEstimatorTrace()
{
    auto trace            = sampleOverlapTrace();
    trace.estimatorCount  = 2;
    auto &quality         = trace.estimatorStatus;
    quality.enabled       = true;
    quality.observed      = ( uint64_t( 1 ) << 54 ) + 1000;
    quality.triggers      = 10;
    quality.selected      = 8;
    quality.drained       = 5;
    quality.dropped       = 1;
    quality.queued        = 2;
    quality.logFailed     = 2;
    quality.history       = dmSyncoSimulation::estimatorHistoryCapacity;
    quality.postRemaining = dmSyncoSimulation::estimatorPostSamples - 1;

    for( size_t index = 0; index < trace.estimatorCount; ++index )
    {
        auto &sample                        = trace.estimator[index];
        sample.id                           = ( uint64_t( 1 ) << 54 ) + 7 + index;
        sample.session                      = ( uint64_t( 1 ) << 54 ) + 17 + index;
        sample.sequence                     = 100 + index;
        sample.counter                      = ( uint64_t( 1 ) << 55 ) + 23 + index;
        sample.observedNs                   = ( int64_t( 1 ) << 56 ) + 1000001 + int64_t( index ) * 500000;
        sample.publicationIntervalNs        = 499500 + int64_t( index ) * 1000;
        sample.sequential                   = true;
        sample.intervalAccepted             = true;
        sample.periodSamples                = 32;
        sample.phaseSamples                 = 32;
        sample.periodBeforeNs               = 500100;
        sample.periodNs                     = 500000;
        sample.predictedNs                  = sample.observedNs + 3500;
        sample.nextCounter                  = sample.counter + 1;
        sample.sourcePlanCounter            = sample.counter;
        sample.sourceEstimatorId            = index ? sample.id : sample.id - 1;
        sample.sourceGridNs                 = sample.predictedNs;
        sample.baseGridNs                   = sample.sourceGridNs + sample.periodNs;
        sample.candidateNs                  = sample.baseGridNs + ( index ? 4500 : -20000 );
        sample.phaseGain                    = 0.1;
        sample.phaseSlewNs                  = 1000;
        sample.rawCorrectionNs              = index ? 450 : -2000;
        sample.appliedCorrectionNs          = index ? 450 : -1000;
        sample.committedGridNs              = sample.baseGridNs + sample.appliedCorrectionNs;
        sample.committed                    = true;
        trace.events[index].planEstimatorId = sample.id - 1;
        trace.events[index].lastEstimatorId = sample.id;
    }

    return trace;
}

/// Construct signed trailing reader measurements and full-width lifetime worst-command provenance.
dmReaderTiming::Summary sampleReader()
{
    dmReaderTiming::Summary reader;
    reader.enabled                  = true;
    reader.session                  = ( uint64_t( 1 ) << 54 ) + 29;
    reader.sequence                 = 77;
    reader.nowNs                    = 20000000000;
    reader.windowSec                = 2;
    reader.capacity                 = 20000;
    reader.queued                   = 3;
    reader.pendingPrimary           = 1;
    reader.pendingCombined          = 2;
    reader.pendingCompletion        = 3;
    reader.span                     = 0.5;
    reader.age                      = 0.1;
    reader.metrics[0]               = { 3, -5e-6, 10e-6, -15e-6, 5e-6, 0.5, true };
    reader.metrics[1]               = { 3, 2e-6, 2e-6, 0, 5e-6, 0.5, true };
    reader.metrics[2]               = { 3, 8e-6, 3e-6, 2e-6, 14e-6, 0.5, true };
    reader.metrics[3]               = { 3, 100e-6, 1e-6, 99e-6, 101e-6, 0.5, true };
    reader.metrics[4]               = { 3, 10e-6, 5e-6, 1e-6, 20e-6, 0.5, true };
    reader.metrics[5]               = { 3, 160e-6, 10e-6, 150e-6, 170e-6, 0.5, true };
    reader.metrics[6]               = { 3, 7e-6, 3e-6, 1e-6, 12e-6, 0.5, true };
    reader.metrics[7]               = { 3, 3e-6, 1e-6, 0, 8e-6, 0.5, true };
    reader.total                    = { 1000, 400, 300, 300, 100, 10, 11, 12, 13, 14, 15, 16,
                                        17,   18,  19,  20,  21,  22, 23, 24, 25, 26, 27, 28 };
    reader.total.receiptUnavailable = 9;
    reader.total.receiptInvalid     = 7;
    reader.worstValid               = true;
    auto &worst                     = reader.worst;
    worst.primary.input             = { ( uint64_t( 1 ) << 56 ) + 3, ( uint64_t( 1 ) << 55 ) + 5, 1700000000, 100000 };
    worst.primary.observedNs        = 999990000;
    worst.combined.input            = { ( uint64_t( 1 ) << 56 ) + 7, ( uint64_t( 1 ) << 55 ) + 9, 1700000000, 200000 };
    worst.combined.primaryInode     = worst.primary.input.inode;
    worst.combined.primaryCounter   = worst.primary.input.counter;
    worst.combined.startNs          = 1700000000000150000;
    worst.combined.observedNs       = 1000000000;
    worst.combined.budgetNs         = 155000;
    auto &completion                = worst.completed.record;
    completion.session              = ( uint64_t( 1 ) << 54 ) + 31;
    completion.epoch                = 4;
    completion.sequence             = ( uint64_t( 1 ) << 55 ) + 39;
    completion.state                = dmCompletion::State::complete;
    completion.inputInode           = worst.combined.input.inode;
    completion.inputCounter         = worst.combined.input.counter;
    completion.publicationSec       = worst.combined.input.sec;
    completion.publicationNsec      = worst.combined.input.nsec;
    completion.startedNs            = 999920000;
    completion.finishedNs           = 1000020000;
    completion.outcome              = dmTiming::Outcome::success;
    completion.publisherPid         = 1234;
    worst.completed.observedNs      = 1000035000;
    worst.completed.receiptNs       = 1000030000;
    worst.baseNs                    = worst.combined.observedNs;
    worst.offsetNs                  = 20000;
    worst.requiredNs                = 20000;
    worst.observedHoldNs            = 35000;
    worst.slackNs                   = 135000;
    reader.maxRequiredHoldNs        = 20000;
    reader.maxObservedHoldNs        = 60000;
    reader.overBudget               = 1;
    return reader;
}

/// Construct conditional three-step readiness with actual main-release evidence and modeled auxiliary endpoints.
dmSyncoSimulation::Summary sampleReadiness()
{
    auto summary                  = sampleSimulation();
    summary.readerReleaseReady    = true;
    summary.virtualReaderNs       = 30000;
    summary.backendNs             = 155000;
    summary.readerCycleFeasible   = true;
    summary.backendCycleFeasible  = true;
    summary.steps                 = 3;
    summary.readiness             = { 10, 7, 1, 1, 1, 2, 1, 1, 1, 3 };
    auto &evidence                = summary.latestRelease;
    evidence.record               = sampleReader().worst.completed.record;
    evidence.record.kind          = dmCompletion::Kind::readers;
    evidence.record.state         = dmCompletion::State::released;
    evidence.record.outcome       = dmTiming::Outcome::unknown;
    evidence.record.backendResult = dmCompletion::unavailableResult;
    evidence.record.startedNs     = ( int64_t( 1 ) << 56 ) + 1000001;
    evidence.record.finishedNs    = evidence.record.startedNs + 20000;
    evidence.observedNs           = evidence.record.finishedNs + 5000;
    evidence.sourceEpoch          = ( uint64_t( 1 ) << 54 ) + 11;
    evidence.coherent             = true;
    summary.outcomes[0]           = { 14, 7, 7, 2, 0, 1, 0, 2, 17, 1, 1, 0, 0 };
    summary.outcomes[1]           = { 12, 6, 6, 0, 1, 1, 2, 0, 3, 0, 0, 1, 1 };
    summary.outcomes[2]           = { 8, 5, 3, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1 };
    return summary;
}

/// Construct independent pipeline resources with reserved future credits and a prior reset discard.
dmSyncoSimulation::Summary samplePipeline()
{
    auto summary                              = sampleReadiness();
    summary.pipeline.enabled                  = true;
    summary.pipeline.inFlight                 = 3;
    summary.pipeline.maximumInFlight          = 3;
    summary.pipeline.mainAdmitted             = ( uint64_t( 1 ) << 55 ) + 11;
    summary.pipeline.auxiliaryAdmitted        = ( uint64_t( 1 ) << 54 ) + 13;
    summary.pipeline.retired                  = summary.pipeline.mainAdmitted + summary.pipeline.auxiliaryAdmitted - 5;
    summary.pipeline.discarded                = 2;
    summary.pipeline.mainSdkWaits             = 17;
    summary.pipeline.mainCreditMisses         = 19;
    summary.pipeline.resets                   = 2;
    summary.pipeline.sdkReadyNs               = summary.nowNs + 50000;
    summary.pipeline.postReadyNs              = summary.nowNs + 100000;
    summary.pipeline.maximumMainWaitNs        = 16000;
    summary.pipeline.mainClearanceShortfallNs = 201000;
    summary.pipeline.sdkCycleFeasible         = true;
    summary.pipeline.postCycleFeasible        = true;
    summary.pipeline.creditCycleFeasible      = true;
    summary.pipeline.mainProtectionFeasible   = false;
    summary.backendCycleFeasible              = false;
    summary.feasible                          = false;
    summary.outcomes[1]                       = { 12, 6, 6, 0, 1, 1, 0, 0, 3, 0, 0, 1, 1, 1, 1 };
    return summary;
}

/// Return all reader populations in their stable common metric order, including optional receipt splits.
std::array<const DmSyncoTimingStats *, static_cast<size_t>( dmReaderTiming::Metric::count )>
readerStats( const DmSyncoReaderTiming *reader /**< [in] Deserialized reader-protection summary. */ )
{
    return { reader->completion_offset(),
             reader->required_hold(),
             reader->observed_hold(),
             reader->processing(),
             reader->notification(),
             reader->model_slack(),
             reader->wake(),
             reader->capture() };
}

/// Populate unchanged scheduler fields with independently sampled primary and combiner timing.
telem_dmsynco::messageT
message( const dmTiming::Summary             &primary /**< [in] coherent primary summary */,
         const dmCombTiming::Summary         &combiner      = {} /**< [in] independent combiner summary */,
         const dmSyncoSimulation::Summary    &simulation    = {} /**< [in] hypothetical shadow prediction */,
         const dmSyncoSimulation::TraceBatch &trace         = {} /**< [in] drained overlap and estimator evidence */,
         const dmReaderTiming::Summary       &reader        = {} /**< [in] optional software-reader timing */,
         const dmReaderTiming::Summary       &readerRelease = {} /**< [in] independent early-release timing */ )
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
                                    trace,
                                    reader,
                                    readerRelease );
}

/// Build a real log envelope for shared verification dispatch.
flatlogs::bufferPtrT envelope( const telem_dmsynco::messageT &input /**< [in] serialized aggregate */ )
{
    flatlogs::bufferPtrT buffer;
    REQUIRE( flatlogs::logHeader::createLog<telem_dmsynco>( buffer, { 1, 2 }, input, flatlogs::logPrio::LOG_TELEM ) ==
             0 );
    return buffer;
}

/// Remove an optional field from a valid table to reproduce an older writer's absent trailing fields.
void omitField( const void            *table /**< [in/out] Valid table owned by the mutable test message. */,
                flatbuffers::voffset_t field /**< [in] Generated optional-field vtable offset. */ )
{
    auto *bytes  = const_cast<uint8_t *>( static_cast<const uint8_t *>( table ) );
    auto *vtable = bytes - flatbuffers::ReadScalar<flatbuffers::soffset_t>( bytes );
    REQUIRE( field < flatbuffers::ReadScalar<flatbuffers::voffset_t>( vtable ) );
    flatbuffers::WriteScalar<flatbuffers::voffset_t>( vtable + field, 0 );
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

/// Check the exact raw identity fields without converting publication time to the local clock.
void checkReaderIdentity( const DmSyncoReaderIdentity    *actual /**< [in] Decoded producer identity. */,
                          const dmReaderTiming::Identity &expected /**< [in] Original producer identity. */ )
{
    REQUIRE( actual != nullptr );
    CHECK( actual->inode() == expected.inode );
    CHECK( actual->counter() == expected.counter );
    CHECK( actual->publication_sec() == expected.sec );
    CHECK( actual->publication_nsec() == expected.nsec );
}

/// Check exact worst-event identities, endpoint arithmetic and raw semaphore/capture observations.
void checkReaderSample( const DmSyncoReaderSample    *sample /**< [in] Decoded matched-command evidence. */,
                        const dmReaderTiming::Sample &worst /**< [in] Original exact endpoint sample. */ )
{
    REQUIRE( sample != nullptr );
    REQUIRE( sample->primary() != nullptr );
    REQUIRE( sample->combined() != nullptr );
    REQUIRE( sample->completed() != nullptr );
    checkReaderIdentity( sample->primary()->input(), worst.primary.input );
    checkReaderIdentity( sample->combined()->input(), worst.combined.input );
    const auto &completion = worst.completed.record;
    checkReaderIdentity(
        sample->completed()->input(),
        { completion.inputInode, completion.inputCounter, completion.publicationSec, completion.publicationNsec } );
    CHECK( sample->primary()->observed_ns() == worst.primary.observedNs );
    CHECK( sample->combined()->primary_inode() == worst.combined.primaryInode );
    CHECK( sample->combined()->primary_counter() == worst.combined.primaryCounter );
    CHECK( sample->combined()->start_ns() == worst.combined.startNs );
    CHECK( sample->combined()->observed_ns() == worst.combined.observedNs );
    CHECK( sample->combined()->budget_ns() == worst.combined.budgetNs );
    CHECK( sample->completed()->protocol_version() == completion.protocolVersion );
    CHECK( sample->completed()->session() == completion.session );
    CHECK( sample->completed()->epoch() == completion.epoch );
    CHECK( sample->completed()->sequence() == completion.sequence );
    CHECK( sample->completed()->state() == static_cast<uint64_t>( completion.state ) );
    CHECK( sample->completed()->started_ns() == completion.startedNs );
    CHECK( sample->completed()->finished_ns() == completion.finishedNs );
    CHECK( sample->completed()->outcome() == static_cast<uint8_t>( completion.outcome ) );
    CHECK( sample->completed()->callback_result() == completion.callbackResult );
    CHECK( sample->completed()->publisher_pid() == completion.publisherPid );
    CHECK( sample->completed()->kind() == static_cast<uint8_t>( completion.kind ) );
    CHECK( sample->completed()->backend_result() == completion.backendResult );
    CHECK( sample->completed()->observed_ns() == worst.completed.observedNs );
    CHECK( sample->completed()->receipt_ns() == worst.completed.receiptNs );
    CHECK( sample->base_ns() == worst.baseNs );
    CHECK( sample->offset_ns() == worst.offsetNs );
    CHECK( sample->required_ns() == worst.requiredNs );
    CHECK( sample->observed_hold_ns() == worst.observedHoldNs );
    CHECK( sample->slack_ns() == worst.slackNs );
}

/// Check every reader population, lifetime quality field and worst-command identity after a reader roundtrip.
void checkReader( const DmSyncoReaderTiming     *actual /**< [in] Decoded reader measurement. */,
                  const dmReaderTiming::Summary &expected /**< [in] Original matched-command summary. */ )
{
    REQUIRE( actual != nullptr );
    CHECK( actual->enabled() == expected.enabled );
    CHECK( actual->kind() == static_cast<uint8_t>( expected.kind ) );
    CHECK( actual->session() == expected.session );
    CHECK( actual->sequence() == expected.sequence );
    CHECK( actual->now_ns() == expected.nowNs );
    CHECK( actual->window_sec() == expected.windowSec );
    CHECK( actual->capacity() == expected.capacity );
    CHECK( actual->queued() == expected.queued );
    CHECK( actual->pending_primary() == expected.pendingPrimary );
    CHECK( actual->pending_combined() == expected.pendingCombined );
    CHECK( actual->pending_completion() == expected.pendingCompletion );
    CHECK( actual->span_sec() == expected.span );
    CHECK( actual->age_sec() == expected.age );
    const auto stats = readerStats( actual );

    for( size_t index = 0; index < stats.size(); ++index )
    {
        checkStats( stats[index], expected.metrics[index] );
    }

    const auto *quality = actual->total();
    REQUIRE( quality != nullptr );
    CHECK( quality->sdk_results() == expected.total.sdkResults );
    CHECK( quality->sdk_success() == expected.total.sdkSuccess );
    CHECK( quality->sdk_failed() == expected.total.sdkFailed );
    CHECK( quality->receipt_unavailable() == expected.total.receiptUnavailable );
    CHECK( quality->receipt_invalid() == expected.total.receiptInvalid );
    CHECK( quality->first_enriched() == expected.total.firstEnriched );
    const std::array<uint64_t, 24> counts         = { quality->events(),
                                                      quality->primary(),
                                                      quality->combined(),
                                                      quality->completion(),
                                                      quality->matched(),
                                                      quality->busy(),
                                                      quality->unavailable(),
                                                      quality->error(),
                                                      quality->unqualified(),
                                                      quality->invalid(),
                                                      quality->incoherent(),
                                                      quality->duplicates(),
                                                      quality->sequence_gaps(),
                                                      quality->counter_gaps(),
                                                      quality->source_resets(),
                                                      quality->stale_epoch(),
                                                      quality->missing_primary(),
                                                      quality->missing_combined(),
                                                      quality->identity_mismatch(),
                                                      quality->evicted(),
                                                      quality->expired(),
                                                      quality->dropped_primary(),
                                                      quality->dropped_combined(),
                                                      quality->dropped_completion() };
    const auto                    &total          = expected.total;
    const std::array<uint64_t, 24> expectedCounts = {
        total.events,     total.primary,        total.combined,        total.completion,       total.matched,
        total.busy,       total.unavailable,    total.error,           total.unqualified,      total.invalid,
        total.incoherent, total.duplicates,     total.sequenceGaps,    total.counterGaps,      total.sourceResets,
        total.staleEpoch, total.missingPrimary, total.missingCombined, total.identityMismatch, total.evicted,
        total.expired,    total.droppedPrimary, total.droppedCombined, total.droppedCompletion };

    for( size_t index = 0; index < counts.size(); ++index )
    {
        CHECK( counts[index] == expectedCounts[index] );
    }

    CHECK( actual->max_required_hold_ns() == expected.maxRequiredHoldNs );
    CHECK( actual->max_observed_hold_ns() == expected.maxObservedHoldNs );
    CHECK( actual->over_budget() == expected.overBudget );
    CHECK( actual->worst_valid() == expected.worstValid );

    CHECK( actual->worst_notification_valid() == expected.worstNotificationValid );

    if( expected.worstNotificationValid )
    {
        checkReaderSample( actual->worst_notification(), expected.worstNotification );
    }
    else
    {
        CHECK( actual->worst_notification() == nullptr );
    }

    if( expected.worstValid )
    {
        checkReaderSample( actual->worst(), expected.worst );
    }
    else
    {
        CHECK( actual->worst() == nullptr );
    }
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

/// Check readiness assumptions and exact real-main release identity without treating virtual slots as measurements.
void checkReadiness( const DmSyncoReadiness           *actual /**< [in] Optional decoded readiness subtable. */,
                     const dmSyncoSimulation::Summary &expected /**< [in] Original shadow policy and evidence. */ )
{
    if( !expected.readerReleaseReady )
    {
        CHECK( actual == nullptr );
        return;
    }

    REQUIRE( actual != nullptr );
    CHECK( actual->reader_release_ready() );
    CHECK( actual->virtual_reader_ns() == expected.virtualReaderNs );
    CHECK( actual->backend_ns() == expected.backendNs );
    CHECK( actual->reader_cycle_feasible() == expected.readerCycleFeasible );
    CHECK( actual->backend_cycle_feasible() == expected.backendCycleFeasible );
    const auto *quality = actual->total();
    REQUIRE( quality != nullptr );
    CHECK( quality->observed() == expected.readiness.observed );
    CHECK( quality->qualified() == expected.readiness.qualified );
    CHECK( quality->invalid() == expected.readiness.invalid );
    CHECK( quality->incoherent() == expected.readiness.incoherent );
    CHECK( quality->duplicates() == expected.readiness.duplicates );
    CHECK( quality->gaps() == expected.readiness.gaps );
    CHECK( quality->source_resets() == expected.readiness.sourceResets );
    CHECK( quality->identity_mismatch() == expected.readiness.identityMismatch );
    CHECK( quality->sdk_failed() == expected.readiness.sdkFailed );
    CHECK( quality->queue_drops() == expected.readiness.queueDrops );

    if( !expected.latestRelease.observedNs )
    {
        CHECK( actual->latest_release() == nullptr );
        return;
    }

    const auto *evidence = actual->latest_release();
    REQUIRE( evidence != nullptr );
    CHECK( evidence->source_epoch() == expected.latestRelease.sourceEpoch );
    CHECK( evidence->coherent() == expected.latestRelease.coherent );
    const auto *completed = evidence->completed();
    REQUIRE( completed != nullptr );
    const auto &record = expected.latestRelease.record;
    checkReaderIdentity( completed->input(),
                         { record.inputInode, record.inputCounter, record.publicationSec, record.publicationNsec } );
    CHECK( completed->protocol_version() == record.protocolVersion );
    CHECK( completed->session() == record.session );
    CHECK( completed->epoch() == record.epoch );
    CHECK( completed->sequence() == record.sequence );
    CHECK( completed->state() == static_cast<uint64_t>( record.state ) );
    CHECK( completed->kind() == static_cast<uint8_t>( record.kind ) );
    CHECK( completed->started_ns() == record.startedNs );
    CHECK( completed->finished_ns() == record.finishedNs );
    CHECK( completed->outcome() == static_cast<uint8_t>( record.outcome ) );
    CHECK( completed->callback_result() == record.callbackResult );
    CHECK( completed->backend_result() == record.backendResult );
    CHECK( completed->publisher_pid() == record.publisherPid );
    CHECK( completed->observed_ns() == expected.latestRelease.observedNs );
    CHECK( completed->receipt_ns() == 0 );
}

/// Check optional resource assumptions, occupancy and all main-service diagnostics without narrowing counts.
void checkPipeline( const DmSyncoPipeline             *actual /**< [in] Serialized optional pipeline model. */,
                    const dmSyncoSimulation::Pipeline &expected /**< [in] Exact source resource snapshot. */ )
{
    if( !expected.enabled )
    {
        REQUIRE( actual == nullptr );
        return;
    }

    REQUIRE( actual != nullptr );
    CHECK( actual->enabled() == expected.enabled );
    CHECK( actual->prepare_ns() == expected.prepareNs );
    CHECK( actual->sdk_service_ns() == expected.sdkServiceNs );
    CHECK( actual->post_ns() == expected.postNs );
    CHECK( actual->capacity() == expected.capacity );
    CHECK( actual->in_flight() == expected.inFlight );
    CHECK( actual->maximum_in_flight() == expected.maximumInFlight );
    CHECK( actual->main_admitted() == expected.mainAdmitted );
    CHECK( actual->auxiliary_admitted() == expected.auxiliaryAdmitted );
    CHECK( actual->retired() == expected.retired );
    CHECK( actual->discarded() == expected.discarded );
    CHECK( actual->main_sdk_waits() == expected.mainSdkWaits );
    CHECK( actual->main_credit_misses() == expected.mainCreditMisses );
    CHECK( actual->resets() == expected.resets );
    CHECK( actual->sdk_ready_ns() == expected.sdkReadyNs );
    CHECK( actual->post_ready_ns() == expected.postReadyNs );
    CHECK( actual->maximum_main_wait_ns() == expected.maximumMainWaitNs );
    CHECK( actual->main_clearance_shortfall_ns() == expected.mainClearanceShortfallNs );
    CHECK( actual->sdk_cycle_feasible() == expected.sdkCycleFeasible );
    CHECK( actual->post_cycle_feasible() == expected.postCycleFeasible );
    CHECK( actual->credit_cycle_feasible() == expected.creditCycleFeasible );
    CHECK( actual->main_protection_feasible() == expected.mainProtectionFeasible );
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
    checkReadiness( actual->readiness(), expected );
    checkPipeline( actual->pipeline(), expected.pipeline );

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
        REQUIRE( step->release_missing() == source.releaseMissing );
        REQUIRE( step->release_late() == source.releaseLate );
        REQUIRE( step->backend_busy() == source.backendBusy );
        REQUIRE( step->backend_wrap() == source.backendWrap );
        REQUIRE( step->preparation_busy() == source.preparationBusy );
        REQUIRE( step->credit_full() == source.creditFull );
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
        CHECK( event->plan_estimator_id() == source.planEstimatorId );
        CHECK( event->last_estimator_id() == source.lastEstimatorId );
        CHECK( event->period_q32() == source.periodQ32 );
        CHECK( event->plan_fraction_q32() == source.planFractionQ32 );
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
        CHECK( event->backend_end_ns() == source.backendEndNs );
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

/// Preserve all selected estimator values and independent lifetime/history coverage through serialization.
void checkEstimatorTrace( const DmSyncoEstimatorTrace         *actual /**< [in] serialized triggered history */,
                          const dmSyncoSimulation::TraceBatch &expected /**< [in] original selected values */ )
{
    REQUIRE( actual != nullptr );
    const auto &quality = expected.estimatorStatus;
    CHECK( actual->enabled() == quality.enabled );
    CHECK( actual->observed() == quality.observed );
    CHECK( actual->triggers() == quality.triggers );
    CHECK( actual->selected() == quality.selected );
    CHECK( actual->drained() == quality.drained );
    CHECK( actual->dropped() == quality.dropped );
    CHECK( actual->queued() == quality.queued );
    CHECK( actual->capacity() == quality.capacity );
    CHECK( actual->log_failed() == quality.logFailed );
    CHECK( actual->history() == quality.history );
    CHECK( actual->history_capacity() == quality.historyCapacity );
    CHECK( actual->post_remaining() == quality.postRemaining );
    REQUIRE( actual->batch_count() == expected.estimatorCount );
    REQUIRE( actual->samples()->size() == expected.estimatorCount );

    for( size_t index = 0; index < expected.estimatorCount; ++index )
    {
        const auto *sample = actual->samples()->Get( index );
        const auto &source = expected.estimator[index];
        CHECK( sample->id() == source.id );
        CHECK( sample->session() == source.session );
        CHECK( sample->sequence() == source.sequence );
        CHECK( sample->counter() == source.counter );
        CHECK( sample->observed_ns() == source.observedNs );
        CHECK( sample->publication_interval_ns() == source.publicationIntervalNs );
        CHECK( sample->sequential() == source.sequential );
        CHECK( sample->interval_accepted() == source.intervalAccepted );
        CHECK( sample->period_samples() == source.periodSamples );
        CHECK( sample->phase_samples() == source.phaseSamples );
        CHECK( sample->period_before_ns() == source.periodBeforeNs );
        CHECK( sample->period_ns() == source.periodNs );
        CHECK( sample->predicted_ns() == source.predictedNs );
        CHECK( sample->next_counter() == source.nextCounter );
        CHECK( sample->source_plan_counter() == source.sourcePlanCounter );
        CHECK( sample->source_estimator_id() == source.sourceEstimatorId );
        CHECK( sample->source_grid_ns() == source.sourceGridNs );
        CHECK( sample->base_grid_ns() == source.baseGridNs );
        CHECK( sample->candidate_ns() == source.candidateNs );
        CHECK( sample->phase_gain() == source.phaseGain );
        CHECK( sample->phase_slew_ns() == source.phaseSlewNs );
        CHECK( sample->raw_correction_ns() == source.rawCorrectionNs );
        CHECK( sample->applied_correction_ns() == source.appliedCorrectionNs );
        CHECK( sample->committed_grid_ns() == source.committedGridNs );
        CHECK( sample->committed() == source.committed );
        CHECK( sample->period_before_q32() == source.periodBeforeQ32 );
        CHECK( sample->source_fraction_q32() == source.sourceFractionQ32 );
        CHECK( sample->base_fraction_q32() == source.baseFractionQ32 );
        CHECK( sample->committed_fraction_q32() == source.committedFractionQ32 );
    }
}

/// Construct a measured slow cadence whose exact fractional period follows its counter/time endpoints.
dmSyncoSimulation::Cadence
sampleCadence( int64_t observed /**< [in] coherent monotonic primary observation */,
               bool    full = true /**< [in] a full ten-second span rather than a growing one-second span */ )
{
    dmSyncoSimulation::Cadence cadence;
    cadence.periodQ32     = uint64_t( 500000 ) * dmSyncoSimulation::periodScale + dmSyncoSimulation::periodScale / 4;
    cadence.state         = full ? dmSyncoSimulation::CadenceState::full : dmSyncoSimulation::CadenceState::growing;
    cadence.epoch         = 7;
    cadence.anchors       = full ? 11 : 2;
    cadence.cycles        = full ? 20000 : 2000;
    cadence.updates       = full ? 10 : 1;
    cadence.rejected      = 2;
    cadence.spanNs        = full ? 10000005000 : 1000000500;
    cadence.windowNs      = 10000000000;
    cadence.updateNs      = 1000000000;
    cadence.observationNs = observed;
    cadence.updatedNs     = observed - 500000;
    return cadence;
}

/// Construct exact target carry across full-width timestamps, queued old epochs and a changed cadence projection.
dmSyncoSimulation::TraceBatch sampleFractionalTrace()
{
    auto          trace = sampleEstimatorTrace();
    const int64_t scale = dmSyncoSimulation::periodScale;

    for( size_t index = 0; index < trace.estimatorCount; ++index )
    {
        auto &sample   = trace.estimator[index];
        sample.cadence = sampleCadence( sample.observedNs, index != 0 );
        sample.cadence.epoch -= index + 1;
        sample.periodBeforeQ32   = uint64_t( sample.periodBeforeNs ) * scale + scale / 4;
        sample.sourceFractionQ32 = index ? -scale / 4 : scale / 4;

        if( index )
        {
            sample.cadence.periodQ32 += scale / 2;
            sample.cadence.spanNs += sample.cadence.cycles / 2;
            sample.periodNs = 500001;
            sample.sourcePlanCounter -= 2;
        }

        const __int128 exact = __int128( sample.sourceGridNs ) * scale + sample.sourceFractionQ32 +
                               ( __int128( sample.nextCounter ) - sample.sourcePlanCounter ) * sample.cadence.periodQ32;
        sample.baseGridNs                   = int64_t( ( exact + scale / 2 ) / scale );
        sample.baseFractionQ32              = int64_t( exact - __int128( sample.baseGridNs ) * scale );
        sample.candidateNs                  = sample.baseGridNs + ( index ? 4500 : -20000 );
        sample.committedGridNs              = sample.baseGridNs + sample.appliedCorrectionNs;
        sample.committedFractionQ32         = sample.baseFractionQ32;
        trace.events[index].periodQ32       = sample.cadence.periodQ32;
        trace.events[index].periodNs        = sample.periodNs;
        trace.events[index].planFractionQ32 = sample.sourceFractionQ32;
    }

    return trace;
}

/// Check every cadence field after binary or JSON serialization without a floating-point period conversion.
void checkCadence( const DmSyncoCadence             *actual /**< [in] serialized slow measurement */,
                   const dmSyncoSimulation::Cadence &expected /**< [in] exact source cadence */ )
{
    REQUIRE( actual != nullptr );
    CHECK( actual->period_q32() == expected.periodQ32 );
    CHECK( int( actual->state() ) == int( expected.state ) );
    CHECK( actual->epoch() == expected.epoch );
    CHECK( actual->anchors() == expected.anchors );
    CHECK( actual->cycles() == expected.cycles );
    CHECK( actual->updates() == expected.updates );
    CHECK( actual->rejected() == expected.rejected );
    CHECK( actual->span_ns() == expected.spanNs );
    CHECK( actual->window_ns() == expected.windowNs );
    CHECK( actual->update_ns() == expected.updateNs );
    CHECK( actual->observation_ns() == expected.observationNs );
    CHECK( actual->updated_ns() == expected.updatedNs );
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
             "cameraWrite_timing=unavailable cameraWrite_pairs=unavailable reader_timing=unavailable "
             "reader_release_timing=unavailable" );
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

/// Distinguish actual-main backend conflicts from channel-reader overlap and preserve half-open endpoints.
/** \ingroup dmSyncoTelemetry_unit_test */
TEST_CASE( "Synco telemetry retains backend overlap independently of reader protection", "[dmSyncoTelemetry]" )
{
    // clang-format off
#ifdef DMSYNCO_TELEMETRY_TEST_DOXYGEN_REF
    MagAOX::logger::telem_dmsynco::messageT;
    MagAOX::logger::telem_dmsynco::verify;
    MagAOX::logger::telem_dmsynco::msgString;
#endif
    // clang-format on
    auto  trace        = sampleOverlapTrace();
    auto &event        = trace.events[1];
    event.window       = dmSyncoSimulation::OverlapWindow::backend;
    event.readerEndNs  = event.observedNs - 1000;
    event.backendEndNs = event.observedNs + 5000;

    SECTION( "Backend-only interval survives binary, text and JSON roundtrip" )
    {
        auto input  = message( samplePrimary(), {}, sampleSimulation(), trace );
        auto buffer = envelope( input );
        REQUIRE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
        checkOverlapTrace( GetTelem_dmsynco_fb( input.builder.GetBufferPointer() )->simulation()->overlap_trace(),
                           trace );
        const auto text = telem_dmsynco::msgString( input.builder.GetBufferPointer(), input.builder.GetSize() );
        CHECK( text.find( "origin:combined,window:backend" ) != std::string::npos );
        CHECK( text.find( "backendEnd_ns:" + std::to_string( event.backendEndNs ) ) != std::string::npos );
        const auto          json = telem_dmsynco::msgJSON( input.builder.GetBufferPointer(),
                                                  input.builder.GetSize(),
                                                  reinterpret_cast<const uint8_t *>( telem_dmsynco_bfbs ),
                                                  telem_dmsynco_bfbs_len );
        flatbuffers::Parser parser;
        REQUIRE(
            parser.Deserialize( reinterpret_cast<const uint8_t *>( telem_dmsynco_bfbs ), telem_dmsynco_bfbs_len ) );
        REQUIRE( parser.Parse( json.c_str() ) );
        checkOverlapTrace( GetTelem_dmsynco_fb( parser.builder_.GetBufferPointer() )->simulation()->overlap_trace(),
                           trace );
    }

    SECTION( "Backend interval includes the reader-end boundary" )
    {
        event.observedNs  = event.readerEndNs;
        event.observation = {};
        auto input        = message( samplePrimary(), {}, sampleSimulation(), trace );
        auto buffer       = envelope( input );
        REQUIRE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
    }

    SECTION( "Reader classification takes precedence while guarded reads remain" )
    {
        event.window      = dmSyncoSimulation::OverlapWindow::readerHold;
        event.observedNs  = event.readerEndNs - 1;
        event.observation = {};
        auto input        = message( samplePrimary(), {}, sampleSimulation(), trace );
        auto buffer       = envelope( input );
        REQUIRE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
    }

    SECTION( "Reader reserve can extend beyond modeled backend ownership" )
    {
        event.window       = dmSyncoSimulation::OverlapWindow::copy;
        event.observedNs   = event.copyBeginNs;
        event.backendEndNs = event.copyEndNs + 1000;
        event.readerEndNs  = event.copyEndNs + 2000;
        event.observation  = {};
        auto input         = message( samplePrimary(), {}, sampleSimulation(), trace );
        auto buffer        = envelope( input );
        REQUIRE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
    }

    SECTION( "Malformed backend classification is rejected" )
    {
        for( unsigned variant = 0; variant < 6; ++variant )
        {
            CAPTURE( variant );
            auto  bad             = trace;
            auto &candidate       = bad.events[1];
            candidate.observation = {};

            switch( variant )
            {
            case 0:
                candidate.observedNs = candidate.backendEndNs;
                break;
            case 1:
                candidate.observedNs = candidate.readerEndNs - 1;
                break;
            case 2:
                candidate.backendEndNs = 0;
                break;
            case 3:
                candidate.backendEndNs = candidate.copyEndNs - 1;
                break;
            case 4:
                candidate.backendEndNs = -1;
                break;
            case 5:
                candidate.step        = 0;
                candidate.readerEndNs = 0;
                break;
            }

            auto input  = message( samplePrimary(), {}, sampleSimulation(), bad );
            auto buffer = envelope( input );
            REQUIRE_FALSE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
        }
    }
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
            event.window = dmSyncoSimulation::OverlapWindow( 3 );
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

/// Preserve exact pre/post history, frozen-plan provenance and signed corrections through every reader path.
/** \ingroup dmSyncoTelemetry_unit_test */
TEST_CASE( "Synco telemetry preserves triggered estimator history across sessions", "[dmSyncoTelemetry]" )
{
    // clang-format off
#ifdef DMSYNCO_TELEMETRY_TEST_DOXYGEN_REF
    MagAOX::logger::telem_dmsynco::messageT;
    MagAOX::logger::telem_dmsynco::verify;
    MagAOX::logger::telem_dmsynco::msgString;
#endif
    // clang-format on
    const auto trace  = sampleEstimatorTrace();
    auto       input  = message( samplePrimary(), {}, sampleSimulation(), trace );
    auto       buffer = envelope( input );
    REQUIRE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
    const auto *simulation = GetTelem_dmsynco_fb( input.builder.GetBufferPointer() )->simulation();
    checkOverlapTrace( simulation->overlap_trace(), trace );
    checkEstimatorTrace( simulation->estimator_trace(), trace );
    REQUIRE( simulation->estimator_trace()->samples()->Get( 0 )->session() < simulation->session() );
    REQUIRE( simulation->estimator_trace()->samples()->Get( 0 )->phase_slew_ns() != simulation->phase_slew_ns() );

    const auto text = telem_dmsynco::msgString( input.builder.GetBufferPointer(), input.builder.GetSize() );
    CHECK( text.find( "estimator_quality={enabled:1,observed:" + std::to_string( trace.estimatorStatus.observed ) ) !=
           std::string::npos );
    CHECK( text.find( ",triggers:10,selected:8,drained:5,dropped:1,queued:2,capacity:4096,logFailed:2,history:256," ) !=
           std::string::npos );
    CHECK( text.find( "historyCapacity:256,postRemaining:63,batch:2}" ) != std::string::npos );
    CHECK( text.find( "planEstimatorId:" + std::to_string( trace.events[0].planEstimatorId ) ) != std::string::npos );
    CHECK( text.find( "lastEstimatorId:" + std::to_string( trace.events[0].lastEstimatorId ) ) != std::string::npos );
    CHECK( text.find( "sim_estimator={id:" + std::to_string( trace.estimator[0].id ) ) != std::string::npos );
    CHECK( text.find( "model_clock:monotonic,observed_ns:" + std::to_string( trace.estimator[0].observedNs ) ) !=
           std::string::npos );
    CHECK( text.find( "interval_clock:producer_elapsed,publicationInterval_ns:499500" ) != std::string::npos );
    CHECK( text.find( "phaseGain:0.10000000000000001,phaseSlew_ns:1000,rawCorrection_ns:-2000," ) !=
           std::string::npos );
    CHECK( text.find( "appliedCorrection_ns:-1000,committedGrid_ns:" +
                      std::to_string( trace.estimator[0].committedGridNs ) ) != std::string::npos );

    const auto          json = telem_dmsynco::msgJSON( input.builder.GetBufferPointer(),
                                              input.builder.GetSize(),
                                              reinterpret_cast<const uint8_t *>( telem_dmsynco_bfbs ),
                                              telem_dmsynco_bfbs_len );
    flatbuffers::Parser parser;
    REQUIRE( parser.Deserialize( reinterpret_cast<const uint8_t *>( telem_dmsynco_bfbs ), telem_dmsynco_bfbs_len ) );
    REQUIRE( parser.Parse( json.c_str() ) );
    const auto *decoded = GetTelem_dmsynco_fb( parser.builder_.GetBufferPointer() )->simulation();
    checkOverlapTrace( decoded->overlap_trace(), trace );
    checkEstimatorTrace( decoded->estimator_trace(), trace );
    REQUIRE( logMemberAccessor( telem_dmsynco::eventCode, "estimator_trace" ).accessor == nullptr );
}

/// Keep older overlap payloads and unavailable/warming/failed estimator computations readable.
/** \ingroup dmSyncoTelemetry_unit_test */
TEST_CASE( "Synco telemetry keeps optional estimator history independent", "[dmSyncoTelemetry]" )
{
    // clang-format off
#ifdef DMSYNCO_TELEMETRY_TEST_DOXYGEN_REF
    MagAOX::logger::telem_dmsynco::messageT;
    MagAOX::logger::telem_dmsynco::verify;
#endif
    // clang-format on
    auto legacy = message( samplePrimary(), {}, sampleSimulation(), sampleOverlapTrace() );
    auto buffer = envelope( legacy );
    REQUIRE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
    const auto *old = GetTelem_dmsynco_fb( legacy.builder.GetBufferPointer() )->simulation();
    REQUIRE( old->estimator_trace() == nullptr );
    REQUIRE( old->overlap_trace()->events()->Get( 0 )->plan_estimator_id() == 0 );
    REQUIRE( old->overlap_trace()->events()->Get( 0 )->last_estimator_id() == 0 );
    CHECK( telem_dmsynco::msgString( legacy.builder.GetBufferPointer(), legacy.builder.GetSize() )
               .find( "estimator_quality=" ) == std::string::npos );

    dmSyncoSimulation::TraceBatch empty;
    empty.estimatorStatus.enabled = true;
    auto noEvents                 = message( samplePrimary(), {}, sampleSimulation(), empty );
    buffer                        = envelope( noEvents );
    REQUIRE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
    checkEstimatorTrace( GetTelem_dmsynco_fb( noEvents.builder.GetBufferPointer() )->simulation()->estimator_trace(),
                         empty );

    auto disabled = message( samplePrimary(), {}, {}, sampleEstimatorTrace() );
    buffer        = envelope( disabled );
    REQUIRE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
    REQUIRE( GetTelem_dmsynco_fb( disabled.builder.GetBufferPointer() )->simulation() == nullptr );

    for( unsigned state = 0; state < 5; ++state )
    {
        INFO( "valid partial estimator state " << state );
        auto trace           = sampleEstimatorTrace();
        trace.count          = 0;
        trace.status.enabled = false;
        trace.estimatorCount = 1;
        auto &sample         = trace.estimator[0];

        if( state == 0 )
        {
            sample                = {};
            sample.id             = 1;
            sample.session        = 1;
            sample.sequence       = 1;
            sample.counter        = 0;
            sample.observedNs     = 1;
            sample.periodBeforeNs = sample.periodNs = 500000;
            sample.phaseSamples                     = 1;
            sample.publicationIntervalNs            = -5;
        }
        else if( state == 1 )
        {
            sample.sourceGridNs = INT64_MAX - 100000;
            sample.baseGridNs = sample.candidateNs = 0;
            sample.rawCorrectionNs = sample.appliedCorrectionNs = sample.committedGridNs = 0;
            sample.committed                                                             = false;
        }
        else if( state == 2 )
        {
            sample.committed       = false;
            sample.committedGridNs = 0;
        }
        else if( state == 3 )
        {
            sample.candidateNs     = 0;
            sample.rawCorrectionNs = sample.appliedCorrectionNs = 0;
            sample.committedGridNs                              = sample.baseGridNs;
        }
        else
        {
            sample.phaseGain       = 0;
            sample.phaseSlewNs     = 0;
            sample.rawCorrectionNs = sample.appliedCorrectionNs = 0;
            sample.committedGridNs                              = sample.baseGridNs;
        }

        auto input = message( samplePrimary(), {}, sampleSimulation(), trace );
        buffer     = envelope( input );
        REQUIRE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
        checkEstimatorTrace( GetTelem_dmsynco_fb( input.builder.GetBufferPointer() )->simulation()->estimator_trace(),
                             trace );
    }
}

/// Reject impossible coverage, lost provenance, invalid estimator arithmetic and malformed nested payloads.
/** \ingroup dmSyncoTelemetry_unit_test */
TEST_CASE( "Synco telemetry rejects malformed estimator history", "[dmSyncoTelemetry]" )
{
    // clang-format off
#ifdef DMSYNCO_TELEMETRY_TEST_DOXYGEN_REF
    MagAOX::logger::telem_dmsynco::verify;
#endif
    // clang-format on
    for( unsigned invalid = 0; invalid < 43; ++invalid )
    {
        INFO( "invalid estimator case " << invalid );
        auto  trace   = sampleEstimatorTrace();
        auto &quality = trace.estimatorStatus;
        auto &sample  = trace.estimator[0];

        switch( invalid )
        {
        case 0:
            quality.capacity = 0;
            break;
        case 1:
            quality.capacity = dmSyncoSimulation::estimatorTraceCapacity + 1;
            break;
        case 2:
            quality.queued   = quality.capacity + 1;
            quality.selected = quality.drained + quality.dropped + quality.queued;
            break;
        case 3:
            quality.observed = quality.selected - 1;
            break;
        case 4:
            quality.drained = quality.selected + 1;
            break;
        case 5:
            quality.dropped = UINT64_MAX;
            break;
        case 6:
            ++quality.queued;
            break;
        case 7:
            quality.logFailed = quality.drained + 1;
            break;
        case 8:
            quality.historyCapacity = 0;
            break;
        case 9:
            quality.historyCapacity = dmSyncoSimulation::estimatorHistoryCapacity + 1;
            break;
        case 10:
            quality.history = quality.historyCapacity + 1;
            break;
        case 11:
            quality.postRemaining = dmSyncoSimulation::estimatorPostSamples + 1;
            break;
        case 12:
            trace.estimatorCount = dmSyncoSimulation::estimatorBatchCapacity + 1;
            break;
        case 13:
            sample.id = 0;
            break;
        case 14:
            sample.id = quality.observed + 1;
            break;
        case 15:
            trace.estimator[1].id = sample.id;
            break;
        case 16:
            sample.session = 0;
            break;
        case 17:
            sample.sequence = 0;
            break;
        case 18:
            sample.phaseSamples = dmSyncoSimulation::estimatorHistoryCapacity + 1;
            break;
        case 19:
            sample.periodSamples = sample.sequence + 1;
            break;
        case 20:
            sample.observedNs = 0;
            break;
        case 21:
            sample.periodBeforeNs = 0;
            break;
        case 22:
            sample.periodNs = 0;
            break;
        case 23:
            sample.predictedNs = -1;
            break;
        case 24:
            sample.phaseGain = std::numeric_limits<double>::quiet_NaN();
            break;
        case 25:
            sample.phaseGain = 1.1;
            break;
        case 26:
            sample.phaseSlewNs = -1;
            break;
        case 27:
            sample.sequential = false;
            break;
        case 28:
            sample.publicationIntervalNs = -1;
            break;
        case 29:
            sample.nextCounter = 0;
            break;
        case 30:
            ++sample.nextCounter;
            break;
        case 31:
            sample.sourceEstimatorId = sample.id + 1;
            break;
        case 32:
            sample.sourcePlanCounter = sample.nextCounter;
            break;
        case 33:
            sample.sourceGridNs = 0;
            break;
        case 34:
            ++sample.baseGridNs;
            break;
        case 35:
            sample.candidateNs = -1;
            break;
        case 36:
            ++sample.rawCorrectionNs;
            break;
        case 37:
            sample.appliedCorrectionNs = -1001;
            break;
        case 38:
            ++sample.committedGridNs;
            break;
        case 39:
            sample.committed = false;
            break;
        case 40:
            trace.events[0].planEstimatorId = trace.events[0].lastEstimatorId + 1;
            break;
        case 41:
            trace.events[0].lastEstimatorId = quality.observed + 1;
            break;
        case 42:
            quality.drained  = 1;
            quality.selected = quality.drained + quality.dropped + quality.queued;
            break;
        }

        auto input  = message( samplePrimary(), {}, sampleSimulation(), trace );
        auto buffer = envelope( input );
        REQUIRE_FALSE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
    }

    auto        input   = message( samplePrimary(), {}, sampleSimulation(), sampleEstimatorTrace() );
    auto        corrupt = envelope( input );
    const auto *decoded = GetTelem_dmsynco_fb( flatlogs::logHeader::messageBuffer( corrupt ) );
    auto       *samples = const_cast<uint8_t *>( decoded->simulation()->estimator_trace()->samples()->Data() );
    std::memset( samples, 0xff, sizeof( flatbuffers::uoffset_t ) );
    REQUIRE_FALSE( verifyLogEntry( telem_dmsynco::eventCode, corrupt.get() ) );
}

/// Preserve long-baseline qualification and fractional frozen-grid carry in binary, text and JSON readers.
/** \ingroup dmSyncoTelemetry_unit_test */
TEST_CASE( "Synco telemetry preserves fractional cadence across queued epochs", "[dmSyncoTelemetry]" )
{
    // clang-format off
#ifdef DMSYNCO_TELEMETRY_TEST_DOXYGEN_REF
    MagAOX::logger::telem_dmsynco::messageT;
    MagAOX::logger::telem_dmsynco::verify;
    MagAOX::logger::telem_dmsynco::msgString;
#endif
    // clang-format on
    auto summary      = sampleSimulation();
    summary.cadence   = sampleCadence( summary.lastMainNs );
    const auto trace  = sampleFractionalTrace();
    auto       input  = message( samplePrimary(), {}, summary, trace );
    auto       buffer = envelope( input );
    REQUIRE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
    const auto *decoded = GetTelem_dmsynco_fb( input.builder.GetBufferPointer() )->simulation();
    checkCadence( decoded->cadence(), summary.cadence );
    checkEstimatorTrace( decoded->estimator_trace(), trace );
    checkOverlapTrace( decoded->overlap_trace(), trace );
    REQUIRE( trace.estimator[0].baseFractionQ32 == -int64_t( dmSyncoSimulation::periodScale / 2 ) );
    REQUIRE( trace.estimator[1].baseGridNs != trace.estimator[1].sourceGridNs + 3 * trace.estimator[1].periodNs );

    const auto text = telem_dmsynco::msgString( input.builder.GetBufferPointer(), input.builder.GetSize() );
    CHECK( text.find( "simulation_cadence={state:full,clock:monotonic,period_q32:" +
                      std::to_string( summary.cadence.periodQ32 ) ) != std::string::npos );
    CHECK( text.find( "period_exact_ns:500000.25" ) != std::string::npos );
    CHECK( text.find( "estimator_cadence={state:growing" ) != std::string::npos );
    CHECK( text.find( "sourceFraction_q32:1073741824,baseFraction_q32:-2147483648" ) != std::string::npos );
    CHECK( text.find( "period_q32:" + std::to_string( trace.events[1].periodQ32 ) ) != std::string::npos );

    const auto          json = telem_dmsynco::msgJSON( input.builder.GetBufferPointer(),
                                              input.builder.GetSize(),
                                              reinterpret_cast<const uint8_t *>( telem_dmsynco_bfbs ),
                                              telem_dmsynco_bfbs_len );
    flatbuffers::Parser parser;
    REQUIRE( parser.Deserialize( reinterpret_cast<const uint8_t *>( telem_dmsynco_bfbs ), telem_dmsynco_bfbs_len ) );
    REQUIRE( parser.Parse( json.c_str() ) );
    decoded = GetTelem_dmsynco_fb( parser.builder_.GetBufferPointer() )->simulation();
    checkCadence( decoded->cadence(), summary.cadence );
    checkEstimatorTrace( decoded->estimator_trace(), trace );
    checkOverlapTrace( decoded->overlap_trace(), trace );

    for( size_t index = 0; index < trace.estimatorCount; ++index )
    {
        checkCadence( decoded->estimator_trace()->samples()->Get( index )->cadence(), trace.estimator[index].cadence );
        CHECK( decoded->estimator_trace()->samples()->Get( index )->cadence()->epoch() < decoded->cadence()->epoch() );
    }
}

/// Keep legacy absence, nominal warmup and unavailable future commitments independent of cadence diagnostics.
/** \ingroup dmSyncoTelemetry_unit_test */
TEST_CASE( "Synco telemetry keeps cadence qualification optional during warmup", "[dmSyncoTelemetry]" )
{
    // clang-format off
#ifdef DMSYNCO_TELEMETRY_TEST_DOXYGEN_REF
    MagAOX::logger::telem_dmsynco::messageT;
    MagAOX::logger::telem_dmsynco::verify;
#endif
    // clang-format on
    auto legacy = message( samplePrimary(), {}, sampleSimulation(), sampleEstimatorTrace() );
    auto buffer = envelope( legacy );
    REQUIRE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
    const auto *old = GetTelem_dmsynco_fb( legacy.builder.GetBufferPointer() )->simulation();
    CHECK( old->cadence() == nullptr );
    CHECK( old->estimator_trace()->samples()->Get( 0 )->cadence() == nullptr );
    CHECK( old->overlap_trace()->events()->Get( 0 )->period_q32() == 0 );

    auto summary              = sampleSimulation();
    summary.cadence           = sampleCadence( summary.lastMainNs );
    summary.cadence.state     = dmSyncoSimulation::CadenceState::nominal;
    summary.cadence.anchors   = 1;
    summary.cadence.cycles    = 0;
    summary.cadence.updates   = 0;
    summary.cadence.spanNs    = 0;
    summary.cadence.updatedNs = 0;
    auto input                = message( samplePrimary(), {}, summary );
    buffer                    = envelope( input );
    REQUIRE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
    checkCadence( GetTelem_dmsynco_fb( input.builder.GetBufferPointer() )->simulation()->cadence(), summary.cadence );

    auto  trace                 = sampleFractionalTrace();
    auto &sample                = trace.estimator[0];
    sample.nextCounter          = 0;
    sample.sourcePlanCounter    = 0;
    sample.sourceEstimatorId    = 0;
    sample.sourceGridNs         = 0;
    sample.baseGridNs           = 0;
    sample.candidateNs          = 0;
    sample.rawCorrectionNs      = 0;
    sample.appliedCorrectionNs  = 0;
    sample.committedGridNs      = 0;
    sample.committed            = false;
    sample.sourceFractionQ32    = 0;
    sample.baseFractionQ32      = 0;
    sample.committedFractionQ32 = 0;
    input                       = message( samplePrimary(), {}, summary, trace );
    buffer                      = envelope( input );
    REQUIRE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
}

/// Reject invalid cadence qualification, nanosecond projections and fractional target arithmetic.
/** \ingroup dmSyncoTelemetry_unit_test */
TEST_CASE( "Synco telemetry rejects malformed fractional cadence", "[dmSyncoTelemetry]" )
{
    // clang-format off
#ifdef DMSYNCO_TELEMETRY_TEST_DOXYGEN_REF
    MagAOX::logger::telem_dmsynco::messageT;
    MagAOX::logger::telem_dmsynco::verify;
#endif
    // clang-format on
    for( int invalid = 0; invalid < 33; ++invalid )
    {
        INFO( "malformed fractional cadence case " << invalid );
        auto summary    = sampleSimulation();
        summary.cadence = sampleCadence( summary.lastMainNs );
        auto  trace     = sampleFractionalTrace();
        auto &cadence   = summary.cadence;
        auto &sample    = trace.estimator[0];

        switch( invalid )
        {
        case 0:
            cadence.periodQ32 = std::numeric_limits<uint64_t>::max();
            break;
        case 1:
            cadence.state = dmSyncoSimulation::CadenceState( 3 );
            break;
        case 2:
            cadence.epoch = 0;
            break;
        case 3:
            cadence.anchors = dmSyncoSimulation::cadenceAnchorCapacity + 1;
            break;
        case 4:
            cadence.anchors = 1;
            break;
        case 5:
            cadence.cycles = 0;
            break;
        case 6:
            cadence.updates = 0;
            break;
        case 7:
            cadence.spanNs = cadence.windowNs - 1;
            break;
        case 8:
            cadence.state = dmSyncoSimulation::CadenceState::growing;
            break;
        case 9:
            cadence.windowNs = 0;
            break;
        case 10:
            cadence.updateNs = cadence.windowNs + 1;
            break;
        case 11:
            cadence.updateNs = 0;
            break;
        case 12:
            cadence.updatedNs = cadence.observationNs + 1;
            break;
        case 13:
            cadence.observationNs = summary.nowNs + 1;
            break;
        case 14:
            cadence.updatedNs = 0;
            break;
        case 15:
            cadence.state = dmSyncoSimulation::CadenceState::nominal;
            break;
        case 16:
            ++summary.periodNs;
            break;
        case 17:
            ++sample.periodBeforeNs;
            break;
        case 18:
            sample.periodBeforeQ32 = 0;
            break;
        case 19:
            ++sample.periodNs;
            break;
        case 20:
            ++sample.cadence.observationNs;
            break;
        case 21:
            sample.cadence.periodQ32 = 0;
            break;
        case 22:
            sample.sourceFractionQ32 = int64_t( dmSyncoSimulation::periodScale / 2 );
            break;
        case 23:
            sample.baseFractionQ32 = -int64_t( dmSyncoSimulation::periodScale / 2 ) - 1;
            break;
        case 24:
            ++sample.baseFractionQ32;
            break;
        case 25:
            ++sample.committedFractionQ32;
            break;
        case 26:
            sample.committed       = false;
            sample.committedGridNs = 0;
            break;
        case 27:
            trace.events[0].planFractionQ32 = int64_t( dmSyncoSimulation::periodScale / 2 );
            break;
        case 28:
            trace.events[0].periodQ32 = 0;
            break;
        case 29:
            ++trace.events[0].periodNs;
            break;
        case 30:
            ++sample.baseGridNs;
            break;
        case 31:
            ++sample.sourceFractionQ32;
            break;
        case 32:
            ++cadence.spanNs;
            break;
        }

        auto input  = message( samplePrimary(), {}, summary, trace );
        auto buffer = envelope( input );
        REQUIRE_FALSE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
    }
}

/// Preserve signed reader measurements and exact lifetime worst-command provenance through all readers.
/** \ingroup dmSyncoTelemetry_unit_test */
TEST_CASE( "Synco telemetry preserves qualified reader-protection measurements", "[dmSyncoTelemetry]" )
{
    // clang-format off
#ifdef DMSYNCO_TELEMETRY_TEST_DOXYGEN_REF
    MagAOX::logger::telem_dmsynco::messageT;
    MagAOX::logger::telem_dmsynco::verify;
    MagAOX::logger::telem_dmsynco::msgString;
#endif
    // clang-format on
    const auto reader = sampleReader();
    auto       input  = message( samplePrimary(), {}, {}, {}, reader );
    auto       buffer = envelope( input );
    REQUIRE( telem_dmsynco::verify( buffer, flatlogs::logHeader::msgLen( buffer ) ) );
    REQUIRE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
    const auto *decoded = GetTelem_dmsynco_fb( input.builder.GetBufferPointer() )->reader_timing();
    checkReader( decoded, reader );
    const auto text = telem_dmsynco::msgString( input.builder.GetBufferPointer(), input.builder.GetSize() );
    CHECK( text.find( "reader_mode=shadow_observation" ) != std::string::npos );
    CHECK( text.find( "reader_pending={primary:1,combined:2,completion:3}" ) != std::string::npos );
    CHECK( text.find( "readerCompletionOffset[n=3 valid=1] mean_s=-5e-06" ) != std::string::npos );
    CHECK( text.find( "readerRequiredHold[n=3 valid=1] mean_s=" ) != std::string::npos );
    CHECK( text.find( "readerObservedHold[n=3 valid=1] mean_s=" ) != std::string::npos );
    CHECK( text.find( "readerProcessing[n=3 valid=1] mean_s=" ) != std::string::npos );
    CHECK( text.find( "readerNotification[n=3 valid=1] mean_s=" ) != std::string::npos );
    CHECK( text.find( "readerModelSlack[n=3 valid=1] mean_s=" ) != std::string::npos );
    CHECK( text.find( "readerWake[n=3 valid=1] mean_s=" ) != std::string::npos );
    CHECK( text.find( "readerCapture[n=3 valid=1] mean_s=" ) != std::string::npos );
    CHECK( text.find( "receiptUnavailable:9,receiptInvalid:7,firstEnriched:0" ) != std::string::npos );
    CHECK( text.find( "receipt_ns:1000030000" ) != std::string::npos );
    CHECK( text.find( "reader_max_required_hold_ns=20000" ) != std::string::npos );
    CHECK( text.find( "reader_max_observed_hold_ns=60000" ) != std::string::npos );
    CHECK( text.find( "reader_over_budget=1" ) != std::string::npos );
    CHECK( text.find( "reader_worst={clock:monotonic,base_ns:1000000000,offset_ns:20000" ) != std::string::npos );
    CHECK( text.find( "budget_ns:155000" ) != std::string::npos );
    CHECK( text.find( "session:" + std::to_string( reader.worst.completed.record.session ) ) != std::string::npos );
    CHECK( text.find( "start_external_ns:1700000000000150000" ) != std::string::npos );
    CHECK( text.find( "droppedCompletion:28" ) != std::string::npos );

    const auto          json = telem_dmsynco::msgJSON( input.builder.GetBufferPointer(),
                                              input.builder.GetSize(),
                                              reinterpret_cast<const uint8_t *>( telem_dmsynco_bfbs ),
                                              telem_dmsynco_bfbs_len );
    flatbuffers::Parser parser;
    REQUIRE( parser.Deserialize( reinterpret_cast<const uint8_t *>( telem_dmsynco_bfbs ), telem_dmsynco_bfbs_len ) );
    REQUIRE( parser.Parse( json.c_str() ) );
    decoded = GetTelem_dmsynco_fb( parser.builder_.GetBufferPointer() )->reader_timing();
    checkReader( decoded, reader );
    CHECK( logMemberAccessor( telem_dmsynco::eventCode, "reader_timing" ).accessor == nullptr );
}

/// Accept legacy absence, new warmup and expired windows without erasing lifetime reader evidence.
/** \ingroup dmSyncoTelemetry_unit_test */
TEST_CASE( "Synco telemetry keeps optional reader timing and lifetime evidence independent", "[dmSyncoTelemetry]" )
{
    // clang-format off
#ifdef DMSYNCO_TELEMETRY_TEST_DOXYGEN_REF
    MagAOX::logger::telem_dmsynco::messageT;
    MagAOX::logger::telem_dmsynco::verify;
#endif
    // clang-format on
    auto legacy = message( samplePrimary() );
    auto buffer = envelope( legacy );
    REQUIRE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
    const auto *record = GetTelem_dmsynco_fb( legacy.builder.GetBufferPointer() );
    CHECK( record->reader_timing() == nullptr );
    CHECK_FALSE( flatbuffers::IsFieldPresent( record, Telem_dmsynco_fb::VT_READER_TIMING ) );
    CHECK( telem_dmsynco::msgString( legacy.builder.GetBufferPointer(), legacy.builder.GetSize() )
               .find( "reader_timing=unavailable" ) != std::string::npos );

    dmReaderTiming::Summary warming;
    warming.enabled   = true;
    warming.session   = 1;
    warming.capacity  = 20;
    warming.windowSec = 2;

    SECTION( "Before the first matched command" )
    {
        warming.queued            = warming.capacity * 3;
        warming.pendingPrimary    = warming.capacity;
        warming.pendingCombined   = warming.capacity;
        warming.pendingCompletion = warming.capacity;
        auto input                = message( samplePrimary(), {}, {}, {}, warming );
        buffer                    = envelope( input );
        REQUIRE( telem_dmsynco::verify( buffer, flatlogs::logHeader::msgLen( buffer ) ) );
        checkReader( GetTelem_dmsynco_fb( input.builder.GetBufferPointer() )->reader_timing(), warming );
    }

    SECTION( "Lifetime worst command remains after all trailing samples expire" )
    {
        auto reader    = sampleReader();
        reader.metrics = {};
        reader.span    = 0;
        reader.age     = -1;
        auto input     = message( samplePrimary(), {}, {}, {}, reader );
        buffer         = envelope( input );
        REQUIRE( telem_dmsynco::verify( buffer, flatlogs::logHeader::msgLen( buffer ) ) );
        REQUIRE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
        checkReader( GetTelem_dmsynco_fb( input.builder.GetBufferPointer() )->reader_timing(), reader );
    }

    SECTION( "Completion before the protection baseline remains signed" )
    {
        auto reader                              = sampleReader();
        reader.worst.completed.record.startedNs  = reader.worst.baseNs - 105000;
        reader.worst.completed.record.finishedNs = reader.worst.baseNs - 5000;
        reader.worst.completed.observedNs        = reader.worst.baseNs + 7000;
        reader.worst.offsetNs                    = -5000;
        reader.worst.requiredNs                  = 0;
        reader.worst.observedHoldNs              = 7000;
        reader.worst.slackNs                     = 160000;
        reader.maxRequiredHoldNs                 = 0;
        reader.maxObservedHoldNs                 = 10000;
        reader.overBudget                        = 0;
        reader.metrics[0]                        = { 3, -10e-6, 5e-6, -15e-6, -5e-6, 0.5, true };
        reader.metrics[1]                        = { 3, 0, 0, 0, 0, 0.5, true };
        reader.metrics[2]                        = { 3, 5e-6, 5e-6, 0, 10e-6, 0.5, true };
        reader.metrics[5]                        = { 3, 165e-6, 5e-6, 160e-6, 170e-6, 0.5, true };
        auto input                               = message( samplePrimary(), {}, {}, {}, reader );
        buffer                                   = envelope( input );
        REQUIRE( telem_dmsynco::verify( buffer, flatlogs::logHeader::msgLen( buffer ) ) );
        REQUIRE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
        checkReader( GetTelem_dmsynco_fb( input.builder.GetBufferPointer() )->reader_timing(), reader );
        const auto text = telem_dmsynco::msgString( input.builder.GetBufferPointer(), input.builder.GetSize() );
        CHECK( text.find( "offset_ns:-5000,required_ns:0,observed_hold_ns:7000,slack_ns:160000" ) !=
               std::string::npos );
    }

    SECTION( "A single qualified completion has unavailable sample variance" )
    {
        auto reader = sampleReader();
        reader.span = 0;

        for( auto &stats : reader.metrics )
        {
            stats.count  = 1;
            stats.valid  = false;
            stats.stddev = 0;
            stats.min    = stats.mean;
            stats.max    = stats.mean;
            stats.span   = 0;
        }

        auto input = message( samplePrimary(), {}, {}, {}, reader );
        buffer     = envelope( input );
        REQUIRE( telem_dmsynco::verify( buffer, flatlogs::logHeader::msgLen( buffer ) ) );
        checkReader( GetTelem_dmsynco_fb( input.builder.GetBufferPointer() )->reader_timing(), reader );
        CHECK( telem_dmsynco::msgString( input.builder.GetBufferPointer(), input.builder.GetSize() )
                   .find( "readerCompletionOffset[n=1 valid=0] mean_s=-5e-06 stddev_s=unavailable" ) !=
               std::string::npos );
    }

    SECTION( "Completion beyond a frozen model budget keeps negative slack" )
    {
        auto reader                              = sampleReader();
        reader.worst.combined.budgetNs           = 20000;
        reader.worst.completed.record.startedNs  = reader.worst.baseNs - 70000;
        reader.worst.completed.record.finishedNs = reader.worst.baseNs + 30000;
        reader.worst.completed.observedNs        = reader.worst.baseNs + 45000;
        reader.worst.offsetNs                    = 30000;
        reader.worst.requiredNs                  = 30000;
        reader.worst.observedHoldNs              = 45000;
        reader.worst.slackNs                     = -10000;
        reader.maxRequiredHoldNs                 = 30000;
        reader.metrics[0]                        = { 3, 20e-6, 10e-6, 10e-6, 30e-6, 0.5, true };
        reader.metrics[1]                        = reader.metrics[0];
        reader.metrics[2]                        = { 3, 35e-6, 10e-6, 25e-6, 45e-6, 0.5, true };
        reader.metrics[5]                        = { 3, 0, 10e-6, -10e-6, 10e-6, 0.5, true };
        auto input                               = message( samplePrimary(), {}, {}, {}, reader );
        buffer                                   = envelope( input );
        REQUIRE( telem_dmsynco::verify( buffer, flatlogs::logHeader::msgLen( buffer ) ) );
        checkReader( GetTelem_dmsynco_fb( input.builder.GetBufferPointer() )->reader_timing(), reader );
        const auto text = telem_dmsynco::msgString( input.builder.GetBufferPointer(), input.builder.GetSize() );
        CHECK( text.find( "readerModelSlack[n=3 valid=1] mean_s=0 stddev_s=1e-05 min_s=-1e-05" ) != std::string::npos );
        CHECK( text.find( "slack_ns:-10000" ) != std::string::npos );
    }
}

/// Reject malformed quality accounting, bounds, identities, outcomes and exact signed completion arithmetic.
/** \ingroup dmSyncoTelemetry_unit_test */
TEST_CASE( "Synco telemetry rejects inconsistent reader-protection evidence", "[dmSyncoTelemetry]" )
{
    // clang-format off
#ifdef DMSYNCO_TELEMETRY_TEST_DOXYGEN_REF
    MagAOX::logger::telem_dmsynco::verify;
#endif
    // clang-format on
    for( unsigned variant = 0; variant < 47; ++variant )
    {
        CAPTURE( variant );
        auto  reader    = sampleReader();
        auto &sample    = reader.worst;
        auto &completed = sample.completed.record;

        switch( variant )
        {
        case 0:
            reader.session = 0;
            break;
        case 1:
            reader.windowSec = std::numeric_limits<double>::quiet_NaN();
            break;
        case 2:
            reader.windowSec = -1;
            break;
        case 3:
            reader.nowNs = -1;
            break;
        case 4:
            reader.capacity = 0;
            break;
        case 5:
            reader.queued = reader.capacity * 3 + 1;
            break;
        case 6:
            reader.pendingPrimary = reader.capacity + 1;
            break;
        case 7:
            reader.pendingCombined = reader.capacity + 1;
            break;
        case 8:
            reader.pendingCompletion = reader.capacity + 1;
            break;
        case 9:
            reader.age = -0.5;
            break;
        case 10:
            reader.span = -1;
            break;
        case 11:
            ++reader.total.events;
            break;
        case 12:
            reader.total.matched = reader.total.completion + 1;
            break;
        case 13:
            reader.total.busy = reader.total.completion;
            break;
        case 14:
            reader.worstValid = false;
            break;
        case 15:
            ++sample.requiredNs;
            break;
        case 16:
            reader.maxObservedHoldNs = reader.maxRequiredHoldNs - 1;
            break;
        case 17:
            reader.overBudget = reader.total.matched + 1;
            break;
        case 18:
            ++completed.inputCounter;
            break;
        case 19:
            completed.publicationNsec = 1000000000;
            break;
        case 20:
            ++sample.combined.primaryInode;
            break;
        case 21:
            sample.combined.startNs = 1700000000000099999;
            break;
        case 22:
            sample.combined.startNs = 1700000000000200001;
            break;
        case 23:
            ++completed.protocolVersion;
            break;
        case 24:
            completed.state = dmCompletion::State::busy;
            break;
        case 25:
            completed.outcome = dmTiming::Outcome::failed;
            break;
        case 26:
            completed.callbackResult = -1;
            break;
        case 27:
            sample.completed.observedNs = completed.finishedNs - 1;
            break;
        case 28:
            ++sample.baseNs;
            break;
        case 29:
            ++sample.offsetNs;
            break;
        case 30:
            ++sample.observedHoldNs;
            break;
        case 31:
            ++sample.slackNs;
            break;
        case 32:
            completed.session = 0;
            break;
        case 33:
            completed.epoch = 0;
            break;
        case 34:
            completed.sequence = 0;
            break;
        case 35:
            completed.startedNs = 0;
            break;
        case 36:
            completed.finishedNs = completed.startedNs - 1;
            break;
        case 37:
            sample.primary.input.sec = std::numeric_limits<int64_t>::max();
            break;
        case 38:
            reader.maxRequiredHoldNs = -1;
            break;
        case 39:
            reader.age = std::numeric_limits<double>::infinity();
            break;
        case 40:
            ++sample.combined.primaryCounter;
            break;
        case 41:
            sample.primary.observedNs = 0;
            break;
        case 42:
            sample.combined.budgetNs = -1;
            break;
        case 43:
            reader.span = reader.windowSec + 1;
            break;
        case 44:
            reader.total.primary    = std::numeric_limits<uint64_t>::max();
            reader.total.combined   = 1;
            reader.total.completion = 1;
            reader.total.events     = 1;
            break;
        case 45:
            reader.span = std::numeric_limits<double>::quiet_NaN();
            break;
        case 46:
            reader.windowSec = 0;
            break;
        }

        auto input  = message( samplePrimary(), {}, {}, {}, reader );
        auto buffer = envelope( input );
        REQUIRE_FALSE( telem_dmsynco::verify( buffer, flatlogs::logHeader::msgLen( buffer ) ) );
        REQUIRE_FALSE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
    }

    auto        input       = message( samplePrimary(), {}, {}, {}, sampleReader() );
    auto        corrupt     = envelope( input );
    const auto *record      = GetTelem_dmsynco_fb( flatlogs::logHeader::messageBuffer( corrupt ) );
    auto       *readerBytes = const_cast<uint8_t *>( reinterpret_cast<const uint8_t *>( record->reader_timing() ) );
    std::memset( readerBytes, 0xff, sizeof( flatbuffers::soffset_t ) );
    REQUIRE_FALSE( telem_dmsynco::verify( corrupt, flatlogs::logHeader::msgLen( corrupt ) ) );
    REQUIRE_FALSE( verifyLogEntry( telem_dmsynco::eventCode, corrupt.get() ) );
}

/// Validate each reader population independently, including signed values, empty windows and unavailable variance.
/** \ingroup dmSyncoTelemetry_unit_test */
TEST_CASE( "Synco telemetry validates every reader timing population", "[dmSyncoTelemetry]" )
{
    // clang-format off
#ifdef DMSYNCO_TELEMETRY_TEST_DOXYGEN_REF
    MagAOX::logger::telem_dmsynco::verify;
#endif
    // clang-format on
    for( size_t metric = 0; metric < static_cast<size_t>( dmReaderTiming::Metric::count ); ++metric )
    {
        for( unsigned variant = 0; variant < 15; ++variant )
        {
            CAPTURE( metric, variant );
            auto  reader = sampleReader();
            auto &stats  = reader.metrics[metric];

            switch( variant )
            {
            case 0:
                stats.count = reader.capacity + 1;
                break;
            case 1:
                stats.count = reader.total.matched + 1;
                break;
            case 2:
                stats.valid = false;
                break;
            case 3:
                stats.stddev = std::numeric_limits<double>::quiet_NaN();
                break;
            case 4:
                stats.stddev = -1;
                break;
            case 5:
                stats.mean = std::numeric_limits<double>::quiet_NaN();
                break;
            case 6:
                stats.min = -std::numeric_limits<double>::infinity();
                break;
            case 7:
                stats.max = std::numeric_limits<double>::infinity();
                break;
            case 8:
                stats.span = std::numeric_limits<double>::quiet_NaN();
                break;
            case 9:
                stats.mean = stats.min - 1;
                break;
            case 10:
                stats.mean = stats.max + 1;
                break;
            case 11:
                stats.count = 1;
                break;
            case 12:
                reader.metrics              = {};
                reader.age                  = -1;
                reader.span                 = 0;
                reader.metrics[metric].mean = std::numeric_limits<double>::quiet_NaN();
                break;
            case 13:
                stats.span = reader.span + 1;
                break;
            case 14:
                stats.count  = 1;
                stats.valid  = false;
                stats.stddev = 0;
                break;
            }

            auto input  = message( samplePrimary(), {}, {}, {}, reader );
            auto buffer = envelope( input );
            REQUIRE_FALSE( telem_dmsynco::verify( buffer, flatlogs::logHeader::msgLen( buffer ) ) );
            REQUIRE_FALSE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
        }
    }
}

/// Retain early-reader populations without requiring successful or observed version-one completion.
/** \ingroup dmSyncoTelemetry_unit_test */
TEST_CASE( "Synco telemetry separates early read release from full SDK completion", "[dmSyncoTelemetry]" )
{
    // clang-format off
#ifdef DMSYNCO_TELEMETRY_TEST_DOXYGEN_REF
    MagAOX::logger::telem_dmsynco::messageT;
    MagAOX::logger::telem_dmsynco::verify;
#endif
    // clang-format on
    auto release                           = sampleReader();
    release.kind                           = dmCompletion::Kind::readers;
    release.worst.completed.record.kind    = dmCompletion::Kind::readers;
    release.worst.completed.record.state   = dmCompletion::State::released;
    release.worst.completed.record.outcome = dmTiming::Outcome::unknown;

    SECTION( "SDK result has not yet been observed" )
    {
        CHECK( release.worst.completed.record.backendResult == dmCompletion::unavailableResult );
    }

    SECTION( "An actual later SDK failure does not undo finished reads" )
    {
        release.worst.completed.record.outcome       = dmTiming::Outcome::failed;
        release.worst.completed.record.backendResult = 7;
        release.total.sdkResults                     = 1;
        release.total.sdkFailed                      = 1;
        release.total.firstEnriched                  = 1;
    }

    auto input  = message( samplePrimary(), {}, {}, {}, {}, release );
    auto buffer = envelope( input );
    REQUIRE( telem_dmsynco::verify( buffer, flatlogs::logHeader::msgLen( buffer ) ) );
    REQUIRE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
    const auto *record = GetTelem_dmsynco_fb( input.builder.GetBufferPointer() );
    CHECK( record->reader_timing() == nullptr );
    checkReader( record->reader_release_timing(), release );
    CHECK( telem_dmsynco::msgString( input.builder.GetBufferPointer(), input.builder.GetSize() )
               .find( "reader_release_mode=shadow_observation" ) != std::string::npos );
    const auto          json = telem_dmsynco::msgJSON( input.builder.GetBufferPointer(),
                                              input.builder.GetSize(),
                                              reinterpret_cast<const uint8_t *>( telem_dmsynco_bfbs ),
                                              telem_dmsynco_bfbs_len );
    flatbuffers::Parser parser;
    REQUIRE( parser.Deserialize( reinterpret_cast<const uint8_t *>( telem_dmsynco_bfbs ), telem_dmsynco_bfbs_len ) );
    REQUIRE( parser.Parse( json.c_str() ) );
    checkReader( GetTelem_dmsynco_fb( parser.builder_.GetBufferPointer() )->reader_release_timing(), release );
}

/// Reject endpoint-protocol confusion and inconsistent opportunistic SDK-result counts.
/** \ingroup dmSyncoTelemetry_unit_test */
TEST_CASE( "Synco telemetry validates independent release protocol and SDK coverage", "[dmSyncoTelemetry]" )
{
    // clang-format off
#ifdef DMSYNCO_TELEMETRY_TEST_DOXYGEN_REF
    MagAOX::logger::telem_dmsynco::verify;
#endif
    // clang-format on
    auto release                           = sampleReader();
    release.kind                           = dmCompletion::Kind::readers;
    release.worst.completed.record.kind    = dmCompletion::Kind::readers;
    release.worst.completed.record.state   = dmCompletion::State::released;
    release.worst.completed.record.outcome = dmTiming::Outcome::unknown;

    SECTION( "Full-completion record cannot stand for early release" )
    {
        release.worst.completed.record.kind = dmCompletion::Kind::completion;
    }

    SECTION( "Full-completion table cannot carry a release population" )
    {
        auto input  = message( samplePrimary(), {}, {}, {}, release );
        auto buffer = envelope( input );
        REQUIRE_FALSE( telem_dmsynco::verify( buffer, flatlogs::logHeader::msgLen( buffer ) ) );
        return;
    }

    SECTION( "SDK count does not equal its actual outcomes" )
    {
        release.total.sdkResults = 2;
        release.total.sdkSuccess = 1;
    }

    SECTION( "SDK results exceed observed release records" )
    {
        release.total.sdkResults = release.total.completion + 1;
        release.total.sdkSuccess = release.total.sdkResults;
    }

    auto input  = message( samplePrimary(), {}, {}, {}, {}, release );
    auto buffer = envelope( input );
    REQUIRE_FALSE( telem_dmsynco::verify( buffer, flatlogs::logHeader::msgLen( buffer ) ) );
    REQUIRE_FALSE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
}

/// Keep six-population legacy records readable while requiring complete optional wake/capture pairs.
/** \ingroup dmSyncoTelemetry_unit_test */
TEST_CASE( "Synco reader telemetry accepts legacy receipt absence and rejects partial splits", "[dmSyncoTelemetry]" )
{
    // clang-format off
#ifdef DMSYNCO_TELEMETRY_TEST_DOXYGEN_REF
    MagAOX::logger::telem_dmsynco::verify;
    MagAOX::logger::telem_dmsynco::msgString;
#endif
    // clang-format on
    auto        input  = message( samplePrimary(), {}, {}, {}, sampleReader() );
    const auto *reader = GetTelem_dmsynco_fb( input.builder.GetBufferPointer() )->reader_timing();

    SECTION( "Legacy writer omits both split populations and receipt provenance" )
    {
        omitField( reader, DmSyncoReaderTiming::VT_WAKE );
        omitField( reader, DmSyncoReaderTiming::VT_CAPTURE );
        omitField( reader->total(), DmSyncoReaderQuality::VT_RECEIPT_UNAVAILABLE );
        omitField( reader->total(), DmSyncoReaderQuality::VT_RECEIPT_INVALID );
        omitField( reader->worst()->completed(), DmSyncoReaderCompletion::VT_RECEIPT_NS );
        auto buffer = envelope( input );
        REQUIRE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
        CHECK( reader->wake() == nullptr );
        CHECK( reader->capture() == nullptr );
        CHECK( reader->total()->receipt_unavailable() == 0 );
        CHECK( reader->total()->receipt_invalid() == 0 );
        CHECK( reader->total()->first_enriched() == 0 );
        CHECK( reader->worst()->completed()->receipt_ns() == 0 );
        CHECK_FALSE( reader->worst_notification_valid() );
        CHECK( reader->worst_notification() == nullptr );
        const auto text = telem_dmsynco::msgString( input.builder.GetBufferPointer(), input.builder.GetSize() );
        CHECK( text.find( "readerWake=unavailable" ) != std::string::npos );
        CHECK( text.find( "readerCapture=unavailable" ) != std::string::npos );
        CHECK( text.find( "readerNotification[n=3 valid=1]" ) != std::string::npos );
    }

    SECTION( "Wake alone is incomplete" )
    {
        omitField( reader, DmSyncoReaderTiming::VT_CAPTURE );
        auto buffer = envelope( input );
        REQUIRE_FALSE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
    }

    SECTION( "Capture alone is incomplete" )
    {
        omitField( reader, DmSyncoReaderTiming::VT_WAKE );
        auto buffer = envelope( input );
        REQUIRE_FALSE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
    }
}

/// Preserve genuine completion statistics when a missing or invalid semaphore timestamp prevents only the split.
/** \ingroup dmSyncoTelemetry_unit_test */
TEST_CASE( "Synco reader telemetry serializes empty splits without discarding raw receipts", "[dmSyncoTelemetry]" )
{
    // clang-format off
#ifdef DMSYNCO_TELEMETRY_TEST_DOXYGEN_REF
    MagAOX::logger::telem_dmsynco::messageT;
    MagAOX::logger::telem_dmsynco::verify;
#endif
    // clang-format on
    auto summary                                                              = sampleReader();
    summary.metrics[dmReaderTiming::index( dmReaderTiming::Metric::wake )]    = {};
    summary.metrics[dmReaderTiming::index( dmReaderTiming::Metric::capture )] = {};

    SECTION( "Legacy producer has no receipt timestamp" )
    {
        summary.worst.completed.receiptNs = 0;
    }

    SECTION( "Bad timestamp remains diagnostic provenance" )
    {
        summary.worst.completed.receiptNs = -17;
    }

    auto input  = message( samplePrimary(), {}, {}, {}, summary );
    auto buffer = envelope( input );
    REQUIRE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
    const auto *reader = GetTelem_dmsynco_fb( input.builder.GetBufferPointer() )->reader_timing();
    checkReader( reader, summary );
    REQUIRE( reader->wake() != nullptr );
    REQUIRE( reader->capture() != nullptr );
    CHECK( reader->wake()->count() == 0 );
    CHECK( reader->capture()->count() == 0 );
    const auto text = telem_dmsynco::msgString( input.builder.GetBufferPointer(), input.builder.GetSize() );
    CHECK( text.find( "readerWake[n=0 valid=0]=unavailable" ) != std::string::npos );
    CHECK( text.find( "readerCapture[n=0 valid=0]=unavailable" ) != std::string::npos );
}

/// Reject split coverage inconsistencies and impossible lifetime receipt-quality accounting.
/** \ingroup dmSyncoTelemetry_unit_test */
TEST_CASE( "Synco reader telemetry validates receipt-split coverage and quality", "[dmSyncoTelemetry]" )
{
    // clang-format off
#ifdef DMSYNCO_TELEMETRY_TEST_DOXYGEN_REF
    MagAOX::logger::telem_dmsynco::verify;
#endif
    // clang-format on
    for( unsigned variant = 0; variant < 6; ++variant )
    {
        CAPTURE( variant );
        auto summary = sampleReader();

        switch( variant )
        {
        case 0:
            ++summary.metrics[dmReaderTiming::index( dmReaderTiming::Metric::wake )].count;
            break;
        case 1:
            summary.metrics[dmReaderTiming::index( dmReaderTiming::Metric::capture )].span /= 2;
            break;
        case 2:
            --summary.metrics[dmReaderTiming::index( dmReaderTiming::Metric::notification )].count;
            break;
        case 3:
            summary.total.receiptUnavailable = summary.total.completion;
            break;
        case 4:
            summary.total.receiptUnavailable = std::numeric_limits<uint64_t>::max();
            summary.total.receiptInvalid     = 1;
            break;
        case 5:
            summary.total.firstEnriched = 1;
            break;
        }

        auto input  = message( samplePrimary(), {}, {}, {}, summary );
        auto buffer = envelope( input );
        REQUIRE_FALSE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
    }
}

/// Retain separate lifetime notification-tail evidence with exact F/W/R stamps through binary and JSON readers.
/** \ingroup dmSyncoTelemetry_unit_test */
TEST_CASE( "Synco reader telemetry retains the independent worst notification", "[dmSyncoTelemetry]" )
{
    // clang-format off
#ifdef DMSYNCO_TELEMETRY_TEST_DOXYGEN_REF
    MagAOX::logger::telem_dmsynco::messageT;
    MagAOX::logger::telem_dmsynco::verify;
    MagAOX::logger::telem_dmsynco::msgString;
#endif
    // clang-format on
    auto summary                   = sampleReader();
    summary.worstNotificationValid = true;
    summary.worstNotification      = summary.worst;
    auto &sample                   = summary.worstNotification;
    ++sample.completed.record.sequence;
    sample.completed.record.finishedNs = sample.baseNs + 10000;
    sample.completed.observedNs        = sample.baseNs + 60000;
    sample.completed.receiptNs         = sample.baseNs + 50000;
    sample.offsetNs = sample.requiredNs = 10000;
    sample.observedHoldNs               = 60000;
    sample.slackNs                      = 145000;

    SECTION( "Distinct maxima roundtrip without changing either selection" )
    {
        auto input  = message( samplePrimary(), {}, {}, {}, summary );
        auto buffer = envelope( input );
        REQUIRE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
        checkReader( GetTelem_dmsynco_fb( input.builder.GetBufferPointer() )->reader_timing(), summary );
        const auto text = telem_dmsynco::msgString( input.builder.GetBufferPointer(), input.builder.GetSize() );
        CHECK( text.find( "reader_worst_notification_valid=1" ) != std::string::npos );
        CHECK( text.find( "reader_worst_notification={clock:monotonic" ) != std::string::npos );
        CHECK( text.find( "receipt_ns:1000050000,observed_ns:1000060000" ) != std::string::npos );
        const auto          json = telem_dmsynco::msgJSON( input.builder.GetBufferPointer(),
                                                  input.builder.GetSize(),
                                                  reinterpret_cast<const uint8_t *>( telem_dmsynco_bfbs ),
                                                  telem_dmsynco_bfbs_len );
        flatbuffers::Parser parser;
        REQUIRE(
            parser.Deserialize( reinterpret_cast<const uint8_t *>( telem_dmsynco_bfbs ), telem_dmsynco_bfbs_len ) );
        REQUIRE( parser.Parse( json.c_str() ) );
        checkReader( GetTelem_dmsynco_fb( parser.builder_.GetBufferPointer() )->reader_timing(), summary );
    }

    SECTION( "Malformed notification evidence is rejected independently of the original worst" )
    {
        ++sample.offsetNs;
        auto input  = message( samplePrimary(), {}, {}, {}, summary );
        auto buffer = envelope( input );
        REQUIRE_FALSE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
    }

    SECTION( "Validity flag cannot outlive its evidence" )
    {
        auto        input  = message( samplePrimary(), {}, {}, {}, summary );
        const auto *reader = GetTelem_dmsynco_fb( input.builder.GetBufferPointer() )->reader_timing();
        omitField( reader, DmSyncoReaderTiming::VT_WORST_NOTIFICATION );
        auto buffer = envelope( input );
        REQUIRE_FALSE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
    }

    SECTION( "Evidence cannot lose its validity flag" )
    {
        auto        input  = message( samplePrimary(), {}, {}, {}, summary );
        const auto *reader = GetTelem_dmsynco_fb( input.builder.GetBufferPointer() )->reader_timing();
        omitField( reader, DmSyncoReaderTiming::VT_WORST_NOTIFICATION_VALID );
        auto buffer = envelope( input );
        REQUIRE_FALSE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
    }
}

/// Preserve opt-in readiness assumptions, four exclusive skip reasons and full-width actual release evidence.
/** \ingroup dmSyncoTelemetry_unit_test */
TEST_CASE( "Synco telemetry preserves conditional reader-release readiness", "[dmSyncoTelemetry]" )
{
    // clang-format off
#ifdef DMSYNCO_TELEMETRY_TEST_DOXYGEN_REF
    MagAOX::logger::telem_dmsynco::messageT;
    MagAOX::logger::telem_dmsynco::verify;
    MagAOX::logger::telem_dmsynco::msgString;
#endif
    // clang-format on
    auto simulation = sampleReadiness();

    SECTION( "Actual release has no SDK result yet" )
    {
        CHECK( simulation.latestRelease.record.outcome == dmTiming::Outcome::unknown );
    }

    SECTION( "Later actual SDK success is not another virtual command" )
    {
        simulation.latestRelease.record.outcome       = dmTiming::Outcome::success;
        simulation.latestRelease.record.backendResult = 0;
    }

    auto input  = message( samplePrimary(), {}, simulation );
    auto buffer = envelope( input );
    REQUIRE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
    const auto *record = GetTelem_dmsynco_fb( input.builder.GetBufferPointer() );
    checkSimulation( record->simulation(), simulation );
    CHECK( record->delivered() == 11 );
    const auto text = telem_dmsynco::msgString( input.builder.GetBufferPointer(), input.builder.GetSize() );
    CHECK( text.find( "simulation_mode=shadow_reader_release_ready" ) != std::string::npos );
    CHECK( text.find( "virtualAuxReader_ns:30000,assumedBackend_ns:155000" ) != std::string::npos );
    CHECK( text.find( "readerCycleFeasible:1,backendCycleFeasible:1" ) != std::string::npos );
    CHECK( text.find( "releaseMissing:1,releaseLate:1,backendBusy:0,backendWrap:0" ) != std::string::npos );
    CHECK( text.find( "simulation_release_evidence={clock:monotonic" ) != std::string::npos );
    CHECK( text.find( "finished_ns:" + std::to_string( simulation.latestRelease.record.finishedNs ) ) !=
           std::string::npos );
    CHECK( text.find( "simulation_delivered" ) == std::string::npos );
    CHECK( text.find( "simulation_armed" ) == std::string::npos );
    const auto          json = telem_dmsynco::msgJSON( input.builder.GetBufferPointer(),
                                              input.builder.GetSize(),
                                              reinterpret_cast<const uint8_t *>( telem_dmsynco_bfbs ),
                                              telem_dmsynco_bfbs_len );
    flatbuffers::Parser parser;
    REQUIRE( parser.Deserialize( reinterpret_cast<const uint8_t *>( telem_dmsynco_bfbs ), telem_dmsynco_bfbs_len ) );
    REQUIRE( parser.Parse( json.c_str() ) );
    checkSimulation( GetTelem_dmsynco_fb( parser.builder_.GetBufferPointer() )->simulation(), simulation );
}

/// Keep legacy fixed-hold records unchanged and allow readiness warmup without invented evidence.
/** \ingroup dmSyncoTelemetry_unit_test */
TEST_CASE( "Synco telemetry keeps readiness optional and distinguishes unobserved releases", "[dmSyncoTelemetry]" )
{
    // clang-format off
#ifdef DMSYNCO_TELEMETRY_TEST_DOXYGEN_REF
    MagAOX::logger::telem_dmsynco::messageT;
    MagAOX::logger::telem_dmsynco::verify;
#endif
    // clang-format on
    SECTION( "Legacy fixed-hold mode has no new readiness table or text" )
    {
        const auto source = sampleSimulation();
        auto       input  = message( samplePrimary(), {}, source );
        auto       buffer = envelope( input );
        REQUIRE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
        const auto *simulation = GetTelem_dmsynco_fb( input.builder.GetBufferPointer() )->simulation();
        checkSimulation( simulation, source );
        CHECK_FALSE( flatbuffers::IsFieldPresent( simulation, DmSyncoSimulation::VT_READINESS ) );
        const auto text = telem_dmsynco::msgString( input.builder.GetBufferPointer(), input.builder.GetSize() );
        CHECK( text.find( "simulation_mode=shadow_conditional" ) != std::string::npos );
        CHECK( text.find( "simulation_readiness" ) == std::string::npos );
        CHECK( text.find( "releaseMissing" ) == std::string::npos );
    }

    SECTION( "New-mode warmup declares assumptions but no real release" )
    {
        dmSyncoSimulation::Summary source;
        source.enabled            = true;
        source.steps              = 3;
        source.readerReleaseReady = true;
        source.virtualReaderNs    = 30000;
        source.backendNs          = 155000;
        auto input                = message( samplePrimary(), {}, source );
        auto buffer               = envelope( input );
        REQUIRE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
        checkSimulation( GetTelem_dmsynco_fb( input.builder.GetBufferPointer() )->simulation(), source );
    }
}

/// Reject malformed readiness assumptions, evidence and skip accounting without weakening legacy validation.
/** \ingroup dmSyncoTelemetry_unit_test */
TEST_CASE( "Synco telemetry validates shadow readiness assumptions and provenance", "[dmSyncoTelemetry]" )
{
    // clang-format off
#ifdef DMSYNCO_TELEMETRY_TEST_DOXYGEN_REF
    MagAOX::logger::telem_dmsynco::verify;
#endif
    // clang-format on
    for( unsigned variant = 0; variant < 20; ++variant )
    {
        CAPTURE( variant );
        auto simulation = sampleReadiness();

        switch( variant )
        {
        case 0:
            simulation.virtualReaderNs = 0;
            break;
        case 1:
            simulation.virtualReaderNs = 1000000001;
            break;
        case 2:
            simulation.backendNs = simulation.virtualReaderNs - 1;
            break;
        case 3:
            simulation.backendNs = 1000000001;
            break;
        case 4:
            simulation.readiness.qualified = simulation.readiness.observed + 1;
            break;
        case 5:
            simulation.readiness.sdkFailed = simulation.readiness.observed + 1;
            break;
        case 6:
            simulation.latestRelease.coherent = false;
            break;
        case 7:
            simulation.latestRelease.sourceEpoch = 0;
            break;
        case 8:
            simulation.latestRelease.observedNs = simulation.latestRelease.record.finishedNs - 1;
            break;
        case 9:
            simulation.latestRelease.record.kind = dmCompletion::Kind::completion;
            break;
        case 10:
            simulation.latestRelease.record.state = dmCompletion::State::busy;
            break;
        case 11:
            simulation.latestRelease.record.outcome       = dmTiming::Outcome::failed;
            simulation.latestRelease.record.backendResult = 7;
            break;
        case 12:
            simulation.latestRelease.record.backendResult = 0;
            break;
        case 13:
            simulation.latestRelease.record.inputCounter = 0;
            break;
        case 14:
            ++simulation.outcomes[0].releaseMissing;
            break;
        case 15:
            simulation.readerReleaseReady = false;
            break;
        case 16:
            simulation.readerCycleFeasible = false;
            break;
        case 17:
            simulation.cadenceMinPeriodNs = simulation.periodNs + 1;
            break;
        case 18:
            simulation.latestRelease.record.publisherPid = 0;
            break;
        case 19:
            simulation.latestRelease.record.publicationSec = std::numeric_limits<int64_t>::max();
            break;
        }

        auto input  = message( samplePrimary(), {}, simulation );
        auto buffer = envelope( input );
        REQUIRE_FALSE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
    }

    SECTION( "Readiness table cannot masquerade as legacy mode" )
    {
        auto        input     = message( samplePrimary(), {}, sampleReadiness() );
        const auto *readiness = GetTelem_dmsynco_fb( input.builder.GetBufferPointer() )->simulation()->readiness();
        omitField( readiness, DmSyncoReadiness::VT_READER_RELEASE_READY );
        auto buffer = envelope( input );
        REQUIRE_FALSE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
    }
}

/// Preserve independent pipeline assumptions, future credit reservations and exclusive auxiliary skip reasons.
/** \ingroup dmSyncoTelemetry_unit_test */
TEST_CASE( "Synco telemetry preserves conditional pipeline resources", "[dmSyncoTelemetry]" )
{
    // clang-format off
#ifdef DMSYNCO_TELEMETRY_TEST_DOXYGEN_REF
    MagAOX::logger::telem_dmsynco::messageT;
    MagAOX::logger::telem_dmsynco::verify;
    MagAOX::logger::telem_dmsynco::msgString;
#endif
    // clang-format on

    const auto simulation = samplePipeline();
    auto       input      = message( samplePrimary(), {}, simulation );
    auto       buffer     = envelope( input );
    REQUIRE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
    const auto *record = GetTelem_dmsynco_fb( input.builder.GetBufferPointer() );
    checkSimulation( record->simulation(), simulation );
    CHECK( record->delivered() == 11 );
    const auto text = telem_dmsynco::msgString( input.builder.GetBufferPointer(), input.builder.GetSize() );
    CHECK( text.find( "simulation_mode=shadow_pipeline" ) != std::string::npos );
    CHECK( text.find( "simulation_pipeline={prepare_ns:35000,sdkService_ns:120000,post_ns:150000" ) !=
           std::string::npos );
    CHECK( text.find( "capacity:3,inFlight:3,maximumInFlight:3" ) != std::string::npos );
    CHECK( text.find( "discarded:2,mainSdkWaits:17,mainCreditMisses:19,resets:2" ) != std::string::npos );
    CHECK( text.find( "maximumMainWait_ns:16000,mainClearanceShortfall_ns:201000" ) != std::string::npos );
    CHECK( text.find( "sdkCycleFeasible:1,postCycleFeasible:1,creditCycleFeasible:1,mainProtectionFeasible:0" ) !=
           std::string::npos );
    CHECK( text.find( "preparationBusy:1,creditFull:1" ) != std::string::npos );
    CHECK( text.find( "assumedBackend_ns" ) == std::string::npos );
    CHECK( text.find( "simulation_delivered" ) == std::string::npos );

    const auto          json = telem_dmsynco::msgJSON( input.builder.GetBufferPointer(),
                                              input.builder.GetSize(),
                                              reinterpret_cast<const uint8_t *>( telem_dmsynco_bfbs ),
                                              telem_dmsynco_bfbs_len );
    flatbuffers::Parser parser;
    REQUIRE( parser.Deserialize( reinterpret_cast<const uint8_t *>( telem_dmsynco_bfbs ), telem_dmsynco_bfbs_len ) );
    REQUIRE( parser.Parse( json.c_str() ) );
    checkSimulation( GetTelem_dmsynco_fb( parser.builder_.GetBufferPointer() )->simulation(), simulation );
}

/// Keep historical fixed-hold/readiness tables compatible and permit an explicitly unobserved pipeline warmup.
/** \ingroup dmSyncoTelemetry_unit_test */
TEST_CASE( "Synco telemetry keeps pipeline modeling optional", "[dmSyncoTelemetry]" )
{
    // clang-format off
#ifdef DMSYNCO_TELEMETRY_TEST_DOXYGEN_REF
    MagAOX::logger::telem_dmsynco::messageT;
    MagAOX::logger::telem_dmsynco::verify;
#endif
    // clang-format on

    for( const bool readiness : { false, true } )
    {
        auto        source     = readiness ? sampleReadiness() : sampleSimulation();
        auto        input      = message( samplePrimary(), {}, source );
        const auto *simulation = GetTelem_dmsynco_fb( input.builder.GetBufferPointer() )->simulation();
        REQUIRE( simulation->pipeline() == nullptr );
        CHECK_FALSE( flatbuffers::IsFieldPresent( simulation, DmSyncoSimulation::VT_PIPELINE ) );

        for( const auto *step : *simulation->outcomes() )
        {
            omitField( step, DmSyncoSimulationStep::VT_PREPARATION_BUSY );
            omitField( step, DmSyncoSimulationStep::VT_CREDIT_FULL );
        }

        auto buffer = envelope( input );
        REQUIRE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
        checkSimulation( simulation, source );
        const auto text = telem_dmsynco::msgString( input.builder.GetBufferPointer(), input.builder.GetSize() );
        CHECK( text.find( "simulation_pipeline" ) == std::string::npos );
        CHECK( text.find( "preparationBusy" ) == std::string::npos );
        CHECK( text.find( readiness ? "simulation_mode=shadow_reader_release_ready"
                                    : "simulation_mode=shadow_conditional" ) != std::string::npos );
    }

    SECTION( "Warmup carries assumptions with no invented admissions" )
    {
        dmSyncoSimulation::Summary source;
        source.enabled            = true;
        source.steps              = 3;
        source.readerReleaseReady = true;
        source.pipeline.enabled   = true;
        auto input                = message( samplePrimary(), {}, source );
        auto buffer               = envelope( input );
        REQUIRE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
        checkSimulation( GetTelem_dmsynco_fb( input.builder.GetBufferPointer() )->simulation(), source );
    }

    SECTION( "Unused legacy backend no longer constrains the virtual reader assumption" )
    {
        auto source      = samplePipeline();
        source.backendNs = source.virtualReaderNs - 1;
        auto input       = message( samplePrimary(), {}, source );
        auto buffer      = envelope( input );
        REQUIRE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
    }
}

/// Reject malformed bounds, credit accounting, feasibility and exclusive skips without weakening older records.
/** \ingroup dmSyncoTelemetry_unit_test */
TEST_CASE( "Synco telemetry validates pipeline resource bounds and credit conservation", "[dmSyncoTelemetry]" )
{
    // clang-format off
#ifdef DMSYNCO_TELEMETRY_TEST_DOXYGEN_REF
    MagAOX::logger::telem_dmsynco::verify;
#endif
    // clang-format on

    for( unsigned variant = 0; variant < 24; ++variant )
    {
        CAPTURE( variant );
        auto source = samplePipeline();

        switch( variant )
        {
        case 0:
            source.pipeline.prepareNs = 0;
            break;
        case 1:
            source.pipeline.prepareNs = 1000000001;
            break;
        case 2:
            source.pipeline.sdkServiceNs = 0;
            break;
        case 3:
            source.pipeline.sdkServiceNs = 1000000001;
            break;
        case 4:
            source.pipeline.postNs = 0;
            break;
        case 5:
            source.pipeline.postNs = 1000000001;
            break;
        case 6:
            source.pipeline.capacity = 2;
            break;
        case 7:
            source.pipeline.maximumInFlight = 2;
            break;
        case 8:
            source.pipeline.inFlight = source.pipeline.maximumInFlight = 4;
            break;
        case 9:
            ++source.pipeline.retired;
            break;
        case 10:
            ++source.pipeline.discarded;
            break;
        case 11:
            source.pipeline.sdkReadyNs = -1;
            break;
        case 12:
            source.pipeline.postReadyNs = -1;
            break;
        case 13:
            source.pipeline.maximumMainWaitNs = -1;
            break;
        case 14:
            source.pipeline.mainClearanceShortfallNs = -1;
            break;
        case 15:
            source.readerReleaseReady = false;
            break;
        case 16:
            source.backendCycleFeasible = true;
            break;
        case 17:
            source.feasible = true;
            break;
        case 18:
            ++source.outcomes[1].preparationBusy;
            break;
        case 19:
            ++source.outcomes[1].creditFull;
            break;
        case 20:
            source.pipeline.enabled = false;
            break;
        case 21:
            source.backendNs = 0;
            break;
        case 22:
            source.backendNs = 1000000001;
            break;
        case 23:
            source.pipeline.mainAdmitted      = std::numeric_limits<uint64_t>::max();
            source.pipeline.auxiliaryAdmitted = 1;
            source.pipeline.retired = source.pipeline.discarded = source.pipeline.inFlight = 0;
            break;
        }

        auto input  = message( samplePrimary(), {}, source );
        auto buffer = envelope( input );
        REQUIRE_FALSE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
    }

    SECTION( "A present resource table cannot disable itself" )
    {
        auto        input    = message( samplePrimary(), {}, samplePipeline() );
        const auto *pipeline = GetTelem_dmsynco_fb( input.builder.GetBufferPointer() )->simulation()->pipeline();
        omitField( pipeline, DmSyncoPipeline::VT_ENABLED );
        auto buffer = envelope( input );
        REQUIRE_FALSE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
    }

    SECTION( "Wide admission totals retain exact conservation" )
    {
        auto source                       = samplePipeline();
        source.pipeline.mainAdmitted      = std::numeric_limits<uint64_t>::max();
        source.pipeline.auxiliaryAdmitted = std::numeric_limits<uint64_t>::max();
        source.pipeline.retired           = std::numeric_limits<uint64_t>::max();
        source.pipeline.discarded         = std::numeric_limits<uint64_t>::max() - source.pipeline.inFlight;
        auto input                        = message( samplePrimary(), {}, source );
        auto buffer                       = envelope( input );
        REQUIRE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
        checkPipeline( GetTelem_dmsynco_fb( input.builder.GetBufferPointer() )->simulation()->pipeline(),
                       source.pipeline );
    }
}

} // namespace dmSyncoTelemetryTest
} // namespace loggerTest
} // namespace libXWCTest
