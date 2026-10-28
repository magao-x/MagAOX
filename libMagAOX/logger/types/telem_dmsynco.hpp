/** \file telem_dmsynco.hpp
 * \brief Syncopation timing, shadow cadence, reader protection and bounded estimator history telemetry.
 */
#ifndef logger_types_telem_dmsynco_hpp
#define logger_types_telem_dmsynco_hpp

#include <algorithm>
#include <iomanip>
#include <limits>
#include <sstream>
#include <vector>

#include <mx/math/floatUtils.hpp>

#include "../../common/dmCombTiming.hpp"
#include "../../common/dmPipelineTiming.hpp"
#include "../../common/dmReaderTiming.hpp"
#include "../../common/dmSyncoSimulation.hpp"
#include "../../common/dmTiming.hpp"
#include "generated/telem_dmsynco_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{

/// Periodic diagnostics recorded outside the real-time publication path.
struct telem_dmsynco : public flatbuffer_log
{
    static constexpr flatlogs::eventCodeT eventCode    = eventCodes::TELEM_DMSYNCO;    ///< Registered event code.
    static constexpr flatlogs::logPrioT   defaultLevel = flatlogs::logPrio::LOG_TELEM; ///< Telemetry priority.
    static inline timespec                lastRecord{}; ///< Last aggregate publication, owned by the telemetry system.

    /// Serialized aggregate message.
    struct messageT : public fbMessage
    {
        /// Serialize a coherent snapshot of scheduler diagnostics.
        messageT(
            int                         state /**< [in] schedule state */,
            int                         steps /**< [in] requested count */,
            int64_t                     period /**< [in] measured period, ns */,
            uint64_t                    delivered /**< [in] confirmed extra combinations */,
            uint64_t                    missed /**< [in] failed timing events */,
            uint64_t                    faults /**< [in] synchronization/resource faults */,
            uint64_t                    cached /**< [in] source-cache fallback count */,
            int64_t                     snapshot /**< [in] maximum whole-snapshot time, ns */,
            int64_t                     copy /**< [in] maximum whole-copy time, ns */,
            const std::vector<int64_t> &residuals /**< [in] latest per-step residuals, ns */,
            const std::vector<int64_t> &advances /**< [in] per-step wait advances, ns */,
            const dmTiming::Summary    &primary = {} /**< [in] observed primary cadence; disabled by default */,
            bool scheduleValid = true /**< [in] false when scheduler fields are unavailable placeholders */,
            const dmCombTiming::Summary     &combiner = {} /**< [in] independently observed combiner timing */,
            const dmTiming::Summary         &camera   = {} /**< [in] independent camera acquisition-stamp intervals */,
            const dmPipelineTiming::Summary &pipeline = {} /**< [in] upstream trace-stage populations */,
            const dmTiming::Summary         &cameraPublication = {} /**< [in] independent camera writetime intervals */,
            const dmSyncoSimulation::Summary    &simulation    = {} /**< [in] conditional shadow predictions */,
            const dmSyncoSimulation::TraceBatch &trace = {} /**< [in] drained overlap/estimator evidence and quality */,
            const dmReaderTiming::Summary &reader = {} /**< [in] independently observed software reader completion */ );
    };

    /// Validate optional cadence and stage populations while accepting legacy records.
    static bool verify( flatlogs::bufferPtrT &buffer /**< [in] complete log entry */,
                        flatlogs::msgLenT     len /**< [in] message length */ );

    /// Format the aggregate for log readers.
    static std::string msgString( void             *buffer /**< [in] message buffer */,
                                  flatlogs::msgLenT len /**< [in] message length, unused */ );

    /// This aggregate has no scalar logMeta projections; read its typed FlatBuffer fields.
    static logMetaDetail getAccessor( const std::string &member /**< [in] requested scalar member, unused */ );

  private:
    /// Serialize one independently counted timing population without changing its units or clock.
    static flatbuffers::Offset<DmSyncoTimingStats>
    timingStats( flatbuffers::FlatBufferBuilder &builder /**< [in/out] destination builder */,
                 const dmTiming::Stats          &stats /**< [in] interval or duration population in seconds */ );

    /// Serialize ordered primary excursions without changing their direction or denominator populations.
    static flatbuffers::Offset<DmSyncoIntervalPairs>
    intervalPairs( flatbuffers::FlatBufferBuilder &builder /**< [in/out] destination builder */,
                   const dmTiming::IntervalPairs  &pairs /**< [in] consecutive primary intervals in seconds */ );

    /// Serialize independently qualified combiner populations, provenance and quality counters.
    static flatbuffers::Offset<DmSyncoCombinerTiming>
    combinerTiming( flatbuffers::FlatBufferBuilder &builder /**< [in/out] destination builder */,
                    const dmCombTiming::Summary    &summary /**< [in] coherent combiner observation summary */ );

    /// Serialize primary publication, camera acquisition or camera publication intervals.
    static flatbuffers::Offset<DmSyncoPrimaryTiming>
    observationTiming( flatbuffers::FlatBufferBuilder &builder /**< [in/out] destination builder */,
                       const dmTiming::Summary        &summary /**< [in] retained external-stamp cadence */ );

    /// Serialize upstream stage identities, conditional durations and signed changes.
    static flatbuffers::Offset<DmSyncoPipelineTiming>
    pipelineTiming( flatbuffers::FlatBufferBuilder  &builder /**< [in/out] destination builder */,
                    const dmPipelineTiming::Summary &summary /**< [in] coherent pipeline observation */ );

    /// Serialize optional reader-protection measurements with signed offsets and lifetime worst provenance.
    static flatbuffers::Offset<DmSyncoReaderTiming>
    readerTiming( flatbuffers::FlatBufferBuilder &builder /**< [in/out] destination builder */,
                  const dmReaderTiming::Summary  &summary /**< [in] matched observation and completion summary */ );

    /// Serialize exact matched command identities and local monotonic evidence for the worst required hold.
    static flatbuffers::Offset<DmSyncoReaderSample>
    readerSample( flatbuffers::FlatBufferBuilder &builder /**< [in/out] destination builder */,
                  const dmReaderTiming::Sample   &sample /**< [in] matched command evidence */ );

    /// Serialize conditional shadow outcomes without claiming auxiliary delivery.
    static flatbuffers::Offset<DmSyncoSimulation>
    simulationTiming( flatbuffers::FlatBufferBuilder      &builder /**< [in/out] destination builder */,
                      const dmSyncoSimulation::Summary    &summary /**< [in] coherent shadow prediction snapshot */,
                      const dmSyncoSimulation::TraceBatch &trace /**< [in] drained overlap evidence */ );

    /// Serialize a bounded overlap batch without converting producer timestamps.
    static flatbuffers::Offset<DmSyncoOverlapTrace>
    overlapTrace( flatbuffers::FlatBufferBuilder      &builder /**< [in/out] destination builder */,
                  const dmSyncoSimulation::TraceBatch &trace /**< [in] lifetime queue quality and drained records */ );

    /// Serialize triggered estimator history without recalculating its medians or corrections.
    static flatbuffers::Offset<DmSyncoEstimatorTrace> estimatorTrace(
        flatbuffers::FlatBufferBuilder      &builder /**< [in/out] destination builder */,
        const dmSyncoSimulation::TraceBatch &trace /**< [in] selected samples and independent queue quality */ );

    /// Serialize optional slow cadence measurement and its qualification on the monotonic clock.
    static flatbuffers::Offset<DmSyncoCadence>
    cadenceTiming( flatbuffers::FlatBufferBuilder   &builder /**< [in/out] destination builder */,
                   const dmSyncoSimulation::Cadence &cadence /**< [in] coherent long-baseline measurement */ );

    /// Check a fractional period and its rounded legacy nanosecond projection.
    static bool validPeriod( uint64_t periodQ32 /**< [in] binary-scaled period; zero denotes legacy absence */,
                             int64_t  periodNs /**< [in] rounded integer period */ );

    /// Check the signed remainder of a target rounded to the nearest nanosecond.
    static bool validFraction( int64_t fractionQ32 /**< [in] binary-scaled signed fractional nanoseconds */ );

    /// Validate optional long-baseline cadence qualification without assuming current-session ownership.
    static bool validCadence( const DmSyncoCadence *cadence /**< [in] optional slow cadence measurement */ );

    /// Check finite populated statistics and sample-variance validity.
    static bool validStats( const DmSyncoTimingStats *stats /**< [in] required nested cadence population */,
                            bool signedValues = false /**< [in] allow negative latency changes */ );

    /// Check independently observed primary or camera cadence without changing legacy requirements.
    static bool validObservation( const DmSyncoPrimaryTiming *summary /**< [in] optional cadence summary */ );

    /// Check retained pipeline topology and conditional-population count bounds.
    static bool validPipeline( const DmSyncoPipelineTiming *summary /**< [in] optional trace-stage summary */ );

    /// Check optional reader matching, signed populations, lifetime maxima and exact worst-command evidence.
    static bool validReader( const DmSyncoReaderTiming *summary /**< [in] optional software reader measurement */ );

    /// Recompute a matched command's signed reader-protection arithmetic without crossing clock domains.
    static bool validReaderSample( const DmSyncoReaderSample *sample /**< [in] lifetime matched-command evidence */ );

    /// Check optional shadow model readiness and independently counted step outcomes.
    static bool validSimulation( const DmSyncoSimulation *summary /**< [in] optional conditional simulation */ );

    /// Check bounded lifetime accounting, event provenance and classified overlap intervals.
    static bool validOverlapTrace( const DmSyncoOverlapTrace *trace /**< [in] optional raw overlap evidence */ );

    /// Check independent triggered coverage and actual estimator arithmetic across configuration epochs.
    static bool validEstimatorTrace( const DmSyncoEstimatorTrace *trace /**< [in] optional estimator history */ );

    /// Format independent primary/camera cadence and ordered excursions with distinct names.
    static void
    formatObservation( std::ostream &output /**< [in/out] formatted aggregate */,
                       const char   *prefix /**< [in] primary, camera acquisition or camera publication identifier */,
                       const char   *intervalName /**< [in] external-stamp interval label */,
                       const DmSyncoPrimaryTiming *summary /**< [in] optional cadence summary */ );

    /// Format retained pipeline stages and their independently counted conditional populations.
    static void formatPipeline( std::ostream                &output /**< [in/out] formatted aggregate */,
                                const DmSyncoPipelineTiming *summary /**< [in] optional pipeline summary */ );

    /// Format signed reader offsets in seconds and retain exact identities and monotonic worst-event evidence.
    static void formatReader( std::ostream              &output /**< [in/out] formatted aggregate */,
                              const DmSyncoReaderTiming *summary /**< [in] optional software reader measurement */ );

    /// Append explicitly hypothetical prediction and per-step outcome diagnostics when enabled.
    static void formatSimulation( std::ostream            &output /**< [in/out] formatted aggregate */,
                                  const DmSyncoSimulation *summary /**< [in] optional conditional simulation */ );

    /// Append exact nanosecond evidence with explicit monotonic and unconverted producer clock labels.
    static void formatOverlapTrace( std::ostream              &output /**< [in/out] formatted aggregate */,
                                    const DmSyncoOverlapTrace *trace /**< [in] optional raw overlap evidence */ );

    /// Append selected pre/post samples and independent coverage/loss accounting with explicit clock labels.
    static void formatEstimatorTrace( std::ostream                &output /**< [in/out] formatted aggregate */,
                                      const DmSyncoEstimatorTrace *trace /**< [in] optional estimator history */ );

    /// Format cadence qualification and retain the exact binary rational period alongside nanoseconds.
    static void formatCadence( std::ostream         &output /**< [in/out] formatted aggregate */,
                               const char           *prefix /**< [in] stable record label */,
                               const DmSyncoCadence *cadence /**< [in] optional slow cadence measurement */ );

    /// Validate optional ordered-pair populations and their conditional count relationships.
    static bool
    validIntervalPairs( const DmSyncoIntervalPairs *pairs /**< [in] optional ordered primary excursions */ );

    /// Append one population, distinguishing unavailable variance from a measured zero.
    static void formatStats( std::ostream             &output /**< [in/out] formatted aggregate */,
                             const char               *name /**< [in] stable endpoint-population label */,
                             const DmSyncoTimingStats *stats /**< [in] validated cadence population */ );
};

inline flatbuffers::Offset<DmSyncoTimingStats> telem_dmsynco::timingStats( flatbuffers::FlatBufferBuilder &builder,
                                                                           const dmTiming::Stats          &stats )
{
    return CreateDmSyncoTimingStats(
        builder, stats.count, stats.mean, stats.stddev, stats.min, stats.max, stats.span, stats.valid );
}

inline flatbuffers::Offset<DmSyncoIntervalPairs> telem_dmsynco::intervalPairs( flatbuffers::FlatBufferBuilder &builder,
                                                                               const dmTiming::IntervalPairs  &pairs )
{
    const auto afterLate    = timingStats( builder, pairs.afterLate );
    const auto afterEarly   = timingStats( builder, pairs.afterEarly );
    const auto lateEarlySum = timingStats( builder, pairs.lateEarlySum );
    const auto earlyLateSum = timingStats( builder, pairs.earlyLateSum );

    return CreateDmSyncoIntervalPairs( builder,
                                       pairs.enabled,
                                       pairs.intervals,
                                       pairs.pairs,
                                       pairs.median,
                                       pairs.threshold,
                                       afterLate,
                                       afterEarly,
                                       lateEarlySum,
                                       earlyLateSum );
}

inline flatbuffers::Offset<DmSyncoCombinerTiming>
telem_dmsynco::combinerTiming( flatbuffers::FlatBufferBuilder &builder, const dmCombTiming::Summary &summary )
{
    const auto publicationInterval =
        timingStats( builder, summary.metrics[dmCombTiming::index( dmCombTiming::Metric::publicationInterval )] );
    const auto receiptInterval =
        timingStats( builder, summary.metrics[dmCombTiming::index( dmCombTiming::Metric::receiptInterval )] );
    const auto execution =
        timingStats( builder, summary.metrics[dmCombTiming::index( dmCombTiming::Metric::execution )] );
    const auto publication =
        timingStats( builder, summary.metrics[dmCombTiming::index( dmCombTiming::Metric::publication )] );
    const auto primaryStart =
        timingStats( builder, summary.metrics[dmCombTiming::index( dmCombTiming::Metric::primaryStart )] );
    const auto primaryPublication =
        timingStats( builder, summary.metrics[dmCombTiming::index( dmCombTiming::Metric::primaryPublication )] );
    const auto &quality = summary.total;

    return CreateDmSyncoCombinerTiming( builder,
                                        summary.session,
                                        summary.sequence,
                                        summary.nowNs,
                                        summary.windowSec,
                                        summary.span,
                                        summary.age,
                                        summary.capacity,
                                        summary.queued,
                                        summary.enabled,
                                        summary.writerPid,
                                        summary.triggerInode,
                                        summary.primaryCounter,
                                        publicationInterval,
                                        receiptInterval,
                                        execution,
                                        publication,
                                        primaryStart,
                                        primaryPublication,
                                        quality.events,
                                        quality.receipts,
                                        quality.synthetic,
                                        quality.rejected,
                                        quality.duplicates,
                                        quality.counterGaps,
                                        quality.invalidMetadata,
                                        quality.missingTrace,
                                        quality.invalidTrace,
                                        quality.repeatedStart,
                                        quality.unlinkedPrimary,
                                        quality.invalidPrimary,
                                        quality.repeatedPrimary,
                                        quality.dropped );
}

inline flatbuffers::Offset<DmSyncoPrimaryTiming>
telem_dmsynco::observationTiming( flatbuffers::FlatBufferBuilder &builder, const dmTiming::Summary &summary )
{
    const auto publication =
        timingStats( builder, summary.metrics[dmTiming::metricIndex( dmTiming::Metric::publicationInterval )] );
    const auto receipt =
        timingStats( builder, summary.metrics[dmTiming::metricIndex( dmTiming::Metric::receiptInterval )] );
    const bool valid = summary.metrics[dmTiming::metricIndex( dmTiming::Metric::publicationInterval )].valid &&
                       summary.metrics[dmTiming::metricIndex( dmTiming::Metric::receiptInterval )].valid;
    const auto  pairs   = intervalPairs( builder, summary.intervalPairs );
    const auto &quality = summary.total;
    const auto  result  = CreateDmSyncoPrimaryTiming( builder,
                                                    summary.session,
                                                    summary.sequence,
                                                    summary.nowNs,
                                                    summary.windowSec,
                                                    summary.span,
                                                    summary.age,
                                                    summary.capacity,
                                                    summary.queued,
                                                    summary.enabled,
                                                    valid,
                                                    publication,
                                                    receipt,
                                                    quality.events,
                                                    quality.receipts,
                                                    quality.synthetic,
                                                    quality.rejected,
                                                    quality.duplicates,
                                                    quality.counterGaps,
                                                    quality.invalid,
                                                    quality.dropped,
                                                    pairs );
    return result;
}

inline flatbuffers::Offset<DmSyncoPipelineTiming>
telem_dmsynco::pipelineTiming( flatbuffers::FlatBufferBuilder &builder, const dmPipelineTiming::Summary &summary )
{
    std::vector<flatbuffers::Offset<DmSyncoPipelineStage>> stages;
    const size_t count = summary.stageCount < summary.stages.size() ? summary.stageCount : summary.stages.size();
    stages.reserve( count );

    for( size_t index = 0; index < count; ++index )
    {
        const auto &stage                = summary.stages[index];
        const auto  execution            = timingStats( builder, stage.execution );
        const auto  handoff              = timingStats( builder, stage.handoff );
        const auto  lateExecution        = timingStats( builder, stage.lateExecution );
        const auto  lateHandoff          = timingStats( builder, stage.lateHandoff );
        const auto  earlyExecution       = timingStats( builder, stage.earlyExecution );
        const auto  earlyHandoff         = timingStats( builder, stage.earlyHandoff );
        const auto  lateExecutionChange  = timingStats( builder, stage.lateExecutionChange );
        const auto  lateHandoffChange    = timingStats( builder, stage.lateHandoffChange );
        const auto  earlyExecutionChange = timingStats( builder, stage.earlyExecutionChange );
        const auto  earlyHandoffChange   = timingStats( builder, stage.earlyHandoffChange );
        stages.push_back( CreateDmSyncoPipelineStage( builder,
                                                      stage.rowIndex,
                                                      stage.pid,
                                                      stage.inputInode,
                                                      execution,
                                                      handoff,
                                                      lateExecution,
                                                      lateHandoff,
                                                      earlyExecution,
                                                      earlyHandoff,
                                                      lateExecutionChange,
                                                      lateHandoffChange,
                                                      earlyExecutionChange,
                                                      earlyHandoffChange ) );
    }

    const auto  rows    = builder.CreateVector( stages );
    const auto &quality = summary.total;
    const auto  total   = CreateDmSyncoPipelineQuality( builder,
                                                     quality.events,
                                                     quality.receipts,
                                                     quality.synthetic,
                                                     quality.rejected,
                                                     quality.duplicates,
                                                     quality.counterGaps,
                                                     quality.invalidMetadata,
                                                     quality.missingTrace,
                                                     quality.invalidTrace,
                                                     quality.topologyChanges,
                                                     quality.truncated,
                                                     quality.dropped );

    const auto sourceInterval         = timingStats( builder, summary.sourceInterval );
    const auto lateSourceInterval     = timingStats( builder, summary.lateSourceInterval );
    const auto earlySourceInterval    = timingStats( builder, summary.earlySourceInterval );
    const auto latePublicationChange  = timingStats( builder, summary.latePublicationChange );
    const auto earlyPublicationChange = timingStats( builder, summary.earlyPublicationChange );

    return CreateDmSyncoPipelineTiming( builder,
                                        summary.enabled,
                                        summary.session,
                                        summary.sequence,
                                        summary.nowNs,
                                        summary.windowSec,
                                        summary.span,
                                        summary.age,
                                        summary.capacity,
                                        summary.queued,
                                        summary.median,
                                        summary.threshold,
                                        summary.intervals,
                                        summary.late,
                                        summary.early,
                                        summary.stageCount,
                                        total,
                                        rows,
                                        sourceInterval,
                                        lateSourceInterval,
                                        earlySourceInterval,
                                        latePublicationChange,
                                        earlyPublicationChange );
}

inline flatbuffers::Offset<DmSyncoOverlapTrace>
telem_dmsynco::overlapTrace( flatbuffers::FlatBufferBuilder &builder, const dmSyncoSimulation::TraceBatch &trace )
{
    if( !trace.status.enabled )
    {
        return {};
    }

    std::vector<flatbuffers::Offset<DmSyncoOverlapEvent>> events;
    const size_t                                          count = std::min<size_t>( trace.count, trace.events.size() );
    events.reserve( count );

    for( size_t index = 0; index < count; ++index )
    {
        const auto &event       = trace.events[index];
        const auto &source      = event.observation;
        const auto  observation = CreateDmSyncoOverlapObservation( builder,
                                                                  source.streamInode,
                                                                  source.triggerInode,
                                                                  source.receiptNs,
                                                                  source.callbackNs,
                                                                  source.lockRequestNs,
                                                                  source.lockAcquiredNs,
                                                                  source.publicationNs,
                                                                  source.primaryPublicationNs,
                                                                  source.startNs,
                                                                  source.updateNs );
        events.push_back( CreateDmSyncoOverlapEvent( builder,
                                                     event.id,
                                                     event.session,
                                                     event.mainCounter,
                                                     event.combinedCounter,
                                                     event.planCounter,
                                                     event.lastMainCounter,
                                                     event.step,
                                                     DmSyncoOverlapOrigin( event.origin ),
                                                     DmSyncoOverlapWindow( event.window ),
                                                     event.observedNs,
                                                     event.mainGridNs,
                                                     event.planGridNs,
                                                     event.periodNs,
                                                     event.copyBeginNs,
                                                     event.copyEndNs,
                                                     event.readerEndNs,
                                                     event.lastMainNs,
                                                     event.lastOutputNs,
                                                     observation,
                                                     event.planEstimatorId,
                                                     event.lastEstimatorId,
                                                     event.periodQ32,
                                                     event.planFractionQ32 ) );
    }

    const auto  records = builder.CreateVector( events );
    const auto &quality = trace.status;

    return CreateDmSyncoOverlapTrace( builder,
                                      quality.enabled,
                                      quality.detected,
                                      quality.drained,
                                      quality.dropped,
                                      quality.queued,
                                      quality.capacity,
                                      quality.logFailed,
                                      trace.count,
                                      records );
}

inline flatbuffers::Offset<DmSyncoCadence> telem_dmsynco::cadenceTiming( flatbuffers::FlatBufferBuilder   &builder,
                                                                         const dmSyncoSimulation::Cadence &cadence )
{
    if( !cadence.periodQ32 )
    {
        return {};
    }

    return CreateDmSyncoCadence( builder,
                                 cadence.periodQ32,
                                 DmSyncoCadenceState( cadence.state ),
                                 cadence.epoch,
                                 cadence.anchors,
                                 cadence.cycles,
                                 cadence.updates,
                                 cadence.rejected,
                                 cadence.spanNs,
                                 cadence.windowNs,
                                 cadence.updateNs,
                                 cadence.observationNs,
                                 cadence.updatedNs );
}

inline flatbuffers::Offset<DmSyncoEstimatorTrace>
telem_dmsynco::estimatorTrace( flatbuffers::FlatBufferBuilder &builder, const dmSyncoSimulation::TraceBatch &trace )
{
    if( !trace.estimatorStatus.enabled )
    {
        return {};
    }

    std::vector<flatbuffers::Offset<DmSyncoEstimatorSample>> samples;
    const size_t count = std::min<size_t>( trace.estimatorCount, trace.estimator.size() );
    samples.reserve( count );

    for( size_t index = 0; index < count; ++index )
    {
        const auto &sample  = trace.estimator[index];
        const auto  cadence = cadenceTiming( builder, sample.cadence );

        samples.push_back( CreateDmSyncoEstimatorSample( builder,
                                                         sample.id,
                                                         sample.session,
                                                         sample.sequence,
                                                         sample.counter,
                                                         sample.observedNs,
                                                         sample.publicationIntervalNs,
                                                         sample.sequential,
                                                         sample.intervalAccepted,
                                                         sample.periodSamples,
                                                         sample.phaseSamples,
                                                         sample.periodBeforeNs,
                                                         sample.periodNs,
                                                         sample.predictedNs,
                                                         sample.nextCounter,
                                                         sample.sourcePlanCounter,
                                                         sample.sourceEstimatorId,
                                                         sample.sourceGridNs,
                                                         sample.baseGridNs,
                                                         sample.candidateNs,
                                                         sample.phaseGain,
                                                         sample.phaseSlewNs,
                                                         sample.rawCorrectionNs,
                                                         sample.appliedCorrectionNs,
                                                         sample.committedGridNs,
                                                         sample.committed,
                                                         sample.periodBeforeQ32,
                                                         sample.sourceFractionQ32,
                                                         sample.baseFractionQ32,
                                                         sample.committedFractionQ32,
                                                         cadence ) );
    }

    const auto  records = builder.CreateVector( samples );
    const auto &quality = trace.estimatorStatus;

    return CreateDmSyncoEstimatorTrace( builder,
                                        quality.enabled,
                                        quality.observed,
                                        quality.triggers,
                                        quality.selected,
                                        quality.drained,
                                        quality.dropped,
                                        quality.queued,
                                        quality.capacity,
                                        quality.logFailed,
                                        quality.history,
                                        quality.historyCapacity,
                                        quality.postRemaining,
                                        trace.estimatorCount,
                                        records );
}

inline flatbuffers::Offset<DmSyncoSimulation>
telem_dmsynco::simulationTiming( flatbuffers::FlatBufferBuilder      &builder,
                                 const dmSyncoSimulation::Summary    &summary,
                                 const dmSyncoSimulation::TraceBatch &trace )
{
    if( !summary.enabled )
    {
        return {};
    }

    std::vector<flatbuffers::Offset<DmSyncoSimulationStep>> steps;

    for( int index = 0; index < summary.steps && index < int( dmSyncoSimulation::maxSteps ); ++index )
    {
        const auto &step = summary.outcomes[index];
        steps.push_back( CreateDmSyncoSimulationStep( builder,
                                                      step.considered,
                                                      step.wouldRun,
                                                      step.skipped,
                                                      step.earlyMain,
                                                      step.missingMain,
                                                      step.missingOutput,
                                                      step.readerHold,
                                                      step.expired,
                                                      step.overlap ) );
    }

    const auto outcomes  = builder.CreateVector( steps );
    const auto evidence  = overlapTrace( builder, trace );
    const auto estimator = estimatorTrace( builder, trace );
    const auto cadence   = cadenceTiming( builder, summary.cadence );

    return CreateDmSyncoSimulation( builder,
                                    summary.enabled,
                                    summary.valid,
                                    summary.feasible,
                                    summary.session,
                                    summary.sequence,
                                    summary.nowNs,
                                    summary.lastMainNs,
                                    summary.counter,
                                    summary.steps,
                                    summary.periodNs,
                                    summary.gridNs,
                                    summary.phaseNs,
                                    summary.phaseMinNs,
                                    summary.phaseMaxNs,
                                    summary.phaseCount,
                                    summary.periodSamples,
                                    summary.phaseSamples,
                                    summary.nextGridNs,
                                    summary.nextActiveNs,
                                    summary.earlyReserveNs,
                                    summary.deliveryNs,
                                    summary.keepoutNs,
                                    summary.guardNs,
                                    summary.snapshotNs,
                                    summary.copyNs,
                                    summary.leadNs,
                                    summary.observationNs,
                                    summary.visibilityNs,
                                    summary.reserveNs,
                                    summary.window,
                                    summary.phaseGain,
                                    summary.phaseSlewNs,
                                    summary.duplicates,
                                    summary.gaps,
                                    summary.resets,
                                    summary.invalid,
                                    summary.excluded,
                                    summary.outputGaps,
                                    summary.associations,
                                    outcomes,
                                    summary.cadenceMinPeriodNs,
                                    evidence,
                                    estimator,
                                    cadence );
}

inline flatbuffers::Offset<DmSyncoReaderSample> telem_dmsynco::readerSample( flatbuffers::FlatBufferBuilder &builder,
                                                                             const dmReaderTiming::Sample   &sample )
{
    const auto &primary         = sample.primary;
    const auto &combined        = sample.combined;
    const auto &record          = sample.completed.record;
    const auto  primaryIdentity = CreateDmSyncoReaderIdentity(
        builder, primary.input.inode, primary.input.counter, primary.input.sec, primary.input.nsec );
    const auto primaryRecord    = CreateDmSyncoReaderPrimary( builder, primaryIdentity, primary.observedNs );
    const auto combinedIdentity = CreateDmSyncoReaderIdentity(
        builder, combined.input.inode, combined.input.counter, combined.input.sec, combined.input.nsec );
    const auto combinedRecord    = CreateDmSyncoReaderCombined( builder,
                                                             combinedIdentity,
                                                             combined.primaryInode,
                                                             combined.primaryCounter,
                                                             combined.startNs,
                                                             combined.observedNs,
                                                             combined.budgetNs );
    const auto completedIdentity = CreateDmSyncoReaderIdentity(
        builder, record.inputInode, record.inputCounter, record.publicationSec, record.publicationNsec );
    const auto completed = CreateDmSyncoReaderCompletion( builder,
                                                          record.protocolVersion,
                                                          record.session,
                                                          record.epoch,
                                                          record.sequence,
                                                          static_cast<uint64_t>( record.state ),
                                                          completedIdentity,
                                                          record.startedNs,
                                                          record.finishedNs,
                                                          static_cast<uint8_t>( record.outcome ),
                                                          record.callbackResult,
                                                          record.publisherPid,
                                                          sample.completed.observedNs );
    return CreateDmSyncoReaderSample( builder,
                                      primaryRecord,
                                      combinedRecord,
                                      completed,
                                      sample.baseNs,
                                      sample.offsetNs,
                                      sample.requiredNs,
                                      sample.observedHoldNs,
                                      sample.slackNs );
}

inline flatbuffers::Offset<DmSyncoReaderTiming> telem_dmsynco::readerTiming( flatbuffers::FlatBufferBuilder &builder,
                                                                             const dmReaderTiming::Summary  &summary )
{
    if( !summary.enabled )
    {
        return 0;
    }

    std::array<flatbuffers::Offset<DmSyncoTimingStats>, static_cast<size_t>( dmReaderTiming::Metric::count )> metrics;

    for( size_t index = 0; index < metrics.size(); ++index )
    {
        metrics[index] = timingStats( builder, summary.metrics[index] );
    }

    const auto &counts  = summary.total;
    const auto  quality = CreateDmSyncoReaderQuality( builder,
                                                     counts.events,
                                                     counts.primary,
                                                     counts.combined,
                                                     counts.completion,
                                                     counts.matched,
                                                     counts.busy,
                                                     counts.unavailable,
                                                     counts.error,
                                                     counts.unqualified,
                                                     counts.invalid,
                                                     counts.incoherent,
                                                     counts.duplicates,
                                                     counts.sequenceGaps,
                                                     counts.counterGaps,
                                                     counts.sourceResets,
                                                     counts.staleEpoch,
                                                     counts.missingPrimary,
                                                     counts.missingCombined,
                                                     counts.identityMismatch,
                                                     counts.evicted,
                                                     counts.expired,
                                                     counts.droppedPrimary,
                                                     counts.droppedCombined,
                                                     counts.droppedCompletion );
    const auto  worst =
        summary.worstValid ? readerSample( builder, summary.worst ) : flatbuffers::Offset<DmSyncoReaderSample>{};
    return CreateDmSyncoReaderTiming( builder,
                                      summary.enabled,
                                      summary.session,
                                      summary.sequence,
                                      summary.nowNs,
                                      summary.windowSec,
                                      summary.capacity,
                                      summary.queued,
                                      summary.pendingPrimary,
                                      summary.pendingCombined,
                                      summary.pendingCompletion,
                                      summary.span,
                                      summary.age,
                                      metrics[0],
                                      metrics[1],
                                      metrics[2],
                                      metrics[3],
                                      metrics[4],
                                      metrics[5],
                                      quality,
                                      summary.worstValid,
                                      worst,
                                      summary.maxRequiredHoldNs,
                                      summary.maxObservedHoldNs,
                                      summary.overBudget );
}

inline telem_dmsynco::messageT::messageT( int                                  state,
                                          int                                  steps,
                                          int64_t                              period,
                                          uint64_t                             delivered,
                                          uint64_t                             missed,
                                          uint64_t                             faults,
                                          uint64_t                             cached,
                                          int64_t                              snapshot,
                                          int64_t                              copy,
                                          const std::vector<int64_t>          &residuals,
                                          const std::vector<int64_t>          &advances,
                                          const dmTiming::Summary             &primary,
                                          bool                                 scheduleValid,
                                          const dmCombTiming::Summary         &combiner,
                                          const dmTiming::Summary             &camera,
                                          const dmPipelineTiming::Summary     &pipeline,
                                          const dmTiming::Summary             &cameraPublication,
                                          const dmSyncoSimulation::Summary    &simulation,
                                          const dmSyncoSimulation::TraceBatch &trace,
                                          const dmReaderTiming::Summary       &reader )
{
    builder.ForceDefaults( true ); // Keep nonfinite input available to verification under fast-math comparisons.

    const auto residualVector          = builder.CreateVector( residuals );
    const auto advanceVector           = builder.CreateVector( advances );
    const auto primaryTiming           = observationTiming( builder, primary );
    const auto cameraTiming            = observationTiming( builder, camera );
    const auto cameraPublicationTiming = observationTiming( builder, cameraPublication );
    const auto upstreamTiming          = pipelineTiming( builder, pipeline );
    const auto combinedTiming          = combinerTiming( builder, combiner );
    const auto simulatedTiming         = simulationTiming( builder, simulation, trace );
    const auto completedTiming         = readerTiming( builder, reader );

    builder.Finish( CreateTelem_dmsynco_fb( builder,
                                            state,
                                            steps,
                                            period,
                                            delivered,
                                            missed,
                                            faults,
                                            cached,
                                            snapshot,
                                            copy,
                                            residualVector,
                                            advanceVector,
                                            primaryTiming,
                                            scheduleValid,
                                            combinedTiming,
                                            cameraTiming,
                                            upstreamTiming,
                                            cameraPublicationTiming,
                                            simulatedTiming,
                                            completedTiming ) );
}

inline bool telem_dmsynco::validStats( const DmSyncoTimingStats *stats, bool signedValues )
{
    if( !stats ||
        ( stats->valid() && ( stats->count() < 2 || !mx::math::isFinite( stats->stddev() ) || stats->stddev() < 0 ) ) )
    {
        return false;
    }

    if( stats->count() && ( !mx::math::isFinite( stats->mean() ) || !mx::math::isFinite( stats->min() ) ||
                            !mx::math::isFinite( stats->max() ) || !mx::math::isFinite( stats->span() ) ||
                            ( !signedValues && stats->min() < 0 ) || stats->mean() < stats->min() ||
                            stats->mean() > stats->max() || stats->span() < 0 ) )
    {
        return false;
    }

    return true;
}

inline bool telem_dmsynco::validIntervalPairs( const DmSyncoIntervalPairs *pairs )
{
    if( !pairs )
    {
        return true;
    }

    if( !( validStats( pairs->after_late() ) && validStats( pairs->after_early() ) &&
           validStats( pairs->late_early_sum() ) && validStats( pairs->early_late_sum() ) &&
           mx::math::isFinite( pairs->median() ) && pairs->median() >= 0 && mx::math::isFinite( pairs->threshold() ) &&
           pairs->threshold() >= 0 ) )
    {
        return false;
    }

    const uint64_t maximumPairs = pairs->intervals() ? pairs->intervals() - 1 : 0;
    const uint64_t afterLate    = pairs->after_late()->count();
    const uint64_t afterEarly   = pairs->after_early()->count();

    if( pairs->pairs() > maximumPairs || afterLate > pairs->pairs() || afterEarly > pairs->pairs() - afterLate ||
        pairs->late_early_sum()->count() > afterLate || pairs->early_late_sum()->count() > afterEarly )
    {
        return false;
    }

    if( !pairs->enabled() )
    {
        return pairs->intervals() == 0 && pairs->pairs() == 0 && pairs->median() == 0 && pairs->threshold() == 0;
    }

    return pairs->threshold() > 0 && ( pairs->intervals() == 0 ? pairs->median() == 0 : pairs->median() > 0 );
}

inline bool telem_dmsynco::validObservation( const DmSyncoPrimaryTiming *summary )
{
    return !summary || ( validStats( summary->publication() ) && validStats( summary->receipt() ) &&
                         summary->valid() == ( summary->publication()->valid() && summary->receipt()->valid() ) &&
                         mx::math::isFinite( summary->window_sec() ) && summary->window_sec() >= 0 &&
                         mx::math::isFinite( summary->span_sec() ) && summary->span_sec() >= 0 &&
                         mx::math::isFinite( summary->age_sec() ) && summary->age_sec() >= -1 &&
                         validIntervalPairs( summary->interval_pairs() ) );
}

inline bool telem_dmsynco::validPipeline( const DmSyncoPipelineTiming *summary )
{
    if( !summary )
    {
        return true;
    }

    if( !summary->stages() || !summary->total() || summary->stage_count() > dmPipelineTiming::maxStages ||
        summary->stage_count() != summary->stages()->size() || summary->queued() > summary->capacity() ||
        !mx::math::isFinite( summary->window_sec() ) || summary->window_sec() < 0 ||
        !mx::math::isFinite( summary->span_sec() ) || summary->span_sec() < 0 ||
        !mx::math::isFinite( summary->age_sec() ) || summary->age_sec() < -1 ||
        !mx::math::isFinite( summary->median() ) || summary->median() < 0 ||
        !mx::math::isFinite( summary->threshold() ) || summary->threshold() < 0 ||
        summary->intervals() > ( summary->capacity() ? summary->capacity() - 1 : 0 ) ||
        summary->late() > summary->intervals() || summary->early() > summary->intervals() - summary->late() ||
        ( summary->threshold() == 0 && ( summary->late() || summary->early() ) ) ||
        ( summary->intervals() ? summary->median() <= 0 : summary->median() != 0 ) )
    {
        return false;
    }

    if( !summary->enabled() && ( summary->stage_count() || summary->intervals() || summary->threshold() != 0 ) )
    {
        return false;
    }

    const DmSyncoTimingStats *boundaries[]     = { summary->source_interval(),
                                                   summary->late_source_interval(),
                                                   summary->early_source_interval(),
                                                   summary->late_publication_change(),
                                                   summary->early_publication_change() };
    const uint64_t            boundaryBounds[] = {
        summary->intervals(), summary->late(), summary->early(), summary->late(), summary->early() };

    for( size_t index = 0; index < 5; ++index )
    {
        const auto *stats = boundaries[index];

        if( !validStats( stats, index >= 3 ) || stats->count() > boundaryBounds[index] ||
            !mx::math::isFinite( stats->stddev() ) || stats->stddev() < 0 )
        {
            return false;
        }
    }

    for( size_t index = 0; index < summary->stages()->size(); ++index )
    {
        const auto *stage = summary->stages()->Get( index );

        if( !stage || stage->row_index() != index || stage->pid() <= 0 ||
            ( index + 1 < summary->stage_count() && !stage->input_inode() ) )
        {
            return false;
        }

        const DmSyncoTimingStats *populations[] = { stage->execution(),
                                                    stage->handoff(),
                                                    stage->late_execution(),
                                                    stage->late_handoff(),
                                                    stage->early_execution(),
                                                    stage->early_handoff(),
                                                    stage->late_execution_change(),
                                                    stage->late_handoff_change(),
                                                    stage->early_execution_change(),
                                                    stage->early_handoff_change() };
        const uint64_t            bounds[]      = { summary->capacity(),
                                                    summary->capacity(),
                                                    summary->late(),
                                                    summary->late(),
                                                    summary->early(),
                                                    summary->early(),
                                                    stage->late_execution()->count(),
                                                    stage->late_handoff()->count(),
                                                    stage->early_execution()->count(),
                                                    stage->early_handoff()->count() };

        for( size_t population = 0; population < 10; ++population )
        {
            const auto *stats = populations[population];

            if( !validStats( stats, population >= 6 ) || stats->count() > bounds[population] ||
                !mx::math::isFinite( stats->stddev() ) || stats->stddev() < 0 )
            {
                return false;
            }
        }
    }

    return true;
}

inline bool telem_dmsynco::validReaderSample( const DmSyncoReaderSample *sample )
{
    if( !sample || !sample->primary() || !sample->combined() || !sample->completed() )
    {
        return false;
    }

    const auto *primary   = sample->primary();
    const auto *combined  = sample->combined();
    const auto *completed = sample->completed();
    const auto *source    = primary->input();
    const auto *input     = combined->input();
    const auto *command   = completed->input();

    if( !source || !input || !command || !source->inode() || !source->counter() || source->publication_sec() <= 0 ||
        source->publication_nsec() < 0 || source->publication_nsec() >= 1000000000 || !input->inode() ||
        !input->counter() || input->publication_sec() <= 0 || input->publication_nsec() < 0 ||
        input->publication_nsec() >= 1000000000 || primary->observed_ns() <= 0 || combined->observed_ns() <= 0 ||
        combined->budget_ns() < 0 || combined->primary_inode() != source->inode() ||
        combined->primary_counter() != source->counter() || command->inode() != input->inode() ||
        command->counter() != input->counter() || command->publication_sec() != input->publication_sec() ||
        command->publication_nsec() != input->publication_nsec() ||
        completed->observed_ns() < completed->finished_ns() )
    {
        return false;
    }

    MagAOX::dmCompletion::Record record;
    record.protocolVersion = completed->protocol_version();
    record.session         = completed->session();
    record.epoch           = completed->epoch();
    record.sequence        = completed->sequence();
    record.state           = static_cast<MagAOX::dmCompletion::State>( completed->state() );
    record.inputInode      = command->inode();
    record.inputCounter    = command->counter();
    record.publicationSec  = command->publication_sec();
    record.publicationNsec = command->publication_nsec();
    record.startedNs       = completed->started_ns();
    record.finishedNs      = completed->finished_ns();
    record.outcome         = static_cast<MagAOX::dmTiming::Outcome>( completed->outcome() );
    record.callbackResult  = completed->callback_result();
    record.publisherPid    = completed->publisher_pid();

    if( !MagAOX::dmCompletion::qualified( record ) )
    {
        return false;
    }

    const __int128 sourcePublication = __int128( source->publication_sec() ) * 1000000000 + source->publication_nsec();
    const __int128 inputPublication  = __int128( input->publication_sec() ) * 1000000000 + input->publication_nsec();
    const int64_t  base              = std::max( primary->observed_ns(), combined->observed_ns() );
    const __int128 offset            = __int128( completed->finished_ns() ) - base;
    const __int128 observed          = __int128( completed->observed_ns() ) - base;
    const __int128 slack             = __int128( combined->budget_ns() ) - offset;
    return sourcePublication <= combined->start_ns() && combined->start_ns() <= inputPublication &&
           inputPublication <= std::numeric_limits<int64_t>::max() && sample->base_ns() == base &&
           sample->offset_ns() == offset && sample->required_ns() == std::max<__int128>( 0, offset ) &&
           sample->observed_hold_ns() == std::max<__int128>( 0, observed ) && sample->slack_ns() == slack;
}

inline bool telem_dmsynco::validReader( const DmSyncoReaderTiming *summary )
{
    if( !summary )
    {
        return true;
    }

    const auto *quality = summary->total();

    if( !quality || summary->now_ns() < 0 || !mx::math::isFinite( summary->window_sec() ) ||
        summary->window_sec() < 0 || !mx::math::isFinite( summary->span_sec() ) || summary->span_sec() < 0 ||
        summary->span_sec() > summary->window_sec() + 1e-9 || !mx::math::isFinite( summary->age_sec() ) ||
        ( summary->age_sec() < 0 && summary->age_sec() != -1 ) ||
        __uint128_t( summary->queued() ) > __uint128_t( summary->capacity() ) * 3 ||
        summary->pending_primary() > summary->capacity() || summary->pending_combined() > summary->capacity() ||
        summary->pending_completion() > summary->capacity() || summary->max_required_hold_ns() < 0 ||
        summary->max_observed_hold_ns() < summary->max_required_hold_ns() ||
        summary->over_budget() > quality->matched() ||
        __uint128_t( quality->primary() ) + quality->combined() + quality->completion() != quality->events() ||
        quality->matched() > quality->combined() || quality->matched() > quality->completion() ||
        __uint128_t( quality->busy() ) + quality->unavailable() + quality->error() + quality->unqualified() >
            quality->completion() ||
        summary->worst_valid() != ( quality->matched() > 0 ) ||
        ( summary->enabled() && ( !summary->session() || !summary->capacity() || summary->window_sec() <= 0 ) ) )
    {
        return false;
    }

    const DmSyncoTimingStats *populations[] = { summary->completion_offset(),
                                                summary->required_hold(),
                                                summary->observed_hold(),
                                                summary->processing(),
                                                summary->notification(),
                                                summary->model_slack() };
    uint64_t                  maximumCount  = 0;

    for( size_t index = 0; index < static_cast<size_t>( dmReaderTiming::Metric::count ); ++index )
    {
        const auto *stats = populations[index];

        if( !validStats( stats,
                         index == dmReaderTiming::index( dmReaderTiming::Metric::completionOffset ) ||
                             index == dmReaderTiming::index( dmReaderTiming::Metric::modelSlack ) ) ||
            !mx::math::isFinite( stats->mean() ) || !mx::math::isFinite( stats->min() ) ||
            !mx::math::isFinite( stats->max() ) || !mx::math::isFinite( stats->span() ) ||
            !mx::math::isFinite( stats->stddev() ) || stats->stddev() < 0 || stats->count() > summary->capacity() ||
            stats->count() > quality->matched() || stats->valid() != ( stats->count() > 1 ) ||
            ( stats->count() < 2 && stats->stddev() != 0 ) || stats->span() > summary->span_sec() + 1e-9 ||
            ( stats->count() == 1 && stats->span() != 0 ) ||
            ( !stats->count() &&
              ( stats->mean() != 0 || stats->min() != 0 || stats->max() != 0 || stats->span() != 0 ) ) )
        {
            return false;
        }

        maximumCount = std::max( maximumCount, stats->count() );
    }

    if( ( !maximumCount && ( summary->age_sec() != -1 || summary->span_sec() != 0 ) ) ||
        ( maximumCount && summary->age_sec() < 0 ) ||
        populations[0]->max() > summary->max_required_hold_ns() * 1e-9 + 1e-15 ||
        populations[1]->max() > summary->max_required_hold_ns() * 1e-9 + 1e-15 ||
        populations[2]->max() > summary->max_observed_hold_ns() * 1e-9 + 1e-15 ||
        ( !summary->enabled() && ( maximumCount || summary->capacity() || summary->queued() ||
                                   summary->pending_primary() || summary->pending_combined() ||
                                   summary->pending_completion() || quality->events() || quality->matched() ) ) )
    {
        return false;
    }

    if( !summary->worst_valid() )
    {
        return !summary->worst() && !summary->max_required_hold_ns() && !summary->max_observed_hold_ns() &&
               !summary->over_budget();
    }

    return validReaderSample( summary->worst() ) &&
           summary->worst()->required_ns() == summary->max_required_hold_ns() &&
           summary->worst()->observed_hold_ns() <= summary->max_observed_hold_ns() &&
           ( summary->worst()->slack_ns() >= 0 || summary->over_budget() > 0 ) &&
           ( summary->max_required_hold_ns() > 0 || !summary->over_budget() );
}

inline bool telem_dmsynco::validPeriod( uint64_t periodQ32, int64_t periodNs )
{
    return !periodQ32 || ( periodQ32 <= uint64_t( 1000000000 ) * dmSyncoSimulation::periodScale &&
                           periodNs == int64_t( ( periodQ32 + dmSyncoSimulation::periodScale / 2 ) /
                                                dmSyncoSimulation::periodScale ) &&
                           periodNs > 0 );
}

inline bool telem_dmsynco::validFraction( int64_t fractionQ32 )
{
    const int64_t half = int64_t( dmSyncoSimulation::periodScale / 2 );

    return fractionQ32 >= -half && fractionQ32 < half;
}

inline bool telem_dmsynco::validCadence( const DmSyncoCadence *cadence )
{
    if( !cadence )
    {
        return true;
    }

    if( !cadence->period_q32() ||
        !validPeriod( cadence->period_q32(),
                      int64_t( ( cadence->period_q32() + dmSyncoSimulation::periodScale / 2 ) /
                               dmSyncoSimulation::periodScale ) ) ||
        !cadence->epoch() || cadence->anchors() > dmSyncoSimulation::cadenceAnchorCapacity ||
        cadence->window_ns() <= 0 || cadence->update_ns() <= 0 || cadence->update_ns() > cadence->window_ns() ||
        cadence->observation_ns() < 0 || cadence->updated_ns() < 0 ||
        cadence->updated_ns() > cadence->observation_ns() || cadence->span_ns() < 0 )
    {
        return false;
    }

    if( cadence->state() == DmSyncoCadenceState_nominal )
    {
        return !cadence->span_ns() && !cadence->cycles() && !cadence->updates() && !cadence->updated_ns();
    }

    return ( cadence->state() == DmSyncoCadenceState_growing || cadence->state() == DmSyncoCadenceState_full ) &&
           cadence->anchors() >= 2 && cadence->cycles() && cadence->updates() && cadence->updated_ns() > 0 &&
           cadence->span_ns() >= cadence->update_ns() && cadence->span_ns() <= cadence->updated_ns() &&
           ( __int128( cadence->span_ns() ) * dmSyncoSimulation::periodScale + cadence->cycles() / 2 ) /
                   cadence->cycles() ==
               cadence->period_q32() &&
           ( cadence->state() == DmSyncoCadenceState_full ? cadence->span_ns() >= cadence->window_ns()
                                                          : cadence->span_ns() < cadence->window_ns() );
}

inline bool telem_dmsynco::validOverlapTrace( const DmSyncoOverlapTrace *trace )
{
    if( !trace )
    {
        return true;
    }

    if( !trace->enabled() || !trace->events() || trace->capacity() == 0 ||
        trace->capacity() > dmSyncoSimulation::traceCapacity || trace->queued() > trace->capacity() ||
        trace->drained() > trace->detected() || trace->dropped() > trace->detected() - trace->drained() ||
        trace->queued() != trace->detected() - trace->drained() - trace->dropped() ||
        trace->log_failed() > trace->drained() || trace->batch_count() != trace->events()->size() ||
        trace->batch_count() > dmSyncoSimulation::traceBatchCapacity || trace->batch_count() > trace->drained() )
    {
        return false;
    }

    uint64_t previousId = 0;

    for( const auto *event : *trace->events() )
    {
        if( !event || event->id() <= previousId || event->id() > trace->detected() || !event->session() ||
            event->plan_estimator_id() > event->last_estimator_id() || event->step() < 0 ||
            event->step() >= int( dmSyncoSimulation::maxSteps ) ||
            ( event->origin() != DmSyncoOverlapOrigin_primary && event->origin() != DmSyncoOverlapOrigin_combined ) ||
            ( event->window() != DmSyncoOverlapWindow_copy && event->window() != DmSyncoOverlapWindow_readerHold ) ||
            event->observed_ns() <= 0 || event->main_grid_ns() < 0 || event->plan_grid_ns() <= 0 ||
            event->period_ns() <= 0 || event->copy_begin_ns() <= 0 || event->copy_end_ns() <= event->copy_begin_ns() ||
            event->last_main_ns() < 0 || event->last_main_ns() > event->observed_ns() || event->last_output_ns() < 0 ||
            event->last_output_ns() > event->observed_ns() || !validPeriod( event->period_q32(), event->period_ns() ) ||
            !validFraction( event->plan_fraction_q32() ) || ( !event->period_q32() && event->plan_fraction_q32() ) ||
            ( event->step() == 0 ? event->reader_end_ns() != 0 : event->reader_end_ns() < event->copy_end_ns() ) )
        {
            return false;
        }

        previousId = event->id();

        if( event->window() == DmSyncoOverlapWindow_copy )
        {
            if( event->observed_ns() < event->copy_begin_ns() || event->observed_ns() >= event->copy_end_ns() )
            {
                return false;
            }
        }
        else if( event->step() == 0 || event->reader_end_ns() <= event->copy_end_ns() ||
                 event->observed_ns() < event->copy_end_ns() || event->observed_ns() >= event->reader_end_ns() )
        {
            return false;
        }

        const auto *observation = event->observation();

        if( !observation || observation->publication_ns() < 0 || observation->primary_publication_ns() < 0 ||
            observation->start_ns() < 0 || observation->update_ns() < 0 )
        {
            return false;
        }

        int64_t previousTime = 0;

        for( int64_t stamp : { observation->receipt_ns(),
                               observation->callback_ns(),
                               observation->lock_request_ns(),
                               observation->lock_acquired_ns(),
                               event->observed_ns() } )
        {
            if( stamp < 0 || ( stamp && stamp < previousTime ) )
            {
                return false;
            }

            if( stamp )
            {
                previousTime = stamp;
            }
        }
    }

    return true;
}

inline bool telem_dmsynco::validEstimatorTrace( const DmSyncoEstimatorTrace *trace )
{
    if( !trace )
    {
        return true;
    }

    if( !trace->enabled() || !trace->samples() || !trace->capacity() ||
        trace->capacity() > dmSyncoSimulation::estimatorTraceCapacity || trace->queued() > trace->capacity() ||
        trace->selected() > trace->observed() || trace->drained() > trace->selected() ||
        trace->dropped() > trace->selected() - trace->drained() ||
        trace->queued() != trace->selected() - trace->drained() - trace->dropped() ||
        trace->log_failed() > trace->drained() || !trace->history_capacity() ||
        trace->history_capacity() > dmSyncoSimulation::estimatorHistoryCapacity ||
        trace->history() > trace->history_capacity() || trace->history() > trace->observed() ||
        trace->post_remaining() > dmSyncoSimulation::estimatorPostSamples ||
        trace->batch_count() != trace->samples()->size() ||
        trace->batch_count() > dmSyncoSimulation::estimatorBatchCapacity || trace->batch_count() > trace->drained() )
    {
        return false;
    }

    uint64_t previousId = 0;

    for( const auto *sample : *trace->samples() )
    {
        if( !sample || sample->id() <= previousId || sample->id() > trace->observed() || !sample->session() ||
            !sample->sequence() || sample->sequence() > sample->id() || sample->observed_ns() <= 0 ||
            sample->period_samples() > dmSyncoSimulation::estimatorHistoryCapacity ||
            sample->phase_samples() > dmSyncoSimulation::estimatorHistoryCapacity || !sample->phase_samples() ||
            sample->period_samples() > sample->sequence() || sample->phase_samples() > sample->sequence() ||
            sample->period_before_ns() <= 0 || sample->period_ns() <= 0 || sample->predicted_ns() < 0 ||
            sample->source_grid_ns() < 0 || sample->base_grid_ns() < 0 || sample->candidate_ns() < 0 ||
            sample->committed_grid_ns() < 0 || !mx::math::isFinite( sample->phase_gain() ) ||
            sample->phase_gain() < 0 || sample->phase_gain() > 1 || sample->phase_slew_ns() < 0 ||
            ( sample->sequential() && sample->sequence() < 2 ) ||
            ( sample->interval_accepted() &&
              ( !sample->sequential() || sample->publication_interval_ns() <= 0 || !sample->period_samples() ) ) ||
            !validPeriod( sample->period_before_q32(), sample->period_before_ns() ) ||
            !validCadence( sample->cadence() ) || !validFraction( sample->source_fraction_q32() ) ||
            !validFraction( sample->base_fraction_q32() ) || !validFraction( sample->committed_fraction_q32() ) ||
            ( sample->cadence() ? !sample->period_before_q32() ||
                                      !validPeriod( sample->cadence()->period_q32(), sample->period_ns() ) ||
                                      sample->cadence()->observation_ns() != sample->observed_ns()
                                : sample->period_before_q32() || sample->source_fraction_q32() ||
                                      sample->base_fraction_q32() || sample->committed_fraction_q32() ) )
        {
            return false;
        }

        previousId = sample->id();

        if( !sample->next_counter() )
        {
            if( sample->source_plan_counter() || sample->source_estimator_id() || sample->source_grid_ns() ||
                sample->base_grid_ns() || sample->candidate_ns() || sample->raw_correction_ns() ||
                sample->applied_correction_ns() || sample->committed_grid_ns() || sample->committed() ||
                sample->source_fraction_q32() || sample->base_fraction_q32() || sample->committed_fraction_q32() )
            {
                return false;
            }

            continue;
        }

        if( sample->counter() == std::numeric_limits<uint64_t>::max() ||
            sample->next_counter() != sample->counter() + 1 ||
            sample->source_plan_counter() >= sample->next_counter() || !sample->source_grid_ns() ||
            !sample->source_estimator_id() || sample->source_estimator_id() > sample->id() )
        {
            return false;
        }

        const __int128 scale   = dmSyncoSimulation::periodScale;
        const __int128 baseQ32 = __int128( sample->source_grid_ns() ) * scale + sample->source_fraction_q32() +
                                 ( __int128( sample->next_counter() ) - sample->source_plan_counter() ) *
                                     ( sample->cadence() ? __int128( sample->cadence()->period_q32() )
                                                         : __int128( sample->period_ns() ) * scale );
        const __int128 base             = ( baseQ32 + scale / 2 ) / scale;
        const int64_t  expectedBase     = base > 0 && base <= std::numeric_limits<int64_t>::max() ? int64_t( base ) : 0;
        const int64_t  expectedFraction = expectedBase && sample->cadence() ? int64_t( baseQ32 - base * scale ) : 0;
        const int64_t  raw              = sample->base_grid_ns() && sample->candidate_ns()
                                              ? int64_t( sample->phase_gain() * ( static_cast<long double>( sample->candidate_ns() ) -
                                                                    sample->base_grid_ns() ) )
                                              : 0;
        const int64_t  applied          = std::clamp( raw, -sample->phase_slew_ns(), sample->phase_slew_ns() );
        const __int128 committed        = __int128( sample->base_grid_ns() ) + applied;

        if( sample->base_grid_ns() != expectedBase || sample->base_fraction_q32() != expectedFraction ||
            sample->raw_correction_ns() != raw || sample->applied_correction_ns() != applied ||
            ( sample->committed()
                  ? !sample->base_grid_ns() || committed <= 0 || committed > std::numeric_limits<int64_t>::max() ||
                        sample->committed_grid_ns() != int64_t( committed ) ||
                        sample->committed_fraction_q32() != sample->base_fraction_q32()
                  : sample->committed_grid_ns() != 0 || sample->committed_fraction_q32() != 0 ) )
        {
            return false;
        }
    }

    return true;
}

inline bool telem_dmsynco::validSimulation( const DmSyncoSimulation *summary )
{
    if( !summary )
    {
        return true;
    }

    if( !summary->outcomes() || summary->steps() < 0 || summary->steps() > int( dmSyncoSimulation::maxSteps ) ||
        summary->outcomes()->size() != size_t( summary->steps() ) ||
        ( summary->enabled() && ( summary->steps() == 0 || summary->session() == 0 ) ) ||
        ( !summary->enabled() && ( summary->valid() || summary->feasible() ) ) || summary->now_ns() < 0 ||
        summary->last_main_ns() < 0 || summary->last_main_ns() > summary->now_ns() || summary->period_ns() < 0 ||
        summary->grid_ns() < 0 || summary->next_grid_ns() < 0 || summary->next_active_ns() < 0 ||
        summary->early_reserve_ns() < 0 || summary->delivery_ns() < 0 || summary->keepout_ns() < 0 ||
        summary->guard_ns() < 0 || summary->snapshot_ns() < 0 || summary->copy_ns() < 0 || summary->lead_ns() < 0 ||
        summary->observation_ns() < 0 || summary->visibility_ns() < 0 || summary->reserve_ns() < 0 ||
        !validOverlapTrace( summary->overlap_trace() ) || !validEstimatorTrace( summary->estimator_trace() ) ||
        !validCadence( summary->cadence() ) ||
        ( summary->cadence() && ( !validPeriod( summary->cadence()->period_q32(), summary->period_ns() ) ||
                                  summary->cadence()->observation_ns() > summary->now_ns() ) ) ||
        !mx::math::isFinite( summary->phase_gain() ) || summary->phase_gain() < 0 || summary->phase_gain() > 1 ||
        summary->phase_slew_ns() < 0 || summary->cadence_min_period_ns() < 0 ||
        summary->phase_count() > summary->sequence() || summary->period_samples() > summary->sequence() ||
        summary->phase_samples() > summary->sequence() || summary->period_samples() > summary->window() ||
        summary->phase_samples() > summary->window() ||
        ( summary->valid() && ( !summary->enabled() || summary->period_ns() == 0 || summary->grid_ns() == 0 ||
                                summary->last_main_ns() == 0 || summary->window() == 0 ) ) )
    {
        return false;
    }

    if( summary->phase_count() == 0 )
    {
        if( summary->phase_ns() != 0 || summary->phase_min_ns() != 0 || summary->phase_max_ns() != 0 )
        {
            return false;
        }
    }
    else if( summary->phase_min_ns() > summary->phase_ns() || summary->phase_ns() > summary->phase_max_ns() ||
             ( summary->phase_count() == 1 &&
               ( summary->phase_min_ns() != summary->phase_ns() || summary->phase_max_ns() != summary->phase_ns() ) ) )
    {
        return false;
    }

    if( summary->overlap_trace() && summary->estimator_trace() )
    {
        for( const auto *event : *summary->overlap_trace()->events() )
        {
            if( event->last_estimator_id() > summary->estimator_trace()->observed() )
            {
                return false;
            }
        }
    }

    for( const auto *step : *summary->outcomes() )
    {
        if( !step || step->would_run() > step->considered() ||
            step->skipped() != step->considered() - step->would_run() )
        {
            return false;
        }

        uint64_t unclassified = step->skipped();

        for( uint64_t count : { step->early_main(),
                                step->missing_main(),
                                step->missing_output(),
                                step->reader_hold(),
                                step->expired() } )
        {
            if( count > unclassified )
            {
                return false;
            }

            unclassified -= count;
        }

        if( unclassified != 0 )
        {
            return false;
        }
    }

    return true;
}

inline bool telem_dmsynco::verify( flatlogs::bufferPtrT &buffer, flatlogs::msgLenT len )
{
    if( !buffer )
    {
        return false;
    }

    flatbuffers::Verifier verifier( static_cast<uint8_t *>( flatlogs::logHeader::messageBuffer( buffer ) ), len );

    if( !VerifyTelem_dmsynco_fbBuffer( verifier ) )
    {
        return false;
    }

    const auto *record = GetTelem_dmsynco_fb( flatlogs::logHeader::messageBuffer( buffer ) );

    if( !validObservation( record->primary_timing() ) || !validObservation( record->camera_timing() ) ||
        !validPipeline( record->pipeline_timing() ) || !validObservation( record->camera_publication_timing() ) ||
        !validSimulation( record->simulation() ) || !validReader( record->reader_timing() ) )
    {
        return false;
    }

    const auto *combiner = record->combiner_timing();

    return !combiner ||
           ( validStats( combiner->publication_interval() ) && validStats( combiner->receipt_interval() ) &&
             validStats( combiner->execution() ) && validStats( combiner->publication() ) &&
             validStats( combiner->primary_start() ) && validStats( combiner->primary_publication() ) &&
             mx::math::isFinite( combiner->window_sec() ) && combiner->window_sec() >= 0 &&
             mx::math::isFinite( combiner->span_sec() ) && combiner->span_sec() >= 0 &&
             mx::math::isFinite( combiner->age_sec() ) && combiner->age_sec() >= -1 );
}

inline void telem_dmsynco::formatStats( std::ostream &output, const char *name, const DmSyncoTimingStats *stats )
{
    output << ' ' << name << "[n=" << stats->count() << " valid=" << stats->valid() << ']';

    if( !stats->count() )
    {
        output << "=unavailable";
        return;
    }

    output << " mean_s=" << stats->mean() << " stddev_s=";

    if( stats->valid() )
    {
        output << stats->stddev();
    }
    else
    {
        output << "unavailable";
    }

    output << " min_s=" << stats->min() << " max_s=" << stats->max() << " span_s=" << stats->span();
}

inline void telem_dmsynco::formatObservation( std::ostream               &output,
                                              const char                 *prefix,
                                              const char                 *intervalName,
                                              const DmSyncoPrimaryTiming *summary )
{
    const std::string name = prefix;

    if( summary )
    {
        output << " " << name << "_enabled=" << summary->enabled() << " " << name << "_valid=" << summary->valid()
               << " " << name << "_session=" << summary->session() << " " << name << "_sequence=" << summary->sequence()
               << " " << name << "_now_ns=" << summary->now_ns() << " " << name << "_window_s=" << summary->window_sec()
               << " " << name << "_span_s=" << summary->span_sec() << " " << name << "_age_s=" << summary->age_sec()
               << " " << name << "_capacity=" << summary->capacity() << " " << name << "_queued=" << summary->queued()
               << " " << name << "_quality={events:" << summary->events() << ",receipts:" << summary->receipts()
               << ",synthetic:" << summary->synthetic() << ",rejected:" << summary->rejected()
               << ",duplicates:" << summary->duplicates() << ",counterGaps:" << summary->counter_gaps()
               << ",invalid:" << summary->invalid() << ",dropped:" << summary->dropped() << '}';

        formatStats( output, intervalName, summary->publication() );
        formatStats( output, ( name + "ReceiptInterval" ).c_str(), summary->receipt() );
    }
    else
    {
        output << " " << name << "_timing=unavailable";
    }

    const auto *pairs = summary ? summary->interval_pairs() : nullptr;

    if( pairs )
    {
        output << " " << name << "_pairs={enabled:" << pairs->enabled() << ",intervals:" << pairs->intervals()
               << ",pairs:" << pairs->pairs() << ",median_s:" << pairs->median()
               << ",threshold_s:" << pairs->threshold() << ",lateEarly:" << pairs->late_early_sum()->count() << '/'
               << pairs->after_late()->count() << ",earlyLate:" << pairs->early_late_sum()->count() << '/'
               << pairs->after_early()->count() << '}';

        formatStats( output, ( name + "AfterLateInterval" ).c_str(), pairs->after_late() );
        formatStats( output, ( name + "AfterEarlyInterval" ).c_str(), pairs->after_early() );
        formatStats( output, ( name + "LateEarlySum" ).c_str(), pairs->late_early_sum() );
        formatStats( output, ( name + "EarlyLateSum" ).c_str(), pairs->early_late_sum() );
    }
    else
    {
        output << " " << name << "_pairs=unavailable";
    }
}

inline void telem_dmsynco::formatPipeline( std::ostream &output, const DmSyncoPipelineTiming *summary )
{
    if( !summary )
    {
        output << " pipeline_timing=unavailable";
        return;
    }

    const auto *quality = summary->total();
    output << " pipeline_enabled=" << summary->enabled() << " pipeline_session=" << summary->session()
           << " pipeline_sequence=" << summary->sequence() << " pipeline_now_ns=" << summary->now_ns()
           << " pipeline_window_s=" << summary->window_sec() << " pipeline_span_s=" << summary->span_sec()
           << " pipeline_age_s=" << summary->age_sec() << " pipeline_capacity=" << summary->capacity()
           << " pipeline_queued=" << summary->queued() << " pipeline_median_s=" << summary->median()
           << " pipeline_threshold_s=" << summary->threshold() << " pipeline_intervals=" << summary->intervals()
           << " pipeline_late=" << summary->late() << " pipeline_early=" << summary->early()
           << " pipeline_stages=" << summary->stage_count() << " pipeline_quality={events:" << quality->events()
           << ",receipts:" << quality->receipts() << ",synthetic:" << quality->synthetic()
           << ",rejected:" << quality->rejected() << ",duplicates:" << quality->duplicates()
           << ",counterGaps:" << quality->counter_gaps() << ",invalidMetadata:" << quality->invalid_metadata()
           << ",missingTrace:" << quality->missing_trace() << ",invalidTrace:" << quality->invalid_trace()
           << ",topologyChanges:" << quality->topology_changes() << ",truncated:" << quality->truncated()
           << ",dropped:" << quality->dropped() << '}';

    formatStats( output, "pipelineSourceInterval", summary->source_interval() );
    formatStats( output, "pipelineLateSourceInterval", summary->late_source_interval() );
    formatStats( output, "pipelineEarlySourceInterval", summary->early_source_interval() );
    formatStats( output, "pipelineLatePublicationChange", summary->late_publication_change() );
    formatStats( output, "pipelineEarlyPublicationChange", summary->early_publication_change() );

    for( const auto *stage : *summary->stages() )
    {
        output << " pipeline_stage={row:" << stage->row_index() << ",pid:" << stage->pid()
               << ",inputInode:" << stage->input_inode() << '}';
        formatStats( output, "stageExecution", stage->execution() );
        formatStats( output, "stageHandoff", stage->handoff() );
        formatStats( output, "stageLateExecution", stage->late_execution() );
        formatStats( output, "stageLateHandoff", stage->late_handoff() );
        formatStats( output, "stageEarlyExecution", stage->early_execution() );
        formatStats( output, "stageEarlyHandoff", stage->early_handoff() );
        formatStats( output, "stageLateExecutionChange", stage->late_execution_change() );
        formatStats( output, "stageLateHandoffChange", stage->late_handoff_change() );
        formatStats( output, "stageEarlyExecutionChange", stage->early_execution_change() );
        formatStats( output, "stageEarlyHandoffChange", stage->early_handoff_change() );
    }
}

inline void telem_dmsynco::formatReader( std::ostream &output, const DmSyncoReaderTiming *summary )
{
    if( !summary )
    {
        output << " reader_timing=unavailable";
        return;
    }

    const auto *quality = summary->total();
    output << " reader_enabled=" << summary->enabled() << " reader_mode=shadow_observation"
           << " reader_session=" << summary->session() << " reader_sequence=" << summary->sequence()
           << " reader_now_ns=" << summary->now_ns() << " reader_window_s=" << summary->window_sec()
           << " reader_capacity=" << summary->capacity() << " reader_queued=" << summary->queued()
           << " reader_pending={primary:" << summary->pending_primary() << ",combined:" << summary->pending_combined()
           << ",completion:" << summary->pending_completion() << "} reader_span_s=" << summary->span_sec()
           << " reader_age_s=" << summary->age_sec()
           << " reader_max_required_hold_ns=" << summary->max_required_hold_ns()
           << " reader_max_observed_hold_ns=" << summary->max_observed_hold_ns()
           << " reader_over_budget=" << summary->over_budget() << " reader_quality={events:" << quality->events()
           << ",primary:" << quality->primary() << ",combined:" << quality->combined()
           << ",completion:" << quality->completion() << ",matched:" << quality->matched()
           << ",busy:" << quality->busy() << ",unavailable:" << quality->unavailable() << ",error:" << quality->error()
           << ",unqualified:" << quality->unqualified() << ",invalid:" << quality->invalid()
           << ",incoherent:" << quality->incoherent() << ",duplicates:" << quality->duplicates()
           << ",sequenceGaps:" << quality->sequence_gaps() << ",counterGaps:" << quality->counter_gaps()
           << ",sourceResets:" << quality->source_resets() << ",staleEpoch:" << quality->stale_epoch()
           << ",missingPrimary:" << quality->missing_primary() << ",missingCombined:" << quality->missing_combined()
           << ",identityMismatch:" << quality->identity_mismatch() << ",evicted:" << quality->evicted()
           << ",expired:" << quality->expired() << ",droppedPrimary:" << quality->dropped_primary()
           << ",droppedCombined:" << quality->dropped_combined()
           << ",droppedCompletion:" << quality->dropped_completion() << '}';
    formatStats( output, "readerCompletionOffset", summary->completion_offset() );
    formatStats( output, "readerRequiredHold", summary->required_hold() );
    formatStats( output, "readerObservedHold", summary->observed_hold() );
    formatStats( output, "readerProcessing", summary->processing() );
    formatStats( output, "readerNotification", summary->notification() );
    formatStats( output, "readerModelSlack", summary->model_slack() );
    output << " reader_worst_valid=" << summary->worst_valid();

    if( !summary->worst_valid() )
    {
        return;
    }

    const auto *sample    = summary->worst();
    const auto *primary   = sample->primary();
    const auto *combined  = sample->combined();
    const auto *completed = sample->completed();
    const auto *source    = primary->input();
    const auto *input     = combined->input();
    const auto *command   = completed->input();
    output << " reader_worst={clock:monotonic,base_ns:" << sample->base_ns() << ",offset_ns:" << sample->offset_ns()
           << ",required_ns:" << sample->required_ns() << ",observed_hold_ns:" << sample->observed_hold_ns()
           << ",slack_ns:" << sample->slack_ns() << ",primary:{inode:" << source->inode()
           << ",counter:" << source->counter() << ",publication_sec:" << source->publication_sec()
           << ",publication_nsec:" << source->publication_nsec() << ",observed_ns:" << primary->observed_ns()
           << "},combined:{inode:" << input->inode() << ",counter:" << input->counter()
           << ",publication_sec:" << input->publication_sec() << ",publication_nsec:" << input->publication_nsec()
           << ",primary_inode:" << combined->primary_inode() << ",primary_counter:" << combined->primary_counter()
           << ",start_external_ns:" << combined->start_ns() << ",observed_ns:" << combined->observed_ns()
           << ",budget_ns:" << combined->budget_ns()
           << "},completion:{protocol_version:" << completed->protocol_version() << ",session:" << completed->session()
           << ",epoch:" << completed->epoch() << ",sequence:" << completed->sequence()
           << ",state:" << completed->state() << ",input_inode:" << command->inode()
           << ",input_counter:" << command->counter() << ",publication_sec:" << command->publication_sec()
           << ",publication_nsec:" << command->publication_nsec() << ",started_ns:" << completed->started_ns()
           << ",finished_ns:" << completed->finished_ns() << ",outcome:" << unsigned( completed->outcome() )
           << ",callback_result:" << completed->callback_result() << ",publisher_pid:" << completed->publisher_pid()
           << ",observed_ns:" << completed->observed_ns() << "}}";
}

inline void telem_dmsynco::formatSimulation( std::ostream &output, const DmSyncoSimulation *summary )
{
    if( !summary || !summary->enabled() )
    {
        return;
    }

    output << " simulation_enabled=1 simulation_valid=" << summary->valid()
           << " simulation_feasible=" << summary->feasible() << " simulation_mode=shadow_conditional"
           << " simulation_session=" << summary->session() << " simulation_sequence=" << summary->sequence()
           << " simulation_now_ns=" << summary->now_ns() << " simulation_last_main_ns=" << summary->last_main_ns()
           << " simulation_counter=" << summary->counter() << " simulation_steps=" << summary->steps()
           << " simulation_period_ns=" << summary->period_ns() << " simulation_grid_ns=" << summary->grid_ns()
           << " simulation_phase_ns=" << summary->phase_ns() << " simulation_phase_min_ns=" << summary->phase_min_ns()
           << " simulation_phase_max_ns=" << summary->phase_max_ns()
           << " simulation_phase_count=" << summary->phase_count()
           << " simulation_period_samples=" << summary->period_samples()
           << " simulation_phase_samples=" << summary->phase_samples()
           << " simulation_next_grid_ns=" << summary->next_grid_ns()
           << " simulation_next_active_ns=" << summary->next_active_ns()
           << " simulation_early_reserve_ns=" << summary->early_reserve_ns()
           << " simulation_model={delivery_ns:" << summary->delivery_ns() << ",keepout_ns:" << summary->keepout_ns()
           << ",guard_ns:" << summary->guard_ns() << ",snapshot_ns:" << summary->snapshot_ns()
           << ",copy_ns:" << summary->copy_ns() << ",lead_ns:" << summary->lead_ns()
           << ",observation_ns:" << summary->observation_ns() << ",visibility_ns:" << summary->visibility_ns()
           << ",reserve_ns:" << summary->reserve_ns() << ",window:" << summary->window()
           << ",phaseGain:" << summary->phase_gain() << ",phaseSlew_ns:" << summary->phase_slew_ns()
           << ",cadenceMinPeriod_ns:" << summary->cadence_min_period_ns() << '}'
           << " simulation_quality={duplicates:" << summary->duplicates() << ",gaps:" << summary->gaps()
           << ",resets:" << summary->resets() << ",invalid:" << summary->invalid()
           << ",excluded:" << summary->excluded() << ",outputGaps:" << summary->output_gaps()
           << ",associations:" << summary->associations() << '}';

    formatCadence( output, "simulation_cadence", summary->cadence() );

    for( size_t index = 0; index < summary->outcomes()->size(); ++index )
    {
        const auto *step = summary->outcomes()->Get( index );
        output << " sim_step={index:" << index << ",considered:" << step->considered()
               << ",wouldRun:" << step->would_run() << ",skipped:" << step->skipped()
               << ",earlyMain:" << step->early_main() << ",missingMain:" << step->missing_main()
               << ",missingOutput:" << step->missing_output() << ",readerHold:" << step->reader_hold()
               << ",expired:" << step->expired() << ",overlap:" << step->overlap() << '}';
    }

    formatOverlapTrace( output, summary->overlap_trace() );
    formatEstimatorTrace( output, summary->estimator_trace() );
}

inline void telem_dmsynco::formatOverlapTrace( std::ostream &output, const DmSyncoOverlapTrace *trace )
{
    if( !trace || !trace->enabled() )
    {
        return;
    }

    output << " trace_quality={enabled:1,detected:" << trace->detected() << ",drained:" << trace->drained()
           << ",dropped:" << trace->dropped() << ",queued:" << trace->queued() << ",capacity:" << trace->capacity()
           << ",logFailed:" << trace->log_failed() << ",batch:" << trace->batch_count() << '}';

    for( const auto *event : *trace->events() )
    {
        const auto *source = event->observation();
        output << " sim_overlap={id:" << event->id() << ",session:" << event->session()
               << ",origin:" << ( event->origin() == DmSyncoOverlapOrigin_primary ? "primary" : "combined" )
               << ",window:" << ( event->window() == DmSyncoOverlapWindow_copy ? "copy" : "readerHold" )
               << ",mainCounter:" << event->main_counter() << ",combinedCounter:" << event->combined_counter()
               << ",planCounter:" << event->plan_counter() << ",lastMainCounter:" << event->last_main_counter()
               << ",planEstimatorId:" << event->plan_estimator_id() << ",lastEstimatorId:" << event->last_estimator_id()
               << ",step:" << event->step() << ",model_clock:monotonic,observed_ns:" << event->observed_ns()
               << ",mainGrid_ns:" << event->main_grid_ns() << ",planGrid_ns:" << event->plan_grid_ns()
               << ",period_ns:" << event->period_ns() << ",copyBegin_ns:" << event->copy_begin_ns()
               << ",copyEnd_ns:" << event->copy_end_ns() << ",readerEnd_ns:" << event->reader_end_ns()
               << ",lastMain_ns:" << event->last_main_ns() << ",lastOutput_ns:" << event->last_output_ns()
               << ",streamInode:" << source->stream_inode() << ",triggerInode:" << source->trigger_inode()
               << ",receipt_ns:" << source->receipt_ns() << ",callback_ns:" << source->callback_ns()
               << ",lockRequest_ns:" << source->lock_request_ns() << ",lockAcquired_ns:" << source->lock_acquired_ns()
               << ",producer_clock:unconverted,publication_ns:" << source->publication_ns()
               << ",primaryPublication_ns:" << source->primary_publication_ns() << ",start_ns:" << source->start_ns()
               << ",update_ns:" << source->update_ns();

        if( event->period_q32() )
        {
            output << ",period_q32:" << event->period_q32() << ",planFraction_q32:" << event->plan_fraction_q32()
                   << ",period_exact_ns:" << std::setprecision( std::numeric_limits<long double>::max_digits10 )
                   << static_cast<long double>( event->period_q32() ) / dmSyncoSimulation::periodScale
                   << std::setprecision( 10 );
        }

        output << '}';
    }
}

inline void telem_dmsynco::formatEstimatorTrace( std::ostream &output, const DmSyncoEstimatorTrace *trace )
{
    if( !trace || !trace->enabled() )
    {
        return;
    }

    output << " estimator_quality={enabled:1,observed:" << trace->observed() << ",triggers:" << trace->triggers()
           << ",selected:" << trace->selected() << ",drained:" << trace->drained() << ",dropped:" << trace->dropped()
           << ",queued:" << trace->queued() << ",capacity:" << trace->capacity() << ",logFailed:" << trace->log_failed()
           << ",history:" << trace->history() << ",historyCapacity:" << trace->history_capacity()
           << ",postRemaining:" << trace->post_remaining() << ",batch:" << trace->batch_count() << '}';

    for( const auto *sample : *trace->samples() )
    {
        output << " sim_estimator={id:" << sample->id() << ",session:" << sample->session()
               << ",sequence:" << sample->sequence() << ",counter:" << sample->counter()
               << ",model_clock:monotonic,observed_ns:" << sample->observed_ns()
               << ",interval_clock:producer_elapsed,publicationInterval_ns:" << sample->publication_interval_ns()
               << ",sequential:" << sample->sequential() << ",intervalAccepted:" << sample->interval_accepted()
               << ",periodSamples:" << sample->period_samples() << ",phaseSamples:" << sample->phase_samples()
               << ",periodBefore_ns:" << sample->period_before_ns() << ",period_ns:" << sample->period_ns()
               << ",predicted_ns:" << sample->predicted_ns() << ",nextCounter:" << sample->next_counter()
               << ",sourcePlanCounter:" << sample->source_plan_counter()
               << ",sourceEstimatorId:" << sample->source_estimator_id()
               << ",sourceGrid_ns:" << sample->source_grid_ns() << ",baseGrid_ns:" << sample->base_grid_ns()
               << ",candidate_ns:" << sample->candidate_ns()
               << ",phaseGain:" << std::setprecision( std::numeric_limits<double>::max_digits10 )
               << sample->phase_gain() << std::setprecision( 10 ) << ",phaseSlew_ns:" << sample->phase_slew_ns()
               << ",rawCorrection_ns:" << sample->raw_correction_ns()
               << ",appliedCorrection_ns:" << sample->applied_correction_ns()
               << ",committedGrid_ns:" << sample->committed_grid_ns() << ",committed:" << sample->committed();

        if( sample->cadence() )
        {
            output << ",periodBefore_q32:" << sample->period_before_q32()
                   << ",sourceFraction_q32:" << sample->source_fraction_q32()
                   << ",baseFraction_q32:" << sample->base_fraction_q32()
                   << ",committedFraction_q32:" << sample->committed_fraction_q32();
        }

        output << '}';
        formatCadence( output, "estimator_cadence", sample->cadence() );
    }
}

inline void telem_dmsynco::formatCadence( std::ostream &output, const char *prefix, const DmSyncoCadence *cadence )
{
    if( !cadence )
    {
        return;
    }

    const char *state = cadence->state() == DmSyncoCadenceState_nominal   ? "nominal"
                        : cadence->state() == DmSyncoCadenceState_growing ? "growing"
                                                                          : "full";
    output << ' ' << prefix << "={state:" << state << ",clock:monotonic,period_q32:" << cadence->period_q32()
           << ",period_scale:" << dmSyncoSimulation::periodScale
           << ",period_exact_ns:" << std::setprecision( std::numeric_limits<long double>::max_digits10 )
           << static_cast<long double>( cadence->period_q32() ) / dmSyncoSimulation::periodScale
           << std::setprecision( 10 ) << ",epoch:" << cadence->epoch() << ",anchors:" << cadence->anchors()
           << ",cycles:" << cadence->cycles() << ",updates:" << cadence->updates()
           << ",rejected:" << cadence->rejected() << ",span_ns:" << cadence->span_ns()
           << ",window_ns:" << cadence->window_ns() << ",update_ns:" << cadence->update_ns()
           << ",observation_ns:" << cadence->observation_ns() << ",updated_ns:" << cadence->updated_ns() << '}';
}

inline std::string telem_dmsynco::msgString( void *buffer, flatlogs::msgLenT len )
{
    static_cast<void>( len );

    const auto       *record = GetTelem_dmsynco_fb( buffer );
    const std::string prefix =
        "[dmsynco] state=" + std::to_string( record->state() ) + " steps=" + std::to_string( record->steps() ) +
        " period_ns=" + std::to_string( record->period_ns() ) + " delivered=" + std::to_string( record->delivered() ) +
        " missed=" + std::to_string( record->missed() ) + " faults=" + std::to_string( record->faults() );
    std::ostringstream output;
    output << prefix << " schedule_valid=" << record->schedule_valid() << std::setprecision( 10 );

    formatObservation( output, "primary", "primaryPublicationInterval", record->primary_timing() );
    const auto *combiner = record->combiner_timing();

    if( !combiner )
    {
        output << " combiner_timing=unavailable";
    }
    else
    {
        output << " combiner_enabled=" << combiner->enabled() << " combiner_session=" << combiner->session()
               << " combiner_sequence=" << combiner->sequence() << " combiner_now_ns=" << combiner->now_ns()
               << " combiner_window_s=" << combiner->window_sec() << " combiner_span_s=" << combiner->span_sec()
               << " combiner_age_s=" << combiner->age_sec() << " combiner_capacity=" << combiner->capacity()
               << " combiner_queued=" << combiner->queued() << " combiner_writer_pid=" << combiner->writer_pid()
               << " combiner_trigger_inode=" << combiner->trigger_inode()
               << " combiner_primary_counter=" << combiner->primary_counter()
               << " combiner_quality={events:" << combiner->events() << ",receipts:" << combiner->receipts()
               << ",synthetic:" << combiner->synthetic() << ",rejected:" << combiner->rejected()
               << ",duplicates:" << combiner->duplicates() << ",counterGaps:" << combiner->counter_gaps()
               << ",invalidMetadata:" << combiner->invalid_metadata() << ",missingTrace:" << combiner->missing_trace()
               << ",invalidTrace:" << combiner->invalid_trace() << ",repeatedStart:" << combiner->repeated_start()
               << ",unlinkedPrimary:" << combiner->unlinked_primary()
               << ",invalidPrimary:" << combiner->invalid_primary()
               << ",repeatedPrimary:" << combiner->repeated_primary() << ",dropped:" << combiner->dropped() << '}';

        formatStats( output, "combinerPublicationInterval", combiner->publication_interval() );
        formatStats( output, "combinerReceiptInterval", combiner->receipt_interval() );
        formatStats( output, "combinerExecution", combiner->execution() );
        formatStats( output, "combinerPublication", combiner->publication() );
        formatStats( output, "combinerPrimaryStart", combiner->primary_start() );
        formatStats( output, "combinerPrimaryPublication", combiner->primary_publication() );
    }

    formatObservation( output, "camera", "cameraAcquisitionInterval", record->camera_timing() );
    formatPipeline( output, record->pipeline_timing() );
    formatObservation( output, "cameraWrite", "cameraWriteInterval", record->camera_publication_timing() );
    formatSimulation( output, record->simulation() );
    formatReader( output, record->reader_timing() );

    return output.str();
}

inline logMetaDetail telem_dmsynco::getAccessor( const std::string &member )
{
    static_cast<void>( member );
    return {};
}

} // namespace logger
} // namespace MagAOX
#endif
