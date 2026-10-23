/** \file telem_dmsynco.hpp
 * \brief Aggregate syncopation timing, ordered excursions, outcomes and per-step feedback telemetry.
 */
#ifndef logger_types_telem_dmsynco_hpp
#define logger_types_telem_dmsynco_hpp

#include <iomanip>
#include <sstream>

#include <mx/math/floatUtils.hpp>

#include "../../common/dmCombTiming.hpp"
#include "../../common/dmPipelineTiming.hpp"
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
        messageT( int                         state /**< [in] schedule state */,
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
                  const dmCombTiming::Summary &combiner = {} /**< [in] independently observed combiner timing */,
                  const dmTiming::Summary     &camera = {} /**< [in] independent camera acquisition-stamp intervals */,
                  const dmPipelineTiming::Summary &pipeline = {} /**< [in] upstream trace-stage populations */ );
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

    /// Serialize either primary publication or independent camera acquisition intervals.
    static flatbuffers::Offset<DmSyncoPrimaryTiming>
    observationTiming( flatbuffers::FlatBufferBuilder &builder /**< [in/out] destination builder */,
                       const dmTiming::Summary        &summary /**< [in] retained external-stamp cadence */ );

    /// Serialize upstream stage identities, conditional durations and signed changes.
    static flatbuffers::Offset<DmSyncoPipelineTiming>
    pipelineTiming( flatbuffers::FlatBufferBuilder  &builder /**< [in/out] destination builder */,
                    const dmPipelineTiming::Summary &summary /**< [in] coherent pipeline observation */ );

    /// Check finite populated statistics and sample-variance validity.
    static bool validStats( const DmSyncoTimingStats *stats /**< [in] required nested cadence population */,
                            bool signedValues = false /**< [in] allow negative latency changes */ );

    /// Check independently observed primary or camera cadence without changing legacy requirements.
    static bool validObservation( const DmSyncoPrimaryTiming *summary /**< [in] optional cadence summary */ );

    /// Check retained pipeline topology and conditional-population count bounds.
    static bool validPipeline( const DmSyncoPipelineTiming *summary /**< [in] optional trace-stage summary */ );

    /// Format independent primary/camera cadence and ordered excursions with distinct names.
    static void formatObservation( std::ostream &output /**< [in/out] formatted aggregate */,
                                   const char   *prefix /**< [in] primary or camera identifier */,
                                   const char   *intervalName /**< [in] publication/acquisition interval label */,
                                   const DmSyncoPrimaryTiming *summary /**< [in] optional cadence summary */ );

    /// Format retained pipeline stages and their independently counted conditional populations.
    static void formatPipeline( std::ostream                &output /**< [in/out] formatted aggregate */,
                                const DmSyncoPipelineTiming *summary /**< [in] optional pipeline summary */ );

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

inline telem_dmsynco::messageT::messageT( int                              state,
                                          int                              steps,
                                          int64_t                          period,
                                          uint64_t                         delivered,
                                          uint64_t                         missed,
                                          uint64_t                         faults,
                                          uint64_t                         cached,
                                          int64_t                          snapshot,
                                          int64_t                          copy,
                                          const std::vector<int64_t>      &residuals,
                                          const std::vector<int64_t>      &advances,
                                          const dmTiming::Summary         &primary,
                                          bool                             scheduleValid,
                                          const dmCombTiming::Summary     &combiner,
                                          const dmTiming::Summary         &camera,
                                          const dmPipelineTiming::Summary &pipeline )
{
    builder.ForceDefaults( true ); // Keep nonfinite input available to verification under fast-math comparisons.

    const auto residualVector = builder.CreateVector( residuals );
    const auto advanceVector  = builder.CreateVector( advances );
    const auto primaryTiming  = observationTiming( builder, primary );
    const auto cameraTiming   = observationTiming( builder, camera );
    const auto upstreamTiming = pipelineTiming( builder, pipeline );
    const auto combinedTiming = combinerTiming( builder, combiner );

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
                                            upstreamTiming ) );
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
        !validPipeline( record->pipeline_timing() ) )
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
