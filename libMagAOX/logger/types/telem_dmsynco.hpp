/** \file telem_dmsynco.hpp
 * \brief Aggregate syncopation timing, ordered excursions, outcomes and per-step feedback telemetry.
 */
#ifndef logger_types_telem_dmsynco_hpp
#define logger_types_telem_dmsynco_hpp

#include <iomanip>
#include <sstream>

#include <mx/math/floatUtils.hpp>

#include "../../common/dmCombTiming.hpp"
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
                  const dmCombTiming::Summary &combiner = {} /**< [in] independently observed combiner timing */ );
    };

    /// Validate optional primary and combiner statistics while accepting legacy records.
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

    /// Check finite populated statistics and sample-variance validity.
    static bool validStats( const DmSyncoTimingStats *stats /**< [in] required nested cadence population */ );

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

inline telem_dmsynco::messageT::messageT( int                          state,
                                          int                          steps,
                                          int64_t                      period,
                                          uint64_t                     delivered,
                                          uint64_t                     missed,
                                          uint64_t                     faults,
                                          uint64_t                     cached,
                                          int64_t                      snapshot,
                                          int64_t                      copy,
                                          const std::vector<int64_t>  &residuals,
                                          const std::vector<int64_t>  &advances,
                                          const dmTiming::Summary     &primary,
                                          bool                         scheduleValid,
                                          const dmCombTiming::Summary &combiner )
{
    builder.ForceDefaults( true ); // Keep nonfinite input available to verification under fast-math comparisons.

    const auto residualVector = builder.CreateVector( residuals );
    const auto advanceVector  = builder.CreateVector( advances );
    const auto publication =
        timingStats( builder, primary.metrics[dmTiming::metricIndex( dmTiming::Metric::publicationInterval )] );
    const auto receipt =
        timingStats( builder, primary.metrics[dmTiming::metricIndex( dmTiming::Metric::receiptInterval )] );
    const bool primaryValid = primary.metrics[dmTiming::metricIndex( dmTiming::Metric::publicationInterval )].valid &&
                              primary.metrics[dmTiming::metricIndex( dmTiming::Metric::receiptInterval )].valid;
    const auto  primaryPairs   = intervalPairs( builder, primary.intervalPairs );
    const auto &quality        = primary.total;
    const auto  primaryTiming  = CreateDmSyncoPrimaryTiming( builder,
                                                           primary.session,
                                                           primary.sequence,
                                                           primary.nowNs,
                                                           primary.windowSec,
                                                           primary.span,
                                                           primary.age,
                                                           primary.capacity,
                                                           primary.queued,
                                                           primary.enabled,
                                                           primaryValid,
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
                                                           primaryPairs );
    const auto  combinedTiming = combinerTiming( builder, combiner );

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
                                            combinedTiming ) );
}

inline bool telem_dmsynco::validStats( const DmSyncoTimingStats *stats )
{
    if( !stats ||
        ( stats->valid() && ( stats->count() < 2 || !mx::math::isFinite( stats->stddev() ) || stats->stddev() < 0 ) ) )
    {
        return false;
    }

    if( stats->count() &&
        ( !mx::math::isFinite( stats->mean() ) || !mx::math::isFinite( stats->min() ) ||
          !mx::math::isFinite( stats->max() ) || !mx::math::isFinite( stats->span() ) || stats->min() < 0 ||
          stats->mean() < stats->min() || stats->mean() > stats->max() || stats->span() < 0 ) )
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

    const auto *record  = GetTelem_dmsynco_fb( flatlogs::logHeader::messageBuffer( buffer ) );
    const auto *primary = record->primary_timing();

    if( primary && !( validStats( primary->publication() ) && validStats( primary->receipt() ) &&
                      primary->valid() == ( primary->publication()->valid() && primary->receipt()->valid() ) &&
                      mx::math::isFinite( primary->window_sec() ) && primary->window_sec() >= 0 &&
                      mx::math::isFinite( primary->span_sec() ) && primary->span_sec() >= 0 &&
                      mx::math::isFinite( primary->age_sec() ) && primary->age_sec() >= -1 &&
                      validIntervalPairs( primary->interval_pairs() ) ) )
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

inline std::string telem_dmsynco::msgString( void *buffer, flatlogs::msgLenT len )
{
    static_cast<void>( len );

    const auto       *record = GetTelem_dmsynco_fb( buffer );
    const std::string prefix =
        "[dmsynco] state=" + std::to_string( record->state() ) + " steps=" + std::to_string( record->steps() ) +
        " period_ns=" + std::to_string( record->period_ns() ) + " delivered=" + std::to_string( record->delivered() ) +
        " missed=" + std::to_string( record->missed() ) + " faults=" + std::to_string( record->faults() );
    const auto *primary = record->primary_timing();

    std::ostringstream output;
    output << prefix << " schedule_valid=" << record->schedule_valid() << std::setprecision( 10 );

    if( primary )
    {
        output << " primary_enabled=" << primary->enabled() << " primary_valid=" << primary->valid()
               << " primary_session=" << primary->session() << " primary_sequence=" << primary->sequence()
               << " primary_now_ns=" << primary->now_ns() << " primary_window_s=" << primary->window_sec()
               << " primary_span_s=" << primary->span_sec() << " primary_age_s=" << primary->age_sec()
               << " primary_capacity=" << primary->capacity() << " primary_queued=" << primary->queued()
               << " primary_quality={events:" << primary->events() << ",receipts:" << primary->receipts()
               << ",synthetic:" << primary->synthetic() << ",rejected:" << primary->rejected()
               << ",duplicates:" << primary->duplicates() << ",counterGaps:" << primary->counter_gaps()
               << ",invalid:" << primary->invalid() << ",dropped:" << primary->dropped() << '}';

        formatStats( output, "primaryPublicationInterval", primary->publication() );
        formatStats( output, "primaryReceiptInterval", primary->receipt() );
    }
    else
    {
        output << " primary_timing=unavailable";
    }

    const auto *pairs = primary ? primary->interval_pairs() : nullptr;

    if( pairs )
    {
        output << " primary_pairs={enabled:" << pairs->enabled() << ",intervals:" << pairs->intervals()
               << ",pairs:" << pairs->pairs() << ",median_s:" << pairs->median()
               << ",threshold_s:" << pairs->threshold() << ",lateEarly:" << pairs->late_early_sum()->count() << '/'
               << pairs->after_late()->count() << ",earlyLate:" << pairs->early_late_sum()->count() << '/'
               << pairs->after_early()->count() << '}';

        formatStats( output, "primaryAfterLateInterval", pairs->after_late() );
        formatStats( output, "primaryAfterEarlyInterval", pairs->after_early() );
        formatStats( output, "primaryLateEarlySum", pairs->late_early_sum() );
        formatStats( output, "primaryEarlyLateSum", pairs->early_late_sum() );
    }
    else
    {
        output << " primary_pairs=unavailable";
    }

    const auto *combiner = record->combiner_timing();

    if( !combiner )
    {
        output << " combiner_timing=unavailable";
        return output.str();
    }

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
           << ",unlinkedPrimary:" << combiner->unlinked_primary() << ",invalidPrimary:" << combiner->invalid_primary()
           << ",repeatedPrimary:" << combiner->repeated_primary() << ",dropped:" << combiner->dropped() << '}';

    formatStats( output, "combinerPublicationInterval", combiner->publication_interval() );
    formatStats( output, "combinerReceiptInterval", combiner->receipt_interval() );
    formatStats( output, "combinerExecution", combiner->execution() );
    formatStats( output, "combinerPublication", combiner->publication() );
    formatStats( output, "combinerPrimaryStart", combiner->primary_start() );
    formatStats( output, "combinerPrimaryPublication", combiner->primary_publication() );

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
