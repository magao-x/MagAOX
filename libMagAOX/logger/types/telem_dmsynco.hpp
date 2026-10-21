/** \file telem_dmsynco.hpp
 * \brief Aggregate syncopation timing, outcomes and per-step feedback telemetry.
 */
#ifndef logger_types_telem_dmsynco_hpp
#define logger_types_telem_dmsynco_hpp

#include <iomanip>
#include <sstream>

#include <mx/math/floatUtils.hpp>

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
                  bool scheduleValid = true /**< [in] false when scheduler fields are unavailable placeholders */ );
    };

    /// Validate the FlatBuffer and optional primary statistics while accepting legacy records.
    static bool verify( flatlogs::bufferPtrT &buffer /**< [in] complete log entry */,
                        flatlogs::msgLenT     len /**< [in] message length */ );

    /// Format the aggregate for log readers.
    static std::string msgString( void             *buffer /**< [in] message buffer */,
                                  flatlogs::msgLenT len /**< [in] message length, unused */ );

    /// This aggregate has no scalar logMeta projections; read its typed FlatBuffer fields.
    static logMetaDetail getAccessor( const std::string &member /**< [in] requested scalar member, unused */ );

  private:
    /// Serialize one independently counted cadence population without changing its units or clock.
    static flatbuffers::Offset<DmSyncoTimingStats>
    timingStats( flatbuffers::FlatBufferBuilder &builder /**< [in/out] destination builder */,
                 const dmTiming::Stats          &stats /**< [in] cadence population in seconds */ );

    /// Check finite populated statistics and sample-variance validity.
    static bool validStats( const DmSyncoTimingStats *stats /**< [in] required nested cadence population */ );

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

inline telem_dmsynco::messageT::messageT( int                         state,
                                          int                         steps,
                                          int64_t                     period,
                                          uint64_t                    delivered,
                                          uint64_t                    missed,
                                          uint64_t                    faults,
                                          uint64_t                    cached,
                                          int64_t                     snapshot,
                                          int64_t                     copy,
                                          const std::vector<int64_t> &residuals,
                                          const std::vector<int64_t> &advances,
                                          const dmTiming::Summary    &primary,
                                          bool                        scheduleValid )
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
    const auto &quality       = primary.total;
    const auto  primaryTiming = CreateDmSyncoPrimaryTiming( builder,
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
                                                           quality.dropped );

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
                                            scheduleValid ) );
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

    const auto *primary = GetTelem_dmsynco_fb( flatlogs::logHeader::messageBuffer( buffer ) )->primary_timing();

    if( !primary )
    {
        return true;
    }

    return validStats( primary->publication() ) && validStats( primary->receipt() ) &&
           primary->valid() == ( primary->publication()->valid() && primary->receipt()->valid() ) &&
           mx::math::isFinite( primary->window_sec() ) && primary->window_sec() >= 0 &&
           mx::math::isFinite( primary->span_sec() ) && primary->span_sec() >= 0 &&
           mx::math::isFinite( primary->age_sec() ) && primary->age_sec() >= -1;
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

    if( !primary )
    {
        return prefix + " schedule_valid=" + std::to_string( record->schedule_valid() ) + " primary_timing=unavailable";
    }

    std::ostringstream output;
    output << prefix << " schedule_valid=" << record->schedule_valid() << std::setprecision( 10 )
           << " primary_enabled=" << primary->enabled() << " primary_valid=" << primary->valid()
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
