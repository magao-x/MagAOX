/** \file telem_fgstage.hpp
 * \brief Optional frame-grabber stage durations, matched publication excursions and collection quality.
 * \ingroup logger_types_files
 */
#ifndef logger_types_telem_fgstage_hpp
#define logger_types_telem_fgstage_hpp

#include <array>
#include <iomanip>
#include <limits>
#include <sstream>
#include <type_traits>

#include <mx/math/floatUtils.hpp>

#include "../../common/frameGrabberTiming.hpp"
#include "generated/telem_fgstage_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{

/// Independent camera diagnostics; monotonic segment attribution does not establish external timestamp provenance.
/** \ingroup logger_types */
struct telem_fgstage : public flatbuffer_log
{
    static constexpr flatlogs::eventCodeT eventCode    = eventCodes::TELEM_FGSTAGE;    ///< Stable logger dispatch code.
    static constexpr flatlogs::logPrioT   defaultLevel = flatlogs::logPrio::LOG_TELEM; ///< Telemetry priority.
    static inline timespec                lastRecord{}; ///< Last emission time, owned by the ordinary telemeter.

    /// Serialized consumer-owned snapshot, with separate raw and qualified change populations.
    struct messageT : public fbMessage
    {
        /// Serialize frame coverage, independent metric populations and control-path observations.
        messageT( const fgTiming::Summary &summary /**< [in] coherent ordinary-thread statistics */ );
    };

    /// Validate complete metric-ID sets, population signs, classification bounds and snapshot coverage.
    static bool verify( flatlogs::bufferPtrT &buffer /**< [in] complete log entry */,
                        flatlogs::msgLenT     len /**< [in] serialized message length */ );

    /// Format populated segments, bracket ambiguity, serial association and quality in seconds.
    static std::string msgString( void             *buffer /**< [in] serialized message */,
                                  flatlogs::msgLenT len /**< [in] serialized message length, unused */ );

    /// Project typed snapshot metadata, metric populations and quality without interpolating across windows.
    static logMetaDetail getAccessor( const std::string &member /**< [in] scalar or population_field name */ );

  private:
    static constexpr size_t metricCount = size_t( fgTiming::Metric::count ); ///< Complete stable metric identifier set.

    /// Serialize one raw, signed-change or serial-duration population in seconds.
    static flatbuffers::Offset<FgStageStats>
    stats( flatbuffers::FlatBufferBuilder &builder /**< [in/out] destination builder */,
           const fgTiming::Stats          &value /**< [in] independently counted population */ );

    /// Serialize one complete metric population without relying on implicit vector positions.
    static flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<FgStageMetricStats>>>
    metrics( flatbuffers::FlatBufferBuilder                 &builder /**< [in/out] destination builder */,
             const std::array<fgTiming::Stats, metricCount> &values /**< [in] complete metric set */ );

    /// Serialize one cumulative or since-summary quality population.
    static flatbuffers::Offset<FgStageCounters>
    counters( flatbuffers::FlatBufferBuilder &builder /**< [in/out] destination builder */,
              const fgTiming::Counters       &value /**< [in] observation-quality population */ );

    /// Validate one population, allowing negative values only for explicit clock context or changes.
    static bool validStats( const FgStageStats *value /**< [in] required population */,
                            bool                signedValues /**< [in] permit signed mean and extrema */ );

    /// Validate a full unique metric-ID set and its maximum independent population count.
    static bool validMetrics( const flatbuffers::Vector<flatbuffers::Offset<FgStageMetricStats>> *rows
                              /**< [in] required metric population */,
                              uint64_t maximum /**< [in] retained frame or qualified cohort bound */,
                              bool     changes /**< [in] every metric is a signed frame-to-frame change */ );

    /// Format a population while distinguishing a missing sample from unavailable sample variance.
    static void formatStats( std::ostream       &output /**< [in/out] formatted record */,
                             const std::string  &name /**< [in] stable population label */,
                             const FgStageStats *value /**< [in] verified population */ );

    /// Extract a typed summary scalar for logMeta.
    template <typename valueT, auto member>
    static valueT scalar( void *buffer /**< [in] serialized message */ );

    /// Describe a typed scalar without interpolating across observation windows.
    template <typename valueT, auto member>
    static logMetaDetail scalarAccessor( const std::string &name /**< [in] metadata label */ );

    /// Extract one metric field using stable identity rather than its vector position.
    template <size_t population, size_t index, typename valueT, auto member>
    static valueT metric( void *buffer /**< [in] serialized message */ );

    /// Resolve a raw, late-change or early-change metric and its typed field.
    template <size_t population, size_t index = 0>
    static logMetaDetail metricAccessor( const std::string &name /**< [in] complete metadata label */ );

    /// Extract one quality counter from the cumulative or since-summary population.
    template <bool delta, auto member>
    static uint64_t counter( void *buffer /**< [in] serialized message */ );

    /// Resolve a cumulative or since-summary counter to its metadata accessor.
    template <bool delta>
    static logMetaDetail
    counterAccessor( const std::string &name /**< [in] counter label without population prefix */ );
};

inline flatbuffers::Offset<FgStageStats> telem_fgstage::stats( flatbuffers::FlatBufferBuilder &builder,
                                                               const fgTiming::Stats          &value )
{
    return CreateFgStageStats(
        builder, value.count, value.mean, value.stddev, value.min, value.max, value.span, value.valid );
}

inline flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<FgStageMetricStats>>>
telem_fgstage::metrics( flatbuffers::FlatBufferBuilder                 &builder,
                        const std::array<fgTiming::Stats, metricCount> &values )
{
    std::array<flatbuffers::Offset<FgStageMetricStats>, metricCount> rows;

    for( size_t index = 0; index < rows.size(); ++index )
    {
        rows[index] = CreateFgStageMetricStats( builder, FgStageMetric( index ), stats( builder, values[index] ) );
    }

    return builder.CreateVector( rows.data(), rows.size() );
}

inline flatbuffers::Offset<FgStageCounters> telem_fgstage::counters( flatbuffers::FlatBufferBuilder &builder,
                                                                     const fgTiming::Counters       &value )
{
    return CreateFgStageCounters( builder,
                                  value.events,
                                  value.published,
                                  value.rejected,
                                  value.failed,
                                  value.duplicates,
                                  value.counterGaps,
                                  value.frameGaps,
                                  value.invalid,
                                  value.bracketWide,
                                  value.dropped,
                                  value.controls,
                                  value.controlFailed,
                                  value.controlInvalid,
                                  value.controlDropped );
}

inline telem_fgstage::messageT::messageT( const fgTiming::Summary &summary )
{
    static_assert( size_t( fgTiming::Metric::acquisitionInterval ) == size_t( FgStageMetric_acquisitionInterval ) );
    static_assert( size_t( fgTiming::Metric::publicationInterval ) == size_t( FgStageMetric_publicationInterval ) );
    static_assert( size_t( fgTiming::Metric::acquisitionToWrite ) == size_t( FgStageMetric_acquisitionToWrite ) );
    static_assert( size_t( fgTiming::Metric::waitReturnInterval ) == size_t( FgStageMetric_waitReturnInterval ) );
    static_assert( size_t( fgTiming::Metric::writeEndpointInterval ) == size_t( FgStageMetric_writeEndpointInterval ) );
    static_assert( size_t( fgTiming::Metric::wait ) == size_t( FgStageMetric_wait ) );
    static_assert( size_t( fgTiming::Metric::rearm ) == size_t( FgStageMetric_rearm ) );
    static_assert( size_t( fgTiming::Metric::validation ) == size_t( FgStageMetric_validation ) );
    static_assert( size_t( fgTiming::Metric::conversion ) == size_t( FgStageMetric_conversion ) );
    static_assert( size_t( fgTiming::Metric::writeBracket ) == size_t( FgStageMetric_writeBracket ) );
    static_assert( size_t( fgTiming::Metric::mainPost ) == size_t( FgStageMetric_mainPost ) );
    static_assert( size_t( fgTiming::Metric::syncPost ) == size_t( FgStageMetric_syncPost ) );
    static_assert( size_t( fgTiming::Metric::loopTail ) == size_t( FgStageMetric_loopTail ) );
    static_assert( size_t( fgTiming::Metric::waitToWrite ) == size_t( FgStageMetric_waitToWrite ) );
    static_assert( size_t( fgTiming::Metric::previousTail ) == size_t( FgStageMetric_previousTail ) );
    static_assert( size_t( fgTiming::Metric::serialOverlap ) == size_t( FgStageMetric_serialOverlap ) );
    static_assert( size_t( fgTiming::Metric::count ) == 16 );
    static_assert( size_t( fgTiming::Availability::disabled ) == size_t( FgStageAvailability_disabled ) );
    static_assert( size_t( fgTiming::Availability::warmingUp ) == size_t( FgStageAvailability_warmingUp ) );
    static_assert( size_t( fgTiming::Availability::valid ) == size_t( FgStageAvailability_valid ) );
    static_assert( size_t( fgTiming::Availability::stale ) == size_t( FgStageAvailability_stale ) );
    static_assert( size_t( fgTiming::Availability::unavailable ) == size_t( FgStageAvailability_unavailable ) );
    builder.ForceDefaults( true ); // Preserve nonfinite input for validation under fast-math comparisons.

    const auto raw                        = metrics( builder, summary.metrics );
    const auto lateChanges                = metrics( builder, summary.lateChange );
    const auto earlyChanges               = metrics( builder, summary.earlyChange );
    const auto serial                     = stats( builder, summary.serialDuration );
    const auto total                      = counters( builder, summary.total );
    const auto delta                      = counters( builder, summary.delta );
    const auto lateWaitReturnInterval     = stats( builder, summary.lateWaitReturnInterval );
    const auto earlyWaitReturnInterval    = stats( builder, summary.earlyWaitReturnInterval );
    const auto lateWriteEndpointInterval  = stats( builder, summary.lateWriteEndpointInterval );
    const auto earlyWriteEndpointInterval = stats( builder, summary.earlyWriteEndpointInterval );
    const auto latePreviousTail           = stats( builder, summary.latePreviousTail );
    const auto earlyPreviousTail          = stats( builder, summary.earlyPreviousTail );

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
                                            lateChanges,
                                            earlyChanges,
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

inline bool telem_fgstage::validStats( const FgStageStats *value, bool signedValues )
{
    if( !value || ( value->valid() && value->count() < 2 ) )
    {
        return false;
    }

    return !value->count() ||
           ( mx::math::isFinite( value->mean() ) && mx::math::isFinite( value->stddev() ) &&
             mx::math::isFinite( value->min() ) && mx::math::isFinite( value->max() ) &&
             mx::math::isFinite( value->span() ) && value->stddev() >= 0 && value->span() >= 0 &&
             ( signedValues || value->min() >= 0 ) && value->mean() >= value->min() && value->mean() <= value->max() );
}

inline bool telem_fgstage::validMetrics( const flatbuffers::Vector<flatbuffers::Offset<FgStageMetricStats>> *rows,
                                         uint64_t                                                            maximum,
                                         bool                                                                changes )
{
    if( !rows || rows->size() != metricCount )
    {
        return false;
    }

    std::array<bool, metricCount> seen{};

    for( const auto *row : *rows )
    {
        if( !row || size_t( row->metric() ) >= seen.size() || seen[size_t( row->metric() )] )
        {
            return false;
        }

        seen[size_t( row->metric() )] = true;
        const bool signedValues       = changes || row->metric() == FgStageMetric_acquisitionToWrite;

        if( !validStats( row->value(), signedValues ) || row->value()->count() > maximum )
        {
            return false;
        }
    }

    return true;
}

inline bool telem_fgstage::verify( flatlogs::bufferPtrT &buffer, flatlogs::msgLenT len )
{
    if( !buffer )
    {
        return false;
    }

    auto                 *payload = static_cast<uint8_t *>( flatlogs::logHeader::messageBuffer( buffer ) );
    flatbuffers::Verifier verifier( payload, len );

    if( !VerifyTelem_fgstage_fbBuffer( verifier ) )
    {
        return false;
    }

    const auto *record = GetTelem_fgstage_fb( payload );

    if( record->state() < FgStageAvailability_MIN || record->state() > FgStageAvailability_MAX ||
        !mx::math::isFinite( record->window_sec() ) || record->window_sec() < 0 ||
        !mx::math::isFinite( record->span_sec() ) || record->span_sec() < 0 ||
        !mx::math::isFinite( record->age_sec() ) || record->age_sec() < -1 || !mx::math::isFinite( record->median() ) ||
        record->median() < 0 || !mx::math::isFinite( record->threshold() ) || record->threshold() < 0 ||
        !mx::math::isFinite( record->bracket_limit() ) || record->bracket_limit() < 0 ||
        record->queued() > record->capacity() || record->control_queued() > record->capacity() ||
        record->serial_overlaps() > record->capacity() ||
        record->intervals() > ( record->capacity() ? record->capacity() - 1 : 0 ) ||
        record->qualified() > record->intervals() || record->ambiguous() != record->intervals() - record->qualified() ||
        record->late() > record->qualified() || record->early() > record->qualified() - record->late() ||
        ( record->intervals() ? record->median() <= 0 : record->median() != 0 ) ||
        ( record->threshold() == 0 && ( record->late() || record->early() ) ) ||
        ( record->bracket_limit() == 0 && record->qualified() ) ||
        ( !record->stages_enabled() && record->intervals() ) )
    {
        return false;
    }

    const FgStageStats *conditional[] = { record->late_wait_return_interval(),
                                          record->early_wait_return_interval(),
                                          record->late_write_endpoint_interval(),
                                          record->early_write_endpoint_interval(),
                                          record->late_previous_tail(),
                                          record->early_previous_tail() };

    for( size_t index = 0; index < 6; ++index )
    {
        if( !validStats( conditional[index], false ) ||
            conditional[index]->count() > ( index % 2 == 0 ? record->late() : record->early() ) )
        {
            return false;
        }
    }

    return validMetrics( record->metrics(), record->capacity(), false ) &&
           validMetrics( record->late_change(), record->late(), true ) &&
           validMetrics( record->early_change(), record->early(), true ) &&
           validStats( record->serial_duration(), false ) &&
           record->serial_duration()->count() <= record->total()->controls();
}

inline void telem_fgstage::formatStats( std::ostream &output, const std::string &name, const FgStageStats *value )
{
    output << ' ' << name << "[n=" << value->count() << " valid=" << value->valid() << "]=";

    if( !value->count() )
    {
        output << "unavailable";
        return;
    }

    output << value->mean() << "+/-";

    if( value->valid() )
    {
        output << value->stddev();
    }
    else
    {
        output << "unavailable";
    }

    output << " s min=" << value->min() << " max=" << value->max() << " span=" << value->span();
}

inline std::string telem_fgstage::msgString( void *buffer, flatlogs::msgLenT len )
{
    static_cast<void>( len );
    const auto        *record = GetTelem_fgstage_fb( buffer );
    std::ostringstream output;
    output << std::setprecision( 10 ) << "[fgstage] session=" << record->session() << " sequence=" << record->sequence()
           << " state=" << EnumNameFgStageAvailability( record->state() ) << " enabled=" << record->enabled()
           << " stages_enabled=" << record->stages_enabled() << " fg_valid=" << record->fg_valid()
           << " window_s=" << record->window_sec() << " span_s=" << record->span_sec() << " age_s=" << record->age_sec()
           << " queued=" << record->queued() << '/' << record->capacity()
           << " control_queued=" << record->control_queued() << " median_s=" << record->median()
           << " threshold_s=" << record->threshold() << " bracket_limit_s=" << record->bracket_limit()
           << " intervals=" << record->intervals() << " qualified=" << record->qualified()
           << " ambiguous=" << record->ambiguous() << " late=" << record->late() << " early=" << record->early()
           << " serial_overlaps=" << record->serial_overlaps() << " stream_inode=" << record->stream_inode()
           << " last_frame=" << record->last_frame() << " last_counter=" << record->last_counter()
           << " mode=" << record->mode() << " bin=" << record->bin_x() << 'x' << record->bin_y();

    const std::pair<const char *, const flatbuffers::Vector<flatbuffers::Offset<FgStageMetricStats>> *> populations[] =
        { { "", record->metrics() },
          { "lateChange_", record->late_change() },
          { "earlyChange_", record->early_change() } };

    for( const auto &population : populations )
    {
        for( const auto *row : *population.second )
        {
            formatStats( output,
                         std::string( population.first ) + fgTiming::metricName( fgTiming::Metric( row->metric() ) ),
                         row->value() );
        }
    }

    formatStats( output, "serialDuration", record->serial_duration() );
    formatStats( output, "lateWaitReturnInterval", record->late_wait_return_interval() );
    formatStats( output, "earlyWaitReturnInterval", record->early_wait_return_interval() );
    formatStats( output, "lateWriteEndpointInterval", record->late_write_endpoint_interval() );
    formatStats( output, "earlyWriteEndpointInterval", record->early_write_endpoint_interval() );
    formatStats( output, "latePreviousTail", record->late_previous_tail() );
    formatStats( output, "earlyPreviousTail", record->early_previous_tail() );

    for( const bool delta : { false, true } )
    {
        const auto *value = delta ? record->delta() : record->total();
        output << ' ' << ( delta ? "delta" : "total" ) << "={events:" << value->events()
               << ",published:" << value->published() << ",rejected:" << value->rejected()
               << ",failed:" << value->failed() << ",duplicates:" << value->duplicates()
               << ",counterGaps:" << value->counter_gaps() << ",frameGaps:" << value->frame_gaps()
               << ",invalid:" << value->invalid() << ",bracketWide:" << value->bracket_wide()
               << ",dropped:" << value->dropped() << ",controls:" << value->controls()
               << ",controlFailed:" << value->control_failed() << ",controlInvalid:" << value->control_invalid()
               << ",controlDropped:" << value->control_dropped() << '}';
    }

    return output.str();
}

template <typename valueT, auto member>
inline valueT telem_fgstage::scalar( void *buffer )
{
    return valueT( ( GetTelem_fgstage_fb( buffer )->*member )() );
}

template <typename valueT, auto member>
inline logMetaDetail telem_fgstage::scalarAccessor( const std::string &name )
{
    return { "FG stage " + name,
             mx::fits::fitsType<valueT>(),
             logMeta::metaTypes::State,
             reinterpret_cast<void *>( &scalar<valueT, member> ) };
}

template <size_t population, size_t index, typename valueT, auto member>
inline valueT telem_fgstage::metric( void *buffer )
{
    const auto         *record = GetTelem_fgstage_fb( buffer );
    const FgStageStats *value  = nullptr;

    if constexpr( population >= 3 )
    {
        const FgStageStats *populations[] = { record->serial_duration(),
                                              record->late_wait_return_interval(),
                                              record->early_wait_return_interval(),
                                              record->late_write_endpoint_interval(),
                                              record->early_write_endpoint_interval(),
                                              record->late_previous_tail(),
                                              record->early_previous_tail() };
        value                             = populations[population - 3];
    }
    else
    {
        const auto *rows = population == 0   ? record->metrics()
                           : population == 1 ? record->late_change()
                                             : record->early_change();

        for( const auto *row : *rows )
        {
            if( size_t( row->metric() ) == index )
            {
                value = row->value();
                break;
            }
        }
    }

    if constexpr( std::is_floating_point_v<valueT> )
    {
        if( !value || !value->count() )
        {
            return std::numeric_limits<valueT>::quiet_NaN();
        }

        if constexpr( member == &FgStageStats::stddev )
        {
            if( !value->valid() )
            {
                return std::numeric_limits<valueT>::quiet_NaN();
            }
        }
    }

    return value ? valueT( ( value->*member )() ) : valueT{};
}

template <size_t population, size_t index>
inline logMetaDetail telem_fgstage::metricAccessor( const std::string &name )
{
    const char       *names[] = { "serialDuration_",
                                  "lateWaitReturnInterval_",
                                  "earlyWaitReturnInterval_",
                                  "lateWriteEndpointInterval_",
                                  "earlyWriteEndpointInterval_",
                                  "latePreviousTail_",
                                  "earlyPreviousTail_" };
    const std::string prefix  = population >= 3 ? names[population - 3]
                                                : std::string( population == 0   ? ""
                                                              : population == 1 ? "lateChange_"
                                                                                : "earlyChange_" ) +
                                                     fgTiming::metricName( fgTiming::Metric( index ) ) + '_';

    if( name.starts_with( prefix ) )
    {
        const std::string field    = name.substr( prefix.size() );
        void             *accessor = nullptr;
        int               type     = mx::fits::fitsType<double>();

        if( field == "mean" )
        {
            type     = mx::fits::fitsType<double>();
            accessor = reinterpret_cast<void *>( &metric<population, index, double, &FgStageStats::mean> );
        }

        else if( field == "stddev" )
        {
            type     = mx::fits::fitsType<double>();
            accessor = reinterpret_cast<void *>( &metric<population, index, double, &FgStageStats::stddev> );
        }

        else if( field == "min" )
        {
            type     = mx::fits::fitsType<double>();
            accessor = reinterpret_cast<void *>( &metric<population, index, double, &FgStageStats::min> );
        }

        else if( field == "max" )
        {
            type     = mx::fits::fitsType<double>();
            accessor = reinterpret_cast<void *>( &metric<population, index, double, &FgStageStats::max> );
        }

        else if( field == "span" )
        {
            type     = mx::fits::fitsType<double>();
            accessor = reinterpret_cast<void *>( &metric<population, index, double, &FgStageStats::span> );
        }

        else if( field == "count" )
        {
            type     = mx::fits::fitsType<uint64_t>();
            accessor = reinterpret_cast<void *>( &metric<population, index, uint64_t, &FgStageStats::count> );
        }

        else if( field == "valid" )
        {
            type     = mx::fits::fitsType<bool>();
            accessor = reinterpret_cast<void *>( &metric<population, index, bool, &FgStageStats::valid> );
        }

        return accessor ? logMetaDetail{ "FG stage " + name, type, logMeta::metaTypes::State, accessor }
                        : logMetaDetail{};
    }

    if constexpr( population < 3 && index + 1 < metricCount )
    {
        return metricAccessor<population, index + 1>( name );
    }

    return {};
}

template <bool delta, auto member>
inline uint64_t telem_fgstage::counter( void *buffer )
{
    const auto *record = GetTelem_fgstage_fb( buffer );
    const auto *value  = delta ? record->delta() : record->total();
    return ( value->*member )();
}

template <bool delta>
inline logMetaDetail telem_fgstage::counterAccessor( const std::string &name )
{
    void *accessor = nullptr;

    if( name == "events" )
    {
        accessor = reinterpret_cast<void *>( &counter<delta, &FgStageCounters::events> );
    }

    else if( name == "published" )
    {
        accessor = reinterpret_cast<void *>( &counter<delta, &FgStageCounters::published> );
    }

    else if( name == "rejected" )
    {
        accessor = reinterpret_cast<void *>( &counter<delta, &FgStageCounters::rejected> );
    }

    else if( name == "failed" )
    {
        accessor = reinterpret_cast<void *>( &counter<delta, &FgStageCounters::failed> );
    }

    else if( name == "duplicates" )
    {
        accessor = reinterpret_cast<void *>( &counter<delta, &FgStageCounters::duplicates> );
    }

    else if( name == "counterGaps" )
    {
        accessor = reinterpret_cast<void *>( &counter<delta, &FgStageCounters::counter_gaps> );
    }

    else if( name == "frameGaps" )
    {
        accessor = reinterpret_cast<void *>( &counter<delta, &FgStageCounters::frame_gaps> );
    }

    else if( name == "invalid" )
    {
        accessor = reinterpret_cast<void *>( &counter<delta, &FgStageCounters::invalid> );
    }

    else if( name == "bracketWide" )
    {
        accessor = reinterpret_cast<void *>( &counter<delta, &FgStageCounters::bracket_wide> );
    }

    else if( name == "dropped" )
    {
        accessor = reinterpret_cast<void *>( &counter<delta, &FgStageCounters::dropped> );
    }

    else if( name == "controls" )
    {
        accessor = reinterpret_cast<void *>( &counter<delta, &FgStageCounters::controls> );
    }

    else if( name == "controlFailed" )
    {
        accessor = reinterpret_cast<void *>( &counter<delta, &FgStageCounters::control_failed> );
    }

    else if( name == "controlInvalid" )
    {
        accessor = reinterpret_cast<void *>( &counter<delta, &FgStageCounters::control_invalid> );
    }

    else if( name == "controlDropped" )
    {
        accessor = reinterpret_cast<void *>( &counter<delta, &FgStageCounters::control_dropped> );
    }

    return accessor ? logMetaDetail{ std::string( delta ? "FG stage delta_" : "FG stage total_" ) + name,
                                     mx::fits::fitsType<uint64_t>(),
                                     logMeta::metaTypes::State,
                                     accessor }
                    : logMetaDetail{};
}

inline logMetaDetail telem_fgstage::getAccessor( const std::string &member )
{
    if( member == "session" )
    {
        return scalarAccessor<uint64_t, &Telem_fgstage_fb::session>( member );
    }

    if( member == "sequence" )
    {
        return scalarAccessor<uint64_t, &Telem_fgstage_fb::sequence>( member );
    }

    if( member == "nowNs" )
    {
        return scalarAccessor<int64_t, &Telem_fgstage_fb::now_ns>( member );
    }

    if( member == "newestNs" )
    {
        return scalarAccessor<int64_t, &Telem_fgstage_fb::newest_ns>( member );
    }

    if( member == "windowSec" )
    {
        return scalarAccessor<double, &Telem_fgstage_fb::window_sec>( member );
    }

    if( member == "span" )
    {
        return scalarAccessor<double, &Telem_fgstage_fb::span_sec>( member );
    }

    if( member == "age" )
    {
        return scalarAccessor<double, &Telem_fgstage_fb::age_sec>( member );
    }

    if( member == "capacity" )
    {
        return scalarAccessor<uint64_t, &Telem_fgstage_fb::capacity>( member );
    }

    if( member == "queued" )
    {
        return scalarAccessor<uint64_t, &Telem_fgstage_fb::queued>( member );
    }

    if( member == "controlQueued" )
    {
        return scalarAccessor<uint64_t, &Telem_fgstage_fb::control_queued>( member );
    }

    if( member == "enabled" )
    {
        return scalarAccessor<bool, &Telem_fgstage_fb::enabled>( member );
    }

    if( member == "stagesEnabled" )
    {
        return scalarAccessor<bool, &Telem_fgstage_fb::stages_enabled>( member );
    }

    if( member == "fgValid" )
    {
        return scalarAccessor<bool, &Telem_fgstage_fb::fg_valid>( member );
    }

    if( member == "state" )
    {
        return scalarAccessor<int, &Telem_fgstage_fb::state>( member );
    }

    if( member == "median" )
    {
        return scalarAccessor<double, &Telem_fgstage_fb::median>( member );
    }

    if( member == "threshold" )
    {
        return scalarAccessor<double, &Telem_fgstage_fb::threshold>( member );
    }

    if( member == "bracketLimit" )
    {
        return scalarAccessor<double, &Telem_fgstage_fb::bracket_limit>( member );
    }

    if( member == "intervals" )
    {
        return scalarAccessor<uint64_t, &Telem_fgstage_fb::intervals>( member );
    }

    if( member == "qualified" )
    {
        return scalarAccessor<uint64_t, &Telem_fgstage_fb::qualified>( member );
    }

    if( member == "ambiguous" )
    {
        return scalarAccessor<uint64_t, &Telem_fgstage_fb::ambiguous>( member );
    }

    if( member == "late" )
    {
        return scalarAccessor<uint64_t, &Telem_fgstage_fb::late>( member );
    }

    if( member == "early" )
    {
        return scalarAccessor<uint64_t, &Telem_fgstage_fb::early>( member );
    }

    if( member == "serialOverlaps" )
    {
        return scalarAccessor<uint64_t, &Telem_fgstage_fb::serial_overlaps>( member );
    }

    if( member == "streamInode" )
    {
        return scalarAccessor<uint64_t, &Telem_fgstage_fb::stream_inode>( member );
    }

    if( member == "lastFrame" )
    {
        return scalarAccessor<uint64_t, &Telem_fgstage_fb::last_frame>( member );
    }

    if( member == "lastCounter" )
    {
        return scalarAccessor<uint64_t, &Telem_fgstage_fb::last_counter>( member );
    }

    if( member == "mode" )
    {
        return scalarAccessor<uint32_t, &Telem_fgstage_fb::mode>( member );
    }

    if( member == "binX" )
    {
        return scalarAccessor<uint32_t, &Telem_fgstage_fb::bin_x>( member );
    }

    if( member == "binY" )
    {
        return scalarAccessor<uint32_t, &Telem_fgstage_fb::bin_y>( member );
    }

    if( member.starts_with( "total_" ) )
    {
        return counterAccessor<false>( member.substr( 6 ) );
    }

    if( member.starts_with( "delta_" ) )
    {
        return counterAccessor<true>( member.substr( 6 ) );
    }

    if( member.starts_with( "lateChange_" ) )
    {
        return metricAccessor<1>( member );
    }

    if( member.starts_with( "earlyChange_" ) )
    {
        return metricAccessor<2>( member );
    }

    if( member.starts_with( "serialDuration_" ) )
    {
        return metricAccessor<3>( member );
    }

    if( member.starts_with( "lateWaitReturnInterval_" ) )
    {
        return metricAccessor<4>( member );
    }

    if( member.starts_with( "earlyWaitReturnInterval_" ) )
    {
        return metricAccessor<5>( member );
    }

    if( member.starts_with( "lateWriteEndpointInterval_" ) )
    {
        return metricAccessor<6>( member );
    }

    if( member.starts_with( "earlyWriteEndpointInterval_" ) )
    {
        return metricAccessor<7>( member );
    }

    if( member.starts_with( "latePreviousTail_" ) )
    {
        return metricAccessor<8>( member );
    }

    if( member.starts_with( "earlyPreviousTail_" ) )
    {
        return metricAccessor<9>( member );
    }

    return metricAccessor<0>( member );
}

} // namespace logger
} // namespace MagAOX
#endif
