/** \file telem_dmtimings.hpp
 * \brief Controller-local timing statistics, coverage and collection quality.
 * \ingroup logger_types_files
 */
#ifndef logger_types_telem_dmtimings_hpp
#define logger_types_telem_dmtimings_hpp

#include <array>
#include <iomanip>
#include <limits>
#include <sstream>
#include <type_traits>

#include <mx/math/floatUtils.hpp>

#include "../../common/dmTiming.hpp"
#include "generated/telem_dmtimings_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{

/// Coherent controller statistics; SDK return is not physical DM completion.
/** \ingroup logger_types */
struct telem_dmtimings : public flatbuffer_log
{
    static constexpr flatlogs::eventCodeT eventCode    = eventCodes::TELEM_DMTIMINGS;  ///< Logger dispatch code.
    static constexpr flatlogs::logPrioT   defaultLevel = flatlogs::logPrio::LOG_TELEM; ///< Telemetry priority.
    static inline timespec                lastRecord{}; ///< Last emission time, maintained by the telemeter.

    /// Serialized snapshot of the consumer-owned timing summary.
    struct messageT : public fbMessage
    {
        /// Serialize coverage, every metric row and both quality-counter populations.
        messageT( const dmTiming::Summary &summary /**< [in] coherent controller-local statistics */ );
    };

    /// Verify the FlatBuffer and its complete stable metric-ID contract.
    static bool verify( flatlogs::bufferPtrT &buffer /**< [in] complete log entry */,
                        flatlogs::msgLenT     len /**< [in] serialized message length */ );

    /// Format validity, coverage, metric populations and collection quality in seconds.
    static std::string msgString( void             *buffer /**< [in] serialized message */,
                                  flatlogs::msgLenT len /**< [in] serialized message length, unused */ );

    /// Return snapshot metadata; invalid metric durations project as NaN, never measured zero.
    static logMetaDetail getAccessor( const std::string &member /**< [in] scalar or metric_field name */ );

  private:
    static constexpr size_t metricCount = size_t( dmTiming::Metric::count ); ///< Number of stable metric identifiers.

    /// Serialize one quality-counter population without changing its collection interval.
    static flatbuffers::Offset<DmTimingCounters>
    counters( flatbuffers::FlatBufferBuilder &builder /**< [in/out] destination builder */,
              const dmTiming::Counters       &values /**< [in] cumulative or since-summary counts */ );

    /// Extract a typed summary scalar for logMeta.
    template <typename valueT, auto member>
    static valueT scalar( void *buffer /**< [in] serialized message */ );

    /// Find one stable metric ID without relying on vector order.
    static const DmTimingStats *metricRow( void  *buffer /**< [in] serialized message */,
                                           size_t index /**< [in] stable metric identifier */ );

    /// Extract one metric field, preserving count/validity when durations are unavailable.
    template <size_t index, typename valueT, auto member>
    static valueT metric( void *buffer /**< [in] serialized message */ );

    /// Extract one cumulative or since-summary quality counter.
    template <bool delta, auto member>
    static uint64_t counter( void *buffer /**< [in] serialized message */ );

    /// Describe a typed snapshot scalar for logMeta without interpolating across windows.
    template <typename valueT, auto member>
    static logMetaDetail scalarAccessor( const std::string &name /**< [in] metadata name */ );

    /// Resolve a named metric and its field to a typed metadata accessor.
    template <size_t index = 0>
    static logMetaDetail metricAccessor( const std::string &member /**< [in] metric_field name */ );

    /// Resolve a cumulative or since-summary counter name to its metadata accessor.
    template <bool delta>
    static logMetaDetail counterAccessor( const std::string &member /**< [in] counter name */ );
};

inline flatbuffers::Offset<DmTimingCounters> telem_dmtimings::counters( flatbuffers::FlatBufferBuilder &builder,
                                                                        const dmTiming::Counters       &values )
{
    return CreateDmTimingCounters( builder,
                                   values.events,
                                   values.receipts,
                                   values.submitted,
                                   values.failed,
                                   values.partial,
                                   values.noSend,
                                   values.synthetic,
                                   values.rejected,
                                   values.duplicates,
                                   values.counterGaps,
                                   values.invalid,
                                   values.dropped );
}

inline telem_dmtimings::messageT::messageT( const dmTiming::Summary &summary )
{
    static_assert( size_t( dmTiming::Metric::count ) == 12 );
    builder.ForceDefaults( true ); // Preserve nonfinite input for verification even under fast-math comparisons.
    std::array<flatbuffers::Offset<DmTimingStats>, size_t( dmTiming::Metric::count )> rows;
    for( size_t index = 0; index < rows.size(); ++index )
    {
        const auto &value = summary.metrics[index];
        rows[index]       = CreateDmTimingStats( builder,
                                           DmTimingMetric( index ),
                                           value.count,
                                           value.mean,
                                           value.stddev,
                                           value.min,
                                           value.max,
                                           value.span,
                                           value.valid );
    }
    const auto metrics = builder.CreateVector( rows.data(), rows.size() );
    const auto total   = counters( builder, summary.total );
    const auto delta   = counters( builder, summary.delta );
    builder.Finish( CreateTelem_dmtimings_fb( builder,
                                              summary.session,
                                              summary.sequence,
                                              summary.nowNs,
                                              summary.newestNs,
                                              summary.windowSec,
                                              summary.span,
                                              summary.age,
                                              summary.capacity,
                                              summary.queued,
                                              summary.enabled,
                                              summary.fgValid,
                                              DmTimingAvailability( summary.state ),
                                              metrics,
                                              total,
                                              delta ) );
}

inline bool telem_dmtimings::verify( flatlogs::bufferPtrT &buffer, flatlogs::msgLenT len )
{
    if( !buffer )
        return false;
    auto                 *payload = static_cast<uint8_t *>( flatlogs::logHeader::messageBuffer( buffer ) );
    flatbuffers::Verifier verifier( payload, len );
    if( !VerifyTelem_dmtimings_fbBuffer( verifier ) )
        return false;
    const auto *record = GetTelem_dmtimings_fb( payload );
    if( record->state() < DmTimingAvailability_MIN || record->state() > DmTimingAvailability_MAX ||
        record->metrics()->size() != metricCount )
        return false;
    std::array<bool, metricCount> seen{};
    for( const auto *row : *record->metrics() )
    {
        const size_t index = size_t( row->metric() );
        if( index >= seen.size() || seen[index] )
            return false;
        seen[index] = true;
        if( row->valid() &&
            ( row->count() < 2 || !mx::math::isFinite( row->mean() ) || !mx::math::isFinite( row->stddev() ) ||
              !mx::math::isFinite( row->min() ) || !mx::math::isFinite( row->max() ) ||
              !mx::math::isFinite( row->span() ) || row->stddev() < 0 || row->min() < 0 || row->mean() < row->min() ||
              row->mean() > row->max() || row->span() < 0 ) )
            return false;
    }
    return true;
}

inline std::string telem_dmtimings::msgString( void *buffer, flatlogs::msgLenT len )
{
    static_cast<void>( len );
    const auto        *record = GetTelem_dmtimings_fb( buffer );
    std::ostringstream output;
    output << std::setprecision( 10 ) << "[dmtimes] session=" << record->session() << " sequence=" << record->sequence()
           << " state=" << EnumNameDmTimingAvailability( record->state() ) << " enabled=" << record->enabled()
           << " fg_valid=" << record->fg_valid() << " window_s=" << record->window_sec()
           << " span_s=" << record->span_sec() << " age_s=" << record->age_sec() << " queued=" << record->queued()
           << '/' << record->capacity();
    if( record->metrics() )
        for( const auto *row : *record->metrics() )
        {
            const size_t index = size_t( row->metric() );
            output << ' ' << dmTiming::metricName( dmTiming::Metric( index ) ) << "[n=" << row->count() << "]=";
            if( row->valid() )
                output << row->mean() << "+/-" << row->stddev() << " s min=" << row->min() << " max=" << row->max()
                       << " span=" << row->span();
            else
                output << "unavailable";
        }
    for( const bool delta : { false, true } )
    {
        const auto *value = delta ? record->delta() : record->total();
        if( value )
            output << ' ' << ( delta ? "delta" : "total" ) << "={events:" << value->events()
                   << ",receipts:" << value->receipts() << ",submitted:" << value->submitted()
                   << ",failed:" << value->failed() << ",partial:" << value->partial() << ",noSend:" << value->no_send()
                   << ",synthetic:" << value->synthetic() << ",rejected:" << value->rejected()
                   << ",duplicates:" << value->duplicates() << ",counterGaps:" << value->counter_gaps()
                   << ",invalid:" << value->invalid() << ",dropped:" << value->dropped() << '}';
    }
    return output.str();
}

template <typename valueT, auto member>
inline valueT telem_dmtimings::scalar( void *buffer )
{
    return valueT( ( GetTelem_dmtimings_fb( buffer )->*member )() );
}

inline const DmTimingStats *telem_dmtimings::metricRow( void *buffer, size_t index )
{
    const auto *rows = GetTelem_dmtimings_fb( buffer )->metrics();
    if( rows )
        for( const auto *row : *rows )
            if( size_t( row->metric() ) == index )
                return row;
    return nullptr;
}

template <size_t index, typename valueT, auto member>
inline valueT telem_dmtimings::metric( void *buffer )
{
    const auto *row = metricRow( buffer, index );
    if constexpr( std::is_floating_point_v<valueT> )
    {
        if( !row || !row->valid() )
            return std::numeric_limits<valueT>::quiet_NaN();
    }
    return row ? valueT( ( row->*member )() ) : valueT{};
}

template <bool delta, auto member>
inline uint64_t telem_dmtimings::counter( void *buffer )
{
    const auto *record = GetTelem_dmtimings_fb( buffer );
    const auto *values = delta ? record->delta() : record->total();
    return values ? ( values->*member )() : 0;
}

template <typename valueT, auto member>
inline logMetaDetail telem_dmtimings::scalarAccessor( const std::string &name )
{
    return { "DM " + name,
             mx::fits::fitsType<valueT>(),
             logMeta::metaTypes::State,
             reinterpret_cast<void *>( &scalar<valueT, member> ) };
}

template <size_t index>
inline logMetaDetail telem_dmtimings::metricAccessor( const std::string &member )
{
    const std::string prefix = std::string( dmTiming::metricName( dmTiming::Metric( index ) ) ) + '_';
    if( member.starts_with( prefix ) )
    {
        const std::string field    = member.substr( prefix.size() );
        void             *accessor = nullptr;
        int               type     = logMeta::valTypes::Double;
        if( field == "mean" )
            accessor = reinterpret_cast<void *>( &metric<index, double, &DmTimingStats::mean> );
        else if( field == "stddev" )
            accessor = reinterpret_cast<void *>( &metric<index, double, &DmTimingStats::stddev> );
        else if( field == "min" )
            accessor = reinterpret_cast<void *>( &metric<index, double, &DmTimingStats::min> );
        else if( field == "max" )
            accessor = reinterpret_cast<void *>( &metric<index, double, &DmTimingStats::max> );
        else if( field == "span" )
            accessor = reinterpret_cast<void *>( &metric<index, double, &DmTimingStats::span> );
        else if( field == "count" )
        {
            type     = mx::fits::fitsType<uint64_t>();
            accessor = reinterpret_cast<void *>( &metric<index, uint64_t, &DmTimingStats::count> );
        }
        else if( field == "valid" )
        {
            type     = logMeta::valTypes::Bool;
            accessor = reinterpret_cast<void *>( &metric<index, bool, &DmTimingStats::valid> );
        }
        if( accessor )
            return { "DM " + member, type, logMeta::metaTypes::State, accessor };
        return {};
    }
    if constexpr( index + 1 < metricCount )
        return metricAccessor<index + 1>( member );
    return {};
}

template <bool delta>
inline logMetaDetail telem_dmtimings::counterAccessor( const std::string &member )
{
    void *accessor = nullptr;
    if( member == "events" )
        accessor = reinterpret_cast<void *>( &counter<delta, &DmTimingCounters::events> );
    else if( member == "receipts" )
        accessor = reinterpret_cast<void *>( &counter<delta, &DmTimingCounters::receipts> );
    else if( member == "submitted" )
        accessor = reinterpret_cast<void *>( &counter<delta, &DmTimingCounters::submitted> );
    else if( member == "failed" )
        accessor = reinterpret_cast<void *>( &counter<delta, &DmTimingCounters::failed> );
    else if( member == "partial" )
        accessor = reinterpret_cast<void *>( &counter<delta, &DmTimingCounters::partial> );
    else if( member == "noSend" )
        accessor = reinterpret_cast<void *>( &counter<delta, &DmTimingCounters::no_send> );
    else if( member == "synthetic" )
        accessor = reinterpret_cast<void *>( &counter<delta, &DmTimingCounters::synthetic> );
    else if( member == "rejected" )
        accessor = reinterpret_cast<void *>( &counter<delta, &DmTimingCounters::rejected> );
    else if( member == "duplicates" )
        accessor = reinterpret_cast<void *>( &counter<delta, &DmTimingCounters::duplicates> );
    else if( member == "counterGaps" )
        accessor = reinterpret_cast<void *>( &counter<delta, &DmTimingCounters::counter_gaps> );
    else if( member == "invalid" )
        accessor = reinterpret_cast<void *>( &counter<delta, &DmTimingCounters::invalid> );
    else if( member == "dropped" )
        accessor = reinterpret_cast<void *>( &counter<delta, &DmTimingCounters::dropped> );
    if( !accessor )
        return {};
    return { std::string( delta ? "DM delta_" : "DM total_" ) + member,
             mx::fits::fitsType<uint64_t>(),
             logMeta::metaTypes::State,
             accessor };
}

inline logMetaDetail telem_dmtimings::getAccessor( const std::string &member )
{
    if( member == "session" )
        return scalarAccessor<uint64_t, &Telem_dmtimings_fb::session>( member );
    if( member == "sequence" )
        return scalarAccessor<uint64_t, &Telem_dmtimings_fb::sequence>( member );
    if( member == "nowNs" )
        return scalarAccessor<int64_t, &Telem_dmtimings_fb::now_ns>( member );
    if( member == "newestNs" )
        return scalarAccessor<int64_t, &Telem_dmtimings_fb::newest_ns>( member );
    if( member == "windowSec" )
        return scalarAccessor<double, &Telem_dmtimings_fb::window_sec>( member );
    if( member == "span" )
        return scalarAccessor<double, &Telem_dmtimings_fb::span_sec>( member );
    if( member == "age" )
        return scalarAccessor<double, &Telem_dmtimings_fb::age_sec>( member );
    if( member == "capacity" )
        return scalarAccessor<uint64_t, &Telem_dmtimings_fb::capacity>( member );
    if( member == "queued" )
        return scalarAccessor<uint64_t, &Telem_dmtimings_fb::queued>( member );
    if( member == "enabled" )
        return scalarAccessor<bool, &Telem_dmtimings_fb::enabled>( member );
    if( member == "fgValid" )
        return scalarAccessor<bool, &Telem_dmtimings_fb::fg_valid>( member );
    if( member == "state" )
        return scalarAccessor<int, &Telem_dmtimings_fb::state>( member );
    if( member.starts_with( "total_" ) )
        return counterAccessor<false>( member.substr( 6 ) );
    if( member.starts_with( "delta_" ) )
    {
        const auto detail = counterAccessor<true>( member.substr( 6 ) );
        if( detail.accessor )
            return detail;
    }
    return metricAccessor( member );
}

} // namespace logger
} // namespace MagAOX

#endif
