/** \file telem_fgtrace.hpp
 * \brief Chronological camera excursion records for correlation with monotonic scheduler and IRQ traces.
 * \ingroup logger_types_files
 */
#ifndef logger_types_telem_fgtrace_hpp
#define logger_types_telem_fgtrace_hpp

#include <array>
#include <sstream>

#include "../../common/frameGrabberTiming.hpp"
#include "generated/telem_fgtrace_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{

/// Five camera attempts around an excursion, with exact clock domains and diagnostic-loss context.
/** \ingroup logger_types */
struct telem_fgtrace : public flatbuffer_log
{
    static constexpr flatlogs::eventCodeT eventCode    = eventCodes::TELEM_FGTRACE;    ///< Stable dispatch identifier.
    static constexpr flatlogs::logPrioT   defaultLevel = flatlogs::logPrio::LOG_TELEM; ///< Telemetry priority.
    static inline timespec                lastRecord{}; ///< Ordinary telemeter's last emission time.

    /// Serialized ordinary-thread excursion neighborhood; no acquisition-thread logging is required.
    struct messageT : public fbMessage
    {
        /// Preserve integer endpoints, neighbor presence, original stamps and cumulative losses.
        messageT( const fgTiming::TraceRecord &record /**< [in] finalized chronological neighborhood */,
                  const fgTiming::TraceStatus &status /**< [in] consumer-owned cumulative trace quality */,
                  uint64_t                     processPid /**< [in] process identifier for trace correlation */,
                  uint64_t                     acquisitionTid /**< [in] Linux acquisition-thread identifier */,
                  int64_t emittedNs /**< [in] CLOCK_MONOTONIC time sampled immediately before emission */ );
    };

    /// Reject malformed shape, unknown identifiers and impossible classification or endpoint ordering.
    static bool verify( flatlogs::bufferPtrT &buffer /**< [in] complete log entry */,
                        flatlogs::msgLenT     len /**< [in] serialized message length */ );

    /// Format every available neighbor using absolute monotonic ns and explicit original-clock stamps.
    static std::string msgString( void             *buffer /**< [in] verified serialized message */,
                                  flatlogs::msgLenT len /**< [in] message length, unused */ );

    /// Resolve typed trace, quality, neighbor and marker metadata without narrowing integer timestamps.
    static logMetaDetail getAccessor( const std::string &member /**< [in] scalar, quality_ or frameN_ field */ );

  private:
    static constexpr size_t neighborCount = 5; ///< Fixed chronological slots, with the trigger at slot two.
    static constexpr size_t markerCount   = size_t( fgTiming::Marker::count ); ///< Stable local endpoint count.

    /// Serialize one neighbor, preserving slot identity even when its presence bit is clear.
    static flatbuffers::Offset<FgTraceEvent>
    event( flatbuffers::FlatBufferBuilder &builder /**< [in/out] destination builder */,
           const fgTiming::Event          &value /**< [in] unmodified producer/consumer event */,
           uint8_t                         slot /**< [in] fixed chronological slot */ );

    /// Resolve one stable local endpoint name.
    static const char *markerName( size_t index /**< [in] local endpoint array position */ );

    /// Extract one top-level typed trace scalar.
    template <typename valueT, auto member>
    static valueT scalar( void *buffer /**< [in] serialized message */ );

    /// Extract one cumulative trace-quality scalar.
    template <typename valueT, auto member>
    static valueT quality( void *buffer /**< [in] serialized message */ );

    /// Extract one neighbor scalar using fixed identity rather than vector order.
    template <size_t slot, typename valueT, auto member>
    static valueT frame( void *buffer /**< [in] serialized message */ );

    /// Extract one marker endpoint, returning -1 for an unavailable neighbor or marker.
    template <size_t slot, size_t marker>
    static int64_t endpoint( void *buffer /**< [in] serialized message */ );

    /// Test whether a chronological neighbor was captured.
    template <size_t slot>
    static bool present( void *buffer /**< [in] serialized message */ );

    /// Extract one completed-controls-only overlap, returning -1 when the neighbor is absent.
    template <size_t slot>
    static int64_t overlap( void *buffer /**< [in] serialized message */ );

    /// Describe one exact typed metadata getter.
    template <typename valueT, auto getter>
    static logMetaDetail accessor( const std::string &name /**< [in] full metadata label */ );

    /// Resolve one neighbor scalar or recurse to the remaining slots.
    template <size_t slot = 0>
    static logMetaDetail frameAccessor( const std::string &name /**< [in] complete metadata label */ );

    /// Resolve one local endpoint within a known chronological neighbor.
    template <size_t slot, size_t marker = 0>
    static logMetaDetail markerAccessor( const std::string &name /**< [in] suffix after frameN_ */ );
};

inline flatbuffers::Offset<FgTraceEvent>
telem_fgtrace::event( flatbuffers::FlatBufferBuilder &builder, const fgTiming::Event &value, uint8_t slot )
{
    const auto markers = builder.CreateVector( value.markers.data(), value.markers.size() );
    return CreateFgTraceEvent( builder,
                               slot,
                               value.sequence,
                               value.session,
                               value.frameNumber,
                               value.outputCounter,
                               value.streamInode,
                               value.mode,
                               value.binX,
                               value.binY,
                               value.frameValid,
                               value.outputValid,
                               value.atime.tv_sec,
                               value.atime.tv_nsec,
                               value.writetime.tv_sec,
                               value.writetime.tv_nsec,
                               markers,
                               value.valid,
                               value.invalidMarkers,
                               FgTraceOutcome( value.outcome ),
                               value.anchorNs );
}

inline telem_fgtrace::messageT::messageT( const fgTiming::TraceRecord &record,
                                          const fgTiming::TraceStatus &status,
                                          uint64_t                     processPid,
                                          uint64_t                     acquisitionTid,
                                          int64_t                      emittedNs )
{
    static_assert( size_t( fgTiming::TraceDirection::late ) == size_t( FgTraceDirection_late ) );
    static_assert( size_t( fgTiming::TraceDirection::early ) == size_t( FgTraceDirection_early ) );
    static_assert( size_t( fgTiming::TraceEnd::complete ) == size_t( FgTraceEnd_complete ) );
    static_assert( size_t( fgTiming::TraceEnd::boundary ) == size_t( FgTraceEnd_boundary ) );
    static_assert( size_t( fgTiming::TraceEnd::expired ) == size_t( FgTraceEnd_expired ) );
    static_assert( size_t( fgTiming::Outcome::published ) == size_t( FgTraceOutcome_published ) );
    static_assert( size_t( fgTiming::Outcome::rejected ) == size_t( FgTraceOutcome_rejected ) );
    static_assert( size_t( fgTiming::Outcome::failed ) == size_t( FgTraceOutcome_failed ) );
    static_assert( markerCount == 9 );
    static_assert( record.events.size() == neighborCount );
    builder.ForceDefaults( true );
    std::array<flatbuffers::Offset<FgTraceEvent>, neighborCount> neighbors;

    for( size_t index = 0; index < neighborCount; ++index )
    {
        neighbors[index] = event( builder, record.events[index], index );
    }

    const auto events   = builder.CreateVector( neighbors.data(), neighbors.size() );
    const auto overlaps = builder.CreateVector( record.serialOverlapNs.data(), record.serialOverlapNs.size() );
    const auto quality  = CreateFgTraceStatus( builder,
                                              status.enabled,
                                              status.capacity,
                                              status.ready,
                                              status.pending,
                                              status.detected,
                                              status.recorded,
                                              status.dropped,
                                              status.incomplete,
                                              status.unclassified,
                                              status.discontinuities,
                                              status.eventDropped,
                                              status.controlDropped );
    builder.Finish( CreateTelem_fgtrace_fb( builder,
                                            record.id,
                                            record.present,
                                            record.qualified,
                                            FgTraceDirection( record.direction ),
                                            FgTraceEnd( record.end ),
                                            record.medianNs,
                                            record.intervalNs,
                                            record.thresholdNs,
                                            record.bracketLimitNs,
                                            record.evaluatedNs,
                                            record.controlCoverageNs,
                                            events,
                                            overlaps,
                                            quality,
                                            processPid,
                                            acquisitionTid,
                                            emittedNs,
                                            record.baselineIntervals ) );
}

inline bool telem_fgtrace::verify( flatlogs::bufferPtrT &buffer, flatlogs::msgLenT len )
{
    if( !buffer )
    {
        return false;
    }

    auto                 *payload = static_cast<uint8_t *>( flatlogs::logHeader::messageBuffer( buffer ) );
    flatbuffers::Verifier verifier( payload, len );

    if( !VerifyTelem_fgtrace_fbBuffer( verifier ) )
    {
        return false;
    }

    const auto        *record     = GetTelem_fgtrace_fb( payload );
    const auto        *status     = record->status();
    constexpr uint32_t allMarkers = ( uint32_t( 1 ) << markerCount ) - 1;

    if( record->events()->size() != neighborCount || record->serial_overlap_ns()->size() != neighborCount ||
        record->present() > 31 || ( record->present() & 6 ) != 6 || record->median_ns() <= 0 ||
        record->interval_ns() <= 0 || record->threshold_ns() <= 0 || record->baseline_intervals() < 3 ||
        record->bracket_limit_ns() < 0 || record->evaluated_ns() < 0 || record->control_coverage_ns() < 0 ||
        record->control_coverage_ns() > record->evaluated_ns() || record->emitted_ns() < record->evaluated_ns() ||
        !record->process_pid() || !record->acquisition_tid() || record->direction() < FgTraceDirection_MIN ||
        record->direction() > FgTraceDirection_MAX || record->end() < FgTraceEnd_MIN ||
        record->end() > FgTraceEnd_MAX || status->ready() > status->capacity() ||
        status->pending() > status->capacity() - status->ready() ||
        ( record->qualified() && ( record->bracket_limit_ns() == 0 || !( record->present() & 2 ) ) ) )
    {
        return false;
    }

    const bool late = record->interval_ns() > record->median_ns() &&
                      record->interval_ns() - record->median_ns() > record->threshold_ns();
    const bool early = record->median_ns() > record->interval_ns() &&
                       record->median_ns() - record->interval_ns() > record->threshold_ns();

    if( record->direction() == FgTraceDirection_late ? !late : !early )
    {
        return false;
    }

    std::array<bool, neighborCount>                 seen{};
    std::array<const FgTraceEvent *, neighborCount> neighbors{};

    for( const auto *value : *record->events() )
    {
        if( !value || value->slot() >= neighborCount || seen[value->slot()] ||
            value->markers_ns()->size() != markerCount || value->valid() > allMarkers ||
            value->outcome() < FgTraceOutcome_MIN || value->outcome() > FgTraceOutcome_MAX )
        {
            return false;
        }

        const size_t slot = value->slot();
        seen[slot]        = true;
        neighbors[slot]   = value;

        if( record->serial_overlap_ns()->Get( slot ) < -1 )
        {
            return false;
        }

        if( !( record->present() & ( 1u << slot ) ) )
        {
            continue;
        }

        int64_t previous = -1;

        for( size_t index = 0; index < markerCount; ++index )
        {
            if( value->valid() & ( 1u << index ) )
            {
                const int64_t marker = value->markers_ns()->Get( index );

                if( marker < 0 || marker < previous || marker > record->evaluated_ns() )
                {
                    return false;
                }

                previous = marker;
            }
        }
    }

    const auto      *previous  = neighbors[1];
    const auto      *trigger   = neighbors[2];
    constexpr size_t write     = size_t( fgTiming::Marker::writeStampDone );
    constexpr size_t load      = size_t( fgTiming::Marker::loadDone );
    const bool       frameStep = !previous->frame_valid() || !trigger->frame_valid() ||
                           trigger->frame_number() == previous->frame_number() + 1 ||
                           ( previous->frame_number() == UINT32_MAX && trigger->frame_number() == 0 );

    if( previous->outcome() != FgTraceOutcome_published || trigger->outcome() != FgTraceOutcome_published ||
        !previous->output_valid() || !trigger->output_valid() || previous->session() != trigger->session() ||
        trigger->sequence() != previous->sequence() + 1 || trigger->output_counter() <= previous->output_counter() ||
        trigger->output_counter() != previous->output_counter() + 1 ||
        trigger->stream_inode() != previous->stream_inode() || !frameStep || !( previous->valid() & ( 1u << write ) ) ||
        !( trigger->valid() & ( 1u << write ) ) ||
        trigger->markers_ns()->Get( write ) <= previous->markers_ns()->Get( write ) ||
        trigger->markers_ns()->Get( write ) - previous->markers_ns()->Get( write ) != record->interval_ns() )
    {
        return false;
    }

    if( record->qualified() )
    {
        for( const auto *value : { previous, trigger } )
        {
            const bool validWrite = value->writetime_sec() >= 0 && value->writetime_nsec() >= 0 &&
                                    value->writetime_nsec() < 1000000000 &&
                                    ( value->writetime_sec() > 0 || value->writetime_nsec() > 0 );

            if( value->invalid_markers() || !validWrite || !( value->valid() & ( 1u << load ) ) ||
                value->markers_ns()->Get( write ) - value->markers_ns()->Get( load ) > record->bracket_limit_ns() )
            {
                return false;
            }
        }
    }

    return true;
}

inline const char *telem_fgtrace::markerName( size_t index )
{
    constexpr const char *names[] = { "waitEnterNs",
                                      "waitReturnNs",
                                      "rearmReturnNs",
                                      "validatedNs",
                                      "loadDoneNs",
                                      "writeStampDoneNs",
                                      "mainPostDoneNs",
                                      "syncPostDoneNs",
                                      "loopTailDoneNs" };
    return index < markerCount ? names[index] : "unknown";
}

inline std::string telem_fgtrace::msgString( void *buffer, flatlogs::msgLenT len )
{
    static_cast<void>( len );
    const auto        *record = GetTelem_fgtrace_fb( buffer );
    const auto        *status = record->status();
    std::ostringstream output;
    output << "[fgtrace] id=" << record->id() << " pid=" << record->process_pid()
           << " tid=" << record->acquisition_tid() << " direction=" << EnumNameFgTraceDirection( record->direction() )
           << " qualified=" << record->qualified() << " end=" << EnumNameFgTraceEnd( record->end() )
           << " present=" << unsigned( record->present() ) << " median_ns=" << record->median_ns()
           << " baseline_intervals=" << record->baseline_intervals() << " interval_ns=" << record->interval_ns()
           << " threshold_ns=" << record->threshold_ns() << " bracket_limit_ns=" << record->bracket_limit_ns()
           << " evaluated_ns=" << record->evaluated_ns() << " emitted_ns=" << record->emitted_ns()
           << " control_coverage_ns=" << record->control_coverage_ns();

    for( const auto *value : *record->events() )
    {
        const unsigned slot = value->slot();
        output << " frame" << slot << "={present:" << bool( record->present() & ( 1u << slot ) );

        if( record->present() & ( 1u << slot ) )
        {
            output << ",sequence:" << value->sequence() << ",session:" << value->session()
                   << ",frame:" << value->frame_number() << ",counter:" << value->output_counter()
                   << ",inode:" << value->stream_inode() << ",mode:" << value->mode() << ",binX:" << value->bin_x()
                   << ",binY:" << value->bin_y() << ",frameValid:" << value->frame_valid()
                   << ",outputValid:" << value->output_valid()
                   << ",outcome:" << EnumNameFgTraceOutcome( value->outcome() ) << ",atime_sec:" << value->atime_sec()
                   << ",atime_nsec:" << value->atime_nsec() << ",writetime_sec:" << value->writetime_sec()
                   << ",writetime_nsec:" << value->writetime_nsec() << ",valid:" << value->valid()
                   << ",invalidMarkers:" << value->invalid_markers() << ",anchorNs:" << value->anchor_ns();

            for( size_t marker = 0; marker < markerCount; ++marker )
            {
                output << ',' << markerName( marker ) << ':';

                if( value->valid() & ( 1u << marker ) )
                {
                    output << value->markers_ns()->Get( marker );
                }
                else
                {
                    output << "unavailable";
                }
            }

            output << ",serialOverlapNs:" << record->serial_overlap_ns()->Get( slot );
        }

        output << '}';
    }

    output << " quality={enabled:" << status->enabled() << ",capacity:" << status->capacity()
           << ",ready:" << status->ready() << ",pending:" << status->pending() << ",detected:" << status->detected()
           << ",recorded:" << status->recorded() << ",dropped:" << status->dropped()
           << ",incomplete:" << status->incomplete() << ",unclassified:" << status->unclassified()
           << ",discontinuities:" << status->discontinuities() << ",eventDropped:" << status->event_dropped()
           << ",controlDropped:" << status->control_dropped() << '}';
    return output.str();
}

template <typename valueT, auto member>
inline valueT telem_fgtrace::scalar( void *buffer )
{
    return valueT( ( GetTelem_fgtrace_fb( buffer )->*member )() );
}

template <typename valueT, auto member>
inline valueT telem_fgtrace::quality( void *buffer )
{
    return valueT( ( GetTelem_fgtrace_fb( buffer )->status()->*member )() );
}

template <size_t slot, typename valueT, auto member>
inline valueT telem_fgtrace::frame( void *buffer )
{
    const auto *record = GetTelem_fgtrace_fb( buffer );

    for( const auto *value : *record->events() )
    {
        if( value->slot() == slot && ( record->present() & ( 1u << slot ) ) )
        {
            return valueT( ( value->*member )() );
        }
    }

    return valueT{};
}

template <size_t slot, size_t marker>
inline int64_t telem_fgtrace::endpoint( void *buffer )
{
    const auto *record = GetTelem_fgtrace_fb( buffer );

    for( const auto *value : *record->events() )
    {
        if( value->slot() == slot && ( record->present() & ( 1u << slot ) ) && ( value->valid() & ( 1u << marker ) ) )
        {
            return value->markers_ns()->Get( marker );
        }
    }

    return -1;
}

template <size_t slot>
inline int64_t telem_fgtrace::overlap( void *buffer )
{
    const auto *record = GetTelem_fgtrace_fb( buffer );
    return record->present() & ( 1u << slot ) ? record->serial_overlap_ns()->Get( slot ) : -1;
}

template <typename valueT, auto getter>
inline logMetaDetail telem_fgtrace::accessor( const std::string &name )
{
    return { "FG trace " + name,
             mx::fits::fitsType<valueT>(),
             logMeta::metaTypes::State,
             reinterpret_cast<void *>( getter ) };
}

template <size_t slot, size_t marker>
inline logMetaDetail telem_fgtrace::markerAccessor( const std::string &name )
{
    if( name == markerName( marker ) )
    {
        return accessor<int64_t, &endpoint<slot, marker>>( "frame" + std::to_string( slot ) + '_' + name );
    }

    if constexpr( marker + 1 < markerCount )
    {
        return markerAccessor<slot, marker + 1>( name );
    }

    return {};
}

template <size_t slot>
inline bool telem_fgtrace::present( void *buffer )
{
    return GetTelem_fgtrace_fb( buffer )->present() & ( 1u << slot );
}

template <size_t slot>
inline logMetaDetail telem_fgtrace::frameAccessor( const std::string &name )
{
    const std::string prefix = "frame" + std::to_string( slot ) + '_';

    if( name.starts_with( prefix ) )
    {
        const auto field = name.substr( prefix.size() );

        if( field == "present" )
        {
            return accessor<bool, &present<slot>>( name );
        }

        if( field == "serialOverlapNs" )
        {
            return accessor<int64_t, &overlap<slot>>( name );
        }

        if( field == "sequence" )
        {
            return accessor<uint64_t, &frame<slot, uint64_t, &FgTraceEvent::sequence>>( name );
        }

        if( field == "session" )
        {
            return accessor<uint64_t, &frame<slot, uint64_t, &FgTraceEvent::session>>( name );
        }

        if( field == "frameNumber" )
        {
            return accessor<uint64_t, &frame<slot, uint64_t, &FgTraceEvent::frame_number>>( name );
        }

        if( field == "outputCounter" )
        {
            return accessor<uint64_t, &frame<slot, uint64_t, &FgTraceEvent::output_counter>>( name );
        }

        if( field == "streamInode" )
        {
            return accessor<uint64_t, &frame<slot, uint64_t, &FgTraceEvent::stream_inode>>( name );
        }

        if( field == "mode" )
        {
            return accessor<uint32_t, &frame<slot, uint32_t, &FgTraceEvent::mode>>( name );
        }

        if( field == "binX" )
        {
            return accessor<uint32_t, &frame<slot, uint32_t, &FgTraceEvent::bin_x>>( name );
        }

        if( field == "binY" )
        {
            return accessor<uint32_t, &frame<slot, uint32_t, &FgTraceEvent::bin_y>>( name );
        }

        if( field == "frameValid" )
        {
            return accessor<bool, &frame<slot, bool, &FgTraceEvent::frame_valid>>( name );
        }

        if( field == "outputValid" )
        {
            return accessor<bool, &frame<slot, bool, &FgTraceEvent::output_valid>>( name );
        }

        if( field == "atimeSec" )
        {
            return accessor<int64_t, &frame<slot, int64_t, &FgTraceEvent::atime_sec>>( name );
        }

        if( field == "atimeNsec" )
        {
            return accessor<int64_t, &frame<slot, int64_t, &FgTraceEvent::atime_nsec>>( name );
        }

        if( field == "writetimeSec" )
        {
            return accessor<int64_t, &frame<slot, int64_t, &FgTraceEvent::writetime_sec>>( name );
        }

        if( field == "writetimeNsec" )
        {
            return accessor<int64_t, &frame<slot, int64_t, &FgTraceEvent::writetime_nsec>>( name );
        }

        if( field == "valid" )
        {
            return accessor<uint32_t, &frame<slot, uint32_t, &FgTraceEvent::valid>>( name );
        }

        if( field == "invalidMarkers" )
        {
            return accessor<uint32_t, &frame<slot, uint32_t, &FgTraceEvent::invalid_markers>>( name );
        }

        if( field == "outcome" )
        {
            return accessor<int, &frame<slot, int, &FgTraceEvent::outcome>>( name );
        }

        if( field == "anchorNs" )
        {
            return accessor<int64_t, &frame<slot, int64_t, &FgTraceEvent::anchor_ns>>( name );
        }

        return markerAccessor<slot>( field );
    }

    if constexpr( slot + 1 < neighborCount )
    {
        return frameAccessor<slot + 1>( name );
    }

    return {};
}

inline logMetaDetail telem_fgtrace::getAccessor( const std::string &member )
{
    if( member == "id" )
    {
        return accessor<uint64_t, &scalar<uint64_t, &Telem_fgtrace_fb::id>>( member );
    }

    if( member == "present" )
    {
        return accessor<uint32_t, &scalar<uint32_t, &Telem_fgtrace_fb::present>>( member );
    }

    if( member == "qualified" )
    {
        return accessor<bool, &scalar<bool, &Telem_fgtrace_fb::qualified>>( member );
    }

    if( member == "direction" )
    {
        return accessor<int, &scalar<int, &Telem_fgtrace_fb::direction>>( member );
    }

    if( member == "end" )
    {
        return accessor<int, &scalar<int, &Telem_fgtrace_fb::end>>( member );
    }

    if( member == "medianNs" )
    {
        return accessor<int64_t, &scalar<int64_t, &Telem_fgtrace_fb::median_ns>>( member );
    }

    if( member == "baselineIntervals" )
    {
        return accessor<uint64_t, &scalar<uint64_t, &Telem_fgtrace_fb::baseline_intervals>>( member );
    }

    if( member == "intervalNs" )
    {
        return accessor<int64_t, &scalar<int64_t, &Telem_fgtrace_fb::interval_ns>>( member );
    }

    if( member == "thresholdNs" )
    {
        return accessor<int64_t, &scalar<int64_t, &Telem_fgtrace_fb::threshold_ns>>( member );
    }

    if( member == "bracketLimitNs" )
    {
        return accessor<int64_t, &scalar<int64_t, &Telem_fgtrace_fb::bracket_limit_ns>>( member );
    }

    if( member == "evaluatedNs" )
    {
        return accessor<int64_t, &scalar<int64_t, &Telem_fgtrace_fb::evaluated_ns>>( member );
    }

    if( member == "controlCoverageNs" )
    {
        return accessor<int64_t, &scalar<int64_t, &Telem_fgtrace_fb::control_coverage_ns>>( member );
    }

    if( member == "processPid" )
    {
        return accessor<uint64_t, &scalar<uint64_t, &Telem_fgtrace_fb::process_pid>>( member );
    }

    if( member == "acquisitionTid" )
    {
        return accessor<uint64_t, &scalar<uint64_t, &Telem_fgtrace_fb::acquisition_tid>>( member );
    }

    if( member == "emittedNs" )
    {
        return accessor<int64_t, &scalar<int64_t, &Telem_fgtrace_fb::emitted_ns>>( member );
    }

    if( member == "quality_enabled" )
    {
        return accessor<bool, &quality<bool, &FgTraceStatus::enabled>>( member );
    }

    if( member == "quality_capacity" )
    {
        return accessor<uint64_t, &quality<uint64_t, &FgTraceStatus::capacity>>( member );
    }

    if( member == "quality_ready" )
    {
        return accessor<uint64_t, &quality<uint64_t, &FgTraceStatus::ready>>( member );
    }

    if( member == "quality_pending" )
    {
        return accessor<uint64_t, &quality<uint64_t, &FgTraceStatus::pending>>( member );
    }

    if( member == "quality_detected" )
    {
        return accessor<uint64_t, &quality<uint64_t, &FgTraceStatus::detected>>( member );
    }

    if( member == "quality_recorded" )
    {
        return accessor<uint64_t, &quality<uint64_t, &FgTraceStatus::recorded>>( member );
    }

    if( member == "quality_dropped" )
    {
        return accessor<uint64_t, &quality<uint64_t, &FgTraceStatus::dropped>>( member );
    }

    if( member == "quality_incomplete" )
    {
        return accessor<uint64_t, &quality<uint64_t, &FgTraceStatus::incomplete>>( member );
    }

    if( member == "quality_unclassified" )
    {
        return accessor<uint64_t, &quality<uint64_t, &FgTraceStatus::unclassified>>( member );
    }

    if( member == "quality_discontinuities" )
    {
        return accessor<uint64_t, &quality<uint64_t, &FgTraceStatus::discontinuities>>( member );
    }

    if( member == "quality_eventDropped" )
    {
        return accessor<uint64_t, &quality<uint64_t, &FgTraceStatus::event_dropped>>( member );
    }

    if( member == "quality_controlDropped" )
    {
        return accessor<uint64_t, &quality<uint64_t, &FgTraceStatus::control_dropped>>( member );
    }

    return frameAccessor( member );
}

} // namespace logger
} // namespace MagAOX

#endif // logger_types_telem_fgtrace_hpp
