/** \file frameGrabberTraceTelemetry_test.cpp
 * \brief Exact chronological camera trace serialization, typed metadata and malformed-record guards.
 * \ingroup logger_files
 */
#include "../../../tests/testXWC.hpp"
#include "../logBinarySchemata.hpp"
#include "../generated/logMemberAccessor.hpp"

namespace libXWCTest
{
namespace loggerTest
{
/// Individual camera excursion telemetry tests.
/** \ingroup frameGrabberTraceTelemetry_unit_test */
namespace frameGrabberTraceTelemetryTest
{
using namespace MagAOX;
using namespace MagAOX::logger;

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
/// Construct an exact 40 us late event above a 20 us threshold, with integer identities beyond double precision.
fgTiming::TraceRecord sampleRecord()
{
    fgTiming::TraceRecord record;
    record.id                = ( uint64_t( 1 ) << 54 ) + 3;
    record.present           = 31;
    record.qualified         = true;
    record.direction         = fgTiming::TraceDirection::late;
    record.end               = fgTiming::TraceEnd::complete;
    record.medianNs          = 500000;
    record.baselineIntervals = 27;
    record.intervalNs        = 540000;
    record.thresholdNs       = 20000;
    record.bracketLimitNs    = 1000;
    const int64_t base       = ( int64_t( 1 ) << 54 ) + 9;
    record.evaluatedNs       = base + 3000000;
    record.controlCoverageNs = record.evaluatedNs;

    for( size_t index = 0; index < record.events.size(); ++index )
    {
        auto &event         = record.events[index];
        event.sequence      = record.id + index;
        event.session       = 17;
        event.frameNumber   = 100 + index;
        event.outputCounter = ( uint64_t( 1 ) << 54 ) + 10 + index;
        event.streamInode   = ( uint64_t( 1 ) << 54 ) + 20;
        event.mode          = 3;
        event.binX          = 2;
        event.binY          = 3;
        event.frameValid    = true;
        event.outputValid   = true;
        event.atime         = { 1700000000, long( index * 500000 + 31 ) };
        event.writetime     = { 1700000000, long( index * 500000 + 50031 ) };
        event.valid         = 511;
        event.outcome       = fgTiming::Outcome::published;

        for( size_t marker = 0; marker < event.markers.size(); ++marker )
        {
            event.markers[marker] = base + index * 500000 + marker * 1000 + ( index >= 2 ? 40000 : 0 );
        }

        event.anchorNs                = event.markers.back();
        record.serialOverlapNs[index] = index * 100;
    }

    return record;
}

/// Construct independently identifiable cumulative source and bounded trace losses.
fgTiming::TraceStatus sampleStatus()
{
    fgTiming::TraceStatus status;
    status.enabled         = true;
    status.capacity        = 32;
    status.ready           = 1;
    status.pending         = 2;
    status.detected        = 7;
    status.recorded        = 5;
    status.dropped         = 1;
    status.incomplete      = 1;
    status.unclassified    = 13;
    status.discontinuities = 3;
    status.eventDropped    = ( uint64_t( 1 ) << 54 ) + 4;
    status.controlDropped  = 5;
    return status;
}

/// Wrap a typed payload with the real logger envelope and event dispatch identity.
flatlogs::bufferPtrT envelope( const telem_fgtrace::messageT &message /**< [in] serialized excursion record */ )
{
    flatlogs::bufferPtrT buffer;
    REQUIRE( flatlogs::logHeader::createLog<telem_fgtrace>( buffer, { 1, 2 }, message, flatlogs::logPrio::LOG_TELEM ) ==
             0 );
    return buffer;
}

/// Project metadata through the generated event-code dispatcher with exact value types.
template <typename valueT>
valueT project( const std::string &name /**< [in] selected trace metadata */,
                void              *payload /**< [in] serialized excursion record */ )
{
    const auto detail = logMemberAccessor( telem_fgtrace::eventCode, name );
    REQUIRE( detail.accessor != nullptr );
    REQUIRE( detail.valType == mx::fits::fitsType<valueT>() );
    REQUIRE( detail.metaType == logMeta::metaTypes::State );
    return reinterpret_cast<valueT ( * )( void * )>( detail.accessor )( payload );
}
/// \endcond

/// Preserve all raw neighbor identities, integer clocks and original external timestamps through binary and JSON paths.
/** \ingroup frameGrabberTraceTelemetry_unit_test
 * \see MagAOX::logger::telem_fgtrace::messageT
 * \see MagAOX::logger::telem_fgtrace::verify
 * \see MagAOX::logger::telem_fgtrace::msgString
 */
TEST_CASE( "Camera excursion records preserve chronological clocks and losses", "[frameGrabberTraceTelemetry]" )
{
    const auto              source = sampleRecord();
    const auto              status = sampleStatus();
    telem_fgtrace::messageT message( source, status, 123, 456, source.evaluatedNs + 7000 );
    auto                    buffer = envelope( message );
    REQUIRE( verifyLogEntry( telem_fgtrace::eventCode, buffer.get() ) );
    REQUIRE( eventCodeName( telem_fgtrace::eventCode ) == "telem_fgtrace" );
    auto       *payload = message.builder.GetBufferPointer();
    const auto *record  = GetTelem_fgtrace_fb( payload );
    REQUIRE( record->id() == source.id );
    REQUIRE( record->present() == 31 );
    REQUIRE( record->qualified() );
    REQUIRE( record->direction() == FgTraceDirection_late );
    REQUIRE( record->end() == FgTraceEnd_complete );
    REQUIRE( record->median_ns() == 500000 );
    REQUIRE( record->baseline_intervals() == 27 );
    REQUIRE( record->interval_ns() == 540000 );
    REQUIRE( record->threshold_ns() == 20000 );
    REQUIRE( record->bracket_limit_ns() == 1000 );
    REQUIRE( record->evaluated_ns() == source.evaluatedNs );
    REQUIRE( record->control_coverage_ns() == source.controlCoverageNs );
    REQUIRE( record->emitted_ns() == source.evaluatedNs + 7000 );
    REQUIRE( record->process_pid() == 123 );
    REQUIRE( record->acquisition_tid() == 456 );
    REQUIRE( record->events()->size() == 5 );
    REQUIRE( record->serial_overlap_ns()->size() == 5 );

    for( size_t slot = 0; slot < 5; ++slot )
    {
        const auto *event    = record->events()->Get( slot );
        const auto &expected = source.events[slot];
        REQUIRE( event->slot() == slot );
        REQUIRE( event->sequence() == expected.sequence );
        REQUIRE( event->session() == expected.session );
        REQUIRE( event->frame_number() == expected.frameNumber );
        REQUIRE( event->output_counter() == expected.outputCounter );
        REQUIRE( event->stream_inode() == expected.streamInode );
        REQUIRE( event->mode() == expected.mode );
        REQUIRE( event->bin_x() == 2 );
        REQUIRE( event->bin_y() == 3 );
        REQUIRE( event->frame_valid() );
        REQUIRE( event->output_valid() );
        REQUIRE( event->atime_sec() == expected.atime.tv_sec );
        REQUIRE( event->atime_nsec() == expected.atime.tv_nsec );
        REQUIRE( event->writetime_sec() == expected.writetime.tv_sec );
        REQUIRE( event->writetime_nsec() == expected.writetime.tv_nsec );
        REQUIRE( event->valid() == 511 );
        REQUIRE( event->invalid_markers() == 0 );
        REQUIRE( event->outcome() == FgTraceOutcome_published );
        REQUIRE( event->anchor_ns() == expected.anchorNs );
        REQUIRE( record->serial_overlap_ns()->Get( slot ) == source.serialOverlapNs[slot] );

        for( size_t marker = 0; marker < expected.markers.size(); ++marker )
        {
            REQUIRE( event->markers_ns()->Get( marker ) == expected.markers[marker] );
        }
    }

    REQUIRE( record->status()->enabled() );
    REQUIRE( record->status()->capacity() == 32 );
    REQUIRE( record->status()->ready() == 1 );
    REQUIRE( record->status()->pending() == 2 );
    REQUIRE( record->status()->detected() == 7 );
    REQUIRE( record->status()->recorded() == 5 );
    REQUIRE( record->status()->dropped() == 1 );
    REQUIRE( record->status()->incomplete() == 1 );
    REQUIRE( record->status()->unclassified() == 13 );
    REQUIRE( record->status()->discontinuities() == 3 );
    REQUIRE( record->status()->event_dropped() == status.eventDropped );
    REQUIRE( record->status()->control_dropped() == 5 );
    const auto text = telem_fgtrace::msgString( payload, message.builder.GetSize() );
    REQUIRE( text.starts_with( "[fgtrace] id=18014398509481987 pid=123 tid=456 direction=late" ) );
    REQUIRE( text.find( "baseline_intervals=27 interval_ns=540000" ) != std::string::npos );
    REQUIRE( text.find( "writeStampDoneNs:18014398510526993" ) != std::string::npos );
    REQUIRE( text.find( "eventDropped:18014398509481988" ) != std::string::npos );
    const auto          json = telem_fgtrace::msgJSON( payload,
                                              message.builder.GetSize(),
                                              reinterpret_cast<const uint8_t *>( telem_fgtrace_bfbs ),
                                              telem_fgtrace_bfbs_len );
    flatbuffers::Parser parser;
    REQUIRE( parser.Deserialize( reinterpret_cast<const uint8_t *>( telem_fgtrace_bfbs ), telem_fgtrace_bfbs_len ) );
    REQUIRE( parser.Parse( json.c_str() ) );
    const auto *decoded = GetTelem_fgtrace_fb( parser.builder_.GetBufferPointer() );
    REQUIRE( decoded->events()->Get( 2 )->output_counter() == source.events[2].outputCounter );
    REQUIRE( decoded->events()->Get( 2 )->markers_ns()->Get( 5 ) == source.events[2].markers[5] );
    REQUIRE( decoded->status()->event_dropped() == status.eventDropped );
}

/// Expose exact metadata for correlation while distinguishing absent neighbors and unavailable endpoint captures.
/** \ingroup frameGrabberTraceTelemetry_unit_test
 * \see MagAOX::logger::telem_fgtrace::getAccessor
 */
TEST_CASE( "Camera excursion metadata distinguishes absent neighbors and markers", "[frameGrabberTraceTelemetry]" )
{
    auto source      = sampleRecord();
    source.present   = 15;
    source.end       = fgTiming::TraceEnd::boundary;
    source.qualified = false;
    source.events[1].valid &= ~( 1u << 4 );
    source.events[1].invalidMarkers = 1;
    telem_fgtrace::messageT message( source, sampleStatus(), 123, 456, source.evaluatedNs );
    auto                    buffer = envelope( message );
    REQUIRE( verifyLogEntry( telem_fgtrace::eventCode, buffer.get() ) );
    auto *payload = message.builder.GetBufferPointer();
    REQUIRE( project<uint64_t>( "id", payload ) == source.id );
    REQUIRE( project<uint32_t>( "present", payload ) == 15 );
    REQUIRE_FALSE( project<bool>( "qualified", payload ) );
    REQUIRE( project<int>( "direction", payload ) == 0 );
    REQUIRE( project<int>( "end", payload ) == 1 );
    REQUIRE( project<int64_t>( "medianNs", payload ) == 500000 );
    REQUIRE( project<uint64_t>( "baselineIntervals", payload ) == 27 );
    REQUIRE( project<int64_t>( "intervalNs", payload ) == 540000 );
    REQUIRE( project<int64_t>( "thresholdNs", payload ) == 20000 );
    REQUIRE( project<int64_t>( "bracketLimitNs", payload ) == 1000 );
    REQUIRE( project<int64_t>( "evaluatedNs", payload ) == source.evaluatedNs );
    REQUIRE( project<int64_t>( "controlCoverageNs", payload ) == source.controlCoverageNs );
    REQUIRE( project<int64_t>( "emittedNs", payload ) == source.evaluatedNs );
    REQUIRE( project<uint64_t>( "processPid", payload ) == 123 );
    REQUIRE( project<uint64_t>( "acquisitionTid", payload ) == 456 );
    REQUIRE( project<bool>( "quality_enabled", payload ) );
    REQUIRE( project<uint64_t>( "quality_eventDropped", payload ) == sampleStatus().eventDropped );
    REQUIRE( project<uint64_t>( "quality_controlDropped", payload ) == 5 );

    for( size_t slot = 0; slot < source.events.size(); ++slot )
    {
        const std::string prefix  = "frame" + std::to_string( slot ) + '_';
        const bool        present = slot != 4;
        REQUIRE( project<bool>( prefix + "present", payload ) == present );
        REQUIRE( project<uint64_t>( prefix + "sequence", payload ) == ( present ? source.events[slot].sequence : 0 ) );
        REQUIRE( project<uint64_t>( prefix + "outputCounter", payload ) ==
                 ( present ? source.events[slot].outputCounter : 0 ) );
        REQUIRE( project<int64_t>( prefix + "atimeSec", payload ) ==
                 ( present ? source.events[slot].atime.tv_sec : 0 ) );
        REQUIRE( project<int64_t>( prefix + "serialOverlapNs", payload ) ==
                 ( present ? source.serialOverlapNs[slot] : -1 ) );
        REQUIRE( project<int64_t>( prefix + "writeStampDoneNs", payload ) ==
                 ( present ? source.events[slot].markers[5] : -1 ) );
    }

    REQUIRE( project<int64_t>( "frame1_loadDoneNs", payload ) == -1 );
    REQUIRE( telem_fgtrace::getAccessor( "frame5_sequence" ).accessor == nullptr );
    REQUIRE( telem_fgtrace::getAccessor( "frame2_unknown" ).accessor == nullptr );
    REQUIRE( telem_fgtrace::getAccessor( "unknown" ).accessor == nullptr );
    const auto text = telem_fgtrace::msgString( payload, message.builder.GetSize() );
    REQUIRE( text.find( "loadDoneNs:unavailable" ) != std::string::npos );
    REQUIRE( text.find( "frame4={present:0}" ) != std::string::npos );
}

/// Keep raw ambiguous or incomplete records available while rejecting impossible classification and malformed metadata.
/** \ingroup frameGrabberTraceTelemetry_unit_test
 * \see MagAOX::logger::telem_fgtrace::verify
 */
TEST_CASE( "Camera excursion telemetry validates classification and shape", "[frameGrabberTraceTelemetry]" )
{
    for( unsigned variant = 0; variant < 29; ++variant )
    {
        INFO( "malformed trace variant " << variant );
        auto     source = sampleRecord();
        auto     status = sampleStatus();
        uint64_t pid = 123, tid = 456;
        int64_t  emitted = source.evaluatedNs;

        switch( variant )
        {
        case 0:
            source.present = 32;
            break;
        case 1:
            source.present = 27;
            break;
        case 2:
            source.direction = fgTiming::TraceDirection( 255 );
            break;
        case 3:
            source.end = fgTiming::TraceEnd( 255 );
            break;
        case 4:
            source.medianNs = 0;
            break;
        case 5:
            source.intervalNs = 0;
            break;
        case 6:
            source.thresholdNs = -1;
            break;
        case 7:
            source.bracketLimitNs = -1;
            break;
        case 8:
            source.bracketLimitNs = 0;
            break;
        case 9:
            source.evaluatedNs = -1;
            break;
        case 10:
            source.controlCoverageNs = source.evaluatedNs + 1;
            break;
        case 11:
            source.baselineIntervals = 2;
            break;
        case 12:
            --emitted;
            break;
        case 13:
            pid = 0;
            break;
        case 14:
            tid = 0;
            break;
        case 15:
            status.ready = status.capacity + 1;
            break;
        case 16:
            status.pending = status.capacity + 1;
            break;
        case 17:
            source.direction = fgTiming::TraceDirection::early;
            break;
        case 18:
            source.events[2].valid = 512;
            break;
        case 19:
            source.events[2].markers[3] = source.events[2].markers[2] - 1;
            break;
        case 20:
            source.events[2].outcome = fgTiming::Outcome( 255 );
            break;
        case 21:
            ++source.intervalNs;
            break;
        case 22:
            source.events[1].valid &= ~( 1u << 5 );
            break;
        case 23:
            source.events[1].invalidMarkers = 1;
            break;
        case 24:
            source.events[1].writetime = {};
            break;
        case 25:
            --source.events[2].markers[4];
            break;
        case 26:
            ++source.events[2].outputCounter;
            break;
        case 27:
            ++source.events[2].session;
            break;
        case 28:
            status.ready   = status.capacity;
            status.pending = 1;
            break;
        }

        telem_fgtrace::messageT message( source, status, pid, tid, emitted );
        auto                    buffer = envelope( message );
        REQUIRE_FALSE( verifyLogEntry( telem_fgtrace::eventCode, buffer.get() ) );
    }

    auto source       = sampleRecord();
    source.direction  = fgTiming::TraceDirection::early;
    source.intervalNs = 460000;

    for( auto &marker : source.events[2].markers )
    {
        marker -= 80000;
    }

    source.qualified      = false;
    source.bracketLimitNs = 0;
    source.end            = fgTiming::TraceEnd::expired;
    source.present        = 7;
    telem_fgtrace::messageT message( source, sampleStatus(), 123, 456, source.evaluatedNs );
    auto                    buffer = envelope( message );
    REQUIRE( verifyLogEntry( telem_fgtrace::eventCode, buffer.get() ) );
    REQUIRE_FALSE( telem_fgtrace::verify( buffer, 1 ) );
    std::memset( flatlogs::logHeader::messageBuffer( buffer ), 0xff, 4 );
    REQUIRE_FALSE( verifyLogEntry( telem_fgtrace::eventCode, buffer.get() ) );
    flatlogs::bufferPtrT absent;
    REQUIRE_FALSE( telem_fgtrace::verify( absent, 0 ) );
}

/// Reject incomplete vector shapes and duplicate slot identities instead of relying on implicit vector position.
/** \ingroup frameGrabberTraceTelemetry_unit_test
 * \see MagAOX::logger::telem_fgtrace::verify
 */
TEST_CASE( "Camera excursion telemetry requires five identified neighbors and nine markers",
           "[frameGrabberTraceTelemetry]" )
{
    const auto source = sampleRecord();

    for( unsigned variant = 0; variant < 4; ++variant )
    {
        telem_fgtrace::messageT message( source, sampleStatus(), 123, 456, source.evaluatedNs );
        auto                   *record = GetTelem_fgtrace_fb( message.builder.GetBufferPointer() );

        if( variant == 0 )
        {
            flatbuffers::WriteScalar<flatbuffers::uoffset_t>(
                const_cast<void *>( static_cast<const void *>( record->events() ) ), 4 );
        }
        else if( variant == 1 )
        {
            flatbuffers::WriteScalar<flatbuffers::uoffset_t>(
                const_cast<void *>( static_cast<const void *>( record->serial_overlap_ns() ) ), 4 );
        }
        else if( variant == 2 )
        {
            flatbuffers::WriteScalar<flatbuffers::uoffset_t>(
                const_cast<void *>( static_cast<const void *>( record->events()->Get( 2 )->markers_ns() ) ), 8 );
        }
        else
        {
            auto *table =
                reinterpret_cast<flatbuffers::Table *>( const_cast<FgTraceEvent *>( record->events()->Get( 2 ) ) );
            REQUIRE( table->GetAddressOf( FgTraceEvent::VT_SLOT ) != nullptr );
            flatbuffers::WriteScalar<uint8_t>( table->GetAddressOf( FgTraceEvent::VT_SLOT ), 1 );
        }

        auto buffer = envelope( message );
        REQUIRE_FALSE( verifyLogEntry( telem_fgtrace::eventCode, buffer.get() ) );
    }
}

} // namespace frameGrabberTraceTelemetryTest
} // namespace loggerTest
} // namespace libXWCTest
