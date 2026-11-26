/** \file telem_alignloop.hpp
 * \brief Binary telemetry for alignment loop state and processed measurements.
 *
 * \ingroup logger_types_files
 */
#ifndef logger_types_telem_alignloop_hpp
#define logger_types_telem_alignloop_hpp

#include <sstream>

#include "generated/telem_alignloop_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{

/// State or processed-frame telemetry for a two-axis alignment loop.
/** \ingroup logger_types
 */
struct telem_alignloop : public flatbuffer_log
{
    /// Event code for alignment loop telemetry.
    static constexpr flatlogs::eventCodeT eventCode = eventCodes::TELEM_ALIGNLOOP;

    /// Priority used by the telemetry writer.
    static constexpr flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_TELEM;

    /// Timestamp used for periodic state refreshes.
    static timespec lastRecord;

    /// FlatBuffer message constructed from the loop's state and optional frame.
    struct messageT : public fbMessage
    {
        /// Serialize the record, including targets only on an attempted send.
        messageT( bool                   closed,    /**< [in] whether the loop is closed */
                  float                  gain,      /**< [in] global loop gain */
                  const AlignLoopAxes   &axisGains, /**< [in] controller-axis gains */
                  const AlignLoopAxes   &offsets,   /**< [in] measurement setpoints */
                  const AlignLoopSample *sample,    /**< [in] processed input frame or null for state only */
                  const AlignLoopAxes   *targets,   /**< [in] attempted INDI targets or null */
                  uint8_t                sentMask /**< [in] bit mask of successful local sends */ );
    };

    /// Verify a serialized alignment loop record.
    static bool verify( flatlogs::bufferPtrT &logBuff, /**< [in] complete flatlogs record */
                        flatlogs::msgLenT     len /**< [in] payload length in bytes */ );

    /// Format the record for a text log reader.
    static std::string msgString( void             *msgBuffer, /**< [in] verified FlatBuffer payload */
                                  flatlogs::msgLenT len /**< [in] payload length in bytes */ );

    /// Look up scalar metadata for the loop state.
    static logMetaDetail getAccessor( const std::string &member /**< [in] metadata field name */ );

  private:
    /// Read the closed-loop flag.
    static bool closed( void *msgBuffer /**< [in] verified FlatBuffer payload */ );

    /// Read the global loop gain.
    static float gain( void *msgBuffer /**< [in] verified FlatBuffer payload */ );

    /// Read the first measurement setpoint.
    static float offset0( void *msgBuffer /**< [in] verified FlatBuffer payload */ );

    /// Read the second measurement setpoint.
    static float offset1( void *msgBuffer /**< [in] verified FlatBuffer payload */ );
};

inline telem_alignloop::messageT::messageT( bool                   closed,
                                            float                  gain,
                                            const AlignLoopAxes   &axisGains,
                                            const AlignLoopAxes   &offsets,
                                            const AlignLoopSample *sample,
                                            const AlignLoopAxes   *targets,
                                            uint8_t                sentMask )
{
    builder.Finish(
        CreateTelem_alignloop_fb( builder, closed, gain, &axisGains, &offsets, sample, targets, sentMask ) );
}

inline bool telem_alignloop::verify( flatlogs::bufferPtrT &logBuff, flatlogs::msgLenT len )
{
    auto                 *payload = static_cast<uint8_t *>( flatlogs::logHeader::messageBuffer( logBuff ) );
    flatbuffers::Verifier verifier( payload, len );

    if( !VerifyTelem_alignloop_fbBuffer( verifier ) )
    {
        return false;
    }

    const auto *record = GetTelem_alignloop_fb( payload );
    return record->targets() ? record->sample() && record->closed() && record->sent_mask() <= 3
                             : record->sent_mask() == 0;
}

inline std::string telem_alignloop::msgString( void *msgBuffer, flatlogs::msgLenT len )
{
    static_cast<void>( len );
    const auto        *record = GetTelem_alignloop_fb( msgBuffer );
    std::ostringstream out;
    out << "[alignloop] closed=" << std::boolalpha << record->closed() << " gain=" << record->gain() << " axis_gains=("
        << record->axis_gains()->axis0() << ", " << record->axis_gains()->axis1() << ") offsets=("
        << record->offsets()->axis0() << ", " << record->offsets()->axis1() << ")";

    if( const auto *sample = record->sample() )
    {
        out << " frame=" << sample->frame() << " acquired=(" << sample->acq_sec() << ", " << sample->acq_nsec()
            << ") measurement=(" << sample->measurement().axis0() << ", " << sample->measurement().axis1()
            << ") residual=(" << sample->residual().axis0() << ", " << sample->residual().axis1() << ")";
    }

    if( const auto *targets = record->targets() )
    {
        out << " targets=(" << targets->axis0() << ", " << targets->axis1()
            << ") sent_mask=" << static_cast<unsigned>( record->sent_mask() );
    }

    return out.str();
}

inline bool telem_alignloop::closed( void *msgBuffer )
{
    return GetTelem_alignloop_fb( msgBuffer )->closed();
}

inline float telem_alignloop::gain( void *msgBuffer )
{
    return GetTelem_alignloop_fb( msgBuffer )->gain();
}

inline float telem_alignloop::offset0( void *msgBuffer )
{
    return GetTelem_alignloop_fb( msgBuffer )->offsets()->axis0();
}

inline float telem_alignloop::offset1( void *msgBuffer )
{
    return GetTelem_alignloop_fb( msgBuffer )->offsets()->axis1();
}

inline logMetaDetail telem_alignloop::getAccessor( const std::string &member )
{
    if( member == "closed" )
    {
        return { "CLOSED", logMeta::valTypes::Bool, logMeta::metaTypes::State, reinterpret_cast<void *>( &closed ) };
    }

    if( member == "gain" )
    {
        return { "GAIN", logMeta::valTypes::Float, logMeta::metaTypes::State, reinterpret_cast<void *>( &gain ) };
    }

    if( member == "offset0" )
    {
        return { "OFFSET0", logMeta::valTypes::Float, logMeta::metaTypes::State, reinterpret_cast<void *>( &offset0 ) };
    }

    if( member == "offset1" )
    {
        return { "OFFSET1", logMeta::valTypes::Float, logMeta::metaTypes::State, reinterpret_cast<void *>( &offset1 ) };
    }

    return {};
}

} // namespace logger
} // namespace MagAOX

#endif // logger_types_telem_alignloop_hpp
