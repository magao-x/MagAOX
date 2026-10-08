/** \file telem_pdu.hpp
 * \brief Electrical sample telemetry with explicit acquisition validity.
 * \ingroup logger_types_files
 */
#ifndef logger_types_telem_pdu_hpp
#define logger_types_telem_pdu_hpp

#include "generated/telem_pdu_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{
/// Observed electrical sample; invalid samples must not be treated as measurements.
/** \ingroup logger_types */
struct telem_pdu : public flatbuffer_log
{
    /// Unique logger event code.
    static constexpr flatlogs::eventCodeT eventCode = eventCodes::TELEM_PDU;

    /// Default telemetry priority.
    static constexpr flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_TELEM;

    /// Most recent record time used by the telemetry scheduler.
    static timespec lastRecord;

    /// Serialized electrical sample input.
    struct messageT : public fbMessage
    {
        /// Construct a complete electrical sample or its invalidation.
        messageT( float frequency /**< [in] line frequency in Hz */,
                  float voltage /**< [in] line voltage in V */,
                  float current /**< [in] total current in A */,
                  bool valid /**< [in] whether this is a complete successful measurement */ );
    };

    /// Verify a serialized log payload.
    static bool verify( flatlogs::bufferPtrT &logBuff /**< [in] complete log buffer */,
                        flatlogs::msgLenT len /**< [in] payload length */ );

    /// Format a sample with units and validity for human-readable logs.
    static std::string msgString( void *msgBuffer /**< [in] serialized payload */,
                                  flatlogs::msgLenT len /**< [in] unused payload length */ );

    /// Get line frequency [Hz].
    static float frequency( void *msgBuffer /**< [in] serialized payload */ );

    /// Get line voltage [V].
    static float voltage( void *msgBuffer /**< [in] serialized payload */ );

    /// Get total current [A].
    static float current( void *msgBuffer /**< [in] serialized payload */ );

    /// Get whether the complete electrical sample is valid.
    static bool valid( void *msgBuffer /**< [in] serialized payload */ );

    /// Get metadata for a supported field, or an empty descriptor for an unknown field.
    static logMetaDetail getAccessor( const std::string &member /**< [in] field name */ );
};

inline telem_pdu::messageT::messageT( float frequency, float voltage, float current, bool valid )
{
    builder.Finish( CreateTelem_pdu_fb( builder, frequency, voltage, current, valid ) );
}

inline bool telem_pdu::verify( flatlogs::bufferPtrT &logBuff, flatlogs::msgLenT len )
{
    flatbuffers::Verifier verifier( static_cast<uint8_t *>( flatlogs::logHeader::messageBuffer( logBuff ) ), len );
    return VerifyTelem_pdu_fbBuffer( verifier );
}

inline std::string telem_pdu::msgString( void *msgBuffer, flatlogs::msgLenT len )
{
    static_cast<void>( len );
    return "[pdu] " + std::to_string( frequency( msgBuffer ) ) + " Hz " + std::to_string( voltage( msgBuffer ) ) +
           " V " + std::to_string( current( msgBuffer ) ) + " A valid: " + std::to_string( valid( msgBuffer ) );
}

inline float telem_pdu::frequency( void *msgBuffer )
{
    return GetTelem_pdu_fb( msgBuffer )->frequency();
}

inline float telem_pdu::voltage( void *msgBuffer )
{
    return GetTelem_pdu_fb( msgBuffer )->voltage();
}

inline float telem_pdu::current( void *msgBuffer )
{
    return GetTelem_pdu_fb( msgBuffer )->current();
}

inline bool telem_pdu::valid( void *msgBuffer )
{
    return GetTelem_pdu_fb( msgBuffer )->valid();
}

inline logMetaDetail telem_pdu::getAccessor( const std::string &member )
{
    if( member == "frequency" )
        return { "PDU FREQUENCY", "line frequency [Hz]", logMeta::valTypes::Float,
                 logMeta::metaTypes::State, reinterpret_cast<void *>( &frequency ), true };
    if( member == "voltage" )
        return { "PDU VOLTAGE", "line voltage [V]", logMeta::valTypes::Float,
                 logMeta::metaTypes::State, reinterpret_cast<void *>( &voltage ), true };
    if( member == "current" )
        return { "PDU CURRENT", "total current [A]", logMeta::valTypes::Float,
                 logMeta::metaTypes::State, reinterpret_cast<void *>( &current ), true };
    if( member == "valid" )
        return { "PDU VALID", "whether the complete electrical sample is valid", logMeta::valTypes::Bool,
                 logMeta::metaTypes::State, reinterpret_cast<void *>( &valid ), true };
    return {};
}
} // namespace logger
} // namespace MagAOX
#endif // logger_types_telem_pdu_hpp
