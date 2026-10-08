/** \file telem_outlet.hpp
 * \brief Variable-length observed outlet-state telemetry and FITS metadata.
 * \ingroup logger_types_files
 */
#ifndef logger_types_telem_outlet_hpp
#define logger_types_telem_outlet_hpp

#include "generated/telem_outlet_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{
/// Snapshot of observed outlet states; -1 unknown, 0 off, 1 intermediate, 2 on.
/** \ingroup logger_types */
struct telem_outlet : public flatbuffer_log
{
    /// Unique logger event code.
    static constexpr flatlogs::eventCodeT eventCode = eventCodes::TELEM_OUTLET;

    /// Default telemetry priority.
    static constexpr flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_TELEM;

    /// Most recent record time used by the telemetry scheduler.
    static timespec lastRecord;

    /// Serialized snapshot input.
    struct messageT : public fbMessage
    {
        /// Construct an observed-state snapshot.
        messageT( uint8_t first /**< [in] first displayed outlet number, zero or one */,
                  const std::vector<int8_t> &states /**< [in] observed states in internal outlet order */ );
    };

    /// Verify a serialized log payload.
    static bool verify( flatlogs::bufferPtrT &logBuff /**< [in] complete log buffer */,
                        flatlogs::msgLenT len /**< [in] payload length */ );

    /// Format a snapshot for human-readable logs.
    static std::string msgString( void *msgBuffer /**< [in] serialized payload */,
                                  flatlogs::msgLenT len /**< [in] unused payload length */ );

    /// Get the first displayed outlet number.
    static unsigned char first_outlet( void *msgBuffer /**< [in] serialized payload */ );

    /// Get the signed observed states without converting them to characters.
    static std::vector<int8_t> states( void *msgBuffer /**< [in] serialized payload */ );

    /// Format observed states as comma-separated decimal integers for FITS string cards.
    static std::string stateString( void *msgBuffer /**< [in] serialized payload */ );

    /// Get metadata for a supported field, or an empty descriptor for an unknown field.
    static logMetaDetail getAccessor( const std::string &member /**< [in] field name */ );
};

inline telem_outlet::messageT::messageT( uint8_t first, const std::vector<int8_t> &states )
{
    auto values = builder.CreateVector( states );
    builder.Finish( CreateTelem_outlet_fb( builder, first, values ) );
}

inline bool telem_outlet::verify( flatlogs::bufferPtrT &logBuff, flatlogs::msgLenT len )
{
    flatbuffers::Verifier verifier( static_cast<uint8_t *>( flatlogs::logHeader::messageBuffer( logBuff ) ), len );
    return VerifyTelem_outlet_fbBuffer( verifier );
}

inline std::string telem_outlet::msgString( void *msgBuffer, flatlogs::msgLenT len )
{
    static_cast<void>( len );
    return "[outlet] first: " + std::to_string( first_outlet( msgBuffer ) ) + " states: " + stateString( msgBuffer );
}

inline unsigned char telem_outlet::first_outlet( void *msgBuffer )
{
    return GetTelem_outlet_fb( msgBuffer )->first_outlet();
}

inline std::vector<int8_t> telem_outlet::states( void *msgBuffer )
{
    auto values = GetTelem_outlet_fb( msgBuffer )->states();
    if( !values ) return {};
    return { values->begin(), values->end() };
}

inline std::string telem_outlet::stateString( void *msgBuffer )
{
    std::string value;
    for( auto state : states( msgBuffer ) )
    {
        if( !value.empty() ) value += ',';
        value += std::to_string( static_cast<int>( state ) );
    }
    return value;
}

inline logMetaDetail telem_outlet::getAccessor( const std::string &member )
{
    if( member == "first_outlet" )
        return { "OUTLET FIRST", "first displayed outlet number", logMeta::valTypes::UChar,
                 logMeta::metaTypes::State, reinterpret_cast<void *>( &first_outlet ), true };
    if( member == "states" )
        return { "OUTLET STATES", "-1 unknown,0 off,1 intermediate,2 on", logMeta::valTypes::String,
                 logMeta::metaTypes::State, reinterpret_cast<void *>( &stateString ), true };
    return {};
}
} // namespace logger
} // namespace MagAOX
#endif // logger_types_telem_outlet_hpp
