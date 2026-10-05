/** \file telem_flowrpm.hpp
 * \brief The MagAO-X logger telem_flowrpm log type.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup logger_types_files
 */

#ifndef logger_types_telem_flowrpm_hpp
#define logger_types_telem_flowrpm_hpp

#include "generated/telem_flowrpm_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{

/// Log entry recording the displayed flow value and source age.
/** \ingroup logger_types
 */
struct telem_flowrpm : public flatbuffer_log
{
    /// The event code.
    static const flatlogs::eventCodeT eventCode = eventCodes::TELEM_FLOWRPM;

    /// The default level.
    static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_TELEM;

    static timespec lastRecord; ///< The time of the last time this log was recorded. Used by the telemetry system.

    /// The type of the input message.
    struct messageT : public fbMessage
    {
        /// Construct from components.
        messageT( const double flowRate, /**< [in] displayed flow rate in LPM */
                  const double age,      /**< [in] displayed age in seconds */
                  const bool   valid     /**< [in] whether the displayed value is valid */
        )
        {
            auto fp = CreateTelem_flowrpm_fb( builder, flowRate, age, valid );
            builder.Finish( fp );
        }
    };

    static bool verify( flatlogs::bufferPtrT &logBuff, ///< [in] Buffer containing the flatbuffer serialized message.
                        flatlogs::msgLenT     len      ///< [in] length of msgBuffer.
    )
    {
        auto verifier = flatbuffers::Verifier( static_cast<uint8_t *>( flatlogs::logHeader::messageBuffer( logBuff ) ),
                                               static_cast<size_t>( len ) );
        return VerifyTelem_flowrpm_fbBuffer( verifier );
    }

    /// Get the message formatted for human consumption.
    static std::string msgString( void *msgBuffer, /**< [in] Buffer containing the flatbuffer serialized message. */
                                  flatlogs::msgLenT len /**< [in] [unused] length of msgBuffer. */
    )
    {
        static_cast<void>( len );

        auto        fbs = GetTelem_flowrpm_fb( msgBuffer );
        std::string msg = "[flowrpm] ";

        msg += "flow: ";
        msg += std::to_string( fbs->flowRate() );
        msg += " LPM age: ";
        msg += std::to_string( fbs->age() );
        msg += " s valid: ";
        msg += ( fbs->valid() ? "true" : "false" );

        return msg;
    }

    /// Recover the flow rate from a serialized message.
    static double flowRate( void *msgBuffer )
    {
        return GetTelem_flowrpm_fb( msgBuffer )->flowRate();
    }

    /// Recover the age from a serialized message.
    static double age( void *msgBuffer )
    {
        return GetTelem_flowrpm_fb( msgBuffer )->age();
    }

    /// Recover the validity flag from a serialized message.
    static bool valid( void *msgBuffer )
    {
        return GetTelem_flowrpm_fb( msgBuffer )->valid();
    }

    /// Get the logMetaDetail for a member by name.
    static logMetaDetail getAccessor( const std::string &member /**< [in] the name of the member */ )
    {
        if( member == "flowRate" )
        {
            return logMetaDetail( { "FLOW RATE",
                                    "LPM",
                                    logMeta::valTypes::Double,
                                    logMeta::metaTypes::Continuous,
                                    reinterpret_cast<void *>( &flowRate ),
                                    true } );
        }
        else if( member == "age" )
        {
            return logMetaDetail( { "FLOW AGE",
                                    "s",
                                    logMeta::valTypes::Double,
                                    logMeta::metaTypes::Continuous,
                                    reinterpret_cast<void *>( &age ),
                                    true } );
        }
        else if( member == "valid" )
        {
            return logMetaDetail( { "FLOW VALID",
                                    "",
                                    logMeta::valTypes::Bool,
                                    logMeta::metaTypes::State,
                                    reinterpret_cast<void *>( &valid ),
                                    true } );
        }
        else
        {
            std::cerr << "No member " << member << " in telem_flowrpm\n";
            return logMetaDetail();
        }
    }
};

} // namespace logger
} // namespace MagAOX

#endif // logger_types_telem_flowrpm_hpp
