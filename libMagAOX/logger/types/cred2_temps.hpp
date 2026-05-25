/** \file cred2_temps.hpp
 * \brief The MagAO-X logger cred2_temps log type.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup logger_types_files
 *
 * History:
 * - 2026-03-27 created by Codex
 */
#ifndef logger_types_cred2_temps_hpp
#define logger_types_cred2_temps_hpp

#include "generated/cred2_temps_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{

/// Log entry recording the C-RED 2 detailed temperature channels.
/** \ingroup logger_types
 */
struct cred2_temps : public flatbuffer_log
{
    /// The event code
    static const flatlogs::eventCodeT eventCode = eventCodes::CRED2_TEMPS;

    /// The default level
    static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_TELEM;

    static timespec lastRecord; ///< The timestamp of the last time this log was recorded. Used by telemetry.

    /// The type of the input message
    struct messageT : public fbMessage
    {
        /// Construct from components
        messageT( const float &motherboard, ///< [in] motherboard temperature [C]
                  const float &frontend,    ///< [in] front-end temperature [C]
                  const float &powerboard,  ///< [in] power-board temperature [C]
                  const float &snake,       ///< [in] detector temperature [C]
                  const float &setpoint,    ///< [in] detector setpoint temperature [C]
                  const float &peltier,     ///< [in] external TEC temperature [C]
                  const float &heatsink     ///< [in] heatsink temperature [C]
        )
        {
            auto fp =
                CreateCred2_temps_fb( builder, motherboard, frontend, powerboard, snake, setpoint, peltier, heatsink );
            builder.Finish( fp );
        }
    };

    static bool verify( flatlogs::bufferPtrT &logBuff, ///< [in] buffer containing the flatbuffer serialized message
                        flatlogs::msgLenT     len      ///< [in] length of msgBuffer
    )
    {
        auto verifier = flatbuffers::Verifier( static_cast<uint8_t *>( flatlogs::logHeader::messageBuffer( logBuff ) ),
                                               static_cast<size_t>( len ) );
        return VerifyCred2_temps_fbBuffer( verifier );
    }

    /// Get the message formatted for human consumption.
    static std::string msgString( void *msgBuffer,      /**< [in] buffer containing the flatbuffer serialized message */
                                  flatlogs::msgLenT len /**< [in] [unused] length of msgBuffer */
    )
    {
        static_cast<void>( len );

        auto fbs = GetCred2_temps_fb( msgBuffer );

        std::string msg = "mb: ";
        msg += std::to_string( fbs->motherboard() ) + " C ";

        msg += "fe: ";
        msg += std::to_string( fbs->frontend() ) + " C ";

        msg += "pwr: ";
        msg += std::to_string( fbs->powerboard() ) + " C ";

        msg += "snake: ";
        msg += std::to_string( fbs->snake() ) + " C ";

        msg += "setpt: ";
        msg += std::to_string( fbs->setpoint() ) + " C ";

        msg += "pelt: ";
        msg += std::to_string( fbs->peltier() ) + " C ";

        msg += "sink: ";
        msg += std::to_string( fbs->heatsink() ) + " C ";

        return msg;
    }

    static float motherboard( void *msgBuffer /**< [in] buffer containing the flatbuffer serialized message */ )
    {
        auto fbs = GetCred2_temps_fb( msgBuffer );
        return fbs->motherboard();
    }

    static float frontend( void *msgBuffer /**< [in] buffer containing the flatbuffer serialized message */ )
    {
        auto fbs = GetCred2_temps_fb( msgBuffer );
        return fbs->frontend();
    }

    static float powerboard( void *msgBuffer /**< [in] buffer containing the flatbuffer serialized message */ )
    {
        auto fbs = GetCred2_temps_fb( msgBuffer );
        return fbs->powerboard();
    }

    static float snake( void *msgBuffer /**< [in] buffer containing the flatbuffer serialized message */ )
    {
        auto fbs = GetCred2_temps_fb( msgBuffer );
        return fbs->snake();
    }

    static float setpoint( void *msgBuffer /**< [in] buffer containing the flatbuffer serialized message */ )
    {
        auto fbs = GetCred2_temps_fb( msgBuffer );
        return fbs->setpoint();
    }

    static float peltier( void *msgBuffer /**< [in] buffer containing the flatbuffer serialized message */ )
    {
        auto fbs = GetCred2_temps_fb( msgBuffer );
        return fbs->peltier();
    }

    static float heatsink( void *msgBuffer /**< [in] buffer containing the flatbuffer serialized message */ )
    {
        auto fbs = GetCred2_temps_fb( msgBuffer );
        return fbs->heatsink();
    }

    /// Get the logMetaDetail for a member by name.
    /**
     * \returns a logMetaDetail filled in with the appropriate details
     * \returns an empty logMetaDetail if member not recognized
     */
    static logMetaDetail getAccessor( const std::string &member /**< [in] the name of the member */ )
    {
        if( member == "motherboard" )
            return logMetaDetail( { "MB TEMP",
                                    "motherboard temperature [C]",
                                    logMeta::valTypes::Float,
                                    logMeta::metaTypes::Continuous,
                                    reinterpret_cast<void *>( &motherboard ),
                                    false } );
        else if( member == "frontend" )
            return logMetaDetail( { "FE TEMP",
                                    "front-end temperature [C]",
                                    logMeta::valTypes::Float,
                                    logMeta::metaTypes::Continuous,
                                    reinterpret_cast<void *>( &frontend ),
                                    false } );
        else if( member == "powerboard" )
            return logMetaDetail( { "PWR TEMP",
                                    "power-board temperature [C]",
                                    logMeta::valTypes::Float,
                                    logMeta::metaTypes::Continuous,
                                    reinterpret_cast<void *>( &powerboard ),
                                    false } );
        else if( member == "snake" )
            return logMetaDetail( { "SNAKE TEMP",
                                    "detector temperature [C]",
                                    logMeta::valTypes::Float,
                                    logMeta::metaTypes::Continuous,
                                    reinterpret_cast<void *>( &snake ),
                                    false } );
        else if( member == "setpoint" )
            return logMetaDetail( { "SET TEMP",
                                    "detector setpoint temperature [C]",
                                    logMeta::valTypes::Float,
                                    logMeta::metaTypes::Continuous,
                                    reinterpret_cast<void *>( &setpoint ),
                                    false } );
        else if( member == "peltier" )
            return logMetaDetail( { "PELTIER TEMP",
                                    "external TEC temperature [C]",
                                    logMeta::valTypes::Float,
                                    logMeta::metaTypes::Continuous,
                                    reinterpret_cast<void *>( &peltier ),
                                    false } );
        else if( member == "heatsink" )
            return logMetaDetail( { "SINK TEMP",
                                    "heatsink temperature [C]",
                                    logMeta::valTypes::Float,
                                    logMeta::metaTypes::Continuous,
                                    reinterpret_cast<void *>( &heatsink ),
                                    false } );
        else
        {
            std::cerr << "No member " << member << " in cred2_temps\n";
            return logMetaDetail();
        }
    }
};

} // namespace logger
} // namespace MagAOX

#endif // logger_types_cred2_temps_hpp
