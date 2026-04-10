/** \file telem_tcsi_labmode.hpp
 * \brief The MagAO-X logger type for tcsInterface lab-mode telemetry.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup logger_types_files
 *
 */
#ifndef logger_types_telem_tcsi_labmode_hpp
#define logger_types_telem_tcsi_labmode_hpp

#include "generated/telem_tcsi_labmode_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{

/// tcsInterface lab-mode telemetry.
/** \ingroup logger_types
 */
struct telem_tcsi_labmode : public flatbuffer_log
{
    /// The event code
    static const flatlogs::eventCodeT eventCode = eventCodes::TELEM_TCSI_LABMODE;

    /// The default level
    static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_TELEM;

    static timespec lastRecord; ///< The time of the last time this log was recorded. Used by the telemetry system.

    /// The type of the input message
    struct messageT : public fbMessage
    {
        /// Construct from components.
        messageT( const bool &labMode ///< [in] whether tcsInterface is in lab mode
        )
        {
            auto fp = CreateTelem_tcsi_labmode_fb( builder, labMode );
            builder.Finish( fp );
        }
    };

    static bool verify( flatlogs::bufferPtrT &logBuff, ///< [in] Buffer containing the flatbuffer serialized message.
                        flatlogs::msgLenT     len      ///< [in] length of msgBuffer.
    )
    {
        auto verifier = flatbuffers::Verifier( static_cast<uint8_t *>( flatlogs::logHeader::messageBuffer( logBuff ) ),
                                               static_cast<size_t>( len ) );
        return VerifyTelem_tcsi_labmode_fbBuffer( verifier );
    }

    /// Format the message for human consumption.
    static std::string msgString( void *msgBuffer,      /**< [in] Buffer containing the flatbuffer serialized message.*/
                                  flatlogs::msgLenT len /**< [in] [unused] length of msgBuffer.*/
    )
    {
        static_cast<void>( len );

        auto fbs = GetTelem_tcsi_labmode_fb( msgBuffer );

        std::string msg = "[tcsi_labmode] ";
        msg += "labMode: ";
        msg += std::to_string( fbs->labMode() );

        return msg;
    }

    /// Access the lab-mode state.
    static bool labMode( void *msgBuffer /**< [in] Buffer containing the flatbuffer serialized message.*/ )
    {
        auto fbs = GetTelem_tcsi_labmode_fb( msgBuffer );
        return fbs->labMode();
    }

    /// Get the logMetaDetail for a member by name.
    /**
     * \returns the a logMetaDetail filled in with the appropriate details
     * \returns an empty logMetaDetail if member not recognized
     */
    static logMetaDetail getAccessor( const std::string &member /**< [in] the name of the member */ )
    {
        if( member == "labMode" )
        {
            return logMetaDetail( { "LABMODE",
                                    logMeta::valTypes::Bool,
                                    logMeta::metaTypes::State,
                                    reinterpret_cast<void *>( &labMode ) } );
        }
        else
        {
            std::cerr << "No member " << member << " in telem_tcsi_labmode\n";
            return logMetaDetail();
        }
    }
};

} // namespace logger
} // namespace MagAOX

#endif // logger_types_telem_tcsi_labmode_hpp
