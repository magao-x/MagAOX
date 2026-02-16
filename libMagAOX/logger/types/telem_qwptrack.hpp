/** \file telem_qwptrack.hpp
 * \brief The MagAO-X logger telem_qwptrack log type.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup logger_types_files
 *
 */
#ifndef logger_types_telem_qwptrack_hpp
#define logger_types_telem_qwptrack_hpp

#include "generated/telem_qwptrack_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{

/// Log entry recording qwptrack stage specific status.
/** \ingroup logger_types
 */
struct telem_qwptrack : public flatbuffer_log
{
    /// The event code
    static const flatlogs::eventCodeT eventCode = eventCodes::TELEM_QWPTRACK;

    /// The default level
    static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_TELEM;

    static timespec
        lastRecord; ///< The timestamp of the last time this log was recorded.  Used by the telemetry system.

    /// The type of the input message
    struct messageT : public fbMessage
    {
        /// Construct from components
        messageT( const float &qwp1_angle,   /**<[in] the HWP set angle */
                  const float &qwp2_angle, /**<[in] the actual HWP angle */
                  const bool &tracking
        )
        {

            auto fp = CreateTelem_qwptrack_fb( builder, qwp1_angle, qwp2_angle, tracking);
            builder.Finish( fp );
        }
    };

    static bool verify( flatlogs::bufferPtrT &logBuff, ///< [in] Buffer containing the flatbuffer serialized message.
                        flatlogs::msgLenT     len      ///< [in] length of msgBuffer.
    )
    {
        auto verifier = flatbuffers::Verifier( static_cast<uint8_t *>( flatlogs::logHeader::messageBuffer( logBuff ) ),
                                               static_cast<size_t>( len ) );
        return VerifyTelem_qwptrack_fbBuffer( verifier );
    }

    /// Get the message formatte for human consumption.
    static std::string msgString( void *msgBuffer,      /**< [in] Buffer containing the flatbuffer serialized message.*/
                                  flatlogs::msgLenT len /**< [in] [unused] length of msgBuffer.*/
    )
    {
        static_cast<void>( len );

        auto fbs = GetTelem_qwptrack_fb( msgBuffer );

        std::string msg = "[qwptrack] ";

        msg += "qwp1: ";
        msg += std::to_string( fbs->qwp1_angle() ) + " ";

        msg += "qwp2: ";
        msg += std::to_string( fbs->qwp2_angle() ) + " ";

        msg += "tracking: ";
        if (fbs->tracking())
        {
            msg += "SYNCHRO_IMR ";
        }
        else
        {
            msg += "NONE ";
        }

        return msg;
    }

    static float qwp1_angle( void *msgBuffer )
    {
        auto fbs = GetTelem_qwptrack_fb( msgBuffer );
        return fbs->qwp1_angle();
    }

    static float qwp2_angle( void *msgBuffer )
    {
        auto fbs = GetTelem_qwptrack_fb( msgBuffer );
        return fbs->qwp2_angle();
    }

    static bool tracking( void *msgBuffer )
    {
        auto fbs = GetTelem_qwptrack_fb( msgBuffer );
        return fbs->tracking();

    }

    /// Get the logMetaDetail for a member by name
    /**
     * \returns the a logMetaDetail filled in with the appropriate details
     * \returns an empty logMetaDetail if member not recognized
     */
    static logMetaDetail getAccessor( const std::string &member /**< [in] the name of the member */ )
    {
        if( member == "qwp1_angle" )
        {
            return logMetaDetail( { "QWP1 ANGLE",
                                    logMeta::valTypes::Float,
                                    logMeta::metaTypes::State,
                                    reinterpret_cast<void *>( &qwp1_angle ) } );
        }
        else if( member == "qwp2_angle" )
        {
            return logMetaDetail( { "QWP2 ANGLE",
                                    logMeta::valTypes::Float,
                                    logMeta::metaTypes::State,
                                    reinterpret_cast<void *>( &qwp2_angle ) } );
        }
        else if( member == "tracking" )
        {
            return logMetaDetail( { "TRACKING",
                                    logMeta::valTypes::Bool,
                                    logMeta::metaTypes::State,
                                    reinterpret_cast<void *>( &tracking ) } );
        }
        else
        {
            std::cerr << "No member " << member << " in telem_qwptrack\n";
            return logMetaDetail();
        }
    }

}; // telem_qwptrack

} // namespace logger
} // namespace MagAOX

#endif // logger_types_telem_qwptrack_hpp
