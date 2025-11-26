/** \file telem_poltrack.hpp
 * \brief The MagAO-X logger telem_poltrack log type.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup logger_types_files
 *
 */
#ifndef logger_types_telem_poltrack_hpp
#define logger_types_telem_poltrack_hpp

#include "generated/telem_poltrack_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{

/// Log entry recording poltrack stage specific status.
/** \ingroup logger_types
 */
struct telem_poltrack : public flatbuffer_log
{
    /// The event code
    static const flatlogs::eventCodeT eventCode = eventCodes::TELEM_POLTRACK;

    /// The default level
    static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_TELEM;

    static timespec
        lastRecord; ///< The timestamp of the last time this log was recorded.  Used by the telemetry system.

    /// The type of the input message
    struct messageT : public fbMessage
    {
        /// Construct from components
        messageT( const float &set_angle,   /**<[in] the HWP set angle */
                  const float &actual_angle, /**<[in] the actual HWP angle */
                  const std::string &pos_name, /**<[in] the name of the HWP position */
                  const bool &tracking
        )
        {
            auto _pos_name = builder.CreateString(pos_name);
         
            auto fp = CreateTelem_poltrack_fb( builder, set_angle, actual_angle, _pos_name, tracking);
            builder.Finish( fp );
        }
    };

    static bool verify( flatlogs::bufferPtrT &logBuff, ///< [in] Buffer containing the flatbuffer serialized message.
                        flatlogs::msgLenT     len      ///< [in] length of msgBuffer.
    )
    {
        auto verifier = flatbuffers::Verifier( static_cast<uint8_t *>( flatlogs::logHeader::messageBuffer( logBuff ) ),
                                               static_cast<size_t>( len ) );
        return VerifyTelem_poltrack_fbBuffer( verifier );
    }

    /// Get the message formatte for human consumption.
    static std::string msgString( void *msgBuffer,      /**< [in] Buffer containing the flatbuffer serialized message.*/
                                  flatlogs::msgLenT len /**< [in] [unused] length of msgBuffer.*/
    )
    {
        static_cast<void>( len );

        auto fbs = GetTelem_poltrack_fb( msgBuffer );

        std::string msg = "[poltrack] ";

        msg += "set: ";
        msg += std::to_string( fbs->set_angle() ) + " ";

        msg += "act: ";
        msg += std::to_string( fbs->actual_angle() ) + " ";

        msg += "name: ";
        if ( fbs->pos_name() )
        {
            msg += std::string(fbs->pos_name()->c_str()) + " ";
        }

        msg += "tracking: ";
        if (fbs->tracking()) 
        {
            msg += "SYNCHRO_ADI ";
        } 
        else 
        {
            msg += "NONE ";
        }

        return msg;
    }

    static float set_angle( void *msgBuffer )
    {
        auto fbs = GetTelem_poltrack_fb( msgBuffer );
        return fbs->set_angle();
    }

    static float actual_angle( void *msgBuffer )
    {
        auto fbs = GetTelem_poltrack_fb( msgBuffer );
        return fbs->actual_angle();
    }

    static std::string pos_name( void *msgBuffer )
    {
        auto fbs = GetTelem_poltrack_fb(msgBuffer);
        if(fbs->pos_name())
        {
            return std::string(fbs->pos_name()->c_str());
        }
        else return std::string();
    }

    static bool tracking( void *msgBuffer )
    {
        auto fbs = GetTelem_poltrack_fb( msgBuffer );
        return fbs->tracking();

    }

    /// Get the logMetaDetail for a member by name
    /**
     * \returns the a logMetaDetail filled in with the appropriate details
     * \returns an empty logMetaDetail if member not recognized
     */
    static logMetaDetail getAccessor( const std::string &member /**< [in] the name of the member */ )
    {
        if( member == "set_angle" )
        {
            return logMetaDetail( { "SET ANGLE",
                                    logMeta::valTypes::Float,
                                    logMeta::metaTypes::Continuous,
                                    reinterpret_cast<void *>( &set_angle ) } );
        }
        else if( member == "actual_angle" )
        {
            return logMetaDetail( { "ACTUAL ANGLE",
                                    logMeta::valTypes::Float,
                                    logMeta::metaTypes::Continuous,
                                    reinterpret_cast<void *>( &actual_angle ) } );
        }
        else if( member == "pos_name" )
        {
            return logMetaDetail( { "POS NAME",
                                    logMeta::valTypes::String,
                                    logMeta::metaTypes::State,
                                    reinterpret_cast<void *>( &pos_name ) } );
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
            std::cerr << "No member " << member << " in telem_poltrack\n";
            return logMetaDetail();
        }
    }

}; // telem_poltrack

} // namespace logger
} // namespace MagAOX

#endif // logger_types_telem_poltrack_hpp
