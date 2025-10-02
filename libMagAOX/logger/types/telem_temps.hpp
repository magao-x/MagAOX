/** \file telem_temps.hpp
 * \brief The MagAO-X logger telem_temps log type.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup logger_types_files
 *
 * History:
 * - 2018-09-06 created by JRM
 */
#ifndef logger_types_telem_temps_hpp
#define logger_types_telem_temps_hpp

#include "generated/telem_temps_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{

/// Log entry recording electronics rack temperature
/** \ingroup logger_types
 */
struct telem_temps : public flatbuffer_log
{
    /// The event code
    static const flatlogs::eventCodeT eventCode = eventCodes::TELEM_TEMPS;

    /// The default level
    static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_TELEM;

    static timespec lastRecord; ///< The time of the last time this log was recorded.  Used by the telemetry system.

    /// The type of the input message
    struct messageT : public fbMessage
    {
        /// Construct from components
        messageT( const std::vector<float> &temps ///<[in] vector of temperatures
        )
        {
            auto _temps = builder.CreateVector( temps );
            auto fp     = CreateTelem_temps_fb( builder, _temps );

            builder.Finish( fp );
        }
    };

    static bool verify( flatlogs::bufferPtrT &logBuff, ///< [in] Buffer containing the flatbuffer serialized message.
                        flatlogs::msgLenT     len      ///< [in] length of msgBuffer.
    )
    {
        auto verifier = flatbuffers::Verifier( static_cast<uint8_t *>( flatlogs::logHeader::messageBuffer( logBuff ) ),
                                               static_cast<size_t>( len ) );
        return VerifyTelem_temps_fbBuffer( verifier );
    }

    /// Get the message formatted for human consumption.
    static std::string msgString( void *msgBuffer,      /**< [in] Buffer containing the flatbuffer serialized message.*/
                                  flatlogs::msgLenT len /**< [in] [unused] length of msgBuffer.*/
    )
    {
        static_cast<void>( len );

        auto fbs = GetTelem_temps_fb( msgBuffer );

        std::string msg = "[temps] ";

        if( fbs->temps() )
        {
            for( size_t i = 0; i < fbs->temps()->size(); ++i )
            {
                msg += " ";
                msg += std::to_string( fbs->temps()->Get( i ) );
            }
        }

        return msg;
    }

    static std::vector<float> temps( void *msgBuffer /**< [in] Buffer containing the flatbuffer serialized message.*/ )
    {
        std::vector<float> p;

        auto fbs = GetTelem_temps_fb( msgBuffer );

        if( fbs->temps() != nullptr )
        {
            for( auto it = fbs->temps()->begin(); it != fbs->temps()->end(); ++it )
            {
                p.push_back( *it );
            }
        }
        return p;
    }

    /// Get the logMetaDetail for a member by name
    /**
     * \returns the a logMetaDetail filled in with the appropriate details
     * \returns an empty logMetaDetail if member not recognized
     */
    static logMetaDetail getAccessor( const std::string &member /**< [in] the name of the member */ )
    {
        if( member == "temps" )
            return logMetaDetail( { "TEMPS",
                                    "temperatures",
                                    logMeta::valTypes::Vector_Float,
                                    logMeta::metaTypes::State,
                                    reinterpret_cast<void *>( &temps ),
                                    false } );
        else
        {
            std::cerr << "No member " << member << " in telem_temps\n";
            return logMetaDetail();
        }
    }

}; // telem_temps

} // namespace logger
} // namespace MagAOX

#endif // logger_types_telem_temps_hpp
