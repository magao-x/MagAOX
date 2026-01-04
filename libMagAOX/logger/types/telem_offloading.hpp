/** \file telem_offloading.hpp
 * \brief The MagAO-X logger telem_offloading log type.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup logger_types_files
 *
 */
#ifndef logger_types_telem_offloading_hpp
#define logger_types_telem_offloading_hpp

#include "generated/telem_offloading_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{

/// Log entry recording the build-time git state.
/** \ingroup logger_types
 */
struct telem_offloading : public flatbuffer_log
{
    /// The event code
    static const flatlogs::eventCodeT eventCode = eventCodes::TELEM_OFFLOADING;

    /// The default level
    static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_TELEM;

    static timespec lastRecord; ///< The time of the last time this log was recorded.  Used by the telemetry system.

    /// The type of the input message
    struct messageT : public fbMessage
    {
        /// Construct from components
        messageT( const uint32_t &num_modes,   ///< [in] the number of modes being offloaded
                  const uint32_t &num_average, ///< [in] the number of frames being averaged
                  const float    &fps          ///< [in] the rate of offloading, in frames per second
        )
        {
            auto fp = CreateTelem_offloading_fb( builder, num_modes, num_average, fps );
            builder.Finish( fp );
        }
    };

    static bool verify( flatlogs::bufferPtrT &logBuff, ///< [in] Buffer containing the flatbuffer serialized message.
                        flatlogs::msgLenT     len      ///< [in] length of msgBuffer.
    )
    {
        auto verifier = flatbuffers::Verifier( static_cast<uint8_t *>( flatlogs::logHeader::messageBuffer( logBuff ) ),
                                               static_cast<size_t>( len ) );
        return VerifyTelem_offloading_fbBuffer( verifier );
    }

    /// Get the message formatte for human consumption.
    static std::string msgString( void *msgBuffer,      /**< [in] Buffer containing the flatbuffer serialized message.*/
                                  flatlogs::msgLenT len /**< [in] [unused] length of msgBuffer.*/
    )
    {
        static_cast<void>( len );

        auto fbs = GetTelem_offloading_fb( msgBuffer );

        std::string msg = "[offloading] ";

        msg += "modes: ";
        msg += std::to_string( fbs->num_modes() ) + " ";

        msg += "n-avg: ";
        msg += std::to_string( fbs->num_average() ) + " ";

        msg += "fps: ";
        msg += std::to_string( fbs->fps() ) + " ";

        return msg;
    }

    static uint32_t num_modes( void *msgBuffer )
    {
        auto fbs = GetTelem_offloading_fb( msgBuffer );
        return fbs->num_modes();
    }

    static uint32_t num_average( void *msgBuffer )
    {
        auto fbs = GetTelem_offloading_fb( msgBuffer );
        return fbs->num_average();
    }

    static float fps( void *msgBuffer )
    {
        auto fbs = GetTelem_offloading_fb( msgBuffer );
        return fbs->fps();
    }

    /// Get the logMetaDetail for a member by name
    /**
     * \returns the a logMetaDetail filled in with the appropriate details
     * \returns an empty logMetaDetail if member not recognized
     */
    static logMetaDetail getAccessor( const std::string &member /**< [in] the name of the member */ )
    {
        if( member == "num_modes" )
            return logMetaDetail( { "NUMBER OF MODES",
                                    logMeta::valTypes::Int,
                                    logMeta::metaTypes::State,
                                    reinterpret_cast<void *>( &num_modes ) } );
        else if( member == "num_average" )
            return logMetaDetail( { "FRAMES AVERAGED",
                                    logMeta::valTypes::Int,
                                    logMeta::metaTypes::State,
                                    reinterpret_cast<void *>( &num_average ) } );
        else if( member == "fps" )
            return logMetaDetail(
                { "FPS", logMeta::valTypes::Float, logMeta::metaTypes::State, reinterpret_cast<void *>( &fps ) } );
        else
        {
            std::cerr << "No member " << member << " in telem_offloading\n";
            return logMetaDetail();
        }
    }

}; // telem_offloading

} // namespace logger
} // namespace MagAOX

#endif // logger_types_telem_offloading_hpp
