/** \file telem_tcsi_offload.hpp
 * \brief Shared MagAO-X logger type for tcsInterface offload-control telemetry.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup logger_types_files
 *
 */
#ifndef logger_types_telem_tcsi_offload_hpp
#define logger_types_telem_tcsi_offload_hpp

#include "generated/telem_tcsi_offload_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{

/// Shared base class for tcsInterface offload-control telemetry.
/** This provides the common payload and accessors for both tip/tilt and focus
 * offload-control telemetry records. It is not intended to be used directly as
 * a logged type.
 *
 * \ingroup logger_types_basic
 */
struct telem_tcsi_offload : public flatbuffer_log
{
    /// The type of the input message
    struct messageT : public fbMessage
    {
        /// Construct from components.
        messageT( const bool  &enabled, ///< [in] whether the offload path is enabled
                  const float &avgInt,  ///< [in] the current averaging interval
                  const float &gain,    ///< [in] the current offload gain
                  const float &thresh   ///< [in] the current offload threshold
        )
        {
            auto fp = CreateTelem_tcsi_offload_fb( builder, enabled, avgInt, gain, thresh );
            builder.Finish( fp );
        }
    };

    static bool verify( flatlogs::bufferPtrT &logBuff, ///< [in] Buffer containing the flatbuffer serialized message.
                        flatlogs::msgLenT     len      ///< [in] length of msgBuffer.
    )
    {
        auto verifier = flatbuffers::Verifier( static_cast<uint8_t *>( flatlogs::logHeader::messageBuffer( logBuff ) ),
                                               static_cast<size_t>( len ) );
        return VerifyTelem_tcsi_offload_fbBuffer( verifier );
    }

    /// Format the message for human consumption.
    static std::string msgString( void *msgBuffer,      /**< [in] Buffer containing the flatbuffer serialized message.*/
                                  flatlogs::msgLenT len /**< [in] [unused] length of msgBuffer.*/
    )
    {
        static_cast<void>( len );

        auto fbs = GetTelem_tcsi_offload_fb( msgBuffer );

        std::string msg = "[tcsi_offload] ";

        msg += "enabled: ";
        msg += std::to_string( fbs->enabled() );
        msg += " avgInt: ";
        msg += std::to_string( fbs->avgInt() );
        msg += " gain: ";
        msg += std::to_string( fbs->gain() );
        msg += " thresh: ";
        msg += std::to_string( fbs->thresh() );

        return msg;
    }

    /// Access the enabled state.
    static bool enabled( void *msgBuffer /**< [in] Buffer containing the flatbuffer serialized message.*/ )
    {
        auto fbs = GetTelem_tcsi_offload_fb( msgBuffer );
        return fbs->enabled();
    }

    /// Access the offload gain.
    static float gain( void *msgBuffer /**< [in] Buffer containing the flatbuffer serialized message.*/ )
    {
        auto fbs = GetTelem_tcsi_offload_fb( msgBuffer );
        return fbs->gain();
    }

    /// Access the offload averaging interval.
    static float avgInt( void *msgBuffer /**< [in] Buffer containing the flatbuffer serialized message.*/ )
    {
        auto fbs = GetTelem_tcsi_offload_fb( msgBuffer );
        return fbs->avgInt();
    }

    /// Access the offload threshold.
    static float thresh( void *msgBuffer /**< [in] Buffer containing the flatbuffer serialized message.*/ )
    {
        auto fbs = GetTelem_tcsi_offload_fb( msgBuffer );
        return fbs->thresh();
    }

    /// Get the logMetaDetail for a member by name.
    /**
     * \returns the a logMetaDetail filled in with the appropriate details
     * \returns an empty logMetaDetail if member not recognized
     */
    static logMetaDetail getAccessor( const std::string &member /**< [in] the name of the member */ )
    {
        if( member == "enabled" )
        {
            return logMetaDetail( { "ENABLED",
                                    logMeta::valTypes::Bool,
                                    logMeta::metaTypes::State,
                                    reinterpret_cast<void *>( &enabled ) } );
        }
        else if( member == "avgInt" )
        {
            return logMetaDetail( { "AVGINT",
                                    logMeta::valTypes::Float,
                                    logMeta::metaTypes::State,
                                    reinterpret_cast<void *>( &avgInt ) } );
        }
        else if( member == "gain" )
        {
            return logMetaDetail(
                { "GAIN", logMeta::valTypes::Float, logMeta::metaTypes::State, reinterpret_cast<void *>( &gain ) } );
        }
        else if( member == "thresh" )
        {
            return logMetaDetail( { "THRESH",
                                    logMeta::valTypes::Float,
                                    logMeta::metaTypes::State,
                                    reinterpret_cast<void *>( &thresh ) } );
        }
        else
        {
            std::cerr << "No member " << member << " in telem_tcsi_offload\n";
            return logMetaDetail();
        }
    }
};

} // namespace logger
} // namespace MagAOX

#endif // logger_types_telem_tcsi_offload_hpp
