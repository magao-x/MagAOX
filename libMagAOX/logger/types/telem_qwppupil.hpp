/** \file telem_qwppupil.hpp
 * \brief The MagAO-X logger telem_qwppupil log type.
 *
 * \ingroup logger_types_files
 *
 */
#ifndef logger_types_telem_qwppupil_hpp
#define logger_types_telem_qwppupil_hpp

#include "generated/telem_qwppupil_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{

/// Log entry recording the QWP pupil tracker's predicted pupil offset and ttmperi correction.
/** \ingroup logger_types
 */
struct telem_qwppupil : public flatbuffer_log
{
    /// The event code
    static const flatlogs::eventCodeT eventCode = eventCodes::TELEM_QWPPUPIL;

    /// The default level
    static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_TELEM;

    static timespec
        lastRecord; ///< The timestamp of the last time this log was recorded.  Used by the telemetry system.

    /// The type of the input message
    struct messageT : public fbMessage
    {
        /// Construct from components
        messageT( const bool  &tracking, /**< [in] whether the ttmperi correction is being applied */
                  const float &dx,       /**< [in] the predicted camwfs pupil offset in x, in pixels */
                  const float &dy,       /**< [in] the predicted camwfs pupil offset in y, in pixels */
                  const float &v1,       /**< [in] the ttmperi axis 1 correction relative to the baseline, in V */
                  const float &v2        /**< [in] the ttmperi axis 2 correction relative to the baseline, in V */
        )
        {
            auto fp = CreateTelem_qwppupil_fb( builder, tracking, dx, dy, v1, v2 );
            builder.Finish( fp );
        }
    };

    /// Verify a serialized message.
    /**
     * \returns true if the buffer holds a valid message
     */
    static bool verify( flatlogs::bufferPtrT &logBuff, ///< [in] Buffer containing the flatbuffer serialized message.
                        flatlogs::msgLenT     len      ///< [in] length of msgBuffer.
    )
    {
        auto verifier = flatbuffers::Verifier( static_cast<uint8_t *>( flatlogs::logHeader::messageBuffer( logBuff ) ),
                                               static_cast<size_t>( len ) );
        return VerifyTelem_qwppupil_fbBuffer( verifier );
    }

    /// Get the message formatted for human consumption.
    /**
     * \returns the formatted message
     */
    static std::string msgString( void *msgBuffer,      /**< [in] Buffer containing the flatbuffer serialized message.*/
                                  flatlogs::msgLenT len /**< [in] [unused] length of msgBuffer.*/
    )
    {
        static_cast<void>( len );

        auto fbs = GetTelem_qwppupil_fb( msgBuffer );

        std::string msg = "[qwppupil] ";

        msg += "tracking: ";
        msg += fbs->tracking() ? "On " : "Off ";

        msg += "dx: " + std::to_string( fbs->dx() ) + " ";
        msg += "dy: " + std::to_string( fbs->dy() ) + " ";
        msg += "v1: " + std::to_string( fbs->v1() ) + " ";
        msg += "v2: " + std::to_string( fbs->v2() ) + " ";

        return msg;
    }

    /// Get the tracking flag.
    /**
     * \returns whether tracking was on
     */
    static bool tracking( void *msgBuffer /**< [in] Buffer containing the flatbuffer serialized message.*/ )
    {
        return GetTelem_qwppupil_fb( msgBuffer )->tracking();
    }

    /// Get the x offset.
    /**
     * \returns the predicted camwfs pupil offset in x, in pixels
     */
    static float dx( void *msgBuffer /**< [in] Buffer containing the flatbuffer serialized message.*/ )
    {
        return GetTelem_qwppupil_fb( msgBuffer )->dx();
    }

    /// Get the y offset.
    /**
     * \returns the predicted camwfs pupil offset in y, in pixels
     */
    static float dy( void *msgBuffer /**< [in] Buffer containing the flatbuffer serialized message.*/ )
    {
        return GetTelem_qwppupil_fb( msgBuffer )->dy();
    }

    /// Get the axis 1 correction.
    /**
     * \returns the ttmperi axis 1 correction relative to the baseline, in V
     */
    static float v1( void *msgBuffer /**< [in] Buffer containing the flatbuffer serialized message.*/ )
    {
        return GetTelem_qwppupil_fb( msgBuffer )->v1();
    }

    /// Get the axis 2 correction.
    /**
     * \returns the ttmperi axis 2 correction relative to the baseline, in V
     */
    static float v2( void *msgBuffer /**< [in] Buffer containing the flatbuffer serialized message.*/ )
    {
        return GetTelem_qwppupil_fb( msgBuffer )->v2();
    }

    /// Get the logMetaDetail for a member by name
    /**
     * \returns the a logMetaDetail filled in with the appropriate details
     * \returns an empty logMetaDetail if member not recognized
     */
    static logMetaDetail getAccessor( const std::string &member /**< [in] the name of the member */ )
    {
        if( member == "tracking" )
        {
            return logMetaDetail( { "TRACKING",
                                    logMeta::valTypes::Bool,
                                    logMeta::metaTypes::State,
                                    reinterpret_cast<void *>( &tracking ) } );
        }
        else if( member == "dx" )
        {
            return logMetaDetail(
                { "DX", logMeta::valTypes::Float, logMeta::metaTypes::State, reinterpret_cast<void *>( &dx ) } );
        }
        else if( member == "dy" )
        {
            return logMetaDetail(
                { "DY", logMeta::valTypes::Float, logMeta::metaTypes::State, reinterpret_cast<void *>( &dy ) } );
        }
        else if( member == "v1" )
        {
            return logMetaDetail(
                { "V1", logMeta::valTypes::Float, logMeta::metaTypes::State, reinterpret_cast<void *>( &v1 ) } );
        }
        else if( member == "v2" )
        {
            return logMetaDetail(
                { "V2", logMeta::valTypes::Float, logMeta::metaTypes::State, reinterpret_cast<void *>( &v2 ) } );
        }
        else
        {
            std::cerr << "No member " << member << " in telem_qwppupil\n";
            return logMetaDetail();
        }
    }

}; // telem_qwppupil

} // namespace logger
} // namespace MagAOX

#endif // logger_types_telem_qwppupil_hpp
