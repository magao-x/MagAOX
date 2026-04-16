/** \file telem_w2tcsoffloader.hpp
 * \brief The MagAO-X logger telem_w2tcsoffloader log type.
 *
 * \ingroup logger_types_files
 *
 */
#ifndef logger_types_telem_w2tcsoffloader_hpp
#define logger_types_telem_w2tcsoffloader_hpp

#include "generated/telem_w2tcsoffloader_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{

/// Log entry recording the woofer-to-TCS Zernike coefficient vector.
/** \ingroup logger_types
 */
struct telem_w2tcsoffloader : public flatbuffer_log
{
    /// The event code.
    static const flatlogs::eventCodeT eventCode = eventCodes::TELEM_W2TCSOFFLOADER;

    /// The default level.
    static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_TELEM;

    static timespec lastRecord; ///< The time of the last time this log was recorded. Used by the telemetry system.

    /// The type of the input message.
    struct messageT : public fbMessage
    {
        /// Construct from components.
        explicit messageT( const std::vector<float> &coeffs /**< [in] the offloaded Zernike coefficients */ )
        {
            auto coeffVec = builder.CreateVector( coeffs );
            auto fp       = CreateTelem_w2tcsoffloader_fb( builder, coeffVec );

            builder.Finish( fp );
        }
    };

    static bool verify( flatlogs::bufferPtrT &logBuff, ///< [in] Buffer containing the flatbuffer serialized message.
                        flatlogs::msgLenT     len      ///< [in] length of msgBuffer.
    )
    {
        auto verifier = flatbuffers::Verifier( static_cast<uint8_t *>( flatlogs::logHeader::messageBuffer( logBuff ) ),
                                               static_cast<size_t>( len ) );
        return VerifyTelem_w2tcsoffloader_fbBuffer( verifier );
    }

    /// Get the message formatted for human consumption.
    static std::string msgString( void *msgBuffer, /**< [in] Buffer containing the flatbuffer serialized message. */
                                  flatlogs::msgLenT len /**< [in] [unused] length of msgBuffer. */
    )
    {
        static_cast<void>( len );

        auto fbs = GetTelem_w2tcsoffloader_fb( msgBuffer );

        std::string msg = "[w2tcsoffloader coeffs] ";

        if( fbs->coeffs() != nullptr )
        {
            for( flatbuffers::Vector<float>::const_iterator it = fbs->coeffs()->begin(); it != fbs->coeffs()->end();
                 ++it )
            {
                msg += std::to_string( *it );
                msg += " ";
            }
        }

        return msg;
    }

    /// Recover the coefficient vector from a serialized message.
    static std::vector<float> coeffs( void *msgBuffer )
    {
        std::vector<float> coeffVec;
        auto               fbs = GetTelem_w2tcsoffloader_fb( msgBuffer );

        if( fbs->coeffs() != nullptr )
        {
            for( flatbuffers::Vector<float>::const_iterator it = fbs->coeffs()->begin(); it != fbs->coeffs()->end();
                 ++it )
            {
                coeffVec.push_back( *it );
            }
        }

        return coeffVec;
    }

    /// Get the logMetaDetail for a member by name.
    /**
     * \returns a logMetaDetail filled in with the appropriate details
     * \returns an empty logMetaDetail if member not recognized
     */
    static logMetaDetail getAccessor( const std::string &member /**< [in] the name of the member */ )
    {
        if( member == "coeffs" )
        {
            return logMetaDetail( { "COEFFS",
                                    logMeta::valTypes::Vector_Float,
                                    logMeta::metaTypes::Continuous,
                                    reinterpret_cast<void *>( &coeffs ),
                                    false } );
        }
        else
        {
            std::cerr << "No member " << member << " in telem_w2tcsoffloader\n";
            return logMetaDetail();
        }
    }

}; // struct telem_w2tcsoffloader

} // namespace logger
} // namespace MagAOX

#endif // logger_types_telem_w2tcsoffloader_hpp
