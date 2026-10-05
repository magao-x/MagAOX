/** \file telem_adctrack.hpp
 * \brief The MagAO-X logger telem_adctrack log type.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup logger_types_files
 *
 */
#ifndef logger_types_telem_adctrack_hpp
#define logger_types_telem_adctrack_hpp

#include "generated/telem_adctrack_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{

/// Log entry recording ADC tracker operator-adjustable state.
/** \ingroup logger_types
 */
struct telem_adctrack : public flatbuffer_log
{
    /// The event code
    static const flatlogs::eventCodeT eventCode = eventCodes::TELEM_ADCTRACK;

    /// The default level
    static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_TELEM;

    static timespec
        lastRecord; ///< The timestamp of the last time this log was recorded.  Used by the telemetry system.

    /// The type of the input message
    struct messageT : public fbMessage
    {
        /// Construct from components
        messageT( const bool  &tracking,    /**< [in] whether ADC tracking is enabled */
                  const float &delta_angle, /**< [in] the shared ADC delta angle */
                  const float &adc1_delta,  /**< [in] the ADC 1-specific delta angle */
                  const float &adc2_delta,  /**< [in] the ADC 2-specific delta angle */
                  const float &min_zd       /**< [in] the minimum tracked zenith distance */
        )
        {
            auto fp = CreateTelem_adctrack_fb( builder, tracking, delta_angle, adc1_delta, adc2_delta, min_zd );
            builder.Finish( fp );
        }
    };

    static bool verify( flatlogs::bufferPtrT &logBuff, ///< [in] Buffer containing the flatbuffer serialized message.
                        flatlogs::msgLenT     len      ///< [in] length of msgBuffer.
    )
    {
        auto verifier = flatbuffers::Verifier( static_cast<uint8_t *>( flatlogs::logHeader::messageBuffer( logBuff ) ),
                                               static_cast<size_t>( len ) );
        return VerifyTelem_adctrack_fbBuffer( verifier );
    }

    /// Get the message formatted for human consumption.
    static std::string msgString( void *msgBuffer,      /**< [in] Buffer containing the flatbuffer serialized message.*/
                                  flatlogs::msgLenT len /**< [in] [unused] length of msgBuffer.*/
    )
    {
        static_cast<void>( len );

        auto fbs = GetTelem_adctrack_fb( msgBuffer );

        std::string msg = "[adctrack] ";

        msg += "tracking: ";
        msg += fbs->tracking() ? "On " : "Off ";

        msg += "delta_angle: ";
        msg += std::to_string( fbs->delta_angle() ) + " ";

        msg += "adc1_delta: ";
        msg += std::to_string( fbs->adc1_delta() ) + " ";

        msg += "adc2_delta: ";
        msg += std::to_string( fbs->adc2_delta() ) + " ";

        msg += "min_zd: ";
        msg += std::to_string( fbs->min_zd() ) + " ";

        return msg;
    }

    static bool tracking( void *msgBuffer )
    {
        auto fbs = GetTelem_adctrack_fb( msgBuffer );
        return fbs->tracking();
    }

    static float delta_angle( void *msgBuffer )
    {
        auto fbs = GetTelem_adctrack_fb( msgBuffer );
        return fbs->delta_angle();
    }

    static float adc1_delta( void *msgBuffer )
    {
        auto fbs = GetTelem_adctrack_fb( msgBuffer );
        return fbs->adc1_delta();
    }

    static float adc2_delta( void *msgBuffer )
    {
        auto fbs = GetTelem_adctrack_fb( msgBuffer );
        return fbs->adc2_delta();
    }

    static float min_zd( void *msgBuffer )
    {
        auto fbs = GetTelem_adctrack_fb( msgBuffer );
        return fbs->min_zd();
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
        else if( member == "delta_angle" )
        {
            return logMetaDetail( { "DELTA ANGLE",
                                    logMeta::valTypes::Float,
                                    logMeta::metaTypes::State,
                                    reinterpret_cast<void *>( &delta_angle ) } );
        }
        else if( member == "adc1_delta" )
        {
            return logMetaDetail( { "ADC1 DELTA",
                                    logMeta::valTypes::Float,
                                    logMeta::metaTypes::State,
                                    reinterpret_cast<void *>( &adc1_delta ) } );
        }
        else if( member == "adc2_delta" )
        {
            return logMetaDetail( { "ADC2 DELTA",
                                    logMeta::valTypes::Float,
                                    logMeta::metaTypes::State,
                                    reinterpret_cast<void *>( &adc2_delta ) } );
        }
        else if( member == "min_zd" )
        {
            return logMetaDetail( { "MIN ZD",
                                    logMeta::valTypes::Float,
                                    logMeta::metaTypes::State,
                                    reinterpret_cast<void *>( &min_zd ) } );
        }
        else
        {
            std::cerr << "No member " << member << " in telem_adctrack\n";
            return logMetaDetail();
        }
    }

}; // telem_adctrack

} // namespace logger
} // namespace MagAOX

#endif // logger_types_telem_adctrack_hpp
