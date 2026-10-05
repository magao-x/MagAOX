/** \file telem_modalgainopt.hpp
 * \brief The MagAO-X logger telem_modalgainopt log type.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup logger_types_files
 */

#ifndef logger_types_telem_modalgainopt_hpp
#define logger_types_telem_modalgainopt_hpp

#include "generated/telem_modalgainopt_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{

/// Log entry recording modalGainOpt control state.
/** \ingroup logger_types
 */
struct telem_modalgainopt : public flatbuffer_log
{
    static const flatlogs::eventCodeT eventCode = eventCodes::TELEM_MODALGAINOPT;
    static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_TELEM;

    static timespec lastRecord; ///< The time of the last telemetry record.

    struct messageT : public fbMessage
    {
        messageT( const bool updateAuto,          ///< [in] whether update_auto is enabled
                  const bool opticalGainTracking, ///< [in] whether optical gain tracking is enabled
                  const float opticalGain,        ///< [in] in-use optical gain
                  const float gainGain,           ///< [in] SI gain integrator gain coefficient
                  const float gainLeak            ///< [in] SI gain integrator leak coefficient
        )
        {
            auto fp = CreateTelem_modalgainopt_fb( builder,
                                                   updateAuto,
                                                   opticalGainTracking,
                                                   opticalGain,
                                                   gainGain,
                                                   gainLeak );
            builder.Finish( fp );
        }
    };

    static bool verify( flatlogs::bufferPtrT &logBuff, flatlogs::msgLenT len )
    {
        auto verifier = flatbuffers::Verifier( static_cast<uint8_t *>( flatlogs::logHeader::messageBuffer( logBuff ) ),
                                               static_cast<size_t>( len ) );
        return VerifyTelem_modalgainopt_fbBuffer( verifier );
    }

    static std::string msgString( void *msgBuffer, flatlogs::msgLenT len )
    {
        static_cast<void>( len );

        auto fbs = GetTelem_modalgainopt_fb( msgBuffer );
        std::string msg = "[modalgainopt] ";

        msg += "updateAuto: ";
        msg += ( fbs->update_auto() ? "true" : "false" );
        msg += " opticalGainTracking: ";
        msg += ( fbs->optical_gain_tracking() ? "true" : "false" );
        msg += " opticalGain: ";
        msg += std::to_string( fbs->optical_gain() );
        msg += " gainGain: ";
        msg += std::to_string( fbs->gain_gain() );
        msg += " gainLeak: ";
        msg += std::to_string( fbs->gain_leak() );

        return msg;
    }

    static bool updateAuto( void *msgBuffer )
    {
        return GetTelem_modalgainopt_fb( msgBuffer )->update_auto();
    }

    static bool opticalGainTracking( void *msgBuffer )
    {
        return GetTelem_modalgainopt_fb( msgBuffer )->optical_gain_tracking();
    }

    static float opticalGain( void *msgBuffer )
    {
        return GetTelem_modalgainopt_fb( msgBuffer )->optical_gain();
    }

    static float gainGain( void *msgBuffer )
    {
        return GetTelem_modalgainopt_fb( msgBuffer )->gain_gain();
    }

    static float gainLeak( void *msgBuffer )
    {
        return GetTelem_modalgainopt_fb( msgBuffer )->gain_leak();
    }

    static logMetaDetail getAccessor( const std::string &member )
    {
        if( member == "updateAuto" )
        {
            return logMetaDetail( { "UPDATE AUTO",
                                    "",
                                    logMeta::valTypes::Bool,
                                    logMeta::metaTypes::State,
                                    reinterpret_cast<void *>( &updateAuto ),
                                    true } );
        }
        else if( member == "opticalGainTracking" )
        {
            return logMetaDetail( { "OPTICAL GAIN TRACKING",
                                    "",
                                    logMeta::valTypes::Bool,
                                    logMeta::metaTypes::State,
                                    reinterpret_cast<void *>( &opticalGainTracking ),
                                    true } );
        }
        else if( member == "opticalGain" )
        {
            return logMetaDetail( { "OPTICAL GAIN",
                                    "",
                                    logMeta::valTypes::Float,
                                    logMeta::metaTypes::State,
                                    reinterpret_cast<void *>( &opticalGain ),
                                    true } );
        }
        else if( member == "gainGain" )
        {
            return logMetaDetail( { "GAIN GAIN",
                                    "",
                                    logMeta::valTypes::Float,
                                    logMeta::metaTypes::State,
                                    reinterpret_cast<void *>( &gainGain ),
                                    true } );
        }
        else if( member == "gainLeak" )
        {
            return logMetaDetail( { "GAIN LEAK",
                                    "",
                                    logMeta::valTypes::Float,
                                    logMeta::metaTypes::State,
                                    reinterpret_cast<void *>( &gainLeak ),
                                    true } );
        }
        else
        {
            std::cerr << "No member " << member << " in telem_modalgainopt\n";
            return logMetaDetail();
        }
    }
};

} // namespace logger
} // namespace MagAOX

#endif // logger_types_telem_modalgainopt_hpp
