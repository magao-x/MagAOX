/** \file telem_pokeloop.hpp
 * \brief The MagAO-X logger telem_pokeloop log type.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup logger_types_files
 *
 */
#ifndef logger_types_telem_pokeloop_hpp
#define logger_types_telem_pokeloop_hpp

#include "generated/telem_pokeloop_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{

/// Log entry recording DM poke centering results
/** \ingroup logger_types
 */
struct telem_pokeloop : public flatbuffer_log
{
    /// The event code
    static const flatlogs::eventCodeT eventCode = eventCodes::TELEM_POKELOOP;

    /// The default level
    static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_TELEM;

    static timespec lastRecord; ///< The time of the last time this log was recorded.  Used by the telemetry system.

    /// The type of the input message
    struct messageT : public fbMessage
    {
        /// Construct from components
        messageT( const uint8_t & measuring, ///< [in] flag indicating if the WFS is measuring
                  const float & deltaX,      ///< [in] the x delta position
                  const float & deltaY,      ///< [in] the y delta position
                  const uint64_t & counter   ///< [in] the loop counter
                )
        {

            if(measuring == 0)
            {
                Telem_pokeloop_fbBuilder telem_pokeloop_builder(builder);
                telem_pokeloop_builder.add_measuring(measuring);
                auto fb = telem_pokeloop_builder.Finish();
                builder.Finish(fb);
                return;
            }

            auto fb = CreateTelem_pokeloop_fb(builder, measuring, deltaX, deltaY, counter);

            builder.Finish(fb);
        }
    };

    static bool verify( flatlogs::bufferPtrT &logBuff, ///< [in] Buffer containing the flatbuffer serialized message.
                        flatlogs::msgLenT len          ///< [in] length of msgBuffer.
                      )
    {
        auto verifier = flatbuffers::Verifier(static_cast<uint8_t *>(flatlogs::logHeader::messageBuffer(logBuff)), static_cast<size_t>(len));
        return VerifyTelem_pokeloop_fbBuffer(verifier);
    }

    /// Get the message formatted for human consumption.
    static std::string msgString( void *msgBuffer,      /**< [in] Buffer containing the flatbuffer serialized message.*/
                                  flatlogs::msgLenT len /**< [in] [unused] length of msgBuffer.*/
                                )
    {
        static_cast<void>(len);

        auto fbs = GetTelem_pokeloop_fb(msgBuffer);

        std::string msg;
        if(fbs->measuring() == 0)
        {
            msg = "[pokeloop] not measuring";
        }
        else
        {
            msg = "[pokeloop] X: ";

            msg += std::to_string(fbs->deltaX());
            msg += " Y: ";
            msg += std::to_string(fbs->deltaY());

            msg += " ctr: ";
            msg += std::to_string(fbs->counter());
        }

        return msg;
    }

    static bool measuring( void * msgBuffer /**< [in] Buffer containing the flatbuffer serialized message.*/ )
    {
        auto fbs = GetTelem_pokeloop_fb(msgBuffer);
        return fbs->measuring();
    }

    static float deltaX( void * msgBuffer /**< [in] Buffer containing the flatbuffer serialized message.*/ )
    {
        auto fbs = GetTelem_pokeloop_fb(msgBuffer);
        return fbs->deltaX();
    }
    static float deltaY( void * msgBuffer /**< [in] Buffer containing the flatbuffer serialized message.*/ )
    {
        auto fbs = GetTelem_pokeloop_fb(msgBuffer);
        return fbs->deltaY();
    }

    static unsigned long long counter( void * msgBuffer /**< [in] Buffer containing the flatbuffer serialized message.*/ )
    {
        auto fbs = GetTelem_pokeloop_fb(msgBuffer);
        return fbs->counter();
    }

    /// Get the logMetaDetail for a member by name
    /**
      * \returns the a logMetaDetail filled in with the appropriate details
      * \returns an empty logMetaDetail if member not recognized
      */
    static logMetaDetail getAccessor( const std::string & member /**< [in] the name of the member */ )
    {
       if( member == "measuring") return logMetaDetail({"MEASURING", "measuring state", logMeta::valTypes::Bool, logMeta::metaTypes::State, reinterpret_cast<void*>(&measuring), false});
       else if( member == "deltaX") return logMetaDetail({"DELTA X", "loop delta x", logMeta::valTypes::Float, logMeta::metaTypes::State, reinterpret_cast<void*>(&deltaX), false});
       else if( member == "deltaY") return logMetaDetail({"DELTA Y", "loop delta y", logMeta::valTypes::Float, logMeta::metaTypes::State, reinterpret_cast<void*>(&deltaY), false});
       else if( member == "counter") return logMetaDetail({"LOOP COUNTER", "loop counter", logMeta::valTypes::ULongLong, logMeta::metaTypes::State, reinterpret_cast<void*>(&counter), false});
       else
       {
          std::cerr << "No member " << member << " in telem_pokeloop\n";
          return logMetaDetail();
       }
    }

}; // telem_pokeloop

} // namespace logger
} // namespace MagAOX

#endif // logger_types_telem_pokeloop_hpp
