/** \file telem_loopgain.hpp
  * \brief The MagAO-X logger telem_loopgain log type.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_types_files
  * 
  * History:
  * - 2022-11-28 created by JRM
  */
#ifndef logger_types_telem_loopgain_hpp
#define logger_types_telem_loopgain_hpp

#include "generated/telem_loopgain_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{


/// Log entry recording the build-time git state.
/** \ingroup logger_types
  */
struct telem_loopgain : public flatbuffer_log
{
   ///The event code
   static const flatlogs::eventCodeT eventCode = eventCodes::TELEM_LOOPGAIN;

   ///The default level
   static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_TELEM;

   static timespec lastRecord; ///< The time of the last time this log was recorded.  Used by the telemetry system.
   
   ///The type of the input message
   struct messageT : public fbMessage
   {
      ///Construct from components
      messageT( const uint8_t & state,  ///< [in] 
                const float & gain,     ///< [in]
                const float & multcoef, ///< [in]
                const float & limit     ///< [in] 
              )
      {
         auto fp = CreateTelem_loopgain_fb(builder, state, gain, multcoef, limit);
         builder.Finish(fp);
      }

   };
                 
   static bool verify( flatlogs::bufferPtrT & logBuff,  ///< [in] Buffer containing the flatbuffer serialized message.
                       flatlogs::msgLenT len            ///< [in] length of msgBuffer.
                     )
   {
      auto verifier = flatbuffers::Verifier( static_cast<uint8_t*>(flatlogs::logHeader::messageBuffer(logBuff)), static_cast<size_t>(len));
      return VerifyTelem_loopgain_fbBuffer(verifier);
   }

   ///Get the message formatte for human consumption.
   static std::string msgString( void * msgBuffer,  /**< [in] Buffer containing the flatbuffer serialized message.*/
                                 flatlogs::msgLenT len  /**< [in] [unused] length of msgBuffer.*/
                               )
   {
      static_cast<void>(len);

      auto fbs = GetTelem_loopgain_fb(msgBuffer);

      std::string msg = "[loopgain] ";
     
      
      
      msg += "state: ";
      msg += std::to_string(fbs->state()) + " ";
      
      msg += "gain: ";
      msg += std::to_string(fbs->gain()) + " ";
      
      msg += "multcoef: ";
      msg += std::to_string(fbs->multcoef()) + " ";
      
      msg += "limit: ";
      msg += std::to_string(fbs->limit());

      return msg;
   
   }
      
   static uint8_t state(void * msgBuffer)
   {
      auto fbs = GetTelem_loopgain_fb(msgBuffer);
      return fbs->state();
   }
   
   static float gain(void * msgBuffer)
   {
      auto fbs = GetTelem_loopgain_fb(msgBuffer);
      return fbs->gain();
   }
   
   static float multcoef(void * msgBuffer)
   {
      auto fbs = GetTelem_loopgain_fb(msgBuffer);
      return fbs->multcoef();
   }

   static float limit(void * msgBuffer)
   {
      auto fbs = GetTelem_loopgain_fb(msgBuffer);
      return fbs->limit();
   }

   /// Get the logMetaDetail for a member by name
   /**
     * \returns the a logMetaDetail filled in with the appropriate details
     * \returns an empty logmegaDetail if member not recognized
     */ 
   static logMetaDetail getAccessor( const std::string & member /**< [in] the name of the member */ )
   {
      if(      member == "state")    return logMetaDetail({"LOOP STATE", logMeta::valTypes::Int, logMeta::metaTypes::State, reinterpret_cast<void*>(&state)});
      else if( member == "gain")     return logMetaDetail({"LOOP GAIN", logMeta::valTypes::Float, logMeta::metaTypes::State, reinterpret_cast<void*>(&gain)});
      else if( member == "multcoef") return logMetaDetail({"LOOP MULTCOEF", logMeta::valTypes::Float, logMeta::metaTypes::State, reinterpret_cast<void*>(&multcoef)});
      else if( member == "limit")    return logMetaDetail({"LOOP LIMIT", logMeta::valTypes::Float, logMeta::metaTypes::State, reinterpret_cast<void*>(&limit)});
      else
      {
         std::cerr << "No string member " << member << " in telem_loopgain\n";
         return logMetaDetail();
      }

   }
   

}; //telem_loopgain



} //namespace logger
} //namespace MagAOX

#endif //logger_types_telem_loopgain_hpp

