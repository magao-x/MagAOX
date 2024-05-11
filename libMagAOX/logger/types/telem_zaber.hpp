/** \file telem_zaber.hpp
  * \brief The MagAO-X logger telem_zaber log type.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_types_files
  * 
  * History:
  * - 2018-09-06 created by JRM
  */
#ifndef logger_types_telem_zaber_hpp
#define logger_types_telem_zaber_hpp

#include "generated/telem_zaber_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{


/// Log entry recording zaber stage specific status.
/** \ingroup logger_types
  */
struct telem_zaber : public flatbuffer_log
{
   ///The event code
   static const flatlogs::eventCodeT eventCode = eventCodes::TELEM_ZABER;

   ///The default level
   static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_TELEM;

   static timespec lastRecord; ///< The timestamp of the last time this log was recorded.  Used by the telemetry system.

   ///The type of the input message
   struct messageT : public fbMessage
   {
      ///Construct from components
      messageT( const float & pos,     ///<[in] stage position in mm
                const float & rawPos,  ///<[in] stage raw position, in counts
                const float & temp     ///<[in] stage temperature
              )
      {         
         auto fp = CreateTelem_zaber_fb(builder, pos, rawPos, temp);
         builder.Finish(fp);
      }

   };
                 
 
   static bool verify( flatlogs::bufferPtrT & logBuff,  ///< [in] Buffer containing the flatbuffer serialized message.
                       flatlogs::msgLenT len            ///< [in] length of msgBuffer.
                     )
   {
      auto verifier = flatbuffers::Verifier( static_cast<uint8_t*>(flatlogs::logHeader::messageBuffer(logBuff)), static_cast<size_t>(len));
      return VerifyTelem_zaber_fbBuffer(verifier);
   }

   ///Get the message formatte for human consumption.
   static std::string msgString( void * msgBuffer,  /**< [in] Buffer containing the flatbuffer serialized message.*/
                                 flatlogs::msgLenT len  /**< [in] [unused] length of msgBuffer.*/
                               )
   {
      static_cast<void>(len);

      auto fbs = GetTelem_zaber_fb(msgBuffer);

      std::string msg = "[zaber] ";
      
      msg += "pos: ";
      msg += std::to_string(fbs->pos()) + " ";
      
      msg += "rawPos: ";
      msg += std::to_string(fbs->rawPos()) + " ";
      
      msg += "temp: ";
      msg += std::to_string(fbs->temp());
      
      return msg;
   
   }

   static float pos( void * msgBuffer )
   {
      auto fbs = GetTelem_zaber_fb(msgBuffer);
      return fbs->pos();
   }

   static float rawPos( void * msgBuffer )
   {
      auto fbs = GetTelem_zaber_fb(msgBuffer);
      return fbs->rawPos();
   }

   static float temp( void * msgBuffer )
   {
      auto fbs = GetTelem_zaber_fb(msgBuffer);
      return fbs->temp();
   }

   /// Get the logMetaDetail for a member by name
   /**
     * \returns the a logMetaDetail filled in with the appropriate details
     * \returns an empty logmegaDetail if member not recognized
     */ 
   static logMetaDetail getAccessor( const std::string & member /**< [in] the name of the member */ )
   {
      if(     member == "pos") return logMetaDetail({"POS", logMeta::valTypes::Float, logMeta::metaTypes::Continuous, reinterpret_cast<void*>(&pos)});
      else if(member == "rawPos") return logMetaDetail({"COUNTS", logMeta::valTypes::Float, logMeta::metaTypes::Continuous, reinterpret_cast<void*>(&rawPos)}); 
      else if(member == "temp") return logMetaDetail({"TEMP", logMeta::valTypes::Float, logMeta::metaTypes::Continuous, reinterpret_cast<void*>(&temp)}); 
      else
      {
         std::cerr << "No string member " << member << " in telem_zaber\n";
         return logMetaDetail();
      }
   }

}; //telem_zaber



} //namespace logger
} //namespace MagAOX

#endif //logger_types_telem_zaber_hpp

