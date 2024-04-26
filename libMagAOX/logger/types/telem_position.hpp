/** \file telem_position.hpp
  * \brief The MagAO-X logger telem_position log type.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_types_files
  * 
  */
#ifndef logger_types_telem_position_hpp
#define logger_types_telem_position_hpp

#include "generated/telem_position_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{


/// Log entry recording position stage specific status.
/** \ingroup logger_types
  */
struct telem_position : public flatbuffer_log
{
   ///The event code
   static const flatlogs::eventCodeT eventCode = eventCodes::TELEM_POSITION;

   ///The default level
   static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_TELEM;

   static timespec lastRecord; ///< The timestamp of the last time this log was recorded.  Used by the telemetry system.

   ///The type of the input message
   struct messageT : public fbMessage
   {
      ///Construct from components
      messageT( const float & pos     /**<[in] stage position in mm */ )
      {         
         auto fp = CreateTelem_position_fb(builder, pos);
         builder.Finish(fp);
      }

   };
                 
 
   static bool verify( flatlogs::bufferPtrT & logBuff,  ///< [in] Buffer containing the flatbuffer serialized message.
                       flatlogs::msgLenT len            ///< [in] length of msgBuffer.
                     )
   {
      auto verifier = flatbuffers::Verifier( static_cast<uint8_t*>(flatlogs::logHeader::messageBuffer(logBuff)), static_cast<size_t>(len));
      return VerifyTelem_position_fbBuffer(verifier);
   }

   ///Get the message formatte for human consumption.
   static std::string msgString( void * msgBuffer,  /**< [in] Buffer containing the flatbuffer serialized message.*/
                                 flatlogs::msgLenT len  /**< [in] [unused] length of msgBuffer.*/
                               )
   {
      static_cast<void>(len);

      auto fbs = GetTelem_position_fb(msgBuffer);

      std::string msg = "[position] ";
      
      msg += "pos: ";
      msg += std::to_string(fbs->pos()) + " ";
            
      return msg;
   
   }

   static float pos( void * msgBuffer )
   {
      auto fbs = GetTelem_position_fb(msgBuffer);
      return fbs->pos();
   }


   /// Get the logMetaDetail for a member by name
   /**
     * \returns the a logMetaDetail filled in with the appropriate details
     * \returns an empty logmegaDetail if member not recognized
     */ 
   static logMetaDetail getAccessor( const std::string & member /**< [in] the name of the member */ )
   {
      if(     member == "pos") return logMetaDetail({"POS", logMeta::valTypes::Float, logMeta::metaTypes::Continuous, reinterpret_cast<void*>(&pos)});
      else
      {
         std::cerr << "No string member " << member << " in telem_position\n";
         return logMetaDetail();
      }
   }

}; //telem_position



} //namespace logger
} //namespace MagAOX

#endif //logger_types_telem_position_hpp

