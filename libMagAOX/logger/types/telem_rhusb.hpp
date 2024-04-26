/** \file telem_rhusb.hpp
  * \brief The MagAO-X logger telem_rhusb log type.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_types_files
  * 
  * History:
  * - 2021-08-22 created by JRM
  */
#ifndef logger_types_telem_rhusb_hpp
#define logger_types_telem_rhusb_hpp

#include "generated/telem_rhusb_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{


/// Log entry recording the build-time git state.
/** \ingroup logger_types
  */
struct telem_rhusb : public flatbuffer_log
{
   ///The event code
   static const flatlogs::eventCodeT eventCode = eventCodes::TELEM_RHUSB;

   ///The default level
   static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_TELEM;

   static timespec lastRecord; ///< The time of the last time this log was recorded.  Used by the telemetry system.
   
   ///The type of the input message
   struct messageT : public fbMessage
   {
      ///Construct from components
      messageT( const float & temp, ///< [in] 
                const float & rh    ///< [in] 
              )
      {
         auto fp = CreateTelem_rhusb_fb(builder, temp, rh);
         builder.Finish(fp);
      }

   };
                 
   static bool verify( flatlogs::bufferPtrT & logBuff,  ///< [in] Buffer containing the flatbuffer serialized message.
                       flatlogs::msgLenT len            ///< [in] length of msgBuffer.
                     )
   {
      auto verifier = flatbuffers::Verifier( static_cast<uint8_t*>(flatlogs::logHeader::messageBuffer(logBuff)), static_cast<size_t>(len));
      return VerifyTelem_rhusb_fbBuffer(verifier);
   }

   ///Get the message formatte for human consumption.
   static std::string msgString( void * msgBuffer,  /**< [in] Buffer containing the flatbuffer serialized message.*/
                                 flatlogs::msgLenT len  /**< [in] [unused] length of msgBuffer.*/
                               )
   {
      static_cast<void>(len);

      auto fbs = GetTelem_rhusb_fb(msgBuffer);

      std::string msg = "[rhusb] ";
     
      
      
      msg += "temp: ";
      msg += std::to_string(fbs->temp()) + "C ";
      
      msg += "RH: ";
      msg += std::to_string(fbs->rh()) + "%";
      
      
      return msg;
   
   }

   static float temp(void * msgBuffer)
   {
      auto fbs = GetTelem_rhusb_fb(msgBuffer);
      return fbs->temp();
   }
   
   static float rh(void * msgBuffer)
   {
      auto fbs = GetTelem_rhusb_fb(msgBuffer);
      return fbs->rh();
   }
   
   /// Get pointer to the accessor for a member by name 
   /**
     * \returns the function pointer cast to void*
     * \returns -1 for an unknown member
     */ 
   static void * getAccessor( const std::string & member /**< [in] the name of the member */ )
   {
      if(member == "temp") return reinterpret_cast<void*>(&temp);
      else if(member == "rh") return reinterpret_cast<void*>(&rh);
      else
      {
         std::cerr << "No string member " << member << " in telem_rhusb\n";
         return 0;
      }
   }
   

}; //telem_rhusb



} //namespace logger
} //namespace MagAOX

#endif //logger_types_telem_rhusb_hpp

