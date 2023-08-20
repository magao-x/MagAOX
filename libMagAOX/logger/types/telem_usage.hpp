/** \file telem_usage.hpp
  * \brief The MagAO-X logger telem_usage log type.
  * \author Chris Bohlman (cbohlmanaz@gmail.com)
  *
  * \ingroup logger_types_files
  *
  * History:
  * - 2018-10-15 created by CJB
  */
#ifndef logger_types_telem_usage_hpp
#define logger_types_telem_usage_hpp

#include "generated/telem_usage_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{


/// Log entry recording hdd temperatures
/** \ingroup logger_types
  */
struct telem_usage : public flatbuffer_log
{

  static const flatlogs::eventCodeT eventCode = eventCodes::TELEM_USAGE;
  static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_TELEM;

  static timespec lastRecord; ///< The time of the last time this log was recorded.  Used by the telemetry system.
  
   ///The type of the input message
   struct messageT : public fbMessage
   {
      ///Construct from components
      messageT( float ram,
                float boot,
                float root,
                float data
              )
      {
         
         auto fp = CreateTelem_usage_fb(builder, ram, boot, root, data );
         
         builder.Finish(fp);

      }

   };

   static bool verify( flatlogs::bufferPtrT & logBuff,  ///< [in] Buffer containing the flatbuffer serialized message.
                       flatlogs::msgLenT len            ///< [in] length of msgBuffer.
                     )
   {
      auto verifier = flatbuffers::Verifier( (uint8_t*) flatlogs::logHeader::messageBuffer(logBuff), static_cast<size_t>(len));
      return VerifyTelem_usage_fbBuffer(verifier);
   }

   ///Get the message formatted for human consumption.
   static std::string msgString( void * msgBuffer,  /**< [in] Buffer containing the flatbuffer serialized message.*/
                                 flatlogs::msgLenT len  /**< [in] [unused] length of msgBuffer.*/
                               )
   {

      static_cast<void>(len); // unused by most log types
   
      auto rgs = GetTelem_usage_fb(msgBuffer);  
      
      std::string msg = "[usage] ram: ";

      msg += std::to_string(rgs->ramUsage());
      msg += " boot: ";
      msg += std::to_string(rgs->bootUsage());
      msg += " root: ";
      msg += std::to_string(rgs->rootUsage());
      msg += " data: ";
      msg += std::to_string(rgs->dataUsage());
      
      return msg;

   }

}; //telem_usage



} //namespace logger
} //namespace MagAOX

#endif //logger_types_telem_usage_hpp
