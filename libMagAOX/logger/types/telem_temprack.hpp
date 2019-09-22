/** \file telem_temprack.hpp
  * \brief The MagAO-X logger telem_temprack log type.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_types_files
  * 
  * History:
  * - 2018-09-06 created by JRM
  */
#ifndef logger_types_telem_temprack_hpp
#define logger_types_telem_temprack_hpp

#include "generated/telem_temprack_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{


/// Log entry recording electronics rack temperature
/** \ingroup logger_types
  */
struct telem_temprack : public flatbuffer_log
{
   ///The event code
   static const flatlogs::eventCodeT eventCode = eventCodes::TELEM_TELPOS;

   ///The default level
   static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_TELEM;

   static timespec lastRecord; ///< The time of the last time this log was recorded.  Used by the telemetry system.

   ///The type of the input message
   struct messageT : public fbMessage
   {
      ///Construct from components
      messageT( const double & lower,  ///<[in] lower rack temp
                const double & middle, ///<[in] middle rack temp
                const double & upper   ///<[in] upper rack temp
              )
      {
         auto fp = CreateTelem_temprack_fb(builder, lower, middle, upper);
         builder.Finish(fp);
      }

   };
                 
 
   ///Get the message formatte for human consumption.
   static std::string msgString( void * msgBuffer,  /**< [in] Buffer containing the flatbuffer serialized message.*/
                                 flatlogs::msgLenT len  /**< [in] [unused] length of msgBuffer.*/
                               )
   {
      static_cast<void>(len);

      auto fbs = GetTelem_temprack_fb(msgBuffer);

      std::string msg = "[temprack] ";
      
      msg += "lr: ";
      msg += std::to_string(fbs->lower()) + " ";
      
      msg += "md: ";
      msg += std::to_string(fbs->middle()) + " ";
      
      msg += "up: ";
      msg += std::to_string(fbs->upper()) + " ";
      
      return msg;
   
   }
   
}; //telem_temprack

timespec telem_temprack::lastRecord = {0,0};

} //namespace logger
} //namespace MagAOX

#endif //logger_types_telem_temprack_hpp

