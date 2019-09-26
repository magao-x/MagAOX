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
      messageT( const double & pos,     ///<[in] stage position in mm
                const double & rawPos,  ///<[in] stage raw position, in counts
                const double & temp     ///<[in] stage temperature
              )
      {         
         auto fp = CreateTelem_zaber_fb(builder, pos, rawPos, temp);
         builder.Finish(fp);
      }

   };
                 
 
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
      msg += fbs->temp();
      
      return msg;
   
   }
   
}; //telem_zaber

timespec telem_zaber::lastRecord = {0,0};

} //namespace logger
} //namespace MagAOX

#endif //logger_types_telem_zaber_hpp

