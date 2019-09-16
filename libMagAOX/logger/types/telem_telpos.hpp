/** \file telem_telpos.hpp
  * \brief The MagAO-X logger telem_telpos log type.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_types_files
  * 
  * History:
  * - 2018-09-06 created by JRM
  */
#ifndef logger_types_telem_telpos_hpp
#define logger_types_telem_telpos_hpp

#include "generated/telem_telpos_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{


/// Log entry recording the build-time git state.
/** \ingroup logger_types
  */
struct telem_telpos : public flatbuffer_log
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
      messageT( const double & epoch, ///<[in] epoch
                const double & ra,    ///<[in] right ascension
                const double & dec,   ///<[in] declination
                const double & el,    ///<[in] elevation
                const double & ha,    ///<[in] hour angle
                const double & am,    ///<[in] air mass
                const double & rotoff ///<[in] rotoff
              )
      {
         auto fp = CreateTelem_telpos_fb(builder, epoch, ra, dec, el, ha, am, rotoff);
         builder.Finish(fp);
      }

   };
                 
 
   ///Get the message formatte for human consumption.
   static std::string msgString( void * msgBuffer,  /**< [in] Buffer containing the flatbuffer serialized message.*/
                                 flatlogs::msgLenT len  /**< [in] [unused] length of msgBuffer.*/
                               )
   {
      static_cast<void>(len);

      auto fbs = GetTelem_telpos_fb(msgBuffer);

      std::string msg = "[telpos] ";
      
      msg += "ep: ";
      msg += std::to_string(fbs->epoch()) + " ";
      
      msg += "ra: ";
      msg += std::to_string(fbs->ra()) + " ";
      
      msg += "dec: ";
      msg += std::to_string(fbs->dec()) + " ";
      
      msg += "el: ";
      msg += std::to_string(fbs->el()) + " ";
      
      msg += "ha: ";
      msg += std::to_string(fbs->ha()) + " ";
      
      msg += "am: ";
      msg += std::to_string(fbs->am()) + " ";
      
      msg += "ro: ";
      msg += std::to_string(fbs->rotoff());
      
      return msg;
   
   }
   
}; //telem_telpos

timespec telem_telpos::lastRecord = {0,0};

} //namespace logger
} //namespace MagAOX

#endif //logger_types_telem_telpos_hpp

