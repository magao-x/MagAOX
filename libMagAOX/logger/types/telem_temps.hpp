/** \file telem_temps.hpp
  * \brief The MagAO-X logger telem_temps log type.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_types_files
  * 
  * History:
  * - 2018-09-06 created by JRM
  */
#ifndef logger_types_telem_temps_hpp
#define logger_types_telem_temps_hpp

#include "generated/telem_temps_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{


/// Log entry recording electronics rack temperature
/** \ingroup logger_types
  */
struct telem_temps : public flatbuffer_log
{
   ///The event code
   static const flatlogs::eventCodeT eventCode = eventCodes::TELEM_TEMPS;

   ///The default level
   static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_TELEM;

   static timespec lastRecord; ///< The time of the last time this log was recorded.  Used by the telemetry system.

   ///The type of the input message
   struct messageT : public fbMessage
   {
      ///Construct from components
      messageT( const std::vector<float> & temps  ///<[in] vector of temperatures
              )
      {
         auto _temps = builder.CreateVector(temps);
         auto fp = CreateTelem_temps_fb(builder, _temps);
         
         builder.Finish(fp);
      }

   };

   static bool verify( flatlogs::bufferPtrT & logBuff,  ///< [in] Buffer containing the flatbuffer serialized message.
                       flatlogs::msgLenT len            ///< [in] length of msgBuffer.
                     )
   {
      auto verifier = flatbuffers::Verifier( (uint8_t*) flatlogs::logHeader::messageBuffer(logBuff), static_cast<size_t>(len));
      return VerifyTelem_temps_fbBuffer(verifier);
   }
 
   ///Get the message formatte for human consumption.
   static std::string msgString( void * msgBuffer,  /**< [in] Buffer containing the flatbuffer serialized message.*/
                                 flatlogs::msgLenT len  /**< [in] [unused] length of msgBuffer.*/
                               )
   {
      static_cast<void>(len);

      auto fbs = GetTelem_temps_fb(msgBuffer);

      std::string msg = "[temps] ";
      
      
      if( fbs->temps() )
      {
         for(size_t i=0; i< fbs->temps()->Length(); ++i)
         {
            msg += " ";
            msg += std::to_string(fbs->temps()->Get(i));
         }
      }
            
      return msg;
   
   }
   
}; //telem_temps



} //namespace logger
} //namespace MagAOX

#endif //logger_types_telem_temps_hpp

