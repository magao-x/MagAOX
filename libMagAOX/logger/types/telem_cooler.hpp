/** \file telem_cooler.hpp
  * \brief The MagAO-X logger telem_cooler log type.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_types_files
  * 
  * History:
  * - 2018-09-06 created by JRM
  */
#ifndef logger_types_telem_cooler_hpp
#define logger_types_telem_cooler_hpp

#include "generated/telem_cooler_generated.h"
#include "flatbuffer_log.hpp"

#include <cmath>

namespace MagAOX
{
namespace logger
{


/// Log entry recording the build-time git state.
/** \ingroup logger_types
  */
struct telem_cooler : public flatbuffer_log
{
   ///The event code
   static const flatlogs::eventCodeT eventCode = eventCodes::TELEM_COOLER;

   ///The default level
   static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_TELEM;

   static timespec lastRecord; ///< The time of the last time this log was recorded.  Used by the telemetry system.

   enum member{ em_liquidTemp, em_flowRate, em_pumpLevel, em_pumpSpeed, em_fanLevel, em_fanSpeed };
   
   ///The type of the input message
   struct messageT : public fbMessage
   {
      ///Construct from components
      messageT( const float & liqTemp,    ///<[in] liquid temperature
                const float & flowRate,   ///<[in] flow rate
                const uint8_t & pumpLvl,  ///<[in] pump level
                const uint16_t & pumpSpd, ///<[in] pump speed
                const uint8_t & fanLvl,   ///<[in] fan level
                const uint16_t & fanSpd   ///<[in] fan speed
              )
      {
         auto fp = CreateTelem_cooler_fb(builder, liqTemp, flowRate, pumpLvl, pumpSpd, fanLvl, fanSpd);
         builder.Finish(fp);
      }

   };
                 
 
   static bool verify( flatlogs::bufferPtrT & logBuff,  ///< [in] Buffer containing the flatbuffer serialized message.
                       flatlogs::msgLenT len            ///< [in] length of msgBuffer.
                     )
   {
      auto verifier = flatbuffers::Verifier( (uint8_t*) flatlogs::logHeader::messageBuffer(logBuff), static_cast<size_t>(len));
      return VerifyTelem_cooler_fbBuffer(verifier);
   }

   ///Get the message formatte for human consumption.
   static std::string msgString( void * msgBuffer,  /**< [in] Buffer containing the flatbuffer serialized message.*/
                                 flatlogs::msgLenT len  /**< [in] [unused] length of msgBuffer.*/
                               )
   {
      static_cast<void>(len);

      auto fbs = GetTelem_cooler_fb(msgBuffer);

      std::string msg = "[cooler] ";
      
      msg += "temp: ";
      msg += std::to_string(fbs->liquidTemp()) + " C ";
      
      msg += "flow: ";
      msg += std::to_string(fbs->flowRate()) + " L/min ";
      
      msg += "pump lvl: ";
      msg += std::to_string(fbs->pumpLevel()) + " ";
      
      msg += "spd: ";
      msg += std::to_string(fbs->pumpSpeed()) + " RPM ";
      
      msg += "fan lvl: ";
      msg += std::to_string(fbs->fanLevel()) + " ";
      
      msg += "speed: ";
      msg += std::to_string(fbs->fanSpeed()) + " RPM ";
      
      return msg;
   
   }

   static double getDouble( flatlogs::bufferPtrT & buffer,
                            member m 
                          )
   {
      switch(m)
      {
         case em_liquidTemp:
            return GetTelem_cooler_fb(flatlogs::logHeader::messageBuffer(buffer))->liquidTemp();
         case em_flowRate:
            return GetTelem_cooler_fb(flatlogs::logHeader::messageBuffer(buffer))->flowRate();
         case em_pumpLevel:
            return GetTelem_cooler_fb(flatlogs::logHeader::messageBuffer(buffer))->pumpLevel();
         case em_pumpSpeed:
            return GetTelem_cooler_fb(flatlogs::logHeader::messageBuffer(buffer))->pumpSpeed();
         case em_fanLevel:
            return GetTelem_cooler_fb(flatlogs::logHeader::messageBuffer(buffer))->fanLevel();
         case em_fanSpeed:
            return GetTelem_cooler_fb(flatlogs::logHeader::messageBuffer(buffer))->fanSpeed();
         default:
            return nan("");
      }
   }
   
   
}; //telem_cooler



} //namespace logger
} //namespace MagAOX

#endif //logger_types_telem_cooler_hpp

