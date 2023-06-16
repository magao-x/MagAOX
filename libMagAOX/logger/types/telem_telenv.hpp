/** \file telem_telenv.hpp
  * \brief The MagAO-X logger telem_telenv log type.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_types_files
  * 
  * History:
  * - 2018-09-06 created by JRM
  */
#ifndef logger_types_telem_telenv_hpp
#define logger_types_telem_telenv_hpp

#include "generated/telem_telenv_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{


/// Log entry recording the build-time git state.
/** \ingroup logger_types
  */
struct telem_telenv : public flatbuffer_log
{
   ///The event code
   static const flatlogs::eventCodeT eventCode = eventCodes::TELEM_TELENV;

   ///The default level
   static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_TELEM;

   static timespec lastRecord; ///< The time of the last time this log was recorded.  Used by the telemetry system.

   ///The type of the input message
   struct messageT : public fbMessage
   {
      ///Construct from components
      messageT( const double & tempout,     ///< [in] 
                const double & pressure,    ///< [in] 
                const double & humidity,    ///< [in] 
                const double & wind,        ///< [in] 
                const double & winddir,     ///< [in] 
                const double & temptruss,   ///< [in] 
                const double & tempcell,    ///< [in] 
                const double & tempseccell, ///< [in] 
                const double & tempamb,     ///< [in] 
                const double & dewpoint     ///< [in] 
              )
      {
         auto fp = CreateTelem_telenv_fb(builder, tempout, pressure,  humidity, wind,winddir,temptruss,tempcell,tempseccell,tempamb, dewpoint);
         builder.Finish(fp);
      }

   };
                 
   static bool verify( flatlogs::bufferPtrT & logBuff,  ///< [in] Buffer containing the flatbuffer serialized message.
                       flatlogs::msgLenT len            ///< [in] length of msgBuffer.
                     )
   {
      auto verifier = flatbuffers::Verifier( (uint8_t*) flatlogs::logHeader::messageBuffer(logBuff), static_cast<size_t>(len));
      return VerifyTelem_telenv_fbBuffer(verifier);
   }

   ///Get the message formatte for human consumption.
   static std::string msgString( void * msgBuffer,  /**< [in] Buffer containing the flatbuffer serialized message.*/
                                 flatlogs::msgLenT len  /**< [in] [unused] length of msgBuffer.*/
                               )
   {
      static_cast<void>(len);

      auto fbs = GetTelem_telenv_fb(msgBuffer);

      std::string msg = "[telenv] ";
      
      msg += "tempout: ";
      msg += std::to_string(fbs->tempout()) + " ";
      
      msg += "press: ";
      msg += std::to_string(fbs->pressure()) + " ";
      
      msg += "humidity: ";
      msg += std::to_string(fbs->humidity()) + " ";
      
      msg += "wind: ";
      msg += std::to_string(fbs->wind()) + " ";
      
      msg += "winddir: ";
      msg += std::to_string(fbs->winddir()) + " ";
      
      msg += "temptruss: ";
      msg += std::to_string(fbs->temptruss()) + " ";
      
      msg += "tempcell: ";
      msg += std::to_string(fbs->tempcell()) + " ";
      
      msg += "tempseccell: ";
      msg += std::to_string(fbs->tempseccell()) + " ";
      
      msg += "tempamb: ";
      msg += std::to_string(fbs->tempamb()) + " ";
      
      msg += "dewpoint: ";
      msg += std::to_string(fbs->dewpoint()) + " ";
      
      
      return msg;
   
   }
   
}; //telem_telenv


} //namespace logger
} //namespace MagAOX

#endif //logger_types_telem_telenv_hpp

