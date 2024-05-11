/** \file config_log.hpp
  * \brief The MagAO-X logger config_log log type.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_types_files
  * 
  * History:
  * - 2018-09-06 created by JRM
  */
#ifndef logger_types_config_log_hpp
#define logger_types_config_log_hpp

#include "generated/config_log_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{


/// Log entry recording configuration settings at startup
/** \ingroup logger_types
  */
struct config_log : public flatbuffer_log
{
   ///The event code
   static const flatlogs::eventCodeT eventCode = eventCodes::CONFIG_LOG;

   ///The default level
   static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_INFO;


   ///The type of the input message
   struct messageT : public fbMessage
   {
      ///Construct from components
      messageT( const std::string & name,
                const int & code,
                const std::string & value,
                const std::string & source
              )
      {
         auto _name = builder.CreateString(name);
         auto _value = builder.CreateString(value);
         auto _source = builder.CreateString(source);
         
         
         auto fp = CreateConfig_log_fb(builder, _name, code, _value, _source);
         builder.Finish(fp);

      }

   };
   
   static bool verify( flatlogs::bufferPtrT & logBuff,  ///< [in] Buffer containing the flatbuffer serialized message.
                       flatlogs::msgLenT len            ///< [in] length of msgBuffer.
                     )
   {
      auto verifier = flatbuffers::Verifier( static_cast<uint8_t*>(flatlogs::logHeader::messageBuffer(logBuff)), static_cast<size_t>(len));
      return VerifyConfig_log_fbBuffer(verifier);
   }

   ///Get the message formatte for human consumption.
   static std::string msgString( void * msgBuffer,  /**< [in] Buffer containing the flatbuffer serialized message.*/
                                 flatlogs::msgLenT len  /**< [in] [unused] length of msgBuffer.*/
                               )
   {
      static_cast<void>(len);

      auto fbs = GetConfig_log_fb(msgBuffer);

      std::string msg = "Config: ";
      
      if(fbs->name())
      {
         msg += fbs->name()->c_str();
      }
      
      msg += "=";
      
      if(fbs->value())
      {
         msg += fbs->value()->c_str();
      }
      
      msg += " [";
      
      if(fbs->source())
      {
         msg += fbs->source()->c_str();
      }
      
      msg += "]";
      
      return msg;
   
   }
   
}; //config_log


} //namespace logger
} //namespace MagAOX

#endif //logger_types_config_log_hpp

