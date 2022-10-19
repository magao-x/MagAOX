/** \file user_log.hpp
  * \brief The MagAO-X logger user_log log type.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_types_files
  * 
  * History:
  * - 2018-08-18 created by JRM
  */
#ifndef logger_types_user_log_hpp
#define logger_types_user_log_hpp

#include "string_log.hpp"

namespace MagAOX
{
namespace logger
{

///User entered log, a string-type log.
/** \ingroup logger_types
  */
struct user_log : public string_log
{
   ///The event code
   static const flatlogs::eventCodeT eventCode = eventCodes::USER_LOG;

   ///The default level
   static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_INFO;

   ///Get the message formatte for human consumption.
   static std::string msgString( void * msgBuffer,  /**< [in] Buffer containing the flatbuffer serialized message.*/
                                 flatlogs::msgLenT len  /**< [in] [unused] length of msgBuffer.*/
                               )
   {
      std::string msg;
      msg = string_log::msgString(msgBuffer, len);
      
      std::string nmsg = "USER: ";
      return nmsg + msg;
   }
};


} //namespace logger
} //namespace MagAOX

#endif //logger_types_user_log_hpp
