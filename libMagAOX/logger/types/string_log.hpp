/** \file string_log.hpp
  * \brief The MagAO-X logger string_log log type.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_types_files
  * 
  * History:
  * - 2018-08-18 created by JRM
  */
#ifndef logger_types_string_log_hpp
#define logger_types_string_log_hpp

#include "generated/string_log_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{

namespace logger
{
   
   
   
/// Base class for logs consisting of a string message.
/** Does not have eventCode or defaultLevel, so this can not be used as a log type in logger.
  *
  * \ingroup logtypesbasics
  */
struct string_log : public flatbuffer_log
{
   ///The type of the message
   struct messageT : public fbMessage
   {
      messageT( const char * msg )
      {
         auto _msg = builder.CreateString(msg);
         
         auto gs = CreateString_log_fb(builder, _msg);
         builder.Finish(gs);
      }
      
      messageT( const std::string & msg )
      {
         auto _msg = builder.CreateString(msg);
         
         auto gs = CreateString_log_fb(builder, _msg);
         builder.Finish(gs);
      }
   };

  
   static std::string msgString(void * msgBuffer, flatlogs::msgLenT len)
   {
      static_cast<void>(len);
      
      auto rgs = GetString_log_fb(msgBuffer);
      
      return rgs->message()->c_str();
   }

};




} //namespace logger
} //namespace MagAOX

#endif //logger_types_string_log_hpp
