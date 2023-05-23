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
  * \ingroup logger_types_basic
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

   static bool verify( flatlogs::bufferPtrT & logBuff,  ///< [in] Buffer containing the flatbuffer serialized message.
                       flatlogs::msgLenT len            ///< [in] length of msgBuffer.
                     )
   {
      auto verifier = flatbuffers::Verifier( (uint8_t*) flatlogs::logHeader::messageBuffer(logBuff), static_cast<size_t>(len));
      return VerifyString_log_fbBuffer(verifier);
   }

   ///Get the message formatted for human consumption.
   static std::string msgString( void * msgBuffer,  /**< [in] Buffer containing the flatbuffer serialized message.*/
                                 flatlogs::msgLenT len  /**< [in] [unused] length of msgBuffer.*/
                               )
   {
      static_cast<void>(len);
      
      auto rgs = GetString_log_fb(msgBuffer);
      
      if(rgs->message() == nullptr) return "";
      else return rgs->message()->c_str();
   }

};




} //namespace logger
} //namespace MagAOX

#endif //logger_types_string_log_hpp
