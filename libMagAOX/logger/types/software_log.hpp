/** \file software_log.hpp
  * \brief The MagAO-X logger software_log log type.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_types_files
  * 
  * History:
  * - 2018-08-18 created by JRM
  */
#ifndef logger_types_software_log_hpp
#define logger_types_software_log_hpp

#include "generated/software_log_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{

///Base class for software logs
/** Such logs are used to log software status, warnings, and errors. Does not have eventCode or defaultLevel, so this can not be used as a log type in logger.
  *
  * \ingroup logtypesbasics
  */
struct software_log : public flatbuffer_log
{
   ///The event code
   static const flatlogs::eventCodeT eventCode = eventCodes::SOFTWARE_LOG;
   
   ///The type of the message
   struct messageT : public fbMessage
   {
      /// C'tor with full specification.
      messageT( const char * file,
                const uint32_t line,
                const int32_t  code_errno,
                const int32_t  code_other,
                const char * expl 
              )
      {
         auto _file = builder.CreateString(file);
         auto _expl = builder.CreateString(expl);
                                    
         auto gs = CreateSoftware_log_fb(builder, _file, line, code_errno, code_other, _expl);
         builder.Finish(gs);
      }
      
      /// C'tor with full specification, overloaded for a std::string in explanation.
      messageT( const char * file,
                const uint32_t line,
                const int32_t  code_errno,
                const int32_t  code_other,
                const std::string & expl 
              )
      {
         auto _file = builder.CreateString(file);
         auto _expl = builder.CreateString(expl);
                                    
         auto gs = CreateSoftware_log_fb(builder, _file, line, code_errno, code_other, _expl);
         builder.Finish(gs);
      }
      
      /// C'tor for errno only -- code explanation can be looked up later.
      messageT( const char * file,
                const uint32_t line,
                const int32_t  code_errno
              )
      {
         auto _file = builder.CreateString(file);
                                    
         auto gs = CreateSoftware_log_fb(builder, _file, line, code_errno, 0, 0);
         builder.Finish(gs);
      }
      
      /// C'tor for errno with additional explanation.
      messageT( const char * file,
                const uint32_t line,
                const int32_t  code_errno,
                const char * expl
              )
      {
         auto _file = builder.CreateString(file);
         auto _expl = builder.CreateString(expl);                           
         
         auto gs = CreateSoftware_log_fb(builder, _file, line, code_errno, 0, _expl);
         builder.Finish(gs);
      }
      
      /// C'tor for errno with additional explanation, std::string overload.
      messageT( const char * file,
                const uint32_t line,
                const int32_t  code_errno,
                const std::string & expl
              )
      {
         auto _file = builder.CreateString(file);
         auto _expl = builder.CreateString(expl);                           
         
         auto gs = CreateSoftware_log_fb(builder, _file, line, code_errno, 0, _expl);
         builder.Finish(gs);
      }
      
      /// C'tor with no codes, just the explanation.
      messageT( const char * file,
                const uint32_t line,
                const std::string & expl
              )
      {
         auto _file = builder.CreateString(file);
         auto _expl = builder.CreateString(expl);
         
         auto gs = CreateSoftware_log_fb(builder, _file, line, 0,0,_expl);
         builder.Finish(gs);        
      }

      /// C'tor for a trace log, only the file and line.
      messageT( const char * file,
                const uint32_t line
              )
      {
         auto _file = builder.CreateString(file);
         
         auto gs = CreateSoftware_log_fb(builder, _file, line, 0,0,0);
         builder.Finish(gs);        
      }
      
      
   };



   static std::string msgString( void * msgBuffer, flatlogs::msgLenT len) 
   {
      
      static_cast<void>(len);
      
      auto rgs = GetSoftware_log_fb(msgBuffer);
      
      std::string ret = "SW FILE: ";
      ret += rgs->file()->c_str();
      ret += " LINE: ";
      ret += mx::ioutils::convertToString(rgs->line());
      if(rgs->errnoCode())
      {
         ret += "  ERRNO: ";
         ret += mx::ioutils::convertToString(rgs->errnoCode());
      }
      if(rgs->otherCode())
      {
         ret += "  OTHER CODE: ";
         ret += mx::ioutils::convertToString(rgs->otherCode());
      }
      
      if(rgs->explanation())
      {
         ret += " >";
         ret += rgs->explanation()->c_str();
      }
      return ret;
   }
};


///Software EMERGENCY log entry
/** \ingroup logtypes
  */
struct software_emergency : public software_log
{
   ///The default level
   static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_EMERGENCY;
};

///Software ALERT log entry
/** \ingroup logtypes
  */
struct software_alert : public software_log
{
   ///The default level
   static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_ALERT;
};

///Software CRIT log entry
/** \ingroup logtypes
  */
struct software_critical : public software_log
{
   ///The default level
   static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_CRITICAL;
};

///Software ERR log entry
/** \ingroup logtypes
  */
struct software_error : public software_log
{
   ///The default level
   static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_ERROR;
};

///Software WARN log entry
/** \ingroup logtypes
  */
struct software_warning : public software_log
{
   ///The default level
   static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_WARNING;
};

///Software NOTICE log entry
/** \ingroup logtypes
  */
struct software_notice : public software_log
{
   ///The default level
   static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_NOTICE;
};

///Software INFO log entry
/** \ingroup logtypes
  */
struct software_info : public software_log
{
   ///The default level
   static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_INFO;
};

///Software DEBUG log entry
/** \ingroup logtypes
  */
struct software_debug : public software_log
{
   ///The default level
   static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_DEBUG;
};

///Software DEBUG2 log entry
/** \ingroup logtypes
  */
struct software_debug2 : public software_log
{
   ///The default level
   static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_DEBUG2;
};

} //namespace logger
} //namespace MagAOX

#endif //logger_types_software_log_hpp
