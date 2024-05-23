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
/** Such logs are used to log software status, warnings, and errors. Does not have defaultLevel, so this can not be used as a log type in logger.
  *
  * \includedoc sw_logs.dox.inc
  * 
  * 
  * \ingroup logger_types__basic
  */
struct software_log : public flatbuffer_log
{
   ///The event code
   static const flatlogs::eventCodeT eventCode = eventCodes::SOFTWARE_LOG;
   
   ///The type of the message
   struct messageT : public fbMessage
   {
      /// C'tor with full specification.
      messageT( const char * file, ///< [in] The file of the error, should always be \c \_\_FILE\_\_
                const uint32_t line, ///< [in] The line number of the error, should always be \c \_\_LINE\_\_ 
                const int32_t  errnoCode, ///< [in] The errno code at the time of the log entry. Only errno should be passed here, so strerror can be used later.
                const int32_t  otherCode, ///< [in] Some other error code, such as a return value or library code.
                const char * explanation  ///< [in] explanatory text about the software event
              )
      {
         auto _file = builder.CreateString(file);
         auto _expl = builder.CreateString(explanation);
                                    
         auto gs = CreateSoftware_log_fb(builder, _file, line, errnoCode, otherCode, _expl);
         builder.Finish(gs);
      }
      
      /// C'tor with full specification, overloaded for a std::string in explanation.
      /** \overload
        */
      messageT( const char * file, ///< [in] The file of the error, should always be \c \_\_FILE\_\_
                const uint32_t line, ///< [in] The line number of the error, should always be \c \_\_LINE\_\_
                const int32_t  errnoCode, ///< [in] The errno code at the time of the log entry. Only errno should be passed here, so strerror can be used later.
                const int32_t  otherCode, ///< [in] Some other error code, such as a return value or library code.
                const std::string & explanation ///< [in] explanatory text about the software event
              )
      {
         auto _file = builder.CreateString(file);
         auto _expl = builder.CreateString(explanation);
                                    
         auto gs = CreateSoftware_log_fb(builder, _file, line, errnoCode, otherCode, _expl);
         builder.Finish(gs);
      }
      
      /// C'tor for errno only -- code explanation can be looked up later.
      messageT( const char * file, ///< [in] The file of the error, should always be \c \_\_FILE\_\_
                const uint32_t line, ///< [in] The line number of the error, should always be \c \_\_LINE\_\_
                const int32_t  errnoCode ///< [in] The errno code at the time of the log entry. Only errno should be passed here, so strerror can be used later.
              )
      {
         auto _file = builder.CreateString(file);
                                    
         auto gs = CreateSoftware_log_fb(builder, _file, line, errnoCode, 0, 0);
         builder.Finish(gs);
      }
      
      /// C'tor for errno with additional explanation.
      messageT( const char * file, ///< [in] The file of the error, should always be \c \_\_FILE\_\_
                const uint32_t line, ///< [in] The line number of the error, should always be \c \_\_LINE\_\_
                const int32_t  errnoCode, ///< [in] The errno code at the time of the log entry. Only errno should be passed here, so strerror can be used later.
                const char * explanation ///< [in] explanatory text about the software event
              )
      {
         auto _file = builder.CreateString(file);
         auto _expl = builder.CreateString(explanation);                           
         
         auto gs = CreateSoftware_log_fb(builder, _file, line, errnoCode, 0, _expl);
         builder.Finish(gs);
      }
      
      /// C'tor for errno with additional explanation, std::string overload.
      messageT( const char * file, ///< [in] The file of the error, should always be \c \_\_FILE\_\_
                const uint32_t line, ///< [in] The line number of the error, should always be \c \_\_LINE\_\_
                const int32_t  errnoCode, ///< [in] The errno code at the time of the log entry. Only errno should be passed here, so strerror can be used later.
                const std::string & explanation ///< [in] explanatory text about the software event
              )
      {
         auto _file = builder.CreateString(file);
         auto _expl = builder.CreateString(explanation);                           
         
         auto gs = CreateSoftware_log_fb(builder, _file, line, errnoCode, 0, _expl);
         builder.Finish(gs);
      }
      
      /// C'tor with no codes, just the explanation.
      messageT( const char * file, ///< [in] The file of the error, should always be \c \_\_FILE\_\_
                const uint32_t line, ///< [in] The line number of the error, should always be \c \_\_LINE\_\_
                const std::string & explanation ///< [in] explanatory text about the software event
              )
      {
         auto _file = builder.CreateString(file);
         auto _expl = builder.CreateString(explanation);
         
         auto gs = CreateSoftware_log_fb(builder, _file, line, 0,0,_expl);
         builder.Finish(gs);        
      }

      /// C'tor for a trace log, only the file and line.
      messageT( const char * file, ///< [in] The file of the error, should always be \c \_\_FILE\_\_
                const uint32_t line ///< [in] The line number of the error, should always be \c \_\_LINE\_\_
              )
      {
         auto _file = builder.CreateString(file);
         
         auto gs = CreateSoftware_log_fb(builder, _file, line, 0,0,0);
         builder.Finish(gs);        
      }
      
      
   };

   static bool verify( flatlogs::bufferPtrT & logBuff,  ///< [in] Buffer containing the flatbuffer serialized message.
                       flatlogs::msgLenT len            ///< [in] length of msgBuffer.
                     )
   {
      auto verifier = flatbuffers::Verifier( static_cast<uint8_t*>(flatlogs::logHeader::messageBuffer(logBuff)), static_cast<size_t>(len));
      return VerifySoftware_log_fbBuffer(verifier);
   }

   ///Get the message formatted for human consumption.
   static std::string msgString( void * msgBuffer,  /**< [in] Buffer containing the flatbuffer serialized message.*/
                                 flatlogs::msgLenT len  /**< [in] [unused] length of msgBuffer.*/
                               )
   {
      
      static_cast<void>(len);
      
      auto rgs = GetSoftware_log_fb(msgBuffer);
      
      std::string ret = "SW FILE: ";
      if(rgs->file() != nullptr)
      {
         ret += rgs->file()->c_str();
      }
      else
      {
         ret += "????";
      }
      
      ret += " LINE: ";
      ret += mx::ioutils::convertToString(rgs->line());
      if(rgs->errnoCode())
      {
         ret += "  ERRNO: ";
         ret += mx::ioutils::convertToString(rgs->errnoCode());
         ret += " [";
         ret += strerror(rgs->errnoCode());
         ret += "]";
      }
      if(rgs->otherCode())
      {
         ret += "  CODE: ";
         ret += mx::ioutils::convertToString(rgs->otherCode());
         if(rgs->explanation())
         {
            ret += " [";
            ret += rgs->explanation()->c_str();
            ret += "]";
         }
      }
      else if(rgs->explanation())
      {
         ret += " >";
         ret += rgs->explanation()->c_str();
      }
      return ret;
   }

};


///Software EMERGENCY log entry
/** This should only be used for a system-wide emergency requiring operator or automatic shutdown.  Not for a process specific problem.
  * \includedoc sw_logs.dox.inc
  * \ingroup logger_types
  */
struct software_emergency : public software_log
{
   ///The default level
   static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_EMERGENCY;
};

///Software ALERT log entry
/** This should only be used for a system-wide emergency requiring operator or automatic action.  Not for a process specific problem.
  * \includedoc sw_logs.dox.inc
  * \ingroup logger_types
  */
struct software_alert : public software_log
{
   ///The default level
   static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_ALERT;
};

///Software CRITICAL log entry
/** This should only be used if the process is going to shutdown.
  * \includedoc sw_logs.dox.inc
  * \ingroup logger_types
  */
struct software_critical : public software_log
{
   ///The default level
   static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_CRITICAL;
};

///Software ERR log entry
/** Used to record and error that the process will attempt to recover from. 
  * \includedoc sw_logs.dox.inc
  * \ingroup logger_types
  */
struct software_error : public software_log
{
   ///The default level
   static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_ERROR;
};

///Software WARN log entry
/** Used to record an abnormal condition.
  * \includedoc sw_logs.dox.inc
  * \ingroup logger_types
  */
struct software_warning : public software_log
{
   ///The default level
   static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_WARNING;
};

///Software NOTICE log entry
/** Used to record a normal but signficant event or condition.
  * \includedoc sw_logs.dox.inc
  * \ingroup logger_types
  */
struct software_notice : public software_log
{
   ///The default level
   static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_NOTICE;
};

///Software INFO log entry
/** \includedoc sw_logs.dox.inc
 * Used to record a normal event or condition.  This is the lowest priority used in normal operations.
  * \ingroup logger_types
  */
struct software_info : public software_log
{
   ///The default level
   static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_INFO;
};

///Software DEBUG log entry
/** \includedoc sw_logs.dox.inc
 * \ingroup logger_types
  */
struct software_debug : public software_log
{
   ///The default level
   static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_DEBUG;
};

///Software DEBUG2 log entry
/** \includedoc sw_logs.dox.inc
  * \ingroup logger_types
  */
struct software_debug2 : public software_log
{
   ///The default level
   static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_DEBUG2;
};

} //namespace logger
} //namespace MagAOX

#endif //logger_types_software_log_hpp
