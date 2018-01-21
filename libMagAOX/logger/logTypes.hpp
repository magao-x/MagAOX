/** \file logTypes.hpp 
  * \brief The MagAO-X logger log types.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * History:
  * - 2017-06-27 created by JRM
  */ 
#ifndef logger_logTypes_hpp
#define logger_logTypes_hpp

#include "../app/stateCodes.hpp"

#include "logTypesBasics.hpp"

namespace MagAOX
{
namespace logger 
{

   
///A simple text log, a string-type log.
struct text_log : public string_log
{
   //Define the log name for use in the database
   //Event: "Text Log"
   
   ///The event code
   static const eventCodeT eventCode = eventCodes::TEXT_LOG;

   ///The default level 
   static const logLevelT defaultLevel = logLevels::INFO;
   
   static std::string msgString( messageT & msg /**< [in] the message, a std::string */)
   {
      return msg;
   }
};

///User entered log, a string-type log.
struct user_log : public string_log
{
   //Define the log name for use in the database
   //Event: "User Log"
   
   ///The event code
   static const eventCodeT eventCode = eventCodes::USER_LOG;

   ///The default level 
   static const logLevelT defaultLevel = logLevels::INFO;
   
   static std::string msgString( messageT & msg /**< [in] the message, a std::string */)
   {
      std::string nmsg = "USER: ";
      return nmsg + msg;
   }
};


///Software DEBUG log entry
struct software_debug : public software_log
{
   //Define the log name for use in the database
   //Event: "Software Debug"
   
   ///The event code
   static const eventCodeT eventCode = eventCodes::SOFTWARE_DEBUG;

   ///The default level 
   static const logLevelT defaultLevel = logLevels::DEBUG;   
};

///Software DEBUG2 log entry
struct software_debug2 : public software_log
{
   //Define the log name for use in the database
   //Event: "Software Debug2"
   
   ///The event code
   static const eventCodeT eventCode = eventCodes::SOFTWARE_DEBUG2;

   ///The default level 
   static const logLevelT defaultLevel = logLevels::DEBUG2;   
};

///Software INFO log entry
struct software_info : public software_log
{
   //Define the log name for use in the database
   //Event: "Software Info"
   
   ///The event code
   static const eventCodeT eventCode = eventCodes::SOFTWARE_INFO;

   ///The default level 
   static const logLevelT defaultLevel = logLevels::INFO;   
};

///Software WARN log entry
struct software_warning : public software_log
{
   //Define the log name for use in the database
   //Event: "Software Warning"
   
   ///The event code
   static const eventCodeT eventCode = eventCodes::SOFTWARE_WARNING;

   ///The default level 
   static const logLevelT defaultLevel = logLevels::WARNING;   
};

///Software ERR log entry
struct software_error : public software_log
{
   //Define the log name for use in the database
   //Event: "Software Error"
   
   ///The event code
   static const eventCodeT eventCode = eventCodes::SOFTWARE_ERROR;

   ///The default level 
   static const logLevelT defaultLevel = logLevels::ERROR;   
};

///Software CRIT log entry
struct software_critical : public software_log
{
   //Define the log name for use in the database
   //Event: "Software Critical"
   
   ///The event code
   static const eventCodeT eventCode = eventCodes::SOFTWARE_CRITICAL;

   ///The default level 
   static const logLevelT defaultLevel = logLevels::CRITICAL;   
};

///Software FATAL log entry
struct software_fatal : public software_log
{
   //Define the log name for use in the database
   //Event: "Software Fatal"
   
   ///The event code
   static const eventCodeT eventCode = eventCodes::SOFTWARE_FATAL;

   ///The default level 
   static const logLevelT defaultLevel = logLevels::FATAL;   
};


///Loop Closed event log
struct loop_closed : public empty_log
{
   //Define the log name for use in the database
   //Event: "Loop Closed"
   
   ///The event code
   static const eventCodeT eventCode = eventCodes::LOOP_CLOSED;

   ///The default level 
   static const logLevelT defaultLevel = logLevels::INFO;   
   
   static std::string msgString( messageT & msg  /**< [in] [unused] the empty message */ )
   {
      return "LOOP CLOSED";
   }
};

///Loop Paused event log
struct loop_paused : public empty_log
{
   //Define the log name for use in the database
   //Event: "Loop Paused"
   
   ///The event code
   static const eventCodeT eventCode = eventCodes::LOOP_PAUSED;
   
   ///The default level 
   static const logLevelT defaultLevel = logLevels::INFO;   
   
   static std::string msgString( messageT & msg  /**< [in] [unused] the empty message */)
   {
      return "LOOP PAUSED";
   }
};

///Loop Open event log
struct loop_open : public empty_log
{
   //Define the log name for use in the database
   //Event: "Loop Open"
   
   ///The event code
   static const eventCodeT eventCode = eventCodes::LOOP_OPEN;
   
   ///The default level 
   static const logLevelT defaultLevel = logLevels::INFO;   
   
   static std::string msgString( messageT & msg  /**< [in] [unused] the empty message */)
   {
      return "LOOP OPEN";
   }
   
};

///Application State Change 
struct state_change
{
   //Define the log name for use in the database
   //Event: "App State Change"

   //The event code 
   static const eventCodeT eventCode = eventCodes::STATE_CHANGE;
   
   //The default level 
   static const logLevelT defaultLevel = logLevels::INFO;
   
   ///The type of the message
   struct messageT
   {
      int from;
      int to;
   } __attribute__((packed));
   
   ///Get the length of the message.
   static msgLenT length( const messageT & msg /**< [in] [unused] the message itself */ )
   {
      return sizeof(messageT);
   }
  
   ///Format the buffer given the input message (a std::string).
   static int format( void * msgBuffer,    ///< [out] the buffer, must be pre-allocated to size length(msg)
                      const messageT & msg ///< [in] the message, a std::string, which is placed in the buffer
                    )
   {
      int * ibuff = reinterpret_cast<int *>(msgBuffer);
      
      ibuff[0] = msg.from;
      ibuff[1] = msg.to;
      
      return 0;
   }
   
   ///Extract the message from the buffer and fill in the mesage
   /** 
     * \returns 0 on success.
     * \returns -1 on an error.
     */ 
   static int extract( messageT & msg,   ///< [out] the message, an int[2], which is populated with the contents of buffer.
                       void * msgBuffer, ///< [in] the buffer containing the input codes as an int[2].
                       msgLenT len       ///< [in] the length of the string contained in buffer.
                     )
   {
      int * ibuff = reinterpret_cast<int *>(msgBuffer);
      
      msg.from = ibuff[0];
      msg.to = ibuff[1];
      
      return 0;
   }
   
   /// Format the message for text output, including translation of state codes to text form.
   /**
     * \returns the message fromatted as "State changed from UNINITIALIZED to INITIALIZED"
     */ 
   static std::string msgString( messageT & msg /**< [in] the message structure */ )
   {
      std::stringstream s;
      s << "State changed from " << app::stateCodeText(msg.from) << " to " << app::stateCodeText(msg.to);
      return s.str();
   }
};

} //namespace logger
} //namespace MagAOX

#endif //logger_logTypes_hpp

