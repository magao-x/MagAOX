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

/// Log entry recording the build-time git state.
/** \ingroup logtypes
  */
struct git_state
{
   //Define the log name for use in the database
   //Event: "Git State"
   
   ///The event code
   static const eventCodeT eventCode = eventCodes::GIT_STATE;

   ///The default level 
   static const logLevelT defaultLevel = logLevels::INFO;
   
   ///Length of the sha1 hash in ASCII
   static const size_t s_sha1Length = 40;
   
   ///The type of the message
   struct messageT
   {
      char m_sha1[s_sha1Length];
      char m_modified {0};
      std::string m_repoName;
      
      ///Allow default construction
      messageT()
      {
         for(int i=0;i<s_sha1Length;++i) m_sha1[i] = 0;
      }
      
      ///Construct from components
      messageT( const std::string & repoName, ///< [in] the name of the repo
                const std::string & sha1,     ///< [in] the SHA1 hash of the repo
                const bool modified           ///< [in] the modified status (true or false)
              )
      {
         m_repoName = repoName;
         
         int N = sha1.size();
         
         if(sha1.size() != s_sha1Length)
         {
            std::cerr << "SHA-1 incorrect size!\n";
            if(N > s_sha1Length) N = s_sha1Length;
         }
         
         for(int i=0; i<N;++i) m_sha1[i] = sha1[i];
         
         if(modified) m_modified = 1;
         else m_modified = 0;
      }
               
   };
   
   ///Get the length of the message.
   static msgLenT length( const messageT & msg )
   {
      return (s_sha1Length + 1)*sizeof(char) + msg.m_repoName.size();
   }
  
   ///Format the buffer given the input message.
   static int format( void * msgBuffer,    ///< [out] the buffer, must be pre-allocated to size length(msg)
                      const messageT & msg ///< [in] the message, which is placed in the buffer char by char.
                    )
   {
      char * cbuff = reinterpret_cast<char *>(msgBuffer);
      
      int i;
      for(i=0; i< msg.m_repoName.size(); ++i) cbuff[i] = msg.m_repoName[i];
         
      for(int j =0; j< s_sha1Length; ++j)
      {
         cbuff[i] = msg.m_sha1[j];
         ++i;
      }
      cbuff[i] = msg.m_modified;
      
      return 0;
   }
   
   ///Extract the message from the buffer and fill in the mesage
   /** 
     * \returns 0 on success.
     * \returns -1 on an error.
     */ 
   static int extract( messageT & msg, ///< [out] the message which is populated with the contents of buffer.
                       void * msgBuffer,  ///< [in] the buffer containing the GIT state.
                       msgLenT len ///< [in] the length of the string contained in buffer.
                     )
   {
      char * cbuff = reinterpret_cast<char *>(msgBuffer);
      
      int nlen = len - (s_sha1Length + 1);
      
      msg.m_repoName.resize(nlen);
      
      int i;
      for(i =0; i< nlen; ++i) msg.m_repoName[i] = cbuff[i];
      
      for(int j=0; j< s_sha1Length; ++j)
      {
         msg.m_sha1[j] = cbuff[i];
         ++i;
      }
      msg.m_modified = cbuff[i];
      
      return 0;
   }
   
   static std::string msgString( messageT & msg )
   {
      std::string str = msg.m_repoName + " GIT: ";
      for(int i=0;i<s_sha1Length;++i) str += msg.m_sha1[i];
      
      if(msg.m_modified) str += " MODIFIED";
      
      return str;
   }
}; //git_state 
   
///A simple text log, a string-type log.
/** \ingroup logtypes
  */
struct text_log : public string_log
{
   //Define the log name for use in the database
   //Event: "Text Log"
   
   ///The event code
   static const eventCodeT eventCode = eventCodes::TEXT_LOG;

   ///The default level 
   static const logLevelT defaultLevel = logLevels::INFO;
   
};

///User entered log, a string-type log.
/** \ingroup logtypes
  */
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
/** \ingroup logtypes
  */
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
/** \ingroup logtypes
  */
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
/** \ingroup logtypes
  */
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
/** \ingroup logtypes
  */
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
/** \ingroup logtypes
  */
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
/** \ingroup logtypes
  */
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
/** \ingroup logtypes
  */
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
/** \ingroup logtypes
  */
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
/** \ingroup logtypes
  */
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
/** \ingroup logtypes
  */
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
/** \ingroup logtypes
  */
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
     * \returns the message formatted as "State changed from UNINITIALIZED to INITIALIZED"
     */ 
   static std::string msgString( messageT & msg /**< [in] the message structure */ )
   {
      std::stringstream s;
      s << "State changed from " << app::stateCodes::codeText(msg.from) << " to " << app::stateCodes::codeText(msg.to);
      return s.str();
   }
};

} //namespace logger
} //namespace MagAOX

#endif //logger_logTypes_hpp

