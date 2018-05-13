/** \file logTypesBasics.hpp 
  * \brief The MagAO-X logger basic log types.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * History:
  * - 2018-01-01 created by JRM
  */ 
#ifndef logger_logTypesBasics_hpp
#define logger_logTypesBasics_hpp

#include "logBuffer.hpp"

namespace MagAOX
{
namespace logger 
{

   
///Base class for logs consisting of a string message.
/** Does not have eventCode or defaultLevel, so this can not be used as a log type in logger.
  */
struct string_log
{
   //Define the log name for use in the database
   //Event: "---"
      
   ///The type of the message
   typedef std::string messageT;
   
   ///Get the length of the message.
   static msgLenT length( const messageT & msg )
   {
      return msg.size();
   }
  
   ///Format the buffer given the input message (a std::string).
   static int format( void * msgBuffer, ///< [out] the buffer, must be pre-allocated to size length(msg)
                      const messageT & msg ///< [in] the message, a std::string, which is placed in the buffer char by char.
                    )
   {
      char * cbuff = reinterpret_cast<char *>(msgBuffer);
      
      for(int i =0; i< msg.size(); ++i)
      {
         cbuff[i] = msg[i];
      }
      
      return 0;
   }
   
   ///Extract the message from the buffer and fill in the mesage (a std::string)
   /** \returns 0 on success.
     * \returns -1 on an error.
     */ 
   static int extract( messageT & msg, ///< [out] the message, a std::string, which is resize()-ed and populated with the contents of buffer.
                       void * msgBuffer,  ///< [in] the buffer containing the string.
                       msgLenT len ///< [in] the length of the string contained in buffer.
                     )
   {
      char * cbuff = reinterpret_cast<char *>(msgBuffer);
      
      msg.resize(len + 1);
      
      for(int i=0; i< len; ++i)
      {
         msg[i] = cbuff[i];
      }
      msg[len] = '\0'; //I think this might be handled by std::string
      
      return 0;
   }
   
   static std::string msgString( messageT & msg )
   {
      return msg;
   }
   
};


///Empty type for resolving logs with no message.
struct emptyMessage
{
};


///Base class for logs consisting of an empty message.
/** Such logs are used to log events. Does not have eventCode or defaultLevel, so this can not be used as a log type in logger.
  */
struct empty_log
{
   ///The type of the message
   typedef emptyMessage messageT;
   
   ///Get the length of the message.
   static msgLenT length( const messageT & msg)
   {
      return 0;
   }
   
   ///Format the buffer given a message -- a no-op since the message is an emptyMessage.
   /** 
     * \returns 0
     */  
   static int format( void * msgBuffer,  ///< [out] the buffer, which is ignored.
                      const messageT & msg ///< [in] an emptyMessage.
                    )
   {
      return 0;
   }
   
   ///Extract the message from a buffer -- a no-op since it is an emptyMessage.
   /** 
     * \returns 0
     */
   static int extract( messageT & msg, ///< [out] an emptyMessage to which nothing is done.
                       void * msgBuffer, ///< [in] the empty buffer.  Is ignored.
                       msgLenT len ///< [in] ignored length of the empty buffer.
                     )
   {
      return 0;
   }
};

struct softwareMessage
{
   typedef std::string stringT;
   typedef int linenumT;
   typedef int codeT;
   
   typedef int lengthT;
   
   stringT file;
   linenumT linenum;
   codeT code;
   stringT explanation;
};

///Base class for software logs
/** Such logs are used to log software status, warnings, and errors. Does not have eventCode or defaultLevel, so this can not be used as a log type in logger.
  */
struct software_log
{
   ///The type of the message
   typedef softwareMessage messageT;
   
   ///Get the length of the message.
   static msgLenT length( const messageT & msg)
   {
      return ( sizeof(softwareMessage::lengthT) + msg.file.size() 
                   + sizeof(softwareMessage::linenumT) + sizeof(softwareMessage::codeT) 
                       + msg.explanation.size() );
   }
   
   ///Format the buffer given a software message
   /** 
     * \returns 0
     */  
   static int format( void * msgBuffer, ///< [out] the buffer, must be pre-allocated to size length(msg)
                      const messageT & msg ///< [in] a softwareMessage.
                    )
   {
      
      char * cBuffer = reinterpret_cast<char *>(msgBuffer);
      char * cbuff;
      
      //Insert file length and string
      *reinterpret_cast<messageT::lengthT *>(cBuffer) = msg.file.length();
      int offset = sizeof(messageT::lengthT);
      
      cbuff = cBuffer + offset; //reinterpret_cast<char *>(msgBuffer + offset);
      for(int i =0; i< msg.file.size(); ++i)
      {
         cbuff[i] = msg.file[i];
      }
      offset += msg.file.size();
      
      //Insert file length and string
      *reinterpret_cast<messageT::linenumT *>(cBuffer+offset) = msg.linenum;
      offset += sizeof(messageT::linenumT);
      
      *reinterpret_cast<messageT::codeT *>(cBuffer+offset) = msg.code;
      offset += sizeof(messageT::codeT);
      
      cbuff = reinterpret_cast<char *>(cBuffer + offset);
      
      for(int i =0; i< msg.explanation.size(); ++i)
      {
         cbuff[i] = msg.explanation[i];
      }
      
      return 0;
   }
   
   ///Extract the software message from a log buffer.
   /** 
     * \returns 0
     */
   static int extract( messageT & msg, ///< [out] n softwareMessage
                       void * msgBuffer, ///< [in] a log buffer
                       msgLenT len ///< [in] length of the buffer.
                     )
   {
      char * cBuffer = reinterpret_cast<char *>(msgBuffer);
      
      messageT::lengthT strLen = *reinterpret_cast<messageT::lengthT *>(cBuffer);
      int offset = sizeof(messageT::lengthT);
      
      char * cbuff = reinterpret_cast<char *>(cBuffer + offset);
      msg.file.resize(strLen);
      for(int i=0;i<strLen; ++i)
      {
         msg.file[i] = cbuff[i];
      }
      
      offset += strLen;
      msg.linenum = *reinterpret_cast<messageT::linenumT *>(cBuffer+offset);
 
      offset += sizeof(messageT::linenumT);
      msg.code = *reinterpret_cast<messageT::codeT *>(cBuffer+offset);
      
      offset += sizeof(messageT::codeT);
      
      //strLen = *reinterpret_cast<messageT::lengthT *>(cBuffer + offset);
      strLen = len - offset;
//      offset += sizeof(messageT::lengthT);
      
      cbuff = reinterpret_cast<char *>(cBuffer + offset);
      msg.explanation.resize(strLen);
      for(int i=0;i<strLen; ++i)
      {
         msg.explanation[i] = cbuff[i];
      }
      
      return 0;
   }
   
   static std::string msgString( messageT & msg )
   {
      std::string ret = "SW FILE: ";
      ret += msg.file;
      ret += " LINE: ";
      ret += mx::convertToString(msg.linenum);
      ret += "  CODE: ";
      ret += mx::convertToString(msg.code);
      ret += " >";
      ret += msg.explanation;
      
      return ret;
   }
};


} //namespace logger
} //namespace MagAOX

#endif //logger_logTypesBasics_hpp

