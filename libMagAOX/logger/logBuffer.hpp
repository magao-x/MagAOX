/** \file logBuffer.hpp 
  * \brief The MagAO-X logger buffer format and utility functions.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * History:
  * - 2017-08-29 created by JRM
  */ 
#ifndef logger_logBuffer_hpp
#define logger_logBuffer_hpp

#include "../time/timespecX.hpp"
#include "logCodes.hpp"
#include "logLevels.hpp"

namespace MagAOX
{
namespace logger 
{

   
///The type of the message length
typedef uint16_t msgLenT;


/// The log entry header 
/** The log entry is a binary buffer with the following format:
  * \verbatim 
     0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 (22 + len)
    |l| event |time_s            |time_ns                | len | message        |
    \endverbatim
  * where the header consists of everything up to and including the length of the message.
  */
struct logHeaderT
{
   logLevelT logLevel;
   eventCodeT eventCode;
   time::timespecX timespecX;
   msgLenT msgLen;
}  __attribute__((packed)) ;

///The log entry buffer smart pointer.
typedef std::shared_ptr<char> bufferPtrT; 

///Extract the level of a log entry
/**
  * \returns the level
  */
inline
logLevelT logLevel( bufferPtrT & logBuffer /**< [in] a shared_ptr\<char\> containing a raw lag entry buffer.*/)
{
   return reinterpret_cast<logHeaderT *>(logBuffer.get())->logLevel;
}

///Extract the event code of a log entry
/**
  * \returns the event code
  */
inline
eventCodeT eventCode( bufferPtrT & logBuffer /**< [in] a shared_ptr\<char\> containing a raw lag entry buffer.*/)
{
   return reinterpret_cast<logHeaderT *>(logBuffer.get())->eventCode;
}


///Extract the timespec of a log entry
/**
  * \returns the timespec
  */
inline
time::timespecX timespecX( bufferPtrT & logBuffer /**< [in] a shared_ptr\<char\> containing a raw lag entry buffer.*/)
{
   return reinterpret_cast<logHeaderT *>(logBuffer.get())->timespecX;
}

///Extract the message length of a log entry
/**
  * \returns the message length
  */
inline
msgLenT msgLen(  bufferPtrT & logBuffer /**< [in] a shared_ptr\<char\> containing a raw lag entry buffer.*/)
{
   return reinterpret_cast<logHeaderT *>(logBuffer.get())->msgLen;
}

///Extract the basic details of a log entry
/** Convenience wrapper for the other extraction functions.
  * 
  * \returns 0 on success, -1 on error.
  */
inline
int extractBasicLog( logLevelT & lvl,       ///< [out] The log level
                     eventCodeT & ec,       ///< [out] the event code
                     time::timespecX & ts,  ///< [out] the timestamp of the log entry
                     msgLenT & len,         ///< [out] the message length
                     bufferPtrT & logBuffer ///< [in] a shared_ptr\<char\> containing a raw lag entry buffer.
                   )
{
   
   lvl = logLevel(logBuffer); 
   
   ec = eventCode(logBuffer); 

   ts = timespecX(logBuffer);

   len =  msgLen(logBuffer);
      
   return 0;
}
   

///The offset to the log level entry
static const int levelOffset = 0;

///The offset to the log event code
static const int eventOffset = levelOffset + sizeof(logLevelT);

///The offset to the time seconds 
static const int time_sOffset = eventOffset + sizeof(eventCodeT);

///The offset to the time nanoseconds
static const int time_nsOffset = time_sOffset + sizeof(time::timespecX::secT);

///The offset to the message length field
static const int lenOffset = time_nsOffset + sizeof(time::timespecX::nanosecT);

///The offset to the message.
static const int messageOffset = lenOffset + sizeof(msgLenT);
 
///The base size of a message is the size of the header, not including the message.
static const int headerSize = sizeof(logHeaderT);
   
static_assert( headerSize == messageOffset, "headerSize and messageOffset mismatch");

} //namespace logger
} //namespace MagAOX

#endif //logger_logBuffer_hpp

