/** \file logHeader.hpp 
  * \brief The flatlogs buffer header format.
  * \author Jared R. Males (jaredmales@gmail.com)
  * 
  * \ingroup flatlogs_files
  * 
  * History:
  * - 2017-08-29 created by JRM
  * - 2018-08-17 moved to flatlogs
  * 
  */ 
#ifndef flatlogs_logHeader_hpp
#define flatlogs_logHeader_hpp

#include <memory>
#include "logDefs.hpp"
#include "timespecX.hpp"
#include "logPriority.hpp"

namespace flatlogs
{

/** The log entry is a binary buffer with the following format:
  * \addtogroup flatlogs
  * \verbatim 
                         1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 
     0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8
    |p|evt|time_s |time_ns|l| [message (0-254)]
    |p|evt|time_s |time_ns|E|len| [message (255-65534)]
    |p|evt|time_s |time_ns|F|len          | [message (>65535)]
    \endverbatim
  * where the header consists of everything up to the [message]. Table notations are:
  * - p (uint8_t) denotes the log priority.  
  * - evt (uint16_t) denotes the event code.  
  * - time_s (uint32_t) is the time in seconds since the epoch 
  * - time_ns (uint32_t) denotes the nanoseconds past that second. 
  * - l/len is the message length field, which  is variable in length
  *   depending on the size of the message:
  *   - Message length 0-253: uint8_t in a single byte.
  *   - Message length 255-65534: uint8_t = 0xFF-1 in the first byte, uint16_t in the next two bytes.
  *   - Message length 65535 and up: uint8_t=0xFF in the first byte, uint64_t in the next 8 bytes.
  * 
  * Rationale for variable length: this keeps the space used for 0 length and short messages
  * to a minimum, while allowing for progressively larger messages to be saved.  
  * The savings will be large for small messages, while the the cost is proportionally small 
  * for larger messages.  For larger messages, this requires a progressive read to 
  * sniff out the entire length.  Likewise, the cost is small relative to the cost of reading 
  * the larger message.
  *
  * If a non-zero message exists, it is usually a flatlogs serialized buffer. 
  *
  * 
  */
   
///The log entry buffer smart pointer.
/** \ingroup logbuff
  */
typedef std::shared_ptr<char> bufferPtrT; 


/// The log entry header 
/** 
  * This class is designed to work with the log header only as a shared pointer to it, 
  * not directly on the members.  The actual header struct is private so we ensure that it is 
  * accessed properly. As such all of the member methods are static and take a shared_ptr as 
  * first argument.
  *
  * \ingroup logbuff
  */
class logHeader
{
   
public:

   ///The max value in the msgLen0 field.
   constexpr static size_t MAX_LEN0 = std::numeric_limits<msgLen0T>::max() ;

   ///The max value in the msgLen1 field.
   constexpr static size_t MAX_LEN1 = std::numeric_limits<msgLen1T>::max();

   ///The minimum header size
   /** A buffer must be allocated to at least this size.
     */
   constexpr static int minHeadSize = sizeof(logPrioT) + sizeof(eventCodeT) + 
                                     sizeof(secT) + sizeof(nanosecT) + sizeof(msgLen0T);
 
   ///The maximum header size
   /** The header component could be as big as this.
     */
   constexpr static int maxHeadSize = minHeadSize + sizeof(msgLen2T);

private:
   
   struct internal_logHeader
   {
      logPrioT m_logLevel;
      eventCodeT m_eventCode;
      timespecX m_timespecX;
      msgLen0T msgLen0; ///< The short message length.  Always present.
      union
      {
         msgLen1T msgLen1; ///< The intermediate message length.  Only present if msgLen0 == max-1 of msgLen0T.
         msgLen2T msgLen2; ///< The long message length.  Only present if msgLen1 == max-value of msgLen0T.
      };
   } __attribute__((packed));
   
public:
   
   ///Set the level of a log entry in a logBuffer header
   /**
     * \returns 0 on success
     * \returns -1 on error
     * 
     * \ingroup logbuff
     */
   static int logLevel( bufferPtrT & logBuffer, ///< [in/out] a shared_ptr\<char\> containing a raw log entry buffer.
                        const logPrioT & lvl   ///< [in] the new log level.
                      );
   
   ///Extract the level of a log entry
   /**
     * \returns the level
     * 
     * \ingroup logbuff
     */
   static logPrioT logLevel( bufferPtrT & logBuffer /**< [in] a shared_ptr\<char\> containing a raw log entry buffer.*/);
   
   ///Extract the level of a log entry
   /**
     * \returns the level
     * 
     * \ingroup logbuff
     */
   static logPrioT logLevel( char * logBuffer /**< [in] a pointer to a raw log entry buffer.*/);
   
   ///Set the event code of a log entry
   /**
     * \returns 0 on success
     * \returns -1 on error
     * 
     * \ingroup logbuff
     */
   static int eventCode( bufferPtrT & logBuffer, ///< [in,out] a shared_ptr\<char\> containing a raw log entry buffer.
                         const eventCodeT & ec   ///< [in] the new event code.
                       );
   
   ///Extract the event code of a log entry
   /**
     * \returns the event code
     * 
     * \ingroup logbuff
     */
   static eventCodeT eventCode( bufferPtrT & logBuffer /**< [in] a shared_ptr\<char\> containing a raw log entry buffer.*/);
   
   ///Extract the event code of a log entry
   /**
     * \returns the event code
     * 
     * \ingroup logbuff
     */
   static eventCodeT eventCode( char * logBuffer /**< [in] a pointer a raw log entry buffer.*/);
   
   ///Set the timespec of a log entry
   /**
     * \returns 0 on success
     * \returns -1 on error
     * 
     * \ingroup logbuff
     */
   static int timespec( bufferPtrT & logBuffer,    ///< [in, out] a shared_ptr\<char\> containing a raw log entry buffer.*/ 
                        const timespecX & ts ///< [in] the new timespec
                      );
   
   ///Extract the timespec of a log entry
   /**
     * \returns the timespec
     * 
     * \ingroup logbuff
     */
   static timespecX timespec( bufferPtrT & logBuffer /**< [in] a shared_ptr\<char\> containing a raw log entry buffer.*/);
   
   ///Extract the timespec of a log entry
   /**
     * \returns the timespec
     * 
     * \ingroup logbuff
     */
   static timespecX timespec( char * logBuffer /**< [in] a pointer to a raw log entry buffer.*/);

   ///Get the size in bytes of the length field for an existing logBuffer.
   /**
     * \returns the number of bytes in the length field.
     * 
     * \ingroup logbuff
     */
   static size_t lenSize(  bufferPtrT & logBuffer /**< [in] a shared_ptr\<char\> containing a raw log entry buffer.*/);
   
   ///Get the size in bytes of the length field for an existing logBuffer.
   /**
     * \returns the number of bytes in the length field.
     * 
     * \ingroup logbuff
     */
   static size_t lenSize( char * logBuffer /**< [in] a pointer to a raw log entry buffer.*/);
   
   ///Get the size in bytes of the length field for a logBuffer given the intended message length
   /**
     * \returns the number of bytes to be put in the length field.
     * 
     * \ingroup logbuff
     */
   static size_t lenSize( msgLenT & msgSz /**< [in] the size of the intended message.*/);

   ///Set the message length of a log entry message
   /**
     * \note The logBuffer must already be allocated with a header large enough for this message size. 
     * 
     * \returns 0 on success
     * \returns -1 on error
     * 
     * \ingroup logbuff
     */
   static int msgLen( bufferPtrT & logBuffer, ///< [out] a shared_ptr\<char\> containing a raw log entry buffer allocated with large enough header for this message length.
                      const msgLenT & msgLen  ///< [in] the message length to set.
                    );
   
   ///Extract the short message length of a log entry message
   /** This is always safe on a minimally allocated logBuffer, can be used to test for progressive reading.
     * 
     * \returns the short message length field
     * 
     * \ingroup logbuff
     */
   static msgLen0T msgLen0(  bufferPtrT & logBuffer /**< [in] a shared_ptr\<char\> containing a raw log entry buffer.*/);
   
   ///Extract the short message length of a log entry message
   /** This is always safe on a minimally allocated logBuffer, can be used to test for progressive reading.
     * 
     * \returns the short message length field
     * 
     * \ingroup logbuff
     */
   static msgLen0T msgLen0( char * logBuffer /**< [in] a pointer to a raw log entry buffer.*/);
   
   ///Extract the medium message length of a log entry message
   /** This is NOT always safe, and should only be caled if msgLen0 is 0xFE. Can be used to test for progressive reading.
     * 
     * \returns the medium message length field
     * 
     * \ingroup logbuff
     */
   static msgLen1T msgLen1(  bufferPtrT & logBuffer /**< [in] a shared_ptr\<char\> containing a raw log entry buffer.*/);
   
   ///Extract the medium message length of a log entry message
   /** This is NOT always safe, and should only be caled if msgLen0 is 0xFE. Can be used to test for progressive reading.
     * 
     * \returns the medium message length field
     * 
     * \ingroup logbuff
     */
   static msgLen1T msgLen1( char * logBuffer /**< [in] a pointer to a raw log entry buffer.*/);
   
   ///Extract the message length of a log entry message
   /**
     * \returns the message length
     * 
     * \ingroup logbuff
     */
   static msgLenT msgLen(  bufferPtrT & logBuffer /**< [in] a shared_ptr\<char\> containing a raw log entry buffer.*/);
         
   ///Extract the message length of a log entry message
   /**
     * \returns the message length
     * 
     * \ingroup logbuff
     */
   static msgLenT msgLen( char * logBuffer /**< [in] a pointer to a raw log entry buffer.*/);
   
   ///Get the size of the header, including the variable size length field, for an existing logBuffer.
   /**
     * \returns the size of the header in this log entry.
     * 
     * \ingroup logbuff
     */
   static size_t headerSize(  bufferPtrT & logBuffer /**< [in] a shared_ptr\<char\> containing a raw log entry buffer.*/);

   ///Get the size of the header, including the variable size length field, for an existing logBuffer.
   /**
     * \returns the size of the header in this log entry.
     * 
     * \ingroup logbuff
     */
   static size_t headerSize( char* logBuffer /**< [in] a pointer to a raw log entry buffer.*/);
   
   ///Get the size of the header, including the variable size length field, given a message size.
   /**
     * \returns the size to be used for the header.
     * 
     * \ingroup logbuff
     */
   static size_t headerSize(  msgLenT & msgSz /**< [in] the size of the intended message.*/);

   ///Get the total size of the log entry, including the message buffer.
   /**
     * \returns the total size of this log entry.
     * 
     * \ingroup logbuff
     */
   static size_t totalSize(  bufferPtrT & logBuffer /**< [in] a shared_ptr\<char\> containing a raw log entry buffer.*/);
   
   ///Get the total size of the log entry, including the message buffer.
   /**
     * \returns the total size of this log entry.
     * 
     * \ingroup logbuff
     */
   static size_t totalSize( char * logBuffer /**< [in] a pointer to a raw log entry buffer.*/);
   
   ///Get the total size of a log entry, given the message buffer size.
   /**
     * \returns the total size to be used for a log entry.
     * 
     * \ingroup logbuff
     */
   static size_t totalSize(  msgLenT & msgSz /**< [in] the intended size of the message buffer.*/);
   
   ///Get the message buffer address.
   /**
     * \returns the address of the message buffer.
     * 
     * \ingroup logbuff
     */
   static void * messageBuffer(  bufferPtrT & logBuffer /**< [in] a shared_ptr\<char\> containing a raw log entry buffer.*/);
   
   ///Get the message buffer address.
   /**
     * \returns the address of the message buffer.
     * 
     * \ingroup logbuff
     */
   static void * messageBuffer( char * logBuffer /**< [in] a pointer to a raw log entry buffer.*/);
   
   /// Create a formatted log entry, filling in a buffer.
   /** This version has the timestamp provided.
     *
     * \tparam logT is a log entry type
     *
     * \returns 0 on success, -1 on error.
     */
   template<typename logT>
   static int createLog( bufferPtrT & logBuffer,              ///< [out] a shared_ptr\<logBuffer\>, which will be allocated and populated with the log entry 
                         const timespecX & ts,          ///< [in] the timestamp of this log entry.
                         const typename logT::messageT & msg, ///< [in] the message to log (could be of type emptyMessage) 
                         const logPrioT & level              ///< [in] the level (verbosity) of this log
                       );
   
   ///Extract the basic details of a log entry
   /** Convenience wrapper for the other extraction functions.
     * 
     * \returns 0 on success, -1 on error.
     * 
     * \ingroup logbuff
     */
   static int extractBasicLog( logPrioT & lvl,        ///< [out] The log level
                               eventCodeT & ec,       ///< [out] the event code
                               timespecX & ts,        ///< [out] the timestamp of the log entry
                               msgLenT & len,         ///< [out] the message length
                               bufferPtrT & logBuffer ///< [in] a shared_ptr\<char\> containing a raw log entry buffer.
                             );
   
   ///Extract the basic details of a log entry
   /** Convenience wrapper for the other extraction functions.
     * 
     * \returns 0 on success, -1 on error.
     * 
     * \ingroup logbuff
     */
   static int extractBasicLog( logPrioT & lvl,  ///< [out] The log level
                               eventCodeT & ec, ///< [out] the event code
                               timespecX & ts,  ///< [out] the timestamp of the log entry
                               msgLenT & len,   ///< [out] the message length
                               char* logBuffer  ///< [in] a pointer to a raw log entry buffer.
                             );
   
};

inline
int logHeader::logLevel( bufferPtrT & logBuffer,
                          const logPrioT & lvl
                        )
{
   reinterpret_cast<internal_logHeader *>(logBuffer.get())->m_logLevel = lvl;
   
   return 0;
}

inline
logPrioT logHeader::logLevel( bufferPtrT & logBuffer)
{
   return logLevel( logBuffer.get() );
   
   //return reinterpret_cast<internal_logHeader *>(logBuffer.get())->m_logLevel;
}

inline
logPrioT logHeader::logLevel( char * logBuffer)
{
   return reinterpret_cast<internal_logHeader *>(logBuffer)->m_logLevel;
}

inline
int logHeader::eventCode( bufferPtrT & logBuffer,
                           const eventCodeT & ec
                         )
{
   reinterpret_cast<internal_logHeader *>(logBuffer.get())->m_eventCode = ec;
   
   return 0;
}

inline
eventCodeT logHeader::eventCode( bufferPtrT & logBuffer)
{
   return eventCode(logBuffer.get());
   //return reinterpret_cast<internal_logHeader *>(logBuffer.get())->m_eventCode;
}

inline
eventCodeT logHeader::eventCode( char * logBuffer)
{
   return reinterpret_cast<internal_logHeader *>(logBuffer)->m_eventCode;
}

inline
int logHeader::timespec( bufferPtrT & logBuffer,
                          const timespecX & ts
                        )
{
   reinterpret_cast<internal_logHeader *>(logBuffer.get())->m_timespecX = ts;
   
   return 0;
}

inline
timespecX logHeader::timespec( bufferPtrT & logBuffer)
{
   return timespec(logBuffer.get());
   //return reinterpret_cast<internal_logHeader *>(logBuffer.get())->m_timespecX;
}

inline
timespecX logHeader::timespec( char * logBuffer)
{
   return reinterpret_cast<internal_logHeader *>(logBuffer)->m_timespecX;
}

inline
size_t logHeader::lenSize( bufferPtrT & logBuffer )
{
   return lenSize(logBuffer.get());
//    msgLen0T len0 = reinterpret_cast<internal_logHeader *>(logBuffer.get())->msgLen0;
//    
//    if(len0 < MAX_LEN0-1) return sizeof(msgLen0T);
//    
//    if(len0 == MAX_LEN0-1)
//    {
//       return sizeof(msgLen0T) + sizeof(msgLen1T);
//    }
// 
//    return sizeof(msgLen0T) + sizeof(msgLen2T);
}

inline
size_t logHeader::lenSize( char* logBuffer )
{
   msgLen0T len0 = reinterpret_cast<internal_logHeader *>(logBuffer)->msgLen0;
   
   if(len0 < MAX_LEN0-1) return sizeof(msgLen0T);
   
   if(len0 == MAX_LEN0-1)
   {
      return sizeof(msgLen0T) + sizeof(msgLen1T);
   }

   return sizeof(msgLen0T) + sizeof(msgLen2T);
}

inline
size_t logHeader::lenSize(  msgLenT & msgSz /**< [in] the size of the intended message.*/)
{
   if(msgSz < MAX_LEN0-1) return sizeof(msgLen0T);
   
   if(msgSz < MAX_LEN1) return sizeof(msgLen0T) + sizeof(msgLen1T);
   
   return sizeof(msgLen0T) + sizeof(msgLen2T);
}

inline
size_t logHeader::headerSize(  bufferPtrT & logBuffer )
{
   return headerSize(logBuffer.get());
   //return sizeof(logPrioT) + sizeof(eventCodeT) + sizeof(secT) + sizeof(nanosecT) + lenSize(logBuffer);
}

inline
size_t logHeader::headerSize(  char* logBuffer )
{
   return sizeof(logPrioT) + sizeof(eventCodeT) + sizeof(secT) + sizeof(nanosecT) + lenSize(logBuffer);
}

inline
size_t logHeader::headerSize(  msgLenT & msgSz )
{
   return sizeof(logPrioT) + sizeof(eventCodeT) + sizeof(timespecX) + lenSize(msgSz);
}

inline
int logHeader::msgLen( bufferPtrT & logBuffer,
                        const msgLenT & msgLen
                      )
{
   internal_logHeader * lh = reinterpret_cast<internal_logHeader *>(logBuffer.get());
   
   if( msgLen < MAX_LEN0-1 ) //254 for uint8_t
   {
      lh->msgLen0 = msgLen;
      return 0;
   }
   
   if(msgLen < MAX_LEN1) //65535 for uint16_t
   {
      lh->msgLen0 = MAX_LEN0-1; //254 for uint8_t
      lh->msgLen1 = msgLen;
      return 0;
   }
   
   lh->msgLen0 = MAX_LEN0; //255 for  uint8_t
   lh->msgLen2 = msgLen;
   return 0;
}

inline
msgLen0T logHeader::msgLen0( bufferPtrT & logBuffer )
{
   return msgLen0(logBuffer.get());
   //return reinterpret_cast<internal_logHeader *>(logBuffer.get())->msgLen0;
}

inline
msgLen0T logHeader::msgLen0( char * logBuffer )
{
   return reinterpret_cast<internal_logHeader *>(logBuffer)->msgLen0;
}

inline
msgLen1T logHeader::msgLen1( bufferPtrT & logBuffer )
{
   return msgLen1(logBuffer.get());
   //return reinterpret_cast<internal_logHeader *>(logBuffer.get())->msgLen1;
}

inline
msgLen1T logHeader::msgLen1( char* logBuffer )
{
   return reinterpret_cast<internal_logHeader *>(logBuffer)->msgLen1;
}

inline
msgLenT logHeader::msgLen(  bufferPtrT & logBuffer )
{
   return msgLen(logBuffer.get());
//    msgLen0T len0 = reinterpret_cast<internal_logHeader *>(logBuffer.get())->msgLen0;
//    
//    if(len0 < MAX_LEN0-1) return len0;
//    
//    if(len0 == MAX_LEN0-1)
//    {
//       return reinterpret_cast<internal_logHeader *>(logBuffer.get())->msgLen1;
//    }
//    
//    return reinterpret_cast<internal_logHeader *>(logBuffer.get())->msgLen2;
}

inline
msgLenT logHeader::msgLen(  char* logBuffer )
{
   msgLen0T len0 = reinterpret_cast<internal_logHeader *>(logBuffer)->msgLen0;
   
   //std::cerr << "len0: " << (int) len0 << "\n";
   
   if(len0 < MAX_LEN0-1) return len0;
   
   if(len0 == MAX_LEN0-1)
   {
      return reinterpret_cast<internal_logHeader *>(logBuffer)->msgLen1;
   }
   
   return reinterpret_cast<internal_logHeader *>(logBuffer)->msgLen2;
}

inline
size_t logHeader::totalSize(  bufferPtrT & logBuffer )
{
   return totalSize(logBuffer.get());
   //return headerSize(logBuffer) + msgLen(logBuffer);
}

inline
size_t logHeader::totalSize( char* logBuffer )
{
   return headerSize(logBuffer) + msgLen(logBuffer);
}

inline
size_t logHeader::totalSize(  msgLenT & msgSz )
{
   return headerSize(msgSz) + msgSz;
}

inline
void * logHeader::messageBuffer(  bufferPtrT & logBuffer )
{
   return messageBuffer(logBuffer.get());
   //return logBuffer.get() + headerSize(logBuffer);
}

inline
void * logHeader::messageBuffer( char* logBuffer )
{
   return logBuffer + headerSize(logBuffer);
}

template<typename logT>
int logHeader::createLog( bufferPtrT & logBuffer,
                          const timespecX & ts,
                          const typename logT::messageT & msg,
                          const logPrioT & level
                        )
{
   logPrioT lvl;
   if(level == logPrio::LOG_DEFAULT) 
   {
      lvl = logT::defaultLevel;
   }
   else lvl = level;
   
   //We first allocate the buffer.
   msgLenT len = logT::length(msg);
   logBuffer = bufferPtrT( (char *) ::operator new(totalSize(len)*sizeof(char)) );

   //Now load the basics.
   logLevel(logBuffer, lvl);
   eventCode(logBuffer, +logT::eventCode); //The + fixes an issue with undefined references
   timespec(logBuffer, ts);
   
   msgLen( logBuffer, len);


   //Each log-type is responsible for loading its message
   logT::format( messageBuffer(logBuffer), msg);

   return 0;

}

inline
int logHeader::extractBasicLog( logPrioT & lvl,       
                                eventCodeT & ec,       
                                timespecX & ts,  
                                msgLenT & len,
                                bufferPtrT & logBuffer 
                              )
{
   return extractBasicLog(lvl, ec, ts, len, logBuffer.get());
//    lvl = logLevel(logBuffer); 
//    
//    ec = eventCode(logBuffer); 
// 
//    ts = timespec(logBuffer);
// 
//    len =  logHeader::msgLen(logBuffer);
//       
//    return 0;
}
   
inline
int logHeader::extractBasicLog( logPrioT & lvl,       
                                eventCodeT & ec,       
                                timespecX & ts,  
                                msgLenT & len,
                                char* logBuffer 
                              )
{
   lvl = logLevel(logBuffer); 
   
   ec = eventCode(logBuffer); 

   ts = timespec(logBuffer);

   len =  logHeader::msgLen(logBuffer);
      
   return 0;
}
   


} //namespace flatlogs

#endif //flatlogs_logHeader_hpp

