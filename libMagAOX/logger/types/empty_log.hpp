/** \file empty_log.hpp
  * \brief The MagAO-X logger empty log base type.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_types_files
  * 
  * History:
  * - 2018-08-18 created by JRM
  */
#ifndef logger_types_empty_log_hpp
#define logger_types_empty_log_hpp


namespace MagAOX
{
namespace logger
{


///Empty type for resolving logs with no message.
/**
  * \ingroup logger_types_basic
  */
struct emptyMessage
{
};


///Base class for logs consisting of an empty message.
/** Such logs are used to log events. Does not have eventCode or defaultLevel, 
  * so this can not be used as a log type directly.
  *
  *
  * \ingroup logger_types_basic
  */
template<class derivedT>
struct empty_log 
{
   ///The type of the message
   typedef emptyMessage messageT;

   ///Get the length of the message.
   static flatlogs::msgLenT length( const messageT & msg)
   {
      static_cast<void>(msg);
      
      return 0;
   }

   ///Format the buffer given a message -- a no-op since the message is an emptyMessage.
   /**
     * \returns 0
     */
   static int format( void * msgBuffer,    ///< [out] the buffer, which is ignored.
                      const messageT & msg ///< [in] an emptyMessage.
                    )
   {
      static_cast<void>(msgBuffer);
      static_cast<void>(msg);
      
      return 0;
   }

   ///Extract the message from a buffer -- a no-op since it is an emptyMessage.
   /**
     * \returns 0
     */
   static int extract( messageT & msg,      ///< [out] an emptyMessage to which nothing is done.
                       void * msgBuffer,    ///< [in] the empty buffer.  Is ignored.
                       flatlogs::msgLenT len ///< [in] ignored length of the empty buffer.
                     )
   {
      static_cast<void>(msg);
      static_cast<void>(msgBuffer);
      static_cast<void>(len);

      
      return 0;
   }
   
   static bool verify( flatlogs::bufferPtrT & logBuff,  ///< [in] Buffer containing the flatbuffer serialized message.
                       flatlogs::msgLenT len            ///< [in] length of msgBuffer.
                     )
   {
      static_cast<void>(logBuff);
      return (len == 0);
   }

   static std::string msgString( void * msgBuffer, flatlogs::msgLenT len)//messageT & msg  /**< [in] [unused] the empty message */)
   {
      static_cast<void>(msgBuffer);
      static_cast<void>(len);
      
      return derivedT::msg();
   }

   static std::string msgJSON( void * msgBuffer,  /**< [in] Buffer containing the flatbuffer serialized message.*/
                               flatlogs::msgLenT len,  /**< [in] [unused] length of msgBuffer.*/
                               const uint8_t * binarySchema, /**< [in] [unused] */
                               const unsigned int binarySchemaLength /**< [in] [unused] */
                              )
   {
      static_cast<void>(len);
      static_cast<void>(msgBuffer);
      static_cast<void>(binarySchema);
      static_cast<void>(binarySchemaLength);
      return "{}";
   }
};

} //namespace logger
} //namespace MagAOX

#endif //logger_types_empty_log_hpp
