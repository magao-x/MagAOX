/** \file flatbuffer_log.hpp
  * \brief The MagAO-X logger flatbuffer log base type.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_types_files
  * 
  * History:
  * - 2018-08-18 created by JRM
  */
#ifndef logger_types_flatbuffer_log_hpp
#define logger_types_flatbuffer_log_hpp

namespace MagAOX
{
namespace logger
{


///Message type for resolving log messages with a f.b. builder.
/**
  * \ingroup logger_types_basic
  */
struct fbMessage
{
   flatbuffers::FlatBufferBuilder builder;
};


///Base class for logs consisting of a flatbuffer message.
/** Such logs are used to log arbitrary data structures using the flatbuffer protocol. Does not have eventCode or defaultLevel, 
  * so this can not be used as a log type directly.
  *
  *
  * \ingroup logger_types_basic
  */
struct flatbuffer_log 
{
   ///Get the length of the message.
   static flatlogs::msgLenT length( const fbMessage & msg /**< [in] the fbMessage type holding a FlatBufferBuilder */)
   {
      return msg.builder.GetSize();      
   }

   ///Format the buffer given the input message.
   /** \todo this is an unneccesary memcpy from the FlatBufferBuilder, we need to figure out how to not do this.
     */
   static int format( void * msgBuffer,    ///< [out] the buffer, must be pre-allocated to size length(msg)
                      const fbMessage & msg ///< [in] the message which contains a flatbuffer builder, from which the data are memcpy-ed.
                    )
   {
      uint8_t * cbuff = reinterpret_cast<uint8_t *>(msgBuffer);

      memcpy(cbuff, msg.builder.GetBufferPointer(), msg.builder.GetSize());
      
      return 0;
   }

};

} //namespace logger
} //namespace MagAOX

#endif //logger_types_flatbuffer_log_hpp
