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

#include "flatbuffers/flatbuffers.h"
#include "flatbuffers/idl.h"

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

   static std::string msgJSON( void * msgBuffer,  /**< [in] Buffer containing the flatbuffer serialized message.*/
                               flatlogs::msgLenT len,  /**< [in] [unused] length of msgBuffer.*/
                               const uint8_t * binarySchema, /**< [in] flatbuffers binary schema for this log type */
                               const unsigned int binarySchemaLength /**< [in] flatbuffers binary schema length */
                              )
   {
      static_cast<void>(len);
      flatbuffers::Parser parser;
      parser.opts.output_default_scalars_in_json = true;
      parser.opts.output_enum_identifiers = true;
      parser.opts.strict_json = true;
      parser.opts.indent_step = -1;  // also disables line breaking within record
      bool ok = parser.Deserialize(binarySchema, binarySchemaLength);
      if(!ok) {
         std::cerr << __FILE__ << ":" << __LINE__ << " Failed to deserialize binary schema\n";
      }
      std::string output;
      flatbuffers::GenText(parser, msgBuffer, &output);
      return output;
   }

};



} //namespace logger
} //namespace MagAOX

#endif //logger_types_flatbuffer_log_hpp
