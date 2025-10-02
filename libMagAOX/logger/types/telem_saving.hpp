/** \file telem_saving.hpp
  * \brief The MagAO-X logger telem_saving log type.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_types_files
  *
  * History:
  * - 2019-05-04 created by JRM
  */
#ifndef logger_types_telem_saving_hpp
#define logger_types_telem_saving_hpp

#include "generated/telem_saving_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{


/// Log entry recording data saving statistics
/** \ingroup logger_types
  */
struct telem_saving : public flatbuffer_log
{
   ///The event code
   static const flatlogs::eventCodeT eventCode = eventCodes::TELEM_SAVING;

   ///The default level
   static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_TELEM;

   static timespec lastRecord; ///< The time of the last time this log was recorded.  Used by the telemetry system.

   ///The type of the input message
   struct messageT : public fbMessage
   {
      ///Construct from components
      messageT( const uint32_t & rawSize,
                const uint32_t & compressedSize,
                const float & encodeRate,
                const float & differenceRate,
                const float & reorderRate,
                const float & compressRate
              )
      {
         auto fp = Createtelem_saving_fb(builder, rawSize,compressedSize, encodeRate, differenceRate, reorderRate, compressRate);
         builder.Finish(fp);
      }

   };

   static bool verify( flatlogs::bufferPtrT & logBuff,  ///< [in] Buffer containing the flatbuffer serialized message.
                       flatlogs::msgLenT len            ///< [in] length of msgBuffer.
                     )
   {
      auto verifier = flatbuffers::Verifier( static_cast<uint8_t*>(flatlogs::logHeader::messageBuffer(logBuff)), static_cast<size_t>(len));
      return Verifytelem_saving_fbBuffer(verifier);
   }

   ///Get the message formatte for human consumption.
   static std::string msgString( void * msgBuffer,  /**< [in] Buffer containing the flatbuffer serialized message.*/
                                 flatlogs::msgLenT len  /**< [in] [unused] length of msgBuffer.*/
                               )
   {
      static_cast<void>(len);

      auto fbs = Gettelem_saving_fb(msgBuffer);


      std::stringstream s;
      s << "Saved " << ((float)fbs->raw_size())/1048576.0 << " MB @ " << ((float) fbs->compressed_size() )/((float) fbs->raw_size()) << "%.";
      return s.str();

   }

   static unsigned int raw_size( void * msgBuffer  /**< [in] Buffer containing the flatbuffer serialized message.*/ )
   {
      auto fbs = Gettelem_saving_fb(msgBuffer);

      return fbs->raw_size();
   }

   static unsigned int compressed_size( void * msgBuffer  /**< [in] Buffer containing the flatbuffer serialized message.*/ )
   {
      auto fbs = Gettelem_saving_fb(msgBuffer);

      return fbs->compressed_size();
   }

   static float encode_rate( void * msgBuffer  /**< [in] Buffer containing the flatbuffer serialized message.*/ )
   {
      auto fbs = Gettelem_saving_fb(msgBuffer);

      return fbs->encode_rate();
   }

   static float difference_rate( void * msgBuffer  /**< [in] Buffer containing the flatbuffer serialized message.*/ )
   {
      auto fbs = Gettelem_saving_fb(msgBuffer);

      return fbs->difference_rate();
   }

   static float reorder_rate( void * msgBuffer  /**< [in] Buffer containing the flatbuffer serialized message.*/ )
   {
      auto fbs = Gettelem_saving_fb(msgBuffer);

      return fbs->reorder_rate();
   }

   static float compress_rate( void * msgBuffer  /**< [in] Buffer containing the flatbuffer serialized message.*/ )
   {
      auto fbs = Gettelem_saving_fb(msgBuffer);

      return fbs->compress_rate();
   }

   /// Get the logMetaDetail for a member by name
   /**
     * \returns the a logMetaDetail filled in with the appropriate details
     * \returns an empty logMetaDetail if member not recognized
     */
   static logMetaDetail getAccessor( const std::string & member /**< [in] the name of the member */ )
   {
      if(     member == "raw_size") return logMetaDetail({"RAW SIZE", "archive raw sized", logMeta::valTypes::UInt, logMeta::metaTypes::State, reinterpret_cast<void*>(&raw_size), false});
      else if( member == "compressed_size") return logMetaDetail({"COMPRESSED SIZE", "archive compressed sized", logMeta::valTypes::UInt, logMeta::metaTypes::State, reinterpret_cast<void*>(&compressed_size), false});
      else if( member == "encode_sate") return logMetaDetail({"ENCODE RATE", "encoding rate [MB/s]", logMeta::valTypes::Double, logMeta::metaTypes::State, reinterpret_cast<void*>(&encode_rate), false});
      else if( member == "difference_rate") return logMetaDetail({"DIFFERENCING RATE", "differencing rate [MB/s]", logMeta::valTypes::Double, logMeta::metaTypes::State, reinterpret_cast<void*>(&difference_rate), false});
      else if( member == "reorder_rate") return logMetaDetail({"REORDERING RATE", "reordering rate [MB/s]", logMeta::valTypes::Double, logMeta::metaTypes::State, reinterpret_cast<void*>(&reorder_rate), false});
      else if( member == "compress_rate") return logMetaDetail({"COMPRESSION RATE", "compression rate [MB/s]", logMeta::valTypes::Double, logMeta::metaTypes::State, reinterpret_cast<void*>(&compress_rate), false});
      else
      {
         std::cerr << "No member " << member << " in telem_saving\n";
         return logMetaDetail();
      }
    }

}; //telem_saving


} //namespace logger
} //namespace MagAOX

#endif //logger_types_telem_saving_hpp
