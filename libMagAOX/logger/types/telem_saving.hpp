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
      auto verifier = flatbuffers::Verifier( (uint8_t*) flatlogs::logHeader::messageBuffer(logBuff), static_cast<size_t>(len));
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
      s << "Saved " << ((float)fbs->rawSize())/1048576.0 << " MB @ " << ((float) fbs->compressedSize() )/((float) fbs->rawSize()) << "%.";
      return s.str();

   }

}; //telem_saving


} //namespace logger
} //namespace MagAOX

#endif //logger_types_telem_saving_hpp
