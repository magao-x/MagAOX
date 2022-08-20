/** \file saving_stats.hpp
  * \brief The MagAO-X logger saving_stats log type.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_types_files
  *
  * History:
  * - 2019-05-04 created by JRM
  */
#ifndef logger_types_saving_stats_hpp
#define logger_types_saving_stats_hpp

#include "generated/saving_stats_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{


/// Log entry recording data saving statistics
/** \ingroup logger_types
  */
struct saving_stats : public flatbuffer_log
{
   ///The event code
   static const flatlogs::eventCodeT eventCode = eventCodes::SAVING_STATS;

   ///The default level
   static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_INFO;


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
         auto fp = CreateSaving_stats_fb(builder, rawSize,compressedSize, encodeRate, differenceRate, reorderRate, compressRate);
         builder.Finish(fp);

      }

   };

   ///Get the message formatte for human consumption.
   static std::string msgString( void * msgBuffer,  /**< [in] Buffer containing the flatbuffer serialized message.*/
                                 flatlogs::msgLenT len  /**< [in] [unused] length of msgBuffer.*/
                               )
   {
      static_cast<void>(len);

      auto fbs = GetSaving_stats_fb(msgBuffer);


      std::stringstream s;
      s << "Saved " << ((float)fbs->rawSize())/1048576.0 << " MB @ " << ((float) fbs->compressedSize() )/((float) fbs->rawSize()) << "%.";
      return s.str();

   }

}; //saving_stats


} //namespace logger
} //namespace MagAOX

#endif //logger_types_saving_stats_hpp
