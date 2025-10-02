/** \file telem_chrony_stats.hpp
  * \brief The MagAO-X logger telem_chrony_stats log type.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_types_files
  *
  * History:
  * - 2018-09-06 created by JRM
  */
#ifndef logger_types_telem_chrony_stats_hpp
#define logger_types_telem_chrony_stats_hpp

#include "generated/telem_chrony_stats_generated.h"
#include "flatbuffer_log.hpp"

#include <cmath>

namespace MagAOX
{
namespace logger
{


/// Log entry recording the statistics from chrony.
/** \ingroup logger_types
  */
struct telem_chrony_stats : public flatbuffer_log
{
   ///The event code
   static const flatlogs::eventCodeT eventCode = eventCodes::TELEM_CHRONY_STATS;

   ///The default level
   static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_TELEM;

   static timespec lastRecord; ///< The time of the last time this log was recorded.  Used by the telemetry system.

   ///The type of the input message
   struct messageT : public fbMessage
   {
      ///Construct from components
      messageT( const double systemTime,     ///< [in] the error in system time
                const double lastOffset,     ///< [in] the last clock offset
                const double rmsOffset,      ///< [in] the rms avg offset
                const double freq,           ///< [in] freq drift of clock
                const double residFreq,      ///< [in] residual after correction
                const double skew,           ///< [in] skew of the drif
                const double rootDelay,      ///< [in] root delay
                const double rootDispersion, ///< [in] root dispersion
                const double updateInt       ///< [in] the update interval
              )
      {

         auto fp = CreateTelem_chrony_stats_fb(builder, systemTime, lastOffset, rmsOffset, freq, residFreq, skew, rootDelay, rootDispersion, updateInt);
         builder.Finish(fp);
      }

   };

   static bool verify( flatlogs::bufferPtrT & logBuff,  ///< [in] Buffer containing the flatbuffer serialized message.
                       flatlogs::msgLenT len            ///< [in] length of msgBuffer.
                     )
   {
      auto verifier = flatbuffers::Verifier( static_cast<uint8_t*>(flatlogs::logHeader::messageBuffer(logBuff)), static_cast<size_t>(len));
      return VerifyTelem_chrony_stats_fbBuffer(verifier);
   }

   ///Get the message formatte for human consumption.
   static std::string msgString( void * msgBuffer,  /**< [in] Buffer containing the flatbuffer serialized message.*/
                                 flatlogs::msgLenT len  /**< [in] [unused] length of msgBuffer.*/
                               )
   {
      static_cast<void>(len);
      char num[128];

      auto fbs = GetTelem_chrony_stats_fb(msgBuffer);

      std::string msg = "[chrony_stats] ";

      msg += "sys_time: ";
      snprintf(num, sizeof(num), "%g",fbs->systemTime());
      msg += num;

      msg += " last_off: ";
      snprintf(num, sizeof(num), "%g",fbs->lastOffset());
      msg += num;

      msg += " rms_off: ";
      snprintf(num, sizeof(num), "%g",fbs->rmsOffset());
      msg += num;

      msg += " freq: ";
      snprintf(num, sizeof(num), "%g",fbs->freq());
      msg += num;

      msg += " rfreq: ";
      snprintf(num, sizeof(num), "%g",fbs->residFreq());
      msg += num;

      msg += " skew: ";
      snprintf(num, sizeof(num), "%g",fbs->skew());
      msg += num;

      msg += " root_del: ";
      snprintf(num, sizeof(num), "%g",fbs->rootDelay());
      msg += num;

      msg += " root_disp: ";
      snprintf(num, sizeof(num), "%g",fbs->rootDispersion());
      msg += num;

      msg += " upd_int: ";
      msg += std::to_string(fbs->updateInt());

      return msg;

   }

   static double systemTime(void * msgBuffer )
   {
      auto fbs = GetTelem_chrony_stats_fb(msgBuffer);
      return fbs->systemTime();
   }

   static double lastOffset(void * msgBuffer )
   {
      auto fbs = GetTelem_chrony_stats_fb(msgBuffer);
      return fbs->lastOffset();
   }

   static double rmsOffset(void * msgBuffer )
   {
      auto fbs = GetTelem_chrony_stats_fb(msgBuffer);
      return fbs->rmsOffset();
   }

   static double freq(void * msgBuffer )
   {
      auto fbs = GetTelem_chrony_stats_fb(msgBuffer);
      return fbs->freq();
   }

   static double residFreq(void * msgBuffer )
   {
      auto fbs = GetTelem_chrony_stats_fb(msgBuffer);
      return fbs->residFreq();
   }

   static double skew(void * msgBuffer )
   {
      auto fbs = GetTelem_chrony_stats_fb(msgBuffer);
      return fbs->skew();
   }

   static double rootDelay(void * msgBuffer )
   {
      auto fbs = GetTelem_chrony_stats_fb(msgBuffer);
      return fbs->rootDelay();
   }

   static double rootDispersion(void * msgBuffer )
   {
      auto fbs = GetTelem_chrony_stats_fb(msgBuffer);
      return fbs->rootDispersion();
   }

   static double updateInt(void * msgBuffer )
   {
      auto fbs = GetTelem_chrony_stats_fb(msgBuffer);
      return fbs->updateInt();
   }

   /// Get the logMetaDetail for a member by name
   /**
     * \returns the a logMetaDetail filled in with the appropriate details
     * \returns an empty logMetaDetail if member not recognized
     */
   static logMetaDetail getAccessor( const std::string & member /**< [in] the name of the member */ )
   {
      if(     member == "systemTime") return logMetaDetail({"CHRONY SYSTEM TIME", "", logMeta::valTypes::Double, logMeta::metaTypes::State, reinterpret_cast<void*>(&systemTime), true});
      else if( member == "lastOffset") return logMetaDetail({ "CHRONY LAST OFFSET", "", logMeta::valTypes::Double, logMeta::metaTypes::State, reinterpret_cast<void*>(&lastOffset), true});
      else if( member == "rmsOffset") return logMetaDetail({ "CHRONY RMS OFFSET", "", logMeta::valTypes::Double, logMeta::metaTypes::State, reinterpret_cast<void*>(&rmsOffset), true});
      else if( member == "freq") return logMetaDetail({"CHRONY FREQ", "", logMeta::valTypes::Double, logMeta::metaTypes::State, reinterpret_cast<void*>(&freq), true});
      else if( member == "residFreq") return logMetaDetail({"CHRONY RESID FREQ", "", logMeta::valTypes::Double, logMeta::metaTypes::State, reinterpret_cast<void*>(&residFreq), true});
      else if( member == "skew") return logMetaDetail({"CHRONY SKEW", "", logMeta::valTypes::Double, logMeta::metaTypes::State, reinterpret_cast<void*>(&skew), true});
      else if( member == "rootDelay") return logMetaDetail({"CHRONY ROOT DELAY", "", logMeta::valTypes::Double, logMeta::metaTypes::State, reinterpret_cast<void*>(&rootDelay), true});
      else if( member == "rootDispersion") return logMetaDetail({"CHRONY ROOT DISPERSION", "", logMeta::valTypes::Double, logMeta::metaTypes::State, reinterpret_cast<void*>(&rootDispersion), true});
      else if( member == "updateInt") return logMetaDetail({"CHRONY UPDATE INT", "", logMeta::valTypes::Double, logMeta::metaTypes::State, reinterpret_cast<void*>(&updateInt), true});
      else
      {
         std::cerr << "No member " << member << " in telem_chrony_stats\n";
         return logMetaDetail();
      }
    }


}; //telem_chrony_stats

} //namespace logger
} //namespace MagAOX

#endif //logger_types_telem_chrony_stats_hpp

