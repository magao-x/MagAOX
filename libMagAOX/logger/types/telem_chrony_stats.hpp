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
                 
 
   ///Get the message formatte for human consumption.
   static std::string msgString( void * msgBuffer,  /**< [in] Buffer containing the flatbuffer serialized message.*/
                                 flatlogs::msgLenT len  /**< [in] [unused] length of msgBuffer.*/
                               )
   {
      static_cast<void>(len);

      auto fbs = GetTelem_chrony_stats_fb(msgBuffer);

      std::string msg = "[chrony_stats] ";

      msg += "sys_time: ";
      msg += std::to_string(fbs->systemTime());
      
      msg += " last_off: ";
      msg += std::to_string(fbs->lastOffset());
      
      msg += " rms_off: ";
      msg += std::to_string(fbs->rmsOffset());
      
      msg += " freq: ";
      msg += std::to_string(fbs->freq());
      
      msg += " rfreq: ";
      msg += std::to_string(fbs->residFreq());
      
      msg += " skew: ";
      msg += std::to_string(fbs->skew());
      
      msg += " root_del: ";
      msg += std::to_string(fbs->rootDelay());
      
      msg += " root_disp: ";
      msg += std::to_string(fbs->rootDispersion());
      
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
   
   /// Get pointer to the accessor for a member by name 
   /**
     * \returns the function pointer cast to void*
     * \returns -1 for an unknown member
     */ 
   static void * getAccessor( const std::string & member /**< [in] the name of the member */ )
   {
      if(member == "systemTime") return (void *) &systemTime;
      if(member == "lastOffset") return (void *) &lastOffset;
      if(member == "rmsOffset") return (void *) &rmsOffset;
      if(member == "freq") return (void *) &freq;
      if(member == "residFreq") return (void *) &residFreq;
      if(member == "skew") return (void *) &skew;
      if(member == "rootDelay") return (void *) &rootDelay;
      if(member == "rootDispersion") return (void *) &rootDispersion;
      if(member == "updateInt") return (void *) &updateInt;
      else
      {
         std::cerr << "No string member " << member << " in telem_chrony_stats\n";
         return 0;
      }
   }
   
}; //telem_chrony_stats

timespec telem_chrony_stats::lastRecord = {0,0};

} //namespace logger
} //namespace MagAOX

#endif //logger_types_telem_chrony_stats_hpp

