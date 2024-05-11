/** \file telem_fgtimings.hpp
  * \brief The MagAO-X logger telem_fgtimings log type.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_types_files
  * 
  * History:
  * - 2022-10-03 created by JRM
  */
#ifndef logger_types_telem_fgtimings_hpp
#define logger_types_telem_fgtimings_hpp

#include "generated/telem_fgtimings_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{


/// Log entry recording framegrabber timings.
/** \ingroup logger_types
  */
struct telem_fgtimings : public flatbuffer_log
{
   ///The event code
   static const flatlogs::eventCodeT eventCode = eventCodes::TELEM_FGTIMINGS;

   ///The default level
   static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_TELEM;

   static timespec lastRecord; ///< The timestamp of the last time this log was recorded.  Used by the telemetry system.

   ///The type of the input message
   struct messageT : public fbMessage
   {
      ///Construct from components
      messageT( const double & atime,         ///< [in] acquisition time deltas
                const double & atime_jitter,  ///< [in] jitter in acquisition time deltas
                const double & wtime,         ///< [in] acquisition time deltas
                const double & wtime_jitter,  ///< [in] jitter in acquisition time deltas
                const double & mawtime,       ///< [in] acquisition time deltas
                const double & mawtime_jitter ///< [in] jitter in acquisition time deltas
              )
      {  
         auto fp = CreateTelem_fgtimings_fb(builder, atime, atime_jitter, wtime, wtime_jitter, mawtime, mawtime_jitter);
         builder.Finish(fp);
      }

   };
                 
   static bool verify( flatlogs::bufferPtrT & logBuff,  ///< [in] Buffer containing the flatbuffer serialized message.
                       flatlogs::msgLenT len            ///< [in] length of msgBuffer.
                     )
   {
      auto verifier = flatbuffers::Verifier( static_cast<uint8_t*>(flatlogs::logHeader::messageBuffer(logBuff)), static_cast<size_t>(len));
      return VerifyTelem_fgtimings_fbBuffer(verifier);
   }

   ///Get the message formatte for human consumption.
   static std::string msgString( void * msgBuffer,  /**< [in] Buffer containing the flatbuffer serialized message.*/
                                 flatlogs::msgLenT len  /**< [in] [unused] length of msgBuffer.*/
                               )
   {
      static_cast<void>(len);

      char buf[64];

      auto fbs = GetTelem_fgtimings_fb(msgBuffer);

      std::string msg = "[fgtimes]";
      
      msg += " acq: ";

      snprintf(buf, sizeof(buf), "%0.10e", fbs->atime());
      msg += buf;
      msg += " +/- ";
      snprintf(buf, sizeof(buf), "%0.5e", fbs->atime_jitter());
      msg += buf;

      msg += " wrt: ";

      snprintf(buf, sizeof(buf), "%0.10e", fbs->wtime());
      msg += buf;
      msg += " +/- ";
      snprintf(buf, sizeof(buf), "%0.5e", fbs->wtime_jitter());
      msg += buf;

      msg += " wma: ";

      snprintf(buf, sizeof(buf), "%0.10e", fbs->wmatime());
      msg += buf;
      msg += " +/- ";
      snprintf(buf, sizeof(buf), "%0.5e", fbs->wmatime_jitter());
      msg += buf;
      
      return msg;
   
   }

   static double atime( void * msgBuffer )
   {
      auto fbs = GetTelem_fgtimings_fb(msgBuffer);
      return fbs->atime();
   }

   static double atime_jitter( void * msgBuffer )
   {
      auto fbs = GetTelem_fgtimings_fb(msgBuffer);
      return fbs->atime_jitter();
   }
   
   static double wtime( void * msgBuffer )
   {
      auto fbs = GetTelem_fgtimings_fb(msgBuffer);
      return fbs->wtime();
   }

   static double wtime_jitter( void * msgBuffer )
   {
      auto fbs = GetTelem_fgtimings_fb(msgBuffer);
      return fbs->wtime_jitter();
   }

   static double wmatime( void * msgBuffer )
   {
      auto fbs = GetTelem_fgtimings_fb(msgBuffer);
      return fbs->wmatime();
   }

   static double wmatime_jitter( void * msgBuffer )
   {
      auto fbs = GetTelem_fgtimings_fb(msgBuffer);
      return fbs->wmatime_jitter();
   }

   /// Get pointer to the accessor for a member by name 
   /**
     * \returns the function pointer cast to void*
     * \returns -1 for an unknown member
     */ 
   static logMetaDetail getAccessor( const std::string & member /**< [in] the name of the member */ )
   {
      if(member == "atime") return logMetaDetail({"ACQ TIME", logMeta::valTypes::Double, logMeta::metaTypes::Continuous, reinterpret_cast<void*>(&atime)});
      else if(member == "atime_jitter") return logMetaDetail({"ACQ JITTER", logMeta::valTypes::Double, logMeta::metaTypes::Continuous, reinterpret_cast<void*>(&atime_jitter)});
      else if(member == "wtime") return logMetaDetail({"WRT TIME", logMeta::valTypes::Double, logMeta::metaTypes::Continuous, reinterpret_cast<void*>(&wtime)});
      else if(member == "wtime_jitter") return logMetaDetail({"WRT JITTER", logMeta::valTypes::Double, logMeta::metaTypes::Continuous, reinterpret_cast<void*>(&wtime_jitter)});
      else if(member == "wmatime") return logMetaDetail({"WRT-ACQ TIME", logMeta::valTypes::Double, logMeta::metaTypes::Continuous, reinterpret_cast<void*>(&wmatime)});
      else if(member == "wmatime_jitter") return logMetaDetail({"WRT-ACQ JITTER", logMeta::valTypes::Double, logMeta::metaTypes::Continuous, reinterpret_cast<void*>(&wmatime_jitter)});

      else
      {
         std::cerr << "No string member " << member << " in telem_fgtimings\n";
         return logMetaDetail();
      }
   }
   
}; //telem_fgtimings



} //namespace logger
} //namespace MagAOX

#endif //logger_types_telem_fgtimings_hpp

