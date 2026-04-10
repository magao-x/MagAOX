/** \file telem_telsee.hpp
  * \brief The MagAO-X logger telem_telsee log type.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_types_files
  *
  * History:
  * - 2018-09-06 created by JRM
  */
#ifndef logger_types_telem_telsee_hpp
#define logger_types_telem_telsee_hpp

#include "generated/telem_telsee_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{


/// Log entry recording the build-time git state.
/** \ingroup logger_types
  */
struct telem_telsee : public flatbuffer_log
{
   ///The event code
   static const flatlogs::eventCodeT eventCode = eventCodes::TELEM_TELSEE;

   ///The default level
   static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_TELEM;

   static timespec lastRecord; ///< The time of the last time this log was recorded.  Used by the telemetry system.

   ///The type of the input message
   struct messageT : public fbMessage
   {
      ///Construct from components
      messageT( const int & dimm_time,         ///< [in]
                const double & dimm_fwhm_corr, ///< [in]
                const int & mag1_time,         ///< [in]
                const double & mag1_fwhm_corr, ///< [in]
                const int & mag2_time,         ///< [in]
                const double & mag2_fwhm_corr  ///< [in]
              )
      {
         auto fp = CreateTelem_telsee_fb(builder, dimm_time, dimm_fwhm_corr, mag1_time, mag1_fwhm_corr, mag2_time, mag2_fwhm_corr);
         builder.Finish(fp);
      }

   };

   static bool verify( flatlogs::bufferPtrT & logBuff,  ///< [in] Buffer containing the flatbuffer serialized message.
                       flatlogs::msgLenT len            ///< [in] length of msgBuffer.
                     )
   {
      auto verifier = flatbuffers::Verifier( static_cast<uint8_t*>(flatlogs::logHeader::messageBuffer(logBuff)), static_cast<size_t>(len));
      return VerifyTelem_telsee_fbBuffer(verifier);
   }

   ///Get the message formatte for human consumption.
   static std::string msgString( void * msgBuffer,  /**< [in] Buffer containing the flatbuffer serialized message.*/
                                 flatlogs::msgLenT len  /**< [in] [unused] length of msgBuffer.*/
                               )
   {
      static_cast<void>(len);

      auto fbs = GetTelem_telsee_fb(msgBuffer);

      std::string msg = "[telsee] ";

      msg += "dimm[ ";

      msg += "t: ";
      msg += std::to_string(fbs->dimm_time()) + " ";

      msg += "fw-cor: ";
      msg += std::to_string(fbs->dimm_fwhm_corr()) + "] ";

      msg += "mag1[ ";

      msg += "t: ";
      msg += std::to_string(fbs->mag1_time()) + " ";

      msg += "fw-cor: ";
      msg += std::to_string(fbs->mag1_fwhm_corr()) + "] ";


      msg += "mag2[ ";

      msg += "t: ";
      msg += std::to_string(fbs->mag2_time()) + " ";

      msg += "fw-cor: ";
      msg += std::to_string(fbs->mag2_fwhm_corr()) + "] ";
      return msg;

   }

    static int dimm_time( void * msgBuffer  /**< [in] Buffer containing the flatbuffer serialized message.*/ )
    {
        auto fbs = GetTelem_telsee_fb(msgBuffer);
        return fbs->dimm_time();
    }

    static double dimm_fwhm_corr( void * msgBuffer  /**< [in] Buffer containing the flatbuffer serialized message.*/ )
    {
        auto fbs = GetTelem_telsee_fb(msgBuffer);
        return fbs->dimm_fwhm_corr();
    }

    static int mag1_time( void * msgBuffer  /**< [in] Buffer containing the flatbuffer serialized message.*/ )
    {
        auto fbs = GetTelem_telsee_fb(msgBuffer);
        return fbs->mag1_time();
    }

    static double mag1_fwhm_corr( void * msgBuffer  /**< [in] Buffer containing the flatbuffer serialized message.*/ )
    {
        auto fbs = GetTelem_telsee_fb(msgBuffer);
        return fbs->mag1_fwhm_corr();
    }

    static int mag2_time( void * msgBuffer  /**< [in] Buffer containing the flatbuffer serialized message.*/ )
    {
        auto fbs = GetTelem_telsee_fb(msgBuffer);
        return fbs->mag2_time();
    }

    static double mag2_fwhm_corr( void * msgBuffer  /**< [in] Buffer containing the flatbuffer serialized message.*/ )
    {
        auto fbs = GetTelem_telsee_fb(msgBuffer);
        return fbs->mag2_fwhm_corr();
    }

   /// Get the logMetaDetail for a member by name
   /**
     * \returns the a logMetaDetail filled in with the appropriate details
     * \returns an empty logMetaDetail if member not recognized
     */
   static logMetaDetail getAccessor( const std::string & member /**< [in] the name of the member */ )
   {
      if(     member == "dimm_time") return logMetaDetail({"DIMM TIME", "DIMM meas. time [sec since midnight]", logMeta::valTypes::Int, logMeta::metaTypes::State, reinterpret_cast<void*>(&dimm_time), false});
      else if( member == "dimm_fwhm_corr") return logMetaDetail({"DIMM FWHM CORR", "DIMM FWHM corrected to zenith", logMeta::valTypes::Double, logMeta::metaTypes::State, reinterpret_cast<void*>(&dimm_fwhm_corr), false});
      else if( member == "mag1_time") return logMetaDetail({"MAG1 TIME", "Baade meas. time [sec since midnight]", logMeta::valTypes::Int, logMeta::metaTypes::State, reinterpret_cast<void*>(&mag1_time), false});
      else if( member == "mag1_fwhm_corr") return logMetaDetail({"MAG1 FWHM CORR", "Baade FWHM corrected to zenith", logMeta::valTypes::Double, logMeta::metaTypes::State, reinterpret_cast<void*>(&mag1_fwhm_corr), false});
      else if( member == "mag2_time") return logMetaDetail({"MAG2 TIME", "Clay meas. time [sec since midnight]", logMeta::valTypes::Int, logMeta::metaTypes::State, reinterpret_cast<void*>(&mag2_time), false});
      else if( member == "mag2_fwhm_corr") return logMetaDetail({"MAG2 FWHM CORR", "Clay FWHM corrected to zenith", logMeta::valTypes::Double, logMeta::metaTypes::State, reinterpret_cast<void*>(&mag2_fwhm_corr), false});
      else
      {
         std::cerr << "No member " << member << " in telem_saving\n";
         return logMetaDetail();
      }
    }

}; //telem_telsee



} //namespace logger
} //namespace MagAOX

#endif //logger_types_telem_telsee_hpp

