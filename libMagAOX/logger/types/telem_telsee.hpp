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
                const double & dimm_el,        ///< [in] 
                const double & dimm_fwhm,      ///< [in] 
                const double & dimm_fwhm_corr, ///< [in]
                const int & mag1_time,         ///< [in] 
                const double & mag1_el,        ///< [in] 
                const double & mag1_fwhm,      ///< [in] 
                const double & mag1_fwhm_corr, ///< [in]
                const int & mag2_time,         ///< [in] 
                const double & mag2_el,        ///< [in] 
                const double & mag2_fwhm,      ///< [in] 
                const double & mag2_fwhm_corr  ///< [in]
              )
      {
         auto fp = CreateTelem_telsee_fb(builder, dimm_time, dimm_el, dimm_fwhm, dimm_fwhm_corr, mag1_time, mag1_el, mag1_fwhm, mag1_fwhm_corr, mag2_time, mag2_el, mag2_fwhm, mag2_fwhm_corr);
         builder.Finish(fp);
      }

   };
                 
   static bool verify( flatlogs::bufferPtrT & logBuff,  ///< [in] Buffer containing the flatbuffer serialized message.
                       flatlogs::msgLenT len            ///< [in] length of msgBuffer.
                     )
   {
      auto verifier = flatbuffers::Verifier( (uint8_t*) flatlogs::logHeader::messageBuffer(logBuff), static_cast<size_t>(len));
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
      
      msg += "el: ";
      msg += std::to_string(fbs->dimm_el()) + " ";
      
      msg += "fw: ";
      msg += std::to_string(fbs->dimm_fwhm()) + " ";
      
      msg += "fw-cor: ";
      msg += std::to_string(fbs->dimm_fwhm_corr()) + "] ";
      
      msg += "mag1[ ";
      
      msg += "t: ";
      msg += std::to_string(fbs->mag1_time()) + " ";
      
      msg += "el: ";
      msg += std::to_string(fbs->mag1_el()) + " ";
      
      msg += "fw: ";
      msg += std::to_string(fbs->mag1_fwhm()) + " ";
      
      msg += "fw-cor: ";
      msg += std::to_string(fbs->mag1_fwhm_corr()) + "] ";
      
      
      msg += "mag2[ ";
      
      msg += "t: ";
      msg += std::to_string(fbs->mag2_time()) + " ";
      
      msg += "el: ";
      msg += std::to_string(fbs->mag2_el()) + " ";
      
      msg += "fw: ";
      msg += std::to_string(fbs->mag2_fwhm()) + " ";
      
      msg += "fw-cor: ";
      msg += std::to_string(fbs->mag2_fwhm_corr()) + "] ";
      return msg;
   
   }
   
}; //telem_telsee



} //namespace logger
} //namespace MagAOX

#endif //logger_types_telem_telsee_hpp

