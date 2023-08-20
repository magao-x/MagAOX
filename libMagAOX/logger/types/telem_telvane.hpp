/** \file telem_telvane.hpp
  * \brief The MagAO-X logger telem_telvane log type.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_types_files
  * 
  * History:
  * - 2018-09-06 created by JRM
  */
#ifndef logger_types_telem_telvane_hpp
#define logger_types_telem_telvane_hpp

#include "generated/telem_telvane_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{


/// Log entry recording the build-time git state.
/** \ingroup logger_types
  */
struct telem_telvane : public flatbuffer_log
{
   ///The event code
   static const flatlogs::eventCodeT eventCode = eventCodes::TELEM_TELVANE;

   ///The default level
   static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_TELEM;

   static timespec lastRecord; ///< The time of the last time this log was recorded.  Used by the telemetry system.

   ///The type of the input message
   struct messageT : public fbMessage
   {
      ///Construct from components
      messageT( const double & secz, ///< [in] secondary z-position
                const double & encz, ///< [in] secondary z-encoder
                const double & secx, ///< [in] secondary x-position
                const double & encx, ///< [in] secondary x-encoder
                const double & secy, ///< [in] secondary y-position
                const double & ency, ///< [in] secondary y-encoder
                const double & sech, ///< [in] secondary h-position
                const double & ench, ///< [in] secondary h-encoder
                const double & secv, ///< [in] secondary v-position
                const double & encv  ///< [in] secondary v-encoder
              )
      {
         auto fp = CreateTelem_telvane_fb(builder, secz, encz, secx, encx, secy, ency,sech, ench, secv, encv);
         builder.Finish(fp);
      }

   };
                 
 
   static bool verify( flatlogs::bufferPtrT & logBuff,  ///< [in] Buffer containing the flatbuffer serialized message.
                       flatlogs::msgLenT len            ///< [in] length of msgBuffer.
                     )
   {
      auto verifier = flatbuffers::Verifier( (uint8_t*) flatlogs::logHeader::messageBuffer(logBuff), static_cast<size_t>(len));
      return VerifyTelem_telvane_fbBuffer(verifier);
   }

   ///Get the message formatte for human consumption.
   static std::string msgString( void * msgBuffer,  /**< [in] Buffer containing the flatbuffer serialized message.*/
                                 flatlogs::msgLenT len  /**< [in] [unused] length of msgBuffer.*/
                               )
   {
      static_cast<void>(len);

      auto fbs = GetTelem_telvane_fb(msgBuffer);

      std::string msg = "[telvane] ";
      
      msg += "secz: ";
      msg += std::to_string(fbs->secz()) + " ";
      
      msg += "encz: ";
      msg += std::to_string(fbs->encz()) + " ";
      
      msg += "secx: ";
      msg += std::to_string(fbs->secx()) + " ";
      
      msg += "encx: ";
      msg += std::to_string(fbs->encx()) + " ";
      
      msg += "secy: ";
      msg += std::to_string(fbs->secy()) + " ";
      
      msg += "ency: ";
      msg += std::to_string(fbs->ency()) + " ";
      
      msg += "sech: ";
      msg += std::to_string(fbs->sech()) + " ";
      
      msg += "ench: ";
      msg += std::to_string(fbs->ench()) + " ";
      
      msg += "secv: ";
      msg += std::to_string(fbs->secv()) + " ";
      
      msg += "encv: ";
      msg += std::to_string(fbs->encv()) + " ";
      
      return msg;
   
   }
   
}; //telem_telvane



} //namespace logger
} //namespace MagAOX

#endif //logger_types_telem_telvane_hpp

