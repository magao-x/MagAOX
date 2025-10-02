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
      auto verifier = flatbuffers::Verifier( static_cast<uint8_t*>(flatlogs::logHeader::messageBuffer(logBuff)), static_cast<size_t>(len));
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

   static double secz( void * msgBuffer  /**< [in] Buffer containing the flatbuffer serialized message.*/ )
   {
      auto fbs = GetTelem_telvane_fb(msgBuffer);

      return fbs->secz();
   }

   static double encz( void * msgBuffer  /**< [in] Buffer containing the flatbuffer serialized message.*/ )
   {
      auto fbs = GetTelem_telvane_fb(msgBuffer);

      return fbs->encz();
   }

   static double secx( void * msgBuffer  /**< [in] Buffer containing the flatbuffer serialized message.*/ )
   {
      auto fbs = GetTelem_telvane_fb(msgBuffer);

      return fbs->secx();
   }

   static double encx( void * msgBuffer  /**< [in] Buffer containing the flatbuffer serialized message.*/ )
   {
      auto fbs = GetTelem_telvane_fb(msgBuffer);

      return fbs->encx();
   }

   static double secy( void * msgBuffer  /**< [in] Buffer containing the flatbuffer serialized message.*/ )
   {
      auto fbs = GetTelem_telvane_fb(msgBuffer);

      return fbs->secy();
   }

   static double ency( void * msgBuffer  /**< [in] Buffer containing the flatbuffer serialized message.*/ )
   {
      auto fbs = GetTelem_telvane_fb(msgBuffer);

      return fbs->ency();
   }

   static double sech( void * msgBuffer  /**< [in] Buffer containing the flatbuffer serialized message.*/ )
   {
      auto fbs = GetTelem_telvane_fb(msgBuffer);

      return fbs->sech();
   }

   static double ench( void * msgBuffer  /**< [in] Buffer containing the flatbuffer serialized message.*/ )
   {
      auto fbs = GetTelem_telvane_fb(msgBuffer);

      return fbs->ench();
   }

   static double secv( void * msgBuffer  /**< [in] Buffer containing the flatbuffer serialized message.*/ )
   {
      auto fbs = GetTelem_telvane_fb(msgBuffer);

      return fbs->secv();
   }

   static double encv( void * msgBuffer  /**< [in] Buffer containing the flatbuffer serialized message.*/ )
   {
      auto fbs = GetTelem_telvane_fb(msgBuffer);

      return fbs->encv();
   }

   /// Get the logMetaDetail for a member by name
   /**
     * \returns the a logMetaDetail filled in with the appropriate details
     * \returns an empty logMetaDetail if member not recognized
     */
   static logMetaDetail getAccessor( const std::string & member /**< [in] the name of the member */ )
   {
      if(     member == "secz") return logMetaDetail({"SECONDARY Z", "secondary z position", logMeta::valTypes::Double, logMeta::metaTypes::State, reinterpret_cast<void*>(&secz), false});
      else if( member == "encz") return logMetaDetail({"ENCODER Z", "secondary z encoder", logMeta::valTypes::Double, logMeta::metaTypes::State, reinterpret_cast<void*>(&encz), false});
      else if( member == "secx") return logMetaDetail({"SECONDARY X", "secondary X position", logMeta::valTypes::Double, logMeta::metaTypes::State, reinterpret_cast<void*>(&secx), false});
      else if( member == "encx") return logMetaDetail({"ENCODER X", "secondary X encoder", logMeta::valTypes::Double, logMeta::metaTypes::State, reinterpret_cast<void*>(&encx), false});
      else if( member == "secy") return logMetaDetail({"SECONDARY Y", "secondary Y position", logMeta::valTypes::Double, logMeta::metaTypes::State, reinterpret_cast<void*>(&secy), false});
      else if( member == "ency") return logMetaDetail({"ENCODER Y", "secondary Y encoder", logMeta::valTypes::Double, logMeta::metaTypes::State, reinterpret_cast<void*>(&ency), false});
      else if( member == "sech") return logMetaDetail({"SECONDARY H", "secondary H position", logMeta::valTypes::Double, logMeta::metaTypes::State, reinterpret_cast<void*>(&sech), false});
      else if( member == "ench") return logMetaDetail({"ENCODER H", "secondary H encoder", logMeta::valTypes::Double, logMeta::metaTypes::State, reinterpret_cast<void*>(&ench), false});
      else if( member == "secv") return logMetaDetail({"SECONDARY V", "secondary V position", logMeta::valTypes::Double, logMeta::metaTypes::State, reinterpret_cast<void*>(&secv), false});
      else if( member == "encv") return logMetaDetail({"ENCODER V", "secondary V encoder", logMeta::valTypes::Double, logMeta::metaTypes::State, reinterpret_cast<void*>(&encv), false});
      else
      {
         std::cerr << "No member " << member << " in telem_telvane\n";
         return logMetaDetail();
      }
    }
}; //telem_telvane



} //namespace logger
} //namespace MagAOX

#endif //logger_types_telem_telvane_hpp

