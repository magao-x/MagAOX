/** \file ocam_temps.hpp
  * \brief The MagAO-X logger ocam_temps log type.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_types_files
  *
  * History:
  * - 2018-09-06 created by JRM
  */
#ifndef logger_types_ocam_temps_hpp
#define logger_types_ocam_temps_hpp

#include "generated/ocam_temps_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{


/// Log entry recording the build-time git state.
/** \ingroup logger_types
  */
struct ocam_temps : public flatbuffer_log
{
   ///The event code
   static const flatlogs::eventCodeT eventCode = eventCodes::OCAM_TEMPS;

   ///The default level
   static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_TELEM;

   static timespec lastRecord; ///< The time of the last time this log was recorded.  Used by the telemetry system.

   ///The type of the input message
   struct messageT : public fbMessage
   {
      ///Construct from components
      messageT( const float & ccd,    ///<[in] CCD temperature
                const float & cpu,    ///<[in] CPU temperature
                const float & power,  ///<[in] Power unit temperature
                const float & bias,   ///<[in] Bias temperature
                const float & water,  ///<[in] Water temperature
                const float & left,   ///<[in] Left temperature
                const float & right,  ///<[in] Right temperature
                const float & cooling ///<[in] Cooling power
              )
      {
         auto fp = CreateOcam_temps_fb(builder, ccd,cpu, power, bias, water, left, right, cooling);
         builder.Finish(fp);
      }

   };

   static bool verify( flatlogs::bufferPtrT & logBuff,  ///< [in] Buffer containing the flatbuffer serialized message.
                       flatlogs::msgLenT len            ///< [in] length of msgBuffer.
                     )
   {
      auto verifier = flatbuffers::Verifier( static_cast<uint8_t*>(flatlogs::logHeader::messageBuffer(logBuff)), static_cast<size_t>(len));
      return VerifyOcam_temps_fbBuffer(verifier);
   }

   ///Get the message formatte for human consumption.
   static std::string msgString( void * msgBuffer,  /**< [in] Buffer containing the flatbuffer serialized message.*/
                                 flatlogs::msgLenT len  /**< [in] [unused] length of msgBuffer.*/
                               )
   {
      static_cast<void>(len);

      auto fbs = GetOcam_temps_fb(msgBuffer);

      std::string msg = "ccd: ";
      msg += std::to_string(fbs->ccd()) + " C ";

      msg += "cpu: ";
      msg += std::to_string(fbs->cpu()) + " C ";

      msg += "pwr: ";
      msg += std::to_string(fbs->power()) + " C ";

      msg += "bias: ";
      msg += std::to_string(fbs->bias()) + " C ";

      msg += "water: ";
      msg += std::to_string(fbs->water()) + " C ";

      msg += "left: ";
      msg += std::to_string(fbs->left()) + " C ";

      msg += "right: ";
      msg += std::to_string(fbs->right()) + " C ";

      msg += "cool-pwr: ";
      msg += std::to_string(fbs->cooling()) + " mW ";

      return msg;

   }

   static float ccd( void * msgBuffer /**< [in] Buffer containing the flatbuffer serialized message.*/ )
   {
      auto fbs = GetOcam_temps_fb(msgBuffer);
      return fbs->ccd();
   }

   static float cpu( void * msgBuffer /**< [in] Buffer containing the flatbuffer serialized message.*/ )
   {
      auto fbs = GetOcam_temps_fb(msgBuffer);
      return fbs->cpu();
   }

   static float power( void * msgBuffer /**< [in] Buffer containing the flatbuffer serialized message.*/ )
   {
      auto fbs = GetOcam_temps_fb(msgBuffer);
      return fbs->power();
   }

   static float bias( void * msgBuffer /**< [in] Buffer containing the flatbuffer serialized message.*/ )
   {
      auto fbs = GetOcam_temps_fb(msgBuffer);
      return fbs->bias();
   }

   static float water( void * msgBuffer /**< [in] Buffer containing the flatbuffer serialized message.*/ )
   {
      auto fbs = GetOcam_temps_fb(msgBuffer);
      return fbs->water();
   }

   static float left( void * msgBuffer /**< [in] Buffer containing the flatbuffer serialized message.*/ )
   {
      auto fbs = GetOcam_temps_fb(msgBuffer);
      return fbs->left();
   }

   static float right( void * msgBuffer /**< [in] Buffer containing the flatbuffer serialized message.*/ )
   {
      auto fbs = GetOcam_temps_fb(msgBuffer);
      return fbs->right();
   }

   static float cooling( void * msgBuffer /**< [in] Buffer containing the flatbuffer serialized message.*/ )
   {
      auto fbs = GetOcam_temps_fb(msgBuffer);
      return fbs->cooling();
   }

   /// Get the logMetaDetail for a member by name
   /**
     * \returns the a logMetaDetail filled in with the appropriate details
     * \returns an empty logMetaDetail if member not recognized
     */
   static logMetaDetail getAccessor( const std::string & member /**< [in] the name of the member */ )
   {
      if(     member == "ccd") return logMetaDetail({"CCD TEMP", "CCD temperature [C]", logMeta::valTypes::Float, logMeta::metaTypes::Continuous, reinterpret_cast<void*>(&ccd), false});
      else if( member == "cpu") return logMetaDetail({"CPU TEMP", "CPU temperature [C]", logMeta::valTypes::Float, logMeta::metaTypes::Continuous, reinterpret_cast<void*>(&cpu), false});
      else if( member == "power") return logMetaDetail({"PWR TEMP", "power temperature [C]", logMeta::valTypes::Float, logMeta::metaTypes::Continuous, reinterpret_cast<void*>(&power), false});
      else if( member == "bias") return logMetaDetail({"BIAS TEMP", "bias temperature [C]", logMeta::valTypes::Float, logMeta::metaTypes::Continuous, reinterpret_cast<void*>(&bias), false});
      else if( member == "water") return logMetaDetail({"WATER TEMP", "water temperature [C]", logMeta::valTypes::Float, logMeta::metaTypes::Continuous, reinterpret_cast<void*>(water), false});
      else if( member == "left") return logMetaDetail({"LEFT TEMP", "left temperature [C]", logMeta::valTypes::Float, logMeta::metaTypes::Continuous, reinterpret_cast<void*>(&left), false});
      else if( member == "right") return logMetaDetail({"RIGHT TEMP", "right temperature [C]", logMeta::valTypes::Float, logMeta::metaTypes::Continuous, reinterpret_cast<void*>(&right), false});
      else if( member == "cooling") return logMetaDetail({"COOLING PWR", "cooling power [mW]", logMeta::valTypes::Float, logMeta::metaTypes::Continuous, reinterpret_cast<void*>(&cooling), false});
      else
      {
         std::cerr << "No member " << member << " in ocam_temps\n";
         return logMetaDetail();
      }
   }
}; //ocam_temps



} //namespace logger
} //namespace MagAOX

#endif //logger_types_ocam_temps_hpp

