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
      auto verifier = flatbuffers::Verifier( (uint8_t*) flatlogs::logHeader::messageBuffer(logBuff), static_cast<size_t>(len));
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
   
}; //ocam_temps



} //namespace logger
} //namespace MagAOX

#endif //logger_types_ocam_temps_hpp

