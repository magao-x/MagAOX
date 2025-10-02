/** \file ttmmod_params.hpp
  * \brief The MagAO-X logger ttmmod_params log type.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_types_files
  *
  * History:
  * - 2018-09-06 created by JRM
  */
#ifndef logger_types_ttmmod_params_hpp
#define logger_types_ttmmod_params_hpp

#include "generated/ttmmod_params_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{


/// Log entry recording the modulator state
/** \ingroup logger_types
  */
struct ttmmod_params : public flatbuffer_log
{
   ///The event code
   static const flatlogs::eventCodeT eventCode = eventCodes::TTMMOD_PARAMS;

   ///The default level
   static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_INFO;


   ///The type of the input message
   struct messageT : public fbMessage
   {
      ///Construct from components
      messageT( const uint8_t & modState,
                const double & modFreq,
                const double & modRad,
                const double & offset1,
                const double & offset2
              )
      {
         auto fp = CreateTtmmod_params_fb(builder, modState, modFreq, modRad, offset1, offset2);
         builder.Finish(fp);

      }

   };

   static bool verify( flatlogs::bufferPtrT & logBuff,  ///< [in] Buffer containing the flatbuffer serialized message.
                       flatlogs::msgLenT len            ///< [in] length of msgBuffer.
                     )
   {
      auto verifier = flatbuffers::Verifier( static_cast<uint8_t*>(flatlogs::logHeader::messageBuffer(logBuff)), static_cast<size_t>(len));
      return VerifyTtmmod_params_fbBuffer(verifier);
   }

   ///Get the message formatte for human consumption.
   static std::string msgString( void * msgBuffer,  /**< [in] Buffer containing the flatbuffer serialized message.*/
                                 flatlogs::msgLenT len  /**< [in] [unused] length of msgBuffer.*/
                               )
   {
      static_cast<void>(len);

      auto fbs = GetTtmmod_params_fb(msgBuffer);


      std::string msg = "UNK";

      if(fbs->modState() == 0) msg = "OFF";
      if(fbs->modState() == 1) msg = "REST";
      if(fbs->modState() == 2) msg = "INT";
      if(fbs->modState() == 3) msg = "SET";
      if(fbs->modState() == 4)
      {
         msg = "MOD";

         msg += " Freq: ";
         msg += std::to_string(fbs->modFreq()) + " Hz";

         msg += " Rad: ";
         msg += std::to_string(fbs->modRad()) + " lam/D";
      }

      return msg;

   }

   static std::string modState(void * msgBuffer  /**< [in] Buffer containing the flatbuffer serialized message.*/)
   {
      auto fbs = GetTtmmod_params_fb(msgBuffer);

      std::string msg = "UNK";

      if(fbs->modState() == 0) msg = "OFF";
      if(fbs->modState() == 1) msg = "REST";
      if(fbs->modState() == 2) msg = "INT";
      if(fbs->modState() == 3) msg = "SET";
      if(fbs->modState() == 4)
      {
         msg = "MOD";
      }

      return msg;
   }

   static double modFreq(void * msgBuffer  /**< [in] Buffer containing the flatbuffer serialized message.*/)
   {
      auto fbs = GetTtmmod_params_fb(msgBuffer);
      return fbs->modFreq();
   }

   static double modRad(void * msgBuffer  /**< [in] Buffer containing the flatbuffer serialized message.*/)
   {
      auto fbs = GetTtmmod_params_fb(msgBuffer);
      return fbs->modRad();
   }

   static double offset1(void * msgBuffer  /**< [in] Buffer containing the flatbuffer serialized message.*/)
   {
      auto fbs = GetTtmmod_params_fb(msgBuffer);
      return fbs->offset1();
   }

   static double offset2(void * msgBuffer  /**< [in] Buffer containing the flatbuffer serialized message.*/)
   {
      auto fbs = GetTtmmod_params_fb(msgBuffer);
      return fbs->offset2();
   }

   /// Get the logMetaDetail for a member by name
   /**
     * \returns the a logMetaDetail filled in with the appropriate details
     * \returns an empty logMetaDetail if member not recognized
     */
   static logMetaDetail getAccessor( const std::string & member /**< [in] the name of the member */ )
   {
      if(     member == "modState") return logMetaDetail({"MOD STATE", "modulator state", logMeta::valTypes::String, logMeta::metaTypes::State, reinterpret_cast<void*>(&modState), false});
      else if(member == "modFreq") return logMetaDetail({"MOD FREQ", "modulator frequency [Hz]", logMeta::valTypes::Double, logMeta::metaTypes::State, reinterpret_cast<void*>(&modFreq), false});
      else if(member == "modRad") return logMetaDetail({"MOD RAD", "modulator radius [lam/D]", logMeta::valTypes::Double, logMeta::metaTypes::State, reinterpret_cast<void*>(&modRad), false});
      else if(member == "offset1") return logMetaDetail({"MOD OFFSET1", "modulator axis1 offset [V]", logMeta::valTypes::Double, logMeta::metaTypes::State, reinterpret_cast<void*>(&offset1), false});
      else if(member == "offset2") return logMetaDetail({"MOD OFFSET2", "modulator axis2 offset [V]", logMeta::valTypes::Double, logMeta::metaTypes::State, reinterpret_cast<void*>(&offset1), false});
      else
      {
         std::cerr << "No member " << member << " in ttmmod_params\n";
         return logMetaDetail();
      }
    }

}; //ttmmod_params


} //namespace logger
} //namespace MagAOX

#endif //logger_types_ttmmod_params_hpp
