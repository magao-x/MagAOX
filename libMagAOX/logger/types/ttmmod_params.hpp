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


/// Log entry recording the build-time git state.
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

}; //ttmmod_params


} //namespace logger
} //namespace MagAOX

#endif //logger_types_ttmmod_params_hpp
