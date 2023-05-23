/** \file pico_channel.hpp
  * \brief The MagAO-X logger pico_channel log type.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_types_files
  *
  * History:
  * - 2018-08-18 created by JRM
  */
#ifndef logger_types_pico_channel_hpp
#define logger_types_pico_channel_hpp

#include "generated/pico_channel_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{


///Application State Change
/** \ingroup logger_types
  */
struct pico_channel : public flatbuffer_log
{
   //The event code
   static const flatlogs::eventCodeT eventCode = eventCodes::PICO_CHANNEL;

   //The default level
   static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_INFO;

   ///The type of the message
   struct messageT : public fbMessage
   {
      messageT( const std::string & name,
                uint8_t channel
              )
      {
         auto _name = builder.CreateString(name);
         
         auto gs = CreatePico_channel_fb(builder, _name, channel);
         builder.Finish(gs);

      }
   };

   static bool verify( flatlogs::bufferPtrT & logBuff,  ///< [in] Buffer containing the flatbuffer serialized message.
                       flatlogs::msgLenT len            ///< [in] length of msgBuffer.
                     )
   {
      auto verifier = flatbuffers::Verifier( (uint8_t*) flatlogs::logHeader::messageBuffer(logBuff), static_cast<size_t>(len));
      return VerifyPico_channel_fbBuffer(verifier);
   }

   /// Format the message for text output, including translation of state codes to text form.
   /**
     * \returns the message formatted as "State changed from UNINITIALIZED to INITIALIZED"
     */
   static std::string msgString(void * msgBuffer, flatlogs::msgLenT len)
   {
      static_cast<void>(len);

      auto rgs = GetPico_channel_fb(msgBuffer);

      
      std::string s = "Pico Motor: ";
      
      if(rgs->name())
      {
         s += rgs->name()->c_str();
      }
      
      s += " ch: ";
      s += std::to_string(rgs->channel());
         
      return s;
   }
};

} //namespace logger
} //namespace MagAOX

#endif //logger_types_pico_channel_hpp
