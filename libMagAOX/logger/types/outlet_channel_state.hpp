/** \file outlet_channel_state.hpp
  * \brief The MagAO-X logger outlet_channel_state log type.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_types_files
  *
  * History:
  * - 2018-08-18 created by JRM
  */
#ifndef logger_types_outlet_channel_state_hpp
#define logger_types_outlet_channel_state_hpp

#include "generated/outlet_channel_state_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{


///Application State Change
/** \ingroup logger_types
  */
struct outlet_channel_state : public flatbuffer_log
{
   //The event code
   static const flatlogs::eventCodeT eventCode = eventCodes::OUTLET_CHANNEL_STATE;

   //The default level
   static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_NOTICE;

   ///The type of the message
   struct messageT : public fbMessage
   {
      messageT( const std::string & channel,
                uint8_t state
              )
      {
         auto _channel = builder.CreateString(channel);
         
         auto gs = CreateOutlet_channel_state_fb(builder, _channel, state);
         builder.Finish(gs);

      }
   };

   static bool verify( flatlogs::bufferPtrT & logBuff,  ///< [in] Buffer containing the flatbuffer serialized message.
                       flatlogs::msgLenT len            ///< [in] length of msgBuffer.
                     )
   {
      auto verifier = flatbuffers::Verifier( static_cast<uint8_t*>(flatlogs::logHeader::messageBuffer(logBuff)), static_cast<size_t>(len));
      return VerifyOutlet_channel_state_fbBuffer(verifier);
   }

   /// Format the message for text output, including translation of state codes to text form.
   /**
     * \returns the message formatted as "State changed from UNINITIALIZED to INITIALIZED"
     */
   static std::string msgString(void * msgBuffer, flatlogs::msgLenT len)
   {
      static_cast<void>(len);

      auto rgs = GetOutlet_channel_state_fb(msgBuffer);

      std::stringstream s;
      s << "Channel: ";
      
      if(rgs->channel() != nullptr) s << rgs->channel()->c_str() << " ";
      else s << "????" << " ";
      
      if(rgs->state()==2)
      {
         s << "ON";
      }
      else if(rgs->state()==1)
      {
         s << "INT";
      }
      else if(rgs->state()==0)
      {
         s << "OFF";
      }
      else s << "UNK";

      return s.str();
   }

};

} //namespace logger
} //namespace MagAOX

#endif //logger_types_outlet_channel_state_hpp
