/** \file pdu_outlet_state.hpp
  * \brief The MagAO-X logger pdu_outlet_state log type.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_types_files
  *
  * History:
  * - 2018-08-18 created by JRM
  */
#ifndef logger_types_pdu_outlet_state_hpp
#define logger_types_pdu_outlet_state_hpp

#include "generated/pdu_outlet_state_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{


///Application State Change
/** \ingroup logger_types
  */
struct pdu_outlet_state : public flatbuffer_log
{
   //The event code
   static const flatlogs::eventCodeT eventCode = eventCodes::PDU_OUTLET_STATE;

   //The default level
   static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_NOTICE;

   ///The type of the message
   struct messageT : public fbMessage
   {
      messageT( uint8_t outlet,
                uint8_t state
              )
      {
         auto gs = CreatePdu_outlet_state_fb(builder, outlet, state);
         builder.Finish(gs);

      }
   };

   /// Format the message for text output, including translation of state codes to text form.
   /**
     * \returns the message formatted as "State changed from UNINITIALIZED to INITIALIZED"
     */
   static std::string msgString(void * msgBuffer, flatlogs::msgLenT len)
   {
      static_cast<void>(len);

      auto rgs = GetPdu_outlet_state_fb(msgBuffer);

      std::stringstream s;
      s << "Outlet: " << (int) rgs->outlet() << " ";
      if(rgs->state())
      {
         s << "ON";
      }
      else
      {
         s << "OFF";
      }

      return s.str();
   }
};

} //namespace logger
} //namespace MagAOX

#endif //logger_types_pdu_outlet_state_hpp
