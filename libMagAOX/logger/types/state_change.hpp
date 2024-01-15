/** \file state_change.hpp
  * \brief The MagAO-X logger state_change log type.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_types_files
  * 
  * History:
  * - 2018-08-18 created by JRM
  */
#ifndef logger_types_state_change_hpp
#define logger_types_state_change_hpp

#include "../../app/stateCodes.hpp"

#include "generated/state_change_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{


///Application State Change
/** \ingroup logger_types
  */
struct state_change : public flatbuffer_log
{
   //The event code
   static const flatlogs::eventCodeT eventCode = eventCodes::STATE_CHANGE;

   //The default level
   static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_INFO;

   ///The type of the message
   struct messageT : public fbMessage
   {
      messageT( int16_t from,
                int16_t to
              )
      {
         auto gs = CreateState_change_fb(builder, from, to);
         builder.Finish(gs);
      }
   };

   static bool verify( flatlogs::bufferPtrT & logBuff,  ///< [in] Buffer containing the flatbuffer serialized message.
                       flatlogs::msgLenT len            ///< [in] length of msgBuffer.
                     )
   {
      auto verifier = flatbuffers::Verifier( static_cast<uint8_t*>(flatlogs::logHeader::messageBuffer(logBuff)), static_cast<size_t>(len));
      return VerifyState_change_fbBuffer(verifier);
   }

   /// Format the message for text output, including translation of state codes to text form.
   /**
     * \returns the message formatted as "State changed from UNINITIALIZED to INITIALIZED"
     */
   static std::string msgString( void * msgBuffer,  /**< [in] Buffer containing the flatbuffer serialized message.*/
                                 flatlogs::msgLenT len  /**< [in] [unused] length of msgBuffer.*/
                               )   
   {
      static_cast<void>(len);
      
      auto rgs = GetState_change_fb(msgBuffer);
      
      std::stringstream s;
      s << "State changed from " << app::stateCodes::codeText(rgs->from()) << " to " << app::stateCodes::codeText(rgs->to());
      return s.str();
   }

};

} //namespace logger
} //namespace MagAOX

#endif //logger_types_state_change_hpp
