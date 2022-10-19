/** \file saving_state_change.hpp
  * \brief The MagAO-X logger saving_state_change log type.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_types_files
  * 
  * History:
  * - 2019-05-04 created by JRM
  */
#ifndef logger_types_saving_state_change_hpp
#define logger_types_saving_state_change_hpp

#include "../../app/stateCodes.hpp"

#include "generated/saving_state_change_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{


///Saving State Change
/** \ingroup logger_types
  */
struct saving_state_change : public flatbuffer_log
{
   ///The type of the message
   struct messageT : public fbMessage
   {
      messageT( int16_t state,
                uint64_t frameNo
              )
      {
         auto gs = CreateSaving_state_change_fb(builder, state, frameNo);
         builder.Finish(gs);
      }
   };

   /// Format the message for text output, including translation of state codes to text form.
   /**
     * \returns the message formatted as "State changed from UNINITIALIZED to INITIALIZED"
     */
   static std::string msgString( void * msgBuffer,  /**< [in] Buffer containing the flatbuffer serialized message.*/
                                 flatlogs::msgLenT len  /**< [in] [unused] length of msgBuffer.*/
                               )   
   {
      static_cast<void>(len);
      
      auto rgs = GetSaving_state_change_fb(msgBuffer);
      
      std::stringstream s;
      s << "Saving "; 
      
      if(rgs->state() == 0) s << "stopped at frame number ";
      else s << "started at frame number ";
      
      s << rgs->frameNo();
      
      return s.str();
   }
};

} //namespace logger
} //namespace MagAOX

#endif //logger_types_saving_state_change_hpp
