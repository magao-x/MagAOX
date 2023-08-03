/** \file telem_pico.hpp
  * \brief The MagAO-X logger telem_pico log type.
  * \author Chris Bohlman (cbohlmanaz@gmail.com)
  *
  * \ingroup logger_types_files
  *
  * History:
  * - 2018-10-15 created by CJB
  */
#ifndef logger_types_telem_pico_hpp
#define logger_types_telem_pico_hpp

#include "generated/telem_pico_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{


/// Log entry recording CPU temperatures
/** \ingroup logger_types
  */
struct telem_pico : public flatbuffer_log
{

  static const flatlogs::eventCodeT eventCode = eventCodes::TELEM_PICO;
  static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_TELEM;

  static timespec lastRecord; ///< The time of the last time this log was recorded.  Used by the telemetry system.
  
   ///The type of the input message
   struct messageT : public fbMessage
   {
      ///Construct from components
      messageT( std::vector<int64_t> & counts )
      {
         
         auto _countsVec = builder.CreateVector(counts);
         
         auto fp = CreateTelem_pico_fb(builder, _countsVec );
         
         builder.Finish(fp);

      }

   };

   static bool verify( flatlogs::bufferPtrT & logBuff,  ///< [in] Buffer containing the flatbuffer serialized message.
                       flatlogs::msgLenT len            ///< [in] length of msgBuffer.
                     )
   {
      auto verifier = flatbuffers::Verifier( (uint8_t*) flatlogs::logHeader::messageBuffer(logBuff), static_cast<size_t>(len));
      return VerifyTelem_pico_fbBuffer(verifier);
   }

   ///Get the message formatte for human consumption.
   static std::string msgString( void * msgBuffer,  /**< [in] Buffer containing the flatbuffer serialized message.*/
                                 flatlogs::msgLenT len  /**< [in] [unused] length of msgBuffer.*/
                               )
   {

      static_cast<void>(len); // unused by most log types
   
      auto rgs = GetTelem_pico_fb(msgBuffer);  
      
      std::string msg;

      if (rgs->counts() != nullptr) 
      {
         msg+= "[pico pos] ";
         for(flatbuffers::Vector<int64_t>::const_iterator it = rgs->counts()->begin(); it != rgs->counts()->end(); ++it)
         {
            msg+= std::to_string(*it);
            msg+= " ";
         }
      }

      return msg;

   }

}; //telem_pico



} //namespace logger
} //namespace MagAOX

#endif //logger_types_telem_pico_hpp
