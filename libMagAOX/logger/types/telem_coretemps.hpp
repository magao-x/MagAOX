/** \file telem_coretemps.hpp
  * \brief The MagAO-X logger telem_coretemps log type.
  * \author Chris Bohlman (cbohlmanaz@gmail.com)
  *
  * \ingroup logger_types_files
  *
  * History:
  * - 2018-10-15 created by CJB
  */
#ifndef logger_types_telem_coretemps_hpp
#define logger_types_telem_coretemps_hpp

#include "generated/telem_coretemps_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{


/// Log entry recording CPU temperatures
/** \ingroup logger_types
  */
struct telem_coretemps : public flatbuffer_log
{

  static const flatlogs::eventCodeT eventCode = eventCodes::TELEM_CORETEMPS;
  static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_TELEM;

  static timespec lastRecord; ///< The time of the last time this log was recorded.  Used by the telemetry system.
  
   ///The type of the input message
   struct messageT : public fbMessage
   {
      ///Construct from components
      messageT( std::vector<float> & temps )
      {
         
         auto _coreTempsVec = builder.CreateVector(temps);
         
         auto fp = CreateTelem_coretemps_fb(builder, _coreTempsVec );
         
         builder.Finish(fp);

      }

   };

   static bool verify( flatlogs::bufferPtrT & logBuff,  ///< [in] Buffer containing the flatbuffer serialized message.
                       flatlogs::msgLenT len            ///< [in] length of msgBuffer.
                     )
   {
      auto verifier = flatbuffers::Verifier( static_cast<uint8_t*>(flatlogs::logHeader::messageBuffer(logBuff)), static_cast<size_t>(len));
      return VerifyTelem_coretemps_fbBuffer(verifier);
   }

   ///Get the message formatte for human consumption.
   static std::string msgString( void * msgBuffer,  /**< [in] Buffer containing the flatbuffer serialized message.*/
                                 flatlogs::msgLenT len  /**< [in] [unused] length of msgBuffer.*/
                               )
   {

      static_cast<void>(len); // unused by most log types
   
      auto rgs = GetTelem_coretemps_fb(msgBuffer);  
      
      std::string msg;

      if (rgs->temps() != nullptr) 
      {
         msg+= "[cpu temps] ";
         for(flatbuffers::Vector<float>::const_iterator it = rgs->temps()->begin(); it != rgs->temps()->end(); ++it)
         {
            msg+= std::to_string(*it);
            msg+= " ";
         }
      }

      return msg;

   }

}; //telem_coretemps



} //namespace logger
} //namespace MagAOX

#endif //logger_types_telem_coretemps_hpp
