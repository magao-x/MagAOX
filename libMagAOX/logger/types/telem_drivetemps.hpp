/** \file telem_drivetemps.hpp
  * \brief The MagAO-X logger telem_drivetemps log type.
  * \author Chris Bohlman (cbohlmanaz@gmail.com)
  *
  * \ingroup logger_types_files
  *
  * History:
  * - 2018-10-15 created by CJB
  */
#ifndef logger_types_telem_drivetemps_hpp
#define logger_types_telem_drivetemps_hpp

#include "generated/telem_drivetemps_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{


/// Log entry recording hdd temperatures
/** \ingroup logger_types
  */
struct telem_drivetemps : public flatbuffer_log
{

  static const flatlogs::eventCodeT eventCode = eventCodes::TELEM_DRIVETEMPS;
  static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_TELEM;

  static timespec lastRecord; ///< The time of the last time this log was recorded.  Used by the telemetry system.
  
   ///The type of the input message
   struct messageT : public fbMessage
   {
      ///Construct from components
      messageT( std::vector<std::string> & names,
                std::vector<float> & temps )
      {
         
         auto _namesVec = builder.CreateVectorOfStrings(names);
         auto _tempsVec = builder.CreateVector(temps);
         
         auto fp = CreateTelem_drivetemps_fb(builder, _namesVec, _tempsVec );
         
         builder.Finish(fp);

      }

   };

   static bool verify( flatlogs::bufferPtrT & logBuff,  ///< [in] Buffer containing the flatbuffer serialized message.
                       flatlogs::msgLenT len            ///< [in] length of msgBuffer.
                     )
   {
      auto verifier = flatbuffers::Verifier( static_cast<uint8_t*>(flatlogs::logHeader::messageBuffer(logBuff)), static_cast<size_t>(len));
      return VerifyTelem_drivetemps_fbBuffer(verifier);
   }

   ///Get the message formatted for human consumption.
   static std::string msgString( void * msgBuffer,  /**< [in] Buffer containing the flatbuffer serialized message.*/
                                 flatlogs::msgLenT len  /**< [in] [unused] length of msgBuffer.*/
                               )
   {

      static_cast<void>(len); // unused by most log types
   
      auto rgs = GetTelem_drivetemps_fb(msgBuffer);  
      
      std::string msg;

      if(rgs->diskName() != nullptr && rgs->diskTemp() != nullptr) 
      {
         msg+= "[hdd temps] ";
         
         int i=0;
         for(flatbuffers::Vector<float>::const_iterator it = rgs->diskTemp()->begin(); it != rgs->diskTemp()->end(); ++it, ++i)
         {
            msg += rgs->diskName()->GetAsString(i)->c_str();
            msg += ":";
            msg+= std::to_string(*it);
            msg+= "C ";
         }
      }

      return msg;

   }

}; //telem_drivetemps



} //namespace logger
} //namespace MagAOX

#endif //logger_types_telem_drivetemps_hpp
