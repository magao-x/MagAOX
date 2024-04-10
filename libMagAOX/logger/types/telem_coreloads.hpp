/** \file telem_coreloads.hpp
  * \brief The MagAO-X logger telem_coreloads log type.
  * \author Chris Bohlman (cbohlmanaz@gmail.com)
  *
  * \ingroup logger_types_files
  *
  * History:
  * - 2018-10-15 created by CJB
  * - 2019-11-15 refactored by JRM
  */
#ifndef logger_types_telem_coreloads_hpp
#define logger_types_telem_coreloads_hpp

#include "generated/telem_coreloads_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{


/// Log entry recording CPU loads.
/** \ingroup logger_types
  */
struct telem_coreloads : public flatbuffer_log
{

  static const flatlogs::eventCodeT eventCode = eventCodes::TELEM_CORELOADS;
  static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_TELEM;

  static timespec lastRecord; ///< The time of the last time this log was recorded.  Used by the telemetry system.
  
   ///The type of the input message
   struct messageT : public fbMessage
   {
      ///Construct from components
      messageT( std::vector<float> & cpu_core_loads )
      {
         
         auto _coreLoadsVec = builder.CreateVector(cpu_core_loads);

         auto fp = CreateTelem_coreloads_fb(builder, _coreLoadsVec);
         
         builder.Finish(fp);

      }

   };

   static bool verify( flatlogs::bufferPtrT & logBuff,  ///< [in] Buffer containing the flatbuffer serialized message.
                       flatlogs::msgLenT len            ///< [in] length of msgBuffer.
                     )
   {
      auto verifier = flatbuffers::Verifier( static_cast<uint8_t*>(flatlogs::logHeader::messageBuffer(logBuff)), static_cast<size_t>(len));
      return VerifyTelem_coreloads_fbBuffer(verifier);
   }

   ///Get the message formatte for human consumption.
   static std::string msgString( void * msgBuffer,  /**< [in] Buffer containing the flatbuffer serialized message.*/
                                 flatlogs::msgLenT len  /**< [in] [unused] length of msgBuffer.*/
                               )
   {

      static_cast<void>(len); // unused by most log types
   
      auto rgs = GetTelem_coreloads_fb(msgBuffer);  
      
      std::string msg;

      if (rgs->loads() != nullptr) 
      {
         msg+= "[cpu loads] ";
         for(flatbuffers::Vector<float>::const_iterator it = rgs->loads()->begin(); it != rgs->loads()->end(); ++it)
         {
            msg+= std::to_string(*it);
            msg+= " ";
         }
      }
      

      return msg;

   }

}; //telem_coreloads



} //namespace logger
} //namespace MagAOX

#endif //logger_types_telem_coreloads_hpp
