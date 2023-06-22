/** \file telem_dmmodes.hpp
  * \brief The MagAO-X logger telem_dmmodes log type.
  * \author Jared Males (jrmales@arizona.edu)
  *
  * \ingroup logger_types_files
  *
  * History:
  * - 2022-10-09 created by JRM
  */
#ifndef logger_types_telem_dmmodes_hpp
#define logger_types_telem_dmmodes_hpp

#include "generated/telem_dmmodes_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{


/// Log entry recording DM Mode Amplitudes
/** \ingroup logger_types
  */
struct telem_dmmodes : public flatbuffer_log
{

  static const flatlogs::eventCodeT eventCode = eventCodes::TELEM_DMMODES;
  static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_TELEM;

  static timespec lastRecord; ///< The time of the last time this log was recorded.  Used by the telemetry system.
  
   ///The type of the input message
   struct messageT : public fbMessage
   {
      ///Construct from components
      explicit messageT( std::vector<float> & amps )
      {
         auto _ampsVec = builder.CreateVector(amps);
         
         auto fp = CreateTelem_dmmodes_fb(builder, _ampsVec );
         
         builder.Finish(fp);
      }

   };

   static bool verify( flatlogs::bufferPtrT & logBuff,  ///< [in] Buffer containing the flatbuffer serialized message.
                       flatlogs::msgLenT len            ///< [in] length of msgBuffer.
                     )
   {
      auto verifier = flatbuffers::Verifier( (uint8_t*) flatlogs::logHeader::messageBuffer(logBuff), static_cast<size_t>(len));
      return VerifyTelem_dmmodes_fbBuffer(verifier);
   }

   ///Get the message formatte for human consumption.
   static std::string msgString( void * msgBuffer,  /**< [in] Buffer containing the flatbuffer serialized message.*/
                                 flatlogs::msgLenT len  /**< [in] [unused] length of msgBuffer.*/
                               )
   {

      static_cast<void>(len); // unused by most log types
   
      auto rgs = GetTelem_dmmodes_fb(msgBuffer);  
      
      std::string msg;

      msg+= "[dmmodes amps] ";

      if (rgs->amps() != nullptr) 
      {
         for(flatbuffers::Vector<float>::const_iterator it = rgs->amps()->begin(); it != rgs->amps()->end(); ++it)
         {
            msg+= std::to_string(*it);
            msg+= " ";
         }
      }

      return msg;

   }

   static std::vector<float> amps( void * msgBuffer )
   {
      std::vector<float> amps;
      auto fbs = GetTelem_dmmodes_fb(msgBuffer);

      if (fbs->amps() != nullptr) 
      {
         for(flatbuffers::Vector<float>::const_iterator it = fbs->amps()->begin(); it != fbs->amps()->end(); ++it)
         {
            amps.push_back(*it);
         }
      }

      return amps;
   }

   /// Get the logMetaDetail for a member by name
   /**
     * \returns the a logMetaDetail filled in with the appropriate details
     * \returns an empty logmegaDetail if member not recognized
     */ 
   static logMetaDetail getAccessor( const std::string & member /**< [in] the name of the member */ )
   {
      if(member == "amps") return logMetaDetail({"AMPS", logMeta::valTypes::Vector_Float, logMeta::metaTypes::Continuous, (void *) &amps, false});
      else
      {
         std::cerr << "No string member " << member << " in telem_dmmodes\n";
         return logMetaDetail();
      }
   }

}; //telem_dmmodes



} //namespace logger
} //namespace MagAOX

#endif //logger_types_telem_dmmodes_hpp
