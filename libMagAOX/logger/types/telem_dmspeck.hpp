/** \file telem_dmspeck.hpp
  * \brief The MagAO-X logger telem_dmspeck log type.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_types_files
  * 
  * History:
  * - 2022-02-06 created by JRM
  */
#ifndef logger_types_telem_dmspeck_hpp
#define logger_types_telem_dmspeck_hpp

#include "generated/telem_dmspeck_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{


/// Log entry recording stdMotionStage status.
/** \ingroup logger_types
  */
struct telem_dmspeck : public flatbuffer_log
{
   ///The event code
   static const flatlogs::eventCodeT eventCode = eventCodes::TELEM_DMSPECK;

   ///The default level
   static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_TELEM;

   static timespec lastRecord; ///< The timestamp of the last time this log was recorded.  Used by the telemetry system.

   ///The type of the input message
   struct messageT : public fbMessage
   {
      ///Construct from components
      messageT( const bool & modulating,                ///< [in] whether or not the speckle is being modulated
                const bool & trigger,                   ///< [in] whether or not the speckle is being triggered
                const float & frequency,                ///< [in] frequency of modulation is not triggered
                const std::vector<float> & separations, ///< [in] the separations of the speckle(s)
                const std::vector<float> & angles,      ///< [in] the angles of the speckle(s)
                const std::vector<float> & amplitudes,  ///< [in] the amplitudes of the speckle(s)
                const std::vector<bool> & crosses       ///< [in] whether or not the cross speckle(s) are produced
              )
      {  
         auto _separationsVec = builder.CreateVector(separations);
         auto _anglesVec = builder.CreateVector(angles);
         auto _amplitudesVec = builder.CreateVector(amplitudes);
         auto _crossesVec = builder.CreateVector(crosses);

         auto fp = CreateTelem_dmspeck_fb(builder, modulating, trigger, frequency, _separationsVec, _anglesVec, _amplitudesVec, _crossesVec);
         builder.Finish(fp);
      }

   };
                 
   static bool verify( flatlogs::bufferPtrT & logBuff,  ///< [in] Buffer containing the flatbuffer serialized message.
                       flatlogs::msgLenT len            ///< [in] length of msgBuffer.
                     )
   {
      auto verifier = flatbuffers::Verifier( (uint8_t*) flatlogs::logHeader::messageBuffer(logBuff), static_cast<size_t>(len));
      return VerifyTelem_dmspeck_fbBuffer(verifier);
   }

   ///Get the message formatte for human consumption.
   static std::string msgString( void * msgBuffer,  /**< [in] Buffer containing the flatbuffer serialized message.*/
                                 flatlogs::msgLenT len  /**< [in] [unused] length of msgBuffer.*/
                               )
   {
      static_cast<void>(len);

      auto fbs = GetTelem_dmspeck_fb(msgBuffer);

      std::string msg = "[speckles] ";
      
      if(!fbs->modulating())
      {
         msg += "not modulating";
         return msg;
      }
      
      msg += "modulating";
      if(fbs->trigger())
      { 
         msg += " by trigger ";
      }
      else
      {
         msg += " at ";
         msg += std::to_string(fbs->frequency());
         msg += " Hz ";
      }
      
      msg += "seps: ";
      for(flatbuffers::Vector<float>::const_iterator it = fbs->separations()->begin(); it != fbs->separations()->end(); ++it)
      {
         msg+= std::to_string(*it);
         msg+= " ";
      }      
      msg += "angs: ";
      for(flatbuffers::Vector<float>::const_iterator it = fbs->angles()->begin(); it != fbs->angles()->end(); ++it)
      {
         msg+= std::to_string(*it);
         msg+= " ";
      }
      msg += "amps: ";
      for(flatbuffers::Vector<float>::const_iterator it = fbs->amplitudes()->begin(); it != fbs->amplitudes()->end(); ++it)
      {
         msg+= std::to_string(*it);
         msg+= " ";
      }
      for(flatbuffers::Vector<unsigned char>::const_iterator it = fbs->crosses()->begin(); it != fbs->crosses()->end(); ++it)
      {
         if(*it) msg += "+";
         else msg += "-";
      }

      
      return msg;
   
   }

   static bool modulating( void * msgBuffer )
   {
      auto fbs = GetTelem_dmspeck_fb(msgBuffer);
      return fbs->modulating();
   }

   static bool trigger( void * msgBuffer )
   {
      auto fbs = GetTelem_dmspeck_fb(msgBuffer);
      return fbs->trigger();
   }
   
   static float frequency( void * msgBuffer )
   {
      auto fbs = GetTelem_dmspeck_fb(msgBuffer);
      return fbs->frequency();
   }

   static std::vector<float> separations( void * msgBuffer )
   {
      std::vector<float> v;

      auto fbs = GetTelem_dmspeck_fb(msgBuffer);

      for(flatbuffers::Vector<float>::const_iterator it = fbs->separations()->begin(); it != fbs->separations()->end(); ++it)
      {
         v.push_back(*it);
      }      

      return v;
   }
   
   static std::vector<float> angles( void * msgBuffer )
   {
      std::vector<float> v;

      auto fbs = GetTelem_dmspeck_fb(msgBuffer);

      for(flatbuffers::Vector<float>::const_iterator it = fbs->angles()->begin(); it != fbs->angles()->end(); ++it)
      {
         v.push_back(*it);
      }      

      return v;
   }
   
   static std::vector<float> amplitudes( void * msgBuffer )
   {
      std::vector<float> v;

      auto fbs = GetTelem_dmspeck_fb(msgBuffer);

      for(flatbuffers::Vector<float>::const_iterator it = fbs->amplitudes()->begin(); it != fbs->amplitudes()->end(); ++it)
      {
         v.push_back(*it);
      }      

      return v;
   }

   static std::vector<bool> crosses( void * msgBuffer )
   {
      std::vector<bool> v;

      auto fbs = GetTelem_dmspeck_fb(msgBuffer);

      for(flatbuffers::Vector<unsigned char>::const_iterator it = fbs->crosses()->begin(); it != fbs->crosses()->end(); ++it)
      {
         v.push_back(*it);
      }      

      return v;
   }

   /// Get pointer to the accessor for a member by name 
   /**
     * \returns the function pointer cast to void*
     * \returns -1 for an unknown member
     */ 
   static logMetaDetail getAccessor( const std::string & member /**< [in] the name of the member */ )
   {
      if(member == "modulating") return logMetaDetail({"MODULATING", logMeta::valTypes::Bool, logMeta::metaTypes::State, (void *) &modulating});
      else if(member == "trigger") return logMetaDetail({"TRIGGERED", logMeta::valTypes::Bool, logMeta::metaTypes::State, (void *) &trigger});
      else if(member == "frequency") return logMetaDetail({"FREQUENCY", logMeta::valTypes::Float, logMeta::metaTypes::State, (void *) &frequency});
      else if(member == "separations") return logMetaDetail({"SEPARATIONS", logMeta::valTypes::Vector_Float, logMeta::metaTypes::State, (void *) &separations});
      else if(member == "angles") return logMetaDetail({"ANGLES", logMeta::valTypes::Vector_Float, logMeta::metaTypes::State, (void *) &angles});
      else if(member == "amplitudes") return logMetaDetail({"AMPLITUDES", logMeta::valTypes::Vector_Float, logMeta::metaTypes::State, (void *) &amplitudes});
      else if(member == "crosses") return logMetaDetail({"CROSSES", logMeta::valTypes::Vector_Bool, logMeta::metaTypes::State, (void *) &crosses});
      else
      {
         std::cerr << "No string member " << member << " in telem_dmspeck\n";
         return logMetaDetail();
      }
   }
   
}; //telem_dmspeck



} //namespace logger
} //namespace MagAOX

#endif //logger_types_telem_dmspeck_hpp

