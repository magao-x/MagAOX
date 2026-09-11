/** \file telem_sparkleclock.hpp
  * \brief The MagAO-X logger telem_sparkleclock log type.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_types_files
  *
  * History:
  * - 2022-02-06 created by JRM
  */
#ifndef logger_types_telem_sparkleclock_hpp
#define logger_types_telem_sparkleclock_hpp

#include "generated/telem_sparkleclock_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{


/// Log entry recording sparkle clock status
/** \ingroup logger_types
  */
struct telem_sparkleclock : public flatbuffer_log
{
   ///The event code
   // This was TELEM_DMSPECK before. That copy and paste error tagged every sparkle clock log entry
   // with the dmspeck event code. Logs recorded before this fix still carry the old code.
   static const flatlogs::eventCodeT eventCode = eventCodes::TELEM_SPARKLECLOCK;

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
                const float & interval,                 ///< [in] time for one complete cycle of the sparkle clock (e.g. exposure time of some camera)
                const std::vector<float> & separations, ///< [in] the separations of the speckle(s)
                const float & angleOffset,              ///< [in] starting angle offset of the sparkle clock
                const float & amplitude                 ///< [in] starting angle offset of the sparkle clock
              )
      {
         auto _separationsVec = builder.CreateVector(separations);
         // auto _anglesVec = builder.CreateVector(angles);
         // auto _amplitudesVec = builder.CreateVector(amplitudes);
         // auto _crossesVec = builder.CreateVector(crosses);

         auto fp = CreateTelem_sparkleclock_fb(builder, modulating, trigger, frequency, interval, _separationsVec, angleOffset, amplitude);
         builder.Finish(fp);
      }

   };

   static bool verify( flatlogs::bufferPtrT & logBuff,  ///< [in] Buffer containing the flatbuffer serialized message.
                       flatlogs::msgLenT len            ///< [in] length of msgBuffer.
                     )
   {
      auto verifier = flatbuffers::Verifier( static_cast<uint8_t*>(flatlogs::logHeader::messageBuffer(logBuff)), static_cast<size_t>(len));
      return VerifyTelem_sparkleclock_fbBuffer(verifier);
   }

   ///Get the message formatte for human consumption.
   static std::string msgString( void * msgBuffer,  /**< [in] Buffer containing the flatbuffer serialized message.*/
                                 flatlogs::msgLenT len  /**< [in] [unused] length of msgBuffer.*/
                               )
   {
      static_cast<void>(len);

      auto fbs = GetTelem_sparkleclock_fb(msgBuffer);

      std::string msg = "[sparkleclock] ";

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
      // separations is not a required field, so a message can be serialized without it.
      // This null check prevents a crash from iterating a separations vector that is missing.
      if(fbs->separations() != nullptr)
      {
         for(flatbuffers::Vector<float>::const_iterator it = fbs->separations()->begin(); it != fbs->separations()->end(); ++it)
         {
            msg+= std::to_string(*it);
            msg+= " ";
         }
      }
      msg += "angle offset: ";
      msg += std::to_string(fbs->angleOffset());
      msg += "amp: ";
      msg += std::to_string(fbs->amplitude());

      return msg;

   }

   static bool modulating( void * msgBuffer )
   {
      auto fbs = GetTelem_sparkleclock_fb(msgBuffer);
      return fbs->modulating();
   }

   static bool trigger( void * msgBuffer )
   {
      auto fbs = GetTelem_sparkleclock_fb(msgBuffer);
      return fbs->trigger();
   }

   static float frequency( void * msgBuffer )
   {
      auto fbs = GetTelem_sparkleclock_fb(msgBuffer);
      return fbs->frequency();
   }

   static std::vector<float> separations( void * msgBuffer )
   {
      std::vector<float> v;

      auto fbs = GetTelem_sparkleclock_fb(msgBuffer);

      // Bug fix. separations is not a required field, so a message can be serialized
      // without it. The same null check as in formatMessage() above prevents a crash.
      // An empty vector is returned when the field is missing.
      if(fbs->separations() != nullptr)
      {
         for(flatbuffers::Vector<float>::const_iterator it = fbs->separations()->begin(); it != fbs->separations()->end(); ++it)
         {
            v.push_back(*it);
         }
      }

      return v;
   }

   static float angleOffset( void * msgBuffer )
   {
      auto fbs = GetTelem_sparkleclock_fb(msgBuffer);
      return fbs->angleOffset();
   }
   static float amplitude( void * msgBuffer )
   {
      auto fbs = GetTelem_sparkleclock_fb(msgBuffer);
      return fbs->amplitude();
   }

   /// Get the logMetaDetail for a member by name
   /**
     * \returns the a logMetaDetail filled in with the appropriate details
     * \returns an empty logMetaDetail if member not recognized
     */
   static logMetaDetail getAccessor( const std::string & member /**< [in] the name of the member */ )
   {
      if(member == "modulating") return logMetaDetail({"MODULATING", logMeta::valTypes::Bool, logMeta::metaTypes::State, reinterpret_cast<void*>(&modulating)});
      else if(member == "trigger") return logMetaDetail({"TRIGGERED", logMeta::valTypes::Bool, logMeta::metaTypes::State, reinterpret_cast<void*>(&trigger)});
      else if(member == "frequency") return logMetaDetail({"FREQUENCY", logMeta::valTypes::Float, logMeta::metaTypes::State, reinterpret_cast<void*>(&frequency)});
      else if(member == "separations") return logMetaDetail({"SEPARATIONS", logMeta::valTypes::Vector_Float, logMeta::metaTypes::State, reinterpret_cast<void*>(&separations)});
      else if(member == "angleOffset") return logMetaDetail({"ANGLEOFFSET", logMeta::valTypes::Float, logMeta::metaTypes::State, reinterpret_cast<void*>(&angleOffset)});
      else if(member == "amplitude") return logMetaDetail({"AMPLITUDES", logMeta::valTypes::Float, logMeta::metaTypes::State, reinterpret_cast<void*>(&amplitude)});
      else
      {
         std::cerr << "No member " << member << " in telem_sparkleclock\n";
         return logMetaDetail();
      }
   }

}; //telem_sparkleclock



} //namespace logger
} //namespace MagAOX

#endif //logger_types_telem_sparkleclock_hpp

