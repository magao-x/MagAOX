/** \file telem_observer.hpp
  * \brief The MagAO-X logger telem_observer log type.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_types_files
  * 
  * History:
  * - 2018-09-06 created by JRM
  */
#ifndef logger_types_telem_observer_hpp
#define logger_types_telem_observer_hpp

#include "generated/telem_observer_generated.h"
#include "flatbuffer_log.hpp"

#include <cmath>

namespace MagAOX
{
namespace logger
{


/// Log entry recording the build-time git state.
/** \ingroup logger_types
  */
struct telem_observer : public flatbuffer_log
{
   ///The event code
   static const flatlogs::eventCodeT eventCode = eventCodes::TELEM_OBSERVER;

   ///The default level
   static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_TELEM;

   static timespec lastRecord; ///< The time of the last time this log was recorded.  Used by the telemetry system.

   
   ///The type of the input message
   struct messageT : public fbMessage
   {
      ///Construct from components
      messageT( const std::string & email,   /// <[in] observer email
                const std::string & obsName, /// <[in] observer email
                const bool & observing       /// <[in] status of observing
              )
      {
         auto _email = builder.CreateString(email);
         auto _obsName = builder.CreateString(obsName);
         
         auto fp = CreateTelem_observer_fb(builder, _email, _obsName, observing);
         builder.Finish(fp);
      }

   };

   static bool verify( flatlogs::bufferPtrT & logBuff,  ///< [in] Buffer containing the flatbuffer serialized message.
                       flatlogs::msgLenT len            ///< [in] length of msgBuffer.
                     )
   {
      auto verifier = flatbuffers::Verifier( (uint8_t*) flatlogs::logHeader::messageBuffer(logBuff), static_cast<size_t>(len));

      bool ok = VerifyTelem_observer_fbBuffer(verifier); 
      if(!ok) return ok;

      auto fbs = GetTelem_observer_fb((uint8_t*) flatlogs::logHeader::messageBuffer(logBuff));

      if(fbs->email())
      {
         std::string email = fbs->email()->c_str();
         for(size_t n = 0; n < email.size(); ++n)
         {
            if(!isprint(email[n]))
            {
               return false;
            }
         }
      }

      if(fbs->obsName())
      {
         std::string obsn = fbs->obsName()->c_str();
         for(size_t n = 0; n < obsn.size(); ++n)
         {
            if(!isprint(obsn[n]))
            {
               return false;
            }
         }
      }

      return ok;
   }

   ///Get the message formattd for human consumption.
   static std::string msgString( void * msgBuffer,      ///< [in] Buffer containing the flatbuffer serialized message.
                                 flatlogs::msgLenT len  ///< [in] [unused] length of msgBuffer.
                               )
   {
      static_cast<void>(len);

      auto fbs = GetTelem_observer_fb(msgBuffer);

      std::string msg = "[observer] ";
      
      if(fbs->email())
      {
         msg += "email: ";
         msg += fbs->email()->c_str();
         msg += " ";
      }
      
      if(fbs->obsName())
      {
         msg += "obs: ";
         msg += fbs->obsName()->c_str();
         msg += " ";
      }
      
      msg += std::to_string(fbs->observing());
      
      return msg;
   
   }
   
   static std::string email( void * msgBuffer )
   {
      auto fbs = GetTelem_observer_fb(msgBuffer);
      if(fbs->email() != nullptr)
      {
         return std::string(fbs->email()->c_str());
      }
      else return "";
   }

   static std::string obsName( void * msgBuffer )
   {
      auto fbs = GetTelem_observer_fb(msgBuffer);
      if(fbs->email() != nullptr)
      {
         return std::string(fbs->obsName()->c_str());
      }
      else return "";
   }

   static bool observing( void * msgBuffer )
   {
      auto fbs = GetTelem_observer_fb(msgBuffer);
      return fbs->observing();
   }

   /// Get the logMetaDetail for a member by name
   /**
     * \returns the a logMetaDetail filled in with the appropriate details
     * \returns an empty logmegaDetail if member not recognized
     */ 
   static logMetaDetail getAccessor( const std::string & member /**< [in] the name of the member */ )
   {
      if(     member == "email")     return logMetaDetail({"OBSERVER", logMeta::valTypes::String, logMeta::metaTypes::State, (void *) &email, false});
      else if(member == "obsName")   return logMetaDetail({"OBS-NAME", logMeta::valTypes::String, logMeta::metaTypes::State, (void *) &obsName, false});
      else if(member == "observing") return logMetaDetail({"OBSERVING", logMeta::valTypes::Bool, logMeta::metaTypes::State, (void *) &observing}); 
      else
      {
         std::cerr << "No string member " << member << " in telem_observer\n";
         return logMetaDetail();
      }
   }

   
   
}; //telem_observer



} //namespace logger
} //namespace MagAOX

#endif //logger_types_telem_observer_hpp

