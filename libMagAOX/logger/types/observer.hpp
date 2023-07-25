/** \file observer.hpp
  * \brief The MagAO-X logger observer log type.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_types_files
  * 
  * History:
  * - 2018-09-06 created by JRM
  */
#ifndef logger_types_observer_hpp
#define logger_types_observer_hpp

#include "generated/observer_generated.h"
#include "flatbuffer_log.hpp"

#include <cmath>

namespace MagAOX
{
namespace logger
{


/// Log entry recording the observer.
/** \ingroup logger_types
  */
struct observer : public flatbuffer_log
{
   ///The event code
   static const flatlogs::eventCodeT eventCode = eventCodes::OBSERVER;

   ///The default level
   static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_INFO;

   static timespec lastRecord; ///< The time of the last time this log was recorded.  Used by the telemetry system.

   ///The type of the input message
   struct messageT : public fbMessage
   {
      ///Construct from components
      messageT( const std::string & fullName,   ///< [in] the observer's full name
                const std::string & pfoa,       ///< [in] the observer's preferred form of address
                const std::string & email,      ///< [in] the observer's email
                const std::string & institution ///< [in] the observer's institution
              )
      {
         auto _fullName = builder.CreateString(fullName);
         auto _pfoa = builder.CreateString(pfoa);
         auto _email = builder.CreateString(email);
         auto _institution = builder.CreateString(institution);
         
         auto fp = CreateObserver_fb(builder, _fullName, _pfoa, _email, _institution);
         builder.Finish(fp);
      }

   };
                 
   static bool verify( flatlogs::bufferPtrT & logBuff,  ///< [in] Buffer containing the flatbuffer serialized message.
                       flatlogs::msgLenT len            ///< [in] length of msgBuffer.
                     )
   {
      auto verifier = flatbuffers::Verifier( (uint8_t*) flatlogs::logHeader::messageBuffer(logBuff), static_cast<size_t>(len));
      return VerifyObserver_fbBuffer(verifier);
   }

   ///Get the message formatte for human consumption.
   static std::string msgString( void * msgBuffer,  /**< [in] Buffer containing the flatbuffer serialized message.*/
                                 flatlogs::msgLenT len  /**< [in] [unused] length of msgBuffer.*/
                               )
   {
      static_cast<void>(len);

      auto fbs = GetObserver_fb(msgBuffer);

      std::string msg = "Observer Loaded: ";
      
      if(fbs->fullName())
      {
         msg += fbs->fullName()->c_str();
         if(!fbs->pfoa()) msg += ", ";
         else msg += " ";
      }
      
      if(fbs->pfoa())
      {
         msg += "(";
         msg += fbs->pfoa()->c_str();
         msg += "), ";
      }
      
      if(fbs->email())
      {
         msg += fbs->email()->c_str();
         if(fbs->institution()) msg += ", ";
         else msg += " ";
      }
         
      if(fbs->institution())
      {
         msg += fbs->institution()->c_str();
      }
      
      return msg;
   
   }
   
}; //observer


} //namespace logger
} //namespace MagAOX

#endif //logger_types_observer_hpp

