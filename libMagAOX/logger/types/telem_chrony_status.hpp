/** \file telem_chrony_status.hpp
  * \brief The MagAO-X logger telem_chrony_status log type.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_types_files
  * 
  * History:
  * - 2018-09-06 created by JRM
  */
#ifndef logger_types_telem_chrony_status_hpp
#define logger_types_telem_chrony_status_hpp

#include "generated/telem_chrony_status_generated.h"
#include "flatbuffer_log.hpp"

#include <cmath>

namespace MagAOX
{
namespace logger
{


/// Log entry recording the status of chrony.
/** \ingroup logger_types
  */
struct telem_chrony_status : public flatbuffer_log
{
   ///The event code
   static const flatlogs::eventCodeT eventCode = eventCodes::TELEM_CHRONY_STATUS;

   ///The default level
   static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_TELEM;

   static timespec lastRecord; ///< The time of the last time this log was recorded.  Used by the telemetry system.

   ///The type of the input message
   struct messageT : public fbMessage
   {
      ///Construct from components
      messageT( const std::string & mac,  ///< [in] Source MAC
                const std::string & ip,   ///< [in] Source IP
                const std::string & sync, ///< [in] Synch status
                const std::string & leap  ///< [in] leap status
              )
      {
         auto _mac = builder.CreateString(mac);
         auto _ip = builder.CreateString(ip);
         auto _sync = builder.CreateString(sync);
         auto _leap = builder.CreateString(leap);
         
         auto fp = CreateTelem_chrony_status_fb(builder, _mac, _ip, _sync, _leap);
         builder.Finish(fp);
      }

   };
                 
   static bool verify( flatlogs::bufferPtrT & logBuff,  ///< [in] Buffer containing the flatbuffer serialized message.
                       flatlogs::msgLenT len            ///< [in] length of msgBuffer.
                     )
   {
      auto verifier = flatbuffers::Verifier( (uint8_t*) flatlogs::logHeader::messageBuffer(logBuff), static_cast<size_t>(len));
      return VerifyTelem_chrony_status_fbBuffer(verifier);
   }

   ///Get the message formatte for human consumption.
   static std::string msgString( void * msgBuffer,  /**< [in] Buffer containing the flatbuffer serialized message.*/
                                 flatlogs::msgLenT len  /**< [in] [unused] length of msgBuffer.*/
                               )
   {
      static_cast<void>(len);

      auto fbs = GetTelem_chrony_status_fb(msgBuffer);

      std::string msg = "[chrony_status] ";

      msg += "source: ";
      if(fbs->sourceMAC())
      {
         msg += fbs->sourceMAC()->c_str();
         msg +="/";
      }
      
      if(fbs->sourceIP())
      {
         msg += fbs->sourceIP()->c_str();
      }
      
      if(fbs->synch())
      {
         msg += " synch: ";
         msg += fbs->synch()->c_str();
      }
      
      if(fbs->leap())
      {
         msg += " leap: ";
         msg += fbs->leap()->c_str();
      }
      
      return msg;
   
   }
   
   static std::string sourceMAC(void * msgBuffer )
   {
      auto fbs = GetTelem_chrony_status_fb(msgBuffer);
      if(fbs->sourceMAC()) return std::string(fbs->sourceMAC()->c_str());
      else return "";
   }
   
   static std::string sourceIP(void * msgBuffer )
   {
      auto fbs = GetTelem_chrony_status_fb(msgBuffer);
      if(fbs->sourceIP()) return std::string(fbs->sourceIP()->c_str());
      else return "";
   }
   
   static std::string synch(void * msgBuffer )
   {
      auto fbs = GetTelem_chrony_status_fb(msgBuffer);
      if(fbs->synch()) return std::string(fbs->synch()->c_str());
      else return "";
   }

   static std::string leap(void * msgBuffer )
   {
      auto fbs = GetTelem_chrony_status_fb(msgBuffer);
      if(fbs->leap()) return std::string(fbs->leap()->c_str());
      else return "";
   }
   
   /// Get pointer to the accessor for a member by name 
   /**
     * \returns the function pointer cast to void*
     * \returns -1 for an unknown member
     */ 
   static void * getAccessor( const std::string & member /**< [in] the name of the member */ )
   {
      if(member == "sourceMAC") return (void *) &sourceMAC;
      else if(member == "sourceIP") return (void *) &sourceIP;
      else if(member == "synch") return (void *) &synch;
      else if(member == "leap") return (void *) &leap;
      else
      {
         std::cerr << "No string member " << member << " in telem_chrony_status\n";
         return 0;
      }
   }
   
}; //telem_chrony_status


} //namespace logger
} //namespace MagAOX

#endif //logger_types_telem_chrony_status_hpp

