/** \file telem_teldata.hpp
  * \brief The MagAO-X logger telem_teldata log type.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_types_files
  * 
  * History:
  * - 2018-09-06 created by JRM
  */
#ifndef logger_types_telem_teldata_hpp
#define logger_types_telem_teldata_hpp

#include "generated/telem_teldata_generated.h"
#include "flatbuffer_log.hpp"
#include "../logMeta.hpp"

namespace MagAOX
{
namespace logger
{


/// Log entry recording the build-time git state.
/** \ingroup logger_types
  */
struct telem_teldata : public flatbuffer_log
{
   ///The event code
   static const flatlogs::eventCodeT eventCode = eventCodes::TELEM_TELDATA;

   ///The default level
   static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_TELEM;

   static timespec lastRecord; ///< The time of the last time this log was recorded.  Used by the telemetry system.

   ///The type of the input message
   struct messageT : public fbMessage
   {
      ///Construct from components
      messageT( const int & roi,          ///<[in] rotator of interest
                const int & tracking,     ///<[in] tracking state
                const int & guiding,      ///<[in] guiding state
                const int & slewing,      ///<[in] slewing state
                const int & guiderMoving, ///<[in] guider moving state
                const double & az,        ///<[in] azimuth
                const double & zd,        ///<[in] zenith distance
                const double & pa,        ///<[in] parallactic angle
                const double & domeAz,    ///<[in] dome azimuth
                const int & domeStat      ///<[in] dome status
              )
      {
         auto fp = CreateTelem_teldata_fb(builder, roi, tracking, guiding, slewing, guiderMoving, az, zd, pa, domeAz, domeStat);
         builder.Finish(fp);
      }

   };

   static bool verify( flatlogs::bufferPtrT & logBuff,  ///< [in] Buffer containing the flatbuffer serialized message.
                       flatlogs::msgLenT len            ///< [in] length of msgBuffer.
                     )
   {
      auto verifier = flatbuffers::Verifier( (uint8_t*) flatlogs::logHeader::messageBuffer(logBuff), static_cast<size_t>(len));
      return VerifyTelem_teldata_fbBuffer(verifier);
   }
 
   ///Get the message formatte for human consumption.
   static std::string msgString( void * msgBuffer,  /**< [in] Buffer containing the flatbuffer serialized message.*/
                                 flatlogs::msgLenT len  /**< [in] [unused] length of msgBuffer.*/
                               )
   {
      static_cast<void>(len);

      auto fbs = GetTelem_teldata_fb(msgBuffer);

      std::string msg = "[teldata] ";
      
      msg += "roi: ";
      msg += std::to_string(fbs->roi()) + " ";
      
      msg += "tr: ";
      msg += std::to_string(fbs->tracking()) + " ";
      
      msg += "gd: ";
      msg += std::to_string(fbs->guiding()) + " ";
      
      msg += "sl: ";
      msg += std::to_string(fbs->slewing()) + " ";
      
      msg += "gm: ";
      msg += std::to_string(fbs->guiderMoving()) + " ";
      
      msg += "az: ";
      msg += std::to_string(fbs->az()) + " ";
      
      msg += "zd: ";
      msg += std::to_string(fbs->zd()) + " ";
      
      msg += "pa: ";
      msg += std::to_string(fbs->pa()) + " ";
      
      msg += "da: ";
      msg += std::to_string(fbs->domeAz()) + " ";
      
      msg += "ds: ";
      msg += std::to_string(fbs->domeStat());
      
      return msg;
   
   }
   
   static int roi( void * msgBuffer )
   {
       auto fbs = GetTelem_teldata_fb(msgBuffer);
       return fbs->roi();
   }
   
   static int tracking( void * msgBuffer )
   {
       auto fbs = GetTelem_teldata_fb(msgBuffer);
       return fbs->tracking();
   }
   
   static int guiding( void * msgBuffer )
   {
       auto fbs = GetTelem_teldata_fb(msgBuffer);
       return fbs->guiding();
   }
   
   static int slewing( void * msgBuffer )
   {
       auto fbs = GetTelem_teldata_fb(msgBuffer);
       return fbs->slewing();
   }      
   
   static int guiderMoving( void * msgBuffer )
   {
       auto fbs = GetTelem_teldata_fb(msgBuffer);
       return fbs->guiderMoving();
   }
   
   static double az( void * msgBuffer )
   {
       auto fbs = GetTelem_teldata_fb(msgBuffer);
       return fbs->az();
   }
   
   static double zd( void * msgBuffer )
   {
       auto fbs = GetTelem_teldata_fb(msgBuffer);
       return fbs->zd();
   }
   
   static double pa( void * msgBuffer )
   {
       auto fbs = GetTelem_teldata_fb(msgBuffer);
       return fbs->pa();
   }
   
   static double domeAz( void * msgBuffer )
   {
       auto fbs = GetTelem_teldata_fb(msgBuffer);
       return fbs->domeAz();
   }
   
   static int domeStat( void * msgBuffer )
   {
       auto fbs = GetTelem_teldata_fb(msgBuffer);
       return fbs->domeStat();
   }
   
   /// Get the logMetaDetail for a member by name
   /**
     * \returns the function pointer cast to void*
     * \returns -1 for an unknown member
     */ 
   static logMetaDetail getAccessor( const std::string & member /**< [in] the name of the member */ )
   {
      if(     member == "roi") return logMetaDetail({"ROI", logMeta::valTypes::Int, logMeta::metaTypes::State, (void *) &roi});
      else if(member == "tracking") return logMetaDetail({"TRACKING", logMeta::valTypes::Int, logMeta::metaTypes::State, (void *) &tracking}); 
      else if(member == "guiding") return logMetaDetail({"GUIDING", logMeta::valTypes::Int, logMeta::metaTypes::State, (void *) &guiding}); 
      else if(member == "slewing") return logMetaDetail({"SLEWING", logMeta::valTypes::Int, logMeta::metaTypes::State, (void *) &slewing}); 
      else if(member == "guiderMoving") return logMetaDetail({"GUIDER MOVING", logMeta::valTypes::Int, logMeta::metaTypes::State, (void *) &guiderMoving}); 
      else if(member == "az") return logMetaDetail({"AZ", logMeta::valTypes::Double, logMeta::metaTypes::Continuous, (void *) &az, false});
      else if(member == "zd") return logMetaDetail({"ZD", logMeta::valTypes::Double, logMeta::metaTypes::Continuous, (void *) &zd, false});
      else if(member == "pa") return logMetaDetail({"PARANG", logMeta::valTypes::Double, logMeta::metaTypes::Continuous, (void *) &pa, false});
      else if(member == "domeAz") return logMetaDetail({"DOME AZ", logMeta::valTypes::Double, logMeta::metaTypes::Continuous, (void *) &domeAz}); 
      else if(member == "domeStat") return logMetaDetail({"DOME STATUS", logMeta::valTypes::Int, logMeta::metaTypes::State, (void *) &domeStat});
      else
      {
         std::cerr << "No string member " << member << " in telem_teldata\n";
         return logMetaDetail();
      }
   }
   
}; //telem_teldata



} //namespace logger
} //namespace MagAOX

#endif //logger_types_telem_teldata_hpp

