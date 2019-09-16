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

   static timespec lastRecord; ///< The time of the last time this log wastelem_telpos             20000    telem_telpos recorded.  Used by the telemetry system.

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
   
}; //telem_teldata

timespec telem_teldata::lastRecord = {0,0};

} //namespace logger
} //namespace MagAOX

#endif //logger_types_telem_teldata_hpp

