/** \file telem_stdcam.hpp
  * \brief The MagAO-X logger telem_stdcam log type.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_types_files
  * 
  * History:
  * - 2018-09-06 created by JRM
  */
#ifndef logger_types_telem_stdcam_hpp
#define logger_types_telem_stdcam_hpp

#include "generated/telem_stdcam_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{


/// Log entry recording stdcam stage specific status.
/** \ingroup logger_types
  */
struct telem_stdcam : public flatbuffer_log
{
   ///The event code
   static const flatlogs::eventCodeT eventCode = eventCodes::TELEM_STDCAM;

   ///The default level
   static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_TELEM;

   static timespec lastRecord; ///< The timestamp of the last time this log was recorded.  Used by the telemetry system.

   ///The type of the input message
   struct messageT : public fbMessage
   {
      ///Construct from components
      messageT( const std::string & mode,     ///<[in]
                const float & xcen,           ///<[in]
                const float & ycen,           ///<[in] 
                const int & width,            ///<[in]
                const int & height,           ///<[in]
                const int & xbin,             ///<[in]
                const int & ybin,             ///<[in]
                const float & exptime,        ///<[in]
                const float & fps,            ///<[in]
                const float & emGain,
                const float & adcSpeed,
                const float & temp,           ///<[in]
                const float & setpt,          ///<[in]
                const uint8_t & status,       ///<[in]
                const uint8_t & ontarget,     ///<[in]
                const std::string & statusStr ///<[in]
              )
      {         
         auto _mode = builder.CreateString(mode);
         auto _roi = CreateROI(builder,xcen, ycen, width, height, xbin, ybin);
         auto _statusStr = builder.CreateString(statusStr);
         auto _tempCtrl = CreateTempCtrl(builder, temp, setpt, status, ontarget, _statusStr);
         
         
         
         auto fp = CreateTelem_stdcam_fb(builder, _mode, _roi, exptime, fps, emGain, adcSpeed, _tempCtrl);
         builder.Finish(fp);
      }

   };
                 
 
   ///Get the message formatte for human consumption.
   static std::string msgString( void * msgBuffer,  /**< [in] Buffer containing the flatbuffer serialized message.*/
                                 flatlogs::msgLenT len  /**< [in] [unused] length of msgBuffer.*/
                               )
   {
      static_cast<void>(len);

      auto fbs = GetTelem_stdcam_fb(msgBuffer);

      std::string msg = "[stdcam] ";
      
      if(fbs->mode() != nullptr)
      {
         msg+= "mode: ";
         msg += fbs->mode()->c_str();
         msg += " ";
      }
      
      if(fbs->roi() != nullptr)
      {
         msg += "ROI-x: ";
         msg += std::to_string(fbs->roi()->xcen());
         msg += " y: ";
         msg += std::to_string(fbs->roi()->ycen());
         msg += " w: ";
         msg += std::to_string(fbs->roi()->w());
         msg += " xbin: ";
         msg += std::to_string(fbs->roi()->xbin());
         msg += " ybin: ";
         msg += std::to_string(fbs->roi()->ybin());
         msg += " ";
      }
      
      msg += "expt: ";
      msg += std::to_string(fbs->exptime());
      msg += " fps: ";
      msg += std::to_string(fbs->fps());
      msg += " emG: ";
      msg += std::to_string(fbs->emGain());
      msg += " adc: ";
      msg += std::to_string(fbs->adcSpeed());
      
      if(fbs->tempCtrl() != nullptr)
      { 
         msg += " temp: ";
         msg += std::to_string(fbs->tempCtrl()->temp());
         msg += " setpt: ";
         msg += std::to_string(fbs->tempCtrl()->setpt());
         msg += " tempctr-stat: ";
         msg += std::to_string(fbs->tempCtrl()->status());
         msg += " tempctr-ontgt: ";
         msg += std::to_string(fbs->tempCtrl()->ontarget());
         
         if(fbs->tempCtrl()->statusStr())
         {
            msg += " tempctr-statstr: ";
            msg += fbs->tempCtrl()->statusStr()->c_str();
         }
      }
      
      return msg;
   
   }
   
}; //telem_stdcam

timespec telem_stdcam::lastRecord = {0,0};

} //namespace logger
} //namespace MagAOX

#endif //logger_types_telem_stdcam_hpp

