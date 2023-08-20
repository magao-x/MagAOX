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
#include "../logMeta.hpp"

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
      messageT( const std::string & mode,            ///<[in]
                const float & xcen,                  ///<[in]
                const float & ycen,                  ///<[in] 
                const int & width,                   ///<[in]
                const int & height,                  ///<[in]
                const int & xbin,                    ///<[in]
                const int & ybin,                    ///<[in]
                const float & exptime,               ///<[in]
                const float & fps,                   ///<[in]
                const float & emGain,                ///<[in]
                const float & adcSpeed,              ///<[in]
                const float & temp,                  ///<[in]
                const float & setpt,                 ///<[in]
                const uint8_t & status,              ///<[in]
                const uint8_t & ontarget,            ///<[in]
                const std::string & statusStr,       ///<[in]
                const std::string & shutterStatusSr, ///<[in]
                const int8_t & shutterState,         ///<[in]
                const uint8_t & synchro,             ///<[in]
                const float & vshift,                ///<[in]
                const uint8_t & cropMode             ///<[in]
              )
      {         
         auto _mode = builder.CreateString(mode);
         auto _roi = CreateROI(builder,xcen, ycen, width, height, xbin, ybin);
         
         auto _statusStr = builder.CreateString(statusStr);
         auto _tempCtrl = CreateTempCtrl(builder, temp, setpt, status, ontarget, _statusStr);
         
         auto _shutterStatusStr = builder.CreateString(shutterStatusSr);
         auto _shutter = CreateShutter(builder, _shutterStatusStr, shutterState);
                  
         auto fp = CreateTelem_stdcam_fb(builder, _mode, _roi, exptime, fps, emGain, adcSpeed, _tempCtrl, _shutter, synchro, vshift, cropMode);
         builder.Finish(fp);
      }

      ///Construct from components, without vShift and cropMode for backwards compat.
      messageT( const std::string & mode,            ///<[in]
                const float & xcen,                  ///<[in]
                const float & ycen,                  ///<[in] 
                const int & width,                   ///<[in]
                const int & height,                  ///<[in]
                const int & xbin,                    ///<[in]
                const int & ybin,                    ///<[in]
                const float & exptime,               ///<[in]
                const float & fps,                   ///<[in]
                const float & emGain,                ///<[in]
                const float & adcSpeed,              ///<[in]
                const float & temp,                  ///<[in]
                const float & setpt,                 ///<[in]
                const uint8_t & status,              ///<[in]
                const uint8_t & ontarget,            ///<[in]
                const std::string & statusStr,       ///<[in]
                const std::string & shutterStatusSr, ///<[in]
                const int8_t & shutterState,         ///<[in]
                const uint8_t & synchro              ///<[in]
              )
      {         
         auto _mode = builder.CreateString(mode);
         auto _roi = CreateROI(builder,xcen, ycen, width, height, xbin, ybin);
         
         auto _statusStr = builder.CreateString(statusStr);
         auto _tempCtrl = CreateTempCtrl(builder, temp, setpt, status, ontarget, _statusStr);
         
         auto _shutterStatusStr = builder.CreateString(shutterStatusSr);
         auto _shutter = CreateShutter(builder, _shutterStatusStr, shutterState);
                  
         auto fp = CreateTelem_stdcam_fb(builder, _mode, _roi, exptime, fps, emGain, adcSpeed, _tempCtrl, _shutter, synchro, 0, -1);
         builder.Finish(fp);
      }

   };
                 
   static bool verify( flatlogs::bufferPtrT & logBuff,  ///< [in] Buffer containing the flatbuffer serialized message.
                       flatlogs::msgLenT len            ///< [in] length of msgBuffer.
                     )
   {
      auto verifier = flatbuffers::Verifier( (uint8_t*) flatlogs::logHeader::messageBuffer(logBuff), static_cast<size_t>(len));
      return VerifyTelem_stdcam_fbBuffer(verifier);
   }

   ///Get the message formatted for human consumption.
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
         msg += " h: ";
         msg += std::to_string(fbs->roi()->h());
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
         
      if(fbs->shutter() != nullptr)
      {
         if(fbs->shutter()->statusStr())
         {
            msg += " shutter-statstr: ";
            msg += fbs->shutter()->statusStr()->c_str();
         }
         msg+= " shutter: ";
         if( fbs->shutter()->state() == -1)
         {
            msg += "UNKN";
         }
         else if( fbs->shutter()->state() == 0)
         {
            msg += "SHUT";
         }
         else if( fbs->shutter()->state() == 1)
         {
            msg += "OPEN";
         }
      }

      msg += " Sync: ";
      if(fbs->synchro()) msg += "ON";
      else msg += "OFF";

      msg += " vshift: ";
      if(fbs->vshift() == 0) msg += "---";
      else msg += std::to_string(fbs->vshift());

      msg += " crop: ";
      if(fbs->cropMode()==-1) msg += "---";
      else if(fbs->cropMode()==1) msg += "ON";
      else msg += "OFF";

      return msg;
   
   }

   static std::string mode( void * msgBuffer )
   {
      auto fbs = GetTelem_stdcam_fb(msgBuffer);
      if(fbs->mode() != nullptr)
      {
         return std::string(fbs->mode()->c_str());
      }
      else return "";
   }

   static float xcen( void * msgBuffer )
   {
      auto fbs = GetTelem_stdcam_fb(msgBuffer);
      if(fbs->roi() != nullptr) return fbs->roi()->xcen();
      else return -1;
   }

   static float ycen( void * msgBuffer )
   {
      auto fbs = GetTelem_stdcam_fb(msgBuffer);
      if(fbs->roi() != nullptr) return fbs->roi()->ycen();
      else return -1;
   }

   static int width( void * msgBuffer )
   {
      auto fbs = GetTelem_stdcam_fb(msgBuffer);
      if(fbs->roi() != nullptr) return fbs->roi()->w();
      else return -1;
   }

   static int height( void * msgBuffer )
   {
      auto fbs = GetTelem_stdcam_fb(msgBuffer);
      if(fbs->roi() != nullptr) return fbs->roi()->h();
      else return -1;
   }

   static int xbin( void * msgBuffer )
   {
      auto fbs = GetTelem_stdcam_fb(msgBuffer);
      if(fbs->roi() != nullptr) return fbs->roi()->xbin();
      else return -1;
   }

   static int ybin( void * msgBuffer )
   {
      auto fbs = GetTelem_stdcam_fb(msgBuffer);
      if(fbs->roi() != nullptr) return fbs->roi()->ybin();
      else return -1;
   }

   static float exptime( void * msgBuffer )
   {
      auto fbs = GetTelem_stdcam_fb(msgBuffer);
      return fbs->exptime();
   }
   
   static float fps( void * msgBuffer )
   {
      auto fbs = GetTelem_stdcam_fb(msgBuffer);
      return fbs->fps();
   }

   static float emGain( void * msgBuffer )
   {
      auto fbs = GetTelem_stdcam_fb(msgBuffer);
      return fbs->emGain();
   }

   static float adcSpeed( void * msgBuffer )
   {
      auto fbs = GetTelem_stdcam_fb(msgBuffer);
      return fbs->adcSpeed();
   }
   
   static float temp( void * msgBuffer )
   {
      auto fbs = GetTelem_stdcam_fb(msgBuffer);
      if(fbs->tempCtrl() != nullptr) return fbs->tempCtrl()->temp();
      else return -9999;
   }

   static float tempSetpt( void * msgBuffer )
   {
      auto fbs = GetTelem_stdcam_fb(msgBuffer);
      if(fbs->tempCtrl() != nullptr) return fbs->tempCtrl()->setpt();
      else return -9999;
   }

   static int tempStatus( void * msgBuffer )
   {
      auto fbs = GetTelem_stdcam_fb(msgBuffer);
      if(fbs->tempCtrl() != nullptr) return fbs->tempCtrl()->status();
      else return -9999;
   }

   static int tempOnTarget( void * msgBuffer )
   {
      auto fbs = GetTelem_stdcam_fb(msgBuffer);
      if(fbs->tempCtrl() != nullptr) return fbs->tempCtrl()->ontarget();
      else return -9999;
   }

   static std::string tempStatusStr( void * msgBuffer )
   {
      auto fbs = GetTelem_stdcam_fb(msgBuffer);
      if(fbs->tempCtrl() != nullptr)
      { 
         if(fbs->tempCtrl()->statusStr()) return fbs->tempCtrl()->statusStr()->c_str();
         else return "";
      }
      else return "";
   }

   static std::string shutterStatusStr( void * msgBuffer)
   {
      auto fbs = GetTelem_stdcam_fb(msgBuffer);
      if(fbs->shutter() != nullptr)
      {
         if(fbs->shutter()->statusStr()) return fbs->shutter()->statusStr()->c_str();
         else return "";
      }
      else return "";
   }

   static std::string shutterState( void * msgBuffer )
   {
      auto fbs = GetTelem_stdcam_fb(msgBuffer);
      if(fbs->shutter() != nullptr)
      {
         if( fbs->shutter()->state() == -1) return "UNKNOWN";
         else if( fbs->shutter()->state() == 0) return "SHUT";
         else if( fbs->shutter()->state() == 1) return "OPEN";
         else return "INVALID";
      }
      else return "INVALID";
   }

   static bool synchro( void * msgBuffer )
   {
      auto fbs = GetTelem_stdcam_fb(msgBuffer);
      return fbs->synchro();
   }

   static float vshift( void * msgBuffer )
   {
      auto fbs = GetTelem_stdcam_fb(msgBuffer);
      return fbs->vshift();
   }

   static bool cropMode( void * msgBuffer )
   {
      auto fbs = GetTelem_stdcam_fb(msgBuffer);

      //slightly different because define a default to indicated not-used.
      if(fbs->cropMode() == 1) return true;
      else return false;
   }

   /// Get the logMetaDetail for a member by name
   /**
     * \returns the function pointer cast to void*
     * \returns -1 for an unknown member
     */ 
   static logMetaDetail getAccessor( const std::string & member /**< [in] the name of the member */ )
   {
      if(member == "mode") return logMetaDetail({"MODE", logMeta::valTypes::String, logMeta::metaTypes::State, (void *) &mode});
      else if(member == "xcen") return logMetaDetail({"ROI XCEN", logMeta::valTypes::Float, logMeta::metaTypes::State, (void *) &xcen});
      else if(member == "ycen") return logMetaDetail({"ROI YCEN", logMeta::valTypes::Float, logMeta::metaTypes::State, (void *) &ycen});
      else if(member == "width") return logMetaDetail({"ROI WIDTH", logMeta::valTypes::Int, logMeta::metaTypes::State, (void *) &width});
      else if(member == "height") return logMetaDetail({"ROI HEIGHT", logMeta::valTypes::Int, logMeta::metaTypes::State, (void *) &height});
      else if(member == "xbin") return logMetaDetail({"ROI XBIN", logMeta::valTypes::Int, logMeta::metaTypes::State, (void *) &xbin});
      else if(member == "ybin") return logMetaDetail({"ROI YBIN", logMeta::valTypes::Int, logMeta::metaTypes::State, (void *) &ybin});
      else if(member == "exptime") return logMetaDetail({"EXPTIME", logMeta::valTypes::Float, logMeta::metaTypes::State, (void *) &exptime});
      else if(member == "fps") return logMetaDetail({"FPS", logMeta::valTypes::Float, logMeta::metaTypes::State, (void *) &fps});
      else if(member == "emGain") return logMetaDetail({"EMGAIN", logMeta::valTypes::Float, logMeta::metaTypes::State, (void *) &emGain});
      else if(member == "adcSpeed") return logMetaDetail({"ADC SPEED", logMeta::valTypes::Float, logMeta::metaTypes::State, (void *) &adcSpeed});
      else if(member == "temp") return logMetaDetail({"TEMP", logMeta::valTypes::Float, logMeta::metaTypes::Continuous, (void *) &temp});
      else if(member == "tempSetpt") return logMetaDetail({"TEMP SETPT", logMeta::valTypes::Float, logMeta::metaTypes::State, (void *) &tempSetpt});
      else if(member == "tempStatus") return logMetaDetail({"TEMP STATUS", logMeta::valTypes::Int, logMeta::metaTypes::Continuous, (void *) &tempStatus});
      else if(member == "tempOnTarget") return logMetaDetail({"TEMP ONTGT", logMeta::valTypes::Int, logMeta::metaTypes::Continuous, (void *) &tempOnTarget});
      else if(member == "tempStatusStr") return logMetaDetail({"TEMP STATUSSTR", logMeta::valTypes::String, logMeta::metaTypes::State, (void *) &tempStatusStr});
      else if(member == "shutterStatusStr") return logMetaDetail({"SHUTTER STATUS", logMeta::valTypes::String, logMeta::metaTypes::State, (void *) &shutterStatusStr});
      else if(member == "shutterState") return logMetaDetail({"SHUTTER", logMeta::valTypes::String, logMeta::metaTypes::State, (void *) &shutterState});
      else if(member == "synchro") return logMetaDetail({"SYNCHRO", logMeta::valTypes::Bool, logMeta::metaTypes::State, (void *) &synchro});
      else if(member == "vshift") return logMetaDetail({"VSHIFTSPD", logMeta::valTypes::Float, logMeta::metaTypes::State, (void *) &vshift});
      else if(member == "cropMode") return logMetaDetail({"CROPMODE", logMeta::valTypes::Bool, logMeta::metaTypes::State, (void *) &cropMode});
      else
      {
         std::cerr << "No string member " << member << " in telem_stdcam\n";
         return logMetaDetail();
      }
   }
      
}; //telem_stdcam



} //namespace logger
} //namespace MagAOX

#endif //logger_types_telem_stdcam_hpp

