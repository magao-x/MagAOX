/** \file picamCtrl.hpp
  * \brief The MagAO-X Princeton Instruments EMCCD camera controller.
  *
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup picamCtrl_files
  */

#ifndef picamCtrl_hpp
#define picamCtrl_hpp


#include <ImageStruct.h>
#include <ImageStreamIO.h>

#include <picam_advanced.h>

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

#define DEBUG

#ifdef DEBUG
#define BREADCRUMB  std::cerr << __FILE__ << " " << __LINE__ << "\n";
#else
#define BREADCRUMB
#endif

inline
std::string PicamEnum2String( PicamEnumeratedType type, piint value )
{
    const pichar* string;
    Picam_GetEnumerationString( type, value, &string );
    std::string str(string);
    Picam_DestroyString( string );

    return str;
}

namespace MagAOX
{
namespace app
{



/** \defgroup picamCtrl Princeton Instruments EMCCD Camera
  * \brief Control of a Princeton Instruments EMCCD Camera.
  *
  * <a href="../handbook/apps/picamCtrl.html">Application Documentation</a>
  *
  * \ingroup apps
  *
  */

/** \defgroup picamCtrl_files Princeton Instruments EMCCD Camera Files
  * \ingroup picamCtrl
  */

/** MagAO-X application to control a Princeton Instruments EMCCD
  *
  * \ingroup picamCtrl
  *
  * \todo Config item for ImageStreamIO name filename
  * \todo implement ImageStreamIO circular buffer, with config setting
  */
class picamCtrl : public MagAOXApp<>, public dev::stdCamera<picamCtrl>, public dev::frameGrabber<picamCtrl>, public dev::dssShutter<picamCtrl>
{

   friend class dev::stdCamera<picamCtrl>;
   friend class dev::frameGrabber<picamCtrl>;
   friend class dev::dssShutter<picamCtrl>;

   typedef MagAOXApp<> MagAOXAppT;

protected:

   /** \name configurable parameters
     *@{
     */
   std::string m_serialNumber; ///< The camera's identifying serial number


   ///@}

   int m_depth {0};
   

   std::vector<std::string> m_adcSpeeds = {"00100_kHz", "01_MHz", "05_MHz", "10_MHz", "20_MHz", "30_MHz"};
   std::vector<std::string> m_adcSpeedLabels = {"100 kHz", "1 MHz", "5 MHz", "10 MHz", "20 MHz", "30 MHz"};
   
   std::vector<float> m_adcSpeedValues = {0.1, 1, 5,10,20,30};
   
   int m_adcSpeed {30}; ///< The ADC speed, 5, 10, 20, or 30.

   

   
   


   PicamHandle m_cameraHandle {0};
   PicamHandle m_modelHandle {0};

   PicamAcquisitionBuffer m_acqBuff;
   PicamAvailableData m_available;

   std::string m_cameraName;
   std::string m_cameraModel;

public:

   ///Default c'tor
   picamCtrl();

   ///Destructor
   ~picamCtrl() noexcept;

   /// Setup the configuration system (called by MagAOXApp::setup())
   virtual void setupConfig();

   /// load the configuration system results (called by MagAOXApp::setup())
   virtual void loadConfig();

   /// Startup functions
   /** Sets up the INDI vars.
     *
     */
   virtual int appStartup();

   /// Implementation of the FSM for the Siglent SDG
   virtual int appLogic();

   /// Implementation of the on-power-off FSM logic
   virtual int onPowerOff();

   /// Implementation of the while-powered-off FSM
   virtual int whilePowerOff();

   /// Do any needed shutdown tasks.  Currently nothing in this app.
   virtual int appShutdown();

protected:
   int getPicamParameter( piint & value,
                          PicamParameter parameter
                        );

   int getPicamParameter( piflt & value,
                          PicamParameter parameter
                        );

   int setPicamParameter( PicamParameter parameter,
                          pi64s value
                        );

   int setPicamParameter( PicamHandle handle,
                          PicamParameter parameter,
                          piflt value
                        );

   int setPicamParameter( PicamParameter parameter,
                          piflt value
                        );

   int setPicamParameterOnline( PicamHandle handle,
                                PicamParameter parameter,
                                piflt value
                              );

   int setPicamParameterOnline( PicamParameter parameter,
                                piflt value
                              );

   int connect();

   int getAcquisitionState();

   int getTemps();

   int adcSpeed();

   int setAdcSpeed(int newspd);

   

   // stdCamera interface:
   
   //This must set the power-on default values of
   /* -- m_ccdTempSetpt
    * -- m_currentROI 
    */
   int powerOnDefaults();
   
   int setTempControl();
   int setTempSetPt();
   int setExpTime();
   int setFPS();
   int setNextROI();
   
   //Framegrabber interface:
   int configureAcquisition();
   int startAcquisition();
   int acquireAndCheckValid();
   int loadImageIntoStream(void * dest);
   int reconfig();


   //INDI:
protected:

   pcf::IndiProperty m_indiP_adcspeed;

public:
   INDI_NEWCALLBACK_DECL(picamCtrl, m_indiP_adcspeed);

};

inline
picamCtrl::picamCtrl() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   m_powerMgtEnabled = true;

   m_acqBuff.memory_size = 0;
   m_acqBuff.memory = 0;

   m_usesFPS = false;
   m_usesModes = false;
   
   return;
}

inline
picamCtrl::~picamCtrl() noexcept
{
   if(m_acqBuff.memory)
   {
      free(m_acqBuff.memory);
   }

   return;
}

inline
void picamCtrl::setupConfig()
{


   config.add("camera.serialNumber", "", "camera.serialNumber", argType::Required, "camera", "serialNumber", false, "int", "The identifying serial number of the camera.");

   dev::stdCamera<picamCtrl>::setupConfig(config);
   dev::frameGrabber<picamCtrl>::setupConfig(config);
   dev::dssShutter<picamCtrl>::setupConfig(config);
}

inline
void picamCtrl::loadConfig()
{

   config(m_serialNumber, "camera.serialNumber");

   dev::stdCamera<picamCtrl>::loadConfig(config);
   dev::frameGrabber<picamCtrl>::loadConfig(config);
   dev::dssShutter<picamCtrl>::loadConfig(config);

}

inline
int picamCtrl::appStartup()
{

   createStandardIndiSelectionSw(m_indiP_adcspeed, "adcspeed", m_adcSpeeds, "ADC Speed");
   for(size_t n=0; n< m_adcSpeeds.size(); ++n) m_indiP_adcspeed[m_adcSpeeds[n]].setLabel(m_adcSpeedLabels[n]);
   registerIndiPropertyNew(m_indiP_adcspeed, INDI_NEWCALLBACK(m_indiP_adcspeed));
   
   
   m_minTemp = -55;
   m_maxTemp = 25;
   m_stepTemp = 0;
   
   m_minROIx = 0;
   m_maxROIx = 1023;
   m_stepROIx = 0;
   
   m_minROIy = 0;
   m_maxROIy = 1023;
   m_stepROIy = 0;
   
   m_minROIWidth = 1;
   m_maxROIWidth = 1024;
   m_stepROIWidth = 4;
   
   m_minROIHeight = 1;
   m_maxROIHeight = 1024;
   m_stepROIHeight = 1;
   
   m_minROIBinning_x = 1;
   m_maxROIBinning_x = 32;
   m_stepROIBinning_x = 1;
   
   m_minROIBinning_y = 1;
   m_maxROIBinning_y = 1024;
   m_stepROIBinning_y = 1;
   
   if(dev::stdCamera<picamCtrl>::appStartup() < 0)
   {
      return log<software_critical,-1>({__FILE__,__LINE__});
   }
   
   
   if(dev::frameGrabber<picamCtrl>::appStartup() < 0)
   {
      return log<software_critical,-1>({__FILE__,__LINE__});
   }

   if(dev::dssShutter<picamCtrl>::appStartup() < 0)
   {
      return log<software_critical,-1>({__FILE__,__LINE__});
   }

   return 0;

}

inline
int picamCtrl::appLogic()
{
   //and run stdCamera's appLogic
   if(dev::stdCamera<picamCtrl>::appLogic() < 0)
   {
      return log<software_error, -1>({__FILE__, __LINE__});
   }
   
   //first run frameGrabber's appLogic to see if the f.g. thread has exited.
   if(dev::frameGrabber<picamCtrl>::appLogic() < 0)
   {
      return log<software_error, -1>({__FILE__, __LINE__});
   }

   //and run dssShutter's appLogic
   if(dev::dssShutter<picamCtrl>::appLogic() < 0)
   {
      return log<software_error, -1>({__FILE__, __LINE__});
   }


   if( state() == stateCodes::NOTCONNECTED || state() == stateCodes::NODEVICE || state() == stateCodes::ERROR)
   {
      m_reconfig = true; //Trigger a f.g. thread reconfig.

      //Might have gotten here because of a power off.
      if(MagAOXAppT::m_powerState == 0) return 0;

      std::unique_lock<std::mutex> lock(m_indiMutex);
      if(connect() < 0)
      {
         log<software_error>({__FILE__, __LINE__});
      }

      if(state() != stateCodes::CONNECTED) return 0;
   }

   if( state() == stateCodes::CONNECTED )
   {
      //Get a lock
      std::unique_lock<std::mutex> lock(m_indiMutex);

      if( getAcquisitionState() < 0 )
      {
         return log<software_error,0>({__FILE__,__LINE__});
      }

      if( setTempSetPt() < 0 ) //m_ccdTempSetpt already set on power on
      {
         return log<software_error,0>({__FILE__,__LINE__});
      }

      
      if(frameGrabber<picamCtrl>::updateINDI() < 0)
      {
         return log<software_error,0>({__FILE__,__LINE__});
      }
      
   }

   if( state() == stateCodes::READY || state() == stateCodes::OPERATING )
   {
      //Get a lock if we can
      std::unique_lock<std::mutex> lock(m_indiMutex, std::try_to_lock);

      //but don't wait for it, just go back around.
      if(!lock.owns_lock()) return 0;

      if(getAcquisitionState() < 0)
      {
         if(MagAOXAppT::m_powerState == 0) return 0;

         state(stateCodes::ERROR);
         return 0;
      }

      if(getTemps() < 0)
      {
         if(MagAOXAppT::m_powerState == 0) return 0;

         state(stateCodes::ERROR);
         return 0;
      }

      if(stdCamera<picamCtrl>::updateINDI() < 0)
      {
         return log<software_error,0>({__FILE__,__LINE__});
      }
      
      if(frameGrabber<picamCtrl>::updateINDI() < 0)
      {
         return log<software_error,0>({__FILE__,__LINE__});
      }

      if(dssShutter<picamCtrl>::updateINDI() < 0)
      {
         log<software_error>({__FILE__, __LINE__});
         state(stateCodes::ERROR);
         return 0;
      }

   }

   //Fall through check?

   return 0;

}

inline
int picamCtrl::onPowerOff()
{
   std::lock_guard<std::mutex> lock(m_indiMutex);

   if(m_cameraHandle)
   {
      Picam_CloseCamera(m_cameraHandle);
      m_cameraHandle = 0;
   }

   Picam_UninitializeLibrary();

   ///\todo error check these base class fxns.
   dssShutter<picamCtrl>::onPowerOff();

   stdCamera<picamCtrl>::onPowerOff();
   
   return 0;
}

inline
int picamCtrl::whilePowerOff()
{
   ///\todo error check these base class fxns.

   dssShutter<picamCtrl>::whilePowerOff();


   return 0;
}

inline
int picamCtrl::appShutdown()
{
   dev::frameGrabber<picamCtrl>::appShutdown();

   if(m_cameraHandle)
   {
      Picam_CloseCamera(m_cameraHandle);
      m_cameraHandle = 0;
   }

   Picam_UninitializeLibrary();

   ///\todo error check these base class fxns.
   dev::frameGrabber<picamCtrl>::appShutdown();
   dev::dssShutter<picamCtrl>::appShutdown();

   return 0;
}

inline
int picamCtrl::getPicamParameter( piint & value,
                                  PicamParameter parameter
                                )
{
   PicamError error = Picam_GetParameterIntegerValue( m_cameraHandle, parameter, &value );

   if(MagAOXAppT::m_powerState == 0) return -1; //Flag error but don't log

   if(error != PicamError_None)
   {
      log<software_error>({__FILE__, __LINE__, 0, error, PicamEnum2String(PicamEnumeratedType_Error, error)});
      return -1;
   }

   return 0;
}

inline
int picamCtrl::getPicamParameter( piflt & value,
                                  PicamParameter parameter
                                )
{
   PicamError error = Picam_GetParameterFloatingPointValue( m_cameraHandle, parameter, &value );

   if(MagAOXAppT::m_powerState == 0) return -1; //Flag error but don't log

   if(error != PicamError_None)
   {
      log<software_error>({__FILE__, __LINE__, 0, error, PicamEnum2String(PicamEnumeratedType_Error, error)});
      return -1;
   }

   return 0;
}

inline
int picamCtrl::setPicamParameter( PicamParameter parameter,
                                  pi64s value
                                )
{
   PicamError error = Picam_SetParameterLargeIntegerValue( m_cameraHandle, parameter, value );
   if(error != PicamError_None)
   {
      log<software_error>({__FILE__, __LINE__, 0, error, PicamEnum2String(PicamEnumeratedType_Error, error)});
      return -1;
   }

   const PicamParameter* failed_parameters;
   piint failed_parameters_count;

   error = Picam_CommitParameters( m_cameraHandle, &failed_parameters, &failed_parameters_count );
   if(error != PicamError_None)
   {
      log<software_error>({__FILE__, __LINE__, 0, error, PicamEnum2String(PicamEnumeratedType_Error, error)});
      return -1;
   }

   for( int i=0; i< failed_parameters_count; ++i)
   {
      if( failed_parameters[i] ==  parameter)
      {
         Picam_DestroyParameters( failed_parameters );
         return log<text_log,-1>( "Parameter not committed");
      }
   }

   Picam_DestroyParameters( failed_parameters );

   return 0;
}

inline
int picamCtrl::setPicamParameter( PicamHandle handle,
                                  PicamParameter parameter,
                                  piflt value
                                )
{
   PicamError error = Picam_SetParameterFloatingPointValue( handle, parameter, value );
   if(error != PicamError_None)
   {
      log<software_error>({__FILE__, __LINE__, 0, error, PicamEnum2String(PicamEnumeratedType_Error, error)});
      return -1;
   }

   const PicamParameter* failed_parameters;
   piint failed_parameters_count;

   error = Picam_CommitParameters( handle, &failed_parameters, &failed_parameters_count );
   if(error != PicamError_None)
   {
      log<software_error>({__FILE__, __LINE__, 0, error, PicamEnum2String(PicamEnumeratedType_Error, error)});
      return -1;
   }

   for( int i=0; i< failed_parameters_count; ++i)
   {
      if( failed_parameters[i] ==  parameter)
      {
         Picam_DestroyParameters( failed_parameters );
         return log<text_log,-1>( "Parameter not committed");
      }
   }

   Picam_DestroyParameters( failed_parameters );

   return 0;
}

inline
int picamCtrl::setPicamParameter( PicamParameter parameter,
                                  piflt value
                                )
{
   return setPicamParameter( m_cameraHandle, parameter, value);
}

inline
int picamCtrl::setPicamParameterOnline( PicamHandle handle,
                                        PicamParameter parameter,
                                        piflt value
                                       )
{
   PicamError error = Picam_SetParameterFloatingPointValueOnline( handle, parameter, value );
   if(error != PicamError_None)
   {
      log<software_error>({__FILE__, __LINE__, 0, error, PicamEnum2String(PicamEnumeratedType_Error, error)});
      return -1;
   }

   return 0;
}

inline
int picamCtrl::setPicamParameterOnline( PicamParameter parameter,
                                        piflt value
                                       )
{
   return setPicamParameterOnline(m_cameraHandle, parameter, value);
}
inline
int picamCtrl::connect()
{

   PicamError error;
   PicamCameraID * id_array;
   piint id_count;

   if(m_acqBuff.memory)
   {
      std::cerr << "Clearing\n";
      free(m_acqBuff.memory);
      m_acqBuff.memory = NULL;
      m_acqBuff.memory_size = 0;
   }

   BREADCRUMB

   Picam_UninitializeLibrary();

   BREADCRUMB

   //Have to initialize the library every time.  Otherwise we won't catch a newly booted camera.
   Picam_InitializeLibrary();

   BREADCRUMB

   if(m_cameraHandle)
   {
      BREADCRUMB
      Picam_CloseCamera(m_cameraHandle);
      m_cameraHandle = 0;
   }

   BREADCRUMB

   Picam_GetAvailableCameraIDs((const PicamCameraID **) &id_array, &id_count);

   BREADCRUMB

   if(id_count == 0)
   {
      BREADCRUMB

      Picam_DestroyCameraIDs(id_array);

      BREADCRUMB

      Picam_UninitializeLibrary();

      BREADCRUMB

      state(stateCodes::NODEVICE);
      if(!stateLogged())
      {
         log<text_log>("no P.I. Cameras available.");
      }
      return 0;
   }

   BREADCRUMB

   for(int i=0; i< id_count; ++i)
   {
      BREADCRUMB

      if( std::string(id_array[i].serial_number) == m_serialNumber )
      {
         BREADCRUMB
         std::cerr << "Camera was found.  Now connecting.\n";

         error = PicamAdvanced_OpenCameraDevice(&id_array[i], &m_cameraHandle);
         if(error == PicamError_None)
         {
            m_cameraName = id_array[i].sensor_name;
            m_cameraModel = PicamEnum2String(PicamEnumeratedType_Model, id_array[i].model);

            BREADCRUMB

            error = PicamAdvanced_GetCameraModel( m_cameraHandle, &m_modelHandle );
            if( error != PicamError_None )
            {
               std::cerr << "failed to get camera model\n";
            }

            state(stateCodes::CONNECTED);
            log<text_log>("Connected to " + m_cameraName + " [S/N " + m_serialNumber + "]");

            BREADCRUMB

            Picam_DestroyCameraIDs(id_array);

            BREADCRUMB

            return 0;
         }
         else
         {
            BREADCRUMB

            state(stateCodes::ERROR);
            if(!stateLogged())
            {
               log<software_error>({__FILE__,__LINE__, 0, error, "Error connecting to camera."});
            }

            BREADCRUMB

            Picam_DestroyCameraIDs(id_array);

            BREADCRUMB

            Picam_UninitializeLibrary();

            BREADCRUMB
            return -1;
         }
      }
   }

   state(stateCodes::NODEVICE);
   if(!stateLogged())
   {
      log<text_log>("Camera not found in available ids.");
   }

   Picam_DestroyCameraIDs(id_array);


   Picam_UninitializeLibrary();


   return 0;
}


inline
int picamCtrl::getAcquisitionState()
{
   pibln running = false;

   PicamError error = Picam_IsAcquisitionRunning(m_cameraHandle, &running);

   if(MagAOXAppT::m_powerState == 0) return 0;

   if(error != PicamError_None)
   {
      log<software_error>({__FILE__, __LINE__, 0, error, PicamEnum2String(PicamEnumeratedType_Error, error)});
      state(stateCodes::ERROR);
      return -1;
   }

   if(running) state(stateCodes::OPERATING);
   else state(stateCodes::READY);

   return 0;

}

inline
int picamCtrl::getTemps()
{
   piflt currTemperature;

   if(getPicamParameter(currTemperature, PicamParameter_SensorTemperatureReading) < 0)
   {
      if(MagAOXAppT::m_powerState == 0) return 0;

      log<software_error>({__FILE__, __LINE__});
      state(stateCodes::ERROR);
      return -1;
   }

   m_ccdTemp = currTemperature;
   
   //PicamSensorTemperatureStatus
   piint status;

   if(getPicamParameter( status, PicamParameter_SensorTemperatureStatus ) < 0)
   {
      if(MagAOXAppT::m_powerState == 0) return 0;

      log<software_error>({__FILE__, __LINE__});
      state(stateCodes::ERROR);
      return -1;
   }

   if(status == 1) 
   {
      m_tempControlStatus = true;
      m_tempControlOnTarget = false;
      m_tempControlStatusStr = "UNLOCKED";
   }
   else if(status == 2) 
   {
      m_tempControlStatus = true;
      m_tempControlOnTarget = true;
      m_tempControlStatusStr = "LOCKED";
   }
   else if(status == 3) 
   {
      m_tempControlStatus = false;
      m_tempControlOnTarget = false;
      m_tempControlStatusStr = "FAULTED";
      log<text_log>("temperature control faulted", logPrio::LOG_ALERT);
   }
   else
   {
      m_tempControlStatus = false;
      m_tempControlOnTarget = false;
      m_tempControlStatusStr = "UNKNOWN";
   }   


   return 0;

}

inline
int picamCtrl::setAdcSpeed(int newspd)
{
   if(newspd != 5 && newspd != 10 && newspd != 20 && newspd != 30)
   {
      log<text_log>("Invalid ADC speed requested.", logPrio::LOG_ERROR);
      return -1;
   }

   m_adcSpeed = newspd;

   m_reconfig = true;

   return 0;
}




inline
int picamCtrl::setFPS()
{
   return 0;
}

inline 
int picamCtrl::powerOnDefaults()
{
   m_ccdTempSetpt = -55; //This is the power on setpoint

   m_currentROI.x = 511.5;
   m_currentROI.y = 511.5;
   m_currentROI.w = 1024;
   m_currentROI.h = 1024;
   m_currentROI.bin_x = 1;
   m_currentROI.bin_y = 1;

   return 0;
}

inline 
int picamCtrl::setTempControl()
{
   //Always on
   m_tempControlStatus = true;
   m_tempControlStatusSet = true;
   updateSwitchIfChanged(m_indiP_tempcont, "toggle", pcf::IndiElement::On, INDI_IDLE);
   return 0;
}

inline 
int picamCtrl::setTempSetPt()
{
   ///\todo bounds check here.
   m_reconfig = true;

   return 0;
}

inline
int picamCtrl::setExpTime()
{
   long intexptime = m_expTimeSet * 1000 * 10000 + 0.5;
   piflt exptime = ((double)intexptime)/10000;

   int rv;
   
   std::string mode;
   if(state() == stateCodes::OPERATING)
   {
      mode = "online";
      
      rv = setPicamParameterOnline(m_modelHandle, PicamParameter_ExposureTime, exptime);      
   }
   else
   {
      mode = "offline";
      
      rv = setPicamParameter(m_modelHandle, PicamParameter_ExposureTime, exptime);
   }

   if(rv < 0)
   {
      log<software_error>({__FILE__, __LINE__, "Error setting exposure time"});
      return -1;
   }

   m_expTime = exptime/1000.0;
   
   log<text_log>( "Set exposure time " + mode + " to: " + std::to_string(exptime/1000.0) + " sec");

   updateIfChanged(m_indiP_exptime, "current", m_expTime, INDI_IDLE);

   return 0;
}

//Set ROI property to busy if accepted, set toggle to Off and Idlw either way.
//Set ROI actual 
//Update current values (including struct and indiP) and set to OK when done
inline 
int picamCtrl::setNextROI()
{
   std::cerr << "setNextROI:\n";
   std::cerr << "  m_nextROI.x = " << m_nextROI.x << "\n";
   std::cerr << "  m_nextROI.y = " << m_nextROI.y << "\n";
   std::cerr << "  m_nextROI.w = " << m_nextROI.w << "\n";
   std::cerr << "  m_nextROI.h = " << m_nextROI.h << "\n";
   std::cerr << "  m_nextROI.bin_x = " << m_nextROI.bin_x << "\n";
   std::cerr << "  m_nextROI.bin_y = " << m_nextROI.bin_y << "\n";
   
   m_reconfig = true;

   updateSwitchIfChanged(m_indiP_roi_set, "request", pcf::IndiElement::Off, INDI_IDLE);
   
   return 0;
   
}

inline
int picamCtrl::configureAcquisition()
{

   piint readoutStride;
   piint framesPerReadout;
   piint frameStride;
   piint frameSize;
   piint pixelBitDepth;



   std::unique_lock<std::mutex> lock(m_indiMutex);


   //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
   //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
   // Check Frame Transfer
   //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
   //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

   piint cmode;
   if(getPicamParameter(cmode, PicamParameter_ReadoutControlMode) < 0)
   {
      log<software_error>({__FILE__,__LINE__, "could not get Readout Control Mode"});
      return -1;
   }

   if( cmode != PicamReadoutControlMode_FrameTransfer)
   {
      log<software_error>({__FILE__,__LINE__, "Readout Control Mode not configured for frame transfer"}) ;
      return -1;
   }

   //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
   //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
   // Temperature
   //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
   //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

   if(setPicamParameter( PicamParameter_SensorTemperatureSetPoint, m_ccdTempSetpt) < 0)
   {
      log<software_error>({__FILE__, __LINE__, "Error setting temperature setpoint"});
      state(stateCodes::ERROR);
      return -1;
   }

   log<text_log>( "Set temperature set point: " + std::to_string(m_ccdTempSetpt) + " C");

   //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
   //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
   // ADC Speed
   //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
   //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

   if( setPicamParameter(m_modelHandle, PicamParameter_AdcSpeed, m_adcSpeed) < 0)
   {
      log<software_error>({__FILE__, __LINE__, "Error setting ADC Speed"});
      state(stateCodes::ERROR);
      return -1;
   }

   std::string adcstr;
   for(size_t n=0; n<m_adcSpeeds.size(); ++n)
   {
      if(m_adcSpeedValues[n] == m_adcSpeed)
      {
         adcstr = m_adcSpeeds[n];
         break;
      }
   }
   indi::updateSelectionSwitchIfChanged( m_indiP_adcspeed, adcstr, m_indiDriver, INDI_OK);
   
//    updateIfChanged(m_indiP_adcspeed, "current", m_adcSpeed);
//    updateIfChanged(m_indiP_adcspeed, "target", std::string(""));

   log<text_log>( "ADC Speed: " + std::to_string(m_adcSpeed) + " MHz");

   //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
   //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
   // Dimensions
   //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
   //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
   
   PicamRois  nextrois;
   PicamRoi nextroi;
   
   nextrois.roi_array = &nextroi;
   nextrois.roi_count = 1;
   
   nextroi.x = (m_nextROI.x - 0.5*( (float) m_nextROI.w - 1.0));
   nextroi.y = (m_nextROI.y - 0.5*( (float) m_nextROI.h - 1.0));
   nextroi.width = m_nextROI.w;
   nextroi.height = m_nextROI.h;
   nextroi.x_binning = m_nextROI.bin_x;
   nextroi.y_binning = m_nextROI.bin_y;
   
   PicamError error = Picam_SetParameterRoisValue( m_cameraHandle, PicamParameter_Rois, &nextrois);   
   if( error != PicamError_None )
   {
      std::cerr << PicamEnum2String(PicamEnumeratedType_Error, error) << "\n";
      log<software_error>({__FILE__, __LINE__, 0, error, PicamEnum2String(PicamEnumeratedType_Error, error)});
      state(stateCodes::ERROR);
      return -1;
   }
   
   if(getPicamParameter(readoutStride, PicamParameter_ReadoutStride) < 0)
   {
      log<software_error>({__FILE__, __LINE__, "Error getting readout stride"});
      state(stateCodes::ERROR);
      return -1;
   }

   if(getPicamParameter(frameStride, PicamParameter_FrameStride) < 0)
   {
      log<software_error>({__FILE__, __LINE__, "Error getting frame stride"});
      state(stateCodes::ERROR);

      return -1;
   }

   if(getPicamParameter(framesPerReadout, PicamParameter_FramesPerReadout) < 0)
   {
      log<software_error>({__FILE__, __LINE__, "Error getting frames per readout"});
      state(stateCodes::ERROR);
      return -1;
   }

   if(getPicamParameter( frameSize, PicamParameter_FrameSize) < 0)
   {
      log<software_error>({__FILE__, __LINE__, "Error getting frame size"});
      state(stateCodes::ERROR);
      return -1;
   }

   if(getPicamParameter( pixelBitDepth, PicamParameter_PixelBitDepth) < 0)
   {
      log<software_error>({__FILE__, __LINE__,"Error getting pixel bit depth"});
      state(stateCodes::ERROR);
      return -1;
   }
   m_depth = pixelBitDepth;

   const PicamRois* rois;
   error = Picam_GetParameterRoisValue( m_cameraHandle, PicamParameter_Rois, &rois );
   if( error != PicamError_None )
   {
      log<software_error>({__FILE__, __LINE__, 0, error, PicamEnum2String(PicamEnumeratedType_Error, error)});
      state(stateCodes::ERROR);
      return -1;
   }
   m_xbinning = rois->roi_array[0].x_binning;
   m_currentROI.bin_x = m_xbinning;
   m_ybinning = rois->roi_array[0].y_binning;
   m_currentROI.bin_y = m_ybinning;
   
   std::cerr << rois->roi_array[0].x << "\n";
   std::cerr << (rois->roi_array[0].x-1) << "\n";
   std::cerr << rois->roi_array[0].width << "\n";
   std::cerr << 0.5*( (float) (rois->roi_array[0].width - 1.0)) << "\n";
   
   m_currentROI.x = (rois->roi_array[0].x) + 0.5*( (float) (rois->roi_array[0].width - 1.0)) ;
   m_currentROI.y = (rois->roi_array[0].y) + 0.5*( (float) (rois->roi_array[0].height - 1.0)) ;
   
   m_currentROI.w = rois->roi_array[0].width;
   m_currentROI.h = rois->roi_array[0].height;
   
   m_width  = rois->roi_array[0].width  / rois->roi_array[0].x_binning;
   m_height = rois->roi_array[0].height / rois->roi_array[0].y_binning;
   Picam_DestroyRois( rois );


   updateIfChanged( m_indiP_roi_x, "current", m_currentROI.x, INDI_OK);
   updateIfChanged( m_indiP_roi_y, "current", m_currentROI.y, INDI_OK);
   updateIfChanged( m_indiP_roi_w, "current", m_currentROI.w, INDI_OK);
   updateIfChanged( m_indiP_roi_h, "current", m_currentROI.h, INDI_OK);
   updateIfChanged( m_indiP_roi_bin_x, "current", m_currentROI.bin_x, INDI_OK);
   updateIfChanged( m_indiP_roi_bin_y, "current", m_currentROI.bin_y, INDI_OK);
   
   //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
   //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
   // Exposure Time and Frame Rate
   //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
   //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

   const PicamRangeConstraint * constraint_array;
   piint constraint_count;
   PicamAdvanced_GetParameterRangeConstraints( m_modelHandle, PicamParameter_ExposureTime, &constraint_array, &constraint_count);

   if(constraint_count != 1)
   {
      log<text_log>("Constraint count is not 1: " + std::to_string(constraint_count) + " constraints",logPrio::LOG_ERROR);
   }
   else
   {
      m_minExpTime = constraint_array[0].minimum;
      m_maxExpTime = constraint_array[0].maximum;
      m_stepExpTime = constraint_array[0].increment;

      m_indiP_exptime["current"].setMin(m_minExpTime);
      m_indiP_exptime["current"].setMax(m_maxExpTime);
      m_indiP_exptime["current"].setStep(m_stepExpTime);
   
      m_indiP_exptime["target"].setMin(m_minExpTime);
      m_indiP_exptime["target"].setMax(m_maxExpTime);
      m_indiP_exptime["target"].setStep(m_stepExpTime);
   }

   if(m_expTimeSet > 0)
   {
      long intexptime = m_expTimeSet * 1000 * 10000 + 0.5;
      piflt exptime = ((double)intexptime)/10000;
   
      std::cerr << "Setting exposure time to " << m_expTimeSet << "\n";
      int rv = setPicamParameter(m_modelHandle, PicamParameter_ExposureTime, exptime);
   
      if(rv < 0)
      {
         log<software_error>({__FILE__, __LINE__, "Error setting exposure time"});
         return -1;
      }
   }
   
   piflt exptime;
   if(getPicamParameter(exptime, PicamParameter_ExposureTime) < 0)
   {
      std::cerr << "could not get Exposuretime\n";
   }
   else
   {
      m_expTime = exptime/1000.0;
      m_expTimeSet = m_expTime; //At this point it must be true.
      updateIfChanged(m_indiP_exptime, "current", m_expTime, INDI_IDLE);
      updateIfChanged(m_indiP_exptime, "target", m_expTimeSet, INDI_IDLE);
   }

//    piflt FrameRateCalculation;
//    if(getPicamParameter(FrameRateCalculation, PicamParameter_FrameRateCalculation) < 0)
//    {
//       std::cerr << "could not get FrameRateCalculation\n";
//    }
   
   piint AdcQuality;
   if(getPicamParameter(AdcQuality, PicamParameter_AdcQuality) < 0)
   {
      std::cerr << "could not get AdcQuality\n";
   }

   std::string adcqStr = PicamEnum2String( PicamEnumeratedType_AdcQuality, AdcQuality );

   std::cerr << "AdcQuality is: " << adcqStr << "\n";




   piflt AdcSpeed;
   if(getPicamParameter(AdcSpeed, PicamParameter_AdcSpeed) < 0)
   {
      std::cerr << "could not get AdcSpeed\n";
   }

   std::cerr << "AdcSpeed is: " << AdcSpeed << "\n";


   std::cerr << "************************************************************\n";



/*


   std::cerr << "Onlineable:\n";
   pibln onlineable;
   Picam_CanSetParameterOnline(m_modelHandle, PicamParameter_ReadoutControlMode,&onlineable);
   std::cerr << "ReadoutControlMode: " << onlineable << "\n"; //0

   Picam_CanSetParameterOnline(m_modelHandle, PicamParameter_AdcQuality,&onlineable);
   std::cerr << "AdcQuality: " << onlineable << "\n"; //0

   Picam_CanSetParameterOnline(m_modelHandle, PicamParameter_AdcAnalogGain,&onlineable);
   std::cerr << "AdcAnalogGain: " << onlineable << "\n"; //1

   Picam_CanSetParameterOnline(m_modelHandle, PicamParameter_DisableCoolingFan,&onlineable);
   std::cerr << "DisableCoolingFan: " << onlineable << "\n";//0

   Picam_CanSetParameterOnline(m_modelHandle, PicamParameter_SensorTemperatureSetPoint,&onlineable);
   std::cerr << "SensorTemperatureSetPoint: " << onlineable << "\n"; //0

   Picam_CanSetParameterOnline(m_modelHandle, PicamParameter_AdcEMGain,&onlineable);
   std::cerr << "AdcEMGain: " << onlineable << "\n"; //1

   Picam_CanSetParameterOnline(m_modelHandle, PicamParameter_FrameRateCalculation,&onlineable);
   std::cerr << "FrameRateCalculation: " << onlineable << "\n"; //0

   std::cerr << "************************************************************\n";
*/

   //If not previously allocated, allocate a nice big buffer to play with
   pi64s newbuffsz = framesPerReadout*readoutStride*10; //Save room for 10 frames
   if( newbuffsz >  m_acqBuff.memory_size)
   {
      if(m_acqBuff.memory)
      {
         std::cerr << "Clearing\n";
         free(m_acqBuff.memory);
         m_acqBuff.memory = NULL;
         PicamAdvanced_SetAcquisitionBuffer(m_cameraHandle, NULL);
      }

      m_acqBuff.memory_size = newbuffsz;
      std::cerr << "m_acqBuff.memory_size: " << m_acqBuff.memory_size << "\n";
      m_acqBuff.memory = malloc(m_acqBuff.memory_size);

      error = PicamAdvanced_SetAcquisitionBuffer(m_cameraHandle, &m_acqBuff);
      if(error != PicamError_None)
      {
         log<software_error>({__FILE__, __LINE__, 0, error, PicamEnum2String(PicamEnumeratedType_Error, error)});
         state(stateCodes::ERROR);

         std::cerr << "-->" << PicamEnum2String(PicamEnumeratedType_Error, error) << "\n";
      }
   }

   //Start continuous acquisition
   if(setPicamParameter(PicamParameter_ReadoutCount,(pi64s) 0) < 0)
   {
      log<software_error>({__FILE__, __LINE__, "Error setting readouts=0"});
      state(stateCodes::ERROR);
      return -1;
   }

    error = Picam_StartAcquisition(m_cameraHandle);
    if(error != PicamError_None)
    {
       log<software_error>({__FILE__, __LINE__, 0, error, PicamEnum2String(PicamEnumeratedType_Error, error)});
       state(stateCodes::ERROR);

       return -1;
    }

    m_dataType = _DATATYPE_UINT16; //Where does this go?
    
    return 0;

}

inline
int picamCtrl::startAcquisition()
{
   return 0;
}

inline
int picamCtrl::acquireAndCheckValid()
{
   piint camTimeOut = 1000; //1 second keeps us responsive without busy-waiting too much

   PicamAcquisitionStatus status;

   PicamAvailableData available;

   PicamError error;
   error = Picam_WaitForAcquisitionUpdate(m_cameraHandle, camTimeOut, &available, &status);

   if(error == PicamError_TimeOutOccurred) 
   {
      return 1; //This sends it back to framegrabber to cehck for reconfig, etc.
   }

   clock_gettime(CLOCK_REALTIME, &m_currImageTimestamp);

   if(error != PicamError_None)
   {
      log<software_error>({__FILE__, __LINE__, 0, error, PicamEnum2String(PicamEnumeratedType_Error, error)});
      state(stateCodes::ERROR);

      return -1;
   }
      
   m_available.initial_readout = available.initial_readout;
   m_available.readout_count = available.readout_count;

   if(m_available.initial_readout == 0)
   {
      return 1;
   }

   return 0;

}

inline
int picamCtrl::loadImageIntoStream(void * dest)
{
   memcpy(dest, m_available.initial_readout, m_width*m_height*m_typeSize);

   return 0;
}

inline
int picamCtrl::reconfig()
{
   ///\todo clean this up.  Just need to wait on acquisition update the first time probably.
   
   PicamError error = Picam_StopAcquisition(m_cameraHandle);
   if(error != PicamError_None)
   {
      log<software_error>({__FILE__, __LINE__, 0, error, PicamEnum2String(PicamEnumeratedType_Error, error)});
      state(stateCodes::ERROR);

      return -1;
   }

   pibln running = true;

   error = Picam_IsAcquisitionRunning(m_cameraHandle, &running);

   while(running)
   {
      if(MagAOXAppT::m_powerState == 0) return 0;
      sleep(1);

      error = Picam_StopAcquisition(m_cameraHandle);

      if(error != PicamError_None)
      {
         log<software_error>({__FILE__, __LINE__, 0, error, PicamEnum2String(PicamEnumeratedType_Error, error)});
         state(stateCodes::ERROR);
         return -1;
      }

      piint camTimeOut = 1000;

      PicamAcquisitionStatus status;

      PicamAvailableData available;

      error = Picam_WaitForAcquisitionUpdate(m_cameraHandle, camTimeOut, &available, &status);

//       if(! status.running )
//       {
//          std::cerr << "Not running \n";
// 
//          std::cerr << "status.running: " << status.running << "\n";
//          std::cerr << "status.errors: " << status.errors << "\n";
//          std::cerr << "CameraFaulted: " << (int)(status.errors & PicamAcquisitionErrorsMask_CameraFaulted) << "\n";
//          std::cerr << "CannectionLost: " << (int)(status.errors & PicamAcquisitionErrorsMask_ConnectionLost) << "\n";
//          std::cerr << "DataLost: " << (int)(status.errors & PicamAcquisitionErrorsMask_DataLost) << "\n";
//          std::cerr << "DataNotArriving: " << (int)(status.errors & PicamAcquisitionErrorsMask_DataNotArriving) << "\n";
//          std::cerr << "None: " << (int)(status.errors & PicamAcquisitionErrorsMask_None) << "\n";
//          std::cerr << "ShutterOverheated: " << (int)(status.errors & PicamAcquisitionErrorsMask_ShutterOverheated) << "\n";
//          std::cerr << "status.readout_rate: " << status.readout_rate << "\n";
//       }

      error = Picam_IsAcquisitionRunning(m_cameraHandle, &running);
      if(error != PicamError_None)
      {
         log<software_error>({__FILE__, __LINE__, 0, error, PicamEnum2String(PicamEnumeratedType_Error, error)});
         state(stateCodes::ERROR);
         return -1;
      }
   }

   return 0;
}

INDI_NEWCALLBACK_DEFN(picamCtrl, m_indiP_adcspeed)(const pcf::IndiProperty &ipRecv)
{
   if(ipRecv.getName() != m_indiP_adcspeed.getName())
   {
      log<software_error>({__FILE__, __LINE__, "invalid indi property received"});
      return -1;
   }
   
   std::string newspeed;
   int newn = -1;
   size_t i;
   for(i=0; i< m_adcSpeeds.size(); ++i) 
   {
      if(!ipRecv.find(m_adcSpeeds[i])) continue;
      
      if(ipRecv[m_adcSpeeds[i]].getSwitchState() == pcf::IndiElement::On)
      {
         if(newspeed != "")
         {
            log<text_log>("More than one ADC speed selected", logPrio::LOG_ERROR);
            return -1;
         }
         
         newspeed = m_adcSpeeds[i];
         newn = i;         
      }
   }
   
   if(newspeed == "" || newn < 0)
   {
      return 0; //This is just a reset of current probably
   }
   
   if(newspeed == "00100_kHz" || newspeed == "01_MHz") return 0; //just silently ignore
   
   std::lock_guard<std::mutex> guard(m_indiMutex);
   return setAdcSpeed(m_adcSpeedValues[newn]);
   
}






}//namespace app
} //namespace MagAOX
#endif
