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
  * \brief Control of a Princeton Instruments OCAM2K EMCCD Camera.
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
class picamCtrl : public MagAOXApp<>, public dev::frameGrabber<picamCtrl>, public dev::dssShutter<picamCtrl>
{

   friend class dev::frameGrabber<picamCtrl>;
   friend class dev::dssShutter<picamCtrl>;

   typedef MagAOXApp<> MagAOXAppT;

protected:

   /** \name configurable parameters
     *@{
     */
   std::string m_serialNumber; ///< The camera's identifying serial number

   unsigned long m_powerOnWait {2}; ///< Time in sec to wait for camera boot after power on.


   float m_startupTemp {-55}; ///< The temperature to set after a power-on.

   ///@}

   int m_depth {0};
   float m_ccdTemp;
   float m_ccdTempSetpt; ///< The desired temperature.

   int m_adcSpeed {30}; ///< The ADC speed, 5, 10, 20, or 30.

   float m_expTimeSet {0}; ///< The exposure time, in seconds, as set by user.
   float m_expTimeMin {0}; ///<
   float m_expTimeMax {0};
   float m_expTimeDelta {0};


   float m_fpsSet {0}; ///< The commanded fps, as set by user.

   float m_fpsCurr {0}; ///< The current FPS as reported by the camera.

   int m_powerOnCounter {0}; ///< Counts numer of loops after power on, implements delay for camera bootup.

   std::string m_modeName;
   std::string m_nextMode;






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

   int setTemp(piflt temp);

   int adcSpeed();

   int setAdcSpeed(int newspd);

   int setExpTime(piflt exptime);

   int setFPS(piflt fps);


   //Framegrabber interface:
   int configureAcquisition();
   int startAcquisition();
   int acquireAndCheckValid();
   int loadImageIntoStream(void * dest);
   int reconfig();









   //INDI:
protected:
   //declare our properties
   pcf::IndiProperty m_indiP_ccdTemp;
   pcf::IndiProperty m_indiP_ccdTempLock;

   pcf::IndiProperty m_indiP_adcspeed;

//   pcf::IndiProperty m_indiP_mode;

   pcf::IndiProperty m_indiP_exptime;
   pcf::IndiProperty m_indiP_fps;

public:
   INDI_NEWCALLBACK_DECL(picamCtrl, m_indiP_ccdTemp);
   INDI_NEWCALLBACK_DECL(picamCtrl, m_indiP_adcspeed);

//   INDI_NEWCALLBACK_DECL(picamCtrl, m_indiP_mode);
   INDI_NEWCALLBACK_DECL(picamCtrl, m_indiP_exptime);
   INDI_NEWCALLBACK_DECL(picamCtrl, m_indiP_fps);


};

inline
picamCtrl::picamCtrl() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   m_powerMgtEnabled = true;

   m_acqBuff.memory_size = 0;
   m_acqBuff.memory = 0;

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

   config.add("camera.powerOnWait", "", "camera.powerOnWait", argType::Required, "camera", "powerOnWait", false, "int", "Time after power-on to begin attempting connections [sec].  Default is 10 sec.");
   config.add("camera.startupTemp", "", "camera.startupTemp", argType::Required, "camera", "startupTemp", false, "float", "The temperature setpoint to set after a power-on [C].  Default is -55 C.");

   dev::frameGrabber<picamCtrl>::setupConfig(config);
   dev::dssShutter<picamCtrl>::setupConfig(config);
}

inline
void picamCtrl::loadConfig()
{

   config(m_serialNumber, "camera.serialNumber");
   config(m_powerOnWait, "camera.powerOnWait");
   config(m_startupTemp, "camera.startupTemp");

   dev::frameGrabber<picamCtrl>::loadConfig(config);
   dev::dssShutter<picamCtrl>::loadConfig(config);

}

inline
int picamCtrl::appStartup()
{
   // set up the  INDI properties
   REG_INDI_NEWPROP(m_indiP_ccdTemp, "ccdtemp", pcf::IndiProperty::Number);
   m_indiP_ccdTemp.add (pcf::IndiElement("current"));
   m_indiP_ccdTemp["current"].set(0);
   m_indiP_ccdTemp.add (pcf::IndiElement("target"));

   REG_INDI_NEWPROP(m_indiP_adcspeed, "adcspeed", pcf::IndiProperty::Number);
   m_indiP_adcspeed.add (pcf::IndiElement("current"));
   m_indiP_adcspeed.add (pcf::IndiElement("target"));

   REG_INDI_NEWPROP_NOCB(m_indiP_ccdTempLock, "ccdtempctrl", pcf::IndiProperty::Text);
   m_indiP_ccdTempLock.add (pcf::IndiElement("state"));

//    REG_INDI_NEWPROP(m_indiP_mode, "mode", pcf::IndiProperty::Text);
//    m_indiP_mode.add (pcf::IndiElement("current"));
//    m_indiP_mode.add (pcf::IndiElement("target"));

   REG_INDI_NEWPROP(m_indiP_fps, "fps", pcf::IndiProperty::Number);
   m_indiP_fps.add (pcf::IndiElement("current"));
   m_indiP_fps["current"].set(0);
   m_indiP_fps.add (pcf::IndiElement("set"));
   m_indiP_fps.add (pcf::IndiElement("target"));
   m_indiP_fps.add (pcf::IndiElement("measured"));

   REG_INDI_NEWPROP(m_indiP_exptime, "exptime", pcf::IndiProperty::Number);
   m_indiP_exptime.add (pcf::IndiElement("current"));
   m_indiP_exptime["current"].set(m_expTimeSet);
   m_indiP_exptime.add (pcf::IndiElement("target"));
   m_indiP_exptime["target"].set(0);

   m_indiP_exptime.add (pcf::IndiElement("min"));
   m_indiP_exptime["min"].set(m_expTimeMin);
   m_indiP_exptime.add (pcf::IndiElement("max"));
   m_indiP_exptime["max"].set(m_expTimeMax);
   m_indiP_exptime.add (pcf::IndiElement("delta"));
   m_indiP_exptime["delta"].set(m_expTimeDelta);

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

   if( state() == stateCodes::POWERON )
   {
      if(m_powerOnCounter*m_loopPause > ((double) m_powerOnWait)*1e9)
      {
         //=================================


         state(stateCodes::NOTCONNECTED);
         m_reconfig = true; //Trigger a f.g. thread reconfig.
         m_powerOnCounter = 0;
      }
      else
      {
         ++m_powerOnCounter;
         return 0;
      }
   }

   if( state() == stateCodes::NOTCONNECTED || state() == stateCodes::NODEVICE || state() == stateCodes::ERROR)
   {
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

      if( setTemp(m_startupTemp) < 0 )
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

      updateIfChanged(m_indiP_fps, "current", m_fpsCurr);
   }

   //Fall through check?

   return 0;

}

inline
int picamCtrl::onPowerOff()
{
   m_powerOnCounter = 0;

   std::lock_guard<std::mutex> lock(m_indiMutex);

   updateIfChanged(m_indiP_ccdTemp, "current", -999);
   updateIfChanged(m_indiP_ccdTemp, "target", -999);
   updateIfChanged(m_indiP_ccdTempLock, "state", std::string(""));

//    updateIfChanged(m_indiP_mode, "current", std::string(""));
//    updateIfChanged(m_indiP_mode, "target", std::string(""));

   updateIfChanged(m_indiP_fps, "current", 0);
   updateIfChanged(m_indiP_fps, "target", 0);
   updateIfChanged(m_indiP_fps, "measured", 0);

   if(m_cameraHandle)
   {
      Picam_CloseCamera(m_cameraHandle);
      m_cameraHandle = 0;
   }

   Picam_UninitializeLibrary();

   ///\todo error check these base class fxns.
   dssShutter<picamCtrl>::onPowerOff();

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

   //PicamSensorTemperatureStatus
   piint status;

   if(getPicamParameter( status, PicamParameter_SensorTemperatureStatus ) < 0)
   {
      if(MagAOXAppT::m_powerState == 0) return 0;

      log<software_error>({__FILE__, __LINE__});
      state(stateCodes::ERROR);
      return -1;
   }

   std::string lockstr = "unknown";
   if(status == 1) lockstr = "unlocked";
   else if(status == 2) lockstr = "locked";
   else if(status == 3) lockstr = "faulted";

   updateIfChanged(m_indiP_ccdTemp, "current", currTemperature);

   updateIfChanged(m_indiP_ccdTempLock, "state", lockstr);

   return 0;

}

inline
int picamCtrl::setTemp(piflt temp)
{
   m_ccdTempSetpt = temp;

   m_reconfig = true;

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

   updateIfChanged(m_indiP_adcspeed, "target", m_adcSpeed);

   m_reconfig = true;

   return 0;
}

inline
int picamCtrl::setExpTime(piflt exptime)
{

   long intexptime = exptime * 1000 * 10000 + 0.5;
   exptime = ((double)intexptime)/10000;

   int rv;
   if(state() == stateCodes::OPERATING)
   {
      std::cerr << "setting online...\n";
      rv = setPicamParameterOnline(m_modelHandle, PicamParameter_ExposureTime, exptime);
   }
   else
   {
      std::cerr << "setting offline...\n";
      rv = setPicamParameter(m_modelHandle, PicamParameter_ExposureTime, exptime);
   }

   if(rv < 0)
   {
      log<software_error>({__FILE__, __LINE__, "Error setting exposure time"});
      return -1;
   }

   log<text_log>( "Set exposure time: " + std::to_string(exptime/1000.0) + " sec");

   return 0;
}


inline
int picamCtrl::setFPS(piflt fps)
{
   return setExpTime(1.0/fps);
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

   updateIfChanged(m_indiP_ccdTemp, "target", m_ccdTempSetpt);

   log<text_log>( "Set temperature set point: " + std::to_string(m_ccdTempSetpt) + " C");

   //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
   //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
   // Speed
   //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
   //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

   if( setPicamParameter(m_modelHandle, PicamParameter_AdcSpeed, m_adcSpeed) < 0)
   {
      log<software_error>({__FILE__, __LINE__, "Error setting ADC Speed"});
      state(stateCodes::ERROR);
      return -1;
   }

   updateIfChanged(m_indiP_adcspeed, "current", m_adcSpeed);
   updateIfChanged(m_indiP_adcspeed, "target", std::string(""));

   log<text_log>( "ADC Speed: " + std::to_string(m_adcSpeed) + " MHz");

   //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
   //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
   // Dimensions
   //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
   //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

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
   PicamError error = Picam_GetParameterRoisValue( m_cameraHandle, PicamParameter_Rois, &rois );
   if( error != PicamError_None )
   {
      log<software_error>({__FILE__, __LINE__, 0, error, PicamEnum2String(PicamEnumeratedType_Error, error)});
      state(stateCodes::ERROR);
      return -1;
   }
   m_xbinning = rois->roi_array[0].x_binning;
   m_ybinning = rois->roi_array[0].y_binning;
   m_width  = rois->roi_array[0].width  / rois->roi_array[0].x_binning;
   m_height = rois->roi_array[0].height / rois->roi_array[0].y_binning;
   Picam_DestroyRois( rois );


   //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
   //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
   // Exposure Time and Frame Rate
   //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
   //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

   piflt exptime;
   if(getPicamParameter(exptime, PicamParameter_ExposureTime) < 0)
   {
      std::cerr << "could not get Exposuretime\n";
   }
   else
   {
      m_expTimeSet = exptime/1000.0;
      updateIfChanged(m_indiP_exptime, "current", m_expTimeSet);
      std::cerr << "ExposureTime is: " << exptime << "\n";
   }

   const PicamRangeConstraint * constraint_array;
   piint constraint_count;
   PicamAdvanced_GetParameterRangeConstraints( m_modelHandle, PicamParameter_ExposureTime, &constraint_array, &constraint_count);

   if(constraint_count != 1)
   {
      std::cerr << "Constraint count is not 1: " << (int) constraint_count << " constraints" << std::endl;

      log<text_log>("Constraint count is not 1: " + std::to_string(constraint_count) + " constraints",logPrio::LOG_ERROR);

      updateIfChanged(m_indiP_exptime, "min", std::string("UNK"));
      updateIfChanged(m_indiP_exptime, "max", std::string("UNK"));
      updateIfChanged(m_indiP_exptime, "delta", std::string("UNK"));
   }
   else
   {
      m_expTimeMin = constraint_array[0].minimum;
      updateIfChanged(m_indiP_exptime, "min", m_expTimeMin);
      m_expTimeMax = constraint_array[0].maximum;
      updateIfChanged(m_indiP_exptime, "max", m_expTimeMax);
      m_expTimeDelta = constraint_array[0].increment;
      updateIfChanged(m_indiP_exptime, "delta", m_expTimeDelta);
   }


   piflt FrameRateCalculation;
   if(getPicamParameter(FrameRateCalculation, PicamParameter_FrameRateCalculation) < 0)
   {
      std::cerr << "could not get FrameRateCalculation\n";
   }

   //std::cerr << "FrameRateCalculation is: " << FrameRateCalculation << "\n";

   //std::cerr << "Available readouts: " << available.readout_count << "\n";
   std::cerr << "Frame rate: " << FrameRateCalculation << "\n";
   m_fpsSet = FrameRateCalculation;
   updateIfChanged(m_indiP_fps, "set", m_fpsSet);












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

//          Picam_CanSetParameterOnline(m_modelHandle, PicamParameter_ ,&onlineable);
//          std::cerr << ": " << onlineable << "\n";


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

    m_dataType = _DATATYPE_INT16; //Where does this go?
    std::cerr << "Acquisition Started\n";
    sleep(1);
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

   piint camTimeOut = 1000;

   PicamAcquisitionStatus status;

   PicamAvailableData available;

   PicamError error;
   error = Picam_WaitForAcquisitionUpdate(m_cameraHandle, camTimeOut, &available, &status);



   if(! status.running )
   {
      std::cerr << "Not running \n";

      std::cerr << "status.running: " << status.running << "\n";
      std::cerr << "status.errors: " << status.errors << "\n";
      std::cerr << "CameraFaulted: " << (int)(status.errors & PicamAcquisitionErrorsMask_CameraFaulted) << "\n";
      std::cerr << "CannectionLost: " << (int)(status.errors & PicamAcquisitionErrorsMask_ConnectionLost) << "\n";
      std::cerr << "DataLost: " << (int)(status.errors & PicamAcquisitionErrorsMask_DataLost) << "\n";
      std::cerr << "DataNotArriving: " << (int)(status.errors & PicamAcquisitionErrorsMask_DataNotArriving) << "\n";
      std::cerr << "None: " << (int)(status.errors & PicamAcquisitionErrorsMask_None) << "\n";
      std::cerr << "ShutterOverheated: " << (int)(status.errors & PicamAcquisitionErrorsMask_ShutterOverheated) << "\n";
      std::cerr << "status.readout_rate: " << status.readout_rate << "\n";

      error = Picam_StartAcquisition(m_cameraHandle);
      if(error != PicamError_None)
      {
         log<software_error>({__FILE__, __LINE__, 0, error, PicamEnum2String(PicamEnumeratedType_Error, error)});
         state(stateCodes::ERROR);

         return -1;
      }
   }


   clock_gettime(CLOCK_REALTIME, &m_currImageTimestamp);

   m_available.initial_readout = available.initial_readout;
   m_available.readout_count = available.readout_count;

   if(error == PicamError_TimeOutOccurred)
   {
      return 1;
   }
   else if(error != PicamError_None)
   {
      log<software_error>({__FILE__, __LINE__, 0, error, PicamEnum2String(PicamEnumeratedType_Error, error)});
      state(stateCodes::ERROR);

      return -1;
   }
   if(m_available.initial_readout == 0)
   {
      return 1;
   }

   m_fpsCurr = status.readout_rate;

   return 0;

   piflt FrameRateCalculation;
   if(getPicamParameter(FrameRateCalculation, PicamParameter_FrameRateCalculation) < 0)
   {
      std::cerr << "could not get FrameRateCalculation\n";
   }

   //std::cerr << "FrameRateCalculation is: " << FrameRateCalculation << "\n";

   //std::cerr << "Available readouts: " << available.readout_count << "\n";
   std::cerr << "Frame rate: " << status.readout_rate <<" " << FrameRateCalculation << "\r";
   if(available.readout_count <= 0) return 1;

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
   std::cerr << "In reconfig " << std::endl;

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
      std::cerr << "running..." << std::endl;

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



   if(! status.running )
   {
      std::cerr << "Not running \n";

      std::cerr << "status.running: " << status.running << "\n";
      std::cerr << "status.errors: " << status.errors << "\n";
      std::cerr << "CameraFaulted: " << (int)(status.errors & PicamAcquisitionErrorsMask_CameraFaulted) << "\n";
      std::cerr << "CannectionLost: " << (int)(status.errors & PicamAcquisitionErrorsMask_ConnectionLost) << "\n";
      std::cerr << "DataLost: " << (int)(status.errors & PicamAcquisitionErrorsMask_DataLost) << "\n";
      std::cerr << "DataNotArriving: " << (int)(status.errors & PicamAcquisitionErrorsMask_DataNotArriving) << "\n";
      std::cerr << "None: " << (int)(status.errors & PicamAcquisitionErrorsMask_None) << "\n";
      std::cerr << "ShutterOverheated: " << (int)(status.errors & PicamAcquisitionErrorsMask_ShutterOverheated) << "\n";
      std::cerr << "status.readout_rate: " << status.readout_rate << "\n";


   }

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









INDI_NEWCALLBACK_DEFN(picamCtrl, m_indiP_ccdTemp)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_ccdTemp.getName())
   {
      float current = 99, target = 99;

      try
      {
         current = ipRecv["current"].get<float>();
      }
      catch(...){}

      try
      {
         target = ipRecv["target"].get<float>();
      }
      catch(...){}


      //Check if target is empty
      if( target == 99 ) target = current;

      //Now check if it's valid?
      ///\todo implement more configurable max-set-able temperature
      if( target > 30 ) return 0;


      //Lock the mutex, waiting if necessary
      std::unique_lock<std::mutex> lock(m_indiMutex);

      return setTemp(target);
   }
   return -1;
}

INDI_NEWCALLBACK_DEFN(picamCtrl, m_indiP_adcspeed)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_adcspeed.getName())
   {
      int current = 0, target = 0;

      if(ipRecv.find("current"))
      {
         current = ipRecv["current"].get<int>();
      }

      if(ipRecv.find("target"))
      {
         target = ipRecv["target"].get<int>();
      }


      //Check if target is empty
      if( target == 0 ) target = current;

      if( target == 0 ) return 0;

      std::unique_lock<std::mutex> lock(m_indiMutex);
      return setAdcSpeed(target);
   }
   return -1;
}

#if 0
INDI_NEWCALLBACK_DEFN(picamCtrl, m_indiP_mode)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_mode.getName())
   {
      std::cerr << "New mode\n";
      std::string current;
      std::string target;
      try
      {
         current = ipRecv["current"].get();
      }
      catch(...)
      {
         current = "";
      }

      try
      {
         target = ipRecv["target"].get();
      }
      catch(...)
      {
         target = "";
      }

      if(target == "") target = current;

      if(m_cameraModes.count(target) == 0 )
      {
         return log<text_log, -1>("Unrecognized mode requested: " + target, logPrio::LOG_ERROR);
      }

      updateIfChanged(m_indiP_mode, "target", target);

      //Now signal the f.g. thread to reconfigure
      m_nextMode = target;
      m_reconfig = true;

      return 0;
   }
   return -1;
}
#endif

INDI_NEWCALLBACK_DEFN(picamCtrl, m_indiP_fps)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_fps.getName())
   {
      float current = -99, target = -99;

      try
      {
         current = ipRecv["current"].get<float>();
      }
      catch(...){}

      try
      {
         target = ipRecv["target"].get<float>();
      }
      catch(...){}

      if(target == -99) target = current;

      if(target <= 0) return 0;

      //Lock the mutex, waiting if necessary
      std::unique_lock<std::mutex> lock(m_indiMutex);

      m_ccdTempSetpt = target;
      updateIfChanged(m_indiP_fps, "target", target);

      return setFPS(target);

   }
   return -1;
}

INDI_NEWCALLBACK_DEFN(picamCtrl, m_indiP_exptime)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_exptime.getName())
   {
      float current = -99, target = -99;

      try
      {
         current = ipRecv["current"].get<float>();
      }
      catch(...){}

      try
      {
         target = ipRecv["target"].get<float>();
      }
      catch(...){}

      if(target == -99) target = current;

      if(target <= 0) return 0;

      //Lock the mutex, waiting if necessary
      std::unique_lock<std::mutex> lock(m_indiMutex);

      updateIfChanged(m_indiP_exptime, "target", target);

      return setExpTime(target);

   }
   return -1;
}
}//namespace app
} //namespace MagAOX
#endif
