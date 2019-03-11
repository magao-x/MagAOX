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

#include <picam.h>

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"


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

#define CAMCTRL_E_NOCONFIGS (-10)
   
///\todo create cameraConfig in libMagAOX
struct cameraConfig 
{
   std::string m_configFile;
   std::string m_serialCommand;
   unsigned m_binning {0};
   unsigned m_sizeX {0};
   unsigned m_sizeY {0};
   float m_maxFPS {0};
};

typedef std::unordered_map<std::string, cameraConfig> cameraConfigMap;

inline
int loadCameraConfig( cameraConfigMap & ccmap,
                      mx::app::appConfigurator & config 
                    )
{
   std::vector<std::string> sections;

   config.unusedSections(sections);

   if( sections.size() == 0 )
   {
      return CAMCTRL_E_NOCONFIGS;
   }
   
   for(size_t i=0; i< sections.size(); ++i)
   {
      bool fileset = config.isSetUnused(mx::app::iniFile::makeKey(sections[i], "configFile" ));
      /*bool binset = config.isSetUnused(mx::app::iniFile::makeKey(sections[i], "binning" ));
      bool sizeXset = config.isSetUnused(mx::app::iniFile::makeKey(sections[i], "sizeX" ));
      bool sizeYset = config.isSetUnused(mx::app::iniFile::makeKey(sections[i], "sizeY" ));
      bool maxfpsset = config.isSetUnused(mx::app::iniFile::makeKey(sections[i], "maxFPS" ));
      */
      
      //The configuration file tells us most things for EDT, so it's our current requirement. 
      if( !fileset ) continue;
      
      std::string configFile;
      config.configUnused(configFile, mx::app::iniFile::makeKey(sections[i], "configFile" ));
      
      std::string serialCommand;
      config.configUnused(serialCommand, mx::app::iniFile::makeKey(sections[i], "serialCommand" ));
      
      unsigned binning = 0;
      config.configUnused(binning, mx::app::iniFile::makeKey(sections[i], "binning" ));
      
      unsigned sizeX = 0;
      config.configUnused(sizeX, mx::app::iniFile::makeKey(sections[i], "sizeX" ));
      
      unsigned sizeY = 0;
      config.configUnused(sizeY, mx::app::iniFile::makeKey(sections[i], "sizeY" ));
      
      float maxFPS = 0;
      config.configUnused(maxFPS, mx::app::iniFile::makeKey(sections[i], "maxFPS" ));
      
      ccmap[sections[i]] = cameraConfig({configFile, serialCommand, binning, sizeX, sizeY, maxFPS});
   }
   
   return 0;
}

/** \defgroup picamCtrl Princeton Instruments EMCCD Camera
  * \brief Control of a Princeton Instruments OCAM2K EMCCD Camera.
  *
  *  <a href="../apps_html/page_module_picamCtrl.html">Application Documentation</a>
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
class picamCtrl : public MagAOXApp<>, public dev::ioDevice
{

protected:

   /** \name configurable parameters 
     *@{
     */ 
   std::string m_serialNumber; ///< The camera's identifying serial number
   
   unsigned long m_powerOnWait {2}; ///< Time in sec to wait for camera boot after power on.
   
   cameraConfigMap m_cameraModes; ///< Map holding the possible camera mode configurations
   
   float m_startupTemp {99}; ///< The temperature to set after a power-on.
   
   ///@}

   float m_expTime {0}; ///< The exposure time, in seconds, as returned by the camera.
   float m_fpsSet {0}; ///< The commanded fps, as returned by the camera

   int m_powerOnCounter {0}; ///< Counts numer of loops after power on, implements delay for camera bootup.
   
   std::string m_modeName;
   std::string m_nextMode;
   
   int m_width {0}; ///< The width of the image according to the framegrabber, not necessarily the true image width.
   int m_height {0}; ///< The height of the image frame according the framegrabber, not necessarily the true image height.
   int m_depth {0}; ///< The pixel bit depth according to the framegrabber
   std::string m_cameraType; ///< The camera type according to the framegrabber
          
   long m_currImageNumber {-1};
   double m_currImageTimestamp {0};
   double m_currImageDMATimestamp {0};
      
   long m_lastImageNumber {-1};
      
   long m_firstGoodImageNumber {-1};
   double m_firstGoodImageTimestamp {0};
   double m_firstGoodImageDMATimestamp {0};
   
   long m_framesSkipped = 0;
      
   bool m_resetFPS {false};
   bool m_reconfig {false};
   

   PicamHandle m_cameraHandle;
   bool m_cameraConnected;
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

   int connect();
   
   int getAcquisitionState();
   
   int getTemps();
   int setTemp(float temp);

   int getExpTime();
   int setExpTime(float exptime);
   
   int getFPS();
   int setFPS(float fps);
   
protected:
   
   int m_fgThreadPrio {1}; ///< Priority of the framegrabber thread, should normally be > 00.

   std::thread m_fgThread; ///< A separate thread for the actual framegrabbings

   ///Thread starter, called by fgThreadStart on thread construction.  Calls fgThreadExec.
   static void _fgThreadStart( picamCtrl * o /**< [in] a pointer to an picamCtrl instance (normally this) */);

   /// Start the log capture.
   int fgThreadStart();

   /// Execute the log capture.
   void fgThreadExec();

   
   //INDI:
protected:
   //declare our properties
   pcf::IndiProperty m_indiP_ccdTemp;
   pcf::IndiProperty m_indiP_ccdTempLock;
   
//   pcf::IndiProperty m_indiP_mode;
   
   pcf::IndiProperty m_indiP_exptime;
   pcf::IndiProperty m_indiP_fps;

public:
   INDI_NEWCALLBACK_DECL(picamCtrl, m_indiP_ccdTemp);
//   INDI_NEWCALLBACK_DECL(picamCtrl, m_indiP_mode);
   INDI_NEWCALLBACK_DECL(picamCtrl, m_indiP_exptime);
   INDI_NEWCALLBACK_DECL(picamCtrl, m_indiP_fps);
   

};

inline
picamCtrl::picamCtrl() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   m_powerMgtEnabled = true;
   
   return;
}

inline
picamCtrl::~picamCtrl() noexcept
{
   return;
}

inline
void picamCtrl::setupConfig()
{
   config.add("framegrabber.threadPrio", "", "framegrabber.threadPrio", argType::Required, "framegrabber", "threadPrio", false, "int", "The real-time priority of the fraemgrabber thread.");
   
   
   config.add("camera.serialNumber", "", "camera.serialNumber", argType::Required, "camera", "serialNumber", false, "int", "The identifying serial number of the camera.");
   
   config.add("camera.powerOnWait", "", "camera.powerOnWait", argType::Required, "camera", "powerOnWait", false, "int", "Time after power-on to begin attempting connections [sec].  Default is 10 sec.");
   config.add("camera.startupTemp", "", "camera.startupTemp", argType::Required, "camera", "startupTemp", false, "float", "The temperature setpoint to set after a power-on [C].  Default is -55 C.");
   
   dev::ioDevice::setupConfig(config);
}



inline
void picamCtrl::loadConfig()
{
   config(m_fgThreadPrio, "framegrabber.threadPrio");
   
   config(m_serialNumber, "camera.serialNumber");
   config(m_powerOnWait, "camera.powerOnWait");
   config(m_startupTemp, "camera.startupTemp");
   
   int rv = loadCameraConfig(m_cameraModes, config);
   
   if(rv < 0)
   {
      if(rv == CAMCTRL_E_NOCONFIGS)
      {
         log<text_log>("No camera configurations found.", logPrio::LOG_CRITICAL);
      }
      
      m_shutdown = true;
   }
   
   m_readTimeout = 1000;
   m_writeTimeout = 1000;
   dev::ioDevice::loadConfig(config);
}



inline
int picamCtrl::appStartup()
{
   // set up the  INDI properties
   REG_INDI_NEWPROP(m_indiP_ccdTemp, "ccdtemp", pcf::IndiProperty::Number);
   m_indiP_ccdTemp.add (pcf::IndiElement("current"));
   m_indiP_ccdTemp["current"].set(0);
   m_indiP_ccdTemp.add (pcf::IndiElement("target"));
   
   REG_INDI_NEWPROP_NOCB(m_indiP_ccdTempLock, "ccdtempctrl", pcf::IndiProperty::Text);
   m_indiP_ccdTempLock.add (pcf::IndiElement("state"));
   
//    REG_INDI_NEWPROP(m_indiP_mode, "mode", pcf::IndiProperty::Text);
//    m_indiP_mode.add (pcf::IndiElement("current"));
//    m_indiP_mode.add (pcf::IndiElement("target"));

   REG_INDI_NEWPROP(m_indiP_fps, "fps", pcf::IndiProperty::Number);
   m_indiP_fps.add (pcf::IndiElement("current"));
   m_indiP_fps["current"].set(0);
   m_indiP_fps.add (pcf::IndiElement("target"));
   m_indiP_fps.add (pcf::IndiElement("measured"));
   
   REG_INDI_NEWPROP(m_indiP_exptime, "exptime", pcf::IndiProperty::Number);
   m_indiP_exptime.add (pcf::IndiElement("current"));
   m_indiP_exptime["current"].set(0);
   m_indiP_exptime.add (pcf::IndiElement("target"));


   
   //=================================
   // Do camera configuration here
   //=================================
   Picam_InitializeLibrary();
   
   if(fgThreadStart() < 0)
   {
      log<software_error>({__FILE__, __LINE__});
      return -1;
   }
   
   return 0;

}



inline
int picamCtrl::connect()
{
   PicamError error;
   PicamCameraID * id_array;
   piint id_count;

   Picam_GetAvailableCameraIDs((const PicamCameraID **) &id_array, &id_count);
   
   if(id_count == 0)
   {
      state(stateCodes::NODEVICE);
      if(!stateLogged())
      {
         log<text_log>("no P.I. Cameras available.");
      }
      return 0;
   }
   
   for(int i=0; i< id_count; ++i)
   {
      if( std::string(id_array[i].serial_number) == m_serialNumber )
      {  
         error = Picam_OpenCamera(&id_array[i], &m_cameraHandle);
         if(error == PicamError_None) 
         {
            m_cameraConnected = true;
            m_cameraName = id_array[i].sensor_name;
            m_cameraModel = PicamEnum2String(PicamEnumeratedType_Model, id_array[i].model);
            
            state(stateCodes::CONNECTED);
            log<text_log>("Connected to " + m_cameraName + " [S/N " + m_serialNumber + "]");
            
            return 0;
         }
         else
         {
            state(stateCodes::ERROR);
            if(!stateLogged())
            {
               log<software_error>({__FILE__,__LINE__, 0, error, "Error connecting to camera."});
            }
            
            return -1;
         }
      }
   }
   
   state(stateCodes::NODEVICE);
   if(!stateLogged())
   {
      log<text_log>("Camera not found in available ids.");
   }
   return 0;
}

inline
int picamCtrl::appLogic()
{
   //first do a join check to see if other threads have exited.
   if(pthread_tryjoin_np(m_fgThread.native_handle(),0) == 0)
   {
      log<software_error>({__FILE__, __LINE__, "framegrabber thread has exited"});
      
      return -1;
   }
   
   if( state() == stateCodes::POWERON )
   {
      if(m_powerOnCounter*m_loopPause > ((double) m_powerOnWait)*1e9)
      {
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

   if( state() == stateCodes::NOTCONNECTED || state() == stateCodes::ERROR)
   {
      //Might have gotten here because of a power off.
      if(m_powerState == 0) return 0;
      
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
   }

   if( state() == stateCodes::READY || state() == stateCodes::OPERATING )
   {
      //Get a lock if we can
      std::unique_lock<std::mutex> lock(m_indiMutex, std::try_to_lock);

      //but don't wait for it, just go back around.
      if(!lock.owns_lock()) return 0;
      
      if(getTemps() < 0)
      {
         if(m_powerState == 0) return 0;
         
         state(stateCodes::ERROR);
         return 0;
      }

      if(getFPS() < 0)
      {
         if(m_powerState == 0) return 0;
         
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
   
   return 0;
}

inline
int picamCtrl::whilePowerOff()
{
   return 0;
}

inline
int picamCtrl::appShutdown()
{
   if(m_fgThread.joinable())
   {
      m_fgThread.join();
   }
   return 0;
}


inline
int picamCtrl::getAcquisitionState()
{
   pibln running = false;
   
   PicamError error = Picam_IsAcquisitionRunning(m_cameraHandle, &running);

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
   
   PicamError error = Picam_ReadParameterFloatingPointValue( m_cameraHandle, PicamParameter_SensorTemperatureReading, &currTemperature );

   if(error != PicamError_None)
   {
      log<software_error>({__FILE__, __LINE__, 0, error, PicamEnum2String(PicamEnumeratedType_Error, error)});
      state(stateCodes::ERROR);
      return -1;
   }
   
   piflt setTemperature;
   
   error = Picam_ReadParameterFloatingPointValue( m_cameraHandle, PicamParameter_SensorTemperatureSetPoint, &setTemperature );

   if(error != PicamError_None)
   {
      log<software_error>({__FILE__, __LINE__, 0, error, PicamEnum2String(PicamEnumeratedType_Error, error)});
      state(stateCodes::ERROR);
      return -1;
   }
   
   PicamSensorTemperatureStatus status;
   error = Picam_ReadParameterIntegerValue( m_cameraHandle, PicamParameter_SensorTemperatureStatus, reinterpret_cast<piint*>( &status ) );
   if(error != PicamError_None)
   {
      log<software_error>({__FILE__, __LINE__, 0, error, PicamEnum2String(PicamEnumeratedType_Error, error)});
      state(stateCodes::ERROR);
      return -1;
   }
   
   std::string lockstr = "unknown";
   if(status == 1) lockstr = "unlocked";
   else if(status == 2) lockstr = "locked";
   else if(status == 3) lockstr = "faulted";
      
   updateIfChanged(m_indiP_ccdTemp, "current", currTemperature);
   updateIfChanged(m_indiP_ccdTemp, "target", setTemperature);
   
   updateIfChanged(m_indiP_ccdTempLock, "state", lockstr);
   
   return 0;

}

inline
int picamCtrl::setTemp(float temp)
{
   PicamError error = Picam_SetParameterFloatingPointValue( m_cameraHandle, PicamParameter_SensorTemperatureSetPoint, temp );
   if(error != PicamError_None)
   {
      log<software_error>({__FILE__, __LINE__, 0, error, PicamEnum2String(PicamEnumeratedType_Error, error)});
      state(stateCodes::ERROR);
      return -1;
   }
   
   const PicamParameter* failed_parameters;
   piint failed_parameters_count;
    
   error = Picam_CommitParameters( m_cameraHandle, &failed_parameters, &failed_parameters_count );
   if(error != PicamError_None)
   {
      log<software_error>({__FILE__, __LINE__, 0, error, PicamEnum2String(PicamEnumeratedType_Error, error)});
      state(stateCodes::ERROR);
      return -1;
   }
   
   for( int i=0; i< failed_parameters_count; ++i)
   {
      if( failed_parameters[i] ==  PicamParameter_SensorTemperatureSetPoint)
      {
         Picam_DestroyParameters( failed_parameters );
         return log<text_log,-1>( "Camera refused new set point: " + std::to_string(temp) + " C");
      }
   }
   
   Picam_DestroyParameters( failed_parameters );
   
   log<text_log>( "Set temperature set point: " + std::to_string(temp) + " C");
   
   return 0;

}

inline
int picamCtrl::getExpTime()
{
   piflt exptime;
   
   PicamError error = Picam_ReadParameterFloatingPointValue( m_cameraHandle, PicamParameter_ExposureTime, &exptime );

   if(error != PicamError_None)
   {
      log<software_error>({__FILE__, __LINE__, 0, error, PicamEnum2String(PicamEnumeratedType_Error, error)});
      state(stateCodes::ERROR);
      return -1;
   }
   

   m_expTime = exptime/1000.0;
   
   m_fpsSet = 1.0/m_expTime;
   
   updateIfChanged(m_indiP_exptime, "current", m_expTime);
   updateIfChanged(m_indiP_fps, "current", m_fpsSet);
   

   float target_exptime = m_indiP_exptime["target"].get<float>();
   if( fabs(m_expTime - target_exptime) < 1e-5) updateIfChanged(m_indiP_exptime, "target", 0);
   
   float target_fps = m_indiP_fps["target"].get<float>();
   if( fabs(m_fpsSet - target_fps) < 1e-5) updateIfChanged(m_indiP_fps, "target", 0);
   
   return 0;

}

inline
int picamCtrl::setExpTime(float exptime)
{
   PicamError error = Picam_SetParameterFloatingPointValue( m_cameraHandle, PicamParameter_ExposureTime, exptime*1000.0 );
   if(error != PicamError_None)
   {
      log<software_error>({__FILE__, __LINE__, 0, error, PicamEnum2String(PicamEnumeratedType_Error, error)});
      state(stateCodes::ERROR);
      return -1;
   }
   
   const PicamParameter* failed_parameters;
   piint failed_parameters_count;
    
   error = Picam_CommitParameters( m_cameraHandle, &failed_parameters, &failed_parameters_count );
   if(error != PicamError_None)
   {
      log<software_error>({__FILE__, __LINE__, 0, error, PicamEnum2String(PicamEnumeratedType_Error, error)});
      state(stateCodes::ERROR);
      return -1;
   }
   
   for( int i=0; i< failed_parameters_count; ++i)
   {
      if( failed_parameters[i] ==  PicamParameter_ExposureTime)
      {
         Picam_DestroyParameters( failed_parameters );
         return log<text_log,-1>( "Camera refused new exposure time: " + std::to_string(exptime) + " sec");
      }
   }
   
   Picam_DestroyParameters( failed_parameters );
   
   log<text_log>( "Set exposure time: " + std::to_string(exptime) + " sec");
   
   return 0;
}

inline
int picamCtrl::getFPS()
{
   return getExpTime();
}

inline
int picamCtrl::setFPS(float fps)
{
   return setExpTime(1.0/fps);
}

inline
void picamCtrl::_fgThreadStart( picamCtrl * o)
{
   o->fgThreadExec();
}

inline
int picamCtrl::fgThreadStart()
{
   try
   {
      m_fgThread  = std::thread( _fgThreadStart, this);
   }
   catch( const std::exception & e )
   {
      log<software_error>({__FILE__,__LINE__, std::string("Exception on framegrabber thread start: ") + e.what()});
      return -1;
   }
   catch( ... )
   {
      log<software_error>({__FILE__,__LINE__, "Unkown exception on framegrabber thread start"});
      return -1;
   }

   if(!m_fgThread.joinable())
   {
      log<software_error>({__FILE__, __LINE__, "framegrabber thread did not start"});
      return -1;
   }

   //Now set the RT priority.
   
   int prio=m_fgThreadPrio;
   if(prio < 0) prio = 0;
   if(prio > 99) prio = 99;

   sched_param sp;
   sp.sched_priority = prio;

   //Get the maximum privileges available
   if( euidCalled() < 0 )
   {
      log<software_error>({__FILE__, __LINE__, "Setting euid to called failed."});
      return -1;
   }
   
   //We set return value based on result from sched_setscheduler
   //But we make sure to restore privileges no matter what happens.
   errno = 0;
   int rv = 0;
   if(prio > 0) rv = pthread_setschedparam(m_fgThread.native_handle(), MAGAOX_RT_SCHED_POLICY, &sp);
   else rv = pthread_setschedparam(m_fgThread.native_handle(), SCHED_OTHER, &sp);
   
   //Go back to regular privileges
   if( euidReal() < 0 )
   {
      log<software_error>({__FILE__, __LINE__, "Setting euid to real failed."});
   }
   
   if(rv < 0)
   {
      return log<software_error,-1>({__FILE__, __LINE__, errno, "Setting F.G. thread scheduler priority to " + std::to_string(prio) + " failed."});
   }
   else
   {
      return log<text_log,0>("F.G. thread scheduler priority (framegrabber.threadPrio) set to " + std::to_string(prio));
   }
   

}

inline
void picamCtrl::fgThreadExec()
{

   while(m_shutdown == 0)
   {
      while(!m_shutdown && (!( state() == stateCodes::READY || state() == stateCodes::OPERATING) || m_powerState <= 0 ) )
      {
         sleep(1);
      }
      
      //*****&&&&&&&@@@@@@@@@@@   TEMPORARY
      while(! m_shutdown) sleep(1);
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
      
#if 0
      
      
      if(m_shutdown) continue;
      else //This gives a nice scope for the mutex
      {
         // call static_cast<derived>::preAcqModeConfig() or whatever
         
//          std::unique_lock<std::mutex> lock(m_indiMutex);
      
         //Send command to camera to place it in the correct mode, if needed.
         
         
//          log<text_log>("Send command to set mode: " + m_cameraModes[m_modeName].m_serialCommand);
//          log<text_log>("Response was: " + response);
//          
//          updateIfChanged(m_indiP_mode, "current", m_modeName);
//          updateIfChanged(m_indiP_mode, "target", std::string(""));
      }
   
      ///\todo check response
   
   
      /* Initialize the Camera?
       */

      
      /* Initialize ImageStreamIO
       */
      IMAGE imageStream;
      uint32_t imsize[3];
      imsize[0] = 1024; //MODE DETERMINED SIZE?
      imsize[1] = OCAM_SZ;
      imsize[2] = 1;
      ImageStreamIO_createIm(&imageStream, "ocam2k", 2, imsize, _DATATYPE_INT16, 1, 0);

      
      //This completes the reconfiguration.
      m_reconfig = false;
                  
      //Trigger an FPS reset.
      m_lastImageNumber = -1;
      
      //This is the main image grabbing loop.
      
      while(!m_shutdown && !m_reconfig && m_powerState > 0)
      {
         if( state() != stateCodes::OPERATING || m_powerState <= 0)
         {
            sleep(1);
            continue;
         }
         
         
         //==================
         //Get next image, process validity.
         //====================
         
         //Ok, no timeout, so we process the image and publish it.
         imageStream.md[0].write=1;
         ocam2_descramble(id, &currImageNumber, imageStream.array.SI16, (short int *) image_p);
         imageStream.md[0].cnt0++;
         imageStream.md[0].cnt1++;
         imageStream.md[0].write=0;
         ImageStreamIO_sempost(&imageStream,-1);
 

      }
    
      ImageStreamIO_destroyIm( &imageStream );
    
      if(m_reconfig && !m_shutdown)
      {
         //lock mutex
         std::unique_lock<std::mutex> lock(m_indiMutex);
         
         if(1 /*Do reconfigure here */) //pdvConfig(m_nextMode) < 0)
         {
            log<text_log>("error trying to re-configure with " + m_nextMode, logPrio::LOG_ERROR);
            sleep(1);
         }
         else
         {
            m_nextMode = "";
         }
      }
#endif
   } //outer loop, will exit if m_shutdown==true

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
      
      updateIfChanged(m_indiP_ccdTemp, "target", target);
      
      return setTemp(target);
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
