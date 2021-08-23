/** \file zylaCtrl.hpp
  * \brief The MagAO-X Andor sCMOS camera controller.
  *
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup zylaCtrl_files
  */

#ifndef zylaCtrl_hpp
#define zylaCtrl_hpp






#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"


#include "atcore.h"
#include "atutility.h"

namespace MagAOX
{
namespace app
{


/** \defgroup zylaCtrl Andor sCMOS Camera
  * \brief Control of an Andor sCMOS Camera.
  *
  * <a href="../handbook/operating/software/apps/zylaCtrl.html">Application Documentation</a>
  *
  * \ingroup apps
  *
  */

/** \defgroup zylaCtrl_files Andor sCMOS Camera Files
  * \ingroup zylaCtrl
  */

/** MagAO-X application to control an Andor sCMOS Camera
  *
  * \ingroup zylaCtrl
  *
  */
class zylaCtrl : public MagAOXApp<>, public dev::stdCamera<zylaCtrl>, public dev::frameGrabber<zylaCtrl>, public dev::telemeter<zylaCtrl>
{

   friend class dev::stdCamera<zylaCtrl>;
   friend class dev::frameGrabber<zylaCtrl>;
   friend class dev::telemeter<zylaCtrl>;

public:
   /** \name app::dev Configurations
     *@{
     */
   static constexpr bool c_stdCamera_tempControl = true; ///< app::dev config to tell stdCamera to expose temperature controls
   
   static constexpr bool c_stdCamera_temp = true; ///< app::dev config to tell stdCamera to expose temperature
   
   static constexpr bool c_stdCamera_readoutSpeed = false; ///< app::dev config to tell stdCamera to expose readout speed controls
   
   static constexpr bool c_stdCamera_vShiftSpeed = false; ///< app:dev config to tell stdCamera to expose vertical shift speed control
   
   static constexpr bool c_stdCamera_emGain = false; ///< app::dev config to tell stdCamera to expose EM gain controls 

   static constexpr bool c_stdCamera_exptimeCtrl = true; ///< app::dev config to tell stdCamera to expose exposure time controls
   
   static constexpr bool c_stdCamera_fpsCtrl = true; ///< app::dev config to tell stdCamera to expose FPS controls

   static constexpr bool c_stdCamera_fps = true; ///< app::dev config to tell stdCamera not to expose FPS status
   
   static constexpr bool c_stdCamera_usesModes = false; ///< app:dev config to tell stdCamera not to expose mode controls
   
   static constexpr bool c_stdCamera_usesROI = true; ///< app:dev config to tell stdCamera to expose ROI controls

   static constexpr bool c_stdCamera_cropMode = false; ///< app:dev config to tell stdCamera to expose Crop Mode controls
   
   static constexpr bool c_stdCamera_hasShutter = false; ///< app:dev config to tell stdCamera to expose shutter controls
   
   static constexpr bool c_frameGrabber_flippable = false; ///< app:dev config to tell framegrabber this camera can not be flipped
   
   ///@}
   
protected:

   /** \name configurable parameters
     *@{
     */

   std::string m_serial; ///< The camera serial number.  This is a required configuration parameter.
   
   unsigned int m_imageTimeout {1000}; ///< Timeout for waiting on images [msec].  Default is 1000 msec.
   
   ///@}

   bool m_libInit {false}; ///< Flag indicating whether the AT library is initialized.
   
   AT_H m_handle {AT_HANDLE_UNINITIALISED}; ///< The Andor API handle to the camera

   std::vector<unsigned char*> m_inputBuffers;
   size_t m_nextBuffer {0};
   
   int m_inputBufferSize {0};
   
   unsigned char* m_outputBuffer {nullptr};
   
   int m_outputBufferSize {0};
   
   wchar_t m_pixelEncoding[256];

   int m_stride;
   
public:

   ///Default c'tor
   zylaCtrl();

   ///Destructor
   ~zylaCtrl() noexcept;

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


   /// Select the camera with the desired serial number.
   int cameraSelect();

   
   int getTemp();

   int getExpTime();

   int getFPS();

   
   /** \name stdCamera Interface 
     * 
     * @{
     */
   
   /// Set defaults for a power on state.
   /** 
     * \returns 0 on success
     * \returns -1 on error
     */ 
   int powerOnDefaults();
   
   /// Turn temperature control on or off.
   /** Sets temperature control on or off based on the current value of m_tempControlStatus
     * \returns 0 on success
     * \returns -1 on error
     */ 
   int setTempControl();
   
   /// Set the CCD temperature setpoint [stdCamera interface].
   /** Sets the temperature to m_ccdTempSetpt.
     * \returns 0 on success
     * \returns -1 on error
     */
   int setTempSetPt();
   
   /// Set the frame rate. [stdCamera interface]
   /** Sets the frame rate to m_fpsSet.
     * 
     * \returns 0 on success
     * \returns -1 on error
     */
   int setFPS();
   
   /// Required by stdCamera, but this does not do anything for this camera [stdCamera interface]
   /**
     * \returns 0 always
     */ 
   int setExpTime();
   
   /// Required by stdCamera, checks the next ROI [stdCamera interface]
   /** Checks if the target values are valid and adjusts them to the closest valid values if needed.
     *
     * \returns 0 if successful
     * \returns -1 on error
     */
   int checkNextROI();

   /// Required by stdCamera, but this does not do anything for this camera [stdCamera interface]
   /**
     * \returns 0 always
     */
   int setNextROI();
   
   /// Required by stdCamera, but this does not do anything for this camera [stdCamera interface]
   /**
     * \returns 0 always
     */
   int setShutter(int sh);
   
   ///@}
   
   
   
   /** \name framegrabber Interface 
     * 
     * @{
     */
   
   int configureAcquisition();
   float fps()
   {
      return m_fps;
   }
   
   int startAcquisition();
   int acquireAndCheckValid();
   int loadImageIntoStream(void * dest);
   int reconfig();

   ///@}
   
   /** \name Telemeter Interface
     * 
     * @{
     */ 
   int checkRecordTimes();
   
   int recordTelem( const telem_stdcam * );
   
   
   ///@}
   
};

inline
zylaCtrl::zylaCtrl() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   m_powerMgtEnabled = true;
   m_powerOnWait = 10;
   

   m_startupTemp = 20;
   
   m_expTimeSet = 0.05; //Set default for startup
   m_fpsSet = 20; //Set default for startup
   
   m_startup_x = 1075;
   m_startup_y = 975;
   m_startup_w = 128;
   m_startup_h = 128;
   m_startup_bin_x = 1;
   m_startup_bin_y = 1;
   
   m_full_x = 1023.5;
   m_full_y = 1023.5;
   m_full_w = 2048;
   m_full_h = 2048;
   
   return;
}

inline
zylaCtrl::~zylaCtrl() noexcept
{
   for(size_t n=0; n < m_inputBuffers.size(); ++n)
   {
      if(m_inputBuffers[n]) free(m_inputBuffers[n]);
   }
   
   return;
}

inline
void zylaCtrl::setupConfig()
{
   dev::stdCamera<zylaCtrl>::setupConfig(config);
   
   config.add("camera.serial", "", "camera.serial", argType::Required, "camera", "serial", false, "string", "The camera serial number.");
   
   dev::frameGrabber<zylaCtrl>::setupConfig(config);
   dev::telemeter<zylaCtrl>::setupConfig(config);

}



inline
void zylaCtrl::loadConfig()
{
   dev::stdCamera<zylaCtrl>::loadConfig(config);

   config(m_serial, "camera.serial");

   dev::frameGrabber<zylaCtrl>::loadConfig(config);
   dev::telemeter<zylaCtrl>::loadConfig(config);
}



inline
int zylaCtrl::appStartup()
{
   m_minROIx = 0;
   m_maxROIx = 2047;
   m_stepROIx = 0;
   
   m_minROIy = 0;
   m_maxROIy = 2047;
   m_stepROIy = 0;
   
   m_minROIWidth = 1;
   m_maxROIWidth = 2048;
   m_stepROIWidth = 4;
   
   m_minROIHeight = 1;
   m_maxROIHeight = 2048;
   m_stepROIHeight = 1;
   
   m_minROIBinning_x = 1;
   m_maxROIBinning_x = 32;
   m_stepROIBinning_x = 1;
   
   m_minROIBinning_y = 1;
   m_maxROIBinning_y = 1024;
   m_stepROIBinning_y = 1;
   
   if(dev::stdCamera<zylaCtrl>::appStartup() < 0)
   {
      return log<software_critical,-1>({__FILE__,__LINE__});
   }

   if(dev::frameGrabber<zylaCtrl>::appStartup() < 0)
   {
      return log<software_critical,-1>({__FILE__,__LINE__});
   }

   if(dev::telemeter<zylaCtrl>::appStartup() < 0)
   {
      return log<software_error,-1>({__FILE__,__LINE__});
   }
   
   m_inputBuffers.resize(3);
   for(size_t n =0; n < m_inputBuffers.size(); ++n)
   {
      m_inputBuffers[n] = nullptr;
   }
   m_nextBuffer = 0;
   
   state(stateCodes::NOTCONNECTED);

   return 0;

}


inline
int zylaCtrl::appLogic()
{
   //and run stdCamera's appLogic
   if(dev::stdCamera<zylaCtrl>::appLogic() < 0)
   {
      return log<software_error, -1>({__FILE__, __LINE__});
   }
   
   //first run frameGrabber's appLogic to see if the f.g. thread has exited.
   if(dev::frameGrabber<zylaCtrl>::appLogic() < 0)
   {
      return log<software_error, -1>({__FILE__, __LINE__});
   }

   if( state() == stateCodes::POWERON) return 0;
   
   if( state() == stateCodes::NOTCONNECTED || state() == stateCodes::ERROR)
   {
      //Might have gotten here because of a power off.
      if(m_powerState == 0) return 0;

      int ret = cameraSelect(); 

      if( ret != 0) //Probably not powered on yet.
      {
         sleep(1);
         return 0;
      }

      state(stateCodes::CONNECTED);


   }

   if( state() == stateCodes::CONNECTED )
   {
      //Get a lock
      std::unique_lock<std::mutex> lock(m_indiMutex);

      state(stateCodes::READY);
      
      m_tempControlStatusSet = true;
      setTempControl();
      
   }

   if( state() == stateCodes::READY || state() == stateCodes::OPERATING )
   {
      //Get a lock if we can
      std::unique_lock<std::mutex> lock(m_indiMutex, std::try_to_lock);

      //but don't wait for it, just go back around.
      if(!lock.owns_lock()) return 0;

      if(getTemp() < 0)
      {
         if(m_powerState == 0) return 0;

         state(stateCodes::ERROR);
         return 0;
      }

      if(getExpTime() < 0)
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
      
      if(stdCamera<zylaCtrl>::updateINDI() < 0)
      {
         return log<software_error,0>({__FILE__,__LINE__});
      }
      
      if(frameGrabber<zylaCtrl>::updateINDI() < 0)
      {
         return log<software_error,0>({__FILE__,__LINE__});
      }

      if(dev::telemeter<zylaCtrl>::appLogic() < 0)
      {
         log<software_error>({__FILE__, __LINE__});
         return 0;
      }

   }

   //Fall through check?

   return 0;

}

inline
int zylaCtrl::onPowerOff()
{
   m_powerOnCounter = 0;

   if(m_handle != AT_HANDLE_UNINITIALISED)
   {
      AT_Close(m_handle);
      m_handle = AT_HANDLE_UNINITIALISED;
   }
   
   if(m_libInit)
   {
      AT_FinaliseLibrary();
      AT_FinaliseUtilityLibrary();

      m_libInit = false;
   }

   std::lock_guard<std::mutex> lock(m_indiMutex);

   stdCamera<zylaCtrl>::onPowerOff();
   
   return 0;
}

inline
int zylaCtrl::whilePowerOff()
{
   std::lock_guard<std::mutex> lock(m_indiMutex);

   stdCamera<zylaCtrl>::whilePowerOff();
   
   return 0;
}

inline
int zylaCtrl::appShutdown()
{
   dev::stdCamera<zylaCtrl>::appShutdown();
   dev::frameGrabber<zylaCtrl>::appShutdown();

   if(m_handle != AT_HANDLE_UNINITIALISED)
   {
      AT_Close(m_handle);
      m_handle = AT_HANDLE_UNINITIALISED;
   }
   
   if(m_libInit)
   {
      AT_FinaliseLibrary();
      AT_FinaliseUtilityLibrary();

      m_libInit = false;
   }
   
   
   
   return 0;
}

inline
int zylaCtrl::cameraSelect()
{
   int iErr;
   
   if(m_handle != AT_HANDLE_UNINITIALISED)
   {
      log<software_warning>({__FILE__, __LINE__, "handle initialized on call to cameraSelect.  Attempting to close and go on."});
      
      iErr = AT_Close(m_handle);
      if(iErr != AT_SUCCESS)
      {
         log<software_error>({__FILE__, __LINE__,  "Error from AT_Close: " + std::to_string(iErr) + ". Attempting to go on." });
         m_handle = AT_HANDLE_UNINITIALISED;
      }
   }
      
   if(m_libInit)
   {
      iErr = AT_FinaliseLibrary();
      if(iErr != AT_SUCCESS )
      {
         return log<software_critical,-1>({__FILE__, __LINE__,  "Error from AT_FinaliseLibrary: " + std::to_string(iErr)});
      }
      iErr = AT_FinaliseUtilityLibrary();
      if(iErr != AT_SUCCESS )
      {
         return log<software_critical,-1>({__FILE__, __LINE__,  "Error from AT_FinaliseUtilityLibrary: " + std::to_string(iErr)});
      }
      
      m_libInit = false;
   }
   
   iErr = AT_InitialiseLibrary();
   if( iErr != AT_SUCCESS ) 
   {
      return log<software_critical,-1>({__FILE__, __LINE__,  "Error from AT_InitialiseLibrary: " + std::to_string(iErr)});
   }
   
   iErr = AT_InitialiseUtilityLibrary();
   if( iErr != AT_SUCCESS ) 
   {
      return log<software_critical,-1>({__FILE__, __LINE__,  "Error from AT_InitialiseUtilityLibrary: " + std::to_string(iErr)});
   }
   
   m_libInit = true;
  
   long long DeviceCount = 0;
  
   iErr = AT_GetInt(AT_HANDLE_SYSTEM, L"Device Count", &DeviceCount);
  
   if (iErr != AT_SUCCESS) 
   {
      return log<software_critical,-1>({__FILE__,__LINE__, "Error from AT_GetInt('Device Count'): " + std::to_string(iErr)});
   }

   std::cout << "Found " << DeviceCount << " Devices." << std::endl;

   for (long long i=0; i<DeviceCount; i++) 
   {
      AT_H Hndl  = AT_HANDLE_UNINITIALISED;
    
      iErr = AT_Open(static_cast<int>(i), &Hndl);
  
      if (iErr != AT_SUCCESS) 
      {
         return log<software_critical,-1>({__FILE__,__LINE__, "Error from AT_Open(): " + std::to_string(iErr)});
      }
      
      AT_WC CameraSerial[128];
      
      iErr = AT_GetString(Hndl, L"SerialNumber", CameraSerial, 128);
      
      if (iErr != AT_SUCCESS) 
      {
         return log<software_critical,-1>({__FILE__,__LINE__, "Error from AT_GetString('SerialNumber'): " + std::to_string(iErr)});
      }

      char camSerial[128];
      wcstombs(camSerial, CameraSerial, sizeof(camSerial));
      
      if(m_serial != camSerial)
      {
         iErr = AT_Close(Hndl);
         if (iErr != AT_SUCCESS) 
         {
            log<software_error>({__FILE__,__LINE__, "Error from AT_Close(): " + std::to_string(iErr)});
         }
         
         continue;
      }
      
      AT_WC CameraModel[128];
      
      iErr = AT_GetString(Hndl, L"Camera Model", CameraModel, 128);
      
      if (iErr != AT_SUCCESS) 
      {
         return log<software_critical,-1>({__FILE__,__LINE__, "Error from AT_GetString('Camera Model'): " + std::to_string(iErr)});
      }

      char camModel[128];
      wcstombs(camModel, CameraModel, sizeof(camModel));

      log<text_log>({std::string("Found ") + camModel + " serial number " + m_serial}, logPrio::LOG_NOTICE);
      
      m_handle = Hndl;
      return 0;
   }
   
   log<text_log>({"Camera with serial number " + m_serial + " not found in " + std::to_string(DeviceCount) + "devices."}, logPrio::LOG_WARNING);
   
   m_handle = AT_HANDLE_UNINITIALISED;
   AT_FinaliseLibrary();
   AT_FinaliseUtilityLibrary();

   m_libInit = false;
   
   return -1;
   
}

inline
int zylaCtrl::getTemp()
{
   int temperatureStatusIndex = 0;
   wchar_t temperatureStatus[256];
   int rv = AT_GetEnumIndex(m_handle, L"TemperatureStatus", &temperatureStatusIndex);
   if (rv != AT_SUCCESS) 
   {
      return log<software_error,-1>({__FILE__,__LINE__, "Error from AT_EnumIndex('TemperatureStatus'): " + std::to_string(rv)});
   }
      
   rv = AT_GetEnumStringByIndex(m_handle, L"TemperatureStatus", temperatureStatusIndex, temperatureStatus, 256);
   if (rv != AT_SUCCESS) 
   {
      return log<software_error,-1>({__FILE__,__LINE__, "Error from AT_EnumStringByIndex('TemperatureStatus'): " + std::to_string(rv)});
   }
   
   if(wcscmp(L"Stabilised",temperatureStatus) == 0)
   {
      m_tempControlStatusStr="Stabilised";
      m_tempControlStatus = true;
      m_tempControlOnTarget = true;
   }
   else if(wcscmp(L"Cooler Off",temperatureStatus) == 0)
   {
      m_tempControlStatusStr="Cooler Off";
      m_tempControlStatus = false;
      m_tempControlOnTarget = false;
   }
   else if(wcscmp(L"Cooling",temperatureStatus) == 0)
   {
      m_tempControlStatusStr="Cooling";
      m_tempControlStatus = true;
      m_tempControlOnTarget = false;
   }
   else if(wcscmp(L"Drift",temperatureStatus) == 0)
   {
      m_tempControlStatusStr="Drift";
      m_tempControlStatus = true;
      m_tempControlOnTarget = false;
   }
   else if(wcscmp(L"Not Stabilised",temperatureStatus) == 0)
   {
      m_tempControlStatusStr="Not Stabilised";
      m_tempControlStatus = true;
      m_tempControlOnTarget = false;
   }
   else if(wcscmp(L"Fault",temperatureStatus) == 0)
   {
      m_tempControlStatusStr="Fault";
      m_tempControlStatus = false;
      m_tempControlOnTarget = false;
   }
   else
   {
      m_tempControlStatusStr="Unknown";
      m_tempControlStatus = false;
      m_tempControlOnTarget = false;
   }
   
   double val;
   rv = AT_GetFloat(m_handle, L"SensorTemperature", &val);
   if (rv != AT_SUCCESS) 
   {
      return log<software_error,-1>({__FILE__,__LINE__, "Error from AT_GetFloat('SensorTemperature'): " + std::to_string(rv)});
   }
   
   m_ccdTemp = val;

   //Check if we have the right target, and set it if  not.
   rv = AT_GetFloat(m_handle, L"TargetSensorTemperature", &val);
   if (rv != AT_SUCCESS) 
   {
      return log<software_error,-1>({__FILE__,__LINE__, "Error from AT_GetFloat('TargetSensorTemperature'): " + std::to_string(rv)});
   }
   
   m_ccdTempSetpt = val;
   
   recordCamera();
   
   return 0;
}

inline
int zylaCtrl::getExpTime()
{
   return 0;

}

inline
int zylaCtrl::getFPS()
{
   return 0;

}


//------------------------------------------------------------------------
//-----------------------  stdCamera interface ---------------------------
//------------------------------------------------------------------------

inline
int zylaCtrl::powerOnDefaults()
{
   //Camera boots up with this true in most cases.
   m_tempControlStatusSet = false;
   m_tempControlStatus =false;
      
   m_ccdTempSetpt = 0; //This is the power on setpoint

   m_currentROI.x = m_startup_x;
   m_currentROI.y = m_startup_y;
   m_currentROI.w = m_startup_w;
   m_currentROI.h = m_startup_h;
   m_currentROI.bin_x = m_startup_bin_x;
   m_currentROI.bin_y = m_startup_bin_y;
   
   return 0;
}

inline
int zylaCtrl::setTempControl()
{  
   if(m_tempControlStatusSet == true)
   {
      int rv = AT_SetBool(m_handle, L"SensorCooling", AT_TRUE);
      if(rv != AT_SUCCESS)
      {
         return log<software_error,-1>({__FILE__,__LINE__, "Error from AT_SetBool(<SensorCooling>): " + std::to_string(rv)});
      }
      log<text_log>({"cooling on"}, logPrio::LOG_NOTICE);
   }
   else
   {
      int rv = AT_SetBool(m_handle, L"SensorCooling", AT_FALSE);
      if(rv != AT_SUCCESS)
      {
         return log<software_error,-1>({__FILE__,__LINE__, "Error from AT_SetBool(<SensorCooling>): " + std::to_string(rv)});
         log<text_log>({"cooling off"}, logPrio::LOG_NOTICE);
      }
   }
   
   recordCamera();
   return 0;
}

inline
int zylaCtrl::setTempSetPt()
{
   std::cerr << "setTempSetPt is not implemented\n";
   return 0;
}

inline 
int zylaCtrl::setExpTime()
{
   std::cerr << "Set exposure time\n";
   m_reconfig = true;
   return 0;
}

inline
int zylaCtrl::setFPS()
{
   std::cerr << "setFPS\n";
   m_reconfig = true;
   return 0;
}

inline 
int zylaCtrl::checkNextROI()
{
   return 0;
}

inline 
int zylaCtrl::setNextROI()
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
int zylaCtrl::setShutter(int sh)
{
   static_cast<void>(sh);
   
   return 0;
}

//------------------------------------------------------------------------
//-------------------   framegrabber interface ---------------------------
//------------------------------------------------------------------------

inline
int zylaCtrl::configureAcquisition()
{
   int rv;
   
   if(m_handle == AT_HANDLE_UNINITIALISED || m_libInit == false)
   {
      log<software_error>({__FILE__, __LINE__, "camer or AT not initialized on configureAcquisition()."}); 
      return -1;
   }
   
   //lock mutex
   std::unique_lock<std::mutex> lock(m_indiMutex);


   AT_BOOL faoi;
   AT_GetBool(m_handle, L"FullAOIControl", &faoi);
   std::cerr << "FullAOIControl: " << std::boolalpha << faoi << "\n";
   
   //Configure ROI:
   AT_64 xbin = m_nextROI.bin_x;
   AT_64 ybin = m_nextROI.bin_y;
   AT_64 left= (m_nextROI.x - 0.5*( (float) m_nextROI.w - 1.0)) + 1;
   AT_64 top =  (m_nextROI.y - 0.5*( (float) m_nextROI.h - 1.0)) + 1;
   AT_64 width = m_nextROI.w;
   AT_64 height = m_nextROI.h;
   
   std::cerr << xbin << " " << ybin << " " << left << " " << top << " " << width << " " << height << " " << "\n";
   
   rv = AT_SetInt(m_handle, L"AOIHBin", xbin);
   if(rv != AT_SUCCESS)
   {
      return log<software_error,-1>({__FILE__,__LINE__, "Error from AT_SetInt(<AOIHBin>): [" + std::to_string(xbin) + "] err: " + std::to_string(rv)});
   }
   
   rv = AT_SetInt(m_handle, L"AOIVBin", ybin);
   if(rv != AT_SUCCESS)
   {
      return log<software_error,-1>({__FILE__,__LINE__, "Error from AT_SetInt(<AOIVBin>): [" + std::to_string(ybin) + "] err: " + std::to_string(rv)});
   }
   
   rv = AT_SetInt(m_handle, L"AOIWidth", width);
   if(rv != AT_SUCCESS)
   {
      return log<software_error,-1>({__FILE__,__LINE__, "Error from AT_SetInt(<AOIWidth>): [" + std::to_string(width) + "] err: " + std::to_string(rv)});
   }
   
   rv = AT_SetInt(m_handle, L"AOILeft", left);
   if(rv != AT_SUCCESS)
   {
      return log<software_error,-1>({__FILE__,__LINE__, "Error from AT_SetInt(<AOILeft>): [" + std::to_string(left) + "] err: " + std::to_string(rv)});
   }
   
   rv = AT_SetInt(m_handle, L"AOIHeight", height);
   if(rv != AT_SUCCESS)
   {
      return log<software_error,-1>({__FILE__,__LINE__, "Error from AT_SetInt(<AOIHeight>): [" + std::to_string(height) + "] err: " + std::to_string(rv)});
   }
   
   rv = AT_SetInt(m_handle, L"AOITop", top);
   if(rv != AT_SUCCESS)
   {
      return log<software_error,-1>({__FILE__,__LINE__, "Error from AT_SetInt(<AOITop>): [" + std::to_string(top) + "] err: " + std::to_string(rv)});
   }
   
   //Get Detector dimensions
   AT_64 stride;
    
   rv = AT_GetInt(m_handle, L"AOI Left", &left);    
   if(rv != AT_SUCCESS)
   {
      return log<software_error,-1>({__FILE__,__LINE__, "Error from AT_GetInt(<AOI Left>): " + std::to_string(rv)});
   }

   rv = AT_GetInt(m_handle, L"AOI Top", &top);    
   if(rv != AT_SUCCESS)
   {
      return log<software_error,-1>({__FILE__,__LINE__, "Error from AT_GetInt(<AOI Top>): " + std::to_string(rv)});
   }
   
   rv = AT_GetInt(m_handle, L"AOI Width", &width);    
   if(rv != AT_SUCCESS)
   {
      return log<software_error,-1>({__FILE__,__LINE__, "Error from AT_GetInt(<AOI Width>): " + std::to_string(rv)});
   }
   
   rv = AT_GetInt(m_handle, L"AOI Height", &height);    
   if(rv != AT_SUCCESS)
   {
      return log<software_error,-1>({__FILE__,__LINE__, "Error from AT_GetInt(<AOI Height>): " + std::to_string(rv)});
   }
   
   m_currentROI.x = left + 0.5*( (float) (width - 1.0)) ;
   m_currentROI.y = top + 0.5*( (float) (height - 1.0)) ;
   
   m_currentROI.w = width;
   m_currentROI.h = height;
   
   updateIfChanged( m_indiP_roi_x, "current", m_currentROI.x, INDI_OK);
   updateIfChanged( m_indiP_roi_y, "current", m_currentROI.y, INDI_OK);
   updateIfChanged( m_indiP_roi_w, "current", m_currentROI.w, INDI_OK);
   updateIfChanged( m_indiP_roi_h, "current", m_currentROI.h, INDI_OK);
   updateIfChanged( m_indiP_roi_bin_x, "current", m_currentROI.bin_x, INDI_OK);
   updateIfChanged( m_indiP_roi_bin_y, "current", m_currentROI.bin_y, INDI_OK);
   
   rv = AT_GetInt(m_handle, L"AOI Stride", &stride);    
   if(rv != AT_SUCCESS)
   {
      return log<software_error,-1>({__FILE__,__LINE__, "Error from AT_GetInt(<AOI Stride>): " + std::to_string(rv)});
   }
   
   m_width = static_cast<int>(width);
   m_height = static_cast<int>(height);
   m_stride = static_cast<int>(stride);
   m_dataType = _DATATYPE_UINT16;

   //Free the API buffer
   for(size_t n =0; n < m_inputBuffers.size(); ++n)
   {
      if(m_inputBuffers[n])
      {
         free(m_inputBuffers[n]);
         m_inputBuffers[n] = nullptr;
      }
      m_inputBufferSize = 0;
   }
   
   //Get the number of bytes required to store one frame
   AT_64 ImageSizeBytes;
   rv = AT_GetInt(m_handle, L"ImageSizeBytes", &ImageSizeBytes);
   if(rv != AT_SUCCESS)
   {
      return log<software_error,-1>({__FILE__,__LINE__, "Error from AT_GetInt(<ImageSizeBytes>): " + std::to_string(rv)});
   }
   
   m_inputBufferSize = static_cast<int>(ImageSizeBytes);
   
   //Allocate a memory buffer to store one frame
   for(size_t n =0; n < m_inputBuffers.size(); ++n)
   {
      m_inputBuffers[n] = (unsigned char *) malloc(m_inputBufferSize * sizeof(unsigned char));
   }
   
   rv = AT_Flush(m_handle);
   if(rv != AT_SUCCESS)
   {
      return log<software_error,-1>({__FILE__,__LINE__, "Error from AT_Flush " + std::to_string(rv)});
   }
      
   //Pass this buffer to the SDK
   for(size_t n =0; n < m_inputBuffers.size(); ++n)
   {
      rv = AT_QueueBuffer(m_handle, m_inputBuffers[n], m_inputBufferSize);
      if(rv != AT_SUCCESS)
      {
         return log<software_error,-1>({__FILE__,__LINE__, "Error from AT_QueueBuffer: " + std::to_string(rv)});
      }
   }
   m_nextBuffer = 0;
   
   AT_SetFloat(m_handle, L"ExposureTime", m_expTimeSet);
   if(rv != AT_SUCCESS)
   {
      return log<software_error,-1>({__FILE__,__LINE__, "Error from AT_SetFloat(<ExposureTime>): " + std::to_string(rv)});
   }
   m_expTime = m_expTimeSet;
   
   AT_SetFloat(m_handle, L"FrameRate", m_fpsSet);
   if(rv != AT_SUCCESS)
   {
      return log<software_error,-1>({__FILE__,__LINE__, "Error from AT_SetFloat(<FrameRate>): " + std::to_string(rv)});
   }
   m_fps = m_fpsSet;
   
   int pixelEncodingIndex = 0;

   AT_GetEnumIndex(m_handle, L"PixelEncoding", &pixelEncodingIndex);
   AT_GetEnumStringByIndex(m_handle, L"PixelEncoding", pixelEncodingIndex, m_pixelEncoding, sizeof(m_pixelEncoding));

   std::wcout << m_pixelEncoding << "\n";
   
   //Set the camera to continuously acquire frames
   rv = AT_SetEnumString(m_handle, L"CycleMode", L"Continuous");
   if(rv != AT_SUCCESS)
   {
      return log<software_error,-1>({__FILE__,__LINE__, "Error from AT_SetEnumString(<CycleMode-Continuous>): " + std::to_string(rv)});
   }
   
   log<text_log>({"Camera configured for continous acquistion with " + std::to_string(m_width) + "x" + std::to_string(m_height)});
   
   recordCamera(true); //Force so it is logged before starting acq.
   
   return 0;
}

inline
int zylaCtrl::startAcquisition()
{
   //Start the Acquisition running
   int rv = AT_Command(m_handle, L"AcquisitionStart");
   
   if(rv != AT_SUCCESS)
   {
      return log<software_error,-1>({__FILE__,__LINE__, "Error from AT_Command(<AcquisitionStart>): " + std::to_string(rv)});
   }
   
   log<text_log>("Acqusition started");
   
   return 0;
}

inline
int zylaCtrl::acquireAndCheckValid()
{
   int rv = AT_WaitBuffer(m_handle, &m_outputBuffer, &m_outputBufferSize, m_imageTimeout);
   
   if(rv == AT_ERR_TIMEDOUT) 
   {
      return 1;
   }
   
   clock_gettime(CLOCK_REALTIME, &m_currImageTimestamp);
      
   
   if(rv != AT_SUCCESS )
   {
      return log<software_error,-1>({__FILE__,__LINE__, "Error from AT_WaitBuffer: " + std::to_string(rv)});
   }
   
   if(m_outputBufferSize != m_inputBufferSize)
   {
      return log<software_error,-1>({__FILE__,__LINE__, "Wrong buffer size returned"});
   }
   
   return 0;
}

inline
int zylaCtrl::loadImageIntoStream(void * dest)
{
   if(m_outputBuffer == nullptr) return -1;

   AT_ConvertBuffer(m_outputBuffer, static_cast<AT_U8*>(dest), m_width, m_height, m_stride, m_pixelEncoding, L"Mono16");

   if(m_outputBuffer != m_inputBuffers[m_nextBuffer]) 
   {
      std::cerr << "buffer skip!\n";
      while(m_outputBuffer != m_inputBuffers[m_nextBuffer])
      {
         ++m_nextBuffer;
         if(m_nextBuffer >= m_inputBuffers.size()) m_nextBuffer = 0;
      }
   }
   
   int rv = AT_QueueBuffer(m_handle, m_inputBuffers[m_nextBuffer], m_inputBufferSize);
   if(rv != AT_SUCCESS)
   {
      return log<software_error,-1>({__FILE__,__LINE__, "Error from AT_QueueBuffer: " + std::to_string(rv)});
   }

   //Pass the buffer to the SDK
   ++m_nextBuffer;
   if(m_nextBuffer >= m_inputBuffers.size()) m_nextBuffer = 0;

   
   return 0;
}

inline
int zylaCtrl::reconfig()
{
   //lock mutex
   std::unique_lock<std::mutex> lock(m_indiMutex);

   recordCamera(true); //force so it is logged before stopping acq.
   
    //Start the Acquisition running
   int rv = AT_Command(m_handle, L"AcquisitionStop");
   if(rv != AT_SUCCESS)
   {
      return log<software_error,-1>({__FILE__,__LINE__, "Error from AT_Command(<AcquisitionStop>): " + std::to_string(rv)});
   }
   log<text_log>("Acqusition stopped");
   
   rv = AT_Flush(m_handle);
   if(rv != AT_SUCCESS)
   {
      return log<software_error,-1>({__FILE__,__LINE__, "Error from AT_Fluxh  : " + std::to_string(rv)});
   }
   
   return 0;//edtCamera<zylaCtrl>::pdvReconfig();
}

int zylaCtrl::checkRecordTimes()
{
   return telemeter<zylaCtrl>::checkRecordTimes(telem_stdcam());
}
   
int zylaCtrl::recordTelem(const telem_stdcam *)
{
   return recordCamera(true);
}

}//namespace app
} //namespace MagAOX
#endif
