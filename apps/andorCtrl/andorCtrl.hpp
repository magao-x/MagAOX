/** \file andorCtrl.hpp
  * \brief The MagAO-X Andor EMCCD camera controller.
  *
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup andorCtrl_files
  */

#ifndef andorCtrl_hpp
#define andorCtrl_hpp






#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"


#include "atmcdLXd.h"

namespace MagAOX
{
namespace app
{

#define CAMCTRL_E_NOCONFIGS (-10)


/** \defgroup andorCtrl Andor EMCCD Camera
  * \brief Control of the Andor EMCCD Camera.
  *
  * <a href="../handbook/operating/software/apps/andorCtrl.html">Application Documentation</a>
  *
  * \ingroup apps
  *
  */

/** \defgroup andorCtrl_files Andor EMCCD Camera Files
  * \ingroup andorCtrl
  */

/** MagAO-X application to control the Andor EMCCD
  *
  * \ingroup andorCtrl
  *
  */
class andorCtrl : public MagAOXApp<>, public dev::stdCamera<andorCtrl>, public dev::edtCamera<andorCtrl>, 
                                          public dev::frameGrabber<andorCtrl>, public dev::telemeter<andorCtrl>
{

   friend class dev::stdCamera<andorCtrl>;
   friend class dev::edtCamera<andorCtrl>;
   friend class dev::frameGrabber<andorCtrl>;
   friend class dev::telemeter<andorCtrl>;

protected:

   /** \name configurable parameters
     *@{
     */

   //Camera:
   unsigned long m_powerOnWait {10}; ///< Time in sec to wait for camera boot after power on.

   float m_startupTemp {20.0}; ///< The temperature to set after a power-on.

   unsigned m_maxEMGain {600};

   ///@}

   std::string m_configFile; ///< The path, relative to configDir, where to write and read the temporary config file.
   
   bool m_libInit {false}; ///< Whether or not the Andor SDK library is initialized.

   unsigned m_emGain {1};


public:

   ///Default c'tor
   andorCtrl();

   ///Destructor
   ~andorCtrl() noexcept;

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




   int cameraSelect();

   
   int getTemp();

   int getFPS();

   int setFPS(float fps);

   int getEMGain();

   int setEMGain( unsigned emg );

   int getShutter();

   int setShutter( unsigned os);

   
   int writeConfig();
   
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
   
   /// Required by stdCamera, but this does not do anything for this camera [stdCamera interface]
   /**
     * \returns 0 always
     */
   int setNextROI();
   
   ///@}
   
   
   
   /** \name framegrabber Interface 
     * 
     * @{
     */
   
   int configureAcquisition();
   float fps();
   int startAcquisition();
   int acquireAndCheckValid();
   int loadImageIntoStream(void * dest);
   int reconfig();


   //INDI:
protected:
  
   pcf::IndiProperty m_indiP_emGain;

public:
   INDI_NEWCALLBACK_DECL(andorCtrl, m_indiP_emGain);

   
   /** \name Telemeter Interface
     * 
     * @{
     */ 
   int checkRecordTimes();
   
   int recordTelem( const telem_stdcam * );
      
   ///@}
};

inline
andorCtrl::andorCtrl() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   m_powerMgtEnabled = true;
   m_powerOnWait = 10;
   
   m_startupTemp = 20;
   
   m_hasShutter = true;
   
   m_startup_x = 255.5; 
   m_startup_y = 255.5; 
   m_startup_w = 512;  
   m_startup_h = 512;  
      
   m_nextROI.x = m_startup_x;
   m_nextROI.y = m_startup_y;
   m_nextROI.w = m_startup_w;
   m_nextROI.h = m_startup_h;
   m_nextROI.bin_x = 1;
   m_nextROI.bin_y = 1;
   
   m_full_x = 255.5; 
   m_full_y = 255.5; 
   m_full_w = 512; 
   m_full_h = 512; 
   
   //--- frameGrabber ---
   m_flippable = true;
   
   return;
}

inline
andorCtrl::~andorCtrl() noexcept
{
   return;
}

inline
void andorCtrl::setupConfig()
{
   //Here we override edt camera and take over mode management.
   m_usesModes = false;

   dev::stdCamera<andorCtrl>::setupConfig(config);
   dev::edtCamera<andorCtrl>::setupConfig(config);
   
   config.add("camera.maxEMGain", "", "camera.maxEMGain", argType::Required, "camera", "maxEMGain", false, "unsigned", "The maximum EM gain which can be set by  user. Default is 600.  Min is 1, max is 600.");

   dev::frameGrabber<andorCtrl>::setupConfig(config);

   dev::telemeter<andorCtrl>::setupConfig(config);
   

}



inline
void andorCtrl::loadConfig()
{
   dev::stdCamera<andorCtrl>::loadConfig(config);
   
   m_configFile = "../../../tmp/andor_";
   m_configFile += configName();
   m_configFile += ".cfg";
   m_cameraModes["onlymode"] = dev::cameraConfig({m_configFile, "", 255, 255, 512, 512, 1, 1, 1000});
   m_startupMode = "onlymode";
   
   if(writeConfig() < 0)
   {
      log<software_critical>({__FILE__,__LINE__});
      m_shutdown = true;
      return;
   }
   
   dev::edtCamera<andorCtrl>::loadConfig(config);

   config(m_maxEMGain, "camera.maxEMGain");

   if(m_maxEMGain < 1)
   {
      m_maxEMGain = 1;
      log<text_log>("maxEMGain set to 1");
   }

   if(m_maxEMGain > 600)
   {
      m_maxEMGain = 600;
      log<text_log>("maxEMGain set to 600");
   }

   dev::frameGrabber<andorCtrl>::loadConfig(config);
   
   dev::telemeter<andorCtrl>::loadConfig(config);



}



inline
int andorCtrl::appStartup()
{
   
   REG_INDI_NEWPROP(m_indiP_emGain, "emgain", pcf::IndiProperty::Number);
   m_indiP_emGain.add (pcf::IndiElement("current"));
   m_indiP_emGain["current"].set(m_emGain);
   m_indiP_emGain.add (pcf::IndiElement("target"));

   if(dev::stdCamera<andorCtrl>::appStartup() < 0)
   {
      return log<software_critical,-1>({__FILE__,__LINE__});
   }

   if(dev::edtCamera<andorCtrl>::appStartup() < 0)
   {
      return log<software_critical,-1>({__FILE__,__LINE__});
   }

   if(dev::frameGrabber<andorCtrl>::appStartup() < 0)
   {
      return log<software_critical,-1>({__FILE__,__LINE__});
   }

   if(dev::telemeter<andorCtrl>::appStartup() < 0)
   {
      return log<software_error,-1>({__FILE__,__LINE__});
   }
   
   state(stateCodes::NOTCONNECTED);

   return 0;

}



inline
int andorCtrl::appLogic()
{
   //and run stdCamera's appLogic
   if(dev::stdCamera<andorCtrl>::appLogic() < 0)
   {
      return log<software_error, -1>({__FILE__, __LINE__});
   }
   
   //and run edtCamera's appLogic
   if(dev::edtCamera<andorCtrl>::appLogic() < 0)
   {
      return log<software_error, -1>({__FILE__, __LINE__});
   }
   
   //first run frameGrabber's appLogic to see if the f.g. thread has exited.
   if(dev::frameGrabber<andorCtrl>::appLogic() < 0)
   {
      return log<software_error, -1>({__FILE__, __LINE__});
   }

   if( state() == stateCodes::POWERON) return 0;
   
   if( state() == stateCodes::NOTCONNECTED || state() == stateCodes::NODEVICE || state() == stateCodes::ERROR)
   {
      //Might have gotten here because of a power off.
      if(m_powerState == 0) return 0;
      
      int ret = cameraSelect();

      if( ret != 0) 
      {
         return log<software_critical,-1>({__FILE__, __LINE__});
      }
   }

   if( state() == stateCodes::CONNECTED )
   {
      m_shutterStatus = "READY";

      state(stateCodes::READY);
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

     if(getFPS() < 0)
      {
         if(m_powerState == 0) return 0;

         state(stateCodes::ERROR);
         return 0;
      }

      if(getEMGain () < 0)
      {
         if(m_powerState == 0) return 0;

         state(stateCodes::ERROR);
         return 0;
      }

      if(frameGrabber<andorCtrl>::updateINDI() < 0)
      {
         log<software_error>({__FILE__, __LINE__});
         state(stateCodes::ERROR);
         return 0;
      }
      
      if(stdCamera<andorCtrl>::updateINDI() < 0)
      {
         log<software_error>({__FILE__, __LINE__});
         state(stateCodes::ERROR);
         return 0;
      }
      
      if(edtCamera<andorCtrl>::updateINDI() < 0)
      {
         log<software_error>({__FILE__, __LINE__});
         state(stateCodes::ERROR);
         return 0;
      }
      
      if(telemeter<andorCtrl>::appLogic() < 0)
      {
         log<software_error>({__FILE__, __LINE__});
         return 0;
      }
      
   }

   //Fall through check?

   return 0;

}

inline
int andorCtrl::onPowerOff()
{
   if(m_libInit)
   {
      ShutDown();
      m_libInit = false;
   }
      
   m_powerOnCounter = 0;

   std::lock_guard<std::mutex> lock(m_indiMutex);

   updateIfChanged(m_indiP_emGain, "current", 0);
   updateIfChanged(m_indiP_emGain, "target", 0);

   m_shutterStatus = "POWEROFF";
   m_shutterState = 0;
   
   if(stdCamera<andorCtrl>::onPowerOff() < 0)
   {
      log<software_error>({__FILE__, __LINE__});
   }
   
   if(edtCamera<andorCtrl>::onPowerOff() < 0)
   {
      log<software_error>({__FILE__, __LINE__});
   }
   
   if(frameGrabber<andorCtrl>::onPowerOff() < 0)
   {
      log<software_error>({__FILE__, __LINE__});
   }
   
   return 0;
}

inline
int andorCtrl::whilePowerOff()
{
   m_shutterStatus = "POWEROFF";
   m_shutterState = 0;
   
   if(stdCamera<andorCtrl>::whilePowerOff() < 0)
   {
      log<software_error>({__FILE__, __LINE__});
   }
   
   if(edtCamera<andorCtrl>::whilePowerOff() < 0)
   {
      log<software_error>({__FILE__, __LINE__});
   }
   
   return 0;
}


inline
int andorCtrl::appShutdown()
{
   if(m_libInit)
   {
      ShutDown();
      m_libInit = false;
   }
      
   dev::frameGrabber<andorCtrl>::appShutdown();

   dev::telemeter<andorCtrl>::appShutdown();
   
   return 0;
}

std::string andorSDKErrorName(unsigned int error)
{
   switch(error)
   {
      case DRV_SUCCESS:
         return "DRV_SUCCESS";
      case DRV_VXDNOTINSTALLED:
         return "DRV_VXDNOTINSTALLED";
      case DRV_INIERROR:
         return "DRV_INIERROR";
      case DRV_COFERROR:
         return "DRV_COFERROR";
      case DRV_FLEXERROR:
         return "DRV_FLEXERROR";
      case DRV_ERROR_ACK:
         return "DRV_ERROR_ACK";
      case DRV_ERROR_FILELOAD:
         return "DRV_ERROR_FILELOAD";
      case DRV_ERROR_PAGELOCK:
         return "DRV_ERROR_PAGELOCK";
      case DRV_USBERROR:
         return "DRV_USBERROR";
      case DRV_ERROR_NOCAMERA:
         return "DRV_ERROR_NOCAMERA";
      case DRV_NOT_INITIALIZED:
         return "DRV_NOT_INITIALIZED";
      case DRV_ACQUIRING:
         return "DRV_ACQUIRING";
      case DRV_P1INVALID:
         return "DRV_P1INVALID";
      case DRV_NOT_SUPPORTED:
         return "DRV_NOT_SUPPORTED";
      default:
         return "UNKNOWN";
   }
}
 
inline
int andorCtrl::cameraSelect()
{
   unsigned int error;
   
   if(!m_libInit)
   {
      char path[] = "/usr/local/etc/andor/";
      error = Initialize(path);

      if(error == DRV_USBERROR || error == DRV_ERROR_NOCAMERA || error == DRV_VXDNOTINSTALLED)
      {
         state(stateCodes::NODEVICE);
         if(!stateLogged())
         {
            log<text_log>("No Andor USB camera found", logPrio::LOG_WARNING);
         }
         
         //Not an error, appLogic should just go on.
         return 0;
      }
      else if(error!=DRV_SUCCESS)
      {
         log<software_critical>({__FILE__, __LINE__, "ANDOR SDK initialization failed:" + andorSDKErrorName(error)});
         return -1;
      }
      
      m_libInit = true;
   }
   
   at_32 lNumCameras = 0;
   error = GetAvailableCameras(&lNumCameras);

   if(error != DRV_SUCCESS)
   {
      log<software_critical>({__FILE__, __LINE__, "ANDOR SDK GetAvailableCameras failed."});
      return -1;
   }
   
   if(lNumCameras < 1)
   {
      if(!stateLogged())
      {
         log<text_log>("No Andor cameras found after initialization", logPrio::LOG_WARNING);
      }
      state(stateCodes::NODEVICE);
      return 0;
   }
   
   int iSelectedCamera = 0; //We're hard-coded for just one camera!

   int serialNumber = 0;
   error = GetCameraSerialNumber(&serialNumber);
   
   if(error != DRV_SUCCESS)
   {
      log<software_critical>({__FILE__, __LINE__, "ANDOR SDK GetCameraSerialNumber failed."});
      return -1;
   }
   
   log<text_log>(std::string("Found Andor USB Camera with serial number ") + std::to_string(serialNumber));
   
   at_32 lCameraHandle;
   error = GetCameraHandle(iSelectedCamera, &lCameraHandle);

   if(error != DRV_SUCCESS)
   {
      log<software_critical>({__FILE__, __LINE__, "ANDOR SDK GetCameraHandle failed."});
      return -1;
   }
   
   error = SetCurrentCamera(lCameraHandle);

   if(error != DRV_SUCCESS)
   {
      log<software_critical>({__FILE__, __LINE__, "ANDOR SDK SetCurrentCamera failed."});
      return -1;
   }
   
   char name[MAX_PATH];
   
   error = GetHeadModel(name);
   if(error != DRV_SUCCESS)
   {
      log<software_critical>({__FILE__, __LINE__, "ANDOR SDK GetHeadModel failed."});
      return -1;
   }

   state(stateCodes::CONNECTED);
   log<text_log>(std::string("Connected to ") + name +  " with serial number " + std::to_string(serialNumber));
   
   //Initialize Shutter to SHUT
   int ss = 2;
   if(m_shutterState == 1) ss = 1;
   else m_shutterState = 0; //handles startup case
   error = SetShutter(1,ss,50,50);
   if(error != DRV_SUCCESS)
   {
      log<software_critical>({__FILE__, __LINE__, "ANDOR SDK SetShutter failed."});
      return -1;
   }
   
   // Set CameraLink
   error = SetCameraLinkMode(1);
   if(error != DRV_SUCCESS)
   {
      log<software_critical>({__FILE__, __LINE__, "ANDOR SDK SetCameraLinkMode failed."});
      return -1;
   }
   
   //Set Read Mode to --Image--
   /* 0 - Full Vertical Binning
    * 1 - Multi-Track; Need to call SetMultiTrack(int NumTracks, int height, int offset, int* bottom, int *gap)
    * 2 - Random-Track; Need to call SetRandomTracks
    * 3 - Single-Track; Need to call SetSingleTrack(int center, int height)
    * 4 - Image; See SetImage, need shutter during readout
    */
   error = SetReadMode(4);
   if(error != DRV_SUCCESS)
   {
      return log<software_error,-1>({__FILE__, __LINE__, "Andor SDK Error from SetReadMode: " + andorSDKErrorName(error)});
   }
   
   //Set Acquisition mode to --Run Till Abort--
   /* 1 - Single Scan
    * 2 - Accumulate
    * 3 - Kinetic Series
    * 5 - Run Till Abort
    *
    * See Page 53 of SDK User's Guide for Frame Transfer Info
    */
   error = SetAcquisitionMode(5);
   if(error != DRV_SUCCESS)
   {
      return log<software_error,-1>({__FILE__, __LINE__, "Andor SDK Error from SetAcquisitionMode: " + andorSDKErrorName(error)});
   }

   //Set to frame transfer mode
   /* See Page 53 of SDK User's Guide for Frame Transfer Info
    */
   error = SetFrameTransferMode(1);
   if(error != DRV_SUCCESS)
   {
      return log<software_error,-1>({__FILE__, __LINE__, "Andor SDK Error from SetFrameTransferMode: " + andorSDKErrorName(error)});
   }
   
   //Set initial exposure time
   error = SetExposureTime(0.1);
   if(error != DRV_SUCCESS)
   {
      return log<software_error,-1>({__FILE__, __LINE__, "Andor SDK Error from SetExposureTime: " + andorSDKErrorName(error)});
   }
   
   return 0;

}

inline
int andorCtrl::getTemp()
{
   //unsigned int error;
   //int temp_low {999}, temp_high {999};
   //error = GetTemperatureRange(&temp_low, &temp_high); 

   float temp = -999;
   unsigned int status = GetTemperatureF(&temp);
   
   std::string cooling;
   switch(status)
   {
      case DRV_TEMPERATURE_OFF: 
         m_tempControlStatusStr =  "OFF"; 
         m_tempControlStatus = false;
         m_tempControlOnTarget = false;
         break;
      case DRV_TEMPERATURE_STABILIZED: 
         m_tempControlStatusStr = "STABILIZED"; 
         m_tempControlStatus = true;
         m_tempControlOnTarget = true;
         break;
      case DRV_TEMPERATURE_NOT_REACHED: 
         m_tempControlStatusStr = "COOLING";
         m_tempControlStatus = true;
         m_tempControlOnTarget = false;
         break;
      case DRV_TEMPERATURE_NOT_STABILIZED: 
         m_tempControlStatusStr = "NOT STABILIZED";
         m_tempControlStatus = true;
         m_tempControlOnTarget = false;
         break;
      case DRV_TEMPERATURE_DRIFT: 
         m_tempControlStatusStr = "DRIFTING";
         m_tempControlStatus = true;
         m_tempControlOnTarget = false;
         break;
      default: 
         m_tempControlStatusStr =  "UNKOWN";
         m_tempControlStatus = false;
         m_tempControlOnTarget = false;
         m_ccdTemp = -999;
         log<software_error>({__FILE__, __LINE__, "ANDOR SDK GetTemperatureF:" + andorSDKErrorName(status)});
         return -1;
   }

   m_ccdTemp = temp;
   recordCamera();
      
   return 0;

}

inline
int andorCtrl::getEMGain()
{
   int state;
   int gain;
   int low, high;

   ///\todo what is EM advanced?
   if(GetEMAdvanced(&state) != DRV_SUCCESS)
   {
      log<software_error>({__FILE__,__LINE__, "error getting em advanced"});
      return -1;
   }

   if(GetEMCCDGain(&gain) !=DRV_SUCCESS)
   {
      log<software_error>({__FILE__,__LINE__, "error getting em gain"});
      return -1;
   }

   m_emGain = gain;

   ///\todo this needs to be done on connection, and the max/min field updated.
   if(GetEMGainRange(&low, &high) !=DRV_SUCCESS)
   {
      log<software_error>({__FILE__,__LINE__, "error getting em gain range"});
      return -1;
   }

   return 0;
}

inline
int andorCtrl::setEMGain( unsigned emg )
{

}

inline
int andorCtrl::setShutter( unsigned os )
{
   recordCamera(true);
   AbortAcquisition();
   state(stateCodes::CONFIGURING);

   if(os == 0) //Shut
   {
      SetShutter(1,2,50,50);
      m_shutterState = 0;
   }
   else //Open
   {
      SetShutter(1,1,50,50);
      m_shutterState = 1;
   }

   m_nextMode = m_modeName;
   m_reconfig = true;

   return 0;
}

inline 
int andorCtrl::writeConfig()
{
   std::ofstream fout;
   fout.open(m_configFile);
   
   if(fout.fail())
   {
      log<software_error>({__FILE__, __LINE__, "error opening config file for writing"});
      return -1;
   }
   
   fout << "camera_class:                  \"Andor\"\n";
   fout << "camera_model:                  \"iXon Ultra 897\"\n";
   fout << "camera_info:                   \"512x512 (1-tap, freerun)\"\n";
   fout << "width:                         " << m_nextROI.w << "\n";
   fout << "height:                        " << m_nextROI.h << "\n";
   fout << "depth:                         16\n";
   fout << "extdepth:                      16\n";
   fout << "CL_DATA_PATH_NORM:             0f       # single tap\n";
   fout << "CL_CFG_NORM:                   02\n";
   
   fout.close();
   
   return 0;

}
//------------------------------------------------------------------------
//-----------------------  stdCamera interface ---------------------------
//------------------------------------------------------------------------

inline
int andorCtrl::powerOnDefaults()
{
   //Camera boots up with this true in most cases.
   m_tempControlStatus = false;
   m_tempControlStatusSet = false;
   m_tempControlStatusStr =  "OFF"; 
   m_tempControlOnTarget = false;
      
   m_currentROI.x = m_startup_x;
   m_currentROI.y = m_startup_y;
   m_currentROI.w = m_startup_w;
   m_currentROI.h = m_startup_h;
   m_currentROI.bin_x = 1;
   m_currentROI.bin_y = 1;
   
   m_nextROI.x = m_startup_x;
   m_nextROI.y = m_startup_y;
   m_nextROI.w = m_startup_w;
   m_nextROI.h = m_startup_h;
   m_nextROI.bin_x = 1;
   m_nextROI.bin_y = 1;
   
   return 0;
}

inline
int andorCtrl::setTempControl()
{  
   if(m_tempControlStatusSet)
   {
      unsigned int error = CoolerON();
      if(error != DRV_SUCCESS)
      {
         log<software_critical>({__FILE__, __LINE__, "ANDOR SDK CoolerOFF failed: " + andorSDKErrorName(error)});
         return -1;
      }
      m_tempControlStatus = true;
      m_tempControlStatusStr = "COOLING";
      recordCamera();
      log<text_log>("enabled temperature control");
      return 0;
   }
   else
   {
      unsigned int error = CoolerOFF();
      if(error != DRV_SUCCESS)
      {
         log<software_critical>({__FILE__, __LINE__, "ANDOR SDK CoolerOFF failed: " + andorSDKErrorName(error)});
         return -1;
      }
      m_tempControlStatus = false;
      m_tempControlStatusStr = "OFF";
      recordCamera();
      log<text_log>("disabled temperature control");
      return 0;
   }
}

inline
int andorCtrl::setTempSetPt()
{
   int temp = m_ccdTempSetpt + 0.5;
   
   unsigned int error = SetTemperature(temp);
   
   if(error != DRV_SUCCESS)
   {
      log<software_critical>({__FILE__, __LINE__, "ANDOR SDK setTemperature failed: " + andorSDKErrorName(error)});
      return -1;
   }
   
  return 0;

}

inline
int andorCtrl::getFPS()
{
   float exptime;
   float accumCycletime;
   float kinCycletime;

   unsigned int error = GetAcquisitionTimings(&exptime, &accumCycletime, &kinCycletime);
   if(error != DRV_SUCCESS)
   {
      return log<software_error,-1>({__FILE__, __LINE__, "ANDOR SDK error from GetAcquisitionTimings: " + andorSDKErrorName(error)});
   }

   m_expTime = exptime;
   
   //std::cerr << accumCycletime << " " << kinCycletime << "\n";
   
   float readoutTime;
   error = GetReadOutTime(&readoutTime);
   if(error != DRV_SUCCESS)
   {
      return log<software_error,-1>({__FILE__, __LINE__, "ANDOR SDK error from GetReadOutTime: " + andorSDKErrorName(error)});
   }
   
   //if(readoutTime < exptime) m_fps = 1./m_expTime;
   //else m_fps = 1.0/readoutTime;
   m_fps = 1.0/accumCycletime;
   
   return 0;

}

inline
int andorCtrl::setFPS()
{
   recordCamera(true);
   AbortAcquisition();
   state(stateCodes::CONFIGURING);

   unsigned long err = SetExposureTime(1.0/m_fpsSet);

   if(err != DRV_SUCCESS)
   {
      return log<software_error, -1>({__FILE__, __LINE__, "error from SetExposureTime"});
   }
   m_nextMode = m_modeName;
   m_reconfig = true;
   
   return 0;

}



inline 
int andorCtrl::setExpTime()
{
   recordCamera(true);
   AbortAcquisition();
   state(stateCodes::CONFIGURING);
   
   unsigned int error = SetExposureTime(m_expTimeSet);
   if(error != DRV_SUCCESS)
   {
      log<software_critical>({__FILE__, __LINE__, "ANDOR SDK SetExposureTime failed: " + andorSDKErrorName(error)});
      return -1;
   }
   m_nextMode = m_modeName;
   m_reconfig = true;
   return 0;
}
   
inline 
int andorCtrl::setNextROI()
{
   std::cerr << "setNextROI:\n";
   std::cerr << "  m_nextROI.x = " << m_nextROI.x << "\n";
   std::cerr << "  m_nextROI.y = " << m_nextROI.y << "\n";
   std::cerr << "  m_nextROI.w = " << m_nextROI.w << "\n";
   std::cerr << "  m_nextROI.h = " << m_nextROI.h << "\n";
   std::cerr << "  m_nextROI.bin_x = " << m_nextROI.bin_x << "\n";
   std::cerr << "  m_nextROI.bin_y = " << m_nextROI.bin_y << "\n";
 
   recordCamera(true);
   AbortAcquisition();
   state(stateCodes::CONFIGURING);
   
   m_nextMode = m_modeName;
   m_reconfig = true;

   updateSwitchIfChanged(m_indiP_roi_set, "request", pcf::IndiElement::Off, INDI_IDLE);
   
   return 0;
}

//------------------------------------------------------------------------
//-------------------   framegrabber interface ---------------------------
//------------------------------------------------------------------------

inline
int andorCtrl::configureAcquisition()
{
   //lock mutex
   std::unique_lock<std::mutex> lock(m_indiMutex);

   unsigned int error;
   
//    //Get Detector dimensions
//    int width, height;
//    error = GetDetector(&width, &height);
//    if(error != DRV_SUCCESS)
//    {
//       return log<software_error,-1>({__FILE__, __LINE__, "Andor SDK Error from GetDetector: " + andorSDKErrorName(error)});
//    }

   //SetNumberAccumulations(1);
   //SetKineticCycleTime(0);

   // Set Output Amplifier
   SetOutputAmplifier(1);

    
   int x0 = (m_nextROI.x - 0.5*(m_nextROI.w - 1)) + 1;
   int y0 = (m_nextROI.y - 0.5*(m_nextROI.h - 1)) + 1;
    
   //Setup Image dimensions
   /* SetImage(int hbin, int vbin, int hstart, int hend, int vstart, int vend)
    * hbin: number of pixels to bin horizontally
    * vbin: number of pixels to bin vertically
    * hstart: Starting Column (inclusive)
    * hend: End column (inclusive)
    * vstart: Start row (inclusive)
    * vend: End row (inclusive)
    */
   error = SetImage(m_nextROI.bin_x, m_nextROI.bin_y, x0, x0 + m_nextROI.w - 1, y0, y0 + m_nextROI.h - 1);
   if(error != DRV_SUCCESS)
   {
      if(error == DRV_P1INVALID)
      {
         log<text_log>(std::string("invalid x-binning: ") + std::to_string(m_nextROI.bin_x), logPrio::LOG_ERROR);
      }
      else if(error == DRV_P2INVALID)
      {
         log<text_log>(std::string("invalid y-binning: ") + std::to_string(m_nextROI.bin_y), logPrio::LOG_ERROR);
      }
      else if(error == DRV_P3INVALID)
      {
         log<text_log>(std::string("invalid x-center: ") + std::to_string(m_nextROI.x) + "/" + std::to_string(x0), logPrio::LOG_ERROR);
      }
      else if(error == DRV_P4INVALID)
      {
         log<text_log>(std::string("invalid width: ") + std::to_string(m_nextROI.w), logPrio::LOG_ERROR);
      }
      else if(error == DRV_P5INVALID)
      {
         log<text_log>(std::string("invalid y-center: ") + std::to_string(m_nextROI.y) + "/" + std::to_string(y0), logPrio::LOG_ERROR);
      }
      else if(error == DRV_P6INVALID)
      {
         log<text_log>(std::string("invalid height: ") + std::to_string(m_nextROI.h), logPrio::LOG_ERROR);
      }
      else
      {
         return log<software_error,-1>({__FILE__, __LINE__, "Andor SDK Error from SetImage: " + andorSDKErrorName(error)});
      }
   
      m_nextROI.x = m_currentROI.x;
      m_nextROI.y = m_currentROI.y;
      m_nextROI.w = m_currentROI.w;
      m_nextROI.h = m_currentROI.h;
      m_nextROI.bin_x = m_currentROI.bin_x;
      m_nextROI.bin_y = m_currentROI.bin_y;
            
      m_nextMode = m_modeName;

      state(stateCodes::ERROR);
      return -1;
   
   }
   
    = m_nextROI.x 
   
   m_currentROI.bin_x = m_nextROI.bin_x;
   m_currentROI.bin_y = m_nextROI.bin_y;
   m_currentROI.x = x0 - 1.0 +  0.5*(m_nextROI.w - 1);
   m_currentROI.y = y0 - 1.0 +  0.5*(m_nextROI.h - 1);
   m_currentROI.w = m_nextROI.w;
   m_currentROI.h = m_nextROI.h;
   
   updateIfChanged( m_indiP_roi_x, "current", m_currentROI.x, INDI_OK);
   updateIfChanged( m_indiP_roi_y, "current", m_currentROI.y, INDI_OK);
   updateIfChanged( m_indiP_roi_w, "current", m_currentROI.w, INDI_OK);
   updateIfChanged( m_indiP_roi_h, "current", m_currentROI.h, INDI_OK);
   updateIfChanged( m_indiP_roi_bin_x, "current", m_currentROI.bin_x, INDI_OK);
   updateIfChanged( m_indiP_roi_bin_y, "current", m_currentROI.bin_y, INDI_OK);
   
       ///\todo This should check whether we have a match between EDT and the camera right?
   m_width = m_currentROI.w;
   m_height = m_currentROI.h;
   m_dataType = _DATATYPE_INT16;

   
   // Print Detector Frame Size
   //std::cout << "Detector Frame is: " << width << "x" << height << "\n";

    

   return 0;
}

inline
float andorCtrl::fps()
{
   return m_fps;
}

inline
int andorCtrl::startAcquisition()
{
   StartAcquisition();
   state(stateCodes::OPERATING);
   recordCamera();
   
   return edtCamera<andorCtrl>::pdvStartAcquisition();
}

inline
int andorCtrl::acquireAndCheckValid()
{
   return edtCamera<andorCtrl>::pdvAcquire( m_currImageTimestamp );

}

inline
int andorCtrl::loadImageIntoStream(void * dest)
{
   memcpy(dest, m_image_p, m_width*m_height*m_typeSize);

   return 0;
}

inline
int andorCtrl::reconfig()
{
   //lock mutex
   std::unique_lock<std::mutex> lock(m_indiMutex);

   writeConfig();
   
   int rv = edtCamera<andorCtrl>::pdvReconfig();
   if(rv < 0) return rv;
   
   state(stateCodes::READY);
   return 0;
}

INDI_NEWCALLBACK_DEFN(andorCtrl, m_indiP_emGain)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_emGain.getName())
   {
      unsigned current = 0, target = 0;

      if(ipRecv.find("current"))
      {
         current = ipRecv["current"].get<unsigned>();
      }

      if(ipRecv.find("target"))
      {
         target = ipRecv["target"].get<unsigned>();
      }

      if(target == 0) target = current;

      if(target == 0) return 0;

      //Lock the mutex, waiting if necessary
      std::unique_lock<std::mutex> lock(m_indiMutex);

      updateIfChanged(m_indiP_emGain, "target", target);

      return setEMGain(target);

   }
   return -1;
}

inline
int andorCtrl::checkRecordTimes()
{
   return telemeter<andorCtrl>::checkRecordTimes(telem_stdcam());
}
  
inline
int andorCtrl::recordTelem( const telem_stdcam * )
{
   return recordCamera(true);
}

}//namespace app
} //namespace MagAOX
#endif
