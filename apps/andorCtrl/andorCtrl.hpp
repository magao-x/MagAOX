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
class andorCtrl : public MagAOXApp<>, public dev::stdCamera<andorCtrl>, public dev::edtCamera<andorCtrl>, public dev::frameGrabber<andorCtrl>
{

   friend class dev::stdCamera<andorCtrl>;
   friend class dev::edtCamera<andorCtrl>;
   friend class dev::frameGrabber<andorCtrl>;

protected:

   /** \name configurable parameters
     *@{
     */

   //Camera:
   unsigned long m_powerOnWait {10}; ///< Time in sec to wait for camera boot after power on.

   float m_startupTemp {20.0}; ///< The temperature to set after a power-on.

   unsigned m_maxEMGain {600};

   ///@}


   unsigned m_emGain {1};

   float m_expTime {0};
   float m_fpsSet {0};
   float m_fpsTgt {0};
   float m_fpsMeasured {0};

   bool m_shutter {false};

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




   int cameraSelect(int camNo);

   
   
   
   
   
   
   
   
   
   
   
   
   
   int getTemp();

   int setTemp(float temp);

   int getFPS();

   int setFPS(float fps);

   int getEMGain();

   int setEMGain( unsigned emg );

   int getShutter();

   int setShutter( unsigned os);

   
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
   int startAcquisition();
   int acquireAndCheckValid();
   int loadImageIntoStream(void * dest);
   int reconfig();


   //INDI:
protected:
  
   pcf::IndiProperty m_indiP_emGain;

   pcf::IndiProperty m_indiP_shutter;

public:
   INDI_NEWCALLBACK_DECL(andorCtrl, m_indiP_emGain);

   INDI_NEWCALLBACK_DECL(andorCtrl, m_indiP_shutter);
};

inline
andorCtrl::andorCtrl() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   m_powerMgtEnabled = true;
   m_powerOnWait = 10;
   
   m_startupTemp = 20;
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
   dev::stdCamera<andorCtrl>::setupConfig(config);
   dev::edtCamera<andorCtrl>::setupConfig(config);
   
   config.add("camera.maxEMGain", "", "camera.maxEMGain", argType::Required, "camera", "maxEMGain", false, "unsigned", "The maximum EM gain which can be set by  user. Default is 600.  Min is 1, max is 600.");

   dev::frameGrabber<andorCtrl>::setupConfig(config);

   

}



inline
void andorCtrl::loadConfig()
{
   dev::stdCamera<andorCtrl>::loadConfig(config);
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




}



inline
int andorCtrl::appStartup()
{
   
   REG_INDI_NEWPROP(m_indiP_emGain, "emgain", pcf::IndiProperty::Number);
   m_indiP_emGain.add (pcf::IndiElement("current"));
   m_indiP_emGain["current"].set(m_emGain);
   m_indiP_emGain.add (pcf::IndiElement("target"));

   REG_INDI_NEWPROP(m_indiP_shutter, "shutter", pcf::IndiProperty::Text);
   m_indiP_shutter.add (pcf::IndiElement("current"));
   m_indiP_shutter["current"].set("UNKNOWN");
   m_indiP_shutter.add (pcf::IndiElement("target"));

//    if(pdvConfig(m_startupMode) < 0)
//    {
//       log<software_error>({__FILE__, __LINE__});
//       return -1;
//    }
   
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
   
   if( state() == stateCodes::NOTCONNECTED || state() == stateCodes::ERROR)
   {
      std::string response;

      //Might have gotten here because of a power off.
      if(m_powerState == 0) return 0;

      int ret = cameraSelect(0); ///\todo make camera number configurable

      if( ret != 0) //Probably not powered on yet.
      {
         sleep(1);
         return 0;
      }

      int error = Initialize((char *)"/usr/local/etc/andor");

      std::cout << "The Error code is " << error << std::endl;

      if(error!=DRV_SUCCESS)
      {
		std::cerr << "Initialisation error...exiting" << std::endl;
		return -1;
	}

      state(stateCodes::CONNECTED);


   }

   if( state() == stateCodes::CONNECTED )
   {
      //Get a lock
      std::unique_lock<std::mutex> lock(m_indiMutex);



      state(stateCodes::READY);
//       if( getFPS() == 0 )
//       {
//          if(m_fpsSet == 0) state(stateCodes::READY);
//          else state(stateCodes::OPERATING);
//
//          if(setTemp(m_startupTemp) < 0)
//          {
//             return log<software_error,0>({__FILE__,__LINE__});
//          }
//       }
//       else
//       {
//          state(stateCodes::ERROR);
//          return log<software_error,0>({__FILE__,__LINE__});
//       }
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

      if(m_shutter == false)
      {
         updateIfChanged(m_indiP_shutter, "current", std::string("SHUT"));
         updateIfChanged(m_indiP_shutter, "target", std::string(""));
      }
      else
      {
         updateIfChanged(m_indiP_shutter, "current", std::string("OPEN"));
         updateIfChanged(m_indiP_shutter, "target", std::string(""));
      }
      /*if(frameGrabber<andorCtrl>::updateINDI() < 0)
      {
         log<software_error>({__FILE__, __LINE__});
         state(stateCodes::ERROR);
         return 0;
      }*/
   }

   //Fall through check?

   return 0;

}

inline
int andorCtrl::onPowerOff()
{
   m_powerOnCounter = 0;

   std::lock_guard<std::mutex> lock(m_indiMutex);

   updateIfChanged(m_indiP_emGain, "current", 0);
   updateIfChanged(m_indiP_emGain, "target", 0);

   edtCamera<andorCtrl>::onPowerOff();
   return 0;
}

inline
int andorCtrl::whilePowerOff()
{
   return 0;
}

inline
int andorCtrl::appShutdown()
{
   dev::frameGrabber<andorCtrl>::appShutdown();

   return 0;
}

inline
int andorCtrl::cameraSelect(int camNo)
{
   std::cerr << "In cameraSelect(0) \n";
   at_32 lNumCameras;
   GetAvailableCameras(&lNumCameras);

   std::cerr << "Number of cameras: " << lNumCameras << "\n";

   int iSelectedCamera = camNo;

   if (iSelectedCamera < lNumCameras && iSelectedCamera >= 0)
   {
      at_32 lCameraHandle;
      GetCameraHandle(iSelectedCamera, &lCameraHandle);

      SetCurrentCamera(lCameraHandle);

      return 0;
   }
   else
   {
      return log<text_log,-1>("No Andor cameras found.");
   }

}

inline
int andorCtrl::getTemp()
{
   std::string response;

   int temp {999}, temp_low {999}, temp_high {999};
   unsigned long error=GetTemperatureRange(&temp_low, &temp_high); ///\todo need error check

//   std::cerr << error << "\n";
   unsigned long status=GetTemperature(&temp);


//    std::cout << "Current Temperature: " << temp << " C" << std::endl;
//    std::cout << "Temp Range: {" << temp_low << "," << temp_high << "}" << std::endl;
//    std::cout << "Status             : ";
   std::string cooling;
   switch(status)
   {
      case DRV_TEMPERATURE_OFF: cooling =  "OFF"; break;
      case DRV_TEMPERATURE_STABILIZED: cooling = "STABILIZED"; break;
      case DRV_TEMPERATURE_NOT_REACHED: cooling = "COOLING"; break;
      case DRV_TEMPERATURE_NOT_STABILIZED: cooling = "NOT STABILIZED"; break;
      case DRV_TEMPERATURE_DRIFT: cooling = "DRIFTING"; break;
      default: cooling =  "UNKOWN";
   }
  
   return 0;


}

inline
int andorCtrl::setTemp(float temp)
{
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
   AbortAcquisition();

   if(os == 0)
   {
      SetShutter(1,2,50,50);
      m_shutter = 0;
   }
   else
   {
      SetShutter(1,1,50,50);
      m_shutter = 1;
   }

   StartAcquisition();

   return 0;
}

//------------------------------------------------------------------------
//-----------------------  stdCamera interface ---------------------------
//------------------------------------------------------------------------

inline
int andorCtrl::powerOnDefaults()
{
   //Camera boots up with this true in most cases.
   m_tempControlStatusSet = false;
   m_tempControlStatus =false;
      
   return 0;
}

inline
int andorCtrl::setTempControl()
{  
   return 0;
}

inline
int andorCtrl::setTempSetPt()
{
  return 0;

}

inline
int andorCtrl::getFPS()
{
   float exptime;
   float accumCycletime;
   float kinCycletime;

   unsigned long error = GetAcquisitionTimings(&exptime, &accumCycletime, &kinCycletime);
   if(error != DRV_SUCCESS)
   {
      return log<software_error,-1>({__FILE__, __LINE__, "Error from GetAcquisitionTimings"});
   }

   m_fps = 1./exptime;

   return 0;

}

inline
int andorCtrl::setFPS()
{
   AbortAcquisition();

   unsigned long err = SetExposureTime(1.0/m_fpsSet);

   StartAcquisition();


   if(err != DRV_SUCCESS)
   {
      return log<software_error, -1>({__FILE__, __LINE__, "error from SetExposureTime"});
   }
   
   log<text_log>({"set fps " + std::to_string(m_fpsSet)});
   
   return 0;

}



inline 
int andorCtrl::setExpTime()
{
   return 0;
}
   
inline 
int andorCtrl::setNextROI()
{
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






    //Get Detector dimensions
    int width, height;
    GetDetector(&width, &height);

    ///\todo This should check whether we have a match between EDT and the camera right?
    m_width = width;
    m_height = height;
    m_dataType = _DATATYPE_INT16;

    //Set Read Mode to --Image--
    SetReadMode(4);
    /* 0 - Full Vertical Binning
     * 1 - Multi-Track; Need to call SetMultiTrack(int NumTracks, int height, int offset, int* bottom, int *gap)
     * 2 - Random-Track; Need to call SetRandomTracks
     * 3 - Single-Track; Need to call SetSingleTrack(int center, int height)
     * 4 - Image; See SetImage, need shutter during readout
     */

    //Set Acquisition mode to --Run Till Abort--
    SetAcquisitionMode(5);
    /* 1 - Single Scan
     * 2 - Accumulate
     * 3 - Kinetic Series
     * 5 - Run Till Abort
     *
     * See Page 53 of SDK User's Guide for Frame Transfer Info
     */

    //Set initial exposure time
    SetExposureTime(0.1);

    //Initialize Shutter to SHUT
    int ss = 2;
    if(m_shutter) ss = 1;
    SetShutter(1,ss,50,50);

    SetNumberAccumulations(1);
    SetFrameTransferMode(1);

    // Set CameraLink
    SetCameraLinkMode(1);

    // Set Output Amplifier
    SetOutputAmplifier(1);

    //Setup Image dimensions
    SetImage(1,1,1,width,1,height);
    /* SetImage(int hbin, int vbin, int hstart, int hend, int vstart, int vend)
     * hbin: number of pixels to bin horizontally
     * vbin: number of pixels to bin vertically
     * hstart: Starting Column (inclusive)
     * hend: End column (inclusive)
     * vstart: Start row (inclusive)
     * vend: End row (inclusive)
     */

    // Print Detector Frame Size
    std::cout << "Detector Frame is: " << width << "x" << height << "\n";


   return 0;
}

inline
int andorCtrl::startAcquisition()
{
   SetKineticCycleTime(0);
   StartAcquisition();

   std::cout << "\n" << "Starting Continuous Acquisition" << "\n";

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

   return edtCamera<andorCtrl>::pdvReconfig();
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

INDI_NEWCALLBACK_DEFN(andorCtrl, m_indiP_shutter)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_shutter.getName())
   {
      std::string current, target;

      if(ipRecv.find("current"))
      {
         current = ipRecv["current"].get<std::string>();
      }

      if(ipRecv.find("target"))
      {
         target = ipRecv["target"].get<std::string>();
      }

      if(target == "") target = current;

      target = mx::ioutils::toUpper(target);

      if(target != "OPEN" && target != "SHUT")
      {
         return log<software_error,-1>({__FILE__, __LINE__, "invalid shutter request"});
      }
      else
      {

         //Lock the mutex, waiting if necessary
         std::unique_lock<std::mutex> lock(m_indiMutex);

         updateIfChanged(m_indiP_shutter, "target", target);
      }

      int os = 0;
      if(target == "OPEN") os = 1;
      return setShutter(os);

   }
   return -1;
}

}//namespace app
} //namespace MagAOX
#endif
