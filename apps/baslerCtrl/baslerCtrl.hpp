/** \file baslerCtrl.hpp
  * \brief The MagAO-X basler camera controller.
  *
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup baslerCtrl_files
  */

#ifndef baslerCtrl_hpp
#define baslerCtrl_hpp



#include <pylon/PylonIncludes.h>
#include <pylon/PixelData.h>
#include <pylon/GrabResultData.h>
#include <pylon/usb/BaslerUsbInstantCamera.h>
#include <pylon/usb/_BaslerUsbCameraParams.h>
#include <GenApi/IFloat.h>

typedef Pylon::CBaslerUsbInstantCamera Camera_t;
typedef int16_t pixelT;

using namespace Basler_UsbCameraParams;

using namespace Pylon;

#include <ImageStruct.h>
#include <ImageStreamIO.h>

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

namespace MagAOX
{
namespace app
{

/** \defgroup baslerCtrl Basler USB3 Camera
  * \brief Control of a Basler USB3 Camera
  *
  *  <a href="../apps_html/page_module_baslerCtrl.html">Application Documentation</a>
  *
  * \ingroup apps
  *
  */

/** \defgroup baslerCtrl_files Basler USB3 Camera Files
  * \ingroup baslerCtrl
  */

/** MagAO-X application to control a Basler USB3 Camera
  *
  * \ingroup baslerCtrl
  * 
  */
class baslerCtrl : public MagAOXApp<>, public dev::frameGrabber<baslerCtrl>
{

   friend class dev::frameGrabber<baslerCtrl>;
   
protected:

   /** \name configurable parameters 
     *@{
     */ 
   std::string m_serialNumber; ///< The camera's identifying serial number
   
   
   ///@}

   float m_ccdTemp;
   
   float m_expTimeSet {0}; ///< The exposure time, in seconds, as set by user.
   float m_fpsSet {0}; ///< The commanded fps, as set by user.

   
   CBaslerUsbInstantCamera * camera {nullptr};
   CGrabResultPtr ptrGrabResult;
   
public:

   ///Default c'tor
   baslerCtrl();

   ///Destructor
   ~baslerCtrl() noexcept;

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

   /// Do any needed shutdown tasks.  Currently nothing in this app.
   virtual int appShutdown();

   int connect();
   
   int startAcquisition();
   int acquireAndCheckValid();
   int loadImageIntoStream(void * dest);
   int reconfig();
   
protected:
   
   
   int getTemp();
      
   int getExpTime();
   
   int setExpTime(double exptime);
   
   int setFPS(double fps);
   
   
   //INDI:
protected:
   //declare our properties
   pcf::IndiProperty m_indiP_temp;
   
//   pcf::IndiProperty m_indiP_mode;
   
   pcf::IndiProperty m_indiP_exptime;
   pcf::IndiProperty m_indiP_fps;

public:
   INDI_NEWCALLBACK_DECL(baslerCtrl, m_indiP_exptime);
   INDI_NEWCALLBACK_DECL(baslerCtrl, m_indiP_fps);
   

};

inline
baslerCtrl::baslerCtrl() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   m_powerMgtEnabled = false;
   
   return;
}

inline
baslerCtrl::~baslerCtrl() noexcept
{
   return;
}

inline
void baslerCtrl::setupConfig()
{
   
   config.add("camera.serialNumber", "", "camera.serialNumber", argType::Required, "camera", "serialNumber", false, "int", "The identifying serial number of the camera.");
   
   dev::frameGrabber<baslerCtrl>::setupConfig(config);
}

inline
void baslerCtrl::loadConfig()
{
   config(m_serialNumber, "camera.serialNumber");
   
   dev::frameGrabber<baslerCtrl>::loadConfig(config);
}
   

inline
int baslerCtrl::appStartup()
{
   // set up the  INDI properties
   REG_INDI_NEWPROP_NOCB(m_indiP_temp, "temp", pcf::IndiProperty::Number);
   m_indiP_temp.add (pcf::IndiElement("current"));
   m_indiP_temp["current"].set(0);

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
  
   PylonInitialize(); // Initializes pylon runtime before using any pylon methods

   if(dev::frameGrabber<baslerCtrl>::appStartup() < 0)
   {
      return log<software_critical,-1>({__FILE__,__LINE__});
   }
   
   state(stateCodes::NOTCONNECTED);
   
   return 0;

}

inline
int baslerCtrl::appLogic()
{
   //first run frameGrabber's appLogic to see if the f.g. thread has exited.
   if(dev::frameGrabber<baslerCtrl>::appLogic() < 0)
   {
      return log<software_error, -1>({__FILE__, __LINE__});
   }
   
   
   if( state() == stateCodes::NOTCONNECTED || state() == stateCodes::NODEVICE || state() == stateCodes::ERROR)
   {
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
         if(state() == stateCodes::READY || state() == stateCodes::OPERATING)  state(stateCodes::ERROR);
         return 0;
      }
      
      if(getExpTime() < 0)
      {
         if(state() == stateCodes::READY || state() == stateCodes::OPERATING)  state(stateCodes::ERROR);
         return 0;
      }

      if(frameGrabber<baslerCtrl>::updateINDI() < 0)
      {
         log<software_error>({__FILE__, __LINE__});
         state(stateCodes::ERROR);
         return 0;
      }
   }

   ///\todo Fall through check?

   return 0;

}


inline
int baslerCtrl::appShutdown()
{
   dev::frameGrabber<baslerCtrl>::appShutdown();
   

      
    
   return 0;
}


inline
int baslerCtrl::connect()
{
   CDeviceInfo info;
   //info.SetDeviceClass(Camera_t::DeviceClass());
   info.SetSerialNumber(m_serialNumber.c_str());
   
   try 
   {
      if(camera) delete camera;
      camera = nullptr;
      
      camera = new CBaslerUsbInstantCamera( CTlFactory::GetInstance().CreateFirstDevice(info) );
      
      if(m_shmimName == "")
      {
         m_shmimName = (std::string)camera->GetDeviceInfo().GetModelName() + "_" + (std::string)camera->GetDeviceInfo().GetSerialNumber(); // Gets camera model name and serial number
      }
      
      camera->RegisterConfiguration( new CAcquireContinuousConfiguration , RegistrationMode_ReplaceAll, Cleanup_Delete);
      
      camera->Open(); // Opens camera parameters to grab images and set exposure time
   
      camera->ExposureAuto.SetValue(ExposureAuto_Off); 
   
      camera->PixelFormat.SetValue(PixelFormat_Mono10); // Set to 10 bits
   
      state(stateCodes::CONNECTED);
      if(!stateLogged())
      {
         log<text_log>("Found camera of type " + (std::string)camera->GetDeviceInfo().GetModelName() + " with serial number " + m_serialNumber + ".");
         log<text_log>("Using shared memory name " + m_shmimName + ".");
      }
   }
   catch(...)
   {
      if(camera) delete camera;
      
      camera = nullptr;
      
      state(stateCodes::NODEVICE);
      if(!stateLogged())
      {
         log<text_log>("no camera with serial number " + m_serialNumber + " found.");
      }
   }
   
   return 0;
}


int baslerCtrl::startAcquisition()
{
   m_width = 640;
   m_height = 480;
   m_dataType = _DATATYPE_INT16;

    
   try
   {
      camera->StartGrabbing(GrabStrategy_LatestImageOnly ); // Start grabbing, and always grab just the last image.
   }
   catch(...)
   {
      state(stateCodes::NOTCONNECTED);
      return -1;
   }
   
   return 0;
}

int baslerCtrl::acquireAndCheckValid()
{
   try
   {
      camera->RetrieveResult(1000, ptrGrabResult, TimeoutHandling_ThrowException);
   }
   catch(...)
   {
      state(stateCodes::NOTCONNECTED);
      return -1;
   }
   
   if (ptrGrabResult->GrabSucceeded()) // If image is grabbed successfully 
   {
      clock_gettime(CLOCK_REALTIME, &m_currImageTimestamp);
      return 0;
   }
   else
   {
      state(stateCodes::NOTCONNECTED);
      return -1;
   }
}

int baslerCtrl::loadImageIntoStream(void * dest)
{
   try 
   {
      memcpy( dest, (pixelT *) ptrGrabResult->GetBuffer(), m_width*m_height*sizeof(pixelT));
   }
   catch(...)
   {
      state(stateCodes::NOTCONNECTED);
      return -1;
   }
   
   return 0;
}

int baslerCtrl::reconfig()
{
   return 0;
}
   

inline
int baslerCtrl::getTemp()
{
   if( camera == nullptr) return 0;
   
   float tempcam;
   try 
   {
      tempcam = (float)camera->DeviceTemperature.GetValue();
   }
   catch(...)
   {
      state(stateCodes::NOTCONNECTED);
      return -1;
   }
   
   
   updateIfChanged(m_indiP_temp, "current", tempcam);
   
   
   return 0;

}

inline
int baslerCtrl::getExpTime()
{
   if( camera == nullptr) return 0;
   
   float tempet;
   try 
   {
      tempet = (float)camera->ExposureTime.GetValue();
   }
   catch(...)
   {
      state(stateCodes::NOTCONNECTED);
      return -1;
   }
   
   updateIfChanged(m_indiP_exptime, "current", tempet/1e6);
   
   
   return 0;

}

inline
int baslerCtrl::setExpTime(double exptime)
{
   
   try
   {
      camera->ExposureTime.SetValue(exptime*1e6);
   }
   catch(...)
   {
      log<software_error>({__FILE__, __LINE__, "Error setting exposure time"});
      return -1;
   }
   
   log<text_log>( "Set exposure time: " + std::to_string(exptime) + " sec");
   
   return 0;
}


inline
int baslerCtrl::setFPS(double fps)
{
   return setExpTime(1.0/fps);
}


INDI_NEWCALLBACK_DEFN(baslerCtrl, m_indiP_fps)(const pcf::IndiProperty &ipRecv)
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

INDI_NEWCALLBACK_DEFN(baslerCtrl, m_indiP_exptime)(const pcf::IndiProperty &ipRecv)
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
