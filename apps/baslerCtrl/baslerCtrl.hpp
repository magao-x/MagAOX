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
  * <a href="../handbook/operating/software/apps/baslerCtrl.html">Application Documentation</a>
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
class baslerCtrl : public MagAOXApp<>, public dev::stdCamera<baslerCtrl>, public dev::frameGrabber<baslerCtrl>, public dev::telemeter<baslerCtrl>
{

   friend class dev::stdCamera<baslerCtrl>;
   friend class dev::frameGrabber<baslerCtrl>;
   friend class dev::telemeter<baslerCtrl>;
   
protected:

   /** \name configurable parameters 
     *@{
     */ 
   std::string m_serialNumber; ///< The camera's identifying serial number
   
   
   ///@}

   
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
   
   int configureAcquisition();
   int startAcquisition();
   int acquireAndCheckValid();
   int loadImageIntoStream(void * dest);
   int reconfig();
   
protected:
   
   
   int getTemp();
      
   int getExpTime();
   
   
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
   
   /// Required by stdCamera, but this does not do anything for this camera [stdCamera interface]
   /** 
     * \returns 0 always
     */ 
   int setTempControl();
   
   /// Required by stdCamera, but this does not do anything for this camera [stdCamera interface]
   /** 
     * \returns 0 always
     */
   int setTempSetPt();
   
   /// Required by stdCamera, but this does not do anything for this camera [stdCamera interface]
   /**
     * \returns 0 always
     */ 
   int setFPS();
   
   /// Set the frame rate. [stdCamera interface]
   /** Sets the frame rate to m_fpsSet.
     * 
     * \returns 0 on success
     * \returns -1 on error
     */
   int setExpTime();
   
   /// Required by stdCamera, but this does not do anything for this camera [stdCamera interface]
   /**
     * \returns 0 always
     */
   int setNextROI();
   
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
baslerCtrl::baslerCtrl() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   m_powerMgtEnabled = false;
   
   //--- stdCamera ---
   m_hasTempControl = false;
   m_usesExpTime = true;
   m_usesFPS = false;
   m_usesModes = false;
   m_usesROI = false;
   
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
   dev::stdCamera<baslerCtrl>::setupConfig(config);
   
   dev::frameGrabber<baslerCtrl>::setupConfig(config);
   
   config.add("camera.serialNumber", "", "camera.serialNumber", argType::Required, "camera", "serialNumber", false, "int", "The identifying serial number of the camera.");
   
   dev::telemeter<baslerCtrl>::setupConfig(config);
   
}

inline
void baslerCtrl::loadConfig()
{
   dev::stdCamera<baslerCtrl>::loadConfig(config);
   
   config(m_serialNumber, "camera.serialNumber");
   
   dev::frameGrabber<baslerCtrl>::loadConfig(config);
   
   dev::telemeter<baslerCtrl>::loadConfig(config);
}
   

inline
int baslerCtrl::appStartup()
{
   
   //=================================
   // Do camera configuration here
  
   PylonInitialize(); // Initializes pylon runtime before using any pylon methods

   
   if(dev::stdCamera<baslerCtrl>::appStartup() < 0)
   {
      return log<software_critical,-1>({__FILE__,__LINE__});
   }
   
   if(dev::frameGrabber<baslerCtrl>::appStartup() < 0)
   {
      return log<software_critical,-1>({__FILE__,__LINE__});
   }
   
   
   if(dev::telemeter<baslerCtrl>::appStartup() < 0)
   {
      return log<software_error,-1>({__FILE__,__LINE__});
   }
   
   state(stateCodes::NOTCONNECTED);
   
   return 0;

}

inline
int baslerCtrl::appLogic()
{
   //and run stdCamera's appLogic
   if(dev::stdCamera<baslerCtrl>::appLogic() < 0)
   {
      return log<software_error, -1>({__FILE__, __LINE__});
   }
   
   //and run frameGrabber's appLogic to see if the f.g. thread has exited.
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

      if(stdCamera<baslerCtrl>::updateINDI() < 0)
      {
         log<software_error>({__FILE__, __LINE__});
         state(stateCodes::ERROR);
         return 0;
      }
      
      if(frameGrabber<baslerCtrl>::updateINDI() < 0)
      {
         log<software_error>({__FILE__, __LINE__});
         state(stateCodes::ERROR);
         return 0;
      }
      
      if(telemeter<baslerCtrl>::appLogic() < 0)
      {
         log<software_error>({__FILE__, __LINE__});
         return 0;
      }
   }

   ///\todo Fall through check?

   return 0;

}


inline
int baslerCtrl::appShutdown()
{
   dev::stdCamera<baslerCtrl>::appShutdown();
   
   dev::frameGrabber<baslerCtrl>::appShutdown();
   
   if(camera) camera->Close();
   
   PylonTerminate();
      
   dev::telemeter<baslerCtrl>::appShutdown();
    
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
      if(camera) 
      {
         camera->Close();
         delete camera;
      }
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
      if(camera) 
      {
         camera->Close();
         delete camera;
      }
      camera = nullptr;
      
      state(stateCodes::NODEVICE);
      if(!stateLogged())
      {
         log<text_log>("no camera with serial number " + m_serialNumber + " found.");
      }
   }
   
   return 0;
}


int baslerCtrl::configureAcquisition()
{
   m_width = 640;
   m_height = 480;
   m_dataType = _DATATYPE_INT16;

   return 0;
}

int baslerCtrl::startAcquisition()
{    
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
   
   try 
   {
      m_ccdTemp = (float)camera->DeviceTemperature.GetValue();
      recordCamera();
   }
   catch(...)
   {
      m_ccdTemp = -999;
      recordCamera();
      state(stateCodes::NOTCONNECTED);
      return -1;
   }
      
   return 0;

}

inline
int baslerCtrl::getExpTime()
{
   if( camera == nullptr) return 0;
   
   try 
   {
      m_expTime = (float)camera->ExposureTime.GetValue()/1e6;
      recordCamera();
   }
   catch(...)
   {
      m_expTime = -999;
      recordCamera();
      state(stateCodes::NOTCONNECTED);
      return -1;
   }
      
   return 0;

}

inline
int baslerCtrl::powerOnDefaults()
{
   return 0;
}

inline
int baslerCtrl::setTempControl()
{
   return 0;
}

inline
int baslerCtrl::setTempSetPt()
{
   return 0;
}

inline
int baslerCtrl::setFPS()
{
   return 0;
}

inline
int baslerCtrl::setExpTime()
{
   try
   {
      recordCamera(true);
      camera->ExposureTime.SetValue(m_expTimeSet*1e6);
   }
   catch(...)
   {
      log<software_error>({__FILE__, __LINE__, "Error setting exposure time"});
      return -1;
   }
   
   log<text_log>( "Set exposure time: " + std::to_string(m_expTimeSet) + " sec");
   
   return 0;
}

inline
int baslerCtrl::setNextROI()
{
   return 0;
}

inline
int baslerCtrl::checkRecordTimes()
{
   return telemeter<baslerCtrl>::checkRecordTimes(telem_stdcam());
}

inline
int baslerCtrl::recordTelem( const telem_stdcam * )
{
   return recordCamera(true);
}


}//namespace app
} //namespace MagAOX
#endif
