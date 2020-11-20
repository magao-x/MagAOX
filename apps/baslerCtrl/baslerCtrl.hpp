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
   
   int m_bits {10}; ///< The number of bits used by the camera.
   
   ///@}

   int m_incX {1};
   int m_incY {1};
   int m_maxW {0};
   int m_incW {1};
   int m_maxH {0};
   int m_incH {1};
   
   CBaslerUsbInstantCamera * m_camera {nullptr};
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
   
   /// Required by stdCamera, does not currently do anything. [stdCamera interface]
   /**
     * \returns 0 always
     */ 
   int setShutter(int sh);
 
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
   m_usesROI = true;
   
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
   config.add("camera.bits", "", "camera.bits", argType::Required, "camera", "bits", false, "int", "The number of bits used by the camera.  Default is 10.");
   
   dev::telemeter<baslerCtrl>::setupConfig(config);
   
}

inline
void baslerCtrl::loadConfig()
{
   dev::stdCamera<baslerCtrl>::loadConfig(config);
   
   config(m_serialNumber, "camera.serialNumber");
   config(m_bits, "camera.bits");
   
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
   
   if(m_camera) m_camera->Close();
   
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
      if(m_camera) 
      {
         m_camera->Close();
         delete m_camera;
      }
      m_camera = nullptr;
      
      m_camera = new CBaslerUsbInstantCamera( CTlFactory::GetInstance().CreateFirstDevice(info) );
    }
   catch(...)
   {
      if(m_camera) 
      {
         m_camera->Close();
         delete m_camera;
      }
      m_camera = nullptr;
      
      state(stateCodes::NODEVICE);
      if(!stateLogged())
      {
         log<text_log>("no camera with serial number " + m_serialNumber + " found.");
      }
      return 0;
   }
   
      
   try
   {
      if(m_shmimName == "")
      {
         m_shmimName = (std::string)m_camera->GetDeviceInfo().GetModelName() + "_" + (std::string)m_camera->GetDeviceInfo().GetSerialNumber(); // Gets m_camera model name and serial number
      }
      
      m_camera->RegisterConfiguration( new CAcquireContinuousConfiguration , RegistrationMode_ReplaceAll, Cleanup_Delete);
      
      m_camera->Open(); // Opens camera parameters to grab images and set exposure time
   }
   catch(...)
   {
      if(m_camera) 
      {
         m_camera->Close();
         delete m_camera;
      }
      m_camera = nullptr;
      
      state(stateCodes::NODEVICE);
      if(!stateLogged())
      {
         log<text_log>("error opening camera " + m_serialNumber + ".");
      }
      return -1;
   }
   
   try 
   {
      m_camera->ExposureAuto.SetValue(ExposureAuto_Off); 
   }
   catch(...)
   {
      if(m_camera) 
      {
         m_camera->Close();
         delete m_camera;
      }
      m_camera = nullptr;
      
      state(stateCodes::NODEVICE);
      if(!stateLogged())
      {
         log<text_log>("failed to set exposure auto off for camera  " + m_serialNumber);
      }
      return -1;
   }
   
   try
   {
      if(m_bits == 8)
      {
         m_camera->PixelFormat.SetValue(PixelFormat_Mono8);
      }
      else if(m_bits == 10)
      {
         m_camera->PixelFormat.SetValue(PixelFormat_Mono10); // Set to 10 bits
      }
      else if(m_bits == 12)
      {
         m_camera->PixelFormat.SetValue(PixelFormat_Mono12);
      }
      else
      {
         log<text_log>("unsupported bit depth for camera" + m_serialNumber + "");
      }
   }
   catch(...)
   {
      if(m_camera) 
      {
         m_camera->Close();
         delete m_camera;
      }
      m_camera = nullptr;
      
      state(stateCodes::NODEVICE);
      if(!stateLogged())
      {
         log<text_log>("failed to set bit depth for camera" + m_serialNumber + "");
      }
      return -1;
   }
   
   state(stateCodes::CONNECTED);
   if(!stateLogged())
   {
      log<text_log>("Found camera of type " + (std::string)m_camera->GetDeviceInfo().GetModelName() + " with serial number " + m_serialNumber + ".");
      log<text_log>("Using shared memory name " + m_shmimName + ".");
   }
   
   //set offsets to 0 so full field will be registered.
   m_camera->StopGrabbing();
   m_camera->OffsetX.SetValue(0);
   m_camera->OffsetY.SetValue(0);
      
   if(m_startup_w == 0) m_startup_w = m_camera->Width.GetMax();
   if(m_startup_h == 0) m_startup_h = m_camera->Height.GetMax();
   if(m_startup_x == 0) m_startup_x = 0.5*(m_camera->Width.GetMax()-1);
   if(m_startup_y == 0) m_startup_y = 0.5*(m_camera->Height.GetMax()-1);
   if(m_startup_bin_x == 0) m_startup_bin_x = 1;
   if(m_startup_bin_y == 0) m_startup_bin_y = 1;
      
   m_incX = m_camera->OffsetX.GetInc();
   m_incY = m_camera->OffsetY.GetInc();
   m_maxW = m_camera->Width.GetMax();
   m_incW = m_camera->Width.GetInc();
   m_maxH = m_camera->Height.GetMax();
   m_incH = m_camera->Height.GetInc();
   
   m_nextROI.x = m_startup_x;
   m_nextROI.y = m_startup_y;
   m_nextROI.w = m_startup_w;
   m_nextROI.h = m_startup_h;
   m_nextROI.bin_x = m_startup_bin_x;
   m_nextROI.bin_y = m_startup_bin_y;
   
   return 0;
}


int baslerCtrl::configureAcquisition()
{
   if(!m_camera) return -1;

   try
   {
      recordCamera(true); 
      m_camera->StopGrabbing();
      
      //set offsets to 0 so any valid w/h will work.
      m_camera->OffsetX.SetValue(0);
      m_camera->OffsetY.SetValue(0);
      
      if(m_nextROI.w > m_maxW) m_nextROI.w = m_maxW;
      m_nextROI.w -= m_nextROI.w % m_incW;
      
      if(m_nextROI.h > m_maxH) m_nextROI.h = m_maxH;
      m_nextROI.h -= m_nextROI.h % m_incH;
      
      //Set ROI.
      int xoff = m_nextROI.x - 0.5*((float) m_nextROI.w - 1);
      int yoff = m_nextROI.y - 0.5*((float) m_nextROI.h - 1);
      
      xoff -= xoff % m_incX;
      yoff -= yoff % m_incY;

      m_camera->Width.SetValue(m_nextROI.w);
      m_camera->Height.SetValue(m_nextROI.h);
      
      m_camera->OffsetX.SetValue(xoff);
      m_camera->OffsetY.SetValue(yoff);

      
      
      m_currentROI.w = m_camera->Width.GetValue();
      m_currentROI.h = m_camera->Height.GetValue();
      m_currentROI.x = m_camera->OffsetX.GetValue() + 0.5*((float) m_currentROI.w - 1);
      m_currentROI.y = m_camera->OffsetY.GetValue() + 0.5*((float) m_currentROI.h - 1);
      
      updateIfChanged( m_indiP_roi_x, "current", m_currentROI.x, INDI_OK);
      updateIfChanged( m_indiP_roi_y, "current", m_currentROI.y, INDI_OK);
      updateIfChanged( m_indiP_roi_w, "current", m_currentROI.w, INDI_OK);
      updateIfChanged( m_indiP_roi_h, "current", m_currentROI.h, INDI_OK);
      updateIfChanged( m_indiP_roi_bin_x, "current", m_currentROI.bin_x, INDI_OK);
      updateIfChanged( m_indiP_roi_bin_y, "current", m_currentROI.bin_y, INDI_OK);
   
      m_width = m_currentROI.w;
      m_height = m_currentROI.h;
      m_dataType = _DATATYPE_INT16;
      
      recordCamera(true); 
   }
   catch(...)
   {
      log<software_error>({__FILE__, __LINE__, "invalid ROI specifications"});
      m_camera->OffsetX.SetValue(0);
      m_camera->OffsetY.SetValue(0);
      m_camera->Width.SetValue(m_maxW);
      m_camera->Height.SetValue(m_maxH);
      
      state(stateCodes::NOTCONNECTED);
      return -1;
   }
   
   return 0;
}

int baslerCtrl::startAcquisition()
{    
   try
   {
      m_camera->StartGrabbing(GrabStrategy_LatestImageOnly ); // Start grabbing, and always grab just the last image.
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
      m_camera->RetrieveResult(1000, ptrGrabResult, TimeoutHandling_ThrowException);
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
   if( m_camera == nullptr) return 0;
   
   try 
   {
      m_ccdTemp = (float)m_camera->DeviceTemperature.GetValue();
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
   if( m_camera == nullptr) return 0;
   
   try 
   {
      m_expTime = (float)m_camera->ExposureTime.GetValue()/1e6;
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
   m_nextROI.x = m_startup_x;
   m_nextROI.y = m_startup_y;
   m_nextROI.w = m_startup_w;
   m_nextROI.h = m_startup_h;
   m_nextROI.bin_x = m_startup_bin_x;
   m_nextROI.bin_y = m_startup_bin_y;
   
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
      m_camera->ExposureTime.SetValue(m_expTimeSet*1e6);
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
int baslerCtrl::setShutter(int sh)
{
   static_cast<void>(sh);
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
