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

//#include <ImageStruct.h>
#include <ImageStreamIO/ImageStreamIO.h>

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
   
public:
   /** \name app::dev Configurations
     *@{
     */
   static constexpr bool c_stdCamera_tempControl = false; ///< app::dev config to tell stdCamera to not expose temperature controls
   
   static constexpr bool c_stdCamera_temp = true; ///< app::dev config to tell stdCamera to expose temperature
   
   static constexpr bool c_stdCamera_readoutSpeed = false; ///< app::dev config to tell stdCamera not to  expose readout speed controls
   
   static constexpr bool c_stdCamera_vShiftSpeed = false; ///< app:dev config to tell stdCamera not to expose vertical shift speed control

   static constexpr bool c_stdCamera_emGain = false; ///< app::dev config to tell stdCamera to not expose EM gain controls 
   
   static constexpr bool c_stdCamera_exptimeCtrl = true; ///< app::dev config to tell stdCamera to expose exposure time controls
   
   static constexpr bool c_stdCamera_fpsCtrl = true; ///< app::dev config to tell stdCamera to expose FPS controls
   
   static constexpr bool c_stdCamera_fps = true; ///< app::dev config to tell stdCamera not to expose FPS status (ignored since fpsCtrl=true)
   
   static constexpr bool c_stdCamera_usesModes = false; ///< app:dev config to tell stdCamera not to expose mode controls
   
   static constexpr bool c_stdCamera_usesROI = true; ///< app:dev config to tell stdCamera to expose ROI controls

   static constexpr bool c_stdCamera_cropMode = false; ///< app:dev config to tell stdCamera not to expose Crop Mode controls
   
   static constexpr bool c_stdCamera_hasShutter = false; ///< app:dev config to tell stdCamera to expose shutter controls

   static constexpr bool c_stdCamera_usesStateString = false; ///< app::dev confg to tell stdCamera to expose the state string property
   
   static constexpr bool c_frameGrabber_flippable = true; ///< app:dev config to tell framegrabber that this camera can be flipped
   
   ///@}
   
protected:

   /** \name configurable parameters 
     *@{
     */ 
   std::string m_serialNumber; ///< The camera's identifying serial number
   
   int m_bits {10}; ///< The number of bits used by the camera.
   
   ///@}

   /** \name binning allowed values 
     * @{
     */   
   std::vector<int> m_binXs; ///< The allowed values of binning in X (horizontal)
   std::vector<int> m_binYs; ///< The allowed values of binning in Y (vertical)

   std::vector<int> m_incXs; ///< The allowed increment in X for each X-binning

   std::vector<int> m_minWs; ///< The minimum value of the width for each X-binning
   std::vector<int> m_incWs; ///< The minimum value of the width for each X-binning
   std::vector<int> m_maxWs; ///< The minimum value of the width for each X-binning

   std::vector<int> m_incYs; ///< The allowed increment in Y for each Y-binning

   std::vector<int> m_minHs; ///< The minimum value of the height for each Y-binning
   std::vector<int> m_incHs; ///< The minimum value of the height for each Y-binning
   std::vector<int> m_maxHs; ///< The minimum value of the height for each Y-binning

   ///@}

   CBaslerUsbInstantCamera * m_camera {nullptr}; ///< The library camera handle
   CGrabResultPtr ptrGrabResult; ///< The result of an attempt to grab an image
   
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
   
   /// Get the current detector temperature 
   /** 
     * \returns 0 on success
     * \returns -1 on an error.
     */ 
   int getTemp();
   
   /// Get the current exposure time 
   /** 
     * \returns 0 on success
     * \returns -1 on an error.
     */   
   int getExpTime();
   
   /// Get the current framerate
   /** 
     * \returns 0 on success
     * \returns -1 on an error.
     */
   int getFPS();
   
   float fps();
   
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
   
   /// Set the framerate.
   /** This uses the acquistion framerate feature.  If m_fpsSet is 0, acuisition framerate is disabled
     * and the resultant framerate is based solely on exposure time and ROI.  If non-zero, then the 
     * framerate will be set to m_fpsSet and the camera will maintain this (as long as exposure time 
     * and ROI allow).
     * 
     * \returns 0 always
     */ 
   int setFPS();
   
   /// Set the Exposure Time. [stdCamera interface]
   /** Sets the frame rate to m_expTimeSet.
     * 
     * \returns 0 on success
     * \returns -1 on error
     */
   int setExpTime();
   
   /// Check the next ROI
   /** Checks if the target values are valid and adjusts them to the closest valid values if needed.
     *
     * \returns 0 always
     */
   int checkNextROI();

   /// Set the next ROI
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

      if(getFPS() < 0)
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

   m_camera->BinningHorizontalMode.SetValue(BinningHorizontalMode_Sum);
	m_camera->BinningVerticalMode.SetValue(BinningVerticalMode_Sum);

   // -- Here we interrogate the camera to find valid ROI settings -- //

   // Stop the camera and cycle through settings to get limits for each binning   
   m_camera->StopGrabbing();
   m_camera->OffsetX.SetValue(0); //ensure that all values are valid
   m_camera->OffsetY.SetValue(0);
      
   int minb = m_camera->BinningHorizontal.GetMin();
   int incb = m_camera->BinningHorizontal.GetInc();
   int maxb = m_camera->BinningHorizontal.GetMax();

   m_binXs.clear();
   for(int b = minb; b<=maxb; b+=incb) m_binXs.push_back(b);

   minb = m_camera->BinningVertical.GetMin();
   incb = m_camera->BinningVertical.GetInc();
   maxb = m_camera->BinningVertical.GetMax();

   m_binYs.clear();
   for(int b = minb; b<=maxb; b+=incb) m_binYs.push_back(b);

   m_incXs.clear();
   m_minWs.clear();
   m_incWs.clear();
   m_maxWs.clear();
   for(size_t b=0; b < m_binXs.size(); ++b)
   {
      m_camera->BinningHorizontal.SetValue(m_binXs[b]);
	   m_camera->BinningVertical.SetValue(m_binYs[0]);

      m_incXs.push_back(m_camera->OffsetX.GetInc());
      m_minWs.push_back(m_camera->Width.GetMin());
      m_incWs.push_back(m_camera->Width.GetInc());
      m_maxWs.push_back(m_camera->Width.GetMax());

      /*//Leave for troubleshooting:
      std::cerr << "--------------------\nH-binning: " << m_binXs[b] << "\n";
      std::cerr << "OffsetX: " << 1 << " " << m_camera->OffsetX.GetInc() << " " << m_camera->Width.GetMax() - m_camera->Width.GetMin() << "\n";
      std::cerr << "Width: " << m_camera->Width.GetMin() << " " << m_camera->Width.GetInc() << " " << m_camera->Width.GetMax() << "\n";
      std::cerr << "OffsetY: " << 1 << " " << m_camera->OffsetY.GetInc() << " " << m_camera->Height.GetMax() - m_camera->Height.GetMin() << "\n";
      std::cerr << "Height: " << m_camera->Height.GetMin() << " " << m_camera->Height.GetInc() << " " << m_camera->Height.GetMax() << "\n";      
      */
   }

   m_incYs.clear();
   m_minHs.clear();
   m_incHs.clear();
   m_maxHs.clear();
   for(size_t b=0; b < m_binYs.size(); ++b)
   {
      m_camera->BinningHorizontal.SetValue(m_binXs[0]);
	   m_camera->BinningVertical.SetValue(m_binYs[b]);

      m_incYs.push_back(m_camera->OffsetX.GetInc());
      m_minHs.push_back(m_camera->Height.GetMin());
      m_incHs.push_back(m_camera->Height.GetInc());
      m_maxHs.push_back(m_camera->Height.GetMax());

      /*//Leave for troubleshooting:
      std::cerr << "--------------------\nV-binning: " << m_binYs[b] << "\n";
      std::cerr << "OffsetX: " << 1 << " " << m_camera->OffsetX.GetInc() << " " << m_camera->Width.GetMax() - m_camera->Width.GetMin() << "\n";
      std::cerr << "Width: " << m_camera->Width.GetMin() << " " << m_camera->Width.GetInc() << " " << m_camera->Width.GetMax() << "\n";
      std::cerr << "OffsetY: " << 1 << " " << m_camera->OffsetY.GetInc() << " " << m_camera->Height.GetMax() - m_camera->Height.GetMin() << "\n";
      std::cerr << "Height: " << m_camera->Height.GetMin() << " " << m_camera->Height.GetInc() << " " << m_camera->Height.GetMax() << "\n";
      */
   }
      
   m_full_w = m_camera->SensorWidth.GetValue();
   m_full_h = m_camera->SensorHeight.GetValue();
   m_full_x = 0.5*((float) m_full_w-1.0);
   m_full_y = 0.5*((float) m_full_h-1.0);

   if(m_default_w == 0) m_default_w = m_full_w;
   if(m_default_h == 0) m_default_h = m_full_h;
   if(m_default_x == 0) m_default_x = m_full_x;
   if(m_default_y == 0) m_default_y = m_full_y;
   if(m_default_bin_x == 0) m_binXs[0];
   if(m_default_bin_y == 0) m_binYs[0];

   m_nextROI.x = m_default_x;
   m_nextROI.y = m_default_y;
   m_nextROI.w = m_default_w;
   m_nextROI.h = m_default_h;
   m_nextROI.bin_x = m_default_bin_x;
   m_nextROI.bin_y = m_default_bin_y;
   
   return 0;
}


int baslerCtrl::configureAcquisition()
{
   if(!m_camera) return -1;

   try
   {
      recordCamera(true); 
      m_camera->StopGrabbing();
      /*
	  	The CenterX/Y has to be set to false otherwise the software tries to auto-center the frames.
		See: https://docs.baslerweb.com/image-roi
	   */
	   m_camera->CenterX.SetValue(false);
	   m_camera->CenterY.SetValue(false);

      //set offsets to 0 so any valid w/h will work.
      m_camera->OffsetX.SetValue(0);
      m_camera->OffsetY.SetValue(0);

      if(checkNextROI() < 0)
      {
         log<software_error>({__FILE__, __LINE__, "error from checkNextROI()"});
         return -1;
      }

      //Note: assuming checkNextROI has adjusted m_nextROI to valid values, so not doing any checks
      //First find binning indices
      size_t bx = 0;
      for(size_t b =0; b < m_binXs.size(); ++b)
      {
         if(m_nextROI.bin_x == m_binXs[b])
         {
            bx = b;
            break;
         }
      }

      size_t by = 0;
      for(size_t b =0; b < m_binYs.size(); ++b)
      {
         if(m_nextROI.bin_y == m_binYs[b])
         {
            by = b;
            break;
         }
      }

      //Set ROI.      
      int xoff;
      int yoff;
      if(m_currentFlip == fgFlipLR || m_currentFlip == fgFlipUDLR)
      {
         xoff = (m_maxWs[bx] - 1 - m_nextROI.x) - 0.5*((float) m_nextROI.w - 1);
      }
      else
      {
         xoff = m_nextROI.x - 0.5*((float) m_nextROI.w - 1);
      }
      
      if(m_currentFlip == fgFlipUD || m_currentFlip == fgFlipUDLR)
      {
         yoff = (m_maxHs[by] - 1 - m_nextROI.y) - 0.5*((float) m_nextROI.h - 1);
      }
      else
      {
         yoff = m_nextROI.y - 0.5*((float) m_nextROI.h - 1);
      }

	   m_camera->BinningHorizontal.SetValue(m_nextROI.bin_x);
	   m_camera->BinningVertical.SetValue(m_nextROI.bin_y);
	   //Probably not necessary to do it every time, but just in case:
      m_camera->BinningHorizontalMode.SetValue(BinningHorizontalMode_Sum);
	   m_camera->BinningVerticalMode.SetValue(BinningVerticalMode_Sum);

      m_camera->Width.SetValue(m_nextROI.w);
      m_camera->Height.SetValue(m_nextROI.h);
      
      m_camera->OffsetX.SetValue(xoff);
      m_camera->OffsetY.SetValue(yoff);

	  // Read the parameter from the camera to check if parameter change is successful
  	   m_currentROI.bin_x = m_camera->BinningHorizontal.GetValue();
	   m_currentROI.bin_y = m_camera->BinningVertical.GetValue();

      bx = 0;
      for(size_t b =0; b < m_binXs.size(); ++b)
      {
         if(m_nextROI.bin_x == m_binXs[b])
         {
            bx = b;
            break;
         }
      }

      by = 0;
      for(size_t b =0; b < m_binYs.size(); ++b)
      {
         if(m_nextROI.bin_y == m_binYs[b])
         {
            by = b;
            break;
         }
      }

      m_currentROI.w = m_camera->Width.GetValue();
      m_currentROI.h = m_camera->Height.GetValue();

      if(m_currentFlip == fgFlipLR || m_currentFlip == fgFlipUDLR)
      {
         m_currentROI.x = m_maxWs[bx] - 1 - (m_camera->OffsetX.GetValue() + 0.5*((float) m_currentROI.w - 1));
      }
      else
      {
         m_currentROI.x = m_camera->OffsetX.GetValue() + 0.5*((float) m_currentROI.w - 1);
      }
      
      if(m_currentFlip == fgFlipUD || m_currentFlip == fgFlipUDLR)
      {
         m_currentROI.y = m_maxHs[by] - 1 - (m_camera->OffsetY.GetValue() + 0.5*((float) m_currentROI.h - 1));
      }
      else
      {
         m_currentROI.y = m_camera->OffsetY.GetValue() + 0.5*((float) m_currentROI.h - 1);
      }

      //Set the full window for this binning
      m_full_currbin_w = m_maxWs[bx];
      m_full_currbin_x = 0.5*((float) m_full_currbin_w - 1.0);
      m_full_currbin_h = m_maxHs[by];
      m_full_currbin_y = 0.5*((float) m_full_currbin_h - 1.0);

      //Update binning
      updateIfChanged( m_indiP_roi_x, "current", m_currentROI.x, INDI_OK);
      updateIfChanged( m_indiP_roi_y, "current", m_currentROI.y, INDI_OK);
      updateIfChanged( m_indiP_roi_w, "current", m_currentROI.w, INDI_OK);
      updateIfChanged( m_indiP_roi_h, "current", m_currentROI.h, INDI_OK);
      updateIfChanged( m_indiP_roi_bin_x, "current", m_currentROI.bin_x, INDI_OK);
      updateIfChanged( m_indiP_roi_bin_y, "current", m_currentROI.bin_y, INDI_OK);
   
      //We also update target to the settable values
      m_nextROI.x = m_currentROI.x;
      m_nextROI.y = m_currentROI.y;
      m_nextROI.w = m_currentROI.w;
      m_nextROI.h = m_currentROI.h;
      m_nextROI.bin_x = m_currentROI.bin_x;
      m_nextROI.bin_y = m_currentROI.bin_y;

      updateIfChanged( m_indiP_roi_x, "target", m_nextROI.x, INDI_OK);
      updateIfChanged( m_indiP_roi_y, "target", m_nextROI.y, INDI_OK);
      updateIfChanged( m_indiP_roi_w, "target", m_nextROI.w, INDI_OK);
      updateIfChanged( m_indiP_roi_h, "target", m_nextROI.h, INDI_OK);
      updateIfChanged( m_indiP_roi_bin_x, "target", m_nextROI.bin_x, INDI_OK);
      updateIfChanged( m_indiP_roi_bin_y, "target", m_nextROI.bin_y, INDI_OK);
      
      m_width = m_currentROI.w;
      m_height = m_currentROI.h;
      m_dataType = _DATATYPE_INT16;
      
      getFPS();
      
      recordCamera(true); 
   }
   catch(...)
   {
      log<software_error>({__FILE__, __LINE__, "invalid ROI specifications"});
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
   
   state(stateCodes::OPERATING);
    
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
   pixelT * src = nullptr;
   try 
   {
      src = (pixelT *) ptrGrabResult->GetBuffer();
   }
   catch(...)
   {
      state(stateCodes::NOTCONNECTED);
      return -1;
   }

   if(src == nullptr) return -1;
             
   if( frameGrabber<baslerCtrl>::loadImageIntoStreamCopy(dest, src, m_width, m_height, sizeof(pixelT)) == nullptr) return -1;
   
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
int baslerCtrl::getFPS()
{
   if( m_camera == nullptr) return 0;
   
   try 
   {
      m_fps = m_camera->ResultingFrameRate.GetValue();
      recordCamera();
   }
   catch(...)
   {
      m_fps = -999;
      recordCamera();
      state(stateCodes::NOTCONNECTED);
      return -1;
   }
      
   return 0;

}

inline
float baslerCtrl::fps()
{
   return m_fps;

}

inline
int baslerCtrl::powerOnDefaults()
{
   m_nextROI.x = m_default_x;
   m_nextROI.y = m_default_y;
   m_nextROI.w = m_default_w;
   m_nextROI.h = m_default_h;
   m_nextROI.bin_x = m_default_bin_x;
   m_nextROI.bin_y = m_default_bin_y;
   
   return 0;
}

inline
int baslerCtrl::setFPS()
{
   if( m_camera == nullptr) return 0;
    
   recordCamera(true);
   
   if(m_fpsSet == 0)
   {
      try
      {
         m_camera->AcquisitionFrameRateEnable.SetValue(false);
      }
      catch(...)
      {
         return log<software_error,-1>({__FILE__, __LINE__, "Error disabling frame rate limit."});
      }
   }
   else
   {
      try 
      {
         m_camera->AcquisitionFrameRateEnable.SetValue(true);
         m_camera->AcquisitionFrameRate.SetValue(m_fpsSet);
      }
      catch(...)
      {
         return log<software_error,-1>({__FILE__, __LINE__, "Error setting frame rate limit."});
      }
   }
   
   return 0;
}

inline
int baslerCtrl::setExpTime()
{
   if( m_camera == nullptr) return 0;
    
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
int baslerCtrl::checkNextROI()
{
   std::cerr << "checkNextROI!\n";
   
   //First find binning indices
   size_t bx = 0;
   for(size_t b =0; b < m_binXs.size(); ++b)
   {
      if(m_nextROI.bin_x == m_binXs[b])
      {
         bx = b;
         break;
      }
   }
   std::cerr << "req bin_x: " << m_nextROI.bin_x << " " << "adj bin_x: " << m_binXs[bx] << "\n";
   m_nextROI.bin_x = m_binXs[bx]; //In case no valid value was found.

   size_t by = 0;
   for(size_t b =0; b < m_binYs.size(); ++b)
   {
      if(m_nextROI.bin_y == m_binYs[b])
      {
         by = b;
         break;
      }
   }
   std::cerr << "req bin_y: " << m_nextROI.bin_y << " " << "adj bin_y: " << m_binYs[by] << "\n";
   m_nextROI.bin_y = m_binYs[by]; //In case no valid value was found.

   //Next check width
   //-- round to nearest increment
   //-- check limits
   int w = m_nextROI.w;
   int rw = w % m_incWs[bx];
   if(rw < 0.5*m_incWs[bx]) w -= rw;
   else w += m_incWs[bx] - rw;

   if(w < m_minWs[bx]) w = m_minWs[bx];
   else if(w > m_maxWs[bx]) w = m_maxWs[bx];

   std::cerr << "req w: " << m_nextROI.w << " " << "adj w: " << w << "\n";
   m_nextROI.w = w;

   //Now check x 
   //-- calculate offset from center 
   //-- round to nearest increment 
   //-- recalculate center 
   int x;
   if(m_currentFlip == fgFlipLR || m_currentFlip == fgFlipUDLR)
   {
      x = (m_maxWs[bx] - 1 - m_nextROI.x) - 0.5*((float) w - 1);
   }
   else
   {
      x = m_nextROI.x - 0.5*((float) w - 1);
   }
   
   int rx = x % m_incXs[bx];
   if(rx < 0.5*m_incXs[bx]) x -= rx;
   else x += m_incXs[bx] - rx;

   if(x < 0) x=0;
   else if(x > m_maxWs[bx] - w) x = m_maxWs[bx] - w;

   std::cerr << "req x: " << m_nextROI.x;
   if(m_currentFlip == fgFlipLR || m_currentFlip == fgFlipUDLR)
   {
      m_nextROI.x = m_maxWs[bx] - 1 - (x + 0.5*((float) w - 1.0));
   }
   else
   {
      m_nextROI.x = x + 0.5*((float) w - 1.0);
   }
   std::cerr << " adj x: " << m_nextROI.x << "\n";

   //Next check height
   //-- round to nearest increment
   //-- check limits
   int h = m_nextROI.h;
   int rh = h % m_incHs[by];
   if(rh < 0.5*m_incHs[by]) h -= rh;
   else h += m_incHs[by] - rh;

   if(h < m_minHs[by]) h = m_minHs[by];
   else if(h > m_maxHs[by]) h = m_maxHs[by];

   std::cerr << "req h: " << m_nextROI.h << " " << "adj h: " << h << "\n";
   m_nextROI.h = h;

   //Now check y
   //-- calculate offset from center 
   //-- round to nearest increment 
   //-- recalculate center 
   int y;
   if(m_currentFlip == fgFlipUD || m_currentFlip == fgFlipUDLR)
   {
      y = (m_maxHs[by] - 1 - m_nextROI.y) - 0.5*((float) h - 1);
   }
   else
   {
      y = m_nextROI.y - 0.5*((float) h - 1);
   }
   
   int ry = y % m_incYs[by];
   if(ry < 0.5*m_incYs[by]) y -= ry;
   else y += m_incYs[by] - ry;

   if(y < 0) y=0;
   else if(y > m_maxHs[by] - h) y = m_maxHs[by] - h;

   std::cerr << "req y: " << m_nextROI.y;
   if(m_currentFlip == fgFlipUD || m_currentFlip == fgFlipUDLR)
   {
      m_nextROI.y = m_maxHs[by] - 1 - (y + 0.5*((float) h - 1));
   }
   else
   {
      m_nextROI.y = y + 0.5*((float) h - 1);
   }
   std::cerr << " adj y: " << m_nextROI.y << "\n";

   updateIfChanged( m_indiP_roi_x, "target", m_nextROI.x, INDI_OK);
   updateIfChanged( m_indiP_roi_y, "target", m_nextROI.y, INDI_OK);
   updateIfChanged( m_indiP_roi_w, "target", m_nextROI.w, INDI_OK);
   updateIfChanged( m_indiP_roi_h, "target", m_nextROI.h, INDI_OK);
   updateIfChanged( m_indiP_roi_bin_x, "target", m_nextROI.bin_x, INDI_OK);
   updateIfChanged( m_indiP_roi_bin_y, "target", m_nextROI.bin_y, INDI_OK);

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
   updateSwitchIfChanged(m_indiP_roi_full, "request", pcf::IndiElement::Off, INDI_IDLE);
   updateSwitchIfChanged(m_indiP_roi_last, "request", pcf::IndiElement::Off, INDI_IDLE);
   updateSwitchIfChanged(m_indiP_roi_default, "request", pcf::IndiElement::Off, INDI_IDLE);
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
