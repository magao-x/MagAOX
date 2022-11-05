/** \file qhyCtrl.hpp
  * \brief The MagAO-X QHYCCD camera controller.
  *
  * \author Sebastiaan Y. Haffert (shaffert@arizona.edu)
  *
  * \ingroup qhyCtrl_files
  */

#ifndef qhyCtrl_hpp
#define qhyCtrl_hpp

typedef uint16_t pixelT;

#include <ImageStruct.h>
#include <ImageStreamIO.h>

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include "qhyccd.h"

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

namespace MagAOX
{
namespace app
{

void SDKVersion(){
	unsigned int  YMDS[4];
	unsigned char sVersion[80];

	memset ((char *)sVersion, 0x00, sizeof(sVersion));
	GetQHYCCDSDKVersion(&YMDS[0], &YMDS[1], &YMDS[2], &YMDS[3]);

	if ((YMDS[1] < 10) && (YMDS[2] < 10)){
		sprintf((char *)sVersion, "V20%d0%d0%d_%d\n", YMDS[0], YMDS[1], YMDS[2], YMDS[3]);
	}else if ((YMDS[1] < 10) && (YMDS[2] > 10)){
		sprintf((char *)sVersion, "V20%d0%d%d_%d\n", YMDS[0], YMDS[1], YMDS[2], YMDS[3]);
	}else if ((YMDS[1] > 10) && (YMDS[2] < 10)){
		sprintf((char *)sVersion, "V20%d%d0%d_%d\n", YMDS[0], YMDS[1], YMDS[2], YMDS[3]);
	}else{
		sprintf((char *)sVersion, "V20%d%d%d_%d\n", YMDS[0], YMDS[1], YMDS[2], YMDS[3]);
	}
	fprintf(stderr, "QHYCCD SDK Version: %s\n", sVersion);
}

void FirmWareVersion(qhyccd_handle *h)
{
	unsigned char fwv[32], FWInfo[256];
	unsigned int ret;
	memset (FWInfo, 0x00, sizeof(FWInfo));
	ret = GetQHYCCDFWVersion(h, fwv);
	
	if(ret == QHYCCD_SUCCESS){
		if((fwv[0] >> 4) <= 9){
			sprintf((char *)FWInfo, "Firmware version:20%d_%d_%d\n", ((fwv[0] >> 4) + 0x10), (fwv[0]&~0xf0),fwv[1]);
		}else{
			sprintf((char *)FWInfo, "Firmware version:20%d_%d_%d\n", (fwv[0] >> 4), (fwv[0]&~0xf0), fwv[1]);
		}
	}else{
		sprintf((char *)FWInfo,"Firmware version:Not Found!\n");
	}

	fprintf(stderr,"%s\n", FWInfo);
}

std::string qhyccdSDKErrorName(CONTROL_ID error)
{
	/*
		Fill and complete error messages.
	*/
	
   switch(error)
   {
      case QHYCCD_SUCCESS:
         return "QHT_SUCCES";
      default:
         return "UNKNOWN: " + std::to_string(error);
   }
}

/** \defgroup qhyCtrl QHYCCD USB3 Camera
  * \brief Control of a QHYCCD USB3 Camera
  *
  * <a href="../handbook/operating/software/apps/qhyCtrl.html">Application Documentation</a>
  *
  * \ingroup apps
  *
  */

/** \defgroup qhyCtrl_files QHYCCD USB3 Camera Files
  * \ingroup qhyCtrl
  */

/** MagAO-X application to control a QHYCCD USB3 Camera
  *
  * \ingroup qhyCtrl
  * 
  */
class qhyCtrl : public MagAOXApp<>, public dev::stdCamera<qhyCtrl>, public dev::frameGrabber<qhyCtrl>, public dev::telemeter<qhyCtrl>
{

   friend class dev::stdCamera<qhyCtrl>;
   friend class dev::frameGrabber<qhyCtrl>;
   friend class dev::telemeter<qhyCtrl>;
   
public:
   /** \name app::dev Configurations
     *@{
     */
   static constexpr bool c_stdCamera_tempControl = true; ///< app::dev config to tell stdCamera to not expose temperature controls
   
   static constexpr bool c_stdCamera_temp = true; ///< app::dev config to tell stdCamera to expose temperature
   
   static constexpr bool c_stdCamera_readoutSpeed = false; ///< app::dev config to tell stdCamera not to  expose readout speed controls
   
   static constexpr bool c_stdCamera_vShiftSpeed = false; ///< app:dev config to tell stdCamera not to expose vertical shift speed control

   static constexpr bool c_stdCamera_emGain = false; ///< app::dev config to tell stdCamera to not expose EM gain controls 
   
   static constexpr bool c_stdCamera_exptimeCtrl = true; ///< app::dev config to tell stdCamera to expose exposure time controls
   
   static constexpr bool c_stdCamera_fpsCtrl = false; ///< app::dev config to tell stdCamera to expose FPS controls
   
   static constexpr bool c_stdCamera_fps = false; ///< app::dev config to tell stdCamera not to expose FPS status (ignored since fpsCtrl=true)
   
   static constexpr bool c_stdCamera_usesModes = false; ///< app:dev config to tell stdCamera not to expose mode controls
   
   static constexpr bool c_stdCamera_usesROI = false; ///< app:dev config to tell stdCamera to expose ROI controls

   static constexpr bool c_stdCamera_cropMode = false; ///< app:dev config to tell stdCamera not to expose Crop Mode controls
   
   static constexpr bool c_stdCamera_hasShutter = false; ///< app:dev config to tell stdCamera to expose shutter controls

   static constexpr bool c_stdCamera_usesStateString = false; ///< app::dev confg to tell stdCamera to expose the state string property
   
   static constexpr bool c_frameGrabber_flippable = false; ///< app:dev config to tell framegrabber that this camera can be flipped
   
   ///@}
   
protected:

	/** \name configurable parameters 
	*@{
	*/ 

	std::string m_serialNumber; ///< The camera's identifying serial number
	char m_camId[32]; ///< The camera's ID
	uint32_t m_bits {16}; ///< The number of bits used by the camera.

	///@}

	unsigned int m_retVal {0}; ///< Return code for QHYCCD cameras
	double m_ccdTemp;
	
	double m_expTimeSet;
	double m_expTime;
	
	unsigned int channels {1};
	uint32_t m_frame_length;
	uint8_t * m_frame_data;

	qhyccd_handle *m_camera {nullptr}; ///< The library camera handle
   
public:

   ///Default c'tor
   qhyCtrl();

   ///Destructor
   ~qhyCtrl() noexcept;

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
   int AbortAcquisition();
   int acquireAndCheckValid();
   int loadImageIntoStream(void * dest);
   int reconfig();
   /*
* The derived class must implement:
  * \code
  * int powerOnDefaults(); // called on power-on after powerOnWaitElapsed has occurred.
  * \endcode
  * 
  * Calls to this class's `setupConfig`, `loadConfig`, `appStartup`, `appLogic`, `appShutdown`
  * `onPowerOff`, and `whilePowerOff`,  must be placed in the derived class's functions of the same name.
  */

protected:
   
   /// Get the current detector temperature 
   /** 
     * \returns 0 on success
     * \returns -1 on an error.
     */ 
   int getTemp();

   /// Set the CCD temperature setpoint [stdCamera interface].
   /** Sets the temperature to m_ccdTempSetpt.
     * \returns 0 on success
     * \returns -1 on error
     */
   int setTempSetPt(){};
   
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
   int getFPS(){return 1/m_expTime;};
   
   float fps(){return m_fps;};
   
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
qhyCtrl::qhyCtrl() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   m_powerMgtEnabled = false;
   
   return;
}

inline
qhyCtrl::~qhyCtrl() noexcept
{
   return;
}

inline
void qhyCtrl::setupConfig()
{
   dev::stdCamera<qhyCtrl>::setupConfig(config);
   
   dev::frameGrabber<qhyCtrl>::setupConfig(config);
   
   config.add("camera.serialNumber", "", "camera.serialNumber", argType::Required, "camera", "serialNumber", false, "string", "The identifying serial number of the camera.");
   config.add("camera.bits", "", "camera.bits", argType::Required, "camera", "bits", false, "int", "The number of bits used by the camera.  Default is 16.");
   
   dev::telemeter<qhyCtrl>::setupConfig(config);
}

inline
void qhyCtrl::loadConfig()
{
   dev::stdCamera<qhyCtrl>::loadConfig(config);
   
   config(m_serialNumber, "camera.serialNumber");
   // m_camId = &m_serialNumber[0];
   for(int i=0; i < 32; ++i){
	   m_camId[i] = m_serialNumber[i];
   }

   config(m_bits, "camera.bits");
   
   dev::frameGrabber<qhyCtrl>::loadConfig(config);
   
   dev::telemeter<qhyCtrl>::loadConfig(config);
}
   

inline
int qhyCtrl::appStartup()
{
   
	//=================================
	// Do camera configuration here
	unsigned int m_retVal = InitQHYCCDResource();
	if (QHYCCD_SUCCESS == m_retVal){
		printf("SDK resources initialized.\n");
	}else{
		printf("Cannot initialize SDK resources, error: %d\n", m_retVal);
		return 1;
	}
   
   if(dev::stdCamera<qhyCtrl>::appStartup() < 0)
   {
      return log<software_critical,-1>({__FILE__,__LINE__});
   }
   
   if(dev::frameGrabber<qhyCtrl>::appStartup() < 0)
   {
      return log<software_critical,-1>({__FILE__,__LINE__});
   }
   
   if(dev::telemeter<qhyCtrl>::appStartup() < 0)
   {
      return log<software_error,-1>({__FILE__,__LINE__});
   }
   
   state(stateCodes::NOTCONNECTED);
   
   return 0;

}

inline
int qhyCtrl::appLogic()
{
   //and run stdCamera's appLogic
   if(dev::stdCamera<qhyCtrl>::appLogic() < 0)
   {
      return log<software_error, -1>({__FILE__, __LINE__});
   }
   
   //and run frameGrabber's appLogic to see if the f.g. thread has exited.
   if(dev::frameGrabber<qhyCtrl>::appLogic() < 0)
   {
      return log<software_error, -1>({__FILE__, __LINE__});
   }
   

   ///\todo Fall through check?

   return 0;

}


inline
int qhyCtrl::appShutdown()
{
	dev::stdCamera<qhyCtrl>::appShutdown();
	
	dev::frameGrabber<qhyCtrl>::appShutdown();
	
	if(m_camera){
		// close camera handle
		m_retVal = CloseQHYCCD(m_camera);
		if (QHYCCD_SUCCESS == m_retVal){
			printf("Close QHYCCD success.\n");
		}else{
			printf("Close QHYCCD failure, error: %d\n", m_retVal);
		}
	}

	// release sdk resources
	m_retVal = ReleaseQHYCCDResource();
	if (QHYCCD_SUCCESS == m_retVal){
		printf("SDK resources released.\n");
	}else{
		printf("Cannot release SDK resources, error %d.\n", m_retVal);
		return 1;
	}

      
   dev::telemeter<qhyCtrl>::appShutdown();
    
   return 0;
}


inline
int qhyCtrl::connect()
{
	try {
		if(m_camera) 
			m_retVal = CloseQHYCCD(m_camera);

		m_camera = nullptr;
		m_camera = OpenQHYCCD(m_camId);

	}catch(...){
		if(m_camera) 
			m_retVal = CloseQHYCCD(m_camera);
		m_camera = nullptr;
		
		state(stateCodes::NODEVICE);
		if(!stateLogged())
			log<text_log>("no camera with serial number " + m_serialNumber + " found.");

		return 0;
	}
   
   return 0;
}


int qhyCtrl::configureAcquisition()
{
   if(!m_camera) return -1;

   /*
	Setup m_frame_data here!
   */
   //lock mutex
   std::unique_lock<std::mutex> lock(m_indiMutex);

   uint8_t buf;
   m_retVal = GetQHYCCDCameraStatus(&m_camera, &buf);
   if(m_retVal != QHYCCD_SUCCESS)
   {
      state(stateCodes::ERROR);
      return log<software_error,-1>({__FILE__, __LINE__, "QHYCCD SDK Error from GetStatus: " });
   }
   
	m_retVal = SetQHYCCDBinMode(&m_camera, m_nextROI.bin_x, m_nextROI.bin_y);
	if(m_retVal != QHYCCD_SUCCESS){
		// Do error handling
	}
	
	//QHYCCD expects the top left corner as starting point
	int x0 = (m_nextROI.x - 0.5 * (m_nextROI.w - 1)) + 1;
	int y0 = (m_nextROI.y - 0.5 * (m_nextROI.h - 1)) + 1;
	m_retVal = SetQHYCCDResolution(&m_camera, x0, y0, m_nextROI.w, m_nextROI.h);

	if(m_retVal != QHYCCD_SUCCESS){
		// Do error handling
	}

	m_width = m_currentROI.w;
	m_height = m_currentROI.h;
	m_dataType = _DATATYPE_INT16;

	uint32_t new_frame_length = GetQHYCCDMemLength(&m_camera);
	
	// Only allocate memory if the frame size has changed
	if(new_frame_length != m_frame_length){
		
		m_frame_length = new_frame_length;

		if(m_frame_data){
			delete[] m_frame_data;
		}

		m_frame_data = new uint8_t[m_frame_length];

	}

	//Update binning
	updateIfChanged( m_indiP_roi_x, "current", m_currentROI.x, INDI_OK);
	updateIfChanged( m_indiP_roi_y, "current", m_currentROI.y, INDI_OK);
	updateIfChanged( m_indiP_roi_w, "current", m_currentROI.w, INDI_OK);
	updateIfChanged( m_indiP_roi_h, "current", m_currentROI.h, INDI_OK);
	updateIfChanged( m_indiP_roi_bin_x, "current", m_currentROI.bin_x, INDI_OK);
	updateIfChanged( m_indiP_roi_bin_y, "current", m_currentROI.bin_y, INDI_OK);

	updateIfChanged( m_indiP_roi_x, "target", m_nextROI.x, INDI_OK);
	updateIfChanged( m_indiP_roi_y, "target", m_nextROI.y, INDI_OK);
	updateIfChanged( m_indiP_roi_w, "target", m_nextROI.w, INDI_OK);
	updateIfChanged( m_indiP_roi_h, "target", m_nextROI.h, INDI_OK);
	updateIfChanged( m_indiP_roi_bin_x, "target", m_nextROI.bin_x, INDI_OK);
	updateIfChanged( m_indiP_roi_bin_y, "target", m_nextROI.bin_y, INDI_OK);

	getFPS();

	recordCamera(true); 

	return 0;
}

int qhyCtrl::startAcquisition()
{    
	
   try
   {
      //m_camera->StartGrabbing(GrabStrategy_LatestImageOnly); // Start grabbing, and always grab just the last image.
   }
   catch(...)
   {
      state(stateCodes::NOTCONNECTED);
      return -1;
   }
   
   state(stateCodes::OPERATING);
    
   return 0;
}

int qhyCtrl::AbortAcquisition()
{    
	m_retVal = CancelQHYCCDExposing(&m_camera);
	if(m_retVal != QHYCCD_SUCCESS){
		// Error handling
	}

	// Call readout image

	return 0;
}

int qhyCtrl::acquireAndCheckValid()
{
   try
   {
      // m_camera->RetrieveResult(1000, ptrGrabResult, TimeoutHandling_ThrowException);
   }
   catch(...)
   {
      state(stateCodes::NOTCONNECTED);
      return -1;
   }
   
   /*
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
   */

}


int qhyCtrl::loadImageIntoStream(void * dest)
{
   	// pixelT * src = nullptr;
	// src = (pixelT *) ptrGrabResult->GetBuffer();

   try 
   {
	  m_retVal = GetQHYCCDSingleFrame(m_camera, &m_width, &m_height, &m_bits, &channels, m_frame_data);  
	  // Do error handling
   }
   catch(...)
   {
      state(stateCodes::NOTCONNECTED);
      return -1;
   }

	// if(src == nullptr) return -1;      
	if( frameGrabber<qhyCtrl>::loadImageIntoStreamCopy(dest, m_frame_data, m_width, m_height, sizeof(pixelT)) == nullptr) return -1;
   
   return 0;
}

int qhyCtrl::reconfig()
{  
   return 0;
}
   

inline
int qhyCtrl::getTemp()
{
   if( m_camera == nullptr) return 0;
   
   try 
   {
      m_ccdTemp = GetQHYCCDParam(&m_camera, CONTROL_CURTEMP);
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
int qhyCtrl::getExpTime()
{
   if( m_camera == nullptr) return 0;
   
   try 
   {
	  m_expTime = GetQHYCCDParam(&m_camera, CONTROL_EXPOSURE); // returns exposure time in us.
	  m_expTime /= 1e6;	// divide by 1e6 to get exposure time in seconds

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
int qhyCtrl::powerOnDefaults()
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
int qhyCtrl::setExpTime()
{
   if( m_camera == nullptr) return 0;
    
   try
   {
      recordCamera(true);
	  m_retVal = SetQHYCCDParam(m_camera, CONTROL_EXPOSURE, m_expTimeSet * 1e6);
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
int qhyCtrl::checkNextROI()
{
   std::cerr << "checkNextROI!\n";
   return 0;
}

inline
int qhyCtrl::setNextROI()
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

inline
int qhyCtrl::checkRecordTimes()
{
   return telemeter<qhyCtrl>::checkRecordTimes(telem_stdcam());
}

inline
int qhyCtrl::recordTelem( const telem_stdcam * )
{
   return recordCamera(true);
}


}//namespace app
} //namespace MagAOX
#endif
