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

std::string andorSDKErrorName(unsigned int error)
{
   switch(error)
   {
      case DRV_ERROR_CODES:
         return "DRV_ERROR_CODES";
      case DRV_SUCCESS:
         return "DRV_SUCCESS";
      case DRV_VXDNOTINSTALLED:
         return "DRV_VXDNOTINSTALLED";
      case DRV_ERROR_SCAN:
         return "DRV_ERROR_SCAN";
      case DRV_ERROR_CHECK_SUM:
         return "DRV_ERROR_CHECK_SUM";
      case DRV_ERROR_FILELOAD:
         return "DRV_ERROR_FILELOAD";
      case DRV_UNKNOWN_FUNCTION:
         return "DRV_UNKNOWN_FUNCTION";
      case DRV_ERROR_VXD_INIT:
         return "DRV_ERROR_VXD_INIT";
      case DRV_ERROR_ADDRESS:
         return "DRV_ERROR_ADDRESS";
      case DRV_ERROR_PAGELOCK:
         return "DRV_ERROR_PAGELOCK";
      case DRV_ERROR_PAGEUNLOCK:
         return "DRV_ERROR_PAGEUNLOCK";
      case DRV_ERROR_BOARDTEST:
         return "DRV_ERROR_BOARDTEST";
      case DRV_ERROR_ACK:
         return "DRV_ERROR_ACK";
      case DRV_ERROR_UP_FIFO:
         return "DRV_ERROR_UP_FIFO";
      case DRV_ERROR_PATTERN:
         return "DRV_ERROR_PATTERN";
      case DRV_ACQUISITION_ERRORS:
         return "DRV_ACQUISITION_ERRORS";
      case DRV_ACQ_BUFFER:
         return "DRV_ACQ_BUFFER";
      case DRV_ACQ_DOWNFIFO_FULL:
         return "DRV_ACQ_DOWNFIFO_FULL";
      case DRV_PROC_UNKONWN_INSTRUCTION:
         return "DRV_PROC_UNKONWN_INSTRUCTION";
      case DRV_ILLEGAL_OP_CODE:
         return "DRV_ILLEGAL_OP_CODE";
      case DRV_KINETIC_TIME_NOT_MET:
         return "DRV_KINETIC_TIME_NOT_MET";
      case DRV_ACCUM_TIME_NOT_MET:
         return "DRV_ACCUM_TIME_NOT_MET";
      case DRV_NO_NEW_DATA:
         return "DRV_NO_NEW_DATA";
      case DRV_SPOOLERROR:
         return "DRV_SPOOLERROR";
      case DRV_SPOOLSETUPERROR:
         return "DRV_SPOOLSETUPERROR";
      case DRV_FILESIZELIMITERROR:
         return "DRV_FILESIZELIMITERROR";
      case DRV_ERROR_FILESAVE:
         return "DRV_ERROR_FILESAVE";
      case DRV_TEMPERATURE_CODES:
         return "DRV_TEMPERATURE_CODES";
      case DRV_TEMPERATURE_OFF:
         return "DRV_TEMPERATURE_OFF";
      case DRV_TEMPERATURE_NOT_STABILIZED:
         return "DRV_TEMPERATURE_NOT_STABILIZED";
      case DRV_TEMPERATURE_STABILIZED:
         return "DRV_TEMPERATURE_STABILIZED";
      case DRV_TEMPERATURE_NOT_REACHED:
         return "DRV_TEMPERATURE_NOT_REACHED";
      case DRV_TEMPERATURE_OUT_RANGE:
         return "DRV_TEMPERATURE_OUT_RANGE";
      case DRV_TEMPERATURE_NOT_SUPPORTED:
         return "DRV_TEMPERATURE_NOT_SUPPORTED";
      case DRV_TEMPERATURE_DRIFT:
         return "DRV_TEMPERATURE_DRIFT";
      case DRV_GENERAL_ERRORS:
         return "DRV_GENERAL_ERRORS";
      case DRV_INVALID_AUX:
         return "DRV_INVALID_AUX";
      case DRV_COF_NOTLOADED:
         return "DRV_COF_NOTLOADED";
      case DRV_FPGAPROG:
         return "DRV_FPGAPROG";
      case DRV_FLEXERROR:
         return "DRV_FLEXERROR";
      case DRV_GPIBERROR:
         return "DRV_GPIBERROR";
      case DRV_EEPROMVERSIONERROR:
         return "DRV_EEPROMVERSIONERROR";
      case DRV_DATATYPE:
         return "DRV_DATATYPE";
      case DRV_DRIVER_ERRORS:
         return "DRV_DRIVER_ERRORS";
      case DRV_P1INVALID:
         return "DRV_P1INVALID";
      case DRV_P2INVALID:
         return "DRV_P2INVALID";
      case DRV_P3INVALID:
         return "DRV_P3INVALID";
      case DRV_P4INVALID:
         return "DRV_P4INVALID";
      case DRV_INIERROR:
         return "DRV_INIERROR";
      case DRV_COFERROR:
         return "DRV_COFERROR";
      case DRV_ACQUIRING:
         return "DRV_ACQUIRING";
      case DRV_IDLE:
         return "DRV_IDLE";
      case DRV_TEMPCYCLE:
         return "DRV_TEMPCYCLE";
      case DRV_NOT_INITIALIZED:
         return "DRV_NOT_INITIALIZED";
      case DRV_P5INVALID:
         return "DRV_P5INVALID";
      case DRV_P6INVALID:
         return "DRV_P6INVALID";
      case DRV_INVALID_MODE:
         return "DRV_INVALID_MODE";
      case DRV_INVALID_FILTER:
         return "DRV_INVALID_FILTER";
      case DRV_I2CERRORS:
         return "DRV_I2CERRORS";
      case DRV_I2CDEVNOTFOUND:
         return "DRV_I2CDEVNOTFOUND";
      case DRV_I2CTIMEOUT:
         return "DRV_I2CTIMEOUT";
      case DRV_P7INVALID:
         return "DRV_P7INVALID";
      case DRV_P8INVALID:
         return "DRV_P8INVALID";
      case DRV_P9INVALID:
         return "DRV_P9INVALID";
      case DRV_P10INVALID:
         return "DRV_P10INVALID";
      case DRV_P11INVALID:
         return "DRV_P11INVALID";
      case DRV_USBERROR:
         return "DRV_USBERROR";
      case DRV_IOCERROR:
         return "DRV_IOCERROR";
      case DRV_VRMVERSIONERROR:
         return "DRV_VRMVERSIONERROR";
      case DRV_GATESTEPERROR:
         return "DRV_GATESTEPERROR";
      case DRV_USB_INTERRUPT_ENDPOINT_ERROR:
         return "DRV_USB_INTERRUPT_ENDPOINT_ERROR";
      case DRV_RANDOM_TRACK_ERROR:
         return "DRV_RANDOM_TRACK_ERROR";
      case DRV_INVALID_TRIGGER_MODE:
         return "DRV_INVALID_TRIGGER_MODE";
      case DRV_LOAD_FIRMWARE_ERROR:
         return "DRV_LOAD_FIRMWARE_ERROR";
      case DRV_DIVIDE_BY_ZERO_ERROR:
         return "DRV_DIVIDE_BY_ZERO_ERROR";
      case DRV_INVALID_RINGEXPOSURES:
         return "DRV_INVALID_RINGEXPOSURES";
      case DRV_BINNING_ERROR:
         return "DRV_BINNING_ERROR";
      case DRV_INVALID_AMPLIFIER:
         return "DRV_INVALID_AMPLIFIER";
      case DRV_INVALID_COUNTCONVERT_MODE:
         return "DRV_INVALID_COUNTCONVERT_MODE";
      case DRV_USB_INTERRUPT_ENDPOINT_TIMEOUT:
         return "DRV_USB_INTERRUPT_ENDPOINT_TIMEOUT";
      case DRV_ERROR_NOCAMERA:
         return "DRV_ERROR_NOCAMERA";
      case DRV_NOT_SUPPORTED:
         return "DRV_NOT_SUPPORTED";
      case DRV_NOT_AVAILABLE:
         return "DRV_NOT_AVAILABLE";
      case DRV_ERROR_MAP:
         return "DRV_ERROR_MAP";
      case DRV_ERROR_UNMAP:
         return "DRV_ERROR_UNMAP";
      case DRV_ERROR_MDL:
         return "DRV_ERROR_MDL";
      case DRV_ERROR_UNMDL:
         return "DRV_ERROR_UNMDL";
      case DRV_ERROR_BUFFSIZE:
         return "DRV_ERROR_BUFFSIZE";
      case DRV_ERROR_NOHANDLE:
         return "DRV_ERROR_NOHANDLE";
      case DRV_GATING_NOT_AVAILABLE:
         return "DRV_GATING_NOT_AVAILABLE";
      case DRV_FPGA_VOLTAGE_ERROR:
         return "DRV_FPGA_VOLTAGE_ERROR";
      case DRV_OW_CMD_FAIL:
         return "DRV_OW_CMD_FAIL";
      case DRV_OWMEMORY_BAD_ADDR:
         return "DRV_OWMEMORY_BAD_ADDR";
      case DRV_OWCMD_NOT_AVAILABLE:
         return "DRV_OWCMD_NOT_AVAILABLE";
      case DRV_OW_NO_SLAVES:
         return "DRV_OW_NO_SLAVES";
      case DRV_OW_NOT_INITIALIZED:
         return "DRV_OW_NOT_INITIALIZED";
      case DRV_OW_ERROR_SLAVE_NUM:
         return "DRV_OW_ERROR_SLAVE_NUM";
      case DRV_MSTIMINGS_ERROR:
         return "DRV_MSTIMINGS_ERROR";
      case DRV_OA_NULL_ERROR:
         return "DRV_OA_NULL_ERROR";
      case DRV_OA_PARSE_DTD_ERROR:
         return "DRV_OA_PARSE_DTD_ERROR";
      case DRV_OA_DTD_VALIDATE_ERROR:
         return "DRV_OA_DTD_VALIDATE_ERROR";
      case DRV_OA_FILE_ACCESS_ERROR:
         return "DRV_OA_FILE_ACCESS_ERROR";
      case DRV_OA_FILE_DOES_NOT_EXIST:
         return "DRV_OA_FILE_DOES_NOT_EXIST";
      case DRV_OA_XML_INVALID_OR_NOT_FOUND_ERROR:
         return "DRV_OA_XML_INVALID_OR_NOT_FOUND_ERROR";
      case DRV_OA_PRESET_FILE_NOT_LOADED:
         return "DRV_OA_PRESET_FILE_NOT_LOADED";
      case DRV_OA_USER_FILE_NOT_LOADED:
         return "DRV_OA_USER_FILE_NOT_LOADED";
      case DRV_OA_PRESET_AND_USER_FILE_NOT_LOADED:
         return "DRV_OA_PRESET_AND_USER_FILE_NOT_LOADED";
      case DRV_OA_INVALID_FILE:
         return "DRV_OA_INVALID_FILE";
      case DRV_OA_FILE_HAS_BEEN_MODIFIED:
         return "DRV_OA_FILE_HAS_BEEN_MODIFIED";
      case DRV_OA_BUFFER_FULL:
         return "DRV_OA_BUFFER_FULL";
      case DRV_OA_INVALID_STRING_LENGTH:
         return "DRV_OA_INVALID_STRING_LENGTH";
      case DRV_OA_INVALID_CHARS_IN_NAME:
         return "DRV_OA_INVALID_CHARS_IN_NAME";
      case DRV_OA_INVALID_NAMING:
         return "DRV_OA_INVALID_NAMING";
      case DRV_OA_GET_CAMERA_ERROR:
         return "DRV_OA_GET_CAMERA_ERROR";
      case DRV_OA_MODE_ALREADY_EXISTS:
         return "DRV_OA_MODE_ALREADY_EXISTS";
      case DRV_OA_STRINGS_NOT_EQUAL:
         return "DRV_OA_STRINGS_NOT_EQUAL";
      case DRV_OA_NO_USER_DATA:
         return "DRV_OA_NO_USER_DATA";
      case DRV_OA_VALUE_NOT_SUPPORTED:
         return "DRV_OA_VALUE_NOT_SUPPORTED";
      case DRV_OA_MODE_DOES_NOT_EXIST:
         return "DRV_OA_MODE_DOES_NOT_EXIST";
      case DRV_OA_CAMERA_NOT_SUPPORTED:
         return "DRV_OA_CAMERA_NOT_SUPPORTED";
      case DRV_OA_FAILED_TO_GET_MODE:
         return "DRV_OA_FAILED_TO_GET_MODE";
      case DRV_OA_CAMERA_NOT_AVAILABLE:
         return "DRV_OA_CAMERA_NOT_AVAILABLE";
      case DRV_PROCESSING_FAILED:
         return "DRV_PROCESSING_FAILED";
      default:
         return "UNKNOWN: " + std::to_string(error);
   }
}

int readoutParams( int & newa,
                   int & newhss,
                   const std::string & ron
                 )
{
   if(ron == "ccd_00_08MHz")
   {
      newa = 1;
      newhss = 2;
   }
   else if(ron == "ccd_01MHz")
   {
      newa = 1;
      newhss = 1;
   }
   else if(ron == "ccd_03MHz")
   {
      newa = 1;
      newhss = 0;
   }
   else if(ron == "emccd_01MHz")
   {
      newa = 0;
      newhss = 3;
   }
   else if(ron == "emccd_05MHz")
   {
      newa = 0;
      newhss = 2;
   }
   else if(ron == "emccd_10MHz")
   {
      newa = 0;
      newhss = 1;
   }
   else if(ron == "emccd_17MHz")
   {
      newa = 0;
      newhss = 0;
   }
   else
   {
      return -1;
   }
   
   return 0;
}

int vshiftParams( int & newvs,
                  const std::string & vssn,
                  float & vs
                )
{
   if(vssn == "0_3us")
   {
      newvs = 0;
      vs = 0.3;
      return 0;
   }
   else if(vssn == "0_5us")
   {
      newvs = 1;
      vs = 0.5;
      return 0;
   }
   else if(vssn == "0_9us")
   {
      newvs = 2;
      vs = 0.9;
      return 0;
   }
   else if(vssn == "1_7us")
   {
      newvs = 3;
      vs = 1.7;
      return 0;
   }
   else if(vssn == "3_3us")
   {
      newvs = 4;
      vs = 3.3;
      return 0;
   }
   else
   {
      newvs = 0;
      vs = 0.3;
      return -1;
   }
}

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

public:
   /** \name app::dev Configurations
     *@{
     */
   static constexpr bool c_stdCamera_tempControl = true; ///< app::dev config to tell stdCamera to expose temperature controls
   
   static constexpr bool c_stdCamera_temp = true; ///< app::dev config to tell stdCamera to expose temperature
   
   static constexpr bool c_stdCamera_readoutSpeed = true; ///< app::dev config to tell stdCamera to expose readout speed controls
   
   static constexpr bool c_stdCamera_vShiftSpeed = true; ///< app:dev config to tell stdCamera to expose vertical shift speed control
   
   static constexpr bool c_stdCamera_emGain = true; ///< app::dev config to tell stdCamera to expose EM gain controls 

   static constexpr bool c_stdCamera_exptimeCtrl = true; ///< app::dev config to tell stdCamera to expose exposure time controls
   
   static constexpr bool c_stdCamera_fpsCtrl = false; ///< app::dev config to tell stdCamera to not expose FPS controls

   static constexpr bool c_stdCamera_fps = true; ///< app::dev config to tell stdCamera not to expose FPS status
   
   static constexpr bool c_stdCamera_synchro = false; ///< app::dev config to tell stdCamera to not expose synchro mode controls

   static constexpr bool c_stdCamera_usesModes = false; ///< app:dev config to tell stdCamera not to expose mode controls
   
   static constexpr bool c_stdCamera_usesROI = true; ///< app:dev config to tell stdCamera to expose ROI controls

   static constexpr bool c_stdCamera_cropMode = true; ///< app:dev config to tell stdCamera to expose Crop Mode controls
   
   static constexpr bool c_stdCamera_hasShutter = true; ///< app:dev config to tell stdCamera to expose shutter controls

   static constexpr bool c_stdCamera_usesStateString = false; ///< app::dev confg to tell stdCamera to expose the state string property

   static constexpr bool c_edtCamera_relativeConfigPath = false; ///< app::dev config to tell edtCamera to use absolute path to camera config file
   
   static constexpr bool c_frameGrabber_flippable = false; ///< app:dev config to tell framegrabber this camera can not be flipped
   
   ///@}
   
protected:

   /** \name configurable parameters
     *@{
     */

   
   

   ///@}

   std::string m_configFile; ///< The path, relative to configDir, where to write and read the temporary config file.
   
   bool m_libInit {false}; ///< Whether or not the Andor SDK library is initialized.

   bool m_poweredOn {false};
   
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

   /// Set the output amplifier and readout speed
   /** Sets according to stdCamera::m_readoutSpeedNameSet
     */
   int setReadoutSpeed();
   
   /// Set the vertical shift speed
   /** Sets according to std::Camera::m_vShiftSpeedNameSet
     */
   int setVShiftSpeed();
   
   int getEMGain();

   int setEMGain();

   int setCropMode();
   
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
   
   /// Required by stdCamera, but this does not do anything for this camera [stdCamera interface]
   /**
     * \returns 0 always
     */ 
   int setExpTime();
   
   /// Check the next ROI
   /** Checks if the target values are valid and adjusts them to the closest valid values if needed.
     *
     * \returns 0 if successfull
     * \returns -1 otherwise
     */
   int checkNextROI();

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
   
public:
   
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
   
   m_startupTemp = -45;
   
   m_defaultReadoutSpeed  = "emccd_17MHz";
   m_readoutSpeedNames = {"ccd_00_08MHz", "ccd_01MHz", "ccd_03MHz", "emccd_01MHz", "emccd_05MHz", "emccd_10MHz", "emccd_17MHz"};
   m_readoutSpeedNameLabels = {"CCD 0.08 MHz", "CCD 1 MHz", "CCD 3 MHz", "EMCCD 1 MHz", "EMCCD 5 MHz", "EMCCD 10 MHz", "EMCCD 17 MHz"};
   
   m_defaultVShiftSpeed = "3_3us";
   m_vShiftSpeedNames = {"0_3us", "0_5us", "0_9us", "1_7us", "3_3us"};
   m_vShiftSpeedNameLabels = {"0.3 us", "0.5 us", "0.9 us", "1.7 us", "3.3 us"};
   
   m_maxEMGain = 300;

      
   m_default_x = 255.5; 
   m_default_y = 255.5; 
   m_default_w = 512;  
   m_default_h = 512;  
      
   m_nextROI.x = m_default_x;
   m_nextROI.y = m_default_y;
   m_nextROI.w = m_default_w;
   m_nextROI.h = m_default_h;
   m_nextROI.bin_x = 1;
   m_nextROI.bin_y = 1;
   
   m_full_x = 255.5; 
   m_full_y = 255.5; 
   m_full_w = 512; 
   m_full_h = 512; 
   
   
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
   
   
   
   dev::frameGrabber<andorCtrl>::setupConfig(config);

   dev::telemeter<andorCtrl>::setupConfig(config);
   

}

inline
void andorCtrl::loadConfig()
{
   dev::stdCamera<andorCtrl>::loadConfig(config);
   
   m_configFile = "/tmp/andor_";
   m_configFile += configName();
   m_configFile += ".cfg";
   m_cameraModes["onlymode"] = dev::cameraConfig({m_configFile, "", 255, 255, 512, 512, 1, 1, 1000});
   m_startupMode = "onlymode";
   
   if(writeConfig() < 0)
   {
      std::cerr << "m_configFile: " << m_configFile << "\n";
      log<software_critical>({__FILE__,__LINE__});
      m_shutdown = true;
      return;
   }
   
   dev::edtCamera<andorCtrl>::loadConfig(config);


   if(m_maxEMGain < 1)
   {
      m_maxEMGain = 1;
      log<text_log>("maxEMGain set to 1");
   }

   if(m_maxEMGain > 300)
   {
      m_maxEMGain = 300;
      log<text_log>("maxEMGain set to 300");
   }

   dev::frameGrabber<andorCtrl>::loadConfig(config);
   
   dev::telemeter<andorCtrl>::loadConfig(config);

}

inline
int andorCtrl::appStartup()
{
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
   //run stdCamera's appLogic
   if(dev::stdCamera<andorCtrl>::appLogic() < 0)
   {
      return log<software_error, -1>({__FILE__, __LINE__});
   }
   
   //run edtCamera's appLogic
   if(dev::edtCamera<andorCtrl>::appLogic() < 0)
   {
      return log<software_error, -1>({__FILE__, __LINE__});
   }
   
   //run frameGrabber's appLogic to see if the f.g. thread has exited.
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

      if(m_poweredOn && m_ccdTempSetpt > -999)
      {
         m_poweredOn = false;
         if(setTempSetPt() < 0)
         {
            if(powerState() != 1 || powerStateTarget() != 1) return 0;
            return log<software_error,0>({__FILE__,__LINE__});
         }
      }
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
   
   //Setting m_poweredOn
   m_poweredOn = true;

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

         ShutDown();

         //Not an error, appLogic should just go on.
         return 0;
      }
      else if(error!=DRV_SUCCESS)
      {
         log<software_critical>({__FILE__, __LINE__, "ANDOR SDK initialization failed: " + andorSDKErrorName(error)});
         ShutDown();
         return -1;
      }
      
      m_libInit = true;
   }
   
   at_32 lNumCameras = 0;
   error = GetAvailableCameras(&lNumCameras);

   if(error != DRV_SUCCESS)
   {
      log<software_critical>({__FILE__, __LINE__, "ANDOR SDK GetAvailableCameras failed: " + andorSDKErrorName(error)});
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
      log<software_critical>({__FILE__, __LINE__, "ANDOR SDK GetCameraSerialNumber failed: " + andorSDKErrorName(error)});
      return -1;
   }
   
   log<text_log>(std::string("Found Andor USB Camera with serial number ") + std::to_string(serialNumber));
   
   at_32 lCameraHandle;
   error = GetCameraHandle(iSelectedCamera, &lCameraHandle);

   if(error != DRV_SUCCESS)
   {
      log<software_critical>({__FILE__, __LINE__, "ANDOR SDK GetCameraHandle failed: " + andorSDKErrorName(error)});
      return -1;
   }
   
   error = SetCurrentCamera(lCameraHandle);

   if(error != DRV_SUCCESS)
   {
      log<software_critical>({__FILE__, __LINE__, "ANDOR SDK SetCurrentCamera failed: "  + andorSDKErrorName(error)});
      return -1;
   }
   
   char name[MAX_PATH];
   
   error = GetHeadModel(name);
   if(error != DRV_SUCCESS)
   {
      log<software_critical>({__FILE__, __LINE__, "ANDOR SDK GetHeadModel failed: " + andorSDKErrorName(error)});
      return -1;
   }

   state(stateCodes::CONNECTED);
   log<text_log>(std::string("Connected to ") + name +  " with serial number " + std::to_string(serialNumber));
   
   unsigned int eprom;
   unsigned int cofFile;
   unsigned int vxdRev;
   unsigned int vxdVer;
   unsigned int dllRev;
   unsigned int dllVer;
   error = GetSoftwareVersion(&eprom, &cofFile, &vxdRev, &vxdVer, &dllRev, &dllVer);
   if(error != DRV_SUCCESS)
   {
      log<software_critical>({__FILE__, __LINE__, "ANDOR SDK GetSoftwareVersion failed: " + andorSDKErrorName(error)});
      return -1;
   }

   log<text_log>(std::string("eprom: ") + std::to_string(eprom));
   log<text_log>(std::string("cofFile: ") + std::to_string(cofFile));
   log<text_log>(std::string("vxd: ") + std::to_string(vxdVer) + "." + std::to_string(vxdRev));
   log<text_log>(std::string("dll: ") + std::to_string(dllVer) + "." + std::to_string(dllRev));
   
   unsigned int PCB;
   unsigned int Decode;
   unsigned int dummy1;
   unsigned int dummy2;
   unsigned int CameraFirmwareVersion;
   unsigned int CameraFirmwareBuild;
   error = GetHardwareVersion(&PCB, &Decode, &dummy1, &dummy2, &CameraFirmwareVersion, &CameraFirmwareBuild);
   if(error != DRV_SUCCESS)
   {
      log<software_critical>({__FILE__, __LINE__, "ANDOR SDK GetHardwareVersion failed: " + andorSDKErrorName(error)});
      return -1;
   }
   
   log<text_log>(std::string("PCB: ") + std::to_string(PCB));
   log<text_log>(std::string("Decode: ") + std::to_string(Decode));
   log<text_log>(std::string("f/w: ") + std::to_string(CameraFirmwareVersion) + "." + std::to_string(CameraFirmwareBuild));
   
#if 0
   int em_speeds;
   error=GetNumberHSSpeeds(0,0, &em_speeds);
   if(error != DRV_SUCCESS)
   {
      return log<software_error,-1>({__FILE__, __LINE__, std::string("Andor SDK Error from GetNumberHSSpeeds: ") + andorSDKErrorName(error)});
   }
   
   std::cerr << "Number of EM HS speeds: " << em_speeds << "\n";
   for(int i=0; i< em_speeds; ++i)
   {  
      float speed;
      error=GetHSSpeed(0,0,i, &speed);
      std::cerr << i << " " << speed << "\n";
   }
   
   int conv_speeds;
   error=GetNumberHSSpeeds(0,1, &conv_speeds);
   if(error != DRV_SUCCESS)
   {
      return log<software_error,-1>({__FILE__, __LINE__, std::string("Andor SDK Error from GetNumberHSSpeeds: ") + andorSDKErrorName(error)});
   }
   
   std::cerr << "Number of Conventional HS speeds: " << conv_speeds << "\n";
   for(int i=0; i< conv_speeds; ++i)
   {  
      float speed;
      error=GetHSSpeed(0,1,i, &speed);
      std::cerr << i << " " << speed << "\n";
   }
#endif
#if 0
   int v_speeds;
   error=GetNumberVSSpeeds(&v_speeds);
   if(error != DRV_SUCCESS)
   {
      return log<software_error,-1>({__FILE__, __LINE__, std::string("Andor SDK Error from GetNumberVSSpeeds: ") + andorSDKErrorName(error)});
   }
   
   std::cerr << "Number of VS speeds: " << v_speeds << "\n";
   for(int i=0; i< v_speeds; ++i)
   {  
      float speed;
      error=GetVSSpeed(i, &speed);
      std::cerr << i << " " << speed << "\n";
   }
#endif
   
   
   //Initialize Shutter to SHUT
   int ss = 2;
   if(m_shutterState == 1) ss = 1;
   else m_shutterState = 0; //handles startup case
   error = SetShutter(1,ss,50,50);
   if(error != DRV_SUCCESS)
   {
      log<software_critical>({__FILE__, __LINE__, "ANDOR SDK SetShutter failed: " + andorSDKErrorName(error)});
      return -1;
   }
   
   // Set CameraLink
   error = SetCameraLinkMode(1);
   if(error != DRV_SUCCESS)
   {
      log<software_critical>({__FILE__, __LINE__, "ANDOR SDK SetCameraLinkMode failed: " + andorSDKErrorName(error)});
      return -1;
   }
   
   //Set Read Mode to --Image--
   error = SetReadMode(4);
   if(error != DRV_SUCCESS)
   {
      return log<software_error,-1>({__FILE__, __LINE__, "Andor SDK Error from SetReadMode: " + andorSDKErrorName(error)});
   }
   
   //Set Acquisition mode to --Run Till Abort--
   error = SetAcquisitionMode(5);
   if(error != DRV_SUCCESS)
   {
      return log<software_error,-1>({__FILE__, __LINE__, "Andor SDK Error from SetAcquisitionMode: " + andorSDKErrorName(error)});
   }

   //Set to frame transfer mode
   error = SetFrameTransferMode(1);
   if(error != DRV_SUCCESS)
   {
      return log<software_error,-1>({__FILE__, __LINE__, "Andor SDK Error from SetFrameTransferMode: " + andorSDKErrorName(error)});
   }
   
   //Set to real gain mode 
   error = SetEMGainMode(3);
   if(error != DRV_SUCCESS)
   {
      return log<software_error,-1>({__FILE__, __LINE__, "Andor SDK Error from SetEMGainMode: " + andorSDKErrorName(error)});
   }
   
   //Set default amplifier and speed
   m_readoutSpeedName = m_defaultReadoutSpeed;
   m_readoutSpeedNameSet = m_readoutSpeedName;

   int newa;
   int newhss;
   
   if(readoutParams(newa, newhss, m_readoutSpeedNameSet) < 0)
   {
      return log<text_log,-1>("invalid default readout speed: " + m_readoutSpeedNameSet, logPrio::LOG_ERROR);
   }
   
   // Set the HSSpeed to first index
   /* See page 284
    */
   error = SetHSSpeed(newa,newhss);
   if(error != DRV_SUCCESS)
   {
      return log<software_error,-1>({__FILE__, __LINE__, std::string("Andor SDK Error from SetHSSpeed: ") + andorSDKErrorName(error)});
   }
   
   m_vShiftSpeedName = m_defaultVShiftSpeed;
   m_vShiftSpeedNameSet = m_vShiftSpeedName;
   
   int newvs;
   float vs;
   if(vshiftParams(newvs,m_vShiftSpeedNameSet, vs) < 0)
   {
      return log<text_log,-1>("invalid default vert. shift speed: " + m_vShiftSpeedNameSet, logPrio::LOG_ERROR);
   }
   
   // Set the VSSpeed to first index
   error = SetVSSpeed(newvs);
   if(error != DRV_SUCCESS)
   {
      return log<software_error,-1>({__FILE__, __LINE__, std::string("Andor SDK Error from SetVSSpeed: ") + andorSDKErrorName(error)});
   }

   m_vshiftSpeed = vs;
   // Set the amplifier
   /* See page 298
    */
   error = SetOutputAmplifier(newa);
   if(error != DRV_SUCCESS)
   {
      return log<software_error,-1>({__FILE__, __LINE__, std::string("Andor SDK Error from SetOutputAmplifier: ") + andorSDKErrorName(error)});
   }
   
   //Set initial exposure time
   error = SetExposureTime(0.1);
   if(error != DRV_SUCCESS)
   {
      return log<software_error,-1>({__FILE__, __LINE__, "Andor SDK Error from SetExposureTime: " + andorSDKErrorName(error)});
   }
   
   //Turn cooling on:
   if(m_ccdTempSetpt > -999)
   {
      error = CoolerON();
      if(error != DRV_SUCCESS)
      {
         log<software_critical,-1>({__FILE__, __LINE__, "ANDOR SDK CoolerON failed: " + andorSDKErrorName(error)});
      }
      m_tempControlStatus = true;
      m_tempControlStatusStr = "COOLING";
      log<text_log>("enabled temperature control");
   }
   
   int nc;
   GetNumberADChannels(&nc);
   std::cout << "NumberADChannels: " << nc << "\n";
   
   GetNumberAmp(&nc);
   std::cout << "NumberAmp; " << nc << "\n";
   
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
   unsigned int error;
   int gain;

   error = GetEMCCDGain(&gain);
   if( error !=DRV_SUCCESS)
   {
      log<software_error>({__FILE__,__LINE__, "Andor SDK error from GetEMCCDGain: " + andorSDKErrorName(error)});
      return -1;
   }

   if(gain == 0) gain = 1;

   m_emGain = gain;
   
   return 0;
}

inline
int andorCtrl::setReadoutSpeed()
{
   recordCamera(true);
   AbortAcquisition();
   state(stateCodes::CONFIGURING);

   int newa;
   int newhss;
   
   if( readoutParams(newa, newhss, m_readoutSpeedNameSet) < 0)
   {
      return log<text_log,-1>("invalid readout speed: " + m_readoutSpeedNameSet);
   }
   
   if(newa == 1 && m_cropMode)
   {
      log<text_log>("disabling crop mode for CCD readout", logPrio::LOG_NOTICE);
      m_cropModeSet = false;
   }
   // Set the HSSpeed to first index
   /* See page 284
    */
   unsigned int error = SetHSSpeed(newa,newhss);
   if(error != DRV_SUCCESS)
   {
      return log<software_error,-1>({__FILE__, __LINE__, std::string("Andor SDK Error from SetHSSpeed: ") + andorSDKErrorName(error)});
   }
   
   // Set the amplifier
   /* See page 298
    */
   error = SetOutputAmplifier(newa);
   if(error != DRV_SUCCESS)
   {
      return log<software_error,-1>({__FILE__, __LINE__, std::string("Andor SDK Error from SetOutputAmplifier: ") + andorSDKErrorName(error)});
   }

   log<text_log>("Set readout speed to " + m_readoutSpeedNameSet + " (" + std::to_string(newa) + "," + std::to_string(newhss) + ")");

      
   m_readoutSpeedName = m_readoutSpeedNameSet;
   
   m_nextMode = m_modeName;
   m_reconfig = true;

   return 0;
}



inline
int andorCtrl::setVShiftSpeed()
{
   recordCamera(true);
   AbortAcquisition();
   state(stateCodes::CONFIGURING);

   int newvs;
   float vs;
   if( vshiftParams(newvs, m_vShiftSpeedNameSet, vs) < 0)
   {
      return log<text_log,-1>("invalid vertical shift speed: " + m_vShiftSpeedNameSet);
   }
   
   // Set the VSSpeed
   unsigned int error = SetVSSpeed(newvs);
   if(error != DRV_SUCCESS)
   {
      return log<software_error,-1>({__FILE__, __LINE__, std::string("Andor SDK Error from SetVSSpeed: ") + andorSDKErrorName(error)});
   }
   

   log<text_log>("Set vertical shift speed to " + m_vShiftSpeedNameSet + " (" + std::to_string(newvs) + ")");

      
   m_vShiftSpeedName = m_vShiftSpeedNameSet;
   m_vshiftSpeed = vs;

   m_nextMode = m_modeName;
   m_reconfig = true;

   return 0;
}

inline
int andorCtrl::setEMGain()
{
   int amp;
   int hss;
      
   readoutParams(amp,hss, m_readoutSpeedName);
      
   if(amp != 0)
   {
      log<text_log>("Attempt to set EM gain while in conventional amplifier.", logPrio::LOG_NOTICE);
      return 0;
   }
   
   int emg = m_emGainSet;

   if(emg == 1) emg = 0;

   if(emg < 0)
   {
      emg = 0;
      log<text_log>("EM gain limited to 0", logPrio::LOG_WARNING);
   }
   
   if(emg > m_maxEMGain)
   {
      emg = m_maxEMGain;
      log<text_log>("EM gain limited to maxEMGain = " + std::to_string(emg), logPrio::LOG_WARNING);
   }
   
   unsigned int error = SetEMCCDGain(emg);
   if( error !=DRV_SUCCESS)
   {
      log<software_error>({__FILE__,__LINE__, "Andor SDK error from SetEMCCDGain: " + andorSDKErrorName(error)});
      return -1;
   }

   log<text_log>("Set EM Gain to: " + std::to_string(emg), logPrio::LOG_WARNING);
   
   return 0;
}

inline
int andorCtrl::setCropMode()
{
   recordCamera(true);
   AbortAcquisition();
   state(stateCodes::CONFIGURING);
   
   //Check if we're in the EMCCD amplifier
   if(m_cropModeSet == true)
   {
      int amp;
      int hss;
      
      readoutParams(amp,hss, m_readoutSpeedName);
   
      if(amp == 1)
      {
         m_cropModeSet = false;
         log<text_log>("Can not set crop mode in CCD mode", logPrio::LOG_ERROR);
      }
   }
   
   m_nextMode = m_modeName;
   m_reconfig = true;
   
   return 0;
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
   std::string configFile = "/tmp/andor_";
   configFile += configName();
   configFile += ".cfg";
   
   std::ofstream fout;
   fout.open(configFile);
   
   if(fout.fail())
   {
      log<software_error>({__FILE__, __LINE__, "error opening config file for writing"});
      return -1;
   }

   int w = m_nextROI.w / m_nextROI.bin_x;
   int h = m_nextROI.h / m_nextROI.bin_y;
   
   fout << "camera_class:                  \"Andor\"\n";
   fout << "camera_model:                  \"iXon Ultra 897\"\n";
   fout << "camera_info:                   \"512x512 (1-tap, freerun)\"\n";
   fout << "width:                         " << w << "\n";
   fout << "height:                        " << h << "\n";
   fout << "depth:                         16\n";
   fout << "extdepth:                      16\n";
   fout << "CL_DATA_PATH_NORM:             0f       # single tap\n";
   fout << "CL_CFG_NORM:                   02\n";
   //fout << "fv_once: 1\n";
   //fout << "method_framesync: EMULATE_TIMEOUT\n";
   
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
      
   m_currentROI.x = m_default_x;
   m_currentROI.y = m_default_y;
   m_currentROI.w = m_default_w;
   m_currentROI.h = m_default_h;
   m_currentROI.bin_x = 1;
   m_currentROI.bin_y = 1;
   
   m_nextROI.x = m_default_x;
   m_nextROI.y = m_default_y;
   m_nextROI.w = m_default_w;
   m_nextROI.h = m_default_h;
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
         log<software_critical>({__FILE__, __LINE__, "ANDOR SDK CoolerON failed: " + andorSDKErrorName(error)});
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
int andorCtrl::checkNextROI()
{
   return 0;
}

inline 
int andorCtrl::setNextROI()
{ 
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

   int status;
   error = GetStatus(&status);
   if(error != DRV_SUCCESS)
   {
      state(stateCodes::ERROR);
      return log<software_error,-1>({__FILE__, __LINE__, "Andor SDK Error from GetStatus: " + andorSDKErrorName(error)});
   }
   
   if(status != DRV_IDLE) return 0;
   
   int x0 = (m_nextROI.x - 0.5*(m_nextROI.w - 1)) + 1;
   int y0 = (m_nextROI.y - 0.5*(m_nextROI.h - 1)) + 1;
    
   if(m_cropModeSet)
   {
      m_cropMode = m_cropModeSet;
      std::cerr << "crop mode on\n";
      
      error = SetIsolatedCropModeEx(1, m_nextROI.h, m_nextROI.w, m_nextROI.bin_y, m_nextROI.bin_x, x0, y0);
      
      if(error != DRV_SUCCESS)
      {
         if(error == DRV_P2INVALID)
         {
            log<text_log>(std::string("crop mode invalid height: ") + std::to_string(m_nextROI.h), logPrio::LOG_ERROR);
         }
         else if(error == DRV_P3INVALID)
         {
            log<text_log>(std::string("crop mode invalid width: ") + std::to_string(m_nextROI.w), logPrio::LOG_ERROR);
         }
         else if(error == DRV_P4INVALID)
         {
            log<text_log>(std::string("crop mode invalid y binning: ") + std::to_string(m_nextROI.bin_y), logPrio::LOG_ERROR);
         }
         else if(error == DRV_P5INVALID)
         {
            log<text_log>(std::string("crop mode invalid x binning: ") + std::to_string(m_nextROI.bin_x), logPrio::LOG_ERROR);
         }
         else if(error == DRV_P6INVALID)
         {
            log<text_log>(std::string("crop mode invalid x center: ") + std::to_string(m_nextROI.x) + "/" + std::to_string(x0), logPrio::LOG_ERROR);
         }
         else if(error == DRV_P7INVALID)
         {
            log<text_log>(std::string("crop mode invalid y center: ") + std::to_string(m_nextROI.y) + "/" + std::to_string(y0), logPrio::LOG_ERROR);
         }
         else
         {
            log<software_error>({__FILE__, __LINE__, "Andor SDK Error from SetIsolatedCropModeEx: " + andorSDKErrorName(error)});
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
      
      //Set low-latency crop mode
      error = SetIsolatedCropModeType(1);
      if(error != DRV_SUCCESS)
      {
         log<software_error>({__FILE__, __LINE__, "SetIsolatedCropModeType: " + andorSDKErrorName(error)});
      }
   }
   else
   {
      m_cropMode = m_cropModeSet;
      
      error = SetIsolatedCropModeEx(0, m_nextROI.h, m_nextROI.w, m_nextROI.bin_y, m_nextROI.bin_x, x0, y0);
      if(error != DRV_SUCCESS)
      {
         log<software_error>({__FILE__, __LINE__, "SetIsolatedCropModeEx(0,): " + andorSDKErrorName(error)});
      }
      
      std::cerr << "crop mode off\n";
      
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
            log<software_error>({__FILE__, __LINE__, "Andor SDK Error from SetImage: " + andorSDKErrorName(error)});
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
      
      
   }
   
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

   //We also update target to the settable values
   m_nextROI.x = m_currentROI.x;
   m_nextROI.y = m_currentROI.y;
   m_nextROI.w = m_currentROI.w;
   m_nextROI.h = m_currentROI.h;
   m_nextROI.bin_x = m_currentROI.bin_x;
   m_nextROI.bin_y = m_currentROI.bin_y;

   updateIfChanged( m_indiP_roi_x, "target", m_currentROI.x, INDI_OK);
   updateIfChanged( m_indiP_roi_y, "target", m_currentROI.y, INDI_OK);
   updateIfChanged( m_indiP_roi_w, "target", m_currentROI.w, INDI_OK);
   updateIfChanged( m_indiP_roi_h, "target", m_currentROI.h, INDI_OK);
   updateIfChanged( m_indiP_roi_bin_x, "target", m_currentROI.bin_x, INDI_OK);
   updateIfChanged( m_indiP_roi_bin_y, "target", m_currentROI.bin_y, INDI_OK);


   ///\todo This should check whether we have a match between EDT and the camera right?
   m_width = m_currentROI.w/m_currentROI.bin_x;
   m_height = m_currentROI.h/m_currentROI.bin_y;
   m_dataType = _DATATYPE_INT16;

   
   // Print Detector Frame Size
   //std::cout << "Detector Frame is: " << width << "x" << height << "\n";

   recordCamera(true);

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
   unsigned int error;
   int status;
   error = GetStatus(&status);
   if(error != DRV_SUCCESS)
   {
      state(stateCodes::ERROR);
      return log<software_error,-1>({__FILE__, __LINE__, "Andor SDK Error from GetStatus: " + andorSDKErrorName(error)});
   }
   
   if(status != DRV_IDLE) 
   {
      state(stateCodes::OPERATING);
      return 0;
   }
   
   error = StartAcquisition();
   if(error != DRV_SUCCESS)
   {
      state(stateCodes::ERROR);
      return log<software_error,-1>({__FILE__, __LINE__, "Andor SDK Error from StartAcquisition: " + andorSDKErrorName(error)});
   }
   
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
   if( frameGrabber<andorCtrl>::loadImageIntoStreamCopy(dest, m_image_p, m_width, m_height, m_typeSize) == nullptr) return -1;

   return 0;
   }

inline
int andorCtrl::reconfig()
{
   //lock mutex
   //std::unique_lock<std::mutex> lock(m_indiMutex);

   writeConfig();
   
   int rv = edtCamera<andorCtrl>::pdvReconfig();
   if(rv < 0) return rv;
   
   state(stateCodes::READY);
   return 0;
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
