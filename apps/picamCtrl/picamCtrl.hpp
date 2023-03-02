/** \file picamCtrl.hpp
  * \brief The MagAO-X Princeton Instruments EMCCD camera controller.
  *
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup picamCtrl_files
  */

#ifndef picamCtrl_hpp
#define picamCtrl_hpp


//#include <ImageStruct.h>
#include <ImageStreamIO/ImageStreamIO.h>

#include <picam_advanced.h>

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

#define DEBUG

#ifdef DEBUG
#define BREADCRUMB  std::cerr << __FILE__ << " " << __LINE__ << "\n";
#else
#define BREADCRUMB
#endif

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

int readoutParams( piint & adcQual,
                   piflt & adcSpeed,
                   const std::string & rosn
                 )
{
   if(rosn == "ccd_00_1MHz")
   {
      adcQual = PicamAdcQuality_LowNoise;
      adcSpeed = 0.1;
   }
   else if(rosn == "ccd_01MHz")
   {
      adcQual = PicamAdcQuality_LowNoise;
      adcSpeed = 1;
   }
   else if(rosn == "emccd_05MHz")
   {
      adcQual = PicamAdcQuality_ElectronMultiplied;
      adcSpeed = 5;
   }
   else if(rosn == "emccd_10MHz")
   {
      adcQual = PicamAdcQuality_ElectronMultiplied;
      adcSpeed = 10;
   }
   else if(rosn == "emccd_20MHz")
   {
      adcQual = PicamAdcQuality_ElectronMultiplied;
      adcSpeed = 20;
   }
   else if(rosn == "emccd_30MHz")
   {
      adcQual = PicamAdcQuality_ElectronMultiplied;
      adcSpeed = 30;
   }
   else
   {
      return -1;
   }
   
   return 0;
}

int vshiftParams( piflt & vss,
                  const std::string & vsn
                )
{
   if(vsn == "0_7us")
   {
      vss = 0.7;
   }
   else if(vsn == "1_2us")
   {
      vss = 1.2;
   }
   else if(vsn == "2_0us")
   {
      vss = 2.0;
   }
   else if(vsn == "5_0us")
   {
      vss = 5.0;
   }
   else
   {
      return -1;
   }
   
   return 0;
}


/** \defgroup picamCtrl Princeton Instruments EMCCD Camera
  * \brief Control of a Princeton Instruments EMCCD Camera.
  *
  * <a href="../handbook/operating/software/apps/picamCtrl.html">Application Documentation</a>
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
class picamCtrl : public MagAOXApp<>, public dev::stdCamera<picamCtrl>, public dev::frameGrabber<picamCtrl>, public dev::dssShutter<picamCtrl>, public dev::telemeter<picamCtrl>
{

   friend class dev::stdCamera<picamCtrl>;
   friend class dev::frameGrabber<picamCtrl>;
   friend class dev::dssShutter<picamCtrl>;
   friend class dev::telemeter<picamCtrl>;

   typedef MagAOXApp<> MagAOXAppT;

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
   
   static constexpr bool c_stdCamera_fpsCtrl = false; ///< app::dev config to tell stdCamera not to expose FPS controls

   static constexpr bool c_stdCamera_fps = true; ///< app::dev config to tell stdCamera not to expose FPS status
   
   static constexpr bool c_stdCamera_synchro = false; ///< app::dev config to tell stdCamera to not expose synchro mode controls
   
   static constexpr bool c_stdCamera_usesModes = false; ///< app:dev config to tell stdCamera not to expose mode controls
   
   static constexpr bool c_stdCamera_usesROI = true; ///< app:dev config to tell stdCamera to expose ROI controls

   static constexpr bool c_stdCamera_cropMode = false; ///< app:dev config to tell stdCamera to expose Crop Mode controls
   
   static constexpr bool c_stdCamera_hasShutter = true; ///< app:dev config to tell stdCamera to expose shutter controls
      
   static constexpr bool c_stdCamera_usesStateString = false; ///< app::dev confg to tell stdCamera to expose the state string property
   
   static constexpr bool c_frameGrabber_flippable = true; ///< app:dev config to tell framegrabber this camera can be flipped
   
   ///@}
   
protected:

   /** \name configurable parameters
     *@{
     */
   std::string m_serialNumber; ///< The camera's identifying serial number


   ///@}

   int m_depth {0};
   
   piint m_timeStampMask {PicamTimeStampsMask_ExposureStarted}; // time stamp at end of exposure
   pi64s m_tsRes; // time stamp resolution
   piint m_frameSize;
   double m_camera_timestamp {0.0};
   piflt m_FrameRateCalculation;
   piflt m_ReadOutTimeCalculation;
   
   
   
   

   

   PicamHandle m_cameraHandle {0};
   PicamHandle m_modelHandle {0};

   PicamAcquisitionBuffer m_acqBuff;
   PicamAvailableData m_available;

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

protected:
   int getPicamParameter( piint & value,
                          PicamParameter parameter
                        );

   int getPicamParameter( piflt & value,
                          PicamParameter parameter
                        );

   int setPicamParameter( PicamParameter parameter,
                          pi64s value,
                          bool commit = true
                        );

   int setPicamParameter( PicamParameter parameter,
                          piint value,
                          bool commit = true
                        );

   int setPicamParameter( PicamHandle handle,
                          PicamParameter parameter,
                          piflt value,
                          bool commit = true
                        );

   int setPicamParameter( PicamHandle handle,
                          PicamParameter parameter,
                          piint value,
                          bool commit = true
                        );


   int setPicamParameter( PicamParameter parameter,
                          piflt value,
                          bool commit = true
                        );

   int setPicamParameterOnline( PicamHandle handle,
                                PicamParameter parameter,
                                piflt value
                              );

   int setPicamParameterOnline( PicamParameter parameter,
                                piflt value
                              );

   int setPicamParameterOnline( PicamHandle handle,
                                PicamParameter parameter,
                                piint value
                              );

   int setPicamParameterOnline( PicamParameter parameter,
                                piint value
                              );
   
   int connect();

   int getAcquisitionState();

   int getTemps();

   // stdCamera interface:
   
   //This must set the power-on default values of
   /* -- m_ccdTempSetpt
    * -- m_currentROI 
    */
   int powerOnDefaults();
   
   int setTempControl();
   int setTempSetPt();
   int setReadoutSpeed();
   int setVShiftSpeed();
   int setEMGain();
   int setExpTime();
   int capExpTime(piflt& exptime);
   int setFPS();

   /// Check the next ROI
   /** Checks if the target values are valid and adjusts them to the closest valid values if needed.
     *
     * \returns 0 if successful
     * \returns -1 otherwise
     */
   int checkNextROI();

   int setNextROI();
   int setShutter(int sh);
   
   //Framegrabber interface:
   int configureAcquisition();
   float fps();
   int startAcquisition();
   int acquireAndCheckValid();
   int loadImageIntoStream(void * dest);
   int reconfig();


   //INDI:
protected:

   pcf::IndiProperty m_indiP_readouttime;

public:
   INDI_NEWCALLBACK_DECL(picamCtrl, m_indiP_adcquality);

   /** \name Telemeter Interface
     * 
     * @{
     */ 
   int checkRecordTimes();
   
   int recordTelem( const telem_stdcam * );
   
   
   ///@}
};

inline
picamCtrl::picamCtrl() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   m_powerMgtEnabled = true;

   m_acqBuff.memory_size = 0;
   m_acqBuff.memory = 0;

   m_defaultReadoutSpeed  = "emccd_05MHz";
   m_readoutSpeedNames = {"ccd_00_1MHz", "ccd_01MHz", "emccd_05MHz", "emccd_10MHz", "emccd_20MHz", "emccd_30MHz"};
   m_readoutSpeedNameLabels = {"CCD 0.1 MHz", "CCD 1 MHz", "EMCCD 5 MHz", "EMCCD 10 MHz", "EMCCD 20 MHz", "EMCCD 30 MHz"};
   
   m_defaultVShiftSpeed = "1_2us";
   m_vShiftSpeedNames = {"0_7us", "1_2us", "2_0us", "5_0us"};
   m_vShiftSpeedNameLabels = {"0.7 us", "1.2 us", "2.0 us", "5.0 us"};
   
   
   m_default_x = 511.5; 
   m_default_y = 511.5; 
   m_default_w = 1024;  
   m_default_h = 1024;  
      
   m_full_x = 511.5; 
   m_full_y = 511.5; 
   m_full_w = 1024; 
   m_full_h = 1024; 
   
   m_maxEMGain = 1000;
   
   return;
}

inline
picamCtrl::~picamCtrl() noexcept
{
   if(m_acqBuff.memory)
   {
      free(m_acqBuff.memory);
   }

   return;
}

inline
void picamCtrl::setupConfig()
{
   config.add("camera.serialNumber", "", "camera.serialNumber", argType::Required, "camera", "serialNumber", false, "int", "The identifying serial number of the camera.");

   dev::stdCamera<picamCtrl>::setupConfig(config);
   dev::frameGrabber<picamCtrl>::setupConfig(config);
   dev::dssShutter<picamCtrl>::setupConfig(config);
   dev::telemeter<picamCtrl>::setupConfig(config);
}

inline
void picamCtrl::loadConfig()
{

   config(m_serialNumber, "camera.serialNumber");

   dev::stdCamera<picamCtrl>::loadConfig(config);
   dev::frameGrabber<picamCtrl>::loadConfig(config);
   dev::dssShutter<picamCtrl>::loadConfig(config);
   dev::telemeter<picamCtrl>::loadConfig(config);
   

}

inline
int picamCtrl::appStartup()
{

   // DELETE ME
   //m_outfile = fopen("/home/xsup/test2.txt", "w");

   createROIndiNumber( m_indiP_readouttime, "readout_time", "Readout Time (s)");
   indi::addNumberElement<float>( m_indiP_readouttime, "value", 0.0, std::numeric_limits<float>::max(), 0.0,  "%0.1f", "readout time");
   registerIndiPropertyReadOnly( m_indiP_readouttime );

   
   m_minTemp = -55;
   m_maxTemp = 25;
   m_stepTemp = 0;
   
   m_minROIx = 0;
   m_maxROIx = 1023;
   m_stepROIx = 0;
   
   m_minROIy = 0;
   m_maxROIy = 1023;
   m_stepROIy = 0;
   
   m_minROIWidth = 1;
   m_maxROIWidth = 1024;
   m_stepROIWidth = 4;
   
   m_minROIHeight = 1;
   m_maxROIHeight = 1024;
   m_stepROIHeight = 1;
   
   m_minROIBinning_x = 1;
   m_maxROIBinning_x = 32;
   m_stepROIBinning_x = 1;
   
   m_minROIBinning_y = 1;
   m_maxROIBinning_y = 1024;
   m_stepROIBinning_y = 1;
   
   if(dev::stdCamera<picamCtrl>::appStartup() < 0)
   {
      return log<software_critical,-1>({__FILE__,__LINE__});
   }
   
   if(dev::frameGrabber<picamCtrl>::appStartup() < 0)
   {
      return log<software_critical,-1>({__FILE__,__LINE__});
   }

   if(dev::dssShutter<picamCtrl>::appStartup() < 0)
   {
      return log<software_critical,-1>({__FILE__,__LINE__});
   }

   if(dev::telemeter<picamCtrl>::appStartup() < 0)
   {
      return log<software_error,-1>({__FILE__,__LINE__});
   }
   
   return 0;

}

inline
int picamCtrl::appLogic()
{
   //and run stdCamera's appLogic
   if(dev::stdCamera<picamCtrl>::appLogic() < 0)
   {
      return log<software_error, -1>({__FILE__, __LINE__});
   }
   
   //first run frameGrabber's appLogic to see if the f.g. thread has exited.
   if(dev::frameGrabber<picamCtrl>::appLogic() < 0)
   {
      return log<software_error, -1>({__FILE__, __LINE__});
   }

   //and run dssShutter's appLogic
   if(dev::dssShutter<picamCtrl>::appLogic() < 0)
   {
      return log<software_error, -1>({__FILE__, __LINE__});
   }


   if( state() == stateCodes::NOTCONNECTED || state() == stateCodes::NODEVICE || state() == stateCodes::ERROR)
   {
      m_reconfig = true; //Trigger a f.g. thread reconfig.

      //Might have gotten here because of a power off.
      if(powerState() != 1 || powerStateTarget() != 1) return 0;

      std::unique_lock<std::mutex> lock(m_indiMutex);
      if(connect() < 0)
      {
         if(powerState() != 1 || powerStateTarget() != 1) return 0;
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
         if(powerState() != 1 || powerStateTarget() != 1) return 0;
         return log<software_error,0>({__FILE__,__LINE__});
      }

      if( setTempSetPt() < 0 ) //m_ccdTempSetpt already set on power on
      {
         if(powerState() != 1 || powerStateTarget() != 1) return 0;
         return log<software_error,0>({__FILE__,__LINE__});
      }

      
      if(frameGrabber<picamCtrl>::updateINDI() < 0)
      {
         return log<software_error,0>({__FILE__,__LINE__});
      }
      
      setPicamParameter(m_modelHandle, PicamParameter_DisableCoolingFan, PicamCoolingFanStatus_On);
      
      
   }

   if( state() == stateCodes::READY || state() == stateCodes::OPERATING )
   {
      //Get a lock if we can
      std::unique_lock<std::mutex> lock(m_indiMutex, std::try_to_lock);

      //but don't wait for it, just go back around.
      if(!lock.owns_lock()) return 0;

      if(getAcquisitionState() < 0)
      {
         if(powerState() != 1 || powerStateTarget() != 1) return 0;

         state(stateCodes::ERROR);
         return 0;
      }

      if(getTemps() < 0)
      {
         if(powerState() != 1 || powerStateTarget() != 1) return 0;

         state(stateCodes::ERROR);
         return 0;
      }

      if(stdCamera<picamCtrl>::updateINDI() < 0)
      {
         return log<software_error,0>({__FILE__,__LINE__});
      }
      
      if(frameGrabber<picamCtrl>::updateINDI() < 0)
      {
         return log<software_error,0>({__FILE__,__LINE__});
      }

      if(telemeter<picamCtrl>::appLogic() < 0)
      {
         log<software_error>({__FILE__, __LINE__});
         return 0;
      }

   }

   //Fall through check?
   return 0;

}

inline
int picamCtrl::onPowerOff()
{
   std::lock_guard<std::mutex> lock(m_indiMutex);

   if(m_cameraHandle)
   {
      Picam_CloseCamera(m_cameraHandle);
      m_cameraHandle = 0;
   }

   Picam_UninitializeLibrary();

   if(dssShutter<picamCtrl>::onPowerOff() < 0)
   {
      log<software_error>({__FILE__, __LINE__});
   }

   if(stdCamera<picamCtrl>::onPowerOff() < 0)
   {
      log<software_error>({__FILE__, __LINE__});
   }
   
   return 0;
}

inline
int picamCtrl::whilePowerOff()
{
   if(dssShutter<picamCtrl>::whilePowerOff() < 0)
   {
      log<software_error>({__FILE__, __LINE__});
   }

   if(stdCamera<picamCtrl>::onPowerOff() < 0 )
   {
      log<software_error>({__FILE__, __LINE__});
   }
   
   return 0;
}

inline
int picamCtrl::appShutdown()
{
   dev::frameGrabber<picamCtrl>::appShutdown();

   if(m_cameraHandle)
   {
      Picam_CloseCamera(m_cameraHandle);
      m_cameraHandle = 0;
   }

   Picam_UninitializeLibrary();

   ///\todo error check these base class fxns.
   dev::frameGrabber<picamCtrl>::appShutdown();
   dev::dssShutter<picamCtrl>::appShutdown();

   return 0;
}

inline
int picamCtrl::getPicamParameter( piint & value,
                                  PicamParameter parameter
                                )
{
   PicamError error = Picam_GetParameterIntegerValue( m_cameraHandle, parameter, &value );

   if(MagAOXAppT::m_powerState == 0) return -1; //Flag error but don't log

   if(error != PicamError_None)
   {
      if(powerState() != 1 || powerStateTarget() != 1) return -1;
      log<software_error>({__FILE__, __LINE__, 0, error, PicamEnum2String(PicamEnumeratedType_Error, error)});
      return -1;
   }

   return 0;
}

inline
int picamCtrl::getPicamParameter( piflt & value,
                                  PicamParameter parameter
                                )
{
   PicamError error = Picam_GetParameterFloatingPointValue( m_cameraHandle, parameter, &value );

   if(MagAOXAppT::m_powerState == 0) return -1; //Flag error but don't log

   if(error != PicamError_None)
   {
      if(powerState() != 1 || powerStateTarget() != 1) return -1;
      log<software_error>({__FILE__, __LINE__, 0, error, PicamEnum2String(PicamEnumeratedType_Error, error)});
      return -1;
   }

   return 0;
}

inline
int picamCtrl::setPicamParameter( PicamParameter parameter,
                                  pi64s value,
                                  bool commit
                                )
{
   PicamError error = Picam_SetParameterLargeIntegerValue( m_cameraHandle, parameter, value );
   if(error != PicamError_None)
   {
      if(powerState() != 1 || powerStateTarget() != 1) return -1;
      log<software_error>({__FILE__, __LINE__, 0, error, PicamEnum2String(PicamEnumeratedType_Error, error)});
      return -1;
   }

   if(!commit) return 0;
   
   const PicamParameter* failed_parameters;
   piint failed_parameters_count;

   error = Picam_CommitParameters( m_cameraHandle, &failed_parameters, &failed_parameters_count );
   if(error != PicamError_None)
   {
      if(powerState() != 1 || powerStateTarget() != 1) return -1;
      log<software_error>({__FILE__, __LINE__, 0, error, PicamEnum2String(PicamEnumeratedType_Error, error)});
      return -1;
   }
   
   for( int i=0; i< failed_parameters_count; ++i)
   {
      if( failed_parameters[i] ==  parameter)
      {
         Picam_DestroyParameters( failed_parameters );
         return log<text_log,-1>( "Parameter not committed");
      }
   }
   
   Picam_DestroyParameters( failed_parameters );

   return 0;
}

inline
int picamCtrl::setPicamParameter( PicamHandle handle,
                                  PicamParameter parameter,
                                  piflt value,
                                  bool commit
                                )
{
   PicamError error = Picam_SetParameterFloatingPointValue( handle, parameter, value );
   if(error != PicamError_None)
   {
      if(powerState() != 1 || powerStateTarget() != 1) return -1;
      log<software_error>({__FILE__, __LINE__, 0, error, PicamEnum2String(PicamEnumeratedType_Error, error)});
      return -1;
   }

   if(!commit) return 0;
   
   const PicamParameter* failed_parameters;
   piint failed_parameters_count;

   error = Picam_CommitParameters( handle, &failed_parameters, &failed_parameters_count );
   if(error != PicamError_None)
   {
      if(powerState() != 1 || powerStateTarget() != 1) return -1;
      log<software_error>({__FILE__, __LINE__, 0, error, PicamEnum2String(PicamEnumeratedType_Error, error)});
      return -1;
   }

   for( int i=0; i< failed_parameters_count; ++i)
   {
      if( failed_parameters[i] ==  parameter)
      {
         Picam_DestroyParameters( failed_parameters );
         return log<text_log,-1>( "Parameter not committed");
      }
   }

   Picam_DestroyParameters( failed_parameters );

   return 0;
}

inline
int picamCtrl::setPicamParameter( PicamHandle handle,
                                  PicamParameter parameter,
                                  piint value,
                                  bool commit
                                )
{
   PicamError error = Picam_SetParameterIntegerValue( handle, parameter, value );
   if(error != PicamError_None)
   {
      if(powerState() != 1 || powerStateTarget() != 1) return -1;
      log<software_error>({__FILE__, __LINE__, 0, error, PicamEnum2String(PicamEnumeratedType_Error, error)});
      return -1;
   }

   if(!commit) return 0;
   
   const PicamParameter* failed_parameters;
   piint failed_parameters_count;

   error = Picam_CommitParameters( handle, &failed_parameters, &failed_parameters_count );
   if(error != PicamError_None)
   {
      if(powerState() != 1 || powerStateTarget() != 1) return -1;
      log<software_error>({__FILE__, __LINE__, 0, error, PicamEnum2String(PicamEnumeratedType_Error, error)});
      return -1;
   }

   for( int i=0; i< failed_parameters_count; ++i)
   {
      if( failed_parameters[i] ==  parameter)
      {
         Picam_DestroyParameters( failed_parameters );
         return log<text_log,-1>( "Parameter not committed");
      }
   }

   Picam_DestroyParameters( failed_parameters );

   return 0;
}

inline
int picamCtrl::setPicamParameter( PicamParameter parameter,
                                  piflt value,
                                  bool commit
                                )
{
   return setPicamParameter( m_cameraHandle, parameter, value, commit);
}

inline
int picamCtrl::setPicamParameter( PicamParameter parameter,
                                  piint value,
                                  bool commit
                                )
{
   return setPicamParameter( m_cameraHandle, parameter, value, commit);
}

inline
int picamCtrl::setPicamParameterOnline( PicamHandle handle,
                                        PicamParameter parameter,
                                        piflt value
                                       )
{
   PicamError error = Picam_SetParameterFloatingPointValueOnline( handle, parameter, value );
   if(error != PicamError_None)
   {
      if(powerState() != 1 || powerStateTarget() != 1) return -1;
      log<software_error>({__FILE__, __LINE__, 0, error, PicamEnum2String(PicamEnumeratedType_Error, error)});
      return -1;
   }

   return 0;
}

inline
int picamCtrl::setPicamParameterOnline( PicamParameter parameter,
                                        piflt value
                                       )
{
   return setPicamParameterOnline(m_cameraHandle, parameter, value);
}

inline
int picamCtrl::setPicamParameterOnline( PicamHandle handle,
                                        PicamParameter parameter,
                                        piint value
                                       )
{
   PicamError error = Picam_SetParameterIntegerValueOnline( handle, parameter, value );
   if(error != PicamError_None)
   {
      if(powerState() != 1 || powerStateTarget() != 1) return -1;
      log<software_error>({__FILE__, __LINE__, 0, error, PicamEnum2String(PicamEnumeratedType_Error, error)});
      return -1;
   }

   return 0;
}

inline
int picamCtrl::setPicamParameterOnline( PicamParameter parameter,
                                        piint value
                                       )
{
   return setPicamParameterOnline(m_cameraHandle, parameter, value);
}

inline
int picamCtrl::connect()
{

   PicamError error;
   PicamCameraID * id_array;
   piint id_count;

   if(m_acqBuff.memory)
   {
      free(m_acqBuff.memory);
      m_acqBuff.memory = NULL;
      m_acqBuff.memory_size = 0;
   }

   Picam_UninitializeLibrary();

   //Have to initialize the library every time.  Otherwise we won't catch a newly booted camera.
   Picam_InitializeLibrary();

   if(m_cameraHandle)
   {
      Picam_CloseCamera(m_cameraHandle);
      m_cameraHandle = 0;
   }

   Picam_GetAvailableCameraIDs((const PicamCameraID **) &id_array, &id_count);

   if(powerState() != 1 || powerStateTarget() != 1) return 0;

   if(id_count == 0)
   {
      Picam_DestroyCameraIDs(id_array);

      Picam_UninitializeLibrary();

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
         log<text_log>("Camera was found.  Now connecting.");

         error = PicamAdvanced_OpenCameraDevice(&id_array[i], &m_cameraHandle);
         if(error == PicamError_None)
         {
            m_cameraName = id_array[i].sensor_name;
            m_cameraModel = PicamEnum2String(PicamEnumeratedType_Model, id_array[i].model);

            error = PicamAdvanced_GetCameraModel( m_cameraHandle, &m_modelHandle );
            if( error != PicamError_None )
            {
               log<software_error>({__FILE__, __LINE__, "failed to get camera model"});
            }

            state(stateCodes::CONNECTED);
            log<text_log>("Connected to " + m_cameraName + " [S/N " + m_serialNumber + "]");

            Picam_DestroyCameraIDs(id_array);

            m_readoutSpeedNameSet = m_defaultReadoutSpeed;
            m_vShiftSpeedNameSet = m_defaultVShiftSpeed;
            
            return 0;
         }
         else
         {
            if(powerState() != 1 || powerStateTarget() != 1) return 0;

            state(stateCodes::ERROR);
            if(!stateLogged())
            {
               log<software_error>({__FILE__,__LINE__, 0, error, "Error connecting to camera."});
            }

            Picam_DestroyCameraIDs(id_array);
            
            Picam_UninitializeLibrary();
            return -1;
         }
      }
   }

   state(stateCodes::NODEVICE);
   if(!stateLogged())
   {
      log<text_log>("Camera not found in available ids.");
   }

   Picam_DestroyCameraIDs(id_array);


   Picam_UninitializeLibrary();


   return 0;
}


inline
int picamCtrl::getAcquisitionState()
{
   pibln running = false;

   PicamError error = Picam_IsAcquisitionRunning(m_cameraHandle, &running);

   if(MagAOXAppT::m_powerState == 0) return 0;

   if(error != PicamError_None)
   {
      if(powerState() != 1 || powerStateTarget() != 1) return -1;
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

   if(getPicamParameter(currTemperature, PicamParameter_SensorTemperatureReading) < 0)
   {
      if(powerState() != 1 || powerStateTarget() != 1) return -1;

      log<software_error>({__FILE__, __LINE__});
      state(stateCodes::ERROR);
      return -1;
   }

   m_ccdTemp = currTemperature;
   
   //PicamSensorTemperatureStatus
   piint status;

   if(getPicamParameter( status, PicamParameter_SensorTemperatureStatus ) < 0)
   {
      if(powerState() != 1 || powerStateTarget() != 1) return -1;

      log<software_error>({__FILE__, __LINE__});
      state(stateCodes::ERROR);
      return -1;
   }

   if(status == 1) 
   {
      m_tempControlStatus = true;
      m_tempControlOnTarget = false;
      m_tempControlStatusStr = "UNLOCKED";
   }
   else if(status == 2) 
   {
      m_tempControlStatus = true;
      m_tempControlOnTarget = true;
      m_tempControlStatusStr = "LOCKED";
   }
   else if(status == 3) 
   {
      m_tempControlStatus = false;
      m_tempControlOnTarget = false;
      m_tempControlStatusStr = "FAULTED";
      log<text_log>("temperature control faulted", logPrio::LOG_ALERT);
   }
   else
   {
      m_tempControlStatus = false;
      m_tempControlOnTarget = false;
      m_tempControlStatusStr = "UNKNOWN";
   }   

   recordCamera();

   return 0;

}

inline
int picamCtrl::setFPS()
{
   return 0;
}

inline 
int picamCtrl::powerOnDefaults()
{
   m_ccdTempSetpt = -55; //This is the power on setpoint

   m_currentROI.x = 511.5;
   m_currentROI.y = 511.5;
   m_currentROI.w = 1024;
   m_currentROI.h = 1024;
   m_currentROI.bin_x = 1;
   m_currentROI.bin_y = 1;

   m_readoutSpeedName = "emccd_05MHz";
   m_vShiftSpeedName = "1_2us";
   return 0;
}

inline 
int picamCtrl::setTempControl()
{
   //Always on
   m_tempControlStatus = true;
   m_tempControlStatusSet = true;
   updateSwitchIfChanged(m_indiP_tempcont, "toggle", pcf::IndiElement::On, INDI_IDLE);
   recordCamera(true);
   return 0;
}

inline 
int picamCtrl::setTempSetPt()
{
   ///\todo bounds check here.
   m_reconfig = true;

   recordCamera(true);
   return 0;
}

inline 
int picamCtrl::setReadoutSpeed()
{
   m_reconfig = true;
   recordCamera(true);
   return 0;
}

inline 
int picamCtrl::setVShiftSpeed()
{
   m_reconfig = true;
   recordCamera(true);
   return 0;
}

inline
int picamCtrl::setEMGain()
{
   piint adcQual;
   piflt adcSpeed;
   
   if(readoutParams(adcQual, adcSpeed, m_readoutSpeedName) < 0)
   {
      log<software_error>({__FILE__, __LINE__, "Invalid readout speed: " + m_readoutSpeedNameSet});
      state(stateCodes::ERROR);
      return -1;
   }
   
   if(adcQual != PicamAdcQuality_ElectronMultiplied)
   {
      m_emGain = 1;
      m_adcSpeed = adcSpeed;
      recordCamera(true);
      log<text_log>("Attempt to set EM gain while in conventional amplifier.", logPrio::LOG_NOTICE);
      return 0;
   }
   
   piint emg = m_emGainSet;
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
   
   recordCamera(true);
   if(setPicamParameterOnline(m_modelHandle, PicamParameter_AdcEMGain, emg) < 0)
   {
      if(powerState() != 1 || powerStateTarget() != 1) return -1;
      log<software_error>({__FILE__, __LINE__, "Error setting EM gain"});
      return -1;
   }
   
   piint AdcEMGain;
   if(getPicamParameter(AdcEMGain, PicamParameter_AdcEMGain) < 0)
   {
      if(powerState() != 1 || powerStateTarget() != 1) return -1;
      return log<software_error,-1>({__FILE__, __LINE__, "could not get AdcEMGain"});
   }
   m_emGain = AdcEMGain;
   m_adcSpeed = adcSpeed;
   recordCamera(true);
   return 0;
}

inline
int picamCtrl::setExpTime()
{
   long intexptime = m_expTimeSet * 1000 * 10000 + 0.5;
   piflt exptime = ((double)intexptime)/10000;
   capExpTime(exptime);

   int rv;
   
   recordCamera(true);

   if(state() == stateCodes::OPERATING)
   {
      rv = setPicamParameterOnline(m_modelHandle, PicamParameter_ExposureTime, exptime);      
   }
   else
   {
      rv = setPicamParameter(m_modelHandle, PicamParameter_ExposureTime, exptime);
   }

   if(rv < 0)
   {
      if(powerState() != 1 || powerStateTarget() != 1) return -1;
      log<software_error>({__FILE__, __LINE__, "Error setting exposure time"});
      return -1;
   }

   m_expTime = exptime/1000.0;

   recordCamera(true);

   updateIfChanged(m_indiP_exptime, "current", m_expTime, INDI_IDLE);

   if(getPicamParameter(m_FrameRateCalculation, PicamParameter_FrameRateCalculation) < 0)
   {
      if(powerState() != 1 || powerStateTarget() != 1) return -1;
      log<software_error>({__FILE__, __LINE__, "could not get FrameRateCalculation"});
   }
   m_fps = m_FrameRateCalculation;
   
   recordCamera(true);

   return 0;
}

inline
int picamCtrl::capExpTime(piflt& exptime)
{
   // cap at minimum possible value
   if(exptime < m_ReadOutTimeCalculation)
   {
      if(powerState() != 1 || powerStateTarget() != 1) return -1;
      log<text_log>("Got exposure time " + std::to_string(exptime) + " ms but capped at " + std::to_string(m_ReadOutTimeCalculation) + " ms");
      long intexptime = m_ReadOutTimeCalculation * 10000 + 0.5;
      exptime = ((double)intexptime)/10000;
   }
   
   return 0;
}

inline
int picamCtrl::checkNextROI()
{
   return 0;
}

//Set ROI property to busy if accepted, set toggle to Off and Idlw either way.
//Set ROI actual 
//Update current values (including struct and indiP) and set to OK when done
inline 
int picamCtrl::setNextROI()
{   
   m_reconfig = true;

   updateSwitchIfChanged(m_indiP_roi_set, "request", pcf::IndiElement::Off, INDI_IDLE);
   
   return 0;
   
}

inline 
int picamCtrl::setShutter( int sh )
{
   return dssShutter<picamCtrl>::setShutter(sh);
}

inline
int picamCtrl::configureAcquisition()
{

   piint readoutStride;
   piint framesPerReadout;
   piint frameStride;
   //piint frameSize;
   piint pixelBitDepth;

   m_camera_timestamp = 0; // reset tracked timestamp

   std::unique_lock<std::mutex> lock(m_indiMutex);


   // Time stamp handling
   if(Picam_SetParameterIntegerValue(m_modelHandle, PicamParameter_TimeStamps,  m_timeStampMask) < 0)
   {
      if(powerState() != 1 || powerStateTarget() != 1) return -1;
      log<software_error>({__FILE__,__LINE__, "Could not set time stamp mask"});
   }
   if(Picam_GetParameterLargeIntegerValue(m_modelHandle, PicamParameter_TimeStampResolution, &m_tsRes) < 0)
   {
      if(powerState() != 1 || powerStateTarget() != 1) return -1;
      log<software_error>({__FILE__,__LINE__, "Could not get timestamp resolution"}) ;
   }

   //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
   //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
   // Check Frame Transfer
   //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
   //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

   piint cmode;
   if(getPicamParameter(cmode, PicamParameter_ReadoutControlMode) < 0)
   {
      if(powerState() != 1 || powerStateTarget() != 1) return -1;
      log<software_error>({__FILE__,__LINE__, "could not get Readout Control Mode"});
      return -1;
   }

   if( cmode != PicamReadoutControlMode_FrameTransfer)
   {
      if(powerState() != 1 || powerStateTarget() != 1) return -1;
      log<software_error>({__FILE__,__LINE__, "Readout Control Mode not configured for frame transfer"}) ;
      return -1;
   }

   //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
   //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
   // Temperature
   //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
   //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

   if(setPicamParameter( PicamParameter_SensorTemperatureSetPoint, m_ccdTempSetpt) < 0)
   {
      if(powerState() != 1 || powerStateTarget() != 1) return -1;
      log<software_error>({__FILE__, __LINE__, "Error setting temperature setpoint"});
      state(stateCodes::ERROR);
      return -1;
   }

   //log<text_log>( "Set temperature set point: " + std::to_string(m_ccdTempSetpt) + " C");

   //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
   //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
   // ADC Speed and Quality
   //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
   //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

   piint adcQual;
   piflt adcSpeed;
   
   if(readoutParams(adcQual, adcSpeed, m_readoutSpeedNameSet) < 0)
   {
      if(powerState() != 1 || powerStateTarget() != 1) return -1;
      log<software_error>({__FILE__, __LINE__, "Invalid readout speed: " + m_readoutSpeedNameSet});
      state(stateCodes::ERROR);
      return -1;
   }
   
   if( setPicamParameter(m_modelHandle, PicamParameter_AdcSpeed, adcSpeed, false) < 0) //don't commit b/c it will error if quality mismatched
   {
      if(powerState() != 1 || powerStateTarget() != 1) return -1;
      log<software_error>({__FILE__, __LINE__, "Error setting ADC Speed"});
      //state(stateCodes::ERROR);
      //return -1;
   }
   
   if( setPicamParameter(m_modelHandle, PicamParameter_AdcQuality, adcQual) < 0)
   {
      if(powerState() != 1 || powerStateTarget() != 1) return -1;
      log<software_error>({__FILE__, __LINE__, "Error setting ADC Quality"});
      state(stateCodes::ERROR);
      return -1;
   }
   m_adcSpeed = adcSpeed;
   m_readoutSpeedName = m_readoutSpeedNameSet;
   log<text_log>( "Readout speed set to: " + m_readoutSpeedNameSet);

   if(adcQual == PicamAdcQuality_LowNoise)
   {
      m_emGain = 1.0;
      m_emGainSet = 1.0;
   }

   //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
   //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
   // Vertical Shift Rate
   //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
   //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
   
   piflt vss;
   if(vshiftParams(vss, m_vShiftSpeedNameSet) < 0)
   {
      if(powerState() != 1 || powerStateTarget() != 1) return -1;
      log<software_error>({__FILE__, __LINE__, "Invalid vertical shift speed: " + m_vShiftSpeedNameSet});
      state(stateCodes::ERROR);
      return -1;
   }
   
   if( setPicamParameter(m_modelHandle, PicamParameter_VerticalShiftRate, vss) < 0)
   {
      if(powerState() != 1 || powerStateTarget() != 1) return -1;
      log<software_error>({__FILE__, __LINE__, "Error setting Vertical Shift Rate"});
      state(stateCodes::ERROR);
      return -1;
   }

   m_vShiftSpeedName = m_vShiftSpeedNameSet;
   log<text_log>( "Vertical Shift Rate set to: " + m_vShiftSpeedName);

   //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
   //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
   // Dimensions
   //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
   //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
   
   PicamRois  nextrois;
   PicamRoi nextroi;
   
   nextrois.roi_array = &nextroi;
   nextrois.roi_count = 1;
   
   int roi_err = false;
   if(m_currentFlip == fgFlipLR || m_currentFlip == fgFlipUDLR)
   {
      nextroi.x = ((1023-m_nextROI.x) - 0.5*( (float) m_nextROI.w - 1.0));
   }
   else
   {
      nextroi.x = (m_nextROI.x - 0.5*( (float) m_nextROI.w - 1.0));
   }
   
   if(nextroi.x < 0)
   {
      log<software_error>({__FILE__, __LINE__, "can't set ROI to x center < 0"});
      roi_err = true;
   }

   if(nextroi.x > 1023) 
   {
      log<software_error>({__FILE__, __LINE__, "can't set ROI to x center > 1023"});
      roi_err = true;
   }
   

   if(m_currentFlip == fgFlipUD || m_currentFlip == fgFlipUDLR)
   {
      nextroi.y = ((1023 - m_nextROI.y) - 0.5*( (float) m_nextROI.h - 1.0));
   }
   else
   {
      nextroi.y = (m_nextROI.y - 0.5*( (float) m_nextROI.h - 1.0));
   }
   
   if(nextroi.y < 0)
   {
      log<software_error>({__FILE__, __LINE__, "can't set ROI to y center < 0"});
      roi_err = true;
   }

   if(nextroi.y > 1023) 
   {
      log<software_error>({__FILE__, __LINE__, "can't set ROI to y center > 1023"});
      roi_err = true;
   }

   nextroi.width = m_nextROI.w;

   if(nextroi.width < 0)
   {
      log<software_error>({__FILE__, __LINE__, "can't set ROI to width to be < 0"});
      roi_err = true;
   }

   if(nextroi.x + nextroi.width  > 1024) 
   {
      log<software_error>({__FILE__, __LINE__, "can't set ROI to width such that edge is > 1023"});
      roi_err = true;
   }

   nextroi.height = m_nextROI.h;

   if(nextroi.y + nextroi.height > 1024) 
   {
      log<software_error>({__FILE__, __LINE__, "can't set ROI to height such that edge is > 1023"});
      roi_err = true;
   }

   if(nextroi.height < 0)
   {
      log<software_error>({__FILE__, __LINE__, "can't set ROI to height to be < 0"});
      roi_err = true;
   }

   nextroi.x_binning = m_nextROI.bin_x;
   
   if(nextroi.x_binning < 0)
   {
      log<software_error>({__FILE__, __LINE__, "can't set ROI x binning < 0"});
      roi_err = true;
   }

   nextroi.y_binning = m_nextROI.bin_y;
   
   if(nextroi.y_binning < 0)
   {
      log<software_error>({__FILE__, __LINE__, "can't set ROI y binning < 0"});
      roi_err = true;
   }

   PicamError error;

   if(!roi_err)
   {
      error = Picam_SetParameterRoisValue( m_cameraHandle, PicamParameter_Rois, &nextrois);   
      if( error != PicamError_None )
      {
         if(powerState() != 1 || powerStateTarget() != 1) return -1;
         std::cerr << PicamEnum2String(PicamEnumeratedType_Error, error) << "\n";
         log<software_error>({__FILE__, __LINE__, 0, error, PicamEnum2String(PicamEnumeratedType_Error, error)});
         state(stateCodes::ERROR);
         return -1;
      }
   }
   
   if(getPicamParameter(readoutStride, PicamParameter_ReadoutStride) < 0)
   {
      if(powerState() != 1 || powerStateTarget() != 1) return -1;
      log<software_error>({__FILE__, __LINE__, "Error getting readout stride"});
      state(stateCodes::ERROR);
      return -1;
   }

   if(getPicamParameter(frameStride, PicamParameter_FrameStride) < 0)
   {
      if(powerState() != 1 || powerStateTarget() != 1) return -1;
      log<software_error>({__FILE__, __LINE__, "Error getting frame stride"});
      state(stateCodes::ERROR);

      return -1;
   }

   if(getPicamParameter(framesPerReadout, PicamParameter_FramesPerReadout) < 0)
   {
      if(powerState() != 1 || powerStateTarget() != 1) return -1;
      log<software_error>({__FILE__, __LINE__, "Error getting frames per readout"});
      state(stateCodes::ERROR);
      return -1;
   }

   if(getPicamParameter( m_frameSize, PicamParameter_FrameSize) < 0)
   {
      if(powerState() != 1 || powerStateTarget() != 1) return -1;
      log<software_error>({__FILE__, __LINE__, "Error getting frame size"});
      state(stateCodes::ERROR);
      return -1;
   }

   if(getPicamParameter( pixelBitDepth, PicamParameter_PixelBitDepth) < 0)
   {
      if(powerState() != 1 || powerStateTarget() != 1) return -1;
      log<software_error>({__FILE__, __LINE__,"Error getting pixel bit depth"});
      state(stateCodes::ERROR);
      return -1;
   }
   m_depth = pixelBitDepth;

   const PicamRois* rois;
   error = Picam_GetParameterRoisValue( m_cameraHandle, PicamParameter_Rois, &rois );
   if( error != PicamError_None )
   {
      if(powerState() != 1 || powerStateTarget() != 1) return -1;
      log<software_error>({__FILE__, __LINE__, 0, error, PicamEnum2String(PicamEnumeratedType_Error, error)});
      state(stateCodes::ERROR);
      return -1;
   }
   m_xbinning = rois->roi_array[0].x_binning;
   m_currentROI.bin_x = m_xbinning;
   m_ybinning = rois->roi_array[0].y_binning;
   m_currentROI.bin_y = m_ybinning;
   
   std::cerr << rois->roi_array[0].x << "\n";
   std::cerr << (rois->roi_array[0].x-1) << "\n";
   std::cerr << rois->roi_array[0].width << "\n";
   std::cerr << 0.5*( (float) (rois->roi_array[0].width - 1.0)) << "\n";
   

   if(m_currentFlip == fgFlipLR || m_currentFlip == fgFlipUDLR)
   {
      m_currentROI.x = (1023.0-rois->roi_array[0].x) - 0.5*( (float) (rois->roi_array[0].width - 1.0)) ;
      //nextroi.x = ((1023-m_nextROI.x) - 0.5*( (float) m_nextROI.w - 1.0));
   }
   else
   {
      m_currentROI.x = (rois->roi_array[0].x) + 0.5*( (float) (rois->roi_array[0].width - 1.0)) ;
   }

   
   if(m_currentFlip == fgFlipUD || m_currentFlip == fgFlipUDLR)
   {
      m_currentROI.y = (1023.0-rois->roi_array[0].y) - 0.5*( (float) (rois->roi_array[0].height - 1.0)) ;
      //nextroi.y = ((1023 - m_nextROI.y) - 0.5*( (float) m_nextROI.h - 1.0));
   }
   else
   {
      m_currentROI.y = (rois->roi_array[0].y) + 0.5*( (float) (rois->roi_array[0].height - 1.0)) ;
   }



   
   
   m_currentROI.w = rois->roi_array[0].width;
   m_currentROI.h = rois->roi_array[0].height;
   
   m_width  = rois->roi_array[0].width  / rois->roi_array[0].x_binning;
   m_height = rois->roi_array[0].height / rois->roi_array[0].y_binning;
   Picam_DestroyRois( rois );


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
   
   
   //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
   //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
   // Exposure Time and Frame Rate
   //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
   //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

   if(getPicamParameter(m_ReadOutTimeCalculation, PicamParameter_ReadoutTimeCalculation) < 0)
   {
      if(powerState() != 1 || powerStateTarget() != 1) return -1; 
      return log<software_error, -1>({__FILE__, __LINE__, "could not get ReadOutTimeCalculation"});
   }
   std::cerr << "Readout time is: " <<  m_ReadOutTimeCalculation << "\n";
   updateIfChanged( m_indiP_readouttime, "value", m_ReadOutTimeCalculation/1000.0, INDI_OK); // convert from msec to sec

   const PicamRangeConstraint * constraint_array;
   piint constraint_count;
   PicamAdvanced_GetParameterRangeConstraints( m_modelHandle, PicamParameter_ExposureTime, &constraint_array, &constraint_count);

   if(constraint_count != 1)
   {
      if(powerState() != 1 || powerStateTarget() != 1) return -1; 
      log<text_log>("Constraint count is not 1: " + std::to_string(constraint_count) + " constraints",logPrio::LOG_ERROR);
   }
   else
   {
      m_minExpTime = constraint_array[0].minimum;
      m_maxExpTime = constraint_array[0].maximum;
      m_stepExpTime = constraint_array[0].increment;

      m_indiP_exptime["current"].setMin(m_minExpTime);
      m_indiP_exptime["current"].setMax(m_maxExpTime);
      m_indiP_exptime["current"].setStep(m_stepExpTime);
   
      m_indiP_exptime["target"].setMin(m_minExpTime);
      m_indiP_exptime["target"].setMax(m_maxExpTime);
      m_indiP_exptime["target"].setStep(m_stepExpTime);
   }

   if(m_expTimeSet > 0)
   {
      long intexptime = m_expTimeSet * 1000 * 10000 + 0.5;
      piflt exptime = ((double)intexptime)/10000;
      capExpTime(exptime);
      std::cerr << "Setting exposure time to " << m_expTimeSet << "\n";
      int rv = setPicamParameter(m_modelHandle, PicamParameter_ExposureTime, exptime);

      if(rv < 0)
      {
         if(powerState() != 1 || powerStateTarget() != 1) return -1; 
         return log<software_error, -1>({__FILE__, __LINE__, "Error setting exposure time"});
      }
   }
   
   piflt exptime;
   if(getPicamParameter(exptime, PicamParameter_ExposureTime) < 0)
   {
      if(powerState() != 1 || powerStateTarget() != 1) return -1; 
      return log<software_error,-1>({__FILE__, __LINE__, "Error getting exposure time"});
   }
   else
   {
      capExpTime(exptime);
      m_expTime = exptime/1000.0;
      m_expTimeSet = m_expTime; //At this point it must be true.
      updateIfChanged(m_indiP_exptime, "current", m_expTime, INDI_IDLE);
      updateIfChanged(m_indiP_exptime, "target", m_expTimeSet, INDI_IDLE);
   }

   if(getPicamParameter(m_FrameRateCalculation, PicamParameter_FrameRateCalculation) < 0)
   {
      if(powerState() != 1 || powerStateTarget() != 1) return -1; 
      return log<software_error,-1>({__FILE__, __LINE__, "Error getting frame rate"});
   }
   else
   {
      m_fps = m_FrameRateCalculation;
      updateIfChanged(m_indiP_fps, "current", m_fps, INDI_IDLE);
   }
   std::cerr << "FrameRate is: " <<  m_FrameRateCalculation << "\n";
   
   piint AdcQuality;
   if(getPicamParameter(AdcQuality, PicamParameter_AdcQuality) < 0)
   {
      std::cerr << "could not get AdcQuality\n";
   }
   std::string adcqStr = PicamEnum2String( PicamEnumeratedType_AdcQuality, AdcQuality );
   std::cerr << "AdcQuality is: " << adcqStr << "\n";

   piflt verticalShiftRate;
   if(getPicamParameter(verticalShiftRate, PicamParameter_VerticalShiftRate) < 0)
   {
      std::cerr << "could not get VerticalShiftRate\n";
   }
   std::cerr << "VerticalShiftRate is: " << verticalShiftRate << "\n";

   piflt AdcSpeed;
   if(getPicamParameter(AdcSpeed, PicamParameter_AdcSpeed) < 0)
   {
      std::cerr << "could not get AdcSpeed\n";
   }
   std::cerr << "AdcSpeed is: " << AdcSpeed << "\n";


   std::cerr << "************************************************************\n";
   
   
   piint AdcAnalogGain;
   if(getPicamParameter(AdcAnalogGain, PicamParameter_AdcAnalogGain) < 0)
   {
      std::cerr << "could not get AdcAnalogGain\n";
   }
   std::string adcgStr = PicamEnum2String( PicamEnumeratedType_AdcAnalogGain, AdcAnalogGain );
   std::cerr << "AdcAnalogGain is: " << adcgStr << "\n";

   if(m_readoutSpeedName == "ccd_00_1MHz" || m_readoutSpeedName == "ccd_01MHz")
   {
      m_emGain = 1;
   }
   else
   {
      piint AdcEMGain;
      if(getPicamParameter(AdcEMGain, PicamParameter_AdcEMGain) < 0)
      {
         std::cerr << "could not get AdcEMGain\n";
      }
      m_emGain = AdcEMGain;
   }
   
/*
   std::cerr << "Onlineable:\n";
   pibln onlineable;
   Picam_CanSetParameterOnline(m_modelHandle, PicamParameter_ReadoutControlMode,&onlineable);
   std::cerr << "ReadoutControlMode: " << onlineable << "\n"; //0

   Picam_CanSetParameterOnline(m_modelHandle, PicamParameter_AdcQuality,&onlineable);
   std::cerr << "AdcQuality: " << onlineable << "\n"; //0

   Picam_CanSetParameterOnline(m_modelHandle, PicamParameter_AdcAnalogGain,&onlineable);
   std::cerr << "AdcAnalogGain: " << onlineable << "\n"; //1

   Picam_CanSetParameterOnline(m_modelHandle, PicamParameter_DisableCoolingFan,&onlineable);
   std::cerr << "DisableCoolingFan: " << onlineable << "\n";//0

   Picam_CanSetParameterOnline(m_modelHandle, PicamParameter_SensorTemperatureSetPoint,&onlineable);
   std::cerr << "SensorTemperatureSetPoint: " << onlineable << "\n"; //0

   Picam_CanSetParameterOnline(m_modelHandle, PicamParameter_AdcEMGain,&onlineable);
   std::cerr << "AdcEMGain: " << onlineable << "\n"; //1

   Picam_CanSetParameterOnline(m_modelHandle, PicamParameter_FrameRateCalculation,&onlineable);
   std::cerr << "FrameRateCalculation: " << onlineable << "\n"; //0

   std::cerr << "************************************************************\n";
*/

   //If not previously allocated, allocate a nice big buffer to play with
   pi64s newbuffsz = framesPerReadout*readoutStride*10; //Save room for 10 frames
   if( newbuffsz >  m_acqBuff.memory_size)
   {
      if(m_acqBuff.memory)
      {
         std::cerr << "Clearing\n";
         free(m_acqBuff.memory);
         m_acqBuff.memory = NULL;
         PicamAdvanced_SetAcquisitionBuffer(m_cameraHandle, NULL);
      }

      m_acqBuff.memory_size = newbuffsz;
      std::cerr << "m_acqBuff.memory_size: " << m_acqBuff.memory_size << "\n";
      m_acqBuff.memory = malloc(m_acqBuff.memory_size);

      error = PicamAdvanced_SetAcquisitionBuffer(m_cameraHandle, &m_acqBuff);
      if(error != PicamError_None)
      {
         log<software_error>({__FILE__, __LINE__, 0, error, PicamEnum2String(PicamEnumeratedType_Error, error)});
         state(stateCodes::ERROR);

         std::cerr << "-->" << PicamEnum2String(PicamEnumeratedType_Error, error) << "\n";
      }
   }

   //Start continuous acquisition
   if(setPicamParameter(PicamParameter_ReadoutCount,(pi64s) 0) < 0)
   {
      log<software_error>({__FILE__, __LINE__, "Error setting readouts=0"});
      state(stateCodes::ERROR);
      return -1;
   }

   recordCamera();
   
   error = Picam_StartAcquisition(m_cameraHandle);
   if(error != PicamError_None)
   {
      log<software_error>({__FILE__, __LINE__, 0, error, PicamEnum2String(PicamEnumeratedType_Error, error)});
      state(stateCodes::ERROR);

      return -1;
   }

   m_dataType = _DATATYPE_UINT16; //Where does this go?
    
   return 0;

}

inline
float picamCtrl::fps()
{
   return m_fps;
}

inline
int picamCtrl::startAcquisition()
{
   return 0;
}

inline
int picamCtrl::acquireAndCheckValid()
{
   piint camTimeOut = 1000; //1 second keeps us responsive without busy-waiting too much

   PicamAcquisitionStatus status;

   PicamAvailableData available;

   PicamError error;
   error = Picam_WaitForAcquisitionUpdate(m_cameraHandle, camTimeOut, &available, &status);

   if(error == PicamError_TimeOutOccurred) 
   {
      return 1; //This sends it back to framegrabber to check for reconfig, etc.
   }

   clock_gettime(CLOCK_REALTIME, &m_currImageTimestamp);

   if(error != PicamError_None)
   {
      log<software_error>({__FILE__, __LINE__, 0, error, PicamEnum2String(PicamEnumeratedType_Error, error)});
      state(stateCodes::ERROR);

      return -1;
   }
      
   m_available.initial_readout = available.initial_readout;
   m_available.readout_count = available.readout_count;

   if(m_available.initial_readout == 0)
   {
      return 1;
   }

   //std::cerr << "readout: " << m_available.initial_readout << " " << m_available.readout_count << "\n";

   // camera time stamp
   pibyte *frame = NULL;
   pi64s metadataOffset;

   frame = (pibyte*) m_available.initial_readout;
   metadataOffset = (pi64s)frame + m_frameSize;

   pi64s *tmpPtr = NULL;
   tmpPtr = (pi64s*)metadataOffset;
   double cam_ts = (double)*tmpPtr/(double)m_tsRes;
   double delta_ts = cam_ts - m_camera_timestamp;

   // check for a frame skip
   if(delta_ts > 1.5 / m_FrameRateCalculation){
      std::cerr << "Skipped frame(s)! (Expected a " << 1000./m_FrameRateCalculation << " ms gap but got " << 1000*delta_ts << " ms)\n";
   }
   // print

   m_camera_timestamp = cam_ts; // update to latest

   //fprintf(m_outfile, "%d %-15.8f\n", m_imageStream->md->cnt0+1, (double)*tmpPtr/(double)m_tsRes);

   return 0;

}

inline
int picamCtrl::loadImageIntoStream(void * dest)
{
   if( frameGrabber<picamCtrl>::loadImageIntoStreamCopy(dest, m_available.initial_readout, m_width, m_height, m_typeSize) == nullptr) return -1;

   return 0;
}

inline
int picamCtrl::reconfig()
{
   ///\todo clean this up.  Just need to wait on acquisition update the first time probably.
   
   PicamError error = Picam_StopAcquisition(m_cameraHandle);
   if(error != PicamError_None)
   {
      log<software_error>({__FILE__, __LINE__, 0, error, PicamEnum2String(PicamEnumeratedType_Error, error)});
      state(stateCodes::ERROR);

      return -1;
   }

   pibln running = true;

   error = Picam_IsAcquisitionRunning(m_cameraHandle, &running);

   while(running)
   {
      if(MagAOXAppT::m_powerState == 0) return 0;
      sleep(1);

      error = Picam_StopAcquisition(m_cameraHandle);

      if(error != PicamError_None)
      {
         log<software_error>({__FILE__, __LINE__, 0, error, PicamEnum2String(PicamEnumeratedType_Error, error)});
         state(stateCodes::ERROR);
         return -1;
      }

      piint camTimeOut = 1000;

      PicamAcquisitionStatus status;

      PicamAvailableData available;

      error = Picam_WaitForAcquisitionUpdate(m_cameraHandle, camTimeOut, &available, &status);

//       if(! status.running )
//       {
//          std::cerr << "Not running \n";
// 
//          std::cerr << "status.running: " << status.running << "\n";
//          std::cerr << "status.errors: " << status.errors << "\n";
//          std::cerr << "CameraFaulted: " << (int)(status.errors & PicamAcquisitionErrorsMask_CameraFaulted) << "\n";
//          std::cerr << "CannectionLost: " << (int)(status.errors & PicamAcquisitionErrorsMask_ConnectionLost) << "\n";
//          std::cerr << "DataLost: " << (int)(status.errors & PicamAcquisitionErrorsMask_DataLost) << "\n";
//          std::cerr << "DataNotArriving: " << (int)(status.errors & PicamAcquisitionErrorsMask_DataNotArriving) << "\n";
//          std::cerr << "None: " << (int)(status.errors & PicamAcquisitionErrorsMask_None) << "\n";
//          std::cerr << "ShutterOverheated: " << (int)(status.errors & PicamAcquisitionErrorsMask_ShutterOverheated) << "\n";
//          std::cerr << "status.readout_rate: " << status.readout_rate << "\n";
//       }

      error = Picam_IsAcquisitionRunning(m_cameraHandle, &running);
      if(error != PicamError_None)
      {
         log<software_error>({__FILE__, __LINE__, 0, error, PicamEnum2String(PicamEnumeratedType_Error, error)});
         state(stateCodes::ERROR);
         return -1;
      }
   }

   return 0;
}



int picamCtrl::checkRecordTimes()
{
   return telemeter<picamCtrl>::checkRecordTimes(telem_stdcam());
}
   
int picamCtrl::recordTelem(const telem_stdcam *)
{
   return recordCamera(true);
}



}//namespace app
} //namespace MagAOX
#endif
