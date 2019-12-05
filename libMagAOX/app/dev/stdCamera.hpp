/** \file stdCamera.hpp
  * \brief Standard camera interface
  *
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup app_files
  */

#ifndef stdCamera_hpp
#define stdCamera_hpp


namespace MagAOX
{
namespace app
{
namespace dev 
{

#define CAMCTRL_E_NOCONFIGS (-10)

/// A camera configuration
/** a.k.a. a mode
 */   
struct cameraConfig 
{
   std::string m_configFile; ///< The file to use for this mode, e.g. an EDT configuration file.
   std::string m_serialCommand; ///< The command to send to the camera to place it in this mode.
   unsigned m_centerX {0};
   unsigned m_centerY {0};
   unsigned m_sizeX {0};
   unsigned m_sizeY {0};
   unsigned m_binningX {0};
   unsigned m_binningY {0};
   
   float m_maxFPS {0};
};

typedef std::unordered_map<std::string, cameraConfig> cameraConfigMap;

///Load the camera configurations contained in the app configuration into a map
inline
int loadCameraConfig( cameraConfigMap & ccmap, ///< [out] the map in which to place the configurations found in config
                      mx::app::appConfigurator & config ///< [in] the application configuration structure
                    )
{
   std::vector<std::string> sections;

   config.unusedSections(sections);

   if( sections.size() == 0 )
   {
      return CAMCTRL_E_NOCONFIGS;
   }
   
   for(size_t i=0; i< sections.size(); ++i)
   {
      bool fileset = config.isSetUnused(mx::app::iniFile::makeKey(sections[i], "configFile" ));
      /*bool binset = config.isSetUnused(mx::app::iniFile::makeKey(sections[i], "binning" ));
      bool sizeXset = config.isSetUnused(mx::app::iniFile::makeKey(sections[i], "sizeX" ));
      bool sizeYset = config.isSetUnused(mx::app::iniFile::makeKey(sections[i], "sizeY" ));
      bool maxfpsset = config.isSetUnused(mx::app::iniFile::makeKey(sections[i], "maxFPS" ));
      */
      
      //The configuration file tells us most things for EDT, so it's our current requirement. 
      if( !fileset ) continue;
      
      std::string configFile;
      config.configUnused(configFile, mx::app::iniFile::makeKey(sections[i], "configFile" ));
      
      std::string serialCommand;
      config.configUnused(serialCommand, mx::app::iniFile::makeKey(sections[i], "serialCommand" ));
      
      unsigned centerX = 0;
      config.configUnused(centerX, mx::app::iniFile::makeKey(sections[i], "centerX" ));
      
      unsigned centerY = 0;
      config.configUnused(centerY, mx::app::iniFile::makeKey(sections[i], "centerY" ));
      
      unsigned sizeX = 0;
      config.configUnused(sizeX, mx::app::iniFile::makeKey(sections[i], "sizeX" ));
      
      unsigned sizeY = 0;
      config.configUnused(sizeY, mx::app::iniFile::makeKey(sections[i], "sizeY" ));
      
      unsigned binningX = 0;
      config.configUnused(binningX, mx::app::iniFile::makeKey(sections[i], "binning" ));
      
      unsigned binningY = 0;
      config.configUnused(binningY, mx::app::iniFile::makeKey(sections[i], "binning" ));
      
      float maxFPS = 0;
      config.configUnused(maxFPS, mx::app::iniFile::makeKey(sections[i], "maxFPS" ));
      
      ccmap[sections[i]] = cameraConfig({configFile, serialCommand, centerX, centerY, sizeX, sizeY, binningX, binningY, maxFPS});
   }
   
   return 0;
}

/// MagAO-X standard camera interface
/** Implements the standard interface to a MagAO-X camera
  * 
  * 
  * The derived class `derivedT` must be a MagAOXApp\<true\>, and should declare this class a friend like so: 
   \code
    friend class dev::stdCamera<derivedT>;
   \endcode
  *
  * The default values of m_currentROI should be set before calling stdCamera::appStartup().
  *
  * Calls to this class's `setupConfig`, `loadConfig`, `appStartup`, `appLogic`, `appShutdown`
  * `onPowerOff`, and `whilePowerOff`,  must be placed in the derived class's functions of the same name.
  *
  * \ingroup appdev
  */
template<class derivedT>
class stdCamera
{
protected:

   /** \name Configurable Parameters
    * @{
    */
   
   cameraConfigMap m_cameraModes; ///< Map holding the possible camera mode configurations
   
   std::string m_startupMode; ///< The camera mode to load during first init after a power-on.
   
   float m_startupTemp {-999}; ///< The temperature to set after a power-on.  Set to <= -999 to not use [default].
   
   ///@}
   
   /** \name Temperature Control Interface 
     * @{
     */ 
   bool m_hasTempControl {true}; ///< Flag to set in constructor determining if this device has temperature control.  Note m_hasTemperature is implied if this is true.
   bool m_hasTemperature {true}; ///< Flag to set in constructor determining if this device has temperature output.  Only meaningful if m_hasTempControl is false.
   
   float m_minTemp {-60};
   float m_maxTemp {30};
   float m_stepTemp {0};
   
   float m_ccdTemp {-999}; ///< The current temperature, in C
   
   float m_ccdTempSetpt {-999}; ///< The desired temperature, in C

   bool m_tempControlStatus {false}; ///< Whether or not temperature control is active 
   bool m_tempControlStatusSet {false}; ///< Desired state of temperature control
   
   bool m_tempControlOnTarget {false}; ///< Whether or not the temperature control system is on its target temperature
   
   std::string m_tempControlStatusStr; ///< Camera specific description of temperature control status.
   
   ///@}
   
   /** \name Exposure Control 
     * @{
     */
   
   bool m_usesExpTime {true}; ///< Set to false in constructor to not expose the exposure time interface
   
   float m_minExpTime {0}; ///< The minimum exposure time, used for INDI attributes
   float m_maxExpTime {std::numeric_limits<float>::max()}; ///< The maximum exposure time, used for INDI attributes
   float m_stepExpTime {0}; ///< The maximum exposure time stepsize, used for INDI attributes
   
   float m_expTime {0}; ///< The current exposure time, in seconds.
   float m_expTimeSet {0}; ///< The exposure time, in seconds, as set by user.
   
   bool m_usesFPS {true}; ///< Set to false in constructor to not expose the FPS interface
   
   float m_minFPS{0};  ///< The minimum FPS, used for INDI attributes
   float m_maxFPS{std::numeric_limits<float>::max()}; ///< The maximum FPS, used for INDI attributes
   float m_stepFPS{0}; ///< The FPS step size, used for INDI attributes
   
   float m_fps {0}; ///< The current FPS.
   float m_fpsSet {0}; ///< The commanded fps, as set by user.
   
   ///@}
   
   bool m_usesModes {true}; ///< Flag to set in constructor determining if modes are offered by this camera
   
   std::string m_modeName; ///< The current mode name
   
   std::string m_nextMode; ///< The mode to be set by the next reconfiguration
   
   bool m_usesROI {true}; ///< Flag to set in constructor determining if ROIs are offered by this camera
   
   struct roi
   {
      float x;
      float y;
      int w;
      int h;
      int bin_x;
      int bin_y;
   };
   
   roi m_currentROI;
   roi m_nextROI;
   
   
   float m_minROIx {0};
   float m_maxROIx {1023};
   float m_stepROIx {0};
   
   float m_minROIy {0};
   float m_maxROIy {1023};
   float m_stepROIy {0};
   
   int m_minROIWidth {1};
   int m_maxROIWidth {1024};
   int m_stepROIWidth {1};
   
   int m_minROIHeight {1};
   int m_maxROIHeight {1024};
   int m_stepROIHeight {1};
   
   int m_minROIBinning_x {1};
   int m_maxROIBinning_x {4};
   int m_stepROIBinning_x {1};
   
   int m_minROIBinning_y {1};
   int m_maxROIBinning_y {4};
   int m_stepROIBinning_y {1};
   
public:

   ///Destructor, destroys the PdvDev structure
   ~stdCamera() noexcept;
   
   /// Setup the configuration system
   /**
     * This should be called in `derivedT::setupConfig` as
     * \code
       stdCamera<derivedT>::setupConfig(config);
       \endcode
     * with appropriate error checking.
     */
   void setupConfig(mx::app::appConfigurator & config /**< [out] the derived classes configurator*/);

   /// load the configuration system results
   /**
     * This should be called in `derivedT::loadConfig` as
     * \code
       stdCamera<derivedT>::loadConfig(config);
       \endcode
     * with appropriate error checking.
     */
   void loadConfig(mx::app::appConfigurator & config /**< [in] the derived classes configurator*/);

   /// Startup function
   /** 
     * This should be called in `derivedT::appStartup` as
     * \code
       stdCamera<derivedT>::appStartup();
       \endcode
     * with appropriate error checking.
     * 
     * You should set the default/startup values of m_currentROI as well as the min/max/step values for the ROI parameters
     * before calling this function.
     *
     * \returns 0 on success
     * \returns -1 on error, which is logged.
     */
   int appStartup();

   /// Application logic 
   /** Checks the stdCamera thread
     * 
     * This should be called from the derived's appLogic() as in
     * \code
       stdCamera<derivedT>::appLogic();
       \endcode
     * with appropriate error checking.
     * 
     * \returns 0 on success
     * \returns -1 on error, which is logged.
     */
   int appLogic();

   /// Actions on power off
   /**
     * This should be called from the derived's onPowerOff() as in
     * \code
       stdCamera<derivedT>::onPowerOff();
       \endcode
     * with appropriate error checking.
     * 
     * \returns 0 on success
     * \returns -1 on error, which is logged.
     */
   int onPowerOff();

   /// Actions while powered off
   /**
     * This should be called from the derived's whilePowerOff() as in
     * \code
       stdCamera<derivedT>::whilePowerOff();
       \endcode
     * with appropriate error checking.
     * 
     * \returns 0 on success
     * \returns -1 on error, which is logged.
     */
   int whilePowerOff();
   
   /// Application the shutdown 
   /** Shuts down the stdCamera thread
     * 
     * \code
       stdCamera<derivedT>::appShutdown();
       \endcode
     * with appropriate error checking.
     * 
     * \returns 0 on success
     * \returns -1 on error, which is logged.
     */
   int appShutdown();
   
protected:
   
   
    /** \name INDI 
      *
      *@{
      */ 
protected:
   //declare our properties
   
   pcf::IndiProperty m_indiP_temp;
   pcf::IndiProperty m_indiP_tempcont;
   pcf::IndiProperty m_indiP_tempstat;
   
   pcf::IndiProperty m_indiP_exptime;
   
   pcf::IndiProperty m_indiP_fps;
   
   pcf::IndiProperty m_indiP_mode; ///< Property used to report the current mode
   
   pcf::IndiProperty m_indiP_roi_x; ///< Property used to set the ROI x center coordinate
   pcf::IndiProperty m_indiP_roi_y; ///< Property used to set the ROI x center coordinate
   pcf::IndiProperty m_indiP_roi_w; ///< Property used to set the ROI width 
   pcf::IndiProperty m_indiP_roi_h; ///< Property used to set the ROI height 
   pcf::IndiProperty m_indiP_roi_bin_x; ///< Property used to set the ROI x binning
   pcf::IndiProperty m_indiP_roi_bin_y; ///< Property used to set the ROI y binning

   pcf::IndiProperty m_indiP_roi_set; ///< Property used to trigger setting the ROI 
   
public:

   /// The static callback function to be registered for stdCamera properties
   /** Dispatches to the relevant handler
     * 
     * \returns 0 on success.
     * \returns -1 on error.
     */
   static int st_newCallBack_stdCamera( void * app, ///< [in] a pointer to this, will be static_cast-ed to derivedT.
                                        const pcf::IndiProperty &ipRecv ///< [in] the INDI property sent with the the new property request.
                                      );
   
   /// Callback to process a NEW CCD temp request
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int newCallBack_temp( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);
   
   /// Callback to process a NEW CCD temp control request
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int newCallBack_temp_controller( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);
   
   /// Callback to process a NEW exposure time request
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int newCallBack_exptime( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);
   
   /// Callback to process a NEW fps request
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int newCallBack_fps( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);
   
   /// Callback to process a NEW mode request
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int newCallBack_mode( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);
   
   /// Callback to process a NEW roi_x request
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int newCallBack_roi_x( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);
   
   /// Callback to process a NEW roi_y request
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int newCallBack_roi_y( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);
   
   /// Callback to process a NEW roi_w request
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int newCallBack_roi_w( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);
   
   /// Callback to process a NEW roi_h request
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int newCallBack_roi_h( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);
   
   /// Callback to process a NEW bin_x request
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int newCallBack_roi_bin_x( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);
   
   /// Callback to process a NEW bin_y request
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int newCallBack_roi_bin_y( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);
   
   
   /// Callback to process a NEW roi_set request
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int newCallBack_roi_set( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);
   
   
   /// Update the INDI properties for this device controller
   /** You should call this once per main loop.
     * It is not called automatically.
     *
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int updateINDI();

   ///@}
   
   /** \name Telemeter Interface 
     * @{
     */
   
   int recordCamera( bool force = false );
   
   ///@}
   
private:
   derivedT & derived()
   {
      return *static_cast<derivedT *>(this);
   }
};

template<class derivedT>
stdCamera<derivedT>::~stdCamera() noexcept
{
   return;
}



template<class derivedT>
void stdCamera<derivedT>::setupConfig(mx::app::appConfigurator & config)
{
   if(m_hasTempControl)
   {
      config.add("camera.startupTemp", "", "camera.startupTemp", argType::Required, "camera", "startupTemp", false, "float", "The temperature setpoint to set after a power-on [C].  Default is 20 C.");
   }
   
   if(m_usesModes)
   {
      config.add("camera.startupMode", "", "camera.startupMode", argType::Required, "camera", "startupMode", false, "string", "The mode to set upon power on or application startup.");
   }
}

template<class derivedT>
void stdCamera<derivedT>::loadConfig(mx::app::appConfigurator & config)
{
   if(m_hasTempControl)
   {
      config(m_startupTemp, "camera.startupTemp");
   }
   
   if(m_usesModes)
   {
      int rv = loadCameraConfig(m_cameraModes, config);
   
      if(rv < 0)
      {
         if(rv == CAMCTRL_E_NOCONFIGS)
         {
            derivedT::template log<text_log>("No camera configurations found.", logPrio::LOG_CRITICAL);
         }
      }
      
      config(m_startupMode, "camera.startupMode");
      
   }
   
}
   


template<class derivedT>
int stdCamera<derivedT>::appStartup()
{
   
   if(m_hasTempControl)
   {
      //The min/max/step values should be set in derivedT before this is called.
      derived().createStandardIndiNumber( m_indiP_temp, "temp_ccd", m_minTemp, m_maxTemp, m_stepTemp, "%0.1f","CCD Temperature", "CCD Temperature");
      m_indiP_temp["current"].set(m_ccdTemp);
      m_indiP_temp["target"].set(m_ccdTempSetpt);
      if( derived().registerIndiPropertyNew( m_indiP_temp, st_newCallBack_stdCamera) < 0)
      {
         #ifndef STDCAMERA_TEST_NOLOG
         derivedT::template log<software_error>({__FILE__,__LINE__});
         #endif
         return -1;
      }
      
      derived().createStandardIndiToggleSw( m_indiP_tempcont, "temp_controller", "CCD Temperature", "Control On/Off");
      m_indiP_tempcont["toggle"].set(pcf::IndiElement::Off);
      if( derived().registerIndiPropertyNew( m_indiP_tempcont, st_newCallBack_stdCamera) < 0)
      {
         #ifndef STDCAMERA_TEST_NOLOG
         derivedT::template log<software_error>({__FILE__,__LINE__});
         #endif
         return -1;
      }
      
      derived().createROIndiText( m_indiP_tempstat, "temp_control", "status", "CCD Temperature", "", "CCD Temperature");
      if( derived().registerIndiPropertyReadOnly( m_indiP_tempstat ) < 0)
      {
         #ifndef STDCAMERA_TEST_NOLOG
         derivedT::template log<software_error>({__FILE__,__LINE__});
         #endif
         return -1;
      }
      
   }
   else if(m_hasTemperature)
   {
   }
   
   if(m_usesExpTime)
   {
      derived().createStandardIndiNumber( m_indiP_exptime, "exptime", m_minExpTime, m_maxExpTime, m_stepExpTime, "%0.3f");
      if( derived().registerIndiPropertyNew( m_indiP_exptime, st_newCallBack_stdCamera) < 0)
      {
         #ifndef STDCAMERA_TEST_NOLOG
         derivedT::template log<software_error>({__FILE__,__LINE__});
         #endif
         return -1;
      }
   }
   
   if(m_usesFPS)
   {
      derived().createStandardIndiNumber( m_indiP_fps, "fps", m_minFPS, m_maxFPS, m_stepFPS, "%0.2f");
      if( derived().registerIndiPropertyNew( m_indiP_fps, st_newCallBack_stdCamera) < 0)
      {
         #ifndef STDCAMERA_TEST_NOLOG
         derivedT::template log<software_error>({__FILE__,__LINE__});
         #endif
         return -1;
      }
   }
   
   if(m_usesModes)
   {
      derived().createStandardIndiText( m_indiP_mode, "mode");
      m_indiP_mode["target"] = m_startupMode;
      if( derived().registerIndiPropertyNew( m_indiP_mode, st_newCallBack_stdCamera) < 0)
      {
         #ifndef STDCAMERA_TEST_NOLOG
         derivedT::template log<software_error>({__FILE__,__LINE__});
         #endif
         return -1;
      }
   }
   
   if(m_usesROI)
   {
      //The min/max/step values should be set in derivedT before this is called.
      derived().createStandardIndiNumber( m_indiP_roi_x, "roi_x", m_minROIx, m_maxROIx, m_stepROIx, "%0.1f");
      if( derived().registerIndiPropertyNew( m_indiP_roi_x, st_newCallBack_stdCamera) < 0)
      {
         #ifndef STDCAMERA_TEST_NOLOG
         derivedT::template log<software_error>({__FILE__,__LINE__});
         #endif
         return -1;
      }
      
      derived().createStandardIndiNumber( m_indiP_roi_y, "roi_y", m_minROIy, m_maxROIy, m_stepROIy, "%0.1f");
      if( derived().registerIndiPropertyNew( m_indiP_roi_y, st_newCallBack_stdCamera) < 0)
      {
         #ifndef STDCAMERA_TEST_NOLOG
         derivedT::template log<software_error>({__FILE__,__LINE__});
         #endif
         return -1;
      }
      
      derived().createStandardIndiNumber( m_indiP_roi_w, "roi_w", m_minROIWidth, m_maxROIWidth, m_stepROIWidth, "%d");
      if( derived().registerIndiPropertyNew( m_indiP_roi_w, st_newCallBack_stdCamera) < 0)
      {
         #ifndef STDCAMERA_TEST_NOLOG
         derivedT::template log<software_error>({__FILE__,__LINE__});
         #endif
         return -1;
      }
      
      derived().createStandardIndiNumber( m_indiP_roi_h, "roi_h", m_minROIHeight, m_maxROIHeight, m_stepROIHeight, "%d");
      if( derived().registerIndiPropertyNew( m_indiP_roi_h, st_newCallBack_stdCamera) < 0)
      {
         #ifndef STDCAMERA_TEST_NOLOG
         derivedT::template log<software_error>({__FILE__,__LINE__});
         #endif
         return -1;
      }
      
      derived().createStandardIndiNumber( m_indiP_roi_bin_x, "roi_bin_x", m_minROIBinning_x, m_maxROIBinning_x, m_stepROIBinning_x, "%d");
      if( derived().registerIndiPropertyNew( m_indiP_roi_bin_x, st_newCallBack_stdCamera) < 0)
      {
         #ifndef STDCAMERA_TEST_NOLOG
         derivedT::template log<software_error>({__FILE__,__LINE__});
         #endif
         return -1;
      }
      
      derived().createStandardIndiNumber( m_indiP_roi_bin_y, "roi_bin_y", m_minROIBinning_y, m_maxROIBinning_y, m_stepROIBinning_y, "%d");
      if( derived().registerIndiPropertyNew( m_indiP_roi_bin_y, st_newCallBack_stdCamera) < 0)
      {
         #ifndef STDCAMERA_TEST_NOLOG
         derivedT::template log<software_error>({__FILE__,__LINE__});
         #endif
         return -1;
      }
   
      derived().createStandardIndiRequestSw( m_indiP_roi_set, "roi_set");
      if( derived().registerIndiPropertyNew( m_indiP_roi_set, st_newCallBack_stdCamera) < 0)
      {
         #ifndef STDCAMERA_TEST_NOLOG
         derivedT::template log<software_error>({__FILE__,__LINE__});
         #endif
         return -1;
      }
      
      
   
   }
   
   return 0;
}

template<class derivedT>
int stdCamera<derivedT>::appLogic()
{
   if( derived().state() == stateCodes::POWERON )
   {
      if(derived().powerOnWaitElapsed()) 
      {
         derived().state(stateCodes::NOTCONNECTED);

         //Set power-on defaults         
         derived().powerOnDefaults();
         
         //then set startupTemp if configured
         if(m_startupTemp > -999) m_ccdTempSetpt = m_startupTemp;
         derived().updateIfChanged(m_indiP_temp, "target", m_ccdTempSetpt, INDI_IDLE);
         
         if(m_usesROI)
         {
            //m_currentROI should be set to default/startup values in derivedT::powerOnDefaults
            m_nextROI.x = m_currentROI.x;
            m_nextROI.y = m_currentROI.y;
            m_nextROI.w = m_currentROI.w;
            m_nextROI.h = m_currentROI.h;
            m_nextROI.bin_x = m_currentROI.bin_x;
            m_nextROI.bin_y = m_currentROI.bin_y;
   
            derived().updateIfChanged(m_indiP_roi_x, "current", m_currentROI.x, INDI_IDLE);
            derived().updateIfChanged(m_indiP_roi_x, "target", m_nextROI.x, INDI_IDLE);
   
            derived().updateIfChanged(m_indiP_roi_y, "current", m_currentROI.y, INDI_IDLE);
            derived().updateIfChanged(m_indiP_roi_y, "target", m_nextROI.y, INDI_IDLE);
   
            derived().updateIfChanged(m_indiP_roi_w, "current", m_currentROI.w, INDI_IDLE);
            derived().updateIfChanged(m_indiP_roi_w, "target", m_nextROI.w, INDI_IDLE);
   
            derived().updateIfChanged(m_indiP_roi_h, "current", m_currentROI.h, INDI_IDLE);
            derived().updateIfChanged(m_indiP_roi_h, "target", m_nextROI.h, INDI_IDLE);
   
            derived().updateIfChanged(m_indiP_roi_bin_x, "current", m_currentROI.bin_x, INDI_IDLE);
            derived().updateIfChanged(m_indiP_roi_bin_x, "target", m_nextROI.bin_x, INDI_IDLE);
   
            derived().updateIfChanged(m_indiP_roi_bin_y, "current", m_currentROI.bin_y, INDI_IDLE);
            derived().updateIfChanged(m_indiP_roi_bin_y, "target", m_nextROI.bin_y, INDI_IDLE);
         }
         
         return 0;
      }
      else
      {
         return 0;
      }
   }
   
   return 0;

}

template<class derivedT>
int stdCamera<derivedT>::onPowerOff()
{
   if( !derived().m_indiDriver ) return 0;
   
   if(m_usesModes)
   {
      indi::updateIfChanged(m_indiP_mode, "current", std::string(""), derived().m_indiDriver, INDI_IDLE);
      indi::updateIfChanged(m_indiP_mode, "target", std::string(""), derived().m_indiDriver, INDI_IDLE);
   }
   
   if(m_usesROI)
   {
      indi::updateIfChanged(m_indiP_roi_x, "current", std::string(""), derived().m_indiDriver, INDI_IDLE);
      indi::updateIfChanged(m_indiP_roi_x, "target", std::string(""), derived().m_indiDriver, INDI_IDLE);
      
      indi::updateIfChanged(m_indiP_roi_y, "current", std::string(""), derived().m_indiDriver, INDI_IDLE);
      indi::updateIfChanged(m_indiP_roi_y, "target", std::string(""), derived().m_indiDriver, INDI_IDLE);
      
      indi::updateIfChanged(m_indiP_roi_w, "current", std::string(""), derived().m_indiDriver, INDI_IDLE);
      indi::updateIfChanged(m_indiP_roi_w, "target", std::string(""), derived().m_indiDriver, INDI_IDLE);
      
      indi::updateIfChanged(m_indiP_roi_h, "current", std::string(""), derived().m_indiDriver, INDI_IDLE);
      indi::updateIfChanged(m_indiP_roi_h, "target", std::string(""), derived().m_indiDriver, INDI_IDLE);
      
      indi::updateIfChanged(m_indiP_roi_bin_x, "current", std::string(""), derived().m_indiDriver, INDI_IDLE);
      indi::updateIfChanged(m_indiP_roi_bin_x, "target", std::string(""), derived().m_indiDriver, INDI_IDLE);
      
      indi::updateIfChanged(m_indiP_roi_bin_y, "current", std::string(""), derived().m_indiDriver, INDI_IDLE);
      indi::updateIfChanged(m_indiP_roi_bin_y, "target", std::string(""), derived().m_indiDriver, INDI_IDLE);
   }
   
   return 0;
}

template<class derivedT>
int stdCamera<derivedT>::whilePowerOff()
{
   return 0;
}

template<class derivedT>
int stdCamera<derivedT>::appShutdown()
{
   return 0;
}


template<class derivedT>
int stdCamera<derivedT>::st_newCallBack_stdCamera( void * app,
                                                   const pcf::IndiProperty &ipRecv
                                                 )
{
   std::string name = ipRecv.getName();
   derivedT * _app = static_cast<derivedT *>(app);
   
   if(name == "temp_ccd") return _app->newCallBack_temp(ipRecv);
   if(name == "temp_controller") return _app->newCallBack_temp_controller(ipRecv);
   if(name == "exptime") return _app->newCallBack_exptime(ipRecv);
   if(name == "fps") return _app->newCallBack_fps(ipRecv);
   if(name == "mode") return _app->newCallBack_mode(ipRecv);
   if(name == "roi_x") return _app->newCallBack_roi_x(ipRecv);
   if(name == "roi_y") return _app->newCallBack_roi_y(ipRecv);
   if(name == "roi_w") return _app->newCallBack_roi_w(ipRecv);
   if(name == "roi_h") return _app->newCallBack_roi_h(ipRecv);
   if(name == "roi_bin_x") return _app->newCallBack_roi_bin_x(ipRecv);
   if(name == "roi_bin_y") return _app->newCallBack_roi_bin_y(ipRecv);
   if(name == "roi_set") return _app->newCallBack_roi_set(ipRecv);
   
   return -1;
}

template<class derivedT>
int stdCamera<derivedT>::newCallBack_temp( const pcf::IndiProperty &ipRecv )
{
   float target;
   
   std::unique_lock<std::mutex> lock(derived().m_indiMutex);
   
   if( derived().indiTargetUpdate( m_indiP_temp, target, ipRecv, true) < 0)
   {
      derivedT::template log<software_error>({__FILE__,__LINE__});
      return -1;
   }

   m_ccdTempSetpt = target;
   return derived().setTempSetPt();
}
   
template<class derivedT>
int stdCamera<derivedT>::newCallBack_temp_controller( const pcf::IndiProperty &ipRecv)
{
   if(ipRecv.getName() != m_indiP_tempcont.getName())
   {
      derivedT::template log<software_error>({__FILE__,__LINE__, "wrong INDI property received."});
      return -1;
   }
   
   if(!ipRecv.find("toggle")) return 0;
   
   
   m_tempControlStatusSet = false;

   std::unique_lock<std::mutex> lock(derived().m_indiMutex);
   
   if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On)
   {
      m_tempControlStatusSet = true;
      derived().updateSwitchIfChanged(m_indiP_tempcont, "toggle", pcf::IndiElement::On, INDI_BUSY);
   }   
   else if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::Off)
   {
      m_tempControlStatusSet = false;
      derived().updateSwitchIfChanged(m_indiP_tempcont, "toggle", pcf::IndiElement::Off, INDI_BUSY);
   }
   
     
   return derived().setTempControl();
   
}

template<class derivedT>
int stdCamera<derivedT>::newCallBack_exptime( const pcf::IndiProperty &ipRecv)
{
   float target;

   std::unique_lock<std::mutex> lock(derived().m_indiMutex);

   if( derived().indiTargetUpdate( m_indiP_exptime, target, ipRecv, true) < 0)
   {
      derivedT::template log<software_error>({__FILE__,__LINE__});
      return -1;
   }
   
   m_expTimeSet = target;
   
   return derived().setExpTime();
   
}

template<class derivedT>
int stdCamera<derivedT>::newCallBack_fps( const pcf::IndiProperty &ipRecv)
{
   float target;

   std::unique_lock<std::mutex> lock(derived().m_indiMutex);

   if( derived().indiTargetUpdate( m_indiP_fps, target, ipRecv, true) < 0)
   {
      derivedT::template log<software_error>({__FILE__,__LINE__});
      return -1;
   }
   
   m_fpsSet = target;
   
   return derived().setFPS();
   
}

template<class derivedT>
int stdCamera<derivedT>::newCallBack_mode( const pcf::IndiProperty &ipRecv )
{
   std::string target;
   
   std::unique_lock<std::mutex> lock(derived().m_indiMutex);
   
   if( derived().indiTargetUpdate( m_indiP_mode, target, ipRecv, true) < 0)
   {
      derivedT::template log<software_error>({__FILE__,__LINE__});
      return -1;
   }
   
   if(m_cameraModes.count(target) == 0 )
   {
      return derivedT::template log<text_log, -1>("Unrecognized mode requested: " + target, logPrio::LOG_ERROR);
   }
   
   //Now signal the f.g. thread to reconfigure
   m_nextMode = target;
   derived().m_reconfig = true;
   
   return 0;
  
}
   
template<class derivedT>
int stdCamera<derivedT>::newCallBack_roi_x( const pcf::IndiProperty &ipRecv )
{
   float target;
   
   std::unique_lock<std::mutex> lock(derived().m_indiMutex);
   
   if( derived().indiTargetUpdate( m_indiP_roi_x, target, ipRecv, false) < 0)
   {
      m_nextROI.x = m_currentROI.x;
      derivedT::template log<software_error>({__FILE__,__LINE__});
      return -1;
   }
   
   m_nextROI.x = target;
   
   return 0;  
}

template<class derivedT>
int stdCamera<derivedT>::newCallBack_roi_y( const pcf::IndiProperty &ipRecv )
{
   float target;
   
   std::unique_lock<std::mutex> lock(derived().m_indiMutex);
   
   if( derived().indiTargetUpdate( m_indiP_roi_y, target, ipRecv, false) < 0)
   {
      m_nextROI.y = m_currentROI.y;
      derivedT::template log<software_error>({__FILE__,__LINE__});
      return -1;
   }
   
   m_nextROI.y = target;
   
   return 0;  
}

template<class derivedT>
int stdCamera<derivedT>::newCallBack_roi_w( const pcf::IndiProperty &ipRecv )
{
   int target;
   
   std::unique_lock<std::mutex> lock(derived().m_indiMutex);
   
   if( derived().indiTargetUpdate( m_indiP_roi_w, target, ipRecv, false) < 0)
   {
      m_nextROI.w = m_currentROI.w;
      derivedT::template log<software_error>({__FILE__,__LINE__});
      return -1;
   }
   
   m_nextROI.w = target;
   
   return 0;  
}

template<class derivedT>
int stdCamera<derivedT>::newCallBack_roi_h( const pcf::IndiProperty &ipRecv )
{
   int target;
   
   std::unique_lock<std::mutex> lock(derived().m_indiMutex);
   
   if( derived().indiTargetUpdate( m_indiP_roi_h, target, ipRecv, false) < 0)
   {
      derivedT::template log<software_error>({__FILE__,__LINE__});
      m_nextROI.h = m_currentROI.h;
      return -1;
   }
   
   m_nextROI.h = target;
   
   return 0;  
}

template<class derivedT>
int stdCamera<derivedT>::newCallBack_roi_bin_x ( const pcf::IndiProperty &ipRecv )
{
   int target;
   
   std::unique_lock<std::mutex> lock(derived().m_indiMutex);
   
   if( derived().indiTargetUpdate( m_indiP_roi_bin_x, target, ipRecv, false) < 0)
   {
      derivedT::template log<software_error>({__FILE__,__LINE__});
      m_nextROI.bin_x = m_currentROI.bin_x;
      return -1;
   }
   
   m_nextROI.bin_x = target;
   
   return 0;  
}

template<class derivedT>
int stdCamera<derivedT>::newCallBack_roi_bin_y( const pcf::IndiProperty &ipRecv )
{
   int target;
   
   std::unique_lock<std::mutex> lock(derived().m_indiMutex);
   
   if( derived().indiTargetUpdate( m_indiP_roi_bin_y, target, ipRecv, false) < 0)
   {
      derivedT::template log<software_error>({__FILE__,__LINE__});
      m_nextROI.bin_y = m_currentROI.bin_y;
      return -1;
   }
   
   m_nextROI.bin_y = target;
   
   return 0;  
}

template<class derivedT>
int stdCamera<derivedT>::newCallBack_roi_set( const pcf::IndiProperty &ipRecv )
{
   if(ipRecv.getName() != m_indiP_roi_set.getName())
   {
      derivedT::template log<software_error>({__FILE__,__LINE__, "wrong INDI property received."});
      return -1;
   }
   
   if(!ipRecv.find("request")) return 0;
   
   
   if( ipRecv["request"].getSwitchState() == pcf::IndiElement::On)
   {
      std::unique_lock<std::mutex> lock(derived().m_indiMutex);
      
      indi::updateSwitchIfChanged(m_indiP_roi_set, "request", pcf::IndiElement::Off, derived().m_indiDriver, INDI_IDLE);
      
      return derived().setNextROI();
   }
   
   return 0;  
}

template<class derivedT>
int stdCamera<derivedT>::updateINDI()
{
   if( !derived().m_indiDriver ) return 0;
   
   if(m_usesExpTime)
   {
      derived().updateIfChanged(m_indiP_exptime, "current", m_expTime, INDI_IDLE);
      derived().updateIfChanged(m_indiP_exptime, "target", m_expTimeSet, INDI_IDLE);
   }
   
   if(m_usesFPS)
   {
      derived().updateIfChanged(m_indiP_fps, "current", m_fps, INDI_IDLE);
      derived().updateIfChanged(m_indiP_fps, "target", m_fpsSet, INDI_IDLE);
   }
   
   if(m_usesModes)
   {
      if(m_nextMode == m_modeName)
      {
         indi::updateIfChanged(m_indiP_mode, "current", m_modeName, derived().m_indiDriver, INDI_IDLE);
         indi::updateIfChanged(m_indiP_mode, "target", m_nextMode, derived().m_indiDriver, INDI_IDLE);
      }
      else
      {
         indi::updateIfChanged(m_indiP_mode, "current", m_modeName, derived().m_indiDriver, INDI_BUSY);
         indi::updateIfChanged(m_indiP_mode, "target", m_nextMode, derived().m_indiDriver, INDI_BUSY);
      }
   }
   
   if(m_hasTempControl)
   {
      if(m_tempControlStatus == false)
      {
         derived().updateSwitchIfChanged( m_indiP_tempcont, "toggle", pcf::IndiElement::Off, INDI_IDLE);
         derived().updateIfChanged(m_indiP_temp, "current", m_ccdTemp, INDI_IDLE);
         derived().updateIfChanged(m_indiP_temp, "target", m_ccdTempSetpt, INDI_IDLE);
         derived().updateIfChanged( m_indiP_tempstat, "status", m_tempControlStatusStr, INDI_IDLE);
      }
      else
      {
         if(m_tempControlOnTarget)
         {
            derived().updateSwitchIfChanged( m_indiP_tempcont, "toggle", pcf::IndiElement::On, INDI_OK);
            derived().updateIfChanged(m_indiP_temp, "current", m_ccdTemp, INDI_OK);
            derived().updateIfChanged(m_indiP_temp, "target", m_ccdTempSetpt, INDI_OK);
            derived().updateIfChanged( m_indiP_tempstat, "status", m_tempControlStatusStr, INDI_OK);
         }
         else
         {
            derived().updateSwitchIfChanged( m_indiP_tempcont, "toggle", pcf::IndiElement::On, INDI_BUSY);
            derived().updateIfChanged(m_indiP_temp, "current", m_ccdTemp, INDI_BUSY);
            derived().updateIfChanged(m_indiP_temp, "target", m_ccdTempSetpt, INDI_BUSY);
            derived().updateIfChanged( m_indiP_tempstat, "status", m_tempControlStatusStr, INDI_BUSY);
         }
      }      
   }
   else if(m_hasTemperature)
   {
      derived().updateIfChanged(m_indiP_temp, "current", m_ccdTemp, INDI_IDLE);
   }
   
   return 0;
}

int recordCamera( bool force )
{
   if(!force)
   {
      return 0;
   }
   
   
   return 0;
   
}


} //namespace dev
} //namespace app
} //namespace MagAOX

#endif //stdCamera_hpp
