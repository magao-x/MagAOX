/** \file stdCamera.hpp
  * \brief Standard camera interface
  *
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup app_files
  */

#ifndef stdCamera_hpp
#define stdCamera_hpp

#include <string>
#include <unordered_map>

#include <mx/app/application.hpp>

#include "../MagAOXApp.hpp"

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

   unsigned m_digitalBinX {0};
   unsigned m_digitalBinY {0};

   float m_maxFPS {0};
};

typedef std::unordered_map<std::string, cameraConfig> cameraConfigMap;

///Load the camera configurations contained in the app configuration into a map
int loadCameraConfig( cameraConfigMap & ccmap, ///< [out] the map in which to place the configurations found in config
                      mx::app::appConfigurator & config ///< [in] the application configuration structure
                    );

/// MagAO-X standard camera interface
/** Implements the standard interface to a MagAO-X camera.  The derived class `derivedT` must
  * meet the following requirements:
  * 
  * - The derived class `derivedT` must be a `MagAOXApp\<true\>`
  *  
  * - Must declare this class a friend like so: 
  *   \code
  *       friend class dev::stdCamera<DERIVEDNAME>;  //replace DERIVEDNAME with derivedT class name
  *   \endcode
  * 
  * - Must declare the following typedef:
  *   \code
  *       typedef dev::stdCamera<DERIVEDNAME> stdCameraT; //replace DERIVEDNAME with derivedT class name
  *   \endcode
  * 
  * - Must declare a series of `static constexpr` flags to manage static compile-time configuration.  Each of these
  *   flags must be defined in derivedT to be either true or false.
  * 
  *     - Temperature Control and Status:
  *     
  *         - A static configuration variable must be defined in derivedT as
  *           \code
  *               static constexpr bool c_stdCamera_tempControl = true; //or: false
  *           \endcode
  *           which determines whether or not temperature controls are exposed. 
  *     
  *         - If `(c_stdCamera_tempControl == true)` then the derived class must implement the following interfaces
  *           \code
  *               int setTempControl(); // set temp control status according to m_tempControlStatusSet
  *               int setTempSetPt(); // set the temperature set point accordin to m_ccdTempSetpt
  *           \endcode
  *     
  *         - A static configuration variable must be defined in derivedT as
  *           \code
  *               static constexpr bool c_stdCamera_temp = true; //or: false
  *           \endcode
  *           which determines whether or not temperature reporting is exposed.  Note that if 
  *           `(c_stdCamera_tempControl == true)`, then the behavior is as if `c_stdCamera_temp == true`, 
  *           but thus constexpr must still be defined.  
  *     
  *         - If either `(c_stdCamera_tempControl == true)` or `c_stdCamera_temp == true` then the INDI property "temp_ccd" 
  *           will be updated from the value of \ref m_ccdTemp.
  *    
  *     - Readout Speed:
  * 
  *       - A static configuration variable must be defined in derivedT as
  *         \code
  *             static constexpr bool c_stdCamera_readoutSpeed = true; //or: false
  *         \endcode
  *         which determines whether or not readout speed controls are exposed.  If true, then the implementation should populate
  *         \ref m_readoutSpeedNames and \ref m_readoutSpeedNameLabels (vectors of strings) on construction to the allowed values.  This 
  *         facility is normally used to control both amplifier and readout/adc speed with names like "ccd_1MHz" and "emccd_17MHz".  
  * 
  *       - If used (and true) then the following interface must be implemented:
  *         \code
  *             int setReadoutSpeed(); // configures camera using m_readoutSpeedNameSet
  *         \endcode
  *         This configures the camera according to \ref m_readoutSpeedNameSet.
  *         The implementation must also manage \ref m_readoutSpeedName, keeping it up to date with the current setting.  
  * 
  *       - If true, the configuration setting "camera.defaultReadoutSpeed"
  *         is also exposed, and \ref m_defaultReadoutSpeed will be set according to it.  The implementation can
  *         set a sensible default on construction.
  *     
  *     - Vertical Shift Speed:
  * 
  *        - A static configuration variable must be defined in derivedT as
  *          \code
  *              static constexpr bool c_stdCamera_vShiftSpeed = true; //or: false
  *          \endcode
  *          which determines whether or not vertical shift speed controls are exposed. 
  * 
  *        - If true, then the implementation should populate \ref m_vShiftSpeedNames and \ref m_vShiftSpeedLabels 
  *          (vectors of strings) on construction to the allowed values.  This 
  *          facility is normally used with names like "0_3us" and "1_3us".  
  * 
  *        - If true then the following interface must be defined: 
  *          \code 
  *              int setVShiftSpeed(); // configures camera according to m_vShiftSpeedNameSet
  *          \endcode
  *          function must be defined which sets the camera according to \ref m_vShiftSpeedNameSet.
  *          The implementation must also manage m_vShiftSpeedName, keeping it up to date.  
  * 
  *        - The configuration setting "camera.defaultVShiftSpeed"
  *          is also exposed, and \ref m_defaultVShiftSpeed will be set accordingly.  derivedT can set a sensible default
  *          on constuction.
  *     
  *     - Exposure Time:
  *        - A static configuration variable must be defined in derivedT as
  *          \code
  *              static constexpr bool c_stdCamera_exptimeCtrl = true; //or: false
  *          \endcode
  *        - If true, the following interface must be implemented:
  *          \code
  *              int setExpTime(); // set camera exposure time according to m_expTimeSet.
  *          \endcode
  *          to configure the camera according to \ref m_expTimeSet.  derivedT must also keep \ref m_expTime up to date.
  * 
  *     - Frames per Second (FPS) Control and Status:
  *         - A static configuration variable must be defined in derivedT as
  *           \code
  *               static constexpr bool c_stdCamera_fpsCtrl = true; //or: false
  *           \endcode
  *   
  *         - If that is set to true the derivedT must implement
  *           \code
  *               int setFPS(); // set camera FS according to m_fps
  *           \endcode
  *           to configure the camera according to \ref m_fpsSet.  
  *   
  *         - A static configuration variable must be defined in derivedT as
  *           \code
  *              static constexpr bool c_stdCamera_fps = true; //or: false
  *           \endcode
  *           Note that the value of c_stdCamera_fps does not matter if c_stdCamera_fpsCtrl == true.
  *   
  *         - If either `c_stdCamera_fpsCtrl == true` or `c_stdCamera_fps == true` then derivedT must also 
  *           keep \ref m_fps up to date.
  *     
  *     - Synchro Control:
  *       
  *       - A static configuration variable must be defined in derivedT as
  *         \code
  *             static constexpr bool c_stdCamera_synchro = true; //or: false
  *         \endcode
  *       - If that is set to true the derivedT must implement
  *         \code
  *             int setSynchro(); // configure the camera based m_synchroSet.
  *         \endcode
  *         to configure the camera based on \ref m_synchroSet.  The implementation should also keep
  *         \ref m_synchro up to date.
  * 
  *     - EM Gain:
  *        - A static configuration variable must be defined in derivedT as
  *          \code
  *              static constexpr bool c_stdCamera_emGain = true; //or: false
  *          \endcode
  *          which determines whether or not EM gain controls are exposed.  
  *  
  *        - If the camera uses EM Gain, then a function 
  *          \code
  *              int setEMGain(); // set EM gain based on m_emGainSet.
  *          \endcode
  *          must be defined which sets the camera EM Gain to \ref m_emGainSet.  
  * 
  *        - If true the implementation must keep \ref m_emGain up to date.  
  *  
  *        - If true the value of \ref m_maxEMGain should be set by the implementation and managed
  *          as needed. Additionally the configuration setting "camera.maxEMGain" is exposed.
  *     
  *     - Camera Modes:
  * 
  *       - A static configuration variable must be defined in derivedT as
  *         \code
  *             static constexpr bool c_stdCamera_usesModes= true; //or: false
  *         \endcode
  * 
  *       - If true, then modes are read from the configuration file.  See \ref loadCameraConfig()
  *     
  *       - If true, then the configuration setting "camera.startupMode" is exposed, which sets the mode at startup by its name.
  * 
  *     - Regions of Interest 
  *       
  *       - A static configuration variable must be defined in derivedT as
  *         \code
  *             static constexpr bool c_stdCamera_usesROI = true; //or: false
  *         \endcode
  *       
  *       - The default values of m_currentROI should be set before calling stdCamera::appStartup().
  *       
  *       - The derived class must implement:
  *         \code
  *             int checkNextROI(); // verifies m_nextROI values and modifies to closest valid values if needed
  *             int setNextROI(); // sets the ROI to the new target values.
  *         \endcode
  *     
  *     - Crop Mode ROIs:
  *       
  *        - A static configuration variable must be defined in derivedT as
  *          \code
  *              static constexpr bool c_stdCamera_cropMode = true; //or: false
  *          \endcode
  * 
  *        - If true the derived class must implement 
  *          \code
  *              int setCropMode(); // set crop mode according to m_cropModeSet
  *          \endcode
  *          which changes the crop mode according to \ref m_cropModeSet. 
  *  
  *        - `derivedT` must also maintain the value of \ref m_cropMode.
  *     
  *     - Shutters:
  * 
  *       - A static configuration variable must be defined in derivedT as
  *         \code
  *             static constexpr bool c_stdCamera_hasShutter = true; //or: false
  *         \endcode
  *       
  *       - If true the following interface must be implemented:
  *         \code
  *             int setShutter(int); // shut the shutter if 0, open the shutter otherwise.
  *         \endcode
  *         which shuts the shutter if the argument is 0, opens it otherwise.
  *     
  *     - State:
  *       
  *       - A static configuration variable must be defined in derivedT as
  *         \code
  *             static constexpr bool c_stdCamera_usesStateString = true; //or: false
  *         \endcode
  *         which determines whether the class provides a state string for dark management.
  *       
  *       - If true, the following functions must be defined in derivedT:
  *         \code
  *             std::string stateString(); //String capturing the current state.  Must not include "__T".
  *             bool stateStringValid(); //Whether or not the current state string is valid, i.e. not changing.
  *         \endcode
  * 
  * 
  * - The derived class must implement:
  *   \code
  *   int powerOnDefaults(); // called on power-on after powerOnWaitElapsed has occurred.
  *   \endcode
  * 
  * - Calls to this class's setupConfig(), loadConfig(), appStartup(), appLogic(), appShutdown()
  *   onPowerOff(), and whilePowerOff(),  must be placed in the derived class's functions of the same name.
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
   
   std::string m_defaultReadoutSpeed; ///< The default readout speed of the camera.
   std::string m_defaultVShiftSpeed; ///< The default readout speed of the camera.
   
   ///@}
   
   /** \name Temperature Control Interface 
     * @{
     */ 
   
   float m_minTemp {-60};
   float m_maxTemp {30};
   float m_stepTemp {0};
   
   float m_ccdTemp {-999}; ///< The current temperature, in C
   
   float m_ccdTempSetpt {-999}; ///< The desired temperature, in C

   bool m_tempControlStatus {false}; ///< Whether or not temperature control is active 
   bool m_tempControlStatusSet {false}; ///< Desired state of temperature control
   
   bool m_tempControlOnTarget {false}; ///< Whether or not the temperature control system is on its target temperature
   
   std::string m_tempControlStatusStr; ///< Camera specific description of temperature control status.
   
   pcf::IndiProperty m_indiP_temp;
   pcf::IndiProperty m_indiP_tempcont;
   pcf::IndiProperty m_indiP_tempstat;

   ///@}
   
   /** \name Readout Control 
     * @{
     */

   std::vector<std::string> m_readoutSpeedNames;
   std::vector<std::string> m_readoutSpeedNameLabels;
   
   std::string m_readoutSpeedName; ///< The current readout speed name
   std::string m_readoutSpeedNameSet; ///< The user requested readout speed name, to be set by derived()
   
   std::vector<std::string> m_vShiftSpeedNames;
   std::vector<std::string> m_vShiftSpeedNameLabels;
   
   std::string m_vShiftSpeedName; ///< The current vshift speed name
   std::string m_vShiftSpeedNameSet; ///< The user requested vshift speed name, to be set by derived()
      
   float m_adcSpeed {0};
   float m_vshiftSpeed {0};

   float m_emGain {1}; ///< The camera's current EM gain (if available).
   float m_emGainSet {1}; ///< The camera's EM gain, as set by the user.
   float m_maxEMGain {1}; ///< The configurable maximum EM gain.  To be enforced in derivedT.

   pcf::IndiProperty m_indiP_readoutSpeed;
   pcf::IndiProperty m_indiP_vShiftSpeed;

   pcf::IndiProperty m_indiP_emGain;

   ///@}
   
   /** \name Exposure Control 
     * @{
     */   
   float m_minExpTime {0}; ///< The minimum exposure time, used for INDI attributes
   float m_maxExpTime {std::numeric_limits<float>::max()}; ///< The maximum exposure time, used for INDI attributes
   float m_stepExpTime {0}; ///< The maximum exposure time stepsize, used for INDI attributes
   
   float m_expTime {0}; ///< The current exposure time, in seconds.
   float m_expTimeSet {0}; ///< The exposure time, in seconds, as set by user.
      
   float m_minFPS{0};  ///< The minimum FPS, used for INDI attributes
   float m_maxFPS{std::numeric_limits<float>::max()}; ///< The maximum FPS, used for INDI attributes
   float m_stepFPS{0}; ///< The FPS step size, used for INDI attributes
   
   float m_fps {0}; ///< The current FPS.
   float m_fpsSet {0}; ///< The commanded fps, as set by user.
   
   pcf::IndiProperty m_indiP_exptime;
   
   pcf::IndiProperty m_indiP_fps;

   ///@}
   
   /** \name External Synchronization 
     * @{ 
     */ 
   bool m_synchroSet {false}; ///< Target status of m_synchro 

   bool m_synchro {false};    ///< Status of synchronization, true is on, false is off.

   pcf::IndiProperty m_indiP_synchro;

   ///@}

   /** \name Modes
     *
     * @{
     */ 
   std::string m_modeName; ///< The current mode name
   
   std::string m_nextMode; ///< The mode to be set by the next reconfiguration
   
   pcf::IndiProperty m_indiP_mode; ///< Property used to report the current mode
   
   pcf::IndiProperty m_indiP_reconfig; ///< Request switch which forces the framegrabber to go through the reconfigure process.

   ///@}
   
   /** \name ROIs 
     * ROI controls are exposed if derivedT::c_stdCamera_usesROI==true
     * @{
     */ 
   struct roi
   {
      float x {0};
      float y {0};
      int w {0};
      int h {0};
      int bin_x {0};
      int bin_y {0};
   };
   
   roi m_currentROI;
   roi m_nextROI;
   roi m_lastROI;
   
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
   
   float m_default_x {0};   ///< Power-on ROI center x coordinate.
   float m_default_y {0};   ///< Power-on ROI center y coordinate.
   int m_default_w {0};     ///< Power-on ROI width.
   int m_default_h {0};     ///< Power-on ROI height.
   int m_default_bin_x {1}; ///< Power-on ROI x binning. 
   int m_default_bin_y {1}; ///< Power-on ROI y binning.
      
   float m_full_x{0}; ///< The full ROI center x coordinate.
   float m_full_y{0}; ///< The full ROI center y coordinate.
   int m_full_w{0}; ///< The full ROI width.
   int m_full_h{0}; ///< The full ROI height.
   int m_full_bin_x{1}; ///< The x-binning in the full ROI. 
   int m_full_bin_y{1}; ///< The y-binning in the full ROI. 

   float m_full_currbin_x{0}; ///< The current-binning full ROI center x coordinate.
   float m_full_currbin_y{0}; ///< The current-binning full ROI center y coordinate.
   int m_full_currbin_w{0}; ///< The current-binning full ROI width.
   int m_full_currbin_h{0}; ///< The current-binning full ROI height.

   pcf::IndiProperty m_indiP_roi_x; ///< Property used to set the ROI x center coordinate
   pcf::IndiProperty m_indiP_roi_y; ///< Property used to set the ROI x center coordinate
   pcf::IndiProperty m_indiP_roi_w; ///< Property used to set the ROI width 
   pcf::IndiProperty m_indiP_roi_h; ///< Property used to set the ROI height 
   pcf::IndiProperty m_indiP_roi_bin_x; ///< Property used to set the ROI x binning
   pcf::IndiProperty m_indiP_roi_bin_y; ///< Property used to set the ROI y binning

   pcf::IndiProperty m_indiP_fullROI; ///< Property used to preset the full ROI dimensions.
   
   pcf::IndiProperty m_indiP_roi_check; ///< Property used to trigger checking the target ROI

   pcf::IndiProperty m_indiP_roi_set; ///< Property used to trigger setting the ROI 
   
   pcf::IndiProperty m_indiP_roi_full; ///< Property used to trigger setting the full ROI.
   pcf::IndiProperty m_indiP_roi_fullbin; ///< Property used to trigger setting the full in current binning ROI.
   pcf::IndiProperty m_indiP_roi_loadlast; ///< Property used to trigger loading the last ROI as the target.
   pcf::IndiProperty m_indiP_roi_last; ///< Property used to trigger setting the last ROI.
   pcf::IndiProperty m_indiP_roi_default; ///< Property used to trigger setting the default and startup ROI.
   
   ///@}
   
   /** \name Crop Mode
     * Crop mode controls are exposed if derivedT::c_stdCamera_cropMode==true
     * @{
     */ 
   bool m_cropMode {false}; ///< Status of crop mode ROIs, if enabled for this camera.
   bool m_cropModeSet {false}; ///< Desired status of crop mode ROIs, if enabled for this camera.
   
   pcf::IndiProperty m_indiP_cropMode; ///< Property used to toggle crop mode on and off.
   ///@}
   
   /** \name Shutter Control
     * Shutter controls are exposed if derivedT::c_stdCamera_hasShutter == true.
     * @{
     */   
   std::string m_shutterStatus {"UNKNOWN"};
   int m_shutterState {-1}; /// State of the shutter.  0 = shut, 1 = open, -1 = unknown.
   
   pcf::IndiProperty m_indiP_shutterStatus; ///< Property to report shutter status
   pcf::IndiProperty m_indiP_shutter; ///< Property used to control the shutter, a switch.
 
   ///@}
   
   /** \name State String
     * The State string is exposed if derivedT::c_stdCamera_usesStateString is true.
     * @{
     */  
   pcf::IndiProperty m_indiP_stateString;
   ///@}

public:

   ///Destructor.
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

protected:
   //workers to create indi variables if needed
   int createReadoutSpeed(const mx::meta::trueFalseT<true> & t);

   int createReadoutSpeed(const mx::meta::trueFalseT<false> & f);

   int createVShiftSpeed(const mx::meta::trueFalseT<true> & t);

   int createVShiftSpeed(const mx::meta::trueFalseT<false> & f);

public:   
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
     * The INDI mutex should be locked before calling.
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
   
   /// Application shutdown 
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
public:

   /// The static callback function to be registered for stdCamera properties
   /** Calls newCallback_stdCamera
     * 
     * \returns 0 on success.
     * \returns -1 on error.
     */
   static int st_newCallBack_stdCamera( void * app, ///< [in] a pointer to this, will be static_cast-ed to derivedT.
                                        const pcf::IndiProperty &ipRecv ///< [in] the INDI property sent with the the new property request.
                                      );
   
   /// The callback function for stdCamera properties
   /** Dispatches to the relevant handler
     * 
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int newCallBack_stdCamera( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);

   /// Interface to setTempSetPt when the derivedT has temperature control
   /** Tag-dispatch resolution of c_stdCamera_tempControl==true will call this function.
     * Calls derivedT::setTempSetPt. 
     */
   int setTempSetPt( const mx::meta::trueFalseT<true> & t);

   /// Interface to setTempSetPt when the derivedT does not have temperature control
   /** Tag-dispatch resolution of c_stdCamera_tempControl==false will call this function.
     * Prevents requiring derivedT::setTempSetPt. 
     */
   int setTempSetPt( const mx::meta::trueFalseT<false> & f);
   
   /// Callback to process a NEW CCD temp request
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int newCallBack_temp( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);

   /// Interface to setTempControl when the derivedT has temperature control
   /** Tag-dispatch resolution of c_stdCamera_tempControl==true will call this function.
     * Calls derivedT::setTempControl. 
     */
   int setTempControl( const mx::meta::trueFalseT<true> & t);

   /// Interface to setTempControl when the derivedT does not have temperature control
   /** Tag-dispatch resolution of c_stdCamera_tempControl==false will call this function.
     * Prevents requiring derivedT::setTempControl. 
     */
   int setTempControl( const mx::meta::trueFalseT<false> & f); 
   
   /// Callback to process a NEW CCD temp control request
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int newCallBack_temp_controller( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);
   
   /// Interface to setReadoutSpeed when the derivedT has readout speed control
   /** Tag-dispatch resolution of c_stdCamera_readoutSpeed==true will call this function.
     * Calls derivedT::setReadoutSpeed. 
     */
   int setReadoutSpeed( const mx::meta::trueFalseT<true> & t);
   
   /// Interface to setReadoutSpeed when the derivedT does not have readout speed control
   /** Tag-dispatch resolution of c_stdCamera_readoutSpeed==false will call this function.
     * Just returns 0.
     */
   int setReadoutSpeed( const mx::meta::trueFalseT<false> & f);
   
   /// Callback to process a NEW readout speed  request
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int newCallBack_readoutSpeed( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);
   
   /// Interface to setVShiftSpeed when the derivedT has vshift speed control
   /** Tag-dispatch resolution of c_stdCamera_vShiftSpeed==true will call this function.
     * Calls derivedT::setVShiftSpeed. 
     */
   int setVShiftSpeed( const mx::meta::trueFalseT<true> & t);
   
   /// Interface to setVShiftSpeed when the derivedT does not have vshift speed control
   /** Tag-dispatch resolution of c_stdCamera_vShiftSpeed==false will call this function.
     * Just returns 0.
     */
   int setVShiftSpeed( const mx::meta::trueFalseT<false> & f);
   
   /// Callback to process a NEW vshift speed  request
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int newCallBack_vShiftSpeed( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);
   
   /// Interface to setEMGain when the derivedT has EM Gain
   /** Tag-dispatch resolution of c_stdCamera_emGain==true will call this function.
     * Calls derivedT::setEMGain. 
     */
   int setEMGain( const mx::meta::trueFalseT<true> & t);
   
   /// Interface to setEMGain when the derivedT does not have EM Gain
   /** Tag-dispatch resolution of c_stdCamera_emGain==false will call this function.
     * This prevents requiring derivedT to have its own setEMGain(). 
     */
   int setEMGain( const mx::meta::trueFalseT<false> & f);
   
   /// Callback to process a NEW EM gain request
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int newCallBack_emgain( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);
   
   /// Interface to setExpTime when the derivedT uses exposure time controls
   /** Tag-dispatch resolution of c_stdCamera_exptimeCtrl==true will call this function.
     * Calls derivedT::setExpTime. 
     */
   int setExpTime( const mx::meta::trueFalseT<true> & t );
   
   /// Interface to setExptime when the derivedT does not use exposure time controls.
   /** Tag-dispatch resolution of c_stdCamera_exptimeCtrl==false will call this function.
     * This prevents requiring derivedT to have its own setExpTime(). 
     */
   int setExpTime( const mx::meta::trueFalseT<false> & f );
   
   /// Callback to process a NEW exposure time request
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int newCallBack_exptime( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);
   
   /// Interface to setFPS when the derivedT uses FPS controls
   /** Tag-dispatch resolution of c_stdCamera_fpsCtrl==true will call this function.
     * Calls derivedT::setFPS. 
     */
   int setFPS( const mx::meta::trueFalseT<true> & t );
   
   /// Interface to setFPS when the derivedT does not use FPS controls.
   /** Tag-dispatch resolution of c_stdCamera_hasFPS==false will call this function.
     * This prevents requiring derivedT to have its own setFPS(). 
     */
   int setFPS( const mx::meta::trueFalseT<false> & f );
   
   /// Callback to process a NEW fps request
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int newCallBack_fps( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);
   
   /// Interface to setSynchro when the derivedT has synchronization
   /** Tag-dispatch resolution of c_stdCamera_synchro==true will call this function.
     * Calls derivedT::setSynchro. 
     */
   int setSynchro( const mx::meta::trueFalseT<true> & t );
   
   /// Interface to setSynchro when the derivedT does not have synchronization
   /** Tag-dispatch resolution of c_stdCamera_ynchro==false will call this function.
     * This prevents requiring derivedT to have its own setSynchro(). 
     */
   int setSynchro( const mx::meta::trueFalseT<false> & f );
   
   /// Callback to process a NEW synchro request
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int newCallBack_synchro( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);

   /// Callback to process a NEW mode request
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int newCallBack_mode( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);
   
   /// Callback to process a NEW reconfigure request
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int newCallBack_reconfigure( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);
   
   /// Interface to setCropMode when the derivedT has crop mode
   /** Tag-dispatch resolution of c_stdCamera_cropMode==true will call this function.
     * Calls derivedT::setCropMode. 
     */
   int setCropMode( const mx::meta::trueFalseT<true> & t);
   
   /// Interface to setCropMode when the derivedT does not have crop mode
   /** Tag-dispatch resolution of c_stdCamera_cropMode==false will call this function.
     * This prevents requiring derivedT to have its own setCropMode(). 
     */
   int setCropMode( const mx::meta::trueFalseT<false> & f);
   
   /// Callback to process a NEW cropMode request
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int newCallBack_cropMode( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);
   
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
   
   /// Interface to checkNextROI when the derivedT uses ROIs
   /** Tag-dispatch resolution of c_stdCamera_usesROI==true will call this function.
     * Calls derivedT::checkNextROI. 
     */
   int checkNextROI( const mx::meta::trueFalseT<true> & t);
   
   /// Interface to checkNextROI when the derivedT does not use ROIs.
   /** Tag-dispatch resolution of c_stdCamera_usesROI==false will call this function.
     * This prevents requiring derivedT to have its own checkNextROI(). 
     */
   int checkNextROI( const mx::meta::trueFalseT<false> & f);
   
   /// Callback to process a NEW roi_check request
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int newCallBack_roi_check( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);

   /// Interface to setNextROI when the derivedT uses ROIs
   /** Tag-dispatch resolution of c_stdCamera_usesROI==true will call this function.
     * Calls derivedT::setNextROI. 
     */
   int setNextROI( const mx::meta::trueFalseT<true> & t);
   
   /// Interface to setNextROI when the derivedT does not use ROIs.
   /** Tag-dispatch resolution of c_stdCamera_usesROI==false will call this function.
     * This prevents requiring derivedT to have its own setNextROI(). 
     */
   int setNextROI( const mx::meta::trueFalseT<false> & f);
   
   /// Callback to process a NEW roi_set request
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int newCallBack_roi_set( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);
   
   /// Callback to process a NEW roi_full request
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int newCallBack_roi_full( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);
   
   /// Callback to process a NEW roi_fullbin request
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int newCallBack_roi_fullbin( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);

   /// Callback to process a NEW roi_loadlast request
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int newCallBack_roi_loadlast( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);

   /// Callback to process a NEW roi_last request
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int newCallBack_roi_last( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);
   
   /// Callback to process a NEW roi_default request
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int newCallBack_roi_default( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);
   
   /// Interface to setShutter when the derivedT has a shutter
   /** Tag-dispatch resolution of c_stdCamera_hasShutter==true will call this function.
     * Calls derivedT::setShutter. 
     */
   int setShutter( int ss,
                   const mx::meta::trueFalseT<true> & t
                 );
   
   /// Interface to setShutter when the derivedT does not have a shutter.
   /** Tag-dispatch resolution of c_stdCamera_hasShutter==false will call this function.
     * This prevents requiring derivedT to have its own setShutter(). 
     */
   int setShutter( int ss,
                   const mx::meta::trueFalseT<false> & f
                 );
   
   /// Callback to process a NEW shutter request
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int newCallBack_shutter( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);
   
   /// Interface to stateString when the derivedT provides it
   /** Tag-dispatch resolution of c_stdCamera_usesStateString==true will call this function.
     * Calls derivedT::stateString. 
     */
   std::string stateString( const mx::meta::trueFalseT<true> & t);
   
   /// Interface to stateString when the derivedT does not provide it
   /** Tag-dispatch resolution of c_stdCamera_usesStateString==false will call this function.
     * returns "". 
     */
   std::string stateString( const mx::meta::trueFalseT<false> & f);

   /// Interface to stateStringValid when the derivedT provides it
   /** Tag-dispatch resolution of c_stdCamera_usesStateString==true will call this function.
     * Calls derivedT::stateStringValid. 
     */
   bool stateStringValid( const mx::meta::trueFalseT<true> & t);
   
   /// Interface to stateStringValid when the derivedT does not provide it
   /** Tag-dispatch resolution of c_stdCamera_usesStateString==false will call this function.
     * returns false. 
     */
   bool stateStringValid( const mx::meta::trueFalseT<false> & f);

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
   if(derivedT::c_stdCamera_tempControl)
   {
      config.add("camera.startupTemp", "", "camera.startupTemp", argType::Required, "camera", "startupTemp", false, "float", "The temperature setpoint to set after a power-on [C].  Default is 20 C.");
   }
   
   if(derivedT::c_stdCamera_readoutSpeed)
   {
      config.add("camera.defaultReadoutSpeed", "", "camera.defaultReadoutSpeed", argType::Required, "camera", "defaultReadoutSpeed", false, "string", "The default amplifier and readout speed.");
   }
   
   if(derivedT::c_stdCamera_vShiftSpeed)
   {
      config.add("camera.defaultVShiftSpeed", "", "camera.defaultVShiftSpeed", argType::Required, "camera", "defaultVShiftSpeed", false, "string", "The default vertical shift speed.");
   }
   
   if(derivedT::c_stdCamera_emGain)
   {
      config.add("camera.maxEMGain", "", "camera.maxEMGain", argType::Required, "camera", "maxEMGain", false, "unsigned", "The maximum EM gain which can be set by the user.");
   }
   
   if(derivedT::c_stdCamera_usesModes)
   {
      config.add("camera.startupMode", "", "camera.startupMode", argType::Required, "camera", "startupMode", false, "string", "The mode to set upon power on or application startup.");
   }
   
   if(derivedT::c_stdCamera_usesROI)
   {
      config.add("camera.default_x", "", "camera.default_x", argType::Required, "camera", "default_x", false, "float", "The default ROI x position.");
      config.add("camera.default_y", "", "camera.default_y", argType::Required, "camera", "default_y", false, "float", "The default ROI y position.");
      config.add("camera.default_w", "", "camera.default_w", argType::Required, "camera", "default_w", false, "int", "The default ROI width.");
      config.add("camera.default_h", "", "camera.default_h", argType::Required, "camera", "default_h", false, "int", "The default ROI height.");
      config.add("camera.default_bin_x", "", "camera.default_bin_x", argType::Required, "camera", "default_bin_x", false, "int", "The default ROI x binning.");
      config.add("camera.default_bin_y", "", "camera.default_bin_y", argType::Required, "camera", "default_bin_y", false, "int", "The default ROI y binning.");
   }
}

template<class derivedT>
void stdCamera<derivedT>::loadConfig(mx::app::appConfigurator & config)
{
   if(derivedT::c_stdCamera_tempControl)
   {
      config(m_startupTemp, "camera.startupTemp");
   }

   if(derivedT::c_stdCamera_readoutSpeed)
   {
      config(m_defaultReadoutSpeed, "camera.defaultReadoutSpeed");
   }

   if(derivedT::c_stdCamera_vShiftSpeed)
   {
      config(m_defaultVShiftSpeed, "camera.defaultVShiftSpeed");
   }
   
   if(derivedT::c_stdCamera_emGain)
   {
      config(m_maxEMGain, "camera.maxEMGain");
   }
   
   if(derivedT::c_stdCamera_usesModes)
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
   
   if(derivedT::c_stdCamera_usesROI)
   {
      config(m_default_x, "camera.default_x");
      config(m_default_y, "camera.default_y");
      config(m_default_w, "camera.default_w");
      config(m_default_h, "camera.default_h");
      config(m_default_bin_x, "camera.default_bin_x");
      config(m_default_bin_y, "camera.default_bin_y");
   }
}
   
template<class derivedT>
int stdCamera<derivedT>::createReadoutSpeed(const mx::meta::trueFalseT<true> & t)
{
   static_cast<void>(t);
   
   derived().createStandardIndiSelectionSw(m_indiP_readoutSpeed, "readout_speed", m_readoutSpeedNames, "Readout Speed");
   
   //Set the labes if provided
   if(m_readoutSpeedNameLabels.size() == m_readoutSpeedNames.size())
   {
      for(size_t n=0; n< m_readoutSpeedNames.size(); ++n) m_indiP_readoutSpeed[m_readoutSpeedNames[n]].setLabel(m_readoutSpeedNameLabels[n]);
   }
   
   derived().registerIndiPropertyNew(m_indiP_readoutSpeed, st_newCallBack_stdCamera);
   
   return 0;
}  

template<class derivedT>
int stdCamera<derivedT>::createReadoutSpeed(const mx::meta::trueFalseT<0> & f)
{
   static_cast<void>(f);
   
   return 0;
}

template<class derivedT>
int stdCamera<derivedT>::createVShiftSpeed(const mx::meta::trueFalseT<true> & t)
{
   static_cast<void>(t);
   
   derived().createStandardIndiSelectionSw(m_indiP_vShiftSpeed, "vshift_speed", m_vShiftSpeedNames, "Vert. Shift Speed");
   
   if(m_vShiftSpeedNameLabels.size() == m_vShiftSpeedNames.size())
   {
      for(size_t n=0; n< m_vShiftSpeedNames.size(); ++n) m_indiP_vShiftSpeed[m_vShiftSpeedNames[n]].setLabel(m_vShiftSpeedNameLabels[n]);
   }
   
   derived().registerIndiPropertyNew(m_indiP_vShiftSpeed, st_newCallBack_stdCamera);
   
   return 0;
}  

template<class derivedT>
int stdCamera<derivedT>::createVShiftSpeed(const mx::meta::trueFalseT<0> & f)
{
   static_cast<void>(f);
   
   return 0;
}

template<class derivedT>
int stdCamera<derivedT>::appStartup()
{
   
   if(derivedT::c_stdCamera_tempControl)
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
   else if(derivedT::c_stdCamera_temp)
   {
      derived().createROIndiNumber( m_indiP_temp, "temp_ccd", "CCD Temperature", "CCD Temperature");
      m_indiP_temp.add(pcf::IndiElement("current"));
      m_indiP_temp["current"].set(m_ccdTemp);
      if( derived().registerIndiPropertyReadOnly( m_indiP_temp) < 0)
      {
         #ifndef STDCAMERA_TEST_NOLOG
         derivedT::template log<software_error>({__FILE__,__LINE__});
         #endif
         return -1;
      }
   }
   
   if(derivedT::c_stdCamera_readoutSpeed)
   {
      mx::meta::trueFalseT<derivedT::c_stdCamera_readoutSpeed> tf;
      if(createReadoutSpeed(tf) < 0)
      {
         #ifndef STDCAMERA_TEST_NOLOG
         derivedT::template log<software_error>({__FILE__,__LINE__});
         #endif
         return -1;
      }
      
   }
   
   if(derivedT::c_stdCamera_vShiftSpeed)
   {
      mx::meta::trueFalseT<derivedT::c_stdCamera_vShiftSpeed> tf;
      if(createVShiftSpeed(tf) < 0)
      {
         #ifndef STDCAMERA_TEST_NOLOG
         derivedT::template log<software_error>({__FILE__,__LINE__});
         #endif
         return -1;
      }
      
   }
   
   if(derivedT::c_stdCamera_emGain)
   {
      derived().createStandardIndiNumber( m_indiP_emGain, "emgain", 0, 1000, 1, "%0.3f");
      if( derived().registerIndiPropertyNew( m_indiP_emGain, st_newCallBack_stdCamera) < 0)
      {
         #ifndef STDCAMERA_TEST_NOLOG
         derivedT::template log<software_error>({__FILE__,__LINE__});
         #endif
         return -1;
      }
   }
   
   if(derivedT::c_stdCamera_exptimeCtrl)
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
   
   if(derivedT::c_stdCamera_fpsCtrl)
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
   else if(derivedT::c_stdCamera_fps)
   {
      derived().createROIndiNumber( m_indiP_fps, "fps");
      m_indiP_fps.add(pcf::IndiElement("current"));
      m_indiP_fps["current"].setMin(m_minFPS);
      m_indiP_fps["current"].setMax(m_maxFPS);
      m_indiP_fps["current"].setStep(m_stepFPS);
      m_indiP_fps["current"].setFormat("%0.2f");

      if( derived().registerIndiPropertyReadOnly( m_indiP_fps ) < 0)
      {
         #ifndef STDCAMERA_TEST_NOLOG
         derivedT::template log<software_error>({__FILE__,__LINE__});
         #endif
         return -1;
      }
   }
   
   if(derivedT::c_stdCamera_synchro)
   {
      derived().createStandardIndiToggleSw( m_indiP_synchro, "synchro", "Synchronization", "Synchronization");  
      if( derived().registerIndiPropertyNew( m_indiP_synchro, st_newCallBack_stdCamera) < 0)
      {
         #ifndef STDCAMERA_TEST_NOLOG
         derivedT::template log<software_error>({__FILE__,__LINE__});
         #endif
         return -1;
      }
   }

   if(derivedT::c_stdCamera_usesModes)
   {
      std::vector<std::string> modeNames;
      for(auto it = m_cameraModes.begin(); it!=m_cameraModes.end(); ++it)
      {
         modeNames.push_back(it->first);
      }
      
      if(derived().createStandardIndiSelectionSw( m_indiP_mode, "mode", modeNames) < 0)
      {
         derivedT::template log<software_critical>({__FILE__, __LINE__});
         return -1;
      }
      if( derived().registerIndiPropertyNew( m_indiP_mode, st_newCallBack_stdCamera) < 0)
      {
         #ifndef STDCAMERA_TEST_NOLOG
         derivedT::template log<software_error>({__FILE__,__LINE__});
         #endif
         return -1;
      }
   }
   
   derived().createStandardIndiRequestSw( m_indiP_reconfig, "reconfigure");
   if( derived().registerIndiPropertyNew( m_indiP_reconfig, st_newCallBack_stdCamera) < 0)
   {
      #ifndef STDCAMERA_TEST_NOLOG
      derivedT::template log<software_error>({__FILE__,__LINE__});
      #endif
      return -1;
   }
   
   if(derivedT::c_stdCamera_usesROI)
   {
      //The min/max/step values should be set in derivedT before this is called.
      derived().createStandardIndiNumber( m_indiP_roi_x, "roi_region_x", m_minROIx, m_maxROIx, m_stepROIx, "%0.1f");
      if( derived().registerIndiPropertyNew( m_indiP_roi_x, st_newCallBack_stdCamera) < 0)
      {
         #ifndef STDCAMERA_TEST_NOLOG
         derivedT::template log<software_error>({__FILE__,__LINE__});
         #endif
         return -1;
      }
      
      derived().createStandardIndiNumber( m_indiP_roi_y, "roi_region_y", m_minROIy, m_maxROIy, m_stepROIy, "%0.1f");
      if( derived().registerIndiPropertyNew( m_indiP_roi_y, st_newCallBack_stdCamera) < 0)
      {
         #ifndef STDCAMERA_TEST_NOLOG
         derivedT::template log<software_error>({__FILE__,__LINE__});
         #endif
         return -1;
      }
      
      derived().createStandardIndiNumber( m_indiP_roi_w, "roi_region_w", m_minROIWidth, m_maxROIWidth, m_stepROIWidth, "%d");
      if( derived().registerIndiPropertyNew( m_indiP_roi_w, st_newCallBack_stdCamera) < 0)
      {
         #ifndef STDCAMERA_TEST_NOLOG
         derivedT::template log<software_error>({__FILE__,__LINE__});
         #endif
         return -1;
      }
      
      derived().createStandardIndiNumber( m_indiP_roi_h, "roi_region_h", m_minROIHeight, m_maxROIHeight, m_stepROIHeight, "%d");
      if( derived().registerIndiPropertyNew( m_indiP_roi_h, st_newCallBack_stdCamera) < 0)
      {
         #ifndef STDCAMERA_TEST_NOLOG
         derivedT::template log<software_error>({__FILE__,__LINE__});
         #endif
         return -1;
      }
      
      derived().createStandardIndiNumber( m_indiP_roi_bin_x, "roi_region_bin_x", m_minROIBinning_x, m_maxROIBinning_x, m_stepROIBinning_x, "%f");
      if( derived().registerIndiPropertyNew( m_indiP_roi_bin_x, st_newCallBack_stdCamera) < 0)
      {
         #ifndef STDCAMERA_TEST_NOLOG
         derivedT::template log<software_error>({__FILE__,__LINE__});
         #endif
         return -1;
      }
      
      derived().createStandardIndiNumber( m_indiP_roi_bin_y, "roi_region_bin_y", m_minROIBinning_y, m_maxROIBinning_y, m_stepROIBinning_y, "%f");
      if( derived().registerIndiPropertyNew( m_indiP_roi_bin_y, st_newCallBack_stdCamera) < 0)
      {
         #ifndef STDCAMERA_TEST_NOLOG
         derivedT::template log<software_error>({__FILE__,__LINE__});
         #endif
         return -1;
      }
   
      derived().createROIndiNumber( m_indiP_fullROI, "roi_full_region");
      m_indiP_fullROI.add(pcf::IndiElement("x"));
      m_indiP_fullROI["x"] = 0;
      m_indiP_fullROI.add(pcf::IndiElement("y"));
      m_indiP_fullROI["y"] = 0;
      m_indiP_fullROI.add(pcf::IndiElement("w"));
      m_indiP_fullROI["w"] = 0;
      m_indiP_fullROI.add(pcf::IndiElement("h"));
      m_indiP_fullROI["h"] = 0;
      if( derived().registerIndiPropertyReadOnly( m_indiP_fullROI ) < 0)
      {
         #ifndef STDCAMERA_TEST_NOLOG
         derivedT::template log<software_error>({__FILE__,__LINE__});
         #endif
         return -1;
      }
      
      derived().createStandardIndiRequestSw( m_indiP_roi_check, "roi_region_check");
      if( derived().registerIndiPropertyNew( m_indiP_roi_check, st_newCallBack_stdCamera) < 0)
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
      
      derived().createStandardIndiRequestSw( m_indiP_roi_full, "roi_set_full");
      if( derived().registerIndiPropertyNew( m_indiP_roi_full, st_newCallBack_stdCamera) < 0)
      {
         #ifndef STDCAMERA_TEST_NOLOG
         derivedT::template log<software_error>({__FILE__,__LINE__});
         #endif
         return -1;
      }
   
      derived().createStandardIndiRequestSw( m_indiP_roi_fullbin, "roi_set_full_bin");
      if( derived().registerIndiPropertyNew( m_indiP_roi_fullbin, st_newCallBack_stdCamera) < 0)
      {
         #ifndef STDCAMERA_TEST_NOLOG
         derivedT::template log<software_error>({__FILE__,__LINE__});
         #endif
         return -1;
      }

      derived().createStandardIndiRequestSw( m_indiP_roi_loadlast, "roi_load_last");
      if( derived().registerIndiPropertyNew( m_indiP_roi_loadlast, st_newCallBack_stdCamera) < 0)
      {
         #ifndef STDCAMERA_TEST_NOLOG
         derivedT::template log<software_error>({__FILE__,__LINE__});
         #endif
         return -1;
      }

      derived().createStandardIndiRequestSw( m_indiP_roi_last, "roi_set_last");
      if( derived().registerIndiPropertyNew( m_indiP_roi_last, st_newCallBack_stdCamera) < 0)
      {
         #ifndef STDCAMERA_TEST_NOLOG
         derivedT::template log<software_error>({__FILE__,__LINE__});
         #endif
         return -1;
      }
      
      derived().createStandardIndiRequestSw( m_indiP_roi_default, "roi_set_default");
      if( derived().registerIndiPropertyNew( m_indiP_roi_default, st_newCallBack_stdCamera) < 0)
      {
         #ifndef STDCAMERA_TEST_NOLOG
         derivedT::template log<software_error>({__FILE__,__LINE__});
         #endif
         return -1;
      }
   }

   if(derivedT::c_stdCamera_cropMode)
   {
      derived().createStandardIndiToggleSw( m_indiP_cropMode, "roi_crop_mode", "Crop Mode", "Crop Mode");  
      if( derived().registerIndiPropertyNew( m_indiP_cropMode, st_newCallBack_stdCamera) < 0)
      {
         #ifndef STDCAMERA_TEST_NOLOG
         derivedT::template log<software_error>({__FILE__,__LINE__});
         #endif
         return -1;
      }
   }
   
   //Set up INDI for shutter
   if(derivedT::c_stdCamera_hasShutter)
   {
      derived().createROIndiText( m_indiP_shutterStatus, "shutter_status", "status", "Shutter Status", "Shutter", "Status");
      m_indiP_shutterStatus["status"] = m_shutterStatus;
      if( derived().registerIndiPropertyReadOnly( m_indiP_shutterStatus ) < 0)
      {
         #ifndef STDCAMERA_TEST_NOLOG
         derivedT::template log<software_error>({__FILE__,__LINE__});
         #endif
         return -1;
      }
      
      derived().createStandardIndiToggleSw( m_indiP_shutter, "shutter", "Shutter", "Shutter");  
      if( derived().registerIndiPropertyNew( m_indiP_shutter, st_newCallBack_stdCamera) < 0)
      {
         #ifndef STDCAMERA_TEST_NOLOG
         derivedT::template log<software_error>({__FILE__,__LINE__});
         #endif
         return -1;
      }
      
   }
   
   if(derivedT::c_stdCamera_usesStateString)
   {
      derived().createROIndiText( m_indiP_stateString, "state_string", "current", "State String", "State", "String");
      m_indiP_stateString.add(pcf::IndiElement("valid"));
      m_indiP_stateString["valid"] = "no";
      if( derived().registerIndiPropertyReadOnly( m_indiP_stateString ) < 0)
      {
         #ifndef STDCAMERA_TEST_NOLOG
         derivedT::template log<software_error>({__FILE__,__LINE__});
         #endif
         return -1;
      }
   }

   return 0;
}//int stdCamera<derivedT>::appStartup()

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
         
         if(derivedT::c_stdCamera_tempControl)
         {
            //then set startupTemp if configured
            if(m_startupTemp > -999) m_ccdTempSetpt = m_startupTemp;
            derived().updateIfChanged(m_indiP_temp, "target", m_ccdTempSetpt, INDI_IDLE);
         }
         
         if(derivedT::c_stdCamera_usesROI)
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
         
         if(derivedT::c_stdCamera_hasShutter)
         {
            if(m_shutterStatus == "OPERATING")
            {
               derived().updateIfChanged(m_indiP_shutterStatus, "status", m_shutterStatus, INDI_BUSY);
            }
            if(m_shutterStatus == "POWERON" || m_shutterStatus == "READY")
            {
               derived().updateIfChanged(m_indiP_shutterStatus, "status", m_shutterStatus, INDI_OK);
            }
            else
            {
               derived().updateIfChanged(m_indiP_shutterStatus, "status", m_shutterStatus, INDI_IDLE);
            }
            
            if(m_shutterState == 1)
            {
               derived().updateSwitchIfChanged(m_indiP_shutter, "toggle", pcf::IndiElement::On, INDI_OK);
            }
            else
            {
               derived().updateSwitchIfChanged(m_indiP_shutter, "toggle", pcf::IndiElement::Off, INDI_IDLE);
            }
         }
            
         return 0;
      }
      else
      {
         return 0;
      }
   }
   else if( derived().state() == stateCodes::READY || derived().state() == stateCodes::OPERATING )
   {
      if(derivedT::c_stdCamera_usesROI)
      {
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
   }
   
   return 0;

}

template<class derivedT>
int stdCamera<derivedT>::onPowerOff()
{
   if( !derived().m_indiDriver ) return 0;
   
   if(derivedT::c_stdCamera_usesModes)
   {
      for(auto it = m_cameraModes.begin();it!=m_cameraModes.end();++it)
      {
         derived().updateSwitchIfChanged(m_indiP_mode, it->first, pcf::IndiElement::Off, INDI_IDLE);
      }
   }
   
   if(derivedT::c_stdCamera_usesROI)
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
   
   //Shutters can be independent pieces of hardware . . .
   if(derivedT::c_stdCamera_hasShutter)
   {
      if(m_shutterStatus == "OPERATING")
      {
         derived().updateIfChanged(m_indiP_shutterStatus, "status", m_shutterStatus, INDI_BUSY);
      }
      if(m_shutterStatus == "POWERON" || m_shutterStatus == "READY")
      {
         derived().updateIfChanged(m_indiP_shutterStatus, "status", m_shutterStatus, INDI_OK);
      }
      else
      {
         derived().updateIfChanged(m_indiP_shutterStatus, "status", m_shutterStatus, INDI_IDLE);
      }
          
      if(m_shutterState == 0)
      {
         derived().updateSwitchIfChanged(m_indiP_shutter, "toggle", pcf::IndiElement::On, INDI_OK);
      }
      else
      {
         derived().updateSwitchIfChanged(m_indiP_shutter, "toggle", pcf::IndiElement::Off, INDI_IDLE);
      }
   }
   
   return 0;
}

template<class derivedT>
int stdCamera<derivedT>::whilePowerOff()
{
   //Shutters can be independent pieces of hardware . . .
   if(derivedT::c_stdCamera_hasShutter)
   {
      if(m_shutterStatus == "OPERATING")
      {
         derived().updateIfChanged(m_indiP_shutterStatus, "status", m_shutterStatus, INDI_BUSY);
      }
      if(m_shutterStatus == "POWERON" || m_shutterStatus == "READY")
      {
         derived().updateIfChanged(m_indiP_shutterStatus, "status", m_shutterStatus, INDI_OK);
      }
      else
      {
         derived().updateIfChanged(m_indiP_shutterStatus, "status", m_shutterStatus, INDI_IDLE);
      }
      
      if(m_shutterState == 0)
      {
         derived().updateSwitchIfChanged(m_indiP_shutter, "toggle", pcf::IndiElement::On, INDI_OK);
      }
      else
      {
         derived().updateSwitchIfChanged(m_indiP_shutter, "toggle", pcf::IndiElement::Off, INDI_IDLE);
      }
   }
   
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
   derivedT * _app = static_cast<derivedT *>(app);
   return _app->newCallBack_stdCamera(ipRecv);
}

template<class derivedT>
int stdCamera<derivedT>::newCallBack_stdCamera( const pcf::IndiProperty &ipRecv )
{
   
   if(ipRecv.getDevice() != derived().configName())
   {
       #ifndef XWCTEST_INDI_CALLBACK_VALIDATION
       derivedT::template log<software_error>({__FILE__, __LINE__, "unknown INDI property"});
       #endif

       return -1;
   }

   std::string name = ipRecv.getName();

   if(                                           name == "reconfigure") return newCallBack_reconfigure(ipRecv);
   else if(derivedT::c_stdCamera_temp &&         name == "temp_ccd") return newCallBack_temp(ipRecv);
   else if(derivedT::c_stdCamera_tempControl &&  name == "temp_ccd") return newCallBack_temp(ipRecv);
   else if(derivedT::c_stdCamera_tempControl &&  name == "temp_controller") return newCallBack_temp_controller(ipRecv);
   else if(derivedT::c_stdCamera_readoutSpeed && name == "readout_speed") return newCallBack_readoutSpeed(ipRecv);
   else if(derivedT::c_stdCamera_vShiftSpeed &&  name == "vshift_speed") return newCallBack_vShiftSpeed(ipRecv);
   else if(derivedT::c_stdCamera_emGain &&       name == "emgain") return newCallBack_emgain(ipRecv);
   else if(derivedT::c_stdCamera_exptimeCtrl &&  name == "exptime") return newCallBack_exptime(ipRecv);
   else if(derivedT::c_stdCamera_fpsCtrl &&      name == "fps") return newCallBack_fps(ipRecv);
   else if(derivedT::c_stdCamera_synchro &&      name == "synchro") return newCallBack_synchro(ipRecv);
   else if(derivedT::c_stdCamera_usesModes &&    name == "mode") return newCallBack_mode(ipRecv);
   else if(derivedT::c_stdCamera_cropMode &&     name == "roi_crop_mode") return newCallBack_cropMode(ipRecv);
   else if(derivedT::c_stdCamera_usesROI &&      name == "roi_region_x") return newCallBack_roi_x(ipRecv);
   else if(derivedT::c_stdCamera_usesROI &&      name == "roi_region_y") return newCallBack_roi_y(ipRecv);
   else if(derivedT::c_stdCamera_usesROI &&      name == "roi_region_w") return newCallBack_roi_w(ipRecv);
   else if(derivedT::c_stdCamera_usesROI &&      name == "roi_region_h") return newCallBack_roi_h(ipRecv);
   else if(derivedT::c_stdCamera_usesROI &&      name == "roi_region_bin_x") return newCallBack_roi_bin_x(ipRecv);
   else if(derivedT::c_stdCamera_usesROI &&      name == "roi_region_bin_y") return newCallBack_roi_bin_y(ipRecv);
   else if(derivedT::c_stdCamera_usesROI &&      name == "roi_region_check") return newCallBack_roi_check(ipRecv);
   else if(derivedT::c_stdCamera_usesROI &&      name == "roi_set") return newCallBack_roi_set(ipRecv);
   else if(derivedT::c_stdCamera_usesROI &&      name == "roi_set_full") return newCallBack_roi_full(ipRecv);
   else if(derivedT::c_stdCamera_usesROI &&      name == "roi_set_full_bin") return newCallBack_roi_fullbin(ipRecv);
   else if(derivedT::c_stdCamera_usesROI &&      name == "roi_load_last") return newCallBack_roi_loadlast(ipRecv);
   else if(derivedT::c_stdCamera_usesROI &&      name == "roi_set_last") return newCallBack_roi_last(ipRecv);
   else if(derivedT::c_stdCamera_usesROI &&      name == "roi_set_default") return newCallBack_roi_default(ipRecv);
   else if(derivedT::c_stdCamera_hasShutter &&   name == "shutter") return newCallBack_shutter(ipRecv);

   #ifndef XWCTEST_INDI_CALLBACK_VALIDATION
   derivedT::template log<software_error>({__FILE__,__LINE__, "unknown INDI property"});
   #endif

   return -1;
}

template<class derivedT>
int stdCamera<derivedT>::setTempSetPt( const mx::meta::trueFalseT<true> & t)
{
   static_cast<void>(t);
   return derived().setTempSetPt();
}

template<class derivedT>
int stdCamera<derivedT>::setTempSetPt( const mx::meta::trueFalseT<false> & f)
{
   static_cast<void>(f);
   return 0;
}

template<class derivedT>
int stdCamera<derivedT>::newCallBack_temp( const pcf::IndiProperty &ipRecv )
{
   if(derivedT::c_stdCamera_tempControl)
   {
       #ifdef XWCTEST_INDI_CALLBACK_VALIDATION
          return 0;
       #endif

       float target;

       std::unique_lock<std::mutex> lock(derived().m_indiMutex);

       if (derived().indiTargetUpdate(m_indiP_temp, target, ipRecv, true) < 0)
       {
           derivedT::template log<software_error>({__FILE__, __LINE__});
           return -1;
       }

       m_ccdTempSetpt = target;

       mx::meta::trueFalseT<derivedT::c_stdCamera_tempControl> tf;
       return setTempSetPt(tf);
   }
   else
   {
      return 0;
   }
}
   
template<class derivedT>
int stdCamera<derivedT>::setTempControl( const mx::meta::trueFalseT<true> & t)
{
   static_cast<void>(t);
   return derived().setTempControl();
}

template<class derivedT>
int stdCamera<derivedT>::setTempControl( const mx::meta::trueFalseT<false> & f)
{
   static_cast<void>(f);
   return 0;
}

template<class derivedT>
int stdCamera<derivedT>::newCallBack_temp_controller( const pcf::IndiProperty &ipRecv)
{
   if(derivedT::c_stdCamera_tempControl)
   {
      #ifdef XWCTEST_INDI_CALLBACK_VALIDATION
          return 0;
      #endif
            
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
      
      mx::meta::trueFalseT<derivedT::c_stdCamera_emGain> tf;
      return setTempControl(tf);
   }
   else
   {
      return 0;
   }
}

template<class derivedT>
int stdCamera<derivedT>::setReadoutSpeed( const mx::meta::trueFalseT<true> & t)
{
   static_cast<void>(t);
   return derived().setReadoutSpeed();
}

template<class derivedT>
int stdCamera<derivedT>::setReadoutSpeed( const mx::meta::trueFalseT<false> & f)
{
   static_cast<void>(f);
   return 0;
}
   
template<class derivedT>
int stdCamera<derivedT>::newCallBack_readoutSpeed( const pcf::IndiProperty &ipRecv)
{
   if(derivedT::c_stdCamera_readoutSpeed)
   {
      #ifdef XWCTEST_INDI_CALLBACK_VALIDATION
          return 0;
      #endif

      std::unique_lock<std::mutex> lock(derived().m_indiMutex);

      std::string newspeed;
   
      for(size_t i=0; i< m_readoutSpeedNames.size(); ++i) 
      {
         if(!ipRecv.find(m_readoutSpeedNames[i])) continue;
         
         if(ipRecv[m_readoutSpeedNames[i]].getSwitchState() == pcf::IndiElement::On)
         {
            if(newspeed != "")
            {
               derivedT::template log<text_log>("More than one readout speed selected", logPrio::LOG_ERROR);
               return -1;
            }
         
            newspeed = m_readoutSpeedNames[i];
         }
      }
   
      if(newspeed == "")
      {
         //We do a reset
         m_readoutSpeedNameSet = m_readoutSpeedName;
      }
      else
      {
         m_readoutSpeedNameSet = newspeed;
      }
      
      mx::meta::trueFalseT<derivedT::c_stdCamera_readoutSpeed> tf;
      return setReadoutSpeed(tf);
   }
   
   return 0;
}

template<class derivedT>
int stdCamera<derivedT>::setVShiftSpeed( const mx::meta::trueFalseT<true> & t)
{
   static_cast<void>(t);
   return derived().setVShiftSpeed();
}

template<class derivedT>
int stdCamera<derivedT>::setVShiftSpeed( const mx::meta::trueFalseT<false> & f)
{
   static_cast<void>(f);
   return 0;
}
   
template<class derivedT>
int stdCamera<derivedT>::newCallBack_vShiftSpeed( const pcf::IndiProperty &ipRecv)
{
   if(derivedT::c_stdCamera_vShiftSpeed)
   {
      #ifdef XWCTEST_INDI_CALLBACK_VALIDATION
          return 0;
      #endif

      std::unique_lock<std::mutex> lock(derived().m_indiMutex);

      std::string newspeed;
   
      for(size_t i=0; i< m_vShiftSpeedNames.size(); ++i) 
      {
         if(!ipRecv.find(m_vShiftSpeedNames[i])) continue;
         
         if(ipRecv[m_vShiftSpeedNames[i]].getSwitchState() == pcf::IndiElement::On)
         {
            if(newspeed != "")
            {
               derivedT::template log<text_log>("More than one vShift speed selected", logPrio::LOG_ERROR);
               return -1;
            }
         
            newspeed = m_vShiftSpeedNames[i];
         }
      }
   
      if(newspeed == "")
      {
         //We do a reset
         m_vShiftSpeedNameSet = m_vShiftSpeedName;
      }
      else
      {
         m_vShiftSpeedNameSet = newspeed;
      }
      
      mx::meta::trueFalseT<derivedT::c_stdCamera_vShiftSpeed> tf;
      return setVShiftSpeed(tf);
   }
   
   return 0;
}

template<class derivedT>
int stdCamera<derivedT>::setEMGain( const mx::meta::trueFalseT<true> & t)
{
   static_cast<void>(t);
   return derived().setEMGain();
}

template<class derivedT>
int stdCamera<derivedT>::setEMGain( const mx::meta::trueFalseT<false> & f)
{
   static_cast<void>(f);
   return 0;
}
   
template<class derivedT>
int stdCamera<derivedT>::newCallBack_emgain( const pcf::IndiProperty &ipRecv)
{
   if(derivedT::c_stdCamera_emGain)
   {
      #ifdef XWCTEST_INDI_CALLBACK_VALIDATION
          return 0;
      #endif

      float target;

      std::unique_lock<std::mutex> lock(derived().m_indiMutex);

      if( derived().indiTargetUpdate( m_indiP_emGain, target, ipRecv, true) < 0)
      {
         derivedT::template log<software_error>({__FILE__,__LINE__});
         return -1;
      }
   
      m_emGainSet = target;
   
      mx::meta::trueFalseT<derivedT::c_stdCamera_emGain> tf;
      return setEMGain(tf);
   }
   
   return 0;
}

template<class derivedT>
int stdCamera<derivedT>::setExpTime( const mx::meta::trueFalseT<true> & t)
{
   static_cast<void>(t);
   return derived().setExpTime();
}

template<class derivedT>
int stdCamera<derivedT>::setExpTime( const mx::meta::trueFalseT<false> & f)
{
   static_cast<void>(f);
   return 0;
}

template<class derivedT>
int stdCamera<derivedT>::newCallBack_exptime( const pcf::IndiProperty &ipRecv)
{
   if(derivedT::c_stdCamera_exptimeCtrl)
   {
      #ifdef XWCTEST_INDI_CALLBACK_VALIDATION
          return 0;
      #endif

      float target;

      std::unique_lock<std::mutex> lock(derived().m_indiMutex);

      if( derived().indiTargetUpdate( m_indiP_exptime, target, ipRecv, true) < 0)
      {
         derivedT::template log<software_error>({__FILE__,__LINE__});
         return -1;
      }
   
      m_expTimeSet = target;
   
      mx::meta::trueFalseT<derivedT::c_stdCamera_exptimeCtrl> tf;
      return setExpTime(tf);
   }
   
   return 0;
}

template<class derivedT>
int stdCamera<derivedT>::setFPS( const mx::meta::trueFalseT<true> & t)
{
   static_cast<void>(t);
   return derived().setFPS();
}

template<class derivedT>
int stdCamera<derivedT>::setFPS( const mx::meta::trueFalseT<false> & f)
{
   static_cast<void>(f);
   return 0;
}

template<class derivedT>
int stdCamera<derivedT>::newCallBack_fps( const pcf::IndiProperty &ipRecv)
{
   if(derivedT::c_stdCamera_fpsCtrl)
   {
      #ifdef XWCTEST_INDI_CALLBACK_VALIDATION
          return 0;
      #endif

      float target;

      std::unique_lock<std::mutex> lock(derived().m_indiMutex);

      if( derived().indiTargetUpdate( m_indiP_fps, target, ipRecv, true) < 0)
      {
         derivedT::template log<software_error>({__FILE__,__LINE__});
         return -1;
      }
   
      m_fpsSet = target;
   
      mx::meta::trueFalseT<derivedT::c_stdCamera_fpsCtrl> tf;
      return setFPS(tf);
   }
   
   return 0;
}

template<class derivedT>
int stdCamera<derivedT>::setSynchro( const mx::meta::trueFalseT<true> & t)
{
   static_cast<void>(t);
   return derived().setSynchro();
}

template<class derivedT>
int stdCamera<derivedT>::setSynchro( const mx::meta::trueFalseT<false> & f)
{
   static_cast<void>(f);
   return 0;
}

template<class derivedT>
int stdCamera<derivedT>::newCallBack_synchro( const pcf::IndiProperty &ipRecv)
{
   if(derivedT::c_stdCamera_synchro)
   {
      #ifdef XWCTEST_INDI_CALLBACK_VALIDATION
          return 0;
      #endif
   
      if(!ipRecv.find("toggle")) return 0;
   
      if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::Off )
      {
         m_synchroSet = false;
      }
   
      if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On )
      {
         m_synchroSet = true;
      }
   
      std::unique_lock<std::mutex> lock(derived().m_indiMutex);

      mx::meta::trueFalseT<derivedT::c_stdCamera_synchro> tf;
      return setSynchro(tf);
   }
   
   return 0;
}

template<class derivedT>
int stdCamera<derivedT>::newCallBack_mode( const pcf::IndiProperty &ipRecv )
{
   if(derivedT::c_stdCamera_usesModes)
   {
      #ifdef XWCTEST_INDI_CALLBACK_VALIDATION
          return 0;
      #endif

      std::unique_lock<std::mutex> lock(derived().m_indiMutex);
      
      if(ipRecv.getName() != m_indiP_mode.getName())
      {
         derivedT::template log<software_error>({__FILE__, __LINE__, "invalid indi property received"});
         return -1;
      }
      
      //look for selected mode switch which matches a known mode.  Make sure only one is selected.
      std::string newName = "";
      for(auto it=m_cameraModes.begin(); it != m_cameraModes.end(); ++it) 
      {
         if(!ipRecv.find(it->first)) continue;
         
         if(ipRecv[it->first].getSwitchState() == pcf::IndiElement::On)
         {
            if(newName != "")
            {
               derivedT::template log<text_log>("More than one camera mode selected", logPrio::LOG_ERROR);
               return -1;
            }
            
            newName = it->first;
         }
      }
      
      if(newName == "")
      {
         return 0; 
      }
      
      //Now signal the f.g. thread to reconfigure
      m_nextMode = newName;
      derived().m_reconfig = true;
      
      return 0;
   }
   else
   {
      return 0;
   }
}
   
template<class derivedT>
int stdCamera<derivedT>::newCallBack_reconfigure( const pcf::IndiProperty &ipRecv )
{
   #ifdef XWCTEST_INDI_CALLBACK_VALIDATION
      return 0;
   #endif
   
   if(!ipRecv.find("request")) return 0;
   
   if( ipRecv["request"].getSwitchState() == pcf::IndiElement::On)
   {
      std::unique_lock<std::mutex> lock(derived().m_indiMutex);
      
      indi::updateSwitchIfChanged(m_indiP_reconfig, "request", pcf::IndiElement::Off, derived().m_indiDriver, INDI_IDLE);
      
      m_nextMode = m_modeName;
      derived().m_reconfig = true;
      return 0;
   }
   
   return 0;  
}

template<class derivedT>
int stdCamera<derivedT>::setCropMode( const mx::meta::trueFalseT<true> & t)
{
   static_cast<void>(t);
   return derived().setCropMode();
}

template<class derivedT>
int stdCamera<derivedT>::setCropMode( const mx::meta::trueFalseT<false> & f)
{
   static_cast<void>(f);
   return 0;
}
   
template<class derivedT>
int stdCamera<derivedT>::newCallBack_cropMode( const pcf::IndiProperty &ipRecv)
{
   if(derivedT::c_stdCamera_cropMode)
   {
      #ifdef XWCTEST_INDI_CALLBACK_VALIDATION
          return 0;
      #endif

      if(!ipRecv.find("toggle")) return 0;
   
      if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::Off )
      {
         m_cropModeSet = false;
      }
   
      if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On )
      {
         m_cropModeSet = true;
      }
   
      std::unique_lock<std::mutex> lock(derived().m_indiMutex);

      mx::meta::trueFalseT<derivedT::c_stdCamera_cropMode> tf;
      return setCropMode(tf);
   }
   
   return 0;
}

///\todo why don't these check if usesROI is true?
template<class derivedT>
int stdCamera<derivedT>::newCallBack_roi_x( const pcf::IndiProperty &ipRecv )
{
   #ifdef XWCTEST_INDI_CALLBACK_VALIDATION
        return 0;
   #endif

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
   #ifdef XWCTEST_INDI_CALLBACK_VALIDATION
          return 0;
   #endif

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
   #ifdef XWCTEST_INDI_CALLBACK_VALIDATION
          return 0;
   #endif

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
   #ifdef XWCTEST_INDI_CALLBACK_VALIDATION
          return 0;
   #endif

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
   #ifdef XWCTEST_INDI_CALLBACK_VALIDATION
      return 0;
   #endif

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
   #ifdef XWCTEST_INDI_CALLBACK_VALIDATION
          return 0;
   #endif
 
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
int stdCamera<derivedT>::checkNextROI( const mx::meta::trueFalseT<true> & t)
{
   static_cast<void>(t);
   return derived().checkNextROI();
}

template<class derivedT>
int stdCamera<derivedT>::checkNextROI( const mx::meta::trueFalseT<false> & f)
{
   static_cast<void>(f);
   return 0;
}

template<class derivedT>
int stdCamera<derivedT>::newCallBack_roi_check( const pcf::IndiProperty &ipRecv )
{
   if(derivedT::c_stdCamera_usesROI)
   {
      #ifdef XWCTEST_INDI_CALLBACK_VALIDATION
          return 0;
      #endif
      
      if(!ipRecv.find("request")) return 0;
      
      if( ipRecv["request"].getSwitchState() == pcf::IndiElement::On)
      {
         std::unique_lock<std::mutex> lock(derived().m_indiMutex);
         
         indi::updateSwitchIfChanged(m_indiP_roi_check, "request", pcf::IndiElement::Off, derived().m_indiDriver, INDI_IDLE);
         
         mx::meta::trueFalseT<derivedT::c_stdCamera_usesROI> tf;
         return checkNextROI(tf);
      }
      
      return 0;  
   }
   
   return 0;
}

template<class derivedT>
int stdCamera<derivedT>::setNextROI( const mx::meta::trueFalseT<true> & t)
{
   static_cast<void>(t);
   return derived().setNextROI();
}

template<class derivedT>
int stdCamera<derivedT>::setNextROI( const mx::meta::trueFalseT<false> & f)
{
   static_cast<void>(f);
   return 0;
}

template<class derivedT>
int stdCamera<derivedT>::newCallBack_roi_set( const pcf::IndiProperty &ipRecv )
{
   if(derivedT::c_stdCamera_usesROI)
   {
      #ifdef XWCTEST_INDI_CALLBACK_VALIDATION
          return 0;
      #endif

      if(!ipRecv.find("request")) return 0;
      
      if( ipRecv["request"].getSwitchState() == pcf::IndiElement::On)
      {
         std::unique_lock<std::mutex> lock(derived().m_indiMutex);
         
         indi::updateSwitchIfChanged(m_indiP_roi_set, "request", pcf::IndiElement::Off, derived().m_indiDriver, INDI_IDLE);
         
         m_lastROI = m_currentROI;
         
         mx::meta::trueFalseT<derivedT::c_stdCamera_usesROI> tf;
         return setNextROI(tf);
      }
      
      return 0;  
   }
   
   return 0;
}

template<class derivedT>
int stdCamera<derivedT>::newCallBack_roi_full( const pcf::IndiProperty &ipRecv )
{
   if(derivedT::c_stdCamera_usesROI)
   {
      #ifdef XWCTEST_INDI_CALLBACK_VALIDATION
          return 0;
      #endif

      if(!ipRecv.find("request")) return 0;
      
      if( ipRecv["request"].getSwitchState() == pcf::IndiElement::On)
      {
         std::unique_lock<std::mutex> lock(derived().m_indiMutex);
         
         indi::updateSwitchIfChanged(m_indiP_roi_full, "request", pcf::IndiElement::Off, derived().m_indiDriver, INDI_IDLE);
      
         m_nextROI.x = m_full_x;
         m_nextROI.y = m_full_y;
         m_nextROI.w = m_full_w;
         m_nextROI.h = m_full_h;
         m_nextROI.bin_x = m_full_bin_x;
         m_nextROI.bin_y = m_full_bin_y;
         m_lastROI = m_currentROI;
         mx::meta::trueFalseT<derivedT::c_stdCamera_usesROI> tf;
         return setNextROI(tf);
      }
      
      return 0;  
   }
   
   return 0;
}

template<class derivedT>
int stdCamera<derivedT>::newCallBack_roi_fullbin( const pcf::IndiProperty &ipRecv )
{
   if(derivedT::c_stdCamera_usesROI)
   {
      #ifdef XWCTEST_INDI_CALLBACK_VALIDATION
          return 0;
      #endif

      if(!ipRecv.find("request")) return 0;
      
      if( ipRecv["request"].getSwitchState() == pcf::IndiElement::On)
      {
         std::unique_lock<std::mutex> lock(derived().m_indiMutex);
         
         indi::updateSwitchIfChanged(m_indiP_roi_fullbin, "request", pcf::IndiElement::Off, derived().m_indiDriver, INDI_IDLE);
      
         bool reset = false;

         if(m_full_currbin_x == 0) //still defaulted 
         {
            derivedT::template log<text_log>("current-binning full ROI not implemented for this camera", logPrio::LOG_WARNING);
            m_full_currbin_x = m_full_x;
            m_full_currbin_y = m_full_y;
            m_full_currbin_w = m_full_w;
            m_full_currbin_h = m_full_h;
            reset = true;
         }

         

         m_nextROI.x = m_full_currbin_x;
         m_nextROI.y = m_full_currbin_y;
         m_nextROI.w = m_full_currbin_w;
         m_nextROI.h = m_full_currbin_h;
         if(reset)
         {
            //Use full binning
            m_nextROI.bin_x = m_full_bin_x;
            m_nextROI.bin_y = m_full_bin_y;

            //restore defaults for next time
            m_full_currbin_x = 0;
            m_full_currbin_y = 0;
            m_full_currbin_w = 0;
            m_full_currbin_h = 0;
         }
         else 
         {
            m_nextROI.bin_x = m_currentROI.bin_x;
            m_nextROI.bin_y = m_currentROI.bin_y;
         }

         m_lastROI = m_currentROI;
         mx::meta::trueFalseT<derivedT::c_stdCamera_usesROI> tf;
         return setNextROI(tf);
      }
      
      return 0;  
   }
   
   return 0;
}

template<class derivedT>
int stdCamera<derivedT>::newCallBack_roi_loadlast( const pcf::IndiProperty &ipRecv )
{
   if(derivedT::c_stdCamera_usesROI)
   {
      #ifdef XWCTEST_INDI_CALLBACK_VALIDATION
          return 0;
      #endif

      if(!ipRecv.find("request")) return 0;      

      if( ipRecv["request"].getSwitchState() == pcf::IndiElement::On)
      {
         std::unique_lock<std::mutex> lock(derived().m_indiMutex);
         
         indi::updateSwitchIfChanged(m_indiP_roi_loadlast, "request", pcf::IndiElement::Off, derived().m_indiDriver, INDI_IDLE);
         
         m_nextROI = m_lastROI;
         return 0;
      }
      
      return 0;  
   }
   
   return 0;
}

template<class derivedT>
int stdCamera<derivedT>::newCallBack_roi_last( const pcf::IndiProperty &ipRecv )
{
   if(derivedT::c_stdCamera_usesROI)
   {
      #ifdef XWCTEST_INDI_CALLBACK_VALIDATION
          return 0;
      #endif
      if(!ipRecv.find("request")) return 0;      

      if( ipRecv["request"].getSwitchState() == pcf::IndiElement::On)
      {
         std::unique_lock<std::mutex> lock(derived().m_indiMutex);
         
         indi::updateSwitchIfChanged(m_indiP_roi_last, "request", pcf::IndiElement::Off, derived().m_indiDriver, INDI_IDLE);
         
         m_nextROI = m_lastROI;
         m_lastROI = m_currentROI;
         mx::meta::trueFalseT<derivedT::c_stdCamera_usesROI> tf;
         return setNextROI(tf);
      }
      
      return 0;  
   }
   
   return 0;
}

template<class derivedT>
int stdCamera<derivedT>::newCallBack_roi_default( const pcf::IndiProperty &ipRecv )
{
   if(derivedT::c_stdCamera_usesROI)
   {
      #ifdef XWCTEST_INDI_CALLBACK_VALIDATION
          return 0;
      #endif

      if(!ipRecv.find("request")) return 0;
      
      if( ipRecv["request"].getSwitchState() == pcf::IndiElement::On)
      {
         std::unique_lock<std::mutex> lock(derived().m_indiMutex);
         
         indi::updateSwitchIfChanged(m_indiP_roi_default, "request", pcf::IndiElement::Off, derived().m_indiDriver, INDI_IDLE);
         
         m_nextROI.x = m_default_x;
         m_nextROI.y = m_default_y;
         m_nextROI.w = m_default_w;
         m_nextROI.h = m_default_h;
         m_nextROI.bin_x = m_default_bin_x;
         m_nextROI.bin_y = m_default_bin_y;
         m_lastROI = m_currentROI;
         mx::meta::trueFalseT<derivedT::c_stdCamera_usesROI> tf;
         return setNextROI(tf);
      }
      
      return 0;  
   }
   
   return 0;
}

template<class derivedT>
int stdCamera<derivedT>::setShutter( int ss,
                                     const mx::meta::trueFalseT<true> & t
                                   )
{
   static_cast<void>(t);
   return derived().setShutter(ss);
}

template<class derivedT>
int stdCamera<derivedT>::setShutter( int ss,
                                     const mx::meta::trueFalseT<false> & f
                                   )
{
   static_cast<void>(ss);
   static_cast<void>(f);
   return 0;
}

template<class derivedT>
int stdCamera<derivedT>::newCallBack_shutter( const pcf::IndiProperty &ipRecv )
{
   if(derivedT::c_stdCamera_hasShutter)
   {
      #ifdef XWCTEST_INDI_CALLBACK_VALIDATION
          return 0;
      #endif
      
      if(!ipRecv.find("toggle")) return 0;
      
       mx::meta::trueFalseT<derivedT::c_stdCamera_hasShutter> tf;
            
      if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::Off )
      {
         setShutter(1, tf);
      }
      
      if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On )
      {
         setShutter(0, tf);
      }
      
      return 0;  
   }
   return 0;
}

template<class derivedT>
std::string stdCamera<derivedT>::stateString( const mx::meta::trueFalseT<true> & t)
{
   static_cast<void>(t);
   return derived().stateString();
}

template<class derivedT>
std::string stdCamera<derivedT>::stateString( const mx::meta::trueFalseT<false> & f)
{
   static_cast<void>(f);
   return "";
}

template<class derivedT>
bool stdCamera<derivedT>::stateStringValid( const mx::meta::trueFalseT<true> & t)
{
   static_cast<void>(t);
   return derived().stateStringValid();
}

template<class derivedT>
bool stdCamera<derivedT>::stateStringValid( const mx::meta::trueFalseT<false> & f)
{
   static_cast<void>(f);
   return false;
}

template<class derivedT>
int stdCamera<derivedT>::updateINDI()
{
   if( !derived().m_indiDriver ) return 0;
   
   if(derivedT::c_stdCamera_readoutSpeed)
   {
      indi::updateSelectionSwitchIfChanged( m_indiP_readoutSpeed, m_readoutSpeedName, derived().m_indiDriver, INDI_OK);
   }
   
   if(derivedT::c_stdCamera_vShiftSpeed)
   {
      indi::updateSelectionSwitchIfChanged( m_indiP_vShiftSpeed, m_vShiftSpeedName, derived().m_indiDriver, INDI_OK);
   }
   
   if(derivedT::c_stdCamera_emGain)
   {
      derived().updateIfChanged(m_indiP_emGain, "current", m_emGain, INDI_IDLE);
      derived().updateIfChanged(m_indiP_emGain, "target", m_emGainSet, INDI_IDLE);
   }
   
   if(derivedT::c_stdCamera_exptimeCtrl)
   {
      derived().updateIfChanged(m_indiP_exptime, "current", m_expTime, INDI_IDLE);
      derived().updateIfChanged(m_indiP_exptime, "target", m_expTimeSet, INDI_IDLE);
   }
   
   if(derivedT::c_stdCamera_fpsCtrl)
   {
      derived().updateIfChanged(m_indiP_fps, "current", m_fps, INDI_IDLE);
      derived().updateIfChanged(m_indiP_fps, "target", m_fpsSet, INDI_IDLE);
   }
   else if(derivedT::c_stdCamera_fps)
   {
      derived().updateIfChanged(m_indiP_fps, "current", m_fps, INDI_IDLE);
   }
   
   if(derivedT::c_stdCamera_synchro)
   {
      if(m_synchro == false) 
      {
         derived().updateSwitchIfChanged(m_indiP_synchro, "toggle", pcf::IndiElement::Off, INDI_IDLE);
      }
      else
      {
         derived().updateSwitchIfChanged(m_indiP_synchro, "toggle", pcf::IndiElement::On, INDI_OK);
      }
   }

   if(derivedT::c_stdCamera_usesModes)
   {
      auto st = pcf::IndiProperty::Ok;
      if(m_nextMode != "") st = pcf::IndiProperty::Busy;
      
      for(auto it = m_cameraModes.begin();it!=m_cameraModes.end();++it)
      {
         if(it->first == m_modeName) derived().updateSwitchIfChanged(m_indiP_mode, it->first, pcf::IndiElement::On, st);
         else derived().updateSwitchIfChanged(m_indiP_mode, it->first, pcf::IndiElement::Off, st);
      }
            
   }
   
   if(derivedT::c_stdCamera_cropMode)
   {
      if(m_cropMode == false) 
      {
         derived().updateSwitchIfChanged(m_indiP_cropMode, "toggle", pcf::IndiElement::Off, INDI_IDLE);
      }
      else
      {
         derived().updateSwitchIfChanged(m_indiP_cropMode, "toggle", pcf::IndiElement::On, INDI_OK);
      }
   }
   
   if(derivedT::c_stdCamera_usesROI)
   {
      //These can't change after initialization, but might not be discoverable until powered on and connected.
      //so we'll check every time I guess.
      derived().updateIfChanged(m_indiP_fullROI, "x", m_full_x, INDI_IDLE);
      derived().updateIfChanged(m_indiP_fullROI, "y", m_full_y, INDI_IDLE);
      derived().updateIfChanged(m_indiP_fullROI, "w", m_full_w, INDI_IDLE);
      derived().updateIfChanged(m_indiP_fullROI, "h", m_full_h, INDI_IDLE);
   }
   
   if(derivedT::c_stdCamera_tempControl)
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
   else if(derivedT::c_stdCamera_temp)
   {
      derived().updateIfChanged(m_indiP_temp, "current", m_ccdTemp, INDI_IDLE);
   }
   
   
   
   if(derivedT::c_stdCamera_hasShutter)
   {
      if(m_shutterStatus == "OPERATING")
      {
         derived().updateIfChanged(m_indiP_shutterStatus, "status", m_shutterStatus, INDI_BUSY);
      }
      if(m_shutterStatus == "POWERON" || m_shutterStatus == "READY")
      {
         derived().updateIfChanged(m_indiP_shutterStatus, "status", m_shutterStatus, INDI_IDLE);
      }
      else
      {
         derived().updateIfChanged(m_indiP_shutterStatus, "status", m_shutterStatus, INDI_IDLE);
      }
          
      if(m_shutterState == 0) //0 shut, 1 open
      {
         derived().updateSwitchIfChanged(m_indiP_shutter, "toggle", pcf::IndiElement::On, INDI_OK);
      }
      else
      {
         derived().updateSwitchIfChanged(m_indiP_shutter, "toggle", pcf::IndiElement::Off, INDI_IDLE);
      }
   }
   
   if(derivedT::c_stdCamera_usesStateString)
   {
      mx::meta::trueFalseT<derivedT::c_stdCamera_usesStateString> tf;
      derived().updateIfChanged(m_indiP_stateString, "current", stateString(tf), INDI_IDLE);
      if(stateStringValid(tf))
      {
         derived().updateIfChanged(m_indiP_stateString, "valid", "yes", INDI_IDLE);
      }
      else
      {
         derived().updateIfChanged(m_indiP_stateString, "valid", "no", INDI_IDLE);
      }
   }
   return 0;
}

template<class derivedT>
int stdCamera<derivedT>::recordCamera( bool force )
{
   static std::string last_mode;
   static roi last_roi;
   static float last_expTime = -1e30; //ensure first one goes
   static float last_fps = 0;
   static float last_adcSpeed = -1;
   static float last_emGain = -1;
   static float last_ccdTemp = 0;
   static float last_ccdTempSetpt = 0;
   static bool last_tempControlStatus = 0;
   static bool last_tempControlOnTarget = 0;
   static std::string last_tempControlStatusStr;
   static std::string last_shutterStatus;
   static int last_shutterState = false;
   static bool last_synchro = false;
   static float last_vshiftSpeed = -1;
   static bool last_cropMode = false;


   if(force || m_modeName != last_mode ||
               m_currentROI.x != last_roi.x ||
               m_currentROI.y != last_roi.y ||
               m_currentROI.w != last_roi.w ||
               m_currentROI.h != last_roi.h ||
               m_currentROI.bin_x != last_roi.bin_x ||
               m_currentROI.bin_y != last_roi.bin_y ||
               m_expTime != last_expTime ||
               m_fps != last_fps ||
               m_emGain != last_emGain ||
               m_adcSpeed != last_adcSpeed ||
               m_ccdTemp != last_ccdTemp ||
               m_ccdTempSetpt != last_ccdTempSetpt ||
               m_tempControlStatus != last_tempControlStatus ||
               m_tempControlOnTarget != last_tempControlOnTarget ||
               m_tempControlStatusStr != last_tempControlStatusStr ||
               m_shutterStatus != last_shutterStatus ||
               m_shutterState != last_shutterState ||
               m_synchro != last_synchro ||
               m_vshiftSpeed != last_vshiftSpeed ||
               m_cropMode != last_cropMode) 
   {
      derived().template telem<telem_stdcam>({m_modeName, m_currentROI.x, m_currentROI.y, 
                                                    m_currentROI.w, m_currentROI.h, m_currentROI.bin_x, m_currentROI.bin_y,
                                                       m_expTime, m_fps, m_emGain, m_adcSpeed, m_ccdTemp, m_ccdTempSetpt, (uint8_t) m_tempControlStatus, 
                                                             (uint8_t) m_tempControlOnTarget, m_tempControlStatusStr, m_shutterStatus, (int8_t) m_shutterState, 
                                                                   (uint8_t) m_synchro, m_vshiftSpeed, (uint8_t) m_cropMode});
      
      last_mode = m_modeName;
      last_roi = m_currentROI;
      last_expTime = m_expTime;
      last_fps = m_fps;
      last_emGain = m_emGain;
      last_adcSpeed = m_adcSpeed;
      last_ccdTemp = m_ccdTemp;
      last_ccdTempSetpt = m_ccdTempSetpt;
      last_tempControlStatus = m_tempControlStatus;
      last_tempControlOnTarget = m_tempControlOnTarget;
      last_tempControlStatusStr = m_tempControlStatusStr;
      last_shutterStatus = m_shutterStatus;
      last_shutterState = m_shutterState;
      last_synchro = m_synchro;
      last_vshiftSpeed = m_vshiftSpeed;
      last_cropMode = m_cropMode;
   }
   
   
   return 0;
   
}


} //namespace dev
} //namespace app
} //namespace MagAOX

#endif //stdCamera_hpp
