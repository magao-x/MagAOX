/** \file orcaCtrl.hpp
 * \brief The MagAO-X Hamamatsu Orca-Quest2 camera controller.
 *
 * \author Joshua Liberman (jliberman54@gmail.com)
 *
 * \ingroup orcaCtrl_files
 */

#ifndef orcaCtrl_hpp
#define orcaCtrl_hpp

// #include <ImageStruct.h>
#include <ImageStreamIO/ImageStreamIO.h>

#include <dcamapi4.h>
#include <dcamprop.h>

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

/// Enables the BREADCRUMB debugging macro.
/** \todo Remove before deployment; debugging output should go through log<>.
 */
#define DEBUG

#ifdef DEBUG
    /// Print the current file and line to std::cerr (debugging only).
    #define BREADCRUMB std::cerr << __FILE__ << " " << __LINE__ << "\n";
#else
    /// No-op when DEBUG is not defined.
    #define BREADCRUMB
#endif

/// Get the DCAM text description of an error code.
/** Uses dcamdev_getstring() so DCAMERR values don't need a local enum-to-string table.
 *
 * \returns the DCAM error text
 * \returns "DCAM error <code>" if DCAM cannot describe the error
 */
inline std::string dcamErrorString( HDCAM   hdcam, /**< [in] camera handle used to look up the error text */
                                    DCAMERR error  /**< [in] the DCAM error code to describe */
)
{
    char text[256]{};

    DCAMDEV_STRING info{};
    info.size      = sizeof( info );
    info.iString   = static_cast<int32>( error );
    info.text      = text;
    info.textbytes = sizeof( text );

    if( failed( dcamdev_getstring( hdcam, &info ) ) )
    {
        return "DCAM error " + std::to_string( static_cast<int32>( error ) );
    }

    return text;
}

/// Get a DCAM device identification string, such as the vendor, model or camera ID.
/** \returns the requested string
 * \returns an empty string if DCAM cannot provide it
 */
inline std::string dcamDeviceString( HDCAM      hdcam,   /**< [in] camera handle, or a device index cast to HDCAM */
                                     DCAM_IDSTR stringID /**< [in] which string to get, e.g. DCAM_IDSTR_CAMERAID */
)
{
    char text[256]{};

    DCAMDEV_STRING info{};
    info.size      = sizeof( info );
    info.iString   = static_cast<int32>( stringID );
    info.text      = text;
    info.textbytes = sizeof( text );

    if( failed( dcamdev_getstring( hdcam, &info ) ) )
    {
        return {};
    }

    return text;
}

namespace MagAOX
{
namespace app
{

/** \defgroup orcaCtrl Hamamatsu Orca Quest 2 Camera
 * \brief Control of a Hamamatsu Orca Quest 2 Camera.
 *
 * <a href="../handbook/operating/software/apps/orcaCtrl.html">Application Documentation</a>
 *
 * \ingroup apps
 *
 */

/** \defgroup orcaCtrl_files Hamamatsu Orca Quest 2 Camera Files
 * \ingroup orcaCtrl
 */

/// MagAO-X application to control a Hamamatsu ORCA-Quest2 (C15550-22UP) qCMOS camera.
/** Controls the camera through the Hamamatsu DCAM-API over CoaXPress, and publishes frames to an
 * ImageStreamIO stream through dev::frameGrabber. Exposure time, ROI and binning, readout speed,
 * temperature status and the cooling fan are exposed through dev::stdCamera. Camera state is
 * recorded through dev::telemeter as telem_stdcam.
 *
 * Configuration specific to this app (in addition to the stdCamera, frameGrabber and telemeter keys):
 * - `camera.serialNumber`: the camera ID reported by DCAM_IDSTR_CAMERAID (e.g. `PHX2`). Required.
 * - `camera.liquidCooling`: declares a liquid-cooled camera. Optional, default `false`, currently only
 *   logged (see agents/plans/2026-09/orcaCtrl-liquidCooling.md).
 *
 * The camera's cooling method (Air or Water, "Cooler Type") is a persistent camera setting. It is
 * changed with the `dcamcfgc` DCAM Configurator, followed by a camera restart, not by this app.
 *
 * \ingroup orcaCtrl
 *
 * \todo Define or remove setorcaParameterOnline( HDCAM, int32, int32 ) and
 *       setorcaParameterOnline( int32, int32 ), which are declared but not defined.
 * \todo Add unit tests under apps/orcaCtrl/tests (AGENTS.md rule 20).
 */
class orcaCtrl : public MagAOXApp<>,
                 public dev::stdCamera<orcaCtrl>,
                 public dev::frameGrabber<orcaCtrl>,
                 //   public dev::dssShutter<orcaCtrl>,
                 public dev::telemeter<orcaCtrl>
{

    friend class dev::stdCamera<orcaCtrl>;
    friend class dev::frameGrabber<orcaCtrl>;
    // friend class dev::dssShutter<orcaCtrl>;
    friend class dev::telemeter<orcaCtrl>;

    /// The stdCamera base class, used by the STDCAMERA_* macros
    typedef dev::stdCamera<orcaCtrl> stdCameraT;

    /// The frameGrabber base class, used by the FRAMEGRABBER_* macros
    typedef dev::frameGrabber<orcaCtrl> frameGrabberT;

    /// The telemeter base class, used by the TELEMETER_* macros
    typedef dev::telemeter<orcaCtrl> telemeterT;

    /// The MagAOXApp base class
    typedef MagAOXApp<> MagAOXAppT;

  public:
    /** \name app::dev Configurations
     *@{
     */

    /// app::dev config to tell stdCamera to expose temperature controls
    static constexpr bool c_stdCamera_tempControl = true;

    /// app::dev config to tell stdCamera to expose temperature
    static constexpr bool c_stdCamera_temp = true;

    /// app::dev config to tell stdCamera to expose readout speed controls
    static constexpr bool c_stdCamera_readoutSpeed = true;

    /// app::dev config to tell stdCamera not to expose vertical shift speed control
    static constexpr bool c_stdCamera_vShiftSpeed = false;

    /// app::dev config to tell stdCamera to expose fan-speed control
    static constexpr bool c_stdCamera_fanSpeed = true;

    /// app::dev config to tell stdCamera not to expose EM gain controls
    static constexpr bool c_stdCamera_emGain = false;

    /// app::dev config to tell stdCamera to expose exposure time controls
    static constexpr bool c_stdCamera_exptimeCtrl = true;

    /// app::dev config to tell stdCamera not to expose FPS controls
    static constexpr bool c_stdCamera_fpsCtrl = false;

    /// app::dev config to tell stdCamera to expose FPS status
    static constexpr bool c_stdCamera_fps = true;

    /// app::dev config to tell stdCamera not to expose synchro mode controls
    static constexpr bool c_stdCamera_synchro = false;

    /// app::dev config to tell stdCamera not to expose mode controls
    static constexpr bool c_stdCamera_usesModes = false;

    /// app::dev config to tell stdCamera to expose ROI controls
    static constexpr bool c_stdCamera_usesROI = true;

    /// app::dev config to tell stdCamera not to expose crop mode controls
    static constexpr bool c_stdCamera_cropMode = false;

    /// app::dev config to tell stdCamera not to expose shutter controls
    static constexpr bool c_stdCamera_hasShutter = false;

    /// app::dev config to tell stdCamera not to expose focus-state reporting and goto-focus control
    static constexpr bool c_stdCamera_hasFocus = false;

    /// app::dev config to tell stdCamera not to expose the state string property
    static constexpr bool c_stdCamera_usesStateString = false;

    /// app::dev config to tell frameGrabber this camera can be flipped
    static constexpr bool c_frameGrabber_flippable = true;

    ///@}

  protected:
    /** \name Configurable Parameters
     *@{
     */

    /// The camera ID to connect to (`camera.serialNumber`), matched against DCAM_IDSTR_CAMERAID.
    std::string m_serialNumber;

    /// True when the camera is declared liquid cooled (`camera.liquidCooling`).
    /** The cooling method itself is a persistent camera setting ("Cooler Type") changed with the
     * `dcamcfgc` DCAM Configurator.  This flag only records the declared mode; the liquid-cooling
     * control in agents/plans/2026-09/orcaCtrl-liquidCooling.md is not yet implemented.
     */
    bool m_liquidCooling{ false };

    ///@}

    /** \name Acquisition State
     *@{
     */

    /// Pixel bit depth reported by DCAM_IDPROP_BITSPERCHANNEL for the current configuration.
    int m_depth{ 0 };

    /// Size of one frame in bytes, from DCAM_IDPROP_IMAGE_FRAMEBYTES.
    int32 m_frameSize;

    /// Number of frames allocated in the DCAM capture buffer by dcambuf_alloc().
    int32 m_frameCount{ 10 };

    /// Camera timestamp of the previous frame [s], used to detect skipped frames. 0 after a reconfigure.
    double m_camera_timestamp{ 0.0 };

    /// Frame rate the camera reports for the current settings, from DCAM_IDPROP_INTERNALFRAMERATE [Hz].
    double m_FrameRateCalculation;

    /// Sensor readout time for the current settings, from DCAM_IDPROP_TIMING_READOUTTIME [s].
    double m_ReadOutTimeCalculation;

    // std::string m_fxngenName{ "fxngensync" }; ///< Default fxngen device name
    // std::string m_fxngenCh{ "C2" };           ///< Default fxngen channel

    /// Unused; kept from the app this one was derived from.
    std::string m_otherCamName;

    ///@}

    /** \name DCAM State
     *@{
     */

    /// Handle to the open DCAM device, or nullptr when no camera is open.
    HDCAM m_cameraHandle{ nullptr };

    /// DCAM wait handle used to wait for frame-ready and capture-stopped events, or nullptr when closed.
    HDCAMWAIT m_waitHandle{ nullptr };

    /// The most recently locked DCAM frame. Its buffer belongs to DCAM and is only valid until overwritten.
    DCAMBUF_FRAME m_currentFrame{};

    /// Camera vendor string (DCAM_IDSTR_VENDOR), set on connect.
    std::string m_cameraName;

    /// Camera model string (DCAM_IDSTR_MODEL), set on connect.
    std::string m_cameraModel;

    /// True while DCAM capture buffers are allocated (by dcambuf_alloc()) and must be released.
    bool m_dcamBuffersAllocated{ false };

    /// True when the camera exposes a writable DCAM_IDPROP_SENSORCOOLERFAN property.
    bool m_fanControlSupported{ false };

    /// True when the camera exposes readable cooling-fan status.
    bool m_fanStatusSupported{ false };

    /// True while the camera reports the cooling fan is forced on for protection.
    bool m_fanForcedOn{ false };

    /// True when the next successful fan apply should emit a notice even without a state change.
    bool m_fanSpeedLogPending{ false };

    ///@}

  public:
    /** \name MagAOXApp Interface
     *@{
     */

    /// Default c'tor.
    /** Enables power management and sets the readout-speed and fan-speed options and the default
     * full-frame ROI.
     */
    orcaCtrl();

    /// D'tor, declared and defined for noexcept.
    /** Releases any DCAM resources still held, via closeCamera().
     */
    ~orcaCtrl() noexcept;

    /// Setup the configuration system (called by MagAOXApp::setup())
    virtual void setupConfig();

    /// Implementation of loadConfig logic, separated for testing.
    /** This is called by loadConfig().
     *
     * \returns 0 on success
     * \returns -1 on error
     */
    int loadConfigImpl(
        mx::app::appConfigurator &_config /**< [in] an application configuration from which to load values*/ );

    /// load the configuration system results (called by MagAOXApp::setup())
    virtual void loadConfig();

    /// Startup function.
    /** Creates the `readout_time` INDI property, sets the temperature and ROI limits, and starts the
     * stdCamera, frameGrabber and telemeter interfaces.
     *
     * \returns 0 on success
     * \returns -1 on an error requiring shutdown
     */
    virtual int appStartup();

    /// Implementation of the FSM for orcaCtrl.
    /** Connects to the camera when not connected, triggers the initial configuration, and while
     * READY/OPERATING polls acquisition state, temperatures and fan state and updates INDI and telemetry.
     *
     * \returns 0 on no critical error
     * \returns -1 on an error requiring shutdown
     */
    virtual int appLogic();

    /// Implementation of the on-power-off FSM logic.
    /** Releases the DCAM resources and resets the stdCamera and frameGrabber state.
     *
     * \returns 0 always
     */
    virtual int onPowerOff();

    /// Implementation of the while-powered-off FSM.
    /**
     * \returns 0 always
     */
    virtual int whilePowerOff();

    /// Shutdown the app.
    /** Stops the framegrabber thread, releases the DCAM resources, then shuts down the stdCamera and
     * telemeter interfaces.
     *
     * \returns 0 on success
     * \returns -1 on error
     */
    virtual int appShutdown();

    ///@}

  protected:
    /** \name DCAM Interface
     *@{
     */

    /// Get an integer-valued DCAM property.
    /**
     * \returns 0 on success
     * \returns -1 on error (logged unless the camera is powering off)
     */
    int getorcaParameter( int32 &value, /**< [out] the property value, truncated to an integer */
                          int32  parameter /**< [in] the DCAM_IDPROP_* property to read */ );

    /// Get a DCAM property as a double.
    /**
     * \returns 0 on success
     * \returns -1 on error (logged unless the camera is powering off)
     */
    int getorcaParameter( double &value, /**< [out] the property value */
                          int32   parameter /**< [in] the DCAM_IDPROP_* property to read */ );

    /// Set an integer-valued DCAM property on the current camera, with dcamprop_setgetvalue().
    /**
     * \returns 0 on success
     * \returns -1 on error (logged unless the camera is powering off)
     */
    int setorcaParameter( int32 parameter, /**< [in] the DCAM_IDPROP_* property to set */
                          int32 value,     /**< [in] the value to set */
                          bool  commit = true /**< [in] currently unused */ );

    /// Set a DCAM property on the given camera handle, with dcamprop_setgetvalue().
    /** DCAM may round the value to the nearest valid one.
     *
     * \returns 0 on success
     * \returns -1 on error (logged unless the camera is powering off)
     */
    int setorcaParameter( HDCAM  handle,    /**< [in] the camera handle */
                          int32  parameter, /**< [in] the DCAM_IDPROP_* property to set */
                          double value,     /**< [in] the value to set */
                          bool   commit = true /**< [in] currently unused */ );

    /// Set an integer-valued DCAM property on the given camera handle.
    /**
     * \returns 0 on success
     * \returns -1 on error (logged unless the camera is powering off)
     */
    int setorcaParameter( HDCAM handle,    /**< [in] the camera handle */
                          int32 parameter, /**< [in] the DCAM_IDPROP_* property to set */
                          int32 value,     /**< [in] the value to set */
                          bool  commit = true /**< [in] currently unused */ );

    /// Set a DCAM property on the current camera, with dcamprop_setgetvalue().
    /**
     * \returns 0 on success
     * \returns -1 on error (logged unless the camera is powering off)
     */
    int setorcaParameter( int32  parameter, /**< [in] the DCAM_IDPROP_* property to set */
                          double value,     /**< [in] the value to set */
                          bool   commit = true /**< [in] currently unused */ );

    /// Set a DCAM property on the given camera handle while capturing, with dcamprop_setvalue().
    /** Use for properties DCAM allows to change while capture is running, e.g. exposure time.
     *
     * \returns 0 on success
     * \returns -1 on error (logged unless the camera is powering off)
     */
    int setorcaParameterOnline( HDCAM  handle,    /**< [in] the camera handle */
                                int32  parameter, /**< [in] the DCAM_IDPROP_* property to set */
                                double value /**< [in] the value to set */ );

    /// Set a DCAM property on the current camera while capturing, with dcamprop_setvalue().
    /**
     * \returns 0 on success
     * \returns -1 on error (logged unless the camera is powering off)
     */
    int setorcaParameterOnline( int32  parameter, /**< [in] the DCAM_IDPROP_* property to set */
                                double value /**< [in] the value to set */ );

    /// Set an integer-valued DCAM property on the given camera handle while capturing.
    /** \todo Declared but not defined.
     */
    int setorcaParameterOnline( HDCAM handle,    /**< [in] the camera handle */
                                int32 parameter, /**< [in] the DCAM_IDPROP_* property to set */
                                int32 value /**< [in] the value to set */ );

    /// Set an integer-valued DCAM property on the current camera while capturing.
    /** \todo Declared but not defined.
     */
    int setorcaParameterOnline( int32 parameter, /**< [in] the DCAM_IDPROP_* property to set */
                                int32 value /**< [in] the value to set */ );

    /// Find and open the camera whose DCAM_IDSTR_CAMERAID matches m_serialNumber.
    /** Closes any previous session, initializes the DCAM-API, opens the matching device and a wait
     * handle, and checks for cooling-fan control. Sets the state to CONNECTED, NODEVICE or ERROR.
     *
     * \returns 0 on success, including when no matching camera is found (state NODEVICE)
     * \returns -1 on error
     */
    int connect();

    /// Stop capture and release all DCAM resources held by the app.
    /** Aborts any pending dcamwait so the framegrabber thread isn't left waiting, stops capture,
     * releases the capture buffers and closes the wait handle. Safe to call when nothing is open.
     */
    void closeCamera( bool uninitAPI /**< [in] if true, also call dcamapi_uninit() */ );

    /// Update the app state from the DCAM capture status.
    /** Sets OPERATING while capture is running, otherwise READY, and requests a reconfigure to
     * restart acquisition if capture has stopped.
     *
     * \returns 0 on success
     * \returns -1 on error
     */
    int getAcquisitionState();

    /// Get the current cooling-fan state from the camera.
    /**
     * \returns 0 on success, or if fan status isn't supported
     * \returns -1 on error
     */
    int getFanSpeed();

    /// Get the sensor temperature and cooler status from the camera.
    /** Updates m_ccdTemp and the temperature-control status, and records telemetry.
     *
     * \returns 0 on success
     * \returns -1 on error
     */
    int getTemps();

    ///@}

    /** \name stdCamera Interface
     *@{
     */

    /// Set defaults for a power-on state. [stdCamera interface]
    /** Sets m_ccdTempSetpt, the readout speed and the fan state to their power-on values.
     *
     * \returns 0 always
     */
    int powerOnDefaults();

    /// Turn temperature control on. Temperature control is always on for this camera. [stdCamera interface]
    /**
     * \returns 0 always
     */
    int setTempControl();

    /// Request the temperature setpoint in m_ccdTempSetpt. [stdCamera interface]
    /** Currently only records telemetry and requests a reconfigure; the setpoint isn't written to the
     * camera.
     *
     * \returns 0 always
     */
    int setTempSetPt();

    /// Request a readout-speed change through the next reconfigure. [stdCamera interface]
    /**
     * \returns 0 always
     */
    int setReadoutSpeed();

    /// Request a cooling-fan state change through the next reconfiguration. [stdCamera interface]
    /**
     * \returns 0 always
     */
    int setFanSpeed();

    /// Set the exposure time from m_expTimeSet. [stdCamera interface]
    /** Writes DCAM_IDPROP_EXPOSURETIME, while capturing if necessary, then updates m_expTime and the
     * frame rate.
     *
     * \returns 0 on success
     * \returns -1 on error
     */
    int setExpTime();

    /// Limit an exposure time to no less than the current readout time.
    /**
     * \returns 0 on success
     * \returns -1 if the camera is powering off
     */
    int capExpTime( double &exptime /**< [in,out] exposure time [s], raised to the readout time if needed */ );

    /// FPS is not settable for this camera. [stdCamera interface]
    /**
     * \returns 0 always
     */
    int setFPS();

    /// Apply a MagAO-X center-referenced ROI with the DCAM subarray properties.
    /** Converts the center to a corner, applies the configured flip, checks the sensor bounds, and
     * writes the binning and subarray properties. Only n×n binning is supported, so binY is not used.
     *
     * \returns 0 on success
     * \returns -1 on invalid parameters or a DCAM error
     */
    int setDcamRoi( int32 xCen,   /**< [in] ROI center x [pixels] */
                    int32 yCen,   /**< [in] ROI center y [pixels] */
                    int32 width,  /**< [in] ROI width [pixels] */
                    int32 height, /**< [in] ROI height [pixels] */
                    int32 binX,   /**< [in] binning factor, applied to both axes */
                    int32 binY /**< [in] y binning factor (unused) */ );

    /// Check the next ROI. [stdCamera interface]
    /** Should check whether the target values are valid and adjust them to the closest valid values.
     * Currently does nothing; DCAM rounds the subarray values when setDcamRoi() writes them.
     *
     * \returns 0 always
     */
    int checkNextROI();

    /// Reports whether the camera is currently in focus. [stdCamera interface]
    /**
     * \returns `true` when the configured external focus switch indicates the camera is in focus.
     * \returns `false` otherwise
     */
    // bool checkFocus();

    /// Request the ROI in m_nextROI through the next reconfigure. [stdCamera interface]
    /** Resets the INDI `roi_set` request switch. configureAcquisition() applies the ROI and updates the
     * current and target ROI values.
     *
     * \returns 0 always
     */
    int setNextROI();

    /// Requests the configured focus preset. [stdCamera interface]
    /**
     * \returns 0 on success
     * \returns -1 if the goto-focus helper is not fully configured
     */
    // int gotoFocus();

    /// Sets the shutter state, via call to dssShutter::setShutterState(int) [stdCamera interface]
    /**
     * \returns 0 always
     */
    // int setShutter( int sh );

    ///@}

    /** \name Framegrabber Interface
     *@{
     */

    /// Configure the camera for the pending settings and start continuous acquisition. [framegrabber interface]
    /** Applies the fan state and the ROI and binning, reads back the frame geometry, readout time,
     * exposure-time limits, exposure time and frame rate, then allocates the DCAM buffers and starts
     * capture.
     *
     * \returns 0 on success
     * \returns -1 on error
     */
    int configureAcquisition();

    /// Get the frame rate the camera reports for the current settings. [framegrabber interface]
    /**
     * \returns the frame rate [Hz]
     */
    float fps();

    /// Start acquisition. Capture is already started by configureAcquisition(). [framegrabber interface]
    /**
     * \returns 0 always
     */
    int startAcquisition();

    /// Wait for the next frame and lock the newest one. [framegrabber interface]
    /** Waits up to 1 s for a frame-ready event, then locks the most recently transferred frame into
     * m_currentFrame and timestamps it. Logs skipped frames using the camera timestamps.
     *
     * \returns 0 when a frame is ready
     * \returns 1 on timeout, abort or capture stopped, so the framegrabber can check for reconfigure or
     *          power off
     * \returns -1 on error
     */
    int acquireAndCheckValid();

    /// Copy the locked frame into the image stream, applying any configured flip. [framegrabber interface]
    /**
     * \returns 0 on success
     * \returns -1 on error
     */
    int loadImageIntoStream( void *dest /**< [in] destination in the image stream */ );

    /// Stop capture so the framegrabber can reconfigure. [framegrabber interface]
    /**
     * \returns 0 on success, or if no camera is open
     * \returns -1 on error
     */
    int reconfig();

    ///@}

    /** \name INDI
     *@{
     */

    /// Read-only INDI property `readout_time` publishing the sensor readout time.
    pcf::IndiProperty m_indiP_readouttime;

    ///@}

  public:
    /** \name Telemeter Interface
     *@{
     */

    /// Check whether telemetry needs to be recorded. [telemeter interface]
    /**
     * \returns the result of telemeter::checkRecordTimes() for telem_stdcam
     */
    int checkRecordTimes();

    /// Record telem_stdcam telemetry. [telemeter interface]
    /**
     * \returns the result of recordCamera()
     */
    int recordTelem( const telem_stdcam * /**< [in] tag selecting the telem_stdcam record */ );

    ///@}
};

inline orcaCtrl::orcaCtrl() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    m_powerMgtEnabled = true;

    // m_acqBuff.memory_size = 0;
    // m_acqBuff.memory      = 0;

    m_defaultReadoutSpeed    = "Standard";
    m_readoutSpeedNames      = { "Standard", "Ultra-quiet" };
    m_readoutSpeedNameLabels = { "Standard", "Ultra-quiet" };

    // m_defaultVShiftSpeed    = "1_2us";
    // m_vShiftSpeedNames      = { "0_7us", "1_2us", "2_0us", "5_0us" };
    // m_vShiftSpeedNameLabels = { "0.7 us", "1.2 us", "2.0 us", "5.0 us" };

    m_defaultFanSpeed    = "on";
    m_fanSpeedNames      = { "on", "off" };
    m_fanSpeedNameLabels = { "On", "Off" };
    m_fanSpeedName       = m_defaultFanSpeed;
    m_fanSpeedNameSet    = m_defaultFanSpeed;

    m_full_x = 511.5;
    m_full_y = 511.5;
    m_full_w = 1024;
    m_full_h = 1024;

    return;
}

inline orcaCtrl::~orcaCtrl() noexcept
{
    // Release anything still held. The framegrabber thread has exited by now, so this is safe.
    // A no-op if appShutdown() has already closed everything.
    closeCamera( false );
}

inline void orcaCtrl::setupConfig()
{

    config.add( "camera.serialNumber",
                "",
                "camera.serialNumber",
                argType::Required,
                "camera",
                "serialNumber",
                false,
                "string",
                "The identifying serial number of the camera." );

    config.add( "camera.liquidCooling",
                "",
                "camera.liquidCooling",
                argType::Required,
                "camera",
                "liquidCooling",
                false,
                "bool",
                "Set true if the camera's Cooler Type is set to Water with dcamcfgc and a chiller is connected. "
                "Currently only logged; liquid-cooling control is not yet implemented. Default is false." );

    STDCAMERA_SETUP_CONFIG( config );

    FRAMEGRABBER_SETUP_CONFIG( config );

    // dev::dssShutter<orcaCtrl>::setupConfig( config );

    TELEMETER_SETUP_CONFIG( config );
}

inline int orcaCtrl::loadConfigImpl( mx::app::appConfigurator &_config )
{
    _config( m_serialNumber, "camera.serialNumber" );

    _config( m_liquidCooling, "camera.liquidCooling" );

    STDCAMERA_LOAD_CONFIG( _config );

    FRAMEGRABBER_LOAD_CONFIG( _config );

    // dev::dssShutter<orcaCtrl>::loadConfig( _config );

    TELEMETER_LOAD_CONFIG( _config );

    return 0;
}

inline void orcaCtrl::loadConfig()
{
    if( loadConfigImpl( config ) != 0 )
    {
        log<text_log>( "error loading config", logPrio::LOG_CRITICAL );
        m_shutdown = true;
    }
}

inline int orcaCtrl::appStartup()
{

    createROIndiNumber( m_indiP_readouttime, "readout_time", "Readout Time (s)" );
    indi::addNumberElement<float>(
        m_indiP_readouttime, "value", 0.0, std::numeric_limits<float>::max(), 0.0, "%0.1f", "readout time" );
    registerIndiPropertyReadOnly( m_indiP_readouttime );

    m_minTemp  = -35;
    m_maxTemp  = 25;
    m_stepTemp = 0;

    m_minROIx  = 0;
    m_maxROIx  = 4096;
    m_stepROIx = 0;

    m_minROIy  = 0;
    m_maxROIy  = 2304;
    m_stepROIy = 0;

    m_minROIWidth  = 1;
    m_maxROIWidth  = 4096;
    m_stepROIWidth = 4;

    m_minROIHeight  = 1;
    m_maxROIHeight  = 2304;
    m_stepROIHeight = 1;

    m_minROIBinning_x  = 1;
    m_maxROIBinning_x  = 4;
    m_stepROIBinning_x = 1;

    m_minROIBinning_y  = 1;
    m_maxROIBinning_y  = 4;
    m_stepROIBinning_y = 1;

    STDCAMERA_APP_STARTUP;

    FRAMEGRABBER_APP_STARTUP;

    TELEMETER_APP_STARTUP;

    return 0;
}

inline int orcaCtrl::appLogic()
{
    // run stdCamera's appLogic
    STDCAMERA_APP_LOGIC;

    // then run frameGrabber's appLogic to see if the f.g. thread has exited.
    FRAMEGRABBER_APP_LOGIC;

    if( state() == stateCodes::NOTCONNECTED || state() == stateCodes::NODEVICE || state() == stateCodes::ERROR )
    {
        m_reconfig = true; // Trigger a f.g. thread reconfig.

        // Might have gotten here because of a power off.
        if( powerState() != 1 || powerStateTarget() != 1 )
            return 0;

        std::cerr << __LINE__ << '\n';
        std::unique_lock<std::mutex> lock( m_indiMutex );
        if( connect() < 0 )
        {
            if( powerState() != 1 || powerStateTarget() != 1 )
                return 0;
            log<software_error>( { __FILE__, __LINE__ } );
        }

        if( state() != stateCodes::CONNECTED )
            return 0;
    }

    if( state() == stateCodes::CONNECTED )
    {
        // Get a lock
        std::unique_lock<std::mutex> lock( m_indiMutex );

        if( getAcquisitionState() < 0 )
        {
            if( powerState() != 1 || powerStateTarget() != 1 )
                return 0;
            return log<software_error, 0>( { __FILE__, __LINE__ } );
        }

        if( setTempSetPt() < 0 ) // m_ccdTempSetpt already set on power on
        {
            if( powerState() != 1 || powerStateTarget() != 1 )
                return 0;
            return log<software_error, 0>( { __FILE__, __LINE__ } );
        }

        if( m_fanSpeedControlEnabled && m_fanStatusSupported && getFanSpeed() < 0 )
        {
            if( powerState() != 1 || powerStateTarget() != 1 )
                return 0;
            return log<software_error, 0>( { __FILE__, __LINE__ } );
        }

        FRAMEGRABBER_UPDATE_INDI;
    }

    if( state() == stateCodes::READY || state() == stateCodes::OPERATING )
    {
        // Get a lock if we can
        std::unique_lock<std::mutex> lock( m_indiMutex, std::try_to_lock );

        // but don't wait for it, just go back around.
        if( !lock.owns_lock() )
            return 0;

        if( getAcquisitionState() < 0 )
        {
            if( powerState() != 1 || powerStateTarget() != 1 )
                return 0;

            state( stateCodes::ERROR );
            return 0;
        }

        if( getTemps() < 0 )
        {
            if( powerState() != 1 || powerStateTarget() != 1 )
                return 0;

            state( stateCodes::ERROR );
            return 0;
        }

        if( m_fanSpeedControlEnabled && m_fanStatusSupported && getFanSpeed() < 0 )
        {
            if( powerState() != 1 || powerStateTarget() != 1 )
                return 0;

            state( stateCodes::ERROR );
            return 0;
        }

        STDCAMERA_UPDATE_INDI;

        FRAMEGRABBER_UPDATE_INDI;

        TELEMETER_APP_LOGIC;
    }

    // Nothing to do in other states.
    return 0;
}

inline int orcaCtrl::onPowerOff()
{
    std::lock_guard<std::mutex> lock( m_indiMutex );

    closeCamera( true );

    if( stdCamera<orcaCtrl>::onPowerOff() < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
    }

    if( frameGrabber<orcaCtrl>::onPowerOff() < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
    }

    return 0;
}

inline int orcaCtrl::whilePowerOff()
{
    std::lock_guard<std::mutex> lock( m_indiMutex );

    ///\todo This should call stdCamera<orcaCtrl>::whilePowerOff(), not onPowerOff().

    if( stdCamera<orcaCtrl>::onPowerOff() < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
    }

    return 0;
}

inline int orcaCtrl::appShutdown()
{
    // Stop the framegrabber thread before releasing the DCAM handles it uses.
    FRAMEGRABBER_APP_SHUTDOWN;

    closeCamera( true );

    STDCAMERA_APP_SHUTDOWN;

    TELEMETER_APP_SHUTDOWN;

    return 0;
}

inline int orcaCtrl::getorcaParameter( double &value, int32 property )
{
    const DCAMERR error = dcamprop_getvalue( m_cameraHandle, property, &value );

    if( failed( error ) )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
        {
            return -1;
        }
        log<software_error>( { __FILE__, __LINE__, 0, error, dcamErrorString( m_cameraHandle, error ) } );
        return -1;
    }

    return 0;
}

inline int orcaCtrl::getorcaParameter( int32 &value, int32 property )
{
    double rawValue = 0.0;

    if( getorcaParameter( rawValue, property ) < 0 )
    {
        return -1;
    }

    value = static_cast<int32>( rawValue );
    return 0;
}

inline int orcaCtrl::setorcaParameter( int32 parameter, double value, bool commit )
{
    DCAMERR error = dcamprop_setgetvalue( m_cameraHandle, parameter, &value );
    if( failed( error ) )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;
        log<software_error>( { __FILE__, __LINE__, 0, error, dcamErrorString( m_cameraHandle, error ) } );
        return -1;
    }

    return 0;
}

inline int orcaCtrl::setorcaParameter( HDCAM handle, int32 parameter, double value, bool commit )
{
    DCAMERR error = dcamprop_setgetvalue( handle, parameter, &value );
    if( failed( error ) )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;
        log<software_error>( { __FILE__, __LINE__, 0, error, dcamErrorString( m_cameraHandle, error ) } );
        return -1;
    }

    return 0;
}

inline int orcaCtrl::setorcaParameter( HDCAM handle, int32 parameter, int32 value, bool commit )
{
    return setorcaParameter( handle, parameter, static_cast<double>( value ), commit );
}

inline int orcaCtrl::setorcaParameter( int32 parameter, int32 value, bool commit )
{
    return setorcaParameter( m_cameraHandle, parameter, value, commit );
}

inline int orcaCtrl::setorcaParameterOnline( HDCAM handle, int32 parameter, double value )
{
    DCAMERR error = dcamprop_setvalue( handle, parameter, value );
    if( failed( error ) )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;
        log<software_error>( { __FILE__, __LINE__, 0, error, dcamErrorString( m_cameraHandle, error ) } );
        return -1;
    }

    return 0;
}

inline int orcaCtrl::setorcaParameterOnline( int32 parameter, double value )
{
    return setorcaParameterOnline( m_cameraHandle, parameter, value );
}

inline int orcaCtrl::connect()
{
    // check if prior session exists and clean it up
    closeCamera( true );

    // Init new API instance
    DCAMAPI_INIT apiInit{};
    apiInit.size = sizeof( apiInit );

    DCAMERR error = dcamapi_init( &apiInit );
    if( failed( error ) )
    {
        log<software_error>( { __FILE__, __LINE__, 0, error, "Failed to initialize DCAM API." } );
        state( stateCodes::ERROR );
        return -1;
    }

    const int32 deviceCount = apiInit.iDeviceCount;

    std::cerr << __LINE__ << '\n';

    if( powerState() != 1 || powerStateTarget() != 1 )
        return 0;

    std::cerr << __LINE__ << '\n';

    if( deviceCount == 0 )
    {
        dcamapi_uninit();

        state( stateCodes::NODEVICE );
        if( !stateLogged() )
        {
            log<text_log>( "no Hamamatsu available.", logPrio::LOG_NOTICE );
        }
        return 0;
    }
    else
    {
        std::cerr << "found " << deviceCount << " Hamamatsu.\n";
    }

    // Loop over device idxs and open each device until cam is detected
    for( int32 index = 0; index < deviceCount; ++index )
    {
        DCAMDEV_OPEN deviceOpen{};
        deviceOpen.size  = sizeof( deviceOpen );
        deviceOpen.index = index;

        // Update log when DCAM device fails to open
        error = dcamdev_open( &deviceOpen );
        if( failed( error ) )
        {
            log<software_error>(
                { __FILE__, __LINE__, 0, error, "Could not open DCAM device " + std::to_string( index ) } );
            continue;
        }
        // Query camera id / model

        const std::string cameraID = dcamDeviceString( deviceOpen.hdcam, DCAM_IDSTR_CAMERAID );

        // debugging
        std::cerr << "DCAM cam ID " << cameraID << ", configured serial: '" << m_serialNumber << "'\n";

        if( cameraID != m_serialNumber )
        {
            dcamdev_close( deviceOpen.hdcam );
            continue;
        }

        if( cameraID == m_serialNumber )
        {
            log<text_log>( "Found camera with ID " + m_serialNumber );
            m_cameraName   = dcamDeviceString( deviceOpen.hdcam, DCAM_IDSTR_VENDOR );
            m_cameraModel  = dcamDeviceString( deviceOpen.hdcam, DCAM_IDSTR_MODEL );
            m_cameraHandle = deviceOpen.hdcam;

            // open a wait handle to the camera
            DCAMWAIT_OPEN waitOpen{};
            waitOpen.size  = sizeof( waitOpen );
            waitOpen.hdcam = m_cameraHandle;

            DCAMERR error = dcamwait_open( &waitOpen );

            if( failed( error ) )
            {
                log<software_error>( { __FILE__, __LINE__, 0, error, dcamErrorString( m_cameraHandle, error ) } );
                dcamdev_close( m_cameraHandle );
                m_cameraHandle = nullptr;
                dcamapi_uninit();
                return -1;
            }

            m_waitHandle = waitOpen.hwait;

            m_fanControlSupported = false;

            // Check for camera cooling fan support
            DCAMPROP_ATTR fanAttr{};
            fanAttr.cbSize = sizeof( fanAttr );
            fanAttr.iProp  = DCAM_IDPROP_SENSORCOOLERFAN;

            error = dcamprop_getattr( m_cameraHandle, &fanAttr );

            if( !failed( error ) )
            {

                m_fanControlSupported = ( fanAttr.attribute & DCAMPROP_ATTR_WRITABLE ) != 0;
            }

            else if( error == DCAMERR_NOTSUPPORT || error == DCAMERR_INVALIDPROPERTYID )
            {
                m_fanControlSupported = false;
            }

            else if( failed( error ) )
            {

                state( stateCodes::ERROR );
                log<software_error>( { __FILE__, __LINE__, 0, error, "Error checking CoolingFan support." } );
                dcamwait_close( m_waitHandle );
                m_waitHandle = nullptr;
                dcamdev_close( m_cameraHandle );
                m_cameraHandle = nullptr;
                return -1;
            }

            if( !m_fanControlSupported )
            {
                log<text_log>( "Cooling fan control is enabled in config but not supported. Disabling fan control..." );
            }
        }

        state( stateCodes::CONNECTED );
        log<text_log>( "Connected to " + m_cameraName + " [S/N " + m_serialNumber + "]" );

        if( m_liquidCooling )
        {
            log<text_log>( "cooling mode: liquid (declared by camera.liquidCooling). orcaCtrl does not yet apply "
                           "liquid-cooling settings; set Cooler Type = Water with dcamcfgc and run the chiller.",
                           logPrio::LOG_WARNING );
        }
        else
        {
            log<text_log>( "cooling mode: air" );
        }

        m_readoutSpeedNameSet = m_defaultReadoutSpeed;

        if( m_fanSpeedControlEnabled && m_fanControlSupported )
        {
            m_fanSpeedNameSet    = m_defaultFanSpeed;
            m_fanSpeedLogPending = true;
        }
        return 0;
    }
    // If no camera was found
    state( stateCodes::NODEVICE );

    if( !stateLogged() )
    {
        log<text_log>( "Camera not found in available IDs." );
    }
    dcamapi_uninit();
    return 0;
}

inline void orcaCtrl::closeCamera( bool uninitAPI ) // if arg is true, also call dcamapi_uninit()
{
    if( m_waitHandle )
    {
        dcamwait_abort( m_waitHandle ); // wake framegrabber thread if it's waiting
    }

    if( m_cameraHandle )
    {
        dcamcap_stop( m_cameraHandle );

        if( m_dcamBuffersAllocated )
        {
            dcambuf_release( m_cameraHandle ); // release allocated capture buffers
        }
    }
    m_dcamBuffersAllocated = false;

    if( m_waitHandle )
    {
        dcamwait_close( m_waitHandle );
        m_waitHandle = nullptr;
    }

    if( m_cameraHandle )
    {
        dcamdev_close( m_cameraHandle );
        m_cameraHandle = nullptr;
    }

    if( uninitAPI )
    {
        dcamapi_uninit();
    }
}

inline int orcaCtrl::getAcquisitionState()
{
    int32 captureStatus = 0;

    DCAMERR error = dcamcap_status( m_cameraHandle, &captureStatus );

    const bool acquisitionRunning = !failed( error ) && captureStatus == DCAMCAP_STATUS_BUSY;

    if( MagAOXAppT::m_powerState == 0 )
        return 0;

    if( failed( error ) )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;
        log<software_error>( { __FILE__, __LINE__, 0, error, dcamErrorString( m_cameraHandle, error ) } );
        state( stateCodes::ERROR );
        return -1;
    }

    if( acquisitionRunning || captureStatus == DCAMCAP_STATUS_UNSTABLE )
        state( stateCodes::OPERATING );
    else
        state( stateCodes::READY );

    if( !acquisitionRunning && captureStatus != DCAMCAP_STATUS_UNSTABLE )
    {
        log<text_log>( "acquisition stopped. restarting", logPrio::LOG_ERROR );
        m_reconfig = true;
    }

    return 0;
}

inline int orcaCtrl::getTemps()
{
    double currTemperature;

    if( getorcaParameter( currTemperature, DCAM_IDPROP_SENSORTEMPERATURE ) < 0 )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;

        log<software_error>( { __FILE__, __LINE__ } );
        state( stateCodes::ERROR );
        return -1;
    }

    m_ccdTemp = currTemperature;

    // orcaSensorTemperatureStatus
    int32 status;

    if( getorcaParameter( status, DCAM_IDPROP_SENSORCOOLERSTATUS ) < 0 )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;

        log<software_error>( { __FILE__, __LINE__ } );
        state( stateCodes::ERROR );
        return -1;
    }

    if( status == DCAMPROP_SENSORCOOLERSTATUS__BUSY )
    {
        m_tempControlStatus    = true;
        m_tempControlOnTarget  = false;
        m_tempControlStatusStr = "UNLOCKED";
    }
    else if( status == DCAMPROP_SENSORCOOLERSTATUS__READY )
    {
        m_tempControlStatus    = true;
        m_tempControlOnTarget  = true;
        m_tempControlStatusStr = "LOCKED";
    }
    else if( status == DCAMPROP_SENSORCOOLERSTATUS__WARNING )
    {
        m_tempControlStatus    = false;
        m_tempControlOnTarget  = false;
        m_tempControlStatusStr = "FAULTED";
        log<text_log>( "temperature control faulted", logPrio::LOG_ALERT );
    }
    else
    {
        m_tempControlStatus    = false;
        m_tempControlOnTarget  = false;
        m_tempControlStatusStr = "UNKNOWN";
    }

    recordCamera();

    return 0;
}

inline int orcaCtrl::getFanSpeed()
{
    if( !m_fanStatusSupported )
        return 0;

    int32 status;

    if( getorcaParameter( status, DCAM_IDPROP_SENSORCOOLER ) < 0 )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;

        log<software_error>( { __FILE__, __LINE__ } );
        state( stateCodes::ERROR );
        return -1;
    }

    // bool fanForcedOn = false;

    if( status == DCAMPROP_SENSORCOOLER__OFF )
    {
        m_fanSpeedName = "off";
    }
    else if( status == DCAMPROP_SENSORCOOLER__ON )
    {
        m_fanSpeedName = "on";
    }
    // else if( status == orcaCoolingFanStatus_ForcedOn )
    // {
    //     m_fanSpeedName = "on";
    //     fanForcedOn    = true;
    // }
    else
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "Unknown cooling-fan status returned by orca." } );
    }

    // if( fanForcedOn && !m_fanForcedOn )
    // {
    //     log<text_log>( "cooling fan forced on by camera", logPrio::LOG_NOTICE );
    // }

    // m_fanForcedOn   = fanForcedOn;
    m_fanSpeedValid = true;
    recordCamera();

    return 0;
}

inline int orcaCtrl::setFPS()
{
    return 0;
}

inline int orcaCtrl::powerOnDefaults()
{
    m_ccdTempSetpt = -35; // This is the power on setpoint

    /*m_currentROI.x = 511.5;
    m_currentROI.y = 511.5;
    m_currentROI.w = 1024;
    m_currentROI.h = 1024;
    m_currentROI.bin_x = 1;
    m_currentROI.bin_y = 1;*/

    m_readoutSpeedName = "Standard";
    // m_vShiftSpeedName  = "1_2us";

    if( m_fanSpeedControlEnabled )
    {
        m_fanSpeedName    = m_defaultFanSpeed;
        m_fanSpeedNameSet = m_defaultFanSpeed;
    }
    else
    {
        m_fanSpeedName.clear();
        m_fanSpeedNameSet.clear();
    }

    m_fanForcedOn         = false;
    m_fanControlSupported = false;
    m_fanStatusSupported  = false;
    m_fanSpeedValid       = false;
    m_fanSpeedLogPending  = m_fanSpeedControlEnabled;

    return 0;
}

inline int orcaCtrl::setTempControl()
{
    // Always on
    m_tempControlStatus    = true;
    m_tempControlStatusSet = true;
    updateSwitchIfChanged( m_indiP_tempcont, "toggle", pcf::IndiElement::On, INDI_IDLE );
    recordCamera( true );
    return 0;
}

inline int orcaCtrl::setTempSetPt()
{
    ///\todo bounds check here.
    recordCamera( true );
    m_reconfig = true;
    return 0;
}

inline int orcaCtrl::setReadoutSpeed()
{
    recordCamera( true );
    m_reconfig = true;
    return 0;
}

inline int orcaCtrl::setFanSpeed()
{
    m_fanSpeedLogPending = true;
    recordCamera( true );
    m_reconfig = true;
    return 0;
}

inline int orcaCtrl::setExpTime()
{
    ///\todo This rounds the exposure time to whole seconds. The DCAM exposure time is in seconds, so
    ///      sub-second exposures are lost.
    long   intexptime = m_expTimeSet + 0.5;
    double exptime    = ( (double)intexptime );
    capExpTime( exptime );

    int rv;

    recordCamera( true );

    if( state() == stateCodes::OPERATING )
    {
        rv = setorcaParameterOnline( m_cameraHandle, DCAM_IDPROP_EXPOSURETIME, exptime );
    }
    else
    {
        rv = setorcaParameter( m_cameraHandle, DCAM_IDPROP_EXPOSURETIME, exptime );
    }

    if( rv < 0 )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;
        log<software_error>( { __FILE__, __LINE__, "Error setting exposure time" } );
        return -1;
    }

    m_expTime = exptime;

    recordCamera( true );

    updateIfChanged( m_indiP_exptime, "current", m_expTime, INDI_IDLE );

    if( getorcaParameter( m_FrameRateCalculation, DCAM_IDPROP_INTERNALFRAMERATE ) < 0 )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;
        log<software_error>( { __FILE__, __LINE__, "could not get FrameRateCalculation" } );
    }
    m_fps = m_FrameRateCalculation;

    recordCamera( true );

    return 0;
}

inline int orcaCtrl::capExpTime( double &exptime )
{
    // cap at minimum possible value
    ///\todo The log message below says "ms" but the values are in seconds, and the cap is rounded to
    ///      whole seconds.
    if( exptime < m_ReadOutTimeCalculation )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;
        log<text_log>( "Got exposure time " + std::to_string( exptime ) + " ms but min value is " +
                       std::to_string( m_ReadOutTimeCalculation ) + " ms" );
        long intexptime = m_ReadOutTimeCalculation + 0.5;
        exptime         = ( (double)intexptime );
    }

    return 0;
}

inline int orcaCtrl::setDcamRoi( int32 xCen, int32 yCen, int32 width, int32 height, int32 binX, int32 binY )
{
    // Define sensor dim constants
    constexpr int32 sensorW = 4096;
    constexpr int32 sensorH = 2304;

    if( width <= 0 || height <= 0 || binX <= 0 || binY <= 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "Invalid ROI parameters or binning." } );
    }

    int32 x = xCen - ( width - 1 ) / 2;
    int32 y = yCen - ( height - 1 ) / 2;

    if( m_defaultFlip == fgFlipLR || m_defaultFlip == fgFlipUDLR )
    {
        x = sensorW - x - width;
    }

    if( m_defaultFlip == fgFlipUD || m_defaultFlip == fgFlipUDLR )
    {
        y = sensorH - y - height;
    }

    if( x < 0 || y < 0 || ( x + width ) > sensorW || ( y + height ) > sensorH )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "ROI out of bounds." } );
    }

    if( setorcaParameter( DCAM_IDPROP_SUBARRAYMODE, DCAMPROP_MODE__OFF ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "Error setting subarray mode to off." } );
    }

    if( setorcaParameter( DCAM_IDPROP_BINNING, binX ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "Error setting bin params." } );
    }

    if( setorcaParameter( DCAM_IDPROP_SUBARRAYHPOS, x ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "Error setting xpos." } );
    }

    if( setorcaParameter( DCAM_IDPROP_SUBARRAYVPOS, y ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "Error setting ypos." } );
    }

    if( setorcaParameter( DCAM_IDPROP_SUBARRAYHSIZE, width ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "Error setting width." } );
    }

    if( setorcaParameter( DCAM_IDPROP_SUBARRAYVSIZE, height ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "Error setting height." } );
    }

    if( setorcaParameter( DCAM_IDPROP_SUBARRAYMODE, DCAMPROP_MODE__ON ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "Error setting subarray mode to 'on.'" } );
    }
    return 0;
}

inline int orcaCtrl::checkNextROI()
{
    return 0;
}

inline int orcaCtrl::setNextROI()
{
    updateSwitchIfChanged( m_indiP_roi_set, "request", pcf::IndiElement::Off, INDI_IDLE );
    m_reconfig = true;

    return 0;
}

inline int orcaCtrl::configureAcquisition()
{

    // Make sure the camera responds before configuring it
    int32   captureStatus = 0;
    DCAMERR error         = dcamcap_status( m_cameraHandle, &captureStatus );

    if( failed( error ) )
    {
        log<software_error>( { __FILE__, __LINE__, 0, error, dcamErrorString( m_cameraHandle, error ) } );
        state( stateCodes::ERROR );
        return -1;
    }

    // int32 readoutStride;
    int32 framesPerReadout;
    int32 frameStride;
    // int32 frameSize;
    int32 pixelBitDepth;

    m_camera_timestamp = 0; // reset tracked timestamp

    std::unique_lock<std::mutex> lock( m_indiMutex );

    //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
    //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
    // Readout Speed
    //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
    //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

    ///\todo The readout speed is read but never set from m_readoutSpeedNameSet.
    int32 cmode;
    if( getorcaParameter( cmode, DCAM_IDPROP_READOUTSPEED ) < 0 )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;
        log<software_error>( { __FILE__, __LINE__, "could not get Readout Control Mode" } );
        return -1;
    }

    //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
    //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
    // Cooling Fan
    //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
    //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

    if( m_fanSpeedControlEnabled && m_fanControlSupported )
    {
        static constexpr int32 c_enableCoolingFan  = 0;
        static constexpr int32 c_disableCoolingFan = 1;

        std::string priorFanSpeed     = m_fanSpeedName;
        int32       disableCoolingFan = c_enableCoolingFan;

        if( m_fanSpeedNameSet == "on" )
        {
            disableCoolingFan = c_enableCoolingFan;
        }
        else if( m_fanSpeedNameSet == "off" )
        {
            disableCoolingFan = c_disableCoolingFan;
        }
        else
        {
            if( powerState() != 1 || powerStateTarget() != 1 )
                return -1;
            log<software_error>( { __FILE__, __LINE__, "Invalid fan speed: " + m_fanSpeedNameSet } );
            state( stateCodes::ERROR );
            return -1;
        }

        if( setorcaParameter( m_cameraHandle, DCAM_IDPROP_SENSORCOOLERFAN, disableCoolingFan ) < 0 )
        {
            if( powerState() != 1 || powerStateTarget() != 1 )
                return -1;
            log<software_error>( { __FILE__, __LINE__, "Error setting cooling-fan state" } );
            state( stateCodes::ERROR );
            return -1;
        }

        m_fanSpeedName  = m_fanSpeedNameSet;
        m_fanSpeedValid = true;
        m_fanForcedOn   = false;

        if( m_fanSpeedName != priorFanSpeed )
        {
            log<text_log>( "fan speed changed from '" + priorFanSpeed + "' to '" + m_fanSpeedName + "'",
                           logPrio::LOG_NOTICE );
        }
        else if( m_fanSpeedLogPending )
        {
            log<text_log>( "fan speed set to '" + m_fanSpeedName + "'", logPrio::LOG_NOTICE );
        }

        m_fanSpeedLogPending = false;
    }

    //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
    //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
    // Temperature
    //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
    //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

    // if( setorcaParameter( DCAM_IDPROP_SENSORTEMPERATURETARGET, m_ccdTempSetpt ) < 0 )
    // {
    //     if( powerState() != 1 || powerStateTarget() != 1 )
    //         return -1;
    //     log<software_error>( { __FILE__, __LINE__, "Error setting temperature setpoint" } );
    //     state( stateCodes::ERROR );
    //     return -1;
    // }

    ///\todo The setpoint is not written to the camera (the block above is disabled), so this log message
    ///      is misleading. In air mode the camera has no SENSORTEMPERATURETARGET property.
    log<text_log>( "Set temperature set point: " + std::to_string( m_ccdTempSetpt ) + " C" );

    //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
    //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
    // Dimensions
    //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
    //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

    if( setDcamRoi( m_nextROI.x, m_nextROI.y, m_nextROI.w, m_nextROI.h, m_nextROI.bin_x, m_nextROI.bin_y ) < 0 )
    {
        state( stateCodes::ERROR );
        return -1;
    }

    if( getorcaParameter( frameStride, DCAM_IDPROP_IMAGE_ROWBYTES ) < 0 )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;
        log<software_error>( { __FILE__, __LINE__, "Error getting frame stride" } );
        state( stateCodes::ERROR );

        return -1;
    }

    if( getorcaParameter( framesPerReadout, DCAM_IDPROP_FRAMEBUNDLE_NUMBER ) < 0 )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;
        log<software_error>( { __FILE__, __LINE__, "Error getting frames per readout" } );
        state( stateCodes::ERROR );
        return -1;
    }

    if( getorcaParameter( m_frameSize, DCAM_IDPROP_IMAGE_FRAMEBYTES ) < 0 )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;
        log<software_error>( { __FILE__, __LINE__, "Error getting frame size" } );
        state( stateCodes::ERROR );
        return -1;
    }

    if( getorcaParameter( pixelBitDepth, DCAM_IDPROP_BITSPERCHANNEL ) < 0 )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;
        log<software_error>( { __FILE__, __LINE__, "Error getting pixel bit depth" } );
        state( stateCodes::ERROR );
        return -1;
    }
    m_depth = pixelBitDepth;

    int32 x           = 0;
    int32 y           = 0;
    int32 imageWidth  = 0;
    int32 imageHeight = 0;

    // get DCAM ROI position and size to update current values
    if( getorcaParameter( x, DCAM_IDPROP_SUBARRAYHPOS ) < 0 || getorcaParameter( y, DCAM_IDPROP_SUBARRAYVPOS ) < 0 ||
        getorcaParameter( m_currentROI.w, DCAM_IDPROP_SUBARRAYHSIZE ) < 0 ||
        getorcaParameter( m_currentROI.h, DCAM_IDPROP_SUBARRAYVSIZE ) < 0 ||
        getorcaParameter( m_currentROI.bin_x, DCAM_IDPROP_BINNING ) < 0 ||
        getorcaParameter( imageWidth, DCAM_IDPROP_IMAGE_WIDTH ) < 0 || imageWidth < 0 ||
        getorcaParameter( imageHeight, DCAM_IDPROP_IMAGE_HEIGHT ) < 0 || imageHeight < 0 ||
        getorcaParameter( m_frameSize, DCAM_IDPROP_IMAGE_FRAMEBYTES ) < 0 ||
        getorcaParameter( m_depth, DCAM_IDPROP_BITSPERCHANNEL ) < 0 )
    {
        log<software_error, -1>( { __FILE__, __LINE__, "Error getting ROI parameters" } );
        state( stateCodes::ERROR );
        return -1;
    }

    m_width  = static_cast<uint32_t>( imageWidth );
    m_height = static_cast<uint32_t>( imageHeight );

    // update current ROI values
    m_xbinning = m_currentROI.bin_x;
    m_ybinning = m_currentROI.bin_y;

    m_currentROI.x = x + ( m_currentROI.w - 1 ) / 2;
    m_currentROI.y = y + ( m_currentROI.h - 1 ) / 2;

    updateIfChanged( m_indiP_roi_x, "current", m_currentROI.x, INDI_OK );
    updateIfChanged( m_indiP_roi_y, "current", m_currentROI.y, INDI_OK );
    updateIfChanged( m_indiP_roi_w, "current", m_currentROI.w, INDI_OK );
    updateIfChanged( m_indiP_roi_h, "current", m_currentROI.h, INDI_OK );
    updateIfChanged( m_indiP_roi_bin_x, "current", m_currentROI.bin_x, INDI_OK );
    updateIfChanged( m_indiP_roi_bin_y, "current", m_currentROI.bin_y, INDI_OK );

    // We also update target to the settable values
    m_nextROI.x     = m_currentROI.x;
    m_nextROI.y     = m_currentROI.y;
    m_nextROI.w     = m_currentROI.w;
    m_nextROI.h     = m_currentROI.h;
    m_nextROI.bin_x = m_currentROI.bin_x;
    m_nextROI.bin_y = m_currentROI.bin_y;

    updateIfChanged( m_indiP_roi_x, "target", m_currentROI.x, INDI_OK );
    updateIfChanged( m_indiP_roi_y, "target", m_currentROI.y, INDI_OK );
    updateIfChanged( m_indiP_roi_w, "target", m_currentROI.w, INDI_OK );
    updateIfChanged( m_indiP_roi_h, "target", m_currentROI.h, INDI_OK );
    updateIfChanged( m_indiP_roi_bin_x, "target", m_currentROI.bin_x, INDI_OK );
    updateIfChanged( m_indiP_roi_bin_y, "target", m_currentROI.bin_y, INDI_OK );

    //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
    //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
    // Exposure Time and Frame Rate
    //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
    //=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

    if( getorcaParameter( m_ReadOutTimeCalculation, DCAM_IDPROP_TIMING_READOUTTIME ) < 0 )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;
        return log<software_error, -1>( { __FILE__, __LINE__, "could not get ReadOutTimeCalculation" } );
    }

    std::cerr << "Readout time is: " << m_ReadOutTimeCalculation << "\n";

    ///\todo DCAM reports the readout time in seconds, so this /1000 makes the INDI value 1000x too small.
    updateIfChanged( m_indiP_readouttime, "value", m_ReadOutTimeCalculation / 1000.0, INDI_OK );

    DCAMPROP_ATTR attr{};
    attr.cbSize = sizeof( attr );
    attr.iProp  = DCAM_IDPROP_EXPOSURETIME;

    error = dcamprop_getattr( m_cameraHandle, &attr );

    if( failed( error ) )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;
        log<text_log>( "Constraint count is not 1: " + std::to_string( attr.cbSize ) + " constraints",
                       logPrio::LOG_ERROR );
    }
    else
    {
        m_minExpTime  = attr.valuemin;
        m_maxExpTime  = attr.valuemax;
        m_stepExpTime = attr.valuestep;

        m_indiP_exptime["current"].setMin( m_minExpTime );
        m_indiP_exptime["current"].setMax( m_maxExpTime );
        m_indiP_exptime["current"].setStep( m_stepExpTime );

        m_indiP_exptime["target"].setMin( m_minExpTime );
        m_indiP_exptime["target"].setMax( m_maxExpTime );
        m_indiP_exptime["target"].setStep( m_stepExpTime );
    }

    if( m_expTimeSet > 0 )
    {
        long   intexptime = m_expTimeSet + 0.5;
        double exptime    = ( (double)intexptime );
        capExpTime( exptime );
        std::cerr << "Setting exposure time to " << m_expTimeSet << "\n";
        int rv = setorcaParameter( m_cameraHandle, DCAM_IDPROP_EXPOSURETIME, exptime );

        if( rv < 0 )
        {
            if( powerState() != 1 || powerStateTarget() != 1 )
                return -1;
            return log<software_error, -1>( { __FILE__, __LINE__, "Error setting exposure time" } );
        }
    }

    double exptime;
    if( getorcaParameter( exptime, DCAM_IDPROP_EXPOSURETIME ) < 0 )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;
        return log<software_error, -1>( { __FILE__, __LINE__, "Error getting exposure time" } );
    }
    else
    {
        capExpTime( exptime );
        m_expTime    = exptime;
        m_expTimeSet = m_expTime; // At this point it must be true.
        updateIfChanged( m_indiP_exptime, "current", m_expTime, INDI_IDLE );
        updateIfChanged( m_indiP_exptime, "target", m_expTimeSet, INDI_IDLE );
    }

    if( getorcaParameter( m_FrameRateCalculation, DCAM_IDPROP_INTERNALFRAMERATE ) < 0 )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return -1;
        return log<software_error, -1>( { __FILE__, __LINE__, "Error getting frame rate" } );
    }
    else
    {
        m_fps = m_FrameRateCalculation;
        updateIfChanged( m_indiP_fps, "current", m_fps, INDI_IDLE );
    }
    std::cerr << "FrameRate is: " << m_FrameRateCalculation << "\n";

    recordCamera();

    // Allocate the capture buffers and start continuous acquisition
    error = dcambuf_alloc( m_cameraHandle, m_frameCount );
    if( failed( error ) )
    {
        log<software_error>( { __FILE__, __LINE__, 0, error, dcamErrorString( m_cameraHandle, error ) } );
        state( stateCodes::ERROR );

        return -1;
    }
    m_dcamBuffersAllocated = true;

    error = dcamcap_start( m_cameraHandle, DCAMCAP_START_SEQUENCE );
    if( failed( error ) )
    {
        log<software_error>( { __FILE__, __LINE__, 0, error, dcamErrorString( m_cameraHandle, error ) } );
        state( stateCodes::ERROR );

        return -1;
    }

    m_dataType = _DATATYPE_UINT16;

    return 0;
}

inline float orcaCtrl::fps()
{
    return m_fps;
}

inline int orcaCtrl::startAcquisition()
{
    return 0;
}

inline int orcaCtrl::acquireAndCheckValid()
{
    int32 camTimeOut = 1000; // 1 second keeps us responsive without busy-waiting too much

    // Create DCAMWait struct for acquisition updates
    DCAMWAIT_START waitStart{};
    waitStart.size      = sizeof( waitStart );
    waitStart.eventmask = DCAMWAIT_CAPEVENT_FRAMEREADY | DCAMWAIT_CAPEVENT_STOPPED;
    waitStart.timeout   = camTimeOut;

    DCAMERR error = dcamwait_start( m_waitHandle, &waitStart );

    if( error == DCAMERR_TIMEOUT || error == DCAMERR_ABORT )
    {
        return 1; // This sends it back to framegrabber to check for reconfig, power-off, etc.
    }

    if( failed( error ) )
    {
        log<software_error>( { __FILE__, __LINE__, 0, error, dcamErrorString( m_cameraHandle, error ) } );
        state( stateCodes::ERROR );

        return -1;
    }

    // check if acq completed
    if( waitStart.eventhappened & DCAMWAIT_CAPEVENT_STOPPED )
    {
        return 1;
    }

    // check if frames transferred to computer

    // define transfer info struct for dcamcap_transferinfo
    DCAMCAP_TRANSFERINFO transferInfo{};
    transferInfo.size  = sizeof( transferInfo );
    transferInfo.iKind = DCAMCAP_TRANSFERKIND_FRAME;

    error = dcamcap_transferinfo( m_cameraHandle, &transferInfo );

    if( failed( error ) )
    {
        log<software_error>( { __FILE__, __LINE__, 0, error, dcamErrorString( m_cameraHandle, error ) } );
        state( stateCodes::ERROR );

        return -1;
    }

    // lock the retrieved data for reading
    DCAMBUF_FRAME frame{};
    frame.size   = sizeof( frame );
    frame.iFrame = transferInfo.nNewestFrameIndex; // get idx of most recently transferred frame

    error = dcambuf_lockframe( m_cameraHandle, &frame );

    if( failed( error ) )
    {
        log<software_error>( { __FILE__, __LINE__, 0, error, dcamErrorString( m_cameraHandle, error ) } );
        state( stateCodes::ERROR );

        return -1;
    }

    m_currentFrame = frame;

    clock_gettime( CLOCK_REALTIME, &m_currImageTimestamp );

    if( m_currentFrame.buf == 0 )
    {
        return 1;
    }

    // std::cerr << "readout: " << frame.buf << " " << transferInfo.nFrameCount << "\n";

    // camera time stamp
    const double cameraTimestamp =
        static_cast<double>( frame.timestamp.sec ) + 1.0e-6 * static_cast<double>( frame.timestamp.microsec );

    if( m_camera_timestamp > 0.0 )
    {
        const double delta_ts = cameraTimestamp - m_camera_timestamp;
        // check for a frame skip
        if( delta_ts > 1.5 / m_FrameRateCalculation )
        {
            std::cerr << "Skipped frame(s)! (Expected a " << 1000. / m_FrameRateCalculation << " ms gap but got "
                      << 1000 * delta_ts << " ms)\n";
        }
    }

    m_camera_timestamp = cameraTimestamp; // update to latest

    return 0;
}

inline int orcaCtrl::loadImageIntoStream( void *dest )
{
    if( frameGrabber<orcaCtrl>::loadImageIntoStreamCopy( dest, m_currentFrame.buf, m_width, m_height, m_typeSize ) ==
        nullptr )
        return -1;

    return 0;
}

inline int orcaCtrl::reconfig()
{

    if( !m_cameraHandle )
    {
        return 0; // don't stop capture if camera is already off
    }
    int32 captureStatus = 0;

    DCAMERR error = dcamcap_stop( m_cameraHandle );
    if( failed( error ) )
    {
        log<software_error>( { __FILE__, __LINE__, 0, error, dcamErrorString( m_cameraHandle, error ) } );
        state( stateCodes::ERROR );

        return -1;
    }

    error                         = dcamcap_status( m_cameraHandle, &captureStatus );
    const bool acquisitionRunning = !failed( error ) && captureStatus == DCAMCAP_STATUS_BUSY;

    while( acquisitionRunning && captureStatus != DCAMCAP_STATUS_UNSTABLE )
    {
        if( MagAOXAppT::m_powerState == 0 )
            return 0;
        sleep( 1 );

        error = dcamcap_stop( m_cameraHandle );

        if( failed( error ) )
        {
            log<software_error>( { __FILE__, __LINE__, 0, error, dcamErrorString( m_cameraHandle, error ) } );
            state( stateCodes::ERROR );
            return -1;
        }

        // release buffer
        dcambuf_release( m_cameraHandle );
        m_dcamBuffersAllocated = False;

        error = dcamcap_status( m_cameraHandle, &captureStatus );
        if( failed( error ) )
        {
            log<software_error>( { __FILE__, __LINE__, 0, error, dcamErrorString( m_cameraHandle, error ) } );
            state( stateCodes::ERROR );
            return -1;
        }
    }

    return 0;
}

///\todo checkRecordTimes() and recordTelem() are defined in the header without `inline`, which causes a
///      multiple-definition link error if the header is included in more than one translation unit.
int orcaCtrl::checkRecordTimes()
{
    return telemeter<orcaCtrl>::checkRecordTimes( telem_stdcam() );
}

int orcaCtrl::recordTelem( const telem_stdcam * )
{
    return recordCamera( true );
}

} // namespace app
} // namespace MagAOX
#endif
