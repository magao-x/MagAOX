/**
 * \file cred2Ctrl.hpp
 * \brief MagAO-X C-RED 2 USB camera controller application
 *
 * \ingroup cred2Ctrl_files
 */

#ifndef cred2Ctrl_hpp
#define cred2Ctrl_hpp

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

typedef MagAOX::app::MagAOXApp<true> MagAOXAppT; //This needs to be before pdvUtils.hpp for logging to work.

#include "fli/cred2_sdk.h"



namespace MagAOX
{
namespace app
{

/** \defgroup cred2Ctrl C-RED2 USB Camera
  * \brief Control of the C-RED2 USB Camera.
  *
  * <a href="../handbook/operating/software/apps/cred2KCtrl.html">Application Documentation</a>
  *
  * \ingroup apps
  *
  */

/** \defgroup cred2Ctrl_files C-RED2 USB Camera Files
  * \ingroup cred2Ctrl
  */

/** MagAO-X application to control the C-RED2 Camera via USB
  *
  * \ingroup cred2Ctrl
  *
  */
class cred2Ctrl : public MagAOXApp<>, public dev::stdCamera<cred2Ctrl>, public dev::frameGrabber<cred2Ctrl>,
                                           public dev::telemeter<cred2Ctrl>
{
    friend class dev::stdCamera<cred2Ctrl>;
    friend class dev::frameGrabber<cred2Ctrl>;
    friend class dev::telemeter<cred2Ctrl>;

    typedef MagAOXApp<> MagAOXAppT;

public:
    /** \name app::dev Configuration Flags
      *
      * These constexpr flags tell the MagAO-X device mixins which camera
      * capabilities exist for the C‑RED2 USB camera.
      *
      *@{
      */

    // C‑RED2 provides temperature telemetry and internal TEC control.
    static constexpr bool c_stdCamera_tempControl = true;
    static constexpr bool c_stdCamera_temp        = true;   // ignored when tempControl==true

    // C‑RED2 has fixed readout speeds and no vertical shift registers.
    static constexpr bool c_stdCamera_readoutSpeed = false;
    static constexpr bool c_stdCamera_vShiftSpeed  = false;

    // C‑RED2 is not an EMCCD and has no EM gain register.
    static constexpr bool c_stdCamera_exptimeCtrl = false;  // Exposure determined by FPS
    static constexpr bool c_stdCamera_fpsCtrl     = true;   // FPS is the primary timing control
    static constexpr bool c_stdCamera_fps         = true;   // expose FPS telemetry

    // Free‑run or external sync modes exist.
    static constexpr bool c_stdCamera_synchro   = true;

    // C‑RED2 supports different readout/gain modes.
    static constexpr bool c_stdCamera_usesModes = true;

    // ROI support can be implemented later; for now leave disabled.
    static constexpr bool c_stdCamera_usesROI  = false;
    static constexpr bool c_stdCamera_cropMode = false;

    // No mechanical shutter in C‑RED2.
    static constexpr bool c_stdCamera_hasShutter = false;

    // Allow stdCamera to publish a descriptive state string.
    static constexpr bool c_stdCamera_usesStateString = true;

    // C‑RED2 frames have a fixed orientation.
    static constexpr bool c_frameGrabber_flippable = false;

    ///@}

protected:

    /** \name configurable parameters
      *@{
      */

    // No descramble file needed for C‑RED2 (OCAM2K only).

    ///@}

    // Handle for the C-RED2 camera returned by the SDK.
    cred2_handle m_camHandle {nullptr};

    // Image counters for tracking dropped or skipped frames.
    long m_currImageNumber {-1};
    long m_lastImageNumber {-1};

    // Latest temperature readings from the camera.
   // cred2_temperatures m_temps {};

    // Digital (software) binning state.
    unsigned m_digitalBinX {1};
    unsigned m_digitalBinY {1};
    bool     m_digitalBin  {false};

    // Working buffer for software binning operations.
    mx::improc::eigenImage<int16_t> m_digitalBinWork;

    // External synchronization device properties (MagAO-X timing generator).
    std::string m_syncDevice   {"fxngensync"};
    std::string m_syncFreqProp {"C1freq"};

    // Current synchronization frequency (Hz).
    float m_syncFreq {0.0f};

public:

public:
    /// Default constructor
    cred2Ctrl();

    /// Destructor
    ~cred2Ctrl() noexcept;

    /// Setup the configuration system (called by MagAOXApp::setup())
    virtual void setupConfig();

    /// Load the configuration system results (called by MagAOXApp::setup())
    virtual void loadConfig();

    /// Application startup
    /** Initializes INDI variables and the frame‑grabber thread. */
    virtual int appStartup();

    /// Main application finite‑state‑machine for the C‑RED2
    virtual int appLogic();


    /// FSM state for remaining powered‑off
    virtual int whilePowerOff();

    /// Clean shutdown procedures
    virtual int appShutdown();

    /// Retrieve current temperature telemetry from the C‑RED2
    /**
     * \returns 0 on success
     * \returns -1 on error
     */
   // int getTemps();

    /// Retrieve current frame rate from the camera
    /**
     * \returns 0 on success
     * \returns -1 on error
     */
    int getFPS();

    /** \name stdCamera Interface
      * @{
      */

    /// Set default camera parameters for a power‑on state
    /**
     * \returns 0 on success
     * \returns -1 on error
     */
    int powerOnDefaults();

    /// Enable or disable temperature control (TEC)
    /** Uses the stdCamera variable m_tempControlStatus.
     * \returns 0 on success
     * \returns -1 on error
     */
    //int setTempControl();

    /// Set the camera temperature setpoint
    /** Uses stdCamera::m_ccdTempSetpt.
     * \returns 0 on success
     * \returns -1 on error
     */
    //int setTempSetPt();

    /// Set the frame rate (FPS)
    /** Uses stdCamera::m_fpsSet.
     * \returns 0 on success
     * \returns -1 on error
     */
   // int setFPS();

    /// Set the synchronization (external trigger / freerun)
    /** Uses stdCamera::m_synchroSet.
     * \returns 0 on success
     * \returns -1 on error
     */
    //int setSynchro();

    /// Required by stdCamera; unused for C‑RED2 (exposure = 1/FPS)
    /**
     * \returns 0 always
     */
    int setExpTime();

    /// Required by stdCamera; ROI not currently implemented for C‑RED2
    /**
     * \returns 0 always
     */
    int setNextROI();

    /// No shutter exists on the C‑RED2; always returns 0
    int setShutter(int sh);

    /// Human‑readable camera state string
    //std::string stateString();

    /// Whether stateString() contains valid content
    //bool stateStringValid();

    ///@}

    /// Configure acquisition for the C‑RED2
    /** Applies mode settings, sets FPS, and initializes the C‑RED2 SDK.
     * \returns 0 on success
     * \returns -1 on error
     */
    int configureAcquisition();

    /// Framegrabber FPS interface
    /** Returns the value stored in m_fps. */
    float fps();

    /// Begin image acquisition
    /** Resets frame counters and starts SDK acquisition.
     * \returns 0 on success
     * \returns -1 on error
     */
    int startAcquisition();

    /// Acquire a frame and validate it
    /** Checks the C‑RED2 frame counter for skips or corruption.
     * \returns 0 on success
     * \returns -1 on error
     */
    int acquireAndCheckValid();

    /// Load a frame into the output stream buffer
    /** Also performs optional digital (software) binning.
     * \returns 0 on success
     * \returns -1 on error
     */
    int loadImageIntoStream(void *dest);

    /// Reconfigure acquisition parameters at runtime
    /**
     * \returns 0 on success
     * \returns -1 on error
     */
    int reconfig();



// INDI:
protected:
    // Declare INDI properties used by this driver
    pcf::IndiProperty m_indiP_temps;        // Temperature telemetry
    pcf::IndiProperty m_indiP_syncFreq;     // External sync frequency command

    // C‑RED2 does NOT have EM protection or EM gain, so these are removed.

public:
    // INDI callbacks
    INDI_SETCALLBACK_DECL(cred2Ctrl, m_indiP_syncFreq);

    /** \name Telemeter Interface
      * @{
      */
    int checkRecordTimes();

    //int recordTelem(const cred2_temperatures *);
    int recordTelem(const telem_stdcam *);
    int recordTelem(const telem_fgtimings *);

    //int recordTemps(bool force = false);
    ///@}
};

//
// ─────────────────────────────────────────────────────────────────────────────
//   CONSTRUCTOR & DESTRUCTOR
// ─────────────────────────────────────────────────────────────────────────────
//

inline
cred2Ctrl::cred2Ctrl() :
    MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
    // --- MagAOXApp Power Management ---
    m_powerMgtEnabled = true;
    m_powerOnWait = 10;   // seconds

    // --- stdCamera defaults ---
    m_startupTemp = 20;   // Default temperature setpoint for TEC

    return;
}

inline
cred2Ctrl::~cred2Ctrl() noexcept
{
    return;
}

//
// ─────────────────────────────────────────────────────────────────────────────
//   CONFIGURATION
// ─────────────────────────────────────────────────────────────────────────────
//

inline
void cred2Ctrl::setupConfig()
{
    dev::stdCamera<cred2Ctrl>::setupConfig(config);

    // C‑RED2 uses USB, NOT EDT — so no dev::edtCamera<>
    // dev::edtCamera<cred2Ctrl>::setupConfig(config);  <-- REMOVED

    // C‑RED2 does not use descrambling files (OCAM only)
    // config.add("camera.cred2DescrambleFile", ... );  <-- REMOVED

    dev::frameGrabber<cred2Ctrl>::setupConfig(config);

    // No mechanical shutter on C‑RED2
    // dev::dssShutter<cred2Ctrl>::setupConfig(config);  <-- REMOVED

    dev::telemeter<cred2Ctrl>::setupConfig(config);
}

inline
void cred2Ctrl::loadConfig()
{
    dev::stdCamera<cred2Ctrl>::loadConfig(config);

    // No EDT camera config → removed
    // dev::edtCamera<cred2Ctrl>::loadConfig(config);   <-- REMOVED

    // No descramble file for C‑RED2 → removed
    // config(m_cred2DescrambleFile, "camera.cred2DescrambleFile"); <-- REMOVED

    dev::frameGrabber<cred2Ctrl>::loadConfig(config);

    // No shutter on this camera → removed
    // dev::dssShutter<cred2Ctrl>::loadConfig(config);  <-- REMOVED

    dev::telemeter<cred2Ctrl>::loadConfig(config);
}

/**TO HERE */
inline
int cred2Ctrl::appStartup()
{
    //
    // ────────────────────────────────────────────────────────
    //  INDI Properties
    // ────────────────────────────────────────────────────────
    //

   m_camHandle = cred2_open(0, nullptr, nullptr);
   if(!m_camHandle) {
    return log<software_error,-1>({"Failed to open C-RED2 camera"});
   }

    // Temperature telemetry
    REG_INDI_NEWPROP_NOCB(m_indiP_temps, "temps", pcf::IndiProperty::Number);
    m_indiP_temps.add(pcf::IndiElement("sensor"));
    m_indiP_temps["sensor"].set(0);
    m_indiP_temps.add(pcf::IndiElement("tec"));
    m_indiP_temps["tec"].set(0);

    // Sync frequency command
    REG_INDI_SETPROP(m_indiP_syncFreq, m_syncDevice, m_syncFreqProp);

    //
    // ────────────────────────────────────────────────────────
    //  Device Mixins Startup
    // ────────────────────────────────────────────────────────
    //

    if (dev::stdCamera<cred2Ctrl>::appStartup() < 0)
        return log<software_critical,-1>({""});

    if (dev::frameGrabber<cred2Ctrl>::appStartup() < 0)
        return log<software_critical,-1>({""});

    if (dev::telemeter<cred2Ctrl>::appStartup() < 0)
        return log<software_error,-1>({""});

    // Initialize temperature structure
   // m_temps.invalidate();

    return 0;
}

inline int cred2Ctrl::appLogic()
{
    // Only run the frameGrabber FSM, since SDK has no camera FSM.
    return dev::frameGrabber<cred2Ctrl>::appLogic();
}

/** NOT CURRENTLY SUPPORTED BY SDK
inline
int cred2Ctrl::appLogic()
{
    //
    // Run stdCamera FSM first
    //
    if (dev::stdCamera<cred2Ctrl>::appLogic() < 0)
        return log<software_error,-1>({""});

    //
    // Run frameGrabber FSM
    //
    if (dev::frameGrabber<cred2Ctrl>::appLogic() < 0)
        return log<software_error,-1>({""});

    //
    // Handle connection and power state transitions
    //
    if (state() == stateCodes::POWERON)
        return 0;

   if (state() == stateCodes::NOTCONNECTED ||
      state() == stateCodes::ERROR)
    {
        m_temps.invalidate();

        // Don’t continue connecting if we are powered off
        if (MagAOXAppT::m_powerState == 0)
            return 0;

        // Try opening camera if not connected
        if (cred2_sdk_is_connected(m_camHandle))
        {
            state(stateCodes::CONNECTED);
        }
        else
        {
            sleep(1);
            return 0;
        }
    }


    //
    // Connection established → configure camera
    //
    if (state() == stateCodes::CONNECTED)
    {
        std::unique_lock<std::mutex> lock(m_indiMutex);

        if (getFPS() == 0)
        {
            state(m_fpsSet == 0 ? stateCodes::READY : stateCodes::OPERATING);

            if (m_poweredOn && m_ccdTempSetpt > -999)
            {
                m_poweredOn = false;
                if (setTempSetPt() < 0)
                {
                    if (powerState() != 1 || powerStateTarget() != 1)
                        return 0;
                    return log<software_error,0>({""});
                }
            }

            // Apply default sync setting after connection
            m_synchroSet = false;
            if (setSynchro() != 0)
                log<software_error>({"error from setSynchro on CONNECT"});
        }
        else
        {
            if (powerState() != 1 || powerStateTarget() != 1)
                return 0;

            state(stateCodes::ERROR);
            return log<software_error,0>({""});
        }
    }

    //
    // Ready or Operating → perform periodic camera health checks
    //
    if (state() == stateCodes::READY || state() == stateCodes::OPERATING)
    {
        std::unique_lock<std::mutex> lock(m_indiMutex, std::try_to_lock);

        if (!lock.owns_lock())
            return 0;

        // Temperature
        if (getTemps() < 0)
        {
            if (powerState() != 1 || powerStateTarget() != 1)
                return 0;

            m_temps.invalidate();
            state(stateCodes::ERROR);
            return 0;
        }

        // Frame rate
        if (getFPS() < 0)
        {
            if (powerState() != 1 || powerStateTarget() != 1)
                return 0;

            state(stateCodes::ERROR);
            return 0;
        }

        // Update frameGrabber INDI stats
        if (frameGrabber<cred2Ctrl>::updateINDI() < 0)
        {
            log<software_error>({""});
            state(stateCodes::ERROR);
            return 0;
        }

        // Update stdCamera INDI
        if (stdCamera<cred2Ctrl>::updateINDI() < 0)
        {
            log<software_error>({""});
            state(stateCodes::ERROR);
            return 0;
        }

        // Telemeter logic
        if (telemeter<cred2Ctrl>::appLogic() < 0)
        {
            log<software_error>({""});
            return 0;
        }
    }

    return 0;
}
*/

inline
int cred2Ctrl::whilePowerOff()
{
    std::lock_guard<std::mutex> lock(m_indiMutex);

    if (stdCamera<cred2Ctrl>::whilePowerOff() < 0)
        log<software_error>({""});

    if (frameGrabber<cred2Ctrl>::whilePowerOff() < 0)
        log<software_error>({""});

    // telemeter doesn’t need whilePowerOff()
    return 0;
}

inline
int cred2Ctrl::appShutdown()
{
    dev::stdCamera<cred2Ctrl>::appShutdown();
    dev::frameGrabber<cred2Ctrl>::appShutdown();
    dev::telemeter<cred2Ctrl>::appShutdown();

   CRED2_close(m_camHandle);

    return 0;
}
/** NOT CURRENTLY SUPPORTED by sdk
inline
int cred2Ctrl::getTemps()
{
    // Temporary structure for SDK call
    cred2_temperatures temps;

    // 1. Query the C‑RED2 SDK for temperatures
    if (cred2_get_temperatures(m_camHandle, &temps) != 0)
    {
        if (powerState() != 1 || powerStateTarget() != 1)
            return -1;

        // Mark data invalid
        m_temps.invalidate();
        m_ccdTemp              = 0.0;
        m_ccdTempSetpt         = 0.0;
        m_tempControlStatus    = false;
        m_tempControlStatusStr = "UNKNOWN";

        // Publish telemetry
        recordTemps();
        recordCamera();

        return log<software_error,0>({"Temperature read error"});
    }

    // 2. Save new temperature values
    m_temps = temps;

    // 3. Update stdCamera interface temperature fields
    m_ccdTemp      = temps.sensor;
    m_ccdTempSetpt = temps.setpoint;

    // 4. Determine TEC control status
    if (temps.tec_power < 0.05f)
        m_tempControlStatus = false;
    else
        m_tempControlStatus = true;

    if (m_tempControlStatus)
    {
        if (fabs(m_temps.sensor - m_temps.setpoint) < 1.0)
        {
            m_tempControlStatusStr = "ON TARGET";
            m_tempControlOnTarget  = true;
        }
        else
        {
            m_tempControlStatusStr = "OFF TARGET";
            m_tempControlOnTarget  = false;
        }
    }
    else
    {
        m_tempControlStatusStr = "TEMP OFF";
        m_tempControlOnTarget  = false;
    }

    // 5. Publish telemetry
    recordTemps();
    recordCamera();

    return 0;
}
*/

/** NOT CURRENTLY SUPPORTED BY SDK
int cred2Ctrl::powerOnDefaults()
{
    m_tempControlStatusSet = false;
    m_tempControlStatus    = false;
    return 0;
}
 */


/** NOT SUPPORTED BY SDK 
inline
int cred2Ctrl::setTempControl()
{
    bool enableTEC = m_tempControlStatusSet;

    if (!enableTEC)
    {
        // Only allow disabling when the detector is close to ambient
        if (m_ccdTemp <= 19.0)
        {
            return log<text_log,-1>(
                "Cannot disable TEC unless detector is above 20°C",
                logPrio::LOG_ERROR
            );
        }
    }

    // Write to camera
    if (cred2_set_tec_enabled(m_camHandle, enableTEC) != 0)
    {
        if (powerState() != 1 || powerStateTarget() != 1)
            return -1;

        return log<software_error,-1>({"TEC enable/disable failed"});
    }

    m_tempControlStatus = enableTEC;

    // If enabling and we have a valid setpoint, push it
    if (enableTEC && m_ccdTempSetpt > -999)
        return setTempSetPt();

    recordCamera();
    return 0;
}

*/

/** NOT CURRENTLY SUPPORTED BY SDK
inline
int cred2Ctrl::getFPS()
{
    // If using external sync, FPS = sync frequency
    if (m_synchro)
    {
        m_fps = m_syncFreq;
        recordCamera();
        return 0;
    }

    float fps = 0;

    if (cred2_get_framerate(m_camHandle, &fps) != 0)
    {
        return log<software_error,-1>({"Failed to read FPS"});
    }

    m_fps = fps;
    recordCamera();
    return 0;
}
*/
/** NOT CURRENTLY SUPPORTED BY SDK 
inline
int cred2Ctrl::setFPS()
{
    //
    // When NOT in external sync mode → set FPS via SDK
    //
    if (!m_synchro)
    {
        float fps = m_fpsSet;

        // Call C‑RED2 SDK to set frame rate
        if (cred2_set_framerate(m_camHandle, fps) != 0)
        {
            if (powerState() != 1 || powerStateTarget() != 1)
                return -1;

            return log<software_error,-1>({"Failed to set FPS via C‑RED2 SDK"});
        }

        // Log success
        log<text_log>({"Set FPS: " + std::to_string(fps)});

        // FrameGrabber circular buffers should always be reset after FPS change
        m_reconfig = true;

        // Optional: store mode name for reconfig
        m_nextMode = m_modeName;

        return 0;
    }

    */

    //
    // When in external sync mode → update INDI property to control external generator
    //
    pcf::IndiProperty ipFreq(pcf::IndiProperty::Number);

    ipFreq.setDevice(m_syncDevice);
    ipFreq.setName(m_syncFreqProp);
    ipFreq.add(pcf::IndiElement("target"));

    ipFreq["target"] = std::to_string(m_fpsSet);

    sendNewProperty(ipFreq);

    return 0;
}

/** NOT CURRENTLY SUPPORTED BY SDK
inline
int cred2Ctrl::setSynchro()
{
    //
    // Determine requested mode
    //
    bool enableExternalSync = m_synchroSet;

    int mode = enableExternalSync ?
               CRED2_SYNC_EXTERNAL :
               CRED2_SYNC_INTERNAL;

    //
    // Apply sync mode using C‑RED2 SDK
    //
    if (cred2_set_sync_mode(m_camHandle, mode) != 0)
    {
        if (powerState() != 1 || powerStateTarget() != 1)
            return -1;

        return log<software_error,-1>({"Failed to set C‑RED2 sync mode"});
    }

    //
    // Update internal state and INDI switch
    //
    m_synchro = enableExternalSync;

    if (!m_synchro)
    {
        updateSwitchIfChanged(
            m_indiP_synchro,
            "toggle",
            pcf::IndiElement::Off,
            INDI_IDLE
        );
    

    }

    //
    // In external sync mode, FPS is determined by the timing generator.
    // In internal mode, we must reapply the desired FPS to the camera.
    //
    return setFPS();
}
 */


//
// Exposure time control (unused for C‑RED2)
//
inline
int cred2Ctrl::setExpTime()
{
    // C‑RED2 exposure time = 1 / FPS.
    // No explicit exposure time register exists.
    return 0;
}

//
// ROI control (not currently implemented)
//

/** NOT CURRENTLY SUPPORTED BY SDK 
inline
int cred2Ctrl::setNextROI()
{
    // ROI support can be added later if needed.
    return 0;
}
 */

//
// Shutter control (C‑RED2 has no shutter)
//
inline
int cred2Ctrl::setShutter(int sh)
{
    // No mechanical shutter exists on the C‑RED2.
    // Simply acknowledge the request and do nothing.
    return 0;
}

inline std::string cred2Ctrl::stateString()
{
    return "CRED2";
}

inline bool cred2Ctrl::stateStringValid()
{
    return true;
}

/** 
inline
std::string cred2Ctrl::stateString()
{
    std::string ss;

    // Include readout / gain / HDR mode name
    ss += m_modeName + "_";

    // Current FPS (internal or externally driven)
    ss += std::to_string(m_fps) + "_";

    // CCD temperature setpoint
    ss += std::to_string(m_ccdTempSetpt) + "_";

    // TEC status (ON or OFF)
    ss += (m_tempControlStatus ? "TEC_ON" : "TEC_OFF");

    return ss;
}

inline
bool cred2Ctrl::stateStringValid()
{
    if (state() != stateCodes::OPERATING)
        return false;

    if (!m_tempControlOnTarget)
        return false;

    // In internal FPS mode → FPS must be nonzero
    if (!m_synchro && m_fps <= 0)
        return false;

    // In external sync mode → accept as long as sync is enabled
    return true;
}
*/

inline
int cred2Ctrl::resetEMProtection()
{
    // C‑RED2 does not have an EM gain register or EM‑protection mechanism.
    // This function is required by the stdCamera interface but is a no‑op.

    log<text_log,0>({"resetEMProtection() called, but C‑RED2 has no EM protection system"});

    return 0;
}

inline
int cred2Ctrl::getEMGain()
{
    // C‑RED2 does not support EM gain.
    // This function exists only because stdCamera requires it.
    // Always return success.

    log<text_log,0>({"getEMGain() called, but C‑RED2 has no EM gain system"});

    return 0;
}

inline
int cred2Ctrl::setEMGain()
{
    // C‑RED2 does not support electron-multiplying gain.
    // This function is required by the stdCamera interface but is a no-op.

    log<text_log,0>({"setEMGain() called, but C‑RED2 has no EM gain system"});

    return 0;
}

/** NOT CURRENTLY SUPPORTED BY SDK
inline
int cred2Ctrl::configureAcquisition()
{
    // 1. Lock INDI mutex for thread‑safe configuration
    std::unique_lock<std::mutex> lock(m_indiMutex);

    // 2. Select C‑RED2 camera mode (gain mode, HDR mode, etc.)
    const auto &modeCfg = m_cameraModes[m_modeName];

    if (cred2_set_mode(m_camHandle, modeCfg.sdkMode) != 0)
    {
        if (powerState() != 1 || powerStateTarget() != 1)
            return -1;
        log<software_error>({"Failed to set C‑RED2 mode: " + m_modeName});
        return -1;
    }

    // 3. Configure ROI and hardware binning
    m_currentROI.x     = modeCfg.roiX;
    m_currentROI.y     = modeCfg.roiY;
    m_currentROI.w     = modeCfg.width;
    m_currentROI.h     = modeCfg.height;
    m_currentROI.bin_x = modeCfg.hwBinX;
    m_currentROI.bin_y = modeCfg.hwBinY;

    if (cred2_set_roi(m_camHandle,
                      m_currentROI.x,
                      m_currentROI.y,
                      m_currentROI.w,
                      m_currentROI.h,
                      m_currentROI.bin_x,
                      m_currentROI.bin_y) != 0)
    {
        log<software_error>({"Failed to configure C‑RED2 ROI"});
        return -1;
    }

    // 4. Digital (software) binning handled by MagAO‑X
    m_digitalBinX = modeCfg.digitalBinX;
    m_digitalBinY = modeCfg.digitalBinY;
    m_digitalBin  = (m_digitalBinX > 1 || m_digitalBinY > 1);

    if (m_digitalBin)
        m_digitalBinWork.resize(m_currentROI.h, m_currentROI.w);

    // 5. Publish camera configuration to telemetry
    recordCamera();

    // 6. Apply FPS (internal) or sync frequency (external)
    if (m_fpsSet > 0)
    {
        if (setFPS() < 0)
        {
            log<software_error>({"Error setting FPS in configureAcquisition"});
            return -1;
        }
    }

    // 7. Apply synchronization mode
    if (setSynchro() < 0)
    {
        log<software_error>({"Error setting sync mode in configureAcquisition"});
        return -1;
    }

    // 8. Compute final output dimensions after digital binning
    m_width  = m_currentROI.w / (m_digitalBinX ? m_digitalBinX : 1);
    m_height = m_currentROI.h / (m_digitalBinY ? m_digitalBinY : 1);

    // 9. C‑RED2 always outputs 16‑bit pixels
    m_dataType = _DATATYPE_UINT16;

    // 10. FSM → OPERATING
    state(stateCodes::OPERATING);

    log<text_log>({"C‑RED2 acquisition configured, mode=" + m_modeName});
    return 0;
}

 */

inline
float cred2Ctrl::fps()
{
    return m_fps;
}

inline 
int cred2Ctrl::startAcquisition()
{
    // Reset internal flags and counters
    m_lastImageNumber = -1;
    m_currImageNumber = -1;
    m_newFrame = false;
    m_image_p = nullptr;

    // Register callback-based acquisition
    int ok = CRED2_startAcquisition(
        m_camHandle,
        m_width,            // configured width
        m_height,           // configured height
        &cred2Ctrl::frameCallback,
        this                // user context pointer
    );

    if (ok != 1)
        return log<software_error,-1>({"Failed to start C-RED2 acquisition"});

    return 0;
}

/* NOT CURRENTLY SUPPORTED BY SDK
inline
int cred2Ctrl::startAcquisition()
{
    // Reset frame counter tracking
    m_lastImageNumber = -1;

    //
    // Start streaming via the C‑RED2 SDK
    //
    if (cred2_start_streaming(m_camHandle) != 0)
    {
        // If camera is powered off during state transitions, don’t hard fail
        if (powerState() != 1 || powerStateTarget() != 1)
            return -1;

        return log<software_error,-1>({"Failed to start C‑RED2 acquisition"});
    }

    return 0;
}
*/

/** NOT CURRENTLY SUPPORTED BY SDK
inline
int cred2Ctrl::acquireAndCheckValid()
{
    // 1. Acquire a frame using the C‑RED2 SDK
    cred2_frame frame;
    if (cred2_acquire_frame(m_camHandle, &frame) != 0)
    {
        if (powerState() != 1 || powerStateTarget() != 1)
            return -1;
        return log<software_error,-1>({"Failed to acquire C‑RED2 frame"});
    }

    // Store buffer pointer and metadata
    m_image_p            = frame.data;
    m_currImageTimestamp = frame.timestamp;
    m_currImageNumber    = frame.counter;

    // 2. First frame after restart
    if (m_lastImageNumber == -1)
    {
        m_lastImageNumber = m_currImageNumber - 1;
    }

    // 3. Detect skipped or corrupted frames
    long delta = m_currImageNumber - m_lastImageNumber;

    if (delta != 1)
    {
        bool wraparound =
            (m_lastImageNumber == std::numeric_limits<unsigned>::max() &&
             m_currImageNumber == 0);

        if (!wraparound)
        {
            // 3a. Small skip (soft error)
            if (delta > 1 && delta < 100)
            {
                long skipped = delta - 1;
                log<text_log>("C‑RED2 frames skipped: " + std::to_string(skipped),
                               logPrio::LOG_ERROR);

                m_lastImageNumber = -1;
                m_nextMode        = m_modeName;
                m_reconfig        = 1;
                return 1;
            }

            // 3b. Large or negative jump (corruption)
            if (powerState() != 1 || powerStateTarget() != 1)
                return -1;

            log<text_log>("C‑RED2 frame number possibly corrupt: curr=" +
                           std::to_string(m_currImageNumber) + " last=" +
                           std::to_string(m_lastImageNumber),
                           logPrio::LOG_ERROR);

            m_lastImageNumber = -1;
            m_nextMode        = m_modeName;
            m_reconfig        = 1;
            return 1;
        }
    }

    // 4. Normal case
    m_lastImageNumber = m_currImageNumber;
    return 0;
}

 */


inline
int cred2Ctrl::loadImageIntoStream(void *dest)
{
    int16_t *src = reinterpret_cast<int16_t*>(m_image_p);
    int16_t *out = reinterpret_cast<int16_t*>(dest);

    // 1. If no digital binning, copy frame directly
    if (!m_digitalBin)
    {
        std::memcpy(out,
                    src,
                    m_currentROI.w * m_currentROI.h * sizeof(int16_t));
        return 0;
    }

    // 2. For digital binning: copy source into working buffer
    std::memcpy(m_digitalBinWork.data(),
                src,
                m_currentROI.w * m_currentROI.h * sizeof(int16_t));

    // Eigen maps for easy binning
    mx::improc::eigenMap<int16_t> srcMap(
        m_digitalBinWork.data(),
        m_currentROI.w,
        m_currentROI.h
    );
    mx::improc::eigenMap<int16_t> dstMap(
        out,
        m_width,
        m_height
    );

    // 3. Apply X/Y digital binning (averaged binning)
    for (int y = 0; y < m_height; ++y)
    {
        for (int x = 0; x < m_width; ++x)
        {
            long sum = 0;
            int count = 0;

            for (unsigned by = 0; by < m_digitalBinY; ++by)
            for (unsigned bx = 0; bx < m_digitalBinX; ++bx)
            {
                sum += srcMap(y * m_digitalBinY + by,
                              x * m_digitalBinX + bx);
                count++;
            }

            dstMap(y, x) = sum / count;
        }
    }

    return 0;
}


inline
int cred2Ctrl::reconfig()
{
    //
    // Stop streaming first (safe reset)
    //
    if (cred2_stop_streaming(m_camHandle) != 0)
    {
        log<software_error>({"Failed to stop C‑RED2 streaming during reconfig"});
        return -1;
    }

    //
    // Reapply full acquisition configuration
    //
    if (configureAcquisition() < 0)
    {
        log<software_error>({"Failed to reconfigure C‑RED2 during reconfig()"});
        return -1;
    }

    //
    // Reset frame counters
    //
    m_lastImageNumber = -1;
    m_currImageNumber = -1;

    //
    // Transition FSM back to READY
    //
    state(stateCodes::READY);

    log<text_log>({"C‑RED2 reconfiguration complete"});
    return 0;
}


INDI_SETCALLBACK_DEFN(cred2Ctrl, m_indiP_syncFreq)(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS(ipRecv, m_indiP_syncFreq);

    // Must have the "current" element
    if (!ipRecv.find("current"))
        return -1;

    // Extract sync frequency
    m_syncFreq = ipRecv["current"].get<double>();

    // If camera is in external-sync mode, sync frequency defines FPS
    if (m_synchro && m_syncFreq != m_fps)
    {
        // Update FPS and notify camera telemetry
        recordCamera(true);

        m_fps = m_syncFreq;

        // Changing FPS requires FG reconfigure (MagAO-X pattern)
        m_nextMode = m_modeName;
        m_reconfig = true;
    }

    return 0;
}

inline
int cred2Ctrl::checkRecordTimes()
{
    return telemeter<cred2Ctrl>::checkRecordTimes(
    //    cred2_temperatures(),    // temperature telemetry struct
        telem_stdcam(),          // stdCamera telemetry
        telem_fgtimings()        // frame‑grabber timing telemetry
    );
}

/** Not current supported by SDK
inline
int cred2Ctrl::recordTelem(const cred2_temperatures *)
{
    return recordTemps(true);
}

 */

inline
int cred2Ctrl::recordTelem(const telem_stdcam *)
{
    return recordCamera(true);
}

inline
int cred2Ctrl::recordTelem(const telem_fgtimings *)
{
    return recordFGTimings(true);
}

/** NOT CURRENTLY SUPPORTED BY SDK
inline
int cred2Ctrl::recordTemps(bool force)
{
    static cred2_temperatures lastTemps;

    // Only publish if changed or forced
    if (!(lastTemps == m_temps) || force)
    {
        telem<cred2_temperatures>({
            m_temps.sensor,      // detector temperature
            m_temps.tec,         // TEC temperature
            m_temps.tec_power    // TEC power (%)
        });

        lastTemps = m_temps;
    }

    return 0;
}
  */


}//namespace app
} //namespace MagAOX

#endif
