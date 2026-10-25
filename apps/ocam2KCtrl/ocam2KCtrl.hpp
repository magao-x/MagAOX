/** \file ocam2KCtrl.hpp
 * \brief The MagAO-X OCAM2K EMCCD camera controller.
 *
 * \ingroup ocam2KCtrl_files
 */

#ifndef ocam2KCtrl_hpp
#define ocam2KCtrl_hpp

#include <fcntl.h>
#include <edtinc.h>
#include <unistd.h>

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

typedef MagAOX::app::MagAOXApp<true> MagAOXAppT; // This needs to be before pdvUtils.hpp for logging to work.

#include "fli/ocam2_sdk.h"
#include "ocamUtils.hpp"

namespace MagAOX
{
namespace app
{

/** \defgroup ocam2KCtrl OCAM2K EMCCD Camera
 * \brief Control of the OCAM2K EMCCD Camera.
 *
 * <a href="../handbook/operating/software/apps/ocam2KCtrl.html">Application Documentation</a>
 *
 * \ingroup apps
 *
 */

/** \defgroup ocam2KCtrl_files OCAM2K EMCCD Camera Files
 * \ingroup ocam2KCtrl
 */

/** MagAO-X application to control the OCAM 2K EMCCD
 *
 * \ingroup ocam2KCtrl
 *
 */
class ocam2KCtrl : public MagAOXApp<>,
                   public dev::stdCamera<ocam2KCtrl>,
                   public dev::edtCamera<ocam2KCtrl>,
                   public dev::frameGrabber<ocam2KCtrl>,
                   public dev::dssShutter<ocam2KCtrl>,
                   public dev::telemeter<ocam2KCtrl>
{
    friend class dev::stdCamera<ocam2KCtrl>;
    friend class dev::edtCamera<ocam2KCtrl>;
    friend class dev::frameGrabber<ocam2KCtrl>;
    friend class dev::dssShutter<ocam2KCtrl>;
    friend class dev::telemeter<ocam2KCtrl>;

    typedef MagAOXApp<> MagAOXAppT;

    typedef dev::stdCamera<ocam2KCtrl> stdCameraT;

  public:
    /** \name app::dev Configurations
     *@{
     */
    static constexpr bool c_stdCamera_tempControl =
        true; ///< app::dev config to tell stdCamera to expose temperature controls

    static constexpr bool c_stdCamera_tempLimits =
        true; ///< app::dev config to enable configurable temperature limits and request validation

    static constexpr bool c_stdCamera_temp =
        true; ///< app::dev config to tell stdCamera to expose temperature (ignored since tempControl==true)

    static constexpr bool c_stdCamera_readoutSpeed =
        false; ///< app::dev config to tell stdCamera not to expose readout speed controls

    static constexpr bool c_stdCamera_vShiftSpeed =
        false; ///< app:dev config to tell stdCamera not to expose vertical shift speed control
    static constexpr bool c_stdCamera_fanSpeed =
        false; ///< app::dev config to tell stdCamera not to expose fan-speed control

    static constexpr bool c_stdCamera_emGain = true; ///< app::dev config to tell stdCamera to expose EM gain controls

    static constexpr bool c_stdCamera_exptimeCtrl =
        false; ///< app::dev config to tell stdCamera not to expose exposure time controls

    static constexpr bool c_stdCamera_fpsCtrl = true; ///< app::dev config to tell stdCamera to expose FPS controls

    static constexpr bool c_stdCamera_fps =
        true; ///< app::dev config to tell stdCamera not to expose FPS status (ignored since fpsCtrl==true)

    static constexpr bool c_stdCamera_synchro =
        true; ///< app::dev config to tell stdCamera to expose synchro mode controls

    static constexpr bool c_stdCamera_usesModes =
        true; ///< app:dev config to tell stdCamera not to expose mode controls

    static constexpr bool c_stdCamera_usesROI = false; ///< app:dev config to tell stdCamera to expose ROI controls

    static constexpr bool c_stdCamera_cropMode =
        false; ///< app:dev config to tell stdCamera to expose Crop Mode controls

    static constexpr bool c_stdCamera_hasShutter =
        true; ///< app:dev config to tell stdCamera to expose shutter controls

    static constexpr bool c_stdCamera_usesStateString =
        true; ///< app::dev confg to tell stdCamera to expose the state string property

    static constexpr bool c_edtCamera_relativeConfigPath =
        true; ///< app::dev config to tell edtCamera to use relative path to camera config file

    static constexpr bool c_frameGrabber_flippable =
        false; ///< app:dev config to tell framegrabber these images can not be flipped

    ///@}
  protected:
    /** \name configurable parameters
     *@{
     */

    std::string m_ocamDescrambleFile; ///< Path to the OCAM 2K pixel descrambling file, relative to the MagAO-X config
                                      ///< directory.

    std::string m_syncShmimName; ///< Name of the sync-only ImageStreamIO output. Defaults to `framegrabber.shmimName +
                                 ///< "_sync"`.

    ///@}

    static constexpr uint32_t c_syncStreamWidth{ 1 };  ///< Width of the sync-only ImageStreamIO stream.
    static constexpr uint32_t c_syncStreamHeight{ 1 }; ///< Height of the sync-only ImageStreamIO stream.
    static constexpr uint32_t c_syncStreamDepth{ 1 };  ///< Depth of the sync-only ImageStreamIO stream.
    static constexpr uint8_t  c_syncStreamDataType{ _DATATYPE_UINT8 }; ///< Data type of the sync-only stream.

    ocam2_id m_ocam2_id{ 0 }; ///< OCAM SDK id.

    bool m_ocam2Initialized{ false }; ///< SDK initialization succeeded; numeric handle zero is valid.

    long m_currImageNumber{ -1 }; ///< The current image number, retrieved from the raw image itself.

    long m_lastImageNumber{ -1 }; ///< The last image number saved from the previous acquisition loop.

    bool m_protectionReset{ false }; ///< Flag indicating that protection has been reset at least once.

    unsigned m_protectionResetConfirmed{ 0 }; ///< Counter indicating the number of times that the protection reset has
                                              ///< been requested within 10 seconds, for confirmation.

    double m_protectionResetReqTime{
        0 }; ///< The time at which protection reset was requested.  You have 10 seconds to confirm.

    bool m_poweredOn{
        false }; ///< Tracks that power-on defaults still need to restore the configured temperature setpoint.

    ocamTemps m_temps; ///< Structure holding the last temperature measurement.

    unsigned m_digitalBinX{ 1 }; ///< Digital x-binning factor applied after descrambling.

    unsigned m_digitalBinY{ 1 }; ///< Digital y-binning factor applied after descrambling.

    bool m_digitalBin{ false }; ///< Indicates whether post-descramble digital binning is active for the current mode.

    mx::improc::eigenImage<int16_t>
        m_digitalBinWork; ///< Scratch image holding the full descrambled frame before digital binning.

    std::string m_syncDevice{
        "fxngensync" }; ///< INDI device providing the external sync frequency when synchro mode is enabled.
    std::string m_syncFreqProp{ "C1freq" }; ///< INDI property name reporting the external sync frequency.

    std::atomic<float> m_workerFPS{ 0 }; ///< Published rate; the worker never reads control-owned FPS storage.

    float m_syncFreq{ 0 }; ///< Current externally supplied sync frequency in Hz.

    IMAGE *m_syncImageStream{ nullptr }; ///< Secondary 1x1 uint8 ImageStreamIO stream used only for frame metadata
                                         ///< and semaphores. Its lifetime is sequenced by the framegrabber thread and
                                         ///< app shutdown so per-frame publication can remain lock-free.

    std::recursive_mutex
        m_cameraMutex; ///< Serializes control/reconfigure access; normal grabs use frame-worker ownership.

    /// Confirmation result for one current request or initial observation.
    enum class ConfirmationPhase
    {
        unavailable,
        confirming,
        confirmed,
        failed
    };

    /// Ordinary-thread request state, protected by m_cameraMutex.
    struct Confirmation
    {
        ConfirmationPhase phase{ ConfirmationPhase::unavailable }; ///< Current confirmation result.
        double            expected{ 0 };      ///< Dispatched value, distinct from the latest observation.
        bool              requested{ false }; ///< Match a target rather than accepting an initial observation.
        uint64_t          generation{ 0 };    ///< Superseding requests invalidate older observations.
        int64_t           due{ 0 };           ///< Next readback opportunity, monotonic ns.
        int64_t           deadline{ 0 };      ///< Fixed acknowledgement deadline, monotonic ns.
    };

    double m_tempIntervalSec{ 10 }; ///< Idle temperature sampling interval.

    double m_tempChangingIntervalSec{ 1 }; ///< Sampling interval during thermal convergence.

    double m_confirmationIntervalSec{ 1 }; ///< Pending readback retry interval.

    double m_confirmationTimeoutSec{ 10 }; ///< Fixed deadline for setting confirmation.

    unsigned m_tempSettleSamples{ 2 }; ///< Consecutive settled observations needed for idle polling.

    Confirmation m_gainConfirmation; ///< Latest gain request or startup observation.

    Confirmation m_fpsConfirmation; ///< Final mode/synchronization rate confirmation.

    Confirmation m_tempConfirmation; ///< Setpoint and cooling-state acknowledgement, separate from settling.

    int64_t m_tempDue{ 0 }; ///< Next temperature query opportunity, monotonic ns.

    bool m_tempChanging{ true }; ///< Thermal convergence requires faster observations.

    unsigned m_tempSettledSamples{ 0 }; ///< Consecutive fresh settled samples.

    int m_tempExpectedControl{ -1 }; ///< Pending cooling state: -1 unspecified, 0 off, 1 on.

    bool m_tempExpectedSet{ false }; ///< A pending temperature request must match its stored setpoint.

    bool m_coolingKnown{ false }; ///< The latest report contains sufficient evidence of cooling state.

    int64_t m_syncFreqTime{ 0 }; ///< Most recent valid external frequency report, monotonic ns.

    uint64_t m_housekeepingTempQueries{ 0 }; ///< Temperature getter attempts, including explicit interlocks.

    uint64_t m_housekeepingGainQueries{ 0 }; ///< Gain getter attempts.

    uint64_t m_housekeepingFpsQueries{ 0 }; ///< Serial FPS getter attempts; external reports are excluded.

    static constexpr int64_t c_housekeepingPauseNs{ 120000000000 }; ///< Maximum diagnostic pause, monotonic ns.

    int64_t m_housekeepingPauseUntil{ 0 }; ///< Pause deadline; zero means polling, protected by m_cameraMutex.

    int64_t m_housekeepingPauseStarted{ 0 }; ///< Last actual pause start, monotonic ns; camera-mutex protected.

    int64_t m_housekeepingPauseEnded{ 0 }; ///< Last actual pause end, monotonic ns; camera-mutex protected.

    int64_t m_housekeepingTempTime{ 0 }; ///< Last successfully parsed temperature reply, monotonic ns.

    int64_t m_housekeepingGainTime{ 0 }; ///< Last successfully parsed gain reply, monotonic ns.

    int64_t m_housekeepingFpsTime{ 0 }; ///< Last successfully parsed serial FPS reply, monotonic ns.

    uint64_t m_housekeepingSkippedTemp{ 0 }; ///< Routine temperature polls suppressed by the diagnostic pause.

    uint64_t m_housekeepingSkippedGain{ 0 }; ///< Legacy pause counter; confirmed gain has no routine polls.

    uint64_t m_housekeepingSkippedFps{ 0 }; ///< Legacy pause counter; confirmed FPS has no routine polls.

    uint64_t m_housekeepingSerialTransactions{ 0 }; ///< Serial attempts through the app, camera-mutex protected.

    std::string m_housekeepingReason{ "scheduled polling" }; ///< Last pause transition, camera-mutex protected.

  public:
    /// Default c'tor
    ocam2KCtrl();

    /// Destructor
    ~ocam2KCtrl() noexcept;

    /// Setup the configuration system (called by MagAOXApp::setup())
    virtual void setupConfig();

    /// load the configuration system results (called by MagAOXApp::setup())
    virtual void loadConfig();

    /// Load and validate configuration, returning -1 on failure.
    int loadConfigImpl();

    /// Startup functions
    /** Sets up the INDI vars, and the f.g. thread.
     *
     */
    virtual int appStartup();

    /// Implementation of the FSM for the OCAM 2K.
    virtual int appLogic();

    /// Implementation of the on-power-off FSM logic
    virtual int onPowerOff();

    /// Implementation of the while-powered-off FSM
    virtual int whilePowerOff();

    /// Do any needed shutdown tasks.
    virtual int appShutdown();

    /// End a quiet interval before issuing an explicit serial transaction on the ordinary control path.
    int pdvSerialWriteRead( std::string       &response /**< [out] Camera response. */,
                            const std::string &command /**< [in] Serial command to send. */,
                            bool               logErrors = true /**< [in] Log transport errors when true. */ );

    /// Get the current device temperatures
    /** A malformed reply leaves the readback unavailable for bounded confirmation retries.
     * \returns 0 after a completed exchange, including a malformed numeric reply.
     * \returns -1 on transport failure.
     */
    int getTemps();

    /// Get the current frame rate.
    /** A malformed reply leaves the readback unavailable for bounded confirmation retries.
     * \returns 0 after a completed exchange, including a malformed numeric reply.
     * \returns -1 on transport failure.
     */
    int getFPS();

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
    /** Sets temperature control from m_tempControlStatusSet and starts bounded state confirmation.
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

    /// Set the frame rate. [stdCamera interface]
    /** Sets the frame rate to m_fpsSet.
     *
     * \returns 0 on success
     * \returns -1 on error
     */
    int setFPS( bool requestReconfiguration = true /**< [in] Queue a same-mode restart for a control request. */ );

    /// Set the synchro state. [stdCamera interface]
    /** Sets the synchro state to m_synchroSet.
     *
     * \returns 0 on success
     * \returns -1 on error
     */
    int setSynchro( bool requestReconfiguration = true /**< [in] Queue a same-mode restart for a control request. */ );

    /// Required by stdCamera, but this does not do anything for this camera [stdCamera interface]
    /**
     * \returns 0 always
     */
    int setExpTime();

    /// Required by stdCamera, but this does not do anything for this camera [stdCamera interface]
    /**
     * \returns 0 always
     */
    int setNextROI();

    /// Sets the shutter state, via call to dssShutter::setShutterState(int) [stdCamera interface]
    /**
     * \returns 0 always
     */
    int setShutter( int sh /**< [in] requested shutter state */ );

    /// Return the current stdCamera state string.
    std::string stateString();

    /// Report whether the current stdCamera state string is valid for persistence and telemetry.
    bool stateStringValid();

    ///@}

    /// Reset the EM Protection
    /**
     * \returns 0 on success
     * \returns -1 on error
     */
    int resetEMProtection();

    /// Get the current EM Gain.
    /** A malformed reply leaves the readback unavailable for bounded confirmation retries.
     * \returns 0 after a completed exchange, including a malformed numeric reply.
     * \returns -1 on transport failure or a received HV trip.
     */
    int getEMGain();

    /// Set the EM gain.
    /** Dispatches the validated integer value of stdCamera::m_emGainSet once, then confirms the readback.
     *
     * \returns 0 after dispatch or an interlock/range refusal
     * \returns -1 on error
     */
    int setEMGain();

    /// Implementation of the framegrabber configureAcquisition interface
    /** Sends the mode command over serial, sets the FPS, and initializes the OCAM SDK.
     *
     * \returns 0 on success
     * \returns -1 on error
     */
    int configureAcquisition();

    /// Implementation of the frameGrabber fps interface
    /** Returns the control-published atomic rate without taking an ordinary-thread mutex.
     */
    float fps();

    /// Implementation of the framegrabber startAcquisition interface
    /** Initializes m_lastImageNumber, and calls edtCamera::pdvStartAcquisition
     *
     * \returns 0 on success
     * \returns -1 on error
     */
    int startAcquisition();

    /// Implementation of the framegrabber acquireAndCheckValid interface
    /** Calls edtCamera::pdvAcquire, then analyzes the OCAM generated framenumber for skips and corruption.
     *
     * \returns 0 on success
     * \returns -1 on error
     */
    int acquireAndCheckValid();

    /// Implementation of the framegrabber loadImageIntoStream interface
    /** Conducts the OCAM descramble.
     *
     * \returns 0 on success
     * \returns -1 on error
     */
    int loadImageIntoStream( void *dest /**< [out] Destination image buffer. */ );

    /// Implementation of the framegrabber reconfig interface
    /** Serializes control access and calls edtCamera::pdvReconfig on the frame worker.
     * \returns 0 on success
     * \returns -1 on error
     */
    int reconfig();

    /// Publish the sync-only stream after the main image, then replenish an eligible EDT ring.
    /** This hot path relies on framegrabber sequencing: configureAcquisition() and this hook both run on the
     * framegrabber thread, while appShutdown() destroys the sync stream only after frameGrabber::appShutdown()
     * joins that thread. Deferred replenishment remains on this same worker and runs after both notifications.
     */
    int frameGrabberPostPublish( IMAGE *imageStream /**< [in] main framegrabber stream that was just posted */ );

  protected:
    /// Read the monotonic clock in nanoseconds for ordinary-thread housekeeping decisions.
    virtual int64_t housekeepingNow() const;

    /// Start a superseding request with one fixed monotonic deadline.
    void beginConfirmation( Confirmation &confirmation /**< [in,out] Request state. */,
                            double        expected /**< [in] Dispatched target or unused initial value. */,
                            bool          requested /**< [in] Whether a matching target is required. */ );

    /// Accept a current numeric observation without reviving an expired request.
    void confirmValue( Confirmation &confirmation /**< [in,out] Current request. */,
                       double        value /**< [in] Valid observed value. */,
                       double        tolerance /**< [in] Numeric representation tolerance. */ );

    /// Invalidate cached confirmations after a camera lifecycle transition.
    void resetHousekeeping();

    /// Validate app-local polling intervals and sample count.
    int validateHousekeepingConfig();

    /// Apply one strictly parsed temperature report and reevaluate confirmation and convergence.
    void acceptTemps( const ocamTemps &temps /**< [in] Valid temperature report. */ );

    /// Register the runtime pause control and read-only housekeeping diagnostics.
    int initHousekeeping();

    /// Start or stop the bounded pause after acquiring the serial transaction mutex.
    int setHousekeepingPause( bool    pause /**< [in] Request a pause when true, otherwise resume. */,
                              int64_t now /**< [in] Monotonic request evaluation time, ns. */ );

    /// End an active quiet interval without changing camera settings.
    void endHousekeepingPause( const std::string &reason /**< [in] Reason recorded for resuming polling. */ );

    /// Expire or cancel the pause and return whether automatic serial polls remain suppressed.
    bool housekeepingPaused( int64_t now /**< [in] Monotonic evaluation time, ns. */ );

    /// Service at most one due serial query and confirmation deadlines; the caller holds m_indiMutex.
    int pollHousekeeping();

    /// Publish confirmation, temperature phase, readback ages and counters; the caller holds m_indiMutex.
    void updateHousekeeping();

    /// Ensure the sync-only ImageStreamIO stream exists with the expected 1x1 uint8 layout.
    int ensureSyncStream();

    /// Destroy the sync-only ImageStreamIO stream owned by this app.
    int destroySyncStream();

    /// Acknowledge configuration under the control lock without erasing a newer requested mode.
    void frameGrabberReconfigureComplete();

    /// Request same-mode recovery without replacing a newer control request.
    void requestAcquisitionRecovery();

    /// Release a successful SDK initialization while the frame worker is stopped or reconfiguring.
    void releaseOcamSdk();

    // INDI:
  protected:
    // declare our properties
    pcf::IndiProperty m_indiP_temps;       ///< INDI property publishing the latest camera temperatures.
    pcf::IndiProperty m_indiP_emProt;      ///< INDI property publishing the EM protection state.
    pcf::IndiProperty m_indiP_emProtReset; ///< INDI property handling operator requests to reset EM protection.

    pcf::IndiProperty m_indiP_housekeepingPause; ///< Runtime toggle for a finite pause of routine serial queries.

    pcf::IndiProperty m_indiP_housekeeping; ///< Read-only monotonic ages, timeout and serial/poll counters.

    pcf::IndiProperty m_indiP_housekeepingStatus; ///< Cached readback validity and protection-polling state.

    pcf::IndiProperty m_indiP_syncFreq; ///< INDI subscription to the external sync-frequency source.

  public:
    /// Start a 120-second quiet interval or explicitly resume routine serial polling.
    INDI_NEWCALLBACK_DECL( ocam2KCtrl, m_indiP_housekeepingPause );

    /// Receive an EM-protection reset request.
    INDI_NEWCALLBACK_DECL( ocam2KCtrl, m_indiP_emProtReset );

    /// Receive the external synchronization frequency.
    INDI_SETCALLBACK_DECL( ocam2KCtrl, m_indiP_syncFreq );

    /** \name Telemeter Interface
     *
     * @{
     */
    /// Check the OCAM, camera, and framegrabber telemetry recording deadlines.
    int checkRecordTimes();

    /// Record OCAM temperature telemetry.
    int recordTelem( const ocam_temps *telem /**< [in] Telemetry type selector. */ );

    /// Record standard camera telemetry.
    int recordTelem( const telem_stdcam *telem /**< [in] Telemetry type selector. */ );

    /// Record framegrabber timing telemetry.
    int recordTelem( const telem_fgtimings *telem /**< [in] Telemetry type selector. */ );

    /// Record a forced cached frame-stage/quality snapshot on the ordinary telemetry path.
    int recordTelem( const telem_fgstage *telem /**< [in] Telemetry type selector. */ );

    /// Record the latest OCAM temperatures when they change or when forced.
    int
    recordTemps( bool force = false /**< [in] whether to force a telemetry record even if values have not changed */ );

    ///@}
};

inline ocam2KCtrl::ocam2KCtrl() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    //--- MagAOXApp Power Mgt. ---
    m_powerMgtEnabled = true;
    m_powerOnWait     = 10;

    //--- stdCamera ---
    m_startupTemp = -45;
    // Preserve OCAM's inclusive lower limit and exclusive 30 C upper limit.
    setTempLimits( -50, std::nextafter( 30.0f, -std::numeric_limits<float>::infinity() ) );

    m_maxEMGain = 600;

    return;
}

inline ocam2KCtrl::~ocam2KCtrl() noexcept
{
    m_shutdown = true;
    dev::frameGrabber<ocam2KCtrl>::appShutdown();
    releaseOcamSdk();
}

inline void ocam2KCtrl::releaseOcamSdk()
{
    if( m_ocam2Initialized )
    {
        ocam2_exit( m_ocam2_id );
        m_ocam2Initialized = false;
    }

    m_ocam2_id = 0;
}

inline void ocam2KCtrl::setupConfig()
{
    STDCAMERA_SETUP_CONFIG( config );

    dev::edtCamera<ocam2KCtrl>::setupConfig( config );

    config.add( "camera.ocamDescrambleFile",
                "",
                "camera.ocamDescrambleFile",
                argType::Required,
                "camera",
                "ocamDescrambleFile",
                false,
                "string",
                "The path of the OCAM descramble file, relative to MagAOX/config." );

    dev::frameGrabber<ocam2KCtrl>::setupConfig( config );

    config.add( "framegrabber.syncShmimName",
                "",
                "framegrabber.syncShmimName",
                argType::Required,
                "framegrabber",
                "syncShmimName",
                false,
                "string",
                "The name of the 1x1 uint8 ImageStreamIO sync stream used only for frame metadata and semaphores. "
                "Defaults to framegrabber.shmimName + \"_sync\"." );

    for( const auto &name : { "tempIntervalSec",
                              "tempChangingIntervalSec",
                              "confirmationIntervalSec",
                              "confirmationTimeoutSec",
                              "tempSettleSamples" } )
    {
        config.add( std::string( "housekeeping." ) + name,
                    "",
                    std::string( "housekeeping." ) + name,
                    argType::Required,
                    "housekeeping",
                    name,
                    false,
                    std::string( name ) == "tempSettleSamples" ? "unsigned" : "double",
                    "Housekeeping polling interval/deadline in seconds, or consecutive settling sample count." );
    }

    dev::dssShutter<ocam2KCtrl>::setupConfig( config );

    dev::telemeter<ocam2KCtrl>::setupConfig( config );
}

inline void ocam2KCtrl::loadConfig()
{
    if( loadConfigImpl() < 0 )
    {
        log<software_critical>( { __FILE__, __LINE__, "Error loading configuration" } );
        m_shutdown = true;
    }
}

inline int ocam2KCtrl::loadConfigImpl()
{
    STDCAMERA_LOAD_CONFIG( config );
    dev::edtCamera<ocam2KCtrl>::loadConfig( config );

    config( m_ocamDescrambleFile, "camera.ocamDescrambleFile" );

    if( m_maxEMGain < 1 )
    {
        m_maxEMGain = 1;
        log<text_log>( "maxEMGain set to 1" );
    }

    if( m_maxEMGain > 600 )
    {
        m_maxEMGain = 600;
        log<text_log>( "maxEMGain set to 600" );
    }

    dev::frameGrabber<ocam2KCtrl>::loadConfig( config );
    config( m_syncShmimName, "framegrabber.syncShmimName" );
    if( m_syncShmimName == "" )
    {
        m_syncShmimName = m_shmimName + "_sync";
    }
    config( m_tempIntervalSec, "housekeeping.tempIntervalSec" );
    config( m_tempChangingIntervalSec, "housekeeping.tempChangingIntervalSec" );
    config( m_confirmationIntervalSec, "housekeeping.confirmationIntervalSec" );
    config( m_confirmationTimeoutSec, "housekeeping.confirmationTimeoutSec" );
    double settleSamples = m_tempSettleSamples;
    config( settleSamples, "housekeeping.tempSettleSamples" );

    if( !mx::math::isFinite( settleSamples ) || settleSamples < 1 ||
        settleSamples > std::numeric_limits<unsigned>::max() || std::floor( settleSamples ) != settleSamples )
    {
        return log<software_error, -1>( { "housekeeping.tempSettleSamples must be a positive integer" } );
    }

    m_tempSettleSamples = settleSamples;

    if( validateHousekeepingConfig() < 0 )
    {
        return -1;
    }

    dev::dssShutter<ocam2KCtrl>::loadConfig( config );
    dev::telemeter<ocam2KCtrl>::loadConfig( config );
    return 0;
}

inline int ocam2KCtrl::ensureSyncStream()
{
    if( m_syncImageStream != nullptr )
    {
        uint32_t syncWidth  = m_syncImageStream->md[0].size[0];
        uint32_t syncHeight = 1;
        uint32_t syncDepth  = 1;

        if( m_syncImageStream->md[0].naxis > 1 )
        {
            syncHeight = m_syncImageStream->md[0].size[1];
        }

        if( m_syncImageStream->md[0].naxis > 2 )
        {
            syncDepth = m_syncImageStream->md[0].size[2];
        }

        if( m_syncImageStream->md[0].datatype == c_syncStreamDataType && syncWidth == c_syncStreamWidth &&
            syncHeight == c_syncStreamHeight && syncDepth == c_syncStreamDepth )
        {
            return 0;
        }

        ImageStreamIO_destroyIm( m_syncImageStream );
        free( m_syncImageStream );
        m_syncImageStream = nullptr;
    }

    char syncFileName[1024];
    ImageStreamIO_filename( syncFileName, sizeof( syncFileName ), m_syncShmimName.c_str() );

    int syncFd = ::open( syncFileName, O_RDWR );
    if( syncFd != -1 )
    {
        ::close( syncFd );

        IMAGE staleSyncStream;
        if( ImageStreamIO_openIm( &staleSyncStream, m_syncShmimName.c_str() ) != IMAGESTREAMIO_SUCCESS )
        {
            return log<software_error, -1>(
                { __FILE__, __LINE__, "error opening existing sync ImageStreamIO stream" } );
        }

        if( ImageStreamIO_destroyIm( &staleSyncStream ) != IMAGESTREAMIO_SUCCESS )
        {
            return log<software_error, -1>(
                { __FILE__, __LINE__, "error destroying existing sync ImageStreamIO stream" } );
        }
    }

    m_syncImageStream = reinterpret_cast<IMAGE *>( malloc( sizeof( IMAGE ) ) );
    if( m_syncImageStream == nullptr )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "error allocating sync ImageStreamIO stream" } );
    }

    uint32_t imsize[3] = { c_syncStreamWidth, c_syncStreamHeight, c_syncStreamDepth };

    if( ImageStreamIO_createIm_gpu( m_syncImageStream,
                                    m_syncShmimName.c_str(),
                                    3,
                                    imsize,
                                    c_syncStreamDataType,
                                    -1,
                                    1,
                                    IMAGE_NB_SEMAPHORE,
                                    0,
                                    CIRCULAR_BUFFER | ZAXIS_TEMPORAL,
                                    0 ) != IMAGESTREAMIO_SUCCESS )
    {
        free( m_syncImageStream );
        m_syncImageStream = nullptr;
        return log<software_error, -1>( { __FILE__, __LINE__, "error creating sync ImageStreamIO stream" } );
    }

    m_syncImageStream->array.UI8[0]      = 0;
    m_syncImageStream->md[0].cnt0        = 0;
    m_syncImageStream->md[0].cnt1        = 0;
    m_syncImageStream->md[0].write       = 0;
    m_syncImageStream->writetimearray[0] = m_syncImageStream->md[0].writetime;
    m_syncImageStream->atimearray[0]     = m_syncImageStream->md[0].atime;
    m_syncImageStream->cntarray[0]       = 0;

    return 0;
}

inline int ocam2KCtrl::destroySyncStream()
{
    if( m_syncImageStream == nullptr )
    {
        return 0;
    }

    int rv = 0;

    if( ImageStreamIO_destroyIm( m_syncImageStream ) != IMAGESTREAMIO_SUCCESS )
    {
        rv = log<software_error, -1>( { __FILE__, __LINE__, "error destroying sync ImageStreamIO stream" } );
    }

    free( m_syncImageStream );
    m_syncImageStream = nullptr;

    return rv;
}

inline int64_t ocam2KCtrl::housekeepingNow() const
{
    timespec stamp{};
    static_cast<void>( mx::sys::get_curr_time<double, CLOCK_MONOTONIC>( stamp ) );

    return int64_t( stamp.tv_sec ) * 1000000000 + stamp.tv_nsec;
}

inline void ocam2KCtrl::beginConfirmation( Confirmation &confirmation, double expected, bool requested )
{
    ++confirmation.generation;
    confirmation.phase     = ConfirmationPhase::confirming;
    confirmation.expected  = expected;
    confirmation.requested = requested;
    confirmation.due       = housekeepingNow();
    confirmation.deadline  = confirmation.due + int64_t( m_confirmationTimeoutSec * 1e9 );
}

inline void ocam2KCtrl::confirmValue( Confirmation &confirmation, double value, double tolerance )
{
    const int64_t now = housekeepingNow();

    if( confirmation.phase == ConfirmationPhase::failed )
    {
        return;
    }

    if( confirmation.phase == ConfirmationPhase::confirming && now >= confirmation.deadline )
    {
        confirmation.phase = ConfirmationPhase::failed;
        return;
    }

    if( !confirmation.requested || std::abs( value - confirmation.expected ) <= tolerance )
    {
        confirmation.phase    = ConfirmationPhase::confirmed;
        confirmation.deadline = 0;
    }
}

inline void ocam2KCtrl::resetHousekeeping()
{
    std::lock_guard<std::recursive_mutex> guard( m_cameraMutex );

    for( auto *confirmation : { &m_gainConfirmation, &m_fpsConfirmation, &m_tempConfirmation } )
    {
        ++confirmation->generation;
        confirmation->phase     = ConfirmationPhase::unavailable;
        confirmation->requested = false;
        confirmation->deadline  = 0;
    }

    m_housekeepingTempTime = 0;
    m_housekeepingGainTime = 0;
    m_housekeepingFpsTime  = 0;
    m_syncFreqTime         = 0;
    m_tempDue              = 0;
    m_tempChanging         = true;
    m_tempSettledSamples   = 0;
    m_tempExpectedControl  = -1;
    m_tempExpectedSet      = false;
    m_coolingKnown         = false;
}

inline int ocam2KCtrl::validateHousekeepingConfig()
{
    for( double interval :
         { m_tempIntervalSec, m_tempChangingIntervalSec, m_confirmationIntervalSec, m_confirmationTimeoutSec } )
    {
        if( !mx::math::isFinite( interval ) || interval <= 0 || interval > std::numeric_limits<int64_t>::max() / 4e9 )
        {
            return log<software_error, -1>(
                { "Housekeeping intervals must be finite, positive and representable as monotonic nanoseconds" } );
        }
    }

    if( m_tempChangingIntervalSec > m_tempIntervalSec || m_confirmationTimeoutSec < m_confirmationIntervalSec ||
        m_tempSettleSamples == 0 )
    {
        return log<software_error, -1>( { "Invalid housekeeping interval ordering or settling sample count" } );
    }

    return 0;
}

inline int ocam2KCtrl::initHousekeeping()
{
    if( createStandardIndiToggleSw( m_indiP_housekeepingPause, "housekeeping_pause" ) < 0 ||
        registerIndiPropertyNew( m_indiP_housekeepingPause, INDI_NEWCALLBACK( m_indiP_housekeepingPause ) ) < 0 )
    {
        return -1;
    }

    REG_INDI_NEWPROP_NOCB( m_indiP_housekeeping, "housekeeping", pcf::IndiProperty::Number );

    for( const auto &name : { "timeoutSec",
                              "remainingSec",
                              "tempAgeSec",
                              "gainAgeSec",
                              "fpsAgeSec",
                              "skippedTemp",
                              "skippedGain",
                              "skippedFps",
                              "serialTransactions",
                              "pauseStartMonoSec",
                              "pauseEndMonoSec",
                              "tempIntervalSec",
                              "tempNextSec",
                              "gainDeadlineSec",
                              "fpsDeadlineSec",
                              "tempDeadlineSec",
                              "tempQueries",
                              "gainQueries",
                              "fpsQueries" } )
    {
        m_indiP_housekeeping.add( pcf::IndiElement( name ) );
        m_indiP_housekeeping[name].set( 0 );
    }

    m_indiP_housekeeping["timeoutSec"].set( c_housekeepingPauseNs * 1e-9 );
    m_indiP_housekeeping["tempAgeSec"].set( -1 );
    m_indiP_housekeeping["gainAgeSec"].set( -1 );
    m_indiP_housekeeping["fpsAgeSec"].set( -1 );

    REG_INDI_NEWPROP_NOCB( m_indiP_housekeepingStatus, "housekeeping_status", pcf::IndiProperty::Text );

    for( const auto &name :
         { "state", "reason", "temperature", "gain", "fps", "protectionPolling", "tempConfirmation" } )
    {
        m_indiP_housekeepingStatus.add( pcf::IndiElement( name ) );
    }

    m_indiP_housekeepingStatus["state"].set( "inactive" );
    m_indiP_housekeepingStatus["reason"].set( m_housekeepingReason );
    m_indiP_housekeepingStatus["temperature"].set( "unavailable" );
    m_indiP_housekeepingStatus["gain"].set( "unavailable" );
    m_indiP_housekeepingStatus["fps"].set( "unavailable" );
    m_indiP_housekeepingStatus["protectionPolling"].set( "confirmation only" );
    m_indiP_housekeepingStatus["tempConfirmation"].set( "unavailable" );

    return 0;
}

inline void ocam2KCtrl::endHousekeepingPause( const std::string &reason )
{
    std::lock_guard<std::recursive_mutex> guard( m_cameraMutex );

    if( m_housekeepingPauseUntil == 0 )
    {
        return;
    }

    m_tempDue                = 0;
    m_housekeepingPauseUntil = 0;
    m_housekeepingPauseEnded = housekeepingNow();
    m_housekeepingReason     = reason;
    log<text_log>( "housekeeping pause ended at monotonic ns " + std::to_string( m_housekeepingPauseEnded ) + ": " +
                   reason );
}

inline bool ocam2KCtrl::housekeepingPaused( int64_t now )
{
    std::lock_guard<std::recursive_mutex> guard( m_cameraMutex );

    if( m_housekeepingPauseUntil == 0 )
    {
        return false;
    }

    if( shutdown() || powerState() != 1 || powerStateTarget() != 1 || m_reconfig || !m_nextMode.empty() ||
        ( state() != stateCodes::READY && state() != stateCodes::OPERATING ) )
    {
        endHousekeepingPause( "camera state changed" );
    }
    else if( now <= 0 || now >= m_housekeepingPauseUntil )
    {
        endHousekeepingPause( "120 second timeout" );
    }

    return m_housekeepingPauseUntil != 0;
}

inline int ocam2KCtrl::setHousekeepingPause( bool pause, int64_t now )
{
    std::lock_guard<std::recursive_mutex> guard( m_cameraMutex );

    if( !pause )
    {
        endHousekeepingPause( "operator resume" );
        return 0;
    }

    if( shutdown() || powerState() != 1 || powerStateTarget() != 1 || m_reconfig || !m_nextMode.empty() ||
        ( state() != stateCodes::READY && state() != stateCodes::OPERATING ) || now <= 0 )
    {
        endHousekeepingPause( "camera state changed" );
        return -1;
    }

    if( housekeepingPaused( now ) )
    {
        return 0;
    }

    const auto pending = []( const Confirmation &confirmation /**< [in] Latest confirmation state. */ )
    {
        return confirmation.phase == ConfirmationPhase::unavailable ||
               confirmation.phase == ConfirmationPhase::confirming;
    };

    if( pending( m_gainConfirmation ) || pending( m_fpsConfirmation ) || pending( m_tempConfirmation ) ||
        m_tempChanging )
    {
        m_housekeepingReason = "confirmation or thermal transition pending";
        return -1;
    }

    m_housekeepingPauseStarted = now;
    m_housekeepingPauseUntil   = now + c_housekeepingPauseNs;
    m_housekeepingReason       = "operator diagnostic pause";
    log<text_log>( "housekeeping pause started at monotonic ns " + std::to_string( now ) +
                   " for 120 seconds; temperature and software protection polling paused" );

    return 0;
}

inline int ocam2KCtrl::pdvSerialWriteRead( std::string &response, const std::string &command, bool logErrors )
{
    std::lock_guard<std::recursive_mutex> guard( m_cameraMutex );
    endHousekeepingPause( "serial command: " + command );
    ++m_housekeepingSerialTransactions;

    return dev::edtCamera<ocam2KCtrl>::pdvSerialWriteRead( response, command, logErrors );
}

inline int ocam2KCtrl::pollHousekeeping()
{
    std::lock_guard<std::recursive_mutex> guard( m_cameraMutex );
    const int64_t                         now    = housekeepingNow();
    const bool                            paused = housekeepingPaused( now );

    if( m_protectionResetConfirmed > 0 && mx::sys::get_curr_time() - m_protectionResetReqTime > 10.0 )
    {
        m_protectionResetConfirmed = 0;
        updateIfChanged( m_indiP_emProt, "status", std::string( "UNCONFIRMED" ) );
        log<text_log>( "protection reset request not confirmed", logPrio::LOG_NOTICE );
    }

    if( m_synchro )
    {
        getFPS(); // External source bookkeeping requires no serial query.
    }

    if( paused )
    {
        ++m_housekeepingSkippedTemp;
        return 0;
    }

    // Initial observations never imply writing default targets to the camera.
    if( m_tempConfirmation.phase == ConfirmationPhase::unavailable && m_tempDue == 0 )
    {
        m_tempDue = now;
    }

    for( auto *confirmation : { &m_gainConfirmation, &m_fpsConfirmation, &m_tempConfirmation } )
    {
        if( confirmation->phase == ConfirmationPhase::unavailable && !m_reconfig && m_nextMode.empty() )
        {
            beginConfirmation( *confirmation, 0, false );
            confirmation->due = now;
        }

        if( confirmation->phase == ConfirmationPhase::confirming && now >= confirmation->deadline )
        {
            confirmation->phase = ConfirmationPhase::failed;

            if( confirmation == &m_tempConfirmation )
            {
                m_tempChanging = false;
                m_tempDue      = now + int64_t( m_tempIntervalSec * 1e9 );
            }
        }
    }

    if( m_reconfig || !m_nextMode.empty() )
    {
        return 0;
    }

    // Oldest due wins. Iteration order breaks an initial tie in favor of confirmation.
    int           operation = -1;
    int64_t       oldest    = now + 1;
    const int64_t dues[]    = {
        m_gainConfirmation.phase == ConfirmationPhase::confirming ? m_gainConfirmation.due : now + 1,
        !m_synchro && m_fpsConfirmation.phase == ConfirmationPhase::confirming ? m_fpsConfirmation.due : now + 1,
        m_tempDue };

    for( int index = 0; index < 3; ++index )
    {
        if( dues[index] <= now && ( operation == -1 || dues[index] < oldest ) )
        {
            operation = index;
            oldest    = dues[index];
        }
    }

    int result = 0;

    if( operation == 0 )
    {
        result                 = getEMGain();
        m_gainConfirmation.due = housekeepingNow() + int64_t( m_confirmationIntervalSec * 1e9 );
    }
    else if( operation == 1 )
    {
        result                = getFPS();
        m_fpsConfirmation.due = housekeepingNow() + int64_t( m_confirmationIntervalSec * 1e9 );
    }
    else if( operation == 2 )
    {
        result                = getTemps();
        const double interval = m_tempConfirmation.phase == ConfirmationPhase::confirming
                                    ? m_confirmationIntervalSec
                                    : ( m_tempChanging ? m_tempChangingIntervalSec : m_tempIntervalSec );
        m_tempDue             = housekeepingNow() + int64_t( interval * 1e9 );
    }

    const int64_t completed = housekeepingNow();

    for( auto *confirmation : { &m_gainConfirmation, &m_fpsConfirmation, &m_tempConfirmation } )
    {
        if( confirmation->phase == ConfirmationPhase::confirming && completed >= confirmation->deadline )
        {
            confirmation->phase = ConfirmationPhase::failed;

            if( confirmation == &m_tempConfirmation )
            {
                m_tempChanging = false;
                m_tempDue      = completed + int64_t( m_tempIntervalSec * 1e9 );
            }
        }
    }

    if( result < 0 && powerState() == 1 && powerStateTarget() == 1 )
    {
        state( stateCodes::ERROR );
    }

    return result;
}

inline void ocam2KCtrl::updateHousekeeping()
{
    std::lock_guard<std::recursive_mutex> guard( m_cameraMutex );
    const int64_t                         now    = housekeepingNow();
    const bool                            paused = housekeepingPaused( now );

    if( !m_indiP_housekeepingPause.find( "toggle" ) )
    {
        return;
    }

    const auto age = [now]( int64_t stamp /**< [in] Last valid observation, ns. */ )
    {
        return stamp > 0 ? std::max( int64_t( 0 ), now - stamp ) * 1e-9 : -1.0;
    };
    const auto remaining = [now]( int64_t deadline /**< [in] Monotonic due time, ns. */ )
    {
        return deadline > 0 ? std::max( int64_t( 0 ), deadline - now ) * 1e-9 : 0.0;
    };
    const auto phase = []( const Confirmation &confirmation /**< [in] Latest request result. */ )
    {
        switch( confirmation.phase )
        {
        case ConfirmationPhase::confirming:
            return std::string( "confirming" );
        case ConfirmationPhase::confirmed:
            return std::string( "confirmed" );
        case ConfirmationPhase::failed:
            return std::string( "unconfirmed/error" );
        default:
            return std::string( "unavailable" );
        }
    };
    const bool active = !shutdown() && powerState() == 1 && powerStateTarget() == 1 && !m_reconfig &&
                        m_nextMode.empty() && ( state() == stateCodes::READY || state() == stateCodes::OPERATING );
    const double      interval = m_tempConfirmation.phase == ConfirmationPhase::confirming
                                     ? m_confirmationIntervalSec
                                     : ( m_tempChanging ? m_tempChangingIntervalSec : m_tempIntervalSec );
    const int64_t     grace    = int64_t( m_loopPause ) + int64_t( m_readTimeout ) * 1000000;
    const std::string validity =
        m_housekeepingTempTime == 0
            ? "unavailable"
            : ( paused || !active || m_housekeepingTempTime < m_housekeepingPauseEnded || now > m_tempDue + grace
                    ? "stale"
                    : "current" );
    const std::string state = paused ? "paused" : ( active ? ( m_tempChanging ? "changing" : "idle" ) : "inactive" );

    updateSwitchIfChanged( m_indiP_housekeepingPause,
                           "toggle",
                           paused ? pcf::IndiElement::On : pcf::IndiElement::Off,
                           paused ? INDI_BUSY : INDI_IDLE );
    updateIfChanged<double>(
        m_indiP_housekeeping,
        { "remainingSec",
          "tempAgeSec",
          "gainAgeSec",
          "fpsAgeSec",
          "skippedTemp",
          "skippedGain",
          "skippedFps",
          "serialTransactions",
          "pauseStartMonoSec",
          "pauseEndMonoSec",
          "tempIntervalSec",
          "tempNextSec",
          "gainDeadlineSec",
          "fpsDeadlineSec",
          "tempDeadlineSec",
          "tempQueries",
          "gainQueries",
          "fpsQueries" },
        { paused ? remaining( m_housekeepingPauseUntil ) : 0.0,
          age( m_housekeepingTempTime ),
          age( m_housekeepingGainTime ),
          age( m_housekeepingFpsTime ),
          double( m_housekeepingSkippedTemp ),
          double( m_housekeepingSkippedGain ),
          double( m_housekeepingSkippedFps ),
          double( m_housekeepingSerialTransactions ),
          m_housekeepingPauseStarted * 1e-9,
          m_housekeepingPauseEnded * 1e-9,
          interval,
          remaining( m_tempDue ),
          m_gainConfirmation.phase == ConfirmationPhase::confirming ? remaining( m_gainConfirmation.deadline ) : 0.0,
          m_fpsConfirmation.phase == ConfirmationPhase::confirming ? remaining( m_fpsConfirmation.deadline ) : 0.0,
          m_tempConfirmation.phase == ConfirmationPhase::confirming ? remaining( m_tempConfirmation.deadline ) : 0.0,
          double( m_housekeepingTempQueries ),
          double( m_housekeepingGainQueries ),
          double( m_housekeepingFpsQueries ) } );
    updateIfChanged<std::string>(
        m_indiP_housekeepingStatus,
        { "state", "reason", "temperature", "gain", "fps", "protectionPolling", "tempConfirmation" },
        { state,
          m_housekeepingReason,
          validity,
          phase( m_gainConfirmation ),
          phase( m_fpsConfirmation ),
          paused ? "paused" : ( active ? "confirmation only" : "inactive" ),
          phase( m_tempConfirmation ) } );
}

inline int ocam2KCtrl::appStartup()
{
    REG_INDI_NEWPROP_NOCB( m_indiP_temps, "temps", pcf::IndiProperty::Number );
    m_indiP_temps.add( pcf::IndiElement( "cpu" ) );
    m_indiP_temps["cpu"].set( 0 );
    m_indiP_temps.add( pcf::IndiElement( "power" ) );
    m_indiP_temps["power"].set( 0 );
    m_indiP_temps.add( pcf::IndiElement( "bias" ) );
    m_indiP_temps["bias"].set( 0 );
    m_indiP_temps.add( pcf::IndiElement( "water" ) );
    m_indiP_temps["water"].set( 0 );
    m_indiP_temps.add( pcf::IndiElement( "left" ) );
    m_indiP_temps["left"].set( 0 );
    m_indiP_temps.add( pcf::IndiElement( "right" ) );
    m_indiP_temps["right"].set( 0 );
    m_indiP_temps.add( pcf::IndiElement( "cooling" ) );
    m_indiP_temps["cooling"].set( 0 );

    REG_INDI_NEWPROP_NOCB( m_indiP_emProt, "emProtection", pcf::IndiProperty::Text );
    m_indiP_emProt.add( pcf::IndiElement( "status" ) );
    m_indiP_emProt["status"].set( "UNKNOWN" );
    m_indiP_emProt.setState( INDI_IDLE );

    createStandardIndiRequestSw( m_indiP_emProtReset, "emProtectionReset", "Reset", "EM Protection" );
    registerIndiPropertyNew( m_indiP_emProtReset, INDI_NEWCALLBACK( m_indiP_emProtReset ) );

    if( initHousekeeping() < 0 )
    {
        return -1;
    }

    REG_INDI_SETPROP( m_indiP_syncFreq, m_syncDevice, m_syncFreqProp );

    if( dev::stdCamera<ocam2KCtrl>::appStartup() < 0 )
    {
        return log<software_critical, -1>( { "" } );
    }

    if( dev::edtCamera<ocam2KCtrl>::appStartup() < 0 )
    {
        return log<software_critical, -1>( { "" } );
    }

    if( dev::frameGrabber<ocam2KCtrl>::appStartup() < 0 )
    {
        return log<software_critical, -1>( { "" } );
    }

    if( dev::dssShutter<ocam2KCtrl>::appStartup() < 0 )
    {
        return log<software_critical, -1>( { "" } );
    }

    m_temps.setInvalid();
    if( dev::telemeter<ocam2KCtrl>::appStartup() < 0 )
    {
        return log<software_error, -1>( { "" } );
    }

    return 0;
}

inline int ocam2KCtrl::appLogic()
{
    { //mutex scope
        std::unique_lock<std::mutex> lock( m_indiMutex, std::try_to_lock );

        if( lock.owns_lock() && ( state() != stateCodes::READY && state() != stateCodes::OPERATING ) )
        {
            updateHousekeeping();
        }
    }

    // and run stdCamera's appLogic
    if( dev::stdCamera<ocam2KCtrl>::appLogic() < 0 )
    {
        endHousekeepingPause( "stdCamera logic failed" );
        return log<software_error, -1>( { "" } );
    }

    // and run edtCamera's appLogic
    if( dev::edtCamera<ocam2KCtrl>::appLogic() < 0 )
    {
        endHousekeepingPause( "edtCamera logic failed" );
        return log<software_error, -1>( { "" } );
    }

    // first run frameGrabber's appLogic to see if the f.g. thread has exited.
    if( dev::frameGrabber<ocam2KCtrl>::appLogic() < 0 )
    {
        endHousekeepingPause( "frameGrabber logic failed" );
        return log<software_error, -1>( { "" } );
    }

    // and run dssShutter's appLogic
    if( dev::dssShutter<ocam2KCtrl>::appLogic() < 0 )
    {
        endHousekeepingPause( "dssShutter logic failed" );
        return log<software_error, -1>( { "" } );
    }

    if( state() == stateCodes::POWERON )
    {
        return 0;
    }

    if( state() == stateCodes::NOTCONNECTED || state() == stateCodes::ERROR )
    {
        m_temps.setInvalid();

        std::string response;

        // Might have gotten here because of a power off.
        if( MagAOXAppT::m_powerState == 0 )
        {
            return 0;
        }

        int ret = 0;
        { //mutex scope
            std::lock_guard<std::recursive_mutex> guard( m_cameraMutex );
            ret = pdvSerialWriteRead( response, "fps" ); // m_pdv, "fps", m_readTimeout);
        }
        if( ret == 0 )
        {
            resetHousekeeping();
            state( stateCodes::CONNECTED );
        }
        else
        {
            sleep( 1 );
            return 0;
        }
    }

    if( state() == stateCodes::CONNECTED )
    {
        // Get a lock
        std::unique_lock<std::mutex> lock( m_indiMutex );

        if( getFPS() == 0 && m_housekeepingFpsTime > 0 )
        {
            if( m_fpsSet == 0 )
            {
                state( stateCodes::READY );
            }
            else
            {
                state( stateCodes::OPERATING );
            }

            if( m_poweredOn && m_ccdTempSetpt > -999 )
            {
                m_poweredOn = false;
                if( setTempSetPt() < 0 )
                {
                    if( powerState() != 1 || powerStateTarget() != 1 )
                    {
                        return 0;
                    }
                    return log<software_error, 0>( { "" } );
                }
            }

            // We always have to set synchro on connecting, b/c otherwise we don't know the state.
            m_synchroSet = false;
            if( setSynchro() != 0 )
            {
                log<software_error>( { "error from setSynchro on CONNECT" } );
            }
        }
        else
        {
            if( powerState() != 1 || powerStateTarget() != 1 )
            {
                return 0;
            }
            state( stateCodes::ERROR );
            return log<software_error, 0>( { "" } );
        }
    }

    if( state() == stateCodes::READY || state() == stateCodes::OPERATING )
    {
        // Get a lock if we can
        std::unique_lock<std::mutex> lock( m_indiMutex, std::try_to_lock );

        // but don't wait for it, just go back around.
        if( !lock.owns_lock() )
        {
            return 0;
        }

        static_cast<void>( pollHousekeeping() );

        if( frameGrabber<ocam2KCtrl>::updateINDI() < 0 )
        {
            log<software_error>( { "" } );
            state( stateCodes::ERROR );
            return 0;
        }

        if( stdCamera<ocam2KCtrl>::updateINDI() < 0 )
        {
            log<software_error>( { "" } );
            state( stateCodes::ERROR );
            return 0;
        }

        if( edtCamera<ocam2KCtrl>::updateINDI() < 0 )
        {
            log<software_error>( { "" } );
            state( stateCodes::ERROR );
            return 0;
        }

        updateHousekeeping();

        if( telemeter<ocam2KCtrl>::appLogic() < 0 )
        {
            log<software_error>( { "" } );
            return 0;
        }
    }

    ///\todo Fall through check?

    return 0;
}

inline int ocam2KCtrl::onPowerOff()
{
    m_powerOnCounter = 0;

    std::lock_guard<std::mutex> lock( m_indiMutex );

    { //mutex scope
        std::lock_guard<std::recursive_mutex> guard( m_cameraMutex );
        endHousekeepingPause( "camera powered off" );
        resetHousekeeping();
        updateHousekeeping();
    }

    updateIfChanged( m_indiP_emProt, "status", std::string( "UNKNOWN" ), INDI_IDLE );

    m_temps.setInvalid();

    updateIfChanged( m_indiP_temps, "cpu", m_temps.CPU );
    updateIfChanged( m_indiP_temps, "power", m_temps.POWER );
    updateIfChanged( m_indiP_temps, "bias", m_temps.BIAS );
    updateIfChanged( m_indiP_temps, "water", m_temps.WATER );
    updateIfChanged( m_indiP_temps, "left", m_temps.LEFT );
    updateIfChanged( m_indiP_temps, "right", m_temps.RIGHT );
    updateIfChanged( m_indiP_temps, "cooling", m_temps.COOLING_POWER );

    if( stdCamera<ocam2KCtrl>::onPowerOff() < 0 )
    {
        log<software_error>( { "" } );
    }

    if( edtCamera<ocam2KCtrl>::onPowerOff() < 0 )
    {
        log<software_error>( { "" } );
    }

    if( frameGrabber<ocam2KCtrl>::onPowerOff() < 0 )
    {
        log<software_error>( { "" } );
    }

    if( dssShutter<ocam2KCtrl>::onPowerOff() < 0 )
    {
        log<software_error>( { "" } );
    }

    // Setting m_poweredOn
    m_poweredOn = true;

    return 0;
}

inline int ocam2KCtrl::whilePowerOff()
{
    std::lock_guard<std::mutex> lock( m_indiMutex );

    if( stdCamera<ocam2KCtrl>::whilePowerOff() < 0 )
    {
        log<software_error>( { "" } );
    }

    if( edtCamera<ocam2KCtrl>::whilePowerOff() < 0 )
    {
        log<software_error>( { "" } );
    }

    if( frameGrabber<ocam2KCtrl>::updateTiming() < 0 || frameGrabber<ocam2KCtrl>::updateINDI() < 0 )
    {
        log<software_error>( { "" } );
    }

    if( dssShutter<ocam2KCtrl>::whilePowerOff() < 0 )
    {
        log<software_error>( { "" } );
    }

    return 0;
}

inline int ocam2KCtrl::appShutdown()
{
    { //mutex scope
        std::lock_guard<std::mutex> lock( m_indiMutex );
        endHousekeepingPause( "application shutdown" );
        updateHousekeeping();
    }

    ///\todo error check these base class fxns.

    dev::stdCamera<ocam2KCtrl>::appShutdown();

    dev::edtCamera<ocam2KCtrl>::appShutdown();

    dev::frameGrabber<ocam2KCtrl>::appShutdown();

    releaseOcamSdk();

    destroySyncStream();

    dev::dssShutter<ocam2KCtrl>::appShutdown();

    dev::telemeter<ocam2KCtrl>::appShutdown();

    return 0;
}

inline void ocam2KCtrl::acceptTemps( const ocamTemps &temps )
{
    m_temps                = temps;
    m_ccdTemp              = temps.CCD;
    m_housekeepingTempTime = housekeepingNow();
    m_coolingKnown         = temps.COOLING_POWER >= 5 || ( temps.COOLING_POWER < 5 && temps.CCD - temps.SET > 2.99 );

    if( m_coolingKnown )
    {
        m_tempControlStatus = temps.COOLING_POWER >= 5;
    }

    else if( m_tempConfirmation.phase == ConfirmationPhase::confirmed )
    {
        beginConfirmation( m_tempConfirmation, 0, false );
    }

    // A reported SET is an observation, not a replacement for an outstanding user target.
    if( !m_tempExpectedSet && m_tempConfirmation.phase != ConfirmationPhase::failed )
    {
        m_ccdTempSetpt = temps.SET;
    }

    if( m_tempConfirmation.phase == ConfirmationPhase::confirming && housekeepingNow() >= m_tempConfirmation.deadline )
    {
        m_tempConfirmation.phase = ConfirmationPhase::failed;
    }

    if( m_tempConfirmation.phase != ConfirmationPhase::failed &&
        ( !m_tempExpectedSet || std::abs( temps.SET - m_tempConfirmation.expected ) < 0.05 ) && m_coolingKnown &&
        ( m_tempExpectedControl < 0 || m_tempControlStatus == bool( m_tempExpectedControl ) ) )
    {
        m_tempConfirmation.phase    = ConfirmationPhase::confirmed;
        m_tempConfirmation.deadline = 0;
        m_tempExpectedSet           = false;
        m_tempExpectedControl       = -1;
    }

    m_tempControlOnTarget = m_coolingKnown && m_tempControlStatus && std::abs( temps.CCD - temps.SET ) < 1.0;
    m_tempControlStatusStr =
        !m_coolingKnown
            ? "UNKNOWN"
            : ( !m_tempControlStatus ? "TEMP OFF" : ( m_tempControlOnTarget ? "ON TARGET" : "OFF TARGET" ) );

    if( m_tempConfirmation.phase == ConfirmationPhase::failed )
    {
        m_tempChanging       = false;
        m_tempSettledSamples = 0;
    }
    else if( m_tempConfirmation.phase != ConfirmationPhase::confirmed || !m_coolingKnown ||
             ( m_tempControlStatus && !m_tempControlOnTarget ) )
    {
        m_tempChanging       = true;
        m_tempSettledSamples = 0;
    }
    else if( !m_tempControlStatus )
    {
        m_tempChanging       = false;
        m_tempSettledSamples = 0;
    }
    else
    {
        if( m_tempSettledSamples < m_tempSettleSamples )
        {
            ++m_tempSettledSamples;
        }

        m_tempChanging = m_tempSettledSamples < m_tempSettleSamples;
    }

    const double interval = m_tempConfirmation.phase == ConfirmationPhase::confirming
                                ? m_confirmationIntervalSec
                                : ( m_tempChanging ? m_tempChangingIntervalSec : m_tempIntervalSec );
    m_tempDue             = housekeepingNow() + int64_t( interval * 1e9 );

    recordTemps();
    recordCamera();
    updateIfChanged( m_indiP_temps, "cpu", temps.CPU );
    updateIfChanged( m_indiP_temps, "power", temps.POWER );
    updateIfChanged( m_indiP_temps, "bias", temps.BIAS );
    updateIfChanged( m_indiP_temps, "water", temps.WATER );
    updateIfChanged( m_indiP_temps, "left", temps.LEFT );
    updateIfChanged( m_indiP_temps, "right", temps.RIGHT );
    updateIfChanged( m_indiP_temps, "cooling", temps.COOLING_POWER );
}

inline int ocam2KCtrl::getTemps()
{
    std::lock_guard<std::recursive_mutex> guard( m_cameraMutex );
    std::string                           response;
    m_housekeepingTempTime = 0;
    ++m_housekeepingTempQueries;

    if( pdvSerialWriteRead( response, "temp" ) != 0 )
    {
        return -1;
    }

    ocamTemps temps;

    if( parseTemps( temps, response ) < 0 )
    {
        m_temps.setInvalid();
        m_ccdTemp              = m_temps.CCD;
        m_coolingKnown         = false;
        m_tempControlOnTarget  = false;
        m_tempControlStatusStr = "UNKNOWN";
        m_tempSettledSamples   = 0;

        if( m_tempConfirmation.phase == ConfirmationPhase::confirmed ||
            m_tempConfirmation.phase == ConfirmationPhase::unavailable )
        {
            beginConfirmation( m_tempConfirmation, 0, false );
        }

        m_tempChanging = m_tempConfirmation.phase != ConfirmationPhase::failed;
        recordTemps();
        recordCamera();

        // Keep the target and confirmation deadline; malformed reports cannot confirm a request.
        return log<software_error, 0>( { "Temperature parse error: " + response } );
    }

    acceptTemps( temps );
    return 0;
}

inline int ocam2KCtrl::powerOnDefaults()
{
    // Camera boots up with this true in most cases.
    m_tempControlStatusSet = false;
    m_tempControlStatus    = false;

    return 0;
}

inline int ocam2KCtrl::setTempControl()
{
    std::lock_guard<std::recursive_mutex> guard( m_cameraMutex );
    const bool                            requested = m_tempControlStatusSet;
    const float                           target    = m_ccdTempSetpt;
    endHousekeepingPause( "cooling command" );

    if( !requested )
    {
        const int result       = getTemps();
        m_tempControlStatusSet = requested;
        m_ccdTempSetpt         = target;

        if( result < 0 || m_housekeepingTempTime == 0 )
        {
            return log<text_log, -1>( "Fresh temperature required before turning cooling off", logPrio::LOG_ERROR );
        }

        if( m_ccdTemp <= 19 )
        {
            return log<text_log, -1>( "Can not turn temp control off when not at 20 C or higher", logPrio::LOG_ERROR );
        }
    }

    std::string       response;
    const std::string command = requested ? "temp on" : "temp off";
    beginConfirmation( m_tempConfirmation, target, true );
    m_tempExpectedControl = requested ? 1 : 0;
    m_tempExpectedSet     = false;
    m_tempChanging        = true;
    m_tempSettledSamples  = 0;
    m_tempDue             = housekeepingNow();

    if( pdvSerialWriteRead( response, command ) != 0 )
    {
        m_tempConfirmation.phase = ConfirmationPhase::failed;
        m_tempChanging           = false;
        return -1;
    }

    ocamTemps temps;

    if( parseTemps( temps, response ) == 0 )
    {
        acceptTemps( temps );
    }

    if( requested && target > -999 )
    {
        m_ccdTempSetpt = target;

        // The setpoint is the final command; retain the cooling-state requirement in that generation.
        return setTempSetPt();
    }

    recordCamera();
    return 0;
}

inline int ocam2KCtrl::setTempSetPt()
{
    std::lock_guard<std::recursive_mutex> guard( m_cameraMutex );

    if( stdCameraT::validateTempSetPt( m_ccdTempSetpt ) < 0 )
    {
        return -1;
    }

    const int expectedControl = m_tempConfirmation.phase == ConfirmationPhase::confirming ? m_tempExpectedControl : -1;
    const std::string tempStr = std::to_string( m_ccdTempSetpt );
    std::string       response;
    beginConfirmation( m_tempConfirmation, m_ccdTempSetpt, true );
    m_tempExpectedSet     = true;
    m_tempExpectedControl = expectedControl;
    m_tempChanging        = true;
    m_tempSettledSamples  = 0;
    m_tempDue             = housekeepingNow();

    if( pdvSerialWriteRead( response, "temp " + tempStr ) != 0 )
    {
        m_tempConfirmation.phase = ConfirmationPhase::failed;
        m_tempChanging           = false;
        return -1;
    }

    ocamTemps temps;

    if( parseTemps( temps, response ) == 0 )
    {
        acceptTemps( temps );
    }

    recordCamera();
    return log<text_log, 0>( { "set temperature: " + tempStr } );
}

inline int ocam2KCtrl::getFPS()
{
    std::lock_guard<std::recursive_mutex> guard( m_cameraMutex );

    if( m_synchro )
    {
        m_fps = m_syncFreq;
        m_workerFPS.store( m_fps, std::memory_order_relaxed );
        recordCamera();
        return 0;
    }

    std::string response;
    m_housekeepingFpsTime = 0;
    ++m_housekeepingFpsQueries;

    if( pdvSerialWriteRead( response, "fps" ) != 0 )
    {
        return -1;
    }

    float fps;

    if( parseFPS( fps, response ) < 0 )
    {
        return log<software_error, 0>( { "FPS parse error: " + response } );
    }

    m_fps = fps;
    m_workerFPS.store( fps, std::memory_order_relaxed );
    m_housekeepingFpsTime = housekeepingNow();
    // Zero requests automatic/max rate; any valid positive reported rate establishes its result.
    const bool automatic = m_fpsConfirmation.requested && m_fpsConfirmation.expected == 0;
    confirmValue(
        m_fpsConfirmation, automatic ? 0 : fps, 4 * std::numeric_limits<float>::epsilon() * std::max( 1.0f, fps ) );
    recordCamera();
    return 0;
}

inline int ocam2KCtrl::setFPS( bool requestReconfiguration )
{
    std::lock_guard<std::recursive_mutex> guard( m_cameraMutex );

    if( !mx::math::isFinite( m_fpsSet ) || m_fpsSet < 0 )
    {
        return log<software_error, -1>( { "FPS target must be finite and nonnegative" } );
    }

    endHousekeepingPause( "FPS command" );
    const std::string fpsStr = std::to_string( m_fpsSet );
    const bool        reuse  = m_synchro && m_fpsConfirmation.phase == ConfirmationPhase::confirmed &&
                       m_fpsConfirmation.requested && m_fpsConfirmation.expected == m_fpsSet && m_syncFreqTime > 0 &&
                       m_syncFreq == m_fpsSet;
    beginConfirmation( m_fpsConfirmation, m_fpsSet, true );

    if( !m_synchro )
    {
        std::string response;

        if( pdvSerialWriteRead( response, "fps " + fpsStr ) != 0 )
        {
            m_fpsConfirmation.phase = ConfirmationPhase::failed;
            return -1;
        }

        float fps;

        if( parseFPS( fps, response ) == 0 )
        {
            m_fps = fps;
            m_workerFPS.store( fps, std::memory_order_relaxed );
            m_housekeepingFpsTime = housekeepingNow();
            confirmValue( m_fpsConfirmation,
                          m_fpsSet == 0 ? 0 : fps,
                          4 * std::numeric_limits<float>::epsilon() * std::max( 1.0f, fps ) );
        }

        if( requestReconfiguration )
        {
            if( m_nextMode.empty() )
            {
                m_nextMode = m_modeName;
            }

            m_reconfig = true;
        }
    }
    else
    {
        pcf::IndiProperty ipFreq( pcf::IndiProperty::Number );
        ipFreq.setDevice( m_syncDevice );
        ipFreq.setName( m_syncFreqProp );
        ipFreq.add( pcf::IndiElement( "target", fpsStr ) );
        if( sendNewProperty( ipFreq ) < 0 )
        {
            m_fpsConfirmation.phase = ConfirmationPhase::failed;
            return -1;
        }

        if( reuse )
        {
            confirmValue( m_fpsConfirmation, m_syncFreq, 0 );
        }
        else if( m_indiDriver )
        {
            // One fresh report request for this generation, including unchanged source targets.
            m_indiDriver->sendGetProperties( m_indiP_syncFreq );
        }
    }

    return 0;
}

inline int ocam2KCtrl::setSynchro( bool requestReconfiguration )
{
    std::string response;

    { //mutex scope
        std::lock_guard<std::recursive_mutex> guard( m_cameraMutex );

        // Invalidate a superseded camera rate before the intermediate maximum-rate command.
        if( !m_synchroSet || !m_synchro )
        {
            ++m_fpsConfirmation.generation;
            m_fpsConfirmation.phase = ConfirmationPhase::unavailable;
        }

        // First set the actual FPS to 0 to get to max
        std::string fpsStr = std::to_string( 0 );
        if( pdvSerialWriteRead( response, "fps " + fpsStr ) == 0 )
        {
            ///\todo check response
            std::cerr << "fps " << fpsStr << " response: " << response << "\n";
            log<text_log>( { "set fps: " + fpsStr } );
        }
        else
        {
            m_fpsConfirmation.phase = ConfirmationPhase::failed;

            if( powerState() != 1 || powerStateTarget() != 1 )
            {
                return -1;
            }
            return log<software_error, -1>( { "" } );
        }

        // Now actually turn synchro on
        std::string sStr;
        if( m_synchroSet )
        {
            sStr = "on";
        }
        else
        {
            sStr = "off";
        }
        if( pdvSerialWriteRead( response, "synchro " + sStr ) == 0 )
        {
            ///\todo check response
            std::cerr << "synchro " << sStr << " resonse: " << response << "\n";
            log<text_log>( { "set synchro: " + sStr } );

            m_synchro = m_synchroSet;

            if( m_synchro == false )
            {
                updateSwitchIfChanged( m_indiP_synchro, "toggle", pcf::IndiElement::Off, INDI_IDLE );
            }
            else
            {
                updateSwitchIfChanged( m_indiP_synchro, "toggle", pcf::IndiElement::On, INDI_OK );
            }
        }
        else
        {
            m_fpsConfirmation.phase = ConfirmationPhase::failed;

            if( powerState() != 1 || powerStateTarget() != 1 )
            {
                return -1;
            }
            return log<software_error, -1>( { "" } );
        }
    }

    // Finally we set the FPS of the synchro device
    return setFPS( requestReconfiguration );
}

inline int ocam2KCtrl::setExpTime()
{
    return 0;
}

inline int ocam2KCtrl::setNextROI()
{
    return 0;
}

inline int ocam2KCtrl::setShutter( int sh )
{
    return dssShutter<ocam2KCtrl>::setShutterState( sh );
}

inline std::string ocam2KCtrl::stateString()
{
    std::string ss;

    ss += m_modeName + "_";
    ss += std::to_string( m_fps ) + "_";
    ss += std::to_string( m_emGain ) + "_";
    ss += std::to_string( m_ccdTempSetpt );

    return ss;
}

inline bool ocam2KCtrl::stateStringValid()
{
    if( state() == stateCodes::OPERATING && m_tempControlOnTarget )
    {
        return true;
    }
    else
    {
        return false;
    }
}

inline int ocam2KCtrl::resetEMProtection()
{
    std::lock_guard<std::recursive_mutex> guard( m_cameraMutex );
    std::string                           response;

    if( pdvSerialWriteRead( response, "protection reset" ) != 0 )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
        {
            return -1;
        }

        return log<software_error, -1>( { "" } );
    }

    std::cerr << "\n******************************************\n";
    std::cerr << "protection reset:\n";
    std::cerr << response << "\n";
    std::cerr << "\n******************************************\n";
    ///\todo check response.

    updateIfChanged( m_indiP_emProt, "status", std::string( "RESET" ), INDI_OK );
    log<text_log>( "overillumination protection has been reset", logPrio::LOG_NOTICE );
    m_protectionResetConfirmed = 0;
    m_protectionReset          = true;
    beginConfirmation( m_gainConfirmation, 0, false );
    return 0;
}

inline int ocam2KCtrl::getEMGain()
{
    std::lock_guard<std::recursive_mutex> guard( m_cameraMutex );
    std::string                           response;
    m_housekeepingGainTime = 0;
    ++m_housekeepingGainQueries;

    if( pdvSerialWriteRead( response, "gain" ) != 0 )
    {
        return -1;
    }

    unsigned gain;

    if( parseEMGain( gain, response ) < 0 )
    {
        if( response.find( "HV" ) != std::string::npos )
        {
            m_emGain                 = 1;
            m_gainConfirmation.phase = ConfirmationPhase::failed;
            updateIfChanged( m_indiP_emProt, "status", std::string( "TRIPPED" ), INDI_ALERT );
            return log<software_warning, -1>( { "EM Gain tripped!" } );
        }

        // A bad numeric reply is retryable within the original fixed deadline.
        return log<software_error, 0>( { "EM Gain parse error: " + response } );
    }

    m_emGain               = gain;
    m_housekeepingGainTime = housekeepingNow();
    confirmValue( m_gainConfirmation, gain, 0 );
    return 0;
}

inline int ocam2KCtrl::setEMGain()
{
    std::lock_guard<std::recursive_mutex> guard( m_cameraMutex );

    if( !mx::math::isFinite( m_emGainSet ) || m_emGainSet < 1 || m_emGainSet >= double( m_maxEMGain ) + 1 )
    {
        return log<text_log, 0>( "EM gain outside limits refused", logPrio::LOG_WARNING );
    }

    const unsigned gain = m_emGainSet; // Preserve the existing integer command for fractional targets.

    if( !m_protectionReset && m_emGainSet > 1 )
    {
        return log<text_log, 0>( "Attempt to set EM gain before protection reset", logPrio::LOG_NOTICE );
    }

    std::string response;
    beginConfirmation( m_gainConfirmation, gain, true );

    if( pdvSerialWriteRead( response, "gain " + std::to_string( gain ) ) != 0 )
    {
        m_gainConfirmation.phase = ConfirmationPhase::failed;
        return -1;
    }

    unsigned actual;

    if( parseEMGain( actual, response ) == 0 )
    {
        m_emGain               = actual;
        m_housekeepingGainTime = housekeepingNow();
        confirmValue( m_gainConfirmation, actual, 0 );
    }
    else if( response.find( "HV" ) != std::string::npos )
    {
        m_emGain                 = 1;
        m_gainConfirmation.phase = ConfirmationPhase::failed;
        updateIfChanged( m_indiP_emProt, "status", std::string( "TRIPPED" ), INDI_ALERT );
        return log<software_warning, -1>( { "EM Gain tripped!" } );
    }

    return log<text_log, 0>( { "set EM Gain: " + std::to_string( gain ) } );
}

inline int ocam2KCtrl::configureAcquisition()
{
    // lock mutex
    std::unique_lock<std::mutex>          lock( m_indiMutex );
    std::lock_guard<std::recursive_mutex> cameraGuard( m_cameraMutex );

    ++m_fpsConfirmation.generation;
    m_fpsConfirmation.phase = ConfirmationPhase::unavailable;

    // Send command to camera to place it in the correct mode
    std::string response;
    if( pdvSerialWriteRead( response, m_cameraModes[m_modeName].m_serialCommand ) != 0 )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
        {
            return -1;
        }

        log<software_error>( { "Error sending command to set mode" } );
        sleep( 1 );
        return -1;
    }

    m_currentROI.x     = 119.5;
    m_currentROI.y     = 119.5;
    m_currentROI.w     = 240;
    m_currentROI.h     = 240;
    m_currentROI.bin_x = m_cameraModes[m_modeName].m_binningX;
    m_currentROI.bin_y = m_cameraModes[m_modeName].m_binningY;

    m_digitalBinX = m_cameraModes[m_modeName].m_digitalBinX;
    m_digitalBinY = m_cameraModes[m_modeName].m_digitalBinY;

    if( m_digitalBinX > 1 )
    {
        m_digitalBin = true;
        std::cerr << "digital binning!\n";
    }
    else
    {
        m_digitalBin = false;
    }

    recordCamera();

    ///\todo check response of pdvSerialWriteRead
    log<text_log>( "camera configured with: " + m_cameraModes[m_modeName].m_serialCommand );

    if( m_fpsSet > 0 )
    {
        setFPS( false );
    }

    log<text_log>( "Send command to set mode: " + m_cameraModes[m_modeName].m_serialCommand );
    log<text_log>( "Response was: " + response );

    if( setSynchro( false ) < 0 )
    {
        log<software_error>( { "Error setting synchro during configureAcquisition" } );
    }

    /* Initialize the OCAM2 SDK
     */

    releaseOcamSdk();
    ocam2_rc   rc;
    ocam2_mode mode;

    int OCAM_SZw;
    int OCAM_SZh;
    if( m_raw_height == 121 )
    {
        mode     = OCAM2_NORMAL;
        OCAM_SZw = 240;
        OCAM_SZh = 240;
    }
    else if( m_raw_height == 62 )
    {
        mode     = OCAM2_BINNING;
        OCAM_SZw = 120;
        OCAM_SZh = 120;
    }
    else if( m_raw_height == 41 )
    {
        mode     = OCAM2_BINNING1x3;
        OCAM_SZw = 240;
        OCAM_SZh = 80;
    }
    else if( m_raw_height == 31 )
    {
        mode     = OCAM2_BINNING1x4;
        OCAM_SZw = 240;
        OCAM_SZh = 60;
    }
    else
    {
        log<text_log>( "Unrecognized OCAM2 mode.", logPrio::LOG_ERROR );
        return -1;
    }

    std::string ocamDescrambleFile = m_configDir + "/" + m_ocamDescrambleFile;

    std::cerr << "ocamDescrambleFile: " << ocamDescrambleFile << std::endl;

    ocam2_id nextOcam2Id = 0;

    rc = ocam2_init( mode, ocamDescrambleFile.c_str(), &nextOcam2Id );

    if( rc != OCAM2_OK )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
        {
            return -1;
        }
        log<text_log>( "ocam2_init error. Failed to initialize OCAM SDK with descramble file: " + ocamDescrambleFile,
                       logPrio::LOG_ERROR );
        return -1;
    }

    m_ocam2_id         = nextOcam2Id;
    m_ocam2Initialized = true;

    log<text_log>( "OCAM2K initialized. id: " + std::to_string( m_ocam2_id ) );

    log<text_log>( std::string( "OCAM2K mode is:" ) + ocam2_modeStr( ocam2_getMode( m_ocam2_id ) ) );

    if( m_digitalBin )
    {
        std::cerr << "and digital binning!\n";
        m_digitalBinWork.resize( OCAM_SZw, OCAM_SZh );

        m_width  = OCAM_SZw / m_digitalBinX;
        m_height = OCAM_SZh / m_digitalBinY;
    }
    else
    {
        m_width  = OCAM_SZw;
        m_height = OCAM_SZh;
    }

    m_dataType = _DATATYPE_UINT16;

    if( ensureSyncStream() < 0 )
    {
        releaseOcamSdk();
        return log<software_error, -1>( { "Error preparing sync ImageStreamIO stream" } );
    }

    m_fgTiming.context( static_cast<uint32_t>( mode ), m_digitalBinX, m_digitalBinY );
    state( stateCodes::OPERATING );

    return 0;
}

inline float ocam2KCtrl::fps()
{
    return m_workerFPS.load( std::memory_order_relaxed );
}

inline int ocam2KCtrl::startAcquisition()
{
    m_lastImageNumber = -1;
    return edtCamera<ocam2KCtrl>::pdvStartAcquisition();
}

inline int ocam2KCtrl::acquireAndCheckValid()
{
    if( edtCamera<ocam2KCtrl>::pdvAcquire( m_currImageTimestamp, true ) < 0 )
    {
        edtCamera<ocam2KCtrl>::pdvFinishAcquisition( false );

        if( powerState() != 1 || powerStateTarget() != 1 )
        {
            return -1;
        }

        m_lastImageNumber = -1;
        requestAcquisitionRecovery();
        return -1;
    }

    // Get the image number to see if this is valid.
    // This is how it is in the ocam2_sdk:
    unsigned currImageNumber = ( reinterpret_cast<int *>( m_image_p ) )[OCAM2_IMAGE_NB_OFFSET / 4]; /* int offset */
    m_currImageNumber        = currImageNumber;
    m_fgTiming.frame( currImageNumber );

    // For the first loop after a restart
    if( m_lastImageNumber == -1 )
    {
        m_lastImageNumber = m_currImageNumber - 1;
    }

    const unsigned delta = currImageNumber - static_cast<unsigned>( m_lastImageNumber );

    if( delta != 1 )
    {
        edtCamera<ocam2KCtrl>::pdvFinishAcquisition( false );

        // Unsigned subtraction permits exactly UINT_MAX -> zero and rejects gaps across wrap.
        if( delta > 1 && delta < 100 )
        {
            // This we handle as a non-timeout -- report how many frames were skipped
            unsigned framesSkipped = delta - 1;

            log<text_log>( "frames skipped: " + std::to_string( framesSkipped ), logPrio::LOG_ERROR );

            m_lastImageNumber = -1;
            requestAcquisitionRecovery();
            return 1;
        }
        else // but if it's any bigger or < 0, it's probably garbage
        {
            if( powerState() != 1 || powerStateTarget() != 1 )
            {
                return -1;
            }

            ///\todo need frame corrupt log type
            log<text_log>( "frame number possibly corrupt: " + std::to_string( m_currImageNumber ) + " - " +
                               std::to_string( m_lastImageNumber ),
                           logPrio::LOG_ERROR );

            requestAcquisitionRecovery();

            // Reset the counters.
            m_lastImageNumber = -1;
            return 1;
        }
    }
    m_lastImageNumber = m_currImageNumber;
    return 0;
}

inline int ocam2KCtrl::loadImageIntoStream( void *dest )
{
    unsigned currImageNumber = 0;

    if( !m_digitalBin )
    {
        ocam2_descramble( m_ocam2_id,
                          &currImageNumber,
                          reinterpret_cast<int16_t *>( dest ),
                          reinterpret_cast<int16_t *>( m_image_p ) );
        // memcpy(dest, m_image_p, 120*120*2); //This is about 10 usec faster -- but we have to descramble.
    }
    else
    {
        ocam2_descramble(
            m_ocam2_id, &currImageNumber, m_digitalBinWork.data(), reinterpret_cast<int16_t *>( m_image_p ) );

        mx::improc::eigenMap<int16_t> destMap( reinterpret_cast<int16_t *>( dest ), m_width, m_height );

        if( m_digitalBinX > 1 )
        {
            for( int cc = 0; cc < destMap.cols(); ++cc )
            {
                for( int rr = 0; rr < destMap.rows(); ++rr )
                {
                    destMap( rr, cc ) = m_digitalBinWork( rr * m_digitalBinX + 0, cc );

                    for( unsigned b = 1; b < m_digitalBinX; ++b )
                    {
                        destMap( rr, cc ) = m_digitalBinWork( rr * m_digitalBinX + b, cc );
                    }
                }
            }
        }
    }

    return 0;
}

inline int ocam2KCtrl::reconfig()
{
    std::lock_guard<std::mutex>           lock( m_indiMutex );
    std::lock_guard<std::recursive_mutex> guard( m_cameraMutex );
    endHousekeepingPause( "camera reconfiguration" );
    ++m_fpsConfirmation.generation;
    m_fpsConfirmation.phase = ConfirmationPhase::unavailable;
    m_reconfig              = false;

    int rv = edtCamera<ocam2KCtrl>::pdvReconfig();

    if( rv < 0 )
    {
        m_reconfig = true;
        state( stateCodes::ERROR );
        return rv;
    }

    state( stateCodes::READY );
    return 0;
}

inline void ocam2KCtrl::frameGrabberReconfigureComplete()
{
    std::lock_guard<std::mutex> lock( m_indiMutex );

    if( m_nextMode.empty() )
    {
        m_reconfig = false;
    }
}

inline void ocam2KCtrl::requestAcquisitionRecovery()
{
    edtCamera<ocam2KCtrl>::pdvFinishAcquisition( false );

    std::lock_guard<std::mutex> lock( m_indiMutex );

    endHousekeepingPause( "acquisition recovery" );
    resetHousekeeping();

    if( m_nextMode.empty() )
    {
        m_nextMode = m_modeName;
    }

    m_reconfig = true;
}

inline int ocam2KCtrl::frameGrabberPostPublish( IMAGE *imageStream )
{
    if( imageStream == nullptr )
    {
        edtCamera<ocam2KCtrl>::pdvFinishAcquisition( false );
        return 0;
    }

    if( m_syncImageStream == nullptr )
    {
        edtCamera<ocam2KCtrl>::pdvFinishAcquisition( false );
        return log<software_error, -1>( { "Sync ImageStreamIO stream unavailable during publication" } );
    }

    m_syncImageStream->md[0].write       = 1;
    m_syncImageStream->array.UI8[0]      = 0;
    m_syncImageStream->md[0].writetime   = imageStream->md[0].writetime;
    m_syncImageStream->md[0].atime       = imageStream->md[0].atime;
    m_syncImageStream->md[0].cnt0        = imageStream->md[0].cnt0;
    m_syncImageStream->md[0].cnt1        = 0;
    m_syncImageStream->writetimearray[0] = imageStream->md[0].writetime;
    m_syncImageStream->atimearray[0]     = imageStream->md[0].atime;
    m_syncImageStream->cntarray[0]       = imageStream->md[0].cnt0;
    m_syncImageStream->md[0].write       = 0;

    ImageStreamIO_sempost( m_syncImageStream, -1 );

    edtCamera<ocam2KCtrl>::pdvFinishAcquisition( !m_reconfig && !shutdown() && powerState() > 0 );

    return 0;
}

INDI_NEWCALLBACK_DEFN( ocam2KCtrl, m_indiP_housekeepingPause )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_housekeepingPause, ipRecv );

    if( !ipRecv.find( "toggle" ) )
    {
        return 0;
    }

    std::lock_guard<std::mutex>           lock( m_indiMutex );
    std::lock_guard<std::recursive_mutex> guard( m_cameraMutex );
    const bool                            pause  = ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On;
    const int                             result = setHousekeepingPause( pause, housekeepingNow() );
    updateHousekeeping();

    return result;
}

INDI_NEWCALLBACK_DEFN( ocam2KCtrl, m_indiP_emProtReset )( const pcf::IndiProperty &ipRecv )
{
    if( MagAOXAppT::m_powerState == 0 )
    {
        return 0;
    }

    if( ipRecv.getName() != m_indiP_emProtReset.getName() )
    {
        log<software_error>( { "wrong INDI property received." } );
        return -1;
    }

    if( !ipRecv.find( "request" ) )
    {
        return 0;
    }

    if( ipRecv["request"].getSwitchState() == pcf::IndiElement::Off )
    {
        return 0;
    }

    std::unique_lock<std::mutex> lock( m_indiMutex );

    if( m_protectionResetConfirmed == 0 )
    {
        updateIfChanged( m_indiP_emProt, "status", std::string( "CONFIRM" ), INDI_BUSY );

        m_protectionResetConfirmed = 1;

        m_protectionResetReqTime = mx::sys::get_curr_time();

        log<text_log>( "protection reset requested", logPrio::LOG_NOTICE );

        return 0;
    }

    // If here, this is a confirmation.
    return resetEMProtection();
}

INDI_SETCALLBACK_DEFN( ocam2KCtrl, m_indiP_syncFreq )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( ipRecv, m_indiP_syncFreq )

    if( !ipRecv.find( "current" ) )
    {
        return -1;
    }

    std::lock_guard<std::mutex>           lock( m_indiMutex );
    std::lock_guard<std::recursive_mutex> guard( m_cameraMutex );
    const double                          frequency = ipRecv["current"].get<double>();

    if( !mx::math::isFinite( frequency ) || frequency <= 0 || frequency > std::numeric_limits<float>::max() )
    {
        return -1;
    }

    m_syncFreq     = frequency;
    m_syncFreqTime = housekeepingNow();

    if( m_synchro )
    {
        m_housekeepingFpsTime  = m_syncFreqTime;
        const double tolerance = 4 * std::numeric_limits<float>::epsilon() * std::max( 1.0f, m_syncFreq );

        if( m_fpsConfirmation.phase == ConfirmationPhase::confirmed && m_fpsConfirmation.requested &&
            std::abs( m_syncFreq - m_fpsConfirmation.expected ) > tolerance )
        {
            // An observed source change invalidates the old target before its required mode reload.
            ++m_fpsConfirmation.generation;
            m_fpsConfirmation.phase = ConfirmationPhase::unavailable;
        }

        confirmValue( m_fpsConfirmation, m_syncFreq, tolerance );
    }

    if( m_synchro && m_syncFreq != m_fps )
    {
        endHousekeepingPause( "synchronization frequency changed" );

        recordCamera( true );
        m_fps = m_syncFreq;
        m_workerFPS.store( m_fps, std::memory_order_relaxed );

        // We always want to reset the latency circular buffers
        if( m_nextMode.empty() )
        {
            m_nextMode = m_modeName;
        }

        m_reconfig = true;
    }

    return 0;
}

inline int ocam2KCtrl::checkRecordTimes()
{
    return telemeter<ocam2KCtrl>::checkRecordTimes( ocam_temps(), telem_stdcam(), telem_fgtimings(), telem_fgstage() );
}

inline int ocam2KCtrl::recordTelem( const ocam_temps * )
{
    return recordTemps( true );
}

inline int ocam2KCtrl::recordTelem( const telem_stdcam * )
{
    return recordCamera( true );
}

inline int ocam2KCtrl::recordTelem( const telem_fgtimings * )
{
    return recordFGTimings( true );
}

inline int ocam2KCtrl::recordTelem( const telem_fgstage * )
{
    return recordFGStageTimings( true );
}

inline int ocam2KCtrl::recordTemps( bool force )
{
    static ocamTemps lastTemps;

    if( !( lastTemps == m_temps ) || force )
    {
        telem<ocam_temps>( { m_temps.CCD,
                             m_temps.CPU,
                             m_temps.POWER,
                             m_temps.BIAS,
                             m_temps.WATER,
                             m_temps.LEFT,
                             m_temps.RIGHT,
                             m_temps.COOLING_POWER } );
        lastTemps = m_temps;
    }

    return 0;
}

} // namespace app
} // namespace MagAOX

#endif
