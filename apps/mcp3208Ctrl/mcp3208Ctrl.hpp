/** \file mcp3208Ctrl.hpp
 * \brief The MagAO-X mcp3208 Controller header file
 *
 * \ingroup mcp3208Ctrl_files
 */

#ifndef mcp3208Ctrl_hpp
#define mcp3208Ctrl_hpp

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"
#include "dependencies/MCP3208.h" // Included for the MCP3208 device interface

/** \defgroup mcp3208Ctrl
 * \brief The MagAO-X application to readout a mcp3208 A/D on a raspberry Pi.
 *
 * <a href="../handbook/operating/software/apps/XXXXXX.html">Application Documentation</a>
 *
 * \ingroup apps
 *
 */

/** \defgroup mcp3208Ctrl_files
 * \ingroup mcp3208Ctrl
 */

namespace MagAOX
{
namespace app
{

/** MagAO-X application to read MCP3208 channels on a Raspberry Pi.
 *
 * The controller can either acquire on its internal timer loop or synchronize reads to an
 * ImageStreamIO semaphore.
 *
 * \ingroup mcp3208Ctrl
 */
class mcp3208Ctrl : public MagAOXApp<true>, public dev::frameGrabber<mcp3208Ctrl>, public dev::telemeter<mcp3208Ctrl>
{

    // Give the test harness access.
    friend class mcp3208Ctrl_test;
    friend class dev::frameGrabber<mcp3208Ctrl>;
    friend class dev::telemeter<mcp3208Ctrl>;

    typedef dev::frameGrabber<mcp3208Ctrl> frameGrabberT;
    typedef dev::telemeter<mcp3208Ctrl>    telemeterT;

    /// The active MCP3208 device interface used for hardware access.
    MCP3208Lib::MCP3208 m_adc;

    static constexpr bool c_frameGrabber_flippable = false; /**< app:dev config indicating these images can not be
                                                                 flipped */

  protected:
    /** \name Configurable Parameters - Data
     * @{
     */

    int m_numChannels{ 2 }; ///< The number of MCP3208 channels read into each output frame.

    std::string m_fpsDevice;               ///< Device name providing external fps metadata for framegrabber sizing.
    std::string m_fpsProperty{ "fps" };    ///< Property name providing external fps metadata.
    std::string m_fpsElement{ "current" }; ///< Property element containing the fps value.

    float m_fpsTol{ 0 }; ///< The tolerance used when monitoring fps metadata changes.

    std::string m_numChannelsDevice;                      ///< Device name for getting numChannels to set circular buffer length.
    std::string m_numChannelsProperty{ "numChannels" };   ///< Property name for getting numChannels to set circular buffer length.
    std::string m_numChannelsElement{ "current" };        ///< Element name for getting numChannels to set circular buffer length.

    float m_numChannelsTol{ 0 }; ///< The tolerance for detecting a change in numChannels.

    std::string m_synchroShmimName; ///< The synchronization ImageStreamIO stream name; empty selects timer mode.
    int m_synchroPostDelay{ 0 }; ///< Signed microsecond phase offset added to synchronized delay-model predictions.

    double m_synchroDtTransfer_ns{ 3e3 }; ///< Transfer-latency term in the synchronized delay model.

    double m_synchroWfsProcess_ns{ 51.5e3 }; ///< WFS processing-latency term in the synchronized delay model.

    double m_synchroDtF_ns{ 10e3 }; ///< Filter/transport-latency term in the synchronized delay model.

    double m_synchroWfsRead_ns{ 276.1e3 }; ///< WFS read-latency term in the synchronized delay model.

    double m_delayLockAbsThreshold_ns{ 50e3 }; ///< Absolute phase-error threshold for declaring delay lock.

    double m_delayLockFracThreshold{ 0.1 }; ///< Fractional period phase-error threshold for declaring delay lock.

    double m_cadenceGuard_ns{ 20e3 }; ///< Reserved per-frame margin protecting synchronized cadence from overruns.

    float m_alpha{ 0.01f }; ///< Global exponential moving-average coefficient applied to timing smoothers.

    ///@}

    /** \name Runtime State - Data
     * @{
     */

    /// Report the configured channel count metadata to the framegrabber.
    int numChannels();

    // Creating INDI property for number of channels to read out
    pcf::IndiProperty m_indiP_numChannels;
    INDI_NEWCALLBACK_DECL( mcp3208Ctrl, m_indiP_numChannels );

    pcf::IndiProperty m_indiP_numChannelsSource;
    INDI_SETCALLBACK_DECL( mcp3208Ctrl, m_indiP_numChannelsSource );
    
    /// INDI property exposing the local fps target.
    pcf::IndiProperty m_indiP_fps;

    /// Handle updates to the local fps target property.
    INDI_NEWCALLBACK_DECL( mcp3208Ctrl, m_indiP_fps );

    float m_fps{ 2000 }; ///< The target acquisition rate in frames per second.

    /// INDI property subscription used to follow an external fps source.
    pcf::IndiProperty m_indiP_fpsSource;

    /// Handle updates from the configured external fps source.
    INDI_SETCALLBACK_DECL( mcp3208Ctrl, m_indiP_fpsSource );

    /// INDI property exposing the global EMA alpha used by timing smoothers.
    pcf::IndiProperty m_indiP_alpha;

    /// Handle updates to the global EMA alpha property.
    INDI_NEWCALLBACK_DECL( mcp3208Ctrl, m_indiP_alpha );

    /// INDI property exposing the synchronized-mode signed phase offset in microseconds.
    pcf::IndiProperty m_indiP_synchroDelay;

    /// Handle updates to the synchronized-mode signed phase offset property.
    INDI_NEWCALLBACK_DECL( mcp3208Ctrl, m_indiP_synchroDelay );

    /// INDI property exposing runtime timing diagnostics for acquisition health checks.
    pcf::IndiProperty m_indiP_timingDiag;

    float m_trigger{ 1e9f / m_fps };       ///< The timer-mode read interval in nanoseconds.
    float m_gain{ .1 };                    ///< The simple integrator gain used for timer and synchro delay control.
    float nano_sec_target{ 1e9f / m_fps }; ///< The timer-mode target interval in nanoseconds.
    float m_synchroDelay{ 0 };             ///< The commanded pre-read delay in synchronized mode, in nanoseconds.
    float m_synchroDelayTarget{ 0 };       ///< The synchronized-mode effective delay target in nanoseconds after applying signed offset and wrap.

    /// Secondary MCP3208 handle retained with the legacy class state.
    MCP3208Lib::MCP3208 adc;

    /// The timer-mode reference point for the next internal acquisition cycle.
    std::chrono::time_point<std::chrono::high_resolution_clock> m_time_start;

    /// The most recently read MCP3208 channel values published to the output stream.
    std::vector<uint16_t> m_values;

    IMAGE  m_synchroStream{};             ///< The opened synchronization stream used for semaphore-triggered reads.
    bool   m_synchroStreamOpen{ false };  ///< Tracks whether the synchronization stream is currently open.
    ino_t  m_synchroStreamInode{ 0 };     ///< Cached inode used to detect synchronization stream recreation.
    int    m_synchroSemaphoreNumber{ 5 }; ///< The claimed semaphore slot for synchronization waits.
    sem_t *m_synchroSemaphore{ nullptr }; ///< Cached pointer to the claimed synchronization semaphore.

    timespec m_atime{}; ///< The most recent semaphore-arrival timestamp on the local realtime clock.

    timespec m_lastAtime{}; ///< The previous semaphore-arrival timestamp used for period estimation.

    double m_avgSemaphorePeriod_ns{ 0.0 }; ///< Exponential moving-average estimate of semaphore period in nanoseconds.

    double m_wfsPeriodMeasured_ns{ 0.0 }; ///< Measured semaphore period used as the synchronized delay-model WFS period.

    timespec m_lastProducerAtime{}; ///< Previous producer atime from the synchronization stream metadata.

    uint64_t m_lastProducerCnt0{ 0 }; ///< Previous producer frame counter from the synchronization stream metadata.

    double m_producerPeriodInst_ns{ 0.0 }; ///< Instantaneous producer period estimate from metadata timestamps.

    double m_avgProducerPeriod_ns{ 0.0 }; ///< Exponential moving-average producer period estimate from metadata.

    bool m_firstProducerSample{ true }; ///< Tracks first-sample initialization for producer-period estimation.

    uint64_t m_localFrameSeq{ 0 }; ///< Monotonic local acquisition sequence incremented on each published frame.

    uint64_t m_syncFramesReceived{ 0 }; ///< Count of synchronized frames received from semaphore wakes.

    uint64_t m_syncFramesWritten{ 0 }; ///< Count of synchronized frames published to the output stream.

    uint64_t m_syncFramesDropped{ 0 }; ///< Count of missing producer frame IDs inferred from positive counter gaps.

    uint64_t m_syncFrameIdGapCount{ 0 }; ///< Number of producer-frame gap events where ID delta exceeded one.

    uint64_t m_syncProducerFrameId{ 0 }; ///< Latest producer frame ID (`cnt0`) observed on the synchronization stream.

    uint64_t m_syncProducerFrameDelta{ 0 }; ///< Latest producer-frame ID delta between consecutive synchronized wakes.

    uint64_t m_lastSyncProducerFrameId{ 0 }; ///< Previous producer frame ID used to detect synchronized frame-ID gaps.

    bool m_syncProducerFrameValid{ false }; ///< Tracks whether producer frame-ID gap tracking has a valid prior sample.

    bool m_firstSemaphore{ true }; ///< Tracks first-arrival initialization for semaphore period estimation.

    double m_avgReadLatency_ns{ 0.0 }; ///< Exponential moving-average estimate of semaphore-to-read latency in nanoseconds.

    bool m_firstReadLatency{ true }; ///< Tracks first-arrival initialization for semaphore-to-read latency estimation.

    double m_wfs_fps{ 0.0 }; ///< WFS frame rate estimate used for timing prediction; initialized from configured fps before callbacks.

    double m_delayModel_ns{ 0.0 }; ///< Current modulo-wrapped delay predicted by the synchronized phase model.

    double m_delayApplied_ns{ 0.0 }; ///< Delay applied to the most recent synchronized ADC read.

    double m_delayBudget_ns{ 0.0 }; ///< Maximum delay allowed this cycle after cadence budgeting.

    double m_nonDelayService_ns{ 0.0 }; ///< Measured wake-to-return service time minus applied delay for the latest cycle.

    double m_avgNonDelayService_ns{ 0.0 }; ///< Exponential moving-average of non-delay synchronized service time.

    bool m_firstNonDelayService{ true }; ///< Tracks first-sample initialization for non-delay service-time averaging.

    double m_delayPhaseError_ns{ 0.0 }; ///< Wrapped phase error between applied and modeled synchronized delay.

    double m_delayLock{ 0.0 }; ///< Delay-lock state exported to diagnostics as 1.0 (locked) or 0.0 (unlocked).

    double m_delayCapped{ 0.0 }; ///< Delay-cap state exported as 1.0 when cadence budgeting limits the applied delay.

    timespec m_triggerTime{}; ///< Computed trigger timestamp aligned to the estimated WFS integration midpoint.

    double m_triggerInterval_ns{ 0.0 }; ///< Measured interval between consecutive trigger events in nanoseconds.

    double m_channelReadoutTime_ns{ 0.0 }; ///< Measured duration of reading all configured channels in nanoseconds.

    timespec m_lastTriggerTime{}; ///< Previous trigger timestamp used to compute the synchronized trigger interval.

    bool m_firstTriggerTime{ true }; ///< Tracks first-trigger initialization for synchronized trigger-interval measurement.

    bool m_firstTimerTrigger{ true }; ///< Tracks first-trigger initialization for timer-mode trigger-interval measurement.

    ///@}

    /** \name Synchronization Helpers
     * @{
     */

    /// Open the synchronization stream when semaphore-driven acquisition is enabled.
    /**
     * \returns 0 on success.
     * \returns 1 when the synchronization stream is not yet ready.
     */
    int openSynchroStream();

    /// Claim a semaphore slot from the synchronization stream.
    /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
    int claimSynchroSemaphore();

    /// Check whether the synchronization stream has disappeared or been recreated.
    /**
     * \returns true when the current synchronization stream handle is stale.
     * \returns false when the synchronization stream still matches the cached inode.
     */
    bool synchroStreamStale();

    /// Release the synchronization semaphore claim and close the synchronization stream.
    void closeSynchroStream();

    /// Acquire one frame using the internal timer loop.
    /**
     * \returns 0 when a new sample is ready.
     * \returns 1 when the loop should continue without publishing.
     */
    int acquireTimerAndCheckValid();

    /// Acquire one frame using the synchronization semaphore.
    /**
     * \returns 0 when a new sample is ready.
     * \returns 1 when the loop should continue without publishing.
     * \returns -1 on a critical timing error.
     */
    int acquireSynchroAndCheckValid();

    /// Read the current realtime clock value.
    /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
    int getRealtime( timespec &ts /**< [out] the current realtime clock value */ );

    /// Wait on the claimed synchronization semaphore until the supplied timeout.
    /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
    int waitOnSemaphore( sem_t    *sem /**< [in] the semaphore to wait on */,
                         timespec &ts /**< [in] the absolute timeout for the wait */ );

    /// Convert a timespec timestamp to nanoseconds.
    static inline double timespecToNs( const timespec &t /**< [in] the timespec value to convert */ );

    /// Convert nanoseconds to a normalized timespec value.
    static inline timespec nsToTimespec( double ns /**< [in] the nanosecond value to convert */ );

    /// Read one MCP3208 channel value.
    /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
    int readChannelValue( int       channel /**< [in] the MCP3208 channel index to read */,
                          uint16_t &value /**< [out] the sampled channel value */ );

    /// Apply the current controlled delay between semaphore wake and ADC read.
    void delayBeforeRead();

    /// Update the synchronized delay command with anti-windup and physical bounds.
    void updateSynchroDelayController( double desiredDelay_ns /**< [in] the unconstrained synchronized-delay command */ );

    /// Update synchronized trigger timing from the current semaphore arrival.
    void updateTriggerTiming( const timespec &atime /**< [in] the semaphore-arrival timestamp */ );

    /// Publish acquisition timing diagnostics to the INDI read-only property.
    /** The exported `trigger_interval_us` value reports measured current-to-previous trigger interval.
     *
     * The exported `trigger_time_us` value is relative to the latest semaphore arrival (`m_atime`).
     */
    void updateTimingDiagnosticsIndi();

    ///@}

  public:
    /** \name Application Lifecycle
     * @{
     */

    /// Construct the application with the current repository version metadata.
    mcp3208Ctrl();

    /// Destroy the application.
    ~mcp3208Ctrl() noexcept
    {
    }

    /// Register configuration entries for the application and helper devices.
    virtual void setupConfig();

    /// Load configuration values into the application state.
    /** This helper is separated from `loadConfig()` to support unit testing.
     */
    int loadConfigImpl( mx::app::appConfigurator &_config /**< [in] the populated application configurator */ );

    /// Load the configured application state.
    virtual void loadConfig();

    /// Start the application and initialize its INDI and hardware state.
    /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
    virtual int appStartup();

    /// Execute one iteration of the application FSM.
    /**
     * \returns 0 on no critical error
     * \returns -1 on an error requiring shutdown
     */
    virtual int appLogic();

    /// Shutdown the application and release synchronization resources.
    /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
    virtual int appShutdown();

    ///@}

    /** \name Framegrabber Interface
     * @{
     */

    /// Configure the output image geometry and acquisition mode.
    /**
     * \returns 0 on success
     * \returns 1 when configuration should be retried
     */
    int configureAcquisition();

    /// Report the current acquisition rate metadata to the framegrabber.
    /**
     * \returns the current fps value.
     */
    float fps();

    /// Prepare the selected acquisition mode to begin producing samples.
    /**
     * \returns 0 on success
     * \returns -1 on error
     */
    int startAcquisition();

    /// Acquire one sample and indicate whether it should be published.
    /**
     * \returns 0 when a new sample is ready.
     * \returns 1 when no new sample should be published.
     * \returns -1 on a critical error.
     */
    int acquireAndCheckValid();

    /// Copy the current MCP3208 values into the output image buffer.
    /**
     * \returns 0 on success
     * \returns -1 on error
     */
    int loadImageIntoStream( void *dest /**< [out] the destination image buffer */ );

    /// Reset synchronization resources before the next acquisition configuration.
    /**
     * \returns 0 on success
     * \returns -1 on error
     */
    int reconfig();

    ///@}

    /** \name Telemeter Interface
     *
     * @{
     */
    /// Check whether framegrabber timing telemetry should be recorded this cycle.
    /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
    int checkRecordTimes();

    /// Record framegrabber timing telemetry.
    /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
    int recordTelem( const telem_fgtimings *telem /**< [in] the telemeter tag requested by the interface */ );

    ///@}
};

mcp3208Ctrl::mcp3208Ctrl() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    return;
}

inline double mcp3208Ctrl::timespecToNs( const timespec &t )
{
    return static_cast<double>( t.tv_sec ) * 1e9 + static_cast<double>( t.tv_nsec );
}

inline timespec mcp3208Ctrl::nsToTimespec( double ns )
{
    timespec t;

    t.tv_sec  = static_cast<time_t>( ns / 1e9 );
    t.tv_nsec = static_cast<long>( ns - static_cast<double>( t.tv_sec ) * 1e9 );

    if( t.tv_nsec >= 1000000000L )
    {
        t.tv_nsec -= 1000000000L;
        ++t.tv_sec;
    }

    if( t.tv_nsec < 0 )
    {
        t.tv_nsec += 1000000000L;
        --t.tv_sec;
    }

    return t;
}

void mcp3208Ctrl::updateTriggerTiming( const timespec &atime )
{
    double dt_ns = 0.0;

    if( !m_firstSemaphore )
    {
        dt_ns = timespecToNs( atime ) - timespecToNs( m_lastAtime );

        const double alpha = static_cast<double>( m_alpha );
        m_avgSemaphorePeriod_ns = alpha * dt_ns + ( 1.0 - alpha ) * m_avgSemaphorePeriod_ns;
    }
    else
    {
        m_avgSemaphorePeriod_ns = 0.0;
        m_firstSemaphore        = false;
    }

    m_lastAtime = atime;

    const double deltaT_wfs_ns = m_avgSemaphorePeriod_ns;
    m_wfsPeriodMeasured_ns     = deltaT_wfs_ns;

    if( deltaT_wfs_ns <= 0.0 )
    {
        m_triggerInterval_ns = 0.0;
        return;
    }

    const double raw_delay_ns =
        0.5 * deltaT_wfs_ns - ( m_synchroDtTransfer_ns + m_synchroWfsProcess_ns + m_synchroDtF_ns + m_synchroWfsRead_ns );

    double t_delay_model_ns = std::fmod( raw_delay_ns, deltaT_wfs_ns );

    if( t_delay_model_ns < 0.0 )
    {
        t_delay_model_ns += deltaT_wfs_ns;
    }

    const double delayOffset_ns = 1e3 * static_cast<double>( m_synchroPostDelay );
    double       t_delay_ns     = std::fmod( t_delay_model_ns + delayOffset_ns, deltaT_wfs_ns );
    if( t_delay_ns < 0.0 )
    {
        t_delay_ns += deltaT_wfs_ns;
    }

    m_delayModel_ns      = t_delay_model_ns;
    m_synchroDelayTarget = static_cast<float>( t_delay_ns );

    const double t_trigger_ns = timespecToNs( atime ) + t_delay_ns;
    m_triggerTime             = nsToTimespec( t_trigger_ns );

    if( m_firstTriggerTime )
    {
        m_triggerInterval_ns = 0.0;
        m_firstTriggerTime   = false;
    }
    else
    {
        m_triggerInterval_ns = timespecToNs( m_triggerTime ) - timespecToNs( m_lastTriggerTime );
        if( m_triggerInterval_ns < 0.0 )
        {
            m_triggerInterval_ns = 0.0;
        }
    }

    m_lastTriggerTime = m_triggerTime;
}

void mcp3208Ctrl::setupConfig()
{
    FRAMEGRABBER_SETUP_CONFIG( config );
    TELEMETER_SETUP_CONFIG( config );

    config.add( "fps.device",
                "",
                "fps.device",
                argType::Required,
                "fps",
                "device",
                false,
                "string",
                "Device name for getting fps to set circular buffer length." );

    config.add( "fps.property",
                "",
                "fps.property",
                argType::Required,
                "fps",
                "property",
                false,
                "string",
                "Property name for getting fps to set circular buffer length. Default is 'fps'." );

    config.add( "fps.element",
                "",
                "fps.element",
                argType::Required,
                "fps",
                "element",
                false,
                "string",
                "Property name for getting fps to set circular buffer length. Default is 'current'." );

    config.add( "fps.tol",
                "",
                "fps.tol",
                argType::Required,
                "fps",
                "tol",
                false,
                "float",
                "Tolerance for detecting a change in FPS.  Default is 0." );

    config.add( "synchro.shmimName",
                "",
                "synchro.shmimName",
                argType::Required,
                "synchro",
                "shmimName",
                false,
                "string",
                "The ImageStreamIO stream used to synchronize acquisition. Default is timer-driven operation." );

    config.add( "synchro.postDelay",
                "",
                "synchro.postDelay",
                argType::Required,
                "synchro",
                "postDelay",
                false,
                "int",
                "Signed phase offset in microseconds added to synchronized delay-model predictions. Default is 0." );

    config.add( "synchro.dtTransfer_ns",
                "",
                "synchro.dtTransfer_ns",
                argType::Required,
                "synchro",
                "dtTransfer_ns",
                false,
                "double",
                "Transfer-latency term in nanoseconds for synchronized delay modeling. Default is 3000." );

    config.add( "synchro.wfsProcess_ns",
                "",
                "synchro.wfsProcess_ns",
                argType::Required,
                "synchro",
                "wfsProcess_ns",
                false,
                "double",
                "WFS processing-latency term in nanoseconds for synchronized delay modeling. Default is 51500." );

    config.add( "synchro.dtF_ns",
                "",
                "synchro.dtF_ns",
                argType::Required,
                "synchro",
                "dtF_ns",
                false,
                "double",
                "Filter/transport-latency term in nanoseconds for synchronized delay modeling. Default is 10000." );

    config.add( "synchro.wfsRead_ns",
                "",
                "synchro.wfsRead_ns",
                argType::Required,
                "synchro",
                "wfsRead_ns",
                false,
                "double",
                "WFS read-latency term in nanoseconds for synchronized delay modeling. Default is 276100." );

    config.add( "synchro.delayLockAbsThreshold_ns",
                "",
                "synchro.delayLockAbsThreshold_ns",
                argType::Required,
                "synchro",
                "delayLockAbsThreshold_ns",
                false,
                "double",
                "Absolute synchronized phase-error threshold in nanoseconds for delay-lock diagnostics. Default is 50000." );

    config.add( "synchro.delayLockFracThreshold",
                "",
                "synchro.delayLockFracThreshold",
                argType::Required,
                "synchro",
                "delayLockFracThreshold",
                false,
                "double",
                "Fractional synchronized phase-error threshold for delay-lock diagnostics. Default is 0.1." );

    config.add( "synchro.cadenceGuard_ns",
                "",
                "synchro.cadenceGuard_ns",
                argType::Required,
                "synchro",
                "cadenceGuard_ns",
                false,
                "double",
                "Reserved nanoseconds in each synchronized cycle for non-delay work. Default is 20000." );

    config.add( "synchro.alpha",
                "",
                "synchro.alpha",
                argType::Required,
                "synchro",
                "alpha",
                false,
                "float",
                "Global EMA coefficient for synchronized timing smoothers. Default is 0.01." );

    config.add( "numChannels.device",
                "",
                "numChannels.device",
                argType::Required,
                "numChannels",
                "device",
                false,
                "int",
                "Setting the number of channels needed to readout accelerometers" );

    config.add( "numChannels.property",
                "",
                "numChannels.property",
                argType::Required,
                "numChannels",
                "property",
                false,
                "string",
                "Property name for getting numChannels to set circular buffer length. Default is 'numChannels'." );

    config.add( "numChannels.element",
                "",
                "numChannels.element",
                argType::Required,
                "numChannels",
                "element",
                false,
                "string",
                "Property name for getting numChannels to set circular buffer length. Default is 'current'." );

    config.add( "numChannels.tol",
                "",
                "numChannels.tol",
                argType::Required,
                "numChannels",
                "tol",
                false,
                "float",
                "Tolerance for detecting a change in numChannels.  Default is 0." );

    config.add( "framegrabber.cpuset",
                "",
                "framegrabber.cpuset",
                argType::Required,
                "framegrabber",
                "cpuset",
                false,
                "string",
                "The cpuset to assign the framegrabber thread to." );
}

int mcp3208Ctrl::loadConfigImpl( mx::app::appConfigurator &_config )
{

    FRAMEGRABBER_LOAD_CONFIG( _config );
    TELEMETER_LOAD_CONFIG( _config );

    _config( m_fpsDevice, "fps.device" );
    _config( m_fpsProperty, "fps.property" );
    _config( m_fpsElement, "fps.element" );
    _config( m_fpsTol, "fps.tol" );
    _config( m_synchroShmimName, "synchro.shmimName" );
    _config( m_synchroPostDelay, "synchro.postDelay" );
    _config( m_synchroDtTransfer_ns, "synchro.dtTransfer_ns" );
    _config( m_synchroWfsProcess_ns, "synchro.wfsProcess_ns" );
    _config( m_synchroDtF_ns, "synchro.dtF_ns" );
    _config( m_synchroWfsRead_ns, "synchro.wfsRead_ns" );
    _config( m_delayLockAbsThreshold_ns, "synchro.delayLockAbsThreshold_ns" );
    _config( m_delayLockFracThreshold, "synchro.delayLockFracThreshold" );
    _config( m_cadenceGuard_ns, "synchro.cadenceGuard_ns" );
    _config( m_alpha, "synchro.alpha" );

    _config( m_numChannels, "accel.numChannels" ); // making number of mcp3208 channels we read out configurable
    _config( m_numChannelsDevice, "numChannels.device" );
    _config( m_numChannelsProperty, "numChannels.property" );
    _config( m_numChannelsElement, "numChannels.element" );
    _config( m_numChannelsTol, "numChannels.tol" );

    _config(m_fgCpuset, "framegrabber.cpuset");

    if( m_synchroDtTransfer_ns < 0.0 )
    {
        m_synchroDtTransfer_ns = 0.0;
    }

    if( m_synchroWfsProcess_ns < 0.0 )
    {
        m_synchroWfsProcess_ns = 0.0;
    }

    if( m_synchroDtF_ns < 0.0 )
    {
        m_synchroDtF_ns = 0.0;
    }

    if( m_synchroWfsRead_ns < 0.0 )
    {
        m_synchroWfsRead_ns = 0.0;
    }

    if( m_delayLockAbsThreshold_ns < 0.0 )
    {
        m_delayLockAbsThreshold_ns = 0.0;
    }

    if( m_delayLockFracThreshold < 0.0 )
    {
        m_delayLockFracThreshold = 0.0;
    }

    if( m_cadenceGuard_ns < 0.0 )
    {
        m_cadenceGuard_ns = 0.0;
    }

    if( m_alpha < 0.0f )
    {
        m_alpha = 0.0f;
    }
    else if( m_alpha > 1.0f )
    {
        m_alpha = 1.0f;
    }

    m_synchroDelayTarget = 1e3f * m_synchroPostDelay;
    m_synchroDelay       = m_synchroDelayTarget;
    m_delayModel_ns      = 0.0;
    m_delayApplied_ns    = static_cast<double>( m_synchroDelayTarget );
    m_delayBudget_ns     = 0.0;
    m_nonDelayService_ns = 0.0;
    m_avgNonDelayService_ns = 0.0;
    m_firstNonDelayService  = true;
    m_delayPhaseError_ns = 0.0;
    m_delayLock          = 0.0;
    m_delayCapped        = 0.0;
    m_wfsPeriodMeasured_ns = 0.0;
    m_lastProducerAtime    = timespec{};
    m_lastProducerCnt0     = 0;
    m_producerPeriodInst_ns = 0.0;
    m_avgProducerPeriod_ns  = 0.0;
    m_firstProducerSample   = true;
    m_localFrameSeq         = 0;
    m_syncFramesReceived    = 0;
    m_syncFramesWritten     = 0;
    m_syncFramesDropped     = 0;
    m_syncFrameIdGapCount   = 0;
    m_syncProducerFrameId   = 0;
    m_syncProducerFrameDelta = 0;
    m_lastSyncProducerFrameId = 0;
    m_syncProducerFrameValid  = false;
    m_wfs_fps            = m_fps;
    m_channelReadoutTime_ns = 0.0;

    return 0;
}

void mcp3208Ctrl::loadConfig()
{
    loadConfigImpl( config );
}

int mcp3208Ctrl::appStartup()
{
    FRAMEGRABBER_APP_STARTUP;
    TELEMETER_APP_STARTUP;

    // INDI prop for user to set fps
    CREATE_REG_INDI_NEW_NUMBERF( m_indiP_fps, "fps", 0, 10000, 1, "%d", "", "" );
    m_indiP_fps["current"].setValue( m_fps );
    m_indiP_fps["target"].setValue( m_fps );

    // INDI prop for user to set global timing EMA alpha
    CREATE_REG_INDI_NEW_NUMBERF( m_indiP_alpha, "alpha", 0.0, 1.0, 0.00001, "%.5f", "", "" );
    m_indiP_alpha["current"].setValue( m_alpha );
    m_indiP_alpha["target"].setValue( m_alpha );

    // INDI prop for user to set number of channels A/D reads out
    CREATE_REG_INDI_NEW_NUMBERF( m_indiP_numChannels, "numChannels", 0, 8, 1, "%d", "", "" );
    m_indiP_numChannels["current"].setValue( m_numChannels );
    m_indiP_numChannels["target"].setValue( m_numChannels );

    // INDI prop for user to set synchronized-mode signed phase offset in microseconds
    CREATE_REG_INDI_NEW_NUMBERF( m_indiP_synchroDelay, "synchroDelay", -1000000, 1000000, 1, "%d", "us", "" );
    m_indiP_synchroDelay["current"].setValue( m_synchroPostDelay );
    m_indiP_synchroDelay["target"].setValue( m_synchroPostDelay );

    CREATE_REG_INDI_RO_NUMBER( m_indiP_timingDiag, "timingDiag", "Timing Diagnostics", "Diagnostics" );
    m_indiP_timingDiag.add( pcf::IndiElement( "avg_read_latency_us" ) );
    m_indiP_timingDiag.add( pcf::IndiElement( "synchro_delay_us" ) );
    m_indiP_timingDiag.add( pcf::IndiElement( "synchro_delay_target_us" ) );
    m_indiP_timingDiag.add( pcf::IndiElement( "delay_applied_us" ) );
    m_indiP_timingDiag.add( pcf::IndiElement( "delay_model_us" ) );
    m_indiP_timingDiag.add( pcf::IndiElement( "delay_phase_error_us" ) );
    m_indiP_timingDiag.add( pcf::IndiElement( "delay_lock" ) );
    m_indiP_timingDiag.add( pcf::IndiElement( "delay_budget_us" ) );
    m_indiP_timingDiag.add( pcf::IndiElement( "avg_non_delay_service_us" ) );
    m_indiP_timingDiag.add( pcf::IndiElement( "delay_capped" ) );
    m_indiP_timingDiag.add( pcf::IndiElement( "read_latency_error_us" ) );
    m_indiP_timingDiag.add( pcf::IndiElement( "avg_semaphore_period_us" ) );
    m_indiP_timingDiag.add( pcf::IndiElement( "channel_readout_us" ) );
    m_indiP_timingDiag.add( pcf::IndiElement( "trigger_interval_us" ) );
    m_indiP_timingDiag.add( pcf::IndiElement( "trigger_time_us" ) );
    m_indiP_timingDiag.add( pcf::IndiElement( "local_frame_seq" ) );
    m_indiP_timingDiag.add( pcf::IndiElement( "sync_frames_received" ) );
    m_indiP_timingDiag.add( pcf::IndiElement( "sync_frames_written" ) );
    m_indiP_timingDiag.add( pcf::IndiElement( "sync_frames_dropped" ) );
    m_indiP_timingDiag.add( pcf::IndiElement( "sync_frame_id_gap_count" ) );
    m_indiP_timingDiag.add( pcf::IndiElement( "sync_producer_frame_id" ) );
    m_indiP_timingDiag.add( pcf::IndiElement( "sync_producer_frame_delta" ) );
    m_indiP_timingDiag.add( pcf::IndiElement( "mode_code" ) );

    if( m_fpsDevice != "" )
    {
        REG_INDI_SETPROP( m_indiP_fpsSource, m_fpsDevice, m_fpsProperty );
    }

    if( m_numChannelsDevice != "" )
    {
        REG_INDI_SETPROP( m_indiP_numChannelsSource, m_numChannelsDevice, m_numChannelsProperty );
    }

    {
        // Get the maximum privileges available
        elevatedPrivileges elPriv( this );

        m_adc.connect();
    }

    updateTimingDiagnosticsIndi();

    state( stateCodes::OPERATING );
    return 0;
}

void mcp3208Ctrl::updateTimingDiagnosticsIndi()
{
    constexpr double c_timerModeCode   = 0.0;
    constexpr double c_synchroModeCode = 1.0;
    constexpr double c_nsToUs          = 1e-3;

    const bool   synchroMode          = !m_synchroShmimName.empty();
    const double readLatencyError_ns  = m_avgReadLatency_ns - static_cast<double>( m_synchroDelayTarget );
    const double modeCode             = synchroMode ? c_synchroModeCode : c_timerModeCode;
    const double delayAppliedDiag_ns  = synchroMode ? m_delayApplied_ns : 0.0;
    const double delayModelDiag_ns    = synchroMode ? m_delayModel_ns : 0.0;
    const double delayBudgetDiag_ns   = synchroMode ? m_delayBudget_ns : 0.0;
    const double avgNonDelayService_ns = synchroMode ? m_avgNonDelayService_ns : 0.0;
    const double delayCappedDiag      = synchroMode ? m_delayCapped : 0.0;
    const double wfsPeriodMeasured_ns = synchroMode ? m_wfsPeriodMeasured_ns : 0.0;

    double triggerTime_ns = 0.0;
    if( ( m_atime.tv_sec != 0 || m_atime.tv_nsec != 0 ) &&
        ( m_triggerTime.tv_sec != 0 || m_triggerTime.tv_nsec != 0 ) )
    {
        triggerTime_ns = timespecToNs( m_triggerTime ) - timespecToNs( m_atime );
        if( triggerTime_ns < 0.0 )
        {
            triggerTime_ns = 0.0;
        }
    }

    double delayPhaseError_ns = 0.0;
    double delayLock          = 0.0;

    if( synchroMode && wfsPeriodMeasured_ns > 0.0 )
    {
        const double wfsPeriod_ns = wfsPeriodMeasured_ns;

        double delayAppliedWrapped_ns = std::fmod( delayAppliedDiag_ns, wfsPeriod_ns );
        if( delayAppliedWrapped_ns < 0.0 )
        {
            delayAppliedWrapped_ns += wfsPeriod_ns;
        }

        double delayModelWrapped_ns = std::fmod( delayModelDiag_ns, wfsPeriod_ns );
        if( delayModelWrapped_ns < 0.0 )
        {
            delayModelWrapped_ns += wfsPeriod_ns;
        }

        delayPhaseError_ns = delayAppliedWrapped_ns - delayModelWrapped_ns;
        if( delayPhaseError_ns <= -0.5 * wfsPeriod_ns )
        {
            delayPhaseError_ns += wfsPeriod_ns;
        }
        else if( delayPhaseError_ns > 0.5 * wfsPeriod_ns )
        {
            delayPhaseError_ns -= wfsPeriod_ns;
        }

        const double absDelayPhaseError_ns = std::fabs( delayPhaseError_ns );
        if( absDelayPhaseError_ns <= m_delayLockAbsThreshold_ns &&
            absDelayPhaseError_ns <= m_delayLockFracThreshold * wfsPeriod_ns )
        {
            delayLock = 1.0;
        }
    }

    m_delayPhaseError_ns = delayPhaseError_ns;
    m_delayLock          = delayLock;

    const double avgReadLatency_us      = m_avgReadLatency_ns * c_nsToUs;
    const double synchroDelay_us        = ( synchroMode ? static_cast<double>( m_synchroDelay ) : 0.0 ) * c_nsToUs;
    const double synchroDelayTarget_us  = static_cast<double>( m_synchroDelayTarget ) * c_nsToUs;
    const double delayAppliedDiag_us    = delayAppliedDiag_ns * c_nsToUs;
    const double delayModelDiag_us      = delayModelDiag_ns * c_nsToUs;
    const double delayPhaseErrorDiag_us = m_delayPhaseError_ns * c_nsToUs;
    const double delayBudgetDiag_us     = delayBudgetDiag_ns * c_nsToUs;
    const double avgNonDelayService_us  = avgNonDelayService_ns * c_nsToUs;
    const double readLatencyError_us    = readLatencyError_ns * c_nsToUs;
    const double avgSemaphorePeriod_us  = m_avgSemaphorePeriod_ns * c_nsToUs;
    const double channelReadout_us      = m_channelReadoutTime_ns * c_nsToUs;
    const double triggerInterval_us     = m_triggerInterval_ns * c_nsToUs;
    const double triggerTime_us         = triggerTime_ns * c_nsToUs;
    const double localFrameSeqDiag      = static_cast<double>( m_localFrameSeq );
    const double syncFramesReceivedDiag = synchroMode ? static_cast<double>( m_syncFramesReceived ) : 0.0;
    const double syncFramesWrittenDiag  = synchroMode ? static_cast<double>( m_syncFramesWritten ) : 0.0;
    const double syncFramesDroppedDiag  = synchroMode ? static_cast<double>( m_syncFramesDropped ) : 0.0;
    const double syncFrameIdGapCountDiag = synchroMode ? static_cast<double>( m_syncFrameIdGapCount ) : 0.0;
    const double syncProducerFrameIdDiag =
        ( synchroMode && m_syncProducerFrameValid ) ? static_cast<double>( m_syncProducerFrameId ) : 0.0;
    const double syncProducerFrameDeltaDiag =
        ( synchroMode && m_syncProducerFrameValid ) ? static_cast<double>( m_syncProducerFrameDelta ) : 0.0;

    updatesIfChanged<double>( m_indiP_timingDiag,
                              { "avg_read_latency_us",
                                "synchro_delay_us",
                                "synchro_delay_target_us",
                                "delay_applied_us",
                                "delay_model_us",
                                "delay_phase_error_us",
                                "delay_lock",
                                "delay_budget_us",
                                "avg_non_delay_service_us",
                                "delay_capped",
                                "read_latency_error_us",
                                "avg_semaphore_period_us",
                                "channel_readout_us",
                                "trigger_interval_us",
                                "trigger_time_us",
                                "local_frame_seq",
                                "sync_frames_received",
                                "sync_frames_written",
                                "sync_frames_dropped",
                                "sync_frame_id_gap_count",
                                "sync_producer_frame_id",
                                "sync_producer_frame_delta",
                                "mode_code" },
                              { avgReadLatency_us,
                                synchroDelay_us,
                                synchroDelayTarget_us,
                                delayAppliedDiag_us,
                                delayModelDiag_us,
                                delayPhaseErrorDiag_us,
                                m_delayLock,
                                delayBudgetDiag_us,
                                avgNonDelayService_us,
                                delayCappedDiag,
                                readLatencyError_us,
                                avgSemaphorePeriod_us,
                                channelReadout_us,
                                triggerInterval_us,
                                triggerTime_us,
                                localFrameSeqDiag,
                                syncFramesReceivedDiag,
                                syncFramesWrittenDiag,
                                syncFramesDroppedDiag,
                                syncFrameIdGapCountDiag,
                                syncProducerFrameIdDiag,
                                syncProducerFrameDeltaDiag,
                                modeCode } );
}

int mcp3208Ctrl::appLogic()
{
    FRAMEGRABBER_APP_LOGIC;
    TELEMETER_APP_LOGIC;

    FRAMEGRABBER_UPDATE_INDI;

    updatesIfChanged<float>( m_indiP_fps, { "current", "target" }, { m_fps, m_fps } );

    updatesIfChanged<float>( m_indiP_alpha, { "current", "target" }, { m_alpha, m_alpha } );

    updatesIfChanged<int>( m_indiP_numChannels, { "current", "target" }, { m_numChannels, m_numChannels } );

    updatesIfChanged<int>(
        m_indiP_synchroDelay, { "current", "target" }, { m_synchroPostDelay, m_synchroPostDelay } );

    updateTimingDiagnosticsIndi();

    return 0;
}

int mcp3208Ctrl::appShutdown()
{
    FRAMEGRABBER_APP_SHUTDOWN;
    TELEMETER_APP_SHUTDOWN;

    closeSynchroStream();

    return 0;
}

int mcp3208Ctrl::configureAcquisition()
{
    m_values.resize( m_numChannels );

    m_width    = m_numChannels;
    m_height   = 1;
    m_dataType = _DATATYPE_UINT16;

    if( !m_synchroShmimName.empty() )
    {
        log<text_log>( "Configuring semaphore-synchronized acquisition from " + m_synchroShmimName +
                           " with phase offset " + std::to_string( m_synchroPostDelay ) + " us.",
                       logPrio::LOG_INFO );

        if( openSynchroStream() != 0 )
        {
            return 1;
        }

        if( claimSynchroSemaphore() != 0 )
        {
            closeSynchroStream();
            return 1;
        }
    }
    else
    {
        log<text_log>( "Configuring timer-driven acquisition.", logPrio::LOG_INFO );
    }

    return 0;
}

int mcp3208Ctrl::numChannels()
{
    return m_numChannels;
}

float mcp3208Ctrl::fps()
{
    return m_fps;
}

int mcp3208Ctrl::startAcquisition()
{
    if( !m_synchroShmimName.empty() )
    {
        if( !m_synchroStreamOpen )
        {
            return -1;
        }

        if( m_synchroSemaphore == nullptr && claimSynchroSemaphore() != 0 )
        {
            return -1;
        }

        ImageStreamIO_semflush( &m_synchroStream, m_synchroSemaphoreNumber );

        m_synchroDelay = m_synchroDelayTarget;
        m_firstSemaphore        = true;
        m_avgSemaphorePeriod_ns = 0.0;
        m_firstReadLatency      = true;
        m_avgReadLatency_ns     = 0.0;
        m_lastAtime             = timespec{};
        m_atime                 = timespec{};
        m_triggerTime           = timespec{};
        m_wfsPeriodMeasured_ns  = 0.0;
        m_lastProducerAtime     = timespec{};
        m_lastProducerCnt0      = 0;
        m_producerPeriodInst_ns = 0.0;
        m_avgProducerPeriod_ns  = 0.0;
        m_firstProducerSample   = true;
        m_syncFramesReceived    = 0;
        m_syncFramesWritten     = 0;
        m_syncFramesDropped     = 0;
        m_syncFrameIdGapCount   = 0;
        m_syncProducerFrameId   = 0;
        m_syncProducerFrameDelta = 0;
        m_lastSyncProducerFrameId = 0;
        m_syncProducerFrameValid  = false;
        m_delayModel_ns         = 0.0;
        m_delayApplied_ns       = 0.0;
        m_delayBudget_ns        = 0.0;
        m_nonDelayService_ns    = 0.0;
        m_avgNonDelayService_ns = 0.0;
        m_firstNonDelayService  = true;
        m_delayPhaseError_ns    = 0.0;
        m_delayLock             = 0.0;
        m_delayCapped           = 0.0;
    }

    m_triggerInterval_ns = 0.0;
    m_channelReadoutTime_ns = 0.0;
    m_lastTriggerTime    = timespec{};
    m_firstTriggerTime   = true;
    m_firstTimerTrigger  = true;
    m_localFrameSeq      = 0;
    m_delayApplied_ns    = 0.0;
    m_delayBudget_ns     = 0.0;
    m_nonDelayService_ns = 0.0;
    m_delayPhaseError_ns = 0.0;
    m_delayLock          = 0.0;
    m_delayCapped        = 0.0;

    m_time_start = std::chrono::high_resolution_clock::now();

    return 0;
}

int mcp3208Ctrl::acquireAndCheckValid()
{
    if( !m_synchroShmimName.empty() )
    {
        return acquireSynchroAndCheckValid();
    }

    return acquireTimerAndCheckValid();
}

int mcp3208Ctrl::loadImageIntoStream( void *dest )
{
    memcpy( dest, m_values.data(), m_values.size() * sizeof( uint16_t ) );
    ++m_localFrameSeq;
    if( !m_synchroShmimName.empty() )
    {
        ++m_syncFramesWritten;
    }
    return 0;
}

int mcp3208Ctrl::reconfig()
{
    closeSynchroStream();
    return 0;
}

int mcp3208Ctrl::openSynchroStream()
{
    char        shmimFilename[1024];
    struct stat buffer;

    if( m_synchroShmimName.empty() || m_synchroStreamOpen )
    {
        return 0;
    }

    if( ImageStreamIO_openIm( &m_synchroStream, m_synchroShmimName.c_str() ) != 0 )
    {
        return 1;
    }

    if( m_synchroStream.md[0].sem < SEMAPHORE_MAXVAL )
    {
        ImageStreamIO_closeIm( &m_synchroStream );
        memset( &m_synchroStream, 0, sizeof( m_synchroStream ) );
        return 1;
    }

    ImageStreamIO_filename( shmimFilename, sizeof( shmimFilename ), m_synchroShmimName.c_str() );
    if( stat( shmimFilename, &buffer ) != 0 )
    {
        ImageStreamIO_closeIm( &m_synchroStream );
        memset( &m_synchroStream, 0, sizeof( m_synchroStream ) );
        return log<software_error, 1>( { __FILE__, __LINE__, errno, "stat" } );
    }

    m_synchroStreamInode = buffer.st_ino;
    m_synchroStreamOpen  = true;
    return 0;
}

int mcp3208Ctrl::claimSynchroSemaphore()
{
    if( !m_synchroStreamOpen )
    {
        return -1;
    }

    if( m_synchroSemaphore != nullptr )
    {
        return 0;
    }

    m_synchroSemaphoreNumber = ImageStreamIO_getsemwaitindex( &m_synchroStream, m_synchroSemaphoreNumber );
    if( m_synchroSemaphoreNumber < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "No valid semaphore found for " + m_synchroShmimName } );
    }

    m_synchroSemaphore = m_synchroStream.semptr[m_synchroSemaphoreNumber];

    if( m_synchroSemaphore == nullptr )
    {
        return log<software_error, -1>(
            { __FILE__, __LINE__, "No valid semaphore pointer found for " + m_synchroShmimName } );
    }

    return 0;
}

bool mcp3208Ctrl::synchroStreamStale()
{
    int         shmimFd;
    char        shmimFilename[1024];
    struct stat buffer;

    if( !m_synchroStreamOpen || m_synchroStream.md[0].sem <= 0 )
    {
        return true;
    }

    ImageStreamIO_filename( shmimFilename, sizeof( shmimFilename ), m_synchroShmimName.c_str() );

    shmimFd = open( shmimFilename, O_RDWR );
    if( shmimFd == -1 )
    {
        return true;
    }

    close( shmimFd );

    if( stat( shmimFilename, &buffer ) != 0 )
    {
        return true;
    }

    return buffer.st_ino != m_synchroStreamInode;
}

void mcp3208Ctrl::closeSynchroStream()
{
    if( m_synchroStreamOpen )
    {
        if( m_synchroSemaphore != nullptr && m_synchroSemaphoreNumber >= 0 )
        {
            m_synchroStream.semReadPID[m_synchroSemaphoreNumber] = 0;
        }

        ImageStreamIO_closeIm( &m_synchroStream );
    }

    memset( &m_synchroStream, 0, sizeof( m_synchroStream ) );
    m_synchroSemaphore       = nullptr;
    m_synchroSemaphoreNumber = 5;
    m_synchroStreamInode     = 0;
    m_synchroStreamOpen      = false;
    m_atime                  = timespec{};
    m_lastAtime              = timespec{};
    m_avgSemaphorePeriod_ns  = 0.0;
    m_wfsPeriodMeasured_ns   = 0.0;
    m_lastProducerAtime      = timespec{};
    m_lastProducerCnt0       = 0;
    m_producerPeriodInst_ns  = 0.0;
    m_avgProducerPeriod_ns   = 0.0;
    m_firstProducerSample    = true;
    m_syncFramesReceived     = 0;
    m_syncFramesWritten      = 0;
    m_syncFramesDropped      = 0;
    m_syncFrameIdGapCount    = 0;
    m_syncProducerFrameId    = 0;
    m_syncProducerFrameDelta = 0;
    m_lastSyncProducerFrameId = 0;
    m_syncProducerFrameValid  = false;
    m_firstSemaphore         = true;
    m_avgReadLatency_ns      = 0.0;
    m_firstReadLatency       = true;
    m_delayModel_ns          = 0.0;
    m_delayApplied_ns        = 0.0;
    m_delayBudget_ns         = 0.0;
    m_nonDelayService_ns     = 0.0;
    m_avgNonDelayService_ns  = 0.0;
    m_firstNonDelayService   = true;
    m_delayPhaseError_ns     = 0.0;
    m_delayLock              = 0.0;
    m_delayCapped            = 0.0;
    m_triggerTime            = timespec{};
    m_triggerInterval_ns     = 0.0;
    m_channelReadoutTime_ns  = 0.0;
    m_localFrameSeq          = 0;
    m_lastTriggerTime        = timespec{};
    m_firstTriggerTime       = true;
    m_firstTimerTrigger      = true;
}

int mcp3208Ctrl::acquireTimerAndCheckValid()
{
    while( !m_shutdown && !m_reconfig )
    {
        // Get current time
        auto now     = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>( now - m_time_start );

        // Read every 500 microseconds
        if( elapsed.count() >= m_trigger )
        {
            if( m_firstTimerTrigger )
            {
                m_triggerInterval_ns = 0.0;
                m_firstTimerTrigger  = false;
            }
            else
            {
                m_triggerInterval_ns = static_cast<double>( elapsed.count() );
            }

            m_time_start = now; // Reset start time

            const auto readStart = std::chrono::high_resolution_clock::now();
            for( int i = 0; i < m_numChannels; ++i )
            {
                if( readChannelValue( i, m_values[i] ) < 0 )
                {
                    return 1;
                }
            }
            const auto readEnd = std::chrono::high_resolution_clock::now();
            m_channelReadoutTime_ns = static_cast<double>(
                std::chrono::duration_cast<std::chrono::nanoseconds>( readEnd - readStart ).count() );

            m_trigger = m_trigger - m_gain * ( elapsed.count() - nano_sec_target );

            return 0;
        }
        else
        {
            mx::sys::nanoSleep( 1000 ); // Sleep for 1 microsecond to prevent busy waiting
        }
    }

    return 0;
}

int mcp3208Ctrl::acquireSynchroAndCheckValid()
{
    timespec ts;

    if( m_synchroSemaphore == nullptr )
    {
        m_reconfig = true;
        return 1;
    }

    errno = 0;
    if( getRealtime( ts ) < 0 )
    {
        return log<software_critical, -1>( { __FILE__, __LINE__, errno, 0, "clock_gettime" } );
    }

    ts.tv_sec += 1;

    errno = 0;
    if( waitOnSemaphore( m_synchroSemaphore, ts ) != 0 )
    {
        if( errno == EINTR )
        {
            return 1;
        }

        if( errno == ETIMEDOUT )
        {
            if( synchroStreamStale() )
            {
                log<text_log>( "Synchronized trigger stream changed, reconfiguring.", logPrio::LOG_NOTICE );
                m_reconfig = true;
            }

            return 1;
        }

        log<software_error>( { __FILE__, __LINE__, errno, "sem_timedwait" } );
        m_reconfig = true;
        return 1;
    }

    ++m_syncFramesReceived;

    if( m_synchroStream.md != nullptr )
    {
        const uint64_t producerCnt0 = m_synchroStream.md[0].cnt0;
        const timespec producerAtime = m_synchroStream.md[0].atime;
        const bool producerAtimeValid = ( producerAtime.tv_sec != 0 || producerAtime.tv_nsec != 0 );

        m_syncProducerFrameId = producerCnt0;
        if( !m_syncProducerFrameValid )
        {
            m_syncProducerFrameDelta = 0;
            m_syncProducerFrameValid = true;
        }
        else if( producerCnt0 >= m_lastSyncProducerFrameId )
        {
            const uint64_t producerFrameDelta = producerCnt0 - m_lastSyncProducerFrameId;
            m_syncProducerFrameDelta          = producerFrameDelta;

            if( producerFrameDelta > 1 )
            {
                ++m_syncFrameIdGapCount;
                m_syncFramesDropped += ( producerFrameDelta - 1 );
            }
        }
        else
        {
            m_syncProducerFrameDelta = 0;
        }
        m_lastSyncProducerFrameId = producerCnt0;

        if( producerAtimeValid )
        {
            if( m_firstProducerSample )
            {
                m_lastProducerCnt0      = producerCnt0;
                m_lastProducerAtime     = producerAtime;
                m_producerPeriodInst_ns = 0.0;
                m_avgProducerPeriod_ns  = 0.0;
                m_firstProducerSample   = false;
            }
            else if( producerCnt0 > m_lastProducerCnt0 )
            {
                const uint64_t producerFrameDelta = producerCnt0 - m_lastProducerCnt0;
                const double   producerDt_ns      = timespecToNs( producerAtime ) - timespecToNs( m_lastProducerAtime );

                if( producerDt_ns > 0.0 )
                {
                    const double producerPeriod_ns = producerDt_ns / static_cast<double>( producerFrameDelta );
                    m_producerPeriodInst_ns        = producerPeriod_ns;

                    const double alphaProducer = static_cast<double>( m_alpha );
                    if( m_avgProducerPeriod_ns > 0.0 )
                    {
                        m_avgProducerPeriod_ns =
                            alphaProducer * producerPeriod_ns + ( 1.0 - alphaProducer ) * m_avgProducerPeriod_ns;
                    }
                    else
                    {
                        m_avgProducerPeriod_ns = producerPeriod_ns;
                    }
                }

                m_lastProducerCnt0  = producerCnt0;
                m_lastProducerAtime = producerAtime;
            }
            else if( producerCnt0 < m_lastProducerCnt0 )
            {
                m_lastProducerCnt0      = producerCnt0;
                m_lastProducerAtime     = producerAtime;
                m_producerPeriodInst_ns = 0.0;
                m_avgProducerPeriod_ns  = 0.0;
            }
        }
    }

    if( getRealtime( m_atime ) < 0 )
    {
        return log<software_critical, -1>( { __FILE__, __LINE__, errno, 0, "clock_gettime" } );
    }

    updateTriggerTiming( m_atime );

    m_synchroDelay = m_synchroDelayTarget;

    double desiredDelay_ns = static_cast<double>( m_synchroDelay );
    if( desiredDelay_ns < 0.0 )
    {
        desiredDelay_ns = 0.0;
    }

    if( m_wfsPeriodMeasured_ns > 0.0 )
    {
        m_delayBudget_ns = m_wfsPeriodMeasured_ns - m_avgNonDelayService_ns - m_cadenceGuard_ns;
        if( m_delayBudget_ns < 0.0 )
        {
            m_delayBudget_ns = 0.0;
        }
    }
    else
    {
        m_delayBudget_ns = 0.0;
    }

    if( desiredDelay_ns > m_delayBudget_ns )
    {
        m_delayApplied_ns = m_delayBudget_ns;
        m_delayCapped     = 1.0;
    }
    else
    {
        m_delayApplied_ns = desiredDelay_ns;
        m_delayCapped     = 0.0;
    }

    delayBeforeRead();

    if( getRealtime( m_currImageTimestamp ) < 0 )
    {
        return log<software_critical, -1>( { __FILE__, __LINE__, errno, 0, "clock_gettime" } );
    }

    const double readLatency_ns = timespecToNs( m_currImageTimestamp ) - timespecToNs( m_atime );
    const double alpha = static_cast<double>( m_alpha );

    if( !m_firstReadLatency )
    {
        m_avgReadLatency_ns = alpha * readLatency_ns + ( 1.0 - alpha ) * m_avgReadLatency_ns;
    }
    else
    {
        m_avgReadLatency_ns = readLatency_ns;
        m_firstReadLatency  = false;
    }

    updateSynchroDelayController( desiredDelay_ns );

    const auto readStart = std::chrono::high_resolution_clock::now();
    for( int i = 0; i < m_numChannels; ++i )
    {
        if( readChannelValue( i, m_values[i] ) < 0 )
        {
            m_reconfig = true;
            return 1;
        }
    }
    const auto readEnd = std::chrono::high_resolution_clock::now();
    m_channelReadoutTime_ns = static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>( readEnd - readStart ).count() );

    timespec cycleEnd;
    if( getRealtime( cycleEnd ) < 0 )
    {
        return log<software_critical, -1>( { __FILE__, __LINE__, errno, 0, "clock_gettime" } );
    }

    double nonDelayService_ns = timespecToNs( cycleEnd ) - timespecToNs( m_atime ) - m_delayApplied_ns;
    if( nonDelayService_ns < 0.0 )
    {
        nonDelayService_ns = 0.0;
    }

    m_nonDelayService_ns = nonDelayService_ns;
    const double alphaService = static_cast<double>( m_alpha );
    if( !m_firstNonDelayService )
    {
        m_avgNonDelayService_ns =
            alphaService * nonDelayService_ns + ( 1.0 - alphaService ) * m_avgNonDelayService_ns;
    }
    else
    {
        m_avgNonDelayService_ns = nonDelayService_ns;
        m_firstNonDelayService  = false;
    }

    return 0;
}

int mcp3208Ctrl::getRealtime( timespec &ts )
{
    return clock_gettime( CLOCK_REALTIME, &ts );
}

int mcp3208Ctrl::waitOnSemaphore( sem_t *sem, timespec &ts )
{
    return sem_timedwait( sem, &ts );
}

int mcp3208Ctrl::readChannelValue( int channel, uint16_t &value )
{
    value = m_adc.read( channel );
    return 0;
}

void mcp3208Ctrl::delayBeforeRead()
{
    if( m_delayApplied_ns > 0.0 )
    {
        mx::sys::nanoSleep( static_cast<unsigned>( m_delayApplied_ns ) );
        return;
    }
}

void mcp3208Ctrl::updateSynchroDelayController( double desiredDelay_ns )
{
    constexpr double c_satEpsilon_ns = 1e-6;

    const double controlStep_ns =
        static_cast<double>( m_gain ) * ( m_avgReadLatency_ns - static_cast<double>( m_synchroDelayTarget ) );
    double updatedDelay_ns = desiredDelay_ns - controlStep_ns;

    if( updatedDelay_ns < 0.0 )
    {
        updatedDelay_ns = 0.0;
    }

    const bool upperSaturated =
        ( m_wfsPeriodMeasured_ns > 0.0 ) && ( m_delayApplied_ns + c_satEpsilon_ns < desiredDelay_ns );
    const bool pushesFurtherIntoSaturation = ( updatedDelay_ns > desiredDelay_ns );

    if( upperSaturated && pushesFurtherIntoSaturation )
    {
        updatedDelay_ns = m_delayApplied_ns;
    }

    if( m_wfsPeriodMeasured_ns > 0.0 && updatedDelay_ns > m_delayBudget_ns )
    {
        updatedDelay_ns = m_delayBudget_ns;
    }

    m_synchroDelay = static_cast<float>( updatedDelay_ns );
}

int mcp3208Ctrl::checkRecordTimes()
{
    return telemeter<mcp3208Ctrl>::checkRecordTimes( telem_fgtimings() );
}

int mcp3208Ctrl::recordTelem( const telem_fgtimings * )
{
    return recordFGTimings( true );
}

// INDI callback handling for fps configuration.
INDI_NEWCALLBACK_DEFN( mcp3208Ctrl, m_indiP_fps )( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getName() != m_indiP_fps.getName() )
    {
        log<software_error>( { __FILE__, __LINE__, "wrong INDI property received." } );
        return -1;
    }

    float target;

    if( indiTargetUpdate( m_indiP_fps, target, ipRecv, true ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        return -1;
    }

    m_fps           = target;
    m_wfs_fps       = m_fps;
    m_trigger       = 1e9f / m_fps; // Update trigger value based off new fps
    nano_sec_target = 1e9f / m_fps;

    log<text_log>( "set fps = " + std::to_string( m_fps ) );
    return 0;
}

// INDI callback handling for global EMA alpha configuration.
INDI_NEWCALLBACK_DEFN( mcp3208Ctrl, m_indiP_alpha )( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getName() != m_indiP_alpha.getName() )
    {
        log<software_error>( { __FILE__, __LINE__, "wrong INDI property received." } );
        return -1;
    }

    float target;
    if( indiTargetUpdate( m_indiP_alpha, target, ipRecv, true ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        return -1;
    }

    if( target < 0.0f )
    {
        target = 0.0f;
    }
    else if( target > 1.0f )
    {
        target = 1.0f;
    }

    m_alpha = target;

    log<text_log>( "set alpha = " + std::to_string( m_alpha ) );
    return 0;
}

// INDI callback handling for synchronized-mode signed phase offset configuration.
INDI_NEWCALLBACK_DEFN( mcp3208Ctrl, m_indiP_synchroDelay )( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getName() != m_indiP_synchroDelay.getName() )
    {
        log<software_error>( { __FILE__, __LINE__, "wrong INDI property received." } );
        return -1;
    }

    int target;
    if( indiTargetUpdate( m_indiP_synchroDelay, target, ipRecv, true ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        return -1;
    }

    m_synchroPostDelay   = target;
    if( m_wfsPeriodMeasured_ns > 0.0 )
    {
        double t_delay_ns = std::fmod( m_delayModel_ns + 1e3 * static_cast<double>( m_synchroPostDelay ), m_wfsPeriodMeasured_ns );
        if( t_delay_ns < 0.0 )
        {
            t_delay_ns += m_wfsPeriodMeasured_ns;
        }

        m_synchroDelayTarget = static_cast<float>( t_delay_ns );
    }
    else
    {
        m_synchroDelayTarget = 1e3f * static_cast<float>( m_synchroPostDelay );
    }
    m_synchroDelay = m_synchroDelayTarget;
    m_delayApplied_ns = static_cast<double>( m_synchroDelay );

    log<text_log>( "set synchroDelay offset = " + std::to_string( m_synchroPostDelay ) + " us" );
    return 0;
}

INDI_SETCALLBACK_DEFN( mcp3208Ctrl, m_indiP_fpsSource )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_fpsSource, ipRecv );

    if( ipRecv.find( m_fpsElement ) != true ) // this isn't valid
    {
        log<software_error>( { __FILE__, __LINE__, "No current property in fps source." } );
        return 0;
    }

    float target = ipRecv[m_fpsElement].get<float>();

    m_fps           = target;
    m_wfs_fps       = m_fps;
    m_trigger       = 1e9f / m_fps; // Update trigger value based off new fps
    nano_sec_target = 1e9f / m_fps;

    log<text_log>( "set fps from " + m_fpsDevice + " = " + std::to_string( m_fps ) );
    return 0;

} // INDI_SETCALLBACK_DEFN(mcp3208Ctrl, m_indiP_fpsSource)

INDI_NEWCALLBACK_DEFN( mcp3208Ctrl, m_indiP_numChannels )( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getName() != m_indiP_numChannels.getName() )
    {
        log<software_error>( { __FILE__, __LINE__, "wrong INDI property received." } );
        return -1;
    }

    float ch_target;

    if( indiTargetUpdate( m_indiP_numChannels, ch_target, ipRecv, true ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        return -1;
    }

    m_numChannels = ch_target;

    log<text_log>( "set numChannels = " + std::to_string( m_numChannels ));
    return 0;
}

INDI_SETCALLBACK_DEFN( mcp3208Ctrl, m_indiP_numChannelsSource )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_numChannelsSource, ipRecv );

    if( ipRecv.find( m_numChannelsElement ) != true ) // this isn't valid
    {
        log<software_error>( { __FILE__, __LINE__, "No current property in numChannels source." } );
        return 0;
    }

    float ch_target = ipRecv[m_numChannelsElement].get<float>();

    m_numChannels = ch_target;

    log<text_log>( "set numChannels from " + m_numChannelsDevice + " = " + std::to_string( m_numChannels ));
    return 0;
}

} // namespace app
} // namespace MagAOX

#endif // mcp3208Ctrl_hpp
