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
#include "../../libMagAOX/app/wfsClock.hpp"
#include "../../libMagAOX/app/wfsScanSchedule.hpp"

#ifdef __linux__
    #include <sys/prctl.h>
#endif

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
 * ImageStreamIO semaphore. Synchronized, `fps` (`accel.fps`, INDI `fps`) is an upper limit: each WFS
 * semaphore arms the highest integer number of scans whose rate does not exceed it (8 kHz at a 2 kHz or
 * 500 Hz WFS, 7.7 kHz at 700 Hz, for an 8 kHz limit), uniformly spaced across that WFS period, with the
 * first scan at the fitted WFS arrival plus the optional signed `synchroDelay` offset (no built-in delay;
 * `wfsScan::syncSchedule`). The WFS period and each frame's
 * scan anchor come from a ten-second least-squares fit of semaphore arrivals on WFS frame IDs
 * (`wfsScan::rollingWfsClock`), so semaphore jitter moves neither; the fit restarts after a persistent
 * WFS rate change. Scans are never free-running: without new semaphores at most one period's carried-over
 * scans follow.
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
    /// Enumeration of runtime acquisition timing-source choices.
    enum class TimingSource
    {
        Integrator = 0, ///< Use the internal timer/integrator cadence.
        Semaphore  = 1  ///< Use synchronization-stream semaphore cadence.
    };

    /// Nominal SPI time of one 24-bit channel transfer at the default rate; a floor on the scan service estimate.
    static constexpr double c_channelReadNominal_ns = 24e9 / MCP3208Lib::MCP3208::DEFAULT_SPI_BAUD;

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

    std::string m_synchroShmimName; ///< The synchronization ImageStreamIO stream name used when semaphore timing is selected.
    int m_synchroPostDelay{ 0 }; ///< Signed microsecond offset of the synchronized scans from the fitted WFS arrival.

    double m_synchroDtTransfer_ns{ 3e3 }; ///< Transfer-latency term of the reported WFS delay model.

    double m_synchroWfsProcess_ns{ 51.5e3 }; ///< WFS processing-latency term of the reported WFS delay model.

    double m_synchroDtF_ns{ 10e3 }; ///< Filter/transport-latency term of the reported WFS delay model.

    double m_synchroWfsRead_ns{ 276.1e3 }; ///< WFS read-latency term of the reported WFS delay model.

    double m_delayLockAbsThreshold_ns{ 50e3 }; ///< Absolute phase-error threshold for declaring delay lock.

    double m_delayLockFracThreshold{ 0.1 }; ///< Fractional period phase-error threshold for declaring delay lock.

    double m_cadenceGuard_ns{ 20e3 }; ///< Reserved per-frame margin protecting synchronized cadence from overruns.

    double m_spinLead_ns{ 50e3 }; ///< Synchronized: poll instead of sleeping this long before each scan deadline.

    double m_lateTolerance_ns{ 50e3 }; ///< Synchronized: a scan at most this late is taken even if it delays the next.

    bool m_completeScans{ true }; ///< Synchronized: never skip a scan (late ones late, unscanned ones next period).

    int m_wfsFitMinEvents{
        16 }; ///< Consistent WFS references that qualify the rolling clock, at start or a rate change.

    double m_wfsCadenceTolerance{ 0.1 }; ///< Rolling clock arrival gate, as a fraction of the WFS period (plus jitter).

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

    float m_fps{ 8000 }; ///< The target acquisition rate in frames per second.

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

    /// INDI property setting the synchronized scan lateness tolerance in microseconds.
    pcf::IndiProperty m_indiP_lateTolerance;

    /// Handle updates to the synchronized scan lateness tolerance.
    INDI_NEWCALLBACK_DECL( mcp3208Ctrl, m_indiP_lateTolerance );

    /// INDI property selecting the runtime timing source.
    pcf::IndiProperty m_indiP_timingSource;

    /// Handle updates to the runtime timing-source selection property.
    INDI_NEWCALLBACK_DECL( mcp3208Ctrl, m_indiP_timingSource );

    /// INDI property exposing runtime timing diagnostics for acquisition health checks.
    pcf::IndiProperty m_indiP_timingDiag;

    /// INDI property exposing the WFS scan schedule diagnostics.
    pcf::IndiProperty m_indiP_syncTiming;

    float m_trigger{ 1e9f / m_fps };       ///< The timer-mode read interval in nanoseconds.
    float m_gain{ .1 };                    ///< The simple integrator gain used for the timer-mode read interval.
    float nano_sec_target{ 1e9f / m_fps }; ///< The timer-mode target interval in nanoseconds.
    float m_synchroDelay{ 0 };             ///< The scans' offset from the fitted WFS arrival, in nanoseconds.
    float m_synchroDelayTarget{ 0 };       ///< The synchronized-mode effective delay target in nanoseconds after applying signed offset and wrap.
    TimingSource m_timingSource{ TimingSource::Integrator }; ///< Operator-selected timing source requested through INDI.
    TimingSource m_effectiveTimingSource{ TimingSource::Integrator }; ///< Currently active timing source after fallback handling.
    bool m_semaphoreFallbackActive{ false }; ///< Tracks whether runtime has fallen back from requested semaphore timing to integrator timing.

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

    double m_delayModel_ns{ 0.0 }; ///< WFS mid-exposure delay from the model, reported only; scans do not use it.

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

    wfsScan::syncSchedule m_syncSchedule; ///< Scans armed by the latest WFS reference, one period's worth.

    uint64_t m_lateScans{ 0 }; ///< Scans that started after their deadline.

    double m_maxLateness_ns{ 0.0 }; ///< Largest scan start lateness since acquisition started, in nanoseconds.

    uint64_t m_coalescedWakes{ 0 }; ///< Semaphore posts drained with another post and handled as one WFS reference.

    uint64_t m_duplicateReferences{ 0 }; ///< WFS references repeating the previous producer frame ID; they arm nothing.

    uint64_t m_scheduleFailures{ 0 }; ///< WFS references with a measured period whose scans could not be fitted.

    uint64_t m_lastArmedId{ 0 }; ///< WFS frame ID of the latest reference that armed scans.

    bool m_haveArmedId{ false }; ///< Whether m_lastArmedId is set, so frames armed by no reference can be counted.

    uint64_t m_missedFrames{ 0 }; ///< WFS frames between armed references that armed nothing themselves.

    /// Ten-second fit of semaphore arrivals on WFS frame IDs: the scans' WFS period and per-frame anchor.
    wfsScan::rollingWfsClock m_wfsClock;

    double m_wfsClockSpan_s{ 0.0 }; ///< Span of the clock's history, copied for INDI after each reference [s].

    /// INDI property exposing the rolling WFS clock fit.
    pcf::IndiProperty m_indiP_wfsClock;

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

    /// Reset timing counters and trigger diagnostics shared by both timing modes.
    void resetAcquisitionTimingState();

    /// Reset synchronized-acquisition state for a fresh semaphore timing session.
    void resetSynchroAcquisitionState();

    /// Attempt to activate requested timing and apply fallback when semaphore timing is unavailable.
    void updateEffectiveTimingSource();

    /// Acquire one frame using the internal timer loop.
    /**
     * \returns 0 when a new sample is ready.
     * \returns 1 when the loop should continue without publishing.
     */
    int acquireTimerAndCheckValid();

    /// Acquire the next scheduled scan, handling WFS semaphore references while waiting for it.
    /**
     * \returns 0 when a new sample is ready.
     * \returns 1 after a second without a reference, a stream change, or a semaphore error.
     * \returns -1 on a critical timing error.
     */
    int acquireSynchroAndCheckValid();

    /// Handle one WFS reference: track producer metadata, update the delay model, and arm this period's scans.
    /**
     * \returns 0 when scans were armed.
     * \returns 1 when the reference armed no scans (repeated frame ID, WFS clock not yet qualified, or no fit).
     * \returns -1 on a critical timing error.
     */
    int processSyncReference( int64_t arrival /**< [in] semaphore arrival on the monotonic clock, in nanoseconds */ );

    /// Read the scan due at `now`, if any, and update the synchronized timing estimates.
    /**
     * \returns 0 when a new sample is ready.
     * \returns 1 when no scan was due, or a channel read failed.
     * \returns -1 on a critical timing error.
     */
    int acquireScan( int64_t now /**< [in] current monotonic time in nanoseconds */ );

    /// Wait for a synchronization semaphore post until a monotonic deadline.
    /**
     * \returns 0 when a post was consumed.
     * \returns -1 with errno ETIMEDOUT at the deadline, EINTR when interrupted, or another error.
     */
    int waitForSync( int64_t deadline /**< [in] monotonic deadline in nanoseconds */ );

    /// Read the monotonic clock, the clock of all scan deadlines.
    /**
     * \returns the current monotonic time in nanoseconds.
     */
    static int64_t monotonicNow();

    /// Read the current realtime clock value.
    /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
    int getRealtime( timespec &ts /**< [out] the current realtime clock value */ );

    /// Convert a timespec timestamp to nanoseconds.
    static inline double timespecToNs( const timespec &t /**< [in] the timespec value to convert */ );

    /// Convert nanoseconds to a normalized timespec value.
    static inline timespec nsToTimespec( double ns /**< [in] the nanosecond value to convert */ );

    /// Read channels 0 to m_numChannels - 1 into m_values in one SPI message.
    /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
    int readScan();

    /// Update synchronized trigger timing from the current semaphore arrival.
    /** Without a fitted period, the period is an EMA of semaphore intervals, starting from the first.
     */
    void updateTriggerTiming( const timespec &atime /**< [in] the semaphore-arrival timestamp */,
                              double fittedPeriod_ns = 0.0 /**< [in] the rolling clock's WFS period, 0 if unfitted */ );

    /// Publish acquisition timing diagnostics to the INDI read-only property.
    /** The exported `trigger_interval_us` value reports measured current-to-previous trigger interval.
     *
     * The exported `trigger_time_us` value is relative to the latest semaphore arrival (`m_atime`).
     */
    void updateTimingDiagnosticsIndi();

    /// Publish the runtime timing-source switch state to the INDI property.
    void updateTimingSourceIndi();

    /// Publish the WFS scan schedule and rolling WFS clock diagnostics to their INDI read-only properties.
    void updateSyncTimingIndi();

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

void mcp3208Ctrl::updateTriggerTiming( const timespec &atime, double fittedPeriod_ns )
{
    double dt_ns = 0.0;

    if( fittedPeriod_ns > 0.0 )
    {
        m_avgSemaphorePeriod_ns = fittedPeriod_ns;
        m_firstSemaphore        = false;
    }
    else if( !m_firstSemaphore )
    {
        dt_ns = timespecToNs( atime ) - timespecToNs( m_lastAtime );

        // A missed WFS notification is not a cadence change.
        if( m_syncProducerFrameValid && m_syncProducerFrameDelta > 1 )
        {
            dt_ns /= static_cast<double>( m_syncProducerFrameDelta );
        }

        // Start from the first measured interval, so the scan count is right from the second reference.
        const double alpha = static_cast<double>( m_alpha );
        m_avgSemaphorePeriod_ns =
            m_avgSemaphorePeriod_ns > 0.0 ? alpha * dt_ns + ( 1.0 - alpha ) * m_avgSemaphorePeriod_ns : dt_ns;
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

    // No built-in delay: the scans are at the WFS arrival plus the signed user offset. The model is reported only.
    const double delayOffset_ns = 1e3 * static_cast<double>( m_synchroPostDelay );
    double       t_delay_ns     = std::fmod( delayOffset_ns, deltaT_wfs_ns );
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
                "Signed offset in microseconds of the synchronized scans from the fitted WFS arrival; there is no "
                "built-in delay. Also INDI synchroDelay. Default is 0." );

    config.add( "synchro.dtTransfer_ns",
                "",
                "synchro.dtTransfer_ns",
                argType::Required,
                "synchro",
                "dtTransfer_ns",
                false,
                "double",
                "Transfer-latency term in nanoseconds of the WFS delay model, reported as timingDiag.delay_model_us "
                "and not applied to the scans. Default is 3000." );

    config.add( "synchro.wfsProcess_ns",
                "",
                "synchro.wfsProcess_ns",
                argType::Required,
                "synchro",
                "wfsProcess_ns",
                false,
                "double",
                "WFS processing-latency term in nanoseconds of the reported WFS delay model. Default is 51500." );

    config.add( "synchro.dtF_ns",
                "",
                "synchro.dtF_ns",
                argType::Required,
                "synchro",
                "dtF_ns",
                false,
                "double",
                "Filter/transport-latency term in nanoseconds of the reported WFS delay model. Default is 10000." );

    config.add( "synchro.wfsRead_ns",
                "",
                "synchro.wfsRead_ns",
                argType::Required,
                "synchro",
                "wfsRead_ns",
                false,
                "double",
                "WFS read-latency term in nanoseconds of the reported WFS delay model. Default is 276100." );

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

    config.add(
        "synchro.spinLead_ns",
        "",
        "synchro.spinLead_ns",
        argType::Required,
        "synchro",
        "spinLead_ns",
        false,
        "double",
        "Nanoseconds before each synchronized scan deadline to stop sleeping and poll, so the core is awake when "
        "the read starts (a read after sleeping is slower). 0 sleeps to the deadline. Default is 50000." );

    config.add( "synchro.completeScans",
                "",
                "synchro.completeScans",
                argType::Required,
                "synchro",
                "completeScans",
                false,
                "bool",
                "Never skip a synchronized scan: late scans are taken late, and a WFS period's unscanned slots are "
                "taken at once in the next (up to one period's worth), so the average rate is exactly the scans per "
                "WFS frame times the WFS rate. false skips scans later than lateTolerance_ns. Default is true." );

    config.add( "synchro.lateTolerance_ns",
                "",
                "synchro.lateTolerance_ns",
                argType::Required,
                "synchro",
                "lateTolerance_ns",
                false,
                "double",
                "A synchronized scan at most this late is still taken, even if that delays the next scan (which then "
                "catches up); later scans that cannot finish before the next are skipped. Only with completeScans "
                "false. INDI scanLateTolerance (us). 0 to 1000000, default 50000." );

    config.add( "synchro.wfsFitMinEvents",
                "",
                "synchro.wfsFitMinEvents",
                argType::Required,
                "synchro",
                "wfsFitMinEvents",
                false,
                "int",
                "Consistent WFS semaphores that qualify the ten-second rolling WFS clock at start and after a WFS rate "
                "change; no scans until then. 2 to 4096, default 16." );

    config.add( "synchro.wfsCadenceTolerance",
                "",
                "synchro.wfsCadenceTolerance",
                argType::Required,
                "synchro",
                "wfsCadenceTolerance",
                false,
                "double",
                "Rolling WFS clock arrival gate as a fraction of the WFS period (at least 20 us and 6x the fitted "
                "jitter). 0.01 to 0.5, default 0.1." );

    config.add( "synchro.alpha",
                "",
                "synchro.alpha",
                argType::Required,
                "synchro",
                "alpha",
                false,
                "float",
                "Global EMA coefficient for synchronized timing smoothers. Default is 0.01." );

    config.add( "accel.fps",
                "",
                "accel.fps",
                argType::Required,
                "accel",
                "fps",
                false,
                "float",
                "ADC rate in timer mode. In semaphore mode the upper limit: each WFS frame gets the highest integer "
                "number of scans not exceeding it (8000 Hz at a 2 kHz or 500 Hz WFS, 7700 Hz at 700 Hz). Also the INDI "
                "fps target. Default is 8000, maximum 10000." );

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
    _config( m_spinLead_ns, "synchro.spinLead_ns" );
    _config( m_completeScans, "synchro.completeScans" );
    _config( m_lateTolerance_ns, "synchro.lateTolerance_ns" );
    _config( m_wfsFitMinEvents, "synchro.wfsFitMinEvents" );
    _config( m_wfsCadenceTolerance, "synchro.wfsCadenceTolerance" );
    _config( m_alpha, "synchro.alpha" );

    _config( m_numChannels, "accel.numChannels" ); // making number of mcp3208 channels we read out configurable

    float fps = m_fps;
    _config( fps, "accel.fps" );

    if( fps > 0 && fps <= 10000 )
    {
        m_fps           = fps;
        m_trigger       = 1e9f / m_fps;
        nano_sec_target = 1e9f / m_fps;
    }
    else
    {
        log<text_log>( "accel.fps must be in (0, 10000]; keeping " + std::to_string( m_fps ), logPrio::LOG_WARNING );
    }
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

    if( m_spinLead_ns < 0.0 )
    {
        m_spinLead_ns = 0.0;
    }

    m_lateTolerance_ns     = std::clamp( m_lateTolerance_ns, 0.0, 1e6 );
    m_wfsFitMinEvents      = std::clamp( m_wfsFitMinEvents, 2, 4096 );
    m_wfsCadenceTolerance  = std::clamp( m_wfsCadenceTolerance, 0.01, 0.5 );
    m_wfsClock.m_minEvents = static_cast<unsigned>( m_wfsFitMinEvents );
    m_wfsClock.m_tolerance = m_wfsCadenceTolerance;

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
    if( m_synchroShmimName.empty() )
    {
        m_timingSource = TimingSource::Integrator;
    }
    else
    {
        m_timingSource = TimingSource::Semaphore;
    }
    m_effectiveTimingSource   = m_timingSource;
    m_semaphoreFallbackActive = false;

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

    // INDI prop for user to set the synchronized scan lateness tolerance in microseconds
    CREATE_REG_INDI_NEW_NUMBERF( m_indiP_lateTolerance, "scanLateTolerance", 0, 1000, 1, "%.1f", "us", "" );
    m_indiP_lateTolerance["current"].setValue( m_lateTolerance_ns * 1e-3 );
    m_indiP_lateTolerance["target"].setValue( m_lateTolerance_ns * 1e-3 );

    if( createStandardIndiSelectionSw( m_indiP_timingSource,
                                       "timingSource",
                                       { "semaphore", "integrator" },
                                       { "Semaphore", "Integrator" },
                                       "Timing Source",
                                       "Acquisition" ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "error from createStandardIndiSelectionSw" } );
    }

    if( registerIndiPropertyNew( m_indiP_timingSource, INDI_NEWCALLBACK( m_indiP_timingSource ) ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "error from registerIndiPropertyNew" } );
    }

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

    CREATE_REG_INDI_RO_NUMBER( m_indiP_syncTiming, "syncTiming", "WFS scan schedule", "Diagnostics" );
    for( const char *name : { "scans_per_wfs",
                              "scheduled_fps",
                              "scan_period_us",
                              "phase_us",
                              "skipped_scans",
                              "late_scans",
                              "max_lateness_us",
                              "coalesced_wakes",
                              "duplicate_references",
                              "schedule_failures",
                              "carried_scans",
                              "missed_frames" } )
    {
        m_indiP_syncTiming.add( pcf::IndiElement( name ) );
    }

    CREATE_REG_INDI_RO_NUMBER( m_indiP_wfsClock, "wfsClock", "Rolling WFS clock", "Diagnostics" );
    for( const char *name : { "valid",
                              "events",
                              "span_s",
                              "fps",
                              "period_us",
                              "jitter_us",
                              "arrival_error_us",
                              "drift_us",
                              "rejected",
                              "resets" } )
    {
        m_indiP_wfsClock.add( pcf::IndiElement( name ) );
    }

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

    updateTimingSourceIndi();
    updateTimingDiagnosticsIndi();
    updateSyncTimingIndi();

    state( stateCodes::OPERATING );
    return 0;
}

void mcp3208Ctrl::updateTimingSourceIndi()
{
    updateSwitchIfChanged( m_indiP_timingSource,
                           "semaphore",
                           m_timingSource == TimingSource::Semaphore ? pcf::IndiElement::On : pcf::IndiElement::Off );

    updateSwitchIfChanged( m_indiP_timingSource,
                           "integrator",
                           m_timingSource == TimingSource::Integrator ? pcf::IndiElement::On : pcf::IndiElement::Off );
}

void mcp3208Ctrl::updateTimingDiagnosticsIndi()
{
    constexpr double c_timerModeCode   = 0.0;
    constexpr double c_synchroModeCode = 1.0;
    constexpr double c_nsToUs          = 1e-3;

    const bool   synchroMode          = ( m_effectiveTimingSource == TimingSource::Semaphore );
    const double readLatencyError_ns  =
        synchroMode ? ( m_avgReadLatency_ns - static_cast<double>( m_synchroDelayTarget ) ) : 0.0;
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

    // Locked: the scans follow a WFS clock fit, at the requested offset.
    if( synchroMode && wfsPeriodMeasured_ns > 0.0 && m_wfsClock.m_ready )
    {
        // Scans an interval apart are equivalent phases; the lock thresholds stay fractions of the WFS period.
        const double wfsPeriod_ns = m_syncSchedule.m_count ? m_syncSchedule.m_interval : wfsPeriodMeasured_ns;

        double delayAppliedWrapped_ns = std::fmod( delayAppliedDiag_ns, wfsPeriod_ns );
        if( delayAppliedWrapped_ns < 0.0 )
        {
            delayAppliedWrapped_ns += wfsPeriod_ns;
        }

        double delayTargetWrapped_ns = std::fmod( static_cast<double>( m_synchroDelayTarget ), wfsPeriod_ns );
        if( delayTargetWrapped_ns < 0.0 )
        {
            delayTargetWrapped_ns += wfsPeriod_ns;
        }

        delayPhaseError_ns = delayAppliedWrapped_ns - delayTargetWrapped_ns;
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
            absDelayPhaseError_ns <= m_delayLockFracThreshold * wfsPeriodMeasured_ns )
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

void mcp3208Ctrl::updateSyncTimingIndi()
{
    const auto &schedule = m_syncSchedule;

    updatesIfChanged<double>( m_indiP_syncTiming,
                              { "scans_per_wfs",
                                "scheduled_fps",
                                "scan_period_us",
                                "phase_us",
                                "skipped_scans",
                                "late_scans",
                                "max_lateness_us",
                                "coalesced_wakes",
                                "duplicate_references",
                                "schedule_failures",
                                "carried_scans",
                                "missed_frames" },
                              { static_cast<double>( schedule.m_count ),
                                schedule.fps(),
                                schedule.m_interval * 1e-3,
                                schedule.m_phase * 1e-3,
                                static_cast<double>( schedule.m_skipped ),
                                static_cast<double>( m_lateScans ),
                                m_maxLateness_ns * 1e-3,
                                static_cast<double>( m_coalescedWakes ),
                                static_cast<double>( m_duplicateReferences ),
                                static_cast<double>( m_scheduleFailures ),
                                static_cast<double>( schedule.m_carried ),
                                static_cast<double>( m_missedFrames ) } );

    const auto &clock = m_wfsClock;

    updatesIfChanged<double>( m_indiP_wfsClock,
                              { "valid",
                                "events",
                                "span_s",
                                "fps",
                                "period_us",
                                "jitter_us",
                                "arrival_error_us",
                                "drift_us",
                                "rejected",
                                "resets" },
                              { clock.valid() ? 1.0 : 0.0,
                                static_cast<double>( clock.m_count ),
                                m_wfsClockSpan_s,
                                clock.m_period > 0 ? 1e9 / clock.m_period : 0.0,
                                clock.m_period * 1e-3,
                                clock.m_jitter * 1e-3,
                                clock.m_arrivalError * 1e-3,
                                clock.m_drift * 1e-3,
                                static_cast<double>( clock.m_rejected ),
                                static_cast<double>( clock.m_resets ) } );
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

    updatesIfChanged<double>(
        m_indiP_lateTolerance, { "current", "target" }, { m_lateTolerance_ns * 1e-3, m_lateTolerance_ns * 1e-3 } );

    updateTimingSourceIndi();

    updateTimingDiagnosticsIndi();

    updateSyncTimingIndi();

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

    if( m_timingSource == TimingSource::Semaphore )
    {
        if( !m_synchroShmimName.empty() )
        {
            log<text_log>( "Configuring semaphore-synchronized acquisition from " + m_synchroShmimName +
                               " with phase offset " + std::to_string( m_synchroPostDelay ) + " us.",
                           logPrio::LOG_INFO );
        }
        else
        {
            log<text_log>( "Semaphore timing requested without synchro.shmimName; falling back to timer timing.",
                           logPrio::LOG_WARNING );
        }

        updateEffectiveTimingSource();
        if( m_effectiveTimingSource != TimingSource::Semaphore )
        {
            log<text_log>( "Proceeding with timer-driven fallback until semaphore timing is available.",
                           logPrio::LOG_WARNING );
        }
    }
    else
    {
        log<text_log>( "Configuring timer-driven acquisition.", logPrio::LOG_INFO );
        closeSynchroStream();
        m_effectiveTimingSource   = TimingSource::Integrator;
        m_semaphoreFallbackActive = false;
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
#ifdef __linux__
    // Scan deadlines are reached through semaphore wait timeouts, which carry the thread's timer slack (50 us by
    // default, applied to SCHED_FIFO threads by older kernels). 1 ns is the minimum; 0 would restore the default.
    prctl( PR_SET_TIMERSLACK, 1UL, 0UL, 0UL, 0UL );
#endif

    updateEffectiveTimingSource();

    if( m_effectiveTimingSource == TimingSource::Semaphore )
    {
        if( !m_synchroStreamOpen )
        {
            return 1;
        }

        if( m_synchroSemaphore == nullptr && claimSynchroSemaphore() != 0 )
        {
            updateEffectiveTimingSource();
            if( m_effectiveTimingSource != TimingSource::Semaphore )
            {
                resetAcquisitionTimingState();
                m_time_start = std::chrono::high_resolution_clock::now();
                return 0;
            }

            return 1;
        }

        ImageStreamIO_semflush( &m_synchroStream, m_synchroSemaphoreNumber );

        m_synchroDelay = m_synchroDelayTarget;

        resetSynchroAcquisitionState();
    }

    resetAcquisitionTimingState();

    m_time_start = std::chrono::high_resolution_clock::now();

    return 0;
}

int mcp3208Ctrl::acquireAndCheckValid()
{
    if( m_timingSource == TimingSource::Semaphore )
    {
        updateEffectiveTimingSource();

        if( m_effectiveTimingSource == TimingSource::Semaphore )
        {
            const int synchroRv = acquireSynchroAndCheckValid();
            if( synchroRv == 0 || synchroRv < 0 )
            {
                return synchroRv;
            }

            m_effectiveTimingSource = TimingSource::Integrator;
            if( !m_semaphoreFallbackActive )
            {
                log<text_log>(
                    "Semaphore timing unavailable during acquisition; using timer-driven fallback.",
                    logPrio::LOG_WARNING );
                m_semaphoreFallbackActive = true;
            }
        }
    }

    return acquireTimerAndCheckValid();
}

int mcp3208Ctrl::loadImageIntoStream( void *dest )
{
    memcpy( dest, m_values.data(), m_values.size() * sizeof( uint16_t ) );
    ++m_localFrameSeq;
    if( m_effectiveTimingSource == TimingSource::Semaphore )
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

    resetSynchroAcquisitionState();
    resetAcquisitionTimingState();
}

void mcp3208Ctrl::resetAcquisitionTimingState()
{
    m_triggerTime           = timespec{};
    m_triggerInterval_ns    = 0.0;
    m_channelReadoutTime_ns = 0.0;
    m_localFrameSeq         = 0;
    m_lastTriggerTime       = timespec{};
    m_firstTriggerTime      = true;
    m_firstTimerTrigger     = true;
}

void mcp3208Ctrl::resetSynchroAcquisitionState()
{
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
    m_syncSchedule            = {};
    m_lateScans               = 0;
    m_maxLateness_ns          = 0.0;
    m_coalescedWakes          = 0;
    m_duplicateReferences     = 0;
    m_scheduleFailures        = 0;
    m_lastArmedId             = 0;
    m_haveArmedId             = false;
    m_missedFrames            = 0;
    m_wfsClock.reset();
    m_wfsClockSpan_s = 0.0;
}

void mcp3208Ctrl::updateEffectiveTimingSource()
{
    if( m_timingSource != TimingSource::Semaphore )
    {
        m_effectiveTimingSource   = TimingSource::Integrator;
        m_semaphoreFallbackActive = false;
        return;
    }

    if( m_synchroShmimName.empty() )
    {
        if( !m_semaphoreFallbackActive )
        {
            log<text_log>( "Semaphore timing requested but synchro.shmimName is empty; using timer-driven fallback.",
                           logPrio::LOG_WARNING );
            m_semaphoreFallbackActive = true;
        }

        m_effectiveTimingSource = TimingSource::Integrator;
        return;
    }

    bool synchroReady = m_synchroStreamOpen;
    if( !synchroReady && openSynchroStream() == 0 )
    {
        synchroReady = true;
    }

    if( synchroReady && m_synchroSemaphore == nullptr && claimSynchroSemaphore() != 0 )
    {
        synchroReady = false;
    }
    else if( synchroReady && m_synchroSemaphore == nullptr )
    {
        synchroReady = false;
    }

    if( !synchroReady )
    {
        m_effectiveTimingSource = TimingSource::Integrator;
        if( !m_semaphoreFallbackActive )
        {
            log<text_log>(
                "Semaphore timing unavailable; using timer-driven fallback until synchronization stream is ready.",
                logPrio::LOG_WARNING );
            m_semaphoreFallbackActive = true;
        }

        return;
    }

    m_effectiveTimingSource = TimingSource::Semaphore;
    if( m_semaphoreFallbackActive )
    {
        log<text_log>( "Semaphore timing restored; leaving timer-driven fallback.", logPrio::LOG_NOTICE );
    }
    m_semaphoreFallbackActive = false;
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

            // Framegrabber latency telemetry requires an acquisition timestamp for every valid frame.
            if( getRealtime( m_currImageTimestamp ) < 0 )
            {
                return log<software_critical, -1>( { __FILE__, __LINE__, errno, 0, "clock_gettime" } );
            }

            const auto readStart = std::chrono::high_resolution_clock::now();
            if( readScan() < 0 )
            {
                return 1;
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
    if( m_synchroSemaphore == nullptr )
    {
        m_reconfig = true;
        return 1;
    }

    bool    wake     = false;
    int64_t lastWake = monotonicNow(); // Start of the one-second silence timeout.

    while( !m_shutdown && !m_reconfig )
    {
        const int64_t now = monotonicNow();

        // A new WFS reference takes precedence over a pending scan of the previous period.
        if( wake || sem_trywait( m_synchroSemaphore ) == 0 )
        {
            wake = false;

            while( sem_trywait( m_synchroSemaphore ) == 0 )
            {
                ++m_syncFramesReceived;
                ++m_coalescedWakes;
            }

            if( processSyncReference( now ) < 0 )
            {
                return -1;
            }

            lastWake = now;
            continue;
        }

        m_syncSchedule.discardLate( now );

        if( m_syncSchedule.due( now ) )
        {
            const int rv = acquireScan( now );

            if( rv != 1 || m_reconfig )
            {
                return rv;
            }

            continue;
        }

        const int64_t timeout = lastWake + 1000000000LL;

        if( now >= timeout )
        {
            m_syncSchedule.cancel();

            if( synchroStreamStale() )
            {
                log<text_log>( "Synchronized trigger stream changed, reconfiguring.", logPrio::LOG_NOTICE );
                m_reconfig = true;
            }

            return 1;
        }

        int64_t until = timeout;

        // Sleep only until spinLead_ns before the next scan, then poll: a read on a core woken from sleep is slower.
        if( m_syncSchedule.m_slot < m_syncSchedule.m_count )
        {
            until = std::min( m_syncSchedule.deadline() - static_cast<int64_t>( m_spinLead_ns ), timeout );

            if( now >= until )
            {
                continue;
            }
        }

        errno = 0;
        if( waitForSync( until ) == 0 )
        {
            wake = true;
        }
        else if( errno != ETIMEDOUT && errno != EINTR )
        {
            log<software_error>( { __FILE__, __LINE__, errno, "WFS semaphore wait" } );
            m_reconfig = true;
            return 1;
        }
    }

    return 1;
}

int mcp3208Ctrl::processSyncReference( int64_t arrival )
{
    ++m_syncFramesReceived;

    if( m_synchroStream.md != nullptr )
    {
        const uint64_t producerCnt0       = m_synchroStream.md[0].cnt0;
        const timespec producerAtime      = m_synchroStream.md[0].atime;
        const bool     producerAtimeValid = ( producerAtime.tv_sec != 0 || producerAtime.tv_nsec != 0 );

        // A repeated frame ID is not a new WFS frame: it neither re-anchors the scans nor updates the cadence.
        if( m_syncProducerFrameValid && producerCnt0 == m_lastSyncProducerFrameId )
        {
            m_syncProducerFrameDelta = 0;
            ++m_duplicateReferences;
            return 1;
        }

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

    // The scans use the ten-second fit: its period, and this frame's fitted arrival as their anchor, so semaphore
    // jitter (e.g. over TCP) moves neither. Rejected arrivals still use the fit, also while consecutive ones are being
    // checked for a WFS rate change: the fit only changes if they confirm one. None until the fit first qualifies.
    const uint64_t id = m_synchroStream.md != nullptr ? m_syncProducerFrameId : m_syncFramesReceived;
    m_wfsClock.observe( id, arrival );
    m_wfsClockSpan_s  = m_wfsClock.span();
    const bool fitted = m_wfsClock.m_ready;

    updateTriggerTiming( m_atime, fitted ? m_wfsClock.m_period : 0.0 );

    if( !fitted )
    {
        return 1;
    }

    const double service = std::max( m_avgNonDelayService_ns, m_numChannels * c_channelReadNominal_ns );
    m_delayBudget_ns     = std::max( 0.0, m_wfsPeriodMeasured_ns - service - m_cadenceGuard_ns );
    m_synchroDelay       = m_synchroDelayTarget;

    m_syncSchedule.m_complete = m_completeScans;

    if( !m_syncSchedule.arm( m_wfsClock.reference( id ),
                             m_wfsPeriodMeasured_ns,
                             m_fps,
                             static_cast<double>( m_synchroDelay ),
                             service,
                             m_cadenceGuard_ns,
                             m_lateTolerance_ns ) )
    {
        if( m_wfsPeriodMeasured_ns > 0.0 )
        {
            ++m_scheduleFailures;

            // Only scans update the service estimate, so restart it from the nominal read time to measure it again.
            m_avgNonDelayService_ns = 0.0;
            m_firstNonDelayService  = true;
        }

        return 1;
    }

    // Frames since the last armed one armed nothing (notifications missed or merged, or a WFS rate change being
    // confirmed). Complete, their scans are owed and read as soon as possible.
    if( m_haveArmedId && id > m_lastArmedId + 1 )
    {
        const uint64_t missed = id - m_lastArmedId - 1;
        m_missedFrames += missed;

        if( m_completeScans )
        {
            m_syncSchedule.owe( static_cast<unsigned>( std::min<uint64_t>( missed, 64 ) ) * m_syncSchedule.m_count );
        }
    }

    m_lastArmedId = id;
    m_haveArmedId = true;

    double requestedPhase = std::fmod( static_cast<double>( m_synchroDelay ), m_syncSchedule.m_interval );

    if( requestedPhase < 0.0 )
    {
        requestedPhase += m_syncSchedule.m_interval;
    }

    m_delayCapped = requestedPhase > m_syncSchedule.m_phase ? 1.0 : 0.0;
    return 0;
}

int mcp3208Ctrl::acquireScan( int64_t now )
{
    m_syncSchedule.discardLate( now );

    // A slot carried over from the previous period is scanned at once and has no phase in this one.
    const bool    carried  = m_syncSchedule.m_backlog > 0;
    const int64_t due      = carried ? now : m_syncSchedule.deadline();
    const bool    previous = m_syncSchedule.m_haveSample;
    const int64_t last     = m_syncSchedule.m_lastSample;

    if( !m_syncSchedule.take( now ) )
    {
        return 1;
    }

    if( now > due )
    {
        ++m_lateScans;
        m_maxLateness_ns = std::max( m_maxLateness_ns, static_cast<double>( now - due ) );
    }

    const unsigned slot  = m_syncSchedule.m_slot - 1;
    m_triggerInterval_ns = previous ? static_cast<double>( now - last ) : 0.0;

    if( !carried )
    {
        m_delayApplied_ns = static_cast<double>( due - m_syncSchedule.m_anchor );
        m_triggerTime     = nsToTimespec( timespecToNs( m_atime ) + m_delayApplied_ns );
    }

    if( getRealtime( m_currImageTimestamp ) < 0 )
    {
        return log<software_critical, -1>( { __FILE__, __LINE__, errno, 0, "clock_gettime" } );
    }

    const int64_t start = monotonicNow();
    if( readScan() < 0 )
    {
        m_reconfig = true;
        return 1;
    }
    const int64_t finished = monotonicNow();

    m_channelReadoutTime_ns = static_cast<double>( finished - start );

    const double alpha = static_cast<double>( m_alpha );

    // Report the read latency (scan start after the fitted WFS arrival) once per WFS period, at the scan nearest the
    // requested offset. It is not fed back.
    const unsigned referenceSlot = static_cast<unsigned>(
        std::clamp( std::floor( ( static_cast<double>( m_synchroDelayTarget ) - m_syncSchedule.m_phase ) /
                                    m_syncSchedule.m_interval +
                                0.5 ),
                    0.0,
                    static_cast<double>( m_syncSchedule.m_count - 1 ) ) );

    if( !carried && slot == referenceSlot )
    {
        const double latency = static_cast<double>( now - m_syncSchedule.m_anchor );
        m_avgReadLatency_ns  = m_firstReadLatency ? latency : alpha * latency + ( 1.0 - alpha ) * m_avgReadLatency_ns;
        m_firstReadLatency   = false;
    }

    m_nonDelayService_ns    = m_channelReadoutTime_ns;
    m_avgNonDelayService_ns = m_firstNonDelayService
                                  ? m_nonDelayService_ns
                                  : alpha * m_nonDelayService_ns + ( 1.0 - alpha ) * m_avgNonDelayService_ns;
    m_firstNonDelayService  = false;

    return 0;
}

int mcp3208Ctrl::waitForSync( int64_t deadline )
{
#if defined( __linux__ ) && defined( __GLIBC__ )
    #if __GLIBC_PREREQ( 2, 30 )
    const timespec until{ static_cast<time_t>( deadline / 1000000000LL ),
                          static_cast<long>( deadline % 1000000000LL ) };
    return sem_clockwait( m_synchroSemaphore, CLOCK_MONOTONIC, &until );
    #endif
#endif

    while( !m_shutdown && !m_reconfig )
    {
        if( sem_trywait( m_synchroSemaphore ) == 0 )
        {
            return 0;
        }

        if( errno != EAGAIN )
        {
            return -1;
        }

        const int64_t now = monotonicNow();

        if( now >= deadline )
        {
            errno = ETIMEDOUT;
            return -1;
        }

        const timespec pause{ 0, static_cast<long>( std::min<int64_t>( deadline - now, 1000000 ) ) };
        nanosleep( &pause, nullptr );
    }

    errno = EINTR;
    return -1;
}

int64_t mcp3208Ctrl::monotonicNow()
{
    timespec ts;
    clock_gettime( CLOCK_MONOTONIC, &ts );
    return static_cast<int64_t>( ts.tv_sec ) * 1000000000LL + ts.tv_nsec;
}

int mcp3208Ctrl::getRealtime( timespec &ts )
{
    return clock_gettime( CLOCK_REALTIME, &ts );
}

int mcp3208Ctrl::readScan()
{
    static constexpr uint8_t c_channels[MCP3208Lib::MCP3208::MAX_CHANNELS] = { 0, 1, 2, 3, 4, 5, 6, 7 };

    if( m_numChannels <= 0 )
    {
        return 0;
    }

    if( m_numChannels > MCP3208Lib::MCP3208::MAX_CHANNELS || m_values.size() < static_cast<size_t>( m_numChannels ) )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "invalid MCP3208 channel count" } );
    }

    try
    {
        m_adc.read( c_channels, m_values.data(), m_numChannels );
    }
    catch( const std::exception &error )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, error.what() } );
    }

    return 0;
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
        double t_delay_ns = std::fmod( 1e3 * static_cast<double>( m_synchroPostDelay ), m_wfsPeriodMeasured_ns );
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

// INDI callback handling for the synchronized scan lateness tolerance.
INDI_NEWCALLBACK_DEFN( mcp3208Ctrl, m_indiP_lateTolerance )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_lateTolerance, ipRecv );

    double target;
    if( indiTargetUpdate( m_indiP_lateTolerance, target, ipRecv, true ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        return -1;
    }

    // Applied from the next WFS reference.
    m_lateTolerance_ns = std::clamp( 1e3 * target, 0.0, 1e6 );

    log<text_log>( "set scanLateTolerance = " + std::to_string( m_lateTolerance_ns * 1e-3 ) + " us" );
    return 0;
}

// INDI callback handling for runtime timing-source selection.
INDI_NEWCALLBACK_DEFN( mcp3208Ctrl, m_indiP_timingSource )( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getName() != m_indiP_timingSource.getName() )
    {
        log<software_error>( { __FILE__, __LINE__, "wrong INDI property received." } );
        return -1;
    }

    bool requestSemaphore  = false;
    bool requestIntegrator = false;

    if( ipRecv.find( "semaphore" ) &&
        ipRecv["semaphore"].getSwitchState() == pcf::IndiElement::On )
    {
        requestSemaphore = true;
    }

    if( ipRecv.find( "integrator" ) &&
        ipRecv["integrator"].getSwitchState() == pcf::IndiElement::On )
    {
        requestIntegrator = true;
    }

    if( requestSemaphore == requestIntegrator )
    {
        log<software_error>( { __FILE__, __LINE__, "timingSource requires exactly one selected mode." } );
        updateTimingSourceIndi();
        return 0;
    }

    const TimingSource requestedSource = requestSemaphore ? TimingSource::Semaphore : TimingSource::Integrator;
    if( requestedSource != m_timingSource )
    {
        m_timingSource = requestedSource;
        updateEffectiveTimingSource();
        m_reconfig = true;
        log<text_log>(
            "set timingSource = " + std::string( requestSemaphore ? "semaphore" : "integrator" ) );
    }

    updateTimingSourceIndi();
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

    m_wfs_fps = target;

    // Synchronized, the scan count follows the measured WFS period; the source must not lower the rate limit.
    if( m_timingSource == TimingSource::Semaphore )
    {
        return 0;
    }

    m_fps           = target;
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
