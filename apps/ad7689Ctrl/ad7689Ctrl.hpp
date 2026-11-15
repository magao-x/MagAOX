/** \file ad7689Ctrl.hpp
 * \brief The MagAO-X AD7689 Controller header file
 *
 * \ingroup ad7689Ctrl_files
 */

#ifndef ad7689Ctrl_hpp
#define ad7689Ctrl_hpp

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"
#include "dependencies/AD7689.h"
#include "ad7689Sync.hpp"
#include <array>
#include <atomic>
#include <mutex>
#include <sstream>

/** \defgroup ad7689Ctrl
 * \brief The MagAO-X application to readout an AD7689 A/D on a raspberry Pi.
 *
 * See apps/ad7689Ctrl/README.md for configuration, timing, and commissioning.
 *
 * \ingroup apps
 *
 */

/** \defgroup ad7689Ctrl_files
 * \ingroup ad7689Ctrl
 */

namespace MagAOX
{
namespace app
{

/** MagAO-X application to read AD7689 channels on a Raspberry Pi.
 *
 * The controller can either acquire on its internal timer loop or synchronize reads to an
 * ImageStreamIO semaphore.
 *
 * \ingroup ad7689Ctrl
 */
class ad7689Ctrl : public MagAOXApp<true>, public dev::frameGrabber<ad7689Ctrl>, public dev::telemeter<ad7689Ctrl>
{

    // Give the test harness access.
    friend class ad7689Ctrl_test;
    friend class dev::frameGrabber<ad7689Ctrl>;
    friend class dev::telemeter<ad7689Ctrl>;

    typedef dev::frameGrabber<ad7689Ctrl> frameGrabberT;
    typedef dev::telemeter<ad7689Ctrl>    telemeterT;

    /// The active AD7689 device interface used for hardware access.
    AD7689Lib::AD7689 m_adc;

    static constexpr bool c_frameGrabber_flippable = false; /**< app:dev config indicating these images can not be
                                                                 flipped */

  protected:
    /** \name Configurable Parameters - Data
     * @{
     */

    int m_numChannels{ 2 }; ///< The number of AD7689 channels read into each output frame.

    std::string m_fpsDevice; ///< External WFS-rate hint in synchronized mode, or ADC-rate source in timer mode.
    std::string m_fpsProperty{ "fps" };    ///< Property name providing external fps metadata.
    std::string m_fpsElement{ "current" }; ///< Property element containing the fps value.

    float m_fpsTol{ 0 }; ///< The tolerance used when monitoring fps metadata changes.

    std::string m_numChannelsDevice; ///< Device name for getting numChannels to set circular buffer length.
    std::string m_numChannelsProperty{
        "numChannels" }; ///< Property name for getting numChannels to set circular buffer length.
    std::string m_numChannelsElement{
        "current" }; ///< Element name for getting numChannels to set circular buffer length.

    float m_numChannelsTol{ 0 }; ///< The tolerance for detecting a change in numChannels.

    std::string m_synchroShmimName; ///< The synchronization ImageStreamIO stream name; empty selects timer mode.
    int m_synchroPostDelay{ 0 };    ///< Signed microsecond phase offset added to synchronized delay-model predictions.

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

    /// INDI property exposing the active and requested scan size.
    pcf::IndiProperty m_indiP_numChannels;
    /// Stage a validated scan-size request.
    INDI_NEWCALLBACK_DECL( ad7689Ctrl, m_indiP_numChannels );

    /// Subscription providing requested scan-size changes.
    pcf::IndiProperty m_indiP_numChannelsSource;
    /// Stage a validated external scan-size request.
    INDI_SETCALLBACK_DECL( ad7689Ctrl, m_indiP_numChannelsSource );

    /// INDI property exposing the local fps target.
    pcf::IndiProperty m_indiP_fps;

    /// Handle updates to the local fps target property.
    INDI_NEWCALLBACK_DECL( ad7689Ctrl, m_indiP_fps );

    float m_fps{ 2000 }; ///< The target acquisition rate in frames per second.

    /// INDI property subscription used to follow an external fps source.
    pcf::IndiProperty m_indiP_fpsSource;

    /// Handle updates from the configured external fps source.
    INDI_SETCALLBACK_DECL( ad7689Ctrl, m_indiP_fpsSource );

    /// INDI property exposing the global EMA alpha used by timing smoothers.
    pcf::IndiProperty m_indiP_alpha;

    /// Handle updates to the global EMA alpha property.
    INDI_NEWCALLBACK_DECL( ad7689Ctrl, m_indiP_alpha );

    /// INDI property exposing the synchronized-mode signed phase offset in microseconds.
    pcf::IndiProperty m_indiP_synchroDelay;

    /// Handle updates to the synchronized-mode signed phase offset property.
    INDI_NEWCALLBACK_DECL( ad7689Ctrl, m_indiP_synchroDelay );

    /// INDI property exposing runtime timing diagnostics for acquisition health checks.
    pcf::IndiProperty m_indiP_timingDiag;

    float m_trigger{ 1e9f / m_fps };       ///< The timer-mode read interval in nanoseconds.
    float m_gain{ .1 };                    ///< The simple integrator gain used for timer and synchro delay control.
    float m_nanoSecTarget{ 1e9f / m_fps }; ///< The timer-mode target interval in nanoseconds.
    float m_synchroDelay{ 0 };             ///< The commanded pre-read delay in synchronized mode, in nanoseconds.
    float m_synchroDelayTarget{
        0 }; ///< Selected delay target [ns], with user offset; only the legacy model wraps within a WFS period.

    /// The timer-mode reference point for the next internal acquisition cycle.
    std::chrono::time_point<std::chrono::steady_clock> m_time_start;

    /// The most recently read AD7689 channel values published to the output stream.
    std::vector<uint16_t> m_values;

    IMAGE  m_synchroStream{};             ///< The opened synchronization stream used for semaphore-triggered reads.
    bool   m_synchroStreamOpen{ false };  ///< Tracks whether the synchronization stream is currently open.
    ino_t  m_synchroStreamInode{ 0 };     ///< Cached inode used to detect synchronization stream recreation.
    int    m_synchroSemaphoreNumber{ 5 }; ///< The claimed semaphore slot for synchronization waits.
    sem_t *m_synchroSemaphore{ nullptr }; ///< Cached pointer to the claimed synchronization semaphore.

    timespec m_atime{}; ///< The most recent semaphore-arrival timestamp on the local realtime clock.

    timespec m_lastAtime{}; ///< The previous semaphore-arrival timestamp used for period estimation.

    double m_avgSemaphorePeriod_ns{ 0.0 }; ///< Exponential moving-average estimate of semaphore period in nanoseconds.

    double m_wfsPeriodMeasured_ns{
        0.0 }; ///< Measured semaphore period used as the synchronized delay-model WFS period.

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

    double m_avgReadLatency_ns{
        0.0 }; ///< Exponential moving-average estimate of semaphore-to-read latency in nanoseconds.

    bool m_firstReadLatency{ true }; ///< Tracks first-arrival initialization for semaphore-to-read latency estimation.

    double m_wfs_fps{
        0.0 }; ///< Startup/source WFS rate hint; measured unique triggers supersede it in synchronized mode.

    double m_delayModel_ns{
        0.0 }; ///< Selected phase-model delay [ns]; legacy wrapped phase or full DDSPC acquisition delay.

    double m_delayApplied_ns{ 0.0 }; ///< Delay applied to the most recent synchronized ADC read.

    double m_delayBudget_ns{
        0.0 }; ///< Selected controller upper bound [ns]; legacy cadence budget or DDSPC causal limit.

    double m_nonDelayService_ns{
        0.0 }; ///< Measured wake-to-return service time minus applied delay for the latest cycle.

    double m_avgNonDelayService_ns{ 0.0 }; ///< Exponential moving-average of non-delay synchronized service time.

    bool m_firstNonDelayService{ true }; ///< Tracks first-sample initialization for non-delay service-time averaging.

    double m_delayPhaseError_ns{ 0.0 }; ///< Wrapped phase error between applied and modeled synchronized delay.

    double m_delayLock{ 0.0 }; ///< Delay-lock state exported to diagnostics as 1.0 (locked) or 0.0 (unlocked).

    double m_delayCapped{ 0.0 }; ///< Delay-cap state exported as 1.0 when cadence budgeting limits the applied delay.

    timespec m_triggerTime{}; ///< Predicted synchronized scan deadline on the realtime diagnostic clock.

    double m_triggerInterval_ns{ 0.0 }; ///< Measured interval between consecutive trigger events in nanoseconds.

    double m_channelReadoutTime_ns{ 0.0 }; ///< Measured duration of reading all configured channels in nanoseconds.

    timespec m_lastTriggerTime{}; ///< Previous trigger timestamp used to compute the synchronized trigger interval.

    bool m_firstTriggerTime{
        true }; ///< Tracks first-trigger initialization for synchronized trigger-interval measurement.

    bool m_firstTimerTrigger{
        true }; ///< Tracks first-trigger initialization for timer-mode trigger-interval measurement.

    ///@}

    /** \name ADC Configuration - Data
     * @{
     */
    AD7689Lib::Settings   m_adcSettings;    ///< Startup-only ADC hardware settings.
    std::vector<unsigned> m_channels;       ///< Configured explicit physical input ordering; empty selects prefix mode.
    std::vector<unsigned> m_activeChannels; ///< Acquisition-owner input ordering for the current geometry.
    bool m_explicitChannels{ false };       ///< Count-only controls cannot alter an explicit physical mapping.
    std::vector<uint16_t> m_staging;        ///< Complete-scan scratch storage, never published on failure.
    bool                  m_adcConnected{ false }; ///< Acquisition-owner connection state, including recovery.
    bool                  m_frameValid{ false };   ///< One successful unpublished complete scan is available.
    unsigned           m_failureAttempts{ 0 };   ///< Consecutive failures without a successful frame; limited to three.
    uint64_t           m_invalidFrames{ 0 };     ///< Entire scans rejected for transport/control/stop changes.
    uint64_t           m_overruns{ 0 };          ///< Completed scans or sync requests exceeding the available cadence.
    uint64_t           m_missedPeriods{ 0 };     ///< Timer periods missed without fabricated replacement frames.
    uint64_t           m_usefulConversions{ 0 }; ///< Useful ADC values committed to complete output frames.
    uint64_t           m_recoveries{ 0 };        ///< Acquisition-owner reconnect attempts after an initial connection.
    std::atomic<float> m_reportedFps{ 2000 };    ///< Thread-safe active rate for framegrabber/main-loop readers.
    std::atomic<int>   m_reportedChannels{ 2 };  ///< Thread-safe active output width.
    ///@}

    /** \name Pending Controls - Data
     * @{
     */
    std::mutex m_controlMutex; ///< Protects requested controls and diagnostic snapshots, never held over hardware I/O.
    double     m_requestedFps{ 2000 };         ///< Validated target applied only at a frame boundary.
    int        m_requestedChannels{ 2 };       ///< Validated target output width, applied through reconfiguration.
    float      m_requestedAlpha{ 0.01f };      ///< Requested timing EMA coefficient.
    float      m_appliedAlpha{ 0.01f };        ///< Applied EMA coefficient, copied under the control mutex.
    int        m_appliedOffset{ 0 };           ///< Applied phase offset, copied under the control mutex.
    int        m_requestedOffset{ 0 };         ///< Requested signed synchronization phase offset in microseconds.
    std::array<double, 23> m_timingSnapshot{}; ///< Acquisition-owner copy of legacy timing diagnostics.
    std::array<double, 9>  m_healthSnapshot{}; ///< Acquisition-owner ADC and loss counters.
    pcf::IndiProperty      m_indiP_adcHealth;  ///< Read-only GPIO/SPI/conversion/recovery diagnostics.
    pcf::IndiProperty      m_indiP_channelMap; ///< Read-only output-slot to physical-channel mapping.
    ///@}

    /** \name Synchronization Helpers - Data
     * @{ */
    ad7689CtrlDetail::syncSchedule m_syncSchedule; ///< Finite acquisition-owner schedule for the current WFS period.
    double   m_referenceFps{ 0 };        ///< Optional startup WFS cadence hint; zero waits for two measured triggers.
    double   m_syncTimeout{ 1 };         ///< Source silence/recreation timeout [s], independent of ADC sampling rate.
    double   m_requestedWfsFps{ 0 };     ///< WFS cadence hint staged by the FPS source callback.
    int64_t  m_lastSyncArrival_ns{ 0 };  ///< Last unique WFS arrival on the monotonic clock.
    int64_t  m_nextSyncProbe_ns{ 0 };    ///< Next bounded stream identity check [monotonic ns].
    bool     m_haveSyncArrival{ false }; ///< Whether an interval can be measured from the previous WFS event.
    bool     m_haveMeasuredCadence{ false }; ///< Measured cadence supersedes the optional startup hint.
    uint64_t m_syncDuplicates{ 0 };          ///< Duplicate source IDs rejected without creating additional scan slots.
    uint64_t m_coalescedWakes{ 0 }; ///< Queued semaphore notifications collapsed into the latest source reference.
    uint64_t m_cadenceResets{ 0 };  ///< Source restarts, silence timeouts, and significant cadence changes.
    uint64_t m_lateScans{ 0 };      ///< Acquisitions starting after their absolute software deadlines.
    double   m_maxLateness_ns{ 0 }; ///< Maximum observed monotonic deadline lateness [ns].
    std::array<double, 11> m_syncSnapshot{};   ///< Synchronized scheduler diagnostics transferred to the main loop.
    pcf::IndiProperty      m_indiP_syncTiming; ///< Read-only actual cadence, subscan spacing, and loss accounting.
    ///@}

    /** \name DDSPC Acquisition - Data
     * @{ */
    ad7689CtrlDetail::ddspcSettings m_ddspc; ///< Acquisition-owner phase model; disabled preserves legacy behavior.
    ad7689CtrlDetail::ddspcSettings
        m_requestedDdspc; ///< Complete validated runtime request, protected by m_controlMutex.
    ad7689CtrlDetail::continuousSchedule m_ddspcSchedule; ///< One absolute clock with finite WFS slot authorization.
    ad7689CtrlDetail::scanSlot           m_ddspcSlot;     ///< Last consumed slot and its original WFS reference.
    double  m_ddspcTarget_ns{ 0 };                        ///< Unwrapped DDSPC formula, before user offset and feedback.
    double  m_ddspcReadError_ns{ 0 };         ///< Smoothed first-slot actual delay error for the existing controller.
    int64_t m_ddspcStart_ns{ 0 };             ///< Actual software scan invocation on the monotonic clock.
    int64_t m_ddspcCompletion_ns{ 0 };        ///< Complete scan returned; data not yet published [monotonic ns].
    int64_t m_ddspcPublished_ns{ 0 };         ///< Sample stream semaphores posted [monotonic ns].
    std::array<double, 18> m_ddspcSnapshot{}; ///< Last timing and source association transferred to the main loop.
    pcf::IndiProperty      m_indiP_ddspcSettings; ///< Runtime enabled/process/latency/margin/slew control.
    pcf::IndiProperty      m_indiP_ddspcTiming;   ///< Calculated phase, actual timestamps, and WFS/ACC association.
    /// Stage an atomic validated DDSPC settings update without touching acquisition hardware.
    INDI_NEWCALLBACK_DECL( ad7689Ctrl, m_indiP_ddspcSettings );
    ///@}

    /** \name ADC Configuration
     * @{
     */
    /// Report the active output width without racing reconfiguration.
    int numChannels();

    /// Parse one complete numeric INDI element; return NaN for malformed or missing values.
    static double controlValue( const pcf::IndiProperty &property /**< [in] incoming property */,
                                const std::string       &element /**< [in] requested numeric element */ );

    /// Preserve accepted controls and publish an INDI alert for a rejected request.
    int rejectControl( pcf::IndiProperty &property /**< [in/out] local control property */ );

    /// Stage a rate/count request after checking the combined ADC budget.
    int requestAcquisition( double rate /**< [in] requested per-channel/frame rate */,
                            double count /**< [in] requested integral output width */ );

    /// Apply requested controls in the acquisition thread; report whether geometry/rate changed.
    bool applyControls( bool reconfigure = false /**< [in] apply width/rate only while acquisition is stopped */ );

    /// Initialize/recover the hardware in the acquisition thread.
    int prepareAdc();

    /// Read an entire validated scan and invalidate it on stop or changed controls.
    int readScan();

    /// Capture acquisition-owned timing/counter fields for main-thread publication.
    void captureTimingDiagnostics();
    ///@}

    /** \name Synchronization Helpers
     * @{
     */

    /// Read monotonic nanoseconds for scan scheduling, independently of realtime stream timestamps.
    virtual int monotonicNow( int64_t &now /**< [out] current monotonic time [ns] */ );

    /// Wait for either a source notification or an absolute monotonic deadline.
    virtual int waitForSync( int64_t deadline /**< [in] absolute monotonic wake deadline [ns] */ );

    /// Track a coalesced WFS observation and update the selected acquisition clock.
    int processSyncReference( int64_t arrival /**< [in] unique reference arrival [monotonic ns] */ );

    /// Abandon pending slots and invalidate source cadence on stop, restart, or reconfiguration.
    void resetSyncSchedule();

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

    /// Consume one DDSPC clock slot, retaining its source association and distinct timing observations.
    int acquireDdspcAndCheckValid( int64_t now /**< [in] actual scan start [monotonic ns] */ );

    /// Read the current realtime clock value.
    /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
    virtual int getRealtime( timespec &ts /**< [out] the current realtime clock value */ );

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

    /// Apply the current controlled delay between semaphore wake and ADC read.
    void delayBeforeRead();

    /// Update the synchronized delay command with anti-windup and physical bounds.
    void
    updateSynchroDelayController( double desiredDelay_ns /**< [in] the unconstrained synchronized-delay command */ );

    /// Recompute the WFS phase model using the current measured cadence.
    void updatePhaseTarget( const timespec &atime /**< [in] latest source arrival [realtime] */ );

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
    ad7689Ctrl();

    /// Destroy the application.
    ~ad7689Ctrl() noexcept;

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

    /// Copy the current AD7689 values into the output image buffer.
    /**
     * \returns 0 on success
     * \returns -1 on error
     */
    int loadImageIntoStream( void *dest /**< [out] the destination image buffer */ );

    /// Record actual data availability after the unchanged frameGrabber stream semaphore posts.
    int frameGrabberPostPublish( IMAGE *stream /**< [in] just-published image stream */ );

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

inline ad7689Ctrl::~ad7689Ctrl() noexcept = default;

inline double ad7689Ctrl::controlValue( const pcf::IndiProperty &property, const std::string &element )
{
    try
    {
        std::istringstream input( property[element].get() );
        double             value = 0;

        if( ( input >> value ) && ( input >> std::ws ).eof() && std::isfinite( value ) )
        {
            return value;
        }
    }
    catch( const std::exception & )
    {
    }

    return std::numeric_limits<double>::quiet_NaN();
}

inline int ad7689Ctrl::rejectControl( pcf::IndiProperty &property )
{
    property.setState( INDI_ALERT );

    if( m_indiDriver )
    {
        m_indiDriver->sendSetProperty( property );
    }

    return -1;
}

inline int ad7689Ctrl::requestAcquisition( double rate, double count )
{
    const double margin = m_synchroShmimName.empty() ? 0 : m_cadenceGuard_ns;

    if( !std::isfinite( rate ) || rate <= 0 || rate > 10000 )
    {
        return log<software_error, -1>( { "Rejected nonfinite or out-of-range frame rate" } );
    }

    const float storedRate = static_cast<float>( rate );

    if( !std::isfinite( count ) || count < 1 || count > 8 || std::floor( count ) != count ||
        ( m_explicitChannels && count != static_cast<double>( m_channels.size() ) ) ||
        !AD7689Lib::AD7689::validRate( m_adcSettings, static_cast<size_t>( count ), rate, margin ) ||
        !AD7689Lib::AD7689::validRate( m_adcSettings, static_cast<size_t>( count ), storedRate, margin ) )
    {
        return log<software_error, -1>( { "Rejected invalid rate/count or ADC timing budget" } );
    }

    { //mutex scope
        std::lock_guard<std::mutex> lock( m_controlMutex );
        m_requestedFps      = storedRate;
        m_requestedChannels = static_cast<int>( count );
    }

    return 0;
}

inline bool ad7689Ctrl::applyControls( bool reconfigure )
{
    bool changed;
    { //mutex scope
        std::lock_guard<std::mutex> lock( m_controlMutex );
        changed = m_requestedFps != m_fps || m_requestedChannels != m_numChannels;

        if( reconfigure )
        {
            m_fps         = static_cast<float>( m_requestedFps );
            m_numChannels = m_requestedChannels;
        }

        m_wfs_fps       = m_synchroShmimName.empty() ? m_requestedFps : m_requestedWfsFps;
        m_alpha         = m_requestedAlpha;
        m_appliedAlpha  = m_alpha;
        m_appliedOffset = m_requestedOffset;

        if( m_ddspc != m_requestedDdspc )
        {
            const bool toggled = m_ddspc.m_enabled != m_requestedDdspc.m_enabled;

            if( toggled || m_requestedDdspc.m_enabled )
            {
                m_firstReadLatency = true;
            }

            m_ddspc = m_requestedDdspc;

            if( toggled )
            {
                m_syncSchedule.cancel();
                m_ddspcSchedule.cancel();
            }
        }

        if( m_ddspc.m_enabled && m_synchroPostDelay != m_requestedOffset )
        {
            m_synchroPostDelay = m_requestedOffset;
            m_firstReadLatency = true;
        }

        if( !m_ddspc.m_enabled && m_synchroPostDelay != m_requestedOffset )
        {
            m_synchroPostDelay = m_requestedOffset;

            if( m_wfsPeriodMeasured_ns > 0 )
            {
                double delay = std::fmod( m_delayModel_ns + 1000.0 * m_synchroPostDelay, m_wfsPeriodMeasured_ns );

                if( delay < 0 )
                {
                    delay += m_wfsPeriodMeasured_ns;
                }

                m_synchroDelayTarget = static_cast<float>( delay );
            }
            else
            {
                m_synchroDelayTarget = 1000.0f * m_synchroPostDelay;
            }

            m_synchroDelay    = m_synchroDelayTarget;
            m_delayApplied_ns = m_synchroDelay;
        }
    }

    if( changed && reconfigure )
    {
        m_trigger = m_nanoSecTarget = 1e9f / m_fps;

        if( m_synchroShmimName.empty() )
        {
            m_wfs_fps = m_fps;
        }
    }

    m_reportedFps      = m_fps;
    m_reportedChannels = m_numChannels;
    return changed;
}

inline int ad7689Ctrl::prepareAdc()
{
    if( m_failureAttempts >= 3 )
    {
        m_shutdown = true;
        state( stateCodes::ERROR );
        return log<software_error, -1>( { "ADC failed three consecutive acquisition/recovery attempts" } );
    }

    try
    {
        if( !m_adcConnected )
        {
            ++m_recoveries;
            elevatedPrivileges privilege( this );

            if( m_adc.connect( m_adcSettings ) < 0 )
            {
                throw std::runtime_error( m_adc.error() );
            }

            m_adcConnected = true;
        }

        if( m_adc.prepare( m_activeChannels ) < 0 )
        {
            throw std::runtime_error( m_adc.error() );
        }
    }
    catch( const std::exception &error )
    {
        ++m_failureAttempts;
        m_adc.disconnect();
        m_adcConnected = false;
        captureTimingDiagnostics();
        return log<software_error, -1>( { std::string( "ADC initialization: " ) + error.what() } );
    }

    return 0;
}

inline int ad7689Ctrl::readScan()
{
    const auto started = std::chrono::steady_clock::now();
    m_frameValid       = false;
    try
    {
        if( m_staging.size() != m_values.size() )
        {
            m_staging.resize( m_values.size() );
        }

        if( m_adc.readScan( m_staging.data(), m_staging.size(), m_currImageTimestamp ) < 0 )
        {
            throw std::runtime_error( m_adc.error() );
        }
    }
    catch( const std::exception &error )
    {
        ++m_invalidFrames;
        ++m_failureAttempts;
        m_adc.disconnect();
        m_adcConnected = false;
        m_reconfig     = true;
        return log<software_error, 1>( { std::string( "ADC scan discarded: " ) + error.what() } );
    }

    m_channelReadoutTime_ns = static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>( std::chrono::steady_clock::now() - started ).count() );

    if( m_channelReadoutTime_ns > 1e9 / m_fps )
    {
        ++m_overruns;
    }

    bool changed;
    { //mutex scope
        std::lock_guard<std::mutex> lock( m_controlMutex );
        changed = m_requestedFps != m_fps || m_requestedChannels != m_numChannels;
    }

    if( m_shutdown || m_reconfig || changed )
    {
        ++m_invalidFrames;

        if( changed )
        {
            m_reconfig = true;
        }

        return 1;
    }

    m_values.swap( m_staging );
    m_failureAttempts = 0;
    m_frameValid      = true;
    return 0;
}

inline ad7689Ctrl::ad7689Ctrl() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    return;
}

inline double ad7689Ctrl::timespecToNs( const timespec &t )
{
    return static_cast<double>( t.tv_sec ) * 1e9 + static_cast<double>( t.tv_nsec );
}

inline timespec ad7689Ctrl::nsToTimespec( double ns )
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

inline void ad7689Ctrl::updateTriggerTiming( const timespec &atime )
{
    double dt_ns = 0.0;

    if( !m_firstSemaphore )
    {
        dt_ns = timespecToNs( atime ) - timespecToNs( m_lastAtime );

        const double alpha      = static_cast<double>( m_alpha );
        m_avgSemaphorePeriod_ns = alpha * dt_ns + ( 1.0 - alpha ) * m_avgSemaphorePeriod_ns;
    }
    else
    {
        m_avgSemaphorePeriod_ns = 0.0;
        m_firstSemaphore        = false;
    }

    m_lastAtime = atime;

    updatePhaseTarget( atime );
}

inline void ad7689Ctrl::updatePhaseTarget( const timespec &atime )
{
    const double deltaT_wfs_ns = m_avgSemaphorePeriod_ns;
    m_wfsPeriodMeasured_ns     = deltaT_wfs_ns;

    if( deltaT_wfs_ns <= 0.0 )
    {
        m_triggerInterval_ns = 0.0;
        return;
    }

    if( m_ddspc.m_enabled )
    {
        m_ddspcTarget_ns     = m_ddspc.delay( deltaT_wfs_ns );
        m_delayModel_ns      = m_ddspcTarget_ns;
        m_synchroDelayTarget = static_cast<float>( m_ddspcTarget_ns + 1000.0 * m_synchroPostDelay );
        return; // Full DDSPC delay is never wrapped or added to the legacy latency model.
    }

    const double raw_delay_ns = 0.5 * deltaT_wfs_ns - ( m_synchroDtTransfer_ns + m_synchroWfsProcess_ns +
                                                        m_synchroDtF_ns + m_synchroWfsRead_ns );

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

inline void ad7689Ctrl::setupConfig()
{
    FRAMEGRABBER_SETUP_CONFIG( config );
    TELEMETER_SETUP_CONFIG( config );

    config.add( "accel.numChannels",
                "",
                "accel.numChannels",
                argType::Required,
                "accel",
                "numChannels",
                false,
                "int",
                "Number of prefix inputs, 1..8; default 2." );
    config.add( "accel.channels",
                "",
                "accel.channels",
                argType::Required,
                "accel",
                "channels",
                false,
                "string",
                "Optional ordered unique inputs, comma separated or bracketed; overrides the default count." );
    config.add( "accel.fps",
                "",
                "accel.fps",
                argType::Required,
                "accel",
                "fps",
                false,
                "double",
                "Initial per-enabled-channel/frame rate, default 2000 Hz." );
    config.add(
        "spi.bus", "", "spi.bus", argType::Required, "spi", "bus", false, "unsigned", "SPI controller, default 0." );
    config.add( "spi.channel",
                "",
                "spi.channel",
                argType::Required,
                "spi",
                "channel",
                false,
                "unsigned",
                "Spidev endpoint, default 0; not the analog input." );
    config.add( "spi.baud",
                "",
                "spi.baud",
                argType::Required,
                "spi",
                "baud",
                false,
                "unsigned",
                "Requested SPI clock, default 1000000 Hz." );
    config.add( "cnv.chip",
                "",
                "cnv.chip",
                argType::Required,
                "cnv",
                "chip",
                false,
                "string",
                "GPIO chip path, default /dev/gpiochip0; verify mapping." );
    config.add( "cnv.line",
                "",
                "cnv.line",
                argType::Required,
                "cnv",
                "line",
                false,
                "unsigned",
                "Chip-local CNV offset, default 8; GPIO8 must be free." );
    config.add( "adc.conversionTime_ns",
                "",
                "adc.conversionTime_ns",
                argType::Required,
                "adc",
                "conversionTime_ns",
                false,
                "uint64",
                "CNV high hold, default 10000 ns, minimum 3200 ns." );
    config.add( "adc.spiSetup_ns",
                "",
                "adc.spiSetup_ns",
                argType::Required,
                "adc",
                "spiSetup_ns",
                false,
                "uint64",
                "CNV-low to SPI guard, default 1000 ns." );
    config.add( "adc.quietTime_ns",
                "",
                "adc.quietTime_ns",
                argType::Required,
                "adc",
                "quietTime_ns",
                false,
                "uint64",
                "Serial quiet guard before CNV high, default 1000 ns." );
    config.add( "adc.vref",
                "",
                "adc.vref",
                argType::Required,
                "adc",
                "vref",
                false,
                "double",
                "External REF calibration voltage, default 3.32 V." );

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

    config.add( "synchro.referenceFps",
                "",
                "synchro.referenceFps",
                argType::Required,
                "synchro",
                "referenceFps",
                false,
                "double",
                "Optional initial WFS rate [Hz]; zero learns cadence from triggers. Does not set ADC fps." );
    config.add( "synchro.timeout",
                "",
                "synchro.timeout",
                argType::Required,
                "synchro",
                "timeout",
                false,
                "double",
                "Maximum WFS silence before rechecking the source [s]." );

    config.add( "synchro.ddspcDelayEnabled",
                "",
                "synchro.ddspcDelayEnabled",
                argType::Required,
                "synchro",
                "ddspcDelayEnabled",
                false,
                "bool",
                "Enable continuous DDSPC phase-aligned scheduling; default false." );
    config.add( "synchro.wfsProcessTime_ns",
                "",
                "synchro.wfsProcessTime_ns",
                argType::Required,
                "synchro",
                "wfsProcessTime_ns",
                false,
                "double",
                "Processing after WFS exposure ends [ns]; DDSPC only, default 75000." );
    config.add( "synchro.accLatency_ns",
                "",
                "synchro.accLatency_ns",
                argType::Required,
                "synchro",
                "accLatency_ns",
                false,
                "double",
                "Calibrated complete ACC-to-publication latency [ns]; DDSPC only, default 15000." );
    config.add( "synchro.ddspcMargin_ns",
                "",
                "synchro.ddspcMargin_ns",
                argType::Required,
                "synchro",
                "ddspcMargin_ns",
                false,
                "double",
                "Desired availability lead before DDSPC [ns]; default 10000." );
    config.add( "synchro.ddspcSlewFraction",
                "",
                "synchro.ddspcSlewFraction",
                argType::Required,
                "synchro",
                "ddspcSlewFraction",
                false,
                "double",
                "Maximum correction per future ACC interval, (0,0.25]; default 0.05." );

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

    config.add(
        "synchro.delayLockAbsThreshold_ns",
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

inline int ad7689Ctrl::loadConfigImpl( mx::app::appConfigurator &_config )
{

    FRAMEGRABBER_LOAD_CONFIG( _config );
    TELEMETER_LOAD_CONFIG( _config );

    _config( m_fpsDevice, "fps.device" );
    _config( m_fpsProperty, "fps.property" );
    _config( m_fpsElement, "fps.element" );
    _config( m_fpsTol, "fps.tol" );
    _config( m_synchroShmimName, "synchro.shmimName" );
    _config( m_referenceFps, "synchro.referenceFps" );
    _config( m_syncTimeout, "synchro.timeout" );
    _config( m_ddspc.m_enabled, "synchro.ddspcDelayEnabled" );
    _config( m_ddspc.m_process, "synchro.wfsProcessTime_ns" );
    _config( m_ddspc.m_latency, "synchro.accLatency_ns" );
    _config( m_ddspc.m_margin, "synchro.ddspcMargin_ns" );
    _config( m_ddspc.m_slew, "synchro.ddspcSlewFraction" );
    _config( m_synchroPostDelay, "synchro.postDelay" );
    _config( m_synchroDtTransfer_ns, "synchro.dtTransfer_ns" );
    _config( m_synchroWfsProcess_ns, "synchro.wfsProcess_ns" );
    _config( m_synchroDtF_ns, "synchro.dtF_ns" );
    _config( m_synchroWfsRead_ns, "synchro.wfsRead_ns" );
    _config( m_delayLockAbsThreshold_ns, "synchro.delayLockAbsThreshold_ns" );
    _config( m_delayLockFracThreshold, "synchro.delayLockFracThreshold" );
    _config( m_cadenceGuard_ns, "synchro.cadenceGuard_ns" );
    _config( m_alpha, "synchro.alpha" );

    double channelCount = m_numChannels;
    _config( channelCount, "accel.numChannels" );
    _config( m_fps, "accel.fps" );
    _config( m_adcSettings.m_bus, "spi.bus" );
    _config( m_adcSettings.m_device, "spi.channel" );
    _config( m_adcSettings.m_baud, "spi.baud" );
    _config( m_adcSettings.m_chip, "cnv.chip" );
    _config( m_adcSettings.m_line, "cnv.line" );
    _config( m_adcSettings.m_conversionTime_ns, "adc.conversionTime_ns" );
    _config( m_adcSettings.m_spiSetup_ns, "adc.spiSetup_ns" );
    _config( m_adcSettings.m_quietTime_ns, "adc.quietTime_ns" );
    _config( m_adcSettings.m_vref, "adc.vref" );

    if( !std::isfinite( m_referenceFps ) || m_referenceFps < 0 || m_referenceFps > 1000000 ||
        !std::isfinite( m_syncTimeout ) || m_syncTimeout <= 0 || m_syncTimeout > 60 )
    {
        return log<software_error, -1>( { "Invalid WFS reference rate or timeout" } );
    }

    if( !m_ddspc.valid() || ( m_ddspc.m_enabled && m_synchroShmimName.empty() ) )
    {
        return log<software_error, -1>( { "Invalid DDSPC settings or missing WFS stream" } );
    }

    if( m_ddspc.m_enabled && m_referenceFps > 0 )
    {
        const double delay = m_ddspc.delay( 1e9 / m_referenceFps, 1000.0 * m_synchroPostDelay );

        if( !std::isfinite( delay ) || delay < 0 || delay > 1e9 )
        {
            return log<software_error, -1>( { "DDSPC acquisition delay must be within [0,1e9] ns" } );
        }
    }

    m_requestedDdspc   = m_ddspc;
    m_explicitChannels = _config.isSet( "accel.channels" );
    m_channels.clear();

    if( m_explicitChannels )
    {
        std::string list;
        _config( list, "accel.channels" );

        if( list.size() >= 2 && list.front() == '[' && list.back() == ']' )
        {
            list = list.substr( 1, list.size() - 2 );
        }

        std::istringstream input( list );
        std::string        token;

        while( std::getline( input, token, ',' ) )
        {
            std::istringstream number( token );
            int                channel = -1;

            if( !( number >> channel ) || !( number >> std::ws ).eof() || channel < 0 || channel > 7 )
            {
                return log<software_error, -1>( { "Invalid accel.channels input" } );
            }

            m_channels.push_back( static_cast<unsigned>( channel ) );
        }

        if( !AD7689Lib::AD7689::validChannels( m_channels ) || list.empty() || list.back() == ',' ||
            ( _config.isSet( "accel.numChannels" ) && channelCount != static_cast<double>( m_channels.size() ) ) )
        {
            return log<software_error, -1>( { "Invalid or conflicting accel.channels/count" } );
        }

        channelCount = m_channels.size();
    }

    if( !std::isfinite( channelCount ) || channelCount < 1 || channelCount > 8 ||
        std::floor( channelCount ) != channelCount )
    {
        return log<software_error, -1>( { "accel.numChannels must be an integer in 1..8" } );
    }

    m_numChannels = static_cast<int>( channelCount );
    _config( m_numChannelsDevice, "numChannels.device" );
    _config( m_numChannelsProperty, "numChannels.property" );
    _config( m_numChannelsElement, "numChannels.element" );
    _config( m_numChannelsTol, "numChannels.tol" );

    _config( m_fgCpuset, "framegrabber.cpuset" );

    for( double timing : { m_synchroDtTransfer_ns,
                           m_synchroWfsProcess_ns,
                           m_synchroDtF_ns,
                           m_synchroWfsRead_ns,
                           m_delayLockAbsThreshold_ns,
                           m_delayLockFracThreshold,
                           m_cadenceGuard_ns,
                           static_cast<double>( m_alpha ) } )
    {
        if( !std::isfinite( timing ) )
        {
            return log<software_error, -1>( { "Nonfinite timing configuration" } );
        }
    }

    const double margin = m_synchroShmimName.empty() ? 0 : std::max( 0.0, m_cadenceGuard_ns );

    if( !AD7689Lib::AD7689::validRate( m_adcSettings, m_numChannels, m_fps, margin ) )
    {
        return log<software_error, -1>( { "Invalid ADC configuration or impossible rate/channel/SPI budget" } );
    }

    if( m_ddspc.m_enabled && m_referenceFps > 0 )
    {
        ad7689CtrlDetail::continuousSchedule admission;
        const double                         period = 1e9 / m_referenceFps;
        const double                         delay  = m_ddspc.delay( period, 1000.0 * m_synchroPostDelay );

        if( !admission.reference( 0,
                                  0,
                                  period,
                                  m_fps,
                                  delay,
                                  delay,
                                  AD7689Lib::AD7689::scanTime_ns( m_adcSettings, m_numChannels ),
                                  margin,
                                  m_ddspc.m_slew ) )
        {
            return log<software_error, -1>( { "DDSPC effective cadence exceeds ADC/guard budget or rate limit" } );
        }
    }

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

    m_synchroDelayTarget      = 1e3f * m_synchroPostDelay;
    m_synchroDelay            = m_synchroDelayTarget;
    m_delayModel_ns           = 0.0;
    m_delayApplied_ns         = static_cast<double>( m_synchroDelayTarget );
    m_delayBudget_ns          = 0.0;
    m_nonDelayService_ns      = 0.0;
    m_avgNonDelayService_ns   = 0.0;
    m_firstNonDelayService    = true;
    m_delayPhaseError_ns      = 0.0;
    m_delayLock               = 0.0;
    m_delayCapped             = 0.0;
    m_wfsPeriodMeasured_ns    = 0.0;
    m_lastProducerAtime       = timespec{};
    m_lastProducerCnt0        = 0;
    m_producerPeriodInst_ns   = 0.0;
    m_avgProducerPeriod_ns    = 0.0;
    m_firstProducerSample     = true;
    m_localFrameSeq           = 0;
    m_syncFramesReceived      = 0;
    m_syncFramesWritten       = 0;
    m_syncFramesDropped       = 0;
    m_syncFrameIdGapCount     = 0;
    m_syncProducerFrameId     = 0;
    m_syncProducerFrameDelta  = 0;
    m_lastSyncProducerFrameId = 0;
    m_syncProducerFrameValid  = false;
    m_wfs_fps                 = m_synchroShmimName.empty() ? m_fps : m_referenceFps;
    m_requestedWfsFps         = m_wfs_fps;
    m_channelReadoutTime_ns   = 0.0;

    m_requestedFps      = m_fps;
    m_requestedChannels = m_numChannels;
    m_requestedAlpha    = m_alpha;
    m_requestedOffset   = m_synchroPostDelay;
    m_appliedAlpha      = m_alpha;
    m_appliedOffset     = m_synchroPostDelay;
    m_reportedFps       = m_fps;
    m_reportedChannels  = m_numChannels;
    m_trigger = m_nanoSecTarget = 1e9f / m_fps;
    return 0;
}

inline void ad7689Ctrl::loadConfig()
{
    try
    {
        if( loadConfigImpl( config ) < 0 )
        {
            m_shutdown = true;
        }
    }
    catch( const std::exception &error )
    {
        log<software_error>( { std::string( "ADC configuration: " ) + error.what() } );
        m_shutdown = true;
    }
}

inline int ad7689Ctrl::appStartup()
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
    CREATE_REG_INDI_NEW_NUMBERF( m_indiP_numChannels, "numChannels", 1, 8, 1, "%d", "", "" );
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

        if( m_adc.connect( m_adcSettings ) < 0 )
        {
            return log<software_error, -1>( { m_adc.error() } );
        }

        m_adcConnected = true;
    }

    CREATE_REG_INDI_RO_NUMBER( m_indiP_adcHealth, "adcHealth", "ADC health", "Diagnostics" );

    for( const char *name : { "conversions",
                              "warmup_discarded",
                              "hardware_errors",
                              "invalid_frames",
                              "overruns",
                              "missed_periods",
                              "recoveries",
                              "scan_us",
                              "useful_conversions" } )
    {
        m_indiP_adcHealth.add( pcf::IndiElement( name ) );
    }

    m_indiP_channelMap = pcf::IndiProperty( pcf::IndiProperty::Text );
    m_indiP_channelMap.setDevice( configName() );
    m_indiP_channelMap.setName( "channelMap" );
    m_indiP_channelMap.setPerm( pcf::IndiProperty::ReadOnly );
    m_indiP_channelMap.add( pcf::IndiElement( "inputs" ) );

    if( registerIndiPropertyNew( m_indiP_channelMap, nullptr ) < 0 )
    {
        return -1;
    }

    CREATE_REG_INDI_RO_NUMBER( m_indiP_syncTiming, "syncTiming", "WFS scan scheduler", "Diagnostics" );

    for( const char *name : { "scans_per_wfs",
                              "scheduled_fps",
                              "scan_period_us",
                              "wfs_period_us",
                              "phase_us",
                              "skipped_scans",
                              "late_scans",
                              "max_lateness_us",
                              "cadence_resets",
                              "duplicate_triggers",
                              "coalesced_wakes" } )
    {
        m_indiP_syncTiming.add( pcf::IndiElement( name ) );
    }

    CREATE_REG_INDI_RO_NUMBER( m_indiP_ddspcTiming, "ddspcTiming", "DDSPC acquisition timing", "Diagnostics" );

    for( const char *name : { "enabled",
                              "valid",
                              "target_delay_us",
                              "applied_delay_us",
                              "sync_correction_us",
                              "phase_error_us",
                              "scheduled_mono_us",
                              "start_mono_us",
                              "completion_mono_us",
                              "published_mono_us",
                              "reference_mono_us",
                              "reference_id",
                              "reference_slot",
                              "nominal_slot",
                              "frame_seq",
                              "start_lateness_us",
                              "availability_us",
                              "phase_acquisitions" } )
    {
        m_indiP_ddspcTiming.add( pcf::IndiElement( name ) );
    }

    m_indiP_ddspcSettings = pcf::IndiProperty( pcf::IndiProperty::Number );
    m_indiP_ddspcSettings.setDevice( configName() );
    m_indiP_ddspcSettings.setName( "ddspcSettings" );
    m_indiP_ddspcSettings.setPerm( pcf::IndiProperty::ReadWrite );

    for( const char *name : { "enabled", "wfs_process_ns", "acc_latency_ns", "margin_ns", "slew_fraction" } )
    {
        m_indiP_ddspcSettings.add( pcf::IndiElement( name ) );
    }

    if( registerIndiPropertyNew( m_indiP_ddspcSettings, st_newCallBack_m_indiP_ddspcSettings ) < 0 )
    {
        return -1;
    }

    captureTimingDiagnostics();
    updateTimingDiagnosticsIndi();

    state( stateCodes::OPERATING );
    return 0;
}

inline void ad7689Ctrl::captureTimingDiagnostics()
{
    constexpr double c_timerModeCode   = 0.0;
    constexpr double c_synchroModeCode = 1.0;
    constexpr double c_nsToUs          = 1e-3;

    const bool   synchroMode           = !m_synchroShmimName.empty();
    const double readLatencyError_ns   = m_avgReadLatency_ns - static_cast<double>( m_synchroDelayTarget );
    const double modeCode              = synchroMode ? c_synchroModeCode : c_timerModeCode;
    const double delayAppliedDiag_ns   = synchroMode ? m_delayApplied_ns : 0.0;
    const double delayModelDiag_ns     = synchroMode ? m_delayModel_ns : 0.0;
    const double delayBudgetDiag_ns    = synchroMode ? m_delayBudget_ns : 0.0;
    const double avgNonDelayService_ns = synchroMode ? m_avgNonDelayService_ns : 0.0;
    const double delayCappedDiag       = synchroMode ? m_delayCapped : 0.0;
    const double wfsPeriodMeasured_ns  = synchroMode ? m_wfsPeriodMeasured_ns : 0.0;

    double triggerTime_ns = 0.0;

    if( ( m_atime.tv_sec != 0 || m_atime.tv_nsec != 0 ) && ( m_triggerTime.tv_sec != 0 || m_triggerTime.tv_nsec != 0 ) )
    {
        triggerTime_ns = timespecToNs( m_triggerTime ) - timespecToNs( m_atime );

        if( triggerTime_ns < 0.0 )
        {
            triggerTime_ns = 0.0;
        }
    }

    double delayPhaseError_ns = 0.0;
    double delayLock          = 0.0;

    if( synchroMode && m_ddspc.m_enabled && m_ddspcSchedule.m_active )
    {
        delayPhaseError_ns = m_ddspcSchedule.m_applied - m_ddspcSchedule.m_target;
        delayLock          = std::abs( delayPhaseError_ns ) <= m_delayLockAbsThreshold_ns &&
                             std::abs( delayPhaseError_ns ) <= m_delayLockFracThreshold * m_ddspcSchedule.m_interval;
    }
    else if( synchroMode && !m_ddspc.m_enabled && wfsPeriodMeasured_ns > 0.0 )
    {
        const double wfsPeriod_ns = m_syncSchedule.m_count ? m_syncSchedule.m_interval : wfsPeriodMeasured_ns;

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

        if( m_syncSchedule.m_count > 0 && absDelayPhaseError_ns <= m_delayLockAbsThreshold_ns &&
            absDelayPhaseError_ns <= m_delayLockFracThreshold * wfsPeriod_ns )
        {
            delayLock = 1.0;
        }
    }

    m_delayPhaseError_ns = delayPhaseError_ns;
    m_delayLock          = delayLock;

    const double avgReadLatency_us       = m_avgReadLatency_ns * c_nsToUs;
    const double synchroDelay_us         = ( synchroMode ? static_cast<double>( m_synchroDelay ) : 0.0 ) * c_nsToUs;
    const double synchroDelayTarget_us   = static_cast<double>( m_synchroDelayTarget ) * c_nsToUs;
    const double delayAppliedDiag_us     = delayAppliedDiag_ns * c_nsToUs;
    const double delayModelDiag_us       = delayModelDiag_ns * c_nsToUs;
    const double delayPhaseErrorDiag_us  = m_delayPhaseError_ns * c_nsToUs;
    const double delayBudgetDiag_us      = delayBudgetDiag_ns * c_nsToUs;
    const double avgNonDelayService_us   = avgNonDelayService_ns * c_nsToUs;
    const double readLatencyError_us     = readLatencyError_ns * c_nsToUs;
    const double avgSemaphorePeriod_us   = m_avgSemaphorePeriod_ns * c_nsToUs;
    const double channelReadout_us       = m_channelReadoutTime_ns * c_nsToUs;
    const double triggerInterval_us      = m_triggerInterval_ns * c_nsToUs;
    const double triggerTime_us          = triggerTime_ns * c_nsToUs;
    const double localFrameSeqDiag       = static_cast<double>( m_localFrameSeq );
    const double syncFramesReceivedDiag  = synchroMode ? static_cast<double>( m_syncFramesReceived ) : 0.0;
    const double syncFramesWrittenDiag   = synchroMode ? static_cast<double>( m_syncFramesWritten ) : 0.0;
    const double syncFramesDroppedDiag   = synchroMode ? static_cast<double>( m_syncFramesDropped ) : 0.0;
    const double syncFrameIdGapCountDiag = synchroMode ? static_cast<double>( m_syncFrameIdGapCount ) : 0.0;
    const double syncProducerFrameIdDiag =
        ( synchroMode && m_syncProducerFrameValid ) ? static_cast<double>( m_syncProducerFrameId ) : 0.0;
    const double syncProducerFrameDeltaDiag =
        ( synchroMode && m_syncProducerFrameValid ) ? static_cast<double>( m_syncProducerFrameDelta ) : 0.0;

    { //mutex scope
        std::lock_guard<std::mutex> lock( m_controlMutex );
        m_timingSnapshot  = { avgReadLatency_us,
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
                              modeCode };
        const auto &stats = m_adc.statistics();
        m_healthSnapshot  = {
            static_cast<double>( stats.m_conversions ),
            static_cast<double>( stats.m_discarded ),
            static_cast<double>( stats.m_errors ),
            static_cast<double>( m_invalidFrames ),
            static_cast<double>( m_overruns ),
            static_cast<double>( m_missedPeriods + m_syncSchedule.m_skipped + m_ddspcSchedule.m_skipped ),
            static_cast<double>( m_recoveries ),
            m_channelReadoutTime_ns * 1e-3,
            static_cast<double>( m_usefulConversions ) };
        const bool ddspc = synchroMode && m_ddspc.m_enabled;
        m_syncSnapshot   = { static_cast<double>( ddspc ? m_ddspcSchedule.m_count : m_syncSchedule.m_count ),
                             ddspc ? m_ddspcSchedule.fps() : m_syncSchedule.fps(),
                             ( ddspc ? m_ddspcSchedule.m_interval : m_syncSchedule.m_interval ) * 1e-3,
                             m_wfsPeriodMeasured_ns * 1e-3,
                             ( ddspc ? m_synchroDelay : m_syncSchedule.m_phase ) * 1e-3,
                             static_cast<double>( m_syncSchedule.m_skipped + m_ddspcSchedule.m_skipped ),
                             static_cast<double>( m_lateScans ),
                             m_maxLateness_ns * 1e-3,
                             static_cast<double>( m_cadenceResets ),
                             static_cast<double>( m_syncDuplicates ),
                             static_cast<double>( m_coalescedWakes ) };
        const auto &slot = m_ddspcSlot;
        m_ddspcSnapshot  = { ddspc ? 1.0 : 0.0,
                             ddspc && m_ddspcSchedule.m_active ? 1.0 : 0.0,
                             m_ddspcTarget_ns * 1e-3,
                             m_delayApplied_ns * 1e-3,
                             ( m_synchroDelay - m_synchroDelayTarget ) * 1e-3,
                             ( m_ddspcSchedule.m_applied - m_ddspcSchedule.m_target ) * 1e-3,
                             slot.m_deadline * 1e-3,
                             m_ddspcStart_ns * 1e-3,
                             m_ddspcCompletion_ns * 1e-3,
                             m_ddspcPublished_ns * 1e-3,
                             slot.m_reference.m_arrival * 1e-3,
                             static_cast<double>( slot.m_reference.m_id ),
                             static_cast<double>( slot.m_slot ),
                             static_cast<double>( slot.m_index ),
                             static_cast<double>( m_localFrameSeq ),
                             m_ddspcStart_ns ? ( m_ddspcStart_ns - slot.m_deadline ) * 1e-3 : 0.0,
                             m_ddspcPublished_ns ? ( m_ddspcPublished_ns - m_ddspcStart_ns ) * 1e-3 : 0.0,
                             static_cast<double>( m_ddspcSchedule.m_epochs ) };
    }
}

inline void ad7689Ctrl::updateTimingDiagnosticsIndi()
{
    std::array<double, 23>          values;
    std::array<double, 9>           health;
    std::array<double, 11>          sync;
    std::array<double, 18>          ddspc;
    ad7689CtrlDetail::ddspcSettings settings;
    { //mutex scope
        std::lock_guard<std::mutex> lock( m_controlMutex );
        values   = m_timingSnapshot;
        health   = m_healthSnapshot;
        sync     = m_syncSnapshot;
        ddspc    = m_ddspcSnapshot;
        settings = m_ddspc;
    }

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
                              { values[0],  values[1],  values[2],  values[3],  values[4],  values[5],
                                values[6],  values[7],  values[8],  values[9],  values[10], values[11],
                                values[12], values[13], values[14], values[15], values[16], values[17],
                                values[18], values[19], values[20], values[21], values[22] } );

    if( m_indiP_adcHealth.find( "conversions" ) )
    {
        updatesIfChanged<double>(
            m_indiP_adcHealth,
            { "conversions",
              "warmup_discarded",
              "hardware_errors",
              "invalid_frames",
              "overruns",
              "missed_periods",
              "recoveries",
              "scan_us",
              "useful_conversions" },
            { health[0], health[1], health[2], health[3], health[4], health[5], health[6], health[7], health[8] } );
    }

    if( m_indiP_syncTiming.find( "scans_per_wfs" ) )
    {
        updatesIfChanged<double>(
            m_indiP_syncTiming,
            { "scans_per_wfs",
              "scheduled_fps",
              "scan_period_us",
              "wfs_period_us",
              "phase_us",
              "skipped_scans",
              "late_scans",
              "max_lateness_us",
              "cadence_resets",
              "duplicate_triggers",
              "coalesced_wakes" },
            { sync[0], sync[1], sync[2], sync[3], sync[4], sync[5], sync[6], sync[7], sync[8], sync[9], sync[10] } );
    }
    if( m_indiP_ddspcTiming.find( "enabled" ) )
    {
        updatesIfChanged<double>( m_indiP_ddspcTiming,
                                  { "enabled",
                                    "valid",
                                    "target_delay_us",
                                    "applied_delay_us",
                                    "sync_correction_us",
                                    "phase_error_us",
                                    "scheduled_mono_us",
                                    "start_mono_us",
                                    "completion_mono_us",
                                    "published_mono_us",
                                    "reference_mono_us",
                                    "reference_id",
                                    "reference_slot",
                                    "nominal_slot",
                                    "frame_seq",
                                    "start_lateness_us",
                                    "availability_us",
                                    "phase_acquisitions" },
                                  { ddspc[0],
                                    ddspc[1],
                                    ddspc[2],
                                    ddspc[3],
                                    ddspc[4],
                                    ddspc[5],
                                    ddspc[6],
                                    ddspc[7],
                                    ddspc[8],
                                    ddspc[9],
                                    ddspc[10],
                                    ddspc[11],
                                    ddspc[12],
                                    ddspc[13],
                                    ddspc[14],
                                    ddspc[15],
                                    ddspc[16],
                                    ddspc[17] } );
    }

    if( m_indiP_ddspcSettings.find( "enabled" ) )
    {
        updatesIfChanged<double>( m_indiP_ddspcSettings,
                                  { "enabled", "wfs_process_ns", "acc_latency_ns", "margin_ns", "slew_fraction" },
                                  { settings.m_enabled ? 1.0 : 0.0,
                                    settings.m_process,
                                    settings.m_latency,
                                    settings.m_margin,
                                    settings.m_slew },
                                  m_indiP_ddspcSettings.getState() == INDI_ALERT ? INDI_ALERT : INDI_OK );
    }
}

inline int ad7689Ctrl::appLogic()
{
    FRAMEGRABBER_APP_LOGIC;
    TELEMETER_APP_LOGIC;
    FRAMEGRABBER_UPDATE_INDI;
    double rate;
    int    count, offset, currentOffset;
    float  alpha, currentAlpha;
    { //mutex scope
        std::lock_guard<std::mutex> lock( m_controlMutex );
        rate          = m_requestedFps;
        count         = m_requestedChannels;
        alpha         = m_requestedAlpha;
        offset        = m_requestedOffset;
        currentAlpha  = m_appliedAlpha;
        currentOffset = m_appliedOffset;
    }

    updatesIfChanged<double>( m_indiP_fps,
                              { "current", "target" },
                              { fps(), rate },
                              m_indiP_fps.getState() == INDI_ALERT ? INDI_ALERT
                              : fps() == rate                      ? INDI_OK
                                                                   : INDI_BUSY );
    updatesIfChanged<int>( m_indiP_numChannels,
                           { "current", "target" },
                           { numChannels(), count },
                           m_indiP_numChannels.getState() == INDI_ALERT ? INDI_ALERT
                           : numChannels() == count                     ? INDI_OK
                                                                        : INDI_BUSY );
    updatesIfChanged<float>( m_indiP_alpha,
                             { "current", "target" },
                             { currentAlpha, alpha },
                             m_indiP_alpha.getState() == INDI_ALERT ? INDI_ALERT
                             : currentAlpha == alpha                ? INDI_OK
                                                                    : INDI_BUSY );
    updatesIfChanged<int>( m_indiP_synchroDelay,
                           { "current", "target" },
                           { currentOffset, offset },
                           m_indiP_synchroDelay.getState() == INDI_ALERT ? INDI_ALERT
                           : currentOffset == offset                     ? INDI_OK
                                                                         : INDI_BUSY );
    std::string mapping;

    for( int i = 0; i < numChannels(); ++i )
    {
        if( i )
        {
            mapping += ',';
        }

        mapping += std::to_string( m_explicitChannels ? m_channels[i] : static_cast<unsigned>( i ) );
    }

    updateIfChanged( m_indiP_channelMap, "inputs", mapping );
    updateTimingDiagnosticsIndi();
    return 0;
}

inline int ad7689Ctrl::appShutdown()
{
    FRAMEGRABBER_APP_SHUTDOWN;
    TELEMETER_APP_SHUTDOWN;

    closeSynchroStream();

    if( m_adc.disconnect() < 0 )
    {
        return log<software_error, -1>( { "ADC cleanup failed" } );
    }

    m_adcConnected = false;
    return 0;
}

inline int ad7689Ctrl::configureAcquisition()
{
    applyControls( true );

    if( m_shutdown )
    {
        return 1;
    }

    m_values.resize( m_numChannels );
    m_staging.resize( m_numChannels );
    m_activeChannels = m_channels;

    if( !m_explicitChannels )
    {
        m_activeChannels.resize( m_numChannels );

        for( int i = 0; i < m_numChannels; ++i )
        {
            m_activeChannels[i] = i;
        }
    }

    if( prepareAdc() < 0 )
    {
        return 1;
    }

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

inline int ad7689Ctrl::numChannels()
{
    return m_reportedChannels.load();
}

inline float ad7689Ctrl::fps()
{
    return m_reportedFps.load();
}

inline int ad7689Ctrl::startAcquisition()
{
    resetSyncSchedule();

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

        m_synchroDelay            = m_synchroDelayTarget;
        m_firstSemaphore          = true;
        m_avgSemaphorePeriod_ns   = 0.0;
        m_firstReadLatency        = true;
        m_avgReadLatency_ns       = 0.0;
        m_lastAtime               = timespec{};
        m_atime                   = timespec{};
        m_triggerTime             = timespec{};
        m_wfsPeriodMeasured_ns    = 0.0;
        m_lastProducerAtime       = timespec{};
        m_lastProducerCnt0        = 0;
        m_producerPeriodInst_ns   = 0.0;
        m_avgProducerPeriod_ns    = 0.0;
        m_firstProducerSample     = true;
        m_syncFramesReceived      = 0;
        m_syncFramesWritten       = 0;
        m_syncFramesDropped       = 0;
        m_syncFrameIdGapCount     = 0;
        m_syncProducerFrameId     = 0;
        m_syncProducerFrameDelta  = 0;
        m_lastSyncProducerFrameId = 0;
        m_syncProducerFrameValid  = false;
        m_delayModel_ns           = 0.0;
        m_delayApplied_ns         = 0.0;
        m_delayBudget_ns          = 0.0;
        m_nonDelayService_ns      = 0.0;
        m_avgNonDelayService_ns   = 0.0;
        m_firstNonDelayService    = true;
        m_delayPhaseError_ns      = 0.0;
        m_delayLock               = 0.0;
        m_delayCapped             = 0.0;
    }

    m_triggerInterval_ns    = 0.0;
    m_channelReadoutTime_ns = 0.0;
    m_lastTriggerTime       = timespec{};
    m_firstTriggerTime      = true;
    m_firstTimerTrigger     = true;
    m_localFrameSeq         = 0;
    m_delayApplied_ns       = 0.0;
    m_delayBudget_ns        = 0.0;
    m_nonDelayService_ns    = 0.0;
    m_delayPhaseError_ns    = 0.0;
    m_delayLock             = 0.0;
    m_delayCapped           = 0.0;

    m_time_start = std::chrono::steady_clock::now();

    return 0;
}

inline int ad7689Ctrl::acquireAndCheckValid()
{
    m_frameValid = false;

    if( m_shutdown || m_reconfig )
    {
        return 1;
    }

    if( applyControls() )
    {
        m_reconfig = true;
        return 1;
    }

    const int result = m_synchroShmimName.empty() ? acquireTimerAndCheckValid() : acquireSynchroAndCheckValid();
    captureTimingDiagnostics();
    return result;
}

inline int ad7689Ctrl::loadImageIntoStream( void *dest )
{
    if( !dest || !m_frameValid || m_shutdown || m_reconfig )
    {
        return -1;
    }

    m_frameValid = false;
    memcpy( dest, m_values.data(), m_values.size() * sizeof( uint16_t ) );
    ++m_localFrameSeq;
    m_usefulConversions += m_values.size();

    if( !m_synchroShmimName.empty() )
    {
        ++m_syncFramesWritten;
    }

    captureTimingDiagnostics();
    return 0;
}

inline int ad7689Ctrl::frameGrabberPostPublish( IMAGE * )
{
    if( !m_ddspc.m_enabled )
    {
        return 0;
    }

    if( monotonicNow( m_ddspcPublished_ns ) < 0 )
    {
        return -1;
    }

    captureTimingDiagnostics();
    return 0;
}

inline int ad7689Ctrl::reconfig()
{
    m_frameValid = false;
    closeSynchroStream();
    return 0;
}

inline int ad7689Ctrl::openSynchroStream()
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

inline int ad7689Ctrl::claimSynchroSemaphore()
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

inline bool ad7689Ctrl::synchroStreamStale()
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

inline void ad7689Ctrl::closeSynchroStream()
{
    resetSyncSchedule();

    if( m_synchroStreamOpen )
    {
        if( m_synchroSemaphore != nullptr && m_synchroSemaphoreNumber >= 0 )
        {
            m_synchroStream.semReadPID[m_synchroSemaphoreNumber] = 0;
        }

        ImageStreamIO_closeIm( &m_synchroStream );
    }

    memset( &m_synchroStream, 0, sizeof( m_synchroStream ) );
    m_synchroSemaphore        = nullptr;
    m_synchroSemaphoreNumber  = 5;
    m_synchroStreamInode      = 0;
    m_synchroStreamOpen       = false;
    m_atime                   = timespec{};
    m_lastAtime               = timespec{};
    m_avgSemaphorePeriod_ns   = 0.0;
    m_wfsPeriodMeasured_ns    = 0.0;
    m_lastProducerAtime       = timespec{};
    m_lastProducerCnt0        = 0;
    m_producerPeriodInst_ns   = 0.0;
    m_avgProducerPeriod_ns    = 0.0;
    m_firstProducerSample     = true;
    m_syncFramesReceived      = 0;
    m_syncFramesWritten       = 0;
    m_syncFramesDropped       = 0;
    m_syncFrameIdGapCount     = 0;
    m_syncProducerFrameId     = 0;
    m_syncProducerFrameDelta  = 0;
    m_lastSyncProducerFrameId = 0;
    m_syncProducerFrameValid  = false;
    m_firstSemaphore          = true;
    m_avgReadLatency_ns       = 0.0;
    m_firstReadLatency        = true;
    m_delayModel_ns           = 0.0;
    m_delayApplied_ns         = 0.0;
    m_delayBudget_ns          = 0.0;
    m_nonDelayService_ns      = 0.0;
    m_avgNonDelayService_ns   = 0.0;
    m_firstNonDelayService    = true;
    m_delayPhaseError_ns      = 0.0;
    m_delayLock               = 0.0;
    m_delayCapped             = 0.0;
    m_triggerTime             = timespec{};
    m_triggerInterval_ns      = 0.0;
    m_channelReadoutTime_ns   = 0.0;
    m_localFrameSeq           = 0;
    m_lastTriggerTime         = timespec{};
    m_firstTriggerTime        = true;
    m_firstTimerTrigger       = true;
}

inline int ad7689Ctrl::acquireTimerAndCheckValid()
{
    while( !m_shutdown && !m_reconfig )
    {
        const auto now     = std::chrono::steady_clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>( now - m_time_start ).count();

        if( elapsed >= m_trigger )
        {
            m_triggerInterval_ns = m_firstTimerTrigger ? 0.0 : static_cast<double>( elapsed );
            m_firstTimerTrigger  = false;
            m_time_start         = now;

            if( elapsed >= 2 * m_nanoSecTarget )
            {
                m_missedPeriods += static_cast<uint64_t>( elapsed / m_nanoSecTarget ) - 1;
            }

            m_trigger = std::clamp(
                m_trigger - m_gain * ( elapsed - m_nanoSecTarget ), 0.5f * m_nanoSecTarget, m_nanoSecTarget );
            return readScan();
        }

        { //mutex scope
            std::lock_guard<std::mutex> lock( m_controlMutex );

            if( m_requestedFps != m_fps || m_requestedChannels != m_numChannels )
            {
                return 1;
            }
        }

        mx::sys::nanoSleep( 1000 );
    }

    return 1;
}

inline int ad7689Ctrl::monotonicNow( int64_t &now )
{
    timespec stamp{};

    if( clock_gettime( CLOCK_MONOTONIC, &stamp ) < 0 )
    {
        return log<software_critical, -1>( { __FILE__, __LINE__, errno, "monotonic scheduler clock" } );
    }

    now = stamp.tv_sec * 1000000000LL + stamp.tv_nsec;
    return 0;
}

inline int ad7689Ctrl::waitForSync( int64_t deadline )
{
#if defined( __linux__ ) && defined( __GLIBC__ )
    #if __GLIBC_PREREQ( 2, 30 )
    const timespec until{ deadline / 1000000000LL, deadline % 1000000000LL };
    return sem_clockwait( m_synchroSemaphore, CLOCK_MONOTONIC, &until );
    #endif
#endif
    // Match the portable monotonic-wait pattern already used by adcStream.
    int64_t now = 0;

    while( !m_shutdown && !m_reconfig )
    {
        if( sem_trywait( m_synchroSemaphore ) == 0 )
        {
            return 0;
        }

        if( errno != EAGAIN || monotonicNow( now ) < 0 )
        {
            return -1;
        }

        if( now >= deadline )
        {
            errno = ETIMEDOUT;
            return -1;
        }

        const int64_t  remaining = std::min<int64_t>( deadline - now, 1000000 );
        const timespec pause{ 0, static_cast<long>( remaining ) };
        nanosleep( &pause, nullptr );
    }

    errno = EINTR;
    return -1;
}

inline void ad7689Ctrl::resetSyncSchedule()
{
    m_ddspcSchedule.cancel();
    m_syncSchedule.cancel();
    m_syncSchedule.m_haveSample = false;
    m_haveSyncArrival = m_haveMeasuredCadence = false;
    m_lastSyncArrival_ns = m_nextSyncProbe_ns = 0;
    m_avgSemaphorePeriod_ns = m_wfsPeriodMeasured_ns = 0;
    m_avgProducerPeriod_ns = m_producerPeriodInst_ns = 0;
    m_firstSemaphore = m_firstProducerSample = m_firstReadLatency = true;
    m_syncProducerFrameValid                                      = false;
}

inline int ad7689Ctrl::processSyncReference( int64_t arrival )
{
    uint64_t producerCnt0 = m_syncFramesReceived;
    timespec producerAtime{};

    if( m_synchroStream.md != nullptr )
    {
        const volatile auto *metadata = m_synchroStream.md;

        if( metadata->write )
        {
            return 1;
        }

        producerCnt0  = metadata->cnt0;
        producerAtime = timespec{ metadata->atime.tv_sec, metadata->atime.tv_nsec };
        std::atomic_thread_fence( std::memory_order_seq_cst );

        if( metadata->write || metadata->cnt0 != producerCnt0 )
        {
            return 1;
        }

        if( m_syncProducerFrameValid && producerCnt0 == m_lastSyncProducerFrameId )
        {
            ++m_syncDuplicates;
            m_syncProducerFrameDelta = 0;
            return 1;
        }

        if( m_syncProducerFrameValid && producerCnt0 < m_lastSyncProducerFrameId )
        {
            ++m_cadenceResets;
            resetSyncSchedule();
        }
    }

    const uint64_t delta = m_syncProducerFrameValid ? producerCnt0 - m_lastSyncProducerFrameId : 0;
    const double   arrivalPeriod =
        m_haveSyncArrival && delta ? static_cast<double>( arrival - m_lastSyncArrival_ns ) / delta : 0;
    double observed = arrivalPeriod;

    if( m_haveSyncArrival && delta && !m_firstProducerSample &&
        ( producerAtime.tv_sec != 0 || producerAtime.tv_nsec != 0 ) )
    {
        const double seconds = static_cast<double>( producerAtime.tv_sec ) - m_lastProducerAtime.tv_sec;
        const double timestampPeriod =
            std::abs( seconds ) <= m_syncTimeout + 1
                ? ( seconds * 1e9 + producerAtime.tv_nsec - m_lastProducerAtime.tv_nsec ) / delta
                : 0;

        // Arrival deadlines stay monotonic; a stepped producer realtime clock is not a cadence change.
        if( timestampPeriod > 0 && arrivalPeriod > 0 && timestampPeriod >= 0.5 * arrivalPeriod &&
            timestampPeriod <= 2.0 * arrivalPeriod )
        {
            observed = timestampPeriod;
        }
    }

    if( m_synchroStream.md != nullptr )
    {
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

    else
    {
        m_syncProducerFrameId     = producerCnt0;
        m_syncProducerFrameDelta  = delta;
        m_lastSyncProducerFrameId = producerCnt0;
        m_syncProducerFrameValid  = true;
    }

    if( getRealtime( m_atime ) < 0 )
    {
        return log<software_critical, -1>( { __FILE__, __LINE__, errno, "WFS arrival timestamp" } );
    }

    if( observed > 0 && std::isfinite( observed ) && observed <= 1e9 )
    {
        const bool changed =
            m_haveMeasuredCadence && std::abs( observed - m_avgSemaphorePeriod_ns ) > 0.1 * m_avgSemaphorePeriod_ns;

        if( !m_haveMeasuredCadence || changed )
        {
            if( changed || m_avgSemaphorePeriod_ns <= 0 )
            {
                m_firstReadLatency = true;
            }

            m_avgSemaphorePeriod_ns = observed;

            if( changed )
            {
                ++m_cadenceResets;
                m_ddspcSchedule.cancel();
            }
        }
        else
        {
            m_avgSemaphorePeriod_ns = m_alpha * observed + ( 1.0 - m_alpha ) * m_avgSemaphorePeriod_ns;
        }

        m_haveMeasuredCadence = true;
    }
    else if( !m_haveMeasuredCadence )
    {
        m_avgSemaphorePeriod_ns = m_wfs_fps > 0 ? 1e9 / m_wfs_fps : 0;
    }

    m_lastAtime             = m_atime;
    m_lastSyncArrival_ns    = arrival;
    m_haveSyncArrival       = true;
    m_firstSemaphore        = false;
    const double correction = m_firstReadLatency ? 0 : m_synchroDelay - m_synchroDelayTarget;
    updatePhaseTarget( m_atime );
    const double service =
        std::max( AD7689Lib::AD7689::scanTime_ns( m_adcSettings, m_numChannels ), m_avgNonDelayService_ns );
    m_delayBudget_ns = m_ddspc.m_enabled ? 1e9 : std::max( 0.0, m_wfsPeriodMeasured_ns - service - m_cadenceGuard_ns );
    m_synchroDelay   = static_cast<float>( m_synchroDelayTarget + correction );

    if( m_ddspc.m_enabled )
    {
        m_delayCapped = 0; // DDSPC phase is never capped at the end of one WFS period.

        if( !m_ddspcSchedule.reference( producerCnt0,
                                        arrival,
                                        m_wfsPeriodMeasured_ns,
                                        m_fps,
                                        m_synchroDelayTarget,
                                        m_synchroDelay,
                                        service,
                                        m_cadenceGuard_ns,
                                        m_ddspc.m_slew ) )
        {
            if( m_wfsPeriodMeasured_ns > 0 )
            {
                ++m_overruns;
            }

            return 1;
        }

        return 0;
    }

    if( !m_syncSchedule.arm( arrival, m_wfsPeriodMeasured_ns, m_fps, m_synchroDelay, service, m_cadenceGuard_ns ) )
    {
        if( m_wfsPeriodMeasured_ns > 0 )
        {
            ++m_overruns;
        }

        return 1;
    }

    m_delayApplied_ns     = m_syncSchedule.m_phase;
    double requestedPhase = std::fmod( m_synchroDelay, m_syncSchedule.m_interval );

    if( requestedPhase < 0 )
    {
        requestedPhase += m_syncSchedule.m_interval;
    }

    m_delayCapped = requestedPhase > m_syncSchedule.m_phase ? 1 : 0;
    return 0;
}

inline int ad7689Ctrl::acquireSynchroAndCheckValid()
{
    bool pendingWake = false;

    if( m_synchroSemaphore == nullptr )
    {
        m_reconfig = true;
        return 1;
    }

    while( !m_shutdown && !m_reconfig )
    {
        if( applyControls() )
        {
            m_reconfig = true;
            return 1;
        }

        int64_t now = 0;

        if( monotonicNow( now ) < 0 )
        {
            return -1;
        }

        if( now >= m_nextSyncProbe_ns )
        {
            m_nextSyncProbe_ns = now + static_cast<int64_t>( std::min( m_syncTimeout * 1e9, 1e8 ) );

            if( synchroStreamStale() )
            {
                resetSyncSchedule();
                m_reconfig = true;
                return 1;
            }
        }

        // References take precedence over a pending old-period scan, including after an overrun.
        if( pendingWake || sem_trywait( m_synchroSemaphore ) == 0 )
        {
            pendingWake    = false;
            unsigned wakes = 1;

            while( wakes < 256 && sem_trywait( m_synchroSemaphore ) == 0 )
            {
                ++wakes;
            }

            m_syncFramesReceived += wakes;
            m_coalescedWakes += wakes - 1;

            if( monotonicNow( now ) < 0 || processSyncReference( now ) < 0 )
            {
                return -1;
            }

            captureTimingDiagnostics();
            return 1; // A WFS observation is not itself an ADC frame; the next call serves its first slot.
        }

        if( errno != EAGAIN && errno != EINTR )
        {
            m_reconfig = true;
            return log<software_error, 1>( { __FILE__, __LINE__, errno, "WFS semaphore poll" } );
        }

        if( m_haveSyncArrival && now - m_lastSyncArrival_ns >= m_syncTimeout * 1e9 )
        {
            ++m_cadenceResets;
            resetSyncSchedule();
            return 1;
        }

        if( m_ddspc.m_enabled )
        {
            m_ddspcSchedule.discardLate( now );

            if( m_ddspcSchedule.pending() && now >= m_ddspcSchedule.deadline() )
            {
                return acquireDdspcAndCheckValid( now );
            }

            if( m_ddspcSchedule.m_active && !m_ddspcSchedule.pending() && now >= m_ddspcSchedule.end() )
            {
                m_ddspcSchedule.cancel();
                m_firstReadLatency = true;
                return 1;
            }
        }

        m_syncSchedule.discardLate( now );

        if( m_syncSchedule.m_slot < m_syncSchedule.m_count && now >= m_syncSchedule.deadline() )
        {
            const int64_t due      = m_syncSchedule.deadline();
            const bool    previous = m_syncSchedule.m_haveSample;
            const int64_t last     = m_syncSchedule.m_lastSample;

            if( !m_syncSchedule.take( now ) )
            {
                continue;
            }

            if( now > due )
            {
                ++m_lateScans;
                m_maxLateness_ns = std::max( m_maxLateness_ns, static_cast<double>( now - due ) );
            }

            const unsigned slot  = m_syncSchedule.m_slot - 1;
            m_delayApplied_ns    = static_cast<double>( due - m_syncSchedule.m_anchor );
            m_triggerTime        = nsToTimespec( timespecToNs( m_atime ) + m_delayApplied_ns );
            m_triggerInterval_ns = previous ? static_cast<double>( now - last ) : 0;

            if( readScan() != 0 )
            {
                m_syncSchedule.cancel();
                return 1;
            }

            // Update the legacy read-latency controller once per WFS period at the nearest model phase.
            const unsigned referenceSlot = static_cast<unsigned>( std::clamp(
                std::floor( ( m_synchroDelayTarget - m_syncSchedule.m_phase ) / m_syncSchedule.m_interval + 0.5 ),
                0.0,
                static_cast<double>( m_syncSchedule.m_count - 1 ) ) );

            if( slot == referenceSlot )
            {
                const double latency = static_cast<double>( now - m_syncSchedule.m_anchor );
                m_avgReadLatency_ns =
                    m_firstReadLatency ? latency : m_alpha * latency + ( 1 - m_alpha ) * m_avgReadLatency_ns;
                m_firstReadLatency = false;
                updateSynchroDelayController( m_synchroDelay );
            }

            int64_t finished = 0;

            if( monotonicNow( finished ) < 0 )
            {
                m_frameValid = false;
                return -1;
            }

            m_nonDelayService_ns    = static_cast<double>( finished - now );
            m_avgNonDelayService_ns = m_firstNonDelayService
                                          ? m_nonDelayService_ns
                                          : m_alpha * m_nonDelayService_ns + ( 1 - m_alpha ) * m_avgNonDelayService_ns;
            m_firstNonDelayService  = false;
            return 0;
        }

        if( m_syncSchedule.m_count && now >= m_syncSchedule.end() )
        {
            m_syncSchedule.cancel();
            m_delayLock = 0;
            return 1;
        }

        int64_t until = now + ( m_ddspc.m_enabled ? 100000000 : 1000000 );

        if( m_ddspc.m_enabled )
        {
            until = std::min( until, m_nextSyncProbe_ns );

            if( m_haveSyncArrival )
            {
                until = std::min( until, m_lastSyncArrival_ns + static_cast<int64_t>( m_syncTimeout * 1e9 ) );
            }
        }

        if( m_syncSchedule.m_count )
        {
            const int64_t next =
                m_syncSchedule.m_slot < m_syncSchedule.m_count ? m_syncSchedule.deadline() : m_syncSchedule.end();
            until = std::min( until, next );
        }

        if( m_ddspc.m_enabled && m_ddspcSchedule.m_active )
        {
            until = std::min( until, m_ddspcSchedule.pending() ? m_ddspcSchedule.deadline() : m_ddspcSchedule.end() );
        }

        if( waitForSync( until ) == 0 )
        {
            pendingWake = true;
        }
        else if( errno == EINTR )
        {
            return 1;
        }
        else if( errno != ETIMEDOUT )
        {
            m_reconfig = true;
            return log<software_error, 1>( { __FILE__, __LINE__, errno, "WFS scheduler wait" } );
        }
    }

    m_syncSchedule.cancel();
    m_ddspcSchedule.cancel();
    return 1;
}

inline int ad7689Ctrl::acquireDdspcAndCheckValid( int64_t now )
{
    const bool    previous = m_ddspcSchedule.m_haveSample;
    const int64_t last     = m_ddspcSchedule.m_lastSample;

    if( !m_ddspcSchedule.take( now, m_ddspcSlot ) )
    {
        return 1;
    }

    const auto &slot     = m_ddspcSlot;
    m_ddspcStart_ns      = now;
    m_ddspcCompletion_ns = m_ddspcPublished_ns = 0;
    m_delayApplied_ns = static_cast<double>( slot.m_deadline - slot.m_reference.m_arrival ) -
                        slot.m_slot * slot.m_reference.m_period / ( slot.m_reference.m_end - slot.m_reference.m_first );
    m_triggerInterval_ns = previous ? static_cast<double>( now - last ) : 0;
    m_triggerTime        = nsToTimespec( timespecToNs( m_atime ) + slot.m_deadline - m_lastSyncArrival_ns );

    if( now > slot.m_deadline )
    {
        ++m_lateScans;
        m_maxLateness_ns = std::max( m_maxLateness_ns, static_cast<double>( now - slot.m_deadline ) );
    }

    if( readScan() != 0 )
    {
        m_ddspcSchedule.cancel();
        return 1;
    }

    if( monotonicNow( m_ddspcCompletion_ns ) < 0 )
    {
        m_frameValid = false;
        return -1;
    }

    m_nonDelayService_ns    = static_cast<double>( m_ddspcCompletion_ns - now );
    m_avgNonDelayService_ns = m_firstNonDelayService
                                  ? m_nonDelayService_ns
                                  : m_alpha * m_nonDelayService_ns + ( 1 - m_alpha ) * m_avgNonDelayService_ns;
    m_firstNonDelayService  = false;

    // Do not let the feedback integrator fight an intentional phase slew. Use the ORIGINAL
    // reference's first slot even when a newer WFS observation has already arrived.
    if( slot.m_slot == 0 && std::abs( m_ddspcSchedule.m_target - m_ddspcSchedule.m_applied ) < 1 )
    {
        const double error  = static_cast<double>( now - slot.m_reference.m_arrival ) - slot.m_reference.m_target;
        m_ddspcReadError_ns = m_firstReadLatency ? error : m_alpha * error + ( 1 - m_alpha ) * m_ddspcReadError_ns;
        m_firstReadLatency  = false;
        m_avgReadLatency_ns = m_synchroDelayTarget + m_ddspcReadError_ns;
        updateSynchroDelayController( m_synchroDelay );
    }

    return 0;
}

inline int ad7689Ctrl::getRealtime( timespec &ts )
{
    return clock_gettime( CLOCK_REALTIME, &ts );
}

inline int ad7689Ctrl::waitOnSemaphore( sem_t *sem, timespec &ts )
{
    return sem_timedwait( sem, &ts );
}

inline void ad7689Ctrl::delayBeforeRead()
{
    if( m_delayApplied_ns > 0.0 )
    {
        mx::sys::nanoSleep( static_cast<unsigned>( m_delayApplied_ns ) );
        return;
    }
}

inline void ad7689Ctrl::updateSynchroDelayController( double desiredDelay_ns )
{
    constexpr double c_satEpsilon_ns = 1e-6;

    const double controlStep_ns =
        static_cast<double>( m_gain ) * ( m_avgReadLatency_ns - static_cast<double>( m_synchroDelayTarget ) );
    double updatedDelay_ns = desiredDelay_ns - controlStep_ns;

    if( updatedDelay_ns < 0.0 )
    {
        updatedDelay_ns = 0.0;
    }

    const bool upperSaturated              = !m_ddspc.m_enabled && ( m_wfsPeriodMeasured_ns > 0.0 ) &&
                                             ( m_delayApplied_ns + c_satEpsilon_ns < desiredDelay_ns );
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

inline int ad7689Ctrl::checkRecordTimes()
{
    return telemeter<ad7689Ctrl>::checkRecordTimes( telem_fgtimings() );
}

inline int ad7689Ctrl::recordTelem( const telem_fgtimings * )
{
    return recordFGTimings( true );
}

inline INDI_NEWCALLBACK_DEFN( ad7689Ctrl, m_indiP_ddspcSettings )( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.createUniqueKey() != m_indiP_ddspcSettings.createUniqueKey() )
    {
        return -1;
    }

    ad7689CtrlDetail::ddspcSettings requested;
    double                          period;
    int                             offset;
    { //mutex scope
        std::lock_guard<std::mutex> lock( m_controlMutex );
        requested = m_requestedDdspc;
        period =
            m_syncSnapshot[3] > 0 ? m_syncSnapshot[3] * 1000 : ( m_requestedWfsFps > 0 ? 1e9 / m_requestedWfsFps : 0 );
        offset = m_requestedOffset;
    }

    bool changed = false;

    if( ipRecv.find( "enabled" ) )
    {
        const double enabled = controlValue( ipRecv, "enabled" );

        if( enabled != 0 && enabled != 1 )
        {
            return rejectControl( m_indiP_ddspcSettings );
        }

        requested.m_enabled = enabled == 1;
        changed             = true;
    }

    for( auto field : { std::pair<const char *, double *>( "wfs_process_ns", &requested.m_process ),
                        { "acc_latency_ns", &requested.m_latency },
                        { "margin_ns", &requested.m_margin },
                        { "slew_fraction", &requested.m_slew } } )
    {
        if( ipRecv.find( field.first ) )
        {
            *field.second = controlValue( ipRecv, field.first );
            changed       = true;
        }
    }

    const double delay = requested.delay( period, 1000.0 * offset );

    if( !changed || !requested.valid() ||
        ( requested.m_enabled && ( m_synchroShmimName.empty() || ( period > 0 && ( delay < 0 || delay > 1e9 ) ) ) ) )
    {
        return rejectControl( m_indiP_ddspcSettings );
    }

    { //mutex scope
        std::lock_guard<std::mutex> lock( m_controlMutex );
        m_requestedDdspc = requested;
    }

    m_indiP_ddspcSettings.setState( INDI_BUSY );
    return 0;
}

// INDI callback handling for fps configuration.
inline INDI_NEWCALLBACK_DEFN( ad7689Ctrl, m_indiP_fps )( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.createUniqueKey() != m_indiP_fps.createUniqueKey() ||
        !( ipRecv.find( "target" ) || ipRecv.find( "current" ) ) )
    {
        return -1;
    }

    double target = controlValue( ipRecv, ipRecv.find( "target" ) ? "target" : "current" );
    int    count;
    { //mutex scope
        std::lock_guard<std::mutex> lock( m_controlMutex );
        count = m_requestedChannels;
    }

    if( requestAcquisition( target, count ) < 0 )
    {
        return rejectControl( m_indiP_fps );
    }

    updateIfChanged( m_indiP_fps, "target", target, INDI_BUSY );
    return 0;
}

// INDI callback handling for global EMA alpha configuration.
inline INDI_NEWCALLBACK_DEFN( ad7689Ctrl, m_indiP_alpha )( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.createUniqueKey() != m_indiP_alpha.createUniqueKey() ||
        !( ipRecv.find( "target" ) || ipRecv.find( "current" ) ) )
    {
        return -1;
    }

    double target = controlValue( ipRecv, ipRecv.find( "target" ) ? "target" : "current" );

    if( !std::isfinite( target ) )
    {
        return rejectControl( m_indiP_alpha );
    }

    { //mutex scope
        std::lock_guard<std::mutex> lock( m_controlMutex );
        m_requestedAlpha = static_cast<float>( std::clamp( target, 0.0, 1.0 ) );
        target           = m_requestedAlpha;
    }

    updateIfChanged( m_indiP_alpha, "target", target, INDI_BUSY );
    return 0;
}

// INDI callback handling for synchronized-mode signed phase offset configuration.
inline INDI_NEWCALLBACK_DEFN( ad7689Ctrl, m_indiP_synchroDelay )( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.createUniqueKey() != m_indiP_synchroDelay.createUniqueKey() ||
        !( ipRecv.find( "target" ) || ipRecv.find( "current" ) ) )
    {
        return -1;
    }

    double target = controlValue( ipRecv, ipRecv.find( "target" ) ? "target" : "current" );

    if( !std::isfinite( target ) || target < -1000000 || target > 1000000 || std::floor( target ) != target )
    {
        return rejectControl( m_indiP_synchroDelay );
    }

    bool valid = true;
    { //mutex scope
        std::lock_guard<std::mutex> lock( m_controlMutex );
        const double                period =
            m_syncSnapshot[3] > 0 ? m_syncSnapshot[3] * 1000 : ( m_requestedWfsFps > 0 ? 1e9 / m_requestedWfsFps : 0 );
        const double delay = m_requestedDdspc.delay( period, 1000.0 * target );
        valid              = !m_requestedDdspc.m_enabled || period <= 0 || ( delay >= 0 && delay <= 1e9 );

        if( valid )
        {
            m_requestedOffset = static_cast<int>( target );
        }
    }

    if( !valid )
    {
        return rejectControl( m_indiP_synchroDelay );
    }

    updateIfChanged( m_indiP_synchroDelay, "target", target, INDI_BUSY );
    return 0;
}

inline INDI_SETCALLBACK_DEFN( ad7689Ctrl, m_indiP_fpsSource )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_fpsSource, ipRecv );

    if( !ipRecv.find( m_fpsElement ) )
    {
        return -1;
    }

    const double target = controlValue( ipRecv, m_fpsElement );

    if( !m_synchroShmimName.empty() )
    {
        if( !std::isfinite( target ) || target < 0 || target > 1000000 )
        {
            return log<software_error, -1>( { "Rejected invalid WFS rate metadata" } );
        }

        { //mutex scope
            std::lock_guard<std::mutex> lock( m_controlMutex );
            m_requestedWfsFps = target;
        }

        return 0;
    }

    int count;
    { //mutex scope
        std::lock_guard<std::mutex> lock( m_controlMutex );
        count = m_requestedChannels;
    }

    if( requestAcquisition( target, count ) < 0 )
    {
        return rejectControl( m_indiP_fps );
    }

    updateIfChanged( m_indiP_fps, "target", target, INDI_BUSY );
    return 0;
} // INDI_SETCALLBACK_DEFN(ad7689Ctrl, m_indiP_fpsSource)

inline INDI_NEWCALLBACK_DEFN( ad7689Ctrl, m_indiP_numChannels )( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.createUniqueKey() != m_indiP_numChannels.createUniqueKey() ||
        !( ipRecv.find( "target" ) || ipRecv.find( "current" ) ) )
    {
        return -1;
    }

    double target = controlValue( ipRecv, ipRecv.find( "target" ) ? "target" : "current" );
    double rate;
    { //mutex scope
        std::lock_guard<std::mutex> lock( m_controlMutex );
        rate = m_requestedFps;
    }

    if( requestAcquisition( rate, target ) < 0 )
    {
        return rejectControl( m_indiP_numChannels );
    }

    updateIfChanged( m_indiP_numChannels, "target", target, INDI_BUSY );
    return 0;
}

inline INDI_SETCALLBACK_DEFN( ad7689Ctrl, m_indiP_numChannelsSource )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_numChannelsSource, ipRecv );

    if( !ipRecv.find( m_numChannelsElement ) )
    {
        return -1;
    }

    const double target = controlValue( ipRecv, m_numChannelsElement );
    double       rate;
    { //mutex scope
        std::lock_guard<std::mutex> lock( m_controlMutex );
        rate = m_requestedFps;
    }

    if( requestAcquisition( rate, target ) < 0 )
    {
        return rejectControl( m_indiP_numChannels );
    }

    updateIfChanged( m_indiP_numChannels, "target", target, INDI_BUSY );
    return 0;
}

} // namespace app
} // namespace MagAOX

#endif // ad7689Ctrl_hpp
