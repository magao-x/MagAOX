/** \file accelSim.hpp
 * \brief Continuous raw-ADC accelerometer simulator with the mcp3208Ctrl interface.
 * \ingroup accelSim_files
 */
#ifndef accelSim_hpp
#define accelSim_hpp

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <sched.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#ifdef __linux__
    #include <sys/prctl.h>
#endif
#include <thread>
#include <unistd.h>

#include <mx/improc/eigenCube.hpp>
#include <mx/ioutils/fits/fitsFile.hpp>

#include "../../libMagAOX/libMagAOX.hpp"
#include "../../magaox_git_version.h"
#include "accelSimModel.hpp"
#include "accelSimTimeline.hpp"
#include "accelSimOutput.hpp"
#include "accelSimDM.hpp"
#include "../../libMagAOX/app/dev/dmSyncopationProducer.hpp"

/** \defgroup accelSim Accelerometer Simulator
 * \brief Continuous physical vibrations published as raw X/Y ADC counts.
 * \ingroup apps
 */
/** \defgroup accelSim_files Accelerometer Simulator Files
 * \ingroup accelSim
 */

namespace MagAOX
{
namespace app
{

namespace accelSimDetail
{
/// Keep the existing producer alive for tagged zero removal after MagAOXApp requests shutdown.
/** This only supplies the CRTP lifetime flag; scheduling, ownership and publication remain in the base. */
class dmProducer : public dev::dmSyncopationProducer<dmProducer>
{
    friend class dev::dmSyncopationProducer<dmProducer>;

  protected:
    bool m_shutdown{ false }; ///< Set only after the output thread has finished its final zero attempt.
};

/// Low-overhead DM publication statistics, accumulated by the DM worker and drained by the app loop.
struct dmStatistics
{
    static constexpr size_t c_steps = 4; ///< Subframes reported individually; later steps are counted in step 3.

    std::array<uint64_t, c_steps> m_advertised{}, m_requests{}, m_published{},
        m_copied{}; ///< Cumulative per-step slots seen in records, returned by poll, committed and copied.

    uint64_t m_preparedHits{ 0 }, m_preparedMisses{ 0 }; ///< Commits using a staged surface / synthesized on demand.

    int64_t m_lastRecordEpoch{ 0 }, m_lastRecordGeneration{ 0 }, m_lastRecordCycle{ -1 }; ///< Last counted record.

    std::array<uint32_t, 256> m_marginHistogram{}; ///< Interval margins in 1-us bins; the last bin collects >=255 us.

    uint64_t m_skippedSlots{ 0 }; ///< Cumulative slots never returned between consecutive requests of one plan.

    int64_t m_lastSlot{ -1 }; ///< Ordinal cycle*steps+step of the last request, or -1 after a plan change.

    int64_t m_lastEpoch{ 0 }, m_lastGeneration{ 0 }; ///< Scheduler plan identity of m_lastSlot.

    int64_t m_lastPoll{ 0 }; ///< Monotonic time of the previous poll while output was serviced [ns].

    /// Per-interval timing in microseconds: count, sum, and extreme values since the last drain.
    struct interval
    {
        uint64_t m_count{ 0 };   ///< Samples in this interval.
        double   m_sum{ 0 };     ///< Sum of samples.
        double   m_min{ 0 };     ///< Smallest sample.
        double   m_max{ 0 };     ///< Largest sample.
        double   m_squares{ 0 }; ///< Sum of squared samples.

        /// Add one sample.
        void add( double value /**< [in] sample [us] */ );

        /// Return the interval mean, or zero when empty.
        double mean() const;

        /// Return the interval standard deviation (RMS jitter about the mean), or zero when empty.
        double rms() const;
    };

    interval m_pollGap; ///< Time between consecutive polls: worker wake and service latency.
    interval m_slack;   ///< Cutoff minus the time a new request was first observed.
    interval m_prepare; ///< Timeline evaluation plus surface synthesis.
    interval m_margin;  ///< Cutoff minus the committed ready time.
    interval m_publish; ///< Producer publish call duration, including its intentional wait to cutoff-writeLead.
    interval m_lead;    ///< Target minus the committed ready time.
    interval m_receipt; ///< Time the copy receipt was first observed after the commit's cutoff.

    /// Return the margin below which the given fraction of interval commits fell, from the histogram [us].
    double marginQuantile( double fraction /**< [in] lower-tail fraction, e.g. 0.01 */ ) const;

    /// Return the reporting index of a scheduler step.
    static size_t index( int step /**< [in] scheduler step */ );
};

inline void dmStatistics::interval::add( double value )
{
    m_min = m_count ? std::min( m_min, value ) : value;
    m_max = m_count ? std::max( m_max, value ) : value;
    m_sum += value;
    m_squares += value * value;
    ++m_count;
}

inline double dmStatistics::interval::mean() const
{
    return m_count ? m_sum / m_count : 0;
}

inline double dmStatistics::interval::rms() const
{
    return m_count ? std::sqrt( std::max( 0.0, m_squares / m_count - mean() * mean() ) ) : 0;
}

inline double dmStatistics::marginQuantile( double fraction ) const
{
    uint64_t total = 0, seen = 0;
    for( auto count : m_marginHistogram )
        total += count;
    for( size_t bin = 0; bin < m_marginHistogram.size(); ++bin )
        if( ( seen += m_marginHistogram[bin] ) > fraction * total )
            return double( bin );
    return 0;
}

/// One privately prepared surface for an exact advertised slot; never written to the source until published.
struct stagedCommand
{
    synco::Request          m_request{};      ///< Exact slot identity, including targetNs and cutoffNs.
    double                  m_scale{ 0 };     ///< dm_scale used for the surface.
    uint64_t                m_timeline{ 0 };  ///< Timeline generation of the evaluated pointing.
    bool                    m_valid{ false }; ///< Whether the surface is complete and finite.
    std::array<float, 2500> m_pixels{};       ///< Preallocated calibrated surface.
};

inline size_t dmStatistics::index( int step )
{
    return std::min<size_t>( std::max( step, 0 ), c_steps - 1 );
}
} // namespace accelSimDetail

/// Simulated accelerometer acquisition with optional DM injection of the same realization at scheduler targets.
/** Acquisition generates one timestamped realization ahead of time. acc000 measures it at each acquisition time;
 * the DM thread evaluates its noise-free pointing at each syncopation request's targetNs.
 */
class accelSim : public MagAOXApp<true>,
                 public dev::frameGrabber<accelSim>,
                 public dev::telemeter<accelSim>,
                 public accelSimDetail::dmProducer
{
    friend class dev::frameGrabber<accelSim>;
    friend class dev::telemeter<accelSim>;
    typedef dev::frameGrabber<accelSim>                            frameGrabberT;
    typedef dev::telemeter<accelSim>                               telemeterT;
    typedef dev::dmSyncopationProducer<accelSimDetail::dmProducer> dmSyncopationProducerT;
    static constexpr bool c_frameGrabber_flippable = false; ///< Preserve fixed X/Y ordering.
    static_assert( std::atomic<double>::is_always_lock_free && std::atomic<uint64_t>::is_always_lock_free );

  protected:
    using MagAOXApp<true>::m_shutdown;

    bool m_zeroOutput{ false }; ///< Startup-only override of published ADC counts, not physics or DM injection.

    accelSimDetail::parameters m_parameters;     ///< Startup-only physics/calibration configuration.
    accelSimDetail::model      m_model;          ///< Acquisition-thread-owned physical state and RNG.
    accelSimDetail::timeline   m_timeline;       ///< Shared realization: acquisition generates, the DM thread reads.
    accelSimDetail::sample     m_acquiredSample; ///< Latest measurement, owned by the acquisition thread.
    accelSimDetail::schedule   m_schedule;       ///< Absolute monotonic timer schedule.
    std::atomic<double>        m_requestedFps{ 6000 };            ///< Validated callback-to-worker rate request.
    std::atomic<double>        m_activeFps{ 6000 };               ///< Rate currently used by the acquisition worker.
    std::atomic<uint64_t> m_rateRequest{ 0 }, m_rateApplied{ 0 }; ///< Requested and worker-acknowledged FPS revisions.
    std::atomic<double>   m_alpha{ .01 };      ///< Timing EMA coefficient, independently updated by INDI.
    std::atomic<int>      m_postDelay{ 0 };    ///< Signed synchronized phase offset in microseconds.
    double                m_maxFrequency{ 0 }; ///< Immutable maximum mode frequency for callback validation.
    std::string           m_fpsDevice, m_fpsProperty{ "fps" }, m_fpsElement{ "current" }; ///< External FPS source.
    std::string           m_channelsDevice, m_channelsProperty{ "numChannels" },
        m_channelsElement{ "current" };            ///< Channel source.
    double      m_fpsTol{ 0 }, m_channelsTol{ 0 }; ///< Compatibility keys, unused by the hardware callbacks as well.
    std::string m_syncName;                        ///< Optional semaphore trigger stream; empty selects timer mode.
    double      m_dtTransfer{ 3000 }, m_wfsProcess{ 51500 }, m_dtF{ 10000 },
        m_wfsRead{ 276100 };                                                ///< Delay-model terms in ns.
    double m_lockAbsolute{ 50000 }, m_lockFraction{ .1 }, m_guard{ 20000 }; ///< Delay lock criteria and cadence margin.
    IMAGE  m_syncImage{};                    ///< Synchronization input only; never the output image.
    bool   m_syncOpen{ false };              ///< Whether the source mapping is owned by this app.
    int    m_syncIndex{ 5 };                 ///< Claimed source semaphore index, initially preferring index five.
    ino_t  m_syncInode{ 0 };                 ///< Backing-file inode used to detect source replacement.
    accelSimDetail::phaseClock m_phaseClock; ///< Continuous synchronized sampling schedule and WFS reference.
    double                     m_syncTimeout{ .01 }, m_maxSlew{ .05 },
        m_referenceFps{ 0 };      ///< Silence seconds, slew fraction, optional WFS Hz.
    int64_t m_nextSyncProbe{ 0 }; ///< Next nonblocking source-open/inode check [monotonic ns].
    int64_t m_lastAcquisition{ 0 }, m_triggerPrediction{ 0 }; ///< Acquisition history and latest WFS phase prediction.
    int64_t m_latencyArrival{ 0 },
        m_latencyPrediction{ 0 }; ///< Pending reference retained across intervening WFS events.
    bool m_measureReference{
        false }; ///< Awaiting a sample at/after the retained WFS prediction, independently of newer references.
    bool m_fresh{ false }; ///< True only for an acquired sample not yet copied.

    using enum accelSimDetail::timingIndex;
    /// Hardware-compatible element names in timingIndex order.
    inline static constexpr std::array<const char *, timingCount> c_timingNames{ "avg_read_latency_us",
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
                                                                                 "mode_code" };

    /// dm_timing element names: interval statistics in microseconds, reset at each app-loop report.
    inline static constexpr std::array<const char *, 19> c_dmTimingNames{ "samples",
                                                                          "poll_gap_mean_us",
                                                                          "poll_gap_max_us",
                                                                          "slack_min_us",
                                                                          "slack_mean_us",
                                                                          "prepare_mean_us",
                                                                          "prepare_max_us",
                                                                          "publish_mean_us",
                                                                          "publish_max_us",
                                                                          "margin_min_us",
                                                                          "margin_mean_us",
                                                                          "margin_rms_us",
                                                                          "margin_p01_us",
                                                                          "margin_p001_us",
                                                                          "lead_min_us",
                                                                          "lead_mean_us",
                                                                          "receipt_mean_us",
                                                                          "receipt_max_us",
                                                                          "slack_rms_us" };

    using diagnostics = accelSimDetail::diagnostics;
    diagnostics         m_diagnostics, m_snapshot; ///< Worker state and synchronized app-loop snapshot.
    std::mutex          m_snapshotMutex;           ///< Protects only snapshot copies, never acquisition work.
    std::mutex          m_controlMutex;            ///< Serializes app-owned INDI properties and their callbacks.
    std::mutex          m_dmMutex; ///< Guards DM flags, bookkeeping and statistics; never held across INDI I/O.
    std::string         m_dmChannel, m_dmModes, m_dmMask; ///< Explicit owned channel and approved calibration paths.
    IMAGE               m_dmImage{}; ///< Existing auxiliary source, written only through dmSyncopationProducer.
    std::thread         m_dmThread;  ///< Sole owner of producer poll/publish and acknowledgement observations.
    bool                m_dmThreadInit{ true }; ///< Startup synchronizer for framework priority and cpuset setup.
    pid_t               m_dmThreadID{ 0 };      ///< DM worker thread ID, set by the worker for threadStart.
    pcf::IndiProperty   m_dmThreadProp;         ///< Framework thread property (th-dm: pid and priority).
    int                 m_dmThreadPrio{ 0 };    ///< DM worker real-time priority; 0 keeps SCHED_OTHER.
    std::string         m_dmThreadCpuset;       ///< DM worker cpuset; empty leaves placement unchanged.
    std::atomic<bool>   m_dmStop{ false };      ///< Request final tagged removal and join independently of acquisition.
    std::atomic<bool>   m_dmCommandValid{ true }; ///< Current numerical validity; a valid synthesis clears the error.
    double              m_dmResidual{ 1e-6 };     ///< Accepted relative physical projection residual.
    bool                m_dmStartupEnabled{ false }; ///< Startup enable remains unsupported; operators use dm_enable.
    bool                m_dmReady{ false }; ///< Immutable calibration preparation result until acquisition has joined.
    std::atomic<double> m_dmScale{ 0 };     ///< DM-only amplitude scale, independent of model amplitudes.
    bool m_dmEnabled{ false }, m_dmZeroPending{ false }; ///< Operator request and required removal, under m_dmMutex.
    bool m_dmActive{ false }, m_dmZeroAcknowledged{ false }; ///< Last copied contribution and explicit zero receipt.
    bool m_dmTransportFault{ false }; ///< Current transport availability, without fault latching.
    bool m_dmAwaitingAck{ false }, m_dmPendingZero{ false }; ///< Receipt bookkeeping for this app's latest commit.
    uint64_t m_dmPollWaits{ 0 }; ///< Attached polls returning zero, including duplicate slots and outstanding receipts.
    uint64_t m_dmRequests{ 0 };  ///< Eligible requests returned by the producer.
    uint64_t m_dmLateReturns{ 0 };                   ///< Failed publish attempts returning at or after their cutoff.
    uint64_t m_dmRejected{ 0 };                      ///< Failed publish attempts returning before their cutoff.
    uint64_t m_dmTimelineMisses{ 0 };                ///< Requests skipped because the target was outside the timeline.
    uint64_t m_dmPublished{ 0 };                     ///< Successful tagged ready commits by this instance.
    uint64_t m_dmCopied{ 0 }, m_dmDiscarded{ 0 };    ///< Exact transport dispositions for this instance's commits.
    struct stat     m_dmModesStat{}, m_dmMaskStat{}; ///< Calibration identities checked while loading.
    std::string     m_dmCalibration{ "fit" };        ///< dm.calibration: "fit" (physical tilt fit) or "coefficients".
    Eigen::Matrix2d m_dmCoefficients{ Eigen::Matrix2d::Zero() };        ///< dm.coefficients K [mode coefficient/rad].
    std::string     m_dmUnavailable{ "dm.channelName not configured" }; ///< Why DM output is unavailable, or "".
    accelSimDetail::dmInjection  m_dmInjection; ///< DM-thread-owned precomputed physical synthesis.
    accelSimDetail::dmStatistics m_dmStats;     ///< Per-step counts and interval timing, under m_dmMutex.
    std::array<accelSimDetail::stagedCommand, accelSimDetail::dmStatistics::c_steps>
                      m_dmStage;    ///< Surfaces prepared ahead for the current record's steps, under m_dmMutex.
    synco::Record     m_dmRecord{}; ///< Last coherent scheduler record read for staging and counting.
    uint64_t          m_dmRecordCounter{ 0 };                              ///< Schedule stream counter of m_dmRecord.
    pcf::IndiProperty m_indiP_dmEnable, m_indiP_dmScale, m_indiP_dmStatus; ///< Minimal low-rate DM controls/status.
    pcf::IndiProperty m_indiP_dmCalibration; ///< Calibration method, K, fit residuals and unavailability reason.
    pcf::IndiProperty m_indiP_dmSteps, m_indiP_dmTiming; ///< Per-step counts and per-report-interval timing.
    /// Compatible accelerometer controls, serialized by m_controlMutex.
    pcf::IndiProperty m_indiP_fps, m_indiP_numChannels, m_indiP_alpha, m_indiP_synchroDelay;
    /// Diagnostic properties and configured external subscriptions.
    pcf::IndiProperty m_indiP_timingDiag, m_indiP_simDiag, m_indiP_fpsSource, m_indiP_numChannelsSource;

    /// Define the accelerometer-owned properties, separated from thread startup for testing.
    int setupProperties();
    /// Send coherent app-owned controls and diagnostics at ordinary app-loop cadence.
    void updateINDI( const diagnostics &snapshot /**< [in] coherent worker snapshot */ );
    /// Open and claim the synchronization source, returning positive to retry.
    int openSync();
    /// Release the claimed source semaphore and mapping.
    void closeSync();
    /// Detect a source mapping that vanished or was replaced.
    bool syncStale();
    /// Wait for the next accelerometer deadline while accepting independent WFS references.
    int waitSync();
    /// Enter holdover and publish unlocked status without stopping the sampling clock.
    void invalidateSync();
    /// Transfer diagnostics; forced transfers wait only for the app-loop snapshot copy.
    void snapshotDiagnostics( bool force /**< [in] require delivery for lifecycle events */ );
    /// Consume a source reference and update its phase/counter diagnostics independently of publication.
    void syncTiming( uint64_t producer /**< [in] current source counter */,
                     int64_t  arrival /**< [in] monotonic semaphore arrival */ );
    /// Report bounded clock correction and lock against the most recent valid phase reference.
    void updateSyncDiagnostics();
    /// Wait interruptibly for an absolute monotonic deadline.
    int waitUntil( int64_t deadline /**< [in] monotonic deadline in ns */ );
    /// Get monotonic nanoseconds without conversion of epoch time to double.
    static int64_t monotonicNow();
    /// Extend the shared timeline and measure one accelerometer tick at its acquisition time.
    void measure( int64_t  acquired /**< [in] monotonic acquisition time, the scheduler's clock domain [ns] */,
                  uint64_t tick /**< [in] consumed accelerometer tick selecting its sensor-noise draw */ );
    /// Send a property state, leaving its last accepted values intact on error.
    int propertyState( pcf::IndiProperty &property /**< [in,out] property to acknowledge */,
                       bool               valid /**< [in] whether the update was accepted */ );
    /// Validate and enqueue an FPS update without touching model state.
    int requestRate( double value /**< [in] requested FPS */ );
    /// Validate the fixed two-channel interface.
    int requestChannels( double value /**< [in] requested channel count */ );
    /// Extract a finite numeric target, falling back to current for compatible clients.
    static double target( const pcf::IndiProperty &property /**< [in] incoming update */ );
    /// Prepare calibration and an explicitly configured owned channel before starting acquisition.
    void prepareDM();
    /// Load the configured calibration and log its method, K and residuals, or record why output is unavailable.
    bool prepareDMCalibration();
    /// Check calibration identity while loading; acquisition uses the validated in-memory calibration.
    bool calibrationUnchanged() const;
    /// Read and validate the mode cube and optional mask, then prepare the configured calibration; throws the reason.
    void loadDMCalibration();
    /// Describe the configured fitting support: dm.mask, uniform weights, or none for explicit coefficients.
    std::string dmSupport() const;
    /// Report controls/status without acquiring the DM writer or publishing images.
    void updateDMINDI();
    /// Service the existing tagged producer once; returns whether a payload was committed.
    bool serviceDM();
    /// Read a newly published scheduler record, count its slots, and optionally stage its surfaces ahead.
    void observeSchedule( bool stage /**< [in] prepare surfaces for the record's remaining steps */ );
    /// Prepare one step's surface privately unless an identical valid surface is already staged.
    void stageStep( int step /**< [in] scheduler step of m_dmRecord */ );
    /// Framework thread starter: record the thread ID and wait for priority/cpuset setup.
    static void dmThreadStart( accelSim *app /**< [in] application owning the DM worker */ );
    /// Publish target-time surfaces or pending zeros without blocking accelerometer acquisition.
    void dmThreadExec();
    /// Join the producer after its final removal attempt and release its mappings.
    void stopDM();

  public:
    /// Construct without opening any hardware or shared-memory stream.
    accelSim();
    /// Join the output worker before its cache and mutex lifetimes end.
    ~accelSim() noexcept override;
    /// Register app and helper configuration keys.
    void setupConfig() override;
    /// Load and validate startup configuration.
    void loadConfig() override;
    /// Load configuration without starting threads, for focused tests.
    int loadConfigImpl( mx::app::appConfigurator &config /**< [in] parsed configuration */ );
    /// Register properties, prepare optional DM injection, and start the existing helper threads.
    int appStartup() override;
    /// Publish diagnostics at ordinary application cadence.
    int appLogic() override;
    /// Join acquisition, zero the owned DM contribution, and release mappings.
    int appShutdown() override;
    /// Set the two-channel stream geometry; a missing synchronization source does not block acquisition.
    int configureAcquisition();
    /// Return the nominal active sampling rate to frameGrabber.
    float fps();
    /// Initialize acquisition pacing and source counters.
    int startAcquisition();
    /// Acquire one fresh sample; positive returns never publish an old pair.
    int acquireAndCheckValid();
    /// Copy one X/Y pair, optionally calibrated zero; framework owns metadata and semaphore posts.
    int loadImageIntoStream( void *destination /**< [out] frameGrabber destination plane */ );
    /// Transfer diagnostics after publication; DM output is independent of this hook.
    int frameGrabberPostPublish( IMAGE *stream /**< [in] framework-owned output, not modified here */ );
    /// Release synchronization input during an explicit framework reconfiguration.
    int reconfig();
    /// Check the existing frame-grabber telemetry cadence.
    int checkRecordTimes();
    /// Record the existing timing telemetry schema.
    int recordTelem( const telem_fgtimings *tag /**< [in] telemetry dispatch tag */ );
    /// Handle a local nominal-rate update.
    INDI_NEWCALLBACK_DECL( accelSim, m_indiP_fps );
    /// Handle the compatible fixed-channel control.
    INDI_NEWCALLBACK_DECL( accelSim, m_indiP_numChannels );
    /// Handle timing EMA changes, independent of AR damping.
    INDI_NEWCALLBACK_DECL( accelSim, m_indiP_alpha );
    /// Handle signed synchronization phase offsets.
    INDI_NEWCALLBACK_DECL( accelSim, m_indiP_synchroDelay );
    /// Follow configured external nominal-rate metadata.
    INDI_SETCALLBACK_DECL( accelSim, m_indiP_fpsSource );
    /// Follow configured external channel metadata, accepting only two.
    INDI_SETCALLBACK_DECL( accelSim, m_indiP_numChannelsSource );
    /// Request scaled output or tagged zero removal through the existing producer.
    INDI_NEWCALLBACK_DECL( accelSim, m_indiP_dmEnable );
    /// Set the independent commissioning scale in [0,100].
    INDI_NEWCALLBACK_DECL( accelSim, m_indiP_dmScale );
};

inline accelSim::accelSim() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    m_shmimName = "acc000";
}

inline accelSim::~accelSim() noexcept
{
    stopDM();
}

inline void accelSim::setupConfig()
{
    DMSYNCOPATIONPRODUCER_SETUP_CONFIG( config );
    FRAMEGRABBER_SETUP_CONFIG( config );
    TELEMETER_SETUP_CONFIG( config );
    const auto add = [this]( const std::string &section,
                             const std::string &key,
                             const std::string &type,
                             const std::string &description )
    {
        const std::string name = section + "." + key;
        config.add( name, "", name, argType::Required, section, key, false, type, description );
    };
    add( "sim", "fps", "double", "Initial nominal sampling rate [Hz], default 6000." );
    add( "sim",
         "zeroOutput",
         "bool",
         "Publish calibrated zero-acceleration ADC counts only; default false, physics/DM unchanged." );
    add( "sim", "seed", "uint64_t", "Numerical random seed, default 42." );
    add( "sim", "modeSeed", "uint64_t", "High-frequency parameter seed, default 123." );
    add( "sim", "leverArm", "double", "Effective lever arm [m], default 10." );
    add( "sensor", "noisePSD", "double", "One-sided acceleration noise PSD, default 1e-8." );
    add( "adc", "zero", "double", "ADC zero point [counts], default 2048." );
    add( "adc", "voltsPerBit", "double", "Empirical volts per count, default 0.0083799." );
    add( "adc", "sensitivity", "double", "Empirical sensitivity, default 1.02." );
    add( "accel", "numChannels", "int", "Compatibility channel count; must be 2." );
    for( const std::string section : { "fps", "numChannels" } )
    {
        add( section, "device", "string", "External INDI source; empty disables subscription." );
        add( section, "property", "string", "External property name." );
        add( section, "element", "string", "External element name, default current." );
        add( section, "tol", "double", "Compatibility tolerance key; hardware callback does not apply it." );
    }
    add( "synchro", "shmimName", "string", "Trigger input stream; empty selects timer acquisition." );
    add( "synchro", "postDelay", "int", "Signed phase offset [us], default 0." );
    add( "synchro", "timeout", "double", "Source silence timeout [s]; sampling continues in holdover, default 0.01." );
    add( "synchro", "maxSlew", "double", "Maximum phase correction per accelerometer interval, fraction in (0,0.25]." );
    add( "synchro",
         "referenceFps",
         "double",
         "Nominal WFS cadence for phase tracking; 0 learns first measured period." );
    add( "synchro", "alpha", "double", "Timing EMA coefficient, default 0.01." );
    for( const std::string key : { "dtTransfer_ns",
                                   "wfsProcess_ns",
                                   "dtF_ns",
                                   "wfsRead_ns",
                                   "delayLockAbsThreshold_ns",
                                   "delayLockFracThreshold",
                                   "cadenceGuard_ns" } )
        add( "synchro", key, "double", "mcp3208Ctrl synchronized timing-model parameter." );
    add( "dm",
         "channelName",
         "string",
         "Existing exclusively owned channel; experiment uses dm01disp10_aux; empty disables access." );
    add( "dm", "modeCube", "string", "Approved surface-micron TT FITS cube; first two planes are used." );
    add( "dm",
         "mask",
         "string",
         "dm.calibration=fit only: optional fitting-weight FITS (dmmask); empty uses weights 1.0 everywhere." );
    add( "dm", "enabled", "bool", "Must be false; use the runtime dm_enable control." );
    add( "dm", "scale", "double", "Independent DM-only linear scale [0,100], default 0." );
    add( "dm", "maxRelativeResidual", "double", "Maximum calibrated physical fit residual, default 1e-6." );
    add( "dm",
         "calibration",
         "string",
         "fit (default): fit K to the physical tilt surface; coefficients: use dm.coefficients without a fit." );
    add( "dm",
         "coefficients",
         "string",
         "K for dm.calibration=coefficients: K00,K01,K10,K11 mode coefficient per radian (rows modes, columns X,Y)." );
    add( "dm", "threadPrio", "int", "DM worker real-time priority [0,99], as dmSpeckle; 0 (default) is SCHED_OTHER." );
    add( "dm", "cpuset", "string", "Dedicated cpuset for the DM worker, as dmSpeckle; empty (default) is unchanged." );
}

inline int accelSim::loadConfigImpl( mx::app::appConfigurator &config )
{
    if( m_dmReady )
    {
        m_shutdown = true;
        return log<software_error, -1>(
            { __FILE__, __LINE__, "DM producer configuration is startup-only; stopping for own-channel removal" } );
    }
    FRAMEGRABBER_LOAD_CONFIG( config );
    TELEMETER_LOAD_CONFIG( config );
    config( m_parameters.m_fps, "sim.fps" );
    config( m_zeroOutput, "sim.zeroOutput" );
    config( m_parameters.m_seed, "sim.seed" );
    config( m_parameters.m_modeSeed, "sim.modeSeed" );
    config( m_parameters.m_leverArm, "sim.leverArm" );
    config( m_parameters.m_noisePSD, "sensor.noisePSD" );
    config( m_parameters.m_zero, "adc.zero" );
    config( m_parameters.m_voltsPerBit, "adc.voltsPerBit" );
    config( m_parameters.m_sensitivity, "adc.sensitivity" );
    config( m_fpsDevice, "fps.device" );
    config( m_fpsProperty, "fps.property" );
    config( m_fpsElement, "fps.element" );
    config( m_fpsTol, "fps.tol" );
    config( m_channelsDevice, "numChannels.device" );
    config( m_channelsProperty, "numChannels.property" );
    config( m_channelsElement, "numChannels.element" );
    config( m_channelsTol, "numChannels.tol" );
    config( m_syncName, "synchro.shmimName" );
    config( m_syncTimeout, "synchro.timeout" );
    config( m_maxSlew, "synchro.maxSlew" );
    config( m_referenceFps, "synchro.referenceFps" );
    config( m_dtTransfer, "synchro.dtTransfer_ns" );
    config( m_wfsProcess, "synchro.wfsProcess_ns" );
    config( m_dtF, "synchro.dtF_ns" );
    config( m_wfsRead, "synchro.wfsRead_ns" );
    config( m_lockAbsolute, "synchro.delayLockAbsThreshold_ns" );
    config( m_lockFraction, "synchro.delayLockFracThreshold" );
    config( m_guard, "synchro.cadenceGuard_ns" );
    double alpha     = .01;
    int    postDelay = 0;
    double channels  = 2;
    config( alpha, "synchro.alpha" );
    config( postDelay, "synchro.postDelay" );
    config( channels, "accel.numChannels" );
    double dmScale = 0;
    DMSYNCOPATIONPRODUCER_LOAD_CONFIG( config );
    config( m_dmChannel, "dm.channelName" );
    if( !m_dmChannel.empty() && ( m_dmChannel != "dm01disp10_aux" || m_syncopationTiming.empty() ) )
        return log<software_error, -1>(
            { __FILE__, __LINE__, "DM output requires dm01disp10_aux and an existing syncopation.timingStream" } );

    config( m_dmModes, "dm.modeCube" );
    config( m_dmMask, "dm.mask" );
    config( m_dmStartupEnabled, "dm.enabled" );
    config( dmScale, "dm.scale" );
    config( m_dmResidual, "dm.maxRelativeResidual" );
    config( m_dmCalibration, "dm.calibration" );
    std::string coefficients;
    config( coefficients, "dm.coefficients" );
    config( m_dmThreadPrio, "dm.threadPrio" );
    config( m_dmThreadCpuset, "dm.cpuset" );
    try
    {
        if( m_dmThreadPrio < 0 || m_dmThreadPrio > 99 )
            throw std::invalid_argument( "dm.threadPrio must be in [0,99]" );
        if( m_dmStartupEnabled || !std::isfinite( dmScale ) || dmScale < 0 || dmScale > accelSimDetail::maxDMScale ||
            !std::isfinite( m_dmResidual ) || m_dmResidual < 0 || m_dmResidual >= 1 ||
            ( !m_dmChannel.empty() && ( m_dmChannel == m_shmimName || m_dmChannel == m_syncName ||
                                        m_dmChannel.find( '/' ) != std::string::npos || m_dmChannel == "dm01disp" ) ) )
            throw std::invalid_argument( "invalid DM configuration; startup enable is prohibited" );
        if( m_dmCalibration == "coefficients" )
        {
            std::stringstream input( coefficients );
            std::string       item;
            int               count = 0;
            while( std::getline( input, item, ',' ) )
            {
                std::size_t used = 0;
                double      value;
                try
                {
                    value = std::stod( item, &used );
                }
                catch( const std::exception & )
                {
                    used = 0;
                }
                if( count >= 4 || !used || item.find_first_not_of( " \t", used ) != std::string::npos ||
                    !std::isfinite( value ) )
                    throw std::invalid_argument( "dm.coefficients must be four finite numbers K00,K01,K10,K11" );
                m_dmCoefficients( count / 2, count % 2 ) = value;
                ++count;
            }
            if( count != 4 )
                throw std::invalid_argument( "dm.coefficients must be four finite numbers K00,K01,K10,K11" );
            if( !m_dmMask.empty() )
                throw std::invalid_argument( "dm.mask applies only to dm.calibration=fit" );
        }
        else if( m_dmCalibration != "fit" || !coefficients.empty() )
            throw std::invalid_argument(
                "dm.calibration must be fit or coefficients; dm.coefficients needs coefficients" );
        if( channels != 2 || !std::isfinite( alpha ) || postDelay < -1000000 || postDelay > 1000000 ||
            ( !m_syncName.empty() && m_syncName == m_shmimName ) )
            throw std::invalid_argument( "invalid channels, alpha, phase offset, or self-trigger stream" );
        if( !std::isfinite( m_syncTimeout ) || m_syncTimeout <= 0 || m_syncTimeout > 3600 ||
            !std::isfinite( m_maxSlew ) || m_maxSlew <= 0 || m_maxSlew > .25 || !std::isfinite( m_referenceFps ) ||
            m_referenceFps < 0 || m_referenceFps > 1e9 ||
            ( m_referenceFps > 0 && 1e9 / m_referenceFps > double( std::numeric_limits<int64_t>::max() ) ) )
            throw std::invalid_argument( "invalid synchronization timeout, slew, or reference FPS" );
        for( double *value :
             { &m_dtTransfer, &m_wfsProcess, &m_dtF, &m_wfsRead, &m_lockAbsolute, &m_lockFraction, &m_guard } )
        {
            if( !std::isfinite( *value ) )
                throw std::invalid_argument( "nonfinite timing configuration" );
            *value = std::max( 0.0, *value );
        }
        if( !std::isfinite( m_dtTransfer + m_wfsProcess + m_dtF + m_wfsRead ) )
            throw std::invalid_argument( "timing-model overflow" );
        m_model.configure( m_parameters, accelSimDetail::referenceModes( m_parameters.m_modeSeed ) );
        m_maxFrequency = m_model.m_modes.back().m_spec.m_frequency;
    }
    catch( const std::exception &error )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, error.what() } );
    }
    m_alpha        = std::clamp( alpha, 0.0, 1.0 );
    m_postDelay    = postDelay;
    m_activeFps    = m_parameters.m_fps;
    m_requestedFps = m_parameters.m_fps;
    m_dmScale      = dmScale;
    return 0;
}

inline void accelSim::loadConfig()
{
    if( loadConfigImpl( config ) < 0 )
        m_shutdown = true;
}

inline int accelSim::setupProperties()
{
    CREATE_REG_INDI_NEW_TOGGLESWITCH( m_indiP_dmEnable, "dm_enable" );
    m_indiP_dmEnable["toggle"] = pcf::IndiElement::Off;
    CREATE_REG_INDI_NEW_NUMBERD(
        m_indiP_dmScale, "dm_scale", 0, accelSimDetail::maxDMScale, .001, "%.3f", "DM scale", "DM" );
    m_indiP_dmScale["current"] = m_dmScale.load();
    m_indiP_dmScale["target"]  = m_dmScale.load();
    CREATE_REG_INDI_RO_NUMBER( m_indiP_dmStatus, "dm_status", "DM readiness", "DM" );
    for( const char *name : { "ready",
                              "attached",
                              "active",
                              "fault",
                              "requested",
                              "zero_acknowledged",
                              "transport_fault",
                              "poll_waits",
                              "requests",
                              "timeline_misses",
                              "late_returns",
                              "rejected",
                              "published",
                              "copied",
                              "discarded",
                              "outstanding" } )
        m_indiP_dmStatus.add( pcf::IndiElement( name ) );
    createROIndiText( m_indiP_dmCalibration, "dm_calibration", "status" );
    for( const char *name :
         { "method", "support", "k00", "k01", "k10", "k11", "residual_x", "residual_y", "fit_pixels" } )
        m_indiP_dmCalibration.add( pcf::IndiElement( name, "" ) );
    if( registerIndiPropertyReadOnly( m_indiP_dmCalibration ) < 0 )
        return -1;
    CREATE_REG_INDI_RO_NUMBER( m_indiP_dmSteps, "dm_steps", "DM per-subframe counts", "DM" );
    for( const std::string kind : { "advertised_", "requests_", "published_", "copied_" } )
        for( size_t step = 0; step < accelSimDetail::dmStatistics::c_steps; ++step )
            m_indiP_dmSteps.add( pcf::IndiElement( kind + std::to_string( step ) ) );
    for( const char *name : { "skipped_slots", "prepared_hits", "prepared_misses" } )
        m_indiP_dmSteps.add( pcf::IndiElement( name ) );
    CREATE_REG_INDI_RO_NUMBER( m_indiP_dmTiming, "dm_timing", "DM timing per report interval", "DM" );
    for( const char *name : c_dmTimingNames )
        m_indiP_dmTiming.add( pcf::IndiElement( name ) );
    CREATE_REG_INDI_NEW_NUMBERF( m_indiP_fps, "fps", 0, 10000, 1, "%d", "", "" );
    CREATE_REG_INDI_NEW_NUMBERF( m_indiP_numChannels, "numChannels", 0, 8, 1, "%d", "", "" );
    CREATE_REG_INDI_NEW_NUMBERF( m_indiP_alpha, "alpha", 0., 1., .00001, "%.5f", "", "" );
    CREATE_REG_INDI_NEW_NUMBERF( m_indiP_synchroDelay, "synchroDelay", -1000000, 1000000, 1, "%d", "us", "" );
    for( const char *element : { "current", "target" } )
    {
        m_indiP_fps[element]          = m_activeFps.load();
        m_indiP_numChannels[element]  = 2;
        m_indiP_alpha[element]        = m_alpha.load();
        m_indiP_synchroDelay[element] = m_postDelay.load();
    }
    CREATE_REG_INDI_RO_NUMBER( m_indiP_timingDiag, "timingDiag", "Timing Diagnostics", "Diagnostics" );
    for( const char *name : c_timingNames )
        m_indiP_timingDiag.add( pcf::IndiElement( name ) );
    CREATE_REG_INDI_RO_NUMBER( m_indiP_simDiag, "sim_diag", "Simulation Diagnostics", "Diagnostics" );
    for( const char *name : { "skipped_publications",
                              "discontinuities",
                              "late_samples",
                              "max_lateness_us",
                              "source_period_ratio",
                              "clip_x_low",
                              "clip_x_high",
                              "clip_y_low",
                              "clip_y_high",
                              "lateness_last_us",
                              "lateness_window_max_us",
                              "lock_phase_ok",
                              "lock_timing_ok",
                              "acq_timer_slack_ns",
                              "acq_sched_policy",
                              "acq_priority" } )
        m_indiP_simDiag.add( pcf::IndiElement( name ) );
    if( !m_fpsDevice.empty() )
    {
        REG_INDI_SETPROP( m_indiP_fpsSource, m_fpsDevice, m_fpsProperty );
    }
    if( !m_channelsDevice.empty() )
    {
        REG_INDI_SETPROP( m_indiP_numChannelsSource, m_channelsDevice, m_channelsProperty );
    }
    return 0;
}

inline bool accelSim::calibrationUnchanged() const
{
    const auto same = []( const std::string &path, const struct stat &original )
    {
        struct stat current;
        if( stat( path.c_str(), &current ) != 0 || current.st_ino != original.st_ino ||
            current.st_size != original.st_size || current.st_mtime != original.st_mtime ||
            current.st_ctime != original.st_ctime )
            return false;
#ifdef __APPLE__
        return current.st_mtimespec.tv_nsec == original.st_mtimespec.tv_nsec &&
               current.st_ctimespec.tv_nsec == original.st_ctimespec.tv_nsec;
#else
        return current.st_mtim.tv_nsec == original.st_mtim.tv_nsec &&
               current.st_ctim.tv_nsec == original.st_ctim.tv_nsec;
#endif
    };
    return same( m_dmModes, m_dmModesStat ) && ( m_dmMask.empty() || same( m_dmMask, m_dmMaskStat ) );
}

inline void accelSim::loadDMCalibration()
{
    if( stat( m_dmModes.c_str(), &m_dmModesStat ) != 0 )
        throw std::invalid_argument( "dm.modeCube not found: " + m_dmModes );
    if( !m_dmMask.empty() && stat( m_dmMask.c_str(), &m_dmMaskStat ) != 0 )
        throw std::invalid_argument( "dm.mask not found: " + m_dmMask );
    mx::fits::fitsFile<double>    ff;
    mx::improc::eigenCube<double> modes, mask;
    if( ff.read( modes, m_dmModes ) != mx::error_t::noerror || modes.rows() != 50 || modes.cols() != 50 ||
        modes.planes() < 2 )
        throw std::invalid_argument( "dm.modeCube must be a readable 50x50 cube with at least two planes" );
    if( !m_dmMask.empty() && ( ff.read( mask, m_dmMask ) != mx::error_t::noerror || mask.rows() != 50 ||
                               mask.cols() != 50 || mask.planes() != 1 ) )
        throw std::invalid_argument( "dm.mask must be a readable single 50x50 plane" );
    if( !calibrationUnchanged() )
        throw std::invalid_argument( "DM calibration files changed while loading" );
    const std::vector<double> mode0( modes.image( 0 ).data(), modes.image( 0 ).data() + 2500 );
    const std::vector<double> mode1( modes.image( 1 ).data(), modes.image( 1 ).data() + 2500 );
    if( m_dmCalibration == "coefficients" )
        m_dmInjection.prepareCoefficients( mode0, mode1, m_dmCoefficients );
    else // Without dm.mask the fit weights are 1.0 on every pixel; the output is the full modes either way.
        m_dmInjection.prepare( mode0,
                               mode1,
                               m_dmMask.empty() ? std::vector<double>()
                                                : std::vector<double>( mask.data(), mask.data() + 2500 ),
                               m_dmResidual );
}

inline std::string accelSim::dmSupport() const
{
    if( m_dmCalibration == "coefficients" )
        return "dm.coefficients, no fit";
    return m_dmMask.empty() ? "uniform weights 1.0 over 50x50" : "dm.mask " + m_dmMask;
}

inline bool accelSim::prepareDMCalibration()
{
    try
    {
        loadDMCalibration();
    }
    catch( const std::exception &error )
    {
        m_dmReady       = false;
        m_dmUnavailable = "calibration " + m_dmCalibration + " (" + dmSupport() + "): " + error.what();
        log<text_log>( "DM injection unavailable: " + m_dmUnavailable, logPrio::LOG_ERROR );
        return false;
    }
    m_dmUnavailable = "";
    const auto &k   = m_dmInjection.mapping();
    char        text[400];
    std::snprintf( text,
                   sizeof( text ),
                   "DM calibration %s (%s): K [coefficient/rad] = [[%.12g, %.12g], [%.12g, %.12g]]; fit pixels %zu; "
                   "relative residuals X %.3g, Y %.3g",
                   accelSimDetail::dmCalibrationName( m_dmInjection.method() ),
                   dmSupport().c_str(),
                   k( 0, 0 ),
                   k( 0, 1 ),
                   k( 1, 0 ),
                   k( 1, 1 ),
                   m_dmInjection.fitPixels(),
                   m_dmInjection.residual()[0],
                   m_dmInjection.residual()[1] );
    log<text_log>( text, logPrio::LOG_NOTICE );
    return true;
}

inline void accelSim::prepareDM()
{
    if( m_dmChannel.empty() || !prepareDMCalibration() )
        return;
    m_dmReady = true;
    // The dmSpeckle worker pattern: framework thread with configured real-time priority and cpuset.
    if( threadStart( m_dmThread,
                     m_dmThreadInit,
                     m_dmThreadID,
                     m_dmThreadProp,
                     m_dmThreadPrio,
                     m_dmThreadCpuset,
                     "dm",
                     this,
                     dmThreadStart ) < 0 )
    {
        stopDM();
        m_dmReady       = false;
        m_dmUnavailable = "DM worker thread priority/cpuset setup failed";
        log<text_log>( "DM injection unavailable: " + m_dmUnavailable, logPrio::LOG_ERROR );
        return;
    }
    log<text_log>( "DM channel prepared disabled: " + m_dmChannel + "; exclusive operator handover required",
                   logPrio::LOG_NOTICE );
}

inline void accelSim::updateDMINDI()
{
    bool                         attached, active, enabled, zeroAcknowledged, transportFault, zeroPending, outstanding;
    std::array<uint64_t, 9>      counts;
    accelSimDetail::dmStatistics stats;
    { //mutex scope
        // Copy only; INDI I/O below never runs while the DM worker could be waiting for this mutex.
        std::lock_guard<std::mutex> guard( m_dmMutex );
        attached         = m_syncopationSchedule.md != nullptr;
        active           = m_dmActive;
        enabled          = m_dmEnabled;
        zeroAcknowledged = m_dmZeroAcknowledged;
        transportFault   = m_dmTransportFault;
        zeroPending      = m_dmZeroPending;
        outstanding      = m_syncopationOutstanding;
        counts           = { m_dmPollWaits,
                             m_dmRequests,
                             m_dmTimelineMisses,
                             m_dmLateReturns,
                             m_dmRejected,
                             m_dmPublished,
                             m_dmCopied,
                             m_dmDiscarded,
                             0 };
        stats            = m_dmStats;
        for( auto *interval : { &m_dmStats.m_pollGap,
                                &m_dmStats.m_slack,
                                &m_dmStats.m_prepare,
                                &m_dmStats.m_margin,
                                &m_dmStats.m_publish,
                                &m_dmStats.m_lead,
                                &m_dmStats.m_receipt } )
            *interval = {};
        m_dmStats.m_marginHistogram = {};
    }
    std::lock_guard<std::mutex> guard( m_controlMutex );
    const bool                  fault     = ( !m_dmChannel.empty() && !m_dmReady ) || !m_dmCommandValid.load();
    m_indiP_dmStatus["ready"]             = static_cast<int>( m_dmReady );
    m_indiP_dmStatus["attached"]          = static_cast<int>( attached );
    m_indiP_dmStatus["active"]            = static_cast<int>( active );
    m_indiP_dmStatus["requested"]         = static_cast<int>( enabled );
    m_indiP_dmStatus["zero_acknowledged"] = static_cast<int>( zeroAcknowledged );
    m_indiP_dmStatus["transport_fault"]   = static_cast<int>( transportFault );
    m_indiP_dmStatus["fault"]             = static_cast<int>( fault );
    m_indiP_dmStatus["poll_waits"]        = counts[0];
    m_indiP_dmStatus["requests"]          = counts[1];
    m_indiP_dmStatus["timeline_misses"]   = counts[2];
    m_indiP_dmStatus["late_returns"]      = counts[3];
    m_indiP_dmStatus["rejected"]          = counts[4];
    m_indiP_dmStatus["published"]         = counts[5];
    m_indiP_dmStatus["copied"]            = counts[6];
    m_indiP_dmStatus["discarded"]         = counts[7];
    m_indiP_dmStatus["outstanding"]       = static_cast<int>( outstanding );
    m_indiP_dmStatus.setState( fault || transportFault ? INDI_ALERT : INDI_OK );
    for( size_t step = 0; step < stats.c_steps; ++step )
    {
        m_indiP_dmSteps["advertised_" + std::to_string( step )] = stats.m_advertised[step];
        m_indiP_dmSteps["requests_" + std::to_string( step )]   = stats.m_requests[step];
        m_indiP_dmSteps["published_" + std::to_string( step )]  = stats.m_published[step];
        m_indiP_dmSteps["copied_" + std::to_string( step )]     = stats.m_copied[step];
    }
    m_indiP_dmSteps["skipped_slots"]   = stats.m_skippedSlots;
    m_indiP_dmSteps["prepared_hits"]   = stats.m_preparedHits;
    m_indiP_dmSteps["prepared_misses"] = stats.m_preparedMisses;
    const std::array<double, c_dmTimingNames.size()> timing{ double( stats.m_margin.m_count ),
                                                             stats.m_pollGap.mean(),
                                                             stats.m_pollGap.m_max,
                                                             stats.m_slack.m_min,
                                                             stats.m_slack.mean(),
                                                             stats.m_prepare.mean(),
                                                             stats.m_prepare.m_max,
                                                             stats.m_publish.mean(),
                                                             stats.m_publish.m_max,
                                                             stats.m_margin.m_min,
                                                             stats.m_margin.mean(),
                                                             stats.m_margin.rms(),
                                                             stats.marginQuantile( .01 ),
                                                             stats.marginQuantile( .001 ),
                                                             stats.m_lead.m_min,
                                                             stats.m_lead.mean(),
                                                             stats.m_receipt.mean(),
                                                             stats.m_receipt.m_max,
                                                             stats.m_slack.rms() };
    for( size_t n = 0; n < timing.size(); ++n )
        m_indiP_dmTiming[c_dmTimingNames[n]] = timing[n];
    // Calibration is fixed after startup: method, K [coefficient/rad], fit quality, or why output is unavailable.
    const auto &k      = m_dmInjection.mapping();
    auto        number = []( double value )
    {
        char text[32];
        std::snprintf( text, sizeof( text ), "%.12g", value );
        return std::string( text );
    };
    m_indiP_dmCalibration["status"]     = m_dmReady ? std::string( "ready" ) : m_dmUnavailable;
    m_indiP_dmCalibration["method"]     = std::string( accelSimDetail::dmCalibrationName( m_dmInjection.method() ) );
    m_indiP_dmCalibration["support"]    = dmSupport();
    m_indiP_dmCalibration["k00"]        = number( k( 0, 0 ) );
    m_indiP_dmCalibration["k01"]        = number( k( 0, 1 ) );
    m_indiP_dmCalibration["k10"]        = number( k( 1, 0 ) );
    m_indiP_dmCalibration["k11"]        = number( k( 1, 1 ) );
    m_indiP_dmCalibration["residual_x"] = number( m_dmInjection.residual()[0] );
    m_indiP_dmCalibration["residual_y"] = number( m_dmInjection.residual()[1] );
    m_indiP_dmCalibration["fit_pixels"] = std::to_string( m_dmInjection.fitPixels() );
    m_indiP_dmCalibration.setState( m_dmReady || m_dmChannel.empty() ? INDI_OK : INDI_ALERT );
    if( m_indiDriver )
    {
        m_indiDriver->sendSetProperty( m_indiP_dmCalibration );
        m_indiDriver->sendSetProperty( m_indiP_dmStatus );
        m_indiDriver->sendSetProperty( m_indiP_dmSteps );
        m_indiDriver->sendSetProperty( m_indiP_dmTiming );
    }
    updateSwitchIfChanged( m_indiP_dmEnable,
                           "toggle",
                           enabled ? pcf::IndiElement::On : pcf::IndiElement::Off,
                           fault || transportFault                 ? INDI_ALERT
                           : zeroPending || ( enabled && !active ) ? INDI_BUSY
                                                                   : INDI_OK );
}

inline bool accelSim::serviceDM()
{
    std::lock_guard<std::mutex> guard( m_dmMutex );
    auto                       &stats = m_dmStats;
    if( !m_dmEnabled && !m_dmZeroPending && !m_syncopationOutstanding )
    {
        stats.m_lastPoll = 0; // Idle periods are not wake latency.
        return false;
    }
    if( !m_dmImage.md )
    {
        if( ImageStreamIO_openIm( &m_dmImage, m_dmChannel.c_str() ) != 0 )
        {
            m_dmImage          = {};
            m_dmTransportFault = true;
            return false;
        }
    }
    if( !syncopationGeometry( m_dmImage ) || m_dmImage.md->size[0] != 50 || m_dmImage.md->size[1] != 50 )
    {
        m_dmTransportFault = true;
        DMSYNCOPATIONPRODUCER_APP_SHUTDOWN;
        ImageStreamIO_closeIm( &m_dmImage );
        m_dmImage = {};
        return false;
    }
    const int64_t polledAt = synco::monotonicNs();
    if( stats.m_lastPoll )
        stats.m_pollGap.add( ( polledAt - stats.m_lastPoll ) / 1e3 );
    stats.m_lastPoll = polledAt;
    synco::Request request{};
    const int      polled = syncopationPoll( request, m_dmImage );
    m_dmTransportFault    = polled < 0;
    // Poll may have released the buffer on a receipt arriving during this call.
    // The prior commit and its echo remain available until we submit another command.
    if( m_dmAwaitingAck && polled >= 0 )
    {
        synco::AckRecord ack{};
        if( synco::readRecord( m_syncopationAck, ack ) && synco::acknowledges( ack, m_syncopationPending ) )
        {
            stats.m_receipt.add( ( polledAt - m_syncopationPending[synco::cmdCutoff] ) / 1e3 );
            if( ack[synco::ackStatus] == synco::copied )
            {
                ++m_dmCopied;
                ++stats.m_copied[stats.index( int( m_syncopationPending[synco::cmdStep] ) )];
                m_dmActive           = !m_dmPendingZero;
                m_dmZeroAcknowledged = m_dmPendingZero;
                if( m_dmPendingZero )
                    m_dmZeroPending = false;
            }
            else
                ++m_dmDiscarded;
            m_dmAwaitingAck = false;
        }
    }
    if( polled == 0 )
        ++m_dmPollWaits;
    else if( polled > 0 )
    {
        ++m_dmRequests;
        ++stats.m_requests[stats.index( request.step )];
        stats.m_slack.add( ( request.cutoffNs - polledAt ) / 1e3 );
        // Slots never returned between consecutive requests of one plan (ordinal cycle*steps+step is consecutive).
        synco::Record record{};
        if( synco::readRecord( m_syncopationSchedule, record ) && record[synco::epoch] == request.epoch &&
            record[synco::generation] == request.generation && record[synco::steps] > 0 )
        {
            const int64_t slot = request.cycle * record[synco::steps] + request.step;
            if( stats.m_lastSlot >= 0 && stats.m_lastEpoch == request.epoch &&
                stats.m_lastGeneration == request.generation && slot > stats.m_lastSlot + 1 )
                stats.m_skippedSlots += slot - stats.m_lastSlot - 1;
            stats.m_lastSlot       = slot;
            stats.m_lastEpoch      = request.epoch;
            stats.m_lastGeneration = request.generation;
        }
        else
            stats.m_lastSlot = -1;
    }
    if( !m_syncopationOutstanding )
        m_dmAwaitingAck = false; // The base also releases ownership on scheduler reset.
    if( polled < 0 )
    {
        DMSYNCOPATIONPRODUCER_APP_SHUTDOWN;
        ImageStreamIO_closeIm( &m_dmImage );
        m_dmImage = {};
    }
    if( polled <= 0 || ( !m_dmEnabled && !m_dmZeroPending ) )
    {
        if( polled == 0 ) // Nothing publishable now: count new records and prepare their surfaces ahead.
            observeSchedule( m_dmEnabled && !m_dmZeroPending );
        return false;
    }
    static const std::array<float, 2500> zeros{};
    const float                         *pixels     = zeros.data();
    bool                                 zero       = m_dmZeroPending;
    const size_t                         stageIndex = size_t( request.step );
    if( !zero && stageIndex < m_dmStage.size() && m_dmStage[stageIndex].m_valid &&
        syncopationSame( m_dmStage[stageIndex].m_request, request ) &&
        m_dmStage[stageIndex].m_scale == m_dmScale.load() &&
        m_dmStage[stageIndex].m_timeline == m_timeline.generation() )
    {
        pixels = m_dmStage[stageIndex].m_pixels.data(); // Prepared ahead for exactly this slot.
        ++stats.m_preparedHits;
        m_dmCommandValid = true;
    }
    else if( !zero )
    {
        ++stats.m_preparedMisses;
        // Same realization as acc000, evaluated at the scheduler's CLOCK_MONOTONIC target, as dmSpeckle does.
        accelSimDetail::sample physical;
        if( !m_timeline.physical( request.targetNs, physical ) )
        {
            ++m_dmTimelineMisses; // Skip this slot; the scheduler retains the last copied contribution.
            return false;
        }
        try
        {
            pixels           = m_dmInjection.synthesize( physical, m_dmScale.load() ).data();
            m_dmCommandValid = true;
        }
        catch( const std::exception & )
        {
            m_dmCommandValid = false;
            zero             = true; // Remove the contribution until a valid command can be synthesized.
        }
    }
    const int64_t prepared  = synco::monotonicNs();
    const bool    published = syncopationPublish( pixels, request, m_dmImage );
    const int64_t returned  = synco::monotonicNs();
    stats.m_prepare.add( ( prepared - polledAt ) / 1e3 );
    stats.m_publish.add( ( returned - prepared ) / 1e3 );
    stats.m_lastPoll = returned; // The intentional wait to cutoff-writeLead is not wake latency.
    if( !published )
    {
        if( returned >= request.cutoffNs )
            ++m_dmLateReturns;
        else
            ++m_dmRejected;
        return false;
    }
    ++m_dmPublished;
    ++stats.m_published[stats.index( request.step )];
    const double margin = ( request.cutoffNs - m_syncopationPending[synco::cmdReadyNs] ) / 1e3;
    stats.m_margin.add( margin );
    ++stats.m_marginHistogram[size_t( std::clamp( margin, 0.0, double( stats.m_marginHistogram.size() - 1 ) ) )];
    stats.m_lead.add( ( request.targetNs - m_syncopationPending[synco::cmdReadyNs] ) / 1e3 );
    m_dmPendingZero      = zero;
    m_dmAwaitingAck      = true;
    m_dmZeroAcknowledged = false;
    return true;
}

inline void accelSim::observeSchedule( bool stage )
{
    if( !m_syncopationSchedule.md )
        return;
    const uint64_t counter = m_syncopationSchedule.md->cnt0;
    synco::Record  record{};
    if( counter != m_dmRecordCounter && synco::readRecord( m_syncopationSchedule, record ) )
    {
        m_dmRecordCounter = counter;
        m_dmRecord        = record;
        auto &stats       = m_dmStats;
        if( synco::validProductionRecord( record, synco::monotonicNs() ) &&
            ( record[synco::epoch] != stats.m_lastRecordEpoch ||
              record[synco::generation] != stats.m_lastRecordGeneration ||
              record[synco::cycle] != stats.m_lastRecordCycle ) )
        {
            for( int step = 0; step < record[synco::steps]; ++step )
                if( record[synco::cutoffs + step] )
                    ++stats.m_advertised[stats.index( step )];
            stats.m_lastRecordEpoch      = record[synco::epoch];
            stats.m_lastRecordGeneration = record[synco::generation];
            stats.m_lastRecordCycle      = record[synco::cycle];
        }
    }
    const int64_t now = synco::monotonicNs();
    if( !stage || !synco::validProductionRecord( m_dmRecord, now ) )
        return;
    for( int step = 0; step < std::min<int>( m_dmRecord[synco::steps], int( m_dmStage.size() ) ); ++step )
        if( m_dmRecord[synco::cutoffs + step] > now )
            stageStep( step );
}

inline void accelSim::stageStep( int step )
{
    const auto request = synco::request( m_dmRecord, step );
    const auto scale   = m_dmScale.load();
    auto      &entry   = m_dmStage[size_t( step )];
    if( entry.m_valid && syncopationSame( entry.m_request, request ) && entry.m_scale == scale &&
        entry.m_timeline == m_timeline.generation() )
        return;
    entry.m_valid = false;
    accelSimDetail::sample physical;
    uint64_t               generation = 0;
    if( !m_timeline.physical( request.targetNs, physical, &generation ) )
        return; // Publication will evaluate on demand and count a timeline miss if still unavailable.
    try
    {
        const auto &surface = m_dmInjection.synthesize( physical, scale );
        std::copy( surface.begin(), surface.end(), entry.m_pixels.begin() );
        entry.m_request  = request;
        entry.m_scale    = scale;
        entry.m_timeline = generation;
        entry.m_valid    = true;
    }
    catch( const std::exception & )
    {
        // An invalid surface is never staged; the publication path reports the numerical fault.
    }
}

inline void accelSim::dmThreadStart( accelSim *app )
{
    app->m_dmThreadID = syscall( SYS_gettid );
    // Wait for threadStart to finish priority and cpuset placement, as frameGrabber and dmSpeckle do.
    while( app->m_dmThreadInit && !app->shutdown() && !app->m_dmStop.load() )
        std::this_thread::sleep_for( std::chrono::milliseconds( 10 ) );
    app->dmThreadExec();
}

inline void accelSim::dmThreadExec()
{
    int64_t stopping = 0;
    while( true )
    {
        if( m_dmStop.load() || shutdown() )
        { //mutex scope
            std::lock_guard<std::mutex> guard( m_dmMutex );
            if( !stopping )
            {
                stopping        = monotonicNow();
                m_dmZeroPending = m_dmZeroPending || m_dmEnabled || m_dmActive || m_syncopationOutstanding;
                m_dmEnabled     = false;
            }
            // Preserve the existing bounded shutdown attempt; never invent an acknowledgement.
            if( !m_dmZeroPending || monotonicNow() - stopping >= 100000000 )
                break;
        }
        if( serviceDM() )
            continue; // Poll again at once for the receipt and the next slot.
        // dmSpeckle's polling intervals: 10 us while awaiting a request or receipt, 1 ms idle, 100 ms after a
        // transport failure. With a real-time priority there is no timer slack on these sleeps.
        std::chrono::microseconds pause{ 10 };
        { //mutex scope
            std::lock_guard<std::mutex> guard( m_dmMutex );
            if( !m_dmEnabled && !m_dmZeroPending && !m_syncopationOutstanding )
                pause = std::chrono::microseconds( 1000 );
            else if( m_dmTransportFault )
                pause = std::chrono::microseconds( 100000 );
        }
        std::this_thread::sleep_for( pause );
    }
}

inline void accelSim::stopDM()
{
    m_dmStop = true;
    if( m_dmThread.joinable() )
        m_dmThread.join();
    accelSimDetail::dmProducer::m_shutdown = true;
    DMSYNCOPATIONPRODUCER_APP_SHUTDOWN;
    if( m_dmImage.md )
    {
        ImageStreamIO_closeIm( &m_dmImage );
        m_dmImage = {};
    }
}

inline int accelSim::appStartup()
{
    if( setupProperties() < 0 )
        return -1;
    prepareDM();
    TELEMETER_APP_STARTUP;
    FRAMEGRABBER_APP_STARTUP;
    state( stateCodes::OPERATING );
    return 0;
}

inline int accelSim::appLogic()
{
    FRAMEGRABBER_APP_LOGIC;
    TELEMETER_APP_LOGIC;
    FRAMEGRABBER_UPDATE_INDI;
    diagnostics snapshot;
    { //mutex scope
        std::lock_guard<std::mutex> guard( m_snapshotMutex );
        snapshot = m_snapshot;
    }
    updateINDI( snapshot );
    updateDMINDI();
    return 0;
}

inline void accelSim::updateINDI( const diagnostics &snapshot )
{
    { //mutex scope
        std::lock_guard<std::mutex> guard( m_controlMutex );
        const auto                  fpsState = m_indiP_fps.getState() == INDI_ALERT           ? INDI_ALERT
                                               : m_rateApplied.load() == m_rateRequest.load() ? INDI_OK
                                                                                              : INDI_BUSY;
        updatesIfChanged<double>(
            m_indiP_fps, { "current", "target" }, { m_activeFps.load(), m_requestedFps.load() }, fpsState );
        updatesIfChanged<double>(
            m_indiP_alpha, { "current", "target" }, { m_alpha.load(), m_alpha.load() }, m_indiP_alpha.getState() );
        updatesIfChanged<int>( m_indiP_synchroDelay,
                               { "current", "target" },
                               { m_postDelay.load(), m_postDelay.load() },
                               m_indiP_synchroDelay.getState() );
        indi::updatesIfChanged<double>( m_indiP_timingDiag,
                                        c_timingNames,
                                        std::vector<double>( snapshot.m_timing.begin(), snapshot.m_timing.end() ),
                                        m_indiDriver );
        updatesIfChanged<double>( m_indiP_simDiag,
                                  { "skipped_publications",
                                    "discontinuities",
                                    "late_samples",
                                    "max_lateness_us",
                                    "source_period_ratio",
                                    "clip_x_low",
                                    "clip_x_high",
                                    "clip_y_low",
                                    "clip_y_high",
                                    "lateness_last_us",
                                    "lateness_window_max_us",
                                    "lock_phase_ok",
                                    "lock_timing_ok",
                                    "acq_timer_slack_ns",
                                    "acq_sched_policy",
                                    "acq_priority" },
                                  { double( snapshot.m_skipped ),
                                    double( snapshot.m_discontinuities ),
                                    double( snapshot.m_late ),
                                    snapshot.m_maxLateUs,
                                    snapshot.m_cadenceRatio,
                                    double( snapshot.m_low[0] ),
                                    double( snapshot.m_high[0] ),
                                    double( snapshot.m_low[1] ),
                                    double( snapshot.m_high[1] ),
                                    snapshot.m_lastLateUs,
                                    std::max( snapshot.m_windowLateUs, snapshot.m_previousLateUs ),
                                    double( snapshot.m_lockPhase ),
                                    double( snapshot.m_lockTiming ),
                                    snapshot.m_timerSlackNs,
                                    double( snapshot.m_policy ),
                                    double( snapshot.m_priority ) } );
    }
}

inline int accelSim::appShutdown()
{
    FRAMEGRABBER_APP_SHUTDOWN;
    stopDM();
    if( m_dmZeroPending )
        log<text_log>( "DM final zero unacknowledged; operator cleanup required", logPrio::LOG_ERROR );
    m_dmReady = false;
    closeSync();
    log<text_log>( "accelSim skipped=" + std::to_string( m_diagnostics.m_skipped ) +
                   " discontinuities=" + std::to_string( m_diagnostics.m_discontinuities ) +
                   " late_samples=" + std::to_string( m_diagnostics.m_late ) +
                   " max_lateness_us=" + std::to_string( m_diagnostics.m_maxLateUs ) +
                   " clip_x_low=" + std::to_string( m_diagnostics.m_low[0] ) +
                   " clip_x_high=" + std::to_string( m_diagnostics.m_high[0] ) +
                   " clip_y_low=" + std::to_string( m_diagnostics.m_low[1] ) +
                   " clip_y_high=" + std::to_string( m_diagnostics.m_high[1] ) );
    TELEMETER_APP_SHUTDOWN;
    return 0;
}

inline int64_t accelSim::monotonicNow()
{
    timespec now;
    if( clock_gettime( CLOCK_MONOTONIC, &now ) != 0 )
        throw std::runtime_error( "CLOCK_MONOTONIC failed" );
    return int64_t( now.tv_sec ) * 1000000000LL + now.tv_nsec;
}

inline int accelSim::waitUntil( int64_t deadline )
{
    while( !shutdown() && !m_reconfig )
    {
        if( m_requestedFps.load() != m_activeFps.load() )
            return 1;
        const int64_t remaining = deadline - monotonicNow();
        if( remaining <= 0 )
            return 0;
        // Absolute schedule, bounded sleeps for shutdown/rate changes; no busy wait.
        const int64_t interval = std::min<int64_t>( remaining, 1000000 );
        timespec      delay{ 0, static_cast<long>( interval ) };
        if( nanosleep( &delay, nullptr ) < 0 && errno != EINTR )
            return -1;
    }
    return 1;
}

inline int accelSim::configureAcquisition()
{
    m_width    = 2;
    m_height   = 1;
    m_dataType = _DATATYPE_UINT16;
    if( !m_syncName.empty() )
        openSync(); // Optional reference: absence must not block acc000 creation.
    return 0;
}

inline float accelSim::fps()
{
    return m_activeFps.load();
}

inline int accelSim::startAcquisition()
{
    // In synchronized mode this worker reaches each deadline through a sem_timedwait timeout. Futex timeouts carry
    // the thread's timer slack (50 us by default), and older kernels apply it to SCHED_FIFO threads as well, which
    // makes every synchronized sample ~50 us late. Timer mode's clock_nanosleep is exempt for real-time threads.
    // 1 ns is the minimum slack; 0 would restore the default. sim_diag reports the effective value.
#ifdef __linux__
    prctl( PR_SET_TIMERSLACK, 1UL, 0UL, 0UL, 0UL );
    const int slack              = prctl( PR_GET_TIMERSLACK, 0UL, 0UL, 0UL, 0UL );
    m_diagnostics.m_timerSlackNs = slack >= 0 ? slack : -1;
    sched_param parameter{};
    m_diagnostics.m_policy   = sched_getscheduler( 0 );
    m_diagnostics.m_priority = sched_getparam( 0, &parameter ) == 0 ? parameter.sched_priority : 0;
#endif
    m_schedule.reset( monotonicNow(), m_activeFps.load() );
    m_timeline.reset( m_schedule.m_epoch, m_model ); // Node k is the nominal time of accelerometer tick k.
    m_lastAcquisition = m_triggerPrediction = 0;
    if( !m_syncName.empty() )
        m_phaseClock.reset( m_schedule.m_epoch, m_activeFps.load() );
    m_nextSyncProbe        = 0;
    m_measureReference     = false;
    m_fresh                = false;
    m_diagnostics.m_timing = {};
    m_diagnostics.invalidateSync( m_postDelay.load() );
    m_diagnostics.m_timing[modeCode]    = m_syncName.empty() ? 0 : 1;
    m_diagnostics.m_timing[delayTarget] = m_postDelay.load();
    if( m_syncOpen )
        ImageStreamIO_semflush( &m_syncImage, m_syncIndex );
    snapshotDiagnostics( true );
    return 0;
}

inline int accelSim::openSync()
{
    if( m_syncOpen )
        return 0;
    char        filename[1024];
    struct stat status;
    ImageStreamIO_filename( filename, sizeof( filename ), m_syncName.c_str() );
    if( stat( filename, &status ) != 0 || ImageStreamIO_openIm( &m_syncImage, m_syncName.c_str() ) != 0 )
        return 1;
    m_syncOpen = true;
    if( m_syncImage.md[0].sem < SEMAPHORE_MAXVAL || stat( filename, &status ) != 0 )
    {
        m_syncIndex = -1;
        closeSync();
        return 1;
    }
    m_syncInode = status.st_ino;
    m_syncIndex = ImageStreamIO_getsemwaitindex( &m_syncImage, 5 );
    if( m_syncIndex < 0 )
    {
        closeSync();
        return 1;
    }
    return 0;
}

inline void accelSim::closeSync()
{
    if( m_syncOpen )
    {
        if( m_syncIndex >= 0 )
            m_syncImage.semReadPID[m_syncIndex] = 0;
        ImageStreamIO_closeIm( &m_syncImage );
    }
    m_syncImage = {};
    m_syncOpen  = false;
    m_syncIndex = -1;
    m_syncInode = 0;
}

inline bool accelSim::syncStale()
{
    if( !m_syncOpen || m_syncImage.md[0].sem <= 0 )
        return true;
    char        filename[1024];
    struct stat status;
    ImageStreamIO_filename( filename, sizeof( filename ), m_syncName.c_str() );
    return stat( filename, &status ) != 0 || status.st_ino != m_syncInode;
}

inline void accelSim::syncTiming( uint64_t producer, int64_t arrival )
{
    if( !m_phaseClock.reference( producer,
                                 arrival,
                                 m_alpha.load(),
                                 m_dtTransfer + m_wfsProcess + m_dtF + m_wfsRead,
                                 m_postDelay.load() * 1000.0,
                                 m_referenceFps ) )
        return;
    if( !m_phaseClock.m_valid )
    {
        m_diagnostics.invalidateSync( m_postDelay.load() );
        m_triggerPrediction = 0;
        m_measureReference  = false;
    }
    auto &t          = m_diagnostics.m_timing;
    t[producerId]    = m_phaseClock.m_producer;
    t[producerDelta] = m_phaseClock.m_delta;
    if( m_phaseClock.m_delta > 1 )
    {
        t[dropped] += m_phaseClock.m_delta - 1;
        ++t[gapCount];
    }
    t[semaphorePeriod] = m_phaseClock.m_sourcePeriod / 1000;
    t[delayModel]      = m_phaseClock.m_modelDelay / 1000;
    t[delayTarget]     = m_phaseClock.m_valid ? m_phaseClock.m_delay / 1000 : m_postDelay.load();
    if( !m_phaseClock.m_valid )
        m_triggerPrediction = 0;
    m_diagnostics.updateTrigger( arrival, m_triggerPrediction );
    if( m_phaseClock.m_valid && !m_measureReference )
    {
        m_measureReference  = true;
        m_latencyArrival    = arrival;
        m_latencyPrediction = m_triggerPrediction;
    }
    m_diagnostics.m_cadenceRatio = m_phaseClock.m_sourcePeriod * m_activeFps.load() / 1e9;
    m_diagnostics.closeLateWindow();
    updateSyncDiagnostics();
}

inline void accelSim::updateSyncDiagnostics()
{
    auto &t = m_diagnostics.m_timing;
    if( !m_phaseClock.m_valid )
    {
        t[lock]                    = 0;
        m_diagnostics.m_lockPhase  = false;
        m_diagnostics.m_lockTiming = false;
        return;
    }
    const double residual = ( m_phaseClock.m_applied - m_phaseClock.m_target ) / 1000;
    t[delayApplied]       = std::fmod( t[delayTarget] + residual, t[semaphorePeriod] );
    if( t[delayApplied] < 0 )
        t[delayApplied] += t[semaphorePeriod];
    t[delay]       = t[delayApplied];
    t[delayBudget] = m_phaseClock.m_budget / 1000;
    t[capped]      = std::abs( residual ) > t[delayBudget] ? 1 : 0;
    m_diagnostics.updatePhaseError();
    m_diagnostics.updateLock( m_lockAbsolute / 1000, m_lockFraction );
}

inline void accelSim::invalidateSync()
{
    m_phaseClock.invalidate();
    m_diagnostics.invalidateSync( m_postDelay.load() );
    m_triggerPrediction = 0;
    m_measureReference  = false;
    snapshotDiagnostics( true );
}

inline int accelSim::waitSync()
{
    const int64_t timeout = std::max<int64_t>( 1, std::llround( m_syncTimeout * 1e9 ) );
    while( !shutdown() && !m_reconfig )
    {
        if( m_requestedFps.load() != m_activeFps.load() )
            return 1;
        int64_t now = monotonicNow();
        if( now >= m_nextSyncProbe )
        {
            m_nextSyncProbe = now + timeout;
            if( m_syncOpen && syncStale() )
            {
                closeSync();
                m_phaseClock.invalidate( true );
                invalidateSync();
            }
            if( !m_syncOpen && openSync() == 0 )
            {
                m_phaseClock.invalidate( true );
                ImageStreamIO_semflush( &m_syncImage, m_syncIndex );
            }
        }
        if( m_phaseClock.expired( now, timeout ) )
            invalidateSync();
        now                     = monotonicNow();
        const int64_t remaining = m_phaseClock.deadline() - now;
        if( remaining <= 0 )
        {
            // Service at most one queued reference even when every acquisition is late;
            // an overloaded producer must not starve source-counter/phase observations.
            if( m_syncOpen && sem_trywait( m_syncImage.semptr[m_syncIndex] ) == 0 )
            {
                ++m_diagnostics.m_timing[received];
                syncTiming( m_syncImage.md[0].cnt0, monotonicNow() );
            }
            return 0;
        }
        if( !m_syncOpen )
            return waitUntil( m_phaseClock.deadline() );
        // Wait for either a WFS reference or the next accelerometer deadline. Receiving a
        // reference never changes that frozen deadline and never publishes a frame itself.
        timespec until;
        if( clock_gettime( CLOCK_REALTIME, &until ) < 0 )
            return -1;
        const int64_t ns = until.tv_nsec + std::min<int64_t>( remaining, 1000000 );
        until.tv_sec += ns / 1000000000;
        until.tv_nsec = ns % 1000000000;
        if( sem_timedwait( m_syncImage.semptr[m_syncIndex], &until ) == 0 )
        {
            ++m_diagnostics.m_timing[received];
            syncTiming( m_syncImage.md[0].cnt0, monotonicNow() );
        }
        else if( errno != ETIMEDOUT && errno != EINTR )
        {
            closeSync();
            m_phaseClock.invalidate( true );
            invalidateSync();
        }
    }
    return 1;
}

inline int accelSim::acquireAndCheckValid()
{
    m_fresh = false;
    if( shutdown() || m_reconfig )
        return 1;
    try
    {
        if( m_rateRequest.load() != m_rateApplied.load() )
        {
            double   requested;
            uint64_t revision;
            { //mutex scope
                std::lock_guard<std::mutex> guard( m_controlMutex );
                requested = m_requestedFps.load();
                revision  = m_rateRequest.load();
            }
            if( requested != m_activeFps.load() )
            {
                m_model.setRate( requested );
                m_activeFps = requested;
                ++m_diagnostics.m_discontinuities;
                m_schedule.reset( monotonicNow(), requested );
                m_timeline.reset( m_schedule.m_epoch, m_model );
                m_lastAcquisition = m_triggerPrediction = 0;
                if( !m_syncName.empty() )
                {
                    m_phaseClock.reset( m_schedule.m_epoch, requested );
                    invalidateSync();
                }
            }
            m_rateApplied = revision; // A newer callback remains Busy until its own boundary.
        }
        uint64_t skipped = 0;
        int64_t  deadline;
        if( m_syncName.empty() )
        {
            deadline         = m_schedule.deadline();
            const int result = waitUntil( deadline );
            if( result != 0 )
                return result;
            const int64_t now = monotonicNow();
            skipped           = m_schedule.consume( now );
            // Lateness is measured against the original deadline, including skipped periods.
        }
        else
        {
            const int result = waitSync();
            if( result != 0 )
                return result;
            deadline = m_phaseClock.deadline();
        }
        if( shutdown() || m_reconfig || ( !m_syncName.empty() && m_requestedFps.load() != m_activeFps.load() ) )
        {
            if( !m_syncName.empty() )
                invalidateSync();
            return 1;
        }
        const int64_t acquired = monotonicNow();
        if( clock_gettime( CLOCK_REALTIME, &m_currImageTimestamp ) < 0 )
        {
            if( !m_syncName.empty() )
                invalidateSync();
            return -1;
        }
        const double late = std::max( 0.0, ( acquired - deadline ) / 1000.0 );
        if( late > 0 )
            ++m_diagnostics.m_late;
        m_diagnostics.m_maxLateUs = std::max( m_diagnostics.m_maxLateUs, late );
        if( !m_syncName.empty() )
            skipped = m_phaseClock.consume(
                acquired, m_maxSlew, m_phaseClock.m_period - m_diagnostics.m_timing[service] * 1000 - m_guard );
        m_diagnostics.m_skipped += skipped;
        measure( acquired, ( m_syncName.empty() ? m_schedule.m_next : m_phaseClock.m_next ) - 1 );
        const int64_t completed = monotonicNow();
        auto         &t         = m_diagnostics.m_timing;
        t[channelReadout]       = ( completed - acquired ) / 1000.0;
        if( m_syncName.empty() )
        {
            t[triggerInterval] = m_lastAcquisition ? ( acquired - m_lastAcquisition ) / 1000.0 : 0;
            t[delayTarget]     = m_postDelay.load();
        }
        else
        {
            const double alpha = m_alpha.load();
            t[service] = m_lastAcquisition ? alpha * t[channelReadout] + ( 1 - alpha ) * t[service] : t[channelReadout];
            if( m_phaseClock.m_valid && m_measureReference && acquired >= m_latencyPrediction )
            {
                const double latency = ( acquired - m_latencyArrival ) / 1000.0;
                t[readLatency]       = t[readLatency] > 0 ? alpha * latency + ( 1 - alpha ) * t[readLatency] : latency;
                t[latencyError]      = ( acquired - m_latencyPrediction ) / 1000.0;
                m_measureReference   = false;
            }
            // Lateness and phase qualify lock together over whole WFS periods, whichever event updates it last.
            m_diagnostics.acquisitionLate( late );
            updateSyncDiagnostics();
        }
        m_lastAcquisition = acquired;
        m_fresh           = true;
        return 0;
    }
    catch( const std::exception &error )
    {
        m_shutdown = true;
        if( !m_syncName.empty() )
            invalidateSync();
        return log<software_error, -1>( { __FILE__, __LINE__, error.what() } );
    }
}

inline int accelSim::loadImageIntoStream( void *destination )
{
    if( !m_fresh || destination == nullptr )
    {
        return -1;
    }

    const auto counts = accelSimDetail::adcOutput( m_acquiredSample, m_parameters, m_zeroOutput );
    std::memcpy( destination, counts.data(), 2 * sizeof( uint16_t ) );
    m_fresh = false;
    ++m_diagnostics.m_timing[frameSequence];
    if( !m_syncName.empty() )
    {
        ++m_diagnostics.m_timing[written];
    }

    return 0;
}

inline void accelSim::measure( int64_t acquired, uint64_t tick )
{
    if( m_timeline.advance( m_model, acquired ) )
        ++m_diagnostics.m_discontinuities;
    // Clamps count published measurements only, not generated-ahead nodes.
    if( !m_timeline.measurement(
            acquired, tick, m_parameters, m_acquiredSample, m_diagnostics.m_low, m_diagnostics.m_high ) )
        throw std::runtime_error( "acquisition time outside the simulation timeline" );
}

inline void accelSim::snapshotDiagnostics( bool force )
{
    { //mutex scope
        std::unique_lock<std::mutex> guard( m_snapshotMutex, std::defer_lock );
        if( force )
            guard.lock();
        else if( !guard.try_lock() )
            return;
        m_snapshot = m_diagnostics;
    }
}

inline int accelSim::frameGrabberPostPublish( IMAGE *stream )
{
    static_cast<void>( stream );
    snapshotDiagnostics( false );
    return 0;
}

inline int accelSim::reconfig()
{
    m_fresh = false;
    closeSync();
    m_phaseClock.invalidate( true );
    m_model.restart();
    ++m_diagnostics.m_discontinuities;
    if( !m_syncName.empty() )
        invalidateSync();
    return 0;
}

inline int accelSim::checkRecordTimes()
{
    return telemeterT::checkRecordTimes( telem_fgtimings() );
}

inline int accelSim::recordTelem( const telem_fgtimings *tag )
{
    static_cast<void>( tag );
    return recordFGTimings( true );
}

inline int accelSim::propertyState( pcf::IndiProperty &property, bool valid )
{
    property.setState( valid ? pcf::IndiProperty::Ok : pcf::IndiProperty::Alert );
    if( m_indiDriver )
        m_indiDriver->sendSetProperty( property );
    return valid ? 0 : -1;
}

inline double accelSim::target( const pcf::IndiProperty &property )
{
    try
    {
        if( property.find( "target" ) )
            return std::stod( property["target"].getValue() );
        if( property.find( "current" ) )
            return std::stod( property["current"].getValue() );
    }
    catch( const std::exception & )
    {
    }
    return std::numeric_limits<double>::quiet_NaN();
}

inline int accelSim::requestRate( double value )
{
    const bool valid = std::isfinite( value ) && value > 2 * m_maxFrequency && value <= 10000;
    if( !valid )
        return propertyState( m_indiP_fps, false );
    m_requestedFps = value;
    ++m_rateRequest;
    updateIfChanged( m_indiP_fps, "target", value, INDI_BUSY );
    m_indiP_fps["target"] = value;
    m_indiP_fps.setState( INDI_BUSY );
    return 0;
}

inline int accelSim::requestChannels( double value )
{
    return propertyState( m_indiP_numChannels, value == 2 );
}

inline INDI_NEWCALLBACK_DEFN( accelSim, m_indiP_dmEnable )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_dmEnable, ipRecv );
    std::lock_guard<std::mutex> guard( m_controlMutex );
    if( !ipRecv.find( "toggle" ) )
        return propertyState( m_indiP_dmEnable, false );
    const bool enable = ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On;
    if( enable && !m_dmReady )
        return propertyState( m_indiP_dmEnable, false );
    { //mutex scope
        std::lock_guard<std::mutex> dmGuard( m_dmMutex ); // Lock order: control, then DM.
        m_dmEnabled = enable;
        if( !enable && m_dmReady )
            m_dmZeroPending = true;
        for( auto &entry : m_dmStage )
            entry.m_valid = false; // Staged surfaces are regenerated after the next enable.
    }
    m_indiP_dmEnable["toggle"] = enable ? pcf::IndiElement::On : pcf::IndiElement::Off;
    m_indiP_dmEnable.setState( pcf::IndiProperty::Busy );
    if( m_indiDriver )
        m_indiDriver->sendSetProperty( m_indiP_dmEnable );
    return 0;
}

inline INDI_NEWCALLBACK_DEFN( accelSim, m_indiP_dmScale )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_dmScale, ipRecv );
    std::lock_guard<std::mutex> guard( m_controlMutex );
    const double                value = target( ipRecv );
    if( !std::isfinite( value ) || value < 0 || value > accelSimDetail::maxDMScale )
        return propertyState( m_indiP_dmScale, false );
    m_dmScale                  = value;
    m_indiP_dmScale["current"] = value;
    m_indiP_dmScale["target"]  = value;
    return propertyState( m_indiP_dmScale, true );
}

inline INDI_NEWCALLBACK_DEFN( accelSim, m_indiP_fps )( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getName() != "fps" )
        return -1;
    std::lock_guard<std::mutex> guard( m_controlMutex );
    return requestRate( target( ipRecv ) );
}

inline INDI_NEWCALLBACK_DEFN( accelSim, m_indiP_numChannels )( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getName() != "numChannels" )
        return -1;
    std::lock_guard<std::mutex> guard( m_controlMutex );
    return requestChannels( target( ipRecv ) );
}

inline INDI_NEWCALLBACK_DEFN( accelSim, m_indiP_alpha )( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getName() != "alpha" )
        return -1;
    std::lock_guard<std::mutex> guard( m_controlMutex );
    const double                value = target( ipRecv );
    if( !std::isfinite( value ) )
        return propertyState( m_indiP_alpha, false );
    m_alpha                  = std::clamp( value, 0.0, 1.0 );
    m_indiP_alpha["current"] = m_alpha.load();
    m_indiP_alpha["target"]  = m_alpha.load();
    return propertyState( m_indiP_alpha, true );
}

inline INDI_NEWCALLBACK_DEFN( accelSim, m_indiP_synchroDelay )( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getName() != "synchroDelay" )
        return -1;
    std::lock_guard<std::mutex> guard( m_controlMutex );
    const double                value = target( ipRecv );
    if( !std::isfinite( value ) || value < -1000000 || value > 1000000 )
        return propertyState( m_indiP_synchroDelay, false );
    m_postDelay                     = static_cast<int>( value );
    m_indiP_synchroDelay["current"] = m_postDelay.load();
    m_indiP_synchroDelay["target"]  = m_postDelay.load();
    return propertyState( m_indiP_synchroDelay, true );
}

inline INDI_SETCALLBACK_DEFN( accelSim, m_indiP_fpsSource )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_fpsSource, ipRecv );
    std::lock_guard<std::mutex> guard( m_controlMutex );
    if( !ipRecv.find( m_fpsElement ) )
        return propertyState( m_indiP_fps, false );
    try
    {
        return requestRate( ipRecv[m_fpsElement].get<double>() );
    }
    catch( const std::exception & )
    {
        return propertyState( m_indiP_fps, false );
    }
}

inline INDI_SETCALLBACK_DEFN( accelSim, m_indiP_numChannelsSource )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_numChannelsSource, ipRecv );
    std::lock_guard<std::mutex> guard( m_controlMutex );
    if( !ipRecv.find( m_channelsElement ) )
        return propertyState( m_indiP_numChannels, false );
    try
    {
        return requestChannels( ipRecv[m_channelsElement].get<double>() );
    }
    catch( const std::exception & )
    {
        return propertyState( m_indiP_numChannels, false );
    }
}

} // namespace app
} // namespace MagAOX
#endif
