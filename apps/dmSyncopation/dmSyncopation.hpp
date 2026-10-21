/** \file dmSyncopation.hpp
 * \brief Main-loop synchronized, persistent auxiliary DM command scheduler.
 */
#ifndef dmSyncopation_hpp
#define dmSyncopation_hpp

#include <atomic>
#include <cstring>
#include <memory>
#include <mutex>
#include <sstream>
#include <sys/random.h>

#include <mx/improc/milkImage.hpp>
#include <mx/math/floatUtils.hpp>

#include "../../libMagAOX/libMagAOX.hpp"
#include "../../magaox_git_version.h"
#include "syncoSchedule.hpp"

namespace MagAOX
{
namespace app
{

/// Access metadata while retaining milkImage's existing passive/normal publication implementation.
class syncoImage : public mx::improc::milkImage<float>
{
  public:
    /// Get the attached ImageStreamIO handle; ownership stays with milkImage.
    IMAGE *image();
};

inline IMAGE *syncoImage::image()
{
    return m_image;
}

/** \defgroup dmSyncopation DM Syncopation
 * \brief Adds evenly spaced DM combinations between genuine main-loop publications.
 * \ingroup apps
 */

/// One primary-monitor callback owns all timed copies and auxiliary counter publications.
/** \ingroup dmSyncopation */
class dmSyncopation : public MagAOXApp<true>,
                      public dev::shmimMonitor<dmSyncopation>,
                      public dev::telemeter<dmSyncopation>
{
    friend class dev::shmimMonitor<dmSyncopation>;
    friend class dev::telemeter<dmSyncopation>;
    friend class dmSyncopation_test;
    using shmimMonitorT = dev::shmimMonitor<dmSyncopation>;
    using telemeterT    = dev::telemeter<dmSyncopation>;

  protected:
    /// Fixed fault classes counted without formatting or logging in the timing callback.
    enum class Fault
    {
        none,
        primary,
        resource,
        output,
        trigger,
        count
    };

    /// Persistent stream handles and one-pass source staging, owned by the scheduler.
    struct Mapping
    {
        std::unique_ptr<syncoImage> m_destination;           ///< Existing numbered main channel.
        std::unique_ptr<syncoImage> m_source;                ///< Persistent matching _aux image.
        std::vector<float>          m_cache;                 ///< Last complete source snapshot.
        std::vector<float>          m_stage;                 ///< Frozen snapshot for one copy.
        uint64_t                    m_counter{ 0 };          ///< Last destination counter owned/accepted by this app.
        ino_t                       m_destinationInode{ 0 }; ///< File identity at attachment.
        ino_t                       m_sourceInode{ 0 };      ///< File identity at attachment.
    };

    /// Coherent camera measurement cached by the cycle without waiting for the INDI input thread.
    struct CameraReport
    {
        double  m_fps{ 0 };       ///< Measured acquisition frequency, Hz.
        int64_t m_received{ 0 };  ///< Original monotonic receipt time, ns.
        double  m_jitterUs{ -1 }; ///< Acquisition interval standard deviation, us.
        double  m_minUs{ -1 };    ///< Minimum reported acquisition interval, us.
        double  m_maxUs{ -1 };    ///< Maximum reported acquisition interval, us.
    };

    /** \name Configuration - Data
     * @{ */
    std::vector<std::string> m_destinations{ "dm01disp07", "dm01disp08" }; ///< Ordered destination mapping.
    std::string              m_combinedName{ "dm01disp" };                 ///< dmcomb output, observation only.
    std::string              m_timingName; ///< Integer timing record; defaults to instance name plus _timing.
    std::string              m_cameraDevice{ "camwfs" };      ///< Device publishing measured acquisition timing.
    std::string              m_cameraProperty{ "fg_timing" }; ///< Measured camera timing property.
    std::string              m_cameraElement{ "acq_fps" };    ///< Measured acquisition-rate element.
    int                      m_requested{ 3 };                ///< Fixed requested count, including passive step 0.
    int                      m_primarySemaphore{ 5 }; ///< Required reader semaphore; reject allocation fallback.
    unsigned                 m_lockFrames{ 32 };  ///< Clean main observations before preparing the first active cycle.
    unsigned                 m_failureLimit{ 3 }; ///< Consecutive original event failures before reacquisition.
    unsigned      m_rateFrames{ 32 }; ///< Cadence violation persistence and primary-rate qualification interval count.
    double        m_nominalRate{ 2000 };   ///< Acquisition fallback, Hz.
    double        m_maxCadence{ 6000 };    ///< Nominal sanity rating, Hz, including step 0.
    double        m_rateTolerance{ 0.01 }; ///< Relative cadence-rating tolerance.
    double        m_minRate{ 500 };        ///< Minimum accepted main rate, Hz.
    double        m_maxRate{ 4000 };       ///< Maximum accepted main rate, Hz.
    double        m_gain{ 0.1 };           ///< Per-event wait integrator gain.
    double        m_maxAdvanceUs{ 20 };    ///< Absolute wait correction limit, us.
    double        m_outputTimeoutUs{ 50 }; ///< Combination watchdog, separate from delivery allowance.
    synco::Bounds m_bounds;                ///< Explicit copy, observation and reader bounds, ns.
    ///@}

    std::mutex           m_cycleMutex;       ///< Serializes lifecycle/configuration with a complete callback.
    std::atomic<bool>    m_enabled{ false }; ///< Cancellation request, checked at bounded waits and publications.
    std::mutex           m_cameraMutex; ///< Protects only the pending camera report, never a timed cycle or INDI I/O.
    std::atomic<double>  m_cameraFps{ 0 };       ///< Latest validated pending acquisition frequency, Hz.
    std::atomic<int64_t> m_cameraTime{ 0 };      ///< Original receipt time of the pending camera report.
    double               m_cameraJitterUs{ -1 }; ///< Acquisition interval standard deviation, us; -1 when omitted.
    double               m_cameraMinUs{ -1 };    ///< Minimum reported acquisition interval, us; -1 when omitted.
    double               m_cameraMaxUs{ -1 };    ///< Maximum reported acquisition interval, us; -1 when omitted.
    CameraReport         m_cameraReport;    ///< Cycle-owned snapshot; retained if the mailbox is briefly unavailable.
    std::vector<Mapping> m_mappings;        ///< Open mappings, fixed while attached.
    std::unique_ptr<syncoImage> m_combined; ///< Read-only combined-output attachment.
    ino_t                       m_combinedInode{ 0 }; ///< Combined-output file identity.
    IMAGE                       m_timing{};           ///< Owned timing publisher handle, never a DM input.
    ino_t                       m_timingInode{ 0 };   ///< File identity of the published provider schedule.
    synco::Record               m_record{};           ///< Current coherent provider schedule.
    bool                        m_resources{ false }; ///< Mappings and timing stream are attached and validated.
    int64_t                     m_nextAttach{ 0 }; ///< Bounded retry interval while unarmed resources are unavailable.
    bool        m_trackingOutput{ false };         ///< A baseline is available to associate the next main combination.
    bool        m_expectOutput{ false };           ///< Exactly one pending combination, or the initial baseline drain.
    bool        m_primed{ false };                 ///< The upcoming cycle has an on-time passive preparation.
    int         m_scheduleState{ synco::invalid }; ///< Provider-facing acquisition/armed state.
    const char *m_reason{ "disabled" };            ///< Static health reason, formatted only by appLogic.
    const char *m_lastFaultReason{ "none" };       ///< Latched fault cause, retained while the current phase changes.
    int64_t     m_lastFaultNs{ 0 };   ///< Monotonic timestamp of the latched synchronization/resource fault.
    int64_t     m_faultInterval{ 0 }; ///< Latched interval at the latest interval fault; zero for other faults.
    int64_t     m_faultPeriod{ 0 };   ///< Prediction used at the latest interval fault; zero for other faults.
    std::array<uint64_t, size_t( Fault::count )> m_faultCounts{}; ///< Fixed per-cause fault totals.
    bool     m_havePrimaryCounter{ false }; ///< Last-seen counter is initialized for the current primary attachment.
    uint64_t m_primaryCounter{ 0 };         ///< Last seen primary counter, retained through reacquisition.
    int64_t  m_lastPrimaryDelta{ 0 };       ///< Counter change at the latest distinct primary observation.
    int64_t  m_lastPrimaryInterval{ 0 };    ///< Observed interval at the latest distinct primary event.
    int64_t  m_lastOutputDelta{ 0 }; ///< Observed combined counter minus the last confirmed counter at an output fault.
    uint64_t m_duplicatePosts{ 0 };  ///< Unchanged-counter callbacks ignored while draining queued notifications.
    uint64_t m_consumed{ 0 };        ///< Last confirmed combined counter.
    int64_t  m_lastMain{ 0 };        ///< Previous main callback timestamp, monotonic ns.
    int64_t  m_previousPrimaryTime{ 0 };     ///< Last distinct observation, retained across phase reacquisition.
    int64_t  m_rateWindowStart{ 0 };         ///< First timestamp of the candidate primary-rate window.
    int64_t  m_rateWindowMin{ 0 };           ///< Shortest sequential interval in the candidate window.
    int64_t  m_rateWindowMax{ 0 };           ///< Longest sequential interval in the candidate window.
    unsigned m_rateWindowCount{ 0 };         ///< Number of coherent intervals accumulated for rate qualification.
    int64_t  m_measuredPeriod{ 0 };          ///< Most recent qualified primary-window mean period, ns.
    int64_t  m_period{ 500000 };             ///< Selected camera, qualified primary or configured period, ns.
    const char *m_periodSource{ "nominal" }; ///< Current period reference, independent of primary phase.
    int64_t     m_submission{ 0 };           ///< Actual last submission, or conservative main-publication upper bound.
    int64_t     m_epoch{ 0 };                ///< Restart-unique timing record identity.
    int64_t     m_generation{ 1 };           ///< Changes on invalidation/reacquisition.
    unsigned    m_clean{ 0 };                ///< Consecutive clean primary observations.
    unsigned    m_failures{ 0 };       ///< Original consecutive timing failures; dependent cancellations do not count.
    unsigned    m_rateViolations{ 0 }; ///< Consecutive nominal rate incompatibilities.
    int         m_feasible{ 0 };       ///< Reported maximum feasible count, never an automatic count selector.
    uint64_t    m_delivered{ 0 };      ///< Confirmed later-step combinations.
    uint64_t    m_mainDelivered{ 0 };  ///< Confirmed primary combinations, separate from extra updates.
    uint64_t    m_reportedDelivered{ 0 }; ///< Receipt count at the previous slow status update.
    int64_t     m_reportTime{ 0 };        ///< Monotonic reference for measured delivered cadence.
    double      m_deliveredHz{ 0 };       ///< Main plus auxiliary confirmed cadence over the last status interval.
    uint64_t    m_missed{ 0 };            ///< Original timing failures.
    uint64_t    m_faults{ 0 };            ///< Synchronization/resource invalidations.
    uint64_t    m_cached{ 0 };            ///< Source fallback uses, not deadline misses.
    uint64_t    m_notStarted{ 0 };        ///< Step-0 copies suppressed before mutation.
    uint64_t    m_copiedLate{ 0 };        ///< Step-0 copies completed after the effective cutoff.
    int64_t     m_snapshotMax{ 0 };       ///< Observed maximum complete snapshot batch time.
    int64_t     m_copyMax{ 0 };           ///< Observed maximum complete destination copy time.
    std::array<int64_t, synco::maxSteps> m_advances{};  ///< Independent bounded wait integrators.
    std::array<int64_t, synco::maxSteps> m_residuals{}; ///< Latest actual-minus-target timings, ns.
    int64_t ( *m_now )(){ synco::monotonicNs };         ///< Local clock adapter for deterministic tests.
    int ( *m_wait )( const timespec * ){ nullptr };     ///< Optional absolute-wait adapter used by deterministic tests.
    bool    m_copyStarted{ false }; ///< Per-call outcome, independent of historical maximum copy duration.
    int64_t m_copyAttemptedAt{ 0 }; ///< Actual copy-attempt start, including safely rejected attempts.
    int64_t m_copyCompleted{ 0 };   ///< Per-call completion timestamp for passive-copy classification.
    int64_t m_step0Wake{ 0 };       ///< Latest passive-copy start lateness relative to the requested wake, ns.
    int64_t m_step0WakeMax{ 0 };    ///< Maximum observed passive-copy wake/check lateness, ns.

    pcf::IndiProperty m_indiP_enable;    ///< Enable request and acknowledged quiescence.
    pcf::IndiProperty m_indiP_reacquire; ///< Request fresh synchronization.
    pcf::IndiProperty m_indiP_settings;  ///< Disabled-only essential numeric controls.
    pcf::IndiProperty m_indiP_status;    ///< Compact rates/counts/outcome summary.
    pcf::IndiProperty m_indiP_health;    ///< Schedule reason and stream identities.
    pcf::IndiProperty m_indiP_camera;    ///< External camera FPS subscription.

  public:
    /// Construct without publishing or enabling any DM command.
    dmSyncopation();

    /// Preserve the app's noexcept destruction contract; normal shutdown joins before releasing handles.
    ~dmSyncopation() noexcept;

    /// Register app and helper configuration.
    void setupConfig();

    /// Parse and validate configuration without opening shared memory.
    int loadConfigImpl( mx::app::appConfigurator &config /**< [in] parsed app configuration */ );

    /// Load configuration, requesting shutdown on invalid input.
    void loadConfig();

    /// Register INDI controls and start the one primary monitor.
    int appStartup();

    /// Publish slow status/telemetry and invalidate silent schedules.
    int appLogic();

    /// Cancel, join the monitor and close handles while retaining commands.
    int appShutdown();

    /// Validate the primary reader assignment and attach persistent resources.
    int allocate( const dev::shmimT &tag /**< [in] primary-monitor tag */ );

    /// Execute one main cycle; never writes or posts to the primary input.
    int processImage( void              *source /**< [in] unused primary pixel pointer */,
                      const dev::shmimT &tag /**< [in] primary-monitor tag */ );

    /// Schedule aggregate telemetry outside the timed callback.
    int checkRecordTimes();

    /// Record one aggregate timing/feedback sample.
    int recordTelem( const telem_dmsynco *record /**< [in] telemetry type tag */ );

    /// Accept enable/disable, acknowledging disable only after the callback quiesces.
    INDI_NEWCALLBACK_DECL( dmSyncopation, m_indiP_enable );

    /// Request acquisition without clearing persistent contributions.
    INDI_NEWCALLBACK_DECL( dmSyncopation, m_indiP_reacquire );

    /// Validate an atomic disabled-only timing/count setting update.
    INDI_NEWCALLBACK_DECL( dmSyncopation, m_indiP_settings );

    /// Accept measured camera FPS and optional acquisition statistics as one coherent report.
    INDI_SETCALLBACK_DECL( dmSyncopation, m_indiP_camera );

  protected:
    /// Attach existing destinations/output and create only missing auxiliary/timing streams.
    bool openResources();

    /// Close this app's handles without destroying persistent DM/source images.
    void closeResources();

    /// Check shape/counter ownership, optionally including on-disk stream identity.
    bool resourcesValid( bool files /**< [in] include bounded-cycle-start inode checks */ );

    /// Refresh the coherent camera cache only when its short mailbox lock is immediately available.
    void readCamera();

    /// Select a measured period without learning individual callback outliers as rate changes.
    void updatePeriod( int64_t observed /**< [in] distinct primary observation time */,
                       int64_t interval /**< [in] interval from the prior distinct observation, or zero */,
                       bool    sequential /**< [in] whether primary counters progressed by exactly one */ );

    /// Invalidate scheduling while retaining the last complete contributions.
    void
    invalidate( const char *reason /**< [in] static diagnostic reason */,
                Fault       fault  = Fault::none /**< [in] synchronization/resource fault class, if any */,
                bool resetFeedback = true /**< [in] reset learned waits for a changed synchronization contract */ );

    /// Wait with bounded cancellation and primary-counter checkpoints.
    bool waitUntil( int64_t target /**< [in] absolute monotonic target */ );

    /// Confirm exactly one expected combined-output update before a bounded timeout.
    bool consume();

    /// Freeze all source contributions once, using prior complete cache on inconsistency.
    bool snapshot();

    /// Always copy every staged mapping and passively notify observers.
    bool copyBatch( int64_t cutoff /**< [in] hard completion cutoff */ );

    /// Prepare/trigger one later step or the next cycle's passive step 0.
    bool executeStep( int     step /**< [in] zero for next passive preparation */,
                      int64_t target /**< [in] desired publication/completion time */,
                      int64_t cutoff /**< [in] hard copy/publication limit */ );

    /// Count one original missed event and apply the consecutive-failure policy.
    void missedStep();

    /// Publish future cutoffs and targets or invalidate the provider record.
    void publishSchedule( int64_t observed /**< [in] current main callback reference */,
                          int     completed /**< [in] last frozen step; -1 before the cycle */ );

    /// Resolve a stream inode, returning zero when absent.
    static ino_t inode( const std::string &name /**< [in] ImageStreamIO stream name */ );

    /// Reject nonfinite numeric input even in the repository's fast-math build.
    static bool finite( double value /**< [in] numeric input */ );
};

inline dmSyncopation::dmSyncopation() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    shmimMonitorT::m_smThreadPrio = 30;
    shmimMonitorT::m_smCpuset     = "spare61";
    shmimMonitorT::m_targetState  = stateCodes::READY;
}

inline bool dmSyncopation::finite( double value )
{
    return mx::math::isFinite( value );
}

inline dmSyncopation::~dmSyncopation() noexcept
{
}

inline void dmSyncopation::setupConfig()
{
    SHMIMMONITOR_SETUP_CONFIG( config );
    shmimMonitorT::m_shmimName = "dmC";
    TELEMETER_SETUP_CONFIG( config );
    const std::vector<std::pair<std::string, std::string>> keys = {
        { "streams.destinations", "Ordered numbered DM destinations; sources append _aux; first triggers." },
        { "streams.combined", "Combined output to observe." },
        { "streams.timing", "Same-host signed-int64 schedule stream." },
        { "tracking.cameraDevice", "Camera device publishing measured acquisition timing." },
        { "tracking.cameraProperty", "Measured camera timing property." },
        { "tracking.cameraElement", "Measured camera FPS element." },
        { "scheduler.maxSubframes", "Fixed requested steps, including passive step zero (1-16)." },
        { "scheduler.maxCadenceHz", "Nominal total cadence sanity rating, Hz." },
        { "scheduler.keepoutUs", "Controller processing/last-reader bound, us." },
        { "scheduler.guardUs", "Extra reserve beyond explicit delivery/reader bounds, us." },
        { "scheduler.step0LeadUs", "Nominal step-zero completion lead, us." },
        { "scheduler.deliveryUs", "Qualified maximum command delivery latency, us." },
        { "scheduler.snapshotBudgetUs", "Whole-batch one-pass source snapshot budget, us." },
        { "scheduler.copyBudgetUs", "Whole-batch destination copy budget, us." },
        { "scheduler.visibilityUs", "Positive margin before earliest main publication, us." },
        { "scheduler.preparationReserveUs", "Preparation target reserve before hard cutoff, us." },
        { "scheduler.outputTimeoutUs", "Combined output watchdog, us." },
        { "tracking.nominalMainRateHz", "Acquisition fallback main rate, Hz." },
        { "tracking.minMainRateHz", "Minimum valid main rate, Hz." },
        { "tracking.maxMainRateHz", "Maximum valid main rate, Hz." },
        { "tracking.earlyMainUs", "Earliest next-main allowance, us." },
        { "tracking.observationLagUs", "Qualified maximum publication-to-callback lag, us." },
        { "tracking.lockFrames", "Clean main observations before priming." },
        { "tracking.rateTolerance", "Relative nominal cadence tolerance." },
        { "tracking.rateFrames", "Cadence violation persistence and primary-rate qualification interval count." },
        { "tracking.primarySemaphore", "Required monitor reader index; fallback is rejected." },
        { "recovery.consecutiveMisses", "Consecutive original missed steps before reacquisition." },
        { "feedback.gain", "Independent per-step wait integrator gain (0-1)." },
        { "feedback.maxAdvanceUs", "Maximum absolute wait correction, us." } };
    for( const auto &key : keys )
    {
        const auto dot = key.first.find( '.' );
        config.add( key.first,
                    "",
                    key.first,
                    argType::Required,
                    key.first.substr( 0, dot ),
                    key.first.substr( dot + 1 ),
                    false,
                    "value",
                    key.second );
    }
}

inline int dmSyncopation::loadConfigImpl( mx::app::appConfigurator &config )
{
    SHMIMMONITOR_LOAD_CONFIG( config );
    TELEMETER_LOAD_CONFIG( config );
    shmimMonitorT::m_getExistingFirst = false;
    config( m_destinations, "streams.destinations" );
    config( m_combinedName, "streams.combined" );
    config( m_timingName, "streams.timing" );
    config( m_cameraDevice, "tracking.cameraDevice" );
    config( m_cameraProperty, "tracking.cameraProperty" );
    config( m_cameraElement, "tracking.cameraElement" );
    config( m_requested, "scheduler.maxSubframes" );
    config( m_maxCadence, "scheduler.maxCadenceHz" );
    config( m_nominalRate, "tracking.nominalMainRateHz" );
    config( m_minRate, "tracking.minMainRateHz" );
    config( m_maxRate, "tracking.maxMainRateHz" );
    config( m_lockFrames, "tracking.lockFrames" );
    config( m_rateTolerance, "tracking.rateTolerance" );
    config( m_rateFrames, "tracking.rateFrames" );
    config( m_primarySemaphore, "tracking.primarySemaphore" );
    config( m_failureLimit, "recovery.consecutiveMisses" );
    config( m_gain, "feedback.gain" );
    config( m_maxAdvanceUs, "feedback.maxAdvanceUs" );
    config( m_outputTimeoutUs, "scheduler.outputTimeoutUs" );
    const std::pair<const char *, int64_t synco::Bounds::*> bounds[] = {
        { "scheduler.keepoutUs", &synco::Bounds::keepout },
        { "scheduler.guardUs", &synco::Bounds::guard },
        { "scheduler.deliveryUs", &synco::Bounds::delivery },
        { "scheduler.step0LeadUs", &synco::Bounds::lead },
        { "scheduler.snapshotBudgetUs", &synco::Bounds::snapshot },
        { "scheduler.copyBudgetUs", &synco::Bounds::copy },
        { "scheduler.visibilityUs", &synco::Bounds::visibility },
        { "scheduler.preparationReserveUs", &synco::Bounds::reserve },
        { "tracking.earlyMainUs", &synco::Bounds::early },
        { "tracking.observationLagUs", &synco::Bounds::observation } };
    for( const auto &entry : bounds )
    {
        double value = double( m_bounds.*entry.second ) / 1000;
        config( value, entry.first );
        if( !finite( value ) || value < 0 || value > 1000000 )
            return -1;
        m_bounds.*entry.second = int64_t( std::ceil( value * 1000 ) );
    }
    if( m_timingName.empty() )
        m_timingName = configName() + "_timing";
    if( m_combinedName.size() != 8 || m_combinedName.substr( 0, 2 ) != "dm" || m_combinedName.substr( 4 ) != "disp" ||
        m_combinedName.find_first_not_of( "0123456789", 2 ) != 4 ||
        ( m_timingName.size() >= 8 && m_timingName.substr( 0, 2 ) == "dm" && m_timingName.substr( 4, 4 ) == "disp" ) )
        return -1;
    if( m_requested < 1 || m_requested > synco::maxSteps || m_destinations.empty() || m_destinations.size() > 32 ||
        !finite( m_maxCadence ) || m_maxCadence <= 0 || !finite( m_nominalRate ) || !finite( m_minRate ) ||
        !finite( m_maxRate ) || m_minRate < 1 || m_maxRate > 100000 || m_minRate > m_nominalRate ||
        m_nominalRate > m_maxRate || !finite( m_rateTolerance ) || m_rateTolerance < 0 || m_rateTolerance > 1 ||
        !finite( m_gain ) || m_gain < 0 || m_gain > 1 || !finite( m_maxAdvanceUs ) || m_maxAdvanceUs < 0 ||
        m_maxAdvanceUs > 1000 || !finite( m_outputTimeoutUs ) || m_outputTimeoutUs <= 0 ||
        m_outputTimeoutUs > 1000000 || !m_lockFrames || !m_failureLimit || !m_rateFrames || m_primarySemaphore < 0 ||
        m_primarySemaphore == 9 || m_primarySemaphore >= SEMAPHORE_MAXVAL || m_bounds.copy == 0 ||
        m_bounds.snapshot == 0 || m_bounds.visibility == 0 || m_bounds.reserve == 0 ||
        ( !m_cameraDevice.empty() && ( m_cameraProperty.empty() || m_cameraElement.empty() ) ) )
        return -1;
    std::vector<std::string> names{ shmimMonitorT::m_shmimName, m_combinedName, m_timingName };
    for( const auto &name : m_destinations )
    {
        if( name.size() != 10 || name.substr( 0, 2 ) != "dm" || name.substr( 4, 4 ) != "disp" ||
            name.substr( 0, 8 ) != m_combinedName || name.find_first_not_of( "0123456789", 2 ) != 4 ||
            name.find_first_not_of( "0123456789", 8 ) != std::string::npos )
            return -1;
        names.push_back( name );
        names.push_back( name + "_aux" );
    }
    for( const auto &name : names )
        if( name.empty() || name.find_first_of( "/\\ \t\n" ) != std::string::npos )
            return -1;
    std::sort( names.begin(), names.end() );
    if( std::adjacent_find( names.begin(), names.end() ) != names.end() )
        return -1;
    m_period                         = int64_t( 1e9 / m_nominalRate );
    shmimMonitorT::m_semaphoreNumber = m_primarySemaphore;
    return 0;
}

inline void dmSyncopation::loadConfig()
{
    if( loadConfigImpl( config ) < 0 )
    {
        log<software_critical>( { __FILE__, __LINE__, "Invalid syncopation configuration" } );
        m_shutdown = true;
    }
}

inline ino_t dmSyncopation::inode( const std::string &name )
{
    char filename[PATH_MAX];
    ImageStreamIO_filename( filename, sizeof( filename ), name.c_str() );
    struct stat info{};
    return stat( filename, &info ) == 0 ? info.st_ino : 0;
}

inline bool dmSyncopation::openResources()
{
    closeResources();
    try
    {
        m_combined = std::make_unique<syncoImage>();
        m_combined->open( m_combinedName );
        m_combinedInode = inode( m_combinedName );
        std::vector<ino_t> identities{ inode( shmimMonitorT::m_shmimName ), m_combinedInode };
        const auto         width  = shmimMonitorT::m_width;
        const auto         height = shmimMonitorT::m_height;
        if( width == 0 || height == 0 || uint64_t( width ) * height > 1048576 ||
            shmimMonitorT::m_dataType != _DATATYPE_FLOAT || shmimMonitorT::m_depth > 1 ||
            shmimMonitorT::m_imageStream.md->naxis != 2 || shmimMonitorT::m_imageStream.md->location != -1 ||
            m_combined->image()->md->naxis != 2 || m_combined->image()->md->location != -1 ||
            m_combined->rows() != width || m_combined->cols() != height )
            throw std::runtime_error( "resource attachment failed" );
        for( const auto &name : m_destinations )
        {
            Mapping mapping;
            mapping.m_destination = std::make_unique<syncoImage>();
            mapping.m_source      = std::make_unique<syncoImage>();
            mapping.m_destination->open( name );
            mapping.m_destination->passive( true );
            if( mapping.m_destination->rows() != width || mapping.m_destination->cols() != height ||
                mapping.m_destination->image()->md->naxis != 2 || mapping.m_destination->image()->md->location != -1 )
                throw std::runtime_error( "resource attachment failed" );
            if( inode( name + "_aux" ) )
                mapping.m_source->open( name + "_aux" );
            else
            {
                // Seed once from a complete stable destination; never destroy/recreate an existing source.
                const auto counter = mapping.m_destination->image()->md->cnt0;
                if( mapping.m_destination->image()->md->write )
                    throw std::runtime_error( "resource attachment failed" );
                std::vector<float> seed( width * height );
                std::memcpy( seed.data(), mapping.m_destination->data(), seed.size() * sizeof( float ) );
                if( mapping.m_destination->image()->md->write || counter != mapping.m_destination->image()->md->cnt0 )
                    throw std::runtime_error( "resource attachment failed" );
                IMAGE    created{};
                uint32_t sizes[]{ width, height };
                // milkImage::create uses a three-axis circular buffer; controller auxiliary sources are plain 2D.
                if( ImageStreamIO_createIm_gpu( &created,
                                                ( name + "_aux" ).c_str(),
                                                2,
                                                sizes,
                                                _DATATYPE_FLOAT,
                                                -1,
                                                1,
                                                IMAGE_NB_SEMAPHORE,
                                                0,
                                                0,
                                                0 ) != 0 )
                    throw std::runtime_error( "resource attachment failed" );
                ImageStreamIO_closeIm( &created );
                mapping.m_source->open( name + "_aux" );
                mapping.m_source->setWrite( true );
                std::memcpy( mapping.m_source->data(), seed.data(), seed.size() * sizeof( float ) );
                mapping.m_source->post();
            }
            if( mapping.m_source->rows() != width || mapping.m_source->cols() != height ||
                mapping.m_source->image()->md->naxis != 2 || mapping.m_source->image()->md->location != -1 )
                throw std::runtime_error( "resource attachment failed" );
            mapping.m_counter          = mapping.m_destination->image()->md->cnt0;
            mapping.m_destinationInode = inode( name );
            mapping.m_sourceInode      = inode( name + "_aux" );
            identities.push_back( mapping.m_destinationInode );
            identities.push_back( mapping.m_sourceInode );
            mapping.m_cache.resize( width * height );
            mapping.m_stage.resize( width * height );
            const auto count = mapping.m_source->image()->md->cnt0;
            if( mapping.m_source->image()->md->write )
                throw std::runtime_error( "resource attachment failed" );
            std::memcpy( mapping.m_cache.data(), mapping.m_source->data(), mapping.m_cache.size() * sizeof( float ) );
            if( mapping.m_source->image()->md->write || count != mapping.m_source->image()->md->cnt0 )
                throw std::runtime_error( "resource attachment failed" );
            m_mappings.push_back( std::move( mapping ) );
        }
        if( inode( m_timingName ) )
        {
            if( ImageStreamIO_openIm( &m_timing, m_timingName.c_str() ) != 0 )
                throw std::runtime_error( "resource attachment failed" );
            if( m_timing.md->datatype != _DATATYPE_INT64 || m_timing.md->naxis != 1 || m_timing.md->location != -1 ||
                m_timing.md->size[0] != synco::words )
                throw std::runtime_error( "resource attachment failed" );
        }
        else
        {
            uint32_t size = synco::words;
            if( ImageStreamIO_createIm_gpu(
                    &m_timing, m_timingName.c_str(), 1, &size, _DATATYPE_INT64, -1, 1, IMAGE_NB_SEMAPHORE, 0, 0, 0 ) !=
                0 )
                throw std::runtime_error( "resource attachment failed" );
        }
        identities.push_back( inode( m_timingName ) );
        m_timingInode = inode( m_timingName );
        std::sort( identities.begin(), identities.end() );
        if( identities.front() == 0 || std::adjacent_find( identities.begin(), identities.end() ) != identities.end() )
            throw std::runtime_error( "resource attachment failed" );
        m_consumed       = m_combined->image()->md->cnt0;
        m_trackingOutput = false;
        m_resources      = true;
        publishSchedule( 0, -1 );
        return true;
    }
    catch( ... )
    {
        closeResources();
        return false;
    }
}

inline void dmSyncopation::closeResources()
{
    m_resources = false;
    m_mappings.clear();
    m_combined.reset();
    if( m_timing.md )
        ImageStreamIO_closeIm( &m_timing );
    m_timing = {};
}

inline bool dmSyncopation::resourcesValid( bool files )
{
    if( !m_resources || !m_combined || !m_combined->valid() || m_combined->image()->md->sem == 0 || !m_timing.md ||
        m_timing.md->sem == 0 || m_timing.md->datatype != _DATATYPE_INT64 || m_timing.md->naxis != 1 ||
        m_timing.md->size[0] != synco::words || m_timing.md->location != -1 ||
        shmimMonitorT::m_imageStream.md->sem == 0 || m_combined->image()->md->naxis != 2 ||
        m_combined->image()->md->location != -1 )
        return false;
    if( files && ( inode( m_combinedName ) != m_combinedInode || inode( m_timingName ) != m_timingInode ||
                   inode( shmimMonitorT::m_shmimName ) != shmimMonitorT::m_inode ) )
        return false;
    for( size_t j = 0; j < m_mappings.size(); ++j )
    {
        const auto &mapping = m_mappings[j];
        if( !mapping.m_destination->valid() || !mapping.m_source->valid() ||
            mapping.m_destination->image()->md->naxis != 2 || mapping.m_destination->image()->md->location != -1 ||
            mapping.m_source->image()->md->naxis != 2 || mapping.m_source->image()->md->location != -1 ||
            mapping.m_destination->image()->md->cnt0 != mapping.m_counter ||
            mapping.m_destination->image()->md->write || mapping.m_destination->image()->md->sem == 0 ||
            mapping.m_source->image()->md->sem == 0 )
            return false;
        if( files && ( inode( m_destinations[j] ) != mapping.m_destinationInode ||
                       inode( m_destinations[j] + "_aux" ) != mapping.m_sourceInode ) )
            return false;
    }
    return true;
}

inline void dmSyncopation::readCamera()
{
    std::unique_lock<std::mutex> lock( m_cameraMutex, std::try_to_lock );
    if( lock.owns_lock() )
        m_cameraReport = { m_cameraFps.load(), m_cameraTime.load(), m_cameraJitterUs, m_cameraMinUs, m_cameraMaxUs };
}

inline void dmSyncopation::updatePeriod( int64_t observed, int64_t interval, bool sequential )
{
    if( !sequential || interval < int64_t( 1e9 / m_maxRate ) || interval > int64_t( 1e9 / m_minRate ) )
    {
        m_rateWindowCount = 0;
        if( !sequential )
            m_measuredPeriod = 0;
    }
    else
    {
        const int64_t shortest = std::min( m_rateWindowMin, interval );
        const int64_t longest  = std::max( m_rateWindowMax, interval );
        if( !m_rateWindowCount || longest - shortest > 2 * m_bounds.early )
        {
            m_rateWindowStart = observed - interval;
            m_rateWindowMin   = interval;
            m_rateWindowMax   = interval;
            m_rateWindowCount = 0;
        }
        else
        {
            m_rateWindowMin = shortest;
            m_rateWindowMax = longest;
        }
        if( ++m_rateWindowCount >= m_rateFrames )
        {
            m_measuredPeriod  = ( observed - m_rateWindowStart ) / m_rateWindowCount;
            m_rateWindowCount = 0;
        }
    }
    readCamera();
    const auto cameraTime    = m_cameraReport.m_received;
    const auto cameraFps     = m_cameraReport.m_fps;
    const auto referenceTime = m_now();
    const bool camera = cameraTime > 0 && referenceTime >= cameraTime && referenceTime - cameraTime < 5000000000LL &&
                        cameraFps >= m_minRate && cameraFps <= m_maxRate;
    const char   *source = camera ? "camera acquisition" : ( m_measuredPeriod > 0 ? "primary window" : "nominal" );
    const int64_t period =
        camera ? int64_t( std::llround( 1e9 / cameraFps ) )
               : ( m_measuredPeriod > 0 ? m_measuredPeriod : int64_t( std::llround( 1e9 / m_nominalRate ) ) );
    if( std::strcmp( source, m_periodSource ) != 0 || period < m_period - m_bounds.early ||
        period > m_period + m_bounds.early )
    {
        invalidate( m_enabled ? "period reference changed" : "disabled" );
        m_lastMain       = 0;
        m_trackingOutput = false;
    }
    m_period       = period;
    m_periodSource = source;
}

inline void dmSyncopation::invalidate( const char *reason, Fault fault, bool resetFeedback )
{
    if( fault != Fault::none )
    {
        ++m_faults;
        ++m_faultCounts[size_t( fault )];
        m_lastFaultReason = reason;
        m_lastFaultNs     = m_now();
        m_faultInterval   = 0;
        m_faultPeriod     = 0;
        m_trackingOutput  = false;
        m_lastMain        = 0;
    }
    m_reason        = reason;
    m_scheduleState = synco::invalid;
    m_clean         = 0;
    m_primed        = false;
    m_expectOutput  = false;
    m_failures      = 0;
    if( resetFeedback )
    {
        m_advances.fill( 0 );
        // Start conservatively early; valid wake samples tune this toward actual wait/check latency.
        m_advances[0] = int64_t( m_maxAdvanceUs * 1000 );
    }
    ++m_generation;
    publishSchedule( 0, -1 );
}

inline bool dmSyncopation::waitUntil( int64_t target )
{
    while( m_enabled && !shutdown() )
    {
        if( shmimMonitorT::m_imageStream.md->cnt0 != m_primaryCounter )
        {
            invalidate( "primary changed during cycle", Fault::primary );
            return false;
        }
        if( !resourcesValid( false ) )
        {
            invalidate( "resource changed during cycle", Fault::resource );
            return false;
        }
        if( !m_expectOutput && m_combined->image()->md->cnt0 != m_consumed )
        {
            m_lastOutputDelta = int64_t( m_combined->image()->md->cnt0 ) - int64_t( m_consumed );
            invalidate( "unexpected combination", Fault::output );
            return false;
        }
        const int64_t now = m_now();
        if( now >= target )
            return true;
        const auto wake  = synco::toTimespec( std::min( target, now + 10000 ) );
        const int result = m_wait ? m_wait( &wake ) : clock_nanosleep( CLOCK_MONOTONIC, TIMER_ABSTIME, &wake, nullptr );
        if( result != 0 && result != EINTR )
        {
            invalidate( "monotonic wait failed", Fault::resource );
            return false;
        }
    }
    return false;
}

inline bool dmSyncopation::consume()
{
    m_expectOutput         = true;
    const int64_t deadline = m_now() + int64_t( m_outputTimeoutUs * 1000 );
    while( m_now() <= deadline )
    {
        const auto counter = m_combined->image()->md->cnt0;
        if( counter == m_consumed + 1 && !m_combined->image()->md->write )
        {
            m_consumed     = counter;
            m_expectOutput = false;
            return true;
        }
        if( counter != m_consumed && counter != m_consumed + 1 )
            break;
        if( !waitUntil( std::min( deadline, m_now() + 1000 ) ) )
            return false;
    }
    m_lastOutputDelta = int64_t( m_combined->image()->md->cnt0 ) - int64_t( m_consumed );
    invalidate( "missing/ambiguous combined output", Fault::output );
    return false;
}

inline bool dmSyncopation::snapshot()
{
    const int64_t start = m_now();
    for( auto &mapping : m_mappings )
    {
        auto      *metadata = mapping.m_source->image()->md;
        const auto counter  = metadata->cnt0;
        bool       complete = !metadata->write;
        if( complete )
        {
            std::memcpy( mapping.m_stage.data(), mapping.m_source->data(), mapping.m_stage.size() * sizeof( float ) );
            std::atomic_thread_fence( std::memory_order_acquire );
            complete = !metadata->write && metadata->cnt0 == counter;
        }
        if( complete )
            std::copy( mapping.m_stage.begin(), mapping.m_stage.end(), mapping.m_cache.begin() );
        else
        {
            std::copy( mapping.m_cache.begin(), mapping.m_cache.end(), mapping.m_stage.begin() );
            ++m_cached;
        }
    }
    const int64_t elapsed = m_now() - start;
    m_snapshotMax         = std::max( m_snapshotMax, elapsed );
    if( elapsed > m_bounds.snapshot )
    {
        m_bounds.snapshot = elapsed + 1000;
        invalidate( "snapshot budget exceeded; enlarged for reacquisition", Fault::none, false );
        return false;
    }
    return true;
}

inline bool dmSyncopation::copyBatch( int64_t cutoff )
{
    m_copyStarted       = false;
    m_copyCompleted     = 0;
    const int64_t start = m_now();
    m_copyAttemptedAt   = start;
    if( start + m_bounds.copy > cutoff || !waitUntil( start ) || m_combined->image()->md->cnt0 != m_consumed )
        return false;
    m_copyStarted = true;
    for( auto &mapping : m_mappings )
    {
        // Once a passive batch starts, finish it on an ordinary disable so retained
        // destinations do not become a mixture of old and newly staged mappings.
        if( shmimMonitorT::m_imageStream.md->cnt0 != m_primaryCounter )
        {
            invalidate( "main event during batch copy", Fault::primary );
            return false;
        }
        mapping.m_destination->setWrite( true );
        std::memcpy( mapping.m_destination->data(), mapping.m_stage.data(), mapping.m_stage.size() * sizeof( float ) );
        mapping.m_destination->post(); // passive: stamps metadata and notifies viewers, preserves cnt0
        m_copyCompleted = m_now();
    }
    const int64_t completed = m_now();
    m_copyCompleted         = completed;
    m_copyMax               = std::max( m_copyMax, completed - start );
    if( !waitUntil( completed ) )
        return false;
    if( completed - start > m_bounds.copy )
    {
        m_bounds.copy = completed - start + 1000;
        invalidate( "copy budget exceeded; enlarged for reacquisition", Fault::none, false );
        return false;
    }
    if( completed > cutoff )
        return false;
    return true;
}

inline void dmSyncopation::missedStep()
{
    ++m_missed;
    if( ++m_failures >= m_failureLimit )
        invalidate( "consecutive missed timing events", Fault::none, false );
}

inline bool dmSyncopation::executeStep( int step, int64_t target, int64_t cutoff )
{
    const uint64_t faultsBefore = m_faults;
    const int64_t  advanceLimit = int64_t( m_maxAdvanceUs * 1000 );
    const int64_t  snapshotAt   = target - m_bounds.copy - advanceLimit - m_bounds.snapshot - m_bounds.reserve;
    if( !waitUntil( snapshotAt ) || !snapshot() )
    {
        if( !m_enabled || m_faults != faultsBefore )
            return false;
        if( step == 0 )
            ++m_notStarted;
        missedStep();
        return false;
    }
    publishSchedule( m_lastMain, step );
    const int64_t hold         = m_submission + m_bounds.delivery + m_bounds.keepout + m_bounds.guard;
    const int64_t desiredStart = target - m_bounds.copy;
    const int64_t advance      = step == 0 ? m_advances[0] : advanceLimit;
    // Step zero never trades its positive preparation reserve for a later planned start.
    const int64_t copyAt = std::max( hold, desiredStart - advance );
    if( copyAt + m_bounds.copy > cutoff )
    {
        if( !m_enabled || m_faults != faultsBefore )
            return false;
        if( step == 0 )
            ++m_notStarted;
        missedStep();
        return false;
    }
    if( !waitUntil( copyAt ) )
        return false;
    const bool copied = copyBatch( cutoff );
    if( step == 0 && m_enabled && m_faults == faultsBefore && copyAt == desiredStart - advance )
    {
        m_step0Wake    = std::max<int64_t>( 0, m_copyAttemptedAt - copyAt );
        m_step0WakeMax = std::max( m_step0WakeMax, m_step0Wake );
        // A rejected copy still supplies a valid wait observation, not a successful command sample.
        m_advances[0] = std::clamp<int64_t>(
            synco::feedback( advance, m_copyAttemptedAt - desiredStart, m_gain, advanceLimit, true ), 0, advanceLimit );
    }
    if( !copied )
    {
        if( !m_enabled || m_faults != faultsBefore )
            return false;
        if( step == 0 )
        {
            if( m_copyStarted && m_copyCompleted > cutoff )
                ++m_copiedLate;
            else if( !m_copyStarted )
                ++m_notStarted;
        }
        missedStep();
        return false;
    }
    int64_t actual = m_now();
    if( step > 0 )
    {
        if( !waitUntil( target - m_advances[step] ) )
            return false;
        const int64_t latest = m_lastMain + m_period - m_bounds.early - m_bounds.observation;
        if( m_now() + m_bounds.delivery + m_bounds.keepout + m_bounds.guard > latest || m_now() > cutoff ||
            m_combined->image()->md->cnt0 != m_consumed )
        {
            missedStep();
            return false;
        }
        auto &trigger = m_mappings.front();
        if( !waitUntil( m_now() ) )
            return false;
        const uint64_t expected = trigger.m_counter + 1;
        trigger.m_destination->passive( false );
        trigger.m_destination->post();
        trigger.m_destination->passive( true );
        if( trigger.m_destination->image()->md->cnt0 != expected )
        {
            invalidate( "trigger destination counter changed unexpectedly", Fault::resource );
            return false;
        }
        trigger.m_counter = expected;
        actual            = m_now();
        m_submission      = actual;
        if( actual + m_bounds.delivery + m_bounds.keepout + m_bounds.guard > latest || actual > cutoff )
        {
            invalidate( "actual trigger outside safe window", Fault::trigger );
            return false;
        }
        if( !consume() )
            return false;
        ++m_delivered;
    }
    m_residuals[step] = actual - target;
    if( step > 0 )
        m_advances[step] = synco::feedback( m_advances[step], m_residuals[step], m_gain, advanceLimit, true );
    m_failures = 0;
    return true;
}

inline void dmSyncopation::publishSchedule( int64_t observed, int completed )
{
    if( !m_timing.md || m_timing.md->datatype != _DATATYPE_INT64 || m_timing.md->naxis != 1 ||
        m_timing.md->size[0] != synco::words || m_timing.md->location != -1 )
        return;
    m_record.fill( 0 );
    m_record[synco::version]    = synco::layoutVersion;
    m_record[synco::epoch]      = m_epoch;
    m_record[synco::generation] = m_generation;
    m_record[synco::state]      = observed > 0 ? m_scheduleState : synco::invalid;
    m_record[synco::steps]      = m_requested;
    m_record[synco::period]     = m_period;
    m_record[synco::reference]  = observed;
    m_record[synco::expires]    = observed > 0 ? observed + 2 * m_period : 0;
    m_record[synco::cycle]      = int64_t( m_primaryCounter );
    for( int j = 0; observed > 0 && j < m_requested; ++j )
    {
        if( m_scheduleState == synco::acquiring && j > 0 )
            continue;
        int64_t target = j == 0 ? observed + m_period : observed + j * m_period / m_requested;
        if( completed == 0 || ( j > 0 && j <= completed ) )
            target += m_period;
        const int64_t preparation = j == 0 ? synco::copyCutoff( target, m_bounds ) - m_bounds.reserve : target;
        const int64_t cutoff =
            preparation - m_bounds.copy - int64_t( m_maxAdvanceUs * 1000 ) - m_bounds.snapshot - m_bounds.reserve;
        if( cutoff > m_now() )
        {
            m_record[synco::cutoffs + j] = cutoff;
            m_record[synco::targets + j] = target;
        }
    }
    m_timing.md->write = 1;
    std::memcpy( m_timing.array.raw, m_record.data(), sizeof( m_record ) );
    ImageStreamIO_UpdateIm( &m_timing );
}

inline int dmSyncopation::allocate( const dev::shmimT &tag )
{
    static_cast<void>( tag );
    std::lock_guard<std::mutex> lock( m_cycleMutex );
    if( shmimMonitorT::m_semaphoreNumber != m_primarySemaphore || m_primarySemaphore == 9 )
    {
        const int assigned = shmimMonitorT::m_semaphoreNumber;
        if( assigned >= 0 && shmimMonitorT::m_imageStream.semReadPID[assigned] == getpid() )
            shmimMonitorT::m_imageStream.semReadPID[assigned] = 0;
        return -1;
    }
    invalidate( "acquiring resources" );
    m_lastMain            = 0;
    m_primaryCounter      = shmimMonitorT::m_imageStream.md->cnt0;
    m_havePrimaryCounter  = true;
    m_previousPrimaryTime = 0;
    m_rateWindowCount     = 0;
    m_measuredPeriod      = 0;
    openResources(); // Other streams may appear after the primary; callbacks retry while unarmed.
    return 0;
}

inline int dmSyncopation::processImage( void *source, const dev::shmimT &tag )
{
    static_cast<void>( source );
    static_cast<void>( tag );
    const int64_t               observed = m_now();
    std::lock_guard<std::mutex> lock( m_cycleMutex );
    if( !m_resources )
    {
        if( observed < m_nextAttach )
            return 0;
        m_nextAttach = observed + 1000000000;
        if( !openResources() )
        {
            invalidate( "resources unavailable" );
            return 0;
        }
    }
    if( !resourcesValid( true ) )
    {
        invalidate( "resource replacement/external destination write", Fault::resource );
        closeResources(); // Reattach and refresh caches, including a controller's cleared auxiliary values.
        return 0;
    }
    const auto counter = shmimMonitorT::m_imageStream.md->cnt0;
    if( m_havePrimaryCounter && counter == m_primaryCounter )
    {
        ++m_duplicatePosts;
        return 0; // A queued semaphore token is not a new physical main event.
    }
    const bool sequential = !m_havePrimaryCounter || counter == m_primaryCounter + 1;
    m_lastPrimaryDelta    = m_havePrimaryCounter ? int64_t( counter ) - int64_t( m_primaryCounter ) : 0;
    m_primaryCounter      = counter; // Adopt jumps/resets once; queued duplicates must not re-enter acquisition.
    m_havePrimaryCounter  = true;
    m_lastPrimaryInterval = m_previousPrimaryTime ? observed - m_previousPrimaryTime : 0;
    m_previousPrimaryTime = observed;
    updatePeriod( observed, m_lastPrimaryInterval, sequential );
    if( !m_enabled )
    {
        m_lastMain       = 0;
        m_consumed       = m_combined->image()->md->cnt0;
        m_trackingOutput = false;
        return 0;
    }
    const int64_t previous = m_lastMain;
    if( !sequential )
    {
        invalidate( "primary counter jump/reset", Fault::primary );
        return 0;
    }
    if( previous &&
        ( observed <= previous || observed - previous < int64_t( 1e9 / m_maxRate ) ||
          observed - previous > int64_t( 1e9 / m_minRate ) || observed < previous + m_period - m_bounds.early ||
          observed > previous + m_period + m_bounds.early ) )
    {
        invalidate( "primary interval anomaly", Fault::primary );
        m_faultInterval = observed - previous;
        m_faultPeriod   = m_period;
        return 0;
    }
    m_lastMain   = observed;
    m_submission = observed; // Conservative upper bound on the current main's publication.
    if( !m_trackingOutput )
    {
        // The current main may already be combined, or still pending. Drain its qualified
        // delivery window before choosing a baseline, so it cannot become the next main's acknowledgement.
        m_expectOutput = true;
        if( !waitUntil( observed + int64_t( std::max( m_outputTimeoutUs * 1000, double( m_bounds.delivery ) ) ) ) )
            return 0;
        if( m_combined->image()->md->write )
        {
            invalidate( "initial output baseline incomplete", Fault::output );
            return 0;
        }
        m_consumed       = m_combined->image()->md->cnt0;
        m_expectOutput   = false;
        m_trackingOutput = true;
        m_primed         = false;
        return 0;
    }
    if( !consume() )
    {
        m_trackingOutput = false;
        return 0;
    }
    ++m_mainDelivered;
    ++m_clean;
    m_feasible           = synco::maximumSteps( m_period, m_bounds );
    const double nominal = 1e9 / m_period;
    if( nominal * m_requested > m_maxCadence * ( 1 + m_rateTolerance ) )
        ++m_rateViolations;
    else
        m_rateViolations = 0;
    if( !synco::feasible( m_period, m_requested, m_bounds ) || m_rateViolations >= m_rateFrames )
    {
        invalidate( "requested count/cadence infeasible" );
        return 0;
    }
    if( m_clean < m_lockFrames )
    {
        m_reason        = "acquiring main progression";
        m_scheduleState = synco::acquiring;
        publishSchedule( observed, -1 );
        return 0;
    }
    const bool active = m_primed;
    m_primed          = false;
    m_scheduleState   = active ? synco::armed : synco::acquiring;
    m_reason          = active ? "armed" : "preparing first step zero";
    publishSchedule( observed, -1 );
    if( active )
    {
        for( int j = 1; j < m_requested; ++j )
        {
            const int64_t target = observed + j * m_period / m_requested;
            const int64_t latest = observed + m_period - m_bounds.early - m_bounds.observation - m_bounds.delivery -
                                   m_bounds.keepout - m_bounds.guard;
            executeStep( j, target, latest );
            if( m_scheduleState == synco::invalid || !m_enabled )
                return 0;
        }
    }
    const int64_t cutoff = synco::copyCutoff( observed + m_period, m_bounds );
    m_primed             = executeStep( 0, cutoff - m_bounds.reserve, cutoff );
    return 0;
}

inline int dmSyncopation::appStartup()
{
    if( getrandom( &m_epoch, sizeof( m_epoch ), 0 ) != sizeof( m_epoch ) )
        return -1;
    m_epoch &= INT64_MAX;
    if( m_epoch == 0 )
        m_epoch = 1;
    if( createStandardIndiToggleSw( m_indiP_enable, "enable" ) < 0 ||
        registerIndiPropertyNew( m_indiP_enable, INDI_NEWCALLBACK( m_indiP_enable ) ) < 0 ||
        createStandardIndiRequestSw( m_indiP_reacquire, "reacquire" ) < 0 ||
        registerIndiPropertyNew( m_indiP_reacquire, INDI_NEWCALLBACK( m_indiP_reacquire ) ) < 0 )
        return -1;
    REG_INDI_NEWPROP( m_indiP_settings, "settings", pcf::IndiProperty::Number );
    for( const auto &name : { "steps", "maxCadenceHz", "keepoutUs", "guardUs", "step0LeadUs" } )
    {
        m_indiP_settings.add( pcf::IndiElement( name ) );
        m_indiP_settings[name].setMin( std::string( name ) == "steps" ? 1 : 0 );
        m_indiP_settings[name].setMax( std::string( name ) == "steps" ? synco::maxSteps : 1000000 );
        m_indiP_settings[name].setStep( std::string( name ) == "steps" ? 1 : 0 );
        m_indiP_settings[name].setFormat( "%g" );
    }
    REG_INDI_NEWPROP_NOCB( m_indiP_status, "timing", pcf::IndiProperty::Number );
    m_indiP_status.setPerm( pcf::IndiProperty::ReadOnly );
    for( const auto &name : { "mainHz",           "cameraHz",
                              "plannedHz",        "deliveredHz",
                              "delivered",        "requested",
                              "active",           "feasible",
                              "missed",           "faults",
                              "cached",           "notStarted",
                              "copiedLate",       "snapshotMaxUs",
                              "copyMaxUs",        "primaryFaults",
                              "resourceFaults",   "outputFaults",
                              "triggerFaults",    "duplicatePosts",
                              "step0WakeUs",      "step0WakeMaxUs",
                              "step0AdvanceUs",   "copyBudgetUs",
                              "snapshotBudgetUs", "lastPrimaryDelta",
                              "lastIntervalUs",   "lastOutputDelta",
                              "lastFaultAgeSec",  "cameraJitterUs",
                              "cameraMinUs",      "cameraMaxUs",
                              "cameraAgeSec",     "primaryMeasuredHz",
                              "faultIntervalUs",  "faultPeriodUs",
                              "faultResidualUs" } )
        m_indiP_status.add( pcf::IndiElement( name ) );
    REG_INDI_NEWPROP_NOCB( m_indiP_health, "health", pcf::IndiProperty::Text );
    m_indiP_health.setPerm( pcf::IndiProperty::ReadOnly );
    for( const auto &name :
         { "reason", "lastFault", "periodSource", "primary", "combined", "timingStream", "mapping" } )
        m_indiP_health.add( pcf::IndiElement( name ) );
    if( !m_cameraDevice.empty() )
    {
        REG_INDI_SETPROP( m_indiP_camera, m_cameraDevice, m_cameraProperty );
    }
    state( stateCodes::READY );
    TELEMETER_APP_STARTUP;
    SHMIMMONITOR_APP_STARTUP;
    return 0;
}

inline int dmSyncopation::appLogic()
{
    SHMIMMONITOR_APP_LOGIC;
    { // mutex scope
        std::unique_lock<std::mutex> lock( m_cycleMutex, std::try_to_lock );
        if( lock.owns_lock() )
        {
            readCamera();
            if( m_enabled && m_lastMain && m_now() - m_lastMain > int64_t( 2e9 / m_minRate ) )
            {
                invalidate( "main timeout", Fault::primary );
            }
            const int64_t  now       = m_now();
            const uint64_t delivered = m_mainDelivered + m_delivered;
            if( m_reportTime && now > m_reportTime )
                m_deliveredHz = 1e9 * ( delivered - m_reportedDelivered ) / ( now - m_reportTime );
            m_reportTime                          = now;
            m_reportedDelivered                   = delivered;
            const auto                     status = m_scheduleState == synco::armed ? INDI_OK : INDI_IDLE;
            const std::vector<std::string> settingNames{
                "steps", "maxCadenceHz", "keepoutUs", "guardUs", "step0LeadUs" };
            const std::vector<double>      settingValues{ double( m_requested ),
                                                     m_maxCadence,
                                                     m_bounds.keepout / 1000.0,
                                                     m_bounds.guard / 1000.0,
                                                     m_bounds.lead / 1000.0 };
            const std::vector<std::string> names{ "mainHz",           "cameraHz",
                                                  "plannedHz",        "deliveredHz",
                                                  "delivered",        "requested",
                                                  "active",           "feasible",
                                                  "missed",           "faults",
                                                  "cached",           "notStarted",
                                                  "copiedLate",       "snapshotMaxUs",
                                                  "copyMaxUs",        "primaryFaults",
                                                  "resourceFaults",   "outputFaults",
                                                  "triggerFaults",    "duplicatePosts",
                                                  "step0WakeUs",      "step0WakeMaxUs",
                                                  "step0AdvanceUs",   "copyBudgetUs",
                                                  "snapshotBudgetUs", "lastPrimaryDelta",
                                                  "lastIntervalUs",   "lastOutputDelta",
                                                  "lastFaultAgeSec",  "cameraJitterUs",
                                                  "cameraMinUs",      "cameraMaxUs",
                                                  "cameraAgeSec",     "primaryMeasuredHz",
                                                  "faultIntervalUs",  "faultPeriodUs",
                                                  "faultResidualUs" };
            const std::vector<double>      values{ 1e9 / m_period,
                                              m_cameraReport.m_fps,
                                              m_requested * 1e9 / m_period,
                                              m_deliveredHz,
                                              double( m_delivered ),
                                              double( m_requested ),
                                              m_primed ? double( m_requested ) : 0,
                                              double( m_feasible ),
                                              double( m_missed ),
                                              double( m_faults ),
                                              double( m_cached ),
                                              double( m_notStarted ),
                                              double( m_copiedLate ),
                                              m_snapshotMax / 1000.0,
                                              m_copyMax / 1000.0,
                                              double( m_faultCounts[size_t( Fault::primary )] ),
                                              double( m_faultCounts[size_t( Fault::resource )] ),
                                              double( m_faultCounts[size_t( Fault::output )] ),
                                              double( m_faultCounts[size_t( Fault::trigger )] ),
                                              double( m_duplicatePosts ),
                                              m_step0Wake / 1000.0,
                                              m_step0WakeMax / 1000.0,
                                              m_advances[0] / 1000.0,
                                              m_bounds.copy / 1000.0,
                                              m_bounds.snapshot / 1000.0,
                                              double( m_lastPrimaryDelta ),
                                              m_lastPrimaryInterval / 1000.0,
                                              double( m_lastOutputDelta ),
                                              m_lastFaultNs ? ( now - m_lastFaultNs ) / 1e9 : -1,
                                              m_cameraReport.m_jitterUs,
                                              m_cameraReport.m_minUs,
                                              m_cameraReport.m_maxUs,
                                              m_cameraReport.m_received > 0 ? ( now - m_cameraReport.m_received ) / 1e9
                                                                                 : -1,
                                              m_measuredPeriod > 0 ? 1e9 / m_measuredPeriod : 0,
                                              m_faultInterval / 1000.0,
                                              m_faultPeriod / 1000.0,
                                              m_faultPeriod ? ( m_faultInterval - m_faultPeriod ) / 1000.0 : 0 };
            const std::string              reason       = m_reason;
            const std::string              lastFault    = m_lastFaultReason;
            const std::string              periodSource = m_periodSource;
            std::string                    mappings;
            for( const auto &name : m_destinations )
                mappings += ( mappings.empty() ? "" : ", " ) + name + "_aux -> " + name;
            lock.unlock(); // Never hold the cycle mutex across INDI driver I/O.
            updateIfChanged( m_indiP_settings, settingNames, settingValues );
            updateIfChanged( m_indiP_status, names, values, status );
            updateIfChanged( m_indiP_health, "reason", reason );
            updateIfChanged( m_indiP_health, "lastFault", lastFault );
            updateIfChanged( m_indiP_health, "periodSource", periodSource );
            updateIfChanged( m_indiP_health, "primary", shmimMonitorT::m_shmimName );
            updateIfChanged( m_indiP_health, "combined", m_combinedName );
            updateIfChanged( m_indiP_health, "timingStream", m_timingName );
            updateIfChanged( m_indiP_health, "mapping", mappings );
        }
    }
    SHMIMMONITOR_UPDATE_INDI;
    TELEMETER_APP_LOGIC;
    return 0;
}

inline int dmSyncopation::appShutdown()
{
    m_enabled = false;
    SHMIMMONITOR_APP_SHUTDOWN;
    TELEMETER_APP_SHUTDOWN;
    std::lock_guard<std::mutex> lock( m_cycleMutex );
    invalidate( "shutdown" );
    closeResources();
    return 0;
}

inline int dmSyncopation::checkRecordTimes()
{
    return telemeterT::checkRecordTimes( telem_dmsynco() );
}

inline int dmSyncopation::recordTelem( const telem_dmsynco *record )
{
    static_cast<void>( record );
    std::unique_lock<std::mutex> lock( m_cycleMutex, std::try_to_lock );
    if( !lock.owns_lock() )
        return 0;
    std::vector<int64_t> residuals( m_residuals.begin(), m_residuals.begin() + m_requested );
    std::vector<int64_t> advances( m_advances.begin(), m_advances.begin() + m_requested );
    const int            state     = m_scheduleState;
    const int            count     = m_requested;
    const int64_t        period    = m_period;
    const uint64_t       delivered = m_delivered, missed = m_missed, faults = m_faults, cached = m_cached;
    const int64_t        snapshot = m_snapshotMax, copy = m_copyMax;
    lock.unlock();
    telem_dmsynco::messageT message(
        state, count, period, delivered, missed, faults, cached, snapshot, copy, residuals, advances );
    telem<telem_dmsynco>( message );
    return 0;
}

INDI_NEWCALLBACK_DEFN( dmSyncopation, m_indiP_enable )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_enable, ipRecv );
    if( ipRecv.getType() != pcf::IndiProperty::Switch || !ipRecv.find( "toggle" ) || ipRecv.getElements().size() != 1 )
        return -1;
    const bool enable = ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On;
    if( !enable )
        m_enabled = false;
    { // mutex scope
        std::lock_guard<std::mutex> lock( m_cycleMutex );
        invalidate( enable ? "acquiring" : "disabled" );
        m_lastMain       = 0;
        m_trackingOutput = false;
        m_enabled        = enable;
    }
    updateSwitchIfChanged( m_indiP_enable,
                           "toggle",
                           enable ? pcf::IndiElement::On : pcf::IndiElement::Off,
                           enable ? INDI_BUSY : INDI_IDLE );
    return 0;
}

INDI_NEWCALLBACK_DEFN( dmSyncopation, m_indiP_reacquire )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_reacquire, ipRecv );
    if( ipRecv.getType() != pcf::IndiProperty::Switch || !ipRecv.find( "request" ) || ipRecv.getElements().size() != 1 )
        return -1;
    if( ipRecv["request"].getSwitchState() == pcf::IndiElement::On )
    {
        std::lock_guard<std::mutex> lock( m_cycleMutex );
        invalidate( "reacquire requested" );
        m_lastMain       = 0;
        m_trackingOutput = false;
    }
    updateSwitchIfChanged( m_indiP_reacquire, "request", pcf::IndiElement::Off, INDI_IDLE );
    return 0;
}

INDI_NEWCALLBACK_DEFN( dmSyncopation, m_indiP_settings )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_settings, ipRecv );
    if( ipRecv.getType() != pcf::IndiProperty::Number || ipRecv.getElements().empty() )
        return -1;
    std::lock_guard<std::mutex> lock( m_cycleMutex );
    if( m_enabled )
        return -1;
    int    requested = m_requested;
    double cadence   = m_maxCadence;
    auto   bounds    = m_bounds;
    for( const auto &element : ipRecv.getElements() )
    {
        const auto  &name  = element.first;
        const double value = element.second.get<double>();
        if( !finite( value ) || value < 0 || value > 1e6 )
            return -1;
        if( name == "steps" )
        {
            if( value < 1 || value > synco::maxSteps || value != std::floor( value ) )
                return -1;
            requested = int( value );
        }
        else if( name == "maxCadenceHz" )
        {
            if( value == 0 )
                return -1;
            cadence = value;
        }
        else if( name == "keepoutUs" )
            bounds.keepout = int64_t( std::ceil( value * 1000 ) );
        else if( name == "guardUs" )
            bounds.guard = int64_t( std::ceil( value * 1000 ) );
        else if( name == "step0LeadUs" )
            bounds.lead = int64_t( std::ceil( value * 1000 ) );
        else
            return -1;
    }
    m_requested  = requested;
    m_maxCadence = cadence;
    m_bounds     = bounds;
    invalidate( "disabled settings changed" );
    return 0;
}

INDI_SETCALLBACK_DEFN( dmSyncopation, m_indiP_camera )( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getType() != pcf::IndiProperty::Number || m_cameraDevice.empty() ||
        ipRecv.getDevice() != m_cameraDevice || ipRecv.getName() != m_cameraProperty ||
        !ipRecv.find( m_cameraElement ) )
        return -1;
    double      values[]{ 0, -1, -1, -1 };
    const char *names[]{ m_cameraElement.c_str(), "acq_jitter", "acq_min", "acq_max" };
    for( size_t j = 0; j < 4; ++j )
    {
        if( !ipRecv.find( names[j] ) )
            continue;
        // IndiElement's numeric getter does not report failed or partial conversions.
        std::istringstream input( ipRecv[names[j]].getValue() );
        if( !( input >> values[j] ) || !( input >> std::ws ).eof() || !finite( values[j] ) ||
            ( j == 0 ? ( values[j] <= 0 || values[j] > 100000 ) : ( values[j] < 0 || values[j] > 1000000 ) ) )
            return -1;
    }
    if( values[2] >= 0 && values[3] >= 0 && values[2] > values[3] )
        return -1;
    std::lock_guard<std::mutex> lock( m_cameraMutex );
    m_cameraFps      = values[0];
    m_cameraTime     = m_now();
    m_cameraJitterUs = values[1] >= 0 ? values[1] * 1e6 : -1;
    m_cameraMinUs    = values[2] >= 0 ? values[2] * 1e6 : -1;
    m_cameraMaxUs    = values[3] >= 0 ? values[3] * 1e6 : -1;
    return 0;
}

} // namespace app
} // namespace MagAOX
#endif
