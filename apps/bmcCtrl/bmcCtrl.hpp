/** \file bmcCtrl.hpp
 * \brief The MagAO-X BMC DM controller header file
 *
 * \ingroup bmcCtrl_files
 */

/*
Open questions:
* Fix actuator mapping issue (don't want a bunch of if statements)


Test:
* bias code

*/

// #define _GLIBCXX_USE_CXX11_ABI 0

#ifndef bmcCtrl_hpp
#define bmcCtrl_hpp

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"
#include "../../libMagAOX/app/dev/telemeter.hpp"

#include <array>
#include <condition_variable>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>

/* BMC SDK C Header */
#include <BMCApi.h>

/** \defgroup bmcCtrl
 * \brief The MagAO-X application to control a BMC DM
 *
 * <a href="..//apps_html/page_module_bmcCtrl.html">Application Documentation</a>
 *
 * \ingroup apps
 *
 */

/** \defgroup bmcCtrl_files
 * \ingroup bmcCtrl
 */

namespace MagAOX
{
namespace app
{

/// The MagAO-X BMC DM Controller
/**
 * \ingroup bmcCtrl
 */
class bmcCtrl : public MagAOXApp<true>,
                public dev::dm<bmcCtrl, float>,
                public dev::shmimMonitor<bmcCtrl>,
                public dev::telemeter<bmcCtrl>
{

    // Give the test harness access.
    friend class bmcCtrl_test;

    friend class dev::dm<bmcCtrl, float>;

    friend class dev::shmimMonitor<bmcCtrl>;

    friend class dev::telemeter<bmcCtrl>;

    typedef dev::telemeter<bmcCtrl> telemeterT;

    typedef float realT; ///< This defines the datatype used to signal the DM using the ImageStreamIO library.

    typedef dev::dm<bmcCtrl, float> dmT;

    typedef dev::shmimMonitor<bmcCtrl> shmimMonitorT;

  protected:
    /** \name Configurable Parameters - Data
     *@{
     */

    std::string m_serialNumber; ///< The BMC serial number used to open the correct DM profile

    long m_satThresh{ 100000 }; ///< Threshold above which to log saturation.

    ///@}

    std::atomic<long> m_nsat{ 0 }; ///< Saturated actuator count accumulated for ordinary app reporting.

    /** \name SDK Worker - Data
     * @{ */

    bool m_sdkWorkerEnabled{ false }; ///< Startup-only opt-in to overlapped SDK submission.

    int m_sdkThreadPrio{ 0 }; ///< Requested SDK-worker scheduling priority.

    std::string m_sdkCpuset; ///< Requested SDK-worker cpuset; empty retains inherited affinity.

    double m_sdkTimeoutSec{ 1.0 }; ///< Diagnostic normal-command deadline; never cancels or relinquishes SDK ownership.

    std::thread m_sdkThread; ///< Sole SDK owner while worker mode is enabled.

    bool m_sdkThreadInit{ true }; ///< threadStart compatibility flag; the worker waits on m_sdkConfigured instead.

    pid_t m_sdkThreadID{ 0 }; ///< Worker TID used by the standard thread readback.

    pcf::IndiProperty m_sdkThreadProp; ///< Standard requested scheduling/TID readback.

    pcf::IndiProperty m_indiP_sdkWorker; ///< Ordinary-thread snapshot of worker admission and result diagnostics.

    /// SDK operations serialized on the same handle owner.
    enum class SdkOperation
    {
        open,
        loadMap,
        set,
        clear,
        close
    };

    /// Fixed request whose pointees remain owned until its matching result is consumed.
    struct SdkRequest
    {
        SdkOperation  operation{ SdkOperation::set }; ///< Operation selected by the serialized caller.
        const char   *serial{ nullptr };              ///< Control-owned serial string for open.
        const double *values{ nullptr };              ///< Immutable command or zero vector for set.
        uint32_t     *map{ nullptr };                 ///< Control-owned mapping output for loadMap.
        uint64_t      sequence{ 0 };                  ///< Unique request sequence, shared with its result.
        uint64_t      generation{ 0 };                ///< Handle generation frozen at dispatch.
        bool    normalCommand{ false }; ///< Only normal command submissions participate in the diagnostic watchdog.
        int64_t dispatchNs{ 0 };        ///< Existing monitor dispatch timestamp; control operations remain exempt.
    };

    /// Fixed SDK result imported by the joined monitor or the detached post owner.
    struct SdkResult
    {
        BMCRC    result{ NO_ERR };    ///< Actual vendor result, or -1 when the SDK throws.
        uint32_t actuatorCount{ 0 };  ///< Owned copy of handle metadata following open.
        uint32_t driverType{ 0 };     ///< SDK-owner snapshot of the active transport interface ID.
        uint32_t amplifierType{ 0 };  ///< SDK-owner snapshot of the PCIe amplifier configuration ID.
        int64_t  startNs{ 0 };        ///< Actual monotonic SDK entry boundary.
        int64_t  endNs{ 0 };          ///< Actual monotonic SDK return/exception boundary.
        uint64_t sequence{ 0 };       ///< Matching request sequence.
        uint64_t generation{ 0 };     ///< Matching handle generation.
        bool     exception{ false };  ///< SDK raised a C++ exception instead of returning a code.
        bool     timedOut{ false };   ///< A diagnostic deadline was observed; actual SDK ownership was retained.
        bool     newTimeout{ false }; ///< Finalization detected a violation not already counted by ordinary app logic.
    };

    std::mutex
        m_sdkMutex; ///< Protects fixed request/result visibility; nested locking always takes SDK before command mutex.

    std::condition_variable m_sdkWake; ///< Wakes the sole worker for a pending request or orderly stop.

    std::condition_variable m_sdkDone; ///< Wakes the exact request owner after actual SDK return.

    SdkRequest m_sdkRequest; ///< One fixed request slot, never overwritten while occupied.

    SdkResult m_sdkResult; ///< One fixed result slot, consumed by its exact request owner.

    bool m_sdkPending{ false }; ///< A request has been dispatched and not yet consumed.

    bool m_sdkReady{ false }; ///< The pending request has returned from the SDK.

    bool m_sdkTimedOut{ false }; ///< Once-per-job timeout claim protected by the SDK handoff mutex.

    bool m_sdkStop{ false }; ///< Stop requested only after control and command owners drain.

    bool m_sdkConfigured{ false }; ///< Mutex-protected worker startup barrier after actual placement verification.

    int m_sdkPolicy{ -1 }; ///< Verified worker scheduling policy, recorded before SDK access is enabled.

    int m_sdkActualPrio{ -1 }; ///< Verified worker scheduling priority.

    int m_sdkAllowedCpus{ 0 }; ///< Verified worker affinity-mask population.

    uint64_t m_sdkSequence{ 0 }; ///< Monotonically increasing request sequence under the SDK mutex.

    uint64_t m_sdkGeneration{ 0 }; ///< Control-owned handle generation, changed with admission closed.

    std::mutex m_controlMutex; ///< Serializes whole control operations, including their drain and setup.

    std::mutex
        m_commandMutex; ///< Protects admission/callback ownership; never acquire SDK mutex while holding this mutex.

    std::condition_variable m_commandRetired; ///< Control waits until the entire monitor callback retires.

    bool m_admissionClosed{ true }; ///< Controls close admission before waiting for callback retirement.

    bool m_commandActive{ false }; ///< Remains true through timing and full-completion publication.

    bool m_sdkFault{ false }; ///< Fault latch; only release and successful reinitialization recover it.

    bool m_haveInput{ false }; ///< A coherent input has been admitted in this handle generation.

    ino_t m_inputInode{ 0 }; ///< Input stream identity used for duplicate/coalesced-post rejection.

    uint64_t m_inputCounter{ 0 }; ///< Last admitted input counter; never replay semaphore backlog.

    uint64_t m_rejectedInputs{ 0 }; ///< Attempts rejected while unavailable, incoherent or duplicate.

    uint64_t m_coalescedInputs{ 0 }; ///< Counter advances skipped while the monitor was occupied.

    uint64_t m_sdkFailures{ 0 }; ///< Command SDK failures/exceptions observed by finalization.

    uint64_t m_completedCommands{ 0 }; ///< Matched command results collected at callback tail.

    uint64_t m_sdkTimeouts{ 0 }; ///< Diagnostic deadline violations, counted once per normal command.

    double m_sdkActiveAgeSec{ 0 }; ///< Ordinary-thread observed age of a pending normal SDK call.

    std::vector<double> m_sdkValues; ///< Preallocated immutable SDK input, separate from saturation input.

    std::vector<uint8_t> m_stagedSat; ///< Preallocated private instantaneous saturation results.

    long m_stagedSatCount{ 0 }; ///< Saturation count committed only after all prepublication processing succeeds.

    uint64_t m_commandJob{ 0 }; ///< Monitor-owned pending SDK sequence, zero when no command was dispatched.

    int64_t m_sdkDispatchNs{ 0 }; ///< Monitor-side handoff timestamp for the active command.

    /// Startup-only full pipeline selection; the existing SDK worker is required.
    bool m_pipelineEnabled{ false };

    int m_pipelineThreadPrio{ 0 }; ///< Requested post-processing worker priority.

    std::string m_pipelineCpuset; ///< Post-processing worker CPU set; independently verified at startup.

    std::string m_pipelineResultStream; ///< Optional dedicated actual SDK-result stream.

    /// One bounded immutable command, retained until preparation, SDK execution and post-processing all retire.
    struct PipelineJob
    {
        dmT::PipelineCommand         command;  ///< Generic private shape, staged products, timing and source identity.
        MagAOX::dmCompletion::Record identity; ///< Immutable source identity copied before SDK dispatch.
        std::vector<double>          values;   ///< Dedicated immutable SDK actuator vector.
        SdkResult                    result;   ///< Actual SDK result and timestamps, consumed only after sdkDone.
        uint64_t                     order{ 0 };    ///< Admission sequence for ordered submission and publication.
        bool                         busy{ false }; ///< Pool credit remains owned through post retirement.
        bool staged{ false };     ///< Monitor finished every channel read and detached reader release.
        bool queued{ false };     ///< SDK owner may select this prepared vector.
        bool running{ false };    ///< Native SDK call currently owns this job's vector.
        bool sdkDone{ false };    ///< Actual SDK result or explicit cancellation is available.
        bool submitted{ false };  ///< This command actually entered the native SDK.
        bool posting{ false };    ///< Sole post worker owns finalization outside the SDK mutex.
        bool cancelled{ false };  ///< No SDK submission is allowed for this admitted command.
        bool timedOut{ false };   ///< Diagnostic timeout counted once while SDK ownership is retained.
        bool superseded{ false }; ///< A newer observed command cancelled this unsent job without a backend fault.
    };

    static constexpr size_t pipelineCapacity{ 3 }; ///< Fixed preparation/SDK/post credits; never an unbounded FIFO.

    std::array<PipelineJob, pipelineCapacity>
        m_pipelineJobs; ///< Protected state under m_sdkMutex; payload owners differ.

    uint64_t m_pipelineOrder{ 0 }; ///< Last admitted pipeline job sequence.

    size_t m_pipelineInFlight{ 0 }; ///< Busy pool credits; protected by m_sdkMutex and reflected by commandActive.

    size_t m_pipelineSdkActive{ pipelineCapacity }; ///< SDK-owned pool slot, or pipelineCapacity when idle.

    uint64_t m_pipelineAdmitted{ 0 }; ///< Commands admitted into private pipeline storage.

    uint64_t m_pipelineRetired{ 0 }; ///< Whole jobs whose ordered post work finished.

    uint64_t m_pipelineOverloaded{ 0 }; ///< Fresh receipts rejected because every bounded credit was owned.

    uint64_t m_pipelineCancelled{ 0 }; ///< Admitted commands cancelled before entering the SDK.

    uint64_t m_pipelineSuperseded{
        0 }; ///< Unsent commands superseded by newer observations or unstable source identity.

    uint64_t m_pipelineVectorRejected{
        0 }; ///< Source identity rejections before native dispatch, including unstable initial capture.

    uint64_t m_pipelineReaderUnqualified{ 0 }; ///< Admitted jobs lacking stable late reader/full provenance.

    uint64_t m_pipelinePreparationFailures{
        0 }; ///< Actual conversion/staging errors or unexpected preparation exceptions.

    MagAOX::dmCompletion::Record m_pipelineLatest; ///< Monitor-announced pointer-free identity under m_sdkMutex.

    bool m_pipelineHaveLatest{ false }; ///< Latest observed source remains attached and coherently identified.

    uint64_t m_pipelinePostFailures{
        0 }; ///< Post-processing failures, including failures discovered after a later send.

    std::thread m_pipelineThread; ///< Sole ordered post-processing and generic timing/completion owner.

    bool m_pipelineThreadInit{ true }; ///< Standard threadStart compatibility; worker uses configured handshake.

    pid_t m_pipelineThreadID{ 0 }; ///< Post-processing worker TID.

    pcf::IndiProperty m_pipelineThreadProp; ///< Standard post-worker thread readback.

    pcf::IndiProperty m_indiP_pipeline; ///< Ordinary-thread bounded pipeline diagnostics.

    bool m_pipelineConfigured{ false }; ///< Post worker may run only after verified initialization.

    bool m_pipelineStop{ false }; ///< Stop only after all pipeline credits have retired.

    std::condition_variable m_pipelineWake; ///< SDK/preparation changes wake the ordered post worker.

    int m_pipelinePolicy{ -1 }; ///< Verified post-worker scheduling policy.

    int m_pipelineActualPrio{ -1 }; ///< Verified post-worker scheduling priority.

    int m_pipelineAllowedCpus{ 0 }; ///< Verified post-worker affinity population.

    dev::dmCompletion m_pipelineResult; ///< SDK-thread-only actual submission-result publisher.

    /// @}

  public:
    /// Default c'tor.
    bmcCtrl();

    /// D'tor.
    ~bmcCtrl() noexcept;

    /// Setup the configuration system.
    virtual void setupConfig();

    /// Implementation of loadConfig logic, separated for testing.
    /** This is called by loadConfig().
     */
    int loadConfigImpl(
        mx::app::appConfigurator &_config /**< [in] an application configuration from which to load values*/ );

    /// Load the configuration
    virtual void loadConfig();

    /// Startup function
    /** Sets up INDI, and starts the shmim thread.
     *
     */
    virtual int appStartup();

    /// Implementation of the FSM for bmcCtrl.
    /**
     * \returns 0 on no critical error
     * \returns -1 on an error requiring shutdown
     */
    virtual int appLogic();

    /// Shutdown the app.
    /**
     *
     */
    virtual int appShutdown();

    /// Cleanup after a power off.
    /**
     */
    virtual int onPowerOff();

    /// Maintenace while powered off.
    /**
     */
    virtual int whilePowerOff();

    /** \name DM Base Class Interface
     *
     *@{
     */

    /// Initialize the DM and prepare for operation.
    /** Application is in state OPERATING upon successful conclusion.
     *
     * \returns 0 on success
     * \returns -1 on error
     */
    int initDM();

    /// Zero all commands on the DM
    /** This does not update the shared memory buffer.
     *
     * \returns 0 on success
     * \returns -1 on error
     */
    int zeroDM();

    /// Select the full pipeline while preserving the inherited synchronous and joined-worker callback paths.
    int processImage( void              *source /**< [in] Current combined command pixels. */,
                      const dev::shmimT &tag /**< [in] Primary monitor tag. */ );

    /// Drain all pipeline owners before inherited allocation changes shared channel/output storage.
    int allocate( const dev::shmimT &tag /**< [in] Reconnected primary monitor. */ );

    /// Preserve one timing producer by routing pipelined monitor events to detached captures.
    void shmimMonitorTiming( const dev::shmimT     &tag /**< [in] Primary monitor tag. */,
                             dev::shmimMonitorEvent event /**< [in] Receipt or lifecycle event. */ );

    /// Expose the startup-only pipeline selection to generic DM helpers.
    bool pipelinedCommand() const;

    /// Send a command to the DM
    /** This is called by the shmim monitoring thread in response to a semaphore trigger.
     * Worker mode returns after dispatch and private staging; finalizeCommand() collects actual SDK success.
     * The default synchronous mode returns after SDK return and saturation processing.
     *
     * \returns 0 on successful synchronous submission or deferred preparation
     * \returns -1 on error
     */
    int commandDM( void *curr_src /**< [in] Combined command pixels. */ );

    /// Admit a fresh coherent semaphore command without waiting on a selected pixel pointer.
    bool admitCommand();

    /// Select generic staged processing only for worker-enabled callbacks.
    bool deferredCommand() const;

    /// Drain an actual SDK result and commit saturation only after successful processing and SDK return.
    int finalizeCommand( int processingResult /**< [in] Preparation/channel-staging result. */ ) noexcept;

    /// Release command ownership after all callback publication and timing have finished.
    void retireCommand() noexcept;

    /// Release the DM, making it safe to turn off power.
    /** The application will be state READY at the conclusion of this.
     *
     * \returns 0 on success
     * \returns -1 on error
     */
    int releaseDM();

    ///@}

    /** \name BMC Interface - Data
     *@{
     */

  protected:
    double   m_act_gain{ 0 };      ///< Actuator gain (microns/volt)
    double   m_volume_factor{ 0 }; ///< the volume factor to convert from displacement to commands
    uint32_t m_nbAct{ 0 };         ///< The number of actuators

    /// Calibrated command-vector address corresponding to each image-grid actuator.
    int *m_actuator_mapping{ nullptr };

    double *m_dminputs{ nullptr }; ///< Monitor-owned preparation and saturation input; never modified by the worker.

    DM m_dm = {}; ///< Handle accessed by the sole SDK owner in worker mode; reset only after quiescence.

    bool m_dmopen{ false }; ///< Track whether the DM connection has been opened

    /// Start and initialize the optional sole SDK owner using standard thread settings.
    int startSdkWorker();

    /// Verify actual scheduling and, when requested, exact cpuset affinity before allowing SDK work.
    int verifySdkThread();

    /// Drain command ownership, wake the idle worker and join before destroying resources.
    void stopSdkWorker() noexcept;

    /// Enter the worker loop after standard thread initialization.
    static void sdkThreadStart( bmcCtrl *app /**< [in] Owning controller. */ );

    /// Execute fixed SDK requests, returning all errors through the matching result slot.
    void sdkThreadExec() noexcept;

    /// Call one SDK operation on the sole owner and capture its actual boundaries.
    SdkResult performSdk( const SdkRequest &request /**< [in] Immutable request and borrowed storage. */ ) noexcept;

    /// Dispatch a fixed request without allocating; zero means the slot was unavailable.
    uint64_t dispatchSdk( SdkRequest request /**< [in] Request whose storage remains valid through waitSdk. */ );

    /// Collect the exact dispatched result without treating notifications as job tokens.
    SdkResult waitSdk( uint64_t sequence /**< [in] Exact outstanding request sequence. */ );

    /// Run a control operation synchronously through the worker, or directly in legacy mode.
    SdkResult runSdk( SdkRequest request /**< [in] Control request with synchronous borrowed storage. */ );

    /// Close admission and drain the whole callback; return whether it was previously open.
    bool closeAdmission();

    /// Reopen admission after a successful control operation when the handle is ready and unfaulted.
    void openAdmission( bool reopen /**< [in] Whether the caller intends to resume commands. */ );

    /// Zero through the selected SDK owner while the caller already holds control ownership.
    int zeroDMImpl();

    /// Observe a pending normal SDK call without waiting for return, retaining all resource ownership on timeout.
    void
    checkSdkTimeout( int64_t nowNs /**< [in] Local monotonic observation time; explicit for deterministic tests. */ );

    /// Update read-only worker diagnostics outside realtime execution.
    void updateSdkWorkerINDI();

    /// Allocate every fixed job only while command admission is closed and all earlier work is drained.
    void allocatePipelineJobs();

    /// Capture, prepare, dispatch and release one job without waiting for its SDK or post result.
    int processPipeline( void *source /**< [in] Current combined command pixels. */ );

    /// Reserve a fresh coherent command and a free private credit; pipelineCapacity means rejection.
    size_t claimPipelineJob();

    /// Return the oldest queued SDK job; caller holds m_sdkMutex.
    size_t nextPipelineSdk() const;

    /// Return the oldest busy job, including one whose staging/SDK result is not ready; caller holds m_sdkMutex.
    size_t nextPipelinePost() const;

    /// Close admission and cancel every not-yet-submitted job; caller holds m_sdkMutex.
    void faultPipeline();

    /// Reject an unstable private source without faulting future admissions; before SDK dispatch only.
    void rejectPipelineVector( PipelineJob &job /**< [in,out] Unsent job whose source cannot be proved stable. */ );

    /// Cancel an unsent job displaced by newer source evidence; requires m_sdkMutex.
    void supersedePipelineJob( PipelineJob &job /**< [in,out] Private unsent command. */ );

    /// Define complete read-only worker diagnostics before any ordinary-loop updates occur.
    int setupSdkDiagnostics();

    /// Start and verify the sole post worker before allowing any pipeline command.
    int startPipelineWorker();

    /// Stop the post owner after all credits drain; caller already closed admission.
    void stopPipelineWorker() noexcept;

    /// Enter the ordered post worker.
    static void pipelineThreadStart( bmcCtrl *app /**< [in] Owning controller. */ );

    /// Publish completed commands in admission order and retire their private storage.
    void pipelineThreadExec() noexcept;

    /// Verify scheduling and CPU placement for either worker using the existing startup contract.
    int verifyWorkerThread( std::thread       &thread /**< [in] Started worker. */,
                            int                priority /**< [in] Requested scheduling priority. */,
                            const std::string &cpuset /**< [in] Requested CPU set, or inherited affinity. */,
                            int               &actualPolicy /**< [out] Observed scheduling policy. */,
                            int               &actualPriority /**< [out] Observed priority. */,
                            int               &allowedCpus /**< [out] Observed allowed CPU count. */ );

    ///@}

  public:
    /** \name BMC Interface
     * @{ */

    /// Parse the BMC calibration file
    /** \returns 0 on success
     * \returns -1 on error
     */
    int parse_calibration_file();

    /// Read the actuator mapping from a FITS file
    /**
     * \todo convert this to use mxlib::fitsFile
     *
     * \returns 0 on success
     * \returns -1 on error
     */
    int get_actuator_mapping();

    ///@}

  public:
    /** \name Telemeter Interface
     * @{ */

    /// Check the common DM timing record's forced-publication interval.
    int checkRecordTimes();

    /// Record the common cached DM timing summary on a forced telemetry request.
    int recordTelem( const telem_dmtimings *telem /**< [in] Telemetry type selector. */ );

    ///@}
};

bmcCtrl::bmcCtrl() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    m_powerMgtEnabled = true;
    return;
}

bmcCtrl::~bmcCtrl() noexcept
{
    stopSdkWorker();

    if( m_actuator_mapping )
    {
        free( m_actuator_mapping );
    }

    if( m_dminputs )
    {
        free( m_dminputs );
    }
}

void bmcCtrl::setupConfig()
{
    config.add( "dm.serialNumber",
                "",
                "dm.serialNumber",
                argType::Required,
                "dm",
                "serialNumber",
                false,
                "string",
                "The BMC serial number used to find correct DM Profile." );
    config.add( "dm.calibRelDir",
                "",
                "dm.calibRelDir",
                argType::Required,
                "dm",
                "calibRelDir",
                false,
                "string",
                "Used to find the default config directory." );
    config.add( "dm.satThresh",
                "",
                "dm.satThresh",
                argType::Required,
                "dm",
                "satThresh",
                false,
                "string",
                "Threshold above which to log saturation." );

    config.add( "sdk.workerEnabled",
                "",
                "sdk.workerEnabled",
                argType::Required,
                "sdk",
                "workerEnabled",
                false,
                "bool",
                "Use the opt-in sole SDK worker and staged reader release. Requires restart." );
    config.add( "sdk.threadPrio",
                "",
                "sdk.threadPrio",
                argType::Required,
                "sdk",
                "threadPrio",
                false,
                "int",
                "Requested SDK-worker FIFO priority (0 selects ordinary scheduling)." );
    config.add( "sdk.cpuset",
                "",
                "sdk.cpuset",
                argType::Required,
                "sdk",
                "cpuset",
                false,
                "string",
                "SDK-worker cpuset; use a separate qualified physical core." );

    config.add( "sdk.timeoutSec",
                "",
                "sdk.timeoutSec",
                argType::Required,
                "sdk",
                "timeoutSec",
                false,
                "double",
                "Positive finite diagnostic normal-command deadline in seconds; never cancels the SDK call." );

    config.add( "sdk.pipelineEnabled",
                "",
                "sdk.pipelineEnabled",
                argType::Required,
                "sdk",
                "pipelineEnabled",
                false,
                "bool",
                "Opt into the three-credit preparation/SDK/post pipeline; requires workerEnabled and restart." );
    config.add( "sdk.pipelineThreadPrio",
                "",
                "sdk.pipelineThreadPrio",
                argType::Required,
                "sdk",
                "pipelineThreadPrio",
                false,
                "int",
                "Post-processing worker priority; zero selects ordinary scheduling." );
    config.add( "sdk.pipelineCpuset",
                "",
                "sdk.pipelineCpuset",
                argType::Required,
                "sdk",
                "pipelineCpuset",
                false,
                "string",
                "Dedicated post-processing worker CPU set." );
    config.add( "sdk.pipelineResultStream",
                "",
                "sdk.pipelineResultStream",
                argType::Required,
                "sdk",
                "pipelineResultStream",
                false,
                "string",
                "Optional distinct stream containing actual SDK results only." );

    DM_SETUP_CONFIG( config );

    TELEMETER_SETUP_CONFIG( config );
}

int bmcCtrl::loadConfigImpl( mx::app::appConfigurator &_config )
{
    _config( m_calibRelDir, "dm.calibRelDir" );
    _config( m_serialNumber, "dm.serialNumber" );
    _config( m_satThresh, "dm.satThresh" );
    _config( m_sdkWorkerEnabled, "sdk.workerEnabled" );
    _config( m_sdkThreadPrio, "sdk.threadPrio" );
    _config( m_sdkCpuset, "sdk.cpuset" );
    _config( m_sdkTimeoutSec, "sdk.timeoutSec" );
    _config( m_pipelineEnabled, "sdk.pipelineEnabled" );
    _config( m_pipelineThreadPrio, "sdk.pipelineThreadPrio" );
    _config( m_pipelineCpuset, "sdk.pipelineCpuset" );
    _config( m_pipelineResultStream, "sdk.pipelineResultStream" );

    if( m_sdkThreadPrio < 0 || m_sdkThreadPrio > 99 || !mx::math::isFinite( m_sdkTimeoutSec ) || m_sdkTimeoutSec <= 0 )
    {
        return -1;
    }

    if( ( m_pipelineEnabled && !m_sdkWorkerEnabled ) || m_pipelineThreadPrio < 0 || m_pipelineThreadPrio > 99 ||
        ( !m_pipelineEnabled && !m_pipelineResultStream.empty() ) )
    {
        return -1;
    }

    DM_LOAD_CONFIG( _config );

    if( m_pipelineEnabled && !m_pipelineResultStream.empty() )
    {
        const std::vector<std::string> reserved{ m_completionStreamName,
                                                 m_readerReleaseStreamName,
                                                 m_shmimFlat,
                                                 m_shmimTest,
                                                 m_shmimSat,
                                                 m_shmimSatPerc,
                                                 m_shmimShape,
                                                 m_shmimDelta,
                                                 m_shmimDiff,
                                                 m_shmimName + "_actmask" };

        if( m_pipelineResultStream.find_first_not_of(
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" ) != std::string::npos ||
            std::find( reserved.begin(), reserved.end(), m_pipelineResultStream ) != reserved.end() ||
            std::find( m_deltaChannels.begin(), m_deltaChannels.end(), m_pipelineResultStream ) !=
                m_deltaChannels.end() ||
            m_pipelineResultStream.starts_with( m_shmimName ) )
        {
            return -1;
        }
    }

    TELEMETER_LOAD_CONFIG( _config );

    return 0;
}

void bmcCtrl::loadConfig()
{
    if( loadConfigImpl( config ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__, "error loading config" } );
        m_shutdown = 1;
    }
}

int bmcCtrl::appStartup()
{
    // This cleanup token does not own the app; it joins a worker on every later startup error return.
    auto stopWorker = []( bmcCtrl *app /**< [in] Controller whose startup did not finish. */ )
    {
        app->stopSdkWorker();
    };
    std::unique_ptr<bmcCtrl, decltype( stopWorker )> workerCleanup( this, stopWorker );

    if( parse_calibration_file() < 0 )
    {
        log<software_critical>( { __FILE__, __LINE__ } );

        return -1;
    }

    if( m_act_gain == 0 || m_volume_factor == 0 )
    {
        log<software_critical>( { __FILE__, __LINE__, "calibration not loaded properly" } );

        return -1;
    }

    if( m_sdkWorkerEnabled && startSdkWorker() < 0 )
    {
        return -1;
    }

    if( m_pipelineEnabled && startPipelineWorker() < 0 )
    {
        return -1;
    }

    if( setupSdkDiagnostics() < 0 )
    {
        return -1;
    }

    const char *version = BMCVersionString();
    log<text_log>( std::string( "BMC SDK version: " ) + ( version ? version : "unavailable" ) );

    DM_APP_STARTUP;

    TELEMETER_APP_STARTUP;

    SHMIMMONITOR_APP_STARTUP;

    workerCleanup.release();
    return 0;
}

int bmcCtrl::appLogic()
{
    DM_APP_LOGIC;

    SHMIMMONITOR_APP_LOGIC;
    TELEMETER_APP_LOGIC;
    checkSdkTimeout( dev::dmTiming::monotonicNowNs() );
    updateSdkWorkerINDI();

    if( state() == stateCodes::POWEROFF )
    {
        return 0;
    }

    if( state() == stateCodes::POWERON )
    {
        sleep( 5 );

        state( stateCodes::NOTHOMED );
    }

    const long saturated = m_nsat.exchange( 0 );

    if( saturated > m_satThresh )
    {
        log<text_log>( "Saturated actuators in last second: " + std::to_string( saturated ), logPrio::LOG_WARNING );
    }

    DM_UPDATE_INDI;
    SHMIMMONITOR_UPDATE_INDI;

    return 0;
}

int bmcCtrl::appShutdown()
{
    if( m_sdkWorkerEnabled || m_dmopen )
    {
        releaseDM();
    }

    DM_APP_SHUTDOWN;

    SHMIMMONITOR_APP_SHUTDOWN;
    TELEMETER_APP_SHUTDOWN;

    stopSdkWorker();

    return 0;
}

int bmcCtrl::onPowerOff()
{
    return dm<bmcCtrl, float>::onPowerOff();
}

int bmcCtrl::whilePowerOff()
{
    return dm<bmcCtrl, float>::whilePowerOff();
}

int bmcCtrl::initDM()
{
    std::unique_lock<std::mutex> control( m_controlMutex, std::defer_lock );
    bool                         reopen = false;

    if( m_sdkWorkerEnabled )
    {
        control.lock();
        reopen = closeAdmission();
    }

    if( m_dmopen )
    {
        if( m_sdkWorkerEnabled )
        {
            openAdmission( reopen );
        }

        log<text_log>( "DM is already initialized.  Release first.", logPrio::LOG_ERROR );

        return -1;
    }

    std::string ser = mx::ioutils::toUpper( m_serialNumber );
    BMCRC       ret = NO_ERR;
    SdkRequest  open;
    open.operation = SdkOperation::open;
    open.serial    = ser.c_str();
    ++m_sdkGeneration;
    const auto opened = runSdk( open );
    ret               = opened.result;

    if( ret == NO_ERR )
    {
        m_dmopen = true; // remember that the DM connection has been opened
    }

    if( ret != NO_ERR )
    {
        const char *err;
        err = BMCErrorString( ret );
        log<text_log>( std::string( "DM initialization failed: " ) + err, logPrio::LOG_ERROR );
        m_dm = {};

        return -1;
    }

    if( !m_dmopen )
    {
        log<text_log>( "DM initialization failed. Couldn't open DM handle.", logPrio::LOG_ERROR );

        return -1;
    }

    // Handle metadata is copied on the SDK owner and read only after its matching result.
    m_nbAct             = opened.actuatorCount;
    const char *version = BMCVersionString();
    log<text_log>( std::string( "BMC SDK " ) + ( version ? version : "unavailable" ) + " device driverType=" +
                   std::to_string( opened.driverType ) + " amplifierType=" + std::to_string( opened.amplifierType ) );

    if( m_nbAct == 0 || m_nbAct > MAX_DM_SIZE )
    {
        return log<software_error, -1>( { "invalid BMC actuator count" } );
    }

    // Load the DM map
    uint32_t *map_lut;
    map_lut = reinterpret_cast<uint32_t *>( malloc( sizeof( uint32_t ) * MAX_DM_SIZE ) );

    if( map_lut == nullptr )
    {
        return log<software_error, -1>( { "allocating BMC map buffer" } );
    }

    SdkRequest load;
    load.operation = SdkOperation::loadMap;
    load.map       = map_lut;
    ret            = runSdk( load ).result;
    free( map_lut );

    if( ret != NO_ERR )
    {
        const char *err;
        err = BMCErrorString( ret );
        log<text_log>( std::string( "DM initialization failed. Couldn't load map." ) + err, logPrio::LOG_ERROR );

        if( !m_sdkWorkerEnabled )
        {
            m_dm = {};
        }

        state( stateCodes::ERROR );

        return -1;
    }

    if( m_dminputs )
    {
        free( m_dminputs );
    }

    m_dminputs = reinterpret_cast<double *>( calloc( m_nbAct, sizeof( double ) ) );

    if( m_dminputs == nullptr )
    {
        return log<software_error, -1>( { "allocating BMC command buffer" } );
    }

    if( zeroDMImpl() < 0 )
    {
        log<text_log>( "DM initialization failed.  Error zeroing DM.", logPrio::LOG_ERROR );
        state( stateCodes::ERROR );

        return -1;
    }

    /* get actuator mapping from 2D cacao image to 1D vector for BMC input */
    if( m_actuator_mapping )
    {
        free( m_actuator_mapping );
    }

    m_actuator_mapping = reinterpret_cast<int *>( malloc( m_nbAct * sizeof( int ) ) ); /* memory for actuator mapping */

    if( m_actuator_mapping == nullptr )
    {
        return log<software_error, -1>( { "allocating BMC actuator mapping" } );
    }

    /* initialize to -1 to allow for handling addressable but ignored actuators */
    for( uint32_t idx = 0; idx < m_nbAct; ++idx )
    {
        m_actuator_mapping[idx] = -1;
    }

    if( get_actuator_mapping() < 0 )
    {
        log<text_log>( "DM initialization failed.  Failed to get actuator mapping.", logPrio::LOG_ERROR );
        state( stateCodes::ERROR );

        return -1;
    }

    if( m_actuator_mapping == nullptr )
    {
        log<text_log>( "DM initialization failed.  null pointer.", logPrio::LOG_ERROR );
        state( stateCodes::ERROR );

        return -1;
    }

    if( m_sdkWorkerEnabled )
    {
        m_sdkValues.assign( m_nbAct, 0 );
        m_stagedSat.assign( m_dmWidth * m_dmHeight, 0 );

        if( m_pipelineEnabled && m_outputShape.data() != nullptr && m_outputShape.rows() == m_dmWidth &&
            m_outputShape.cols() == m_dmHeight )
        {
            try
            {
                allocatePipelineJobs();
            }
            catch( const std::exception &error )
            {
                return log<software_error, -1>(
                    { std::string( "allocating pipeline command storage: " ) + error.what() } );
            }
        }

        { //mutex scope
            std::lock_guard<std::mutex> lock( m_commandMutex );
            m_sdkFault  = false;
            m_haveInput = false;
        }

        openAdmission( true );
    }

    log<text_log>( "BMC " + m_serialNumber + " initialized", logPrio::LOG_NOTICE );
    state( stateCodes::READY );

    return 0;
}

int bmcCtrl::zeroDM()
{
    if( !m_sdkWorkerEnabled )
    {
        return zeroDMImpl();
    }

    std::lock_guard<std::mutex> control( m_controlMutex );
    const bool                  reopen = closeAdmission();
    const int                   result = zeroDMImpl();
    openAdmission( reopen && result == 0 );
    return result;
}

int bmcCtrl::zeroDMImpl()
{
    if( !m_dmopen )
    {
        log<text_log>( "DM not initialized (NULL pointer)", logPrio::LOG_ERROR );

        return -1;
    }

    if( m_nbAct == 0 )
    {
        log<text_log>( "DM not initialized (number of actuators)", logPrio::LOG_ERROR );

        return -1;
    }

    double *dminputs = reinterpret_cast<double *>( calloc( m_nbAct, sizeof( double ) ) );

    if( dminputs == nullptr )
    {
        return log<software_error, -1>( { "allocating BMC zero buffer" } );
    }

    /* Send the all 0 command to the DM */
    SdkRequest zero;
    zero.operation = SdkOperation::set;
    zero.values    = dminputs;
    BMCRC ret      = runSdk( zero ).result;

    /* Release memory */
    free( dminputs );

    if( ret != NO_ERR )
    {
        const char *err;
        err = BMCErrorString( ret );
        log<text_log>( std::string( "Error zeroing DM: " ) + err, logPrio::LOG_ERROR );

        return -1;
    }

    log<text_log>( "DM zeroed" );

    return 0;
}

int bmcCtrl::commandDM( void *curr_src )
{
    // This is based on Kyle Van Gorkoms original sendCommand function.

    /*This loop performs the following steps:
      1) converts from float to double
      2) convert to volume-normalized displacement
      3) convert to squared fractional voltage clamped from 0 to 1.
    */

    dmTimingMark( MagAOX::dmTiming::Marker::actuatorStart );

    double gain_scale     = m_volume_factor / m_act_gain;
    float  inv_gain_scale = 1.0 / gain_scale;

    m_outputShape.setWrite( 1 );

    for( uint32_t idx = 0; idx < m_nbAct; ++idx )
    {
        int address = m_actuator_mapping[idx];

        if( address < 0 )
        {
            m_dminputs[idx] = 0.; // addressable but ignored actuators set to 0
        }
        else
        {
            realT input = reinterpret_cast<realT *>( curr_src )[address];

            m_dminputs[idx] = input * gain_scale;

            if( m_dminputs[idx] > 1 )
            {
                m_dminputs[idx]        = 1;
                m_outputShape[address] = inv_gain_scale;
            }
            else if( m_dminputs[idx] < 0 )
            {
                m_dminputs[idx]        = 0;
                m_outputShape[address] = 0;
            }
            else
            {
                m_dminputs[idx]        = sqrt( m_dminputs[idx] );
                m_outputShape[address] = input;
            }
        }
    }

    m_outputShape.post();

    dmTimingMark( MagAOX::dmTiming::Marker::actuatorEnd );

    if( m_sdkWorkerEnabled )
    {
        if( m_sdkValues.size() != m_nbAct || m_stagedSat.size() != size_t( m_instSatMap.size() ) )
        {
            return -1;
        }

        std::copy_n( m_dminputs, m_nbAct, m_sdkValues.data() );
        SdkRequest request;
        request.operation     = SdkOperation::set;
        request.values        = m_sdkValues.data();
        m_sdkDispatchNs       = dev::dmTiming::monotonicNowNs();
        request.normalCommand = true;
        request.dispatchNs    = m_sdkDispatchNs;
        m_commandJob          = dispatchSdk( request );

        if( m_commandJob == 0 )
        {
            return -1;
        }

        dmTimingMark( MagAOX::dmTiming::Marker::satStart );
        m_stagedSatCount = 0;

        for( uint32_t idx = 0; idx < m_nbAct; ++idx )
        {
            const int address = m_actuator_mapping[idx];

            if( address < 0 )
            {
                continue;
            }

            const bool saturated = m_dminputs[idx] >= 1 || m_dminputs[idx] <= 0;
            m_stagedSat[address] = saturated;
            m_stagedSatCount += saturated;
        }

        dmTimingMark( MagAOX::dmTiming::Marker::satEnd );
        return 0;
    }

    /* Send the command to the DM */
    dmTimingMark( MagAOX::dmTiming::Marker::sdkStart );
    BMCRC ret = BMCSetArray( &m_dm, m_dminputs, NULL );

    dmTimingMark( MagAOX::dmTiming::Marker::sdkEnd );

    dmBackendResult( ret == NO_ERR ? MagAOX::dmTiming::Outcome::success : MagAOX::dmTiming::Outcome::failed, ret );

    /* Return immediately upon error, logging the error
    message first and then return the failure code. */
    if( ret != NO_ERR )
    {
        const char *err;
        err = BMCErrorString( ret );
        log<text_log>( std::string( "DM command failed: " ) + err, logPrio::LOG_ERROR );

        return -1;
    }

    dmTimingMark( MagAOX::dmTiming::Marker::satStart );

    /* Now update the instantaneous sat map */
    for( uint32_t idx = 0; idx < m_nbAct; ++idx )
    {
        int address = m_actuator_mapping[idx];

        if( address == -1 )
        {
            continue;
        }
        else if( m_dminputs[idx] >= 1 || m_dminputs[idx] <= 0 )
        {
            ++m_nsat;
            m_instSatMap.data()[address] = 1;
        }
        else
        {
            m_instSatMap.data()[address] = 0;
        }
    }

    dmTimingMark( MagAOX::dmTiming::Marker::satEnd );

    return ret;
}

int bmcCtrl::releaseDM()
{
    std::unique_lock<std::mutex> control( m_controlMutex, std::defer_lock );

    if( m_sdkWorkerEnabled )
    {
        control.lock();
        closeAdmission();
    }

    // Safe DM shutdown on interrupt

    if( !m_dmopen )
    {
        return 0;
    }

    state( stateCodes::NOTHOMED );

    if( !m_sdkWorkerEnabled )
    {
        if( !shutdown() )
        {
            pthread_kill( m_smThread.native_handle(), SIGUSR1 );
        }

        sleep( 1 );
    }

    if( zeroDMImpl() < 0 )
    {
        log<text_log>( "DM release failed.  Error zeroing DM.", logPrio::LOG_ERROR );
        state( stateCodes::ERROR );

        return -1;
    }

    // Zero all actuators (this is probably redundant after zeroing the DM above)
    BMCRC      ret = NO_ERR;
    SdkRequest clear;
    clear.operation = SdkOperation::clear;
    ret             = runSdk( clear ).result;

    if( ret != NO_ERR )
    {
        const char *err;
        err = BMCErrorString( ret );
        log<text_log>( std::string( "DM reset failed: " ) + err, logPrio::LOG_ERROR );
        state( stateCodes::ERROR );

        return -1;
    }

    // Close BMC connection
    SdkRequest close;
    close.operation = SdkOperation::close;
    ret             = runSdk( close ).result;

    if( ret == NO_ERR )
    {
        m_dmopen = false;
    }

    if( ret != NO_ERR )
    {
        const char *err;
        err = BMCErrorString( ret );
        log<text_log>( std::string( "DM release failed: " ) + err, logPrio::LOG_ERROR );
        state( stateCodes::ERROR );

        return -1;
    }

    m_dm = {};
    ++m_sdkGeneration;

    log<text_log>( "BMC " + m_serialNumber + " reset and released", logPrio::LOG_NOTICE );

    return 0;
}

/* Read in a configuration file with user-calibrated
values to determine the conversion from physical to
fractional stroke as well as the volume displaced by
the influence function. */
int bmcCtrl::parse_calibration_file()
{
    FILE   *fp;
    char   *line = NULL;
    size_t  len  = 0;
    ssize_t read;
    double *calibvals;

    std::string calibpath = m_calibPath + "/" + "bmc_2k_userconfig.txt";

    // open file
    fp = fopen( calibpath.c_str(), "r" );

    if( fp == NULL )
    {
        log<text_log>( "Could not read configuration file at " + calibpath, logPrio::LOG_ERROR );

        return -1;
    }

    calibvals = reinterpret_cast<double *>( malloc( 2 * sizeof( double ) ) );
    int idx   = 0;

    while( ( read = getline( &line, &len, fp ) ) != -1 )
    {
        // grab first value from each line
        calibvals[idx] = strtod( line, NULL );
        idx++;
    }

    fclose( fp );

    // assign stroke and volume factors
    m_act_gain      = calibvals[0];
    m_volume_factor = calibvals[1];

    free( calibvals );

    log<text_log>( "BMC " + m_serialNumber + ": Using stroke and volume calibration from " + calibpath );
    std::cerr << m_act_gain << " " << m_volume_factor << "\n";

    return 0;
}

int bmcCtrl::get_actuator_mapping() // const char * serial, int nbAct, int * actuator_mapping)
{
    /* This function closely follows the CFITSIO imstat
    example */

    fitsfile *fptr;       /* FITS file pointer */
    int       status = 0; /* CFITSIO status value MUST be initialized to zero! */

    // get file path to actuator map
    std::string calibpath = m_calibPath + "/" + "bmc_2k_actuator_mapping.fits";

    if( !fits_open_image( &fptr, calibpath.c_str(), READONLY, &status ) )
    {
        int  hdutype, naxis;
        long naxes[2];

        if( fits_get_hdu_type( fptr, &hdutype, &status ) || hdutype != IMAGE_HDU )
        {
            printf( "Error: this program only works on images, not tables\n" );

            return ( 1 );
        }

        fits_get_img_dim( fptr, &naxis, &status );
        fits_get_img_size( fptr, 2, naxes, &status );

        if( status || naxis != 2 )
        {
            printf( "Error: NAXIS = %d.  Only 2-D images are supported.\n", naxis );

            return ( 1 );
        }

        int *pix = reinterpret_cast<int *>( malloc( naxes[0] * sizeof( int ) ) ); /* memory for 1 row */

        if( pix == NULL )
        {
            printf( "Memory allocation error\n" );

            return ( 1 );
        }

        long fpixel[2];
        // totpix = naxes[0] * naxes[1];
        fpixel[0] = 1; /* read starting with first pixel in each row */

        /* process image one row at a time; increment row # in each loop */
        int ij = 0; /* actuator mapping index */
        for( fpixel[1] = 1; fpixel[1] <= naxes[1]; fpixel[1]++ )
        {
            /* give starting pixel coordinate and number of pixels to read */
            if( fits_read_pix( fptr, TINT, fpixel, naxes[0], 0, pix, 0, &status ) )
            {
                break; /* jump out of loop on error */
            }

            // get indices of active actuators in order
            for( int ii = 0; ii < naxes[0]; ii++ )
            {
                if( pix[ii] > 0 )
                {
                    m_actuator_mapping[pix[ii] - 1] = ij;
                }

                ij++;
            }
        }

        fits_close_file( fptr, &status );

        free( pix );
    }

    if( status )
    {
        fits_report_error( stderr, status ); /* print any error message */
    }

    log<text_log>( "BMC " + m_serialNumber + ": Using actuator mapping from " + calibpath );

    return 0;
}

int bmcCtrl::startSdkWorker()
{
    if( m_sdkThread.joinable() )
    {
        return 0;
    }

    m_sdkStop       = false;
    m_sdkConfigured = false;
    m_sdkPending    = false;
    m_sdkReady      = false;
    m_sdkTimedOut   = false;

    if( threadStart( m_sdkThread,
                     m_sdkThreadInit,
                     m_sdkThreadID,
                     m_sdkThreadProp,
                     m_sdkThreadPrio,
                     m_sdkCpuset,
                     "sdk",
                     this,
                     sdkThreadStart ) < 0 )
    {
        stopSdkWorker();
        return -1;
    }

    if( verifySdkThread() < 0 )
    {
        stopSdkWorker();
        return -1;
    }

    { //mutex scope
        std::lock_guard<std::mutex> lock( m_sdkMutex );
        m_sdkConfigured = true;
    }

    m_sdkWake.notify_one();
    return 0;
}

int bmcCtrl::verifySdkThread()
{
    return verifyWorkerThread(
        m_sdkThread, m_sdkThreadPrio, m_sdkCpuset, m_sdkPolicy, m_sdkActualPrio, m_sdkAllowedCpus );
}

int bmcCtrl::verifyWorkerThread( std::thread       &thread,
                                 int                priority,
                                 const std::string &cpuset,
                                 int               &actualPolicy,
                                 int               &actualPriority,
                                 int               &allowedCpus )
{
    sched_param scheduling{};
    int         policy = -1;
    cpu_set_t   actual;
    CPU_ZERO( &actual );

    if( pthread_getschedparam( thread.native_handle(), &policy, &scheduling ) != 0 ||
        pthread_getaffinity_np( thread.native_handle(), sizeof( actual ), &actual ) != 0 ||
        policy != ( priority == 0 ? SCHED_OTHER : MAGAOX_RT_SCHED_POLICY ) || scheduling.sched_priority != priority ||
        CPU_COUNT( &actual ) == 0 )
    {
        return log<software_error, -1>( { "SDK worker actual scheduling/affinity does not match configuration" } );
    }

    if( !cpuset.empty() )
    {
        std::ifstream file( m_cpusetPath + "/" + cpuset + "/cpuset.cpus" );
        std::string   list;
        std::getline( file, list );
        cpu_set_t expected;
        CPU_ZERO( &expected );

        try
        {
            std::istringstream ranges( list );
            std::string        range;

            while( std::getline( ranges, range, ',' ) )
            {
                const size_t dash  = range.find( '-' );
                const auto   first = std::stoul( range.substr( 0, dash ) );
                const auto   last  = dash == std::string::npos ? first : std::stoul( range.substr( dash + 1 ) );

                if( first > last || last >= CPU_SETSIZE )
                {
                    throw std::runtime_error( "unsupported SDK CPU range" );
                }

                for( auto cpu = first; cpu <= last; ++cpu )
                {
                    CPU_SET( cpu, &expected );
                }
            }
        }
        catch( ... )
        {
            return log<software_error, -1>( { "SDK cpuset CPU list could not be verified" } );
        }

        if( CPU_COUNT( &expected ) == 0 || !CPU_EQUAL( &actual, &expected ) )
        {
            return log<software_error, -1>( { "SDK worker actual affinity does not match requested cpuset" } );
        }
    }

    actualPolicy   = policy;
    actualPriority = scheduling.sched_priority;
    allowedCpus    = CPU_COUNT( &actual );
    return 0;
}

void bmcCtrl::stopSdkWorker() noexcept
{
    std::lock_guard<std::mutex> control( m_controlMutex );
    closeAdmission();
    stopPipelineWorker();

    if( !m_sdkThread.joinable() )
    {
        m_pipelineResult.close();
        return;
    }

    { //mutex scope
        std::unique_lock<std::mutex> lock( m_sdkMutex );
        m_sdkDone.wait( lock,
                        [this]()
                        {
                            return !m_sdkPending || m_sdkReady;
                        } );
        m_sdkStop       = true;
        m_sdkConfigured = false;
    }

    m_sdkWake.notify_one();
    m_sdkThread.join();
    m_pipelineResult.close();
}

void bmcCtrl::sdkThreadStart( bmcCtrl *app )
{
    app->sdkThreadExec();
}

bmcCtrl::SdkResult bmcCtrl::performSdk( const SdkRequest &request ) noexcept
{
    SdkResult result;
    result.sequence   = request.sequence;
    result.generation = request.generation;
    result.startNs    = dev::dmTiming::monotonicNowNs();

    try
    {
        switch( request.operation )
        {
        case SdkOperation::open:
            result.result = BMCOpen( &m_dm, request.serial );

            if( result.result == NO_ERR )
            {
                result.actuatorCount = m_dm.ActCount;
                result.driverType    = m_dm.Driver_Type;
                result.amplifierType = m_dm.HVA_Type;
            }
            break;
        case SdkOperation::loadMap:
            result.result = BMCLoadMap( &m_dm, nullptr, request.map );
            break;
        case SdkOperation::set:
            result.result = BMCSetArray( &m_dm, request.values, nullptr );
            break;
        case SdkOperation::clear:
            result.result = BMCClearArray( &m_dm );
            break;
        case SdkOperation::close:
            result.result = BMCClose( &m_dm );
            break;
        }
    }
    catch( ... )
    {
        result.result    = static_cast<BMCRC>( -1 );
        result.exception = true;
    }

    result.endNs = dev::dmTiming::monotonicNowNs();
    return result;
}

uint64_t bmcCtrl::dispatchSdk( SdkRequest request )
{
    { //mutex scope
        std::lock_guard<std::mutex> lock( m_sdkMutex );

        if( !m_sdkThread.joinable() || !m_sdkConfigured || m_sdkStop || m_sdkPending || m_pipelineInFlight ||
            m_sdkSequence == UINT64_MAX )
        {
            return 0;
        }

        request.sequence   = ++m_sdkSequence;
        request.generation = m_sdkGeneration;
        m_sdkRequest       = request;
        m_sdkPending       = true;
        m_sdkReady         = false;
        m_sdkTimedOut      = false;
    }

    m_sdkWake.notify_one();
    return request.sequence;
}

bmcCtrl::SdkResult bmcCtrl::waitSdk( uint64_t sequence )
{
    std::unique_lock<std::mutex> lock( m_sdkMutex );
    m_sdkDone.wait( lock,
                    [this, sequence]()
                    {
                        return m_sdkReady && m_sdkResult.sequence == sequence &&
                               m_sdkResult.generation == m_sdkRequest.generation;
                    } );
    auto       result  = m_sdkResult;
    const bool elapsed = m_sdkRequest.normalCommand && m_sdkRequest.dispatchNs > 0 &&
                         result.endNs >= m_sdkRequest.dispatchNs &&
                         ( result.endNs - m_sdkRequest.dispatchNs ) * 1e-9 >= m_sdkTimeoutSec;
    result.newTimeout = elapsed && !m_sdkTimedOut;
    result.timedOut   = elapsed || m_sdkTimedOut;
    m_sdkTimedOut     = result.timedOut;
    m_sdkPending      = false;
    m_sdkReady        = false;
    return result;
}

bmcCtrl::SdkResult bmcCtrl::runSdk( SdkRequest request )
{
    if( !m_sdkWorkerEnabled )
    {
        return performSdk( request );
    }

    const auto sequence = dispatchSdk( request );

    if( sequence == 0 )
    {
        SdkResult failed;
        failed.result = static_cast<BMCRC>( -1 );
        return failed;
    }

    const auto result = waitSdk( sequence );

    if( result.result != NO_ERR )
    {
        std::lock_guard<std::mutex> lock( m_commandMutex );
        m_sdkFault        = true;
        m_admissionClosed = true;
    }

    return result;
}

bool bmcCtrl::closeAdmission()
{
    std::unique_lock<std::mutex> lock( m_commandMutex );
    const bool                   reopen = !m_admissionClosed;
    m_admissionClosed                   = true;
    m_commandRetired.notify_all();
    m_commandRetired.wait( lock,
                           [this]()
                           {
                               return !m_commandActive;
                           } );
    return reopen;
}

void bmcCtrl::openAdmission( bool reopen )
{
    std::lock_guard<std::mutex> lock( m_commandMutex );
    m_admissionClosed = !reopen || !m_dmopen || m_sdkFault;
}

bool bmcCtrl::admitCommand()
{
    if( !m_sdkWorkerEnabled )
    {
        return true;
    }

    std::lock_guard<std::mutex> lock( m_commandMutex );
    const auto                 *metadata = m_imageStream.md;

    if( m_admissionClosed || m_sdkFault || m_commandActive || !m_dmopen || !m_completionReceipt ||
        metadata == nullptr || __atomic_load_n( &metadata->write, __ATOMIC_ACQUIRE ) != 0 )
    {
        ++m_rejectedInputs;
        return false;
    }

    const uint64_t inode   = __atomic_load_n( &metadata->inode, __ATOMIC_ACQUIRE );
    const uint64_t counter = __atomic_load_n( &metadata->cnt0, __ATOMIC_ACQUIRE );

    if( inode == 0 || __atomic_load_n( &metadata->write, __ATOMIC_ACQUIRE ) != 0 ||
        inode != __atomic_load_n( &metadata->inode, __ATOMIC_ACQUIRE ) ||
        counter != __atomic_load_n( &metadata->cnt0, __ATOMIC_ACQUIRE ) )
    {
        ++m_rejectedInputs;
        return false;
    }

    if( counter == 0 || ( m_haveInput && inode == m_inputInode && counter <= m_inputCounter ) )
    {
        if( m_haveInput && inode == m_inputInode && counter < m_inputCounter )
        {
            m_haveInput = false;
        }

        ++m_rejectedInputs;
        return false;
    }

    if( m_haveInput && inode == m_inputInode && counter > m_inputCounter + 1 )
    {
        m_coalescedInputs += counter - m_inputCounter - 1;
    }

    m_inputInode    = inode;
    m_inputCounter  = counter;
    m_haveInput     = true;
    m_commandActive = true;
    return true;
}

bool bmcCtrl::deferredCommand() const
{
    return m_sdkWorkerEnabled;
}

int bmcCtrl::finalizeCommand( int processingResult ) noexcept
{
    if( !m_sdkWorkerEnabled )
    {
        return processingResult;
    }

    if( m_commandJob == 0 )
    {
        if( processingResult < 0 )
        {
            std::lock_guard<std::mutex> lock( m_commandMutex );
            m_sdkFault        = true;
            m_admissionClosed = true;
        }

        return processingResult;
    }

    const auto waitStart = dev::dmTiming::monotonicNowNs();
    const auto result    = waitSdk( m_commandJob );
    const auto waitEnd   = dev::dmTiming::monotonicNowNs();
    m_commandJob         = 0;
    const auto outcome =
        result.result == NO_ERR ? MagAOX::dmTiming::Outcome::success : MagAOX::dmTiming::Outcome::failed;
    m_dmTiming.importSdk( result.startNs, result.endNs, outcome, m_sdkDispatchNs, waitStart, waitEnd );
    dmBackendResult( outcome, result.exception ? MagAOX::dmCompletion::unavailableResult : int64_t( result.result ) );

    { //mutex scope
        std::lock_guard<std::mutex> lock( m_commandMutex );
        ++m_completedCommands;
        m_sdkActiveAgeSec = 0;
        m_sdkTimeouts += result.newTimeout;

        if( result.result != NO_ERR || processingResult < 0 || result.timedOut )
        {
            m_sdkFault        = true;
            m_admissionClosed = true;
            m_sdkFailures += result.result != NO_ERR;
        }
    }

    if( result.result != NO_ERR )
    {
        log<text_log>( result.exception ? "DM SDK threw during worker submission"
                                        : std::string( "DM command failed: " ) + BMCErrorString( result.result ),
                       logPrio::LOG_ERROR );
        return -1;
    }

    if( result.timedOut )
    {
        if( result.newTimeout )
        {
            log<text_log>( "DM SDK command exceeded the diagnostic deadline; actual result was drained",
                           logPrio::LOG_ERROR );
        }

        return -1;
    }

    if( processingResult < 0 )
    {
        return processingResult;
    }

    std::copy( m_stagedSat.begin(), m_stagedSat.end(), m_instSatMap.data() );
    m_nsat.fetch_add( m_stagedSatCount );
    return 0;
}

void bmcCtrl::retireCommand() noexcept
{
    if( m_pipelineEnabled || !m_sdkWorkerEnabled )
    {
        return;
    }

    { //mutex scope
        std::lock_guard<std::mutex> lock( m_commandMutex );

        if( m_completionOutcome == MagAOX::dmTiming::Outcome::failed ||
            m_completionOutcome == MagAOX::dmTiming::Outcome::partial ||
            m_completionOutcome == MagAOX::dmTiming::Outcome::rejected )
        {
            m_sdkFault        = true;
            m_admissionClosed = true;
        }

        m_commandActive = false;
    }

    m_commandRetired.notify_all();
}

void bmcCtrl::checkSdkTimeout( int64_t nowNs )
{
    if( !m_sdkWorkerEnabled )
    {
        return;
    }

    bool expired = false;

    { //mutex scope
        // Keep this job's identity protected until its fault/count claim commits. Never reverse this lock order.
        std::lock_guard<std::mutex> sdk( m_sdkMutex );
        double                      ageSec = 0;

        if( m_pipelineEnabled && m_pipelineSdkActive < pipelineCapacity )
        {
            auto &job = m_pipelineJobs[m_pipelineSdkActive];

            if( job.result.endNs == 0 && job.command.dispatchNs > 0 && nowNs >= job.command.dispatchNs )
            {
                ageSec  = ( nowNs - job.command.dispatchNs ) * 1e-9;
                expired = ageSec >= m_sdkTimeoutSec && !job.timedOut;
            }

            if( expired )
            {
                job.timedOut = true;

                { //mutex scope
                    std::lock_guard<std::mutex> command( m_commandMutex );
                    ++m_sdkTimeouts;
                }

                faultPipeline();
            }

            { //mutex scope
                std::lock_guard<std::mutex> command( m_commandMutex );
                m_sdkActiveAgeSec = ageSec;
            }
        }
        else
        {
            if( m_sdkPending && !m_sdkReady && m_sdkRequest.normalCommand && m_sdkRequest.dispatchNs > 0 &&
                nowNs >= m_sdkRequest.dispatchNs )
            {
                ageSec  = ( nowNs - m_sdkRequest.dispatchNs ) * 1e-9;
                expired = ageSec >= m_sdkTimeoutSec && !m_sdkTimedOut;
            }

            std::lock_guard<std::mutex> command( m_commandMutex );
            m_sdkActiveAgeSec = ageSec;

            if( expired )
            {
                m_sdkTimedOut = true;
                ++m_sdkTimeouts;
                m_sdkFault        = true;
                m_admissionClosed = true;
            }
        }
    }

    if( expired )
    {
        log<text_log>( "DM SDK command exceeded the diagnostic deadline; still awaiting its matching SDK result",
                       logPrio::LOG_ERROR );
    }
}

void bmcCtrl::updateSdkWorkerINDI()
{
    if( !m_sdkWorkerEnabled )
    {
        return;
    }

    std::array<double, 18> pipeline{};

    if( m_pipelineEnabled )
    {
        std::lock_guard<std::mutex> lock( m_sdkMutex );
        size_t                      queued  = 0;
        bool                        posting = false;

        for( const auto &job : m_pipelineJobs )
        {
            queued += job.queued;
            posting = posting || job.posting;
        }

        pipeline = { 1,
                     double( pipelineCapacity ),
                     double( m_pipelineInFlight ),
                     double( queued ),
                     m_pipelineSdkActive < pipelineCapacity ? 1.0 : 0.0,
                     posting ? 1.0 : 0.0,
                     double( m_pipelineAdmitted ),
                     double( m_pipelineRetired ),
                     double( m_pipelineOverloaded ),
                     double( m_pipelineCancelled ),
                     double( m_pipelinePostFailures ),
                     double( m_pipelineSuperseded ),
                     double( m_pipelineVectorRejected ),
                     double( m_pipelineReaderUnqualified ),
                     double( m_pipelinePreparationFailures ),
                     double( m_pipelinePolicy ),
                     double( m_pipelineActualPrio ),
                     double( m_pipelineAllowedCpus ) };
    }

    bool     active, closed, fault;
    uint64_t completed, failures, rejected, coalesced, timeouts;
    double   ageSec;

    { //mutex scope
        std::lock_guard<std::mutex> lock( m_commandMutex );
        active    = m_commandActive;
        closed    = m_admissionClosed;
        fault     = m_sdkFault;
        completed = m_completedCommands;
        failures  = m_sdkFailures;
        rejected  = m_rejectedInputs;
        coalesced = m_coalescedInputs;
        timeouts  = m_sdkTimeouts;
        ageSec    = m_sdkActiveAgeSec;
    }

    updateIfChanged<double>( m_indiP_sdkWorker,
                             { "active",
                               "admissionClosed",
                               "fault",
                               "completed",
                               "failures",
                               "rejected",
                               "coalesced",
                               "policy",
                               "priority",
                               "allowedCpus",
                               "activeAgeSec",
                               "timeouts" },
                             { double( active ),
                               double( closed ),
                               double( fault ),
                               double( completed ),
                               double( failures ),
                               double( rejected ),
                               double( coalesced ),
                               double( m_sdkPolicy ),
                               double( m_sdkActualPrio ),
                               double( m_sdkAllowedCpus ),
                               ageSec,
                               double( timeouts ) },
                             fault ? pcf::IndiProperty::Alert : pcf::IndiProperty::Ok );

    if( m_pipelineEnabled )
    {
        updateIfChanged<double>( m_indiP_pipeline,
                                 { "enabled",
                                   "capacity",
                                   "inFlight",
                                   "queued",
                                   "sdkActive",
                                   "postActive",
                                   "admitted",
                                   "retired",
                                   "overloaded",
                                   "cancelled",
                                   "postFailures",
                                   "superseded",
                                   "vectorRejected",
                                   "readerUnqualified",
                                   "preparationFailures",
                                   "policy",
                                   "priority",
                                   "allowedCpus" },
                                 std::vector<double>( pipeline.begin(), pipeline.end() ),
                                 fault ? pcf::IndiProperty::Alert : pcf::IndiProperty::Ok );
    }
}

int bmcCtrl::checkRecordTimes()
{
    return telemeterT::checkRecordTimes( telem_dmtimings() );
}

int bmcCtrl::recordTelem( const telem_dmtimings * )
{
    return dmT::recordDMTimings( true );
}

} // namespace app
} // namespace MagAOX

#include "bmcPipeline.hpp"

#endif // bmcCtrl_hpp
