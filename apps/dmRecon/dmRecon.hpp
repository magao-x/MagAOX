/** \file dmRecon.hpp
 * \brief Reconstruct applied DM modes and diagnose publication timing relative to loop triggers.
 * \ingroup dmRecon_files
 */
#ifndef dmRecon_hpp
#define dmRecon_hpp

#include <atomic>
#include <limits>
#include <memory>
#include <mutex>
#include <vector>

#include <mx/improc/eigenCube.hpp>
#include <mx/improc/eigenImage.hpp>

#include "../../libMagAOX/libMagAOX.hpp"
#include "../../magaox_git_version.h"
#include "dmReconMath.hpp"
#include "dmReconTiming.hpp"

#ifdef MXLIB_CUDA
    #include <mx/math/cuda/cudaPtr.hpp>
    #include <mx/math/cuda/cublasHandle.hpp>
    #include <mx/math/cuda/templateCublas.hpp>
#endif

namespace MagAOX
{
namespace app
{
/** \defgroup dmRecon DM Shape Reconstructor
 * \brief Reconstruct modal state from the applied DM shape.
 * \ingroup apps
 */
/** \defgroup dmRecon_files DM Shape Reconstructor Files
 * \ingroup dmRecon
 */

/// Monitor tag for the complete calibration cube.
struct dmModesShmimT
{
    /// Configuration section for the modes monitor.
    static std::string configSection()
    {
        return "dmModes";
    }

    /// INDI prefix for the modes monitor.
    static std::string indiPrefix()
    {
        return "dmModes";
    }
};
/// Monitor tag for the active-actuator mask.
struct dmMaskShmimT
{
    /// Configuration section for the mask monitor.
    static std::string configSection()
    {
        return "dmMask";
    }

    /// INDI prefix for the mask monitor.
    static std::string indiPrefix()
    {
        return "dmMask";
    }
};
/// Monitor tag for the applied DM delta.
struct dmCommandShmimT
{
    /// Configuration section for the command monitor.
    static std::string configSection()
    {
        return "dmCommand";
    }

    /// INDI prefix for the command monitor.
    static std::string indiPrefix()
    {
        return "dmCommand";
    }
};
/// Metadata-only monitor of the stream which triggers mfilt.
struct dmLoopTriggerShmimT
{
    /// Configuration section for loop-trigger diagnostics.
    static std::string configSection()
    {
        return "loopTrigger";
    }

    /// INDI prefix for loop-trigger diagnostics.
    static std::string indiPrefix()
    {
        return "loopTrigger";
    }
};

/// Reconstruct privately in one worker; observe, but never gate, the AO loop.
/** \ingroup dmRecon */
class dmRecon : public MagAOXApp<true>,
                public dev::shmimMonitor<dmRecon, dmModesShmimT>,
                public dev::shmimMonitor<dmRecon, dmMaskShmimT>,
                public dev::shmimMonitor<dmRecon, dmCommandShmimT>,
                public dev::shmimMonitor<dmRecon, dmLoopTriggerShmimT>,
                public dev::frameGrabber<dmRecon>,
                public dev::telemeter<dmRecon>
{
    friend class dmRecon_test;
    friend class dev::shmimMonitor<dmRecon, dmModesShmimT>;
    typedef dev::shmimMonitor<dmRecon, dmModesShmimT> dmModesSMT;
    friend class dev::shmimMonitor<dmRecon, dmMaskShmimT>;
    typedef dev::shmimMonitor<dmRecon, dmMaskShmimT> dmMaskSMT;
    friend class dev::shmimMonitor<dmRecon, dmCommandShmimT>;
    typedef dev::shmimMonitor<dmRecon, dmCommandShmimT> dmCommandSMT;
    friend class dev::shmimMonitor<dmRecon, dmLoopTriggerShmimT>;
    typedef dev::shmimMonitor<dmRecon, dmLoopTriggerShmimT> loopTriggerSMT;
    friend class dev::frameGrabber<dmRecon>;
    typedef dev::frameGrabber<dmRecon> frameGrabberT;
    friend class dev::telemeter<dmRecon>;
    typedef dev::telemeter<dmRecon> telemeterT;
    typedef float                   realT;                            ///< Reconstruction scalar type.
    static constexpr bool           c_frameGrabber_flippable = false; ///< Modal vectors cannot be flipped.

  protected:
    /** \name Configurable Parameters - Data
     * @{ */
    int         m_loopNumber{ 1 };            ///< Default cacao loop number.
    std::string m_respMPath;                  ///< Optional response matrix converting input modes to DM pixels.
    std::string m_fpsSource{ "camwfs" };      ///< Subscribed FPS source; silence does not invalidate its value.
    int         m_numModes{ 0 };              ///< Number of mode planes, or zero for the complete cube.
    int         m_inverseNumModes{ 0 };       ///< Number of singular modes retained, or zero for all.
    uint16_t    m_gpuIndex{ 0 };              ///< Selected CUDA device.
    bool        m_useGPU{ false };            ///< Worker-owned GPU selection after startup.
    std::string m_triggerClockName{ "isio" }; ///< Clock domain of the monitored trigger's writetime.
    ///@}

    /// A private image and the identities captured alongside its pixels.
    struct Frame
    {
        std::vector<float>     pixels; ///< Owned storage, recycled by swapping the latest-input mailbox.
        uint32_t               width{ 0 }, height{ 0 }, depth{ 0 }; ///< Validated image dimensions.
        uint64_t               counter{ 0 }, inode{ 0 };            ///< Source publication identity.
        uint64_t               generation{ 0 };                     ///< Calibration generation at command capture.
        uint64_t               inputGeneration{ 0 };                ///< Command-stream attachment generation.
        timespec               stamp{}, received{}; ///< Source write time and local realtime acquisition time.
        dmReconTiming::Trigger trigger;             ///< Most recent coherent loop trigger observed at capture.
        int64_t clockLow{ 0 }, clockHigh{ 0 };      ///< Bracket on trigger-clock minus monotonic-clock offset.
        bool    clockValid{ false };                ///< The local clock bracket was captured successfully.
    };

    /// Outcome of the most recent calibration snapshot attempt.
    enum class SnapshotStatus
    {
        unavailable,     ///< No attached pixels or no snapshot attempted yet.
        invalidFormat,   ///< Unsupported datatype, axis count or memory location.
        invalidShape,    ///< Empty, excessive or inconsistent image dimensions.
        invalidCapacity, ///< Selected pixels exceed the declared payload.
        busy,            ///< The producer has not cleared its write flag.
        changed,         ///< Publication identity changed during the copy.
        clockError,      ///< Acquisition timestamp could not be sampled.
        ready            ///< A complete private snapshot was accepted.
    };

    std::mutex m_snapshotMutex;    ///< Protects pending snapshots, their generations, and timing observations.
    std::mutex m_publicationMutex; ///< Serializes final publication with disable and calibration replacement.
    std::unique_lock<std::mutex> m_publicationLock; ///< FG-owned lock from final validation through post-publication.
    Frame m_pendingModes, m_pendingMask, m_pendingCommand; ///< Snapshots shared only under m_snapshotMutex.
    Frame m_modesStage, m_maskStage, m_commandStage;       ///< Each monitor's private reusable copy destination.
    bool  m_modesRetry{ false }, m_maskRetry{ false }; ///< Owning calibration monitor must retry after reattachment.
    Frame m_currentCommand; ///< Input frozen for the FG worker's current calculation/publication.
    bool  m_modesValid{ false }, m_maskValid{ false },
        m_commandPending{ false };                                 ///< Mailbox validity under the snapshot lock.
    uint64_t m_calibrationGeneration{ 0 }, m_inputGeneration{ 0 }; ///< Snapshot generations under the snapshot lock.
    uint64_t m_commandCounter{ 0 }, m_commandInode{ 0 }; ///< Last accepted input identity for duplicate suppression.
    bool     m_haveCommand{ false };                     ///< Input identity is valid for the current attachment.
    uint64_t m_workerGeneration{ 0 };                    ///< Calibration attempted by the FG worker.
    bool     m_calibrationUsable{ false };               ///< Worker buffers contain a complete validated calibration.
    uint32_t m_commandWidth{ 0 }, m_commandHeight{ 0 };  ///< Worker-owned expected command shape.
    mx::improc::eigenImage<float> m_respM;   ///< Immutable optional response matrix loaded before threads start.
    mx::improc::eigenImage<float> m_PInv;    ///< Worker-owned, finite pseudo-inverse.
    std::vector<size_t>           m_maskIDX; ///< Worker-owned column-major active pixel indices.
    mx::improc::eigenImage<float> m_command, m_modevals; ///< Worker-owned private calculation buffers.
    std::atomic<float>            m_fps{ 0 };            ///< Latest subscribed FPS, shared with the worker.
    std::atomic<bool> m_writeDMf{ false }; ///< Operator enable; final acceptance is serialized by the publication lock.
    sem_t             m_smSemaphore{}; ///< Coalesced wake hints for the single pending input, never a queue of results.
    bool              m_semaphoreInitialized{ false }; ///< Owns the local semaphore after successful startup.
    std::string       m_monShmimName;                  ///< Name of the diagnostic reconstructed-value stream.
    mx::improc::milkImage<float> m_modevalMon, m_modeval, m_modevalDiff; ///< Worker-owned diagnostic mappings.
    bool                         m_diagnosticsReady{ false },
        m_differenceReady{ false }; ///< Diagnostic stream availability is independent of feedback.
    std::vector<float>    m_difference, m_diagnosticModes; ///< Private diagnostic input and difference product.
    std::string           m_outputPath;       ///< Immutable feedback backing-file path for app-thread reconnect checks.
    std::atomic<uint64_t> m_outputInode{ 0 }; ///< Worker-published output attachment identity.
    std::atomic<bool>     m_outputChanged{ false };  ///< Inhibits publication after a detected output replacement.
    clockid_t     m_triggerClock{ CLOCK_ISIO };      ///< Explicit clock used for publication-completion observations.
    dmReconTiming m_timing;                          ///< Diagnostic-only event pairing under the snapshot lock.
    dmReconTiming::Trigger m_latestTrigger;          ///< Latest coherent observed trigger, under the snapshot lock.
    uint64_t               m_triggerGeneration{ 0 }; ///< Trigger attachment/reset generation, under the snapshot lock.
    int64_t m_triggerClockLow{ 0 }, m_triggerClockHigh{ 0 }; ///< Last trigger observation's clock offset bracket.
    bool    m_triggerClockValid{ false }; ///< Trigger offset is qualified against the monotonic clock.
    std::atomic<uint64_t> m_beforeNext{ 0 }, m_late{ 0 }, m_unknown{ 0 }; ///< Cumulative timing classifications.
    std::atomic<uint64_t> m_rejected{ 0 }, m_coalesced{ 0 }; ///< Invalid/torn inputs and superseded pending inputs.
    uint64_t              m_reportedLate{ 0 }, m_reportedUnknown{ 0 }; ///< App-thread warning rate limiting.
    std::atomic<SnapshotStatus> m_modesSnapshot{ SnapshotStatus::unavailable },
        m_maskSnapshot{ SnapshotStatus::unavailable }; ///< Calibration-monitor outcomes, readable by the app thread.
    std::atomic<const char *> m_setupStage{ "waiting_for_calibration" }; ///< Worker stage; values are static literals.
    const char               *m_reportedSetupStage{ nullptr };           ///< Last stage logged by the app thread.
    SnapshotStatus            m_reportedModesSnapshot{ SnapshotStatus::unavailable },
        m_reportedMaskSnapshot{ SnapshotStatus::unavailable }; ///< Last calibration outcomes logged by the app thread.

#ifdef MXLIB_CUDA
    /// Allocations replaced as a whole on calibration changes, avoiding cudaPtr's resize/free defects.
    struct GPUData
    {
        mx::cuda::cudaPtr<float> inverse, command, modevals; ///< Owned by the FG worker on its selected device.
    };
    std::unique_ptr<GPUData> m_gpu;               ///< Worker-only allocation generation.
    mx::cuda::cublasHandle   m_cublas;            ///< Worker-only cuBLAS handle.
    bool                     m_gpuReady{ false }; ///< Device and handle initialized on the FG thread.
#endif

    /// Snapshot a float image once and reject an overlapping producer write.
    static bool snapshot( Frame          &frame /**< [out] Private destination. */,
                          const IMAGE    &source /**< [in] Attached source metadata and pixels. */,
                          const void     *pixels /**< [in] Start of the selected frame or complete cube. */,
                          uint32_t        depth /**< [in] Number of planes to copy. */,
                          SnapshotStatus *status = nullptr /**< [out] Optional snapshot outcome. */ );

    /// Describe a snapshot result for logs and INDI without inspecting a live mapping.
    static const char *snapshotStatusName( SnapshotStatus status /**< [in] Snapshot outcome. */ );

    /// Retry calibration snapshots briefly on their owning monitor thread, invalidating rejected data immediately.
    bool captureCalibration( Frame       &frame /**< [out] Private calibration destination. */,
                             const IMAGE &source /**< [in] Monitor-owned source mapping. */,
                             const void  *pixels /**< [in] Start of the selected calibration payload. */,
                             uint32_t     depth /**< [in] Number of planes to copy. */,
                             bool         modes /**< [in] True selects modes, false selects mask. */ );

    /// Require an exact float modal-vector schema before accepting a passive output mapping.
    static bool outputCompatible( const IMAGE_METADATA &metadata /**< [in] Attached output schema. */,
                                  size_t                modes /**< [in] Reconstructed coefficient count. */ );

    /// Sample the configured trigger clock between monotonic reads.
    bool sampleClock( timespec &stamp /**< [out] Selected-clock sample. */,
                      int64_t  &low /**< [out] Lower offset bound. */,
                      int64_t  &high /**< [out] Upper offset bound. */ );

    /// Invalidate a calibration source under publication/snapshot ownership.
    void invalidateCalibration( bool modes /**< [in] True selects modes, false selects mask. */ );

    /// Build private calibration and GPU buffers on the FG thread; return one while unavailable.
    int prepareCalibration();

    /// Take the latest command exactly once, with no semaphore operations.
    bool takeCommand();

    /// Reconstruct the frozen command, rejecting invalid generation, shape or values.
    int reconstruct();

    /// Accumulate timing results without logging on the real-time path.
    void timingUpdate( const dmReconTiming::Update &update /**< [in] Newly resolved diagnostics. */ );

    /// Publish optional diagnostics from private calculation storage.
    void publishDiagnostics();

    /// Start base workers with interface macros so appStartup can unwind any partial failure.
    int startWorkers();

  public:
    /// Initialize passive feedback publication defaults.
    dmRecon();

    /// Release the local semaphore after application workers have stopped.
    ~dmRecon() noexcept;
    /// Register app and base configuration.
    void setupConfig() override;
    /// Load and validate configuration without starting workers.
    int loadConfigImpl( mx::app::appConfigurator &configuration /**< [in] Configuration source. */ );

    /// Load app configuration and request shutdown on invalid settings.
    void loadConfig() override;
    /// Initialize subscriptions, calibration file, semaphore and workers.
    int appStartup() override;
    /// Report state and rate-limited timing warnings.
    int appLogic() override;
    /// Stop all workers using their existing base interfaces.
    int appShutdown() override;
    /// Initialize CUDA on the calling reconstruction worker.
    int setGPU();

    /// Retire modes on stream attachment changes.
    int allocate( const dmModesShmimT &tag /**< [in] Monitor tag. */ );

    /// Copy the complete modes cube from its base, ignoring temporal cnt1 semantics.
    int processImage( void                *pixels /**< [in] Monitor frame pointer, unused for the whole cube. */,
                      const dmModesShmimT &tag /**< [in] Monitor tag. */ );

    /// Retire the mask on stream attachment changes.
    int allocate( const dmMaskShmimT &tag /**< [in] Monitor tag. */ );

    /// Publish a private mask snapshot.
    int processImage( void               *pixels /**< [in] Current mask frame. */,
                      const dmMaskShmimT &tag /**< [in] Monitor tag. */ );

    /// Retire queued input on command-stream attachment changes.
    int allocate( const dmCommandShmimT &tag /**< [in] Monitor tag. */ );

    /// Stage the latest distinct command and emit at most one pending wake hint.
    int processImage( void                  *pixels /**< [in] Current delta frame. */,
                      const dmCommandShmimT &tag /**< [in] Monitor tag. */ );

    /// Reset timing identity when the loop-trigger monitor reconnects.
    int allocate( const dmLoopTriggerShmimT &tag /**< [in] Monitor tag. */ );

    /// Observe trigger metadata without reading its coefficient pixels.
    int processImage( void                      *pixels /**< [in] Unused trigger frame pointer. */,
                      const dmLoopTriggerShmimT &tag /**< [in] Monitor tag. */ );

    /// Prepare worker-owned buffers and attach a compatible passive output stream.
    int configureAcquisition();

    /// Return the latest subscribed FPS.
    float fps();

    /// Begin acquisition with the configured private buffers.
    int startAcquisition();

    /// Select and reconstruct one new command; disabled feedback still produces diagnostics.
    int acquireAndCheckValid();

    /// Validate publication ownership and copy the frozen modal vector.
    int loadImageIntoStream( void *destination /**< [out] Frame-grabber supplied output or staging buffer. */ );

    /// Record completion and release publication ownership after the base posts the stream.
    int frameGrabberPostPublish( IMAGE *stream /**< [in] Published feedback stream. */ );

    /// Release any abandoned publication and retry configuration.
    int reconfig();

  protected:
    pcf::IndiProperty m_indiP_fpsSource;      ///< Startup FPS subscription.
    pcf::IndiProperty m_indiP_fps;            ///< Reported FPS.
    pcf::IndiProperty m_indiP_writeDMf;       ///< Feedback enable control.
    pcf::IndiProperty m_indiP_feedbackTiming; ///< Timing and input-quality counters.
    pcf::IndiProperty m_indiP_reconStatus;    ///< Setup stage and calibration snapshot outcomes.
    /// Accept an FPS update without expiring silent subscriptions.
    INDI_SETCALLBACK_DECL( dmRecon, m_indiP_fpsSource );

    /// Serialize enable/disable with final publication.
    INDI_NEWCALLBACK_DECL( dmRecon, m_indiP_writeDMf );

    /// Check frame-grabber telemetry deadlines.
    int checkRecordTimes();

    /// Record frame-grabber timing telemetry.
    int recordTelem( const telem_fgtimings *tag /**< [in] Telemetry selector. */ );
};

inline dmRecon::dmRecon() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    frameGrabberT::m_ownShmim = false;
    frameGrabberT::m_staged   = true;
}

inline dmRecon::~dmRecon() noexcept
{
    if( m_semaphoreInitialized )
    {
        sem_destroy( &m_smSemaphore );
    }
}

inline void dmRecon::setupConfig()
{
    config.add( "recon.loopNumber",
                "",
                "recon.loopNumber",
                argType::Required,
                "recon",
                "loopNumber",
                false,
                "int",
                "The loop number.  Default is 1 as in aol1." );

    config.add( "recon.respMPath",
                "",
                "recon.respMPath",
                argType::Required,
                "recon",
                "respMPath",
                false,
                "int",
                "Optional response matrix.  If set then CM modes are converted by this response." );

    config.add( "recon.numModes",
                "",
                "recon.numModes",
                argType::Required,
                "recon",
                "numModes",
                false,
                "int",
                "Number of modes to reconstruct.  If 0 (default) all modes in CM are used." );

    config.add( "recon.inverseNumModes",
                "",
                "recon.inverseNumModes",
                argType::Required,
                "recon",
                "inverseNumModes",
                false,
                "int",
                "Number of modes to use for pseudo-inverse truncation.  If 0 (default) all modes are used." );

    config.add( "recon.fpsSource",
                "",
                "recon.fpsSource",
                argType::Required,
                "recon",
                "fpsSource",
                false,
                "string",
                "Device name for getting fps of the loop.  This device should have *.fps.current.  "
                "Default is camwfs" );

    config.add( "recon.gpuIndex",
                "",
                "recon.gpuIndex",
                argType::Required,
                "recon",
                "gpuIndex",
                false,
                "int",
                "Index of the GPU to use for calculations.  Default is 0." );

    config.add( "recon.useGPU",
                "",
                "recon.useGPU",
                argType::Required,
                "recon",
                "useGPU",
                false,
                "bool",
                "Flag controlling whether the GPU is used for calculations. Default is false." );

    config.add( "recon.triggerClock",
                "",
                "recon.triggerClock",
                argType::Required,
                "recon",
                "triggerClock",
                false,
                "string",
                "Clock used by loop-trigger writetime: isio, tai, or realtime. Default is isio." );
    SHMIMMONITORT_SETUP_CONFIG( loopTriggerSMT, config );
    SHMIMMONITORT_SETUP_CONFIG( dmModesSMT, config );

    SHMIMMONITORT_SETUP_CONFIG( dmMaskSMT, config );

    SHMIMMONITORT_SETUP_CONFIG( dmCommandSMT, config );

    FRAMEGRABBER_SETUP_CONFIG( config );

    TELEMETER_SETUP_CONFIG( config );
}

inline int dmRecon::loadConfigImpl( mx::app::appConfigurator &configuration )
{
    configuration( m_loopNumber, "recon.loopNumber" );
    configuration( m_respMPath, "recon.respMPath" );
    configuration( m_numModes, "recon.numModes" );
    configuration( m_inverseNumModes, "recon.inverseNumModes" );
    configuration( m_fpsSource, "recon.fpsSource" );
    configuration( m_gpuIndex, "recon.gpuIndex" );
    configuration( m_useGPU, "recon.useGPU" );
    configuration( m_triggerClockName, "recon.triggerClock" );

    if( m_loopNumber < 0 || m_numModes < 0 || m_inverseNumModes < 0 )
    {
        return -1;
    }

    if( m_triggerClockName == "isio" )
    {
        m_triggerClock = CLOCK_ISIO;
    }
    else if( m_triggerClockName == "realtime" )
    {
        m_triggerClock = CLOCK_REALTIME;
    }
#ifdef CLOCK_TAI
    else if( m_triggerClockName == "tai" )
    {
        m_triggerClock = CLOCK_TAI;
    }
#endif
    else
    {
        return -1;
    }

    const std::string loopName     = std::format( "aol{}", m_loopNumber );
    dmModesSMT::m_shmimName        = loopName + "_CMmodesDM";
    dmModesSMT::m_getExistingFirst = true;
    SHMIMMONITORT_LOAD_CONFIG( dmModesSMT, configuration );
    dmMaskSMT::m_shmimName        = std::format( "dm{:02}disp_actmask", m_loopNumber );
    dmMaskSMT::m_getExistingFirst = true;
    SHMIMMONITORT_LOAD_CONFIG( dmMaskSMT, configuration );
    dmCommandSMT::m_shmimName        = std::format( "dm{:02}disp_delta", m_loopNumber );
    dmCommandSMT::m_getExistingFirst = true;
    SHMIMMONITORT_LOAD_CONFIG( dmCommandSMT, configuration );
    loopTriggerSMT::m_shmimName        = loopName + ( m_loopNumber == 1 ? "_mfiltINPUT" : "_modevalWFS" );
    loopTriggerSMT::m_getExistingFirst = true;
    SHMIMMONITORT_LOAD_CONFIG( loopTriggerSMT, configuration );
    frameGrabberT::m_shmimName = loopName + "_modevalDMf";
    FRAMEGRABBER_LOAD_CONFIG( configuration );
    m_monShmimName = frameGrabberT::m_shmimName + "_mon";
    TELEMETER_LOAD_CONFIG( configuration );
    return 0;
}

inline void dmRecon::loadConfig()
{
    if( loadConfigImpl( config ) < 0 )
    {
        log<software_error>( { "invalid reconstruction counts or trigger clock" } );
        m_shutdown = true;
    }
}

inline int dmRecon::appStartup()
{
    REG_INDI_SETPROP( m_indiP_fpsSource, m_fpsSource, std::string( "fps" ) );
    createROIndiNumber( m_indiP_fps, "fps" );
    m_indiP_fps.add( pcf::IndiElement( "current" ) );

    if( registerIndiPropertyReadOnly( m_indiP_fps ) < 0 )
    {
        return -1;
    }

    CREATE_REG_INDI_NEW_TOGGLESWITCH( m_indiP_writeDMf, "writeDMf" );
    createROIndiNumber( m_indiP_feedbackTiming, "feedback_timing" );

    for( const char *name : { "before_next", "late", "unverified", "rejected", "coalesced" } )
    {
        m_indiP_feedbackTiming.add( pcf::IndiElement( name ) );
    }

    if( registerIndiPropertyReadOnly( m_indiP_feedbackTiming ) < 0 || sem_init( &m_smSemaphore, 0, 0 ) < 0 )
    {
        return -1;
    }

    m_semaphoreInitialized = true;
    createROIndiText( m_indiP_reconStatus, "recon_status", "stage" );
    m_indiP_reconStatus.add( pcf::IndiElement( "modes" ) );
    m_indiP_reconStatus.add( pcf::IndiElement( "mask" ) );

    if( registerIndiPropertyReadOnly( m_indiP_reconStatus ) < 0 )
    {
        return -1;
    }

    char outputPath[1024];
    ImageStreamIO_filename( outputPath, sizeof( outputPath ), frameGrabberT::m_shmimName.c_str() );
    m_outputPath = outputPath;

    if( !m_respMPath.empty() )
    {
        mx::fits::fitsFile<float> file;
        file.read( m_respM, m_respMPath );
    }

    int workerResult = -1;

    try
    {
        workerResult = startWorkers();
    }
    catch( const std::exception &error )
    {
        log<text_log>( std::string( "dmRecon worker startup: " ) + error.what(), logPrio::LOG_ERROR );
    }

    if( workerResult != 0 )
    {
        m_shutdown = true;
        appShutdown();
        return -1;
    }

    log<text_log>( std::format( "Monitoring mfilt trigger {} in {} clock; timing is diagnostic, not exclusion",
                                loopTriggerSMT::m_shmimName,
                                m_triggerClockName ) );
    state( stateCodes::OPERATING );
    return 0;
}

inline int dmRecon::startWorkers()
{
    SHMIMMONITORT_APP_STARTUP( loopTriggerSMT );
    SHMIMMONITORT_APP_STARTUP( dmModesSMT );
    SHMIMMONITORT_APP_STARTUP( dmMaskSMT );
    SHMIMMONITORT_APP_STARTUP( dmCommandSMT );
    FRAMEGRABBER_APP_STARTUP;
    TELEMETER_APP_STARTUP;
    return 0;
}

inline int dmRecon::appLogic()
{
    SHMIMMONITORT_APP_LOGIC( loopTriggerSMT );
    SHMIMMONITORT_APP_LOGIC( dmModesSMT );
    SHMIMMONITORT_APP_LOGIC( dmMaskSMT );
    SHMIMMONITORT_APP_LOGIC( dmCommandSMT );
    FRAMEGRABBER_APP_LOGIC;
    TELEMETER_APP_LOGIC;
    const auto  outputInode = m_outputInode.load();
    struct stat outputStatus{};

    if( outputInode != 0 &&
        ( stat( m_outputPath.c_str(), &outputStatus ) != 0 || uint64_t( outputStatus.st_ino ) != outputInode ) )
    {
        if( !m_outputChanged.exchange( true ) )
        {
            frameGrabberT::m_reconfig = true;
            log<text_log>( "dmRecon feedback stream replaced or missing; reconnecting", logPrio::LOG_WARNING );
        }
    }

    const auto  late          = m_late.load();
    const auto  unknown       = m_unknown.load();
    const char *setupStage    = m_setupStage.load();
    const auto  modesSnapshot = m_modesSnapshot.load(), maskSnapshot = m_maskSnapshot.load();

    if( setupStage != m_reportedSetupStage || modesSnapshot != m_reportedModesSnapshot ||
        maskSnapshot != m_reportedMaskSnapshot )
    {
        log<text_log>( std::format( "dmRecon setup: {}; modes={}, mask={}",
                                    setupStage,
                                    snapshotStatusName( modesSnapshot ),
                                    snapshotStatusName( maskSnapshot ) ) );
        m_reportedSetupStage    = setupStage;
        m_reportedModesSnapshot = modesSnapshot;
        m_reportedMaskSnapshot  = maskSnapshot;
    }

    if( late != m_reportedLate )
    {
        log<text_log>( std::format( "dmRecon feedback missed the next trigger timestamp: {} new, {} total",
                                    late - m_reportedLate,
                                    late ),
                       logPrio::LOG_WARNING );
        m_reportedLate = late;
    }

    if( unknown != m_reportedUnknown )
    {
        log<text_log>(
            std::format( "dmRecon feedback timing unverified: {} new, {} total (trigger gaps, identity or clock)",
                         unknown - m_reportedUnknown,
                         unknown ),
            logPrio::LOG_WARNING );
        m_reportedUnknown = unknown;
    }

    { //mutex scope
        std::lock_guard<std::mutex> lock( m_indiMutex );
        updateSwitchIfChanged(
            m_indiP_writeDMf, "toggle", m_writeDMf.load() ? pcf::IndiElement::On : pcf::IndiElement::Off );
        updateIfChanged( m_indiP_feedbackTiming, "before_next", double( m_beforeNext.load() ) );
        updateIfChanged( m_indiP_feedbackTiming, "late", double( late ) );
        updateIfChanged( m_indiP_feedbackTiming, "unverified", double( unknown ) );
        updateIfChanged( m_indiP_feedbackTiming, "rejected", double( m_rejected.load() ) );
        updateIfChanged( m_indiP_feedbackTiming, "coalesced", double( m_coalesced.load() ) );
        updateIfChanged( m_indiP_reconStatus, "stage", std::string( setupStage ) );
        updateIfChanged( m_indiP_reconStatus, "modes", std::string( snapshotStatusName( modesSnapshot ) ) );
        updateIfChanged( m_indiP_reconStatus, "mask", std::string( snapshotStatusName( maskSnapshot ) ) );
        SHMIMMONITORT_UPDATE_INDI( loopTriggerSMT );
        SHMIMMONITORT_UPDATE_INDI( dmModesSMT );
        SHMIMMONITORT_UPDATE_INDI( dmMaskSMT );
        SHMIMMONITORT_UPDATE_INDI( dmCommandSMT );
        FRAMEGRABBER_UPDATE_INDI;
    }

    return 0;
}

inline int dmRecon::appShutdown()
{
    SHMIMMONITORT_APP_SHUTDOWN( loopTriggerSMT );
    SHMIMMONITORT_APP_SHUTDOWN( dmModesSMT );
    SHMIMMONITORT_APP_SHUTDOWN( dmMaskSMT );
    SHMIMMONITORT_APP_SHUTDOWN( dmCommandSMT );
    FRAMEGRABBER_APP_SHUTDOWN;
    TELEMETER_APP_SHUTDOWN;
    return 0;
}

inline int dmRecon::setGPU()
{
#ifdef MXLIB_CUDA

    if( !m_useGPU )
    {
        return 0;
    }

    int deviceCount;
    int devicecntMax = 100;

    cudaError_t ce = cudaGetDeviceCount( &deviceCount );

    if( ce != cudaSuccess )
    {

        log<software_error>( { std::format( "cudaGetDeviceCount returned error: "
                                            "[{}] {}\nNOT USING GPU",
                                            cudaGetErrorName( ce ),
                                            cudaGetErrorString( ce ) ) } );

        m_useGPU = false;
        state( state(), true );
        return -1;
    }

    std::string msg = std::format( "CUDA: found {} devices\n", deviceCount );

    if( deviceCount > devicecntMax )
    {
        deviceCount = 0;
        msg += "      greater than devicecntMax\n";
    }
    if( deviceCount < 0 )
    {
        msg += "      less than zero\n";
    }

    if( deviceCount == 0 )
    {
        msg += "      no devices found!\nNOT USING GPU";
        log<software_error>( { msg } );
        m_useGPU = false;
        state( state(), true );
        return -1;
    }

    for( int k = 0; k < deviceCount; k++ )
    {
        cudaDeviceProp deviceProp;
        ce = cudaGetDeviceProperties( &deviceProp, k );

        if( ce != cudaSuccess )
        {
            msg += std::format( "cudaGetDeviceProperties returned error: "
                                "[{}] {}\nNOT USING GPU",
                                cudaGetErrorName( ce ),
                                cudaGetErrorString( ce ) );
            log<software_error>( { msg } );
            m_useGPU = false;
            state( state(), true );
            return -1;
        }

        int clockRate;
        ce = cudaDeviceGetAttribute( &clockRate, cudaDevAttrClockRate, k );

        if( ce != cudaSuccess )
        {
            msg += std::format( "cudaGetDeviceAttribute returned error: "
                                "[{}] {}\nNOT USING GPU",
                                cudaGetErrorName( ce ),
                                cudaGetErrorString( ce ) );
            log<software_error>( { msg } );
            m_useGPU = false;
            state( state(), true );
            return -1;
        }

        msg += std::format( "      Device {} / {} [ {} ] has compute capability {}.{}.\n",
                            k + 1,
                            deviceCount,
                            deviceProp.name,
                            deviceProp.major,
                            deviceProp.minor );

        msg += std::format( "          Total amount of global memory: {} MBytes\n",
                            (float)deviceProp.totalGlobalMem / 1048576.0f );

        msg += std::format( "          Multiprocessors: {}\n", deviceProp.multiProcessorCount );
        msg += std::format( "          Clock rate: {} MHz ({} GHz)\n", clockRate * 1e-3f, clockRate * 1e-6f );
    }

    if( m_gpuIndex >= deviceCount )
    {
        msg += std::format( "gpuIndex = {} is not valid for {} devices\nNOT USING GPU", m_gpuIndex, deviceCount );
        log<software_error>( { msg } );
        m_useGPU = false;
        state( state(), true );
        return -1;
    }

    ce = cudaSetDevice( m_gpuIndex );

    if( ce != cudaSuccess )
    {
        msg += std::format( "cudaSetDevice returned error: "
                            "[{}] {}\nNOT USING GPU",
                            cudaGetErrorName( ce ),
                            cudaGetErrorString( ce ) );
        log<software_error>( { msg } );
        m_useGPU = false;
        state( state(), true );
        return -1;
    }

    msg += std::format( "Set GPU Index to device {} ( {} / {})\n", m_gpuIndex, m_gpuIndex + 1, deviceCount );

    cublasStatus_t cbs = m_cublas.create();

    if( cbs != CUBLAS_STATUS_SUCCESS )
    {
        msg += std::format( "cublasHandle create returned error: "
                            "[{}] {}\nNOT USING GPU",
                            cublasGetStatusName( cbs ),
                            cublasGetStatusString( cbs ) );
        log<software_error>( { msg } );
        m_useGPU = false;
        state( state(), true );
        return -1;
    }

    msg += "      cuBLAS initialized";

    log<text_log>( msg );

    return 0;

#else // MXLIB_CUDA

    if( m_useGPU )
    {
        log<software_error>( { "mxlib was compiled without CUDA support. NOT USING GPU" } );

        m_useGPU = false;
        state( state(), true );
        return -1;
    }

    return 0;

#endif // MXLIB_CUDA
}

inline bool
dmRecon::snapshot( Frame &frame, const IMAGE &source, const void *pixels, uint32_t depth, SnapshotStatus *status )
{
    auto result = [status]( SnapshotStatus value )
    {
        if( status )
        {
            *status = value;
        }

        return value == SnapshotStatus::ready;
    };

    if( !source.md || !source.array.raw || !pixels )
    {
        return result( SnapshotStatus::unavailable );
    }

    if( source.md->datatype != _DATATYPE_FLOAT || source.md->naxis < 2 || source.md->naxis > 3 ||
        source.md->location != -1 )
    {
        return result( SnapshotStatus::invalidFormat );
    }

    if( source.md->size[0] == 0 || source.md->size[1] == 0 || depth == 0 )
    {
        return result( SnapshotStatus::invalidShape );
    }

    if( __atomic_load_n( &source.md->write, __ATOMIC_ACQUIRE ) )
    {
        return result( SnapshotStatus::busy );
    }

    const uint64_t  counter = __atomic_load_n( &source.md->cnt0, __ATOMIC_ACQUIRE );
    const size_t    width = source.md->size[0], height = source.md->size[1];
    const uint32_t  availableDepth = source.md->naxis == 3 ? source.md->size[2] : 1;
    const uintptr_t base           = reinterpret_cast<uintptr_t>( source.array.raw );
    const uintptr_t start          = reinterpret_cast<uintptr_t>( pixels );

    if( depth > availableDepth || width > SIZE_MAX / height || width * height > SIZE_MAX / depth / sizeof( float ) ||
        start < base )
    {
        return result( SnapshotStatus::invalidShape );
    }

    const size_t count = width * height * depth, offset = start - base;

    if( offset % sizeof( float ) != 0 || offset > source.md->imdatamemsize ||
        count * sizeof( float ) > source.md->imdatamemsize - offset || offset / sizeof( float ) > source.md->nelement ||
        count > source.md->nelement - offset / sizeof( float ) )
    {
        return result( SnapshotStatus::invalidCapacity );
    }

    frame.pixels.resize( count );
    frame.width   = width;
    frame.height  = height;
    frame.depth   = depth;
    frame.counter = counter;
    frame.inode   = source.md->inode;
    frame.stamp   = source.md->writetime;

    if( clock_gettime( CLOCK_REALTIME, &frame.received ) != 0 )
    {
        return result( SnapshotStatus::clockError );
    }

    memcpy( frame.pixels.data(), pixels, count * sizeof( float ) );
    std::atomic_thread_fence( std::memory_order_acquire );

    if( __atomic_load_n( &source.md->write, __ATOMIC_ACQUIRE ) ||
        counter != __atomic_load_n( &source.md->cnt0, __ATOMIC_ACQUIRE ) || frame.inode != source.md->inode ||
        frame.width != source.md->size[0] || frame.height != source.md->size[1] )
    {
        return result( SnapshotStatus::changed );
    }

    return result( SnapshotStatus::ready );
}

inline const char *dmRecon::snapshotStatusName( SnapshotStatus status )
{
    switch( status )
    {
    case SnapshotStatus::unavailable:
        return "unavailable";
    case SnapshotStatus::invalidFormat:
        return "invalid_format";
    case SnapshotStatus::invalidShape:
        return "invalid_shape";
    case SnapshotStatus::invalidCapacity:
        return "invalid_capacity";
    case SnapshotStatus::busy:
        return "writer_busy";
    case SnapshotStatus::changed:
        return "changed_during_copy";
    case SnapshotStatus::clockError:
        return "clock_error";
    case SnapshotStatus::ready:
        return "ready";
    }

    return "unknown";
}

inline bool
dmRecon::captureCalibration( Frame &frame, const IMAGE &source, const void *pixels, uint32_t depth, bool modes )
{
    auto          &reported = modes ? m_modesSnapshot : m_maskSnapshot;
    SnapshotStatus status;

    for( unsigned attempt = 0; attempt < 4; ++attempt )
    {
        const bool accepted = snapshot( frame, source, pixels, depth, &status );
        reported            = status;

        if( accepted )
        {
            return true;
        }

        if( attempt == 0 )
        {
            ++m_rejected;
            invalidateCalibration( modes );
        }

        if( m_shutdown || ( status != SnapshotStatus::busy && status != SnapshotStatus::changed ) )
        {
            break;
        }

        // Some calibration writers post before clearing write; retry without requiring another post.
        mx::sys::milliSleep( 1 );
    }

    // Back off before the owning monitor reopens; invalid schemas must not create a tight reconnect loop.
    if( !m_shutdown )
    {
        mx::sys::milliSleep( 100 );
    }

    return false;
}

inline bool dmRecon::outputCompatible( const IMAGE_METADATA &metadata, size_t modes )
{
    return modes > 0 && modes <= std::numeric_limits<size_t>::max() / sizeof( float ) &&
           metadata.datatype == _DATATYPE_FLOAT && metadata.location == -1 &&
           ( metadata.naxis == 2 || metadata.naxis == 3 ) && metadata.size[0] == modes && metadata.size[1] == 1 &&
           ( metadata.naxis != 3 || metadata.size[2] == 1 ) && metadata.nelement == modes &&
           metadata.imdatamemsize >= modes * sizeof( float );
}

inline bool dmRecon::sampleClock( timespec &stamp, int64_t &low, int64_t &high )
{
    timespec before{}, after{};

    if( clock_gettime( CLOCK_MONOTONIC, &before ) != 0 || clock_gettime( m_triggerClock, &stamp ) != 0 ||
        clock_gettime( CLOCK_MONOTONIC, &after ) != 0 )
    {
        return false;
    }

    const int64_t selected = int64_t( stamp.tv_sec ) * 1000000000 + stamp.tv_nsec;
    const int64_t first    = int64_t( before.tv_sec ) * 1000000000 + before.tv_nsec;
    const int64_t last     = int64_t( after.tv_sec ) * 1000000000 + after.tv_nsec;
    low                    = selected - last;
    high                   = selected - first;
    return last >= first && last - first <= 100000;
}

inline void dmRecon::timingUpdate( const dmReconTiming::Update &update )
{
    m_beforeNext += update.beforeNextTrigger;
    m_late += update.late;
    m_unknown += update.unknown;
}

inline void dmRecon::invalidateCalibration( bool modes )
{
    { //mutex scope
        std::scoped_lock lock( m_publicationMutex, m_snapshotMutex );
        ( modes ? m_modesValid : m_maskValid ) = false;
        ++m_calibrationGeneration;
        frameGrabberT::m_reconfig = true;
    }
}

inline int dmRecon::allocate( const dmModesShmimT & )
{
    m_modesSnapshot = SnapshotStatus::unavailable;
    invalidateCalibration( true );

    if( m_modesRetry )
    {
        m_modesRetry = false;
        return processImage( nullptr, dmModesShmimT{} );
    }

    return 0;
}

inline int dmRecon::processImage( void *, const dmModesShmimT & )
{
    const uint32_t depth =
        m_numModes == 0 ? dmModesSMT::m_depth : std::min<uint32_t>( m_numModes, dmModesSMT::m_depth );

    if( !captureCalibration(
            m_modesStage, dmModesSMT::m_imageStream, dmModesSMT::m_imageStream.array.raw, depth, true ) )
    {
        // Only this monitor thread changes its restart flag or touches its mapping.
        m_modesRetry          = true;
        dmModesSMT::m_restart = true;
        return 0;
    }

    { //mutex scope
        std::scoped_lock lock( m_publicationMutex, m_snapshotMutex );

        if( m_modesValid && m_pendingModes.counter == m_modesStage.counter &&
            m_pendingModes.inode == m_modesStage.inode )
        {
            return 0;
        }

        std::swap( m_pendingModes, m_modesStage );
        m_modesValid = true;
        ++m_calibrationGeneration;
        frameGrabberT::m_reconfig = true;
    }

    return 0;
}

inline int dmRecon::allocate( const dmMaskShmimT & )
{
    m_maskSnapshot = SnapshotStatus::unavailable;
    invalidateCalibration( false );

    if( m_maskRetry )
    {
        m_maskRetry = false;
        return processImage( dmMaskSMT::m_imageStream.array.raw, dmMaskShmimT{} );
    }

    return 0;
}

inline int dmRecon::processImage( void *pixels, const dmMaskShmimT & )
{
    if( !captureCalibration( m_maskStage, dmMaskSMT::m_imageStream, pixels, dmMaskSMT::m_depth == 1 ? 1 : 0, false ) )
    {
        m_maskRetry          = true;
        dmMaskSMT::m_restart = true;
        return 0;
    }

    { //mutex scope
        std::scoped_lock lock( m_publicationMutex, m_snapshotMutex );

        if( m_maskValid && m_pendingMask.counter == m_maskStage.counter && m_pendingMask.inode == m_maskStage.inode )
        {
            return 0;
        }

        std::swap( m_pendingMask, m_maskStage );
        m_maskValid = true;
        ++m_calibrationGeneration;
        frameGrabberT::m_reconfig = true;
    }

    return 0;
}

inline int dmRecon::allocate( const dmCommandShmimT & )
{
    { //mutex scope
        std::scoped_lock lock( m_publicationMutex, m_snapshotMutex );
        ++m_inputGeneration;
        m_commandPending = false;
        m_haveCommand    = false;
    }

    return 0;
}

inline int dmRecon::processImage( void *pixels, const dmCommandShmimT & )
{
    if( dmCommandSMT::m_depth != 1 || !snapshot( m_commandStage, dmCommandSMT::m_imageStream, pixels, 1 ) )
    {
        ++m_rejected;
        return 0;
    }

    timespec stamp{};
    m_commandStage.clockValid = sampleClock( stamp, m_commandStage.clockLow, m_commandStage.clockHigh );
    bool wake                 = false;

    { //mutex scope
        std::lock_guard<std::mutex> lock( m_snapshotMutex );

        if( m_haveCommand && m_commandCounter == m_commandStage.counter && m_commandInode == m_commandStage.inode )
        {
            return 0;
        }

        m_haveCommand                  = true;
        m_commandCounter               = m_commandStage.counter;
        m_commandInode                 = m_commandStage.inode;
        m_commandStage.generation      = m_calibrationGeneration;
        m_commandStage.inputGeneration = m_inputGeneration;
        m_commandStage.trigger         = m_latestTrigger;
        // A trigger newer than the source command cannot identify the step which produced that command.
        const auto &triggerTime = m_commandStage.trigger.writeTime;

        if( m_triggerClock != CLOCK_ISIO || !m_triggerClockValid || !m_commandStage.clockValid ||
            m_commandStage.clockLow > m_triggerClockHigh + 1000 ||
            m_commandStage.clockHigh < m_triggerClockLow - 1000 || triggerTime.tv_sec > m_commandStage.stamp.tv_sec ||
            ( triggerTime.tv_sec == m_commandStage.stamp.tv_sec &&
              triggerTime.tv_nsec > m_commandStage.stamp.tv_nsec ) )
        {
            m_commandStage.trigger = {};
        }

        wake = !m_commandPending;
        m_coalesced += m_commandPending;
        std::swap( m_pendingCommand, m_commandStage );
        m_commandPending = true;
    }

    if( wake && sem_post( &m_smSemaphore ) < 0 )
    {
        return log<software_error, -1>( { errno, "posting command wake" } );
    }

    return 0;
}

inline int dmRecon::allocate( const dmLoopTriggerShmimT & )
{
    { //mutex scope
        std::lock_guard<std::mutex> lock( m_snapshotMutex );
        timingUpdate( m_timing.reset() );
        ++m_triggerGeneration;
        m_latestTrigger     = {};
        m_triggerClockValid = false;
    }

    return 0;
}

inline int dmRecon::processImage( void *, const dmLoopTriggerShmimT & )
{
    const auto *metadata = loopTriggerSMT::m_imageStream.md;

    if( !metadata || __atomic_load_n( &metadata->write, __ATOMIC_ACQUIRE ) )
    {
        return 0;
    }

    dmReconTiming::Trigger trigger;
    timespec               observed{};
    int64_t                low = 0, high = 0;
    const bool             clockValid = sampleClock( observed, low, high );
    trigger.counter                   = __atomic_load_n( &metadata->cnt0, __ATOMIC_ACQUIRE );
    trigger.inode                     = metadata->inode;
    trigger.writeTime                 = metadata->writetime;
    std::atomic_thread_fence( std::memory_order_acquire );

    if( __atomic_load_n( &metadata->write, __ATOMIC_ACQUIRE ) ||
        trigger.counter != __atomic_load_n( &metadata->cnt0, __ATOMIC_ACQUIRE ) )
    {
        return 0;
    }

    { //mutex scope
        std::lock_guard<std::mutex> lock( m_snapshotMutex );

        if( !clockValid ||
            ( m_triggerClockValid && ( low > m_triggerClockHigh + 1000 || high < m_triggerClockLow - 1000 ) ) ||
            ( m_latestTrigger.generation != 0 &&
              ( trigger.inode != m_latestTrigger.inode || trigger.counter < m_latestTrigger.counter ) ) )
        {
            timingUpdate( m_timing.reset() );
            ++m_triggerGeneration;
            m_latestTrigger = {};
        }

        m_triggerClockValid = clockValid;
        m_triggerClockLow   = low;
        m_triggerClockHigh  = high;

        if( !clockValid )
        {
            return 0;
        }

        trigger.generation = m_triggerGeneration;
        timingUpdate( m_timing.observe( trigger ) );
        m_latestTrigger = trigger;
    }

    return 0;
}

inline int dmRecon::prepareCalibration()
{
    Frame    modes, mask;
    uint64_t generation;

    { //mutex scope
        std::lock_guard<std::mutex> lock( m_snapshotMutex );

        if( !m_modesValid || !m_maskValid )
        {
            m_setupStage = !m_modesValid && !m_maskValid ? "waiting_for_modes_and_mask"
                           : !m_modesValid               ? "waiting_for_modes"
                                                         : "waiting_for_mask";
            return 1;
        }

        generation = m_calibrationGeneration;

        if( generation == m_workerGeneration )
        {
            return m_calibrationUsable ? 0 : 1;
        }

        m_setupStage = "copying_calibration";
        modes        = m_pendingModes;
        mask         = m_pendingMask;
    }

    m_workerGeneration  = generation;
    m_calibrationUsable = false;
    m_setupStage        = "validating_calibration";
    uint32_t width = modes.width, height = modes.height;

    if( m_respM.size() != 0 && !dmReconMath::squareSize( width, m_respM.rows() ) )
    {
        m_setupStage = "calibration_error";
        return log<software_error, 1>( { "response output is not square" } );
    }

    if( m_respM.size() != 0 )
    {
        height = width;
    }

    if( modes.width == 0 || modes.height == 0 || modes.depth == 0 ||
        size_t( modes.width ) > std::numeric_limits<size_t>::max() / modes.height / modes.depth ||
        modes.pixels.size() != size_t( modes.width ) * modes.height * modes.depth || mask.width != width ||
        mask.height != height || mask.depth != 1 || mask.pixels.size() != size_t( width ) * height ||
        !dmReconMath::maskIndices( m_maskIDX, mask.pixels.data(), mask.pixels.size() ) )
    {
        m_setupStage = "calibration_error";
        return log<software_error, 1>( { "invalid mask or calibration dimensions" } );
    }

    mx::improc::eigenCube<float> cube( modes.width, modes.height, modes.depth );
    memcpy( cube.data(), modes.pixels.data(), modes.pixels.size() * sizeof( float ) );
    float       condition = 1;
    int         rejected  = 0;
    std::string error;
    m_setupStage = "building_inverse";
    log<text_log>( std::format( "Building CMmodesDM inverse generation {}: {} active pixels, {} modes",
                                generation,
                                m_maskIDX.size(),
                                modes.depth ) );

    if( dmReconMath::buildInverse( m_PInv, condition, rejected, cube, m_maskIDX, m_respM, m_inverseNumModes, error ) !=
        0 )
    {
        m_setupStage = "calibration_error";
        return log<software_error, 1>( { error } );
    }

    m_commandWidth  = width;
    m_commandHeight = height;
    m_command.resize( m_maskIDX.size(), 1 );
    m_modevals.resize( m_PInv.rows(), 1 );
    m_difference.resize( m_PInv.rows() );
    m_diagnosticModes.resize( m_PInv.rows() );
#ifdef MXLIB_CUDA
    if( m_useGPU && !m_gpuReady )
    {
        m_setupStage = "initializing_gpu";
        m_gpuReady   = setGPU() == 0;
    }

    if( m_useGPU )
    {
        m_setupStage = "allocating_gpu";
        auto gpu     = std::make_unique<GPUData>();

        if( gpu->inverse.upload( m_PInv.data(), m_PInv.rows(), m_PInv.cols() ) != mx::error_t::noerror ||
            gpu->command.resize( m_command.size() ) != mx::error_t::noerror ||
            gpu->modevals.resize( m_modevals.size() ) != mx::error_t::noerror )
        {
            m_setupStage = "gpu_allocation_error";
            return log<software_error, 1>( { "allocating reconstruction GPU buffers" } );
        }

        m_gpu = std::move( gpu );
    }
#else
    if( m_useGPU )
    {
        setGPU();
    }
#endif
    m_calibrationUsable = true;
    m_setupStage        = "calibration_ready";
    log<text_log>( std::format( "Inverted CMmodesDM generation {}: rejected {} of {} modes, condition {}",
                                generation,
                                rejected,
                                modes.depth,
                                condition ) );
    return 0;
}

inline bool dmRecon::takeCommand()
{
    { //mutex scope
        std::lock_guard<std::mutex> lock( m_snapshotMutex );

        if( !m_commandPending )
        {
            return false;
        }

        std::swap( m_currentCommand, m_pendingCommand );
        m_commandPending = false;
    }

    return true;
}

inline int dmRecon::reconstruct()
{
    const auto &input = m_currentCommand;

    if( !m_calibrationUsable || input.generation != m_workerGeneration || input.width != m_commandWidth ||
        input.height != m_commandHeight || input.depth != 1 ||
        input.pixels.size() != size_t( m_commandWidth ) * m_commandHeight ||
        !dmReconMath::allFinite( input.pixels.data(), input.pixels.size() ) )
    {
        ++m_rejected;
        return 1;
    }

    for( size_t n = 0; n < m_maskIDX.size(); ++n )
    {
        m_command( n, 0 ) = input.pixels[m_maskIDX[n]];
    }

#ifdef MXLIB_CUDA
    if( m_useGPU )
    {
        const float alpha = 1, beta = 0;

        if( !m_gpu || m_gpu->command.upload( m_command.data() ) != mx::error_t::noerror ||
            mx::cuda::cublasTgemv( m_cublas,
                                   CUBLAS_OP_N,
                                   m_gpu->inverse.rows(),
                                   m_gpu->inverse.cols(),
                                   &alpha,
                                   m_gpu->inverse.data(),
                                   m_gpu->inverse.rows(),
                                   m_gpu->command.data(),
                                   1,
                                   &beta,
                                   m_gpu->modevals.data(),
                                   1 ) != CUBLAS_STATUS_SUCCESS ||
            m_gpu->modevals.download( m_modevals.data() ) != mx::error_t::noerror )
        {
            ++m_rejected;
            return 1;
        }
    }
    else
#endif
    {
        m_modevals = ( m_PInv.matrix() * m_command.matrix() ).array();
    }

    if( !dmReconMath::allFinite( m_modevals.data(), m_modevals.size() ) )
    {
        ++m_rejected;
        return 1;
    }

    m_currImageTimestamp = input.received;
    return 0;
}

inline int dmRecon::configureAcquisition()
{
    reconfig();
    m_outputChanged = true;
    m_outputInode   = 0;

    if( prepareCalibration() != 0 )
    {
        return 1;
    }

    frameGrabberT::m_width    = m_modevals.rows();
    frameGrabberT::m_height   = 1;
    frameGrabberT::m_dataType = _DATATYPE_FLOAT;
    m_setupStage              = "connecting_output";
    const int rv              = frameGrabberT::openShmim();

    if( rv != 0 )
    {
        return rv;
    }

    const auto *metadata = frameGrabberT::m_imageStream->md;

    if( !outputCompatible( *metadata, m_modevals.size() ) )
    {
        m_setupStage = "output_schema_mismatch";
        return 1;
    }

    struct stat mappedStatus{}, pathStatus{};

    if( fstat( frameGrabberT::m_imageStream->shmfd, &mappedStatus ) != 0 ||
        stat( m_outputPath.c_str(), &pathStatus ) != 0 || mappedStatus.st_ino != pathStatus.st_ino )
    {
        m_setupStage = "output_identity_mismatch";
        return 1;
    }

    frameGrabberT::m_width          = m_modevals.rows();
    frameGrabberT::m_height         = 1;
    frameGrabberT::m_circBuffLength = 1;
    frameGrabberT::m_dataType       = _DATATYPE_FLOAT;
    m_outputInode                   = mappedStatus.st_ino;
    m_outputChanged                 = false;
    m_diagnosticsReady              = false;
    m_differenceReady               = false;

    try
    {
        m_modevalMon.create( m_monShmimName, m_modevals.rows(), 1 );
        m_modevalDiff.create( std::format( "aol{}_modevalDMf_diff", m_loopNumber ), m_modevals.rows(), 1 );
        m_diagnosticsReady = m_modevalMon.rows() == m_modevals.rows() && m_modevalDiff.rows() == m_modevals.rows();
        m_modeval.open( std::format( "aol{}_modevalDM", m_loopNumber ) );
        m_differenceReady = m_modeval.rows() == m_modevals.rows() && m_modeval.cols() == 1;
    }
    catch( const std::exception &error )
    {
        log<text_log>( std::string( "dmRecon diagnostic connection: " ) + error.what(), logPrio::LOG_WARNING );
    }

    m_setupStage = m_diagnosticsReady ? "acquiring" : "acquiring_without_monitor";
    return 0;
}

inline float dmRecon::fps()
{
    return m_fps.load();
}

inline int dmRecon::startAcquisition()
{
    return 0;
}

inline void dmRecon::publishDiagnostics()
{
    if( !m_diagnosticsReady )
    {
        return;
    }

    try
    {
        m_modevalMon.setWrite();
        streamCopy( m_modevalMon.data(), m_modevals.data(), m_modevals.size() * sizeof( float ) );
        m_modevalMon.post();

        if( m_differenceReady )
        {
            memcpy( m_diagnosticModes.data(), m_modeval.data(), m_diagnosticModes.size() * sizeof( float ) );

            for( size_t n = 0; n < m_difference.size(); ++n )
            {
                m_difference[n] = m_modevals( n, 0 ) - m_diagnosticModes[n];
            }

            m_modevalDiff.setWrite();
            streamCopy( m_modevalDiff.data(), m_difference.data(), m_difference.size() * sizeof( float ) );
            m_modevalDiff.post();
        }
    }
    catch( const std::exception & )
    {
        m_diagnosticsReady = false;
        m_setupStage       = "acquiring_without_monitor";
    }
}

inline int dmRecon::acquireAndCheckValid()
{
    timespec deadline{};
    clock_gettime( CLOCK_REALTIME, &deadline );
    ++deadline.tv_sec;

    if( sem_timedwait( &m_smSemaphore, &deadline ) != 0 || !takeCommand() )
    {
        return 1;
    }

    if( reconstruct() != 0 )
    {
        return 1;
    }

    if( !m_writeDMf.load() )
    {
        publishDiagnostics();
        return 1;
    }

    return 0;
}

inline int dmRecon::loadImageIntoStream( void *destination )
{
    m_publicationLock = std::unique_lock<std::mutex>( m_publicationMutex );
    bool valid;

    { //mutex scope
        std::lock_guard<std::mutex> lock( m_snapshotMutex );
        valid = m_writeDMf.load() && !m_outputChanged.load() && m_calibrationUsable &&
                m_currentCommand.generation == m_calibrationGeneration &&
                m_currentCommand.inputGeneration == m_inputGeneration;
    }

    if( !valid || !destination )
    {
        if( frameGrabberT::m_imageStream )
        {
            frameGrabberT::m_imageStream->md->write = 0;
        }

        m_publicationLock.unlock();
        return -1;
    }

    if( frameGrabberT::m_staged )
    {
        memcpy( destination, m_modevals.data(), m_modevals.size() * sizeof( float ) );
    }
    else
    {
        streamCopy( destination, m_modevals.data(), m_modevals.size() * sizeof( float ) );
    }

    return 0;
}

inline int dmRecon::frameGrabberPostPublish( IMAGE * )
{
    timespec   completed{};
    int64_t    low = 0, high = 0;
    const bool clockValid = sampleClock( completed, low, high ) && m_currentCommand.clockValid &&
                            low <= m_currentCommand.clockHigh + 1000 && high >= m_currentCommand.clockLow - 1000;

    { //mutex scope
        std::lock_guard<std::mutex> lock( m_snapshotMutex );

        if( clockValid )
        {
            timingUpdate( m_timing.published( m_currentCommand.trigger, completed, m_currentCommand.stamp ) );
        }
        else
        {
            timingUpdate( m_timing.reset() );
            ++m_unknown;
        }
    }

    if( m_publicationLock.owns_lock() )
    {
        m_publicationLock.unlock();
    }

    publishDiagnostics();
    return 0;
}

inline int dmRecon::reconfig()
{
    if( m_publicationLock.owns_lock() )
    {
        m_publicationLock.unlock();
    }

    return 0;
}

INDI_SETCALLBACK_DEFN( dmRecon, m_indiP_fpsSource )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_fpsSource, ipRecv );

    if( !ipRecv.find( "current" ) )
    {
        return -1;
    }

    const float value = ipRecv["current"].get<float>();

    if( !dmReconMath::finite( value ) || value < 0 )
    {
        return -1;
    }

    { //mutex scope
        std::lock_guard<std::mutex> lock( m_indiMutex );
        m_fps = value;
        updateIfChanged( m_indiP_fps, "current", value );
    }

    return 0;
}

INDI_NEWCALLBACK_DEFN( dmRecon, m_indiP_writeDMf )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_writeDMf, ipRecv );

    if( !ipRecv.find( "toggle" ) )
    {
        return -1;
    }

    { //mutex scope
        std::scoped_lock lock( m_indiMutex, m_publicationMutex );
        m_writeDMf = ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On;
        updateSwitchIfChanged(
            m_indiP_writeDMf, "toggle", m_writeDMf.load() ? pcf::IndiElement::On : pcf::IndiElement::Off );
    }

    return 0;
}

inline int dmRecon::checkRecordTimes()
{
    return telemeterT::checkRecordTimes( telem_fgtimings() );
}

inline int dmRecon::recordTelem( const telem_fgtimings * )
{
    return recordFGTimings( true );
}

} // namespace app
} // namespace MagAOX
#endif // dmRecon_hpp
