/** \file loPredCtrlAcc.hpp
 * \brief Legacy low-order control and independent non-actuating causal DDSPC diagnostics.
 *
 * \ingroup app_files
 */

#ifndef loPredCtrlAcc_hpp
#define loPredCtrlAcc_hpp

#include <iostream>
#include <fstream>
#include <vector>
#include <limits>
#include <chrono>
#include <thread>
#include <random>
#include <semaphore.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <utility>

#include <Eigen/Dense>
#include <mx/improc/eigenCube.hpp>
#include <mx/improc/eigenImage.hpp>
using namespace mx::improc;

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#ifndef MAGAOX_CURRENT_SHA1
    #include "../../magaox_git_version.h"
#endif

#include "ar_controller.hpp"
#include "shadow_controller.hpp"
#include "causal_input.hpp"
#include "adc_reader.hpp"
#include "native_observer.hpp"
#include "shadow_pipeline.hpp"

namespace MagAOX
{
namespace app
{

/// Tag for the optional legacy accelerometer monitor; unused in shadow execution.
struct accelShmimT
{
    /// Return the legacy accelerometer configuration section.
    static std::string configSection();

    /// Return the legacy accelerometer property prefix.
    static std::string indiPrefix();
};

/// Legacy operational controller with an opt-in, output-inhibited pair of causal shadow models.
class loPredCtrlAcc : public MagAOXApp<true>,
                      public dev::shmimMonitor<loPredCtrlAcc>,
                      public dev::shmimMonitor<loPredCtrlAcc, accelShmimT>,
                      public dev::frameGrabber<loPredCtrlAcc>,
                      public dev::telemeter<loPredCtrlAcc>
{
    // Give the test harness access.
    friend class loPredCtrlAcc_test;

    friend class dev::shmimMonitor<loPredCtrlAcc>;
    friend class dev::shmimMonitor<loPredCtrlAcc, accelShmimT>;

    // The base shmimMonitor type
    typedef dev::shmimMonitor<loPredCtrlAcc>              shmimMonitorT;
    typedef dev::shmimMonitor<loPredCtrlAcc, accelShmimT> accelShmimMonitorT;

    friend class dev::frameGrabber<loPredCtrlAcc>;

    typedef dev::frameGrabber<loPredCtrlAcc> frameGrabberT;

    friend class dev::telemeter<loPredCtrlAcc>;

    typedef dev::telemeter<loPredCtrlAcc> telemeterT;

    /// Floating point type in which to do all calculations.
    typedef float realT;

  public:
    /** \name app::dev Configurations
     *@{
     */

    /// This framegrabber can't be flipped
    static constexpr bool c_frameGrabber_flippable = false;

    ///@}

  protected:
    /** \name Configurable Parameters - Data
     *@{
     */

    /// Width of the most recently allocated WFS modal payload.
    uint32_t m_modevalWidth{ 0 };

    /// Height of the most recently allocated WFS modal payload.
    uint32_t m_modevalHeight{ 0 };

    /// Legacy WFS element size for modal copies.
    uint32_t m_modevalTypeSize{ 0 };

    /// Legacy processing-loop sample count.
    long long m_frameCounter{ 0 };

    /// Legacy timing diagnostic epoch.
    std::chrono::high_resolution_clock::time_point m_lastPrintTime{ std::chrono::high_resolution_clock::now() };

    /// Additional legacy feedback gain; excluded from the reference shadow law.
    realT m_gainCtrl{ 0.0 };

    /// Legacy residual-copy gain when predictive correction is disabled.
    realT m_copygainCtrl{ 1.0 };

    /// Preserve historical loading of copy gain from gain; opt out to honor an explicit copy_gain key.
    bool m_legacyCopyGainCompat{ true };

    /// Legacy scaled penalty or literal positive shadow reference-law lambda.
    realT m_regularizationCtrl{ 100.0 };

    /// RLS forgetting factor for each independently owned learner.
    realT m_gammaCtrl{ 1.00 };

    /// Initial RLS information strength (P0 = I / covariance).
    realT m_covarianceCtrl{ 1.0 };

    /// Number of residual/applied-input modes in each causal model.
    int m_num_modes{ 1 };

    /// WFS history length; causal command state uses history minus one.
    int m_history{ 5 };

    /// Future residual/input horizon in decision steps.
    int m_future{ 3 };

    /// Startup-only opt-in; shadow execution never starts or uses the output frameGrabber.
    bool m_shadowMode{ false };

    /// Startup-only shared shadow pipeline; false preserves all existing input/Legacy dispatch.
    bool m_shadowPipelineEnabled{ false };

    /// A failed opt-in loader cannot be bypassed by a later direct startup call.
    bool m_shadowPipelineConfigValid{ false };

    /// Immutable shared component settings; candidate limits must be explicitly configured when enabled.
    DDSPC::FeatureConfig m_shadowFeatures;
    DDSPC::PolicyConfig  m_shadowPolicy;
    DDSPC::ShadowConfig  m_shadowSettings;
    std::string          m_shadowRequested{ "standard" };
    double               m_shadowLimit{ 0 }, m_shadowStep{ 0 };
    std::uint64_t        m_shadowCalibration{ 1 };
    std::size_t          m_shadowAdcCapacity{ 128 };

    /// Serialized admission gate, closed before input helpers are stopped or model storage is destroyed.
    bool m_shadowStopping{ false };

    /// Startup-only native ADC observation, accepted only with shadow execution.
    bool m_adcEnabled{ false };

    /// Immutable ADC transport/calibration; independent of Legacy's FLOAT sensor reader.
    DDSPC::AdcConfig m_adcConfig;

    /// Configured microsecond limits, converted to checked integer nanoseconds at startup.
    double m_adcMaxAgeUs{ 20000 }, m_adcClockToleranceUs{ 1000 }, m_adcMaxIntervalUs{ 1000 };

    /// Owns only an input worker/ring; never feeds model validity or publishes corrections.
    std::unique_ptr<DDSPC::AdcReader> m_adcReader;

    /// Native read-only input path; the installed entrypoint enables continuous acquisition by default.
    bool m_nativeShadowEnabled{ false }, m_nativeShadowConfigValid{ false };

    /// Existing input selections and private local recording limits.
    DDSPC::NativeObservationConfig m_nativeShadowConfig;

    /// Owns observation workers; stopped before its borrowed ADC reader.
    std::unique_ptr<DDSPC::NativeObserver> m_nativeObserver;

    /// Private read-only run state and per-source transport diagnostics.
    pcf::IndiProperty                                        m_indiP_nativeShadow;
    std::array<pcf::IndiProperty, DDSPC::observationSources> m_indiP_nativeSources;
    std::array<pcf::IndiProperty, DDSPC::observationSources>
                      m_indiP_nativeIdentities; ///< Exact IDs/timestamps as text.
    pcf::IndiProperty m_indiP_nativeSafety, m_indiP_nativeTiming,
        m_indiP_nativeInterlocks;                                                   ///< Output and timing gates.
    std::array<std::uint64_t, DDSPC::observationSources> m_nativeReportedSamples{}; ///< New-sample diagnostic pulses.
    std::array<bool, DDSPC::observationSources> m_nativeReportedConnections{}; ///< Log connection transitions once.
    std::array<std::string, 4> m_nativeReportedHealth; ///< Rate-limited compact health-transition logging.
    double m_nativeDiagnosticsMs{ 250 }; ///< Maximum periodic diagnostic rate, independent of acquisition workers.
    std::int64_t m_nativeNextDiagnosticNs{ 0 }; ///< Steady-clock deadline for the next periodic property update.

    bool m_nativeDebugINDI{ false }; ///< Startup-only opt-in to the full native metadata/Legacy diagnostic schema.
    pcf::IndiProperty m_indiP_adcForwarding; ///< Opt-in exact cursor/cutoff evidence; never grants input validity.

    /// Compact native operator groups; detailed source records remain in the optional journal/debug interface.
    enum class NativeOperator : std::size_t
    {
        Health,
        Inputs,
        Rates,
        Adc,
        Losses,
        Buffers,
        Timing,
        Control,
        Performance,
        Safety,
        Count
    };
    std::array<pcf::IndiProperty, std::size_t( NativeOperator::Count )>
        m_indiP_nativeOperator; ///< Operator-only schema.

    /// ADC-only numbers and provenance; separate from synchronized DDSPC validity.
    pcf::IndiProperty m_indiP_adc, m_indiP_adcSource;

    /// Required learned examples per shadow model; zero uses its regressor dimension.
    uint64_t m_shadowMinUpdates{ 0 };

    /// Serialize model processing, selection, and diagnostics; nested only after m_indiMutex.
    std::mutex m_shadowMutex;

    /// Independently owned standard and assisted causal predictors; absent in legacy execution.
    std::unique_ptr<DDSPC::ShadowController> m_shadowController;

    /// Borrowing pipeline is destroyed before its owning model pair; never owns an output sink.
    std::unique_ptr<DDSPC::ShadowPipeline> m_shadowPipeline;

    /// Explicit verified/synthetic timing adapter; no native source is armed by configuration or INDI.
    std::unique_ptr<DDSPC::CausalInputBuilder> m_shadowInputs;

    /// Preserve a rejected selection request in INDI until a subsequent accepted request.
    bool m_shadowModeRejected{ false };

    /// Preserve a rejected training request in INDI until a subsequent accepted request.
    bool m_shadowTrainRejected{ false };

    /// Enables the legacy accelerometer monitor; shadow features require explicit records.
    bool m_accelEnabled{ false };

    /// Number of accelerometer channels in each sample.
    int m_accelChannels{ 2 };

    /// Number of past accelerometer samples in the regressor.
    int m_accelHistory{ 20 };

    /// Enables online per-channel z-score normalization.
    bool m_accelNormalize{ true };

    /// Standard-deviation floor used by normalization.
    realT m_accelStdFloor{ 1.0e-4f };

    /// Optional post-normalization clip threshold in sigma units.
    realT m_accelClipSigma{ 0.0f };

    /// Accelerometer shmim width.
    uint32_t m_accelWidth{ 0 };

    /// Accelerometer shmim height.
    uint32_t m_accelHeight{ 0 };

    /// Most recent normalized accelerometer sample.
    DDSPC::Matrix m_latestAccelSample;

    /// True once at least one accel frame has been ingested.
    bool m_haveAccelSample{ false };

    /// Consecutive WFS frames without a received accelerometer frame.
    int m_accelMissingFrameCount{ 0 };

    /// WFS-frame grace period before disabling accel integration.
    int m_accelMissingFrameLimit{ 20 };

    /// Protects shared accelerometer sample/state access.
    std::mutex m_accelMutex;

    /// Running accelerometer mean for normalization.
    DDSPC::Matrix m_accelMean;

    /// Running second central moment for normalization.
    DDSPC::Matrix m_accelM2;

    /// Number of samples processed by normalization.
    uint64_t m_accelNormCount{ 0 };

    /// Tracks whether the accel shmim monitor was started.
    bool m_accelMonitorStarted{ false };

    /// Legacy candidate command; never treated as a verified applied input in shadow.
    DDSPC::Matrix m_newCommand;

    /// Legacy current modal residual vector.
    DDSPC::Matrix m_newMeasurement;

    /// Legacy output buffer, including optional accumulated candidates.
    DDSPC::Matrix m_fullCommand;

    /// Zero excitation vector for legacy prediction without exploration.
    DDSPC::Matrix m_zeroExpNoise;

    /// Owned legacy learner; absent in shadow execution and deleted at shutdown.
    DDSPC::PredictiveController *m_controller{ nullptr };

    // Process control parameters
    /// Legacy exploration-based adaptation request.
    bool m_isLearning{ false };

    /// Legacy canned training-sequence request.
    bool m_isStdLearning{ false };

    /// Legacy candidate selection flag; rejected in shadow execution.
    bool m_isPredictiveControl{ false };

    /// Legacy output accumulation flag; forced off at shadow configuration.
    bool m_isIntegrating{ false };

    /// Legacy output ownership policy; ignored in shadow execution.
    bool m_ownShmim{ true };

    /// Accumulated legacy processing duration in microseconds.
    double m_loopTimeElapsed{ 0.0 };

    //  Learning variables
    /// Active/inactive legacy exploration amplitudes, first sequence bank.
    std::vector<realT> m_exploration_noise_strength_01;

    /// Remaining samples per stage in the first sequence bank.
    std::vector<int> m_exploration_steps_01;

    /// Legacy penalties per stage in the first sequence bank.
    std::vector<realT> m_regularization_steps_01;

    /// Active/inactive legacy exploration amplitudes, second sequence bank.
    std::vector<realT> m_exploration_noise_strength_02;

    /// Remaining samples per stage in the second sequence bank.
    std::vector<int> m_exploration_steps_02;

    /// Legacy penalties per stage in the second sequence bank.
    std::vector<realT> m_regularization_steps_02;

    /// Pending swap between legacy exploration sequence banks.
    bool m_switchExploration{ false };

    /// Select the first legacy exploration sequence bank.
    bool m_useSet01{ true };

    /// Deferred legacy reset request, consumed by WFS processing.
    bool m_doResetModel{ false };

    /// Deferred legacy restore request, rejected in shadow execution.
    bool m_doTriggerLoad{ false };

    /// Deferred legacy persistence request, rejected in shadow execution.
    bool m_doTriggerSave{ false };

    /// RLS algorithm choice; causal shadow uses the Phase 1 QR correction.
    bool m_useQrd{ false };

    /// Deterministic legacy exploration random-number engine.
    std::default_random_engine m_generator;

    /// Legacy unit normal exploration distribution.
    std::normal_distribution<DDSPC::realT> m_distribution;

    /// Legacy comma-separated exploration-stage request text.
    std::string m_exploration_sequence{ "" };

    /// Legacy model-state filename, without the calibration directory.
    std::string m_filename{ "" };

    std::string m_fpsSource{ "camwfs" }; /**< Device name for getting fps of the loop.
                                              This device must have *.fps.current.  Default is camwfs*/

    /// The current fps
    realT m_fps{ 0 };

    ///@}

    /** \name frameGrabber Interface - Data
     * @{
     */

    /// Flag indicating that the commands have been updated
    bool m_updated{ false };

    /// Semaphore used to synchronize the fg thread and the process thread.
    sem_t m_smSemaphore{ 0 };

    ///@}

    /// Legacy exploration sequence request/status property.
    pcf::IndiProperty m_indiP_exploration;

    /// Legacy persistence filename request/status property.
    pcf::IndiProperty m_indiP_filename;

    /// Legacy exploration-learning request/status property.
    pcf::IndiProperty m_indiP_learningToggle;

    /// Legacy canned-learning request/status property.
    pcf::IndiProperty m_indiP_learningStdToggle;

    /// Legacy accumulation request/status property.
    pcf::IndiProperty m_indiP_integratingToggle;

    /// Legacy predictive-command request/status property.
    pcf::IndiProperty m_indiP_predictingToggle;

    /// Reset request for the legacy model or both independent shadow models.
    pcf::IndiProperty m_indiP_resetToggle;

    /// Legacy save request; rejected in shadow execution.
    pcf::IndiProperty m_indiP_saveToggle;

    /// Legacy restore request; rejected in shadow execution.
    pcf::IndiProperty m_indiP_loadToggle;

    /// Subscription to the configured WFS rate source.
    pcf::IndiProperty m_indiP_fpsSource;

    /// Published nominal loop rate; not an exposure-timing assertion.
    pcf::IndiProperty m_indiP_fps;

    /// Diagnostic predictor selection, independent of the legacy operational path.
    pcf::IndiProperty m_indiP_ddspcMode;

    /// Independent adaptation requests for both shadow models.
    pcf::IndiProperty m_indiP_ddspcTrain;

    /// Read-only startup execution mode; no runtime transition can attach the output publisher.
    pcf::IndiProperty m_indiP_ddspcShadow;

    /// Readiness, training, validity, and error of the independent standard predictor.
    pcf::IndiProperty m_indiP_ddspcStandard;

    /// Readiness, training, validity, and error of the independent assisted predictor.
    pcf::IndiProperty m_indiP_ddspcAssisted;

    /// Input-contract validity assertions; unverified live values remain false.
    pcf::IndiProperty m_indiP_ddspcData;

    /// Execution/output policy, per-model readiness, and explicit input synchronization/provenance diagnostics.
    pcf::IndiProperty m_indiP_ddspcStatus;

    /// Read-only local pipeline eligibility/counters and supervisory reasons; never an output enable.
    pcf::IndiProperty m_indiP_ddspcPipeline, m_indiP_ddspcSupervisor;

  public:
    /// Request the pipeline policy or select a ready direct shadow predictor; never change training/output authority.
    INDI_NEWCALLBACK_DECL( loPredCtrlAcc, m_indiP_ddspcMode );

    /// Independently enable/pause adaptation; no legacy learning/exploration flags are changed.
    INDI_NEWCALLBACK_DECL( loPredCtrlAcc, m_indiP_ddspcTrain );

    /// Request a legacy exploration sequence; rejected in shadow execution.
    INDI_NEWCALLBACK_DECL( loPredCtrlAcc, m_indiP_exploration );

    /// Update the legacy persistence filename.
    INDI_NEWCALLBACK_DECL( loPredCtrlAcc, m_indiP_filename );

    /// Toggle legacy exploration learning; rejected in shadow execution.
    INDI_NEWCALLBACK_DECL( loPredCtrlAcc, m_indiP_learningToggle );

    /// Toggle legacy canned learning; rejected in shadow execution.
    INDI_NEWCALLBACK_DECL( loPredCtrlAcc, m_indiP_learningStdToggle );

    /// Toggle legacy accumulation; rejected in shadow execution.
    INDI_NEWCALLBACK_DECL( loPredCtrlAcc, m_indiP_integratingToggle );

    /// Toggle legacy predictive commands; rejected in shadow execution.
    INDI_NEWCALLBACK_DECL( loPredCtrlAcc, m_indiP_predictingToggle );

    /// Reset legacy or both shadow models without enabling correction.
    INDI_NEWCALLBACK_DECL( loPredCtrlAcc, m_indiP_resetToggle );

    /// Request legacy persistence; rejected in shadow execution.
    INDI_NEWCALLBACK_DECL( loPredCtrlAcc, m_indiP_saveToggle );

    /// Request legacy restore; rejected in shadow execution.
    INDI_NEWCALLBACK_DECL( loPredCtrlAcc, m_indiP_loadToggle );

    /// Receive the nominal WFS rate without asserting frame alignment.
    INDI_SETCALLBACK_DECL( loPredCtrlAcc, m_indiP_fpsSource );

    /// Default c'tor.
    loPredCtrlAcc();

    /// D'tor, declared and defined for noexcept.
    ~loPredCtrlAcc() noexcept;

    /// Declare legacy and startup-only shadow configuration keys.
    virtual void setupConfig();

    /// Implementation of loadConfig logic, separated for testing.
    /** This is called by loadConfig().
     */
    int loadConfigImpl(
        mx::app::appConfigurator &_config /**< [in] an application configuration from which to load values*/ );

    /// Load startup configuration through loadConfigImpl.
    virtual void loadConfig();

    /// Start the WFS reader and selected execution path; shadow omits all output-owning helpers.
    virtual int appStartup();

    /// Implementation of the FSM for loPredCtrlAcc.
    /**
     * \returns 0 on no critical error
     * \returns -1 on an error requiring shutdown
     */
    virtual int appLogic();

    /// Stop only the helpers that were started and release owned learners.
    virtual int appShutdown();

    /** \name frameGrabber Interface
     * @{
     */

    /// Configure legacy output dimensions for frameGrabber; reject shadow acquisition.
    int configureAcquisition();

    /// Gets the frames-per-second readout rate
    /** Used for the latency statistics
     */
    float fps();

    /// Accept legacy acquisition startup; reject shadow acquisition.
    int startAcquisition();

    /// Wait for legacy output; return no frame immediately in shadow execution.
    int acquireAndCheckValid();

    /// Loads the commands into the stream
    int loadImageIntoStream( void *dest /**< [out] Legacy output buffer; untouched in shadow execution. */ );

    /// Take any actions needed to reconfigure the system.  Called if m_reconfig is set to true.
    int reconfig();

    ///@}

    /** \name telemeter Interface
     * @{
     */

    /// Check legacy frame-grabber timing telemetry deadlines.
    int checkRecordTimes();

    /// Record legacy timing telemetry; a no-op in shadow execution.
    int recordTelem( const telem_fgtimings *record /**< [in] Requested telemetry type. */ );

    ///@}

  protected:
    /// Allocate independent causal models from startup configuration, with training disabled.
    int configureShadow();

    /// Register read-only ADC diagnostics without attaching a stream.
    int setupAdcINDI();

    /// Register private observation-only counters without subscribing to operational control properties.
    int setupNativeShadowINDI();

    /// Copy native observer diagnostics outside its source/recording workers.
    void updateNativeShadowINDI();

    /// Admit a bounded-rate diagnostic update without changing acquisition or model state.
    bool nativeDiagnosticsDue( std::int64_t nowNs /**< [in] Current local steady-clock nanoseconds. */ );

    /// Select the compact native schema without changing the historical non-native test/controller interface.
    bool compactNativeINDI() const;

    /// Register the bounded operator schema; metadata is available only through explicit debug opt-in.
    int setupNativeOperatorINDI();

    /// Publish one coherent native snapshot at the diagnostic cadence; never qualify clocks or model inputs.
    void updateNativeOperatorINDI();

    /// Publish a bounded ADC health snapshot at ordinary application cadence.
    void updateAdcINDI();

    /// Register selection and read-only shadow diagnostics without any stream creation.
    int setupShadowINDI();

    /// Refresh model diagnostics; the caller holds m_indiMutex.
    void updateShadowINDI();

    /// Retain the Phase 2 pre-synchronized test seam, rejecting mixed use with the timing builder.
    /** Native telemetry must use the validated timing seam; no live applied-command adapter is provided. */
    bool processShadowRecord( const DDSPC::ShadowRecord &record /**< [in] Explicit synchronized record. */ );

    /// Establish a timing/source session and reset both fits, retaining independent training requests and selection.
    int
    configureShadowInputs( const DDSPC::InputConfig &settings /**< [in] Verified or labeled synthetic contract. */ );

    /// Accept an immutable timestamped WFS observation without generating a command.
    bool processShadowWfs( const DDSPC::TimedWfsRecord &record /**< [in] Validated exposure and residual. */,
                           std::int64_t                 nowNs /**< [in] Established monotonic operation time. */ );

    /// Accept a confirmed applied-input interval, never a requested or calculated command.
    bool processShadowApplied( const DDSPC::AppliedInputRecord &record /**< [in] Applied-input certificate. */,
                               std::int64_t nowNs /**< [in] Established monotonic operation time. */ );

    /// Invalidate stale evidence during silence; the future adapter must supply ticks in the established clock.
    bool expireShadowInputs( std::int64_t nowNs /**< [in] Established monotonic operation time. */ );

    /// Feed owned decoded ADC evidence only to the explicitly configured shadow pipeline.
    bool processShadowAdc( const DDSPC::ShadowAdcRecord &record /**< [in] Decoded sample with explicit mappings. */,
                           std::int64_t                  nowNs /**< [in] Established common monotonic time. */ );

    /// Read scalar requested/effective diagnostics under the same model/input lock.
    DDSPC::ShadowPipelineStatus shadowPipelineStatus();

    /// Copy a qualified in-memory candidate; this has no publication or command-application side effect.
    bool shadowCandidate( DDSPC::Matrix &output /**< [out] Pre-sized modal candidate; zeroed on unavailability. */ );

    /// Feed each paired record once to the two-model manager; caller holds m_shadowMutex.
    bool consumeShadowInputs( DDSPC::InputBuildResult result /**< [in] Pairing result and common discontinuity. */ );

    /// Allocate WFS shape and either legacy buffers or reset shadow histories.
    int allocate( const dev::shmimT &dummy /**< [in] tag to differentiate shmimMonitor parents.*/ );

    /// Process legacy residuals or report the incomplete shadow input contract.
    int processImage( void              *curr_src /**< [in] Pointer to the current WFS modal payload. */,
                      const dev::shmimT &dummy /**< [in] WFS monitor tag. */
    );

    /// Allocate accelerometer stream dimensions/state.
    int allocate( const accelShmimT &dummy /**< [in] tag to differentiate accelerometer shmim monitor parent.*/ );

    /// Ingest one accelerometer frame.
    int processImage( void              *curr_src /**< [in] Legacy accelerometer payload; ignored in shadow. */,
                      const accelShmimT &dummy /**< [in] tag to differentiate accelerometer shmim monitor parent.*/
    );

    /// Reset accelerometer sample and normalization buffers.
    void resetAccelTelemetryState();

    /// Apply optional online normalization to one accelerometer sample.
    void normalizeAccelSample( DDSPC::Matrix &sample /**< [in,out] Legacy channel sample. */ );

    /// Disable accelerometer integration and optionally rebuild legacy m_controller dimensions.
    void disableAccelIntegration( const std::string &reason /**< [in] Legacy fallback explanation. */,
                                  bool rebuildController = false /**< [in] Reallocate the legacy baseline. */ );

    /// Persist the legacy learner only; shadow persistence is not supported.
    void save( std::string directory /**< [in] Legacy state directory. */ );

    /// Restore the legacy learner only; shadow persistence is not supported.
    void load( std::string directory /**< [in] Legacy state directory. */ );
};

inline std::string accelShmimT::configSection()
{
    return "accelShmim";
}

inline std::string accelShmimT::indiPrefix()
{
    return "accel";
}

inline loPredCtrlAcc::~loPredCtrlAcc() noexcept = default;

inline void loPredCtrlAcc::save( std::string directory )
{
    if( m_controller )
        m_controller->save_state( directory );
}

inline void loPredCtrlAcc::load( std::string directory )
{
    if( m_controller )
        m_controller->load_state( directory );
}

inline loPredCtrlAcc::loPredCtrlAcc() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    accelShmimMonitorT::m_getExistingFirst = true;
    return;
}

inline void loPredCtrlAcc::setupConfig()
{
    shmimMonitorT::setupConfig( config );
    accelShmimMonitorT::setupConfig( config );
    FRAMEGRABBER_SETUP_CONFIG( config );
    TELEMETER_SETUP_CONFIG( config );

    for( const auto &entry : std::vector<std::pair<std::string, std::string>>{ { "enabled", "bool" },
                                                                               { "shmimName", "string" },
                                                                               { "profile", "string" },
                                                                               { "min_code", "uint32_t" },
                                                                               { "max_code", "uint32_t" },
                                                                               { "reject_rails", "bool" },
                                                                               { "capacity", "size_t" },
                                                                               { "max_age_us", "double" },
                                                                               { "max_interval_us", "double" },
                                                                               { "clock_tolerance_us", "double" },
                                                                               { "zero0", "double" },
                                                                               { "zero1", "double" },
                                                                               { "volts_per_count0", "double" },
                                                                               { "volts_per_count1", "double" },
                                                                               { "sensitivity0", "double" },
                                                                               { "sensitivity1", "double" },
                                                                               { "polarity0", "double" },
                                                                               { "polarity1", "double" } } )
    {
        config.add( "adc." + entry.first,
                    "",
                    "adc." + entry.first,
                    argType::Optional,
                    "adc",
                    entry.first,
                    false,
                    entry.second,
                    "Startup-only raw UINT16 input/calibration; requires shadow=true. See ADC_INPUT.md." );
    }

    config.add( "native_shadow.enabled",
                "",
                "native_shadow.enabled",
                argType::Optional,
                "native_shadow",
                "enabled",
                false,
                "bool",
                "Read-only native acquisition; default false in the base app, enabled by the installed entrypoint." );
    for( const auto &entry : { std::make_pair( "directory", "string" ),
                               { "duration_sec", "double" },
                               { "record", "bool" },
                               { "history_capacity", "size_t" },
                               { "poll_us", "unsigned" },
                               { "queue_capacity", "size_t" },
                               { "record_dm_commands", "bool" },
                               { "command_queue_capacity", "size_t" },
                               { "command_max_elements", "size_t" },
                               { "modal_index0", "size_t" },
                               { "modal_index1", "size_t" } } )
        config.add( std::string( "native_shadow." ) + entry.first,
                    "",
                    std::string( "native_shadow." ) + entry.first,
                    argType::Optional,
                    "native_shadow",
                    entry.first,
                    false,
                    entry.second,
                    "Private observation startup setting; never enables training or output." );
    config.add( "diagnostics.period_ms",
                "",
                "diagnostics.period_ms",
                argType::Optional,
                "diagnostics",
                "period_ms",
                false,
                "double",
                "Periodic INDI update interval, 50 to 10000 ms." );
    config.add( "diagnostics.debug_indi",
                "",
                "diagnostics.debug_indi",
                argType::Optional,
                "diagnostics",
                "debug_indi",
                false,
                "bool",
                "Startup-only detailed native INDI schema; default false." );
    for( const char *role : { "modal", "camera", "dm", "completion", "sdk" } )
        for( const char *field : { "stream", "acquisition_clock", "publication_clock" } )
        {
            const auto key = std::string( "native_shadow." ) + role + "_" + field;
            config.add( key,
                        "",
                        key,
                        argType::Optional,
                        "native_shadow",
                        std::string( role ) + "_" + field,
                        false,
                        "string",
                        "Existing-only input and explicit raw clock declaration; unknown is retained." );
        }
    config.add( "parameters.shadow",
                "",
                "parameters.shadow",
                argType::Optional,
                "parameters",
                "shadow",
                false,
                "bool",
                "Startup-only non-actuating DDSPC diagnostics; default false preserves legacy execution." );
    config.add( "parameters.shadow_pipeline",
                "",
                "parameters.shadow_pipeline",
                argType::Optional,
                "parameters",
                "shadow_pipeline",
                false,
                "bool",
                "Enable shared non-actuating input/replay pipeline; default false." );
    for( const auto &setting : { std::make_pair( "requested", "string" ),
                                 { "window_ns", "int64" },
                                 { "lag_ns", "int64" },
                                 { "max_gap_ns", "int64" },
                                 { "max_age_ns", "int64" },
                                 { "recovery_frames", "unsigned" },
                                 { "command_limit", "double" },
                                 { "max_step", "double" },
                                 { "adc_calibration", "uint64" },
                                 { "adc_capacity", "size_t" } } )
    {
        const auto key = std::string( "shadow_pipeline." ) + setting.first;
        config.add(
            key,
            "",
            key,
            argType::Optional,
            "shadow_pipeline",
            setting.first,
            false,
            setting.second,
            "Startup-only shared shadow policy; bounds and actual timing/calibration require an explicit contract." );
    }
    config.add( "parameters.shadow_min_updates",
                "",
                "parameters.shadow_min_updates",
                argType::Optional,
                "parameters",
                "shadow_min_updates",
                false,
                "uint64_t",
                "Minimum shadow updates; zero uses each independent model's feature count." );

    config.add( "parameters.fpsSource",
                "",
                "parameters.fpsSource",
                argType::Required,
                "parameters",
                "fpsSource",
                false,
                "string",
                "The device name for getting fps of the loop." );

    config.add( "parameters.gain",
                "",
                "parameters.gain",
                argType::Required,
                "parameters",
                "gain",
                false,
                "float",
                "Legacy additive current-WFS feedback gain; unused by the shadow reference law." );
    config.add( "parameters.copy_gain",
                "",
                "parameters.copy_gain",
                argType::Required,
                "parameters",
                "copy_gain",
                false,
                "float",
                "Legacy prediction-off residual-copy gain; requires legacy_copy_gain_compat=false." );
    config.add( "parameters.legacy_copy_gain_compat",
                "",
                "parameters.legacy_copy_gain_compat",
                argType::Optional,
                "parameters",
                "legacy_copy_gain_compat",
                false,
                "bool",
                "Default true retains historical copy_gain loading from gain; false honors copy_gain." );
    config.add( "parameters.regularization",
                "",
                "parameters.regularization",
                argType::Required,
                "parameters",
                "regularization",
                false,
                "float",
                "Legacy scaled control penalty; literal positive lambda for the shadow reference law." );
    config.add( "parameters.gamma",
                "",
                "parameters.gamma",
                argType::Required,
                "parameters",
                "gamma",
                false,
                "float",
                "The forgetting factor." );
    config.add( "parameters.covariance",
                "",
                "parameters.covariance",
                argType::Required,
                "parameters",
                "covariance",
                false,
                "float",
                "Initial information rho; P0=I/rho and QR R0=sqrt(rho)I." );

    config.add( "parameters.num_modes",
                "",
                "parameters.num_modes",
                argType::Required,
                "parameters",
                "num_modes",
                false,
                "int",
                "The number of modes that will be controlled through predictive control." );
    config.add( "parameters.history",
                "",
                "parameters.history",
                argType::Required,
                "parameters",
                "history",
                false,
                "int",
                "The number of past measurements for the prediction." );
    config.add( "parameters.future",
                "",
                "parameters.future",
                argType::Required,
                "parameters",
                "future",
                false,
                "int",
                "The number of future steps that are predicted." );

    config.add( "parameters.qrd",
                "",
                "parameters.qrd",
                argType::Required,
                "parameters",
                "qrd",
                false,
                "bool",
                "The use QRD-RLS or Classic RLS." );
    config.add( "parameters.own_shmim",
                "",
                "parameters.own_shmim",
                argType::Required,
                "parameters",
                "own_shmim",
                false,
                "bool",
                "Does the predictive control own the output shmim or not." );
    config.add( "parameters.is_integrating",
                "",
                "parameters.is_integrating",
                argType::Required,
                "parameters",
                "is_integrating",
                false,
                "bool",
                "Whether the control signal is integrated or not." );
    config.add( "parameters.accel_enabled",
                "",
                "parameters.accel_enabled",
                argType::Optional,
                "parameters",
                "accel_enabled",
                false,
                "bool",
                "Enable the legacy FLOAT accelerometer monitor, not a UINT16 acc000 decoder; inactive in shadow." );
    config.add( "parameters.accel_channels",
                "",
                "parameters.accel_channels",
                argType::Optional,
                "parameters",
                "accel_channels",
                false,
                "int",
                "Number of accelerometer channels." );
    config.add( "parameters.accel_history",
                "",
                "parameters.accel_history",
                argType::Optional,
                "parameters",
                "accel_history",
                false,
                "int",
                "Number of accelerometer history samples." );
    config.add( "parameters.accel_normalize",
                "",
                "parameters.accel_normalize",
                argType::Optional,
                "parameters",
                "accel_normalize",
                false,
                "bool",
                "Enable online accelerometer normalization." );
    config.add( "parameters.accel_std_floor",
                "",
                "parameters.accel_std_floor",
                argType::Optional,
                "parameters",
                "accel_std_floor",
                false,
                "float",
                "Standard deviation floor for normalization." );
    config.add( "parameters.accel_clip_sigma",
                "",
                "parameters.accel_clip_sigma",
                argType::Optional,
                "parameters",
                "accel_clip_sigma",
                false,
                "float",
                "Optional sigma clipping after normalization." );
    config.add( "parameters.accel_missing_frame_limit",
                "",
                "parameters.accel_missing_frame_limit",
                argType::Optional,
                "parameters",
                "accel_missing_frame_limit",
                false,
                "int",
                "Number of WFS frames to wait for accel before falling back to legacy mode." );
}

inline int loPredCtrlAcc::loadConfigImpl( mx::app::appConfigurator &_config )
{
    _config( m_shadowMode, "parameters.shadow" );
#ifdef DDSPC_SHADOW_ONLY
    if( !m_shadowMode )
    {
        m_shutdown = true;
        return log<software_error, -1>(
            { __FILE__, __LINE__, "Shadow-only build requires parameters.shadow=true; legacy output is forbidden." } );
    }
#endif
    _config( m_shadowPipelineEnabled, "parameters.shadow_pipeline" );
    if( m_shadowPipelineEnabled )
    {
        m_shadowPipelineConfigValid = false;
        if( !m_shadowMode )
        {
            m_shutdown = true;
            return log<software_error, -1>( { __FILE__, __LINE__, "Shadow pipeline requires shadow=true." } );
        }
        _config( m_shadowRequested, "shadow_pipeline.requested" );
        _config( m_shadowFeatures.m_windowNs, "shadow_pipeline.window_ns" );
        _config( m_shadowFeatures.m_lagNs, "shadow_pipeline.lag_ns" );
        _config( m_shadowFeatures.m_maxGapNs, "shadow_pipeline.max_gap_ns" );
        _config( m_shadowFeatures.m_maxAgeNs, "shadow_pipeline.max_age_ns" );
        _config( m_shadowPolicy.m_recoveryFrames, "shadow_pipeline.recovery_frames" );
        _config( m_shadowLimit, "shadow_pipeline.command_limit" );
        _config( m_shadowStep, "shadow_pipeline.max_step" );
        _config( m_shadowCalibration, "shadow_pipeline.adc_calibration" );
        _config( m_shadowAdcCapacity, "shadow_pipeline.adc_capacity" );
        try
        {
            if( m_shadowRequested != "standard" && m_shadowRequested != "accel_assisted" )
                throw std::invalid_argument( "Unknown shadow pipeline request" );
            if( !m_shadowCalibration || m_shadowAdcCapacity < 2 || m_shadowAdcCapacity > 65536 )
                throw std::invalid_argument( "Invalid shadow ADC identity/capacity" );
            m_shadowPolicy.m_limit   = m_shadowLimit;
            m_shadowPolicy.m_maxStep = m_shadowStep;
            DDSPC::ShadowPolicy validatePolicy( DDSPC::PolicyMode::Standard, m_shadowPolicy );
            DDSPC::acceleration_window( {}, 2000000000LL, m_shadowFeatures );
            m_shadowPipelineConfigValid = true;
        }
        catch( const std::exception &error )
        {
            m_shutdown = true;
            return log<software_error, -1>( { __FILE__, __LINE__, error.what() } );
        }
    }
    _config( m_nativeShadowEnabled, "native_shadow.enabled" );
    m_nativeShadowConfigValid = false;
    _config( m_adcEnabled, "adc.enabled" );
    if( m_adcEnabled && !m_shadowMode )
    {
        m_shutdown = true;
        return log<software_error, -1>( { __FILE__, __LINE__, "Native ADC observation requires shadow=true." } );
    }
    if( m_adcEnabled )
    {
        _config( m_adcConfig.m_stream, "adc.shmimName" );
        _config( m_adcConfig.m_profile, "adc.profile" );
        _config( m_adcConfig.m_minCode, "adc.min_code" );
        _config( m_adcConfig.m_maxCode, "adc.max_code" );
        _config( m_adcConfig.m_rejectRails, "adc.reject_rails" );
        _config( m_adcConfig.m_capacity, "adc.capacity" );
        _config( m_adcMaxAgeUs, "adc.max_age_us" );
        _config( m_adcMaxIntervalUs, "adc.max_interval_us" );
        _config( m_adcClockToleranceUs, "adc.clock_tolerance_us" );
        for( std::size_t axis = 0; axis < 2; ++axis )
        {
            const auto suffix = std::to_string( axis );
            _config( m_adcConfig.m_zero[axis], "adc.zero" + suffix );
            _config( m_adcConfig.m_voltsPerCount[axis], "adc.volts_per_count" + suffix );
            _config( m_adcConfig.m_sensitivity[axis], "adc.sensitivity" + suffix );
            _config( m_adcConfig.m_polarity[axis], "adc.polarity" + suffix );
        }
        try
        {
            if( !std::isfinite( m_adcMaxAgeUs ) || m_adcMaxAgeUs < .001 || m_adcMaxAgeUs > 1e9 ||
                !std::isfinite( m_adcMaxIntervalUs ) || m_adcMaxIntervalUs < .001 || m_adcMaxIntervalUs > 1e9 ||
                !std::isfinite( m_adcClockToleranceUs ) || m_adcClockToleranceUs < .001 || m_adcClockToleranceUs > 1e9 )
            {
                throw std::invalid_argument( "Invalid ADC microsecond limits" );
            }
            m_adcConfig.m_maxIntervalNs    = std::llround( m_adcMaxIntervalUs * 1000 );
            m_adcConfig.m_maxAgeNs         = std::llround( m_adcMaxAgeUs * 1000 );
            m_adcConfig.m_clockToleranceNs = std::llround( m_adcClockToleranceUs * 1000 );
            DDSPC::validate_adc_config( m_adcConfig );
        }
        catch( const std::exception &error )
        {
            m_shutdown = true;
            return log<software_error, -1>( { __FILE__, __LINE__, error.what() } );
        }
    }
    shmimMonitorT::loadConfig( _config );
    _config( m_shadowMinUpdates, "parameters.shadow_min_updates" );

    _config( m_accelEnabled, "parameters.accel_enabled" );
    if( m_accelEnabled && !m_shadowMode )
    {
        accelShmimMonitorT::loadConfig( _config );
    }
    else
    {
        accelShmimMonitorT::m_shmimName = "";
    }

    _config( m_fpsSource, "parameters.fpsSource" );

    _config( m_gainCtrl, "parameters.gain" );
    _config( m_legacyCopyGainCompat, "parameters.legacy_copy_gain_compat" );
    _config( m_copygainCtrl, "parameters.gain" );
    if( !m_legacyCopyGainCompat )
        _config( m_copygainCtrl, "parameters.copy_gain" );
    _config( m_regularizationCtrl, "parameters.regularization" );
    _config( m_gammaCtrl, "parameters.gamma" );
    _config( m_covarianceCtrl, "parameters.covariance" );

    _config( m_num_modes, "parameters.num_modes" );
    _config( m_history, "parameters.history" );
    _config( m_future, "parameters.future" );
    _config( m_useQrd, "parameters.qrd" );
    _config( m_ownShmim, "parameters.own_shmim" );
    _config( m_isIntegrating, "parameters.is_integrating" );
    _config( m_accelChannels, "parameters.accel_channels" );
    _config( m_accelHistory, "parameters.accel_history" );
    _config( m_accelNormalize, "parameters.accel_normalize" );
    _config( m_accelStdFloor, "parameters.accel_std_floor" );
    _config( m_accelClipSigma, "parameters.accel_clip_sigma" );
    _config( m_accelMissingFrameLimit, "parameters.accel_missing_frame_limit" );

    m_accelChannels = std::max( 0, m_accelChannels );
    m_accelHistory  = std::max( 0, m_accelHistory );
    if( m_accelMissingFrameLimit < 1 )
    {
        m_accelMissingFrameLimit = 1;
    }

    if( m_accelChannels == 0 || m_accelHistory == 0 )
    {
        m_accelEnabled = false;
    }

    if( !m_accelEnabled && !m_shadowMode )
    {
        m_accelChannels                 = 0;
        m_accelHistory                  = 0;
        accelShmimMonitorT::m_shmimName = "";
    }

    if( !m_shadowMode )
    {
        frameGrabberT::m_ownShmim = m_ownShmim;
        FRAMEGRABBER_LOAD_CONFIG( _config );
        TELEMETER_LOAD_CONFIG( _config );
    }
    else
    {
        m_isIntegrating = m_isPredictiveControl = m_isLearning = m_isStdLearning = false;
    }

    if( m_nativeShadowEnabled )
    {
        try
        {
#ifndef DDSPC_SHADOW_ONLY
            throw std::invalid_argument( "Native observation requires DDSPC_SHADOW_ONLY=yes" );
#endif
            if( !m_shadowMode || m_shadowPipelineEnabled || !m_adcEnabled || m_accelEnabled || m_ownShmim )
                throw std::invalid_argument(
                    "Native observation requires shadow, ADC, no pipeline/Legacy sensor/output ownership" );
            _config( m_nativeShadowConfig.m_directory, "native_shadow.directory" );
            _config( m_nativeShadowConfig.m_durationSec, "native_shadow.duration_sec" );
            _config( m_nativeShadowConfig.m_record, "native_shadow.record" );
            _config( m_nativeShadowConfig.m_historyCapacity, "native_shadow.history_capacity" );
            _config( m_nativeDiagnosticsMs, "diagnostics.period_ms" );
            const bool debugBefore = m_nativeDebugINDI;
            _config( m_nativeDebugINDI, "diagnostics.debug_indi" );
            if( m_nativeObserver && debugBefore != m_nativeDebugINDI )
            {
                m_nativeDebugINDI = debugBefore;
                throw std::invalid_argument( "Changing diagnostic INDI schema requires an application restart" );
            }
            if( !std::isfinite( m_nativeDiagnosticsMs ) || m_nativeDiagnosticsMs < 50 || m_nativeDiagnosticsMs > 10000 )
                throw std::invalid_argument( "Native diagnostic period must be between 50 and 10000 ms" );
            _config( m_nativeShadowConfig.m_pollUs, "native_shadow.poll_us" );
            _config( m_nativeShadowConfig.m_queueCapacity, "native_shadow.queue_capacity" );
            _config( m_nativeShadowConfig.m_recordDmCommands, "native_shadow.record_dm_commands" );
            _config( m_nativeShadowConfig.m_commandQueueCapacity, "native_shadow.command_queue_capacity" );
            _config( m_nativeShadowConfig.m_commandMaxElements, "native_shadow.command_max_elements" );
            _config( m_nativeShadowConfig.m_modalIndices[0], "native_shadow.modal_index0" );
            _config( m_nativeShadowConfig.m_modalIndices[1], "native_shadow.modal_index1" );
            m_nativeShadowConfig.m_streams[0].m_name = m_adcConfig.m_stream;
            for( std::size_t i = 1; i < DDSPC::observationSources; ++i )
            {
                auto      &source = m_nativeShadowConfig.m_streams[i];
                const auto key =
                    std::string( "native_shadow." ) + DDSPC::observation_source_name( DDSPC::ObservationSource( i ) );
                _config( source.m_name, key + "_stream" );
                _config( source.m_acquisitionClock, key + "_acquisition_clock" );
                _config( source.m_publicationClock, key + "_publication_clock" );
            }
            DDSPC::validate_native_observation( m_nativeShadowConfig );
            m_nativeShadowConfigValid = true;
        }
        catch( const std::exception &error )
        {
            m_shutdown = true;
            return log<software_error, -1>( { __FILE__, __LINE__, error.what() } );
        }
    }
    std::cout << "Gain " << m_gainCtrl << std::endl;
    std::cout << "Copy Gain " << m_copygainCtrl << std::endl;
    std::cout << "Regularization " << m_regularizationCtrl << std::endl;
    std::cout << "Covariance " << m_covarianceCtrl << std::endl;
    std::cout << "Gamma " << m_gammaCtrl << std::endl;
    std::cout << "Own shmim " << m_ownShmim << std::endl;

    std::cout << "num modes " << m_num_modes << std::endl;
    std::cout << "History " << m_history << std::endl;
    std::cout << "Future " << m_future << std::endl;
    std::cout << "Use QRD " << m_useQrd << std::endl;
    std::cout << "Accel enabled " << m_accelEnabled << std::endl;
    std::cout << "Accel channels " << m_accelChannels << std::endl;
    std::cout << "Accel history " << m_accelHistory << std::endl;

    std::cout << "Done reading config Impl." << std::endl;
    resetAccelTelemetryState();

    return 0;
}

inline void loPredCtrlAcc::loadConfig()
{
    loadConfigImpl( config );
}

inline int loPredCtrlAcc::configureShadow()
{
    if( !m_shadowMode )
        return -1;
#if defined( __FAST_MATH__ ) || ( defined( __FINITE_MATH_ONLY__ ) && __FINITE_MATH_ONLY__ )
    return log<software_error, -1>( { __FILE__, __LINE__, "Shadow DDSPC requires a complete -fno-fast-math build." } );
#endif
    DDSPC::ShadowConfig settings;
    settings.m_modes         = m_num_modes;
    settings.m_history       = m_history;
    settings.m_future        = m_future;
    settings.m_accelChannels = m_accelChannels;
    settings.m_accelHistory  = m_accelHistory;
    settings.m_gamma         = m_gammaCtrl;
    settings.m_prior         = m_covarianceCtrl;
    settings.m_lambda        = m_regularizationCtrl;
    settings.m_minUpdates    = m_shadowMinUpdates;
    settings.m_useQrd        = m_useQrd;
    try
    {
        { //mutex scope
            std::lock_guard<std::mutex> guard( m_shadowMutex );
            m_shadowPipeline.reset();
            m_shadowStopping       = false;
            m_shadowSettings       = settings;
            m_shadowPolicy.m_modes = settings.m_modes;
            m_shadowController     = std::make_unique<DDSPC::ShadowController>( settings );
            m_shadowInputs.reset();
        }
    }
    catch( const std::exception &e )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, e.what() } );
    }
    return 0;
}

inline bool loPredCtrlAcc::processShadowRecord( const DDSPC::ShadowRecord &record )
{
    if( m_nativeShadowEnabled || !m_shadowMode )
        return false;
    { //mutex scope
        std::lock_guard<std::mutex> guard( m_shadowMutex );
        return !m_shadowStopping && !m_shadowPipelineEnabled && !m_shadowInputs && m_shadowController &&
               m_shadowController->process( record );
    }
}

inline int loPredCtrlAcc::configureShadowInputs( const DDSPC::InputConfig &settings )
{
    if( m_nativeShadowEnabled || !m_shadowMode || settings.m_modes != m_num_modes ||
        settings.m_accelChannels != m_accelChannels )
        return -1;
    try
    {
        { //mutex scope
            std::lock_guard<std::mutex> guard( m_shadowMutex );
            if( !m_shadowController )
                return -1;
            if( m_shadowStopping )
                return -1;
            if( m_shadowPipelineEnabled )
            {
                auto pipeline = std::make_unique<DDSPC::ShadowPipeline>(
                    *m_shadowController,
                    m_shadowSettings,
                    settings,
                    m_shadowFeatures,
                    m_shadowPolicy,
                    m_shadowRequested == "accel_assisted" ? DDSPC::PolicyMode::Assisted : DDSPC::PolicyMode::Standard,
                    m_shadowCalibration,
                    m_shadowAdcCapacity );
                m_shadowController->reset( DDSPC::ShadowMode::Standard );
                m_shadowController->reset( DDSPC::ShadowMode::AccelAssisted );
                m_shadowController->invalidate( DDSPC::ShadowRecord{} );
                m_shadowInputs.reset();
                m_shadowPipeline = std::move( pipeline );
                return 0;
            }
            auto inputs = std::make_unique<DDSPC::CausalInputBuilder>( settings );
            // A new layout/timing contract may change the meaning of every fitted column.
            m_shadowController->reset( DDSPC::ShadowMode::Standard );
            m_shadowController->reset( DDSPC::ShadowMode::AccelAssisted );
            m_shadowController->invalidate( DDSPC::ShadowRecord{} );
            m_shadowInputs = std::move( inputs );
        }
    }
    catch( const std::exception &e )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, e.what() } );
    }
    return 0;
}

inline bool loPredCtrlAcc::consumeShadowInputs( DDSPC::InputBuildResult result )
{
    if( !m_shadowController )
        return false;
    if( result.m_discontinuity )
        m_shadowController->invalidate( DDSPC::ShadowRecord{} );
    bool accepted = result.m_accepted;
    for( const auto &paired : result.m_records )
        accepted = m_shadowController->process( paired.m_record ) && accepted;
    return accepted;
}

inline bool loPredCtrlAcc::processShadowWfs( const DDSPC::TimedWfsRecord &record, std::int64_t nowNs )
{
    if( m_nativeShadowEnabled || !m_shadowMode )
        return false;
    { //mutex scope
        std::lock_guard<std::mutex> guard( m_shadowMutex );
        if( m_shadowStopping )
            return false;
        if( m_shadowPipelineEnabled )
            return m_shadowPipeline && m_shadowPipeline->push_wfs( record, nowNs );
        return m_shadowInputs && consumeShadowInputs( m_shadowInputs->push_wfs( record, nowNs ) );
    }
}

inline bool loPredCtrlAcc::processShadowApplied( const DDSPC::AppliedInputRecord &record, std::int64_t nowNs )
{
    if( m_nativeShadowEnabled || !m_shadowMode )
        return false;
    { //mutex scope
        std::lock_guard<std::mutex> guard( m_shadowMutex );
        if( m_shadowStopping )
            return false;
        if( m_shadowPipelineEnabled )
            return m_shadowPipeline && m_shadowPipeline->push_command( record, nowNs );
        return m_shadowInputs && consumeShadowInputs( m_shadowInputs->push_command( record, nowNs ) );
    }
}

inline bool loPredCtrlAcc::expireShadowInputs( std::int64_t nowNs )
{
    if( !m_shadowMode )
        return false;
    { //mutex scope
        std::lock_guard<std::mutex> guard( m_shadowMutex );
        if( m_shadowStopping )
            return false;
        if( m_shadowPipelineEnabled )
            return m_shadowPipeline && m_shadowPipeline->advance( nowNs );
        return m_shadowInputs && consumeShadowInputs( m_shadowInputs->advance( nowNs ) );
    }
}

inline bool loPredCtrlAcc::processShadowAdc( const DDSPC::ShadowAdcRecord &record, std::int64_t nowNs )
{
    if( m_nativeShadowEnabled || !m_shadowMode || !m_shadowPipelineEnabled )
        return false;
    { //mutex scope
        std::lock_guard<std::mutex> guard( m_shadowMutex );
        return !m_shadowStopping && m_shadowPipeline && m_shadowPipeline->push_adc( record, nowNs );
    }
}

inline DDSPC::ShadowPipelineStatus loPredCtrlAcc::shadowPipelineStatus()
{
    std::lock_guard<std::mutex> guard( m_shadowMutex ); // mutex scope
    return m_shadowPipeline ? m_shadowPipeline->status() : DDSPC::ShadowPipelineStatus{};
}

inline bool loPredCtrlAcc::shadowCandidate( DDSPC::Matrix &output )
{
    std::lock_guard<std::mutex> guard( m_shadowMutex ); // mutex scope
    if( output.rows() != m_num_modes || output.cols() != 1 )
        return false;
    output.setZero();
    return m_shadowMode && !m_shadowStopping && m_shadowPipeline && m_shadowPipeline->candidate( output );
}

inline int loPredCtrlAcc::setupAdcINDI()
{
    if( compactNativeINDI() )
        return 0;
    if( createROIndiNumber( m_indiP_adc, "adc" ) < 0 )
    {
        return -1;
    }
    for( const char *element : { "enabled",       "valid",          "raw0",
                                 "raw1",          "acceleration0",  "acceleration1",
                                 "rate_hz",       "age_us",         "samples",
                                 "dropped",       "invalid",        "duplicates",
                                 "races",         "busy",           "restarts",
                                 "overwrites",    "timing_gaps",    "retained",
                                 "counter",       "generation",     "segment",
                                 "depth",         "width",          "height",
                                 "rail_rejected", "range_rejected", "timestamp_rejected" } )
    {
        m_indiP_adc.add( pcf::IndiElement( element, 0.0 ) );
    }
    if( registerIndiPropertyReadOnly( m_indiP_adc ) < 0 ||
        createROIndiText( m_indiP_adcSource, "adc_source", "state" ) < 0 )
    {
        return -1;
    }
    for( const char *element : { "stream",
                                 "profile",
                                 "datatype",
                                 "device",
                                 "inode",
                                 "acquired_real_ns",
                                 "written_real_ns",
                                 "received_mono_ns",
                                 "sample_counter",
                                 "counter" } )
    {
        m_indiP_adcSource.add( pcf::IndiElement( element ) );
    }
    return registerIndiPropertyReadOnly( m_indiP_adcSource );
}

inline void loPredCtrlAcc::updateAdcINDI()
{
    if( compactNativeINDI() )
        return;
    auto status = m_adcReader ? m_adcReader->status() : DDSPC::AdcStatus{};
    if( !m_adcEnabled )
    {
        status.m_state = DDSPC::AdcState::Disabled;
    }
    const auto &sample = status.m_latest;
    updatesIfChanged<double>( m_indiP_adc,
                              { "enabled",       "valid",          "raw0",
                                "raw1",          "acceleration0",  "acceleration1",
                                "rate_hz",       "age_us",         "samples",
                                "dropped",       "invalid",        "duplicates",
                                "races",         "busy",           "restarts",
                                "overwrites",    "timing_gaps",    "retained",
                                "counter",       "generation",     "segment",
                                "depth",         "width",          "height",
                                "rail_rejected", "range_rejected", "timestamp_rejected" },
                              { double( m_adcEnabled ),
                                double( status.m_valid ),
                                double( sample.m_frame.m_raw[0] ),
                                double( sample.m_frame.m_raw[1] ),
                                sample.m_acceleration[0],
                                sample.m_acceleration[1],
                                status.m_rateHz,
                                status.m_ageUs,
                                double( status.m_samples ),
                                double( status.m_dropped ),
                                double( status.m_invalid ),
                                double( status.m_duplicates ),
                                double( status.m_races ),
                                double( status.m_busy ),
                                double( status.m_restarts ),
                                double( status.m_overwrites ),
                                double( status.m_timingGaps ),
                                double( status.m_retained ),
                                double( status.m_source.m_counter ),
                                double( status.m_generation ),
                                double( status.m_segment ),
                                double( status.m_source.m_depth ),
                                double( status.m_source.m_width ),
                                double( status.m_source.m_height ),
                                double( status.m_railRejected ),
                                double( status.m_rangeRejected ),
                                double( status.m_timestampRejected ) },
                              status.m_valid ? INDI_OK : INDI_IDLE );
    updatesIfChanged<std::string>( m_indiP_adcSource,
                                   { "state",
                                     "stream",
                                     "profile",
                                     "datatype",
                                     "device",
                                     "inode",
                                     "acquired_real_ns",
                                     "written_real_ns",
                                     "received_mono_ns",
                                     "sample_counter",
                                     "counter" },
                                   { DDSPC::adc_state( status.m_state ),
                                     m_adcConfig.m_stream,
                                     m_adcConfig.m_profile,
                                     status.m_source.m_uint16 ? "UINT16" : "unverified",
                                     std::to_string( status.m_device ),
                                     std::to_string( status.m_inode ),
                                     std::to_string( sample.m_frame.m_acquiredRealNs ),
                                     std::to_string( sample.m_frame.m_writtenRealNs ),
                                     std::to_string( sample.m_receivedNs ),
                                     std::to_string( sample.m_frame.m_counter ),
                                     std::to_string( status.m_source.m_counter ) } );
}

inline int loPredCtrlAcc::setupNativeShadowINDI()
{
    if( compactNativeINDI() )
        return setupNativeOperatorINDI();
    if( createROIndiNumber( m_indiP_nativeShadow, "native_shadow" ) < 0 )
        return -1;
    for( const char *name : { "enabled",
                              "training_inhibited",
                              "exposure_qualified",
                              "applied_qualified",
                              "finished",
                              "io_error",
                              "written",
                              "queue_drops",
                              "adc_forwarded",
                              "adc_missing",
                              "adc_breaks",
                              "dm_values_enabled",
                              "dm_recording_enabled",
                              "dm_commands_retained",
                              "dm_history_evictions",
                              "dm_commands_written",
                              "dm_command_drops",
                              "dm_command_oversize",
                              "recording",
                              "queue_occupancy",
                              "command_queue_occupancy",
                              "elapsed_sec" } )
        m_indiP_nativeShadow.add( pcf::IndiElement( name, 0.0 ) );
    if( registerIndiPropertyReadOnly( m_indiP_nativeShadow ) < 0 )
        return -1;
    if( createROIndiText( m_indiP_adcForwarding, "native_adc_forwarding", "reason" ) < 0 )
        return -1;
    for( const char *field : { "reader_counter",
                               "cursor_counter",
                               "reader_generation",
                               "cursor_generation",
                               "reader_segment",
                               "cursor_segment",
                               "pending",
                               "pending_receipt_ns",
                               "cutoff_ns",
                               "reader_receipt_ns",
                               "forward_available_ns" } )
        m_indiP_adcForwarding.add( pcf::IndiElement( field, "" ) );
    if( registerIndiPropertyReadOnly( m_indiP_adcForwarding ) < 0 )
        return -1;
    for( std::size_t i = 0; i < DDSPC::observationSources; ++i )
    {
        auto      &property = m_indiP_nativeSources[i];
        const auto name = std::string( "native_" ) + DDSPC::observation_source_name( DDSPC::ObservationSource( i ) );
        if( createROIndiNumber( property, name ) < 0 )
            return -1;
        for( const char *element : { "connected",
                                     "valid",
                                     "idle",
                                     "new_sample",
                                     "stale",
                                     "received_rate_hz",
                                     "publication_rate_hz",
                                     "sample_age_us",
                                     "timestamp_metadata_valid",
                                     "timestamp_discontinuities",
                                     "clock_qualified",
                                     "history_retained",
                                     "history_capacity",
                                     "history_evictions",
                                     "samples",
                                     "counter",
                                     "generation",
                                     "counter_gaps",
                                     "restarts",
                                     "value0",
                                     "value1",
                                     "publication_age_us",
                                     "clock_unknown",
                                     "clock_mismatches",
                                     "invalid",
                                     "protocol_invalid",
                                     "protocol_incomplete",
                                     "sequence_gaps",
                                     "publisher_restarts",
                                     "association_failures",
                                     "metadata_associations",
                                     "busy",
                                     "races",
                                     "incompatible" } )
            property.add( pcf::IndiElement( element, 0.0 ) );
        if( registerIndiPropertyReadOnly( property ) < 0 )
            return -1;
        auto &identity = m_indiP_nativeIdentities[i];
        if( createROIndiText( identity, name + "_identity", "stream" ) < 0 )
            return -1;
        for( const char *element : { "state",
                                     "device",
                                     "file_inode",
                                     "counter",
                                     "generation",
                                     "image_inode",
                                     "acquired_raw_ns",
                                     "published_raw_ns",
                                     "received_mono_ns",
                                     "available_mono_ns",
                                     "acquisition_clock",
                                     "publication_clock" } )
            identity.add( pcf::IndiElement( element, "" ) );
        if( registerIndiPropertyReadOnly( identity ) < 0 )
            return -1;
    }
    if( createROIndiNumber( m_indiP_nativeSafety, "ddspc_safety" ) < 0 ||
        createROIndiNumber( m_indiP_nativeTiming, "ddspc_timing" ) < 0 ||
        createROIndiText( m_indiP_nativeInterlocks, "ddspc_interlocks", "output" ) < 0 )
        return -1;
    for( const char *name : { "shadow",
                              "active",
                              "dm_write_enabled",
                              "acquisition_enabled",
                              "diagnostics_enabled",
                              "training_enabled",
                              "native_prediction_enabled" } )
        m_indiP_nativeSafety.add( pcf::IndiElement( name, 0.0 ) );
    for( const char *name : { "adc_period_us",
                              "wfs_period_us",
                              "slots_per_wfs",
                              "clock_qualified",
                              "phase_valid",
                              "phase_offset_us",
                              "phase_jitter_us",
                              "synco_deadline_qualified" } )
        m_indiP_nativeTiming.add( pcf::IndiElement( name, 0.0 ) );
    for( const char *name : { "training", "timing", "dm_evidence", "requested_mode", "effective_mode" } )
        m_indiP_nativeInterlocks.add( pcf::IndiElement( name, "" ) );
    if( registerIndiPropertyReadOnly( m_indiP_nativeSafety ) < 0 ||
        registerIndiPropertyReadOnly( m_indiP_nativeTiming ) < 0 ||
        registerIndiPropertyReadOnly( m_indiP_nativeInterlocks ) < 0 )
        return -1;
    return 0;
}

inline bool loPredCtrlAcc::compactNativeINDI() const
{
    return m_nativeShadowEnabled && !m_nativeDebugINDI;
}

inline int loPredCtrlAcc::setupNativeOperatorINDI()
{
    auto add = [&]( NativeOperator index, const char *name, bool numeric, std::initializer_list<const char *> elements )
    {
        auto &p = m_indiP_nativeOperator[std::size_t( index )];
        if( numeric ? createROIndiNumber( p, name ) < 0 : createROIndiText( p, name, *elements.begin() ) < 0 )
            return -1;
        for( const auto *element : elements )
        {
            if( numeric )
                p.add( pcf::IndiElement( element, 0.0 ) );
            else if( !p.find( element ) )
                p.add( pcf::IndiElement( element, "" ) );
        }
        return registerIndiPropertyReadOnly( p );
    };
    if( add(
            NativeOperator::Health, "app_health", false, { "state", "uptime_s", "inputs", "shadow", "error_events" } ) <
            0 ||
        add( NativeOperator::Inputs, "input_health", false, { "adc", "modal", "camera", "dm" } ) < 0 ||
        add( NativeOperator::Rates,
             "input_rates",
             true,
             { "adc_capture_hz", "adc_forward_hz", "modal_hz", "camera_hz", "dm_hz" } ) < 0 ||
        add( NativeOperator::Adc,
             "adc_values",
             true,
             { "raw0", "raw1", "acceleration0", "acceleration1", "invalid", "rail_rejected" } ) < 0 ||
        add( NativeOperator::Losses,
             "input_losses",
             true,
             { "adc_dropped",
               "adc_forward_missing",
               "modal_gaps",
               "camera_gaps",
               "dm_gaps",
               "counter_discontinuities",
               "reader_races",
               "adc_breaks",
               "timing_breaks" } ) < 0 ||
        add( NativeOperator::Buffers,
             "input_buffers",
             false,
             { "history_occupancy",
               "history_evictions",
               "adc_forward_pending",
               "recorder_backlog",
               "recorder_overflow" } ) < 0 ||
        add( NativeOperator::Timing,
             "timing",
             false,
             { "phase_offset_us", "phase_jitter_us", "clock", "exposure", "modal_association", "slots_per_wfs" } ) <
            0 ||
        add( NativeOperator::Control,
             "ddspc_control",
             false,
             { "requested_mode", "effective_mode", "training", "prediction", "standard_ready", "assisted_ready" } ) <
            0 ||
        add( NativeOperator::Performance,
             "ddspc_performance",
             true,
             { "standard_updates",
               "standard_predictions",
               "standard_rmse",
               "assisted_updates",
               "assisted_predictions",
               "assisted_rmse",
               "compute_us" } ) < 0 ||
        add( NativeOperator::Safety,
             "safety",
             false,
             { "dm_write_enabled", "output_reason", "timing_interlock", "model_interlock" } ) < 0 )
        return -1;
    return 0;
}

inline void loPredCtrlAcc::updateNativeOperatorINDI()
{
    const auto          s       = m_nativeObserver->status();
    const auto          adc     = m_adcReader->status();
    const auto         &sources = s.m_sources;
    const auto         &forward = sources[0];
    DDSPC::ShadowStatus standard, assisted;
    { //mutex scope
        std::lock_guard<std::mutex> guard( m_shadowMutex );
        if( m_shadowController )
        {
            standard = m_shadowController->status( DDSPC::ShadowMode::Standard );
            assisted = m_shadowController->status( DDSPC::ShadowMode::AccelAssisted );
        }
    }
    const bool adcGood   = adc.m_valid && forward.m_valid;
    const bool inputGood = adcGood && sources[1].m_valid && sources[2].m_valid;
    auto       text =
        [&]( NativeOperator index, const std::vector<const char *> &keys, const std::vector<std::string> &values )
    { updatesIfChanged<std::string>( m_indiP_nativeOperator[std::size_t( index )], keys, values ); };
    auto number = [&]( NativeOperator index, const std::vector<const char *> &keys, const std::vector<double> &values )
    { updatesIfChanged<double>( m_indiP_nativeOperator[std::size_t( index )], keys, values ); };
    std::uint64_t errors =
        adc.m_dropped + s.m_adcMissing + s.m_queueDrops + s.m_commandDrops + s.m_commandOversize + s.m_ioError;
    std::uint64_t races = adc.m_races, resets = adc.m_restarts;
    std::uint64_t timingBreaks = adc.m_timingGaps + adc.m_timestampRejected;
    for( std::size_t i = 1; i < DDSPC::observationSources; ++i )
    {
        errors += sources[i].m_counterGaps + sources[i].m_invalid + sources[i].m_protocolInvalid;
        races += sources[i].m_races;
        resets += sources[i].m_restarts;
        timingBreaks += sources[i].m_timestampDiscontinuities;
    }
    text( NativeOperator::Health,
          { "state", "uptime_s", "inputs", "shadow", "error_events" },
          { s.m_ioError ? "worker_error" : "observing",
            s.m_nowNs >= s.m_startedNs && s.m_startedNs >= 0
                ? std::to_string( ( s.m_nowNs - s.m_startedNs ) / 1000000000 )
                : "unavailable",
            inputGood ? "adc_wfs_healthy_dm_independent" : "adc_wfs_degraded",
            "true",
            std::to_string( errors ) } );
    for( std::size_t i = 0; i < 4; ++i )
    {
        const auto &v     = sources[i];
        std::string state = !v.m_connected ? "disconnected"
                            : v.m_stale    ? "stale_or_invalid"
                            : v.m_idle     ? "idle_valid"
                            : v.m_valid    ? "valid"
                                           : "awaiting_sample";
        if( i == 0 && adc.m_valid && !v.m_valid )
            state = !v.m_samples ? "awaiting_first_forward"
                    : v.m_stale  ? "reader_healthy_forwarding_stale"
                                 : "reader_healthy_forwarding_invalid";
        if( i == 0 && !adc.m_valid )
            state = std::string( "reader_" ) + DDSPC::adc_state( adc.m_state );
        if( i == 3 && v.m_idle && !v.m_valid )
            state = "idle_no_command_yet";
        state += ";restarts=" + std::to_string( i == 0 ? adc.m_restarts : v.m_restarts );
        text( NativeOperator::Inputs, { DDSPC::observation_source_name( DDSPC::ObservationSource( i ) ) }, { state } );
        if( state != m_nativeReportedHealth[i] )
        {
            log<text_log>( std::string( "native_input_health source=" ) + m_nativeShadowConfig.m_streams[i].m_name +
                           " state=" + state + " counter=" + std::to_string( v.m_latest.m_counter ) +
                           " generation=" + std::to_string( v.m_latest.m_generation ) +
                           " segment=" + std::to_string( v.m_latest.m_segment ) +
                           " adc_cursor=" + std::to_string( s.m_adcCursor.m_counter ) +
                           " adc_pending=" + std::to_string( s.m_adcPending ) +
                           " copy_cutoff_mono_ns=" + std::to_string( s.m_adcCutoffNs ) +
                           " pending_receipt_mono_ns=" + std::to_string( s.m_adcPendingReceiptNs ) );
            m_nativeReportedHealth[i] = state;
        }
        m_nativeReportedConnections[i] = v.m_connected;
    }
    number( NativeOperator::Rates,
            { "adc_capture_hz", "adc_forward_hz", "modal_hz", "camera_hz", "dm_hz" },
            { adc.m_rateHz,
              forward.m_receivedRateHz,
              sources[1].m_receivedRateHz,
              sources[2].m_receivedRateHz,
              sources[3].m_publicationRateHz } );
    number( NativeOperator::Adc,
            { "raw0", "raw1", "acceleration0", "acceleration1", "invalid", "rail_rejected" },
            { double( adc.m_latest.m_frame.m_raw[0] ),
              double( adc.m_latest.m_frame.m_raw[1] ),
              adc.m_latest.m_acceleration[0],
              adc.m_latest.m_acceleration[1],
              double( adc.m_invalid ),
              double( adc.m_railRejected ) } );
    number( NativeOperator::Losses,
            { "adc_dropped",
              "adc_forward_missing",
              "modal_gaps",
              "camera_gaps",
              "dm_gaps",
              "counter_discontinuities",
              "reader_races",
              "adc_breaks",
              "timing_breaks" },
            { double( adc.m_dropped ),
              double( s.m_adcMissing ),
              double( sources[1].m_counterGaps ),
              double( sources[2].m_counterGaps ),
              double( sources[3].m_counterGaps ),
              double( resets ),
              double( races ),
              double( s.m_adcBreaks ),
              double( timingBreaks ) } );
    std::string occupancy, evictions;
    for( std::size_t i = 0; i < 4; ++i )
    {
        const auto prefix =
            std::string( i ? ";" : "" ) + DDSPC::observation_source_name( DDSPC::ObservationSource( i ) ) + "=";
        occupancy += prefix + std::to_string( i == 3 ? s.m_dmCommandsRetained : sources[i].m_retained ) + "/" +
                     std::to_string( i == 3 ? m_nativeShadowConfig.m_commandQueueCapacity
                                            : m_nativeShadowConfig.m_historyCapacity );
        evictions += prefix + std::to_string( i == 3 ? s.m_dmCommandEvictions : sources[i].m_historyEvictions );
    }
    text( NativeOperator::Buffers,
          { "history_occupancy", "history_evictions", "adc_forward_pending", "recorder_backlog", "recorder_overflow" },
          { occupancy,
            evictions,
            std::to_string( s.m_adcPending ),
            "metadata=" + std::to_string( s.m_queueOccupancy ) +
                ";commands=" + std::to_string( s.m_commandQueueOccupancy ),
            "metadata=" + std::to_string( s.m_queueDrops ) + ";commands=" + std::to_string( s.m_commandDrops ) } );
    text( NativeOperator::Timing,
          { "phase_offset_us", "phase_jitter_us", "clock", "exposure", "modal_association", "slots_per_wfs" },
          { "unqualified",
            "unqualified",
            "source_host_mapping_unqualified",
            "exposure_boundaries_unavailable",
            "camera_origin_and_exposure_contract_missing",
            "4_nominal_individual_adc_samples" } );
    text( NativeOperator::Control,
          { "requested_mode", "effective_mode", "training", "prediction", "standard_ready", "assisted_ready" },
          { m_shadowRequested,
            "unavailable",
            "inhibited",
            "inhibited",
            standard.m_ready ? "true" : "false",
            assisted.m_ready ? "true" : "false" } );
    number( NativeOperator::Performance,
            { "standard_updates",
              "standard_predictions",
              "standard_rmse",
              "assisted_updates",
              "assisted_predictions",
              "assisted_rmse",
              "compute_us" },
            { double( standard.m_updates ),
              double( standard.m_predictions ),
              standard.m_predictions ? standard.m_rmse : -1,
              double( assisted.m_updates ),
              double( assisted.m_predictions ),
              assisted.m_predictions ? assisted.m_rmse : -1,
              -1 } );
    text( NativeOperator::Safety,
          { "dm_write_enabled", "output_reason", "timing_interlock", "model_interlock" },
          { "false",
            "compile_time_shadow_only",
            "exposure_clock_and_applied_input_unqualified",
            "native_training_inhibited" } );
}

inline bool loPredCtrlAcc::nativeDiagnosticsDue( std::int64_t nowNs )
{
    const auto interval = std::int64_t( m_nativeDiagnosticsMs * 1000000 );
    if( nowNs < 0 || nowNs < m_nativeNextDiagnosticNs || nowNs > std::numeric_limits<std::int64_t>::max() - interval )
        return false;
    m_nativeNextDiagnosticNs = nowNs + interval;
    return true;
}

inline void loPredCtrlAcc::updateNativeShadowINDI()
{
    if( !m_nativeShadowEnabled || !m_nativeObserver )
        return;
    if( compactNativeINDI() )
    {
        updateNativeOperatorINDI();
        return;
    }
    const auto        s      = m_nativeObserver->status();
    const auto        adc    = m_adcReader->status();
    const std::string reason = !adc.m_valid                                ? "reader_invalid"
                               : s.m_adcPendingReceiptNs > s.m_adcCutoffNs ? "receipt_beyond_copy_cutoff"
                               : s.m_adcPending                            ? "bounded_forward_backlog"
                               : s.m_sources[0].m_stale                    ? "reader_healthy_forwarding_stale"
                                                                           : "up_to_date";
    updatesIfChanged<std::string>( m_indiP_adcForwarding,
                                   { "reason",
                                     "reader_counter",
                                     "cursor_counter",
                                     "reader_generation",
                                     "cursor_generation",
                                     "reader_segment",
                                     "cursor_segment",
                                     "pending",
                                     "pending_receipt_ns",
                                     "cutoff_ns",
                                     "reader_receipt_ns",
                                     "forward_available_ns" },
                                   { reason,
                                     std::to_string( adc.m_latest.m_frame.m_counter ),
                                     std::to_string( s.m_adcCursor.m_counter ),
                                     std::to_string( adc.m_generation ),
                                     std::to_string( s.m_adcCursor.m_generation ),
                                     std::to_string( adc.m_segment ),
                                     std::to_string( s.m_adcCursor.m_segment ),
                                     std::to_string( s.m_adcPending ),
                                     std::to_string( s.m_adcPendingReceiptNs ),
                                     std::to_string( s.m_adcCutoffNs ),
                                     std::to_string( adc.m_latest.m_receivedNs ),
                                     std::to_string( s.m_sources[0].m_latest.m_availableNs ) } );
    updatesIfChanged<double>(
        m_indiP_nativeShadow,
        { "enabled",
          "training_inhibited",
          "exposure_qualified",
          "applied_qualified",
          "finished",
          "io_error",
          "written",
          "queue_drops",
          "adc_forwarded",
          "adc_missing",
          "adc_breaks",
          "dm_values_enabled",
          "dm_recording_enabled",
          "dm_commands_retained",
          "dm_history_evictions",
          "dm_commands_written",
          "dm_command_drops",
          "dm_command_oversize",
          "recording",
          "queue_occupancy",
          "command_queue_occupancy",
          "elapsed_sec" },
        { 1,
          1,
          0,
          0,
          double( s.m_finished ),
          double( s.m_ioError ),
          double( s.m_written ),
          double( s.m_queueDrops ),
          double( s.m_sources[0].m_samples ),
          double( s.m_adcMissing ),
          double( s.m_adcBreaks ),
          1,
          double( m_nativeShadowConfig.m_recordDmCommands ),
          double( s.m_dmCommandsRetained ),
          double( s.m_dmCommandEvictions ),
          double( s.m_commandsWritten ),
          double( s.m_commandDrops ),
          double( s.m_commandOversize ),
          double( m_nativeShadowConfig.m_record ),
          double( s.m_queueOccupancy ),
          double( s.m_commandQueueOccupancy ),
          s.m_nowNs >= s.m_startedNs && s.m_startedNs >= 0 ? double( s.m_nowNs - s.m_startedNs ) / 1e9 : -1 } );
    for( std::size_t i = 0; i < DDSPC::observationSources; ++i )
    {
        const auto &v = s.m_sources[i];
        updatesIfChanged<double>(
            m_indiP_nativeSources[i],
            { "connected",
              "valid",
              "idle",
              "new_sample",
              "stale",
              "received_rate_hz",
              "publication_rate_hz",
              "sample_age_us",
              "timestamp_metadata_valid",
              "timestamp_discontinuities",
              "clock_qualified",
              "history_retained",
              "history_capacity",
              "history_evictions",
              "samples",
              "counter",
              "generation",
              "counter_gaps",
              "restarts",
              "value0",
              "value1",
              "publication_age_us",
              "clock_unknown",
              "clock_mismatches",
              "invalid",
              "protocol_invalid",
              "protocol_incomplete",
              "sequence_gaps",
              "publisher_restarts",
              "association_failures",
              "metadata_associations",
              "busy",
              "races",
              "incompatible" },
            { double( v.m_connected ),
              double( v.m_valid ),
              double( v.m_idle ),
              double( v.m_samples > m_nativeReportedSamples[i] ),
              double( v.m_stale ),
              v.m_receivedRateHz,
              v.m_publicationRateHz,
              v.m_lastReceiptNs >= 0 && s.m_nowNs >= v.m_lastReceiptNs ? double( s.m_nowNs - v.m_lastReceiptNs ) / 1000
                                                                       : -1,
              double( v.m_latest.m_event == DDSPC::ObservationEvent::Sample &&
                      !( v.m_latest.m_flags & ( DDSPC::ObservationInvalidTime | DDSPC::ObservationClockMismatch |
                                                DDSPC::ObservationTimestampDiscontinuity ) ) ),
              double( v.m_timestampDiscontinuities ),
              0, // A declared source clock is not verified source-host/exposure provenance.
              i == 3 ? double( s.m_dmCommandsRetained ) : double( v.m_retained ),
              i == 3  ? double( m_nativeShadowConfig.m_commandQueueCapacity )
              : i < 4 ? double( m_nativeShadowConfig.m_historyCapacity )
                      : 0,
              i == 3 ? double( s.m_dmCommandEvictions ) : double( v.m_historyEvictions ),
              double( v.m_samples ),
              double( v.m_latest.m_counter ),
              double( v.m_latest.m_generation ),
              double( v.m_counterGaps ),
              double( v.m_restarts ),
              v.m_latest.m_values[0],
              v.m_latest.m_values[1],
              v.m_latest.m_publicationAgeNs >= 0 ? double( v.m_latest.m_publicationAgeNs ) / 1000 : -1,
              double( v.m_clockUnknown ),
              double( v.m_clockMismatches ),
              double( v.m_invalid ),
              double( v.m_protocolInvalid ),
              double( v.m_protocolIncomplete ),
              double( v.m_sequenceGaps ),
              double( v.m_publisherRestarts ),
              double( v.m_associationFailures ),
              double( v.m_associations ),
              double( v.m_busy ),
              double( v.m_races ),
              double( v.m_incompatible ) } );
        const auto &r      = v.m_latest;
        const auto &source = m_nativeShadowConfig.m_streams[i];
        if( !source.m_name.empty() && v.m_connected != m_nativeReportedConnections[i] )
            log<text_log>(
                std::string( "Native input " ) + source.m_name +
                    ( v.m_connected ? " attached (observation only)." : " disconnected; retrying read-only." ),
                v.m_connected ? logPrio::LOG_NOTICE : logPrio::LOG_WARNING );
        m_nativeReportedConnections[i] = v.m_connected;
        updatesIfChanged<std::string>( m_indiP_nativeIdentities[i],
                                       { "stream",
                                         "state",
                                         "device",
                                         "file_inode",
                                         "counter",
                                         "generation",
                                         "image_inode",
                                         "acquired_raw_ns",
                                         "published_raw_ns",
                                         "received_mono_ns",
                                         "available_mono_ns",
                                         "acquisition_clock",
                                         "publication_clock" },
                                       { source.m_name,
                                         !v.m_connected ? "disconnected"
                                         : v.m_stale    ? "invalid_or_stale"
                                         : v.m_idle     ? "idle"
                                         : v.m_valid    ? "updating"
                                                        : "awaiting_first_sample",
                                         std::to_string( r.m_device ),
                                         std::to_string( r.m_fileInode ),
                                         std::to_string( r.m_counter ),
                                         std::to_string( r.m_generation ),
                                         std::to_string( r.m_imageInode ),
                                         std::to_string( r.m_acquiredRawNs ),
                                         std::to_string( r.m_publishedRawNs ),
                                         std::to_string( r.m_receivedNs ),
                                         std::to_string( r.m_availableNs ),
                                         source.m_acquisitionClock,
                                         source.m_publicationClock } );
        m_nativeReportedSamples[i] = v.m_samples;
    }
    updatesIfChanged<double>( m_indiP_nativeSafety,
                              { "shadow",
                                "active",
                                "dm_write_enabled",
                                "acquisition_enabled",
                                "diagnostics_enabled",
                                "training_enabled",
                                "native_prediction_enabled" },
                              { 1, 0, 0, double( m_adcEnabled && m_nativeShadowEnabled ), 1, 0, 0 } );
    updatesIfChanged<double>( m_indiP_nativeTiming,
                              { "adc_period_us",
                                "wfs_period_us",
                                "slots_per_wfs",
                                "clock_qualified",
                                "phase_valid",
                                "phase_offset_us",
                                "phase_jitter_us",
                                "synco_deadline_qualified" },
                              { 125, 500, 4, 0, 0, -1, -1, 0 } );
    updatesIfChanged<std::string>(
        m_indiP_nativeInterlocks,
        { "output", "training", "timing", "dm_evidence", "requested_mode", "effective_mode" },
        { "compile_time_shadow_only",
          "exposure_and_applied_input_contract_unqualified",
          "source_clock_and_phase_reference_unqualified",
          "command_observation_not_physical_application",
          m_shadowRequested,
          "unavailable" } );
}

inline int loPredCtrlAcc::setupShadowINDI()
{
    if( createStandardIndiSelectionSw( m_indiP_ddspcMode,
                                       "ddspc_mode",
                                       { "standard", "accel_assisted" },
                                       "Shadow predictor selection",
                                       "DDSPC Shadow" ) < 0 )
        return -1;
    m_indiP_ddspcMode["standard"].setSwitchState( pcf::IndiElement::On );
    if( !m_shadowMode )
        m_indiP_ddspcMode.setPerm( pcf::IndiProperty::ReadOnly );
    if( registerIndiPropertyNew( m_indiP_ddspcMode, INDI_NEWCALLBACK( m_indiP_ddspcMode ) ) < 0 )
        return -1;

    if( compactNativeINDI() )
        return 0;

    if( createStandardIndiSelectionSw( m_indiP_ddspcTrain,
                                       "ddspc_train",
                                       { "standard", "accel_assisted" },
                                       "Independent shadow adaptation",
                                       "DDSPC Shadow" ) < 0 )
        return -1;
    m_indiP_ddspcTrain.setRule( pcf::IndiProperty::AnyOfMany );
    if( !m_shadowMode || m_nativeShadowEnabled )
        m_indiP_ddspcTrain.setPerm( pcf::IndiProperty::ReadOnly );
    if( registerIndiPropertyNew( m_indiP_ddspcTrain, INDI_NEWCALLBACK( m_indiP_ddspcTrain ) ) < 0 )
        return -1;

    if( createStandardIndiToggleSw(
            m_indiP_ddspcShadow, "ddspc_shadow", "Startup-only shadow execution", "DDSPC Shadow" ) < 0 )
        return -1;
    m_indiP_ddspcShadow.setPerm( pcf::IndiProperty::ReadOnly );
    m_indiP_ddspcShadow["toggle"].setSwitchState( m_shadowMode ? pcf::IndiElement::On : pcf::IndiElement::Off );
    if( registerIndiPropertyReadOnly( m_indiP_ddspcShadow ) < 0 )
        return -1;

    for( auto entry : { std::make_pair( &m_indiP_ddspcStandard, "ddspc_standard" ),
                        std::make_pair( &m_indiP_ddspcAssisted, "ddspc_accel_assisted" ) } )
    {
        if( createROIndiNumber( *entry.first, entry.second ) < 0 )
            return -1;
        for( const char *name : { "ready",
                                  "training_enabled",
                                  "training",
                                  "data_valid",
                                  "prediction_valid",
                                  "rmse",
                                  "updates",
                                  "predictions",
                                  "samples",
                                  "feature_rank",
                                  "features" } )
        {
            entry.first->add( pcf::IndiElement( name, 0.0 ) );
        }
        if( registerIndiPropertyReadOnly( *entry.first ) < 0 )
            return -1;
    }
    if( createROIndiNumber( m_indiP_ddspcData, "ddspc_data" ) < 0 )
        return -1;
    for( const char *name : { "wfs_valid", "applied_valid", "timing_valid", "accel_valid" } )
    {
        m_indiP_ddspcData.add( pcf::IndiElement( name, 0.0 ) );
    }
    if( registerIndiPropertyReadOnly( m_indiP_ddspcData ) < 0 )
        return -1;
    if( createROIndiText( m_indiP_ddspcStatus, "ddspc_status", "execution" ) < 0 )
        return -1;
    for( const char *name :
         { "output", "selected", "standard_reason", "accel_assisted_reason", "input_reason", "input_provenance" } )
    {
        m_indiP_ddspcStatus.add( pcf::IndiElement( name, "" ) );
    }
    if( registerIndiPropertyReadOnly( m_indiP_ddspcStatus ) < 0 )
        return -1;
    if( m_shadowPipelineEnabled )
    {
        if( createROIndiNumber( m_indiP_ddspcPipeline, "ddspc_pipeline" ) < 0 )
            return -1;
        for( const char *name : { "configured",
                                  "candidate_valid",
                                  "common_valid",
                                  "qualified_streak",
                                  "decisions",
                                  "fallbacks",
                                  "recoveries",
                                  "index",
                                  "feature_age_us" } )
            m_indiP_ddspcPipeline.add( pcf::IndiElement( name, 0.0 ) );
        if( registerIndiPropertyReadOnly( m_indiP_ddspcPipeline ) < 0 )
            return -1;
        if( createROIndiText( m_indiP_ddspcSupervisor, "ddspc_supervisor", "requested" ) < 0 )
            return -1;
        for( const char *name : { "effective", "reason", "feature_reason" } )
            m_indiP_ddspcSupervisor.add( pcf::IndiElement( name, "" ) );
        if( registerIndiPropertyReadOnly( m_indiP_ddspcSupervisor ) < 0 )
            return -1;
    }
    updateShadowINDI();
    return 0;
}

inline void loPredCtrlAcc::updateShadowINDI()
{
    if( compactNativeINDI() )
    {
        indi::updateSelectionSwitchIfChanged(
            m_indiP_ddspcMode, m_shadowRequested, m_indiDriver, m_shadowModeRejected ? INDI_ALERT : INDI_IDLE );
        auto &control = m_indiP_nativeOperator[std::size_t( NativeOperator::Control )];
        if( !control.getName().empty() )
            updatesIfChanged<std::string>( control, { "requested_mode" }, { m_shadowRequested } );
        return;
    }
    DDSPC::ShadowPipelineStatus pipeline;
    bool                        pipelineConfigured = false;
    DDSPC::ShadowStatus         standard, assisted;
    DDSPC::ShadowRecord         validity;
    DDSPC::ShadowMode           selected      = DDSPC::ShadowMode::Standard;
    DDSPC::InputReason          inputReason   = DDSPC::InputReason::UnverifiedContract;
    DDSPC::InputEvidence        inputEvidence = DDSPC::InputEvidence::Unverified;
    { //mutex scope
        std::lock_guard<std::mutex> guard( m_shadowMutex );
        if( m_shadowController )
        {
            standard = m_shadowController->status( DDSPC::ShadowMode::Standard );
            assisted = m_shadowController->status( DDSPC::ShadowMode::AccelAssisted );
            validity = m_shadowController->validity();
            selected = m_shadowController->selected();
        }
        if( m_shadowPipeline )
        {
            pipelineConfigured = true;
            pipeline           = m_shadowPipeline->status();
            inputReason        = pipeline.m_inputReason;
            inputEvidence      = m_shadowPipeline->evidence();
        }
        if( m_shadowInputs )
        {
            inputReason   = m_shadowInputs->reason();
            inputEvidence = m_shadowInputs->evidence();
        }
    }
    if( m_shadowPipelineEnabled )
    {
        updatesIfChanged<double>(
            m_indiP_ddspcPipeline,
            { "configured",
              "candidate_valid",
              "common_valid",
              "qualified_streak",
              "decisions",
              "fallbacks",
              "recoveries",
              "index",
              "feature_age_us" },
            { double( pipelineConfigured ),
              double( pipeline.m_candidateValid ),
              double( pipeline.m_commonValid ),
              double( pipeline.m_validStreak ),
              double( pipeline.m_decisions ),
              double( pipeline.m_fallbacks ),
              double( pipeline.m_recoveries ),
              double( pipeline.m_index ),
              pipeline.m_featureNs >= 0 ? double( pipeline.m_originNs - pipeline.m_featureNs ) * .001 : -1.0 } );
        updatesIfChanged<std::string>(
            m_indiP_ddspcSupervisor,
            { "requested", "effective", "reason", "feature_reason" },
            { m_shadowRequested,
              pipeline.m_candidateValid ? DDSPC::policy_mode( pipeline.m_effective ) : "unavailable",
              pipeline.m_reason,
              DDSPC::feature_reason( pipeline.m_featureReason ) } );
    }
    const bool isStandard = m_nativeShadowEnabled ? m_shadowRequested == "standard"
                            : pipelineConfigured  ? pipeline.m_requested == DDSPC::PolicyMode::Standard
                                                  : selected == DDSPC::ShadowMode::Standard;
    const bool ready      = isStandard ? standard.m_ready : assisted.m_ready;
    indi::updateSelectionSwitchIfChanged( m_indiP_ddspcMode,
                                          isStandard ? "standard" : "accel_assisted",
                                          m_indiDriver,
                                          m_shadowModeRejected ? INDI_ALERT
                                          : ready              ? INDI_OK
                                                               : INDI_IDLE );
    const auto trainingState = m_shadowTrainRejected ? INDI_ALERT : INDI_OK;
    updateSwitchIfChanged( m_indiP_ddspcTrain,
                           "standard",
                           standard.m_trainingEnabled ? pcf::IndiElement::On : pcf::IndiElement::Off,
                           trainingState );
    updateSwitchIfChanged( m_indiP_ddspcTrain,
                           "accel_assisted",
                           assisted.m_trainingEnabled ? pcf::IndiElement::On : pcf::IndiElement::Off,
                           trainingState );
    auto publish = [&]( pcf::IndiProperty &property, const DDSPC::ShadowStatus &status )
    {
        updatesIfChanged<double>( property,
                                  { "ready",
                                    "training_enabled",
                                    "training",
                                    "data_valid",
                                    "prediction_valid",
                                    "rmse",
                                    "updates",
                                    "predictions",
                                    "samples",
                                    "feature_rank",
                                    "features" },
                                  { double( status.m_ready ),
                                    double( status.m_trainingEnabled ),
                                    double( status.m_training ),
                                    double( status.m_dataValid ),
                                    double( status.m_predictionValid ),
                                    status.m_rmse,
                                    double( status.m_updates ),
                                    double( status.m_predictions ),
                                    double( status.m_samples ),
                                    double( status.m_rank ),
                                    double( status.m_features ) },
                                  status.m_ready       ? INDI_OK
                                  : status.m_dataValid ? INDI_BUSY
                                                       : INDI_IDLE );
    };
    publish( m_indiP_ddspcStandard, standard );
    publish( m_indiP_ddspcAssisted, assisted );
    updatesIfChanged<double>( m_indiP_ddspcData,
                              { "wfs_valid", "applied_valid", "timing_valid", "accel_valid" },
                              { double( validity.m_wfsValid ),
                                double( validity.m_appliedValid ),
                                double( validity.m_timingValid ),
                                double( validity.m_accelValid ) },
                              validity.m_wfsValid && validity.m_appliedValid && validity.m_timingValid ? INDI_OK
                                                                                                       : INDI_IDLE );
    updatesIfChanged<std::string>(
        m_indiP_ddspcStatus,
        { "execution",
          "output",
          "selected",
          "standard_reason",
          "accel_assisted_reason",
          "input_reason",
          "input_provenance" },
        { m_shadowMode ? "shadow" : "legacy",
          m_shadowMode ? "inhibited" : "legacy_compatibility",
          isStandard ? "standard" : "accel_assisted",
          m_shadowMode ? DDSPC::shadow_reason( standard.m_reason ) : "inactive_legacy_execution",
          m_shadowMode ? DDSPC::shadow_reason( assisted.m_reason ) : "inactive_legacy_execution",
          m_shadowMode ? DDSPC::input_reason( inputReason ) : "inactive_legacy_execution",
          m_shadowMode ? DDSPC::input_evidence( inputEvidence ) : "inactive_legacy_execution" } );
}

inline INDI_NEWCALLBACK_DEFN( loPredCtrlAcc, m_indiP_ddspcMode )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_ddspcMode, ipRecv );
    std::lock_guard<std::mutex> lock( m_indiMutex );
    int                         onCount   = 0;
    bool                        valid     = m_shadowMode && ipRecv.getType() == pcf::IndiProperty::Switch;
    DDSPC::ShadowMode           requested = DDSPC::ShadowMode::Standard;
    for( const auto &entry : ipRecv.getElements() )
    {
        valid            = valid && ( entry.first == "standard" || entry.first == "accel_assisted" );
        const auto value = entry.second.getSwitchState();
        valid            = valid && ( value == pcf::IndiElement::On || value == pcf::IndiElement::Off );
        if( value == pcf::IndiElement::On )
        {
            ++onCount;
            requested = entry.first == "standard" ? DDSPC::ShadowMode::Standard : DDSPC::ShadowMode::AccelAssisted;
        }
    }
    valid = valid && onCount == 1;
    { //mutex scope
        std::lock_guard<std::mutex> guard( m_shadowMutex );
        valid = valid && bool( m_shadowController );
        // A pipeline request may remain pending; only the supervisor can qualify an effective candidate.
        // In particular, switching assistance OFF must not require a ready standard model.
        if( valid && !m_shadowPipeline && !m_nativeShadowEnabled )
            valid = m_shadowController->select( requested );
        if( valid )
        {
            m_shadowRequested = requested == DDSPC::ShadowMode::Standard ? "standard" : "accel_assisted";
            if( m_shadowPipeline )
                m_shadowPipeline->request( requested == DDSPC::ShadowMode::Standard ? DDSPC::PolicyMode::Standard
                                                                                    : DDSPC::PolicyMode::Assisted );
        }
    }
    m_shadowModeRejected = !valid;
    updateShadowINDI();
    return valid ? 0 : -1;
}

inline INDI_NEWCALLBACK_DEFN( loPredCtrlAcc, m_indiP_ddspcTrain )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_ddspcTrain, ipRecv );
    std::lock_guard<std::mutex> lock( m_indiMutex );
    bool valid = m_shadowMode && !m_nativeShadowEnabled && ipRecv.getType() == pcf::IndiProperty::Switch &&
                 ipRecv.getNumElements() > 0;
    for( const auto &entry : ipRecv.getElements() )
    {
        const auto value = entry.second.getSwitchState();
        valid            = valid && ( entry.first == "standard" || entry.first == "accel_assisted" ) &&
                           ( value == pcf::IndiElement::On || value == pcf::IndiElement::Off );
    }
    { //mutex scope
        std::lock_guard<std::mutex> guard( m_shadowMutex );
        valid = valid && bool( m_shadowController );
        if( valid )
        {
            for( const auto &entry : ipRecv.getElements() )
            {
                m_shadowController->training( entry.first == "standard" ? DDSPC::ShadowMode::Standard
                                                                        : DDSPC::ShadowMode::AccelAssisted,
                                              entry.second.getSwitchState() == pcf::IndiElement::On );
            }
        }
    }
    m_shadowTrainRejected = !valid;
    updateShadowINDI();
    return valid ? 0 : -1;
}

inline int loPredCtrlAcc::appStartup()
{
    if( m_nativeShadowEnabled &&
        ( !m_nativeShadowConfigValid || !m_shadowMode || !m_adcEnabled || m_shadowPipelineEnabled ) )
        return -1;
#ifndef DDSPC_SHADOW_ONLY
    if( m_nativeShadowEnabled )
        return -1;
#endif
    if( m_shadowPipelineEnabled && ( !m_shadowMode || !m_shadowPipelineConfigValid ) )
        return log<software_error, -1>( { __FILE__, __LINE__, "Shared pipeline startup requires shadow=true." } );
#ifdef DDSPC_SHADOW_ONLY
    if( !m_shadowMode )
        return log<software_error, -1>( { __FILE__, __LINE__, "Shadow-only build refuses legacy startup." } );
#endif
    if( m_adcEnabled && !m_shadowMode )
    {
        return -1; // Native ADC startup never permits Legacy output helpers.
    }
    if( m_shadowMode && configureShadow() < 0 )
        return -1;
    if( m_shadowMode )
        log<text_log>(
            "Shadow output inhibited. Native WFS callbacks are observation-only; ADC capture does not establish "
            "applied-command or exposure timing. Live models remain invalid.",
            logPrio::LOG_WARNING );
    if( !m_nativeShadowEnabled && shmimMonitorT::appStartup() < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__ } );
    }

    if( m_accelEnabled && !m_shadowMode )
    {
        if( accelShmimMonitorT::appStartup() < 0 )
        {
            disableAccelIntegration( "Accelerometer monitor startup failed.", true );
        }
        else
        {
            m_accelMonitorStarted = true;
        }
    }

    if( !compactNativeINDI() )
    {
        CREATE_REG_INDI_NEW_TEXT( m_indiP_exploration, "exploration_sequence", "", "" );
        CREATE_REG_INDI_NEW_TEXT( m_indiP_filename, "filename", "", "" );
    }

    if( setupShadowINDI() < 0 || ( m_shadowMode && setupAdcINDI() < 0 ) )
    {
        return -1;
    }

    if( !compactNativeINDI() )
    {
        createStandardIndiToggleSw( m_indiP_learningToggle, "learn", "Learning State", "Learn Controls" );
        registerIndiPropertyNew( m_indiP_learningToggle, INDI_NEWCALLBACK( m_indiP_learningToggle ) );

        createStandardIndiToggleSw( m_indiP_learningStdToggle, "learn_std", "Learning State", "Learn Controls" );
        registerIndiPropertyNew( m_indiP_learningStdToggle, INDI_NEWCALLBACK( m_indiP_learningStdToggle ) );

        createStandardIndiToggleSw( m_indiP_predictingToggle, "predict", "Predict State", "Predictive Controls" );
        registerIndiPropertyNew( m_indiP_predictingToggle, INDI_NEWCALLBACK( m_indiP_predictingToggle ) );

        createStandardIndiToggleSw(
            m_indiP_integratingToggle, "integrate", "Integration State", "Integration Controls" );
        registerIndiPropertyNew( m_indiP_integratingToggle, INDI_NEWCALLBACK( m_indiP_integratingToggle ) );

        createStandardIndiRequestSw( m_indiP_resetToggle, "reset_model", "Reset the RLS model", "Reset Model" );
        registerIndiPropertyNew( m_indiP_resetToggle, INDI_NEWCALLBACK( m_indiP_resetToggle ) );

        createStandardIndiRequestSw( m_indiP_saveToggle, "save_state", "Save the controller state", "Save State" );
        registerIndiPropertyNew( m_indiP_saveToggle, INDI_NEWCALLBACK( m_indiP_saveToggle ) );

        createStandardIndiRequestSw( m_indiP_loadToggle, "load_state", "Load the controller state", "Load State" );
        registerIndiPropertyNew( m_indiP_loadToggle, INDI_NEWCALLBACK( m_indiP_loadToggle ) );

        if( m_nativeShadowEnabled )
            for( auto *property : { &m_indiP_learningToggle,
                                    &m_indiP_learningStdToggle,
                                    &m_indiP_predictingToggle,
                                    &m_indiP_integratingToggle,
                                    &m_indiP_resetToggle,
                                    &m_indiP_saveToggle,
                                    &m_indiP_loadToggle,
                                    &m_indiP_exploration,
                                    &m_indiP_filename } )
                property->setPerm( pcf::IndiProperty::ReadOnly );

        if( !m_nativeShadowEnabled )
        {
            REG_INDI_SETPROP( m_indiP_fpsSource, m_fpsSource, std::string( "fps" ) );
        }

        createROIndiNumber( m_indiP_fps, "fps" );
        m_indiP_fps.add( pcf::IndiElement( "current" ) );
        if( registerIndiPropertyReadOnly( m_indiP_fps ) < 0 )
        {
            log<software_error>( { "" } );
            return -1;
        }
    }

    if( !m_shadowMode && sem_init( &m_smSemaphore, 0, 0 ) < 0 )
    {
        log<software_critical>( { errno, "Initializing S.M. semaphore" } );
        return -1;
    }

    if( !m_shadowMode )
    {
        FRAMEGRABBER_APP_STARTUP;
        TELEMETER_APP_STARTUP;
    }

    if( m_adcEnabled )
    {
        if( !m_shadowMode )
        {
            return -1; // A direct startup call cannot bypass the configuration guard.
        }
        try
        {
            m_adcReader = std::make_unique<DDSPC::AdcReader>( m_adcConfig );
            m_adcReader->start();
        }
        catch( const std::exception &error )
        {
            m_adcReader.reset();
            return log<software_error, -1>( { __FILE__, __LINE__, error.what() } );
        }
    }
    if( m_shadowMode )
    {
        updateAdcINDI();
    }
    if( m_nativeShadowEnabled )
    {
        if( setupNativeShadowINDI() < 0 )
            return -1;
        try
        {
            m_nativeObserver = std::make_unique<DDSPC::NativeObserver>( m_nativeShadowConfig );
            m_nativeObserver->start( *m_adcReader );
        }
        catch( const std::exception &error )
        {
            m_nativeObserver.reset();
            return log<software_error, -1>( { __FILE__, __LINE__, error.what() } );
        }
        updateNativeShadowINDI();
    }
    state( stateCodes::OPERATING );
    return 0;
}

inline int loPredCtrlAcc::appLogic()
{
    if( m_nativeShadowEnabled && m_nativeObserver )
    {
        const auto status = m_nativeObserver->status();
        if( status.m_finished )
            m_shutdown = true;
        if( status.m_ioError )
            return log<software_error, -1>( { __FILE__, __LINE__, "Native input/recording worker failed" } );
        const auto now =
            std::chrono::duration_cast<std::chrono::nanoseconds>( std::chrono::steady_clock::now().time_since_epoch() )
                .count();
        if( !nativeDiagnosticsDue( now ) )
            return 0;
    }
    if( !m_nativeShadowEnabled && shmimMonitorT::appLogic() < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__ } );
    }

    if( m_accelEnabled && !m_shadowMode )
    {
        if( accelShmimMonitorT::appLogic() < 0 )
        {
            disableAccelIntegration( "Accelerometer monitor thread exited; disabling accelerometer path." );
        }
    }

    if( !m_shadowMode )
    {
        FRAMEGRABBER_APP_LOGIC;
        TELEMETER_APP_LOGIC;
    }

    std::unique_lock<std::mutex> lock( m_indiMutex );

    if( !m_nativeShadowEnabled && shmimMonitorT::updateINDI() < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
    }

    if( m_accelEnabled && !m_shadowMode )
    {
        if( accelShmimMonitorT::updateINDI() < 0 )
        {
            disableAccelIntegration( "Accelerometer INDI update failed; disabling accelerometer path." );
        }
    }

    if( !m_shadowMode )
    {
        FRAMEGRABBER_UPDATE_INDI;
    }
    updateShadowINDI();
    if( m_nativeShadowEnabled )
    {
        updateNativeShadowINDI();
    }
    if( m_shadowMode )
    {
        updateAdcINDI();
    }

    if( compactNativeINDI() )
        return 0;

    updatesIfChanged<std::string>(
        m_indiP_exploration, { "current", "target" }, { m_exploration_sequence, m_exploration_sequence } );

    updatesIfChanged<std::string>( m_indiP_filename, { "current", "target" }, { m_filename, m_filename } );

    if( m_isLearning )
    {
        updateSwitchIfChanged( m_indiP_learningToggle, "toggle", pcf::IndiElement::On, INDI_OK );
    }
    else
    {
        updateSwitchIfChanged( m_indiP_learningToggle, "toggle", pcf::IndiElement::Off, INDI_IDLE );
    }

    if( m_isStdLearning )
    {
        updateSwitchIfChanged( m_indiP_learningStdToggle, "toggle", pcf::IndiElement::On, INDI_OK );
    }
    else
    {
        updateSwitchIfChanged( m_indiP_learningStdToggle, "toggle", pcf::IndiElement::Off, INDI_IDLE );
    }

    if( m_isIntegrating )
    {
        updateSwitchIfChanged( m_indiP_integratingToggle, "toggle", pcf::IndiElement::On, INDI_OK );
    }
    else
    {
        updateSwitchIfChanged( m_indiP_integratingToggle, "toggle", pcf::IndiElement::Off, INDI_IDLE );
    }

    if( m_isPredictiveControl )
    {
        updateSwitchIfChanged( m_indiP_predictingToggle, "toggle", pcf::IndiElement::On, INDI_OK );
    }
    else
    {
        updateSwitchIfChanged( m_indiP_predictingToggle, "toggle", pcf::IndiElement::Off, INDI_IDLE );
    }

    return 0;
}

inline int loPredCtrlAcc::appShutdown()
{
    if( m_shadowMode )
    { //mutex scope
        std::lock_guard<std::mutex> guard( m_shadowMutex );
        m_shadowStopping = true;
        if( m_shadowPipeline )
            m_shadowPipeline->invalidate();
    }
    if( m_nativeObserver )
        m_nativeObserver->stop();
    if( m_adcReader )
    {
        m_adcReader->stop();
    }
    if( !m_nativeShadowEnabled )
        shmimMonitorT::appShutdown();
    if( m_accelMonitorStarted )
    {
        accelShmimMonitorT::appShutdown();
        m_accelMonitorStarted = false;
    }

    if( !m_shadowMode )
    {
        FRAMEGRABBER_APP_SHUTDOWN;
        TELEMETER_APP_SHUTDOWN;
    }

    if( m_controller )
        delete m_controller;

    if( m_shadowMode )
    { //mutex scope
        std::lock_guard<std::mutex> guard( m_shadowMutex );
        m_shadowPipeline.reset();
        m_shadowInputs.reset();
        m_shadowController.reset();
    }

    return 0;
}

inline int loPredCtrlAcc::allocate( const dev::shmimT &dummy )
{
    static_cast<void>( dummy ); // be unused

    m_modevalWidth  = shmimMonitorT::m_width;
    m_modevalHeight = shmimMonitorT::m_height;
    if( m_shadowMode )
    {
        { //mutex scope
            std::lock_guard<std::mutex> guard( m_shadowMutex );
            m_shadowPipeline.reset();
            m_shadowInputs.reset(); // Reconnection/shape changes require a newly verified source session.
            if( m_shadowController )
                m_shadowController->invalidate( DDSPC::ShadowRecord{} );
        }
        if( shmimMonitorT::m_dataType != _DATATYPE_FLOAT || m_modevalHeight != 1 ||
            m_modevalWidth < static_cast<uint32_t>( m_num_modes ) )
            return -1;
        { //mutex scope
            std::lock_guard<std::mutex> guard( m_shadowMutex );
            if( m_shadowController )
            {
                m_shadowController->reset( DDSPC::ShadowMode::Standard );
                m_shadowController->reset( DDSPC::ShadowMode::AccelAssisted );
                m_shadowController->invalidate( DDSPC::ShadowRecord{} );
            }
        }
        return 0;
    }
    m_modevalTypeSize = sizeof( realT );
    std::cout << "m_modevalWidth: " << m_modevalWidth << std::endl;
    std::cout << "m_modevalHeight: " << m_modevalHeight << std::endl;

    // Only resize if dimensions change to avoid Eigen block resize issues
    if( m_fullCommand.rows() != (int)m_modevalWidth || m_fullCommand.cols() != (int)m_modevalHeight )
    {
        m_fullCommand = DDSPC::Matrix( m_modevalWidth, m_modevalHeight );
    }
    if( m_newCommand.rows() != m_num_modes || m_newCommand.cols() != 1 )
    {
        m_newCommand = DDSPC::Matrix( m_num_modes, 1 );
    }
    if( m_newMeasurement.rows() != m_num_modes || m_newMeasurement.cols() != 1 )
    {
        m_newMeasurement = DDSPC::Matrix( m_num_modes, 1 );
    }

    // allocate the exploration noise matrix
    m_zeroExpNoise.resize( m_num_modes, 1 );
    m_zeroExpNoise.setZero();

    m_generator    = std::default_random_engine();
    m_distribution = std::normal_distribution<DDSPC::realT>( 0.0, 1.0 );

    if( m_controller )
    {
        delete m_controller;
        m_controller = nullptr;
    }

    m_controller          = new DDSPC::PredictiveController( m_num_modes,
                                                             m_history,
                                                             m_future,
                                                             m_gainCtrl,
                                                             m_gammaCtrl,
                                                             m_regularizationCtrl,
                                                             m_covarianceCtrl,
                                                             m_accelChannels,
                                                             m_accelHistory );
    m_controller->use_qrd = m_useQrd;

    return 0;
}

inline int loPredCtrlAcc::allocate( const accelShmimT &dummy )
{
    if( m_shadowMode )
        return -1;              // No accelerometer stream attachment in shadow execution.
    static_cast<void>( dummy ); // be unused

    m_accelWidth  = accelShmimMonitorT::m_width;
    m_accelHeight = accelShmimMonitorT::m_height;

    if( m_accelChannels > 0 )
    {
        size_t totalAccelValues = static_cast<size_t>( m_accelWidth ) * static_cast<size_t>( m_accelHeight );
        if( static_cast<size_t>( m_accelChannels ) > totalAccelValues )
        {
            return log<software_error, -1>(
                { __FILE__, __LINE__, "Accelerometer channel count exceeds accel shmim size." } );
        }
    }

    resetAccelTelemetryState();
    return 0;
}

inline void loPredCtrlAcc::resetAccelTelemetryState()
{
    std::lock_guard<std::mutex> guard( m_accelMutex ); // mutex scope
    m_accelNormCount         = 0;
    m_haveAccelSample        = false;
    m_accelMissingFrameCount = 0;

    if( m_accelChannels <= 0 )
    {
        m_accelMean.resize( 0, 1 );
        m_accelM2.resize( 0, 1 );
        m_latestAccelSample.resize( 0, 1 );
        return;
    }

    m_accelMean.resize( m_accelChannels, 1 );
    m_accelMean.setZero();
    m_accelM2.resize( m_accelChannels, 1 );
    m_accelM2.setZero();
    m_latestAccelSample.resize( m_accelChannels, 1 );
    m_latestAccelSample.setZero();
}

inline void loPredCtrlAcc::normalizeAccelSample( DDSPC::Matrix &sample )
{
    if( !m_accelNormalize )
    {
        return;
    }

    if( m_accelChannels <= 0 || sample.rows() != m_accelChannels || sample.cols() != 1 )
    {
        return;
    }

    m_accelNormCount++;
    realT stdFloor = std::max( m_accelStdFloor, static_cast<realT>( 1.0e-8 ) );

    for( int i = 0; i < m_accelChannels; ++i )
    {
        realT value = sample( i, 0 );
        realT delta = value - m_accelMean( i, 0 );
        m_accelMean( i, 0 ) += delta / static_cast<realT>( m_accelNormCount );
        realT delta2 = value - m_accelMean( i, 0 );
        m_accelM2( i, 0 ) += delta * delta2;

        if( m_accelNormCount < 2 )
        {
            sample( i, 0 ) = 0.0;
            continue;
        }

        realT variance = m_accelM2( i, 0 ) / static_cast<realT>( m_accelNormCount - 1 );
        realT sigma    = std::sqrt( std::max( variance, stdFloor * stdFloor ) );
        sample( i, 0 ) = ( value - m_accelMean( i, 0 ) ) / sigma;

        if( m_accelClipSigma > 0.0f )
        {
            sample( i, 0 ) = std::max( -m_accelClipSigma, std::min( m_accelClipSigma, sample( i, 0 ) ) );
        }
    }
}

inline void loPredCtrlAcc::disableAccelIntegration( const std::string &reason, bool rebuildController )
{
    if( !m_accelEnabled )
    {
        return;
    }

    log<text_log>( reason + " Falling back to legacy non-accelerometer behavior.", logPrio::LOG_WARNING );

    m_accelEnabled                  = false;
    m_accelChannels                 = 0;
    m_accelHistory                  = 0;
    accelShmimMonitorT::m_shmimName = "";

    if( m_accelMonitorStarted )
    {
        accelShmimMonitorT::appShutdown();
        m_accelMonitorStarted = false;
    }

    if( rebuildController && m_controller )
    {
        delete m_controller;
        m_controller = new DDSPC::PredictiveController(
            m_num_modes, m_history, m_future, m_gainCtrl, m_gammaCtrl, m_regularizationCtrl, m_covarianceCtrl, 0, 0 );
        m_controller->use_qrd = m_useQrd;
    }

    resetAccelTelemetryState();
}

inline int loPredCtrlAcc::processImage( void *curr_src, const accelShmimT &dummy )
{
    if( m_shadowMode )
        return 0;               // No raw stream ingestion in this phase.
    static_cast<void>( dummy ); // be unused
    if( !m_accelEnabled || m_accelChannels <= 0 )
    {
        return 0;
    }

    Eigen::Map<eigenImage<realT>> accelFrame( static_cast<realT *>( curr_src ), m_accelWidth, m_accelHeight );
    DDSPC::Matrix                 accelSample;
    accelSample.resize( m_accelChannels, 1 );

    size_t availableValues = static_cast<size_t>( m_accelWidth ) * static_cast<size_t>( m_accelHeight );
    for( int i = 0; i < m_accelChannels; ++i )
    {
        if( static_cast<size_t>( i ) >= availableValues )
        {
            accelSample( i, 0 ) = 0.0;
            continue;
        }

        uint32_t row        = static_cast<uint32_t>( i ) % m_accelWidth;
        uint32_t col        = static_cast<uint32_t>( i ) / m_accelWidth;
        accelSample( i, 0 ) = accelFrame( row, col );
    }

    normalizeAccelSample( accelSample );

    std::lock_guard<std::mutex> guard( m_accelMutex ); // mutex scope
    m_latestAccelSample      = accelSample;
    m_haveAccelSample        = true;
    m_accelMissingFrameCount = 0;

    return 0;
}

inline int loPredCtrlAcc::processImage( void *curr_src, const dev::shmimT &dummy )
{
    static_cast<void>( dummy ); // be unused
    if( m_shadowMode )
    {
        DDSPC::ShadowRecord validity;
        validity.m_wfsValid = curr_src != nullptr && shmimMonitorT::m_dataType == _DATATYPE_FLOAT &&
                              m_modevalHeight == 1 && m_modevalWidth >= static_cast<uint32_t>( m_num_modes );
        if( validity.m_wfsValid )
        {
            const realT *values = static_cast<const realT *>( curr_src );
            for( int mode = 0; mode < m_num_modes; ++mode )
                validity.m_wfsValid = validity.m_wfsValid && std::isfinite( values[mode] );
        }
        // This callback has no established applied-command or exposure-timing contract.
        { //mutex scope
            std::lock_guard<std::mutex> guard( m_shadowMutex );
            m_shadowPipeline.reset();
            m_shadowInputs.reset(); // This raw callback still lacks a validated exposure/application contract.
            if( m_shadowController )
                m_shadowController->invalidate( validity );
        }
        return 0;
    }
    // record arrival time as the atime
    if( clock_gettime( CLOCK_REALTIME, &( frameGrabberT::m_currImageTimestamp ) ) < 0 )
    {
        m_shutdown = true;
        return log<software_critical, -1>( { errno, "clock_gettime" } );
    }

    auto start = std::chrono::high_resolution_clock::now();
    // static_cast<void>( dummy ); // be unused
    // This could be made more efficient by doing only a single copy statement.
    Eigen::Map<eigenImage<realT>> m_modeval( static_cast<realT *>( curr_src ), m_modevalWidth, m_modevalHeight );

    DDSPC::Matrix exp_noise;
    exp_noise.resize( m_num_modes, 1 );
    exp_noise.setZero();

    if( m_doResetModel )
    {
        m_controller->reset();
        m_doResetModel = false;
    }

    if( m_doTriggerLoad )
    {
        load( "/opt/MagAOX/calib/loPredCtrlAcc/" + m_filename );
        m_doTriggerLoad = false;
    }

    if( m_doTriggerSave )
    {
        save( "/opt/MagAOX/calib/loPredCtrlAcc/" + m_filename );
        m_doTriggerSave = false;
    }

    if( m_switchExploration )
    {
        m_useSet01          = !m_useSet01;
        m_switchExploration = false;

        if( m_useSet01 )
        {
            m_controller->set_regularization( m_regularization_steps_01[0] );
        }
        else
        {
            m_controller->set_regularization( m_regularization_steps_02[0] );
        }
    }

    if( m_isLearning )
    {
        if( m_useSet01 )
        {
            if( !m_exploration_steps_01.empty() and !m_exploration_noise_strength_01.empty() )
            {
                for( int i = 0; i < m_num_modes; i++ )
                {
                    exp_noise( i, 0 ) = m_exploration_noise_strength_01[0] * m_distribution( m_generator );
                }

                // If no more steps are left pop it!
                m_exploration_steps_01[0]--;
                if( m_exploration_steps_01[0] == 0 )
                {
                    m_exploration_steps_01.erase( m_exploration_steps_01.begin() );
                    m_exploration_noise_strength_01.erase( m_exploration_noise_strength_01.begin() );

                    // Erase and apply the next regularization step?
                    m_regularization_steps_01.erase( m_regularization_steps_01.begin() );
                    if( !m_regularization_steps_01.empty() )
                        m_controller->set_regularization( m_regularization_steps_01[0] );
                }
            }
        }
        else
        {
            if( !m_exploration_steps_02.empty() and !m_exploration_noise_strength_02.empty() )
            {
                for( int i = 0; i < m_num_modes; i++ )
                {
                    exp_noise( i, 0 ) = m_exploration_noise_strength_02[0] * m_distribution( m_generator );
                }

                // If no more steps are left pop it!
                m_exploration_steps_02[0]--;
                if( m_exploration_steps_02[0] == 0 )
                {
                    m_exploration_steps_02.erase( m_exploration_steps_02.begin() );
                    m_exploration_noise_strength_02.erase( m_exploration_noise_strength_02.begin() );

                    // Erase and apply the next regularization step?
                    m_regularization_steps_02.erase( m_regularization_steps_02.begin() );
                    if( !m_regularization_steps_02.empty() )
                        m_controller->set_regularization( m_regularization_steps_02[0] );
                }
            }
        }
    }

    for( int i = 0; i < m_num_modes; i++ )
    {
        m_newMeasurement( i, 0 ) = m_modeval( i, 0 );
    }

    if( m_controller && m_accelEnabled )
    {
        DDSPC::Matrix accelSample;
        accelSample.resize( m_accelChannels, 1 );
        accelSample.setZero();

        bool haveAccel = false;
        { //mutex scope
            std::lock_guard<std::mutex> guard( m_accelMutex );
            if( m_haveAccelSample && m_latestAccelSample.rows() == m_accelChannels && m_latestAccelSample.cols() == 1 )
            {
                accelSample              = m_latestAccelSample;
                haveAccel                = true;
                m_accelMissingFrameCount = 0;
            }
            else
            {
                m_accelMissingFrameCount++;
            }
        }

        if( !haveAccel && m_accelMissingFrameCount >= m_accelMissingFrameLimit )
        {
            disableAccelIntegration( "No accelerometer frames received; disabling accelerometer path.", true );
        }
        else if( m_accelEnabled )
        {
            m_controller->push_accelerometer_sample( accelSample );
        }
    }

    if( m_isPredictiveControl )
    {
        if( m_isLearning )
        {
            m_newCommand = m_controller->calculate_command( m_newMeasurement, exp_noise );
        }
        else
        {
            m_newCommand = m_controller->calculate_command( m_newMeasurement, m_zeroExpNoise );
        }
    }
    else
    {
        for( int i = 0; i < m_num_modes; i++ )
        {
            m_newCommand( i, 0 ) = m_copygainCtrl * m_newMeasurement( i, 0 );
        }
    }

    //
    for( size_t i = 0; i < m_modevalWidth; i++ )
    {
        if( i < static_cast<size_t>( m_num_modes ) )
        {
            if( m_isIntegrating )
            {
                m_fullCommand( i, 0 ) = m_fullCommand( i, 0 ) + m_newCommand( i, 0 );
            }
            else
            {
                m_fullCommand( i, 0 ) = m_newCommand( i, 0 );
            }
        }
        else
        {
            m_fullCommand( i, 0 ) = m_modeval( i, 0 );
        }
    }

    // Send modal coefficients to the correct stream
    m_updated = true;

    // trigger framegrabber
    if( sem_post( &m_smSemaphore ) < 0 )
    {
        log<software_critical>( { errno, 0, "Error posting to semaphore" } );
        return -1;
    }

    if( m_isLearning )
    {
        m_controller->update_system();
        m_controller->update_controller();
    }

    auto end = std::chrono::high_resolution_clock::now();
    m_loopTimeElapsed += std::chrono::duration_cast<std::chrono::duration<double, std::micro>>( end - start ).count();

    if( m_frameCounter % 4000 == 0 )
    {
        std::cout << "HOWDY elapsed us: " << m_loopTimeElapsed / 4000.0 << " us" << std::endl;
        m_loopTimeElapsed = 0.0;
    }

    m_frameCounter++;
    return 0;
}

inline int loPredCtrlAcc::configureAcquisition()
{
#ifdef DDSPC_SHADOW_ONLY
    return -1; // Unconditional commissioning-build inhibition, including invalid runtime state.
#endif
    if( m_shadowMode )
        return -1;
    frameGrabberT::m_width    = m_modevalWidth;
    frameGrabberT::m_height   = 1;
    frameGrabberT::m_dataType = _DATATYPE_FLOAT;

    return 0;
}

inline float loPredCtrlAcc::fps()
{
    return m_fps;
}

inline int loPredCtrlAcc::startAcquisition()
{
#ifdef DDSPC_SHADOW_ONLY
    return -1; // Unconditional commissioning-build inhibition, including invalid runtime state.
#endif
    if( m_shadowMode )
        return -1;
    return 0;
}

inline int loPredCtrlAcc::acquireAndCheckValid()
{
#ifdef DDSPC_SHADOW_ONLY
    return 1; // Unconditional commissioning-build inhibition, including invalid runtime state.
#endif
    if( m_shadowMode )
        return 1;
    timespec ts;

    errno = 0;
    if( clock_gettime( CLOCK_REALTIME, &ts ) < 0 )
    {
        log<software_critical>( { errno, "clock_gettime" } );
        return -1;
    }

    ts.tv_sec += 1;

    if( sem_timedwait( &m_smSemaphore, &ts ) == 0 )
    {
        if( m_updated )
        {
            return 0;
        }
        else
        {
            return 1;
        }
    }
    else
    {
        return 1;
    }
}

inline int loPredCtrlAcc::loadImageIntoStream( void *dest )
{
#ifdef DDSPC_SHADOW_ONLY
    return -1; // Unconditional commissioning-build inhibition, including invalid runtime state.
#endif
    if( m_shadowMode )
        return -1;
    memcpy( dest, m_fullCommand.data(), m_fullCommand.rows() * frameGrabberT::m_typeSize );

    m_updated = false;

    return 0;
}

inline int loPredCtrlAcc::reconfig()
{
    return 0;
}

inline int loPredCtrlAcc::checkRecordTimes()
{
    if( m_shadowMode )
        return 0;
    return telemeterT::checkRecordTimes( telem_fgtimings() );
}

inline int loPredCtrlAcc::recordTelem( const telem_fgtimings * )
{
    if( m_shadowMode )
        return 0;
    return recordFGTimings( true );
}

inline INDI_NEWCALLBACK_DEFN( loPredCtrlAcc, m_indiP_exploration )( const pcf::IndiProperty &ipRecv )
{
    if( m_shadowMode )
        return -1; // Legacy controls cannot affect shadow execution.
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_exploration, ipRecv );
    // Called in indi like: num_explore, std, regularization, num_explore, std, regularization, ....

    std::string target;

    std::unique_lock<std::mutex> lock( m_indiMutex );

    if( indiTargetUpdate( m_indiP_exploration, target, ipRecv, true ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        return -1;
    }

    // Now we need to parse the string!
    m_exploration_sequence = target;
    std::cout << target << std::endl;

    std::stringstream csvStringStream( m_exploration_sequence );
    std::string       entry;

    int k = 0;
    while( getline( csvStringStream, entry, ',' ) )
    {
        if( k % 3 == 0 )
        {
            std::cout << std::stoi( entry ) << std::endl;
            if( m_useSet01 )
            {
                m_exploration_steps_02.push_back( std::stoi( entry ) );
            }
            else
            {
                m_exploration_steps_01.push_back( std::stoi( entry ) );
            }
        }
        else if( k % 3 == 1 )
        {
            std::cout << static_cast<DDSPC::realT>( std::stod( entry ) ) << std::endl;
            if( m_useSet01 )
            {
                m_exploration_noise_strength_02.push_back( std::stod( entry ) );
            }
            else
            {
                m_exploration_noise_strength_01.push_back( std::stod( entry ) );
            }
        }
        else
        {
            std::cout << static_cast<DDSPC::realT>( std::stof( entry ) ) << std::endl;
            if( m_useSet01 )
            {
                m_regularization_steps_02.push_back( std::stof( entry ) );
            }
            else
            {
                m_regularization_steps_01.push_back( std::stof( entry ) );
            }
        }
        k++;
    }
    m_switchExploration = true;

    return 0;
}

inline INDI_NEWCALLBACK_DEFN( loPredCtrlAcc, m_indiP_filename )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_filename, ipRecv );

    std::string target;

    std::unique_lock<std::mutex> lock( m_indiMutex );

    if( indiTargetUpdate( m_indiP_filename, target, ipRecv, true ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        return -1;
    }

    m_filename = target;
    log<text_log>( "Filename set to: " + m_filename, logPrio::LOG_NOTICE );

    return 0;
}

inline INDI_NEWCALLBACK_DEFN( loPredCtrlAcc, m_indiP_learningToggle )( const pcf::IndiProperty &ipRecv )
{
    if( m_shadowMode )
        return -1; // Legacy controls cannot affect shadow execution.
    if( ipRecv.getName() != m_indiP_learningToggle.getName() )
    {
        log<software_error>( { __FILE__, __LINE__, "invalid indi property received" } );
        return -1;
    }

    // switch is toggled to on
    if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On )
    {
        if( !m_isLearning ) // is actively learning so change it
        {
            m_isLearning = true;
            log<text_log>( "started learning", logPrio::LOG_NOTICE );
            updateSwitchIfChanged( m_indiP_learningToggle, "toggle", pcf::IndiElement::On, INDI_BUSY );
        }
        return 0;
    }

    // switch is toggle to off
    if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::Off )
    {
        if( m_isLearning ) // is actively learning so change it
        {
            m_isLearning = false;
            log<text_log>( "stopped learning", logPrio::LOG_NOTICE );
            updateSwitchIfChanged( m_indiP_learningToggle, "toggle", pcf::IndiElement::Off, INDI_IDLE );
        }
        return 0;
    }

    return 0;
}

inline INDI_NEWCALLBACK_DEFN( loPredCtrlAcc, m_indiP_learningStdToggle )( const pcf::IndiProperty &ipRecv )
{
    if( m_shadowMode )
        return -1; // Legacy controls cannot affect shadow execution.
    if( ipRecv.getName() != m_indiP_learningStdToggle.getName() )
    {
        log<software_error>( { __FILE__, __LINE__, "invalid indi property received" } );
        return -1;
    }

    // switch is toggled to on
    if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On )
    {
        if( !m_isLearning ) // is actively learning so change it
        {

            if( m_useSet01 )
            {
                m_exploration_steps_02.push_back( 1000 );
                m_exploration_noise_strength_02.push_back( 0.1 );
                m_regularization_steps_02.push_back( 100.0 );

                m_exploration_steps_02.push_back( 1000 );
                m_exploration_noise_strength_02.push_back( 0.1 );
                m_regularization_steps_02.push_back( 3.0 );

                m_exploration_steps_02.push_back( 1000 );
                m_exploration_noise_strength_02.push_back( 0.1 );
                m_regularization_steps_02.push_back( 0.1 );
            }
            else
            {
                m_exploration_steps_01.push_back( 1000 );
                m_exploration_noise_strength_01.push_back( 0.1 );
                m_regularization_steps_01.push_back( 100.0 );

                m_exploration_steps_01.push_back( 1000 );
                m_exploration_noise_strength_01.push_back( 0.1 );
                m_regularization_steps_01.push_back( 3.0 );

                m_exploration_steps_01.push_back( 1000 );
                m_exploration_noise_strength_01.push_back( 0.1 );
                m_regularization_steps_01.push_back( 0.1 );
            }

            // Setup all the correct triggers
            m_switchExploration   = true;
            m_isPredictiveControl = true;
            m_isLearning          = true;

            log<text_log>( "started standard learning", logPrio::LOG_NOTICE );
            updateSwitchIfChanged( m_indiP_learningStdToggle, "toggle", pcf::IndiElement::On, INDI_BUSY );
        }
        return 0;
    }

    // switch is toggle to off
    if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::Off )
    {
        if( m_isLearning ) // is actively learning so change it
        {
            m_isLearning = false;
            log<text_log>( "stopped STD learning", logPrio::LOG_NOTICE );
            updateSwitchIfChanged( m_indiP_learningStdToggle, "toggle", pcf::IndiElement::Off, INDI_IDLE );
        }
        return 0;
    }

    return 0;
}

inline INDI_NEWCALLBACK_DEFN( loPredCtrlAcc, m_indiP_integratingToggle )( const pcf::IndiProperty &ipRecv )
{
    if( m_shadowMode )
        return -1; // Legacy controls cannot affect shadow execution.
    if( ipRecv.getName() != m_indiP_integratingToggle.getName() )
    {
        log<software_error>( { __FILE__, __LINE__, "invalid indi property received" } );
        return -1;
    }

    // switch is toggled to on
    if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On )
    {
        if( !m_isIntegrating ) // is actively integrating so change it
        {
            m_isIntegrating = true;
            log<text_log>( "started integrating", logPrio::LOG_NOTICE );
            updateSwitchIfChanged( m_indiP_integratingToggle, "toggle", pcf::IndiElement::On, INDI_BUSY );
        }
        return 0;
    }

    // switch is toggle to off
    if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::Off )
    {
        if( m_isIntegrating ) // is actively integrating so change it
        {
            m_isIntegrating = false;
            log<text_log>( "stopped integrating", logPrio::LOG_NOTICE );
            updateSwitchIfChanged( m_indiP_integratingToggle, "toggle", pcf::IndiElement::Off, INDI_IDLE );
        }
        return 0;
    }

    return 0;
}

inline INDI_NEWCALLBACK_DEFN( loPredCtrlAcc, m_indiP_predictingToggle )( const pcf::IndiProperty &ipRecv )
{
    if( m_shadowMode )
        return -1; // Legacy controls cannot affect shadow execution.
    if( ipRecv.getName() != m_indiP_predictingToggle.getName() )
    {
        log<software_error>( { __FILE__, __LINE__, "invalid indi property received" } );
        return -1;
    }

    // switch is toggled to on
    if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On )
    {
        if( !m_isPredictiveControl ) // is actively learning so change it
        {
            m_isPredictiveControl = true;
            log<text_log>( "started predicting", logPrio::LOG_NOTICE );
            updateSwitchIfChanged( m_indiP_predictingToggle, "toggle", pcf::IndiElement::On, INDI_BUSY );
        }
        return 0;
    }

    // switch is toggle to off
    if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::Off )
    {
        if( m_isPredictiveControl ) // is actively learning so change it
        {
            m_isPredictiveControl = false;
            log<text_log>( "stopped predicting", logPrio::LOG_NOTICE );
            updateSwitchIfChanged( m_indiP_predictingToggle, "toggle", pcf::IndiElement::Off, INDI_IDLE );
        }
        return 0;
    }

    return 0;
}

inline INDI_NEWCALLBACK_DEFN( loPredCtrlAcc, m_indiP_resetToggle )( const pcf::IndiProperty &ipRecv )
{
    if( m_shadowMode )
    {
        INDI_VALIDATE_CALLBACK_PROPS( m_indiP_resetToggle, ipRecv );
        if( !ipRecv.find( "request" ) )
            return 0;
        if( ipRecv["request"].getSwitchState() == pcf::IndiElement::On )
        {
            { //mutex scope
                std::lock_guard<std::mutex> guard( m_shadowMutex );
                if( m_shadowController )
                {
                    m_shadowController->reset( DDSPC::ShadowMode::Standard );
                    m_shadowController->reset( DDSPC::ShadowMode::AccelAssisted );
                    m_shadowController->invalidate( DDSPC::ShadowRecord{} );
                    if( m_shadowPipeline )
                        m_shadowPipeline->invalidate();
                    if( m_shadowInputs )
                        m_shadowInputs->invalidate();
                }
            }
            std::lock_guard<std::mutex> lock( m_indiMutex );
            updateSwitchIfChanged( m_indiP_resetToggle, "request", pcf::IndiElement::Off, INDI_OK );
            updateShadowINDI();
        }
        return 0;
    }

    if( ipRecv.getName() != m_indiP_resetToggle.getName() )
    {
        log<software_error>( { __FILE__, __LINE__, "invalid indi property received" } );
        return -1;
    }

    if( !ipRecv.find( "request" ) )
        return 0;

    if( ipRecv["request"].getSwitchState() == pcf::IndiElement::On )
    {
        std::lock_guard<std::mutex> guard( m_indiMutex );

        // m_controller->reset();
        m_doResetModel = true;
        log<text_log>( "request reset.", logPrio::LOG_NOTICE );
        updateSwitchIfChanged( m_indiP_resetToggle, "request", pcf::IndiElement::Off, INDI_IDLE );
    }

    return 0;
}

inline INDI_NEWCALLBACK_DEFN( loPredCtrlAcc, m_indiP_saveToggle )( const pcf::IndiProperty &ipRecv )
{
    if( m_shadowMode )
        return -1; // Legacy controls cannot affect shadow execution.
    if( ipRecv.getName() != m_indiP_saveToggle.getName() )
    {
        log<software_error>( { __FILE__, __LINE__, "invalid indi property received" } );
        return -1;
    }

    if( !ipRecv.find( "request" ) )
        return 0;

    if( ipRecv["request"].getSwitchState() == pcf::IndiElement::On )
    {
        std::lock_guard<std::mutex> guard( m_indiMutex );

        // save(m_filename);
        m_doTriggerSave = true;
        log<text_log>( "saved state to " + m_filename, logPrio::LOG_NOTICE );
        updateSwitchIfChanged( m_indiP_saveToggle, "request", pcf::IndiElement::Off, INDI_IDLE );
    }

    return 0;
}

inline INDI_NEWCALLBACK_DEFN( loPredCtrlAcc, m_indiP_loadToggle )( const pcf::IndiProperty &ipRecv )
{
    if( m_shadowMode )
        return -1; // Legacy controls cannot affect shadow execution.
    if( ipRecv.getName() != m_indiP_loadToggle.getName() )
    {
        log<software_error>( { __FILE__, __LINE__, "invalid indi property received" } );
        return -1;
    }

    if( !ipRecv.find( "request" ) )
        return 0;

    if( ipRecv["request"].getSwitchState() == pcf::IndiElement::On )
    {
        std::lock_guard<std::mutex> guard( m_indiMutex );

        // load(m_filename);
        m_doTriggerLoad = true;
        log<text_log>( "loaded state from " + m_filename, logPrio::LOG_NOTICE );
        updateSwitchIfChanged( m_indiP_loadToggle, "request", pcf::IndiElement::Off, INDI_IDLE );
    }

    return 0;
}

inline INDI_SETCALLBACK_DEFN( loPredCtrlAcc, m_indiP_fpsSource )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_fpsSource, ipRecv );

    if( ipRecv.find( "current" ) != true ) // this isn't valid
    {
        return -1;
    }

    std::lock_guard<std::mutex> guard( m_indiMutex );

    realT fps = ipRecv["current"].get<float>();

    if( fps != m_fps )
    {
        m_fps = fps;
        updateIfChanged( m_indiP_fps, "current", m_fps );
    }

    return 0;
}

} // namespace app
} // namespace MagAOX

#endif // loPredCtrlAcc_hpp
