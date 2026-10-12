/** \file po4ao.hpp
 * \brief Real-time modal PO4AO inference and replay publication for MagAO-X.
 *
 * \ingroup po4ao_files
 */

#ifndef po4ao_hpp
#define po4ao_hpp

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"
#include "po4ao_utils.hpp"

/** \defgroup po4ao PO4AO
 * \brief Real-time modal policy control with asynchronous model loading and replay publication.
 * \ingroup apps
 */

/** \defgroup po4ao_files PO4AO Files
 * \ingroup po4ao
 */

/// TensorRT logger which suppresses informational messages in the real-time app.
class Logger : public nvinfer1::ILogger
{
    /// Forward TensorRT warnings and errors to standard output during model loading.
    void log( Severity    severity /**< [in] TensorRT message severity. */,
              const char *msg /**< [in] TensorRT message text. */ ) noexcept override;
};

inline void Logger::log( Severity severity, const char *msg ) noexcept
{
    if( severity <= Severity::kWARNING )
    {
        std::cout << msg << std::endl;
    }
}

/// Release either ordinary or CUDA-pinned host memory.
struct HostBufferDeleter
{
    bool pinned{ false }; ///< Whether CUDA allocated the buffer.

    /// Release one host buffer with its matching allocator.
    void operator()( float *pointer /**< [in] Buffer to release. */ ) const;
};

inline void HostBufferDeleter::operator()( float *pointer ) const
{
    if( pointer == nullptr )
        return;
    if( pinned )
        cudaFreeHost( pointer );
    else
        delete[] pointer;
}

/// Validated CPU-linear or TensorRT policy and its tensor contract.
struct EngineBundle
{
    nvinfer1::IRuntime          *runtime{ nullptr };   ///< Runtime which deserialized the engine.
    nvinfer1::ICudaEngine       *engine{ nullptr };    ///< Immutable TensorRT engine.
    nvinfer1::IExecutionContext *context{ nullptr };   ///< Loader warms this context before exclusive real-time use.
    cudaGraphExec_t              graphExec{ nullptr }; ///< Executable graph launched by the real-time thread.
    std::string                  inputName;            ///< Input tensor name.
    std::string                  outputName;           ///< Output tensor name.
    int                          inputChannels{ 0 };   ///< Policy input feature planes.
    int                          inputHeight{ 0 };     ///< Policy input height.
    int                          inputWidth{ 0 };      ///< Policy input width.
    int                          inputSize{ 0 };       ///< Total policy input floats.
    int                          outputSize{ 0 };      ///< Total policy output floats.
    int                          trainedModes{ 0 };    ///< Modal prefix represented by this engine.
    int                          historyLength{ 0 };   ///< Observation history inferred from the model.
    uint64_t                     modelId{ 0 };         ///< Stable FNV-1a hash of serialized engine bytes.
    uint64_t                     generation{ 0 };      ///< Reset generation in which this model was requested.
    std::vector<float>           linearWeights;        ///< Optional feature-major CPU linear-policy weights.

    /// Destroy TensorRT objects outside the real-time thread.
    ~EngineBundle();
};

inline EngineBundle::~EngineBundle()
{
    if( graphExec != nullptr )
        cudaGraphExecDestroy( graphExec );
    delete context;
    delete engine;
    delete runtime;
}

namespace MagAOX
{
namespace app
{

/// Apply a modal policy upstream of mfilt while publishing complete replay snapshots.
/** \ingroup po4ao
 */
class po4ao : public MagAOXApp<true>, public dev::shmimMonitor<po4ao>
{
    // Give the test harness access.
    friend class po4ao_test;

    friend class dev::shmimMonitor<po4ao>;

    /// The monitored modal-input base type.
    typedef dev::shmimMonitor<po4ao> shmimMonitorT;

    /// Floating-point type used by the controller.
    typedef float realT;

  protected:
    /** \name Configuration - Data
     *@{
     */
    std::string m_dataDirs;                      ///< Directory containing training artifacts.
    std::string m_engineName;                    ///< TensorRT engine filename.
    std::string m_engineDirs;                    ///< Directory containing the TensorRT engine.
    std::string m_linearInferenceBackend;        ///< Linear-policy backend: auto, cpu, or tensorrt.
    std::string m_outputChannel;                 ///< Differential modal command stream consumed by mfilt.
    std::string m_bufferChannel;                 ///< Rolling replay-buffer stream.
    std::string m_bufferMetadataChannel;         ///< Rolling replay publication metadata stream.
    std::string m_warmupBufferChannel;           ///< Frozen warm-up replay-buffer stream.
    std::string m_warmupMetadataChannel;         ///< Frozen warm-up publication metadata stream.
    std::string m_exploreNoiseFile;              ///< Exploration-noise CSV loaded at startup.
    int         m_configVersion{ 0 };            ///< Runtime-configuration schema version.
    int         m_nModes{ 0 };                   ///< Number of modes in each shared-memory stream.
    int         m_nModesControlled{ 0 };         ///< Leading modes actively controlled by PO4AO.
    int         m_historyCapacity{ 0 };          ///< Maximum history retained for hot-swappable policies.
    int         m_replayBufferSize{ 0 };         ///< Samples retained in the rolling replay buffer.
    uint32_t    m_iterationsPerEpisode{ 0 };     ///< Frames in one publication episode.
    uint32_t    m_warmupEpisodes{ 0 };           ///< Episodes which use exploratory integrator commands.
    float       m_integratorGain{ 0.F };         ///< Proportional fallback gain; mfilt owns downstream integration.
    float       m_policyBlend{ 1.F };            ///< Learned-policy fraction relative to fallback control.
    float       m_explorationSigmaStart{ 0.F };  ///< Exploration-noise scale at warm-up start.
    float       m_explorationSigmaEnd{ 0.F };    ///< Exploration-noise scale at warm-up end.
    float       m_maxCommandAbs{ 0.F };          ///< Per-mode differential-command limit.
    int         m_cudaDevice{ 0 };               ///< CUDA device used by the inference stream.
    uint32_t    m_latencyDeadlineUs{ 0 };        ///< Callback duration counted as a deadline miss.
    double      m_statsPeriodSeconds{ 10.0 };    ///< Runtime-statistics publication period.
    bool        m_controlEnabledConfig{ false }; ///< Initial learned-policy selection state.
    ///@}

    /** \name Inference and History - Data
     *@{
     */
    Logger                      m_logger;                        ///< TensorRT deserialization logger.
    EngineBundle               *m_activeEngine{ nullptr };       ///< Model used only by the real-time callback.
    std::atomic<EngineBundle *> m_pendingEngine{ nullptr };      ///< Fully validated model awaiting activation.
    std::atomic<EngineBundle *> m_retiredEngine{ nullptr };      ///< Old model awaiting background destruction.
    std::atomic<bool>           m_reloadRequested{ false };      ///< Background-load request flag.
    std::atomic<bool>           m_loaderShutdown{ false };       ///< Model-loader thread shutdown flag.
    std::atomic<int>            m_reloadStatus{ 0 };             ///< Reload state: idle, loading, ready, or failed.
    std::atomic<uint64_t>       m_modelGeneration{ 1 };          ///< Generation accepted for future model loads.
    std::atomic<uint64_t>       m_reloadGeneration{ 1 };         ///< Generation attached to the queued reload.
    std::atomic<uint64_t>       m_modelResetRequest{ 0 };        ///< Generation awaiting fallback activation.
    std::thread                 m_loaderThread;                  ///< Non-real-time TensorRT loader.
    std::mutex                  m_loaderMutex;                   ///< Protects loader sleep and wake-up.
    std::condition_variable     m_loaderCondition;               ///< Wakes the model-loader thread.
    std::mutex                  m_reloadErrorMutex;              ///< Protects the reload-status detail string.
    std::string                 m_reloadError;                   ///< Most recent model-load failure.
    cudaStream_t                m_cudaStream{ nullptr };         ///< Dedicated inference stream.
    cudaStream_t                m_engineWarmupStream{ nullptr }; ///< Low-priority candidate-engine warm-up stream.
    float                      *m_deviceInput{ nullptr };        ///< Device policy-input storage.
    float                      *m_deviceOutput{ nullptr };       ///< Device policy-output storage.
    float                      *m_deviceWarmupInput{ nullptr };  ///< Loader-owned candidate input storage.
    float                      *m_deviceWarmupOutput{ nullptr }; ///< Loader-owned candidate output storage.
    std::unique_ptr<float[], HostBufferDeleter> m_inputState{ nullptr, HostBufferDeleter{} }; ///< Host policy input.
    std::unique_ptr<float[], HostBufferDeleter> m_output{ nullptr, HostBufferDeleter{} };     ///< Host policy output.
    std::unique_ptr<float[]>                    m_integratorCommands;        ///< Safe proportional fallback command.
    std::unique_ptr<CircularBuffer>             m_commandHistory;            ///< Commands used for policy history.
    std::unique_ptr<CircularBuffer>             m_observationHistory;        ///< Observations used for policy history.
    std::vector<float>                          m_explorationNoise;          ///< Flattened exploration-noise table.
    size_t                                      m_explorationNoiseRows{ 0 }; ///< Exploration-noise row count.
    size_t                                      m_explorationNoiseCols{ 0 }; ///< Exploration-noise column count.
    std::atomic<bool>     m_controlEnabled{ false };   ///< Whether the learned policy may replace fallback.
    std::atomic<bool>     m_useNeuralNetwork{ false }; ///< Whether the policy may replace warm-up control.
    uint64_t              m_frameCounter{ 0 };         ///< Frames processed since allocation or warm-up reset.
    std::atomic<uint64_t> m_episodeCounter{ 0 };       ///< Completed episode count.
    ///@}

    /** \name Replay Publication - Data
     *@{
     */
    static constexpr size_t                      c_episodeSlots = 3; ///< Preallocated producer-consumer episode slots.
    std::array<EpisodeBuffer, c_episodeSlots>    m_episodeBuffers;   ///< Real-time handoff storage.
    std::array<std::atomic<int>, c_episodeSlots> m_episodeStates;    ///< 0 free, 1 active, 2 ready, 3 publishing.
    std::array<int, c_episodeSlots>              m_readyQueue{}; ///< Fixed-size single-producer/single-consumer queue.
    std::atomic<uint64_t>                        m_readyWrite{ 0 };        ///< Queue producer position.
    std::atomic<uint64_t>                        m_readyRead{ 0 };         ///< Queue consumer position.
    int                                          m_activeEpisodeSlot{ 0 }; ///< Slot written by the real-time callback.

    std::atomic<uint64_t> m_collectionGeneration{ 1 };       ///< Latest requested warm-up collection generation.
    uint64_t              m_activeCollectionGeneration{ 1 }; ///< Generation stamped by the real-time producer.
    uint64_t              m_publicationGeneration{ 1 };      ///< Generation currently accumulated by the publisher.
    std::atomic<uint64_t> m_warmupRecollectRequest{ 0 };     ///< Generation awaiting a real-time collection reset.

    std::thread                     m_replayThread;            ///< Non-real-time replay publisher.
    std::atomic<bool>               m_replayShutdown{ false }; ///< Replay-publisher shutdown flag.
    std::mutex                      m_replayMutex;             ///< Protects publisher sleep and wake-up.
    std::condition_variable         m_replayCondition;         ///< Wakes the replay-publisher thread.
    std::mutex                      m_publicationMutex;        ///< Serializes stream reconfiguration and publication.
    std::unique_ptr<CircularBuffer> m_publishCommands;         ///< Publisher-owned rolling commands.
    std::unique_ptr<CircularBuffer> m_publishObservations;     ///< Publisher-owned rolling observations.
    ///@}

    /** \name Shared-memory Streams - Data
     *@{
     */
    IMAGE m_outputStream{};                ///< Differential modal command stream.
    IMAGE m_replayStream{};                ///< Rolling replay-buffer stream.
    IMAGE m_replayMetadataStream{};        ///< Rolling replay publication metadata stream.
    IMAGE m_warmupStream{};                ///< Frozen warm-up replay-buffer stream.
    IMAGE m_warmupMetadataStream{};        ///< Frozen warm-up publication metadata stream.
    bool  m_outputOpened{ false };         ///< Whether the command stream is open and validated.
    bool  m_replayOpened{ false };         ///< Whether the replay stream is open and validated.
    bool  m_replayMetadataOpened{ false }; ///< Whether rolling replay metadata is open and validated.
    bool  m_warmupOpened{ false };         ///< Whether the warm-up stream is open and validated.
    bool  m_warmupMetadataOpened{ false }; ///< Whether warm-up metadata is open and validated.
    ///@}

    /** \name Runtime Telemetry - Data
     *@{
     */
    static constexpr size_t                             c_latencySamples = 1024; ///< Callback-latency ring length.
    std::array<std::atomic<uint32_t>, c_latencySamples> m_latencySamples;        ///< Recent callback durations.
    std::atomic<uint64_t>                               m_latencyCount{ 0 }; ///< Number of recorded callback durations.
    std::atomic<uint64_t>                               m_missedFrames{ 0 }; ///< Source counter discontinuities.
    std::atomic<uint64_t> m_deadlineMisses{ 0 };                   ///< Callbacks exceeding the configured deadline.
    std::atomic<uint64_t> m_inferenceFailures{ 0 };                ///< Inference or result-validation failures.
    std::atomic<uint64_t> m_replayDrops{ 0 };                      ///< Episodes dropped because the publisher lagged.
    std::atomic<uint64_t> m_activeModelId{ 0 };                    ///< Selected policy hash; zero reports fallback.
    std::atomic<bool>     m_activeCpuInference{ false };           ///< Whether the active policy runs on CPU.
    uint64_t              m_lastInputCounter{ 0 };                 ///< Previous source cnt0 used for gap detection.
    bool                  m_haveInputCounter{ false };             ///< Whether the previous source counter is valid.
    std::chrono::steady_clock::time_point m_lastStatsUpdate;       ///< Previous telemetry publication time.
    pcf::IndiProperty                     m_indiP_reloadToggle;    ///< Asynchronous model-reload request.
    pcf::IndiProperty                     m_indiP_resetModel;      ///< Active-model reset request.
    pcf::IndiProperty                     m_indiP_controlToggle;   ///< Select learned control or proportional fallback.
    pcf::IndiProperty                     m_indiP_recollectWarmup; ///< Fresh warm-up collection request.
    pcf::IndiProperty                     m_indiP_runtimeStats;    ///< Latency and failure counters.
    pcf::IndiProperty                     m_indiP_modelStatus;     ///< Active model ID and reload state.
                                                                   ///@}

  public:
    /** \name Configuration
     *@{
     */
    /// Construct the application with repository-version metadata.
    po4ao();

    /// Destroy application resources after appShutdown has stopped worker threads.
    ~po4ao() noexcept;

    /// Register application configuration keys.
    virtual void setupConfig();

    /// Load monitor and PO4AO settings, propagating monitor configuration errors.
    int loadConfigImpl(
        mx::app::appConfigurator &_config /**< [in] Application configuration containing runtime values. */ );

    /// Load application configuration and request shutdown if loading fails.
    virtual void loadConfig();
    ///@}

    /// Validate settings, register INDI properties, and start the monitor and workers.
    /** \returns 0 on success, or -1 after startup failure and worker rollback.
     */
    virtual int appStartup();

    /// Check monitor health and publish control, model, and runtime statistics.
    /**
     * \returns 0 on no critical error
     * \returns -1 on an error requiring shutdown
     */
    virtual int appLogic();
    /// Stop the monitor and workers, publish zero output, and release owned resources.
    /** \returns 0 on success, or -1 if monitor shutdown failed after cleanup.
     */
    virtual int appShutdown();

    /// Publish the current differential modal command.
    int sendOutput();

    /// Load a serialized model and validate its CPU sidecar or TensorRT tensor contract.
    std::unique_ptr<EngineBundle> createEngineBundle( const std::string &filename /**< [in] Engine pathname. */,
                                                      std::string       &error /**< [out] Failure detail. */ );

    /// Load an optional CPU linear-policy sidecar matching one TensorRT engine.
    bool loadLinearPolicy( const std::string &filename /**< [in] Linear-policy sidecar pathname. */,
                           EngineBundle      &bundle /**< [in,out] Bundle receiving validated weights. */,
                           std::string       &error /**< [out] Failure detail. */ );

    /// Allocate CUDA inference buffers before activating a TensorRT candidate.
    int prepareEngineMemory();

    /// Release CUDA buffers and the dedicated inference stream.
    void cleanupEngineMemory();

    /// Exercise an engine on the selected CUDA stream before real-time use.
    bool warmEngineBundle( EngineBundle &bundle /**< [in,out] Candidate engine and execution context. */,
                           cudaStream_t  stream /**< [in] CUDA stream used for the warm-up. */,
                           float        *deviceInput /**< [in,out] Device input storage. */,
                           float        *deviceOutput /**< [in,out] Device output storage. */,
                           std::string  &error /**< [out] Failure detail. */ );

    /// Capture and upload one candidate's complete inference graph.
    bool captureEngineGraph( EngineBundle &bundle /**< [in,out] Warmed candidate engine. */,
                             std::string  &error /**< [out] Failure detail. */ );

    /// Load exploration noise from the configured CSV.
    bool loadExplorationNoise();

    /// Background model-loader loop.
    void modelLoaderLoop();

    /// Activate a ready model from the current reset generation at a frame boundary.
    void activatePendingEngine();

    /// Select fallback control and clear histories at a frame boundary.
    void resetActiveModel( uint64_t generation /**< [in] Reset generation to apply. */ );

    /// Background replay-publication loop.
    void replayPublisherLoop();

    /// Queue a completed episode without waiting for the publisher.
    void finishEpisode();

    /// Publish one rolling replay snapshot and its validity metadata.
    int publishReplay( IMAGE   &stream /**< [in,out] Destination replay stream. */,
                       bool     opened /**< [in] Whether replay destination is usable. */,
                       IMAGE   &metadataStream /**< [in,out] Destination metadata stream. */,
                       bool     metadataOpened /**< [in] Whether metadata destination is usable. */,
                       uint64_t episodeIdentifier /**< [in] Completed episode ID. */ );

    /// Mark one replay stream invalid until a fresh snapshot is published.
    int invalidateReplay( IMAGE &stream /**< [in,out] Replay stream to invalidate. */,
                          bool   opened /**< [in] Whether the replay stream is usable. */,
                          IMAGE &metadataStream /**< [in,out] Paired metadata stream. */,
                          bool   metadataOpened /**< [in] Whether the metadata stream is usable. */ );

    /// Restart warm-up collection at a real-time frame boundary.
    void beginWarmupCollection( uint64_t generation /**< [in] New collection generation. */ );

    /// Assemble modal history and run CPU-linear or dedicated-stream TensorRT inference.
    bool runInference( const float *observation /**< [in] Current full-sized modal observation. */ );

    /// Fill the proportional fallback command, optionally with exploration noise.
    void makeIntegratorCommand( const float *observation /**< [in] Current modal observation. */,
                                bool         addNoise /**< [in] Whether to add scheduled exploration. */ );

    /// Publish latency percentiles and failure counters to INDI and the log.
    void updateRuntimeStatistics();

    /// Open and validate command and replay streams.
    int openOutputStreams();

    /// Close command and replay streams.
    void closeOutputStreams();

  protected:
    /** \name INDI Callbacks
     *@{
     */
    /// Queue an explicit model reload on the background loader.
    INDI_NEWCALLBACK_DECL( po4ao, m_indiP_reloadToggle );

    /// Request fallback control and invalidate reloads from older generations.
    INDI_NEWCALLBACK_DECL( po4ao, m_indiP_resetModel );

    /// Select learned-policy control or the proportional fallback.
    INDI_NEWCALLBACK_DECL( po4ao, m_indiP_controlToggle );

    /// Invalidate replay snapshots and request a fresh exploratory warm-up.
    INDI_NEWCALLBACK_DECL( po4ao, m_indiP_recollectWarmup );
    ///@}

    /// Allocate fixed-size controller state after the monitored stream opens.
    int allocate( const dev::shmimT &dummy /**< [in] Tag identifying the monitored input. */ );

    /// Process one modal observation and publish one differential command.
    int processImage( void              *curr_src /**< [in] Pointer to the current full-sized modal frame. */,
                      const dev::shmimT &dummy /**< [in] Tag identifying the monitored input. */
    );
};

#include "po4ao_impl.hpp"

} // namespace app
} // namespace MagAOX

#endif // po4ao_hpp
