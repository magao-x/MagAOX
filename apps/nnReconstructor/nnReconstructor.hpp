/** \file nnReconstructor.hpp
 * \brief TensorRT WFS reconstruction with checked model loading and acknowledged publication controls
 *
 * \ingroup app_files
 */

#ifndef nnReconstructor_hpp
#define nnReconstructor_hpp

#ifdef NNRECONSTRUCTOR_TEST
    #include "tests/nnTestSupport.hpp"
#else
    #include <NvInfer.h>
    #include <cuda_runtime_api.h>
    #include <iostream>
    #include <fstream>
    #include <vector>
    #include <limits>

    #include <mx/improc/eigenCube.hpp>
    #include <mx/improc/eigenImage.hpp>

    #include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
    #include "../../magaox_git_version.h"
#endif

#include <algorithm>
#include <cstring>
#include <array>
#include <atomic>
#include <bit>
#include <condition_variable>
#include <cstdint>
#include <fstream>
#include <filesystem>
#include <memory>
#include <mutex>
#include <cmath>
#include <cctype>
#include <limits>
#include <stdexcept>
#include <vector>
#include <thread>
#include <chrono>
#include <sys/stat.h>

using namespace nvinfer1;

/// Forward TensorRT warnings and errors to the existing diagnostic stream.
class Logger : public ILogger
{
    /// Emit warnings/errors without changing TensorRT logger exception guarantees.
    void log( Severity    severity /**< [in] TensorRT severity. */,
              const char *msg /**< [in] TensorRT-owned message. */ ) noexcept override;
};

inline void Logger::log( Severity severity, const char *msg ) noexcept
{
    if( severity <= Severity::kWARNING )
        std::cout << msg << std::endl;
}

namespace MagAOX
{
namespace app
{

/// Reconstruct normalized WFS pupils into an externally owned modal stream.
/** The monitor owns inference; the framegrabber owns claimed snapshots and timestamps.
 * Runtime controls share only the protected handoff state. Model replacement is allowed
 * while disabled and never changes activation. Loop state and writer ownership remain
 * operator responsibilities; this app does not provide an inter-process loop interlock.
 */
class nnReconstructor : public MagAOXApp<true>,
                        public dev::shmimMonitor<nnReconstructor>,
                        public dev::frameGrabber<nnReconstructor>,
                        public dev::telemeter<nnReconstructor>
{
    /// Allow focused app-local tests to exercise the production methods.
    friend class nnReconstructor_test;

    friend class dev::shmimMonitor<nnReconstructor>;

    /// WFS stream monitor interface.
    typedef dev::shmimMonitor<nnReconstructor> shmimMonitorT;

    friend class dev::frameGrabber<nnReconstructor>;

    typedef dev::frameGrabber<nnReconstructor> frameGrabberT;

    friend class dev::telemeter<nnReconstructor>;

    typedef dev::telemeter<nnReconstructor> telemeterT;

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
    std::string engineName; ///< Configured startup engine filename.

    std::string engineDirs; ///< Directory containing the configured serialized engines.

    bool rebuildEngine{ false }; ///< Legacy flag retained for compatibility; no engine building occurs.

    std::string m_fpsSource{ "camwfs" }; /**< Device name for getting fps of the loop.
                                              This device must have *.fps.current.  Default is camwfs*/

    ///@}

    Logger logger; ///< Logger retained for the lifetime of all TensorRT resources.

    std::vector<char> engineData; ///< Serialized bytes belonging to the loaded model.

    IRuntime *runtime{ nullptr }; ///< Owned runtime; destroyed after its engine.

    ICudaEngine *engine{ nullptr }; ///< Owned engine; replaced only while workers are quiescent.

    IExecutionContext *context{ nullptr }; ///< Owned execution context used only by the admitted inference.

    int inputC{ 0 }; ///< Validated image-channel count.

    int inputH{ 0 }; ///< Validated padded pupil height.

    int inputW{ 0 }; ///< Validated padded pupil width.

    int inputSize{ 0 }; ///< Validated input element count; immutable after startup.

    uint32_t outputSize{ 0 }; ///< Modal count fixed by the startup engine and preserved across replacements.

    int zeroPad{ 2 }; ///< Existing zero-padding width on each pupil edge.

    float *d_input{ nullptr }; ///< Owned device input allocation; holds FP32 elements.

    float *d_output{ nullptr }; ///< Owned device modal-output allocation.

    float imageNorm{
        std::numeric_limits<float>::quiet_NaN() }; ///< Existing multiplicative scale applied to cropped WFS pixels.

    float modalNorm{ 1 }; ///< Legacy value loaded but intentionally not applied.

    int m_pupPix{ -1 }; ///< Padded pupil side length supplied to the network.

    int pup_offset1_x{ -1 }; ///< First pupil-column origin in the source image.

    int pup_offset1_y{ -1 }; ///< First pupil-row origin in the source image.

    int pup_offset2_x{ -1 }; ///< Second pupil-column origin in the source image.

    int pup_offset2_y{ -1 }; ///< Second pupil-row origin in the source image.

    int pixels_per_quadrant{ -1 }; ///< Fixed stride between flattened padded pupils.

    int Npup{ 4 }; ///< Four pupils in the existing channel order.

    eigenImage<float> modeval; ///< Inference-owned download buffer; never read by the publisher.

    float *pp_image{ nullptr }; ///< Owned CPU preprocessing buffer, reused on WFS reconnection.

    size_t m_pwfsWidth{ 0 }; ///< The width of the image

    size_t m_pwfsHeight{ 0 }; ///< The height of the image.

    uint8_t m_pwfsDataType{ 0 }; ///< The ImageStreamIO type code.

    size_t m_pwfsTypeSize{ 0 }; ///< The size of the type, in bytes.

    std::atomic<float> m_fps{ 0 }; ///< Current FPS from the FPS source.

    std::mutex m_resultMutex; ///< Protects the result handoff and worker admission.

    std::condition_variable m_resultReady; ///< Wakes publication for a result or shutdown.

    std::vector<float> m_pendingModes; ///< Latest complete inference, protected by m_resultMutex.

    std::vector<float> m_publishModes; ///< Snapshot owned exclusively by the framegrabber.

    timespec m_pendingTime{ 0, 0 }; ///< Timestamp paired with m_pendingModes.

    bool m_updated{ false }; ///< A complete result is waiting under m_resultMutex.

    bool m_inferencing{ false }; ///< An admitted inference has not yet returned.

    bool m_publishing{ false }; ///< A claimed publication has not reached the base hook.

    bool m_stopping{ false }; ///< Prevents new work during cleanup.

    bool m_sourceReady{ false }; ///< The current WFS attachment passed validation.

    std::string m_fault; ///< Latched runtime failure; no further inference is admitted.

    std::array<void *, 2> m_bindings{}; ///< Device addresses in the engine's validated I/O order.

    bool m_smStarted{ false }; ///< Monitor startup was attempted and needs shutdown.

    bool m_fgStarted{ false }; ///< Framegrabber startup was attempted and needs shutdown.

    bool m_telemStarted{ false }; ///< Telemeter startup was attempted and needs shutdown.

    std::vector<std::string> m_models; ///< Validated startup catalog; configured filenames plus discovered engines.

    bool m_startEnabled{ true }; ///< Preserve legacy automatic startup unless explicitly disabled.

    double m_transitionTimeout{ 30 }; ///< Seconds before an unfinished transition is reported as a fault.

    bool m_enabled{ true }; ///< Admission gate for new inference, protected by m_resultMutex.

    bool m_enableReadback{ true }; ///< Last acknowledged enable state.

    bool m_disabling{ false }; ///< Disable awaits worker quiescence and its one zero publication.

    bool m_zeroPending{ false }; ///< The explicit disable zero has not been claimed.

    bool m_publishZero{ false }; ///< The publisher-owned snapshot completes a disable request.

    bool m_loading{ false }; ///< A queued or running model replacement blocks inference.

    bool m_loadPending{ false }; ///< appLogic must launch the loader for this request.

    bool m_cancelLoad{ false }; ///< A timed-out load must not later replace the model.

    bool m_waitForFreshFrame{ false }; ///< Anchor the source counter on first callback after enabling.

    bool m_requireNewCounter{ false }; ///< Wait for a newer counter than the enable/reconnect anchor.

    uint64_t m_resumeCounter{ 0 }; ///< Counter sampled after enable, on the monitor thread.

    bool m_hadSource{ false }; ///< Distinguishes initial attachment from a reconnection.

    std::thread m_modelThread; ///< Loads candidates only while disabled; joined before resource cleanup.

    std::chrono::steady_clock::time_point m_transitionStart{}; ///< Monotonic request time.

    std::string m_requestedModel; ///< Most recent accepted selection, including failed loads.

    std::string m_loadedModel; ///< Successfully installed engine filename.

    std::string m_activeModel; ///< Model that supplied the last publication while enabled.

    std::chrono::steady_clock::time_point m_outputCheck{}; ///< Publisher-owned output identity check deadline.

    std::string m_requestError; ///< Rejected operator request, separate from a runtime fault.

  public:
    /// Default c'tor.
    nnReconstructor();

    /// Unwind partial startup as well as normal shutdown without retaining owned resources.
    ~nnReconstructor() noexcept;

    /// Register existing parameters and app-local control configuration.
    virtual void setupConfig();

    /// Implementation of loadConfig logic, separated for testing.
    /** This is called by loadConfig().
     */
    int loadConfigImpl( mx::app::appConfigurator &_config /**< [in] an application configuration
                        from which to load values*/
    );

    /// Load configuration and prevent startup when its contract is invalid.
    virtual void loadConfig();

    /// Load the initial engine, register controls and start the existing base workers.
    virtual int appStartup();

    /// Implementation of the FSM for nnReconstructor.
    /**
     * \returns 0 on no critical error
     * \returns -1 on an error requiring shutdown
     */
    virtual int appLogic();

    /// Cancel unclaimed work, join workers and release resources; safe to call repeatedly.
    virtual int appShutdown();

    /// Read the configured startup engine, throwing on incomplete or invalid file input.
    void load_engine( const std::string filename /**< [in] Serialized engine path. */ );

    /// Deserialize and validate the startup engine, then allocate fixed inference buffers.
    void create_engine_context();

    /// Allocate device and host buffers; partial failures are unwound by shutdown.
    void prepare_engine_memory();

    /// Free device allocations after the workers stop; repeated calls are harmless.
    void cleanup_engine_memory();

    /// Destroy the context, engine and runtime in dependency order.
    void cleanup_engine_context();

    /// Validate a candidate without changing the currently installed engine.
    std::array<int, 2> validateEngine( ICudaEngine *candidate /**< [in] Candidate engine. */,
                                       uint32_t    &outputs /**< [out] Modal count. */ );

    /// Discover and validate engine files before registering controls or starting workers.
    void discoverModels();

    /// Build a readable, unique INDI element name from a configured filename.
    std::string modelId( size_t index /**< [in] Configured model index. */ ) const;

    /// Read a complete, nonempty serialized engine.
    std::vector<char> readEngineFile( const std::string &filename /**< [in] Engine path. */ );

    /// Load a replacement while disabled; failure leaves the old model installed.
    void replaceModel( const std::string &name /**< [in] Configured model filename. */ );

    /// Detect incompatible metadata or a replaced external output before claiming a publication.
    bool verifyOutput();

    /// Launch pending model work, report timeouts and update control readbacks.
    void updateControls();

    /// Release host allocations after both data-path threads have stopped.
    void cleanup_host_memory();

    /// Check a CUDA result and throw with the failing operation on error.
    void checkCuda( cudaError_t result /**< [in] CUDA status. */, const char *operation /**< [in] Operation name. */ );

    /// Check IEEE float exponent bits even when fast-math assumes finite arithmetic.
    static bool isFinite( const float &value /**< [in] Configuration or downloaded coefficient. */ );

    /// Check IEEE double exponent bits without changing the app's arithmetic flags.
    static bool isFinite( const double &value /**< [in] Configured transition timeout. */ );

    /// Latch a fault and discard unclaimed results without freeing in-use resources.
    void fail( const std::string &reason /**< [in] Failure description. */ );

    /// Perform the existing preprocessing and synchronous inference on one valid frame.
    void infer( void *curr_src /**< [in] CPU-backed float WFS frame. */ );

    /// Acknowledge publication only after the base has updated metadata and posted.
    int frameGrabberPostPublish( IMAGE *imageStream /**< [in] Published stream. */ );

  protected:
    /// Validate a WFS attachment and invalidate pending results without reallocating buffers.
    int allocate( const dev::shmimT &dummy /**< [in] tag to differentiate shmimMonitor parents.*/ );

    /// Admit one frame, run inference and hand off a complete result or latch its failure.
    int processImage( void              *curr_src /**< [in] Pointer to the current CPU-backed WFS frame. */,
                      const dev::shmimT &dummy /**< [in] Monitor tag. */
    );

    /** \name frameGrabber Interface
     * @{
     */

    /// Attach to a compatible existing output stream.
    /** Attaches only to an existing external stream and checks its size/type/location.
     * Sets the base width, height and datatype; never creates or owns the stream.
     */
    int configureAcquisition();

    /// Gets the frames-per-second readout rate
    /** Used for the latency statistics
     */
    float fps();

    /// Start acquisition.
    /** A no-op in this class.
     */
    int startAcquisition();

    /// Acquire data.
    /** Waits for a protected result or the explicit disable zero, then claims a publisher snapshot.
     */
    int acquireAndCheckValid();

    /// Loads the modevals into the stream
    int loadImageIntoStream( void *dest /**< [out] Destination supplied by the framegrabber. */ );

    /// Complete the base reconfiguration interface; attachment occurs in configureAcquisition.
    int reconfig();

    ///@}

    /** \name INDI interface
     * @{
     */

    pcf::IndiProperty m_indiP_model; ///< Configured model selection, independent of activation.

    /// Queue exactly one configured model only when reconstruction and publication are quiescent.
    INDI_NEWCALLBACK_DECL( nnReconstructor, m_indiP_model );

    pcf::IndiProperty m_indiP_enable; ///< Explicit enable/disable request and acknowledged state.

    /// Request fresh-frame activation or a single acknowledged zero publication.
    INDI_NEWCALLBACK_DECL( nnReconstructor, m_indiP_enable );

    pcf::IndiProperty m_indiP_modelStatus; ///< Requested, loaded, active and transition/error descriptions.

    pcf::IndiProperty m_indiP_fpsSource; ///< Subscription to the configured WFS frame-rate source.

    /// Update the atomic frame-rate estimate used by framegrabber telemetry.
    INDI_SETCALLBACK_DECL( nnReconstructor, m_indiP_fpsSource );

    pcf::IndiProperty m_indiP_fps; ///< Read-only copy of the source frame rate.

    ///@}

    /** \name Telemeter Interface
     * @{
     */

    /// Request the existing framegrabber timing telemetry at its configured cadence.
    int checkRecordTimes();

    /// Record the existing framegrabber timing statistics without redefining their scope.
    int recordTelem( const telem_fgtimings *record /**< [in] Telemetry dispatch tag. */ );

    ///@}
};

inline bool nnReconstructor::isFinite( const float &value )
{
    // Inspect storage by reference; fast-math can assume by-value FP arguments are finite.
    static_assert( sizeof( float ) == sizeof( uint32_t ) && std::numeric_limits<float>::is_iec559 );
    return ( std::bit_cast<uint32_t>( value ) & 0x7f800000U ) != 0x7f800000U;
}

inline bool nnReconstructor::isFinite( const double &value )
{
    static_assert( sizeof( double ) == sizeof( uint64_t ) && std::numeric_limits<double>::is_iec559 );
    return ( std::bit_cast<uint64_t>( value ) & 0x7ff0000000000000ULL ) != 0x7ff0000000000000ULL;
}

inline std::string nnReconstructor::modelId( size_t index ) const
{
    std::string id = "model" + std::to_string( index ) + "_" + m_models[index];
    for( char &character : id )
        if( !std::isalnum( static_cast<unsigned char>( character ) ) && character != '_' )
            character = '_';
    return id;
}

inline std::vector<char> nnReconstructor::readEngineFile( const std::string &filename )
{
    std::ifstream file( filename, std::ios::binary | std::ios::ate );
    if( !file )
        throw std::runtime_error( "cannot open engine: " + filename );
    const auto size = file.tellg();
    if( size <= 0 || static_cast<uintmax_t>( size ) > std::numeric_limits<size_t>::max() )
        throw std::runtime_error( "invalid engine file size: " + filename );
    std::vector<char> data( static_cast<size_t>( size ) );
    file.seekg( 0 );
    if( !file.read( data.data(), size ) )
        throw std::runtime_error( "incomplete engine read: " + filename );
    return data;
}

inline void nnReconstructor::load_engine( const std::string filename )
{
    engineData = readEngineFile( filename );
}

inline void nnReconstructor::create_engine_context()
{
    runtime = createInferRuntime( logger );
    if( !runtime )
        throw std::runtime_error( "createInferRuntime failed" );
    engine = runtime->deserializeCudaEngine( engineData.data(), engineData.size() );
    if( !engine )
        throw std::runtime_error( "engine deserialization failed" );
    const auto indices    = validateEngine( engine, outputSize );
    const int  imageIndex = indices[0], outputIndex = indices[1];
    inputC = Npup;
    inputH = inputW = m_pupPix;
    inputSize       = Npup * pixels_per_quadrant;
    context         = engine->createExecutionContext();
    if( !context )
        throw std::runtime_error( "createExecutionContext failed" );
    prepare_engine_memory();
    m_bindings[imageIndex]  = d_input;
    m_bindings[outputIndex] = d_output;
    log<text_log>( std::format( "Tensor input dimensions: {}x{}x{}; tensor output dimensions: {}; I/O precision: FP32",
                                inputC,
                                inputH,
                                inputW,
                                outputSize ),
                   logPrio::LOG_INFO );
}

inline std::array<int, 2> nnReconstructor::validateEngine( ICudaEngine *candidate, uint32_t &outputs )
{
    constexpr int count = 2;
    if( candidate->getNbIOTensors() != count )
        throw std::runtime_error( "unexpected engine I/O count" );
    int imageIndex = -1, outputIndex = -1;
    for( int i = 0; i < count; ++i )
    {
        const char *name = candidate->getIOTensorName( i );
        if( !name || candidate->getTensorDataType( name ) != DataType::kFLOAT ||
            candidate->getTensorFormat( name ) != TensorFormat::kLINEAR ||
            candidate->getTensorLocation( name ) != TensorLocation::kDEVICE )
            throw std::runtime_error( "unsupported engine I/O dtype, layout or location" );
        const auto dims   = candidate->getTensorShape( name );
        size_t     volume = 1;
        if( dims.nbDims < 1 )
            throw std::runtime_error( "invalid tensor rank" );
        for( int d = 0; d < dims.nbDims; ++d )
        {
            if( dims.d[d] <= 0 || volume > size_t( std::numeric_limits<int>::max() ) / dims.d[d] )
                throw std::runtime_error( "dynamic, empty or oversized tensor" );
            volume *= dims.d[d];
        }
        if( candidate->getTensorIOMode( name ) == TensorIOMode::kOUTPUT )
        {
            if( outputIndex >= 0 || dims.nbDims != 2 || dims.d[0] != 1 )
                throw std::runtime_error( "expected one batch-one modal output" );
            outputIndex = i;
            outputs     = dims.d[1];
        }
        else if( candidate->getTensorIOMode( name ) == TensorIOMode::kINPUT && dims.nbDims == 4 )
        {
            if( imageIndex >= 0 || dims.d[0] != 1 || dims.d[1] != Npup || dims.d[2] != m_pupPix ||
                dims.d[3] != m_pupPix )
                throw std::runtime_error( "engine image shape does not match configured padded pupils" );
            imageIndex = i;
        }
        else
            throw std::runtime_error( "unsupported engine tensor role or shape" );
    }
    if( imageIndex < 0 || outputIndex < 0 )
        throw std::runtime_error( "missing required engine tensor" );
    return { imageIndex, outputIndex };
}

inline void nnReconstructor::discoverModels()
{
    // Preserve explicit entries (including nonstandard extensions), then append sorted discoveries.
    auto                     candidates = m_models;
    std::vector<std::string> discovered;
    for( const auto &entry : std::filesystem::directory_iterator( engineDirs ) )
    {
        const auto extension = entry.path().extension().string();
        if( entry.is_regular_file() && ( extension == ".trt" || extension == ".engine" || extension == ".plan" ) )
            discovered.push_back( entry.path().filename().string() );
    }
    std::sort( discovered.begin(), discovered.end() );
    candidates.insert( candidates.end(), discovered.begin(), discovered.end() );
    m_models = { engineName }; // Already loaded and validated; never select another startup model implicitly.
    std::vector<std::string> checked{ engineName };
    for( const auto &name : candidates )
    {
        if( std::find( checked.begin(), checked.end(), name ) != checked.end() )
            continue;
        checked.push_back( name );
        try
        {
            auto                         data = readEngineFile( engineDirs + "/" + name );
            std::unique_ptr<ICudaEngine> candidate( runtime->deserializeCudaEngine( data.data(), data.size() ) );
            if( !candidate )
                throw std::runtime_error( "engine deserialization failed" );
            uint32_t outputs = 0;
            validateEngine( candidate.get(), outputs );
            if( outputs != outputSize )
                throw std::runtime_error( "modal count differs from startup model" );
            std::unique_ptr<IExecutionContext> candidateContext( candidate->createExecutionContext() );
            if( !candidateContext )
                throw std::runtime_error( "createExecutionContext failed" );
            m_models.push_back( name );
        }
        catch( const std::exception &error )
        {
            log<text_log>( "Skipping model " + name + ": " + error.what(), logPrio::LOG_WARNING );
        }
    }
}

inline void nnReconstructor::replaceModel( const std::string &name )
{
    auto                      data = readEngineFile( engineDirs + "/" + name );
    std::unique_ptr<IRuntime> nextRuntime( createInferRuntime( logger ) );
    if( !nextRuntime )
        throw std::runtime_error( "createInferRuntime failed" );
    std::unique_ptr<ICudaEngine> nextEngine( nextRuntime->deserializeCudaEngine( data.data(), data.size() ) );
    if( !nextEngine )
        throw std::runtime_error( "engine deserialization failed" );
    uint32_t   outputs = 0;
    const auto indices = validateEngine( nextEngine.get(), outputs );
    if( outputs != outputSize )
        throw std::runtime_error( "replacement must preserve the configured modal count" );
    std::unique_ptr<IExecutionContext> nextContext( nextEngine->createExecutionContext() );
    if( !nextContext )
        throw std::runtime_error( "createExecutionContext failed" );
    { // mutex scope
        std::lock_guard<std::mutex> lock( m_resultMutex );
        if( m_stopping || m_cancelLoad ||
            std::chrono::duration<double>( std::chrono::steady_clock::now() - m_transitionStart ).count() >
                m_transitionTimeout )
            throw std::runtime_error( "model load cancelled or timed out before installation" );
        // Admission is closed throughout loading. Swapping cannot race inference or publication.
        auto *oldRuntime = runtime;
        runtime          = nextRuntime.release();
        nextRuntime.reset( oldRuntime );
        auto *oldEngine = engine;
        engine          = nextEngine.release();
        nextEngine.reset( oldEngine );
        auto *oldContext = context;
        context          = nextContext.release();
        nextContext.reset( oldContext );
        m_bindings.fill( nullptr );
        m_bindings[indices[0]] = d_input;
        m_bindings[indices[1]] = d_output;
        engineData.swap( data );
        m_loadedModel = name;
        m_activeModel.clear();
        m_updated           = false;
        m_waitForFreshFrame = true;
        m_fault.clear();
    }
}

inline void nnReconstructor::checkCuda( cudaError_t result, const char *operation )
{
    if( result != cudaSuccess )
        throw std::runtime_error( std::string( operation ) + ": " + cudaGetErrorString( result ) );
}

inline void nnReconstructor::prepare_engine_memory()
{
    constexpr size_t size = sizeof( float );
    checkCuda( cudaMalloc( reinterpret_cast<void **>( &d_input ), inputSize * size ), "allocate image input" );
    checkCuda( cudaMalloc( reinterpret_cast<void **>( &d_output ), outputSize * size ), "allocate modal output" );
    pp_image = new float[inputSize]{};
    modeval.resize( outputSize, 1 );
    m_pendingModes.resize( outputSize );
    m_publishModes.resize( outputSize );
}

inline void nnReconstructor::cleanup_engine_memory()
{
    for( auto pointer : { d_input, d_output } )
        if( pointer && cudaFree( pointer ) != cudaSuccess )
            log<software_error>( { "cudaFree failed during cleanup" } );
    d_input = d_output = nullptr;
    m_bindings.fill( nullptr );
}

inline void nnReconstructor::cleanup_engine_context()
{
    delete context;
    context = nullptr;
    delete engine;
    engine = nullptr;
    delete runtime;
    runtime = nullptr;
}

inline void nnReconstructor::cleanup_host_memory()
{
    delete[] pp_image;
    pp_image = nullptr;
}

inline nnReconstructor::~nnReconstructor() noexcept
{
    appShutdown();
}

inline void nnReconstructor::fail( const std::string &reason )
{
    { // mutex scope
        std::lock_guard<std::mutex> lock( m_resultMutex );
        m_fault   = reason;
        m_enabled = false;
        m_updated = false;
    }
    m_resultReady.notify_all();
    log<software_error>( { reason } );
}

inline nnReconstructor::nnReconstructor() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    return;
}

inline void nnReconstructor::setupConfig()
{
    std::cout << "setupConfig()" << std::endl;

    SHMIMMONITOR_SETUP_CONFIG( config );
    FRAMEGRABBER_SETUP_CONFIG( config );
    TELEMETER_SETUP_CONFIG( config );

    config.add( "parameters.engineDirs",
                "",
                "parameters.engineDirs",
                argType::Required,
                "parameters",
                "engineDirs",
                false,
                "string",
                "The path to the directory with the TRT engine." );

    config.add( "parameters.engineName",
                "",
                "parameters.engineName",
                argType::Required,
                "parameters",
                "engineName",
                false,
                "string",
                "Name of the TRT engine." );

    config.add( "parameters.rebuildEngine",
                "",
                "parameters.rebuildEngine",
                argType::Required,
                "parameters",
                "rebuildEngine",
                false,
                "bool",
                "If true the engine will be rebuild." );

    config.add( "parameters.imageNorm",
                "",
                "parameters.imageNorm",
                argType::Required,
                "parameters",
                "imageNorm",
                false,
                "float",
                "Normalization term for the preprocessed images." );

    config.add( "parameters.modalNorm",
                "",
                "parameters.modalNorm",
                argType::Required,
                "parameters",
                "modalNorm",
                false,
                "float",
                "Normalization term for the modal coefficients." );

    config.add( "parameters.m_pupPix",
                "",
                "parameters.m_pupPix",
                argType::Required,
                "parameters",
                "m_pupPix",
                false,
                "int",
                "Number of pixels across a PWFS pupil." );

    config.add( "parameters.pup_offset1_x",
                "",
                "parameters.pup_offset1_x",
                argType::Required,
                "parameters",
                "pup_offset1_x",
                false,
                "int",
                "Horizontal offset to the top left of the closest set op PWFS pupils." );

    config.add( "parameters.pup_offset1_y",
                "",
                "parameters.pup_offset1_y",
                argType::Required,
                "parameters",
                "pup_offset1_y",
                false,
                "int",
                "Vertical offset to the top left of the closest set op PWFS pupils." );

    config.add( "parameters.pup_offset2_x",
                "",
                "parameters.pup_offset2_x",
                argType::Required,
                "parameters",
                "pup_offset2_x",
                false,
                "int",
                "Horizontal offset to the top left of the furthest set op PWFS pupils." );

    config.add( "parameters.pup_offset2_y",
                "",
                "parameters.pup_offset2_y",
                argType::Required,
                "parameters",
                "pup_offset2_y",
                false,
                "int",
                "Vertical offset to the top left of the furthest set op PWFS pupils." );

    config.add( "parameters.models",
                "",
                "parameters.models",
                argType::Required,
                "parameters",
                "models",
                false,
                "vector<string>",
                "Optional extra engine filenames; compatible files in engineDirs are also discovered at startup." );
    config.add( "parameters.startEnabled",
                "",
                "parameters.startEnabled",
                argType::Required,
                "parameters",
                "startEnabled",
                false,
                "bool",
                "Start reconstructing automatically (legacy default true)." );
    config.add(
        "parameters.transitionTimeout",
        "",
        "parameters.transitionTimeout",
        argType::Required,
        "parameters",
        "transitionTimeout",
        false,
        "double",
        "Seconds before an unfinished disable/load is reported as a fault; does not cancel native CUDA calls." );
}

inline int nnReconstructor::loadConfigImpl( mx::app::appConfigurator &_config )
{
    std::cout << "loadConfigImpl()" << std::endl;

    SHMIMMONITOR_LOAD_CONFIG( _config );

    frameGrabberT::m_ownShmim = false;
    FRAMEGRABBER_LOAD_CONFIG( _config );

    _config( engineDirs, "parameters.engineDirs" );
    _config( engineName, "parameters.engineName" );
    _config( rebuildEngine, "parameters.rebuildEngine" );
    _config( imageNorm, "parameters.imageNorm" );
    _config( modalNorm, "parameters.modalNorm" );

    _config( m_pupPix, "parameters.m_pupPix" );
    _config( pup_offset1_x, "parameters.pup_offset1_x" );
    _config( pup_offset1_y, "parameters.pup_offset1_y" );
    _config( pup_offset2_x, "parameters.pup_offset2_x" );
    _config( pup_offset2_y, "parameters.pup_offset2_y" );

    _config( m_fpsSource, "recon.fpsSource" );
    _config( m_models, "parameters.models" );
    _config( m_startEnabled, "parameters.startEnabled" );
    _config( m_transitionTimeout, "parameters.transitionTimeout" );
    if( engineName.empty() || !isFinite( m_transitionTimeout ) || m_transitionTimeout <= 0 )
        return log<software_error, -1>( { "engineName and a positive finite transitionTimeout are required" } );
    if( std::find( m_models.begin(), m_models.end(), engineName ) == m_models.end() )
        m_models.insert( m_models.begin(), engineName );
    for( size_t i = 0; i < m_models.size(); ++i )
        if( m_models[i].empty() ||
            std::find( m_models.begin(), m_models.begin() + i, m_models[i] ) != m_models.begin() + i )
            return log<software_error, -1>( { "model filenames must be nonempty and unique" } );

    TELEMETER_LOAD_CONFIG( _config );

    if( true )
    {
        std::cout << "Debug configuration loading: " << std::endl;
        std::cout << "engineDirs: " << engineDirs << std::endl;
        std::cout << "engineName: " << engineName << std::endl;
        std::cout << "rebuildEngine: " << rebuildEngine << std::endl;
        std::cout << "imageNorm: " << imageNorm << std::endl;
        std::cout << "modalNorm: " << modalNorm << std::endl;
        std::cout << "modeval Channel: " << frameGrabberT::m_shmimName << std::endl;

        std::cout << "m_pupPix: " << m_pupPix << std::endl;
        std::cout << "pup_offset1_x: " << pup_offset1_x << std::endl;
        std::cout << "pup_offset1_y: " << pup_offset1_y << std::endl;
        std::cout << "pup_offset2_x: " << pup_offset2_x << std::endl;
        std::cout << "pup_offset2_y: " << pup_offset2_y << std::endl;
    }

    if( m_pupPix <= 2 * zeroPad || m_pupPix > 23170 || !isFinite( imageNorm ) || pup_offset1_x < 0 ||
        pup_offset1_y < 0 || pup_offset2_x < 0 || pup_offset2_y < 0 )
        return log<software_error, -1>( { "invalid pupil crop or image normalization" } );
    pixels_per_quadrant = m_pupPix * m_pupPix;
    return 0;
}

inline void nnReconstructor::loadConfig()
{
    if( loadConfigImpl( config ) < 0 )
        m_shutdown = true;
}

inline int nnReconstructor::appStartup()
{

    REG_INDI_SETPROP( m_indiP_fpsSource, m_fpsSource, std::string( "fps" ) );

    createROIndiNumber( m_indiP_fps, "fps" );
    m_indiP_fps.add( pcf::IndiElement( "current" ) );
    if( registerIndiPropertyReadOnly( m_indiP_fps ) < 0 )
    {
        log<software_error>( { "" } );
        return -1;
    }

    try
    {
        load_engine( engineDirs + "/" + engineName );
        create_engine_context();
        discoverModels();
        m_requestedModel = m_loadedModel = engineName;
        m_enabled = m_enableReadback = m_startEnabled;
        std::vector<std::string> ids;
        for( size_t i = 0; i < m_models.size(); ++i )
            ids.push_back( modelId( i ) );
        if( createStandardIndiSelectionSw( m_indiP_model, "model", ids, m_models ) < 0 ||
            registerIndiPropertyNew( m_indiP_model, INDI_NEWCALLBACK( m_indiP_model ) ) < 0 )
            throw std::runtime_error( "cannot register model selection" );
        createStandardIndiToggleSw( m_indiP_enable, "enable" );
        if( registerIndiPropertyNew( m_indiP_enable, INDI_NEWCALLBACK( m_indiP_enable ) ) < 0 )
            throw std::runtime_error( "cannot register enable control" );
        createROIndiText( m_indiP_modelStatus, "model_status", "requested" );
        for( const char *element : { "loaded", "active", "status", "error" } )
            m_indiP_modelStatus.add( pcf::IndiElement( element ) );
        if( registerIndiPropertyReadOnly( m_indiP_modelStatus ) < 0 )
            throw std::runtime_error( "cannot register model status" );
        updateControls();
        // The base startup macros return on failure; the destructor also unwinds partial startup.
        m_fgStarted = true;
        FRAMEGRABBER_APP_STARTUP;
        m_smStarted = true;
        SHMIMMONITOR_APP_STARTUP;
        m_telemStarted = true;
        TELEMETER_APP_STARTUP;
    }
    catch( const std::exception &error )
    {
        fail( error.what() );
        appShutdown();
        return -1;
    }

    state( stateCodes::OPERATING );
    return 0;
}

inline int nnReconstructor::appLogic()
{
    SHMIMMONITOR_APP_LOGIC;

    FRAMEGRABBER_APP_LOGIC;

    TELEMETER_APP_LOGIC;

    updateControls();

    std::unique_lock<std::mutex> lock( m_indiMutex );

    SHMIMMONITOR_UPDATE_INDI;

    FRAMEGRABBER_UPDATE_INDI;

    return 0;
}

inline int nnReconstructor::appShutdown()
{
    { // mutex scope
        std::lock_guard<std::mutex> lock( m_resultMutex );
        m_stopping    = true;
        m_enabled     = false;
        m_zeroPending = false;
        m_updated     = false;
    }
    m_shutdown = true;
    m_resultReady.notify_all();
    if( m_smStarted )
    {
        shmimMonitorT::appShutdown();
        m_smStarted = false;
    }
    if( m_fgStarted )
    {
        frameGrabberT::appShutdown();
        m_fgStarted = false;
    }
    if( m_telemStarted )
    {
        telemeterT::appShutdown();
        m_telemStarted = false;
    }
    if( m_modelThread.joinable() )
        m_modelThread.join();
    cleanup_engine_context();
    cleanup_engine_memory();
    cleanup_host_memory();
    return 0;
}

inline int nnReconstructor::allocate( const dev::shmimT &dummy )
{
    static_cast<void>( dummy );
    const auto  &md    = shmimMonitorT::m_imageStream.md[0];
    const size_t crop  = m_pupPix - 2 * zeroPad;
    const size_t width = shmimMonitorT::m_width, height = shmimMonitorT::m_height;
    const bool   valid = md.datatype == _DATATYPE_FLOAT && md.location == -1 && ( md.naxis == 2 || md.naxis == 3 ) &&
                       shmimMonitorT::m_imageStream.array.raw && size_t( pup_offset1_x ) + crop <= width &&
                       size_t( pup_offset2_x ) + crop <= width && size_t( pup_offset1_y ) + crop <= height &&
                       size_t( pup_offset2_y ) + crop <= height;
    { // mutex scope
        std::lock_guard<std::mutex> lock( m_resultMutex );
        m_updated     = false;
        m_sourceReady = valid;
        if( m_hadSource )
            m_waitForFreshFrame = true;
        m_hadSource  = true;
        m_pwfsWidth  = width;
        m_pwfsHeight = height;
    }
    // Buffers belong to the loaded engine, not to a WFS attachment.
    if( !valid )
        fail( "incompatible WFS stream or crop; reconnect a valid stream and reload before resuming" );
    return 0;
}

inline int nnReconstructor::processImage( void *curr_src, const dev::shmimT &dummy )
{
    static_cast<void>( dummy );
    { // mutex scope
        std::lock_guard<std::mutex> lock( m_resultMutex );
        if( m_stopping || !m_enabled || m_loading || !m_fault.empty() || !m_sourceReady )
            return 0;
        const uint64_t counter = shmimMonitorT::m_imageStream.md[0].cnt0;
        if( m_waitForFreshFrame )
        {
            m_resumeCounter     = counter;
            m_waitForFreshFrame = false;
            m_requireNewCounter = true;
            return 0;
        }
        if( m_requireNewCounter && counter == m_resumeCounter )
            return 0;
        m_requireNewCounter = false;
        m_inferencing       = true;
    }
    timespec timestamp{};
    try
    {
        if( clock_gettime( CLOCK_REALTIME, &timestamp ) != 0 )
            throw std::runtime_error( "clock_gettime failed" );
        infer( curr_src );
        { // mutex scope
            std::lock_guard<std::mutex> lock( m_resultMutex );
            if( !m_stopping && m_enabled && m_fault.empty() )
            {
                std::copy( modeval.data(), modeval.data() + outputSize, m_pendingModes.begin() );
                m_pendingTime = timestamp;
                m_updated     = true;
            }
        }
    }
    catch( const std::exception &error )
    {
        fail( error.what() );
    }
    catch( ... )
    {
        fail( "unknown inference failure" );
    }
    { // mutex scope
        std::lock_guard<std::mutex> lock( m_resultMutex );
        m_inferencing = false;
    }
    m_resultReady.notify_all();
    return 0;
}

inline void nnReconstructor::infer( void *curr_src )
{
    // aol_imwfs2 is reference and dark subtracted and is power normalized.

    Eigen::Map<eigenImage<float>> pwfsIm( reinterpret_cast<float *>( curr_src ), m_pwfsHeight, m_pwfsWidth );

    // Split up the four pupils for the Neural Network.
    int ki = 0;

    for( int col_i = -zeroPad; col_i < ( m_pupPix - zeroPad ); ++col_i )
    {
        for( int row_i = -zeroPad; row_i < ( m_pupPix - zeroPad ); ++row_i )
        {
            if( ( col_i < 0 ) or ( col_i >= ( m_pupPix - 2 * zeroPad ) ) or ( row_i < 0 ) or
                ( row_i >= ( m_pupPix - 2 * zeroPad ) ) )
            {
                pp_image[ki]                           = 0;
                pp_image[ki + pixels_per_quadrant]     = 0;
                pp_image[ki + 2 * pixels_per_quadrant] = 0;
                pp_image[ki + 3 * pixels_per_quadrant] = 0;
            }
            else
            {
                pp_image[ki] = imageNorm * (realT)pwfsIm( pup_offset1_y + row_i, pup_offset1_x + col_i );
                pp_image[ki + pixels_per_quadrant] =
                    imageNorm * (realT)pwfsIm( pup_offset1_y + row_i, pup_offset2_x + col_i );
                pp_image[ki + 2 * pixels_per_quadrant] =
                    imageNorm * (realT)pwfsIm( pup_offset2_y + row_i, pup_offset1_x + col_i );
                pp_image[ki + 3 * pixels_per_quadrant] =
                    imageNorm * (realT)pwfsIm( pup_offset2_y + row_i, pup_offset2_x + col_i );
            }
            ++ki;
        }
    }

    // Copy input data to device
    checkCuda( cudaMemcpy( d_input, pp_image, inputSize * sizeof( float ), cudaMemcpyHostToDevice ), "cudaMemcpy" );

    // Keep synchronous execution; tensor roles determine the binding slots.
    if( !context->executeV2( m_bindings.data() ) )
        throw std::runtime_error( "executeV2 failed" );

    // Copy output data back to host
    checkCuda( cudaMemcpy( modeval.data(), d_output, outputSize * sizeof( float ), cudaMemcpyDeviceToHost ),
               "cudaMemcpy" );

    for( uint32_t i = 0; i < outputSize; ++i )
        if( !isFinite( modeval.data()[i] ) )
            throw std::runtime_error( "non-finite modal output" );
}

inline int nnReconstructor::configureAcquisition()
{
    static bool logged = false;

    int rv = openShmim();
    if( rv != 0 )
    {
        return rv;
    }

    if( frameGrabberT::m_width != outputSize || frameGrabberT::m_height != 1 ||
        frameGrabberT::m_dataType != _DATATYPE_FLOAT || !frameGrabberT::m_imageStream ||
        !frameGrabberT::m_imageStream->array.raw || frameGrabberT::m_imageStream->md[0].location != -1 ||
        frameGrabberT::m_imageStream->md[0].naxis < 1 || frameGrabberT::m_imageStream->md[0].naxis > 3 )
    {
        if( !logged )
        {
            log<text_log>( std::format( "{} is wrong size ({}x{}) or type ({})",
                                        frameGrabberT::m_shmimName,
                                        frameGrabberT::m_width,
                                        frameGrabberT::m_height,
                                        frameGrabberT::m_dataType ),
                           logPrio::LOG_INFO );
            logged = true;
        }
        return 1;
    }

    logged        = false;
    m_outputCheck = std::chrono::steady_clock::now();

    return 0;
}

inline float nnReconstructor::fps()
{
    return m_fps.load();
}

inline int nnReconstructor::startAcquisition()
{
    return 0;
}

inline bool nnReconstructor::verifyOutput()
{
    const auto *stream = frameGrabberT::m_imageStream;
    if( !stream || !stream->md || !stream->array.raw )
    {
        fail( "output mapping unavailable" );
        return false;
    }
    const auto &md = stream->md[0];
    if( md.datatype != _DATATYPE_FLOAT || md.location != -1 || md.size[0] != outputSize || md.naxis < 1 ||
        md.naxis > 3 || ( md.naxis > 1 && md.size[1] != 1 ) || md.sem <= 0 ||
        ( md.naxis == 3 && md.size[2] != frameGrabberT::m_circBuffLength ) )
    {
        fail( "output metadata changed; publication stopped" );
        return false;
    }
    const auto now = std::chrono::steady_clock::now();
    if( now >= m_outputCheck )
    {
        m_outputCheck = now + std::chrono::seconds( 1 );
        char filename[1024];
        ImageStreamIO_filename( filename, sizeof( filename ), frameGrabberT::m_shmimName.c_str() );
        struct stat info;
        if( ::stat( filename, &info ) != 0 || info.st_ino != frameGrabberT::m_inode )
        {
            fail( "output stream removed or replaced; reattachment and explicit reload required" );
            return false;
        }
    }
    return true;
}

inline int nnReconstructor::acquireAndCheckValid()
{
    std::unique_lock<std::mutex> lock( m_resultMutex );
    m_resultReady.wait_for( lock,
                            std::chrono::seconds( 1 ),
                            [this] { return m_stopping || m_updated || ( m_zeroPending && !m_inferencing ); } );
    if( m_stopping )
        return 1;
    lock.unlock();
    const bool outputValid = verifyOutput();
    lock.lock();
    if( !outputValid )
        return -1;
    if( m_stopping )
        return 1;
    m_publishZero = m_zeroPending && !m_inferencing;
    if( m_publishZero )
    {
        if( clock_gettime( CLOCK_REALTIME, &( frameGrabberT::m_currImageTimestamp ) ) != 0 )
        {
            m_fault       = "disable incomplete: cannot timestamp zero publication; retry Off";
            m_zeroPending = false;
            return 1;
        }
        std::fill( m_publishModes.begin(), m_publishModes.end(), 0.F );
        m_zeroPending = false;
    }
    else
    {
        if( !m_enabled || m_loading || !m_fault.empty() || !m_updated )
            return 1;
        std::copy( m_pendingModes.begin(), m_pendingModes.end(), m_publishModes.begin() );
        frameGrabberT::m_currImageTimestamp = m_pendingTime;
    }
    m_updated    = false;
    m_publishing = true;
    return 0;
}

inline int nnReconstructor::loadImageIntoStream( void *dest )
{
    memcpy( dest, m_publishModes.data(), m_publishModes.size() * sizeof( float ) );
    return 0;
}

inline int nnReconstructor::frameGrabberPostPublish( IMAGE *imageStream )
{
    static_cast<void>( imageStream );
    { // mutex scope
        std::lock_guard<std::mutex> lock( m_resultMutex );
        m_publishing = false;
        if( m_publishZero )
        {
            m_disabling      = false;
            m_enableReadback = false;
            m_activeModel.clear();
        }
        else if( m_enabled && m_fault.empty() )
            m_activeModel = m_loadedModel;
        else
            m_activeModel.clear();
    }
    m_resultReady.notify_all();
    return 0;
}

inline int nnReconstructor::reconfig()
{
    return 0;
}

inline void nnReconstructor::updateControls()
{
    bool        launch = false, finished = false;
    std::string name;
    { // mutex scope
        std::lock_guard<std::mutex> lock( m_resultMutex );
        finished = !m_loading;
        if( m_loadPending && !m_stopping )
        {
            launch        = true;
            name          = m_requestedModel;
            m_loadPending = false;
        }
        if( ( m_loading || m_disabling ) &&
            std::chrono::duration<double>( std::chrono::steady_clock::now() - m_transitionStart ).count() >
                m_transitionTimeout )
        {
            m_cancelLoad = m_loading;
            if( m_fault.empty() )
                m_fault = m_loading ? "load incomplete: worker must finish, then reload explicitly"
                                    : "disable incomplete: zero publication not acknowledged";
        }
    }
    if( ( finished || launch ) && m_modelThread.joinable() )
        m_modelThread.join();
    if( launch )
    {
        try
        {
            m_modelThread = std::thread(
                [this, name]
                {
                    try
                    {
                        replaceModel( name );
                    }
                    catch( const std::exception &error )
                    {
                        fail( error.what() );
                    }
                    catch( ... )
                    {
                        fail( "unknown model loading failure" );
                    }
                    { // mutex scope
                        std::lock_guard<std::mutex> lock( m_resultMutex );
                        m_loading = false;
                    }
                    m_resultReady.notify_all();
                } );
        }
        catch( const std::exception &error )
        {
            fail( error.what() );
            std::lock_guard<std::mutex> lock( m_resultMutex );
            m_loading = false;
        }
    }
    bool                                 enabled;
    pcf::IndiProperty::PropertyStateType status;
    std::string                          phase, requested, loaded, active, error;
    { // mutex scope
        std::lock_guard<std::mutex> lock( m_resultMutex );
        if( !m_fault.empty() && !m_inferencing && !m_publishing && !m_disabling )
        {
            m_enableReadback = false;
            m_activeModel.clear();
        }
        const bool busy = m_disabling || m_loading;
        status          = !m_fault.empty() || !m_requestError.empty() ? INDI_ALERT : busy ? INDI_BUSY : INDI_OK;
        phase           = m_stopping              ? "stopping"
                          : m_disabling           ? "disabling: zero pending"
                          : m_loading             ? "loading"
                          : !m_fault.empty()      ? "fault: reload required"
                          : !m_enabled            ? "disabled"
                          : m_activeModel.empty() ? "waiting for fresh result"
                                                  : "active";
        enabled         = m_enableReadback;
        requested       = m_requestedModel;
        loaded          = m_loadedModel;
        active          = m_activeModel;
        error           = m_fault.empty() ? m_requestError : m_fault;
    }
    // Network/property updates must not hold the real-time handoff lock.
    std::lock_guard<std::mutex> indiLock( m_indiMutex );
    updateSwitchIfChanged( m_indiP_enable, "toggle", enabled ? pcf::IndiElement::On : pcf::IndiElement::Off, status );
    for( size_t i = 0; i < m_models.size(); ++i )
        updateSwitchIfChanged( m_indiP_model,
                               modelId( i ),
                               m_models[i] == requested ? pcf::IndiElement::On : pcf::IndiElement::Off,
                               status );
    updateIfChanged( m_indiP_modelStatus, "requested", requested, status );
    updateIfChanged( m_indiP_modelStatus, "loaded", loaded, status );
    updateIfChanged( m_indiP_modelStatus, "active", active, status );
    updateIfChanged( m_indiP_modelStatus, "status", phase, status );
    updateIfChanged( m_indiP_modelStatus, "error", error, status );
}

INDI_NEWCALLBACK_DEFN( nnReconstructor, m_indiP_enable )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_enable, ipRecv );
    if( !ipRecv.find( "toggle" ) )
        return 0;
    { // mutex scope
        std::lock_guard<std::mutex> lock( m_resultMutex );
        const auto                  value = ipRecv["toggle"].getSwitchState();
        if( value != pcf::IndiElement::On && value != pcf::IndiElement::Off )
        {
            m_requestError = "enable requires On or Off";
            return -1;
        }
        const bool on = value == pcf::IndiElement::On;
        if( m_stopping || ( on && ( m_loading || m_disabling || !m_fault.empty() || !m_sourceReady ) ) )
        {
            m_requestError = "enable rejected: shutdown, transition, source validation or fault recovery pending";
            return -1;
        }
        m_requestError.clear();
        if( on && !m_enabled )
        {
            m_enabled           = true;
            m_enableReadback    = true;
            m_waitForFreshFrame = true;
            m_activeModel.clear();
        }
        else if( !on && m_disabling && !m_zeroPending && !m_publishing && !m_inferencing )
        {
            m_zeroPending     = true;
            m_transitionStart = std::chrono::steady_clock::now();
        }
        else if( !on && !m_disabling && ( m_enabled || m_enableReadback ) )
        {
            m_enabled   = false;
            m_disabling = m_zeroPending = true;
            m_transitionStart           = std::chrono::steady_clock::now();
            m_updated                   = false;
        }
    }
    m_resultReady.notify_all();
    return 0;
}

INDI_NEWCALLBACK_DEFN( nnReconstructor, m_indiP_model )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_model, ipRecv );
    std::lock_guard<std::mutex> lock( m_resultMutex );
    int                         selected = -1;
    for( const auto &element : ipRecv.getElements() )
    {
        if( !m_indiP_model.find( element.first ) )
        {
            m_requestError = "unknown model selection element";
            return -1;
        }
        if( element.second.getSwitchState() != pcf::IndiElement::On )
            continue;
        if( selected >= 0 )
        {
            m_requestError = "select exactly one model";
            return -1;
        }
        for( size_t i = 0; i < m_models.size(); ++i )
            if( element.first == modelId( i ) )
                selected = i;
    }
    if( selected < 0 )
    {
        m_requestError = "select exactly one model";
        return -1;
    }
    if( m_stopping || m_enabled || m_disabling || m_loading || m_publishing || m_inferencing )
    {
        m_requestError = "model load requires completed disable and quiescent workers";
        return -1;
    }
    m_requestError.clear();
    m_requestedModel = m_models[selected];
    m_loading = m_loadPending = true;
    m_cancelLoad              = false;
    m_transitionStart         = std::chrono::steady_clock::now();
    return 0;
}

INDI_SETCALLBACK_DEFN( nnReconstructor, m_indiP_fpsSource )( const pcf::IndiProperty &ipRecv )
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
        updateIfChanged( m_indiP_fps, "current", m_fps.load() );
    }

    return 0;
}

inline int nnReconstructor::checkRecordTimes()
{
    return telemeterT::checkRecordTimes( telem_fgtimings() );
}

inline int nnReconstructor::recordTelem( const telem_fgtimings * )
{
    return recordFGTimings( true );
}

} // namespace app
} // namespace MagAOX

#endif // nnReconstructor_hpp
