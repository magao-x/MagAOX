/** \file nnReconstructor.hpp
 * \brief The MagAO-X generic ImageStreamIO stream integrator
 *
 * \ingroup app_files
 */

#ifndef nnReconstructor_hpp
#define nnReconstructor_hpp

#include <NvInfer.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <limits>

#include <mx/improc/eigenCube.hpp>
#include <mx/improc/eigenImage.hpp>

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

using namespace nvinfer1;

// Logger for TensorRT info/warning/errors
class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << msg << std::endl;
        }
    }
};

// #define MAGAOX_CURRENT_SHA1 0
// #define MAGAOX_REPO_MODIFIED 0

void halfToFloatArray(float* dst, const half* src, size_t num_elements) {
    for (size_t i = 0; i < num_elements; ++i) {
        dst[i] = __half2float(src[i]);
    }
}

void floatToHalfArray(half* dst, const float* src, size_t num_elements) {
    for (size_t i = 0; i < num_elements; ++i) {
        dst[i] = __float2half(src[i]);  // Convert each float to half-precision
    }
}


namespace MagAOX
{
namespace app
{

class nnReconstructor : public MagAOXApp<true>, public dev::shmimMonitor<nnReconstructor>, public dev::frameGrabber<nnReconstructor>, public dev::telemeter<nnReconstructor>
{
    // Give the test harness access.
    friend class nnReconstructor_test;

    friend class dev::shmimMonitor<nnReconstructor>;

    // The base shmimMonitor type
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
    /** \name Configurable Parameters
     *@{
     */
    std::string dataDirs;     // Location where the data (onnx file, engine, WFS reference) is stored
    std::string engineName;   // Name of the engine
    std::string engineDirs;   // Name of the engine
    bool rebuildEngine {false};       // If true, it will rebuild the engine and save it at engineName

    std::string m_fpsSource{ "camwfs" }; /**< Device name for getting fps of the loop.
                                              This device must have *.fps.current.  Default is camwfs*/

                                              ///@}

    Logger logger;			  // The tensorRT logger
    std::vector<char> engineData; // for loading the engine file.

    IRuntime* runtime {nullptr};
    ICudaEngine* engine {nullptr};
    IExecutionContext* context {nullptr};
    int inputC {0};
    int inputH {0};
    int inputW {0};
    int inputSize {0};
    int input2Size {4};
    uint32_t outputSize {0};
    int zeroPad { 2 };

    float* d_input {nullptr};
    float* d_input2 {nullptr};
    float* d_output {nullptr};
    bool use_fp16 = false;
    bool explicit_tt = false;

    float imageNorm; // Normalization constant for the image intensities
    float modalNorm; // Normalization constant for the modal coefficients

    int m_pupPix;      // Number of pixels in the pupil used for the Neural Network
    int pup_offset1_x; // Horizontal offset to the first set of pupils
    int pup_offset1_y; // Vertical offset to the first set of pupils
    int pup_offset2_x; // Horizontal offset to the second set of pupils
    int pup_offset2_y; // Horizontal offset to the second set of pupils
    int pixels_per_quadrant;

    int Npup{ 4 };        // Number of pupils

    eigenImage<float> modeval;

    half *modeval_half{ nullptr };

    float *pp_image{ nullptr };
    float *pup_Is{ nullptr };
    half *pp_image_half{ nullptr };
    half *pup_Is_half{ nullptr };

    size_t m_pwfsWidth{ 0 };  ///< The width of the image
    size_t m_pwfsHeight{ 0 }; ///< The height of the image.

    uint8_t m_pwfsDataType{ 0 }; ///< The ImageStreamIO type code.
    size_t m_pwfsTypeSize{ 0 };  ///< The size of the type, in bytes.

    float m_fps{ 0 }; ///< Current FPS from the FPS source.

    sem_t m_smSemaphore{ 0 }; ///< Semaphore used to synchronize the fg thread and the dm command thread.

    bool m_updated{ false }; ///< Flag indicating that the mode vals have been updated


  public:
    /// Default c'tor.
    nnReconstructor();

    /// D'tor, declared and defined for noexcept.
    ~nnReconstructor() noexcept
    {
    }

    virtual void setupConfig();

    /// Implementation of loadConfig logic, separated for testing.
    /** This is called by loadConfig().
     */
    int loadConfigImpl(
        mx::app::appConfigurator &_config /**< [in] an application configuration
        from which to load values*/
    );

    virtual void loadConfig();

    /// Startup function
    /**
     *
     */
    virtual int appStartup();

    /// Implementation of the FSM for nnReconstructor.
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

    // Custom functions
    //int send_to_shmim();

    void load_engine(const std::string filename);
    void create_engine_context();
    void prepare_engine_memory();
    void cleanup_engine_memory();
    void cleanup_engine_context();

    // void build_engine(){};

  protected:
    int allocate( const dev::shmimT &dummy /**< [in] tag to differentiate shmimMonitor parents.*/ );

    int processImage( void *curr_src,          ///< [in] pointer to start of current frame.
                      const dev::shmimT &dummy ///< [in] tag to differentiate shmimMonitor parents.
    );


    /** \name frameGrabber Interface
     * @{
     */

    /// Configure the output stream for acquistion.
    /** Tests if stream exists and is expected size.  Creates it if needed.
     *  will set m_width, m_height, and m_dataType.
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
    /** Here just waits on the semaphore.
     */
    int acquireAndCheckValid();

    /// Loads the modevals into the stream
    int loadImageIntoStream(void * dest);

    ///Take any actions needed to reconfigure the system.  Called if m_reconfig is set to true.
    int reconfig();

    ///@}

    /** \name INDI interface
     * @{
    */

    pcf::IndiProperty m_indiP_fpsSource;
    INDI_SETCALLBACK_DECL( nnReconstructor, m_indiP_fpsSource );

    pcf::IndiProperty m_indiP_fps;

    ///@}

    /** \name Telemeter Interface
     * @{
     */

    int checkRecordTimes();

    int recordTelem( const telem_fgtimings * );

    ///@}
};

void nnReconstructor::load_engine(const std::string filename) {

    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cout << "Error opening " << filename << std::endl;
    }

    file.seekg(0, std::ios::end);

    engineData = std::vector<char>(file.tellg());
    file.seekg(0, std::ios::beg);
    file.read(engineData.data(), engineData.size());

}

void nnReconstructor::create_engine_context(){

    // Create the runtime and deserialize the engine
    runtime = createInferRuntime(logger);
    if (!runtime) {
        std::cout << "Failed to createInferRuntime\n";
    }

    engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size());
    if (!engine) {
        std::cout << "Failed to deserialize CUDA engine.\n";
    } else {
        std::cout << "Deserialized CUDA engine.\n";
    }

    context = engine->createExecutionContext();


    int numIOTensors = engine->getNbIOTensors();
    std::cout << "Number of IO Tensors: " << numIOTensors << std::endl;


    auto inputName = engine->getIOTensorName(0);
    auto outputName = engine->getIOTensorName(1);
    std::cout << "Tensor IO names: " << inputName << ", " << outputName << std::endl;


    const auto inputDims = engine->getTensorShape(inputName);
    const auto outputDims = engine->getTensorShape(outputName);
    inputC = inputDims.d[1];
    inputH = inputDims.d[2];
    inputW = inputDims.d[3];
    inputSize = inputC * inputH * inputW;

    outputSize = outputDims.d[1];
    std::cout << "Tensor input dimensions: " << inputC << "x" << inputH << "x" << inputW << std::endl;
    std::cout << "Tensor output dimensions: " << outputSize << std::endl;

}

void nnReconstructor::prepare_engine_memory(){

    // Allocate device memory for input and output
    if (use_fp16) {
        // Allocate FP16 memory
        cudaMalloc((void**)&d_input, inputSize * sizeof(half));
        if (explicit_tt) {
            cudaMalloc((void**)&d_input2, input2Size * sizeof(half));
        }
        cudaMalloc((void**)&d_output, outputSize * sizeof(half));
    } else {
        // Allocate FP32 memory
        cudaMalloc((void**)&d_input, inputSize * sizeof(float));
        if (explicit_tt) {
            cudaMalloc((void**)&d_input2, input2Size * sizeof(float));
        }
        cudaMalloc((void**)&d_output, outputSize * sizeof(float));
    }

    //cudaMalloc((void**)&d_input, inputSize * sizeof(float));
    //cudaMalloc((void**)&d_output, outputSize * sizeof(float));

}

void nnReconstructor::cleanup_engine_memory(){
    if(d_input)
        cudaFree(d_input);
    if(d_input2)
        cudaFree(d_input2);

    if(d_output)
        cudaFree(d_output);
}

void nnReconstructor::cleanup_engine_context(){
    if(context)
        delete context;
    if(engine)
        delete engine;
    if(runtime)
        delete runtime;
};

inline nnReconstructor::nnReconstructor() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    return;
}

inline void nnReconstructor::setupConfig()
{
    std::cout << "setupConfig()" << std::endl;

    SHMIMMONITOR_SETUP_CONFIG( config );
    FRAMEGRABBER_SETUP_CONFIG( config );
    TELEMETER_SETUP_CONFIG(config);

    config.add( "parameters.dataDirs",
                "",
                "parameters.dataDirs",
                argType::Required,
                "parameters",
                "dataDirs",
                false,
                "string",
                "The path to the directory with the onnx model." );

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

    config.add( "parameters.use_fp16",
                "",
                "parameters.use_fp16",
                argType::Required,
                "parameters",
                "use_fp16",
                false,
                "bool",
                "If true the half precision mode will be used." );

    config.add( "parameters.explicit_tt",
                "",
                "parameters.explicit_tt",
                argType::Required,
                "parameters",
                "explicit_tt",
                false,
                "bool",
                "If true the model will additionally give the pupil intensities as input to the NN." );

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

}

inline int nnReconstructor::loadConfigImpl( mx::app::appConfigurator &_config )
{
    std::cout << "loadConfigImpl()" << std::endl;

    SHMIMMONITOR_LOAD_CONFIG(_config);

    frameGrabberT::m_ownShmim = false;
    FRAMEGRABBER_LOAD_CONFIG(_config);

    _config( dataDirs, "parameters.dataDirs" );
    _config( engineDirs, "parameters.engineDirs" );
    _config( engineName, "parameters.engineName" );
    _config( rebuildEngine, "parameters.rebuildEngine" );
    _config( imageNorm, "parameters.imageNorm" );
    _config( modalNorm, "parameters.modalNorm" );
    _config( use_fp16, "parameters.use_fp16" );
    _config( explicit_tt, "parameters.explicit_tt" );

    _config( m_pupPix, "parameters.m_pupPix" );
    _config( pup_offset1_x, "parameters.pup_offset1_x" );
    _config( pup_offset1_y, "parameters.pup_offset1_y" );
    _config( pup_offset2_x, "parameters.pup_offset2_x" );
    _config( pup_offset2_y, "parameters.pup_offset2_y" );

    _config( m_fpsSource, "recon.fpsSource" );

    TELEMETER_LOAD_CONFIG(_config);

    if( true )
    {
        std::cout << "Debug configuration loading: " << std::endl;
        std::cout << "dataDirs: " << dataDirs << std::endl;
        std::cout << "engineDirs: " << engineDirs << std::endl;
        std::cout << "engineName: " << engineName << std::endl;
        std::cout << "rebuildEngine: " << rebuildEngine << std::endl;
        std::cout << "imageNorm: " << imageNorm << std::endl;
        std::cout << "modalNorm: " << modalNorm << std::endl;
        std::cout << "use_fp16: " << use_fp16 << std::endl;
        std::cout << "explicit_tt: " << explicit_tt << std::endl;
        std::cout << "modeval Channel: " << frameGrabberT::m_shmimName << std::endl;

        std::cout << "m_pupPix: " << m_pupPix << std::endl;
        std::cout << "pup_offset1_x: " << pup_offset1_x << std::endl;
        std::cout << "pup_offset1_y: " << pup_offset1_y << std::endl;
        std::cout << "pup_offset2_x: " << pup_offset2_x << std::endl;
        std::cout << "pup_offset2_y: " << pup_offset2_y << std::endl;
    }

    return 0;
}

inline void nnReconstructor::loadConfig()
{
    loadConfigImpl( config );
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

    if( sem_init( &m_smSemaphore, 0, 0 ) < 0 )
    {
        log<software_critical>( { errno, "Initializing S.M. semaphore" } );
        return -1;
    }

    std::string full_filepath = engineDirs + "/" + engineName;
    std::cout << "file: " << full_filepath << std::endl;

    load_engine(full_filepath);
    create_engine_context();
    prepare_engine_memory();

    //Do this after everything is configured so we can check sizes propertly
    FRAMEGRABBER_APP_STARTUP;

    SHMIMMONITOR_APP_STARTUP;

    TELEMETER_APP_STARTUP;

    state( stateCodes::OPERATING );
    return 0;
}

inline int nnReconstructor::appLogic()
{
    SHMIMMONITOR_APP_LOGIC;

    FRAMEGRABBER_APP_LOGIC;

    TELEMETER_APP_LOGIC;

    std::unique_lock<std::mutex> lock( m_indiMutex );

    SHMIMMONITOR_UPDATE_INDI;

    FRAMEGRABBER_UPDATE_INDI;


    return 0;
}

inline int nnReconstructor::appShutdown()
{
    SHMIMMONITOR_APP_SHUTDOWN;

    FRAMEGRABBER_APP_SHUTDOWN;

    TELEMETER_APP_SHUTDOWN;

    if( pp_image )
    {
        delete[] pp_image;
    }
    if( pp_image_half )
    {
        delete[] pp_image_half;
    }
    if( pup_Is )
    {
        delete[] pup_Is;
    }
    if( pup_Is_half)
    {
        delete[] pup_Is_half;
    }

    if( modeval_half )
    {
        delete[] modeval_half;
    }

    cleanup_engine_context();
    cleanup_engine_memory();

    return 0;
}

inline int nnReconstructor::allocate( const dev::shmimT &dummy )
{
    std::cout << "allocate()" << std::endl;
    static_cast<void>( dummy ); // be unused

    // Wavefront sensor setup
    m_pwfsWidth = shmimMonitorT::m_width;
    m_pwfsHeight = shmimMonitorT::m_height;
    std::cout << "Width: " << m_pwfsWidth << " Height: " << m_pwfsHeight << std::endl;

    pixels_per_quadrant = m_pupPix * m_pupPix;
    std::cout << "Pixels: " << pixels_per_quadrant << std::endl;
    pp_image = new float[Npup * pixels_per_quadrant];
    pup_Is = new float[4];
    //modeval = new float[outputSize];
    modeval.resize(outputSize, 1);

    memset( pp_image, 0, sizeof( float ) * Npup * pixels_per_quadrant );
        if (explicit_tt){
            memset( pup_Is, 0, sizeof( float) * 4);
        }
    //memset( modeval, 0, sizeof( float) * outputSize);
    modeval.setZero();

    if (use_fp16){
        pp_image_half = new half[Npup * pixels_per_quadrant];
        modeval_half = new half[outputSize];
        pup_Is_half = new half[4];
        memset( pp_image_half, 0, sizeof( half ) * Npup * pixels_per_quadrant );
        if (explicit_tt){
            memset( pup_Is_half, 0, sizeof( half ) * 4);
        }
        memset( modeval_half, 0, sizeof( half) * outputSize);
    }

    return 0;
}

inline int nnReconstructor::processImage( void *curr_src, const dev::shmimT &dummy )
{
    static_cast<void>( dummy ); // be unused

    //record arrival time as the atime
    if( clock_gettime( CLOCK_REALTIME, &(frameGrabberT::m_currImageTimestamp) ) < 0 )
    {
        m_shutdown = true;
        return log<software_critical,-1>( { errno, "clock_gettime" } );
    }

    // aol_imwfs2 is reference and dark subtracted and is power normalized.

    Eigen::Map<eigenImage<float>> pwfsIm(reinterpret_cast<float*>(curr_src), m_pwfsHeight, m_pwfsWidth);

    // Split up the four pupils for the Neural Network.
    int ki = 0;


    for( int col_i = -zeroPad; col_i < (m_pupPix - zeroPad); ++col_i )
    {
        for( int row_i = -zeroPad; row_i < (m_pupPix - zeroPad); ++row_i )
        {
            if ((col_i < 0) or (col_i >= (m_pupPix - 2*zeroPad)) or (row_i < 0) or (row_i >= (m_pupPix - 2*zeroPad))){
                pp_image[ki] = 0;
                pp_image[ki + pixels_per_quadrant] = 0;
                pp_image[ki + 2 * pixels_per_quadrant] = 0;
                pp_image[ki + 3 * pixels_per_quadrant] = 0;
            }
            else {
                pp_image[ki] = imageNorm * (realT)pwfsIm(pup_offset1_y + row_i, pup_offset1_x + col_i );
                pp_image[ki + pixels_per_quadrant] = imageNorm * (realT)pwfsIm( pup_offset1_y + row_i, pup_offset2_x + col_i );
                pp_image[ki + 2 * pixels_per_quadrant] = imageNorm * (realT)pwfsIm( pup_offset2_y + row_i, pup_offset1_x + col_i );
                pp_image[ki + 3 * pixels_per_quadrant] = imageNorm * (realT)pwfsIm( pup_offset2_y + row_i, pup_offset2_x + col_i );
            }
            ++ki;
        }
    }

    // Copy input data to device
    if (use_fp16){
        floatToHalfArray(pp_image_half, pp_image, inputSize);
        cudaMemcpy(d_input, pp_image_half, inputSize * sizeof(half), cudaMemcpyHostToDevice);
        if (explicit_tt){
            floatToHalfArray(pup_Is_half, pup_Is, input2Size);
            cudaMemcpy(d_input2, pup_Is_half, input2Size * sizeof(half), cudaMemcpyHostToDevice);
        }
    }
    else {
        cudaMemcpy(d_input, pp_image, inputSize * sizeof(float), cudaMemcpyHostToDevice);
        if (explicit_tt){
            cudaMemcpy(d_input2, pup_Is, input2Size * sizeof(float), cudaMemcpyHostToDevice);

        }
    }


    // Run inference
    if (explicit_tt){
        void* buffers[] = {d_input, d_input2, d_output};
        context->executeV2(buffers);
    }
    else {
        void* buffers[] = {d_input, d_output};
        context->executeV2(buffers);
    }

    // Copy output data back to host
    if (use_fp16){
        cudaMemcpy(modeval_half, d_output, outputSize * sizeof(half), cudaMemcpyDeviceToHost);
        halfToFloatArray(modeval.data(), modeval_half, outputSize);
    }
    else {
        cudaMemcpy(modeval.data(), d_output, outputSize * sizeof(float), cudaMemcpyDeviceToHost);
    }

    // Send modal coefficients to the correct stream
    m_updated = true;

    // trigger framegrabber
    if( sem_post( &m_smSemaphore ) < 0 )
    {
        log<software_critical>( { errno, 0, "Error posting to semaphore" } );
        return -1;
    }

    return 0;
}

int nnReconstructor::configureAcquisition()
{
    static bool logged = false;

    int rv = openShmim();
    if(rv != 0)
    {
        return rv;
    }

    if( frameGrabberT::m_width != outputSize || frameGrabberT::m_height != 1 ||
        frameGrabberT::m_dataType != _DATATYPE_FLOAT )
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


    logged = false;

    return 0;
}

float nnReconstructor::fps()
{
    return m_fps;
}

int nnReconstructor::startAcquisition()
{
    return 0;
}

int nnReconstructor::acquireAndCheckValid()
{
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

int nnReconstructor::loadImageIntoStream(void * dest)
{
    memcpy( dest, modeval.data(), modeval.rows() * frameGrabberT::m_typeSize );

    m_updated = false;

    return 0;
}

int nnReconstructor::reconfig()
{
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
        updateIfChanged( m_indiP_fps, "current", m_fps );
    }

    return 0;
}

int nnReconstructor::checkRecordTimes()
{
    return telemeterT::checkRecordTimes( telem_fgtimings() );
}

int nnReconstructor::recordTelem( const telem_fgtimings * )
{
    return recordFGTimings( true );
}

} // namespace app
} // namespace MagAOX

#endif // nnReconstructor_hpp