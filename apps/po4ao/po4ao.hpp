/** \file po4ao.hpp
 * \brief The MagAO-X generic ImageStreamIO stream integrator
 *
 * \ingroup app_files
 */

#ifndef po4ao_hpp
#define po4ao_hpp

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

using namespace nvinfer1;

// Logger for TensorRT info/warning/errors
class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << msg << std::endl;
        }
    }
};


class CircularBuffer {
    private:
        size_t num_rows, num_cols;     // Dimensions of each image (rows x columns)
        size_t historySize;            // Total number of images the buffer can hold
        size_t size;                   // Total number of elements in each image (rows * columns)
        float* buffer;                 // Buffer holding the images
        size_t head;                   // Index for the next write position
        size_t tail;                   // Index for the next read position
        bool is_full;                  // Flag to check if buffer is full

    public:
        CircularBuffer(size_t _num_rows, size_t _num_cols, size_t _historySize)
            : num_rows(_num_rows), num_cols(_num_cols), 
              historySize(_historySize), 
              size(_num_rows * _num_cols), 
              head(0), tail(0), is_full(false) {
            
            buffer = new float[size * historySize];
        }

        ~CircularBuffer(){
            delete buffer;
        }
    
        // Adds an image (array of floats) to the buffer (overwrites if full)
        void add(float* image) {
            //std::cout << "Adding image to buffer" << std::endl;
            memcpy(&buffer[(head % historySize) * size], image, sizeof(float) * size);
            //memcpy(buffer + (head % historySize) * size * sizeof(float), image, sizeof(float) * size);
            head = head + 1;
        }
 
       

        void add_eigenimage(Eigen::Map<eigenImage<float>> image) {
            memcpy(buffer + (head % historySize) * size, &image, sizeof(float) * size);
            //std::cout << "Buffer " << buffer[head] << std::endl;
            head = head + 1;
        }
       
        //void get(float* dest, int index){
        //    if(index < num_elements()){
        //        int data_index = (head - 1 - index) % historySize;
        //        memcpy(dest, buffer[data_index * size], sizeof(float) * size);
        //    }
        //}


        int num_elements(){
            if( head > historySize){
                return historySize;
            }else{
                return head;
            }
        }
        /*
        void reset_head(){
            head = 0;
        }
        */

        float* getBuffer() {
            return buffer;
        }
    

};
/*
class CircularBuffer2 {
    public:
        CircularBuffer2(size_t Rows, size_t Cols, size_t historySize)
            : imageRows(Rows), imageCols(Cols), 
              historySize(historySize), 
              size(Rows * Cols), 
              head(0), tail(0), is_full(false) {
            buffer.resize(historySize, std::vector<float>(size));
        }
    
        // Adds an image (array of floats) to the buffer (overwrites if full)
        void add(float* image) {
            std::cout << "Adding image" << head % historySize << std::endl;
            //memcpy(buffer[(head % historySize)], image, sizeof(float) * size)
            memcpy(buffer[(head % historySize)].data(), image, sizeof(float) * size);

            //for (size_t i = 0; i < imageRows*imageCols; ++i) {
            std::cout << "Done adding image" << std::endl;
            //    buffer[head][i] = image[i];
            //}
            head = (head + 1) % historySize;
        }

        void add_eigenimage(Eigen::Map<eigenImage<unsigned short>> image) {
            for (size_t i = 0; i < image.size(); ++i) {
                buffer[head][i] = image(i); // Cast to float if needed
            }
            head = (head + 1) % historySize;
        }

        void reset_head(){
            head = 0;
        }
    
    
        // Returns the n-th image in the buffer in the order of addition
        std::vector<float> getItem(size_t n) const {
            if (n >= head) {
                std::cout << "Index: " << n << std::endl;
                std::cout << "Index2: " << head - 1 << std::endl;
                throw std::runtime_error("Index out of bounds!");
                return {}; // Return empty vector as error
            }
    
            // Calculate the actual index in the buffer
            //size_t index = (tail + n) % historySize;
            size_t index = head  - n - 1;
            return buffer[index];
        }

        const std::vector<float>* getBuffer() const {
            return buffer.data();
        }
    
    private:
        size_t imageRows, imageCols;   // Dimensions of each image (rows x columns)
        size_t historySize;            // Total number of images the buffer can hold
        size_t size;                   // Total number of elements in each image (rows * columns)
        std::vector<std::vector<float>> buffer; // Buffer holding the images
        size_t head;                   // Index for the next write position
        size_t tail;                   // Index for the next read position
        bool is_full;                  // Flag to check if buffer is full
};

*/

// #define MAGAOX_CURRENT_SHA1 0
// #define MAGAOX_REPO_MODIFIED 0

namespace MagAOX
{
namespace app
{

class po4ao : public MagAOXApp<true>, public dev::shmimMonitor<po4ao>
{
    // Give the test harness access.
    friend class po4ao_test;

    friend class dev::shmimMonitor<po4ao>;

    // The base shmimMonitor type
    typedef dev::shmimMonitor<po4ao> shmimMonitorT;

    /// Floating point type in which to do all calculations.engine
    typedef float realT;

  public:
    /** \name app::dev Configurations
     *@{reconstructed_buffer
     */

    ///@}

  protected:
    /** \name Configurable Parameters
     *@{
     */
    std::string dataDirs;     // Location where the data (onnx file, engine, WFS reference) is stored
    std::string engineName;   // Name of the engine
    std::string engineDirs;   // Name of the engine
    
    Logger logger;			  // The tensorRT logger
    std::vector<char> engineData; // for loading the engine file.

    IRuntime* runtime {nullptr};
    ICudaEngine* engine {nullptr};
    ICudaEngine* engine2 {nullptr};
    IExecutionContext* context {nullptr};
    IExecutionContext* context2 {nullptr};
    int inputC {0};
    int inputH {0};
    int inputW {0};
    int inputSize {0};
    int outputSize {0};
    int Nact {0};
    int Nact_across {0};
    int Nfeatures {0};
    float max_sigma {0};

    bool reloadEngine {false};
    bool engineReloaded {false};

    CircularBuffer* command_buffer {nullptr};
    CircularBuffer* reconstructed_buffer {nullptr};

    float* d_input {nullptr};
    float* d_output {nullptr};
    float* integrator_commands {nullptr};

    unsigned long frame_counter{ 0 };
    unsigned long episode_counter{ 0 };

    uint32_t Nmodes{ 0 }; // Number of modes to reconstruct
    float *m_output{ nullptr };
    float *inputState{ nullptr };

    // variables for sending the output to aol_outputs
    std::string m_outputChannel;
    IMAGE m_outputStream;
    uint32_t m_outputWidth {0}; ///< The width of the shmim
    uint32_t m_outputHeight {0}; ///< The height of the shmim
    uint8_t m_outputDataType {0}; ///< The ImageStreamIO type code.
    size_t m_outputTypeSize {0};  ///< The size of the type, in bytes.

    // variables for sending the output to po4ao_observations
    std::string po4ao_obs_Channel;
    IMAGE po4ao_obs_Stream;
    uint32_t po4ao_obs_Width {0}; ///< The width of the shmim
    uint32_t po4ao_obs_Height {0}; ///< The height of the shmim
    uint8_t po4ao_obs_DataType {0}; ///< The ImageStreamIO type code.
    size_t po4ao_obs_TypeSize {0};  ///< The size of the type, in bytes.

    int Nhist {0};
    int replay_buffer_size {0};
    uint32_t iterations_per_ep {0};
    uint32_t warmup_episodes {0};
    float integrator_gain {0};

    std::string po4ao_act_Channel;


    bool m_outputOpened {false};
    bool m_outputRestart {false};

    bool po4ao_obs_Opened {false};
    bool po4ao_obs_Restart {false};

    pcf::IndiProperty m_indiP_reloadToggle;
    INDI_NEWCALLBACK_DECL(po4ao, m_indiP_reloadToggle);

  public:
    /// Default c'tor.
    po4ao();

    /// D'tor, declared and defined for noexcept.
    ~po4ao() noexcept
    {
    }

    virtual void setupConfig();

    /// Implementation of loadConfig logic, separated for testing.
    /** This is called by loadConfig().
     */
    int loadConfigImpl(
        mx::app::appConfigurator &_config /**< [in] an application configuration from which to load values*/ );

    virtual void loadConfig();

    /// Startup function
    /**
     *
     */
    virtual int appStartup();

    /// Implementation of the FSM for po4ao.
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
    int send_to_shmim();
    int send_obs_to_shmim();
    
    void load_engine(const std::string filename);
    void create_engine_context();
    void prepare_engine_memory();
    void cleanup_engine_memory();
    void cleanup_engine_context();
    void reload_engine();
    void switch_engine();

    // void build_engine(){};

  protected:
    int allocate( const dev::shmimT &dummy /**< [in] tag to differentiate shmimMonitor parents.*/ );

    int processImage( void *curr_src,          ///< [in] pointer to start of current frame.
                      const dev::shmimT &dummy ///< [in] tag to differentiate shmimMonitor parents.
    );
};

void po4ao::load_engine(const std::string filename) {
    
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cout << "Error opening " << filename << std::endl;
    }

    file.seekg(0, std::ios::end);
    
    engineData = std::vector<char>(file.tellg());
    file.seekg(0, std::ios::beg);
    file.read(engineData.data(), engineData.size());

}

void po4ao::create_engine_context(){

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
    std::cout << "Tensor IO names: " << inputName << " " << outputName << std::endl;

    const auto inputDims = engine->getTensorShape(inputName);
    const auto outputDims = engine->getTensorShape(outputName);
    inputC = inputDims.d[1];
    inputH = inputDims.d[2];
    inputW = inputDims.d[3];
    inputSize = inputC * inputH * inputW;

    outputSize = outputDims.d[1];
    std::cout << "Tensor input dimensions: " << inputC << "x" << inputH << "x" << inputW << std::endl;
    std::cout << "Tensor output dimensions: " << outputSize << std::endl;

    Nact = outputSize;
    Nact_across = inputH;
    Nfeatures = inputC;

}

void po4ao::reload_engine(){
    std::string full_filepath = engineDirs + "/" + engineName;
    load_engine(full_filepath);

    engine2 = runtime->deserializeCudaEngine(engineData.data(), engineData.size());
    if (!engine2) {
        std::cout << "Failed to deserialize CUDA engine.\n";
    } else {
        std::cout << "Deserialized CUDA engine.\n";
    }
    context2 = engine2->createExecutionContext();
    engineReloaded = true;
}

void po4ao::switch_engine(){
    std::cout << "Start switching engine" << std::endl;
    auto temp_engine = engine;
    auto temp_context = context;
    engine = engine2;
    context = context2;
    engine2 = temp_engine;
    context2 = temp_context;
    engineReloaded = false;
    std::cout << "Done switching engine" << std::endl;
}

void po4ao::prepare_engine_memory(){

    // Allocate device memory for input and output
    cudaMalloc((void**)&d_input, inputSize * sizeof(float));
    cudaMalloc((void**)&d_output, outputSize * sizeof(float));

}

void po4ao::cleanup_engine_memory(){
    if(d_input)
        cudaFree(d_input);
    
    if(d_output)
        cudaFree(d_output);
}

void po4ao::cleanup_engine_context(){
    if(context)
        delete context;
    if(engine)
        delete engine;
    if(runtime)
        delete runtime;
    if(context2)
        delete context2;
    if(engine2)
        delete engine2;
};

inline int po4ao::send_to_shmim()
{
    // Check if processImage is running
    // while(m_dmStream.md[0].write == 1);

    m_outputStream.md[0].write = 1;
    memcpy( m_outputStream.array.raw, m_output, outputSize * m_outputTypeSize );
    m_outputStream.md[0].cnt0++;
    m_outputStream.md[0].write = 0;

    ImageStreamIO_sempost( &m_outputStream, -1 );

    return 0;
}

inline int po4ao::send_obs_to_shmim()
{
    // Check if processImage is running
    // while(m_dmStream.md[0].write == 1);

    po4ao_obs_Stream.md[0].write = 1;
    //reconstructed_buffer->get(po4ao_obs_Stream.array.raw, 0);
    memcpy( po4ao_obs_Stream.array.raw, reconstructed_buffer->getBuffer(), outputSize * po4ao_obs_TypeSize );
    po4ao_obs_Stream.md[0].cnt0++;
    po4ao_obs_Stream.md[0].write = 0;

    ImageStreamIO_sempost( &po4ao_obs_Stream, -1 );

    return 0;
}

inline po4ao::po4ao() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    return;
}

inline void po4ao::setupConfig()
{
    std::cout << "setupConfig()" << std::endl;
    shmimMonitorT::setupConfig( config );

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

    config.add( "parameters.channel",
                "",
                "parameters.channel",
                argType::Required,
                "parameters",
                "channel",
                false,
                "string",
                "The output channel." );

    config.add( "parameters.observation_channel",
                "",
                "parameters.observation_channel",
                argType::Required,
                "parameters",
                "observation_channel",
                false,
                "string",
                "The output channel for the observations." );

    config.add( "parameters.action_channel",
                "",
                "parameters.action_channel",
                argType::Required,
                "parameters",
                "action_channel",
                false,
                "string",
                "The output channel for the actions." );

    config.add( "parameters.iterations_per_ep",
                "",
                "parameters.iterations_per_ep",
                argType::Required,
                "parameters",
                "iterations_per_ep",
                false,
                "int",
                "Number of iterations per episode." );

    config.add( "parameters.integrator_gain",
                "",
                "parameters.integrator_gain",
                argType::Required,
                "parameters",
                "integrator_gain",
                false,
                "float",
                "Gain of the integrator during warmup." );

    config.add( "parameters.warmup_episodes",
                "",
                "parameters.warmup_episodes",
                argType::Required,
                "parameters",
                "warmup_episodes",
                false,
                "int",
                "Number of episodes for the warmup." );

    config.add( "parameters.replay_buffer_size",
                "",
                "parameters.replay_buffer_size",
                argType::Required,
                "parameters",
                "replay_buffer_size",
                false,
                "int",
                "Number of samples in the real-time buffer." );

    config.add( "parameters.max_sigma",
                "",
                "parameters.max_sigma",
                argType::Required,
                "parameters",
                "max_sigma",
                false,
                "float",
                "Standard deviation of the exploration noise." );

    config.add( "parameters.Nhist",
                "",
                "parameters.Nhist",
                argType::Required,
                "parameters",
                "Nhist",
                false,
                "int",
                "Number of frames in the history vector." );

    config.add( "parameters.modal_filt_matrix",
                "",
                "parameters.modal_filt_matrix",
                argType::Required,
                "parameters",
                "modal_filt_matrix",
                false,
                "string",
                "Name of the model filtering matrix file." );
                
    config.add( "parameters.reloadEngine",
                "",
                "parameters.reloadEngine",
                argType::Required,
                "parameters",
                "reloadEngine",
                false,
                "bool",
                "Whether to reload the engine." );

}

inline int po4ao::loadConfigImpl( mx::app::appConfigurator &_config )
{
    std::cout << "loadConfigImpl()" << std::endl;
    shmimMonitorT::loadConfig( config );

    _config( dataDirs, "parameters.dataDirs" );
    _config( engineDirs, "parameters.engineDirs" );
    _config( engineName, "parameters.engineName" );
    _config( m_outputChannel, "parameters.channel");
    _config( po4ao_obs_Channel, "parameters.observation_channel");
    _config( po4ao_act_Channel, "parameters.observation_channel");

    _config( Nhist, "parameters.Nhist" );
    _config( iterations_per_ep, "parameters.iterations_per_ep" );
    _config( warmup_episodes, "parameters.warmup_episodes" );
    _config( replay_buffer_size, "parameters.replay_buffer_size" );
    _config( max_sigma, "parameters.max_sigma" );
    _config( integrator_gain, "parameters.integrator_gain" );
    _config( reloadEngine, "parameters.reloadEngine" );

    if( true )
    {
        std::cout << "Debug configuration loading: " << std::endl;
        std::cout << "dataDirs: " << dataDirs << std::endl;
        std::cout << "engineDirs: " << engineDirs << std::endl;
        std::cout << "engineName: " << engineName << std::endl;
        std::cout << "output Channel: " << m_outputChannel << std::endl;
    }

    return 0;
}

inline void po4ao::loadConfig()
{
    loadConfigImpl( config );
}

inline int po4ao::appStartup()
{
    if( shmimMonitorT::appStartup() < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__ } );
    }
    createStandardIndiToggleSw( m_indiP_reloadToggle, "reload_engine");
	registerIndiPropertyNew( m_indiP_reloadToggle, INDI_NEWCALLBACK(m_indiP_reloadToggle) ); 
  
    std::string full_filepath = engineDirs + "/" + engineName;
    std::cout << "file: " << full_filepath << std::endl;

    load_engine(full_filepath);
    create_engine_context();
    prepare_engine_memory();

    // state(stateCodes::READY);
    state( stateCodes::OPERATING );
    return 0;
}

inline int po4ao::appLogic()
{
    if( shmimMonitorT::appLogic() < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__ } );
    }

    std::unique_lock<std::mutex> lock( m_indiMutex );

    if( shmimMonitorT::updateINDI() < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
    }

    if(reloadEngine){
		updateSwitchIfChanged(m_indiP_reloadToggle, "toggle", pcf::IndiElement::On, INDI_OK);
	}else{
		updateSwitchIfChanged(m_indiP_reloadToggle, "toggle", pcf::IndiElement::Off, INDI_IDLE);
	}

    return 0;
}

inline int po4ao::appShutdown()
{
    shmimMonitorT::appShutdown();

    if( inputState )
    {
        delete inputState;
    }
    if( integrator_commands )
    {
        delete integrator_commands;
    }
    if ( command_buffer )
    {
        delete command_buffer;
    }
    if ( reconstructed_buffer )
    {
        delete reconstructed_buffer;
    }

    cleanup_engine_context();
    cleanup_engine_memory();

    return 0;
}

inline int po4ao::allocate( const dev::shmimT &dummy )
{
    std::cout << "allocate()" << std::endl;
    static_cast<void>( dummy ); // be unused

    inputState = new float[Nfeatures * Nact];
    memset( inputState, 0, sizeof( float ) * Nfeatures * Nact );

    integrator_commands = new float[Nact];
    memset( integrator_commands, 0, sizeof( float ) * Nact );

    m_output = new float[Nact];
    memset( m_output, 0, sizeof( float ) * Nact );

    command_buffer = new CircularBuffer( 1, Nact, replay_buffer_size);
    reconstructed_buffer = new CircularBuffer( 1, Nact, replay_buffer_size);

    std::cout << "Close shmims" << std::endl;
    // Allocate the DM shmim interface
    if(m_outputOpened){
        ImageStreamIO_closeIm(&m_outputStream);
    }

    std::cout << "Open shmims" << std::endl;
    m_outputOpened = false;
    m_outputRestart = false; //Set this up front, since we're about to restart.

    if( ImageStreamIO_openIm(&m_outputStream, m_outputChannel.c_str()) == 0){
        if(m_outputStream.md[0].sem < 10){
            ImageStreamIO_closeIm(&m_outputStream);
        }else{
            m_outputOpened = true;
        }
    }

    std::cout << "Done!" << std::endl;
    if(!m_outputOpened){
        log<text_log>( m_outputChannel + " not opened.", logPrio::LOG_NOTICE);
        return -1;
    }else{
        m_outputWidth = m_outputStream.md->size[0];
        m_outputHeight = m_outputStream.md->size[1];

        m_outputDataType = m_outputStream.md->datatype;
        m_outputTypeSize = sizeof(float);

        log<text_log>( "Opened " + m_outputChannel + " " + std::to_string(m_outputWidth) + " x " + std::to_string(m_outputHeight) + " with data type: " + std::to_string(m_outputDataType), logPrio::LOG_NOTICE);
    }

    // Open PO4AO observation stream
    if(po4ao_obs_Opened){
        ImageStreamIO_closeIm(&po4ao_obs_Stream);
    }

    po4ao_obs_Opened = false;
    po4ao_obs_Restart = false; //Set this up front, since we're about to restart.

    if( ImageStreamIO_openIm(&po4ao_obs_Stream, po4ao_obs_Channel.c_str()) == 0){
        if(po4ao_obs_Stream.md[0].sem < 10){
            ImageStreamIO_closeIm(&po4ao_obs_Stream);
        }else{
            po4ao_obs_Opened = true;
        }
    }

    if(!po4ao_obs_Opened){
        log<text_log>( po4ao_obs_Channel + " not opened.", logPrio::LOG_NOTICE);
        return -1;
    }else{
        po4ao_obs_Width = po4ao_obs_Stream.md->size[0];
        po4ao_obs_Height = po4ao_obs_Stream.md->size[1];

        po4ao_obs_DataType = po4ao_obs_Stream.md->datatype;
        po4ao_obs_TypeSize = sizeof(float);

        log<text_log>( "Opened " + po4ao_obs_Channel + " " + std::to_string(po4ao_obs_Width) + " x " + std::to_string(po4ao_obs_Height) + " with data type: " + std::to_string(po4ao_obs_DataType), logPrio::LOG_NOTICE);
    }


    return 0;
}

inline int po4ao::processImage( void *curr_src, const dev::shmimT &dummy )
{
    //std::cout << "processImage()" << std::endl;
    static_cast<void>( dummy ); // be unused
    Eigen::Map<eigenImage<float>> ReconstructedMap(static_cast<float *>( curr_src ), 1, Nact );
    //if(false){
    reconstructed_buffer->add_eigenimage(ReconstructedMap);
    //reconstructed_buffer->getItem(1);
    // Assign values to InputState in correct order
    if (episode_counter > warmup_episodes){
        for( int feat_i =0; feat_i < Nfeatures; ++feat_i)
            {
            int ki = 0;
            for( int col_i = 0; col_i < Nact_across; ++col_i )
            {
                for( int row_i = 0; row_i < Nact_across; ++row_i )
                {
                    inputState[ki + feat_i * Nact] = 0;
                    /*
                        //std::cout << "HELLO:" << reconstructed_buffer->getItem(Nhist - feat_i) << std::endl;
                        //std::cout << "Buffer index: " << Nhist - feat_i << std::endl;
                        //std::cout << "Image index: " << col_i * Nact_across + row_i << std::endl;
                        if (frame_counter > Nhist + 1) {
                            std::cout << Nhist - feat_i << std::endl;
                            std::cout << "Z" << std::endl;
                            if (feat_i < Nhist){
                                std::cout << "A" << std::endl;
                                auto item = reconstructed_buffer->getItem(Nhist - feat_i);
                                std::cout << "B" << std::endl;
                                //This is where the error now is!!!
                                inputState[ki + feat_i * Nact] = item[ki];
                                //inputState[ki + feat_i * Nact] = reconstructed_buffer->getItem(feat_i)[row_i, col_i];
                            }
                            else{
                                std::cout << "C" << std::endl;
                                inputState[ki + feat_i * Nact] = command_buffer->getItem(Nhist - (feat_i - Nhist))[ki];
                            }
                        }nsigned short>> ReconstructedMap(static_cas
                        else {
                            //std::cout << "D" << std::endl;
                            inputState[ki + feat_i * Nact] = 0;
                        }
                    */
                    ++ki;
                    }
            }

        }

        // Copy input data to device
        cudaMemcpy(d_input, inputState, inputSize * sizeof(float), cudaMemcpyHostToDevice);

        // Run inference
        void* buffers[] = {d_input, d_output};
        context->executeV2(buffers);


        cudaMemcpy(m_output, d_output, outputSize * sizeof(float), cudaMemcpyDeviceToHost);
        
    }
    else {
        // Run with integrator
        for( int act_i = 0; act_i < Nact; ++act_i ){
            integrator_commands[act_i] = integrator_gain * ReconstructedMap(0, act_i);
        }

        memcpy(m_output, integrator_commands, outputSize * sizeof(float));
    }

   
    //reconstructed_buffer->add(m_output);
    if(frame_counter % iterations_per_ep == 0){
        std::cout << "Done with episode " << episode_counter << std::endl;
        episode_counter = episode_counter + 1;
        send_obs_to_shmim();
    }

    if(engineReloaded){
        switch_engine();  
    }
    // Send control commands to the correct stream
    //send_to_shmim();


    frame_counter++;
    return 0;
}

INDI_NEWCALLBACK_DEFN(po4ao, m_indiP_reloadToggle )(const pcf::IndiProperty &ipRecv)
{
   if(ipRecv.getName() != m_indiP_reloadToggle.getName())
   {
      log<software_error>({__FILE__, __LINE__, "invalid indi property received"});
      return -1;
   }
   
   //switch is toggled to on
   if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On)
   {
        std::cout << "Reloading model" << std::endl;
	    log<text_log>("Reloading model", logPrio::LOG_NOTICE);
        reload_engine();
        engineReloaded = true;
	    updateSwitchIfChanged(m_indiP_reloadToggle, "toggle", pcf::IndiElement::Off, INDI_BUSY);
        
        return 0;
   }

   //switch is toggle to off
   if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::Off)
   {
      std::cout << "No new model" << std::endl;
      return 0;
   }
   
   return 0;
}



} // namespace app
} // namespace MagAOX

#endif // po4ao_hpp
