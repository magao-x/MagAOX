/** \file po4ao_impl.hpp
 * \brief Header-only implementation of the PO4AO real-time application.
 *
 * This file is included by po4ao.hpp inside MagAOX::app.
 * \ingroup po4ao_files
 */

#ifndef po4ao_impl_hpp
#define po4ao_impl_hpp

inline po4ao::po4ao() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
}

inline po4ao::~po4ao() noexcept
{
}

inline void po4ao::setupConfig()
{
    SHMIMMONITOR_SETUP_CONFIG( config );

    config.add( "parameters.dataDirs",
                "",
                "parameters.dataDirs",
                argType::Required,
                "parameters",
                "dataDirs",
                false,
                "string",
                "Directory containing PO4AO training artifacts." );
    config.add( "parameters.engineDirs",
                "",
                "parameters.engineDirs",
                argType::Required,
                "parameters",
                "engineDirs",
                false,
                "string",
                "Directory containing the TensorRT engine." );
    config.add( "parameters.engineName",
                "",
                "parameters.engineName",
                argType::Required,
                "parameters",
                "engineName",
                false,
                "string",
                "TensorRT engine filename." );
    config.add( "parameters.linear_inference_backend",
                "",
                "parameters.linear_inference_backend",
                argType::Required,
                "parameters",
                "linear_inference_backend",
                false,
                "string",
                "Linear-policy inference backend: auto, cpu, or tensorrt." );
    config.add( "parameters.channel",
                "",
                "parameters.channel",
                argType::Required,
                "parameters",
                "channel",
                false,
                "string",
                "Differential modal command stream consumed by mfilt." );
    config.add( "parameters.buffer_channel",
                "",
                "parameters.buffer_channel",
                argType::Required,
                "parameters",
                "buffer_channel",
                false,
                "string",
                "Rolling replay-buffer stream." );
    config.add( "parameters.buffer_metadata_channel",
                "",
                "parameters.buffer_metadata_channel",
                argType::Required,
                "parameters",
                "buffer_metadata_channel",
                false,
                "string",
                "Rolling replay publication metadata stream." );
    config.add( "parameters.warmup_buffer_channel",
                "",
                "parameters.warmup_buffer_channel",
                argType::Required,
                "parameters",
                "warmup_buffer_channel",
                false,
                "string",
                "Frozen warm-up replay-buffer stream." );
    config.add( "parameters.warmup_metadata_channel",
                "",
                "parameters.warmup_metadata_channel",
                argType::Required,
                "parameters",
                "warmup_metadata_channel",
                false,
                "string",
                "Frozen warm-up publication metadata stream." );
    config.add( "parameters.iterations_per_ep",
                "",
                "parameters.iterations_per_ep",
                argType::Required,
                "parameters",
                "iterations_per_ep",
                false,
                "int",
                "Frames in one training episode." );
    config.add( "parameters.warmup_episodes",
                "",
                "parameters.warmup_episodes",
                argType::Required,
                "parameters",
                "warmup_episodes",
                false,
                "int",
                "Exploratory warm-up episode count." );
    config.add( "parameters.replay_buffer_size",
                "",
                "parameters.replay_buffer_size",
                argType::Required,
                "parameters",
                "replay_buffer_size",
                false,
                "int",
                "Samples retained in the rolling replay buffer." );
    config.add( "parameters.integrator_gain",
                "",
                "parameters.integrator_gain",
                argType::Required,
                "parameters",
                "integrator_gain",
                false,
                "float",
                "Proportional fallback command gain." );
    config.add( "parameters.policy_blend",
                "",
                "parameters.policy_blend",
                argType::Required,
                "parameters",
                "policy_blend",
                false,
                "float",
                "Learned-policy fraction; zero is fallback control and one is the full policy." );
    config.add( "parameters.exploration_sigma_start",
                "",
                "parameters.exploration_sigma_start",
                argType::Required,
                "parameters",
                "exploration_sigma_start",
                false,
                "float",
                "Exploration-noise scale at the first warm-up frame." );
    config.add( "parameters.exploration_sigma_end",
                "",
                "parameters.exploration_sigma_end",
                argType::Required,
                "parameters",
                "exploration_sigma_end",
                false,
                "float",
                "Exploration-noise scale at the final warm-up frame." );
    config.add( "parameters.Nmodes",
                "",
                "parameters.Nmodes",
                argType::Required,
                "parameters",
                "Nmodes",
                false,
                "int",
                "Number of modes in the input, output, and replay streams." );
    config.add( "parameters.Nmodes_controlled",
                "",
                "parameters.Nmodes_controlled",
                argType::Required,
                "parameters",
                "Nmodes_controlled",
                false,
                "int",
                "Leading modal count actively controlled by PO4AO." );
    config.add( "parameters.max_Nhist",
                "",
                "parameters.max_Nhist",
                argType::Required,
                "parameters",
                "max_Nhist",
                false,
                "int",
                "Maximum retained history; loaded policies may use any shorter history." );
    config.add( "parameters.exploreNoiseFile",
                "",
                "parameters.exploreNoiseFile",
                argType::Required,
                "parameters",
                "exploreNoiseFile",
                false,
                "string",
                "Exploration-noise CSV pathname." );
    config.add( "parameters.config_version",
                "",
                "parameters.config_version",
                argType::Required,
                "parameters",
                "config_version",
                false,
                "int",
                "Runtime-configuration schema version." );
    config.add( "parameters.control_enabled",
                "",
                "parameters.control_enabled",
                argType::Required,
                "parameters",
                "control_enabled",
                false,
                "bool",
                "Whether the learned policy is initially selected instead of the proportional fallback." );
    config.add( "parameters.max_command_abs",
                "",
                "parameters.max_command_abs",
                argType::Required,
                "parameters",
                "max_command_abs",
                false,
                "float",
                "Maximum absolute value of one modal command." );
    config.add( "parameters.cuda_device",
                "",
                "parameters.cuda_device",
                argType::Required,
                "parameters",
                "cuda_device",
                false,
                "int",
                "CUDA device index." );
    config.add( "parameters.latency_deadline_us",
                "",
                "parameters.latency_deadline_us",
                argType::Required,
                "parameters",
                "latency_deadline_us",
                false,
                "int",
                "Callback duration counted as a deadline miss." );
    config.add( "parameters.stats_period_seconds",
                "",
                "parameters.stats_period_seconds",
                argType::Required,
                "parameters",
                "stats_period_seconds",
                false,
                "float",
                "Runtime-statistics publication period." );
}

inline int po4ao::loadConfigImpl( mx::app::appConfigurator &_config )
{
    SHMIMMONITOR_LOAD_CONFIG( _config );
    _config( m_dataDirs, "parameters.dataDirs" );
    _config( m_engineDirs, "parameters.engineDirs" );
    _config( m_engineName, "parameters.engineName" );
    _config( m_linearInferenceBackend, "parameters.linear_inference_backend" );
    _config( m_outputChannel, "parameters.channel" );
    _config( m_bufferChannel, "parameters.buffer_channel" );
    _config( m_bufferMetadataChannel, "parameters.buffer_metadata_channel" );
    _config( m_warmupBufferChannel, "parameters.warmup_buffer_channel" );
    _config( m_warmupMetadataChannel, "parameters.warmup_metadata_channel" );
    _config( m_iterationsPerEpisode, "parameters.iterations_per_ep" );
    _config( m_warmupEpisodes, "parameters.warmup_episodes" );
    _config( m_replayBufferSize, "parameters.replay_buffer_size" );
    _config( m_integratorGain, "parameters.integrator_gain" );
    _config( m_policyBlend, "parameters.policy_blend" );
    _config( m_explorationSigmaStart, "parameters.exploration_sigma_start" );
    _config( m_explorationSigmaEnd, "parameters.exploration_sigma_end" );
    _config( m_nModes, "parameters.Nmodes" );
    _config( m_nModesControlled, "parameters.Nmodes_controlled" );
    _config( m_historyCapacity, "parameters.max_Nhist" );
    _config( m_exploreNoiseFile, "parameters.exploreNoiseFile" );
    _config( m_configVersion, "parameters.config_version" );
    _config( m_controlEnabledConfig, "parameters.control_enabled" );
    _config( m_maxCommandAbs, "parameters.max_command_abs" );
    _config( m_cudaDevice, "parameters.cuda_device" );
    _config( m_latencyDeadlineUs, "parameters.latency_deadline_us" );
    _config( m_statsPeriodSeconds, "parameters.stats_period_seconds" );
    return 0;
}

inline void po4ao::loadConfig()
{
    if( loadConfigImpl( config ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__, "PO4AO configuration load failed" } );
        m_shutdown = true;
    }
}

inline std::unique_ptr<EngineBundle> po4ao::createEngineBundle( const std::string &filename, std::string &error )
{
    std::ifstream file( filename, std::ios::binary );
    if( !file )
    {
        error = "could not open " + filename;
        return nullptr;
    }
    file.seekg( 0, std::ios::end );
    const std::streamoff fileSize = file.tellg();
    if( fileSize <= 0 )
    {
        error = "engine file is empty";
        return nullptr;
    }
    std::vector<char> bytes( static_cast<size_t>( fileSize ) );
    file.seekg( 0, std::ios::beg );
    file.read( bytes.data(), fileSize );
    if( !file )
    {
        error = "engine file read failed";
        return nullptr;
    }

    auto     bundle = std::make_unique<EngineBundle>();
    uint64_t hash   = 14695981039346656037ULL;
    for( unsigned char byte : bytes )
    {
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
    bundle->modelId = hash;
    if( m_linearInferenceBackend != "tensorrt" )
    {
        if( !loadLinearPolicy( filename + ".linear", *bundle, error ) )
            return nullptr;
        if( !bundle->linearWeights.empty() )
            return bundle;
        if( m_linearInferenceBackend == "cpu" )
        {
            error = "CPU linear inference requested but no matching sidecar is available";
            return nullptr;
        }
    }

    if( cudaSetDevice( m_cudaDevice ) != cudaSuccess )
    {
        error = "CUDA inference-device selection failed";
        return nullptr;
    }
    bundle->runtime = nvinfer1::createInferRuntime( m_logger );
    if( bundle->runtime == nullptr )
    {
        error = "TensorRT runtime creation failed";
        return nullptr;
    }
    bundle->engine = bundle->runtime->deserializeCudaEngine( bytes.data(), bytes.size() );
    if( bundle->engine == nullptr )
    {
        error = "TensorRT engine deserialization failed";
        return nullptr;
    }
    if( bundle->engine->getNbIOTensors() != 2 )
    {
        error = "engine must contain exactly one input and one output";
        return nullptr;
    }

    for( int index = 0; index < bundle->engine->getNbIOTensors(); ++index )
    {
        const char *name = bundle->engine->getIOTensorName( index );
        if( bundle->engine->getTensorDataType( name ) != nvinfer1::DataType::kFLOAT )
        {
            error = "engine tensors must use float32";
            return nullptr;
        }
        if( bundle->engine->getTensorIOMode( name ) == nvinfer1::TensorIOMode::kINPUT )
            bundle->inputName = name;
        else
            bundle->outputName = name;
    }
    if( bundle->inputName.empty() || bundle->outputName.empty() )
    {
        error = "engine input/output tensor discovery failed";
        return nullptr;
    }

    const nvinfer1::Dims inputDims  = bundle->engine->getTensorShape( bundle->inputName.c_str() );
    const nvinfer1::Dims outputDims = bundle->engine->getTensorShape( bundle->outputName.c_str() );
    if( inputDims.nbDims != 4 || inputDims.d[0] != 1 || outputDims.nbDims < 2 || outputDims.d[0] != 1 )
    {
        error = "engine must have fixed batch-one NCHW input and batch-one output";
        return nullptr;
    }
    bundle->inputChannels = inputDims.d[1];
    bundle->inputHeight   = inputDims.d[2];
    bundle->inputWidth    = inputDims.d[3];
    bundle->inputSize     = 1;
    for( int dimension = 1; dimension < inputDims.nbDims; ++dimension )
    {
        if( inputDims.d[dimension] <= 0 )
        {
            error = "dynamic or invalid input dimensions are unsupported";
            return nullptr;
        }
        bundle->inputSize *= inputDims.d[dimension];
    }
    bundle->outputSize = 1;
    for( int dimension = 1; dimension < outputDims.nbDims; ++dimension )
    {
        if( outputDims.d[dimension] <= 0 )
        {
            error = "dynamic or invalid output dimensions are unsupported";
            return nullptr;
        }
        bundle->outputSize *= outputDims.d[dimension];
    }
    bundle->trainedModes  = bundle->inputHeight * bundle->inputWidth;
    bundle->historyLength = ( bundle->inputChannels + 1 ) / 2;
    if( bundle->inputChannels != 2 * bundle->historyLength - 1 || bundle->outputSize != bundle->trainedModes ||
        bundle->trainedModes < m_nModesControlled || bundle->trainedModes > m_nModes ||
        bundle->historyLength > m_historyCapacity )
    {
        error = "engine tensor contract does not cover Nmodes_controlled, exceeds Nmodes, or exceeds max_Nhist";
        return nullptr;
    }
    bundle->context = bundle->engine->createExecutionContext();
    if( bundle->context == nullptr )
    {
        error = "TensorRT execution-context creation failed";
        return nullptr;
    }
    return bundle;
}

inline bool po4ao::loadLinearPolicy( const std::string &filename, EngineBundle &bundle, std::string &error )
{
    std::ifstream file( filename, std::ios::binary );
    if( !file )
        return true;

    std::array<char, 8> magic{};
    uint32_t            version  = 0;
    uint32_t            features = 0;
    uint32_t            modes    = 0;
    uint64_t            modelId  = 0;
    file.read( magic.data(), magic.size() );
    file.read( reinterpret_cast<char *>( &version ), sizeof( version ) );
    file.read( reinterpret_cast<char *>( &features ), sizeof( features ) );
    file.read( reinterpret_cast<char *>( &modes ), sizeof( modes ) );
    file.read( reinterpret_cast<char *>( &modelId ), sizeof( modelId ) );
    const std::array<char, 8> expectedMagic{ 'P', 'O', '4', 'A', 'O', 'L', 'I', 'N' };
    if( !file || magic != expectedMagic || version != 1 || features == 0 || features % 2 == 0 ||
        modes < static_cast<uint32_t>( m_nModesControlled ) || modes > static_cast<uint32_t>( m_nModes ) ||
        modelId != bundle.modelId )
    {
        error = "linear-policy sidecar has an invalid tensor contract or model ID";
        return false;
    }

    const uint32_t historyLength = ( features + 1 ) / 2;
    const uint64_t inputSize     = static_cast<uint64_t>( features ) * modes;
    if( historyLength > static_cast<uint32_t>( m_historyCapacity ) ||
        inputSize > static_cast<uint64_t>( std::numeric_limits<int>::max() ) )
    {
        error = "linear-policy tensor size exceeds the configured capacity";
        return false;
    }
    bundle.inputChannels = static_cast<int>( features );
    bundle.inputHeight   = 1;
    bundle.inputWidth    = static_cast<int>( modes );
    bundle.inputSize     = static_cast<int>( inputSize );
    bundle.outputSize    = static_cast<int>( modes );
    bundle.trainedModes  = static_cast<int>( modes );
    bundle.historyLength = static_cast<int>( historyLength );

    const size_t weightCount = static_cast<size_t>( features ) * modes;
    bundle.linearWeights.resize( weightCount );
    file.read( reinterpret_cast<char *>( bundle.linearWeights.data() ), weightCount * sizeof( float ) );
    if( !file || file.peek() != std::ifstream::traits_type::eof() )
    {
        error = "linear-policy sidecar has an invalid size";
        return false;
    }
    for( float weight : bundle.linearWeights )
    {
        if( !finiteFloat( weight ) )
        {
            error = "linear-policy sidecar contains a non-finite weight";
            return false;
        }
    }
    return true;
}

inline int po4ao::prepareEngineMemory()
{
    if( m_cudaStream != nullptr )
        return 0;
    const size_t inputCapacity = ( 2 * static_cast<size_t>( m_historyCapacity ) - 1 ) * static_cast<size_t>( m_nModes );
    int          lowPriority   = 0;
    if( cudaSetDevice( m_cudaDevice ) != cudaSuccess ||
        cudaDeviceGetStreamPriorityRange( &lowPriority, nullptr ) != cudaSuccess ||
        cudaStreamCreateWithFlags( &m_cudaStream, cudaStreamNonBlocking ) != cudaSuccess ||
        cudaStreamCreateWithPriority( &m_engineWarmupStream, cudaStreamNonBlocking, lowPriority ) != cudaSuccess ||
        cudaMalloc( reinterpret_cast<void **>( &m_deviceInput ), inputCapacity * sizeof( float ) ) != cudaSuccess ||
        cudaMalloc( reinterpret_cast<void **>( &m_deviceOutput ), static_cast<size_t>( m_nModes ) * sizeof( float ) ) !=
            cudaSuccess ||
        cudaMalloc( reinterpret_cast<void **>( &m_deviceWarmupInput ), inputCapacity * sizeof( float ) ) !=
            cudaSuccess ||
        cudaMalloc( reinterpret_cast<void **>( &m_deviceWarmupOutput ),
                    static_cast<size_t>( m_nModes ) * sizeof( float ) ) != cudaSuccess )
    {
        cleanupEngineMemory();
        return -1;
    }
    return 0;
}

inline void po4ao::cleanupEngineMemory()
{
    if( m_deviceInput != nullptr )
        cudaFree( m_deviceInput );
    if( m_deviceOutput != nullptr )
        cudaFree( m_deviceOutput );
    if( m_deviceWarmupInput != nullptr )
        cudaFree( m_deviceWarmupInput );
    if( m_deviceWarmupOutput != nullptr )
        cudaFree( m_deviceWarmupOutput );
    if( m_cudaStream != nullptr )
        cudaStreamDestroy( m_cudaStream );
    if( m_engineWarmupStream != nullptr )
        cudaStreamDestroy( m_engineWarmupStream );
    m_deviceInput        = nullptr;
    m_deviceOutput       = nullptr;
    m_deviceWarmupInput  = nullptr;
    m_deviceWarmupOutput = nullptr;
    m_cudaStream         = nullptr;
    m_engineWarmupStream = nullptr;
}

inline bool po4ao::warmEngineBundle(
    EngineBundle &bundle, cudaStream_t stream, float *deviceInput, float *deviceOutput, std::string &error )
{
    if( stream == nullptr || deviceInput == nullptr || deviceOutput == nullptr )
    {
        error = "engine warm-up resources are unavailable";
        return false;
    }

    const size_t inputBytes = static_cast<size_t>( bundle.inputSize ) * sizeof( float );
    if( cudaMemsetAsync( deviceInput, 0, inputBytes, stream ) != cudaSuccess )
    {
        error = "engine warm-up input initialization failed";
        return false;
    }
    if( !bundle.context->setTensorAddress( bundle.inputName.c_str(), deviceInput ) ||
        !bundle.context->setTensorAddress( bundle.outputName.c_str(), deviceOutput ) ||
        !bundle.context->enqueueV3( stream ) )
    {
        error = "engine warm-up enqueue failed";
        return false;
    }
    if( cudaStreamSynchronize( stream ) != cudaSuccess )
    {
        error = "engine warm-up synchronization failed";
        return false;
    }
    if( !bundle.context->setTensorAddress( bundle.inputName.c_str(), m_deviceInput ) ||
        !bundle.context->setTensorAddress( bundle.outputName.c_str(), m_deviceOutput ) )
    {
        error = "candidate-engine inference-address binding failed";
        return false;
    }
    return true;
}

inline bool po4ao::captureEngineGraph( EngineBundle &bundle, std::string &error )
{
    if( m_engineWarmupStream == nullptr || !m_inputState || !m_output )
    {
        error = "inference-graph resources are unavailable";
        return false;
    }

    cudaGraph_t graph = nullptr;
    if( cudaStreamBeginCapture( m_engineWarmupStream, cudaStreamCaptureModeThreadLocal ) != cudaSuccess )
    {
        error = "inference-graph capture start failed";
        return false;
    }

    const size_t inputBytes  = static_cast<size_t>( bundle.inputSize ) * sizeof( float );
    const size_t outputBytes = static_cast<size_t>( bundle.outputSize ) * sizeof( float );
    const bool   captured =
        cudaMemcpyAsync(
            m_deviceInput, m_inputState.get(), inputBytes, cudaMemcpyHostToDevice, m_engineWarmupStream ) ==
            cudaSuccess &&
        bundle.context->enqueueV3( m_engineWarmupStream ) &&
        cudaMemcpyAsync( m_output.get(), m_deviceOutput, outputBytes, cudaMemcpyDeviceToHost, m_engineWarmupStream ) ==
            cudaSuccess;
    const cudaError_t captureStatus = cudaStreamEndCapture( m_engineWarmupStream, &graph );
    if( !captured || captureStatus != cudaSuccess || graph == nullptr )
    {
        if( graph != nullptr )
            cudaGraphDestroy( graph );
        error = "inference-graph capture failed";
        return false;
    }

    cudaGraphExec_t graphExec = nullptr;
    if( cudaGraphInstantiate( &graphExec, graph, 0 ) != cudaSuccess )
    {
        cudaGraphDestroy( graph );
        error = "inference-graph instantiation failed";
        return false;
    }
    cudaGraphDestroy( graph );
    if( cudaGraphUpload( graphExec, m_engineWarmupStream ) != cudaSuccess ||
        cudaStreamSynchronize( m_engineWarmupStream ) != cudaSuccess )
    {
        cudaGraphExecDestroy( graphExec );
        error = "inference-graph upload failed";
        return false;
    }

    bundle.graphExec = graphExec;
    return true;
}

inline bool po4ao::loadExplorationNoise()
{
    std::ifstream file( m_exploreNoiseFile );
    if( !file )
        return false;

    m_explorationNoise.clear();
    m_explorationNoiseRows = 0;
    m_explorationNoiseCols = 0;
    std::string line;
    while( std::getline( file, line ) )
    {
        if( line.empty() )
            continue;
        std::stringstream  rowStream( line );
        std::string        token;
        std::vector<float> row;
        while( std::getline( rowStream, token, ',' ) )
        {
            try
            {
                row.push_back( std::stof( token ) );
            }
            catch( const std::exception & )
            {
                return false;
            }
        }
        if( row.empty() )
            continue;
        if( m_explorationNoiseRows == 0 )
            m_explorationNoiseCols = row.size();
        else if( row.size() != m_explorationNoiseCols )
            return false;
        m_explorationNoise.insert( m_explorationNoise.end(), row.begin(), row.end() );
        ++m_explorationNoiseRows;
    }
    return m_explorationNoiseRows > 0 && m_explorationNoiseCols >= static_cast<size_t>( m_nModesControlled );
}

inline void po4ao::modelLoaderLoop()
{
    while( !m_loaderShutdown.load( std::memory_order_acquire ) )
    {
        { // mutex scope
            std::unique_lock<std::mutex> lock( m_loaderMutex );
            m_loaderCondition.wait( lock,
                                    [this]()
                                    {
                                        return m_loaderShutdown.load( std::memory_order_acquire ) ||
                                               m_reloadRequested.load( std::memory_order_acquire ) ||
                                               m_retiredEngine.load( std::memory_order_acquire ) != nullptr;
                                    } );
        }
        delete m_retiredEngine.exchange( nullptr, std::memory_order_acq_rel );
        if( m_loaderShutdown.load( std::memory_order_acquire ) )
            break;
        if( !m_reloadRequested.exchange( false, std::memory_order_acq_rel ) )
            continue;

        const uint64_t requestGeneration = m_reloadGeneration.load( std::memory_order_acquire );
        std::string    error;
        auto           candidate = createEngineBundle( m_engineDirs + "/" + m_engineName, error );
        if( candidate != nullptr && candidate->linearWeights.empty() && m_cudaStream == nullptr &&
            prepareEngineMemory() < 0 )
        {
            error = "CUDA inference-buffer allocation failed";
            candidate.reset();
        }
        if( candidate != nullptr && candidate->linearWeights.empty() &&
            !warmEngineBundle( *candidate, m_engineWarmupStream, m_deviceWarmupInput, m_deviceWarmupOutput, error ) )
            candidate.reset();
        if( candidate != nullptr && candidate->linearWeights.empty() && !captureEngineGraph( *candidate, error ) )
            candidate.reset();
        if( candidate == nullptr )
        {
            { // mutex scope
                std::lock_guard<std::mutex> lock( m_reloadErrorMutex );
                m_reloadError = error;
            }
            m_reloadStatus.store( 3, std::memory_order_release );
            continue;
        }
        if( requestGeneration != m_modelGeneration.load( std::memory_order_acquire ) )
        {
            m_reloadStatus.store( 0, std::memory_order_release );
            continue;
        }
        candidate->generation = requestGeneration;
        if( m_pendingEngine.load( std::memory_order_acquire ) != nullptr )
        {
            std::lock_guard<std::mutex> lock( m_reloadErrorMutex );
            m_reloadError = "a validated model is already awaiting activation";
            m_reloadStatus.store( 3, std::memory_order_release );
            continue;
        }
        // Publish ready before the pointer so activation cannot be followed by a stale ready status.
        m_reloadStatus.store( 2, std::memory_order_release );
        m_pendingEngine.store( candidate.release(), std::memory_order_release );
    }
    delete m_retiredEngine.exchange( nullptr, std::memory_order_acq_rel );
}

inline void po4ao::activatePendingEngine()
{
    if( m_pendingEngine.load( std::memory_order_acquire ) == nullptr )
        return;
    EngineBundle *candidate = takePendingModel(
        m_pendingEngine, m_retiredEngine, m_reloadStatus, m_modelGeneration.load( std::memory_order_acquire ) );
    if( candidate == nullptr )
    {
        if( m_retiredEngine.load( std::memory_order_acquire ) != nullptr )
            m_loaderCondition.notify_one();
        return;
    }

    EngineBundle *old = m_activeEngine;
    m_activeEngine    = candidate;
    m_activeModelId.store( candidate->modelId, std::memory_order_release );
    m_activeCpuInference.store( !candidate->linearWeights.empty(), std::memory_order_release );
    m_useNeuralNetwork.store( true, std::memory_order_release );
    m_reloadStatus.store( 0, std::memory_order_release );
    m_retiredEngine.store( old, std::memory_order_release );
    m_loaderCondition.notify_one();
}

inline void po4ao::resetActiveModel( uint64_t generation )
{
    if( m_observationHistory )
        m_observationHistory->reset();
    if( m_commandHistory )
        m_commandHistory->reset();
    m_episodeBuffers[static_cast<size_t>( m_activeEpisodeSlot )].reset();

    if( m_activeEngine != nullptr && m_activeEngine->generation >= generation )
        return;

    m_activeModelId.store( 0, std::memory_order_release );
    m_activeCpuInference.store( false, std::memory_order_release );
    m_useNeuralNetwork.store( false, std::memory_order_release );
}

inline bool po4ao::runInference( const float *observation )
{
    if( m_activeEngine == nullptr ||
        ( m_activeEngine->linearWeights.empty() && m_activeEngine->graphExec == nullptr ) ||
        !assemblePolicyInput( m_inputState.get(),
                              static_cast<size_t>( m_activeEngine->trainedModes ),
                              static_cast<size_t>( m_activeEngine->historyLength ),
                              *m_observationHistory,
                              *m_commandHistory ) )
        return false;

    if( !m_activeEngine->linearWeights.empty() )
    {
        if( !applyLinearLocalPolicy( m_output.get(),
                                     m_inputState.get(),
                                     m_activeEngine->linearWeights.data(),
                                     static_cast<size_t>( m_activeEngine->inputChannels ),
                                     static_cast<size_t>( m_activeEngine->trainedModes ),
                                     static_cast<size_t>( m_nModesControlled ) ) )
            return false;
    }
    else if( cudaGraphLaunch( m_activeEngine->graphExec, m_cudaStream ) != cudaSuccess ||
             cudaStreamSynchronize( m_cudaStream ) != cudaSuccess )
    {
        return false;
    }
    if( !blendPolicyCommand(
            m_output.get(), observation, static_cast<size_t>( m_nModesControlled ), m_integratorGain, m_policyBlend ) )
        return false;
    return finalizeControlledCommand( m_output.get(),
                                      observation,
                                      static_cast<size_t>( m_nModes ),
                                      static_cast<size_t>( m_nModesControlled ),
                                      m_maxCommandAbs );
}

inline void po4ao::makeIntegratorCommand( const float *observation, bool addNoise )
{
    for( int mode = 0; mode < m_nModesControlled; ++mode )
        m_integratorCommands[mode] = m_integratorGain * observation[mode];
    memcpy( m_output.get(), m_integratorCommands.get(), static_cast<size_t>( m_nModesControlled ) * sizeof( float ) );

    if( addNoise && !m_explorationNoise.empty() )
    {
        const uint64_t warmupFrames = static_cast<uint64_t>( m_warmupEpisodes ) * m_iterationsPerEpisode;
        const float    progress =
            warmupFrames > 1 ? std::min( 1.F, static_cast<float>( m_frameCounter ) / ( warmupFrames - 1 ) ) : 1.F;
        const float  sigma = m_explorationSigmaStart + progress * ( m_explorationSigmaEnd - m_explorationSigmaStart );
        const float *noise =
            m_explorationNoise.data() + ( m_frameCounter % m_explorationNoiseRows ) * m_explorationNoiseCols;
        for( int mode = 0; mode < m_nModesControlled; ++mode )
            m_output[mode] += sigma * noise[mode];
    }
    if( !finalizeControlledCommand( m_output.get(),
                                    observation,
                                    static_cast<size_t>( m_nModes ),
                                    static_cast<size_t>( m_nModesControlled ),
                                    m_maxCommandAbs ) )
        memset( m_output.get(), 0, static_cast<size_t>( m_nModes ) * sizeof( float ) );
}

inline int po4ao::openOutputStreams()
{
    closeOutputStreams();

    auto openOrCreate = [this]( IMAGE                         &stream,
                                const std::string             &name,
                                uint8_t                        dimensions,
                                const std::array<uint32_t, 3> &sizes,
                                uint8_t                        dataType,
                                bool                           recreateIfMismatched )
    {
        auto create = [&stream, &name, dimensions, &sizes, dataType]()
        {
            uint32_t imageSize[3] = { sizes[0], sizes[1], sizes[2] };
            if( ImageStreamIO_createIm_gpu( &stream,
                                            name.c_str(),
                                            dimensions,
                                            imageSize,
                                            dataType,
                                            -1,
                                            1,
                                            IMAGE_NB_SEMAPHORE,
                                            0,
                                            MATH_DATA,
                                            0 ) != IMAGESTREAMIO_SUCCESS )
            {
                return false;
            }
            return true;
        };

        auto matches = [&stream, dimensions, &sizes, dataType]()
        {
            bool valid = stream.md != nullptr && stream.array.raw != nullptr && stream.md[0].naxis == dimensions &&
                         stream.md[0].sem > 0 && stream.md[0].datatype == dataType;
            for( uint8_t axis = 0; valid && axis < dimensions; ++axis )
                valid = stream.md[0].size[axis] == sizes[axis];
            return valid;
        };

        bool created   = false;
        bool recreated = false;
        if( ImageStreamIO_openIm( &stream, name.c_str() ) != IMAGESTREAMIO_SUCCESS )
        {
            if( !create() )
                return false;
            created = true;
        }
        else if( !matches() )
        {
            if( !recreateIfMismatched )
            {
                ImageStreamIO_closeIm( &stream );
                return false;
            }
            if( ImageStreamIO_destroyIm( &stream ) != IMAGESTREAMIO_SUCCESS )
            {
                ImageStreamIO_closeIm( &stream );
                return false;
            }
            if( !create() )
                return false;
            created   = true;
            recreated = true;
        }

        if( !matches() )
        {
            ImageStreamIO_closeIm( &stream );
            return false;
        }
        if( created )
        {
            memset( stream.array.raw, 0, stream.md[0].nelement * ImageStreamIO_typesize( dataType ) );
            log<text_log>(
                ( recreated ? "recreated stale PO4AO shared-memory stream " : "created PO4AO shared-memory stream " ) +
                    name,
                logPrio::LOG_INFO );
        }
        return true;
    };

    const std::array<uint32_t, 3> outputSize = { static_cast<uint32_t>( m_nModes ), 1, 1 };
    const std::array<uint32_t, 3> replaySize = {
        static_cast<uint32_t>( m_nModes ), static_cast<uint32_t>( m_replayBufferSize ), 2 };
    const std::array<uint32_t, 3> metadataSize = { 4, 1, 1 };

    m_outputOpened = openOrCreate( m_outputStream, m_outputChannel, 2, outputSize, _DATATYPE_FLOAT, false );
    m_replayOpened = openOrCreate( m_replayStream, m_bufferChannel, 3, replaySize, _DATATYPE_FLOAT, true );
    m_replayMetadataOpened =
        openOrCreate( m_replayMetadataStream, m_bufferMetadataChannel, 2, metadataSize, _DATATYPE_UINT32, true );
    m_warmupOpened = openOrCreate( m_warmupStream, m_warmupBufferChannel, 3, replaySize, _DATATYPE_FLOAT, true );
    m_warmupMetadataOpened =
        openOrCreate( m_warmupMetadataStream, m_warmupMetadataChannel, 2, metadataSize, _DATATYPE_UINT32, true );
    if( !m_outputOpened || !m_replayOpened || !m_replayMetadataOpened || !m_warmupOpened || !m_warmupMetadataOpened )
    {
        closeOutputStreams();
        return log<software_error, -1>( { __FILE__,
                                          __LINE__,
                                          "could not open or create correctly sized "
                                          "PO4AO shared-memory streams" } );
    }

    memset( m_output.get(), 0, static_cast<size_t>( m_nModes ) * sizeof( float ) );
    sendOutput();

    invalidateReplay( m_replayStream, m_replayOpened, m_replayMetadataStream, m_replayMetadataOpened );
    invalidateReplay( m_warmupStream, m_warmupOpened, m_warmupMetadataStream, m_warmupMetadataOpened );
    return 0;
}

inline void po4ao::closeOutputStreams()
{
    if( m_outputOpened )
        ImageStreamIO_closeIm( &m_outputStream );
    if( m_replayOpened )
        ImageStreamIO_closeIm( &m_replayStream );
    if( m_replayMetadataOpened )
        ImageStreamIO_closeIm( &m_replayMetadataStream );
    if( m_warmupOpened )
        ImageStreamIO_closeIm( &m_warmupStream );
    if( m_warmupMetadataOpened )
        ImageStreamIO_closeIm( &m_warmupMetadataStream );
    m_outputOpened         = false;
    m_replayOpened         = false;
    m_replayMetadataOpened = false;
    m_warmupOpened         = false;
    m_warmupMetadataOpened = false;
}

inline int po4ao::sendOutput()
{
    if( !m_outputOpened )
        return -1;
    m_outputStream.md[0].write = 1;
    memcpy( m_outputStream.array.raw, m_output.get(), static_cast<size_t>( m_nModes ) * sizeof( float ) );
    ++m_outputStream.md[0].cnt0;
    m_outputStream.md[0].write = 0;
    ImageStreamIO_sempost( &m_outputStream, -1 );
    return 0;
}

inline int po4ao::invalidateReplay( IMAGE &stream, bool opened, IMAGE &metadataStream, bool metadataOpened )
{
    if( !opened || !metadataOpened )
        return -1;
    stream.md[0].write         = 1;
    metadataStream.md[0].write = 1;
    memset( metadataStream.array.raw, 0, 4 * sizeof( uint32_t ) );
    const uint64_t publication = stream.md[0].cnt0 + 1;
    metadataStream.md[0].cnt0  = publication;
    stream.md[0].cnt0          = publication;
    stream.md[0].cnt1          = 0;
    stream.md[0].cnt2          = 0;
    stream.md[0].write         = 0;
    metadataStream.md[0].write = 0;
    ImageStreamIO_sempost( &stream, -1 );
    ImageStreamIO_sempost( &metadataStream, -1 );
    return 0;
}

inline int po4ao::publishReplay(
    IMAGE &stream, bool opened, IMAGE &metadataStream, bool metadataOpened, uint64_t episodeIdentifier )
{
    if( !opened || !metadataOpened || !m_publishObservations || !m_publishCommands )
        return -1;
    const size_t valid         = std::min( m_publishObservations->size(), m_publishCommands->size() );
    const size_t planeSize     = static_cast<size_t>( m_nModes ) * m_replayBufferSize;
    stream.md[0].write         = 1;
    metadataStream.md[0].write = 1;
    memset( stream.array.raw, 0, 2 * planeSize * sizeof( float ) );
    auto *destination = static_cast<float *>( stream.array.raw );
    for( size_t sample = 0; sample < valid; ++sample )
    {
        memcpy( destination + sample * m_nModes,
                m_publishObservations->getItem( sample ),
                static_cast<size_t>( m_nModes ) * sizeof( float ) );
        memcpy( destination + planeSize + sample * m_nModes,
                m_publishCommands->getItem( sample ),
                static_cast<size_t>( m_nModes ) * sizeof( float ) );
    }
    const uint64_t publication = stream.md[0].cnt0 + 1;
    const uint64_t modelId     = m_activeModelId.load( std::memory_order_acquire );
    auto          *metadata    = static_cast<uint32_t *>( metadataStream.array.raw );
    metadata[0]                = static_cast<uint32_t>( valid );
    metadata[1]                = static_cast<uint32_t>( episodeIdentifier );
    metadata[2]                = static_cast<uint32_t>( modelId & 0xFFFFFFFFULL );
    metadata[3]                = static_cast<uint32_t>( modelId >> 32 );
    stream.md[0].cnt1          = 0;
    stream.md[0].cnt2          = 0;
    metadataStream.md[0].cnt0  = publication;
    stream.md[0].cnt0          = publication;
    stream.md[0].write         = 0;
    metadataStream.md[0].write = 0;
    ImageStreamIO_sempost( &stream, -1 );
    ImageStreamIO_sempost( &metadataStream, -1 );
    return 0;
}

inline void po4ao::finishEpisode()
{
    EpisodeBuffer &completed = m_episodeBuffers[static_cast<size_t>( m_activeEpisodeSlot )];
    completed.episode        = m_episodeCounter.fetch_add( 1, std::memory_order_acq_rel ) + 1;
    completed.generation     = m_activeCollectionGeneration;

    int next = -1;
    for( size_t slot = 0; slot < c_episodeSlots; ++slot )
    {
        int expected = 0;
        if( m_episodeStates[slot].compare_exchange_strong( expected, 1, std::memory_order_acq_rel ) )
        {
            next = static_cast<int>( slot );
            break;
        }
    }
    if( next < 0 )
    {
        completed.reset();
        m_replayDrops.fetch_add( 1, std::memory_order_relaxed );
        return;
    }

    const uint64_t write                 = m_readyWrite.load( std::memory_order_relaxed );
    m_readyQueue[write % c_episodeSlots] = m_activeEpisodeSlot;
    m_episodeStates[static_cast<size_t>( m_activeEpisodeSlot )].store( 2, std::memory_order_release );
    m_readyWrite.store( write + 1, std::memory_order_release );
    m_activeEpisodeSlot = next;
    m_episodeBuffers[static_cast<size_t>( next )].reset();
    m_replayCondition.notify_one();
}

inline void po4ao::replayPublisherLoop()
{
    while( true )
    {
        std::unique_lock<std::mutex> publicationLock( m_publicationMutex );
        const uint64_t               read  = m_readyRead.load( std::memory_order_relaxed );
        const uint64_t               write = m_readyWrite.load( std::memory_order_acquire );
        if( read == write )
        {
            publicationLock.unlock();
            if( m_replayShutdown.load( std::memory_order_acquire ) )
                break;
            std::unique_lock<std::mutex> lock( m_replayMutex );
            m_replayCondition.wait( lock,
                                    [this]()
                                    {
                                        return m_replayShutdown.load( std::memory_order_acquire ) ||
                                               m_readyRead.load( std::memory_order_relaxed ) !=
                                                   m_readyWrite.load( std::memory_order_acquire );
                                    } );
            continue;
        }

        const int slot = m_readyQueue[read % c_episodeSlots];
        m_readyRead.store( read + 1, std::memory_order_release );
        m_episodeStates[static_cast<size_t>( slot )].store( 3, std::memory_order_release );
        EpisodeBuffer &episode             = m_episodeBuffers[static_cast<size_t>( slot )];
        const uint64_t requestedGeneration = m_collectionGeneration.load( std::memory_order_acquire );
        if( episode.generation != requestedGeneration )
        {
            episode.reset();
            m_episodeStates[static_cast<size_t>( slot )].store( 0, std::memory_order_release );
            continue;
        }
        if( m_publicationGeneration != requestedGeneration )
        {
            m_publishObservations->reset();
            m_publishCommands->reset();
            m_publicationGeneration = requestedGeneration;
        }
        for( size_t frame = 0; frame < episode.count; ++frame )
        {
            m_publishObservations->add( episode.observations.data() + frame * m_nModes );
            m_publishCommands->add( episode.commands.data() + frame * m_nModes );
        }
        publishReplay(
            m_replayStream, m_replayOpened, m_replayMetadataStream, m_replayMetadataOpened, episode.episode );
        if( episode.episode == m_warmupEpisodes )
            publishReplay(
                m_warmupStream, m_warmupOpened, m_warmupMetadataStream, m_warmupMetadataOpened, episode.episode );
        episode.reset();
        m_episodeStates[static_cast<size_t>( slot )].store( 0, std::memory_order_release );
    }
}

inline int po4ao::appStartup()
{
    if( m_configVersion != 2 ||
        ( m_linearInferenceBackend != "auto" && m_linearInferenceBackend != "cpu" &&
          m_linearInferenceBackend != "tensorrt" ) ||
        m_nModes <= 0 || m_nModesControlled <= 0 || m_nModesControlled > m_nModes || m_historyCapacity <= 0 ||
        m_iterationsPerEpisode == 0 || m_replayBufferSize <= 0 || m_warmupEpisodes == 0 ||
        static_cast<uint64_t>( m_warmupEpisodes ) * m_iterationsPerEpisode >
            static_cast<uint64_t>( m_replayBufferSize ) ||
        m_explorationSigmaEnd < 0.F || m_explorationSigmaStart < m_explorationSigmaEnd ||
        !finiteFloat( m_policyBlend ) || m_policyBlend < 0.F || m_policyBlend > 1.F || m_maxCommandAbs <= 0.F ||
        m_latencyDeadlineUs == 0 || m_statsPeriodSeconds <= 0.0 )
        return log<software_error, -1>( { __FILE__, __LINE__, "invalid PO4AO configuration" } );
    if( !CircularBuffer::selfTest() )
        return log<software_error, -1>( { __FILE__, __LINE__, "CircularBuffer self-test failed" } );
    if( !loadExplorationNoise() )
        return log<software_error, -1>( { __FILE__, __LINE__, "exploration-noise load failed" } );

    m_activeModelId.store( 0, std::memory_order_release );
    m_activeCpuInference.store( false, std::memory_order_release );
    m_useNeuralNetwork.store( false, std::memory_order_release );
    log<text_log>( "PO4AO starting in fallback mode; waiting for an explicit model reload", logPrio::LOG_INFO );

    for( auto &sample : m_latencySamples )
        sample.store( 0, std::memory_order_relaxed );
    for( auto &state : m_episodeStates )
        state.store( 0, std::memory_order_relaxed );
    m_controlEnabled.store( m_controlEnabledConfig, std::memory_order_release );
    m_lastStatsUpdate = std::chrono::steady_clock::now();

    createStandardIndiToggleSw( m_indiP_reloadToggle, "reload_engine" );
    registerIndiPropertyNew( m_indiP_reloadToggle, INDI_NEWCALLBACK( m_indiP_reloadToggle ) );
    createStandardIndiToggleSw( m_indiP_resetModel, "reset_model" );
    registerIndiPropertyNew( m_indiP_resetModel, INDI_NEWCALLBACK( m_indiP_resetModel ) );
    createStandardIndiToggleSw( m_indiP_controlToggle, "control_enable" );
    registerIndiPropertyNew( m_indiP_controlToggle, INDI_NEWCALLBACK( m_indiP_controlToggle ) );
    createStandardIndiToggleSw( m_indiP_recollectWarmup, "recollect_warmup" );
    registerIndiPropertyNew( m_indiP_recollectWarmup, INDI_NEWCALLBACK( m_indiP_recollectWarmup ) );

    REG_INDI_NEWPROP_NOCB( m_indiP_runtimeStats, "runtime_stats", pcf::IndiProperty::Number );
    for( const char *element : { "p50_us",
                                 "p95_us",
                                 "p99_us",
                                 "max_us",
                                 "missed_frames",
                                 "deadline_misses",
                                 "inference_failures",
                                 "replay_drops" } )
        m_indiP_runtimeStats.add( pcf::IndiElement( element ) );
    REG_INDI_NEWPROP_NOCB( m_indiP_modelStatus, "model_status", pcf::IndiProperty::Text );
    m_indiP_modelStatus.add( pcf::IndiElement( "active_id" ) );
    m_indiP_modelStatus.add( pcf::IndiElement( "active_backend" ) );
    m_indiP_modelStatus.add( pcf::IndiElement( "reload" ) );

    m_loaderThread = std::thread( [this]() { modelLoaderLoop(); } );
    m_replayThread = std::thread( [this]() { replayPublisherLoop(); } );
    // The standard early-return macro would skip rollback of the two worker threads.
    if( shmimMonitorT::appStartup() < 0 )
    {
        m_loaderShutdown.store( true, std::memory_order_release );
        m_replayShutdown.store( true, std::memory_order_release );
        m_loaderCondition.notify_all();
        m_replayCondition.notify_all();
        m_loaderThread.join();
        m_replayThread.join();
        return log<software_error, -1>( { __FILE__, __LINE__ } );
    }
    state( stateCodes::OPERATING );
    return 0;
}

inline void po4ao::updateRuntimeStatistics()
{
    const auto now = std::chrono::steady_clock::now();
    if( std::chrono::duration<double>( now - m_lastStatsUpdate ).count() < m_statsPeriodSeconds )
        return;
    m_lastStatsUpdate = now;

    const uint64_t        total = m_latencyCount.load( std::memory_order_acquire );
    const size_t          count = static_cast<size_t>( std::min<uint64_t>( total, c_latencySamples ) );
    std::vector<uint32_t> samples;
    samples.reserve( count );
    for( size_t index = 0; index < count; ++index )
    {
        const uint32_t value = m_latencySamples[index].load( std::memory_order_relaxed );
        if( value > 0 )
            samples.push_back( value );
    }
    std::sort( samples.begin(), samples.end() );
    auto percentile = [&samples]( double fraction )
    {
        if( samples.empty() )
            return 0.0;
        const size_t index =
            std::min( samples.size() - 1, static_cast<size_t>( std::ceil( fraction * samples.size() ) ) - 1 );
        return static_cast<double>( samples[index] );
    };
    const double p50     = percentile( 0.50 );
    const double p95     = percentile( 0.95 );
    const double p99     = percentile( 0.99 );
    const double maximum = samples.empty() ? 0.0 : samples.back();
    updateIfChanged( m_indiP_runtimeStats, "p50_us", p50 );
    updateIfChanged( m_indiP_runtimeStats, "p95_us", p95 );
    updateIfChanged( m_indiP_runtimeStats, "p99_us", p99 );
    updateIfChanged( m_indiP_runtimeStats, "max_us", maximum );
    updateIfChanged( m_indiP_runtimeStats,
                     "missed_frames",
                     static_cast<double>( m_missedFrames.load( std::memory_order_relaxed ) ) );
    updateIfChanged( m_indiP_runtimeStats,
                     "deadline_misses",
                     static_cast<double>( m_deadlineMisses.load( std::memory_order_relaxed ) ) );
    updateIfChanged( m_indiP_runtimeStats,
                     "inference_failures",
                     static_cast<double>( m_inferenceFailures.load( std::memory_order_relaxed ) ) );
    updateIfChanged(
        m_indiP_runtimeStats, "replay_drops", static_cast<double>( m_replayDrops.load( std::memory_order_relaxed ) ) );

    const uint64_t     activeModelId = m_activeModelId.load( std::memory_order_acquire );
    std::ostringstream modelId;
    modelId << std::hex << std::setw( 16 ) << std::setfill( '0' ) << activeModelId;
    static const std::array<std::string, 4> reloadNames  = { "idle", "loading", "ready", "failed" };
    const int                               reloadStatus = m_reloadStatus.load( std::memory_order_acquire );
    updateIfChanged( m_indiP_modelStatus, "active_id", modelId.str() );
    std::string activeBackend = "fallback";
    if( activeModelId != 0 )
        activeBackend = m_activeCpuInference.load( std::memory_order_acquire ) ? "cpu" : "tensorrt";
    updateIfChanged( m_indiP_modelStatus, "active_backend", activeBackend );
    updateIfChanged( m_indiP_modelStatus, "reload", reloadNames[static_cast<size_t>( reloadStatus )] );

    std::ostringstream message;
    message << "PO4AO latency(us) p50=" << p50 << " p95=" << p95 << " p99=" << p99 << " max=" << maximum
            << " missed=" << m_missedFrames.load() << " deadline=" << m_deadlineMisses.load()
            << " inference_failures=" << m_inferenceFailures.load() << " replay_drops=" << m_replayDrops.load()
            << " model=" << modelId.str() << " backend=" << activeBackend;
    log<text_log>( message.str(), logPrio::LOG_INFO );
}

inline int po4ao::appLogic()
{
    SHMIMMONITOR_APP_LOGIC;
    std::unique_lock<std::mutex> lock( m_indiMutex );
    SHMIMMONITOR_UPDATE_INDI;
    updateSwitchIfChanged( m_indiP_controlToggle,
                           "toggle",
                           m_controlEnabled.load( std::memory_order_acquire ) ? pcf::IndiElement::On
                                                                              : pcf::IndiElement::Off,
                           INDI_OK );
    updateSwitchIfChanged( m_indiP_reloadToggle,
                           "toggle",
                           pcf::IndiElement::Off,
                           m_reloadStatus.load( std::memory_order_acquire ) == 1 ? INDI_BUSY : INDI_IDLE );
    updateSwitchIfChanged( m_indiP_resetModel,
                           "toggle",
                           pcf::IndiElement::Off,
                           m_modelResetRequest.load( std::memory_order_acquire ) != 0 ? INDI_BUSY : INDI_OK );
    updateSwitchIfChanged( m_indiP_recollectWarmup,
                           "toggle",
                           pcf::IndiElement::Off,
                           m_warmupRecollectRequest.load( std::memory_order_acquire ) != 0 ? INDI_BUSY : INDI_OK );
    updateRuntimeStatistics();
    if( m_reloadStatus.load( std::memory_order_acquire ) == 3 )
    {
        std::lock_guard<std::mutex> errorLock( m_reloadErrorMutex );
        log<text_log>( "PO4AO model reload failed: " + m_reloadError, logPrio::LOG_ERROR );
        m_reloadStatus.store( 0, std::memory_order_release );
    }
    return 0;
}

inline int po4ao::appShutdown()
{
    // Always release worker/model resources, even when the monitor reports a shutdown error.
    const int monitorStatus = shmimMonitorT::appShutdown();
    m_controlEnabled.store( false, std::memory_order_release );
    if( m_outputOpened && m_output )
    {
        memset( m_output.get(), 0, static_cast<size_t>( m_nModes ) * sizeof( float ) );
        sendOutput();
    }

    m_loaderShutdown.store( true, std::memory_order_release );
    m_replayShutdown.store( true, std::memory_order_release );
    m_loaderCondition.notify_all();
    m_replayCondition.notify_all();
    if( m_loaderThread.joinable() )
        m_loaderThread.join();
    if( m_replayThread.joinable() )
        m_replayThread.join();
    closeOutputStreams();

    delete m_pendingEngine.exchange( nullptr, std::memory_order_acq_rel );
    delete m_retiredEngine.exchange( nullptr, std::memory_order_acq_rel );
    delete m_activeEngine;
    m_activeEngine = nullptr;
    cleanupEngineMemory();
    if( monitorStatus < 0 )
        return log<software_error, -1>( { __FILE__, __LINE__, "Error from shmimMonitorT::appShutdown" } );
    return 0;
}

inline int po4ao::allocate( const dev::shmimT &dummy )
{
    static_cast<void>( dummy );
    if( shmimMonitorT::m_dataType != _DATATYPE_FLOAT ||
        static_cast<uint64_t>( shmimMonitorT::m_width ) * shmimMonitorT::m_height != static_cast<uint64_t>( m_nModes ) )
        return log<software_error, -1>( { __FILE__, __LINE__, "input stream must be Nmodes float32 values" } );

    std::lock_guard<std::mutex> publicationLock( m_publicationMutex );
    m_readyRead.store( m_readyWrite.load( std::memory_order_acquire ), std::memory_order_release );
    const size_t inputCapacity = ( 2 * static_cast<size_t>( m_historyCapacity ) - 1 ) * static_cast<size_t>( m_nModes );
    if( !m_inputState || !m_output )
    {
        if( m_linearInferenceBackend != "cpu" )
        {
            if( cudaSetDevice( m_cudaDevice ) != cudaSuccess )
                return log<software_error, -1>( { __FILE__, __LINE__, "CUDA inference-device selection failed" } );
            float *inputState = nullptr;
            float *output     = nullptr;
            if( cudaHostAlloc( reinterpret_cast<void **>( &inputState ),
                               inputCapacity * sizeof( float ),
                               cudaHostAllocDefault ) != cudaSuccess )
                return log<software_error, -1>( { __FILE__, __LINE__, "pinned policy-input allocation failed" } );
            std::unique_ptr<float[], HostBufferDeleter> pinnedInput( inputState, HostBufferDeleter{ true } );
            if( cudaHostAlloc( reinterpret_cast<void **>( &output ),
                               static_cast<size_t>( m_nModes ) * sizeof( float ),
                               cudaHostAllocDefault ) != cudaSuccess )
                return log<software_error, -1>( { __FILE__, __LINE__, "pinned policy-output allocation failed" } );
            std::unique_ptr<float[], HostBufferDeleter> pinnedOutput( output, HostBufferDeleter{ true } );
            m_inputState = std::move( pinnedInput );
            m_output     = std::move( pinnedOutput );
        }
        else
        {
            m_inputState = std::unique_ptr<float[], HostBufferDeleter>( new float[inputCapacity], HostBufferDeleter{} );
            m_output     = std::unique_ptr<float[], HostBufferDeleter>( new float[static_cast<size_t>( m_nModes )],
                                                                    HostBufferDeleter{} );
        }
    }
    m_integratorCommands = std::make_unique<float[]>( static_cast<size_t>( m_nModes ) );
    memset( m_inputState.get(), 0, inputCapacity * sizeof( float ) );
    memset( m_output.get(), 0, static_cast<size_t>( m_nModes ) * sizeof( float ) );
    memset( m_integratorCommands.get(), 0, static_cast<size_t>( m_nModes ) * sizeof( float ) );
    m_observationHistory  = std::make_unique<CircularBuffer>( m_nModes, m_historyCapacity );
    m_commandHistory      = std::make_unique<CircularBuffer>( m_nModes, m_historyCapacity - 1 );
    m_publishObservations = std::make_unique<CircularBuffer>( m_nModes, m_replayBufferSize );
    m_publishCommands     = std::make_unique<CircularBuffer>( m_nModes, m_replayBufferSize );

    for( size_t slot = 0; slot < c_episodeSlots; ++slot )
    {
        m_episodeBuffers[slot].configure( m_iterationsPerEpisode, m_nModes );
        m_episodeStates[slot].store( 0, std::memory_order_relaxed );
    }
    m_activeEpisodeSlot = 0;
    m_episodeStates[0].store( 1, std::memory_order_release );
    m_readyRead.store( 0, std::memory_order_relaxed );
    m_readyWrite.store( 0, std::memory_order_relaxed );
    m_collectionGeneration.store( 1, std::memory_order_relaxed );
    m_activeCollectionGeneration = 1;
    m_publicationGeneration      = 1;
    m_warmupRecollectRequest.store( 0, std::memory_order_relaxed );
    m_frameCounter = 0;
    m_episodeCounter.store( 0, std::memory_order_relaxed );
    m_lastInputCounter = 0;
    m_haveInputCounter = false;
    return openOutputStreams();
}

inline void po4ao::beginWarmupCollection( uint64_t generation )
{
    m_episodeBuffers[static_cast<size_t>( m_activeEpisodeSlot )].reset();
    m_observationHistory->reset();
    m_commandHistory->reset();
    m_frameCounter = 0;
    m_episodeCounter.store( 0, std::memory_order_release );
    m_activeCollectionGeneration = generation;
    m_useNeuralNetwork.store( false, std::memory_order_release );
}

inline int po4ao::processImage( void *curr_src, const dev::shmimT &dummy )
{
    static_cast<void>( dummy );
    const auto started       = std::chrono::steady_clock::now();
    auto       recordLatency = [this, started]()
    {
        const auto elapsed =
            std::chrono::duration_cast<std::chrono::microseconds>( std::chrono::steady_clock::now() - started );
        const uint32_t latency =
            static_cast<uint32_t>( std::min<int64_t>( elapsed.count(), std::numeric_limits<uint32_t>::max() ) );
        const uint64_t sample = m_latencyCount.fetch_add( 1, std::memory_order_acq_rel );
        m_latencySamples[sample % c_latencySamples].store( latency, std::memory_order_release );
        if( latency > m_latencyDeadlineUs )
            m_deadlineMisses.fetch_add( 1, std::memory_order_relaxed );
    };
    if( curr_src == nullptr || !m_inputState || !m_output || !m_outputOpened )
        return -1;

    const uint64_t sourceCounter = shmimMonitorT::m_imageStream.md[0].cnt0;
    // Queued notifications can refer to the same latest image after a processing stall.
    if( sourceCounterIsDuplicate( sourceCounter, m_lastInputCounter, m_haveInputCounter ) )
    {
        recordLatency();
        return 0;
    }

    activatePendingEngine();
    const uint64_t resetGeneration = m_modelResetRequest.exchange( 0, std::memory_order_acq_rel );
    if( resetGeneration != 0 )
        resetActiveModel( resetGeneration );
    const auto *observation = static_cast<const float *>( curr_src );
    const bool  consecutive = sourceCounterIsConsecutive( sourceCounter, m_lastInputCounter, m_haveInputCounter );
    if( m_haveInputCounter && sourceCounter > m_lastInputCounter )
    {
        const uint64_t sourceAdvance = sourceCounter - m_lastInputCounter;
        if( sourceAdvance > 1 )
            m_missedFrames.fetch_add( sourceAdvance - 1, std::memory_order_relaxed );
    }
    m_lastInputCounter = sourceCounter;
    m_haveInputCounter = true;

    const uint64_t recollectGeneration = m_warmupRecollectRequest.exchange( 0, std::memory_order_acq_rel );
    if( recollectGeneration != 0 )
        beginWarmupCollection( recollectGeneration );

    if( !consecutive )
    {
        m_observationHistory->reset();
        m_commandHistory->reset();
        m_episodeBuffers[static_cast<size_t>( m_activeEpisodeSlot )].reset();
        if( m_controlEnabled.load( std::memory_order_acquire ) )
            memcpy( m_output.get(), observation, static_cast<size_t>( m_nModes ) * sizeof( float ) );
        else
            makeIntegratorCommand( observation, false );
        const int result = sendOutput();
        ++m_frameCounter;
        recordLatency();
        return result;
    }

    if( !m_controlEnabled.load( std::memory_order_acquire ) )
    {
        m_observationHistory->reset();
        m_commandHistory->reset();
        m_episodeBuffers[static_cast<size_t>( m_activeEpisodeSlot )].reset();
        makeIntegratorCommand( observation, false );
        const int result = sendOutput();
        recordLatency();
        return result;
    }

    m_observationHistory->add( observation );
    const bool warmup         = m_episodeCounter.load( std::memory_order_acquire ) < m_warmupEpisodes;
    const bool inferenceReady = !warmup && m_activeEngine != nullptr &&
                                m_useNeuralNetwork.load( std::memory_order_acquire ) &&
                                m_observationHistory->size() >= static_cast<size_t>( m_activeEngine->historyLength ) &&
                                m_commandHistory->size() + 1 >= static_cast<size_t>( m_activeEngine->historyLength );
    bool inferenceOk = false;
    if( inferenceReady )
        inferenceOk = runInference( observation );
    if( !inferenceOk )
    {
        if( inferenceReady )
            m_inferenceFailures.fetch_add( 1, std::memory_order_relaxed );
        makeIntegratorCommand( observation, warmup );
    }
    m_commandHistory->add( m_output.get() );
    EpisodeBuffer &episode = m_episodeBuffers[static_cast<size_t>( m_activeEpisodeSlot )];
    if( !episode.add( observation, m_output.get() ) )
        m_replayDrops.fetch_add( 1, std::memory_order_relaxed );
    else if( episode.count == episode.capacity )
        finishEpisode();
    if( sendOutput() < 0 )
        return -1;

    ++m_frameCounter;
    recordLatency();
    return 0;
}

INDI_NEWCALLBACK_DEFN( po4ao, m_indiP_reloadToggle )( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getName() != m_indiP_reloadToggle.getName() )
        return log<software_error, -1>( { __FILE__, __LINE__, "invalid reload property" } );
    if( ipRecv["toggle"].getSwitchState() != pcf::IndiElement::On )
        return 0;
    int expected = 0;
    if( !m_reloadStatus.compare_exchange_strong( expected, 1, std::memory_order_acq_rel ) )
        return 0;
    m_reloadGeneration.store( m_modelGeneration.load( std::memory_order_acquire ), std::memory_order_release );
    m_reloadRequested.store( true, std::memory_order_release );
    m_loaderCondition.notify_one();
    updateSwitchIfChanged( m_indiP_reloadToggle, "toggle", pcf::IndiElement::Off, INDI_BUSY );
    return 0;
}

INDI_NEWCALLBACK_DEFN( po4ao, m_indiP_resetModel )( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getName() != m_indiP_resetModel.getName() )
        return log<software_error, -1>( { __FILE__, __LINE__, "invalid model-reset property" } );
    if( ipRecv["toggle"].getSwitchState() != pcf::IndiElement::On )
        return 0;

    const uint64_t generation = m_modelGeneration.fetch_add( 1, std::memory_order_acq_rel ) + 1;
    m_modelResetRequest.store( generation, std::memory_order_release );
    updateSwitchIfChanged( m_indiP_resetModel, "toggle", pcf::IndiElement::Off, INDI_BUSY );
    log<text_log>( "PO4AO active-model reset requested", logPrio::LOG_INFO );
    return 0;
}

INDI_NEWCALLBACK_DEFN( po4ao, m_indiP_controlToggle )( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getName() != m_indiP_controlToggle.getName() )
        return log<software_error, -1>( { __FILE__, __LINE__, "invalid control property" } );
    m_controlEnabled.store( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On, std::memory_order_release );
    return 0;
}

INDI_NEWCALLBACK_DEFN( po4ao, m_indiP_recollectWarmup )( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getName() != m_indiP_recollectWarmup.getName() )
        return log<software_error, -1>( { __FILE__, __LINE__, "invalid warm-up recollection property" } );
    if( ipRecv["toggle"].getSwitchState() != pcf::IndiElement::On )
        return 0;

    const uint64_t generation = m_collectionGeneration.fetch_add( 1, std::memory_order_acq_rel ) + 1;
    m_warmupRecollectRequest.store( generation, std::memory_order_release );
    { // mutex scope
        std::lock_guard<std::mutex> publicationLock( m_publicationMutex );
        invalidateReplay( m_replayStream, m_replayOpened, m_replayMetadataStream, m_replayMetadataOpened );
        invalidateReplay( m_warmupStream, m_warmupOpened, m_warmupMetadataStream, m_warmupMetadataOpened );
    }
    updateSwitchIfChanged( m_indiP_recollectWarmup, "toggle", pcf::IndiElement::Off, INDI_BUSY );
    log<text_log>( "PO4AO warm-up recollection requested", logPrio::LOG_INFO );
    return 0;
}

#endif // po4ao_impl_hpp
