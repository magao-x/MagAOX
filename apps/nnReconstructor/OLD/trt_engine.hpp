// Define TRT entrypoints used in common code
#define DEFINE_TRT_ENTRYPOINTS 1
#define DEFINE_TRT_LEGACY_PARSER_ENTRYPOINT 0

#include "buffers.hpp"
#include "logger.h"
#include "parserOnnxConfig.hpp"
#include "util.hpp"

#include "/opt/MagAOX/vendor/TensorRT-10.0.0.6/include/NvInfer.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
using namespace nvinfer1;

using preciseStopwatch = Stopwatch<>;
Logger gLogger;

class TensorrtEngine
{
public:
    TensorrtEngine()
        : mRuntime(nullptr)
        , mEngine(nullptr)
        , buffers(nullptr)
        , context(nullptr) // This is different
    {
    }

    bool build(std::string dataDirs, std::string onnxFileName, std::string engineDirs, std::string engineName, bool rebuildEngine);

    bool load(std::string enginePath);

    bool infer(float* inputData);

    bool initializeBuffer();
    float* getOutput();
	inline int32_t getOutputSize(){return outputSize;};

    BufferManager* buffers;

private:

    std::shared_ptr<nvinfer1::IRuntime> mRuntime;   // The TensorRT runtime used to deserialize the engine
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; // The TensorRT engine used to run the network
	std::shared_ptr<nvinfer1::IExecutionContext> context;

    int32_t batch{1};
    int32_t inputC{0};
    int32_t inputH{0};
    int32_t inputW{0};
	int32_t outputSize{0};
    const char * inputName;
    const char * outputName;

    bool constructNetwork(std::unique_ptr<nvinfer1::IBuilder>& builder,
        std::unique_ptr<nvinfer1::INetworkDefinition>& network, std::unique_ptr<nvinfer1::IBuilderConfig>& config,
        std::unique_ptr<nvonnxparser::IParser>& parser, std::string onnxFileName, std::string dataDirs);
};


bool TensorrtEngine::build(std::string dataDirs, std::string onnxFileName, std::string engineDirs, std::string engineName, bool rebuildEngine)
{
    const std::string enginePath = engineDirs + "/" + engineName;
    gLogger = Logger();
    std::cerr << "made a new logger" << std::endl;
	if (!rebuildEngine){
		if (doesFileExist(enginePath)) {
			std::cout << "Engine found, not regenerating..." << std::endl;
			return load(enginePath);
		}else{
			std::cout << "Engine not found... Let's build a new one." << std::endl;
		}
	}

    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
    if (!builder)
    {
        std::cerr << "Couldn't createInferBuilder" << std::endl;
        return false;
    }

    auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        std::cerr << "Couldn't createNetworkV2" << std::endl;
        return false;
    }

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        std::cerr << "Couldn't createBuilderConfig" << std::endl;
        return false;
    }

    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
    if (!parser) 
    {
        std::cerr << "Couldn't createParser" << std::endl;
        return false;
    }

    auto constructed = constructNetwork(builder, network, config, parser, onnxFileName, dataDirs);
    if (!constructed)
    {
        std::cerr << "Couldn't constructNetwork" << std::endl;
        return false;
    }

    // CUDA stream used for profiling by the builder.
    auto profileStream = makeCudaStream();
    if (!profileStream)
    {
        std::cerr << "Couldn't makeCudaStream" << std::endl;
        return false;
    }
    config->setProfileStream(*profileStream);
	
    // Register a single optimization profile
    nvinfer1::IOptimizationProfile *optProfile = builder->createOptimizationProfile();
    const auto input = network->getInput(0);
    const auto output = network->getOutput(0);

    const auto inputDims = input->getDimensions();
    const auto outputDims = output->getDimensions();
	
	
    inputName = input->getName();
    outputName = output->getName();

    inputC = inputDims.d[1];
    inputH = inputDims.d[2];
    inputW = inputDims.d[3];
    outputSize = outputDims.d[1];

    // Specify the optimization profile`
    optProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, inputC, inputH, inputW));
    optProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1, inputC, inputH, inputW));
    optProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(1, inputC, inputH, inputW));
    config->addOptimizationProfile(optProfile);

    std::unique_ptr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan)
    {
        std::cerr << "Couldn't buildSerializedNetwork" << std::endl;
        return false;
    }

    mRuntime = std::shared_ptr<nvinfer1::IRuntime>(createInferRuntime(gLogger));
    if (!mRuntime)
    {
        std::cerr << "Couldn't createInferRuntime" << std::endl;
        return false;
    }

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        mRuntime->deserializeCudaEngine(plan->data(), plan->size()), InferDeleter());
    if (!mEngine)
    {
        std::cerr << "Couldn't deserializeCudaEngine" << std::endl;
        return false;
    }

    std::ofstream outfile(enginePath, std::ofstream::binary);
    outfile.write(reinterpret_cast<const char *>(plan->data()), plan->size());
    std::cout << "Successfully saved engine to " << enginePath << std::endl;

    return true;
}


bool TensorrtEngine::load(const std::string enginePath)
{
	std::cout << enginePath << std::endl;

    std::ifstream file(enginePath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Error, unable to open engine file from " << enginePath << std::endl;
        return false;
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    if (size <= 0) {
        std::cerr << "Error, invalid engine file size for " << enginePath << std::endl;
        return false;
    }

    std::vector<char> engineBuffer(size);
    if (!file.read(engineBuffer.data(), size)) {
        std::cout << "Error, unable to read engine file from " << enginePath << std::endl;
        return false;
    }

    mRuntime = std::shared_ptr<nvinfer1::IRuntime>{nvinfer1::createInferRuntime(gLogger)};
    if (!mRuntime) {
        std::cerr << "Error, failed to create inference runtime." << std::endl;
        return false;
    }
    if (engineBuffer.size() != size) {
        std::cerr << "Error, incomplete read of engine data" << std::endl;
        return false;
    }
    std::cerr << engineBuffer.data() << std::endl;
    std::cerr << engineBuffer.size() << std::endl;
	mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(mRuntime->deserializeCudaEngine(engineBuffer.data(), engineBuffer.size()));
	
	int numIOTensors = mEngine->getNbIOTensors();
	std::cout << "Number of IO Tensors: " << numIOTensors << std::endl;

    inputName = mEngine->getIOTensorName(0);
    outputName = mEngine->getIOTensorName(1);
    const auto inputDims = mEngine->getTensorShape(inputName);
    const auto outputDims = mEngine->getTensorShape(outputName);

	std::cout << inputName << " " << outputName << std::endl;

    inputC = inputDims.d[1];
    inputH = inputDims.d[2];
    inputW = inputDims.d[3];
    outputSize = outputDims.d[1];

    return true;
}

bool TensorrtEngine::constructNetwork(std::unique_ptr<nvinfer1::IBuilder>& builder,
    std::unique_ptr<nvinfer1::INetworkDefinition>& network, std::unique_ptr<nvinfer1::IBuilderConfig>& config,
    std::unique_ptr<nvonnxparser::IParser>& parser, std::string onnxFileName, std::string dataDirs)
{
    const std::string onnxFilePath = dataDirs+ "/" + onnxFileName;
	std::cout << "ONNX file: " << onnxFilePath << std::endl;
    auto parsed = parser->parseFromFile(onnxFilePath.c_str(),0);
    if (!parsed)
    {
        return false;
    }

    config->setFlag(BuilderFlag::kFP16);

    // enableDLA(builder.get(), config.get(), -1);

    return true;
}

bool TensorrtEngine::initializeBuffer(){
  	 // Create RAII buffer manager object
    //buffers = std::shared_ptr<BufferManager>(mEngine);
    // buffers = std::make_shared<BufferManager>(mEngine);
    buffers = new BufferManager(mEngine);

    context = std::shared_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    for (int32_t i = 0, e = mEngine->getNbIOTensors(); i < e; i++)
    {
        auto const name = mEngine->getIOTensorName(i);
        context->setTensorAddress(name, buffers->getDeviceBuffer(name));
    }
	return true;
}

bool TensorrtEngine::infer(float* inputData)
{
    float* hostDataBuffer = static_cast<float*>(buffers->getHostBuffer(inputName));
    for (int i = 0; i < inputC * inputH * inputW; i++)
    {
        hostDataBuffer[i] = inputData[i];
    }

	float test = 0;
	for (int i = 0; i < (inputC * inputH * inputW); i++){
        test += inputData[i] / 10000.0;
    }
	
	float test2 = 0;
	for (int i = 0; i < (inputC * inputH * inputW); i++){
        test2 += hostDataBuffer[i] / 10000.0;
    }
	std::cout << "test: " << test << " " << "test2: "<< test2 << std::endl;
	
	// Memcpy from host input buffers to device input buffers
    buffers->copyInputToDevice();

    // Propagate through network
    bool status = context->executeV2(buffers->getDeviceBindings().data());
    if (!status)
    {
        return false;
    }

	// Memcpy from device output buffers to host output buffers
	buffers->copyOutputToHost();

    return true;
}

float* TensorrtEngine::getOutput()
{
    float* output = static_cast<float*>(buffers->getHostBuffer(outputName));
    return output;
}
