
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <fstream>
#include <vector>

using namespace nvinfer1;

// Function to read the TensorRT engine file
bool readEngineFile(const std::string& filename, std::vector<char>* buffer) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening " << filename << std::endl;
        return false;
    }
    file.seekg(0, std::ios::end);
    std::vector<char> inbuf(file.tellg());
    file.seekg(0, std::ios::beg);
    file.read(inbuf.data(), inbuf.size());
    buffer = &inbuf;
    return true;
}

// Logger for TensorRT info/warning/errors
class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << msg << std::endl;
        }
    }
};

int main() {
    // Initialize Logger
    Logger logger;
    std::cerr << "logger = " << &logger << std::endl;

    // Load the TensorRT engine file
    // const std::string engineFile = "/data/users/xsup/MachineLearning/2024B_OnSky/Engines/test_model_20241114.trt";
	const std::string engineFile = "/data/users/xsup/MachineLearning/2024B_OnSky/Models/test_build_2024b";
    std::vector<char> engineData;
    // bool result = readEngineFile(engineFile, &engineData);
    // if (!result) {
    //     return 1;
    // }
    std::ifstream file(engineFile, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening " << engineFile << std::endl;
        return false;
    }
    file.seekg(0, std::ios::end);
    std::vector<char> inbuf(file.tellg());
    file.seekg(0, std::ios::beg);
    file.read(inbuf.data(), inbuf.size());
    engineData = inbuf;

    // Create the runtime and deserialize the engine
    IRuntime* runtime = createInferRuntime(logger);
    if (!runtime) {
        std::cout << "Failed to createInferRuntime\n";
    }
    ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size());
    if (!engine) {
        std::cout << "Failed to deserialize CUDA engine from " << engineFile << "\n";
        return 1;
    } else {
        std::cout << "Deserialized CUDA engine from " << engineFile << "\n";
    }
    IExecutionContext* context = engine->createExecutionContext();

    // Assume input/output shape and create buffers
    const int inputSize = 4 * 60 * 60; // for example, 224x224 RGB image
    const int outputSize = 1564; // assuming 1000 classes for a classifier

    float inputData[inputSize] = {0}; // Initialize input data as needed
    float outputData[outputSize] = {0};

    // Allocate device memory for input and output
    float* d_input = nullptr;
    float* d_output = nullptr;
    cudaMalloc((void**)&d_input, inputSize * sizeof(float));
    cudaMalloc((void**)&d_output, outputSize * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, inputData, inputSize * sizeof(float), cudaMemcpyHostToDevice);

    // Run inference
    void* buffers[] = {d_input, d_output};
    context->executeV2(buffers);

    // Copy output data back to host
    cudaMemcpy(outputData, d_output, outputSize * sizeof(float), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);
    delete context;
    delete engine;
    delete runtime;

    // Output some results
    std::cout << "Inference done! Output[0] = " << outputData[0] << std::endl;

    return 0;
}
