#include "/opt/MagAOX/vendor/TensorRT-10.0.0.6/include/NvInfer.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cuda_runtime_api.h>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <new>
#include <numeric>
#include <ratio>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <chrono>

// Utility Timer
template <typename Clock = std::chrono::high_resolution_clock> class Stopwatch {
    typename Clock::time_point start_point;

public:
    Stopwatch() : start_point(Clock::now()) {}

    // Returns elapsed time
    template <typename Rep = typename Clock::duration::rep, typename Units = typename Clock::duration> Rep elapsedTime() const {
        std::atomic_thread_fence(std::memory_order_relaxed);
        auto counted_time = std::chrono::duration_cast<Units>(Clock::now() - start_point).count();
        std::atomic_thread_fence(std::memory_order_relaxed);
        return static_cast<Rep>(counted_time);
    }
};



class Logger : public nvinfer1::ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // Only log messages of type INFO and above
        if (severity <= Severity::kINFO)
        {
            std::cout << msg << std::endl;
        }
    }
};

inline bool doesFileExist(const std::string &filepath) {
    std::ifstream f(filepath.c_str());
    return f.good();
}

inline void parseCSV(std::string filepath, float * fileData)
{
    // Open the CSV file
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cout << "Error opening file!" << std::endl;
    }

    std::string line;
    int i =0;
    int arrayLength = sizeof(&fileData) / sizeof(float);
    while (i<arrayLength) 
    {
        getline(file, line);
        std::stringstream ss(line);
        std::string cell;
        while (getline(ss, cell, ',')) {
            fileData[i] = std::stof(cell);
            i++;
        }
    }
    file.close();
}

//! Locate path to file, given its filename or filepath suffix and possible dirs it might lie in.
//! Function will also walk back MAX_DEPTH dirs from CWD to check for such a file path.
inline std::string locateFile(
    const std::string& filepathSuffix, const std::vector<std::string>& directories, bool reportError = true)
{
    const int MAX_DEPTH{10};
    bool found{false};
    std::string filepath;

    for (auto& dir : directories)
    {
        if (!dir.empty() && dir.back() != '/')
        {
#ifdef _MSC_VER
            filepath = dir + "\\" + filepathSuffix;
#else
            filepath = dir + "/" + filepathSuffix;
#endif
        }
        else
        {
            filepath = dir + filepathSuffix;
        }

        for (int i = 0; i < MAX_DEPTH && !found; i++)
        {
            const std::ifstream checkFile(filepath);
            found = checkFile.is_open();
            if (found)
            {
                break;
            }

            filepath = "../" + filepath; // Try again in parent dir
        }

        if (found)
        {
            break;
        }

        filepath.clear();
    }

    // Could not find the file
    if (filepath.empty())
    {
        const std::string dirList = std::accumulate(directories.begin() + 1, directories.end(), directories.front(),
            [](const std::string& a, const std::string& b) { return a + "\n\t" + b; });
        std::cout << "Could not find " << filepathSuffix << " in data directories:\n\t" << dirList << std::endl;

        if (reportError)
        {
            std::cout << "&&&& FAILED" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    return filepath;
}


inline void enableDLA(
    nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config, int useDLACore, bool allowGPUFallback = true)
{
    if (useDLACore >= 0)
    {
        if (builder->getNbDLACores() == 0)
        {
            std::cerr << "Trying to use DLA core " << useDLACore << " on a platform that doesn't have any DLA cores"
                      << std::endl;
            assert("Error: use DLA core on a platfrom that doesn't have any DLA cores" && false);
        }
        if (allowGPUFallback)
        {
            config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
        }
        if (!config->getFlag(nvinfer1::BuilderFlag::kINT8))
        {
            // User has not requested INT8 Mode.
            // By default run in FP16 mode. FP32 mode is not permitted.
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }
        config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
        config->setDLACore(useDLACore);
    }
}

struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
        delete obj;
    }
};

static auto StreamDeleter = [](cudaStream_t* pStream) {
    if (pStream)
    {
        static_cast<void>(cudaStreamDestroy(*pStream));
        delete pStream;
    }
};

std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> makeCudaStream()
{
    std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> pStream(new cudaStream_t, StreamDeleter);
    if (cudaStreamCreateWithFlags(pStream.get(), cudaStreamNonBlocking) != cudaSuccess)
    {
        pStream.reset(nullptr);
    }

    return pStream;
}
