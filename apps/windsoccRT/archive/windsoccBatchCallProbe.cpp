/** \file windsoccBatchCallProbe.cpp
 * \brief Minimal embedded-Python batch-call probe for WindsoCC realtime execution.
 * \author Jay Kueny
 *
 * \ingroup windsoccRT_files
 */

#include <Python.h>

#include <chrono>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace
{

struct ProbeOptions
{
   std::string pythonImportRoot;
   std::string moduleName{"windsocc.realtime"};
   std::string callableName{"run_embedded_batch_buffer"};
   std::string configPath;
   std::string outputRoot{"."};
   size_t frameCount{512};
   size_t frameHeight{120};
   size_t frameWidth{120};
   size_t framesPerCube{512};
   float fillValue{1.0f};
   bool noMovie{false};
   bool saveDistillPNGs{false};
   bool cleanupIntermediate{false};
   bool spawnThread{false};
};

/// Print a brief usage message.
void printUsage(const char *argv0)
{
   std::cerr << "Usage: " << argv0
             << " --config-path PATH [--python-import-root PATH] [--module MODULE] [--callable NAME]"
             << " [--output-root PATH] [--frame-count N] [--frame-height N] [--frame-width N]"
             << " [--frames-per-cube N] [--fill-value FLOAT] [--no-movie] [--save-distill-pngs]"
             << " [--cleanup-intermediate] [--spawn-thread]" << std::endl;
}

/// Prepend a root path to Python's import path.
int prependImportRoot(const std::string &pythonImportRoot)
{
   if(pythonImportRoot.empty())
   {
      return 0;
   }

   PyObject *sysPath = PySys_GetObject("path");
   if(sysPath == nullptr)
   {
      PyErr_Print();
      std::cerr << "windsoccBatchCallProbe: failed to access sys.path" << std::endl;
      return -1;
   }

   PyObject *importRoot = PyUnicode_FromString(pythonImportRoot.c_str());
   if(importRoot == nullptr)
   {
      PyErr_Print();
      std::cerr << "windsoccBatchCallProbe: failed to create import-root string" << std::endl;
      return -1;
   }

   if(PySequence_Contains(sysPath, importRoot) == 0)
   {
      if(PyList_Insert(sysPath, 0, importRoot) != 0)
      {
         Py_DECREF(importRoot);
         PyErr_Print();
         std::cerr << "windsoccBatchCallProbe: failed to prepend import root " << pythonImportRoot << std::endl;
         return -1;
      }
   }

   Py_DECREF(importRoot);
   return 0;
}

/// Format the current UTC time like windsoccRT::formatTimestamp().
std::string formatCurrentTimestamp()
{
   const auto now = std::chrono::system_clock::now();
   const auto secs = std::chrono::time_point_cast<std::chrono::seconds>(now);
   const auto micros = std::chrono::duration_cast<std::chrono::microseconds>(now - secs).count();
   const std::time_t nowTime = std::chrono::system_clock::to_time_t(now);

   struct tm tmUtc;
   gmtime_r(&nowTime, &tmUtc);

   char buffer[64];
   std::snprintf(buffer,
                 sizeof(buffer),
                 "%04d%02d%02dT%02d%02d%02d%06ld",
                 tmUtc.tm_year + 1900,
                 tmUtc.tm_mon + 1,
                 tmUtc.tm_mday,
                 tmUtc.tm_hour,
                 tmUtc.tm_min,
                 tmUtc.tm_sec,
                 static_cast<long>(micros));
   return std::string(buffer);
}

/// Build a deterministic synthetic float batch for one probe call.
std::vector<float> makeSyntheticBatch(const ProbeOptions &options)
{
   const size_t pixelCount = options.frameCount * options.frameHeight * options.frameWidth;
   std::vector<float> batch(pixelCount, options.fillValue);

   for(size_t n = 0; n < pixelCount; ++n)
   {
      batch[n] = options.fillValue + static_cast<float>(n % 1024) / 1024.0f;
   }

   return batch;
}

/// Invoke the configured Python callable using the same ABI as windsoccRT::runPythonBatch().
int invokeBatchCallable(PyObject *callableObj,
                        const ProbeOptions &options,
                        const std::vector<float> &batchData,
                        const std::string &firstTimestamp)
{
   std::cerr << "windsoccBatchCallProbe: before PyGILState_Ensure" << std::endl;
   const PyGILState_STATE gilState = PyGILState_Ensure();
   std::cerr << "windsoccBatchCallProbe: after PyGILState_Ensure" << std::endl;

   const Py_ssize_t byteCount = static_cast<Py_ssize_t>(batchData.size() * sizeof(float));
   PyObject *bufferView =
      PyMemoryView_FromMemory(reinterpret_cast<char *>(const_cast<float *>(batchData.data())), byteCount, PyBUF_READ);
   PyObject *args = PyTuple_New(5);
   PyObject *kwargs = PyDict_New();
   PyObject *result = nullptr;
   int status = -1;

   if(bufferView == nullptr || args == nullptr || kwargs == nullptr)
   {
      PyErr_Print();
      goto cleanup;
   }

   PyTuple_SET_ITEM(args, 0, bufferView);
   PyTuple_SET_ITEM(args, 1, PyLong_FromSize_t(options.frameCount));
   PyTuple_SET_ITEM(args, 2, PyUnicode_FromString(firstTimestamp.c_str()));
   PyTuple_SET_ITEM(args, 3, PyUnicode_FromString(options.configPath.c_str()));
   PyTuple_SET_ITEM(args, 4, PyUnicode_FromString(options.outputRoot.c_str()));

   if(PyTuple_GetItem(args, 1) == nullptr || PyTuple_GetItem(args, 2) == nullptr || PyTuple_GetItem(args, 3) == nullptr ||
      PyTuple_GetItem(args, 4) == nullptr)
   {
      PyErr_Print();
      goto cleanup;
   }

   {
      PyObject *value = PyLong_FromSize_t(options.frameHeight);
      if(value == nullptr || PyDict_SetItemString(kwargs, "frame_height", value) != 0)
      {
         Py_XDECREF(value);
         PyErr_Print();
         goto cleanup;
      }
      Py_DECREF(value);
   }
   {
      PyObject *value = PyLong_FromSize_t(options.frameWidth);
      if(value == nullptr || PyDict_SetItemString(kwargs, "frame_width", value) != 0)
      {
         Py_XDECREF(value);
         PyErr_Print();
         goto cleanup;
      }
      Py_DECREF(value);
   }
   {
      PyObject *value = PyLong_FromSize_t(options.framesPerCube);
      if(value == nullptr || PyDict_SetItemString(kwargs, "frames_per_cube", value) != 0)
      {
         Py_XDECREF(value);
         PyErr_Print();
         goto cleanup;
      }
      Py_DECREF(value);
   }
   {
      PyObject *value = PyBool_FromLong(options.noMovie ? 1 : 0);
      if(value == nullptr || PyDict_SetItemString(kwargs, "no_movie", value) != 0)
      {
         Py_XDECREF(value);
         PyErr_Print();
         goto cleanup;
      }
      Py_DECREF(value);
   }
   {
      PyObject *value = PyBool_FromLong(options.saveDistillPNGs ? 1 : 0);
      if(value == nullptr || PyDict_SetItemString(kwargs, "save_distill_pngs", value) != 0)
      {
         Py_XDECREF(value);
         PyErr_Print();
         goto cleanup;
      }
      Py_DECREF(value);
   }
   {
      PyObject *value = PyBool_FromLong(options.cleanupIntermediate ? 1 : 0);
      if(value == nullptr || PyDict_SetItemString(kwargs, "cleanup_intermediate", value) != 0)
      {
         Py_XDECREF(value);
         PyErr_Print();
         goto cleanup;
      }
      Py_DECREF(value);
   }

   std::cerr << "windsoccBatchCallProbe: args/kwargs prepared frames=" << options.frameCount
             << " dims=" << options.frameHeight << 'x' << options.frameWidth << " bytes=" << byteCount
             << " firstTimestamp=" << firstTimestamp << " configPath=" << options.configPath
             << " outputRoot=" << options.outputRoot << " framesPerCube=" << options.framesPerCube
             << " noMovie=" << (options.noMovie ? "true" : "false")
             << " saveDistillPNGs=" << (options.saveDistillPNGs ? "true" : "false")
             << " cleanupIntermediate=" << (options.cleanupIntermediate ? "true" : "false") << std::endl;

   std::cerr << "windsoccBatchCallProbe: before PyObject_Call" << std::endl;
   result = PyObject_Call(callableObj, args, kwargs);
   if(result == nullptr)
   {
      PyErr_Print();
      std::cerr << "windsoccBatchCallProbe: PyObject_Call returned nullptr" << std::endl;
      goto cleanup;
   }
   std::cerr << "windsoccBatchCallProbe: after PyObject_Call" << std::endl;

   if(PyDict_Check(result))
   {
      PyObject *runDir = PyDict_GetItemString(result, "run_dir");
      if(runDir != nullptr)
      {
         const char *runDirUtf8 = PyUnicode_AsUTF8(runDir);
         if(runDirUtf8 != nullptr)
         {
            std::cerr << "windsoccBatchCallProbe: run_dir=" << runDirUtf8 << std::endl;
         }
         else
         {
            PyErr_Print();
         }
      }
   }

   status = 0;

cleanup:
   if(status != 0)
   {
      std::cerr << "windsoccBatchCallProbe: cleanup after failure" << std::endl;
   }
   else
   {
      std::cerr << "windsoccBatchCallProbe: cleanup after success" << std::endl;
   }

   Py_XDECREF(result);
   Py_XDECREF(kwargs);
   Py_XDECREF(args);

   std::cerr << "windsoccBatchCallProbe: before PyGILState_Release" << std::endl;
   PyGILState_Release(gilState);
   std::cerr << "windsoccBatchCallProbe: after PyGILState_Release" << std::endl;

   return status;
}

} // namespace

/// Entry point for the embedded-Python batch-call probe.
int main(int argc, char **argv)
{
   ProbeOptions options;

   for(int n = 1; n < argc; ++n)
   {
      std::string arg{argv[n]};
      if(arg == "--python-import-root" && n + 1 < argc)
      {
         options.pythonImportRoot = argv[++n];
      }
      else if(arg == "--module" && n + 1 < argc)
      {
         options.moduleName = argv[++n];
      }
      else if(arg == "--callable" && n + 1 < argc)
      {
         options.callableName = argv[++n];
      }
      else if(arg == "--config-path" && n + 1 < argc)
      {
         options.configPath = argv[++n];
      }
      else if(arg == "--output-root" && n + 1 < argc)
      {
         options.outputRoot = argv[++n];
      }
      else if(arg == "--frame-count" && n + 1 < argc)
      {
         options.frameCount = static_cast<size_t>(std::stoull(argv[++n]));
      }
      else if(arg == "--frame-height" && n + 1 < argc)
      {
         options.frameHeight = static_cast<size_t>(std::stoull(argv[++n]));
      }
      else if(arg == "--frame-width" && n + 1 < argc)
      {
         options.frameWidth = static_cast<size_t>(std::stoull(argv[++n]));
      }
      else if(arg == "--frames-per-cube" && n + 1 < argc)
      {
         options.framesPerCube = static_cast<size_t>(std::stoull(argv[++n]));
      }
      else if(arg == "--fill-value" && n + 1 < argc)
      {
         options.fillValue = std::stof(argv[++n]);
      }
      else if(arg == "--no-movie")
      {
         options.noMovie = true;
      }
      else if(arg == "--save-distill-pngs")
      {
         options.saveDistillPNGs = true;
      }
      else if(arg == "--cleanup-intermediate")
      {
         options.cleanupIntermediate = true;
      }
      else if(arg == "--spawn-thread")
      {
         options.spawnThread = true;
      }
      else if(arg == "-h" || arg == "--help")
      {
         printUsage(argv[0]);
         return 0;
      }
      else
      {
         std::cerr << "windsoccBatchCallProbe: unrecognized argument " << arg << std::endl;
         printUsage(argv[0]);
         return 1;
      }
   }

   if(options.configPath.empty())
   {
      std::cerr << "windsoccBatchCallProbe: --config-path is required" << std::endl;
      printUsage(argv[0]);
      return 1;
   }

   std::cerr << "windsoccBatchCallProbe: calling Py_Initialize" << std::endl;
   Py_Initialize();
   if(!Py_IsInitialized())
   {
      std::cerr << "windsoccBatchCallProbe: Py_Initialize failed" << std::endl;
      return 1;
   }

   std::cerr << "windsoccBatchCallProbe: Python version " << Py_GetVersion() << std::endl;
   if(prependImportRoot(options.pythonImportRoot) < 0)
   {
      Py_Finalize();
      return 1;
   }

   if(!options.pythonImportRoot.empty())
   {
      std::cerr << "windsoccBatchCallProbe: prepended sys.path with " << options.pythonImportRoot << std::endl;
   }

   std::cerr << "windsoccBatchCallProbe: importing module " << options.moduleName << std::endl;
   PyObject *module = PyImport_ImportModule(options.moduleName.c_str());
   if(module == nullptr)
   {
      PyErr_Print();
      Py_Finalize();
      return 1;
   }

   std::cerr << "windsoccBatchCallProbe: resolving callable " << options.callableName << std::endl;
   PyObject *callableObj = PyObject_GetAttrString(module, options.callableName.c_str());
   if(callableObj == nullptr || !PyCallable_Check(callableObj))
   {
      PyErr_Print();
      Py_XDECREF(callableObj);
      Py_DECREF(module);
      Py_Finalize();
      return 1;
   }

   std::vector<float> batchData = makeSyntheticBatch(options);
   const std::string firstTimestamp = formatCurrentTimestamp();

   std::cerr << "windsoccBatchCallProbe: built synthetic float32 batch frames=" << options.frameCount
             << " dims=" << options.frameHeight << 'x' << options.frameWidth
             << " values=" << batchData.size() << " firstTimestamp=" << firstTimestamp << std::endl;

   int result = -1;
   if(options.spawnThread)
   {
      std::cerr << "windsoccBatchCallProbe: calling PyEval_SaveThread on main thread" << std::endl;
      PyThreadState *mainThreadState = PyEval_SaveThread();

      std::thread helperThread([&result, &callableObj, &options, &batchData, &firstTimestamp]()
                               { result = invokeBatchCallable(callableObj, options, batchData, firstTimestamp); });

      std::cerr << "windsoccBatchCallProbe: helper worker thread started" << std::endl;
      helperThread.join();
      std::cerr << "windsoccBatchCallProbe: helper worker thread joined" << std::endl;

      std::cerr << "windsoccBatchCallProbe: restoring main thread state" << std::endl;
      PyEval_RestoreThread(mainThreadState);
   }
   else
   {
      result = invokeBatchCallable(callableObj, options, batchData, firstTimestamp);
   }

   Py_XDECREF(callableObj);
   Py_DECREF(module);

   std::cerr << "windsoccBatchCallProbe: calling Py_Finalize" << std::endl;
   Py_Finalize();

   return result == 0 ? 0 : 1;
}
