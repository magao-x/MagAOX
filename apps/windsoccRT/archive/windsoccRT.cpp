/** \file windsoccRT.cpp
  * \brief The MagAO-X windsoccRT main program source file (embedded Python, batch worker, optional debug trace).
  *
  * \ingroup windsoccRT_files
  */

#include "windsoccRT.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <sys/syscall.h>
#include <unistd.h>

namespace MagAOX
{
namespace app
{

windsoccRT::windsoccRT() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   return;
}

void windsoccRT::traceDebug(const std::string &msg)
{
   if(!m_debugTrace)
   {
      return;
   }

   // LOG_NOTICE (5) is stored when default logger.logLevel is LOG_INFO (6): logManager rejects levels with
   // level > m_logLevel. LOG_DEBUG (7) was therefore dropped before; NOTICE is not.
   log<text_log>("windsoccRT trace: " + msg, logPrio::LOG_NOTICE);
}

void windsoccRT::setupConfig()
{
   SHMIMMONITOR_SETUP_CONFIG(config);

   config.add("windsocc.batchFrames",
              "",
              "windsocc.batchFrames",
              argType::Required,
              "windsocc",
              "batchFrames",
              false,
              "size_t",
              "Number of camwfs frames to collect before handing a batch to Python.");

   config.add("windsocc.pythonImportRoot",
              "",
              "windsocc.pythonImportRoot",
              argType::Required,
              "windsocc",
              "pythonImportRoot",
              false,
              "string",
              "Path that should be prepended to sys.path before importing windsocc.");

   config.add("windsocc.pythonModule",
              "",
              "windsocc.pythonModule",
              argType::Required,
              "windsocc",
              "pythonModule",
              false,
              "string",
              "Python module containing the realtime batch callable.");

   config.add("windsocc.pythonCallable",
              "",
              "windsocc.pythonCallable",
              argType::Required,
              "windsocc",
              "pythonCallable",
              false,
              "string",
              "Callable inside the Python module that accepts an embedded batch.");

   config.add("windsocc.configPath",
              "",
              "windsocc.configPath",
              argType::Required,
              "windsocc",
              "configPath",
              false,
              "string",
              "Path to the ws_config.yaml file used by realtime windsocc.");

   config.add("windsocc.outputRoot",
              "",
              "windsocc.outputRoot",
              argType::Required,
              "windsocc",
              "outputRoot",
              false,
              "string",
              "Directory where realtime camwfs_<timestamp> batches should be written.");

   config.add("windsocc.framesPerCube",
              "",
              "windsocc.framesPerCube",
              argType::Required,
              "windsocc",
              "framesPerCube",
              false,
              "size_t",
              "Number of frames to pack into each FITS cube written by Python.");

   config.add("windsocc.noMovie",
              "",
              "windsocc.noMovie",
              argType::Required,
              "windsocc",
              "noMovie",
              false,
              "bool",
              "Disable movie generation for realtime test runs.");

   config.add("windsocc.saveDistillPNGs",
              "",
              "windsocc.saveDistillPNGs",
              argType::Required,
              "windsocc",
              "saveDistillPNGs",
              false,
              "bool",
              "Keep distill PNG products instead of skipping them for latency.");

   config.add("windsocc.cleanupIntermediate",
              "",
              "windsocc.cleanupIntermediate",
              argType::Required,
              "windsocc",
              "cleanupIntermediate",
              false,
              "bool",
              "Delete heavier intermediate products after the Python pipeline completes.");

   config.add("windsocc.workerThreadPrio",
              "",
              "windsocc.workerThreadPrio",
              argType::Required,
              "windsocc",
              "workerThreadPrio",
              false,
              "int",
              "Priority of the windsocc batch worker thread.");

   config.add("windsocc.workerThreadCpuset",
              "",
              "windsocc.workerThreadCpuset",
              argType::Required,
              "windsocc",
              "workerThreadCpuset",
              false,
              "string",
              "Cpuset assigned to the windsocc batch worker thread.");

   // Trace breadcrumbs at LOG_NOTICE (stored with default logger.logLevel=INFO). If aborts occur inside the shmim
   // monitor thread or inside Python/BLAS without returning here, only the last milestone line may appear; combine
   // with e.g. OMP_NUM_THREADS=1 for narrowing.
   config.add("windsocc.debugTrace",
              "",
              "windsocc.debugTrace",
              argType::Required,
              "windsocc",
              "debugTrace",
              false,
              "bool",
              "Enable trace lines (LOG_NOTICE) for embedded Python and batch worker startup. Default is false.");

   config.add("windsocc.debugTraceLoggerDebug",
              "",
              "windsocc.debugTraceLoggerDebug",
              argType::Required,
              "windsocc",
              "debugTraceLoggerDebug",
              false,
              "bool",
              "When true with windsocc.debugTrace, set process minimum log level to DEBUG after config load "
              "(records LOG_DEBUG from all components; overrides logger.logLevel for this run). Default is false.");

   config.add("windsocc.importBeforeShmim",
              "",
              "windsocc.importBeforeShmim",
              argType::Required,
              "windsocc",
              "importBeforeShmim",
              false,
              "bool",
              "When true for debugging, initialize the embedded Python bridge before shmim startup so import-time "
              "crashes can be tested without the shmim thread already running. Default is false.");
}

int windsoccRT::loadConfigImpl(mx::app::appConfigurator &_config)
{
   SHMIMMONITOR_LOAD_CONFIG(_config);

   _config(m_batchFrames, "windsocc.batchFrames");
   _config(m_pythonImportRoot, "windsocc.pythonImportRoot");
   _config(m_pythonModule, "windsocc.pythonModule");
   _config(m_pythonCallable, "windsocc.pythonCallable");
   _config(m_configPath, "windsocc.configPath");
   _config(m_outputRoot, "windsocc.outputRoot");
   _config(m_framesPerCube, "windsocc.framesPerCube");
   _config(m_noMovie, "windsocc.noMovie");
   _config(m_saveDistillPNGs, "windsocc.saveDistillPNGs");
   _config(m_cleanupIntermediate, "windsocc.cleanupIntermediate");
   _config(m_workerThreadPrio, "windsocc.workerThreadPrio");
   _config(m_workerThreadCpuset, "windsocc.workerThreadCpuset");
   _config(m_debugTrace, "windsocc.debugTrace");
   _config(m_debugTraceLoggerDebug, "windsocc.debugTraceLoggerDebug");
   _config(m_importBeforeShmim, "windsocc.importBeforeShmim");

   if(m_debugTrace && m_debugTraceLoggerDebug)
   {
      m_log.logLevel(logPrio::LOG_DEBUG);
      log<text_log>("windsoccRT: minimum log level set to DEBUG (windsocc.debugTraceLoggerDebug=true); overrides "
                    "logger.logLevel for this process.",
                    logPrio::LOG_NOTICE);
   }

   if(m_debugTrace)
   {
      traceDebug("configuration loaded: windsocc.debugTrace enabled");
   }

   return 0;
}

void windsoccRT::loadConfig()
{
   loadConfigImpl(config);
}

int windsoccRT::initializePythonBridge()
{
   traceDebug("initializePythonBridge: enter");

   if(m_pythonImportRoot.empty())
   {
      log<software_error>({__FILE__, __LINE__, "windsocc.pythonImportRoot must be configured"});
      return -1;
   }

   if(m_configPath.empty())
   {
      log<software_error>({__FILE__, __LINE__, "windsocc.configPath must be configured"});
      return -1;
   }

   if(!Py_IsInitialized())
   {
      traceDebug("initializePythonBridge: calling Py_Initialize");
      Py_Initialize();
   }

   if(!Py_IsInitialized())
   {
      log<software_error>({__FILE__, __LINE__, "Failed to initialize embedded CPython"});
      return -1;
   }

   traceDebug(std::string("initializePythonBridge: Py_IsInitialized ok; version ") + Py_GetVersion());

   PyObject *sysPath = PySys_GetObject("path");
   if(sysPath == nullptr)
   {
      PyErr_Print();
      log<software_error>({__FILE__, __LINE__, "Failed to access sys.path"});
      return -1;
   }

   PyObject *importRoot = PyUnicode_FromString(m_pythonImportRoot.c_str());
   if(importRoot == nullptr)
   {
      PyErr_Print();
      log<software_error>({__FILE__, __LINE__, "Failed to create Python import-root string"});
      return -1;
   }

   if(PySequence_Contains(sysPath, importRoot) == 0)
   {
      if(PyList_Insert(sysPath, 0, importRoot) != 0)
      {
         Py_DECREF(importRoot);
         PyErr_Print();
         log<software_error>({__FILE__, __LINE__, "Failed to prepend windsocc import root to sys.path"});
         return -1;
      }
   }
   Py_DECREF(importRoot);

   traceDebug("initializePythonBridge: sys.path prepended with pythonImportRoot=" + m_pythonImportRoot);

   traceDebug("initializePythonBridge: calling ensurePythonCallable");
   if(ensurePythonCallable() < 0)
   {
      return -1;
   }

   traceDebug("initializePythonBridge: calling PyEval_SaveThread");
   m_pyMainThreadState = PyEval_SaveThread();
   m_pythonInitialized = true;

   traceDebug("initializePythonBridge: GIL released; Python bridge ready");

   return 0;
}

int windsoccRT::ensurePythonCallable()
{
   traceDebug("ensurePythonCallable: enter");

   Py_XDECREF(m_pyCallableObj);
   m_pyCallableObj = nullptr;
   Py_XDECREF(m_pyModule);
   m_pyModule = nullptr;

   traceDebug("ensurePythonCallable: importing module " + m_pythonModule);
   m_pyModule = PyImport_ImportModule(m_pythonModule.c_str());
   if(m_pyModule == nullptr)
   {
      PyErr_Print();
      log<software_error>({__FILE__, __LINE__, "Failed to import realtime Python module"});
      return -1;
   }

   traceDebug("ensurePythonCallable: resolving callable " + m_pythonCallable);
   m_pyCallableObj = PyObject_GetAttrString(m_pyModule, m_pythonCallable.c_str());
   if(m_pyCallableObj == nullptr || !PyCallable_Check(m_pyCallableObj))
   {
      PyErr_Print();
      log<software_error>({__FILE__, __LINE__, "Configured realtime Python callable is missing or not callable"});
      return -1;
   }

   traceDebug("ensurePythonCallable: ok");

   return 0;
}

void windsoccRT::shutdownPythonBridge()
{
   if(!m_pythonInitialized)
   {
      return;
   }

   PyEval_RestoreThread(m_pyMainThreadState);
   m_pyMainThreadState = nullptr;

   Py_XDECREF(m_pyCallableObj);
   m_pyCallableObj = nullptr;
   Py_XDECREF(m_pyModule);
   m_pyModule = nullptr;

   if(Py_IsInitialized())
   {
      Py_Finalize();
   }

   m_pythonInitialized = false;
}

int windsoccRT::appStartup()
{
   if(m_importBeforeShmim)
   {
      traceDebug("appStartup: importBeforeShmim enabled; initializing Python bridge before shmim startup");

      if(initializePythonBridge() < 0)
      {
         return -1;
      }
   }

   traceDebug("appStartup: before SHMIMMONITOR_APP_STARTUP");

   SHMIMMONITOR_APP_STARTUP;

   traceDebug("appStartup: after SHMIMMONITOR_APP_STARTUP");

   if(!m_importBeforeShmim)
   {
      if(initializePythonBridge() < 0)
      {
         return -1;
      }
   }
   else
   {
      traceDebug("appStartup: skipping second initializePythonBridge after shmim startup");
   }

   traceDebug("appStartup: before XWCAPP_THREAD_START (batch worker)");

   XWCAPP_THREAD_START(m_workerThread,
                       m_workerThreadInit,
                       m_workerThreadID,
                       m_workerThreadProp,
                       m_workerThreadPrio,
                       m_workerThreadCpuset,
                       "windsoccrt",
                       batchThreadStart);

   traceDebug("appStartup: after XWCAPP_THREAD_START");

   state(stateCodes::OPERATING);

   traceDebug("appStartup: state OPERATING");

   return 0;
}

int windsoccRT::appLogic()
{
   SHMIMMONITOR_APP_LOGIC;

   XWCAPP_THREAD_CHECK(m_workerThread, "windsoccrt");

   std::unique_lock<std::mutex> lock(m_indiMutex);

   SHMIMMONITOR_UPDATE_INDI;

   return 0;
}

int windsoccRT::appShutdown()
{
   { //mutex scope
      std::lock_guard<std::mutex> lock(m_workerMutex);
      m_workerWaiting = false;
      m_processReady = false;
   }
   m_workerCond.notify_all();

   XWCAPP_THREAD_STOP(m_workerThread);

   SHMIMMONITOR_APP_SHUTDOWN;

   shutdownPythonBridge();

   return 0;
}

int windsoccRT::allocate(const dev::shmimT &dummy)
{
   static_cast<void>(dummy);

   m_workerRestarting.store(true, std::memory_order_release);

   { //mutex scope
      std::unique_lock<std::mutex> lock(m_workerMutex);
      m_workerCond.wait(lock, [this]() { return m_workerWaiting || shutdown(); });

      if(shutdown())
      {
         return 0;
      }

      if(shmimMonitorT::m_width == 0 || shmimMonitorT::m_height == 0)
      {
         log<software_error>({__FILE__, __LINE__, "Input shmim dimensions are zero"});
         return -1;
      }

      m_frameWidth = shmimMonitorT::m_width;
      m_frameHeight = shmimMonitorT::m_height;
      m_framePixels = m_frameWidth * m_frameHeight;

      m_inputIsFloat = (shmimMonitorT::m_dataType == IMAGESTRUCT_FLOAT);
      if(!m_inputIsFloat)
      {
         m_pixget = getPixPointer<float>(shmimMonitorT::m_dataType);
         if(m_pixget == nullptr)
         {
            log<software_error>({__FILE__, __LINE__, "Unsupported shmim pixel type for float conversion"});
            return -1;
         }
      }

      if(m_batchFrames == 0)
      {
         log<software_error>({__FILE__, __LINE__, "windsocc.batchFrames must be > 0"});
         return -1;
      }

      m_fillBuffer.assign(m_batchFrames * m_framePixels, 0.0f);
      m_processBuffer.assign(m_batchFrames * m_framePixels, 0.0f);
      m_fillCount = 0;
      m_processFrameCount = 0;
      m_fillHasTimestamp = false;
      m_processReady = false;
      m_pythonActive = false;
      m_workerWaiting = false;
      m_batchesDropped.store(0, std::memory_order_release);

      m_loggedFirstBatch.store(false, std::memory_order_release);

      m_workerRestarting.store(false, std::memory_order_release);
   }

   m_workerCond.notify_all();

   traceDebug("allocate: shmim " + std::to_string(m_frameWidth) + "x" + std::to_string(m_frameHeight) +
              " dataType=" + std::to_string(static_cast<unsigned>(shmimMonitorT::m_dataType)) +
              " inputIsFloat=" + std::string(m_inputIsFloat ? "true" : "false"));

   return 0;
}

int windsoccRT::processImage(void *curr_src, const dev::shmimT &dummy)
{
   static_cast<void>(dummy);

   if(curr_src == nullptr || m_framePixels == 0)
   {
      return 0;
   }

   if(m_workerRestarting.load(std::memory_order_acquire))
   {
      return 0;
   }

   { //mutex scope
      std::lock_guard<std::mutex> lock(m_workerMutex);

      if(m_fillBuffer.empty())
      {
         return 0;
      }

      if(m_fillCount >= m_batchFrames)
      {
         m_batchesDropped.fetch_add(1, std::memory_order_acq_rel);
         return 0;
      }

      if(!m_fillHasTimestamp)
      {
         clock_gettime(CLOCK_REALTIME, &m_fillFirstTimestamp);
         m_fillHasTimestamp = true;
      }

      realT *dest = m_fillBuffer.data() + (m_fillCount * m_framePixels);
      if(m_inputIsFloat)
      {
         std::memcpy(dest, curr_src, m_framePixels * sizeof(realT));
      }
      else
      {
         for(size_t n = 0; n < m_framePixels; ++n)
         {
            dest[n] = m_pixget(curr_src, n);
         }
      }

      ++m_fillCount;

      if(m_fillCount == m_batchFrames)
      {
         if(m_processReady || m_pythonActive)
         {
            m_batchesDropped.fetch_add(1, std::memory_order_acq_rel);
            m_fillCount = 0;
            m_fillHasTimestamp = false;
            return 0;
         }

         std::swap(m_fillBuffer, m_processBuffer);
         m_processFrameCount = m_batchFrames;
         m_processFirstTimestamp = m_fillFirstTimestamp;
         m_processReady = true;
         m_fillCount = 0;
         m_fillHasTimestamp = false;
         m_workerCond.notify_all();
      }
   }

   return 0;
}

void windsoccRT::batchThreadStart(windsoccRT *p)
{
   p->batchThreadExec();
}

std::string windsoccRT::formatTimestamp(const timespec &ts) const
{
   struct tm tmUtc;
   gmtime_r(&ts.tv_sec, &tmUtc);

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
                 ts.tv_nsec / 1000);
   return std::string(buffer);
}

std::string windsoccRT::describeBatchCall(size_t frameCount, const std::string &firstTimestamp, size_t byteCount) const
{
   return "frames=" + std::to_string(frameCount) + " dims=" + std::to_string(m_frameHeight) + "x" +
          std::to_string(m_frameWidth) + " pixels=" + std::to_string(m_framePixels) +
          " bytes=" + std::to_string(byteCount) + " firstTimestamp=" + firstTimestamp +
          " configPath=" + m_configPath + " outputRoot=" + m_outputRoot +
          " framesPerCube=" + std::to_string(m_framesPerCube) +
          " noMovie=" + std::string(m_noMovie ? "true" : "false") +
          " saveDistillPNGs=" + std::string(m_saveDistillPNGs ? "true" : "false") +
          " cleanupIntermediate=" + std::string(m_cleanupIntermediate ? "true" : "false");
}

int windsoccRT::runPythonBatch(const realT *batchData, size_t frameCount, const std::string &firstTimestamp)
{
   if(!m_pythonInitialized || m_pyCallableObj == nullptr)
   {
      log<software_error>({__FILE__, __LINE__, "Python bridge is not initialized"});
      return -1;
   }

   traceDebug("runPythonBatch: enter frameCount=" + std::to_string(frameCount) + " firstTimestamp=" + firstTimestamp);

   traceDebug("runPythonBatch: before PyGILState_Ensure");
   const PyGILState_STATE gilState = PyGILState_Ensure();
   traceDebug("runPythonBatch: after PyGILState_Ensure");

   const Py_ssize_t byteCount = static_cast<Py_ssize_t>(frameCount * m_framePixels * sizeof(realT));
   PyObject *bufferView =
      PyMemoryView_FromMemory(reinterpret_cast<char *>(const_cast<realT *>(batchData)), byteCount, PyBUF_READ);
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
   PyTuple_SET_ITEM(args, 1, PyLong_FromSize_t(frameCount));
   PyTuple_SET_ITEM(args, 2, PyUnicode_FromString(firstTimestamp.c_str()));
   PyTuple_SET_ITEM(args, 3, PyUnicode_FromString(m_configPath.c_str()));
   PyTuple_SET_ITEM(args, 4, PyUnicode_FromString(m_outputRoot.c_str()));

   if(PyTuple_GetItem(args, 1) == nullptr || PyTuple_GetItem(args, 2) == nullptr || PyTuple_GetItem(args, 3) == nullptr ||
      PyTuple_GetItem(args, 4) == nullptr)
   {
      PyErr_Print();
      goto cleanup;
   }

   {
      PyObject *value = PyLong_FromSize_t(m_frameHeight);
      if(value == nullptr || PyDict_SetItemString(kwargs, "frame_height", value) != 0)
      {
         Py_XDECREF(value);
         PyErr_Print();
         goto cleanup;
      }
      Py_DECREF(value);
   }
   {
      PyObject *value = PyLong_FromSize_t(m_frameWidth);
      if(value == nullptr || PyDict_SetItemString(kwargs, "frame_width", value) != 0)
      {
         Py_XDECREF(value);
         PyErr_Print();
         goto cleanup;
      }
      Py_DECREF(value);
   }
   {
      PyObject *value = PyLong_FromSize_t(m_framesPerCube);
      if(value == nullptr || PyDict_SetItemString(kwargs, "frames_per_cube", value) != 0)
      {
         Py_XDECREF(value);
         PyErr_Print();
         goto cleanup;
      }
      Py_DECREF(value);
   }
   {
      PyObject *value = PyBool_FromLong(m_noMovie ? 1 : 0);
      if(value == nullptr || PyDict_SetItemString(kwargs, "no_movie", value) != 0)
      {
         Py_XDECREF(value);
         PyErr_Print();
         goto cleanup;
      }
      Py_DECREF(value);
   }
   {
      PyObject *value = PyBool_FromLong(m_saveDistillPNGs ? 1 : 0);
      if(value == nullptr || PyDict_SetItemString(kwargs, "save_distill_pngs", value) != 0)
      {
         Py_XDECREF(value);
         PyErr_Print();
         goto cleanup;
      }
      Py_DECREF(value);
   }
   {
      PyObject *value = PyBool_FromLong(m_cleanupIntermediate ? 1 : 0);
      if(value == nullptr || PyDict_SetItemString(kwargs, "cleanup_intermediate", value) != 0)
      {
         Py_XDECREF(value);
         PyErr_Print();
         goto cleanup;
      }
      Py_DECREF(value);
   }

   traceDebug("runPythonBatch: args/kwargs prepared " +
              describeBatchCall(frameCount, firstTimestamp, static_cast<size_t>(byteCount)));

   traceDebug("runPythonBatch: before PyObject_Call");
   result = PyObject_Call(m_pyCallableObj, args, kwargs);
   if(result == nullptr)
   {
      PyErr_Print();
      traceDebug("runPythonBatch: PyObject_Call returned nullptr");
      goto cleanup;
   }

   traceDebug("runPythonBatch: after PyObject_Call");

   if(PyDict_Check(result))
   {
      PyObject *runDir = PyDict_GetItemString(result, "run_dir");
      if(runDir != nullptr)
      {
         const char *runDirUtf8 = PyUnicode_AsUTF8(runDir);
         if(runDirUtf8 != nullptr)
         {
            log<text_log>("windsoccRT batch wrote " + std::string(runDirUtf8), logPrio::LOG_NOTICE);
         }
         else
         {
            PyErr_Print();
         }
      }
   }

   traceDebug("runPythonBatch: PyObject_Call returned ok");

   status = 0;

cleanup:
   if(status != 0)
   {
      traceDebug("runPythonBatch: cleanup after failure");
   }
   else
   {
      traceDebug("runPythonBatch: cleanup after success");
   }

   Py_XDECREF(result);
   Py_XDECREF(kwargs);
   Py_XDECREF(args);

   traceDebug("runPythonBatch: before PyGILState_Release");
   PyGILState_Release(gilState);
   traceDebug("runPythonBatch: after PyGILState_Release");

   return status;
}

void windsoccRT::batchThreadExec()
{
   m_workerThreadID = syscall(SYS_gettid);

   traceDebug("batchThreadExec: thread started tid=" + std::to_string(static_cast<long long>(m_workerThreadID)));

   while(m_workerThreadInit == true && shutdown() == 0)
   {
      sleep(1);
   }

   traceDebug("batchThreadExec: worker init gate cleared; entering batch wait loop");

   while(shutdown() == 0)
   {
      const realT *batchData = nullptr;
      size_t frameCount = 0;
      std::string firstTimestamp;

      { //mutex scope
         std::unique_lock<std::mutex> lock(m_workerMutex);

         if(m_workerRestarting.load(std::memory_order_acquire) == true || m_fillBuffer.empty())
         {
            m_workerWaiting = true;
            m_workerCond.notify_all();
         }

         m_workerCond.wait(lock, [this]()
                           { return shutdown() || (m_workerRestarting.load(std::memory_order_acquire) == false &&
                                                   !m_fillBuffer.empty() && m_processReady); });

         m_workerWaiting = false;

         if(shutdown())
         {
            break;
         }

         if(m_workerRestarting.load(std::memory_order_acquire) || !m_processReady)
         {
            continue;
         }

         m_pythonActive = true;
         m_processReady = false;
         frameCount = m_processFrameCount;
         batchData = m_processBuffer.data();
         firstTimestamp = formatTimestamp(m_processFirstTimestamp);
      }

      {
         bool expected = false;
         if(m_debugTrace && m_loggedFirstBatch.compare_exchange_strong(expected, true))
         {
            traceDebug("batchThreadExec: first batch dequeued frames=" + std::to_string(frameCount) + " ts=" +
                       firstTimestamp);
         }
      }

      traceDebug("batchThreadExec: invoking runPythonBatch frames=" + std::to_string(frameCount) + " ts=" +
                 firstTimestamp);

      const double t0 = mx::sys::get_curr_time();
      int pythonStatus = runPythonBatch(batchData, frameCount, firstTimestamp);
      const double t1 = mx::sys::get_curr_time();
      m_lastPythonLatencySec.store(t1 - t0, std::memory_order_release);

      traceDebug("batchThreadExec: runPythonBatch returned status=" + std::to_string(pythonStatus) +
                 " latencySec=" + std::to_string(t1 - t0));

      { //mutex scope
         std::lock_guard<std::mutex> lock(m_workerMutex);
         m_pythonActive = false;
      }

      if(pythonStatus == 0)
      {
         m_batchesProcessed.fetch_add(1, std::memory_order_acq_rel);
      }
      else
      {
         log<software_error>({__FILE__, __LINE__, "Embedded Python batch execution failed"});
      }
   }
}

} // namespace app
} // namespace MagAOX

/// \brief Entry point for the windsoccRT MagAO-X application.
int main(int argc, char **argv)
{
   MagAOX::app::windsoccRT xapp;

   return xapp.main(argc, argv);
}
