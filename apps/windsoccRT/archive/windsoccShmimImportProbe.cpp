/** \file windsoccShmimImportProbe.cpp
 * \brief The MagAO-X windsoccShmimImportProbe main program source file.
 *
 * \ingroup windsoccRT_files
 */

#include "windsoccShmimImportProbe.hpp"

namespace MagAOX
{
namespace app
{

windsoccShmimImportProbe::windsoccShmimImportProbe() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   return;
}

void windsoccShmimImportProbe::traceDebug(const std::string &msg)
{
   if(!m_debugTrace)
   {
      return;
   }

   log<text_log>("windsoccShmimImportProbe trace: " + msg, logPrio::LOG_NOTICE);
}

void windsoccShmimImportProbe::setupConfig()
{
   SHMIMMONITOR_SETUP_CONFIG(config);

   config.add("windsocc.pythonImportRoot",
              "",
              "windsocc.pythonImportRoot",
              argType::Required,
              "windsocc",
              "pythonImportRoot",
              false,
              "string",
              "Path that should be prepended to sys.path before importing the WindsoCC module.");

   config.add("windsocc.pythonModule",
              "",
              "windsocc.pythonModule",
              argType::Required,
              "windsocc",
              "pythonModule",
              false,
              "string",
              "Python module imported by the MagAO-X shmim-mixin probe.");

   config.add("windsocc.pythonCallable",
              "",
              "windsocc.pythonCallable",
              argType::Required,
              "windsocc",
              "pythonCallable",
              false,
              "string",
              "Python callable resolved from windsocc.pythonModule when windsocc.resolveCallable is true.");

   config.add("windsocc.resolveCallable",
              "",
              "windsocc.resolveCallable",
              argType::Required,
              "windsocc",
              "resolveCallable",
              false,
              "bool",
              "When true, resolve and verify windsocc.pythonCallable after module import. Default is true.");

   config.add("windsocc.saveThread",
              "",
              "windsocc.saveThread",
              argType::Required,
              "windsocc",
              "saveThread",
              false,
              "bool",
              "When true, call PyEval_SaveThread after the optional callable-resolution stage. Default is false.");

   config.add("windsocc.debugTrace",
              "",
              "windsocc.debugTrace",
              argType::Required,
              "windsocc",
              "debugTrace",
              false,
              "bool",
              "Enable trace lines (LOG_NOTICE) for the MagAO-X shmim-mixin probe. Default is false.");

   config.add("windsocc.debugTraceLoggerDebug",
              "",
              "windsocc.debugTraceLoggerDebug",
              argType::Required,
              "windsocc",
              "debugTraceLoggerDebug",
              false,
              "bool",
              "When true with windsocc.debugTrace, set process minimum log level to DEBUG after config load. "
              "Default is false.");
}

int windsoccShmimImportProbe::loadConfigImpl(mx::app::appConfigurator &_config)
{
   SHMIMMONITOR_LOAD_CONFIG(_config);

   _config(m_pythonImportRoot, "windsocc.pythonImportRoot");
   _config(m_pythonModule, "windsocc.pythonModule");
   _config(m_pythonCallable, "windsocc.pythonCallable");
   _config(m_resolveCallable, "windsocc.resolveCallable");
   _config(m_saveThread, "windsocc.saveThread");
   _config(m_debugTrace, "windsocc.debugTrace");
   _config(m_debugTraceLoggerDebug, "windsocc.debugTraceLoggerDebug");

   if(m_debugTrace && m_debugTraceLoggerDebug)
   {
      m_log.logLevel(logPrio::LOG_DEBUG);
      log<text_log>("windsoccShmimImportProbe: minimum log level set to DEBUG "
                    "(windsocc.debugTraceLoggerDebug=true); overrides logger.logLevel for this process.",
                    logPrio::LOG_NOTICE);
   }

   if(m_debugTrace)
   {
      traceDebug("configuration loaded: windsocc.debugTrace enabled");
   }

   return 0;
}

void windsoccShmimImportProbe::loadConfig()
{
   loadConfigImpl(config);
}

int windsoccShmimImportProbe::initializePythonImport()
{
   traceDebug("initializePythonImport: enter");

   if(m_pythonImportRoot.empty())
   {
      log<software_error>({__FILE__, __LINE__, "windsocc.pythonImportRoot must be configured"});
      return -1;
   }

   if(!Py_IsInitialized())
   {
      traceDebug("initializePythonImport: calling Py_Initialize");
      Py_Initialize();
   }

   if(!Py_IsInitialized())
   {
      log<software_error>({__FILE__, __LINE__, "Failed to initialize embedded CPython"});
      return -1;
   }

   traceDebug(std::string("initializePythonImport: Py_IsInitialized ok; version ") + Py_GetVersion());

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

   traceDebug("initializePythonImport: sys.path prepended with pythonImportRoot=" + m_pythonImportRoot);

   traceDebug("initializePythonImport: importing module " + m_pythonModule);
   m_pyModule = PyImport_ImportModule(m_pythonModule.c_str());
   if(m_pyModule == nullptr)
   {
      PyErr_Print();
      log<software_error>({__FILE__, __LINE__, "Failed to import configured Python module"});
      return -1;
   }

   traceDebug("initializePythonImport: module import succeeded");

   if(m_resolveCallable)
   {
      traceDebug("initializePythonImport: resolving callable " + m_pythonCallable);

      Py_XDECREF(m_pyCallableObj);
      m_pyCallableObj = nullptr;

      m_pyCallableObj = PyObject_GetAttrString(m_pyModule, m_pythonCallable.c_str());
      if(m_pyCallableObj == nullptr || !PyCallable_Check(m_pyCallableObj))
      {
         PyErr_Print();
         log<software_error>({__FILE__, __LINE__, "Configured Python callable is missing or not callable"});
         return -1;
      }

      traceDebug("initializePythonImport: callable resolution succeeded");
   }
   else
   {
      traceDebug("initializePythonImport: skipping callable resolution (windsocc.resolveCallable=false)");
   }

   if(m_saveThread)
   {
      traceDebug("initializePythonImport: calling PyEval_SaveThread");
      m_pyMainThreadState = PyEval_SaveThread();
      traceDebug("initializePythonImport: GIL released via PyEval_SaveThread");
   }
   else
   {
      traceDebug("initializePythonImport: skipping PyEval_SaveThread (windsocc.saveThread=false)");
   }

   m_pythonInitialized = true;

   return 0;
}

void windsoccShmimImportProbe::shutdownPythonImport()
{
   if(!m_pythonInitialized)
   {
      return;
   }

   if(m_pyMainThreadState != nullptr && Py_IsInitialized())
   {
      traceDebug("shutdownPythonImport: restoring main thread state");
      PyEval_RestoreThread(m_pyMainThreadState);
      m_pyMainThreadState = nullptr;
   }

   Py_XDECREF(m_pyCallableObj);
   m_pyCallableObj = nullptr;
   Py_XDECREF(m_pyModule);
   m_pyModule = nullptr;

   if(Py_IsInitialized())
   {
      traceDebug("shutdownPythonImport: calling Py_Finalize");
      Py_Finalize();
   }

   m_pythonInitialized = false;
}

int windsoccShmimImportProbe::appStartup()
{
   traceDebug("appStartup: before initializePythonImport");

   if(initializePythonImport() < 0)
   {
      return -1;
   }

   traceDebug("appStartup: import completed; requesting shutdown before shmim startup");

   m_shutdown = 1;

   return 0;
}

int windsoccShmimImportProbe::appLogic()
{
   return 0;
}

int windsoccShmimImportProbe::appShutdown()
{
   shutdownPythonImport();

   return 0;
}

int windsoccShmimImportProbe::allocate(const dev::shmimT &dummy)
{
   static_cast<void>(dummy);

   return 0;
}

int windsoccShmimImportProbe::processImage(void *curr_src, const dev::shmimT &dummy)
{
   static_cast<void>(curr_src);
   static_cast<void>(dummy);

   return 0;
}

} // namespace app
} // namespace MagAOX

/// \brief Entry point for the windsoccShmimImportProbe MagAO-X application.
int main(int argc, char **argv)
{
   MagAOX::app::windsoccShmimImportProbe xapp;

   return xapp.main(argc, argv);
}
