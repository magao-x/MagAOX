/** \file windsoccShmimImportProbe.hpp
 * \brief The MagAO-X windsoccShmimImportProbe header file.
 *
 * \ingroup windsoccRT_files
 */

#ifndef windsoccShmimImportProbe_hpp
#define windsoccShmimImportProbe_hpp

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

#include <Python.h>

#include <string>

namespace MagAOX
{
namespace app
{

/// Minimal MagAO-X app plus shmimMonitor mixin used to test WindsoCC import behavior.
/**
 * \ingroup windsoccRT
 */
class windsoccShmimImportProbe : public MagAOXApp<true>, public dev::shmimMonitor<windsoccShmimImportProbe>
{
   friend class dev::shmimMonitor<windsoccShmimImportProbe>;

public:
   /// Base shmim monitor type for the probe.
   typedef dev::shmimMonitor<windsoccShmimImportProbe> shmimMonitorT;

protected:
   /** \name Configurable Parameters
    *@{
    */
   std::string m_pythonImportRoot; ///< Root path prepended to `sys.path` before importing the WindsoCC module.
   std::string m_pythonModule{"windsocc.realtime"}; ///< Python module imported during `appStartup()`.
   std::string m_pythonCallable{"run_embedded_batch_buffer"}; ///< Callable resolved inside `m_pythonModule` when enabled.
   bool m_resolveCallable{true}; ///< When true, resolve `m_pythonCallable` after importing the module.
   bool m_saveThread{false}; ///< When true, call `PyEval_SaveThread()` after the optional callable-resolution stage.
   bool m_debugTrace{false}; ///< When true, emit trace breadcrumbs at `LOG_NOTICE`.
   bool m_debugTraceLoggerDebug{false}; ///< When true with `m_debugTrace`, lower process minimum log level to DEBUG.
   ///@}

   bool m_pythonInitialized{false}; ///< True once CPython has been initialized by this probe.
   PyThreadState *m_pyMainThreadState{nullptr}; ///< Main interpreter thread state saved when `m_saveThread` is enabled.
   PyObject *m_pyModule{nullptr}; ///< Borrowed module handle for the imported Python module.
   PyObject *m_pyCallableObj{nullptr}; ///< Borrowed callable handle resolved from `m_pyModule` when enabled.

   /// Import the configured Python module into an embedded interpreter.
   int initializePythonImport();

   /// Tear down CPython state owned by this probe.
   void shutdownPythonImport();

   /// Emit a `LOG_NOTICE` trace line when `m_debugTrace` is true.
   void traceDebug(const std::string &msg /**< [in] message text */);

public:
   /// Default c'tor.
   windsoccShmimImportProbe();

   /// D'tor, declared and defined for noexcept.
   ~windsoccShmimImportProbe() noexcept
   {
   }

   /// Setup the app configuration interface.
   virtual void setupConfig();

   /// Implementation of loadConfig logic, separated for testing.
   /** This is called by loadConfig().
    */
   int loadConfigImpl(mx::app::appConfigurator &_config /**< [in] an application configuration from which to load values */);

   /// Load configuration values into member state.
   virtual void loadConfig();

   /// Startup function.
   virtual int appStartup();

   /// Implementation of the FSM for the import probe.
   /**
    * \returns 0 on no critical error
    * \returns -1 on an error requiring shutdown
    */
   virtual int appLogic();

   /// Shutdown the app.
   virtual int appShutdown();

protected:
   /// Stub allocate required by `shmimMonitor`, intentionally unused by this probe.
   int allocate(const dev::shmimT &dummy /**< [in] tag to differentiate shmimMonitor parents */);

   /// Stub image callback required by `shmimMonitor`, intentionally unused by this probe.
   int processImage(void *curr_src /**< [in] pointer to start of current frame */,
                    const dev::shmimT &dummy /**< [in] tag to differentiate shmimMonitor parents */);
};

} // namespace app
} // namespace MagAOX

#endif // windsoccShmimImportProbe_hpp
