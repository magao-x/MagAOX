/** \file windsoccAppImportProbe.hpp
 * \brief The MagAO-X windsoccAppImportProbe header file.
 *
 * \ingroup windsoccRT_files
 */

#ifndef windsoccAppImportProbe_hpp
#define windsoccAppImportProbe_hpp

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

#include <Python.h>

#include <string>

namespace MagAOX
{
namespace app
{

/// Minimal MagAO-X app used to test `windsocc.realtime` import inside `MagAOXApp`.
/**
 * \ingroup windsoccRT
 */
class windsoccAppImportProbe : public MagAOXApp<true>
{
protected:
   /** \name Configurable Parameters
    *@{
    */
   std::string m_pythonImportRoot; ///< Root path prepended to `sys.path` before importing the WindsoCC module.
   std::string m_pythonModule{"windsocc.realtime"}; ///< Python module imported during `appStartup()`.
   bool m_debugTrace{false}; ///< When true, emit trace breadcrumbs at `LOG_NOTICE`.
   bool m_debugTraceLoggerDebug{false}; ///< When true with `m_debugTrace`, lower process minimum log level to DEBUG.
   ///@}

   bool m_pythonInitialized{false}; ///< True once CPython has been initialized by this probe.
   PyObject *m_pyModule{nullptr}; ///< Borrowed module handle for the imported Python module.

   /// Import the configured Python module into an embedded interpreter.
   int initializePythonImport();

   /// Tear down CPython state owned by this probe.
   void shutdownPythonImport();

   /// Emit a `LOG_NOTICE` trace line when `m_debugTrace` is true.
   void traceDebug(const std::string &msg /**< [in] message text */);

public:
   /// Default c'tor.
   windsoccAppImportProbe();

   /// D'tor, declared and defined for noexcept.
   ~windsoccAppImportProbe() noexcept
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
};

} // namespace app
} // namespace MagAOX

#endif // windsoccAppImportProbe_hpp
