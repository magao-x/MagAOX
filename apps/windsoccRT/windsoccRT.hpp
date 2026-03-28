/** \file windsoccRT.hpp
 * \brief The MagAO-X windsoccRT app header file (embedded Python bridge and optional debug trace).
 *
 * \ingroup windsoccRT_files
 */

#ifndef windsoccRT_hpp
#define windsoccRT_hpp

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

#include <Python.h>

#include <atomic>
#include <condition_variable>
#include <string>
#include <thread>
#include <vector>

/** \defgroup windsoccRT
 * \brief A realtime camwfs batch collector and Python handoff app for WindsoCC
 *
 * \ingroup apps
 */

/** \defgroup windsoccRT_files
 * \ingroup windsoccRT
 */

namespace MagAOX
{
namespace app
{

/// MagAO-X app that batches camwfs shmim frames and hands them to embedded Python.
/**
 * \ingroup windsoccRT
 */
class windsoccRT : public MagAOXApp<true>, public dev::shmimMonitor<windsoccRT>
{
   friend class dev::shmimMonitor<windsoccRT>;

 public:
   /// Floating-point type used for the batch buffer and Python handoff.
   typedef float realT;

   /// Base shmim monitor type for the input camwfs stream.
   typedef dev::shmimMonitor<windsoccRT> shmimMonitorT;

 protected:
   /** \name Configurable Parameters
    *@{
    */
   size_t m_batchFrames{2048}; ///< Number of shmim frames to collect before running one Python batch.
   std::string m_pythonImportRoot; ///< Root path prepended to `sys.path` before importing the windsocc module.
   std::string m_pythonModule{"windsocc.realtime"}; ///< Python module that exposes the embedded batch callable.
   std::string m_pythonCallable{"run_embedded_batch_buffer"}; ///< Callable name inside `m_pythonModule`.
   std::string m_configPath; ///< Path to the realtime `ws_config.yaml` file passed into Python.
   std::string m_outputRoot{"."}; ///< Output root where Python writes `camwfs_<timestamp>` batch directories.
   size_t m_framesPerCube{512}; ///< Number of frames to pack into each FITS cube written by Python.
   bool m_noMovie{false}; ///< Disable movie generation during realtime processing when true.
   bool m_saveDistillPNGs{false}; ///< Preserve distill PNG products when true.
   bool m_cleanupIntermediate{false}; ///< Remove heavier intermediate pipeline products after the batch completes.
   int m_workerThreadPrio{0}; ///< Scheduling priority requested for the batch worker thread.
   std::string m_workerThreadCpuset; ///< Cpuset assigned to the batch worker thread.
   bool m_debugTrace{false}; ///< When true, emit trace breadcrumbs for embedded Python and worker startup (see `windsocc.debugTrace`).
   bool m_debugTraceLoggerDebug{false}; ///< When true with `m_debugTrace`, lower process minimum log level to DEBUG (see `windsocc.debugTraceLoggerDebug`).
   bool m_importBeforeShmim{false}; ///< When true for debugging, initialize the embedded Python bridge before shmim startup (see `windsocc.importBeforeShmim`).
   ///@}

   float (*m_pixget)(void *, size_t){nullptr}; ///< Pixel-conversion helper for non-float shmim data types.
   bool m_inputIsFloat{false}; ///< True when the input shmim already stores float pixels.
   size_t m_frameWidth{0}; ///< Current shmim frame width in pixels.
   size_t m_frameHeight{0}; ///< Current shmim frame height in pixels.
   size_t m_framePixels{0}; ///< Cached total pixel count per frame.

   std::vector<realT> m_fillBuffer; ///< Buffer filled on the shmim callback path until one batch is complete.
   std::vector<realT> m_processBuffer; ///< Completed batch buffer owned by the worker thread during Python execution.
   size_t m_fillCount{0}; ///< Number of frames currently stored in `m_fillBuffer`.
   size_t m_processFrameCount{0}; ///< Number of valid frames currently staged in `m_processBuffer`.
   bool m_fillHasTimestamp{false}; ///< True once the first-frame timestamp has been captured for the fill buffer.
   timespec m_fillFirstTimestamp{0, 0}; ///< Timestamp of the first frame in the active fill buffer.
   timespec m_processFirstTimestamp{0, 0}; ///< Timestamp of the first frame in the batch handed to Python.
   bool m_processReady{false}; ///< True when `m_processBuffer` contains a complete batch ready for Python.
   bool m_pythonActive{false}; ///< True while the worker thread is inside the embedded Python call.

   std::atomic<bool> m_workerRestarting{true}; ///< Synchronization flag set while `allocate()` is resizing state.
   bool m_workerWaiting{false}; ///< Synchronization flag protected by `m_workerMutex`.
   std::mutex m_workerMutex; ///< Protects batch buffers and worker restart handoff state.
   std::condition_variable m_workerCond; ///< Coordinates restart, wakeup, and completed-batch handoff.

   std::atomic<uint64_t> m_batchesProcessed{0}; ///< Count of batches successfully completed by embedded Python.
   std::atomic<uint64_t> m_batchesDropped{0}; ///< Count of batches dropped because the worker could not keep up.
   std::atomic<double> m_lastPythonLatencySec{0.0}; ///< Wall-clock latency of the most recent embedded Python call.
   std::atomic<bool> m_loggedFirstBatch{false}; ///< Set when the first dequeue-to-Python batch is logged under debug trace.

   pid_t m_workerThreadID{0}; ///< Linux thread ID of the batch worker thread.
   std::thread m_workerThread; ///< Dedicated worker thread that invokes the Python pipeline.
   bool m_workerThreadInit{true}; ///< Initialization gate used by the MagAO-X worker-thread macros.
   pcf::IndiProperty m_workerThreadProp; ///< INDI property published for the batch worker thread.

   bool m_pythonInitialized{false}; ///< True once CPython has been initialized for this process.
   PyThreadState *m_pyMainThreadState{nullptr}; ///< Main interpreter thread state saved after initialization.
   PyObject *m_pyModule{nullptr}; ///< Borrowed module handle for the embedded realtime Python module.
   PyObject *m_pyCallableObj{nullptr}; ///< Borrowed callable handle used for batch handoff into Python.

   /// Start trampoline for the batch worker thread.
   static void batchThreadStart(windsoccRT *p /**< [in] pointer to this app instance */);

   /// Main worker-thread loop for batch assembly and Python handoff.
   void batchThreadExec();

   /// Initialize CPython and resolve the configured realtime callable.
   int initializePythonBridge();

   /// Import the configured Python module and callable used for batch processing.
   int ensurePythonCallable();

   /// Tear down CPython state owned by this app.
   void shutdownPythonBridge();

   /// Invoke the configured realtime Python callable on one completed batch.
   int runPythonBatch(const realT *batchData /**< [in] contiguous float batch buffer */,
                      size_t frameCount /**< [in] number of frames stored in `batchData` */,
                      const std::string &firstTimestamp /**< [in] timestamp string for the first batch frame */);

   /// Format a POSIX timestamp into the string layout expected by the Python realtime layer.
   std::string formatTimestamp(const timespec &ts /**< [in] timespec to format */) const;

   /// Emit a LOG_NOTICE trace line when `m_debugTrace` is true (visible with default `logger.logLevel`).
   void traceDebug(const std::string &msg /**< [in] message text */);

 public:
   /// Default c'tor.
   windsoccRT();

   /// D'tor, declared and defined for noexcept.
   ~windsoccRT() noexcept
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

   /// Implementation of the FSM for windsoccRT.
   /**
    * \returns 0 on no critical error
    * \returns -1 on an error requiring shutdown
    */
   virtual int appLogic();

   /// Shutdown the app.
   virtual int appShutdown();

 protected:
   /// Allocate or resize state when the input shmim geometry changes.
   int allocate(const dev::shmimT &dummy /**< [in] tag to differentiate shmimMonitor parents */);

   /// Copy one shmim frame into the active batch buffer.
   int processImage(void *curr_src /**< [in] pointer to start of current frame */,
                    const dev::shmimT &dummy /**< [in] tag to differentiate shmimMonitor parents */);
};

} // namespace app
} // namespace MagAOX

#endif // windsoccRT_hpp
