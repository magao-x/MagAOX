/** \file ocamDescrambleWorker.hpp
 * \brief Hardware-free pinned thread wrapper for the shared OCAM parallel descrambler.
 * \ingroup ocam2KCtrl_files
 */

#ifndef ocamDescrambleWorker_hpp
#define ocamDescrambleWorker_hpp

#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <pthread.h>
#include <sched.h>

#include "../ocamDescrambleParallel.hpp"

namespace MagAOX
{
namespace app
{

/// Own a pinned benchmark thread around the production parallel kernel and completion protocol.
/** The caller must already be pinned to its specified CPU; the harness additionally checks physical-core topology
 * and the original allowed CPU mask. Calls have one owner and must not overlap. run() completes all access to its
 * buffers before returning. Destroy only after calls return. The worker starts parked; park it during reference
 * measurements and resume it before parallel measurements. Production creates its own thread through the framework.
 */
class ocamDescrambleWorker
{
  public:
    /// Output ownership supported by the shared implementation.
    using Partition = ocamDescrambleParallel::Partition;

    /// Optional kernel-only timing fields supported by the shared implementation.
    using Timings = ocamDescrambleParallel::Timings;

  private:
    ocamDescrambleParallel m_parallel; ///< Owns validated mapping and mandatory-job protocol, but no thread.

    int m_cpu; ///< Verified worker CPU; construction fails if this affinity cannot be established.

    bool m_initialized{ false }; ///< Startup affinity result is available under m_mutex.

    bool m_affinityValid{ false }; ///< Startup affinity result, published before construction returns.

    std::mutex m_mutex; ///< Protects publication of the worker's affinity result.

    std::condition_variable m_condition; ///< Acknowledges affinity setup before entering the shared worker loop.

    std::thread m_thread; ///< Owned worker, stopped and joined before any prepared state is destroyed.

    /// Establish and verify affinity, then enter the initially parked shared worker loop.
    void work();

  public:
    /// Prepare immutable mapping data and start a parked worker with verified affinity.
    /** Throws std::runtime_error for unsupported mapping, invalid CPU choice, failed affinity or worker startup.
     * Allocation and thread creation may also throw their standard exceptions.
     */
    ocamDescrambleWorker( const ocamDescrambler &helper /**< [in] Prepared structured bin2 production helper. */,
                          int callerCpu /**< [in] Sole CPU in the calling thread's current affinity. */,
                          int workerCpu /**< [in] Different logical CPU for the benchmark worker. */ );

    /// Stop and join the worker after all owner-thread calls have returned.
    ~ocamDescrambleWorker();

    /// Wake the verified worker and wait until it is ready to receive timed jobs.
    void resume();

    /// Acknowledge worker parking before reference measurements begin.
    void park();

    /// Complete one parallel job without allocating or publishing a stream.
    /** Requires resume() and valid nonoverlapping buffers unless handoffOnly is true. Throws std::logic_error for
     * refusal or worker failure, after the shared protocol has made buffer reuse safe.
     */
    void run( short       *dest /**< [out] At least 14,400 output samples; unused for handoff-only calibration. */,
              const short *raw /**< [in] At least 32,736 raw samples; unused for handoff-only calibration. */,
              Partition    partition /**< [in] Caller/worker row ownership. */,
              bool         handoffOnly = false /**< [in] Measure dispatch/completion without processing pixels. */,
              Timings *timings = nullptr /**< [out] Optional kernel diagnostics; null disables timestamp calls. */ );

    /// Run the shared row kernel on the caller alone for comparison with the single-core production helper.
    void serial( short       *dest /**< [out] At least 14,400 output samples. */,
                 const short *raw /**< [in] At least 32,736 immutable raw samples. */,
                 bool paired = false /**< [in] Process complementary row pairs instead of consecutive rows. */ ) const;

    /// Return the worker's verified logical CPU.
    int cpu() const;
};

inline ocamDescrambleWorker::ocamDescrambleWorker( const ocamDescrambler &helper, int callerCpu, int workerCpu )
    : m_cpu( workerCpu )
{
    cpu_set_t affinity;
    CPU_ZERO( &affinity );

    if( callerCpu < 0 || workerCpu < 0 || callerCpu >= CPU_SETSIZE || workerCpu >= CPU_SETSIZE ||
        callerCpu == workerCpu || pthread_getaffinity_np( pthread_self(), sizeof( affinity ), &affinity ) != 0 ||
        CPU_COUNT( &affinity ) != 1 || !CPU_ISSET( callerCpu, &affinity ) )
    {
        throw std::runtime_error( "benchmark caller must be pinned to a valid CPU distinct from the worker" );
    }

    if( m_parallel.prepare( helper ) < 0 )
    {
        throw std::runtime_error( "parallel benchmark requires a prepared structured bin2 mapping" );
    }

    m_thread = std::thread( &ocamDescrambleWorker::work, this );

    { //mutex scope
        std::unique_lock<std::mutex> lock( m_mutex );
        m_condition.wait( lock,
                          [this]
                          {
                              return m_initialized;
                          } );
    }

    if( !m_affinityValid )
    {
        m_thread.join();
        throw std::runtime_error( "could not establish and verify benchmark worker affinity" );
    }

    if( !m_parallel.waitUntilStarted() )
    {
        m_parallel.requestStop();
        m_thread.join();
        throw std::runtime_error( "benchmark worker did not acknowledge startup" );
    }
}

inline ocamDescrambleWorker::~ocamDescrambleWorker()
{
    m_parallel.requestStop();
    m_thread.join();
}

inline void ocamDescrambleWorker::resume()
{
    if( m_parallel.resume() < 0 )
    {
        throw std::logic_error( "parallel benchmark worker could not resume" );
    }
}

inline void ocamDescrambleWorker::park()
{
    m_parallel.park();
}

inline void
ocamDescrambleWorker::run( short *dest, const short *raw, Partition partition, bool handoffOnly, Timings *timings )
{
    if( m_parallel.run( dest, raw, partition, handoffOnly, timings ) < 0 )
    {
        throw std::logic_error( "parallel benchmark requires an active worker and valid partition" );
    }
}

inline void ocamDescrambleWorker::serial( short *dest, const short *raw, bool paired ) const
{
    m_parallel.serial( dest, raw, paired );
}

inline int ocamDescrambleWorker::cpu() const
{
    return m_cpu;
}

inline void ocamDescrambleWorker::work()
{
    cpu_set_t affinity;
    CPU_ZERO( &affinity );
    CPU_SET( m_cpu, &affinity );
    const int setResult = pthread_setaffinity_np( pthread_self(), sizeof( affinity ), &affinity );
    CPU_ZERO( &affinity );
    const int getResult = pthread_getaffinity_np( pthread_self(), sizeof( affinity ), &affinity );

    { //mutex scope
        std::lock_guard<std::mutex> lock( m_mutex );
        m_affinityValid =
            setResult == 0 && getResult == 0 && CPU_COUNT( &affinity ) == 1 && CPU_ISSET( m_cpu, &affinity );
        m_initialized = true;
    }

    m_condition.notify_all();

    if( m_affinityValid )
    {
        m_parallel.workerMain();
    }
}

} // namespace app
} // namespace MagAOX

#endif // ocamDescrambleWorker_hpp
