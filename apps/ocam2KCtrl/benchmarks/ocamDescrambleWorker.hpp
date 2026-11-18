/** \file ocamDescrambleWorker.hpp
 * \brief Hardware-free, persistent two-core OCAM descramble benchmark support.
 * \ingroup ocam2KCtrl_files
 */

#ifndef ocamDescrambleWorker_hpp
#define ocamDescrambleWorker_hpp

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <pthread.h>
#include <sched.h>

#include "../ocamDescrambler.hpp"

namespace MagAOX
{
namespace app
{

/// Run isolated descramble jobs on a caller and one persistent, explicitly pinned benchmark worker.
/** This is benchmark support, not an application worker. The caller must already be pinned to its specified CPU;
 * the harness must additionally check physical-core topology and the original allowed CPU mask. Calls have one owner
 * and must not overlap. run() completes all access to its buffers before returning. Destroy only after calls return.
 * The worker starts parked; park it during reference measurements and resume it before parallel measurements.
 */
class ocamDescrambleWorker
{
  public:
    /// Output halves or paired output rows divided by their common physical raw row in the observed map.
    enum class Partition
    {
        halves,   ///< Caller processes output rows 0–59; worker processes 60–119.
        paired28, ///< Caller processes pairs y/119-y for y below 28; worker processes the remaining 32 pairs.
        paired30  ///< Caller and worker each process 30 pairs y/119-y.
    };

    /// Optional kernel-only diagnostics; dispatch and completion waiting are excluded.
    struct Timings
    {
        uint64_t mainNs{ 0 };   ///< Caller kernel elapsed time, or zero for a handoff-only job.
        uint64_t workerNs{ 0 }; ///< Worker kernel elapsed time, or zero for a handoff-only job.
    };

  private:
    std::array<std::array<uint16_t, 4>, 120> m_bases{}; ///< Immutable source bases copied from the validated helper.
    std::array<int, 4> m_strides{}; ///< Immutable amplifier strides, retained as data like the production kernel.
    int  m_cpu; ///< Verified worker logical CPU; construction fails if this placement cannot be established.
    bool m_running{ false };         ///< Caller-owned permission to submit jobs after resume acknowledges readiness.
    bool m_initialized{ false };     ///< Startup result is available under m_mutex.
    bool m_parked{ true };           ///< Worker is blocked or entering its condition wait, protected by m_mutex.
    bool m_affinityValid{ false };   ///< Startup affinity result, published under m_mutex before construction returns.
    std::mutex              m_mutex; ///< Protects startup and park/resume acknowledgments outside timed frame work.
    std::condition_variable m_condition; ///< Blocks the worker while inactive and acknowledges control transitions.
    std::thread             m_thread;    ///< Owned worker, joined before any prepared data is destroyed.

    alignas( 64 ) std::atomic<bool> m_stop{ false }; ///< Shutdown request, separate from per-job cache lines.
    std::atomic<bool> m_pause{ true }; ///< Request the worker's blocking wait rather than an idle spinner.

    alignas( 64 ) std::atomic<uint64_t> m_job{ 0 }; ///< Release-published job generation and its following payload.
    short       *m_dest{ nullptr }; ///< Current destination, immutable until the worker acknowledges completion.
    const short *m_raw{ nullptr };  ///< Current raw frame, immutable until completion.
    Partition    m_partition{ Partition::halves }; ///< Current ownership split, published with the job generation.
    bool         m_handoffOnly{ false };           ///< Suppress both pixel kernels for dispatch/completion calibration.
    bool         m_measure{ false };               ///< Enable worker timestamps only for explicit diagnostic jobs.

    alignas( 64 ) std::atomic<uint64_t> m_done{ 0 }; ///< Release-published completion on its own cache line.
    uint64_t m_workerNs{ 0 }; ///< Optional worker duration, read only after acquiring completion.

    /// Run the pinned worker's startup, blocking control, and mandatory-job loop.
    void work();

    /// Apply the mirrored production four-amplifier loop to one output row.
    void row( short       *dest /**< [out] Complete private output image. */,
              const short *raw /**< [in] Complete immutable raw frame. */,
              size_t       index /**< [in] Output row in [0,120). */ ) const;

    /// Process a row interval, optionally following each row with its complementary row.
    void rows( short       *dest /**< [out] Complete private output image. */,
               const short *raw /**< [in] Complete immutable raw frame. */,
               size_t       first /**< [in] First output row or pair index. */,
               size_t       end /**< [in] Exclusive output row or pair bound. */,
               bool         paired /**< [in] Also process row 119-index. */ ) const;

    /// Read diagnostic monotonic elapsed time in nanoseconds.
    static uint64_t now();

    /// Hint to the CPU while polling job, completion, or shutdown state.
    static void relax();

  public:
    /// Prepare immutable mapping data and start a parked worker with verified affinity.
    /** Throws std::runtime_error for an unsupported map, invalid CPU choice, or failed affinity verification.
     * Allocation and thread creation may also throw their standard exceptions.
     */
    ocamDescrambleWorker( const ocamDescrambler &helper /**< [in] Prepared structured bin2 production helper. */,
                          int callerCpu /**< [in] Sole CPU in the calling thread's current affinity. */,
                          int workerCpu /**< [in] Different logical CPU for the benchmark worker. */ );

    /// Wake and join the worker after all owner-thread calls have returned.
    ~ocamDescrambleWorker();

    /// Wake the worker and wait until it is ready to receive timed jobs.
    void resume();

    /// Block the worker and acknowledge parking before reference measurements begin.
    void park();

    /// Complete one parallel job before returning, without allocating or publishing a stream.
    /** Requires resume() and valid, nonoverlapping buffers unless handoffOnly is true. Throws std::logic_error for
     * a parked worker or invalid partition. The owner retains both buffers until this call returns.
     */
    void run( short       *dest /**< [out] At least 14,400 output samples; unused for handoff-only calibration. */,
              const short *raw /**< [in] At least 32,736 raw samples; unused for handoff-only calibration. */,
              Partition    partition /**< [in] Caller/worker row ownership. */,
              bool         handoffOnly = false /**< [in] Measure dispatch/completion without processing pixels. */,
              Timings *timings = nullptr /**< [out] Optional kernel diagnostics; null disables timestamp calls. */ );

    /// Run the same benchmark row kernel on the caller alone for comparison with the production helper.
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

    if( !helper.ready() || !helper.structured() || helper.rawPixels() != OCAM2_PIXELS_RAW_BINNING2x2 ||
        helper.imagePixels() != OCAM2_PIXELS_IMAGE_BINNING2x2 )
    {
        throw std::runtime_error( "parallel benchmark requires a validated structured bin2 mapping" );
    }

    std::vector<short> raw( helper.rawPixels() ), map( helper.imagePixels() );

    for( size_t i = 0; i < raw.size(); ++i )
    {
        const uint16_t index = i;
        std::memcpy( &raw[i], &index, sizeof( index ) );
    }

    helper.mapped( map.data(), raw.data() );

    for( size_t index = 0; index < m_bases.size(); ++index )
    {
        for( size_t amplifier = 0; amplifier < 4; ++amplifier )
        {
            const size_t offset = index * 120 + amplifier * 30;
            std::memcpy( &m_bases[index][amplifier], &map[offset], sizeof( uint16_t ) );

            if( index == 0 )
            {
                uint16_t next;
                std::memcpy( &next, &map[offset + 1], sizeof( next ) );
                m_strides[amplifier] = int( next ) - m_bases[index][amplifier];
            }
        }
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
}

inline ocamDescrambleWorker::~ocamDescrambleWorker()
{
    { //mutex scope
        std::lock_guard<std::mutex> lock( m_mutex );
        m_stop.store( true, std::memory_order_release );
    }

    m_condition.notify_all();
    m_thread.join();
}

inline void ocamDescrambleWorker::resume()
{
    std::unique_lock<std::mutex> lock( m_mutex );
    m_pause.store( false, std::memory_order_release );
    m_condition.notify_all();
    m_condition.wait( lock,
                      [this]
                      {
                          return !m_parked;
                      } );
    m_running = true;
}

inline void ocamDescrambleWorker::park()
{
    std::unique_lock<std::mutex> lock( m_mutex );
    m_pause.store( true, std::memory_order_release );
    m_condition.wait( lock,
                      [this]
                      {
                          return m_parked;
                      } );
    m_running = false;
}

inline void
ocamDescrambleWorker::run( short *dest, const short *raw, Partition partition, bool handoffOnly, Timings *timings )
{
    if( !m_running ||
        ( partition != Partition::halves && partition != Partition::paired28 && partition != Partition::paired30 ) )
    {
        throw std::logic_error( "parallel benchmark requires an active worker and valid partition" );
    }

    m_dest             = dest;
    m_raw              = raw;
    m_partition        = partition;
    m_handoffOnly      = handoffOnly;
    m_measure          = timings != nullptr && !handoffOnly;
    const uint64_t job = m_job.load( std::memory_order_relaxed ) + 1;
    m_job.store( job, std::memory_order_release );

    const uint64_t start = m_measure ? now() : 0;

    if( !handoffOnly )
    {
        rows( dest,
              raw,
              0,
              partition == Partition::halves     ? 60
              : partition == Partition::paired28 ? 28
                                                 : 30,
              partition != Partition::halves );
    }

    const uint64_t mainNs = m_measure ? now() - start : 0;

    while( m_done.load( std::memory_order_acquire ) != job )
    {
        relax();
    }

    if( timings != nullptr )
    {
        timings->mainNs   = mainNs;
        timings->workerNs = m_measure ? m_workerNs : 0;
    }
}

inline void ocamDescrambleWorker::serial( short *dest, const short *raw, bool paired ) const
{
    rows( dest, raw, 0, paired ? 60 : 120, paired );
}

inline int ocamDescrambleWorker::cpu() const
{
    return m_cpu;
}

inline void ocamDescrambleWorker::row( short *__restrict__ dest, const short *__restrict__ raw, size_t index ) const
{
    short      *output = dest + index * 120;
    const auto &bases  = m_bases[index];

    for( int pixel = 0; pixel < 30; ++pixel )
    {
        output[pixel]      = raw[bases[0] + m_strides[0] * pixel];
        output[30 + pixel] = raw[bases[1] + m_strides[1] * pixel];
        output[60 + pixel] = raw[bases[2] + m_strides[2] * pixel];
        output[90 + pixel] = raw[bases[3] + m_strides[3] * pixel];
    }
}

inline void ocamDescrambleWorker::rows( short *dest, const short *raw, size_t first, size_t end, bool paired ) const
{
    for( size_t index = first; index < end; ++index )
    {
        row( dest, raw, index );

        if( paired )
        {
            row( dest, raw, 119 - index );
        }
    }
}

inline uint64_t ocamDescrambleWorker::now()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>( std::chrono::steady_clock::now().time_since_epoch() )
        .count();
}

inline void ocamDescrambleWorker::relax()
{
#if defined( __i386__ ) || defined( __x86_64__ )
    __builtin_ia32_pause();
#else
    std::atomic_signal_fence( std::memory_order_seq_cst );
#endif
}

inline void ocamDescrambleWorker::work()
{
    cpu_set_t affinity;
    CPU_ZERO( &affinity );
    CPU_SET( m_cpu, &affinity );
    const int setResult = pthread_setaffinity_np( pthread_self(), sizeof( affinity ), &affinity );
    CPU_ZERO( &affinity );
    const int getResult = pthread_getaffinity_np( pthread_self(), sizeof( affinity ), &affinity );

    std::unique_lock<std::mutex> lock( m_mutex );
    m_affinityValid = setResult == 0 && getResult == 0 && CPU_COUNT( &affinity ) == 1 && CPU_ISSET( m_cpu, &affinity );
    m_initialized   = true;
    m_condition.notify_all();

    if( !m_affinityValid )
    {
        return;
    }

    uint64_t previous = 0;

    while( !m_stop.load( std::memory_order_acquire ) )
    {
        m_condition.wait( lock,
                          [this]
                          {
                              return !m_pause.load( std::memory_order_acquire ) ||
                                     m_stop.load( std::memory_order_acquire );
                          } );

        if( m_stop.load( std::memory_order_acquire ) )
        {
            break;
        }

        m_parked = false;
        m_condition.notify_all();
        lock.unlock();

        while( !m_pause.load( std::memory_order_acquire ) && !m_stop.load( std::memory_order_acquire ) )
        {
            const uint64_t job = m_job.load( std::memory_order_acquire );

            if( job == previous )
            {
                relax();
                continue;
            }

            const uint64_t start = m_measure ? now() : 0;

            if( !m_handoffOnly )
            {
                rows( m_dest,
                      m_raw,
                      m_partition == Partition::halves     ? 60
                      : m_partition == Partition::paired28 ? 28
                                                           : 30,
                      m_partition == Partition::halves ? 120 : 60,
                      m_partition != Partition::halves );
            }

            if( m_measure )
            {
                m_workerNs = now() - start;
            }

            previous = job;
            m_done.store( job, std::memory_order_release );
        }

        lock.lock();
        m_parked = true;
        m_condition.notify_all();
    }
}

} // namespace app
} // namespace MagAOX

#endif // ocamDescrambleWorker_hpp
