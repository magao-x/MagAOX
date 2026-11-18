/** \file ocamDescrambleParallel.hpp
 * \brief Shared OCAM row kernel and mandatory-job protocol for an externally managed worker.
 * \ingroup ocam2KCtrl_files
 */

#ifndef ocamDescrambleParallel_hpp
#define ocamDescrambleParallel_hpp

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>

#include "ocamDescrambler.hpp"

namespace MagAOX
{
namespace app
{

/// Coordinate one frame owner and an externally started OCAM descramble worker.
/** The owner calls prepare(), clear(), resume(), park(), run(), and serial() without overlapping those calls.
 * Other threads may requestPark() or requestStop() and inspect atomic status. The worker starts parked; its owner
 * must verify actual scheduling and disjoint CPU placement before resume(). Admission and control requests share
 * one atomic word so an accepted frame drains before parking or stopping. No timeout permits buffer reuse.
 * The external thread owner must requestStop() and join the worker before destroying this object.
 */
class ocamDescrambleParallel
{
  public:
    /// Pixel ownership shared by production and benchmark callers.
    enum class Partition
    {
        halves,   ///< Caller owns output rows 0–59; worker owns 60–119.
        paired28, ///< Caller owns pairs y/119-y for y below 28; worker owns the remaining 32 pairs.
        paired30  ///< Caller and worker each own 30 pairs y/119-y.
    };

    /// Optional kernel-only benchmark diagnostics; dispatch and completion waiting are excluded.
    struct Timings
    {
        uint64_t mainNs{ 0 }; ///< Caller kernel duration, or zero for a handoff-only job.

        uint64_t workerNs{ 0 }; ///< Worker kernel duration, or zero for a handoff-only job.
    };

  private:
    /// Admission, requested control, and acknowledged worker state in one atomic modification order.
    enum Control : uint32_t
    {
        parkRequested = 1,
        stopRequested = 2,
        busy          = 4,
        entered       = 8,
        parked        = 16,
        exited        = 32
    };

    std::array<std::array<uint16_t, 4>, 120> m_bases{}; ///< Owned immutable row bases while jobs are admitted.

    std::array<int, 4> m_strides{}; ///< Validated amplifier strides, retained as data like the production kernel.

    std::atomic<bool> m_prepared{ false }; ///< A complete structured bin2 kernel is available.

    std::mutex m_startMutex; ///< Serializes startup publication with the bounded startup wait.

    std::condition_variable m_startWake; ///< Used only to acknowledge entry into the externally created thread.

    alignas( 64 ) std::atomic<uint32_t> m_control{ parkRequested | parked }; ///< Control and job admission.

    alignas( 64 ) std::atomic<uint64_t> m_job{ 0 }; ///< Release-published generation and its following payload.

    short *m_dest{ nullptr }; ///< Current destination, held until completion or definitive worker exit.

    const short *m_raw{ nullptr }; ///< Immutable current input, held until completion or definitive worker exit.

    Partition m_partition{ Partition::halves }; ///< Ownership split published with the job generation.

    bool m_handoffOnly{ false }; ///< Suppress both pixel kernels for benchmark calibration.

    bool m_measure{ false }; ///< Enable timestamps only for explicit benchmark diagnostics.

    alignas( 64 ) std::atomic<uint64_t> m_done{ 0 }; ///< Release-published completion on its own cache line.

    uint64_t m_workerNs{ 0 }; ///< Worker duration, read only after acquiring its completion.

    /// Apply the four-amplifier kernel to one output row.
    void row( short       *dest /**< [out] Complete private output image. */,
              const short *raw /**< [in] Complete immutable raw frame. */,
              size_t       index /**< [in] Output row in [0,120). */ ) const;

    /// Process an interval of output rows or complementary row pairs.
    void rows( short       *dest /**< [out] Complete private output image. */,
               const short *raw /**< [in] Complete immutable raw frame. */,
               size_t       first /**< [in] First output row or pair index. */,
               size_t       end /**< [in] Exclusive row or pair bound. */,
               bool         paired /**< [in] Also process row 119-index. */ ) const;

    /// Read monotonic nanoseconds for optional benchmark diagnostics.
    static uint64_t now();

    /// Hint to the CPU while polling an active job or completion.
    static void relax();

  public:
    /// Park and derive an owned kernel from a prepared structured bin2 helper.
    /** \returns 0 on success, or -1 with no prepared kernel for unsupported input or allocation failure.
     * The external worker may already be running or may not yet have been created.
     */
    int prepare( const ocamDescrambler &helper /**< [in] Validated source mapping. */ );

    /// Park before invalidating prepared state; no pending job retains references to the old kernel.
    void clear();

    /// Enter the externally managed worker once, initially blocked until verified placement permits resume().
    void workerMain() noexcept;

    /// Wait for worker entry without hanging when thread creation or its entry point fails.
    /** \returns true after workerMain() has entered, including when that worker subsequently stopped.
     */
    bool waitUntilStarted( std::chrono::milliseconds timeout = std::chrono::seconds( 1 )
                           /**< [in] Maximum startup wait. */ );

    /// Report whether workerMain() has entered at least once.
    bool started() const;

    /// Report whether a complete structured bin2 kernel is prepared.
    bool prepared() const;

    /// Report an active, prepared worker with no pending park or stop request.
    bool running() const;

    /// Report acknowledged quiescence, including terminal worker exit after the owner call finishes.
    bool isParked() const;

    /// Report definitive worker exit, after its last possible pixel write.
    bool stopped() const;

    /// Report admission of a frame whose owner has not yet completed the call.
    bool inFlight() const;

    /// Explicitly clear a park request after the owner has verified that work is permitted.
    /** \returns 0 after active acknowledgment, or -1 if unprepared, never started, stopped, or concurrently parked.
     * Stop is terminal. This is the only operation that clears a park request; run() never resumes implicitly.
     */
    int resume();

    /// Request parking after any admitted frame, without waiting for the worker or frame owner.
    void requestPark() noexcept;

    /// Request parking and wait for an acknowledged quiescent worker before changing prepared state.
    void park();

    /// Request terminal shutdown after any admitted frame, waking an idle worker without waiting for it.
    void requestStop() noexcept;

    /// Complete one admitted frame without allocating, logging, or publishing a stream.
    /** \returns 0 after both pixel regions complete, or -1 after refusal or definitive worker exit. On either return
     * no worker retains access to the supplied buffers. Requires nonoverlapping, sufficiently large buffers unless
     * handoffOnly is true. A concurrent park/stop drains an admitted frame; it does not abandon an outstanding write.
     */
    int run( short       *dest /**< [out] At least 14,400 samples; unused for handoff-only calibration. */,
             const short *raw /**< [in] At least 32,736 samples; unused for handoff-only calibration. */,
             Partition    partition /**< [in] Caller/worker row ownership. */,
             bool         handoffOnly = false /**< [in] Suppress pixel kernels for benchmark calibration. */,
             Timings     *timings     = nullptr /**< [out] Optional kernel diagnostics; null disables clocks. */ );

    /// Run the shared kernel on the owner alone for benchmark comparisons.
    void serial( short       *dest /**< [out] At least 14,400 output samples. */,
                 const short *raw /**< [in] At least 32,736 immutable raw samples. */,
                 bool paired = false /**< [in] Use complementary row pairs rather than consecutive rows. */ ) const;
};

inline int ocamDescrambleParallel::prepare( const ocamDescrambler &helper )
{
    clear();

    if( !helper.ready() || !helper.structured() || helper.rawPixels() != OCAM2_PIXELS_RAW_BINNING2x2 ||
        helper.imagePixels() != OCAM2_PIXELS_IMAGE_BINNING2x2 )
    {
        return -1;
    }

    try
    {
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
    }
    catch( const std::bad_alloc & )
    {
        return -1;
    }

    m_prepared.store( true, std::memory_order_release );
    return 0;
}

inline void ocamDescrambleParallel::clear()
{
    park();
    m_prepared.store( false, std::memory_order_release );
}

inline bool ocamDescrambleParallel::waitUntilStarted( std::chrono::milliseconds timeout )
{
    std::unique_lock<std::mutex> lock( m_startMutex );
    return m_startWake.wait_for( lock,
                                 timeout,
                                 [this]
                                 {
                                     return started();
                                 } );
}

inline bool ocamDescrambleParallel::started() const
{
    return m_control.load( std::memory_order_acquire ) & entered;
}

inline bool ocamDescrambleParallel::prepared() const
{
    return m_prepared.load( std::memory_order_acquire );
}

inline bool ocamDescrambleParallel::running() const
{
    const uint32_t state = m_control.load( std::memory_order_acquire );
    return prepared() && ( state & ( entered | parked | parkRequested | stopRequested | exited ) ) == entered;
}

inline bool ocamDescrambleParallel::stopped() const
{
    return m_control.load( std::memory_order_acquire ) & exited;
}

inline bool ocamDescrambleParallel::isParked() const
{
    const uint32_t state = m_control.load( std::memory_order_acquire );
    return ( state & ( parked | exited ) ) && !( state & busy );
}

inline bool ocamDescrambleParallel::inFlight() const
{
    return m_control.load( std::memory_order_acquire ) & busy;
}

inline int ocamDescrambleParallel::resume()
{
    if( !prepared() )
    {
        return -1;
    }

    uint32_t state = m_control.load( std::memory_order_acquire );

    for( ;; )
    {
        if( !( state & entered ) || ( state & ( stopRequested | exited | busy ) ) )
        {
            return -1;
        }

        if( !( state & ( parkRequested | parked ) ) )
        {
            return 0;
        }

        if( m_control.compare_exchange_weak( state, state & ~parkRequested, std::memory_order_acq_rel ) )
        {
            m_control.notify_all();
            break;
        }
    }

    for( ;; )
    {
        state = m_control.load( std::memory_order_acquire );

        if( state & ( parkRequested | stopRequested | exited ) )
        {
            return -1;
        }

        if( !( state & parked ) )
        {
            return 0;
        }

        m_control.wait( state, std::memory_order_acquire );
    }
}

inline void ocamDescrambleParallel::requestPark() noexcept
{
    m_control.fetch_or( parkRequested, std::memory_order_release );
    m_control.notify_all();
}

inline void ocamDescrambleParallel::park()
{
    uint32_t state = m_control.load( std::memory_order_acquire );

    if( ( state & ( parkRequested | parked | busy ) ) == ( parkRequested | parked ) )
    {
        return;
    }

    requestPark();

    for( ;; )
    {
        state = m_control.load( std::memory_order_acquire );

        if( ( state & exited ) || !( state & entered ) || ( ( state & parked ) && !( state & busy ) ) )
        {
            return;
        }

        m_control.wait( state, std::memory_order_acquire );
    }
}

inline void ocamDescrambleParallel::requestStop() noexcept
{
    m_control.fetch_or( stopRequested | parkRequested, std::memory_order_release );
    m_control.notify_all();
}

inline int
ocamDescrambleParallel::run( short *dest, const short *raw, Partition partition, bool handoffOnly, Timings *timings )
{
    if( !prepared() ||
        ( partition != Partition::halves && partition != Partition::paired28 && partition != Partition::paired30 ) )
    {
        return -1;
    }

    uint32_t state = entered;

    if( !m_control.compare_exchange_strong( state, entered | busy, std::memory_order_acq_rel ) )
    {
        return -1;
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

    const uint64_t mainNs   = m_measure ? now() - start : 0;
    bool           complete = false;

    while( !( complete = m_done.load( std::memory_order_acquire ) == job ) )
    {
        if( stopped() )
        {
            break;
        }

        relax();
    }

    if( timings != nullptr )
    {
        timings->mainNs   = mainNs;
        timings->workerNs = complete && m_measure ? m_workerNs : 0;
    }

    state = m_control.fetch_and( ~busy, std::memory_order_release );

    if( state & ( parkRequested | stopRequested | exited ) )
    {
        m_control.notify_all();
    }

    return complete ? 0 : -1;
}

inline void ocamDescrambleParallel::serial( short *dest, const short *raw, bool paired ) const
{
    rows( dest, raw, 0, paired ? 60 : 120, paired );
}

inline void ocamDescrambleParallel::row( short *__restrict__ dest, const short *__restrict__ raw, size_t index ) const
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

inline void ocamDescrambleParallel::rows( short *dest, const short *raw, size_t first, size_t end, bool paired ) const
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

inline uint64_t ocamDescrambleParallel::now()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>( std::chrono::steady_clock::now().time_since_epoch() )
        .count();
}

inline void ocamDescrambleParallel::relax()
{
#if defined( __i386__ ) || defined( __x86_64__ )
    __builtin_ia32_pause();
#else
    std::atomic_signal_fence( std::memory_order_seq_cst );
#endif
}

inline void ocamDescrambleParallel::workerMain() noexcept
{
    try
    {
        { //mutex scope
            std::lock_guard<std::mutex> lock( m_startMutex );

            if( m_control.fetch_or( entered, std::memory_order_acq_rel ) & entered )
            {
                return;
            }
        }

        m_startWake.notify_all();
        m_control.notify_all();
        uint64_t previous = 0;

        for( ;; )
        {
            uint32_t state = m_control.load( std::memory_order_acquire );

            if( state & busy )
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
                continue;
            }

            if( state & stopRequested )
            {
                break;
            }

            if( state & parkRequested )
            {
                if( !( state & parked ) )
                {
                    if( m_control.compare_exchange_weak( state, state | parked, std::memory_order_acq_rel ) )
                    {
                        m_control.notify_all();
                    }
                }
                else
                {
                    m_control.wait( state, std::memory_order_acquire );
                }

                continue;
            }

            if( state & parked )
            {
                if( m_control.compare_exchange_weak( state, state & ~parked, std::memory_order_acq_rel ) )
                {
                    m_control.notify_all();
                }

                continue;
            }

            relax();
        }
    }
    catch( ... )
    {
        // Publish definitive exit below before any caller can reuse the buffers after an exceptional worker failure.
    }

    m_control.fetch_or( stopRequested | parkRequested | parked | exited, std::memory_order_release );
    m_control.notify_all();
}

} // namespace app
} // namespace MagAOX

#endif // ocamDescrambleParallel_hpp
