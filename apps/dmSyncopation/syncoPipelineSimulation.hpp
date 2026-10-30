/** \file syncoPipelineSimulation.hpp
 * \brief Fixed-credit conditional preparation, SDK-worker and ordered-post resource simulation.
 */
#ifndef dmSyncopation_syncoPipelineSimulation_hpp
#define dmSyncopation_syncoPipelineSimulation_hpp

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "../../libMagAOX/common/dmSyncoSimulation.hpp"

namespace MagAOX
{
namespace app
{
namespace synco
{

/// Simulate the independent controller stages without threads, allocation or hardware access.
/** The configured bounds are conditional fixed latencies, not measured guarantees. Each admitted
 * command owns one credit until its ordered post stage retires. Future auxiliary credits are reserved
 * at decision time, but resource execution follows input and vector-ready times. Native services are
 * never preempted, and this model does not coalesce or supersede queued commands.
 */
class PipelineSimulation
{
  public:
    /// Startup-only bounds in monotonic nanoseconds.
    struct Config
    {
        int64_t deliveryNs{ 20000 };    ///< Input-anchor to controller-input arrival allowance.
        int64_t prepareNs{ 35000 };     ///< Serialized preparation start to private vector readiness.
        int64_t readerNs{ 60000 };      ///< Total input-anchor to the final shared reader's completion.
        int64_t sdkServiceNs{ 120000 }; ///< SDK entry through worker readiness for its next service.
        int64_t postNs{ 150000 };       ///< Independent serialized ordered-post service duration.
    };

    /// Controller private-vector credits available to preparation, SDK and post stages together.
    static constexpr size_t capacity = 3;

    /// Conditional admission result; no result authorizes a physical auxiliary command.
    enum class Result
    {
        accepted,        ///< All modeled stages preserve the following genuine main.
        prepareBusy,     ///< Candidate preparation would wait behind another shared-reader stage.
        sdkBusy,         ///< Candidate SDK entry would wait after its private vector is ready.
        creditsFull,     ///< No command credit is available at the auxiliary decision.
        mainSdkBusy,     ///< Candidate SDK/readers would still be busy at the earliest following main.
        mainCreditsFull, ///< No command credit would be free at the earliest following main.
        invalid          ///< Bounds, ordering or timestamp arithmetic cannot represent the request.
    };

  private:
    /// One fixed-storage command lease, including its completed times until the slot is reused.
    struct Job
    {
        bool     active{ false };      ///< This slot currently owns one command credit.
        bool     main{ false };        ///< Genuine main rather than conditional auxiliary command.
        bool     prepStarted{ false }; ///< Serialized monitor preparation has started.
        bool     vectorReady{ false }; ///< Private vector can enter the SDK worker.
        bool     staged{ false };      ///< Monitor has finished its last shared read.
        bool     sdkStarted{ false };  ///< Serialized native service has started.
        bool     sdkDone{ false };     ///< SDK worker is ready after this service.
        bool     postStarted{ false }; ///< Ordered post service has started.
        uint64_t counter{ 0 };         ///< Caller-supplied genuine main-cycle identity.
        int      step{ 0 };            ///< Zero for genuine mains, positive for auxiliary slots.
        int64_t  inputNs{ 0 };         ///< Anchor plus delivery, independent of reservation order.
        int64_t  prepNs{ 0 };          ///< Actual monitor-stage start after earlier shared readers clear.
        int64_t  vectorNs{ 0 };        ///< Actual private-vector readiness.
        int64_t  readerNs{ 0 };        ///< Actual final shared-reader completion.
        int64_t  sdkNs{ 0 };           ///< Actual native entry, also defining ordered-post sequence.
        int64_t  sdkReadyNs{ 0 };      ///< Native entry plus bounded worker service.
        int64_t  retiredNs{ 0 };       ///< Ordered-post completion and credit release.
    };

    Config m_config; ///< Bounds changed only by a successful configure operation.

    MagAOX::dmSyncoSimulation::Pipeline m_summary; ///< Current resource state and cumulative diagnostics.

    std::array<Job, capacity> m_jobs{}; ///< Private leases, including reservations for future input arrivals.

    bool m_configured{ false }; ///< No admission is allowed until bounds are validated.

    int64_t m_nowNs{ 0 }; ///< Last processed time; resource execution never advances beyond it.

    int64_t m_horizonNs{ 0 }; ///< Conservative three-job drain bound used to reject arithmetic overflow.

    int64_t m_unknownHorizonNs{ 0 }; ///< Quarantine bound when an observed command's ownership is unknown.

    int64_t m_cooldownNs{ 0 }; ///< Fenced old jobs must have drained before reset credits can be reused.

    /// Reserve one credit after validating every pending job's possible drain arithmetic.
    size_t admit( uint64_t counter /**< [in] Main-cycle identity. */,
                  int      step /**< [in] Zero for a genuine main, positive for an auxiliary. */,
                  int64_t  anchorNs /**< [in] Genuine observation or hypothetical trigger. */ );

    /// Apply simultaneous completions, then start eligible independent stages at the current time.
    void settle();

    /// Select the earliest future transition, or return the maximum timestamp when no work remains.
    int64_t nextEvent() const;

    /// Compare prepared inputs by actual input time, preferring genuine mains for exact ties.
    static bool earlierInput( const Job &left /**< [in] Candidate input. */,
                              const Job &right /**< [in] Current earliest input. */ );

  public:
    /// Validate bounds and begin a fresh model session with zero cumulative counters.
    bool configure( const Config &config /**< [in] Conditional stage and delivery bounds. */ );

    /// Discard modeled leases and fence replacement work until those leases could have drained.
    /** Admission and retirement diagnostics survive this fence. Discarded work is counted separately,
     * preserving admission = retirement + discard + in-flight conservation.
     */
    void reset();

    /// Fence unknown command activity after combined-command or source association is lost.
    void invalidate( int64_t observedNs /**< [in] Latest observed command time, no earlier than model time. */ );

    /// Process only transitions at or before the requested nondecreasing monotonic time.
    void advance( int64_t nowNs /**< [in] Latest time available to the resource model. */ );

    /// Admit a genuine main, recording additional SDK-entry delay from preparation or SDK queues.
    bool main( uint64_t counter /**< [in] Genuine main-cycle identity. */,
               int64_t  observedNs /**< [in] Observed combined-command anchor. */ );

    /// Forecast a conditional auxiliary without permitting any extra following-main SDK wait.
    /** A rejected forecast changes only the required-main-reserve diagnostic. Accepted input owns a
     * credit immediately, while its future preparation, SDK and post work remain unexecuted.
     */
    Result auxiliary( uint64_t counter /**< [in] Genuine main-cycle identity. */,
                      int      step /**< [in] Positive active auxiliary slot. */,
                      int64_t  decisionNs /**< [in] Current decision time, no later than the trigger. */,
                      int64_t  triggerNs /**< [in] Hypothetical command-publication anchor. */,
                      int64_t  earliestMainNs /**< [in] Earliest following main; no preparation credit is assumed. */ );

    /// Return cumulative diagnostics and endpoints of services actually started by the model.
    const MagAOX::dmSyncoSimulation::Pipeline &summary() const;
};

inline bool PipelineSimulation::configure( const Config &config )
{
    const __int128 monitorNs = __int128( config.readerNs ) - config.deliveryNs;
    const __int128 horizonNs = capacity * ( monitorNs + config.sdkServiceNs + config.postNs );
    const __int128 unknownHorizonNs =
        capacity * ( __int128( config.deliveryNs ) + config.readerNs + config.sdkServiceNs + config.postNs );

    if( config.deliveryNs < 0 || config.deliveryNs > 1000000000 || config.prepareNs <= 0 ||
        config.prepareNs > 1000000000 || config.readerNs > 1000000000 || monitorNs < config.prepareNs ||
        config.sdkServiceNs <= 0 || config.sdkServiceNs > 1000000000 || config.postNs <= 0 ||
        config.postNs > 1000000000 || unknownHorizonNs > std::numeric_limits<int64_t>::max() )
    {
        return false;
    }

    m_config               = config;
    m_summary              = {};
    m_summary.enabled      = true;
    m_summary.prepareNs    = config.prepareNs;
    m_summary.sdkServiceNs = config.sdkServiceNs;
    m_summary.postNs       = config.postNs;
    m_summary.capacity     = capacity;
    m_jobs                 = {};
    m_configured           = true;
    m_nowNs                = 0;
    m_horizonNs            = int64_t( horizonNs );
    m_unknownHorizonNs     = int64_t( unknownHorizonNs );
    m_cooldownNs           = 0;
    return true;
}

inline void PipelineSimulation::invalidate( int64_t observedNs )
{
    if( !m_configured || observedNs < m_nowNs || observedNs <= 0 )
    {
        return;
    }

    advance( observedNs );
    reset();
    const __int128 fenceNs = __int128( observedNs ) + m_unknownHorizonNs;
    m_cooldownNs =
        std::max( m_cooldownNs, int64_t( std::min( fenceNs, __int128( std::numeric_limits<int64_t>::max() ) ) ) );
    m_summary.sdkReadyNs  = m_cooldownNs;
    m_summary.postReadyNs = m_cooldownNs;
}

inline void PipelineSimulation::reset()
{
    if( !m_configured )
    {
        return;
    }

    PipelineSimulation drained = *this;
    drained.advance( std::numeric_limits<int64_t>::max() );
    m_cooldownNs = std::max( m_cooldownNs, drained.m_summary.postReadyNs );
    m_summary.discarded += m_summary.inFlight;
    m_summary.inFlight = 0;
    ++m_summary.resets;
    m_jobs                = {};
    m_summary.sdkReadyNs  = m_cooldownNs;
    m_summary.postReadyNs = m_cooldownNs;
}

inline bool PipelineSimulation::earlierInput( const Job &left, const Job &right )
{
    if( left.inputNs != right.inputNs )
    {
        return left.inputNs < right.inputNs;
    }

    if( left.main != right.main )
    {
        return left.main;
    }

    return left.counter < right.counter || ( left.counter == right.counter && left.step < right.step );
}

inline size_t PipelineSimulation::admit( uint64_t counter, int step, int64_t anchorNs )
{
    const __int128 inputNs       = __int128( anchorNs ) + m_config.deliveryNs;
    __int128       latestInputNs = inputNs;
    size_t         slot          = capacity;

    for( size_t i = 0; i < capacity; ++i )
    {
        if( m_jobs[i].active )
        {
            latestInputNs = std::max( latestInputNs, __int128( m_jobs[i].inputNs ) );
        }
        else if( slot == capacity )
        {
            slot = i;
        }
    }

    if( slot == capacity || latestInputNs + m_horizonNs > std::numeric_limits<int64_t>::max() )
    {
        return capacity;
    }

    auto &job   = m_jobs[slot];
    job         = {};
    job.active  = true;
    job.main    = step == 0;
    job.counter = counter;
    job.step    = step;
    job.inputNs = int64_t( inputNs );
    ++m_summary.inFlight;
    m_summary.maximumInFlight = std::max( m_summary.maximumInFlight, m_summary.inFlight );

    if( job.main )
    {
        ++m_summary.mainAdmitted;
    }
    else
    {
        ++m_summary.auxiliaryAdmitted;
    }

    return slot;
}

inline void PipelineSimulation::settle()
{
    for( auto &job : m_jobs )
    {
        if( !job.active )
        {
            continue;
        }

        if( job.postStarted && job.retiredNs <= m_nowNs )
        {
            job.active = false;
            --m_summary.inFlight;
            ++m_summary.retired;
            continue;
        }

        if( job.prepStarted )
        {
            job.vectorReady = job.vectorNs <= m_nowNs;
            job.staged      = job.readerNs <= m_nowNs;
        }

        if( job.sdkStarted && job.sdkReadyNs <= m_nowNs )
        {
            job.sdkDone = true;
        }
    }

    bool   prepBusy = false;
    bool   sdkBusy  = false;
    bool   postBusy = false;
    size_t prep     = capacity;
    size_t sdk      = capacity;
    size_t post     = capacity;

    for( size_t i = 0; i < capacity; ++i )
    {
        const auto &job = m_jobs[i];

        if( !job.active )
        {
            continue;
        }

        prepBusy = prepBusy || ( job.prepStarted && !job.staged );
        sdkBusy  = sdkBusy || ( job.sdkStarted && !job.sdkDone );
        postBusy = postBusy || job.postStarted;

        if( !job.prepStarted && job.inputNs <= m_nowNs && ( prep == capacity || earlierInput( job, m_jobs[prep] ) ) )
        {
            prep = i;
        }

        if( job.vectorReady && !job.sdkStarted &&
            ( sdk == capacity || job.vectorNs < m_jobs[sdk].vectorNs ||
              ( job.vectorNs == m_jobs[sdk].vectorNs && earlierInput( job, m_jobs[sdk] ) ) ) )
        {
            sdk = i;
        }

        if( job.sdkStarted && !job.postStarted && ( post == capacity || job.sdkNs < m_jobs[post].sdkNs ) )
        {
            post = i;
        }
    }

    if( !prepBusy && prep != capacity )
    {
        auto &job       = m_jobs[prep];
        job.prepStarted = true;
        job.prepNs      = m_nowNs;
        job.vectorNs    = m_nowNs + m_config.prepareNs;
        job.readerNs    = m_nowNs + ( m_config.readerNs - m_config.deliveryNs );
    }

    if( !sdkBusy && sdk != capacity )
    {
        auto &job            = m_jobs[sdk];
        job.sdkStarted       = true;
        job.sdkNs            = m_nowNs;
        job.sdkReadyNs       = m_nowNs + m_config.sdkServiceNs;
        m_summary.sdkReadyNs = job.sdkReadyNs;

        const auto waitNs = ( job.prepNs - job.inputNs ) + ( job.sdkNs - job.vectorNs );

        if( job.main && waitNs > 0 )
        {
            ++m_summary.mainSdkWaits;
            m_summary.maximumMainWaitNs = std::max( m_summary.maximumMainWaitNs, waitNs );
        }
    }

    if( !postBusy && post != capacity && m_jobs[post].sdkDone && m_jobs[post].staged )
    {
        auto &job             = m_jobs[post];
        job.postStarted       = true;
        job.retiredNs         = m_nowNs + m_config.postNs;
        m_summary.postReadyNs = job.retiredNs;
    }
}

inline int64_t PipelineSimulation::nextEvent() const
{
    int64_t nextNs = std::numeric_limits<int64_t>::max();

    for( const auto &job : m_jobs )
    {
        if( !job.active )
        {
            continue;
        }

        const std::array<int64_t, 5> events{ !job.prepStarted ? job.inputNs : 0,
                                             job.prepStarted && !job.vectorReady ? job.vectorNs : 0,
                                             job.prepStarted && !job.staged ? job.readerNs : 0,
                                             job.sdkStarted && !job.sdkDone ? job.sdkReadyNs : 0,
                                             job.postStarted ? job.retiredNs : 0 };

        for( const auto timeNs : events )
        {
            if( timeNs > m_nowNs )
            {
                nextNs = std::min( nextNs, timeNs );
            }
        }
    }

    return nextNs;
}

inline void PipelineSimulation::advance( int64_t nowNs )
{
    if( !m_configured || nowNs < m_nowNs || nowNs < 0 )
    {
        return;
    }

    // Each of three leases contributes at most five transitions; no admission occurs in this loop.
    for( size_t transition = 0; transition <= capacity * 5; ++transition )
    {
        settle();
        const auto nextNs = nextEvent();

        if( nextNs > nowNs || nextNs == std::numeric_limits<int64_t>::max() )
        {
            break;
        }

        m_nowNs = nextNs;
    }

    m_nowNs = nowNs;
}

inline bool PipelineSimulation::main( uint64_t counter, int64_t observedNs )
{
    if( !m_configured || observedNs <= 0 || observedNs < m_nowNs )
    {
        return false;
    }

    advance( observedNs );

    if( observedNs < m_cooldownNs || m_summary.inFlight == capacity )
    {
        ++m_summary.mainCreditMisses;
        return false;
    }

    const auto slot = admit( counter, 0, observedNs );

    if( slot == capacity )
    {
        return false;
    }

    settle();
    return true;
}

inline PipelineSimulation::Result PipelineSimulation::auxiliary(
    uint64_t counter, int step, int64_t decisionNs, int64_t triggerNs, int64_t earliestMainNs )
{
    if( !m_configured || step <= 0 || decisionNs <= 0 || decisionNs < m_nowNs || triggerNs < decisionNs ||
        earliestMainNs < triggerNs )
    {
        return Result::invalid;
    }

    PipelineSimulation admitted = *this;
    admitted.advance( decisionNs );

    if( decisionNs < admitted.m_cooldownNs || admitted.m_summary.inFlight == capacity )
    {
        return Result::creditsFull;
    }

    const auto slot = admitted.admit( counter, step, triggerNs );

    if( slot == capacity )
    {
        return Result::invalid;
    }

    admitted.settle();
    PipelineSimulation completed = admitted;
    PipelineSimulation atMain    = admitted;
    completed.advance( std::numeric_limits<int64_t>::max() );
    atMain.advance( earliestMainNs );
    const auto &candidate  = completed.m_jobs[slot];
    int64_t     blockingNs = std::max( candidate.sdkReadyNs, candidate.readerNs );

    for( const auto &job : atMain.m_jobs )
    {
        if( job.active && job.prepStarted && !job.staged )
        {
            blockingNs = std::max( blockingNs, job.readerNs );
        }

        if( job.active && job.sdkStarted && !job.sdkDone )
        {
            blockingNs = std::max( blockingNs, job.sdkReadyNs );
        }
    }

    Result result = Result::accepted;

    if( candidate.prepNs != candidate.inputNs )
    {
        result = Result::prepareBusy;
    }
    else if( candidate.sdkNs != candidate.vectorNs )
    {
        result = Result::sdkBusy;
    }
    else if( blockingNs > earliestMainNs )
    {
        result = Result::mainSdkBusy;
    }
    else if( atMain.m_summary.inFlight == capacity )
    {
        result = Result::mainCreditsFull;
    }

    if( result != Result::accepted )
    {
        if( blockingNs > earliestMainNs )
        {
            m_summary.mainClearanceShortfallNs =
                std::max( m_summary.mainClearanceShortfallNs, blockingNs - earliestMainNs );
        }

        return result;
    }

    *this = admitted;
    return Result::accepted;
}

inline const MagAOX::dmSyncoSimulation::Pipeline &PipelineSimulation::summary() const
{
    return m_summary;
}

} // namespace synco
} // namespace app
} // namespace MagAOX
#endif
