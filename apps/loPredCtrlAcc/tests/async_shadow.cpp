/** \file async_shadow.cpp
 * \brief Offline SPSC training fanout and bounded pinned snapshots with explicit invalidation and age gates.
 */
#include "async_shadow.hpp"
#include "../training_profile.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <ctime>
#include <limits>
#include <stdexcept>
#include <thread>

namespace DDSPC
{
static_assert( std::atomic<std::uint64_t>::is_always_lock_free && std::atomic<int>::is_always_lock_free &&
                   std::atomic<bool>::is_always_lock_free,
               "Offline handoff requires native lock-free atomics" );

std::uint64_t offline_now_ns()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>( std::chrono::steady_clock::now().time_since_epoch() )
        .count();
}

OfflineRecord::OfflineRecord( const ShadowConfig &config )
{
    m_data.m_residual = m_data.m_appliedCommand = Matrix::Zero( config.m_modes, 1 );
    m_data.m_acceleration                       = Matrix::Zero( config.m_accelChannels, 1 );
}

OfflineSnapshot::OfflineSnapshot( const ShadowConfig &config, ShadowMode mode )
{
    const int d      = config.m_modes * config.m_future;
    const int r      = ( 2 * config.m_history - 1 ) * config.m_modes +
                       ( mode == ShadowMode::AccelAssisted ? config.m_accelChannels * config.m_accelHistory : 0 );
    m_theta          = Matrix::Zero( d, d + r );
    m_gain           = Matrix::Zero( d, r );
    m_originState    = Matrix::Zero( r, 1 );
    m_originProposal = Matrix::Zero( config.m_modes, 1 );
}

namespace
{
/// Reject invalid enum values before accessing a lane; controls run outside valid prediction calls.
std::size_t lane_index( ShadowMode mode /**< [in] Validated model selector. */ )
{
    if( mode == ShadowMode::Standard )
        return 0;
    if( mode == ShadowMode::AccelAssisted )
        return 1;
    throw std::invalid_argument( "Invalid offline model selector" );
}

/// Queue entry is immutable from release-publication until its consumer releases the slot.
struct Envelope
{
    OfflineRecord m_record;          ///< Fully copied immutable input record.
    std::uint64_t m_generation{ 0 }; ///< Lane generation captured at ingress.
    /// Allocate fixed queue payload shapes.
    explicit Envelope( const ShadowConfig &cfg /**< [in] Immutable dimensions. */ );
};
Envelope::Envelope( const ShadowConfig &cfg ) : m_record( cfg )
{
}

/// A writer or one reader exclusively owns a slot; published index never exposes an incomplete payload.
struct PublicationSlot
{
    std::atomic<int> m_owner{ 0 }; ///< 0 unpinned, -1 writer, +1 reader.
    OfflineSnapshot  m_snapshot;   ///< Deep matrix storage; no learner pointers.
    /// Allocate one slot outside processing.
    PublicationSlot( const ShadowConfig &cfg /**< [in] Dimensions. */, ShadowMode mode /**< [in] Model. */ );
};
PublicationSlot::PublicationSlot( const ShadowConfig &cfg, ShadowMode mode ) : m_snapshot( cfg, mode )
{
}
} // namespace

/// Sole owner of one adapting model; the inactive half of the existing manager never trains.
/** Reuse the unchanged Phase 7 manager rather than duplicating its readiness/forecast mathematics.
 * The dormant half costs bounded startup memory and history-copy work, measured by the benchmark.
 */
class OfflineTrainingWorker
{
  public:
    ShadowConfig          m_config;  ///< Immutable model dimensions/penalty.
    ShadowMode            m_mode;    ///< The only model enabled for training in this manager.
    ShadowController      m_manager; ///< Private serialized learner storage, never read by the frontend.
    std::vector<Envelope> m_queue;   ///< Fixed capacity and fixed payload shapes.
    alignas( 64 ) std::atomic<std::uint64_t> m_head{ 0 }; ///< Producer release cursor.
    alignas( 64 ) std::atomic<std::uint64_t> m_tail{
        0 }; ///< Consumer release cursor; in-flight entry remains occupied.
    std::atomic<std::uint64_t> m_generation{ 1 }, m_reset{ 0 },
        m_stallUntil{ 0 };                                              ///< Frontend invalidation/control mailbox.
    std::uint64_t              m_segmentFirstSequence{ 0 };             ///< Consumer-owned source segment.
    std::uint64_t              m_accepted{ 0 }, m_overflowDrops{ 0 };   ///< Sole frontend admission counters.
    std::atomic<std::uint64_t> m_threadCpuNs{ 0 }, m_threadWallNs{ 0 }; ///< Whole worker lifetime clocks at shutdown.
    std::uint64_t m_seenGeneration{ 1 }, m_seenReset{ 0 }, m_publications{ 0 }; ///< Consumer-only acknowledged state.
    std::array<std::unique_ptr<PublicationSlot>, 3> m_slots;           ///< At most writer, published, and pinned slots.
    std::atomic<int>                                m_published{ -1 }; ///< Release-published complete slot index.
    std::atomic<bool> m_fault{ false }; ///< Numerical faults reject old snapshots until a full reset is acknowledged.
    std::atomic<std::uint64_t> m_processed{ 0 }, m_discarded{ 0 }, m_publishMisses{ 0 }, m_highWater{ 0 },
        m_traceDrops{ 0 };                             ///< Telemetry.
    std::vector<OfflineWorkerEvent> m_events;          ///< Fixed trace capacity, never grows in a worker.
    std::size_t                     m_eventCount{ 0 }; ///< Worker-owned used trace prefix.

    /// Allocate this lane and enable only its independent learner.
    OfflineTrainingWorker( const ShadowConfig &cfg /**< [in] Dimensions. */,
                           ShadowMode          mode /**< [in] Model. */,
                           std::size_t         capacity /**< [in] Queue bound. */,
                           std::size_t         traces /**< [in] Trace bound. */ );

    /// Service control mailboxes and at most one entry; zero completion time uses the physical monotonic clock.
    bool step( std::uint64_t completionNs /**< [in] Manual synthetic completion or zero in threaded mode. */ );

    /// Try each fixed publication slot once; never overwrite a pinned or currently published slot.
    bool publish( const Envelope &entry /**< [in] Completed record. */,
                  std::uint64_t   completionNs /**< [in] Clock. */ );

    /// Pin once and revalidate the published index; no retry or wait occurs on a busy slot.
    int pin();
};

OfflineTrainingWorker::OfflineTrainingWorker( const ShadowConfig &cfg,
                                              ShadowMode          mode,
                                              std::size_t         capacity,
                                              std::size_t         traces )
    : m_config( cfg ), m_mode( mode ), m_manager( cfg ), m_events( traces )
{
    m_queue.reserve( capacity );
    for( std::size_t i = 0; i < capacity; ++i )
        m_queue.emplace_back( cfg );
    for( auto &slot : m_slots )
        slot = std::make_unique<PublicationSlot>( cfg, mode );
    m_manager.training( mode, true );
}

int OfflineTrainingWorker::pin()
{
    const int slot = m_published.load( std::memory_order_acquire );
    if( slot < 0 )
        return -1;
    int expected = 0;
    if( !m_slots[slot]->m_owner.compare_exchange_strong( expected, 1, std::memory_order_acq_rel ) )
        return -2;
    if( m_published.load( std::memory_order_acquire ) != slot )
    {
        m_slots[slot]->m_owner.store( 0, std::memory_order_release );
        return -2;
    }
    return slot;
}

bool OfflineTrainingWorker::publish( const Envelope &entry, std::uint64_t completionNs )
{
    DDSPC_PROFILE_SCOPE( Snapshot );
    const int current = m_published.load( std::memory_order_acquire );
    for( int i = 0; i < 3; ++i )
    {
        if( i == current )
            continue;
        int expected = 0;
        if( !m_slots[i]->m_owner.compare_exchange_strong( expected, -1, std::memory_order_acq_rel ) )
            continue;
        auto       &s                   = m_slots[i]->m_snapshot;
        const auto &item                = m_manager.model( m_mode );
        s.m_info.m_status               = item.m_status;
        s.m_info.m_generation           = entry.m_generation;
        s.m_info.m_reset                = m_seenReset;
        s.m_info.m_publication          = ++m_publications;
        s.m_info.m_version              = item.m_status.m_updates;
        s.m_info.m_sequence             = entry.m_record.m_data.m_index;
        s.m_info.m_streamEpoch          = entry.m_record.m_streamEpoch;
        s.m_info.m_measurementNs        = entry.m_record.m_measurementNs;
        s.m_info.m_availableNs          = entry.m_record.m_availableNs;
        s.m_info.m_segmentFirstSequence = m_segmentFirstSequence;
        const auto &internal            = item.m_controller->prediction_matrix_view();
        const int   a = m_config.m_modes * m_config.m_history, b = m_config.m_modes * ( m_config.m_history - 1 );
        const int   d = m_config.m_modes * m_config.m_future,
                    c = m_mode == ShadowMode::AccelAssisted ? m_config.m_accelChannels * m_config.m_accelHistory : 0;
        s.m_theta << internal.rightCols( a ), internal.middleCols( d, b ), internal.middleCols( d + b, c ),
            internal.leftCols( d );
        if( item.m_status.m_ready )
        {
            s.m_gain           = item.m_workspace->gain();
            s.m_originState    = item.m_workspace->state();
            s.m_originProposal = item.m_proposal;
        }
        else
        {
            s.m_gain.setZero();
            s.m_originState.setZero();
            s.m_originProposal.setZero();
        }
        s.m_info.m_publishedNs = completionNs ? completionNs : offline_now_ns();
        m_slots[i]->m_owner.store( 0, std::memory_order_release );
        m_published.store( i, std::memory_order_release );
        return true;
    }
    return false;
}

bool OfflineTrainingWorker::step( std::uint64_t completionNs )
{
    DDSPC_PROFILE_BEGIN( Queue, queueWork );
    const auto generation = m_generation.load( std::memory_order_acquire ),
               reset      = m_reset.load( std::memory_order_acquire );
    if( reset > generation )
        return false; // Full-reset publication is in progress; never fit old queued data after reset.
    if( reset != m_seenReset )
    {
        m_manager.reset( m_mode );
        m_fault.store( false, std::memory_order_release );
        m_seenReset = reset;
    }
    if( generation != m_seenGeneration )
    {
        m_manager.invalidate( ShadowRecord{} );
        m_seenGeneration = generation;
    }
    const auto now = completionNs ? completionNs : offline_now_ns();
    if( now < m_stallUntil.load( std::memory_order_relaxed ) )
        return false;
    const auto tail = m_tail.load( std::memory_order_relaxed );
    if( tail == m_head.load( std::memory_order_acquire ) )
        return false;
    const auto &entry = m_queue[tail % m_queue.size()];
    if( now < entry.m_record.m_availableNs )
        return false;
    if( entry.m_generation != generation )
    {
        m_discarded.fetch_add( 1, std::memory_order_relaxed );
        m_tail.store( tail + 1, std::memory_order_release );
        return true;
    }
    OfflineWorkerEvent event;
    DDSPC_PROFILE_END( queueWork );
    event.m_sequence      = entry.m_record.m_data.m_index;
    event.m_measurementNs = entry.m_record.m_measurementNs;
    event.m_queueWaitNs   = now >= entry.m_record.m_availableNs ? now - entry.m_record.m_availableNs : 0;
#ifdef DDSPC_OFFLINE_ALLOCATION_PROBE
    DDSPCBenchmark::eigenRequests = DDSPCBenchmark::cppRequests = 0;
    DDSPCBenchmark::enabled                                     = true;
#endif
    const auto begin = offline_now_ns();
    m_manager.process( entry.m_record.m_data );
    const auto end     = offline_now_ns();
    event.m_trainingNs = end - begin;
    const auto status  = m_manager.status( m_mode );
    if( status.m_samples == 1 )
        m_segmentFirstSequence = event.m_sequence;
    if( !status.m_samples )
        m_segmentFirstSequence = 0;
    if( status.m_reason == ShadowReason::NumericalFault )
        m_fault.store( true, std::memory_order_release );
    event.m_version = status.m_updates;
    if( entry.m_generation != m_generation.load( std::memory_order_acquire ) || !publish( entry, completionNs ) )
        m_publishMisses.fetch_add( 1, std::memory_order_relaxed );
    event.m_publicationNs = offline_now_ns() - end;
    event.m_completedNs   = offline_now_ns();
#ifdef DDSPC_OFFLINE_ALLOCATION_PROBE
    DDSPCBenchmark::enabled = false;
    event.m_eigenRequests   = DDSPCBenchmark::eigenRequests;
    event.m_cppRequests     = DDSPCBenchmark::cppRequests;
#endif
    if( m_eventCount < m_events.size() )
        m_events[m_eventCount++] = event;
    else
        m_traceDrops.fetch_add( 1, std::memory_order_relaxed );
    m_processed.fetch_add( 1, std::memory_order_relaxed );
    m_tail.store( tail + 1, std::memory_order_release );
    return true;
}

struct OfflineAsyncShadow::Impl
{
    ShadowConfig  m_config;                                          ///< Fixed configuration, no runtime resize.
    std::uint64_t m_sensorMaxAgeNs;                                  ///< Explicit offline sample freshness bound.
    std::array<std::unique_ptr<OfflineTrainingWorker>, 2> m_workers; ///< Independent model ownership.
    std::array<std::thread, 2>                            m_threads; ///< Only used in threaded mode.
    std::atomic<bool> m_running{ false }; ///< Worker shutdown, never waited upon in a prediction call.
    bool              m_started{ false }, m_haveRecord{ false }; ///< Frontend-only lifecycle and ingress identity.
    std::uint64_t     m_streamEpoch{ 0 }, m_lastSequence{ 0 }, m_lastTime{ 0 }, m_drops{ 0 },
        m_overflows{ 0 }; ///< Frontend sequencing/counters.

    OfflineOverflowPolicy m_policy; ///< Immutable admission policy.

    /// Allocate workers outside frame execution.
    Impl( const ShadowConfig   &cfg /**< [in] Dimensions. */,
          std::size_t           capacity /**< [in] Queue bound. */,
          std::size_t           traces /**< [in] Trace bound. */,
          std::uint64_t         sensorMaxAgeNs /**< [in] Sample freshness bound. */,
          OfflineOverflowPolicy policy /**< [in] Admission policy. */ );
    /// Invalidate both lane generations immediately, leaving queued prefixes for their consumers to discard.
    void invalidate();
};
OfflineAsyncShadow::Impl::Impl( const ShadowConfig   &cfg,
                                std::size_t           capacity,
                                std::size_t           traces,
                                std::uint64_t         sensorMaxAgeNs,
                                OfflineOverflowPolicy policy )
    : m_config( cfg ), m_sensorMaxAgeNs( sensorMaxAgeNs ), m_policy( policy )
{
    if( policy != OfflineOverflowPolicy::Common && policy != OfflineOverflowPolicy::Independent )
        throw std::invalid_argument( "Unknown offline overflow policy" );
    if( !capacity )
        throw std::invalid_argument( "Offline queue capacity must be positive" );
    m_workers[0] = std::make_unique<OfflineTrainingWorker>( cfg, ShadowMode::Standard, capacity, traces );
    m_workers[1] = std::make_unique<OfflineTrainingWorker>( cfg, ShadowMode::AccelAssisted, capacity, traces );
}
void OfflineAsyncShadow::Impl::invalidate()
{
    for( auto &w : m_workers )
        w->m_generation.fetch_add( 1, std::memory_order_release );
}
OfflineAsyncShadow::OfflineAsyncShadow( const ShadowConfig   &cfg,
                                        std::size_t           capacity,
                                        std::size_t           traces,
                                        std::uint64_t         sensorMaxAgeNs,
                                        OfflineOverflowPolicy policy )
    : m_impl( std::make_unique<Impl>( cfg, capacity, traces, sensorMaxAgeNs, policy ) )
{
}
OfflineAsyncShadow::~OfflineAsyncShadow()
{
    stop();
}
void OfflineAsyncShadow::start()
{
    if( m_impl->m_started )
        throw std::logic_error( "Offline workers already started" );
    m_impl->m_running.store( true, std::memory_order_release );
    m_impl->m_started = true;
    try
    {
        for( std::size_t i = 0; i < 2; ++i )
            m_impl->m_threads[i] = std::thread(
                [this, i]
                {
                    timespec   cpuStart{}, cpuEnd{};
                    const bool cpuSupported = clock_gettime( CLOCK_THREAD_CPUTIME_ID, &cpuStart ) == 0;
                    const auto wallStart    = offline_now_ns();
                    while( m_impl->m_running.load( std::memory_order_acquire ) )
                        if( !m_impl->m_workers[i]->step( 0 ) )
                            std::this_thread::sleep_for( std::chrono::microseconds( 25 ) );
                    auto &worker = *m_impl->m_workers[i];
                    worker.m_threadWallNs.store( offline_now_ns() - wallStart );
                    if( cpuSupported && clock_gettime( CLOCK_THREAD_CPUTIME_ID, &cpuEnd ) == 0 )
                        worker.m_threadCpuNs.store(
                            ( std::uint64_t( cpuEnd.tv_sec ) * 1000000000 + cpuEnd.tv_nsec ) -
                            ( std::uint64_t( cpuStart.tv_sec ) * 1000000000 + cpuStart.tv_nsec ) );
                } );
    }
    catch( ... )
    {
        stop();
        throw;
    }
}
void OfflineAsyncShadow::stop()
{
    m_impl->m_running.store( false, std::memory_order_release );
    for( auto &thread : m_impl->m_threads )
        if( thread.joinable() )
            thread.join();
    m_impl->m_started = false;
}

bool OfflineAsyncShadow::submit( const OfflineRecord &record, std::uint64_t nowNs )
{
    const auto accepted = submit_record( record, nowNs );
    return accepted.m_standard && accepted.m_assisted;
}
OfflineAdmission OfflineAsyncShadow::submit_record( const OfflineRecord &record, std::uint64_t nowNs )
{
    DDSPC_PROFILE_SCOPE( Queue );
    const auto &r   = record.m_data;
    const auto &cfg = m_impl->m_config;
    const bool  common =
        r.m_wfsValid && r.m_appliedValid && r.m_timingValid && r.m_residual.rows() == cfg.m_modes &&
        r.m_residual.cols() == 1 && r.m_appliedCommand.rows() == cfg.m_modes && r.m_appliedCommand.cols() == 1 &&
        r.m_residual.allFinite() && r.m_appliedCommand.allFinite() && record.m_streamEpoch && record.m_measurementNs &&
        record.m_exposureStartNs < record.m_measurementNs && record.m_appliedNs == record.m_measurementNs &&
        record.m_appliedNs <= record.m_appliedAvailableNs && record.m_appliedAvailableNs <= record.m_availableNs &&
        record.m_measurementNs <= record.m_availableNs && record.m_availableNs <= nowNs &&
        r.m_index != std::numeric_limits<std::uint64_t>::max();
    if( !common || ( m_impl->m_haveRecord &&
                     ( record.m_streamEpoch < m_impl->m_streamEpoch || record.m_measurementNs <= m_impl->m_lastTime ||
                       ( record.m_streamEpoch == m_impl->m_streamEpoch && r.m_index <= m_impl->m_lastSequence ) ) ) )
    {
        m_impl->invalidate();
        ++m_impl->m_drops;
        return {};
    }
    if( m_impl->m_haveRecord &&
        ( record.m_streamEpoch != m_impl->m_streamEpoch || r.m_index != m_impl->m_lastSequence + 1 ) )
        m_impl->invalidate();
    m_impl->m_haveRecord   = true;
    m_impl->m_streamEpoch  = record.m_streamEpoch;
    m_impl->m_lastSequence = r.m_index;
    m_impl->m_lastTime     = record.m_measurementNs;
    const bool sensor = r.m_accelValid && r.m_acceleration.rows() == cfg.m_accelChannels &&
                        r.m_acceleration.cols() == 1 && r.m_acceleration.allFinite() && record.m_accelNs &&
                        record.m_accelNs <= record.m_measurementNs && record.m_accelNs <= record.m_accelAvailableNs &&
                        record.m_accelAvailableNs <= record.m_availableNs &&
                        record.m_measurementNs - record.m_accelNs <= m_impl->m_sensorMaxAgeNs;
    if( !sensor )
        m_impl->m_workers[1]->m_generation.fetch_add( 1, std::memory_order_release );
    std::array<bool, 2> space{};
    for( std::size_t lane = 0; lane < 2; ++lane )
    {
        const auto &w = m_impl->m_workers[lane];
        space[lane]   = w->m_head.load( std::memory_order_relaxed ) - w->m_tail.load( std::memory_order_acquire ) <
                        w->m_queue.size();
    }
    if( !space[0] || !space[1] )
    {
        ++m_impl->m_drops;
        ++m_impl->m_overflows;
        if( m_impl->m_policy == OfflineOverflowPolicy::Common )
        {
            m_impl->invalidate();
            for( auto &worker : m_impl->m_workers )
                ++worker->m_overflowDrops;
            return {};
        }
        for( std::size_t lane = 0; lane < 2; ++lane )
            if( !space[lane] )
            {
                m_impl->m_workers[lane]->m_generation.fetch_add( 1, std::memory_order_release );
                ++m_impl->m_workers[lane]->m_overflowDrops;
            }
    }
    for( std::size_t lane = 0; lane < 2; ++lane )
    {
        if( !space[lane] )
            continue;
        auto &w = m_impl->m_workers[lane];
        ++w->m_accepted;
        const auto head              = w->m_head.load( std::memory_order_relaxed );
        auto      &entry             = w->m_queue[head % w->m_queue.size()];
        entry.m_generation           = w->m_generation.load( std::memory_order_relaxed );
        auto &dest                   = entry.m_record;
        dest.m_streamEpoch           = record.m_streamEpoch;
        dest.m_measurementNs         = record.m_measurementNs;
        dest.m_availableNs           = record.m_availableNs;
        dest.m_exposureStartNs       = record.m_exposureStartNs;
        dest.m_appliedNs             = record.m_appliedNs;
        dest.m_appliedAvailableNs    = record.m_appliedAvailableNs;
        dest.m_accelNs               = record.m_accelNs;
        dest.m_accelAvailableNs      = record.m_accelAvailableNs;
        dest.m_data.m_index          = r.m_index;
        dest.m_data.m_residual       = r.m_residual;
        dest.m_data.m_appliedCommand = r.m_appliedCommand;
        if( sensor )
            dest.m_data.m_acceleration = r.m_acceleration;
        else
            dest.m_data.m_acceleration.setZero();
        dest.m_data.m_wfsValid = dest.m_data.m_appliedValid = dest.m_data.m_timingValid = true;
        dest.m_data.m_accelValid                                                        = sensor;
        w->m_head.store( head + 1, std::memory_order_release );
        const auto depth = head + 1 - w->m_tail.load( std::memory_order_acquire );
        w->m_highWater.store( std::max( depth, w->m_highWater.load( std::memory_order_relaxed ) ),
                              std::memory_order_relaxed );
    }
    return { space[0], space[1] };
}
void OfflineAsyncShadow::reset( ShadowMode mode )
{
    auto      &w    = *m_impl->m_workers[lane_index( mode )];
    const auto next = w.m_generation.load( std::memory_order_relaxed ) + 1;
    w.m_reset.store( next, std::memory_order_release ); // Reset incarnation is its generation, not a separate counter.
    w.m_generation.store( next, std::memory_order_release );
}
bool OfflineAsyncShadow::manual_step( ShadowMode mode, std::uint64_t publicationNs )
{
    if( m_impl->m_started )
        throw std::logic_error( "Manual service cannot share a worker thread" );
    return m_impl->m_workers[lane_index( mode )]->step( publicationNs );
}
void OfflineAsyncShadow::stall_until( ShadowMode mode, std::uint64_t untilNs )
{
    m_impl->m_workers[lane_index( mode )]->m_stallUntil.store( untilNs, std::memory_order_relaxed );
}

bool OfflineAsyncShadow::predict( ShadowMode         mode,
                                  std::uint64_t      streamEpoch,
                                  std::uint64_t      sequence,
                                  std::uint64_t      stateNs,
                                  std::uint64_t      nowNs,
                                  std::uint64_t      maxAgeNs,
                                  const Matrix      &state,
                                  Matrix            &output,
                                  OfflinePrediction &result )
{
    result = OfflinePrediction{};
    if( output.rows() != m_impl->m_config.m_modes || output.cols() != 1 || output.data() == state.data() )
    {
        result.m_reason = OfflinePredictionReason::InvalidState;
        return false;
    }
    output.setZero();
    if( mode != ShadowMode::Standard && mode != ShadowMode::AccelAssisted )
    {
        result.m_reason = OfflinePredictionReason::InvalidState;
        return false;
    }
    auto     &w    = *m_impl->m_workers[mode == ShadowMode::Standard ? 0 : 1];
    const int slot = w.pin();
    if( slot < 0 )
    {
        result.m_reason = slot == -1 ? OfflinePredictionReason::NoSnapshot : OfflinePredictionReason::Busy;
        return false;
    }
    const auto &s      = w.m_slots[slot]->m_snapshot;
    result.m_model     = s.m_info;
    const auto &info   = s.m_info;
    auto        reason = OfflinePredictionReason::Valid;
    if( info.m_generation != w.m_generation.load( std::memory_order_acquire ) ||
        info.m_reset != w.m_reset.load( std::memory_order_acquire ) || w.m_fault.load( std::memory_order_acquire ) ||
        streamEpoch != info.m_streamEpoch )
        reason = OfflinePredictionReason::Invalidated;
    else if( !info.m_status.m_ready || !info.m_status.m_dataValid || !info.m_version )
        reason = OfflinePredictionReason::Unready;
    else if( sequence < info.m_sequence || info.m_measurementNs > stateNs || stateNs > nowNs ||
             info.m_publishedNs > nowNs || info.m_availableNs > nowNs )
        reason = OfflinePredictionReason::TimeOrder;
    else if( ( result.m_ageNs = nowNs - info.m_measurementNs ) > maxAgeNs )
        reason = OfflinePredictionReason::Stale;
    else if( state.rows() != s.m_gain.cols() || state.cols() != 1 || !state.allFinite() )
        reason = OfflinePredictionReason::InvalidState;
    else
    {
        output.noalias() = s.m_gain.bottomRows( m_impl->m_config.m_modes ) * state;
        if( !output.allFinite() )
            reason = OfflinePredictionReason::InvalidState;
        else if( info.m_generation != w.m_generation.load( std::memory_order_acquire ) ||
                 w.m_fault.load( std::memory_order_acquire ) )
            reason = OfflinePredictionReason::Invalidated;
    }
    w.m_slots[slot]->m_owner.store( 0, std::memory_order_release );
    result.m_reason = reason;
    if( reason != OfflinePredictionReason::Valid )
        output.setZero();
    return reason == OfflinePredictionReason::Valid;
}
bool OfflineAsyncShadow::read_snapshot( ShadowMode mode, OfflineSnapshot &destination )
{
    auto     &w    = *m_impl->m_workers[lane_index( mode )];
    const int slot = w.pin();
    if( slot < 0 )
        return false;
    const auto &s = w.m_slots[slot]->m_snapshot;
    const bool  valid =
        s.m_info.m_generation == w.m_generation.load( std::memory_order_acquire ) &&
        s.m_info.m_reset == w.m_reset.load( std::memory_order_acquire ) &&
        destination.m_theta.rows() == s.m_theta.rows() && destination.m_theta.cols() == s.m_theta.cols() &&
        destination.m_gain.rows() == s.m_gain.rows() && destination.m_gain.cols() == s.m_gain.cols() &&
        destination.m_originState.rows() == s.m_originState.rows() && destination.m_originState.cols() == 1 &&
        destination.m_originProposal.rows() == s.m_originProposal.rows() && destination.m_originProposal.cols() == 1;
    if( valid )
        destination = s;
    w.m_slots[slot]->m_owner.store( 0, std::memory_order_release );
    return valid;
}
OfflineWorkerStats OfflineAsyncShadow::stats( ShadowMode mode ) const
{
    const auto &w = *m_impl->m_workers[lane_index( mode )];
    return { w.m_processed.load(),
             w.m_discarded.load(),
             w.m_publishMisses.load(),
             w.m_head.load() - w.m_tail.load(),
             w.m_highWater.load(),
             w.m_traceDrops.load(),
             w.m_accepted,
             w.m_overflowDrops,
             w.m_threadCpuNs.load(),
             w.m_threadWallNs.load() };
}
const std::vector<OfflineWorkerEvent> &OfflineAsyncShadow::events( ShadowMode mode ) const
{
    if( m_impl->m_started )
        throw std::logic_error( "Stop before reading worker traces" );
    return m_impl->m_workers[lane_index( mode )]->m_events;
}
std::size_t OfflineAsyncShadow::event_count( ShadowMode mode ) const
{
    if( m_impl->m_started )
        throw std::logic_error( "Stop before reading worker traces" );
    return m_impl->m_workers[lane_index( mode )]->m_eventCount;
}
std::uint64_t OfflineAsyncShadow::drops() const
{
    return m_impl->m_drops;
}
std::uint64_t OfflineAsyncShadow::overflows() const
{
    return m_impl->m_overflows;
}
} // namespace DDSPC
