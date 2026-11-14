/** \file online_shadow.cpp
 * \brief Nonblocking observation transport and bounded-duty native DDSPC processing, without a publisher.
 */
#include "online_shadow.hpp"
#include <algorithm>
#include <chrono>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <pthread.h>

namespace DDSPC
{
namespace
{
/// Local processing/deadline clock; no source clock is inferred from it.
std::int64_t now_ns()
{
    timespec t{};
    if( clock_gettime( CLOCK_MONOTONIC, &t ) != 0 )
        throw std::runtime_error( "Online DDSPC monotonic clock unavailable" );
    return std::int64_t( t.tv_sec ) * 1000000000 + t.tv_nsec;
}
} // namespace

struct OnlineShadowWorker::Impl
{
    struct Event
    {
        bool              m_origin{ false }; ///< Qualified origin or applied-certificate discriminant.
        OnlineOrigin      m_wfs;   ///< Fixed-size observation packet, without Eigen allocations on the producer.
        OnlineAppliedSlot m_input; ///< Fixed-size independent certificate, not a generated command.
    };
    OnlineShadowConfig     m_config; ///< Immutable duty and storage limits.
    MultirateShadow        m_engine; ///< Accessed only on this worker; physical evidence required.
    NativeObservationQueue m_raw;    ///< One native observer producer, one numerical consumer.
    std::vector<Event>     m_events; ///< Bounded adapter mailbox; no resize while running.
    std::size_t            m_head{ 0 }, m_tail{ 0 }, m_count{ 0 }; ///< Qualified queue positions under mailbox lock.
    std::optional<MultirateContract> m_contract; ///< At most one reviewed session awaiting installation.
    mutable std::mutex               m_mailbox,
        m_snapshot;                      ///< Neither is held during Eigen processing; acquisition never takes either.
    OnlineShadowStatus         m_status; ///< Published scalar status.
    std::array<Matrix, 2>      m_plans;  ///< Published owned diagnostic plans, never added to training queues.
    std::atomic<bool>          m_running{ false }; ///< Stop flag, not an actuation enable.
    std::atomic<unsigned>      m_requested{ 0 };   ///< Diagnostic selection only.
    std::atomic<std::uint64_t> m_drops{ 0 };       ///< Qualified queue/contention losses, separate from raw drops.
    std::thread                m_thread;           ///< Joined before engine/queue destruction.

    /// Allocate bounded state before any producer can enqueue.
    explicit Impl( const OnlineShadowConfig &config /**< [in] Fixed queue/model bounds. */ );

    /// Try to copy one adapter event without blocking its producer.
    bool push( const Event &event /**< [in] Fixed-size evidence packet. */ );

    /// Own all model work and enforce the soft cycle budget with explicit overrun invalidation.
    void run();
};

OnlineShadowWorker::Impl::Impl( const OnlineShadowConfig &c )
    : m_config( c ), m_engine( c ), m_raw( c.m_queueCapacity ), m_events( c.m_pendingCapacity * 8 )
{
}
bool OnlineShadowWorker::Impl::push( const Event &event )
{
    std::unique_lock<std::mutex> lock( m_mailbox, std::try_to_lock );
    if( !lock.owns_lock() || m_count == m_events.size() )
    {
        ++m_drops;
        return false;
    }
    m_events[m_head] = event;
    m_head           = ( m_head + 1 ) % m_events.size();
    ++m_count;
    return true;
}

void OnlineShadowWorker::Impl::run()
{
    bool          fault         = false;
    std::uint64_t observedDrops = 0;
    double        compute = -1, peak = -1;
    // Do not inherit a real-time policy from a native application thread.
    sched_param priority{};
    priority.sched_priority = sched_get_priority_min( SCHED_OTHER );
    if( pthread_setschedparam( pthread_self(), SCHED_OTHER, &priority ) != 0 )
        fault = true;
    try
    {
        while( m_running.load( std::memory_order_acquire ) )
        {
            const auto                       begin = now_ns();
            std::optional<MultirateContract> contract;
            { //mutex scope
                std::lock_guard<std::mutex> lock( m_mailbox );
                contract.swap( m_contract );
            }
            if( contract && !fault )
                m_engine.qualify( *contract );
            const auto drops = m_drops.load() + m_raw.drops();
            if( drops != observedDrops )
            {
                m_engine.invalidate( "online_queue_loss_new_history_required" );
                observedDrops = drops;
            }
            NativeObservation raw;
            for( unsigned n = 0; n < 128 && now_ns() - begin < std::int64_t( m_config.m_budgetUs ) * 1000; ++n )
            {
                if( !m_raw.pop( raw ) )
                    break;
                m_engine.observe( raw );
            }
            m_engine.request( m_requested.load() ? ShadowMode::AccelAssisted : ShadowMode::Standard );
            while( !fault && now_ns() - begin < std::int64_t( m_config.m_budgetUs ) * 1000 )
            {
                Event event;
                { //mutex scope
                    std::lock_guard<std::mutex> lock( m_mailbox );
                    if( !m_count )
                        break;
                    event  = m_events[m_tail];
                    m_tail = ( m_tail + 1 ) % m_events.size();
                    --m_count;
                }
                const auto start  = now_ns();
                const auto before = m_engine.status();
                if( event.m_origin )
                    m_engine.origin( event.m_wfs, start );
                else
                    m_engine.applied( event.m_input, start );
                const auto finished = now_ns();
                if( event.m_origin )
                    m_engine.finish_origin( finished );
                const auto after = m_engine.status();
                if( after.m_qualified )
                {
                    const auto work = double( finished - start ) / 1000;
                    if( after.m_records != before.m_records || after.m_candidates != before.m_candidates )
                    {
                        compute = work;
                        peak    = std::max( peak, compute );
                    }
                    if( work > m_config.m_budgetUs )
                    {
                        fault = true;
                        m_engine.invalidate( "single_numerical_job_exceeded_budget_restart_required" );
                    }
                }
            }
            m_engine.advance( now_ns() );
            auto status = m_engine.status();
            if( fault )
            {
                status.m_reason         = "worker_policy_or_compute_budget_fault_restart_required";
                status.m_candidateValid = false;
            }
            status.m_running   = true;
            status.m_overrun   = fault;
            status.m_drops     = observedDrops;
            status.m_computeUs = compute;
            status.m_peakUs    = peak;
            status.m_rawQueued = m_raw.queued();
            { //mutex scope
                std::lock_guard<std::mutex> lock( m_mailbox );
                status.m_qualifiedQueued = m_count;
            }
            { //mutex scope
                std::lock_guard<std::mutex> lock( m_snapshot );
                m_status = status;
                for( unsigned i = 0; i < 2; ++i )
                    if( !m_engine.candidate( i ? ShadowMode::AccelAssisted : ShadowMode::Standard, m_plans[i] ) )
                        m_plans[i].resize( 0, 0 );
            }
            const auto elapsed   = now_ns() - begin;
            const auto remaining = std::int64_t( m_config.m_periodUs ) * 1000 - elapsed;
            // A non-preemptible solve overrun still yields before any next job; it never grants a deadline pass.
            std::this_thread::sleep_for( std::chrono::nanoseconds( std::max<std::int64_t>( 100000, remaining ) ) );
        }
    }
    catch( ... )
    {
        std::lock_guard<std::mutex> lock( m_snapshot );
        m_status.m_reason         = "online_worker_exception_restart_required";
        m_status.m_candidateValid = false;
        m_status.m_overrun        = true;
        m_status.m_qualified      = false;
        m_status.m_standard = m_status.m_assisted = ShadowStatus{};
    }
    std::lock_guard<std::mutex> lock( m_snapshot );
    m_status.m_running        = false;
    m_status.m_candidateValid = false;
    m_running.store( false );
}

OnlineShadowWorker::OnlineShadowWorker( const OnlineShadowConfig &c ) : m_impl( std::make_unique<Impl>( c ) )
{
}
OnlineShadowWorker::~OnlineShadowWorker()
{
    stop();
}
void OnlineShadowWorker::start()
{
    if( m_impl->m_thread.joinable() )
        throw std::logic_error( "Online worker already started" );
    m_impl->m_engine.qualify( {} ); // Restart never implicitly reuses a trusted session or ready model.
    {                               //mutex scope
        std::lock_guard<std::mutex> lock( m_impl->m_snapshot );
        m_impl->m_status = {};
        for( auto &plan : m_impl->m_plans )
            plan.resize( 0, 0 );
    }
    { //mutex scope
        std::lock_guard<std::mutex> lock( m_impl->m_mailbox );
        m_impl->m_contract.reset();
        m_impl->m_head = m_impl->m_tail = m_impl->m_count = 0;
    }
    m_impl->m_running.store( true );
    try
    {
        m_impl->m_thread = std::thread( &Impl::run, m_impl.get() );
    }
    catch( ... )
    {
        m_impl->m_running.store( false );
        throw;
    }
}
void OnlineShadowWorker::stop()
{
    m_impl->m_running.store( false );
    if( m_impl->m_thread.joinable() )
        m_impl->m_thread.join();
}
void OnlineShadowWorker::observe( const NativeObservation &record ) noexcept
{
    m_impl->m_raw.push( record );
}
bool OnlineShadowWorker::origin( const OnlineOrigin &r )
{
    if( !m_impl->m_running.load() )
        return false;
    Impl::Event event;
    event.m_origin = true;
    event.m_wfs    = r;
    return m_impl->push( event );
}
bool OnlineShadowWorker::applied( const OnlineAppliedSlot &r )
{
    if( !m_impl->m_running.load() )
        return false;
    Impl::Event event;
    event.m_input = r;
    return m_impl->push( event );
}
bool OnlineShadowWorker::qualify( const MultirateContract &contract )
{
    if( !m_impl->m_running.load() || contract.m_inputs.m_evidence != InputEvidence::VerifiedApplied ||
        contract.m_provenance.size() > 4096 )
        return false;
    std::unique_lock<std::mutex> lock( m_impl->m_mailbox, std::try_to_lock );
    if( !lock.owns_lock() || m_impl->m_contract )
        return false;
    m_impl->m_contract = contract;
    return true;
}
void OnlineShadowWorker::request( ShadowMode mode )
{
    m_impl->m_requested.store( mode == ShadowMode::AccelAssisted );
}
OnlineShadowStatus OnlineShadowWorker::status() const
{
    std::lock_guard<std::mutex> lock( m_impl->m_snapshot );
    auto                        s = m_impl->m_status;
    s.m_rawQueued                 = m_impl->m_raw.queued();
    s.m_drops                     = m_impl->m_drops.load() + m_impl->m_raw.drops();
    if( s.m_candidateValid && now_ns() > s.m_deadlineNs )
        s.m_candidateValid = false;
    return s;
}
bool OnlineShadowWorker::candidate( ShadowMode mode, Matrix &plan ) const
{
    std::lock_guard<std::mutex> lock( m_impl->m_snapshot );
    if( !m_impl->m_status.m_candidateValid || now_ns() > m_impl->m_status.m_deadlineNs )
        return false;
    const auto &source = m_impl->m_plans[mode == ShadowMode::AccelAssisted ? 1 : 0];
    if( !source.size() )
        return false;
    plan = source;
    return true;
}
} // namespace DDSPC
