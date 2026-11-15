/** \file online_shadow.cpp
 * \brief Nonblocking observation transport and bounded-duty native DDSPC processing, without a publisher.
 */
#include "online_shadow.hpp"
#include <algorithm>
#include <chrono>
#include <mutex>
#include <optional>
#include <string_view>
#include <cstdio>
#include <cerrno>
#include <filesystem>
#include <fcntl.h>
#include <unistd.h>
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

struct OnlineShadowWorker::Impl : public ShadowPredictionSink
{
    struct Event
    {
        bool m_origin{ false };     ///< Qualified origin or applied-certificate discriminant.
        bool m_commands{ false };   ///< Observed combined requests, distinct from physically applied certificates.
        OnlineOrigin       m_wfs;   ///< Fixed-size observation packet, without Eigen allocations on the producer.
        OnlineAppliedSlot  m_input; ///< Fixed-size independent certificate, not a generated command.
        OnlineCommandFrame m_frame; ///< Complete requested inputs for one WFS frame.
    };
    OnlineShadowConfig     m_config; ///< Immutable duty and storage limits.
    MultirateShadow        m_engine; ///< Sole numerical owner; preserves the admitted request/physical input domain.
    NativeObservationQueue m_raw;    ///< One native observer producer, one numerical consumer.
    std::vector<Event>     m_events; ///< Bounded adapter mailbox; no resize while running.
    std::size_t            m_head{ 0 }, m_tail{ 0 }, m_count{ 0 }; ///< Qualified queue positions under mailbox lock.
    std::optional<MultirateContract> m_contract; ///< At most one reviewed session awaiting installation.
    mutable std::mutex               m_mailbox,
        m_snapshot; ///< Neither is held during Eigen processing; acquisition never takes either.
    OnlineShadowStatus                   m_status; ///< Published scalar status.
    std::array<Matrix, 2>                m_plans; ///< Published owned diagnostic plans, never added to training queues.
    std::atomic<bool>                    m_running{ false }; ///< Stop flag, not an actuation enable.
    std::atomic<unsigned>                m_requested{ 0 };   ///< Diagnostic selection only.
    std::atomic<std::uint64_t>           m_drops{ 0 };  ///< Qualified queue/contention losses, separate from raw drops.
    std::thread                          m_thread;      ///< Joined before engine/queue destruction.
    ShadowPredictionQueue                m_predictions; ///< Optional bounded SPSC scalar diagnostics, never commands.
    NativePipelineConfig                 m_nativeConfig; ///< Startup-only live receiver contract.
    std::unique_ptr<NativeInputPipeline> m_native;       ///< Numerical-thread packet assembly, never acquisition work.
    std::shared_ptr<SyncoShadowProducer> m_shadowProducer; ///< Immutable lifetime while acquisition runs.
    std::array<NativeObservation, 4>     m_lastNative;     ///< Sole acquisition-producer continuity high waters.
    std::unique_ptr<NativeCommandQueue>  m_commandInputs;  ///< Separate complete-input SPSC; never model proposals.
    std::vector<float>                   m_commandPayload; ///< Numerical-thread full-image projection workspace.
    std::optional<NativeObservation>     m_commandPending; ///< One future payload retained while metadata catches up.
    std::optional<NativeSyncStatus>      m_syncStatus; ///< Latest bounded ordinary-thread producer heartbeat mailbox.
    std::optional<std::pair<MagAOX::app::synco::Record, std::int64_t>> m_schedule; ///< Read-only schedule mailbox.
    FILE       *m_trace{ nullptr }; ///< Private exclusive output owned by the recorder, never an instrument file.
    std::thread m_recorder;         ///< Joined after the numerical producer and before closing the score queue.
    std::atomic<std::uint64_t> m_traceRows{ 0 }; ///< Successful CSV rows, independently counted.
    std::atomic<bool>          m_traceIoError{ false }, m_traceLimit{ false }; ///< Explicit bounded recording outcomes.
    std::atomic<bool>          m_numericalDone{ true }; ///< Recorder exits only after the sole score producer finishes.

    /// Allocate bounded state before any producer can enqueue.
    explicit Impl( const OnlineShadowConfig &config /**< [in] Fixed queue/model bounds. */ );

    /// Try to copy one adapter event without blocking its producer.
    bool push( const Event &event /**< [in] Fixed-size evidence packet. */ );

    /// Own all model work and enforce the soft cycle budget with explicit overrun invalidation.
    void run();

    /// Copy a scalar without blocking or affecting numerical state.
    void prediction( const ShadowPrediction &sample /**< [in] Frozen-origin score. */ ) noexcept override;

    /// Drain bounded scores to a private CSV without holding numerical/acquisition locks.
    void record();
};

OnlineShadowWorker::Impl::Impl( const OnlineShadowConfig &c )
    : m_config( c ), m_engine( c ), m_raw( c.m_queueCapacity ), m_events( c.m_pendingCapacity * 8 ),
      m_predictions( c.m_predictionCapacity )
{
    if( c.m_predictionCapacity )
        m_engine.prediction_sink( this );
}
void OnlineShadowWorker::Impl::prediction( const ShadowPrediction &sample ) noexcept
{
    m_predictions.push( sample );
}
void OnlineShadowWorker::Impl::record()
{
    sched_param priority{};
    if( pthread_setschedparam( pthread_self(), SCHED_OTHER, &priority ) != 0 )
        m_traceIoError.store( true );
    ShadowPrediction r;
    while( !m_numericalDone.load( std::memory_order_acquire ) || m_predictions.queued() )
    {
        for( unsigned n = 0; n < 256 && m_predictions.pop( r ); ++n )
        {
            if( m_traceRows.load() >= m_config.m_predictionMaxRows )
            {
                m_traceLimit.store( true );
                continue;
            }
            if( m_traceIoError.load() )
                continue;
            if( std::fprintf( m_trace,
                              "%u,%llu,%llu,%u,%u,%.17g,%.17g,%.17g,%u,%u,%llu,%llu,%llu\n",
                              unsigned( r.m_mode ),
                              (unsigned long long)r.m_origin,
                              (unsigned long long)r.m_target,
                              r.m_horizon,
                              r.m_channel,
                              r.m_originPrediction,
                              r.m_prediction,
                              r.m_measurement,
                              unsigned( r.m_originValid ),
                              unsigned( r.m_late ),
                              (unsigned long long)r.m_generation,
                              (unsigned long long)r.m_epoch,
                              (unsigned long long)r.m_layout ) < 0 )
                m_traceIoError.store( true );
            else
                ++m_traceRows;
        }
        std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
    }
    if( std::fflush( m_trace ) != 0 )
        m_traceIoError.store( true );
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
            const auto                                                         begin = now_ns();
            std::optional<MultirateContract>                                   contract;
            std::optional<std::pair<MagAOX::app::synco::Record, std::int64_t>> schedule;
            { //mutex scope
                std::lock_guard<std::mutex> lock( m_mailbox );
                contract.swap( m_contract );
                schedule.swap( m_schedule );
                if( m_native && m_syncStatus )
                {
                    m_native->sync( *m_syncStatus );
                    m_syncStatus.reset();
                }
            }
            if( contract && !fault )
                m_engine.qualify( *contract );
            if( m_native && schedule )
                m_native->schedule( schedule->first, schedule->second, now_ns() );
            m_engine.request( m_requested.load() ? ShadowMode::AccelAssisted : ShadowMode::Standard );
            const auto drops = m_drops.load() + m_raw.drops() + ( m_commandInputs ? m_commandInputs->drops() : 0 );
            if( drops != observedDrops )
            {
                m_engine.invalidate( "online_queue_loss_new_history_required" );
                if( m_native )
                    m_native->invalidate();
                observedDrops = drops;
            }
            NativeObservation raw;
            for( unsigned n = 0; !fault && n < 1024 && now_ns() - begin < std::int64_t( m_config.m_budgetUs ) * 1000;
                 ++n )
            {
                if( !m_raw.pop( raw ) )
                    break;
                const auto start  = now_ns();
                const auto before = m_engine.status();
                if( m_native )
                {
                    if( raw.m_source == ObservationSource::Count )
                        m_native->tick( raw.m_receivedNs );
                    else
                    {
                        if( raw.m_source == ObservationSource::Dm && raw.m_event == ObservationEvent::Sample )
                        {
                            bool matched = false;
                            for( unsigned discarded = 0; discarded < m_config.m_pendingCapacity * 8; ++discarded )
                            {
                                if( !m_commandPending )
                                {
                                    NativeObservation input;
                                    if( !m_commandInputs->pop( input, m_commandPayload ) )
                                        break;
                                    m_commandPending = input;
                                }
                                const auto &input = *m_commandPending;
                                if( input.m_counter == raw.m_counter && input.m_generation == raw.m_generation &&
                                    input.m_device == raw.m_device && input.m_fileInode == raw.m_fileInode &&
                                    input.m_imageInode == raw.m_imageInode &&
                                    input.m_payloadHash == raw.m_payloadHash && input.m_elements == raw.m_elements )
                                {
                                    m_native->command(
                                        input, std::span<const float>( m_commandPayload.data(), input.m_elements ) );
                                    m_commandPending.reset();
                                    matched = true;
                                    break;
                                }
                                if( input.m_availableNs > raw.m_availableNs )
                                    break; // Keep a future payload for its own metadata.
                                m_commandPending
                                    .reset(); // Old metadata was lost; do not shift all following associations.
                            }
                            if( !matched )
                                m_native->invalidate();
                        }
                        m_native->observe( raw, start );
                    }
                }
                else
                    m_engine.observe( raw );
                const auto work = double( now_ns() - start ) / 1000;
                if( m_native && raw.m_source == ObservationSource::Modal && raw.m_event == ObservationEvent::Sample )
                    m_engine.finish_origin( now_ns() );
                const auto after = m_engine.status();
                if( after.m_records != before.m_records || after.m_forecasts != before.m_forecasts )
                {
                    compute = work;
                    peak    = std::max( peak, work );
                    if( work > m_config.m_budgetUs )
                    {
                        fault = true;
                        m_engine.invalidate( "single_numerical_job_exceeded_budget_restart_required" );
                        if( m_native )
                            m_native->invalidate();
                    }
                }
            }
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
                else if( event.m_commands )
                    m_engine.commands( event.m_frame, start );
                else
                    m_engine.applied( event.m_input, start );
                const auto finished = now_ns();
                if( event.m_origin )
                    m_engine.finish_origin( finished );
                const auto after = m_engine.status();
                if( after.m_qualified )
                {
                    const auto work = double( finished - start ) / 1000;
                    if( after.m_records != before.m_records || after.m_candidates != before.m_candidates ||
                        after.m_forecasts != before.m_forecasts )
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
            if( m_native )
                m_native->status( status );
            if( fault )
            {
                status.m_reason          = "worker_policy_or_compute_budget_fault_restart_required";
                status.m_candidateValid  = false;
                status.m_predictionValid = false;
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
        if( m_shadowProducer )
            m_shadowProducer->inhibit( false );
        std::lock_guard<std::mutex> lock( m_snapshot );
        m_status.m_reason          = "online_worker_exception_restart_required";
        m_status.m_candidateValid  = false;
        m_status.m_predictionValid = false;
        m_status.m_overrun         = true;
        m_status.m_qualified       = false;
        m_status.m_standard = m_status.m_assisted = ShadowStatus{};
    }
    std::lock_guard<std::mutex> lock( m_snapshot );
    if( m_shadowProducer )
        m_shadowProducer->inhibit( false );
    m_status.m_running         = false;
    m_status.m_candidateValid  = false;
    m_status.m_predictionValid = false;
    m_numericalDone.store( true, std::memory_order_release );
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
    m_impl->m_predictions.reset();
    m_impl->m_commandPending.reset();
    NativeObservation old;
    while( m_impl->m_raw.pop( old ) )
    {
    } // Both producers are stopped; no old session can be relabelled on restart.
    if( m_impl->m_commandInputs )
        while( m_impl->m_commandInputs->pop( old, m_impl->m_commandPayload ) )
        {
        }
    m_impl->m_traceRows.store( 0 );
    m_impl->m_traceIoError.store( false );
    m_impl->m_traceLimit.store( false );
    if( m_impl->m_nativeConfig.m_enabled )
    {
        m_impl->m_native =
            std::make_unique<NativeInputPipeline>( m_impl->m_engine, m_impl->m_nativeConfig, m_impl->m_config.m_model );
        m_impl->m_shadowProducer = m_impl->m_native->producer();
        m_impl->m_lastNative     = {};
    }
    { //mutex scope
        std::lock_guard<std::mutex> lock( m_impl->m_snapshot );
        m_impl->m_status = {};
        for( auto &plan : m_impl->m_plans )
            plan.resize( 0, 0 );
    }
    { //mutex scope
        std::lock_guard<std::mutex> lock( m_impl->m_mailbox );
        m_impl->m_contract.reset();
        m_impl->m_schedule.reset();
        m_impl->m_syncStatus.reset();
        m_impl->m_head = m_impl->m_tail = m_impl->m_count = 0;
    }
    m_impl->m_running.store( true );
    m_impl->m_numericalDone.store( false );
    try
    {
        if( !m_impl->m_config.m_predictionFile.empty() )
        {
            const auto &path = m_impl->m_config.m_predictionFile;
            if( path[0] != '/' || !m_impl->m_config.m_predictionCapacity || !m_impl->m_config.m_predictionMaxRows ||
                m_impl->m_config.m_predictionMaxRows > 10000000 )
                throw std::invalid_argument(
                    "Prediction CSV requires a private absolute path and bounded score queue" );
            const auto actual = ( std::filesystem::canonical( std::filesystem::path( path ).parent_path() ) /
                                  std::filesystem::path( path ).filename() )
                                    .string();
            for( const auto prefix : { "/opt/MagAOX", "/milk/shm", "/dev/shm" } )
                if( actual == prefix || actual.starts_with( std::string( prefix ) + "/" ) )
                    throw std::invalid_argument( "Prediction CSV cannot use instrument or shared-memory directories" );
            const int fd = open( actual.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600 );
            if( fd < 0 )
                throw std::runtime_error( "Cannot exclusively create private prediction CSV" );
            m_impl->m_trace = fdopen( fd, "w" );
            if( !m_impl->m_trace )
            {
                close( fd );
                throw std::runtime_error( "Cannot open private prediction CSV" );
            }
            if( std::fprintf( m_impl->m_trace,
                              "mode,origin,target,horizon,channel,origin_prediction,conditional_prediction,measurement,"
                              "origin_valid,late,generation,epoch,layout\n" ) < 0 )
                throw std::runtime_error( "Cannot write private prediction CSV header" );
        }
        m_impl->m_thread = std::thread( &Impl::run, m_impl.get() );
        if( m_impl->m_trace )
            m_impl->m_recorder = std::thread( &Impl::record, m_impl.get() );
    }
    catch( ... )
    {
        m_impl->m_running.store( false );
        if( m_impl->m_thread.joinable() )
            m_impl->m_thread.join();
        m_impl->m_numericalDone.store( true );
        if( m_impl->m_trace )
        {
            std::fclose( m_impl->m_trace );
            m_impl->m_trace = nullptr;
        }
        throw;
    }
}
void OnlineShadowWorker::stop()
{
    m_impl->m_running.store( false );
    if( m_impl->m_shadowProducer )
        m_impl->m_shadowProducer->inhibit( false );
    if( m_impl->m_thread.joinable() )
        m_impl->m_thread.join();
    if( m_impl->m_recorder.joinable() )
        m_impl->m_recorder.join();
    if( m_impl->m_trace )
    {
        if( std::fclose( m_impl->m_trace ) != 0 )
            m_impl->m_traceIoError.store( true );
        m_impl->m_trace = nullptr;
    }
}
void OnlineShadowWorker::observe( const NativeObservation &record ) noexcept
{
    const auto n = std::size_t( record.m_source );
    if( m_impl->m_shadowProducer && n < 4 && n != 2 )
    {
        const auto &last = m_impl->m_lastNative[n];
        if( record.m_event != ObservationEvent::Sample || ( record.m_flags & ObservationInvalidPayload ) ||
            ( last.m_counter &&
              ( record.m_counter != last.m_counter + 1 || record.m_generation != last.m_generation ) ) )
            m_impl->m_shadowProducer->inhibit( n == 0, record.m_availableNs );
        m_impl->m_lastNative[n] = record;
    }
    if( !m_impl->m_raw.push( record ) && m_impl->m_shadowProducer )
        m_impl->m_shadowProducer->inhibit( false, record.m_availableNs );
}
void OnlineShadowWorker::native_inputs( const NativePipelineConfig &c )
{
    if( m_impl->m_thread.joinable() || m_impl->m_running.load() )
        throw std::logic_error( "Native input convention is startup-only" );
    m_impl->m_nativeConfig = c;
    if( c.m_enabled )
    {
        m_impl->m_commandInputs = std::make_unique<NativeCommandQueue>( m_impl->m_config.m_pendingCapacity * 8, 4096 );
        m_impl->m_commandPayload.resize( 4096 );
    }
}
void OnlineShadowWorker::command( const NativeObservation &r, std::span<const float> values ) noexcept
{
    if( m_impl->m_commandInputs && !m_impl->m_commandInputs->push( r, values ) && m_impl->m_shadowProducer )
        m_impl->m_shadowProducer->inhibit( false, r.m_availableNs );
}
void OnlineShadowWorker::tick( std::int64_t now ) noexcept
{
    if( !m_impl->m_nativeConfig.m_enabled )
        return;
    if( m_impl->m_shadowProducer )
    {
        if( now - m_impl->m_lastNative[1].m_availableNs > 1000000 )
            m_impl->m_shadowProducer->inhibit( false, now );
        else if( now - m_impl->m_lastNative[0].m_availableNs > 1000000 )
            m_impl->m_shadowProducer->inhibit( true, now );
    }
    NativeObservation tick;
    tick.m_source     = ObservationSource::Count;
    tick.m_receivedNs = tick.m_availableNs = now;
    if( !m_impl->m_raw.push( tick ) && m_impl->m_shadowProducer )
        m_impl->m_shadowProducer->inhibit( false, now );
}
void OnlineShadowWorker::schedule( const MagAOX::app::synco::Record &s, std::int64_t received ) noexcept
{
    if( !m_impl->m_running.load() )
        return;
    if( m_impl->m_shadowProducer )
        m_impl->m_shadowProducer->callback( s, received );
    std::unique_lock<std::mutex> lock( m_impl->m_mailbox, std::try_to_lock );
    if( lock.owns_lock() )
        m_impl->m_schedule = std::make_pair( s, received );
}
bool OnlineShadowWorker::sync_status( const NativeSyncStatus &s )
{
    if( !m_impl->m_running.load() || !m_impl->m_nativeConfig.m_enabled )
        return false;
    if( m_impl->m_shadowProducer && ( !s.m_status.m_locked || !s.m_status.m_synchronized ) )
        m_impl->m_shadowProducer->inhibit( true, s.m_status.m_receivedNs );
    std::unique_lock<std::mutex> lock( m_impl->m_mailbox, std::try_to_lock );
    if( !lock.owns_lock() )
        return false;
    m_impl->m_syncStatus = s;
    return true;
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
bool OnlineShadowWorker::commands( const OnlineCommandFrame &r )
{
    if( !m_impl->m_running.load() || r.m_evidence != InputEvidence::Requested )
        return false;
    Impl::Event event;
    event.m_commands = true;
    event.m_frame    = r;
    return m_impl->push( event );
}

bool OnlineShadowWorker::qualify( const MultirateContract &contract )
{
    if( !m_impl->m_running.load() || !native_shadow_evidence( contract ) ||
        contract.m_upstream.m_enabled != m_impl->m_config.m_expectUpstream || contract.m_provenance.size() > 4096 )
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
    s.m_drops                     = m_impl->m_drops.load() + m_impl->m_raw.drops() +
                                    ( m_impl->m_commandInputs ? m_impl->m_commandInputs->drops() : 0 );
    s.m_traceRows                 = m_impl->m_traceRows.load();
    s.m_traceIoError              = m_impl->m_traceIoError.load();
    s.m_traceLimit                = m_impl->m_traceLimit.load();
    s.m_predictionsQueued         = m_impl->m_predictions.queued();
    s.m_predictionDrops           = m_impl->m_predictions.drops();
    if( m_impl->m_config.m_expectUpstream && !s.m_qualified )
    {
        s.m_upstream = true;
        if( std::string_view( s.m_reason ) == "source_host_mapping_unqualified" )
            s.m_reason = "upstream_sync_and_command_contract_missing";
    }
    if( s.m_candidateValid && now_ns() > s.m_deadlineNs )
        s.m_candidateValid = false;
    return s;
}
bool OnlineShadowWorker::read_prediction( ShadowPrediction &sample )
{
    if( !m_impl->m_config.m_predictionFile.empty() )
        return false; // Exactly one SPSC consumer.
    return m_impl->m_predictions.pop( sample );
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
