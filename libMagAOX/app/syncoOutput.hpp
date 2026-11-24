/** \file syncoOutput.hpp
 * \brief Bounded enable/zero/acknowledgement handling around the existing syncopation producer; no scheduler.
 */
#ifndef MagAOX_app_syncoOutput_hpp
#define MagAOX_app_syncoOutput_hpp
#include "dev/dmSyncopationProducer.hpp"
#include <atomic>
#include <chrono>
#include <cmath>
#include <functional>
#include <mutex>
#include <memory>
#include <stdexcept>
#include <algorithm>
#include <span>
#include <thread>
#include <vector>

namespace MagAOX::app
{
/// Optional commissioning bounds; operational applications may select the existing producer's finite-value/ack policy.
struct SyncoOutputConfig
{
    bool         m_authorized{ false }, m_routingVerified{ false }; ///< Both are startup-only operator assertions.
    double       m_maxAbs{ 0 }, m_maxStep{ 0 }; ///< Approved full-image surface-micron limits; no guessed defaults.
    std::int64_t m_ackTimeoutNs{
        0 }; ///< Strict bounds require a positive timeout; producer defaults permit zero/base-only ack wait.
    bool m_producerDefaults{ false }; ///< App-selected finite-value/base-ack policy; never an INDI/config trust switch.
    std::uint32_t m_width{ 0 },
        m_height{ 0 }; ///< Optional exact geometry from a two-dimensional synthesis calibration.
};

/// Copyable requested/transport-acknowledged state; never a physical application certificate.
struct SyncoOutputStatus
{
    bool m_requested{ false }, m_acknowledged{ false }, m_zeroConfirmed{ true },
        m_fault{ false }; ///< Operator intent, last copied contribution, removal and latched fault.
    std::uint64_t m_published{ 0 }, m_acknowledgements{ 0 },
        m_rejected{ 0 };          ///< Committed payloads, exact receipts and failed bounds.
    std::uint64_t m_skipped{ 0 }; ///< Observed slots not committed by their cutoff; synco retains the last copy.
    const char   *m_reason{ "disabled" }; ///< Stable diagnostic text, without acquisition-thread string allocation.
    const char   *m_faultReason{ "" };    ///< Why the latest fault latched; kept after its zero is acknowledged.
    std::uint64_t m_faults{ 0 };          ///< Faults latched since startup, so a reader can detect each new one.
};

/// Small deterministic output interlock reused by the two native providers and mock consumers.
class SyncoOutputState
{
    SyncoOutputConfig m_config; ///< Immutable authorization and physical limits.
    SyncoOutputStatus m_status; ///< Serialized by its owner, never concurrently accessed without a lock.
    bool m_permitted{ false }, m_pending{ false }, m_pendingZero{ true }, m_initialZero{ false }; ///< Output ownership.
    std::int64_t       m_sent{ -1 }; ///< Actual commit time, used only for acknowledgement timeout.
    std::vector<float> m_last;       ///< Last submitted own contribution, never another application's command.
  public:
    /// Allocate fixed storage before any producer thread starts.
    SyncoOutputState( const SyncoOutputConfig &config /**< [in] Reviewed authorization. */,
                      std::size_t              pixels /**< [in] Complete own auxiliary image. */,
                      bool permitted /**< [in] Compile-time output eligibility supplied by the native wrapper. */ );
    /// Request on only with current qualification; off always schedules removal of this contribution.
    bool request( bool enabled /**< [in] Explicit operator request. */,
                  bool qualified /**< [in] Current input/model gate. */ );
    /// Latch a fault and request zero; a later explicit on request is necessary to recover.
    void fault( const char *reason /**< [in] Stable diagnostic string. */ );
    /// Explicitly request only a tagged initial zero; never permits a nonzero candidate.
    bool initialize();
    /// Require a tagged initial zero after exclusive attachment, before any nonzero contribution.
    void attached();
    /// Release abandoned transport ownership on an observed scheduler incarnation change, without fabricating an ack.
    void transport_reset();
    /// Clear only an acknowledged, removed fault; never enables output.
    bool clear_fault();
    /// Observe an exact matching transport acknowledgement, separately from physical application.
    void acknowledge( bool copied /**< [in] Consumer copied rather than discarded the pending payload. */ );
    /// Check the pending receipt timeout and current qualification without inventing a new write opportunity.
    void check( std::int64_t now /**< [in] Same-host monotonic time. */, bool qualified /**< [in] Current gate. */ );
    /// Validate finite values and any strict DDSPC bounds; zero removal bypasses the active slew limit.
    bool validate( std::span<const float> pixels /**< [in] Prepared complete image. */,
                   bool                   zero /**< [in] Removal command. */ );
    /// Mark a successfully committed command, awaiting its exact matching receipt.
    void submitted( std::span<const float> pixels /**< [in] Submitted own-channel values. */,
                    bool                   zero /**< [in] Explicit removal. */,
                    std::int64_t           now /**< [in] Actual completion. */ );
    /// Count an observed slot that could not be committed by its cutoff; the consumer retains the last copy.
    void skipped();
    /// Return whether a zero must be submitted before another active command.
    bool zero_needed() const;
    /// Return whether this owner can submit a new command.
    bool pending() const;
    /// Copy current public diagnostics.
    SyncoOutputStatus status() const;
};

inline SyncoOutputState::SyncoOutputState( const SyncoOutputConfig &c, std::size_t n, bool permitted )
    : m_config( c ), m_permitted( permitted ), m_last( n, 0 )
{
}
inline bool SyncoOutputState::request( bool enabled, bool qualified )
{
    if( !enabled )
    {
        m_status.m_requested = false;

        m_status.m_reason = m_status.m_zeroConfirmed ? "disabled" : "zero_pending";
        return true;
    }
    if( !m_permitted || !m_config.m_authorized || !m_config.m_routingVerified || !qualified ||
        ( !m_config.m_producerDefaults &&
          ( !std::isfinite( m_config.m_maxAbs ) || !std::isfinite( m_config.m_maxStep ) || m_config.m_maxAbs <= 0 ||
            m_config.m_maxStep <= 0 || m_config.m_ackTimeoutNs <= 0 ) ) ||
        ( m_status.m_fault && !m_status.m_zeroConfirmed ) )
    {
        m_status.m_reason = m_config.m_producerDefaults ? "authorization_route_or_qualification_missing"
                                                        : "authorization_limits_route_or_qualification_missing";
        return false;
    }
    m_status.m_fault     = false;
    m_status.m_requested = true;
    m_status.m_reason    = "enable_requested";
    return true;
}
inline void SyncoOutputState::fault( const char *reason )
{
    m_status.m_fault         = true;
    m_status.m_requested     = false;
    m_status.m_zeroConfirmed = false;
    m_status.m_reason        = reason;
    m_status.m_faultReason   = reason;
    ++m_status.m_faults;
}
inline bool SyncoOutputState::initialize()
{
    if( !m_permitted || !m_config.m_authorized || !m_config.m_routingVerified ||
        ( !m_config.m_producerDefaults &&
          ( !std::isfinite( m_config.m_maxAbs ) || !std::isfinite( m_config.m_maxStep ) || m_config.m_maxAbs <= 0 ||
            m_config.m_maxStep <= 0 || m_config.m_ackTimeoutNs <= 0 ) ) )
        return false;
    if( !m_status.m_requested && m_status.m_zeroConfirmed )
        attached();
    return true;
}
inline void SyncoOutputState::attached()
{
    m_initialZero            = true;
    m_status.m_zeroConfirmed = false;
}
inline void SyncoOutputState::transport_reset()
{
    m_pending = false;
    fault( "scheduler_reset_zero_pending" );
}
inline bool SyncoOutputState::clear_fault()
{
    if( m_pending || !m_status.m_zeroConfirmed || m_status.m_requested )
        return false;
    m_status.m_fault  = false;
    m_status.m_reason = "disabled_zero_acknowledged";
    return true;
}
inline void SyncoOutputState::acknowledge( bool copied )
{
    if( !m_pending )
        return;
    m_pending = false;
    ++m_status.m_acknowledgements;
    if( !copied )
    {
        fault( "command_discarded_zero_pending" );
        return;
    }
    m_status.m_zeroConfirmed = m_pendingZero;
    m_status.m_acknowledged  = !m_pendingZero;
    if( m_pendingZero )
    {
        std::fill( m_last.begin(), m_last.end(), 0 );
        m_initialZero = false;
    }
    m_status.m_reason = m_pendingZero ? ( m_status.m_fault ? "fault_zero_acknowledged" : "disabled_zero_acknowledged" )
                                      : "transport_acknowledged_not_physical_application";
}
inline void SyncoOutputState::check( std::int64_t now, bool qualified )
{
    if( m_status.m_requested && !qualified )
        fault( "input_or_model_qualification_lost" );
    if( m_pending && m_config.m_ackTimeoutNs > 0 && now - m_sent > m_config.m_ackTimeoutNs )
        fault( "ack_timeout_zero_unconfirmed" );
}
inline bool SyncoOutputState::validate( std::span<const float> pixels, bool zero )
{
    if( m_pending || pixels.size() != m_last.size() || ( !zero && !m_status.m_requested ) )
        return false;
    for( std::size_t p = 0; p < pixels.size(); ++p )
        if( !std::isfinite( pixels[p] ) ||
            ( zero ? pixels[p] != 0
                   : ( !m_config.m_producerDefaults && ( std::abs( pixels[p] ) > m_config.m_maxAbs ||
                                                         std::abs( pixels[p] - m_last[p] ) > m_config.m_maxStep ) ) ) )
        {
            ++m_status.m_rejected;
            fault( "nonfinite_or_command_limit" );
            return false;
        }
    return true;
}
inline void SyncoOutputState::submitted( std::span<const float> pixels, bool zero, std::int64_t now )
{
    std::copy( pixels.begin(), pixels.end(), m_last.begin() );
    m_pending                = true;
    m_pendingZero            = zero;
    m_sent                   = now;
    m_status.m_zeroConfirmed = false;
    ++m_status.m_published;
}
inline void SyncoOutputState::skipped()
{
    ++m_status.m_skipped;
}
inline bool SyncoOutputState::zero_needed() const
{
    return m_initialZero || ( !m_status.m_requested && !m_status.m_zeroConfirmed );
}
inline bool SyncoOutputState::pending() const
{
    return m_pending;
}
inline SyncoOutputStatus SyncoOutputState::status() const
{
    return m_status;
}

/// Existing producer transport with its own shutdown lifetime so framework shutdown can still submit a final zero.
/** No output image is created. No source access occurs until explicit authorization and an on or zero-initialization
 * request. The contained CRTP lifetime is necessary because MagAOXApp sets its shutdown flag before appShutdown(). All
 * writes, slot selection, sidecars and cooperative ownership use the unchanged producer base.
 */
class SyncoOutput : protected dev::dmSyncopationProducer<SyncoOutput>
{
    friend class dev::dmSyncopationProducer<SyncoOutput>;
    bool        m_shutdown{ false }; ///< Producer lifetime, independent of the enclosing application's stop request.
    IMAGE       m_source{}; ///< Existing exclusively owned auxiliary mapping; never created or zeroed directly.
    std::string m_name;     ///< One exact configured auxiliary identity, never a combiner destination.
    SyncoOutputConfig                 m_config; ///< Reviewed immutable startup authorization.
    std::unique_ptr<SyncoOutputState> m_state;  ///< Serialized own-channel state, allocated before start.
    std::vector<float>                m_pixels; ///< Producer-owned preallocated whole-image workspace.
    mutable std::mutex                m_mutex;  ///< Serializes intent, receipts and payload commit.
    std::atomic<bool>     m_running{ false }, m_stop{ false }; ///< Thread lifetime and bounded removal request.
    std::thread           m_thread;    ///< Sole base-class transport owner, joined before destruction.
    std::function<bool()> m_qualified; ///< Read current application eligibility without acquiring output intent.
    std::function<bool( const synco::Request &, const synco::Record &, std::int64_t, std::span<float> )>
        m_candidate; ///< Copy-only prepared image callback; never fits a model.
    std::function<void( const synco::CommitRecord &, std::span<const float> )>
        m_copied; ///< Optional owner notification after an exact copied acknowledgement; no physical-application claim.
    /// Own base poll/publish calls; no independent slot clock is generated.
    void run();

  public:
    /// Destruction stops only this provider and attempts bounded protocol removal.
    ~SyncoOutput();
    /// Configure before startup; defaults never permit an output attachment.
    void configure( const SyncoOutputConfig &config /**< [in] Reviewed bounds. */,
                    const std::string       &name /**< [in] Exact own auxiliary name. */,
                    const std::string       &timing /**< [in] Existing scheduler stream. */,
                    double                   leadUs /**< [in] Existing producer lead setting. */,
                    std::size_t              pixels /**< [in] Calibrated full image. */ );
    /// Start only a local producer thread; strict shadow builds never start it or attach streams.
    /** With ownThread false the application runs work() on its own framework thread, e.g. one started by
     * MagAOXApp::threadStart with a configured real-time priority and cpuset.
     */
    void start( std::function<bool()> qualified /**< [in] Current input/model gate. */,
                std::function<bool( const synco::Request &, const synco::Record &, std::int64_t, std::span<float> )>
                    candidate /**< [in] Copy an already prepared exact-slot image. */,
                std::function<void( const synco::CommitRecord &, std::span<const float> )> copied =
                    {} /**< [in] Optional nonblocking notification of the copied commit and own contribution. */,
                bool ownThread = true /**< [in] Start an internal std::thread; false defers to work(). */ );
    /// Run the producer loop on the caller's thread after start(..., false); returns after stop() completes removal.
    void work();
    /// Prepare an authorized own-channel zero after an explicit learning request; does not enable correction.
    bool initialize();
    /// Explicit INDI intent; startup always remains off.
    bool request( bool enabled /**< [in] Operator request. */ );
    /// Stop after zero acknowledgement or the bounded timeout; return false if removal could not be confirmed.
    bool stop();
    /// Explicit reset after confirmed zero; never reactivates output.
    bool clear_fault();
    /// Copy requested and receipt state for ordinary-rate INDI diagnostics.
    SyncoOutputStatus status() const;
};
inline SyncoOutput::~SyncoOutput()
{
    stop();
}
inline void SyncoOutput::configure(
    const SyncoOutputConfig &c, const std::string &name, const std::string &timing, double lead, std::size_t pixels )
{
    std::lock_guard<std::mutex> lock( m_mutex );
    if( m_thread.joinable() )
        throw std::logic_error( "Output configuration is startup-only" );
    m_config            = c;
    m_name              = name;
    m_syncopationTiming = timing;
    m_writeLeadUs       = lead;
    m_pixels.assign( pixels, 0 );
#if defined( DDSPC_OPERATIONAL_BUILD ) && !defined( DDSPC_SHADOW_ONLY )
    constexpr bool permitted = true;
#else
    constexpr bool permitted = false;
#endif
    m_state = std::make_unique<SyncoOutputState>( c, pixels, permitted );
}
inline void SyncoOutput::start(
    std::function<bool()>                                                                                qualified,
    std::function<bool( const synco::Request &, const synco::Record &, std::int64_t, std::span<float> )> candidate,
    std::function<void( const synco::CommitRecord &, std::span<const float> )>                           copied,
    bool                                                                                                 ownThread )
{
    std::lock_guard<std::mutex> lock( m_mutex );
    if( m_thread.joinable() )
        throw std::logic_error( "Output already started" );
    m_qualified = std::move( qualified );
    m_candidate = std::move( candidate );
    m_copied    = std::move( copied );
#if defined( DDSPC_OPERATIONAL_BUILD ) && !defined( DDSPC_SHADOW_ONLY )
    if( m_config.m_authorized && m_config.m_routingVerified && !m_thread.joinable() )
    {
        m_stop.store( false );
        m_running.store( true );
        m_shutdown = false;
        if( ownThread )
            m_thread = std::thread( &SyncoOutput::run, this );
    }
#else
    static_cast<void>( ownThread );
#endif
}
inline void SyncoOutput::work()
{
    run();
}
inline bool SyncoOutput::initialize()
{
    std::lock_guard<std::mutex> lock( m_mutex );
    return m_state && m_state->initialize();
}
inline bool SyncoOutput::request( bool enabled )
{
    std::function<bool()> gate;
    { //mutex scope
        std::lock_guard<std::mutex> lock( m_mutex );
        if( enabled )
            gate = m_qualified;
    }
    bool qualified = !enabled;
    try
    {
        if( enabled )
            qualified = gate && gate();
    }
    catch( ... )
    {
        qualified = false;
    }
    std::lock_guard<std::mutex> lock( m_mutex );
    return m_state && m_state->request( enabled, qualified );
}
inline bool SyncoOutput::clear_fault()
{
    std::lock_guard<std::mutex> lock( m_mutex );
    return m_state && m_state->clear_fault();
}
inline SyncoOutputStatus SyncoOutput::status() const
{
    std::lock_guard<std::mutex> lock( m_mutex );
    return m_state ? m_state->status() : SyncoOutputStatus{};
}
inline bool SyncoOutput::stop()
{
    request( false );
    m_stop.store( true );
    if( m_thread.joinable() )
        m_thread.join();
    return status().m_zeroConfirmed;
}
inline void SyncoOutput::run()
{
#if defined( DDSPC_OPERATIONAL_BUILD ) && !defined( DDSPC_SHADOW_ONLY )
    synco::Request pending;
    bool           haveRequest = false;
    std::int64_t   stopping    = -1;
    // Called with the output mutex held, before the single payload buffer may be reused.
    auto receive = [&]( const synco::CommitRecord &commit )
    {
        synco::AckRecord ack{};
        if( m_state->pending() && synco::readRecord( m_syncopationAck, ack ) && synco::acknowledges( ack, commit ) )
        {
            const bool copied = ack[synco::ackStatus] == synco::copied;
            m_state->acknowledge( copied );
            if( copied && m_copied )
            {
                try
                {
                    m_copied( commit, m_pixels );
                }
                catch( ... )
                {
                    m_state->fault( "copy_callback_exception_zero_pending" );
                }
            }
        }
    };
    while( m_running.load() )
    {
        const auto now = synco::monotonicNs();
        if( m_stop.load() && stopping < 0 )
            stopping = now;
        bool wanted, zero, qualified = false;
        try
        {
            qualified = m_qualified && m_qualified();
        }
        catch( ... )
        {
            qualified = false;
        }
        { //mutex scope
            std::lock_guard<std::mutex> lock( m_mutex );
            if( m_syncopationOutstanding )
                receive( m_syncopationPending );
            m_state->check( now, qualified );
            const auto state = m_state->status();
            wanted           = state.m_requested;
            zero             = m_state->zero_needed();
            if( m_stop.load() && ( state.m_zeroConfirmed ||
                                   now - stopping > std::max<std::int64_t>( 100000000, 2 * m_config.m_ackTimeoutNs ) ) )
                break;
        }
        if( !wanted && !zero )
        {
            std::this_thread::sleep_for( std::chrono::microseconds( 50 ) );
            continue;
        }
        if( !m_source.md )
        {
            if( m_name.empty() || ImageStreamIO_openIm( &m_source, m_name.c_str() ) != 0 )
            {
                { //mutex scope
                    std::lock_guard<std::mutex> lock( m_mutex );
                    m_state->fault( "auxiliary_unavailable" );
                }
                std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
                continue;
            }
            if( !syncopationGeometry( m_source ) || m_source.md->nelement != m_pixels.size() ||
                ( m_config.m_width && m_source.md->size[0] != m_config.m_width ) ||
                ( m_config.m_height && m_source.md->size[1] != m_config.m_height ) )
            {
                std::lock_guard<std::mutex> lock( m_mutex );
                m_state->fault( "auxiliary_geometry" );
                break;
            }
            { //mutex scope
                std::lock_guard<std::mutex> lock( m_mutex );
                m_state->attached();
                zero = true;
            }
        }
        const bool owned          = m_syncopationOutstanding;
        const auto previousCommit = m_syncopationPending;
        const int  polled         = haveRequest ? 0 : syncopationPoll( pending, m_source );
        if( owned && !m_syncopationOutstanding )
        { //mutex scope
            std::lock_guard<std::mutex> lock( m_mutex );
            receive( previousCommit ); // The receipt may have arrived during the base poll.
            zero = m_state->zero_needed();
            if( m_state->pending() )
            {
                m_state->transport_reset();
                zero = true;
            }
        }
        if( polled < 0 )
        {
            std::lock_guard<std::mutex> lock( m_mutex );
            m_state->fault( "syncopation_transport_unavailable" );
        }
        if( polled > 0 )
            haveRequest = true;
        const auto decision = synco::monotonicNs();
        if( haveRequest && decision >= pending.cutoffNs )
        { //mutex scope
            // A late slot is skipped, not faulted: synco retains the last copied contribution for it.
            std::lock_guard<std::mutex> lock( m_mutex );
            m_state->skipped();
            haveRequest = false;
        }
        bool published = false;
        if( haveRequest )
        {
            synco::Record schedule{};
            bool          available = false;
            if( zero )
            {
                std::fill( m_pixels.begin(), m_pixels.end(), 0 );
                available = true;
            }
            else if( synco::readRecord( m_syncopationSchedule, schedule ) )
            {
                try
                {
                    available = m_candidate && m_candidate( pending, schedule, decision, m_pixels );
                }
                catch( ... )
                { //mutex scope
                    std::lock_guard<std::mutex> lock( m_mutex );
                    m_state->fault( "candidate_exception_zero_pending" );
                }
            }
            if( available )
            {
                bool current = zero;
                try
                {
                    if( !zero )
                        current = m_qualified && m_qualified();
                }
                catch( ... )
                {
                    current = false;
                }
                // mutex scope
                std::lock_guard<std::mutex> lock( m_mutex );
                if( !current )
                {
                    m_state->fault( "input_or_model_qualification_lost" );
                    haveRequest = false;
                    continue;
                }
                // Recheck operator intent after preparation; disable never permits a late nonzero publication.
                if( !m_state->status().m_requested && !zero )
                {
                    haveRequest = false;
                    continue;
                }
                if( m_state->validate( m_pixels, zero ) )
                {
                    published = syncopationPublish( m_pixels.data(), pending, m_source );
                    if( published )
                        m_state->submitted( m_pixels, zero, synco::monotonicNs() );
                    else
                        m_state->skipped(); // Cutoff passed or the slot was revoked; transport loss faults in poll.
                    haveRequest = false;
                }
            }
        }
        // Poll promptly, as dmSpeckle does; after a commit the next slot is checked at once.
        if( !published )
            std::this_thread::sleep_for( std::chrono::microseconds( 10 ) );
    }
    m_shutdown = true;
    syncopationClose();
    if( m_source.md )
        ImageStreamIO_closeIm( &m_source );
    m_source = {};
    m_running.store( false );
#endif
}
} // namespace MagAOX::app
#endif
