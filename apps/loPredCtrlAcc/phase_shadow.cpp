/** \file phase_shadow.cpp
 * \brief Bounded native DDSPC phase contexts; upstream counter association, existing synco deadlines and no output.
 */
#include "phase_shadow.hpp"
#include "native_pipeline.hpp"
#include <algorithm>
#include <chrono>
#include <limits>
#include <mutex>
#include <stdexcept>

namespace DDSPC
{
namespace synco = MagAOX::app::synco;
namespace
{
/// Reject nonfinite source modes without interpreting unrelated camera clocks or exposure metadata.
bool finite_modes( const NativeObservation &r /**< [in] Observed payload. */, int modes /**< [in] Required modes. */ )
{
    for( int m = 0; m < modes; ++m )
        if( !std::isfinite( r.m_values[m] ) )
            return false;
    return true;
}
/// Measure elapsed work in the caller's local epoch, including deterministic virtual replay epochs.
std::int64_t elapsed( std::chrono::steady_clock::time_point start /**< [in] Work start. */ )
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>( std::chrono::steady_clock::now() - start ).count();
}
} // namespace

struct PhaseShadow::Impl
{
    struct Frame
    {
        std::uint64_t m_counter{ 0 }, m_epoch{ 0 }, m_adcEpoch{ 0 },
            m_revision{ 0 };                         ///< Real WFS identity and continuity segments.
        std::int64_t         m_available{ -1 };      ///< Actual modal availability, never exposure time.
        std::int64_t         m_closedNs{ -1 };       ///< Actual next WFS availability, distinct from request cutoffs.
        std::array<float, 2> m_y{}, m_single{};      ///< Observed WFS and single-rate requested-input sample.
        std::array<float, 8> m_u{};                  ///< Newest-slot-first original requests at advertised cutoffs.
        std::array<std::array<float, 128>, 4> m_a{}; ///< Distinct rolling raw ADC histories at each phase.
        std::array<bool, 4>                   m_adcValid{}; ///< Phase data available at its own decision.
        std::array<std::int64_t, 4>           m_cutoffs{};  ///< Original scheduler cutoffs; no synthetic sampler grid.
        std::array<unsigned, 4>               m_steps{
            0, 1, 2, 3 };             ///< Chronological target order, independent of protocol step labels.
        synco::Record m_schedule{};   ///< Coherent schedule bound by the declared WFS/cycle convention.
        unsigned      m_sampled{ 0 }; ///< Complete requested-input cutoff coverage.
        bool m_closed{ false }, m_scheduled{ false }, m_singleValid{ false }, m_valid{ false }; ///< Explicit coverage.
    };
    struct Forecast
    {
        std::uint64_t         m_origin{ 0 }, m_adc{ 0 }, m_version{ 0 }; ///< Frozen causal identities.
        std::uint64_t         m_revision{ 0 };           ///< Model-reset incarnation of this frozen forecast.
        unsigned              m_phase{ 0 }, m_mode{ 0 }; ///< Independent prediction context.
        std::int64_t          m_decision{ -1 }, m_completed{ -1 }, m_target{ -1 }; ///< Distinct local events.
        std::array<float, 16> m_prediction{}; ///< F actual future WFS targets, newest first.
        std::array<float, 2>  m_candidate{};  ///< Private next-slot candidate, never a training input.
        synco::Request        m_request;      ///< Actual future request, separate from ADC phase.
        bool                  m_valid{ false }, m_late{ true }; ///< Deadline evidence is separate from causal validity.
    };
    NativePipelineConfig m_inputs; ///< Immutable upstream identity/calibration contract.
    OnlineShadowConfig   m_config; ///< Fixed resource and learner dimensions.
    ShadowConfig         m_model;  ///< Same numerical kernels/objective for both model banks.
    std::array<std::unique_ptr<ShadowController>, 4> m_learners; ///< Standard in bank zero; four assisted phase views.
    std::array<std::array<std::unique_ptr<ShadowSnapshot>, 2>, 4> m_snapshots, m_work; ///< Immutable pointer exchange.
    mutable std::mutex m_snapshotMutex; ///< Short pointer exchange or bounded fast multiplication, never RLS/QR.
    std::mutex         m_syncMutex;     ///< INDI mailbox only; acquisition takes a nonblocking snapshot.
    NativeSyncStatus   m_sync{}, m_activeSync{};  ///< Actual upstream diagnostics, never cadence estimates.
    std::array<NativeObservation, 4> m_last{};    ///< Acquisition-owner source ordering/identity.
    std::array<NativeBinding, 4>     m_sources{}; ///< Initial source incarnations; replacement requires a new contract.
    std::array<NativeObservation, 64> m_adc{};    ///< At most 64 individual X/Y samples (16 WFS histories).
    unsigned m_adcNext{ 0 }, m_adcCount{ 0 }, m_goodAdc{ 0 }; ///< Counter-contiguous recovery, not phase locking.
    std::array<Frame, 64> m_frames{};                         ///< Bounded original WFS, phase and request histories.
    std::uint64_t         m_current{ 0 }, m_nextClose{ 0 };   ///< Real WFS high waters, never advanced by ADC arrival.
    std::array<ProjectedRequest, 256> m_requests{}; ///< Complete chronological request retention; idle is valid.
    unsigned           m_requestNext{ 0 }, m_requestCount{ 0 }; ///< Retention eviction is distinct from queue loss.
    std::int64_t       m_commandObservedThrough{ -1 }; ///< Actual coherent poll witness, including an idle stream.
    std::vector<Frame> m_queue;                        ///< SPSC WFS learning jobs; no per-ADC numerical jobs.
    std::atomic<std::size_t> m_head{ 0 }, m_tail{ 0 }; ///< Acquisition producer and numerical consumer cursors.
    ShadowPredictionQueue    m_scores;                 ///< Fast-owner matured scores, drained by the numerical owner.
    std::vector<Forecast>    m_forecasts; ///< Bounded F-plus-one phase origins, scored only against actual WFS frames.
    std::uint64_t m_epoch{ 1 }, m_adcEpoch{ 1 }, m_learningEpoch{ 0 }, m_learningAdcEpoch{ 0 }; ///< Segment identities.
    bool          m_sourceLost{ false }, m_adcSourceLost{ false }, m_wfsSilent{ false }; ///< Fail-closed restart gates.
    std::atomic<bool>          m_learning{ false }; ///< Explicit learning intent; ADC inference never changes weights.
    std::atomic<std::uint64_t> m_reset{ 0 };        ///< Model incarnation visible immediately to the fast owner.
    std::atomic<std::uint64_t> m_resetPublished{ 0 };                   ///< Fast-owner reset barrier.
    std::uint64_t              m_resetApplied{ 0 };                     ///< Numerical-owner reset high water.
    std::atomic<bool>          m_fault{ false }, m_adcAllowed{ false }; ///< Cross-owner numerical/assisted inhibition.
    std::atomic<bool> m_fastValid{ false }; ///< Latest opportunity remains valid only until input loss/staleness.
    std::atomic<std::uint64_t> m_commonEpoch{ 1 }; ///< Acquisition continuity visible to numerical readback.
    std::atomic<bool> m_bindingValid{ true };      ///< Source replacement cannot silently inherit old counter anchors.
    std::atomic<std::int64_t> m_lastPredictionNs{ -1 }; ///< Diagnostic freshness, independent of lifetime counts.
    std::atomic<std::int64_t> m_wfsAvailableNs{ -1 },
        m_adcAvailableNs{ -1 }; ///< Current source freshness for numerical readback.
    std::int64_t m_scheduleEpoch{ 0 },
        m_scheduleGeneration{ 0 };                    ///< Actual schedule replacement invalidates private plans.
    std::atomic<std::int64_t> m_scheduleExpiry{ -1 }; ///< Latest coherent scheduler expiry, without phase estimation.
    synco::Record m_currentSchedule{}; ///< Latest coherent read-only schedule, revalidated after private preparation.
    int m_firstStep{ -1 }; ///< Bound WFS-to-command phase identity; a changed phase requires a reviewed contract.
    std::atomic<unsigned>                m_requested{ 0 }, m_effective{ 0 }; ///< Selection does not change learning.
    std::shared_ptr<SyncoShadowProducer> m_producer;  ///< Existing private Request/payload cache only.
    OnlineShadowStatus                   m_status;    ///< Numerical-owner published status.
    std::array<double, 2>                m_squared{}; ///< Origin-only errors, never fed into learning.
    std::array<std::uint64_t, 2>         m_scored{};  ///< Finite scalar target counts.
    std::atomic<std::uint64_t>           m_opportunities{ 0 }, m_predictions{ 0 }, m_wfs{ 0 }, m_drops{ 0 };
    std::atomic<std::uint64_t>           m_fallbackPredictions{
        0 };                              ///< Standard-only opportunities from actual scheduler requests.
    synco::Request        m_lastFallback; ///< Exact-request deduplication; no independent DM scheduling clock.
    std::atomic<unsigned> m_assistedReadyMask{ 0 }; ///< All phase fits must qualify before assisted mode is selected.
    std::atomic<std::uint64_t> m_latePredictions{ 0 }; ///< Whole-call deadline misses or unavailable timing.
    std::atomic<bool> m_predictionLate{ true }; ///< Latest data-causal forecast may still miss its actual slot cutoff.
    std::atomic<std::uint64_t> m_raw{ 0 };      ///< Original observer notifications, independent of learning jobs.
    std::array<std::atomic<std::uint64_t>, 4> m_phases{}; ///< Successful ADC contexts, not model adaptations.
    std::atomic<std::int64_t>                 m_fastNs{ -1 }, m_peakNs{ -1 }; ///< Whole fast inference/render time.
    std::atomic<const char *> m_reason{ "native_calibration_unavailable" };   ///< Precise native admission reason.

    /// Allocate all histories, forecasts and native learners before either owner starts.
    Impl( const NativePipelineConfig &inputs /**< [in] Startup contract. */,
          const OnlineShadowConfig   &config /**< [in] Numerical bounds. */ );
    /// Find a retained real WFS frame; never synthesize a missing frame.
    Frame *frame( std::uint64_t counter /**< [in] Real WFS identity. */ );
    /// Find the latest complete request actually available at a declared cutoff.
    const ProjectedRequest *request_at( std::int64_t cutoff /**< [in] Local availability limit. */ ) const;
    /// Break common or ADC-only continuity immediately, retaining coefficients but revoking old private outputs.
    void loss( bool         adcOnly /**< [in] Leave standard input history intact for ADC loss. */,
               std::int64_t now /**< [in] Actual loss detection. */ );
    /// Capture one causally available phase and evaluate immutable gains without allocation or factorization.
    void adc( const NativeObservation &record /**< [in] New individual ADC publication. */,
              std::int64_t             now /**< [in] Actual decision clock. */ );
    /// Apply only immutable cached matrices; called by ADC input or actual scheduler fallback opportunities.
    void predict( Frame        *frame /**< [in,out] Actual WFS context. */,
                  unsigned      phase /**< [in] Upstream/advertised chronological phase. */,
                  std::int64_t  now /**< [in] Actual decision time. */,
                  std::uint64_t adcCounter /**< [in] Actual ADC identity; zero denotes a scheduler-only fallback. */,
                  bool          assisted /**< [in] Assistance has valid individual data at this event. */,
                  std::chrono::steady_clock::time_point started /**< [in] Whole fast-call timing start. */ );
    /// Score only the target counter that has actually arrived.
    void score( const NativeObservation &modal /**< [in] New actual WFS target. */ );
    /// Bind the existing advertised four-slot record, never generate a new schedule.
    void bind_schedule( Frame               &frame /**< [in,out] Pending original WFS context. */,
                        const synco::Record &schedule /**< [in] Existing scheduler record. */,
                        std::int64_t         receipt /**< [in] Schedule availability. */ );
};

PhaseShadow::Impl::Impl( const NativePipelineConfig &inputs, const OnlineShadowConfig &config )
    : m_inputs( inputs ), m_config( config ), m_model( config.m_model ), m_queue( config.m_pendingCapacity + 1 ),
      m_scores( std::max<std::size_t>( 4096, config.m_predictionCapacity ) ),
      m_forecasts( ( config.m_model.m_future + 2 ) * 8 )
{
    m_model.m_accelChannels        = 8 * m_model.m_accelHistory;
    m_model.m_accelHistory         = 1;
    m_model.m_assistedCommandSlots = 4;
    m_model.m_externalOrigins      = true;
    m_model.m_scaledPenalty        = inputs.m_scaledPenalty;
    m_model.m_freezeQualification  = true;
    if( !inputs.m_enabled || m_model.m_accelChannels < 8 || m_model.m_accelChannels > 128 ||
        !std::isfinite( inputs.m_feedback ) || std::abs( inputs.m_feedback ) > 10 || config.m_pendingCapacity < 2 ||
        config.m_pendingCapacity > 128 || m_model.m_modes < 1 || m_model.m_modes > 2 || m_model.m_history < 1 ||
        m_model.m_history > 16 || m_model.m_future < 1 || m_model.m_future > 8 ||
        ( m_model.m_standardCommandSlots != 1 && m_model.m_standardCommandSlots != 4 ) ||
        ( m_model.m_future + m_model.m_history - 1 ) * 4 * m_model.m_modes + m_model.m_history * m_model.m_modes +
                m_model.m_accelChannels >
            128 )
        throw std::invalid_argument( "Invalid native phase receiver" );
    for( unsigned p = 0; p < 4; ++p )
    {
        m_learners[p] = std::make_unique<ShadowController>( m_model );
        m_learners[p]->training( ShadowMode::Standard, false );
        m_learners[p]->training( ShadowMode::AccelAssisted, false );
        for( unsigned m = 0; m < 2; ++m )
        {
            m_snapshots[p][m] = std::make_unique<ShadowSnapshot>();
            m_work[p][m]      = std::make_unique<ShadowSnapshot>();
        }
    }
    const auto &cal = inputs.m_calibration;
    if( cal.m_shapes.size() )
    {
        if( cal.m_shapes.cols() != m_model.m_modes || cal.m_shapes.rows() != cal.m_projection.cols() ||
            !cal.m_shapes.allFinite() )
            throw std::invalid_argument( "Native auxiliary shape extent" );
        m_producer = std::make_shared<SyncoShadowProducer>( cal.m_shapes.rows(), m_model.m_future );
    }
    m_status.m_upstream      = true;
    m_status.m_inputEvidence = InputEvidence::Requested;
    if( inputs.m_calibration.m_layout )
        m_reason.store( "awaiting_native_wfs_and_request_history" );
}
PhaseShadow::Impl::Frame *PhaseShadow::Impl::frame( std::uint64_t counter )
{
    auto &f = m_frames[counter % m_frames.size()];
    return f.m_counter == counter && f.m_valid ? &f : nullptr;
}
const ProjectedRequest *PhaseShadow::Impl::request_at( std::int64_t cutoff ) const
{
    for( unsigned n = 0; n < m_requestCount; ++n )
    {
        const auto &r = m_requests[( m_requestNext + m_requests.size() - n - 1 ) % m_requests.size()];
        if( r.m_source.m_availableNs <= cutoff && r.m_source.m_receivedNs <= cutoff )
            return &r;
    }
    return nullptr;
}
void PhaseShadow::Impl::loss( bool adcOnly, std::int64_t now )
{
    ++m_adcEpoch;
    m_adcNext = m_adcCount = m_goodAdc = 0;
    m_adcAllowed.store( false );
    m_fastValid.store( false );
    m_effective.store( 0 );
    if( !adcOnly )
    {
        ++m_epoch;
        m_commonEpoch.store( m_epoch );
        m_current = m_nextClose = 0;
        for( auto &f : m_frames )
            f.m_valid = false;
    }
    for( auto &f : m_forecasts )
        if( !adcOnly || f.m_mode )
            f.m_valid = false;
    if( m_producer )
        m_producer->inhibit( adcOnly, now );
}
void PhaseShadow::Impl::bind_schedule( Frame &f, const synco::Record &s, std::int64_t receipt )
{
    if( f.m_scheduled || !m_inputs.m_syncoAssociation || !synco::validProductionRecord( s, receipt ) ||
        s[synco::steps] != 4 || s[synco::period] < 450000 || s[synco::period] > 550000 || s[synco::cycle] < 0 ||
        s[synco::cycle] > std::numeric_limits<std::int64_t>::max() - 2 ||
        f.m_counter > std::uint64_t( std::numeric_limits<std::int64_t>::max() - 1000000001 ) ||
        std::int64_t( f.m_counter ) + m_inputs.m_syncoCycleOffset != s[synco::cycle] + 1 )
        return;
    std::sort( f.m_steps.begin(),
               f.m_steps.end(),
               [&]( unsigned a, unsigned b ) { return s[synco::targets + a] < s[synco::targets + b]; } );
    for( unsigned p = 0; p < 4; ++p )
        if( s[synco::cutoffs + f.m_steps[p]] <= receipt ||
            ( p && ( s[synco::cutoffs + f.m_steps[p]] <= s[synco::cutoffs + f.m_steps[p - 1]] ||
                     std::abs( 4 * ( s[synco::targets + f.m_steps[p]] - s[synco::targets + f.m_steps[p - 1]] ) -
                               s[synco::period] ) > 4 ) ) )
            return;
    if( m_firstStep >= 0 && int( f.m_steps[0] ) != m_firstStep )
    {
        if( m_producer )
            m_producer->inhibit( false, receipt );
        m_reason.store( "synco_slot_window_missed_or_association_changed" );
        return;
    }
    m_firstStep   = f.m_steps[0];
    f.m_schedule  = s;
    f.m_scheduled = true;
    for( unsigned p = 0; p < 4; ++p )
        f.m_cutoffs[p] = s[synco::cutoffs + f.m_steps[p]];
}
void PhaseShadow::Impl::score( const NativeObservation &w )
{
    for( auto &f : m_forecasts )
    {
        if( !f.m_valid || f.m_revision != m_reset.load() || w.m_counter <= f.m_origin )
            continue;
        const auto horizon = w.m_counter - f.m_origin;
        if( horizon > unsigned( m_model.m_future ) )
        {
            f.m_valid = false;
            continue;
        }
        for( int m = 0; m < m_model.m_modes; ++m )
        {
            ShadowPrediction out;
            out.m_mode       = f.m_mode ? ShadowMode::AccelAssisted : ShadowMode::Standard;
            out.m_origin     = f.m_origin;
            out.m_target     = w.m_counter;
            out.m_horizon    = horizon;
            out.m_channel    = m;
            out.m_prediction = out.m_originPrediction =
                f.m_prediction[( m_model.m_future - horizon ) * m_model.m_modes + m];
            out.m_measurement      = w.m_values[m];
            out.m_originValid      = true;
            out.m_late             = f.m_late;
            out.m_modelEpoch       = f.m_revision;
            out.m_phase            = f.m_phase;
            out.m_adcCounter       = f.m_adc;
            out.m_version          = f.m_version;
            out.m_decisionNs       = f.m_decision;
            out.m_completedNs      = f.m_completed;
            out.m_slotTargetNs     = f.m_target;
            out.m_candidate        = f.m_candidate[m];
            out.m_generation       = w.m_generation;
            out.m_conditionalValid = false;
            out.m_slotCycle        = f.m_request.cycle;
            out.m_slotStep         = f.m_request.step;
            out.m_slotEpoch        = f.m_request.epoch;
            out.m_slotGeneration   = f.m_request.generation;
            out.m_slotCutoffNs     = f.m_request.cutoffNs;
            out.m_epoch            = m_inputs.m_syncEpoch;
            out.m_layout           = m_inputs.m_calibration.m_layout;
            m_scores.push( out );
        }
        if( horizon == unsigned( m_model.m_future ) )
            f.m_valid = false;
    }
}
void PhaseShadow::Impl::adc( const NativeObservation &r, std::int64_t now )
{
    const auto started = std::chrono::steady_clock::now();
    ++m_opportunities;
    m_fastValid.store( false );
    m_adc[m_adcNext] = r;
    m_adcNext        = ( m_adcNext + 1 ) % m_adc.size();
    m_adcCount       = std::min<unsigned>( m_adcCount + 1, m_adc.size() );
    std::unique_lock<std::mutex> syncLock( m_syncMutex, std::try_to_lock );
    if( syncLock.owns_lock() )
    {
        const auto previous = m_activeSync;
        m_activeSync        = m_sync;
        syncLock.unlock();
        if( previous.m_status.m_receivedNs >= 0 &&
            ( m_activeSync.m_gaps != previous.m_gaps || m_activeSync.m_sequence < previous.m_sequence ||
              m_activeSync.m_referenceCounter < previous.m_referenceCounter ||
              ( previous.m_hasSlotOffset && m_activeSync.m_hasSlotOffset &&
                m_activeSync.m_slotOffset != previous.m_slotOffset ) ) )
        {
            loss( true, r.m_availableNs );
            m_adcSourceLost = true;
        }
    }
    const auto &s = m_activeSync;
    const auto &u = s.m_status;
    if( !m_current || m_sourceLost || m_fault.load() || now - m_last[1].m_availableNs > 1000000 )
        return;
    auto *f = frame( m_current );
    if( !f )
        return;
    const bool haveAnchor = m_inputs.m_adcAnchor && m_inputs.m_modalAnchor && r.m_counter >= m_inputs.m_adcAnchor;
    const auto offset     = haveAnchor ? r.m_counter - m_inputs.m_adcAnchor : 0;
    const bool associated = haveAnchor &&
                            offset / 4 <= std::numeric_limits<std::uint64_t>::max() - m_inputs.m_modalAnchor &&
                            m_inputs.m_modalAnchor + offset / 4 == m_current;
    int        phaseIndex = associated ? int( offset % 4 ) : -1;
    if( !associated && f->m_scheduled )
    {
        const auto step = synco::nextProductionStep( f->m_schedule, now );
        for( unsigned j = 0; j < 4; ++j )
            if( int( f->m_steps[j] ) == step )
                phaseIndex = j;
    }
    if( phaseIndex < 0 || phaseIndex >= 4 )
    {
        m_reason.store( "upstream_counter_reference_or_synco_opportunity_unavailable" );
        return;
    }
    const unsigned phase   = phaseIndex;
    const bool     locked  = associated && !m_adcSourceLost && u.m_synchronized && u.m_locked &&
                             u.m_epoch == m_inputs.m_syncEpoch && u.m_receivedNs >= 0 && u.m_receivedNs <= now &&
                             now - u.m_receivedNs <= m_inputs.m_statusMaxAgeNs;
    const unsigned samples = m_model.m_accelChannels / 2;
    bool           valid   = locked && m_adcCount >= samples && now >= r.m_receivedNs && now - r.m_receivedNs <= 250000;
    for( unsigned n = 0; valid && n < samples; ++n )
    {
        const auto &a = m_adc[( m_adcNext + m_adc.size() - n - 1 ) % m_adc.size()];
        valid = a.m_counter + n == r.m_counter && a.m_generation == r.m_generation && a.m_segment == r.m_segment &&
                a.m_availableNs <= now && a.m_receivedNs <= now && finite_modes( a, 2 );
        f->m_a[phase][2 * n]     = a.m_values[0];
        f->m_a[phase][2 * n + 1] = a.m_values[1];
    }
    m_goodAdc            = valid ? std::min( m_goodAdc + 1, 32u ) : 0;
    f->m_adcValid[phase] = valid && m_goodAdc >= 32;
    f->m_adcEpoch        = m_adcEpoch;
    m_adcAllowed.store( f->m_adcValid[phase] );
    predict( f, phase, now, associated ? r.m_counter : 0, associated, started );
}
void PhaseShadow::Impl::predict( Frame                                *f,
                                 unsigned                              phase,
                                 std::int64_t                          now,
                                 std::uint64_t                         adcCounter,
                                 bool                                  allowAssisted,
                                 std::chrono::steady_clock::time_point started )
{
    if( f->m_revision != m_reset.load() || m_resetPublished.load() != m_reset.load() )
        return;
    const auto *latest = request_at( now );
    if( !latest )
        return;
    std::unique_lock<std::mutex> lock( m_snapshotMutex, std::try_to_lock );
    if( !lock.owns_lock() )
        return;
    bool           ready[2]{};
    int            nextSlot = -1;
    synco::Request nextRequest;
    if( f->m_scheduled && synco::validProductionRecord( m_currentSchedule, now ) &&
        m_currentSchedule[synco::cycle] >= 0 &&
        m_currentSchedule[synco::cycle] < std::numeric_limits<std::int64_t>::max() - 1 &&
        m_currentSchedule[synco::epoch] == f->m_schedule[synco::epoch] &&
        m_currentSchedule[synco::generation] == f->m_schedule[synco::generation] )
    {
        const auto step = synco::nextProductionStep( m_currentSchedule, now );
        if( step >= 0 )
        {
            nextRequest = synco::request( m_currentSchedule, step );
            for( unsigned j = 0; j < 4; ++j )
            {
                const auto old = synco::request( f->m_schedule, f->m_steps[j] );
                if( old.step == step && old.cycle == nextRequest.cycle && old.targetNs == nextRequest.targetNs &&
                    old.cutoffNs == nextRequest.cutoffNs )
                    nextSlot = j;
            }
            const auto first = synco::request( f->m_schedule, f->m_steps[0] );
            if( nextSlot < 0 && step == first.step && nextRequest.cycle == first.cycle + 1 &&
                nextRequest.targetNs > f->m_schedule[synco::targets + f->m_steps[3]] )
                nextSlot = 4;
        }
    }
    for( unsigned mode = 0; mode < 2; ++mode )
    {
        const auto &snap = *m_snapshots[mode ? phase : 0][mode];
        if( !snap.m_status.m_ready || snap.m_inputEpoch != m_epoch || snap.m_dataAvailableNs > now ||
            ( mode && ( m_assistedReadyMask.load() != 15 || !f->m_adcValid[phase] || !allowAssisted ||
                        snap.m_sensorEpoch != m_adcEpoch ) ) )
            continue;
        const int      slots  = snap.m_commands / m_model.m_modes;
        const unsigned prefix = slots == 4 ? f->m_sampled : 0;
        const auto    &gain   = snap.m_gain[prefix];
        if( !gain.size() )
            continue;
        std::array<float, 192> storage{};
        int                    pos      = 0;
        bool                   complete = true;
        for( int h = 0; h < m_model.m_history; ++h )
        {
            auto *old = m_current >= unsigned( h ) ? frame( m_current - h ) : nullptr;
            if( !old || old->m_available > now )
            {
                complete = false;
                break;
            }
            for( int m = 0; m < m_model.m_modes; ++m )
                storage[pos++] = old->m_y[m];
        }
        for( int h = 1; complete && h < m_model.m_history; ++h )
        {
            auto *old = m_current >= unsigned( h ) ? frame( m_current - h ) : nullptr;
            if( !old || ( slots == 4 ? old->m_sampled != 4 : !old->m_singleValid ) )
            {
                complete = false;
                break;
            }
            for( int j = 0; j < slots * m_model.m_modes; ++j )
                storage[pos++] = slots == 4 ? old->m_u[j] : old->m_single[j];
        }
        if( mode )
            for( int j = 0; j < m_model.m_accelChannels; ++j )
                storage[pos++] = f->m_a[phase][j];
        if( !complete || pos != snap.m_stateRows )
            continue;
        for( int j = int( prefix ) - 1; j >= 0; --j )
        {
            if( f->m_sampled <= unsigned( j ) )
            {
                complete = false;
                break;
            }
            for( int m = 0; m < m_model.m_modes; ++m )
                storage[pos++] = f->m_u[( 3 - j ) * m_model.m_modes + m];
        }
        if( !complete || pos != gain.cols() )
            continue;
        Eigen::Map<const Matrix> state( storage.data(), pos, 1 );
        std::array<float, 2>     candidate{};
        Eigen::Map<Matrix>       output( candidate.data(), m_model.m_modes, 1 );
        output.noalias() = gain.bottomRows( m_model.m_modes ) * state;
        for( int m = 0; m < m_model.m_modes; ++m )
            candidate[m] += m_inputs.m_feedback * f->m_y[m];
        if( !output.allFinite() )
            continue;
        auto &forecast = m_forecasts[( ( m_current % ( m_model.m_future + 2 ) ) * 4 + phase ) * 2 + mode];
        forecast       = {};
        Eigen::Map<Matrix> predicted( forecast.m_prediction.data(), m_model.m_future * m_model.m_modes, 1 );
        predicted.noalias() = snap.m_theta.leftCols( snap.m_stateRows ) * state.topRows( snap.m_stateRows );
        for( int block = 0; block < m_model.m_future; ++block )
            for( int slot = 0; slot < slots; ++slot )
                for( int m = 0; m < m_model.m_modes; ++m )
                {
                    const int   chronological = slots * ( m_model.m_future - 1 - block ) + ( slots - 1 - slot );
                    const float v = chronological < int( prefix ) ? f->m_u[( 3 - chronological ) * m_model.m_modes + m]
                                                                  : latest->m_values[m];
                    predicted.noalias() +=
                        snap.m_theta.col( snap.m_stateRows + ( block * slots + slot ) * m_model.m_modes + m ) * v;
                }
        if( !predicted.allFinite() )
            continue;
        forecast.m_revision  = f->m_revision;
        forecast.m_origin    = m_current;
        forecast.m_adc       = adcCounter;
        forecast.m_phase     = phase;
        forecast.m_mode      = mode;
        forecast.m_version   = snap.m_status.m_updates;
        forecast.m_candidate = candidate;
        forecast.m_decision  = now;
        forecast.m_completed = now + elapsed( started );
        forecast.m_valid     = true;
        if( nextSlot >= 0 && ( slots == 4 ? int( prefix ) == nextSlot : nextSlot == 0 ) )
        {
            const auto request = nextRequest;
            forecast.m_request = request;
            forecast.m_target  = request.targetNs;
            forecast.m_late    = forecast.m_completed >= request.cutoffNs;
            if( m_producer && !forecast.m_late )
                m_producer->prepare_slot( mode ? ShadowMode::AccelAssisted : ShadowMode::Standard,
                                          std::span<const float>( candidate.data(), m_model.m_modes ),
                                          m_inputs.m_calibration.m_shapes,
                                          request,
                                          m_currentSchedule,
                                          forecast.m_completed,
                                          now );
        }
        ready[mode] = true;
    }
    lock.unlock();
    const bool assisted = m_requested.load() && ready[1];
    m_effective.store( assisted );
    if( m_producer )
    {
        m_producer->models( assisted ? ShadowMode::AccelAssisted : ShadowMode::Standard, ready[0], ready[1] );
        if( ready[0] || ready[1] )
            m_producer->callback( m_currentSchedule, now + elapsed( started ) );
    }
    if( ready[0] || ready[1] )
    {
        bool       late     = false;
        const auto complete = now + elapsed( started );
        for( unsigned mode = 0; mode < 2; ++mode )
        {
            if( !ready[mode] )
                continue;
            auto &forecast       = m_forecasts[( ( m_current % ( m_model.m_future + 2 ) ) * 4 + phase ) * 2 + mode];
            forecast.m_completed = complete;
            forecast.m_late      = forecast.m_target < 0 || complete >= forecast.m_request.cutoffNs;
            late |= forecast.m_late;
        }
        m_predictionLate.store( late );
        if( late )
            ++m_latePredictions;
        if( adcCounter )
        {
            ++m_predictions;
            ++m_phases[phase];
        }
        else
            ++m_fallbackPredictions;
        m_fastValid.store( assisted || ready[0] );
        m_lastPredictionNs.store( now );
        m_reason.store( adcCounter ? "adc_driven_shadow_prediction" : "standard_synco_fallback_prediction" );
    }
    const auto duration = elapsed( started );
    m_fastNs.store( duration );
    m_peakNs.store( std::max( m_peakNs.load(), duration ) );
}

PhaseShadow::PhaseShadow( const NativePipelineConfig &inputs, const OnlineShadowConfig &config )
    : m_impl( std::make_unique<Impl>( inputs, config ) )
{
}
PhaseShadow::~PhaseShadow() = default;
void PhaseShadow::command( const NativeObservation &r, std::span<const float> values )
{
    auto             &p = *m_impl;
    MultirateContract c;
    c.m_upstream.m_enabled = c.m_upstream.m_combinedRequestModel = c.m_inputs.m_mappingVerified = true;
    c.m_inputs.m_modes                                                                          = p.m_model.m_modes;
    c.m_sources[3] = { r.m_device, r.m_fileInode, r.m_imageInode, r.m_generation };
    ProjectedRequest projected;
    if( !project_request( c, r, values, p.m_inputs.m_calibration.m_projection, projected ) )
    {
        p.loss( false, r.m_availableNs );
        p.m_requestCount = 0;
        return;
    }
    p.m_requests[p.m_requestNext] = projected;
    p.m_requestNext               = ( p.m_requestNext + 1 ) % p.m_requests.size();
    p.m_requestCount              = std::min<unsigned>( p.m_requestCount + 1, p.m_requests.size() );
}
void PhaseShadow::command_poll( bool coherent, std::int64_t now )
{
    if( coherent )
        m_impl->m_commandObservedThrough = now;
}
void PhaseShadow::observe( const NativeObservation &r, std::int64_t now )
{
    auto &p = *m_impl;
    ++p.m_raw;
    const unsigned n = unsigned( r.m_source );
    if( n >= 4 || n == 2 )
        return;
    if( now < 0 )
        now = r.m_availableNs;
    if( r.m_availableNs > now || r.m_receivedNs > now )
    {
        p.loss( n == 0, now );
        return;
    }
    const auto &last = p.m_last[n];
    if( last.m_counter && r.m_event == ObservationEvent::Sample && r.m_availableNs < last.m_availableNs )
    {
        p.loss( n == 0, now );
        p.m_reason.store( "local_input_order_reversed" );
        return;
    }
    if( r.m_event == ObservationEvent::Sample && last.m_event == ObservationEvent::Sample &&
        r.m_counter == last.m_counter && r.m_generation == last.m_generation )
        return;
    const bool changed =
        r.m_event == ObservationEvent::Sample && p.m_sources[n].m_generation && !p.m_sources[n].matches( r );
    const bool bad = r.m_event != ObservationEvent::Sample || ( r.m_flags & ObservationInvalidPayload ) ||
                     !r.m_counter || r.m_counter == std::numeric_limits<std::uint64_t>::max() || !r.m_generation ||
                     !r.m_device || !r.m_fileInode || ( n && !r.m_imageInode ) || r.m_receivedNs < 0 ||
                     r.m_availableNs < r.m_receivedNs || !finite_modes( r, n == 0 ? 2 : p.m_model.m_modes );
    const bool gap = last.m_event == ObservationEvent::Sample && last.m_counter &&
                     r.m_event == ObservationEvent::Sample &&
                     ( r.m_counter != last.m_counter + 1 || ( n == 0 && r.m_segment != last.m_segment ) ||
                       ( n == 1 && r.m_availableNs - last.m_availableNs > 1000000 ) );
    if( bad || gap || changed )
    {
        p.loss( n == 0, r.m_availableNs );
        if( changed )
        {
            if( n == 0 )
                p.m_adcSourceLost = true;
            else
            {
                p.m_sourceLost = true;
                p.m_bindingValid.store( false );
            }
        }
        if( n == 3 )
            p.m_requestCount = 0;
    }
    p.m_last[n] = r;
    if( bad )
        return;
    if( !p.m_sources[n].m_generation )
        p.m_sources[n] = { r.m_device, r.m_fileInode, r.m_imageInode, r.m_generation };
    if( n == 0 )
    {
        p.m_adcAvailableNs.store( r.m_availableNs );
        tick( now );
        p.adc( r, now );
        return;
    }
    if( n != 1 )
        return;
    ++p.m_wfs;
    p.m_wfsAvailableNs.store( r.m_availableNs );
    p.score( r );
    if( p.m_current )
        if( auto *old = p.frame( p.m_current ) )
        {
            old->m_closed   = true;
            old->m_closedNs = r.m_availableNs;
        }
    auto &f = p.m_frames[r.m_counter % p.m_frames.size()];
    if( f.m_valid && !f.m_closed )
    {
        p.loss( false, r.m_availableNs );
        ++p.m_drops;
    }
    f             = {};
    f.m_counter   = r.m_counter;
    f.m_revision  = p.m_reset.load();
    f.m_available = r.m_availableNs;
    f.m_valid     = true;
    f.m_epoch     = p.m_epoch;
    f.m_adcEpoch  = p.m_adcEpoch;
    for( int m = 0; m < p.m_model.m_modes; ++m )
        f.m_y[m] = r.m_values[m];
    p.m_current   = r.m_counter;
    p.m_wfsSilent = false;
    if( !p.m_nextClose )
        p.m_nextClose = r.m_counter;
    tick( r.m_availableNs );
}
void PhaseShadow::tick( std::int64_t now )
{
    auto &p = *m_impl;
    if( p.m_lastPredictionNs.load() >= 0 && now - p.m_lastPredictionNs.load() > 250000 )
        p.m_fastValid.store( false );
    if( p.m_current && now - p.m_last[1].m_availableNs > 1000000 && !p.m_wfsSilent )
    {
        p.loss( false, now );
        p.m_wfsSilent = true;
        p.m_reason.store( "wfs_stale" );
    }
    if( p.m_last[0].m_counter && now - p.m_last[0].m_availableNs > 250000 )
    {
        p.m_adcAllowed.store( false );
        p.m_effective.store( 0 );
        if( p.m_producer )
            p.m_producer->inhibit( true, now );
    }
    for( auto &f : p.m_frames )
    {
        if( !f.m_valid )
            continue;
        while( f.m_scheduled && f.m_sampled < 4 && f.m_cutoffs[f.m_sampled] <= now &&
               f.m_cutoffs[f.m_sampled] <= p.m_commandObservedThrough )
        {
            const auto *u = p.request_at( f.m_cutoffs[f.m_sampled] );
            if( !u )
                break;
            for( int m = 0; m < p.m_model.m_modes; ++m )
                f.m_u[( 3 - f.m_sampled ) * p.m_model.m_modes + m] = u->m_values[m];
            ++f.m_sampled;
        }
        if( f.m_sampled == 4 )
        {
            f.m_singleValid = true;
            for( int m = 0; m < p.m_model.m_modes; ++m )
            {
                f.m_single[m] = f.m_u[m];
                for( int j = 1; j < 4; ++j )
                    f.m_singleValid &= f.m_u[j * p.m_model.m_modes + m] == f.m_single[m];
            }
        }
    }
    for( unsigned n = 0; n < 64 && p.m_nextClose && p.m_nextClose < p.m_current; ++n )
    {
        auto *f = p.frame( p.m_nextClose );
        if( !f )
        {
            ++p.m_nextClose;
            continue;
        }
        if( f->m_sampled != 4 && now - f->m_available < 2000000 )
            break;
        const auto head = p.m_head.load( std::memory_order_relaxed ), next = ( head + 1 ) % p.m_queue.size();
        if( next == p.m_tail.load( std::memory_order_acquire ) )
        {
            ++p.m_drops;
            p.loss( false, now );
            p.m_reason.store( "phase_learning_queue_overflow" );
            break;
        }
        p.m_queue[head] = *f;
        p.m_head.store( next, std::memory_order_release );
        ++p.m_nextClose;
    }
}
void PhaseShadow::schedule( const synco::Record &s, std::int64_t receipt )
{
    auto &p = *m_impl;
    if( !synco::validProductionRecord( s, receipt ) || s[synco::steps] != 4 || s[synco::period] < 450000 ||
        s[synco::period] > 550000 || s[synco::cycle] < 0 ||
        s[synco::cycle] >= std::numeric_limits<std::int64_t>::max() - 1 )
    {
        p.m_currentSchedule = {};
        p.m_scheduleExpiry.store( -1 );
        if( p.m_producer )
            p.m_producer->inhibit( false, receipt );
        for( auto &f : p.m_frames )
            if( f.m_valid && !f.m_closed )
            {
                f.m_scheduled = false;
                f.m_sampled   = 0;
            }
        return;
    }
    if( p.m_scheduleEpoch &&
        ( s[synco::epoch] != p.m_scheduleEpoch || s[synco::generation] != p.m_scheduleGeneration ) )
        p.loss( false, receipt );
    p.m_scheduleEpoch      = s[synco::epoch];
    p.m_scheduleGeneration = s[synco::generation];
    p.m_currentSchedule    = s;
    p.m_scheduleExpiry.store( s[synco::expires] );
    for( auto &f : p.m_frames )
        if( f.m_valid )
            p.bind_schedule( f, s, receipt );
    tick( receipt );
    if( !p.m_fault.load() && !p.m_adcAllowed.load() && !p.m_sourceLost && p.m_current &&
        receipt - p.m_last[1].m_availableNs <= 1000000 )
    {
        if( auto *f = p.frame( p.m_current ); f && f->m_scheduled )
        {
            const auto step = synco::nextProductionStep( s, receipt );
            if( step >= 0 )
            {
                const auto  request = synco::request( s, step );
                const auto &old     = p.m_lastFallback;
                if( request.epoch != old.epoch || request.generation != old.generation || request.cycle != old.cycle ||
                    request.step != old.step || request.targetNs != old.targetNs || request.cutoffNs != old.cutoffNs )
                {
                    for( unsigned phase = 0; phase < 4; ++phase )
                        if( f->m_steps[phase] == unsigned( step ) && f->m_sampled == phase )
                        {
                            p.m_lastFallback = request;
                            p.predict( f, phase, receipt, 0, false, std::chrono::steady_clock::now() );
                        }
                }
            }
        }
    }
}
void PhaseShadow::sync( const NativeSyncStatus &s )
{
    auto &p = *m_impl;
    { //mutex scope
        std::lock_guard<std::mutex> lock( p.m_syncMutex );
        if( s.m_status.m_receivedNs >= p.m_sync.m_status.m_receivedNs )
            p.m_sync = s;
    }
    if( !s.m_status.m_synchronized || !s.m_status.m_locked )
    {
        p.m_adcAllowed.store( false );
        if( p.m_producer )
            p.m_producer->inhibit( true, s.m_status.m_receivedNs );
    }
}
bool PhaseShadow::learn()
{
    auto      &p        = *m_impl;
    const auto revision = p.m_reset.load();
    if( revision != p.m_resetApplied )
    {
        for( auto &bank : p.m_learners )
            for( auto mode : { ShadowMode::Standard, ShadowMode::AccelAssisted } )
                bank->reset( mode );
        { //mutex scope
            std::lock_guard<std::mutex> lock( p.m_snapshotMutex );
            for( auto &phase : p.m_snapshots )
                for( auto &snapshot : phase )
                {
                    snapshot->m_status = {};
                    snapshot->m_theta.setZero();
                    for( auto &gain : snapshot->m_gain )
                        gain.resize( 0, 0 );
                }
        }
        p.m_status.m_standard = p.m_status.m_assisted = {};
        p.m_squared.fill( 0 );
        p.m_scored.fill( 0 );
        p.m_assistedReadyMask.store( 0 );
        p.m_resetApplied = revision;
        p.m_resetPublished.store( revision );
    }
    const auto tail = p.m_tail.load( std::memory_order_relaxed );
    if( tail == p.m_head.load( std::memory_order_acquire ) || p.m_fault.load() || !p.m_bindingValid.load() )
        return false;
    const auto f = p.m_queue[tail];
    p.m_tail.store( ( tail + 1 ) % p.m_queue.size(), std::memory_order_release );
    if( f.m_revision != revision )
        return true;
    if( f.m_epoch != p.m_learningEpoch )
    {
        for( auto &model : p.m_learners )
            model->invalidate( {} );
        p.m_learningEpoch = f.m_epoch;
    }
    if( f.m_adcEpoch != p.m_learningAdcEpoch )
    {
        for( auto &model : p.m_learners )
            model->invalidate_acceleration();
        p.m_learningAdcEpoch = f.m_adcEpoch;
    }
    ShadowRecord row;
    row.m_index = f.m_counter;
    row.m_residual.resize( p.m_model.m_modes, 1 );
    row.m_appliedCommand.resize( p.m_model.m_standardCommandSlots * p.m_model.m_modes, 1 );
    row.m_assistedAppliedCommand.resize( 4 * p.m_model.m_modes, 1 );
    row.m_acceleration.resize( p.m_model.m_accelChannels, 1 );
    for( int m = 0; m < p.m_model.m_modes; ++m )
        row.m_residual( m ) = f.m_y[m];
    for( int m = 0; m < 4 * p.m_model.m_modes; ++m )
        row.m_assistedAppliedCommand( m ) = f.m_u[m];
    for( int m = 0; m < row.m_appliedCommand.rows(); ++m )
        row.m_appliedCommand( m ) = p.m_model.m_standardCommandSlots == 4 ? f.m_u[m] : f.m_single[m];
    row.m_wfsValid = row.m_timingValid = true;
    row.m_appliedValid                 = p.m_model.m_standardCommandSlots == 4 ? f.m_sampled == 4 : f.m_singleValid;
    row.m_assistedAppliedValid         = f.m_sampled == 4;
    const bool training                = p.m_learning.load();
    for( unsigned phase = 0; phase < 4; ++phase )
    {
        for( int j = 0; j < p.m_model.m_accelChannels; ++j )
            row.m_acceleration( j ) = f.m_a[phase][j];
        row.m_accelValid = f.m_adcValid[phase];
        p.m_learners[phase]->training( ShadowMode::Standard, phase == 0 && training );
        p.m_learners[phase]->training( ShadowMode::AccelAssisted, training );
        p.m_learners[phase]->process( row );
        for( unsigned mode = 0; mode < 2; ++mode )
        {
            if( !mode && phase )
                continue;
            const auto  state = p.m_learners[phase]->status( mode ? ShadowMode::AccelAssisted : ShadowMode::Standard );
            const auto &old   = *p.m_snapshots[phase][mode]; // Only this numerical owner changes snapshot pointers.
            if( old.m_status.m_ready == state.m_ready && old.m_status.m_updates == state.m_updates &&
                old.m_inputEpoch == f.m_epoch && ( !mode || old.m_sensorEpoch == f.m_adcEpoch ) )
                continue;
            p.m_learners[phase]->snapshot( mode ? ShadowMode::AccelAssisted : ShadowMode::Standard,
                                           *p.m_work[phase][mode] );
            p.m_work[phase][mode]->m_inputEpoch  = f.m_epoch;
            p.m_work[phase][mode]->m_sensorEpoch = f.m_adcEpoch;
            p.m_work[phase][mode]->m_dataAvailableNs =
                std::max( f.m_closedNs, f.m_sampled == 4 ? f.m_cutoffs[3] : f.m_available );
            { //mutex scope
                std::lock_guard<std::mutex> lock( p.m_snapshotMutex );
                p.m_work[phase][mode].swap( p.m_snapshots[phase][mode] );
            }
        }
    }
    ++p.m_status.m_records;
    p.m_status.m_qualified =
        ( row.m_appliedValid || row.m_assistedAppliedValid ) && row.m_wfsValid && row.m_timingValid;
    p.m_reason.store( !p.m_inputs.m_calibration.m_layout ? "native_calibration_unavailable"
                      : row.m_appliedValid               ? "phase_models_warming_or_unexcited"
                      : row.m_assistedAppliedValid       ? "standard_held_input_contract_not_met"
                                                         : "request_slot_schedule_or_history_incomplete" );
    p.m_status.m_standard = p.m_learners[0]->status( ShadowMode::Standard );
    p.m_status.m_assisted = p.m_learners[0]->status( ShadowMode::AccelAssisted );
    for( unsigned phase = 1; phase < 4; ++phase )
    {
        const auto s = p.m_learners[phase]->status( ShadowMode::AccelAssisted );
        p.m_status.m_assisted.m_ready &= s.m_ready;
        p.m_status.m_assisted.m_updates = std::min( p.m_status.m_assisted.m_updates, s.m_updates );
        p.m_status.m_assisted.m_rank    = std::min( p.m_status.m_assisted.m_rank, s.m_rank );
    }
    unsigned mask = 0;
    for( unsigned phase = 0; phase < 4; ++phase )
        if( p.m_learners[phase]->status( ShadowMode::AccelAssisted ).m_ready )
            mask |= 1u << phase;
    p.m_assistedReadyMask.store( mask );
    return true;
}
std::size_t PhaseShadow::queued() const
{
    const auto &p = *m_impl;
    return ( p.m_head.load() + p.m_queue.size() - p.m_tail.load() ) % p.m_queue.size();
}
bool PhaseShadow::read_prediction( ShadowPrediction &r )
{
    auto &p = *m_impl;
    do
    {
        if( !p.m_scores.pop( r ) )
            return false;
    } while( r.m_modelEpoch != p.m_reset.load() );
    const unsigned m     = r.m_mode == ShadowMode::AccelAssisted;
    const double   error = r.m_originPrediction - r.m_measurement;
    p.m_squared[m] += error * error;
    ++p.m_scored[m];
    return true;
}
OnlineShadowStatus PhaseShadow::status( std::int64_t now ) const
{
    const auto &p     = *m_impl;
    auto        s     = p.m_status;
    s.m_modelEpoch    = p.m_resetApplied;
    s.m_schedule      = p.m_inputs.m_syncoAssociation && p.m_scheduleExpiry.load() >= 0 &&
                        ( now < 0 || now < p.m_scheduleExpiry.load() );
    s.m_running       = true;
    s.m_upstream      = true;
    s.m_inputEvidence = InputEvidence::Requested;
    s.m_qualified     = s.m_qualified && p.m_resetApplied == p.m_reset.load() && p.m_inputs.m_calibration.m_layout &&
                        s.m_records && !p.m_fault.load() && p.m_bindingValid.load() &&
                        p.m_learningEpoch == p.m_commonEpoch.load();
    s.m_requested     = p.m_requested.load() ? ShadowMode::AccelAssisted : ShadowMode::Standard;
    s.m_effective     = p.m_effective.load() ? ShadowMode::AccelAssisted : ShadowMode::Standard;
    s.m_syncValid     = p.m_adcAllowed.load();
    s.m_fastOpportunities = p.m_opportunities.load();
    s.m_fastPredictions   = p.m_predictions.load();
    s.m_wfsReceived       = p.m_wfs.load();
    s.m_forecasts         = s.m_fastPredictions + p.m_fallbackPredictions.load();
    s.m_predictionValid   = p.m_fastValid.load() && !p.m_fault.load();
    s.m_predictionLate    = p.m_predictionLate.load();
    s.m_lateForecasts     = p.m_latePredictions.load();
    s.m_candidateValid    = false;
    s.m_drops             = p.m_drops.load();
    s.m_predictionDrops   = p.m_scores.drops();
    s.m_raw               = p.m_raw.load();
    s.m_adapterReason = s.m_reason = p.m_reason.load();
    s.m_rawQueued                  = ( p.m_head.load() + p.m_queue.size() - p.m_tail.load() ) % p.m_queue.size();
    for( unsigned phase = 0; phase < 4; ++phase )
        s.m_phasePredictions[phase] = p.m_phases[phase].load();
    s.m_fastUs     = p.m_fastNs.load() < 0 ? -1 : p.m_fastNs.load() / 1000.;
    s.m_fastPeakUs = p.m_peakNs.load() < 0 ? -1 : p.m_peakNs.load() / 1000.;
    if( !s.m_qualified )
        s.m_standard.m_ready = s.m_assisted.m_ready = false;
    s.m_assisted.m_ready = s.m_assisted.m_ready && s.m_syncValid;
    if( now >= 0 )
    {
        const auto wfs = p.m_wfsAvailableNs.load(), adc = p.m_adcAvailableNs.load();
        const bool fresh    = wfs >= 0 && now - wfs <= 1000000;
        const bool adcFresh = adc >= 0 && now - adc <= 250000;
        if( !fresh )
        {
            s.m_qualified = s.m_standard.m_ready = s.m_assisted.m_ready = s.m_predictionValid = false;
            if( p.m_inputs.m_calibration.m_layout )
                s.m_reason = "wfs_stale";
        }
        if( !adcFresh )
        {
            s.m_assisted.m_ready = s.m_syncValid = false;
            s.m_effective                        = ShadowMode::Standard;
        }
        if( p.m_lastPredictionNs.load() < 0 || now - p.m_lastPredictionNs.load() > 250000 )
            s.m_predictionValid = false;
    }
    for( unsigned m = 0; m < 2; ++m )
    {
        auto &model               = m ? s.m_assisted : s.m_standard;
        model.m_originPredictions = p.m_scored[m] / p.m_model.m_modes;
        model.m_originRmse        = p.m_scored[m] ? std::sqrt( p.m_squared[m] / p.m_scored[m] ) : 0;
        model.m_rmse              = model.m_originRmse;
    }
    if( p.m_producer )
        p.m_producer->status( s );
    return s;
}
void PhaseShadow::request( ShadowMode mode )
{
    m_impl->m_requested.store( mode == ShadowMode::AccelAssisted );
}
void PhaseShadow::learning( bool enabled )
{
    m_impl->m_learning.store( enabled );
}
void PhaseShadow::reset_models()
{
    ++m_impl->m_reset;
    m_impl->m_fastValid.store( false );
    if( m_impl->m_producer )
        m_impl->m_producer->inhibit( false );
}
void PhaseShadow::inhibit()
{
    m_impl->m_fault.store( true );
    if( m_impl->m_producer )
        m_impl->m_producer->inhibit( false, std::numeric_limits<std::int64_t>::max() );
}
Matrix PhaseShadow::coefficients( ShadowMode mode, unsigned phase ) const
{
    if( phase >= 4 )
        return {};
    const unsigned              m = mode == ShadowMode::AccelAssisted;
    std::lock_guard<std::mutex> lock( m_impl->m_snapshotMutex );
    return m_impl->m_snapshots[m ? phase : 0][m]->m_theta;
}
std::shared_ptr<SyncoShadowProducer> PhaseShadow::producer() const
{
    return m_impl->m_producer;
}
} // namespace DDSPC
